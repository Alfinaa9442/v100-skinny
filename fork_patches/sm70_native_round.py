# SPDX-License-Identifier: Apache-2.0
"""SM70 native speculative round (stage 1: drafter chain).

Captures the k-1 drafter loop iterations of chain-MTP — forwards,
argmax sampling, and inter-iteration input feeding — into ONE CUDA
graph, replayed as a single native launch per round. Eliminates the
measured per-iteration Python: ~0.36 ms metadata + ~0.29 ms sampling
host per iteration plus glue (anatomy 2026-08-10: proposer wall 11.6 ms
vs 6.05 GPU at k=7).

Gate: VLLM_SM70_NATIVE_SPEC_ROUND=1, and only for method=mtp,
batch_size==1, uniform decode, no MM inputs, static drafter shapes.
The Python path is untouched and remains the reference/fallback.

Round-dynamic state is carried in OWNED device buffers refreshed by a
handful of device copies before each replay (`refresh_from`), so the
captured kernels always read current values. Host-baked scalars inside
attention metadata are the known hazard (the session-8 "full-graph
drafter mine"); validation is byte-identity vs the Python path over
full generations before any benchmark is quoted.
"""

from __future__ import annotations

import os
from dataclasses import fields, is_dataclass
from typing import Any

import torch

NATIVE_ROUND_ENABLED = os.environ.get("VLLM_SM70_NATIVE_SPEC_ROUND", "0") == "1"
NATIVE_ROUND_DEBUG = os.environ.get("VLLM_SM70_NATIVE_ROUND_DEBUG", "0") == "1"


def _lockstep_copy(live: Any, owned: Any, stats: dict | None = None,
                   path: str = "") -> None:
    """Walk two structurally-identical metadata trees; copy every live
    device tensor's values into the owned clone the graph reads."""
    if isinstance(live, torch.Tensor):
        if live.is_cuda and isinstance(owned, torch.Tensor) \
                and owned.shape == live.shape:
            owned.copy_(live, non_blocking=True)
            if stats is not None:
                stats["copied"] += 1
        elif live.is_cuda:
            if stats is not None:
                stats["skipped"] += 1
                stats["skipped_paths"].append(
                    f"{path}:{tuple(live.shape)}vs"
                    f"{tuple(owned.shape) if isinstance(owned, torch.Tensor) else type(owned).__name__}")
        return
    if is_dataclass(live) and not isinstance(live, type):
        for f in fields(live):
            _lockstep_copy(getattr(live, f.name), getattr(owned, f.name),
                           stats, f"{path}.{f.name}")
        return
    if isinstance(live, dict):
        for k in live:
            _lockstep_copy(live[k], owned.get(k), stats, f"{path}[{k}]")
        return
    if isinstance(live, (list, tuple)):
        for i, (lv, ov) in enumerate(zip(live, owned)):
            _lockstep_copy(lv, ov, stats, f"{path}[{i}]")
        return


def _clone_device_tensors(obj: Any, registry: list[tuple[torch.Tensor, torch.Tensor]]):
    """Deep-copy a metadata object, replacing every device tensor with an
    owned clone; (live_source, owned_clone) pairs land in `registry` so
    refresh_from() can update them per round with device copies."""
    if isinstance(obj, torch.Tensor):
        if obj.is_cuda:
            owned = obj.clone()
            registry.append((obj, owned))
            return owned
        return obj
    if is_dataclass(obj) and not isinstance(obj, type):
        kwargs = {}
        for f in fields(obj):
            kwargs[f.name] = _clone_device_tensors(getattr(obj, f.name), registry)
        return type(obj)(**kwargs)
    if isinstance(obj, dict):
        return {k: _clone_device_tensors(v, registry) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        vals = [_clone_device_tensors(v, registry) for v in obj]
        return type(obj)(vals) if not isinstance(obj, tuple) else tuple(vals)
    return obj


class NativeDraftRound:
    """Owns the captured drafter-chain graph and its buffers."""

    def __init__(self, proposer) -> None:
        self.proposer = proposer
        self.graph: torch.cuda.CUDAGraph | None = None
        self.registry: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.live_sources_key: list[int] | None = None
        self.out_draft_ids: torch.Tensor | None = None
        self.first_token_slot: torch.Tensor | None = None
        self.failed = False
        self._debug_rounds = 0

    # ---- capture -----------------------------------------------------
    def capture(self, common_attn_metadata, sampling_metadata, batch_size: int,
                input_batch_size: int, batch_size_across_dp,
                hidden_states: torch.Tensor,
                first_draft_ids: torch.Tensor,
                positions_entry: torch.Tensor) -> bool:
        """Capture iterations 1..k-1 as one graph. Returns False (and
        marks failed) if anything refuses capture — caller falls back to
        the Python loop permanently."""
        from vllm.forward_context import set_forward_context

        p = self.proposer
        k = p.num_speculative_tokens
        dev = first_draft_ids.device
        try:
            self.registry = []
            common_owned = _clone_device_tensors(common_attn_metadata,
                                                 self.registry)
            # Prebuild per-iteration metadata from the owned common copy.
            per_iter_meta = []
            positions_owned = []
            slot_owned = []
            positions = positions_entry.clone()
            # Every captured forward must read ITS OWN slot buffer: the
            # Python loop mutates the shared _slot_mapping_buffer between
            # forwards, but a graph has no host code between them — a
            # shared buffer makes all iterations write drafter state at
            # the last iteration's slot (the tau=0 serve-mode defect).
            real_slot_buf = p._slot_mapping_buffer
            try:
                for token_index in range(k - 1):
                    spec_step_idx = token_index + 1
                    own_buf = real_slot_buf.clone()
                    slot_owned.append(own_buf)
                    p._slot_mapping_buffer = own_buf
                    if not p.constant_draft_positions:
                        positions = p._update_positions_dependent_metadata(
                            positions, common_owned, batch_size,
                            input_batch_size, p._native_block_size)
                    _, meta = p.build_per_group_and_layer_attn_metadata(
                        common_owned, draft_index=spec_step_idx)
                    per_iter_meta.append(meta)
                    positions_owned.append(positions.clone())
            finally:
                p._slot_mapping_buffer = real_slot_buf
            self.slot_owned = slot_owned
            self.per_iter_meta_owned = per_iter_meta
            self.common_owned = common_owned

            self.pos_pristine = [t.clone() for t in positions_owned]
            self.pos_owned = positions_owned
            self.captured_entry_pos = positions_entry.clone()
            self.first_token_slot = first_draft_ids.clone()
            self.out_draft_ids = torch.zeros(k - 1, dtype=torch.long,
                                             device=dev)
            hs_owned = hidden_states.clone()
            self.hs_slot = hs_owned

            torch.cuda.synchronize()
            from vllm.distributed import graph_capture as _vllm_graph_capture
            self.graph = torch.cuda.CUDAGraph()
            with _vllm_graph_capture(device=dev) as _gc_ctx, \
                    torch.cuda.graph(self.graph, stream=_gc_ctx.stream):
                prev_ids = self.first_token_slot
                hs = hs_owned
                for token_index in range(k - 1):
                    spec_step_idx = token_index + 1
                    p.input_ids[:batch_size] = prev_ids.int()
                    p.hidden_states[:batch_size] = hs
                    model_kwargs = {
                        "input_ids": p.input_ids[:input_batch_size],
                        "positions": positions_owned[token_index],
                        "inputs_embeds": None,
                    }
                    if p.pass_hidden_states_to_model:
                        model_kwargs["hidden_states"] = (
                            p.hidden_states[:input_batch_size])
                    model_kwargs = p._add_spec_step_idx(model_kwargs,
                                                        spec_step_idx)
                    model_kwargs = p._prepare_model_kwargs_for_aot(model_kwargs)
                    from vllm.config import CUDAGraphMode
                    with set_forward_context(
                        per_iter_meta[token_index],
                        p.vllm_config,
                        num_tokens=input_batch_size,
                        num_tokens_across_dp=batch_size_across_dp,
                        cudagraph_runtime_mode=CUDAGraphMode.NONE,
                        slot_mapping={
                            name: self.slot_owned[token_index]
                            [:input_batch_size]
                            for name in p._draft_attn_layer_names},
                    ):
                        ret = p.model(**model_kwargs)
                    if not p.model_returns_tuple():
                        last_hs, hs_full = ret, ret
                    else:
                        last_hs, hs_full = ret
                    hs = hs_full[:batch_size]
                    ids, _ = p._sample_draft_tokens(
                        last_hs[:batch_size], sampling_metadata,
                        spec_step_idx=spec_step_idx)
                    self.out_draft_ids[token_index] = ids.view(-1)[0]
                    prev_ids = ids
            return True
        except Exception as exc:  # noqa: BLE001 — any refusal => fallback
            from vllm.logger import init_logger
            init_logger(__name__).warning(
                "native round capture failed (%s: %s); using Python path.",
                type(exc).__name__, exc)
            self.failed = True
            self.graph = None
            return False

    # ---- per-round entry --------------------------------------------
    def run(self, common_attn_metadata, hidden_states: torch.Tensor,
            first_draft_ids: torch.Tensor,
            positions_entry: torch.Tensor) -> torch.Tensor:
        """Per-round: refresh the owned common metadata, rebuild the live
        per-iteration derived metadata on the host (stage-1 cost kept),
        lockstep-copy it into the captured trees, then one replay."""
        p = self.proposer
        stats = ({"copied": 0, "skipped": 0, "skipped_paths": []}
                 if NATIVE_ROUND_DEBUG and self._debug_rounds < 3 else None)
        _lockstep_copy(common_attn_metadata, self.common_owned, stats,
                       "common")
        positions = positions_entry.clone()
        for token_index in range(p.num_speculative_tokens - 1):
            spec_step_idx = token_index + 1
            if not p.constant_draft_positions:
                positions = p._update_positions_dependent_metadata(
                    positions, common_attn_metadata, 1, 1,
                    p._native_block_size)
            self.slot_owned[token_index].copy_(
                p._slot_mapping_buffer, non_blocking=True)
            _, live_meta = p.build_per_group_and_layer_attn_metadata(
                common_attn_metadata, draft_index=spec_step_idx)
            _lockstep_copy(live_meta, self.per_iter_meta_owned[token_index],
                           stats, f"iter{token_index}")
            self.pos_owned[token_index].copy_(positions, non_blocking=True)
        if stats is not None:
            self._debug_rounds += 1
            from vllm.logger import init_logger
            init_logger(__name__).info(
                "native round refresh: copied=%d skipped=%d %s",
                stats["copied"], stats["skipped"],
                stats["skipped_paths"][:6])
        self.first_token_slot.copy_(first_draft_ids, non_blocking=True)
        self.hs_slot.copy_(hidden_states, non_blocking=True)
        self.graph.replay()
        return self.out_draft_ids
