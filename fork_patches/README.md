# Fork patches ([1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) 1.2.2)

Reviewable, version-controlled copies of every fork change, tracked here so
the work is diffable and reversible outside the installed package. The fork
is a wheel install (no source checkout), so changes land under
`~/miniconda3/envs/1cat-vllm-122/lib/python3.12/site-packages/` on
c4130-local. **Apply by copying the tracked file over the target once and
keeping the `.pre_*`/`.orig` backup; never hand-edit the installed file.**
Deep/experimental changes are kept as unified-diff `.patch` files rather
than full-file copies, and are meant to be developed in a proper source
checkout, not hot-applied to a live server.

| File | Install path (under site-packages) | Our changes |
|---|---|---|
| `gdn_attn.py` | `vllm/v1/attention/backends/` | Chain-MTP GDN fast metadata build (`VLLM_SM70_GDN_CHAIN_SPEC_FAST_BUILD`, −1.4 ms/step, byte-identical) + env-gated slot-debug instrumentation. Box backup: `.pre_chainfast`. |
| `gpu_model_runner.py` | `vllm/v1/worker/` | Env-gated think-only draft gate (off; blocked on async-scheduler contract) + slot-debug instrumentation. Box backup: `.pre_thinkonly`. |
| `marlin.py` | `vllm/model_executor/kernels/linear/nvfp4/marlin.py` | Skinny-kernel dispatch shim + QPN dispatch/loader-prepack (`VLLM_SKINNY_NVFP4`, `VLLM_SKINNY_QPN`, `VLLM_SKINNY_DROP_CT`, `VLLM_SKINNY_MAX_M`, 4-bit lm_head policy). Box backup: `marlin.py.orig`. |
| `sm70_native_round.py` | *not installed* — lives in `~/flatness-run/`, file-imported | Native speculative-round executor (drafter chain as one CUDA graph). **Experimental, not in production.** |
| `llm_base_proposer.native_round.patch` | unified diff vs `.pre_native` | The proposer hook that selects the native round. **Reverted from the live env** (proposer restored to pristine); kept as a patch for proper source-checkout development. See `../native_round_design.md`. |

## Status of the native-round work (2026-08-11)

Built and validated (byte-identical output vs the Python path, all cells)
but **inert**: served drafts are rejected (τ≈0) because the captured graph
does not correctly persist the drafter's recurrent state across rounds —
the session-8 "full-graph drafter" hazard in a new form. The live env is
back on the stock Python proposer; the flag (`VLLM_SM70_NATIVE_SPEC_ROUND`)
does nothing until the patch is re-applied. Next step belongs in a source
checkout with an editable install and a debugger, not live hot-patching —
design and options in `../native_round_design.md`.
