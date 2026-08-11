#!/usr/bin/env python3
"""Offline test of the patched MarlinNvFp4LinearKernel: skinny vs marlin
on CT-format tensors at the real Qwen3.5-27B TP4 per-rank shapes."""
import os

os.environ["VLLM_SKINNY_NVFP4"] = "1"
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import torch
from vllm.model_executor.kernels.linear.nvfp4 import marlin as mk

dev = torch.device("cuda:0")
torch.manual_seed(0)

# (K, N) per TP4 rank: in_proj_qkvz, out_proj, qkv_proj, gate_up, down
SHAPES = [(5120, 4096), (1536, 5120), (5120, 3584), (5120, 8704), (4352, 5120)]


class FakeLayer(torch.nn.Module):
    pass


kernel = mk.MarlinNvFp4LinearKernel(mk.NvFp4LinearLayerConfig())
all_ok = True
for k, n in SHAPES:
    layer = FakeLayer()
    layer.input_size_per_partition = k
    layer.output_size_per_partition = n
    layer.params_dtype = torch.half
    layer.weight = torch.nn.Parameter(
        torch.randint(0, 256, (n, k // 2), dtype=torch.uint8, device=dev),
        requires_grad=False,
    )
    scales = (torch.rand(n, k // 16, device=dev) * 400 + 8).to(
        torch.float8_e4m3fn
    )
    layer.weight_scale = torch.nn.Parameter(scales, requires_grad=False)
    layer.weight_global_scale = torch.nn.Parameter(
        torch.tensor(0.0021, dtype=torch.float32, device=dev),
        requires_grad=False,
    )
    kernel.process_weights_after_loading(layer)

    for m in (1, 8, 16, 24, 64, 128):
        x = torch.randn(m, k, dtype=torch.float16, device=dev) * 0.1
        mk._self_checks_done = 0  # force the in-patch check every call
        y = kernel.apply_weights(layer, x)
        y_ref = kernel._marlin_apply(layer, x, None)
        rel = ((y.float() - y_ref.float()).abs().max()
               / y_ref.float().abs().max().clamp(min=1e-6)).item()
        route = ("simt" if m in (1, 2, 4, 8) and k % 1024 == 0 else
                 "wmma" if m <= 64 else "marlin")
        ok = rel < 3e-2 and mk._skinny_ok
        all_ok &= ok
        print(f"K={k:<5} N={n:<5} M={m:<3} route={route:<6} "
              f"rel={rel:.2e} [{'ok' if ok else 'FAIL'}]")

print("\nINTEGRATION", "PASS" if all_ok and mk._skinny_ok else "FAIL")
