#!/usr/bin/env python3
"""Skinny kernel vs CT-spec dequant on REAL Qwen3.5-27B-NVFP4 tensors.

Reads individual quantized tensors straight from model.safetensors
(partial reads, no full model load, no vLLM) and checks
    kernel(x, codes, scales, 1/global)  ==  x @ dequant(W).T
Runtime: ~30s.
"""
import os

import torch
from safetensors import safe_open

CKPT = os.path.expanduser("~/models/Qwen3.5-27B-NVFP4/model.safetensors")
dev = torch.device("cuda:0")

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
from torch.utils.cpp_extension import load  # noqa: E402

ext = load(
    name="skinny_nvfp4_v9",
    sources=[os.path.expanduser("~/flatness-run/skinny_kernels.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                       "-gencode=arch=compute_70,code=sm_70"],
    verbose=False,
)

E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=dev)

with safe_open(CKPT, framework="pt") as f:
    names = [n for n in f.keys() if n.endswith("weight_packed")]
    print(f"{len(names)} quantized tensors in checkpoint")
    # a spread of layers/projections
    picks = [n for n in names if any(
        s in n for s in (".0.", ".13.", ".30."))][:8] or names[:8]

    all_ok = True
    for name in picks:
        base = name[: -len("weight_packed")]
        codes = f.get_tensor(name).to(dev)                      # [N, K/2] u8
        scales = f.get_tensor(base + "weight_scale").to(dev)    # [N, K/16] fp8
        graw = f.get_tensor(base + "weight_global_scale").float().to(dev)
        n, k2 = codes.shape
        k = k2 * 2
        gscale = float(1.0 / graw.item())                       # CT stores 1/s

        # CT-spec dequant reference
        lo = (codes & 0x0F).long()
        hi = (codes >> 4).long()
        idx = torch.stack([lo, hi], dim=-1).view(n, k)          # even k = low
        mag = E2M1[idx & 0x7]
        sign = torch.where((idx & 0x8) > 0, -1.0, 1.0)
        eff = scales.float().repeat_interleave(16, dim=1) * gscale
        w_ref = (mag * sign * eff)                              # [N, K] fp32

        for m in (1, 8, 24):
            x = (torch.randn(m, k, dtype=torch.float16, device=dev) * 0.05)
            use_simt = m in (1, 2, 4, 8) and k % 1024 == 0 and n % 8 == 0
            fn = ext.gemm_simt if use_simt else ext.gemm_wmma
            if not use_simt and (k % 128 or n % 64):
                print(f"{base} M={m}: shape not skinny-eligible, skip")
                continue
            y = fn(x, codes.contiguous(),
                   scales.view(torch.uint8).contiguous(), gscale)
            y_ref = x.float() @ w_ref.T
            rel = ((y.float() - y_ref).abs().max()
                   / y_ref.abs().max().clamp(min=1e-6)).item()
            ok = rel < 1e-2
            all_ok &= ok
            print(f"{base[:60]:<60} N={n:<5} K={k:<5} M={m:<2} "
                  f"{'simt' if use_simt else 'wmma'} rel={rel:.2e} "
                  f"[{'ok' if ok else 'FAIL'}]")
        del codes, scales, w_ref, mag, sign, eff, idx, lo, hi
        torch.cuda.empty_cache()

print("\nREAL-WEIGHTS", "PASS" if all_ok else "FAIL")
