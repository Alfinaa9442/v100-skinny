#!/usr/bin/env python3
"""QPN integration check on REAL checkpoint tensors, through the actual
production code: marlin.py's _qpn_prepack + skinny ext's gemm_qpn.
Reads quantized tensors straight from model.safetensors (partial reads).
Gate: rel <= 1e-3 vs CT-spec dequant reference at M {5, 8, 11, 16},
outlier activations included. Runtime ~5 min incl. extension build.
"""
import os

import torch
from safetensors import safe_open

CKPT = os.path.expanduser("~/models/Qwen3.5-27B-NVFP4/model.safetensors")
dev = torch.device("cuda:0")
os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
os.environ.setdefault("VLLM_SKINNY_NVFP4", "1")
os.environ.setdefault("VLLM_SKINNY_QPN", "1")

from vllm.model_executor.kernels.linear.nvfp4 import marlin as mshim  # noqa

ext = mshim._get_skinny_ext()
E2M1 = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=dev)

with safe_open(CKPT, framework="pt") as f:
    names = [n for n in f.keys() if n.endswith("weight_packed")]
    picks = [n for n in names if any(
        s in n for s in (".0.", ".13.", ".30."))][:6] or names[:6]
    all_ok = True
    for name in picks:
        base = name[: -len("weight_packed")]
        codes = f.get_tensor(name).to(dev)
        scales = f.get_tensor(base + "weight_scale").to(dev)
        graw = f.get_tensor(base + "weight_global_scale").float().to(dev)
        n, k2 = codes.shape
        k = k2 * 2
        if n % 32 or k % 64:
            print(f"{base[:56]:<56} shape ineligible, skip")
            continue
        gscale = float(1.0 / graw.item())

        lo = (codes & 0x0F).long()
        hi = (codes >> 4).long()
        idx = torch.stack([lo, hi], dim=-1).view(n, k)
        mag = E2M1[idx & 0x7]
        sign = torch.where((idx & 0x8) > 0, -1.0, 1.0)
        eff = scales.float().repeat_interleave(16, dim=1) * gscale
        w_ref = (mag * sign * eff)

        qc, qs = mshim._qpn_prepack(codes.contiguous(),
                                    scales.view(torch.uint8).contiguous())
        assert qc is not None, "prepack refused an eligible shape"
        for m in (1, 2, 3, 5, 8, 11, 16):
            x = torch.randn(m, k, dtype=torch.float16, device=dev) * 0.5
            x[:, ::256] = 150.0
            fn = ext.gemm_qpn_simt if m <= 3 else ext.gemm_qpn
            y = fn(x, qc, qs, gscale, n)
            y_ref = x.float() @ w_ref.T
            rel = ((y.float() - y_ref).abs().max()
                   / y_ref.abs().max().clamp(min=1e-6)).item()
            ok = rel <= 1e-3
            all_ok &= ok
            print(f"{base[:56]:<56} N={n:<5} K={k:<5} M={m:<2} "
                  f"{'qpn1' if m <= 3 else 'qpn '} "
                  f"rel={rel:.2e} [{'ok' if ok else 'FAIL'}]")
        del codes, scales, w_ref, mag, sign, eff, idx, lo, hi, qc, qs
        torch.cuda.empty_cache()

print("\nQPN-REAL", "PASS" if all_ok else "FAIL")
