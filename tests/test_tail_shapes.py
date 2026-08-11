#!/usr/bin/env python3
"""Validate the generalized (K % 128) SIMT path on per-rank tail shapes,
with activation outliers, and compare M=1 latency vs the WMMA path."""
import os

import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
from torch.utils.cpp_extension import load  # noqa: E402
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.join(
    _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))),
    "benchmarks"))
from _paths import kernel_src  # noqa: E402

dev = torch.device("cuda:0")
ext = load(
    name="skinny_nvfp4_v9",
    sources=[kernel_src()],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                       "-gencode=arch=compute_70,code=sm_70"],
    verbose=False,
)

MAGS = torch.tensor([0, .5, 1, 1.5, 2, 3, 4, 6], device=dev)
MIDS = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev)


def pack(w):
    n, k = w.shape
    wf = w.float().view(n, k // 16, 16)
    scale = wf.abs().amax(-1) / 6.0
    g = float(scale.max().item() / 448.0)
    q8 = (scale / g).to(torch.float8_e4m3fn)
    eff = (q8.float() * g).clamp(min=1e-12).unsqueeze(-1)
    idx = torch.bucketize((wf / eff).abs(), MIDS)
    sgn = (wf / eff) < 0
    ref = (MAGS[idx] * torch.where(sgn, -1., 1.) * eff).half().view(n, k)
    codes = (idx | (sgn.long() << 3)).view(n, k).to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    return packed, q8.view(torch.uint8).contiguous(), g, ref


def t_ms(fn, *a, iters=50):
    for _ in range(10):
        fn(*a)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn(*a)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


all_ok = True
for k, n in [(1536, 5120), (4352, 5120), (5120, 8704), (2176, 5120)]:
    w = torch.randn(n, k, dtype=torch.float16, device=dev)
    codes, scales, g, ref = pack(w)
    del w
    for m in (1, 2, 8):
        x = torch.randn(m, k, dtype=torch.float16, device=dev) * 0.5
        x[:, ::256] = 150.0
        y = ext.gemm_simt(x, codes, scales, g)
        y_ref = x.float() @ ref.t().float()
        rel = ((y.float() - y_ref).abs().max()
               / y_ref.abs().max().clamp(min=1e-6)).item()
        bad = (~torch.isfinite(y.float())).sum().item()
        ok = rel < 1e-2 and bad == 0
        all_ok &= ok
        print(f"K={k:<5} N={n:<5} M={m}: rel={rel:.2e} inf={bad} "
              f"[{'ok' if ok else 'FAIL'}]")
    x1 = torch.randn(1, k, dtype=torch.float16, device=dev)
    ts = t_ms(ext.gemm_simt, x1, codes, scales, g)
    tw = t_ms(ext.gemm_wmma, x1, codes, scales, g)
    gb = (codes.numel() + scales.numel()) / 1e9
    print(f"K={k:<5} N={n:<5} M=1: simt {ts*1000:.0f}us "
          f"({gb/ts*1000:.0f} GB/s) vs wmma {tw*1000:.0f}us "
          f"({gb/tw*1000:.0f} GB/s) -> {tw/ts:.2f}x")
    del codes, scales, ref
    torch.cuda.empty_cache()

print("\nTAIL-SHAPES", "PASS" if all_ok else "FAIL")
