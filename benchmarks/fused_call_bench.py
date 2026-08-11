"""Subgraph replay harness: the lm_head top1 call in isolation.

Replays the drafter's per-iteration sampling subgraph (real lm_head
shard geometry, native NVFP4 codes) without booting a server. Measures
per-call WALL latency (the serving-relevant number: kernels + host glue
+ sync) for:
  A. unfused reference: gemm_simt -> torch .max(dim=-1)   (old path)
  B. fused, shipped glue: gemm_simt_argmax + mx/blk/idx 3-op chain
  C. fused, tight glue:   gemm_simt_argmax + torch.max single-op select
Run in a GPU gap (needs ~300 MB). ~60 s total.
"""
import os
import time

import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
from torch.utils.cpp_extension import load  # noqa: E402

dev = "cuda:0"
torch.manual_seed(3)
HOME = os.path.expanduser("~")
ext = load(name="skinny_nvfp4_v11",
           sources=[f"{HOME}/flatness-run/skinny_kernels.cu"],
           extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                              "-gencode=arch=compute_70,code=sm_70"],
           verbose=False)

N, K = 62080, 5120
VOCAB_START = 124160  # rank-2-style offset, exercised in the math
codes = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
scales = torch.randint(48, 90, (N, K // 16), dtype=torch.uint8, device=dev)
GS = 1e-2
xs = [torch.randn(1, K, dtype=torch.float16, device=dev) * 0.05
      for _ in range(64)]


def timed(fn, iters=400, warmup=50):
    for i in range(warmup):
        fn(xs[i % 64])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for i in range(iters):
        fn(xs[i % 64])
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3


def path_a(x):
    y = ext.gemm_simt(x, codes, scales, GS)
    v, i = y.max(dim=-1)
    return v, i + VOCAB_START


def path_b(x):  # shipped glue
    bv, bi = ext.gemm_simt_argmax(x, codes, scales, GS)
    mx = bv.max()
    blk = (bv == mx).int().argmax()
    idx = bi[blk].to(torch.long) + VOCAB_START
    return mx.view(1), idx.view(1)


def path_c(x):  # tight glue
    bv, bi = ext.gemm_simt_argmax(x, codes, scales, GS)
    mx, blk = torch.max(bv, 0)
    idx = bi[blk].to(torch.long) + VOCAB_START
    return mx.view(1), idx.view(1)


a = timed(path_a)
b = timed(path_b)
c = timed(path_c)
print(f"A unfused (GEMM + max)      : {a:.3f} ms/call")
print(f"B fused, shipped glue       : {b:.3f} ms/call  ({b-a:+.3f} vs A)")
print(f"C fused, tight glue         : {c:.3f} ms/call  ({c-a:+.3f} vs A)")
print(f"per-round delta at 8 calls  : B {8*(b-a):+.2f} ms   C {8*(c-a):+.2f} ms")
# sanity: all three agree on a fresh input
x = torch.randn(1, K, dtype=torch.float16, device=dev) * 0.05
ia = int(path_a(x)[1].item()); ib = int(path_b(x)[1].item()); ic = int(path_c(x)[1].item())
print("index agreement:", "PASS" if ia == ib == ic else f"FAIL {ia}/{ib}/{ic}")
print("FUSED_BENCH_DONE")
