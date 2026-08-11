"""One-shot profile target for the register-gate KILL analysis.

Runs v1 (SHUF=0,NACC=1) then AB2 (SHUF=1,NACC=2) on the grid-healthy
(K=1536, N=5120) shape at M=8 a few times each, for ncu/nsys to attach.
Bench-only; tiny footprint so it can coexist with live serving.
"""
import os

import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
from torch.utils.cpp_extension import load  # noqa: E402
import os as _os
import sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from _paths import kernel_src, out_csv, fixtures_dir  # noqa: E402

dev = "cuda:0"
torch.manual_seed(0)
HOME = os.path.expanduser("~")
rg = load(name="register_gate",
          sources=[kernel_src("register_gate.cu")],
          extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70",
                             "--use_fast_math"], verbose=False)

MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
GSCALE = 0.01
K, N = 1536, 5120


def pack_row_major(w):
    n, k = w.shape
    wf = w.float().view(n, k // 16, 16)
    amax = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    sc = (amax / 6.0 / GSCALE).clamp(min=2**-6)
    q = (wf / (sc * GSCALE)).clamp(-6, 6)
    idx = (q.abs().unsqueeze(-1) - MAGS.view(1, 1, 1, 8)).abs().argmin(-1)
    sgn = q < 0
    e = sc.log2().round().clamp(-6, 8)
    scodes = ((e + 7).long() << 3).clamp(0, 255).to(torch.uint8).view(n, k // 16)
    codes = (idx | (sgn.long() << 3)).view(n, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), scodes.contiguous()


IDX_MAP = torch.tensor([(l & 3) + (4 if l & 16 else 0) for l in range(32)],
                       device=dev)
QP_MAP = torch.tensor([(l >> 2) & 3 for l in range(32)], device=dev)


def fragment_shuffle(codes, scales, n, k):
    ng, nw = n // 8, k // 64
    c8 = codes.view(ng, 8, k // 2)
    s8 = scales.view(ng, 8, k // 16)
    t = torch.arange(nw, device=dev)
    boff = (t.view(nw, 1, 1) * 32 + QP_MAP.view(1, 32, 1) * 8 +
            torch.arange(8, device=dev).view(1, 1, 8))
    rows = IDX_MAP.view(1, 32, 1).expand(nw, 32, 8)
    cshuf = c8[:, rows, boff]
    soff = t.view(nw, 1) * 4 + QP_MAP.view(1, 32)
    srows = IDX_MAP.view(1, 32).expand(nw, 32)
    sshuf = s8[:, srows, soff]
    return cshuf.contiguous().view(-1), sshuf.contiguous().view(-1)


w = torch.randn(N, K, dtype=torch.float16, device=dev) * 0.02
codes, scales = pack_row_major(w)
del w
cshuf, sshuf = fragment_shuffle(codes, scales, N, K)
x = torch.randn(8, K, dtype=torch.float16, device=dev) * 0.5
x[:, ::256] = 150.0

for _ in range(3):
    rg.gemm_mma8_v2(x, codes, scales, GSCALE, N, False, 1)   # v1
for _ in range(3):
    rg.gemm_mma8_v2(x, cshuf, sshuf, GSCALE, N, True, 2)     # AB2
torch.cuda.synchronize()
print("PROF_TARGET_DONE")
