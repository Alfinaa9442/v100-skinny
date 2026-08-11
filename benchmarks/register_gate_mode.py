"""KILL-analysis mode triple (ncu is admin-locked; causal isolation
instead): time v1 and AB2 with pipeline stages surgically removed.
  mode 0 = full   1 = no dequant/pack (loads+mma)   2 = no mma (loads+dequant)
If mode-1 collapses toward the load-rate probe while mode-2 stays near
full, the dequant+fragment-pack chain is the serial bottleneck.
Small footprint; safe alongside idle serving. GPU 0.
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
SHAPES = [(1536, 5120), (4352, 5120)]  # grid-healthy pair, small footprint


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


def bench(fn, iters=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


print(f"{'shape':>14} {'config':>6} {'mode':>16} {'us':>8} {'GB/s':>6}")
for k, n in SHAPES:
    w = torch.randn(n, k, dtype=torch.float16, device=dev) * 0.02
    codes, scales = pack_row_major(w)
    del w
    cshuf, sshuf = fragment_shuffle(codes, scales, n, k)
    gbytes = codes.numel() + scales.numel()
    x = torch.randn(8, k, dtype=torch.float16, device=dev) * 0.5
    x[:, ::256] = 150.0
    for cfg, carg, sarg, shuf, nacc in (
            ("v1", codes, scales, False, 1),
            ("AB2", cshuf, sshuf, True, 2)):
        for mode, label in ((0, "full"), (1, "no-dequant"), (2, "no-mma")):
            t = bench(lambda: rg.gemm_mma8_v2(x, carg, sarg, GSCALE, n,
                                              shuf, nacc, mode))
            gbs = gbytes / (t * 1e-3) / 1e9
            print(f"{f'({k},{n})':>14} {cfg:>6} {label:>16} {t*1000:>8.1f} "
                  f"{gbs:>6.0f}", flush=True)
    del codes, scales, cshuf, sshuf
    torch.cuda.empty_cache()
print("MODE_TRIPLE_DONE")
