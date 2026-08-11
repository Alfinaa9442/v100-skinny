"""Kernel-bandwidth rebench (current kernel + production dispatch).

Effective memory bandwidth (packed weight bytes / GEMM time) vs M, using
the SAME kernel each M that production dispatches: SIMT (M<=3), QPN
m8n8k4 (M 4-16), WMMA (M 17-64). Marlin NVFP4 as the stock-fallback
reference. Shows how far up M the kernel holds near the memory ceiling —
the stale Aug-5 SIMT-only curve went compute-bound by M=8; QPN pushes
that band back to the memory limit.

Aggregate over the 5 real Qwen3.6-27B TP4 per-rank shapes. GPU 0.
Writes results/kernel_bandwidth_YYYYMMDD.csv-style rows to stdout.
"""
import os
import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
from torch.utils.cpp_extension import load  # noqa: E402

dev = "cuda:0"
torch.manual_seed(0)
HOME = os.path.expanduser("~")
ext = load(name="skinny_nvfp4_v11",
           sources=[f"{HOME}/flatness-run/skinny_kernels.cu"],
           extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70",
                              "--use_fast_math"], verbose=False)

MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
SHAPES = [(1536, 5120), (4352, 5120), (5120, 8704), (5120, 4096),
          (5120, 2048)]
GSCALE = 0.01
COPY_CEIL = 825.0  # measured V100 memcpy ceiling (GB/s)


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


KORDER = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7,
                       8, 10, 12, 14, 9, 11, 13, 15], device=dev)
LANE = torch.arange(32, device=dev)
COL = ((LANE >> 2) & 3) * 8 + (LANE & 3) + ((LANE & 16) > 0).long() * 4


def qpn_prepack(codes, scales, n, k):
    T, G = n // 32, k // 16
    nib = torch.stack([codes & 0xF, codes >> 4], dim=-1).view(n, k)
    g = torch.arange(G, device=dev)
    kidx = g.view(G, 1) * 16 + KORDER.view(1, 16)
    ncol = torch.arange(T, device=dev).view(T, 1) * 32 + COL.view(1, 32)
    nb = nib[ncol.view(T, 1, 32, 1).expand(T, G, 32, 16),
             kidx.view(1, G, 1, 16).expand(T, G, 32, 16)]
    qc = (nb[..., 0::2] | (nb[..., 1::2] << 4)).to(torch.uint8)
    qs = scales[ncol.view(T, 1, 32).expand(T, G, 32),
                g.view(1, G, 1).expand(T, G, 32)]
    return qc.contiguous().view(-1), qs.contiguous().view(-1)


def bench(fn, it=200):
    for _ in range(20):
        fn()
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(it):
        fn()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / it  # ms


# Prepack all shapes once.
shape_data = []
for k, n in SHAPES:
    w = torch.randn(n, k, dtype=torch.float16, device=dev) * 0.02
    codes, scales = pack_row_major(w)
    del w
    qc, qs = qpn_prepack(codes, scales, n, k)
    gbytes = codes.numel() + scales.numel()
    shape_data.append((k, n, codes, scales, qc, qs, gbytes))


def eff_gbs(M, kernel):
    """Aggregate effective GB/s over the 5 shapes for the given kernel."""
    tot_bytes = tot_time = 0.0
    for k, n, codes, scales, qc, qs, gbytes in shape_data:
        x = torch.randn(M, k, dtype=torch.float16, device=dev) * 0.5
        x[:, ::256] = 150.0
        if kernel == "simt":
            fn = lambda: ext.gemm_simt(x, codes, scales, GSCALE)
        elif kernel == "qpn":
            fn = lambda: ext.gemm_qpn(x, qc, qs, GSCALE, n)
        elif kernel == "wmma":
            fn = lambda: ext.gemm_wmma(x, codes, scales, GSCALE)
        elif kernel == "marlin":
            fn = None  # marlin measured separately below
        t = bench(fn)
        tot_bytes += gbytes
        tot_time += t
    return tot_bytes / (tot_time * 1e-3) / 1e9


# Production dispatch: simt M<=3, qpn 4-16, wmma 17-64.
def prod_kernel(M):
    if M <= 3:
        return "simt"
    if M <= 16:
        return "qpn"
    return "wmma"


def run_aggregate():
    print("M,prod_kernel,eff_GBs,pct_of_copy_ceiling")
    for M in (1, 2, 3, 4, 5, 8, 11, 16, 24, 32, 64):
        kern = prod_kernel(M)
        # WMMA needs K%128; all shapes qualify. SIMT needs K%128 too.
        gbs = eff_gbs(M, kern)
        print(f"{M},{kern},{gbs:.1f},{gbs/COPY_CEIL*100:.0f}%", flush=True)
    print("KERNEL_BW_DONE")


def run_per_shape():
    """QPN band, per shape, production kernel vs race build on IDENTICAL
    inputs. Per-shape delta = integration loss; cross-shape spread =
    shape mix. Splits the 647-race vs 431-production discrepancy."""
    qe = load(name="qpn_race_v2",
              sources=[f"{HOME}/flatness-run/qpn_race.cu"],
              extra_cuda_cflags=["-O3",
                                 "-gencode=arch=compute_70,code=sm_70",
                                 "--use_fast_math"], verbose=False)
    print("M,K,N,prod_GBs,race_GBs,race_over_prod")
    for M in (4, 8, 11, 16):
        for k, n, codes, scales, qc, qs, gbytes in shape_data:
            x = torch.randn(M, k, dtype=torch.float16, device=dev) * 0.5
            x[:, ::256] = 150.0
            out = torch.empty(M, n, dtype=torch.float16, device=dev)
            t_prod = bench(lambda: ext.gemm_qpn(x, qc, qs, GSCALE, n))
            t_race = bench(
                lambda: qe.gemm_qpn(x, qc, qs, GSCALE, n, out, 0, False))
            g_prod = gbytes / (t_prod * 1e-3) / 1e9
            g_race = gbytes / (t_race * 1e-3) / 1e9
            print(f"{M},{k},{n},{g_prod:.1f},{g_race:.1f},"
                  f"{g_race/g_prod:.2f}", flush=True)
    print("PER_SHAPE_DONE")


import sys
if len(sys.argv) > 1 and sys.argv[1] == "pershape":
    run_per_shape()
else:
    run_aggregate()
