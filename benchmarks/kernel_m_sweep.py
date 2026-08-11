"""Full per-kernel M sweep, M=1..16: SIMT vs QPN m8n8k4 vs WMMA at every M.

Unlike kernel_bw_bench.py (which measures only the production-dispatch
winner per M), this measures ALL THREE kernels at each M so the
crossovers are visible — the dataflow-ownership graph. Effective
bandwidth = packed weight bytes / GEMM time, aggregate over the five
real Qwen3.6-27B TP4 per-rank shapes.

Safe to run beside a live server: weight packing runs on CPU (the GPU
packing path needs a ~1.4 GB intermediate); only the packed tensors
(~100 MB) and activations touch the GPU.

Output: CSV rows M,kernel,eff_GBs,pct_of_copy_ceiling to stdout.
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

MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])  # CPU
SHAPES = [(1536, 5120), (4352, 5120), (5120, 8704), (5120, 4096),
          (5120, 2048)]
GSCALE = 0.01
COPY_CEIL = 825.0  # measured V100 memcpy ceiling (GB/s)


def pack_row_major_cpu(w):
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
                       8, 10, 12, 14, 9, 11, 13, 15])  # CPU
LANE = torch.arange(32)
COL = ((LANE >> 2) & 3) * 8 + (LANE & 3) + ((LANE & 16) > 0).long() * 4


def qpn_prepack_cpu(codes, scales, n, k):
    T, G = n // 32, k // 16
    nib = torch.stack([codes & 0xF, codes >> 4], dim=-1).view(n, k)
    g = torch.arange(G)
    kidx = g.view(G, 1) * 16 + KORDER.view(1, 16)
    ncol = torch.arange(T).view(T, 1) * 32 + COL.view(1, 32)
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


shape_data = []
for k, n in SHAPES:
    w = torch.randn(n, k, dtype=torch.float16) * 0.02  # CPU
    codes, scales = pack_row_major_cpu(w)
    del w
    qc, qs = qpn_prepack_cpu(codes, scales, n, k)
    gbytes = codes.numel() + scales.numel()
    shape_data.append((k, n, codes.to(dev), scales.to(dev),
                       qc.to(dev), qs.to(dev), gbytes))

KERNELS = ("simt", "qpn", "wmma")


def eff_gbs(M, kernel):
    tot_bytes = tot_time = 0.0
    for k, n, codes, scales, qc, qs, gbytes in shape_data:
        x = torch.randn(M, k, dtype=torch.float16, device=dev) * 0.5
        x[:, ::256] = 150.0  # real-activation outliers
        if kernel == "simt":
            fn = lambda: ext.gemm_simt(x, codes, scales, GSCALE)
        elif kernel == "qpn":
            fn = lambda: ext.gemm_qpn(x, qc, qs, GSCALE, n)
        else:
            fn = lambda: ext.gemm_wmma(x, codes, scales, GSCALE)
        t = bench(fn)
        tot_bytes += gbytes
        tot_time += t
    return tot_bytes / (tot_time * 1e-3) / 1e9


OUT_CSV = os.environ.get("M_SWEEP_OUT", "")
rows = ["M,kernel,eff_GBs,pct_of_copy_ceiling"]
print(rows[0], flush=True)
for M in range(1, 17):
    for kern in KERNELS:
        try:
            gbs = eff_gbs(M, kern)
            row = f"{M},{kern},{gbs:.1f},{gbs/COPY_CEIL*100:.0f}%"
        except RuntimeError as exc:
            # A kernel that cannot serve this M is a real dispatch boundary
            # (the SIMT path is compiled for M in 1..8), not a harness failure.
            first = str(exc).strip().splitlines()[0][:80]
            row = f"{M},{kern},unsupported,{first}"
        rows.append(row)
        print(row, flush=True)
if OUT_CSV:
    with open(os.path.expanduser(OUT_CSV), "w") as fh:
        fh.write("\n".join(rows) + "\n")
    print(f"wrote {OUT_CSV}", flush=True)
print("M_SWEEP_DONE")
