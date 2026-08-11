"""dp4a int8 SIMT vs fp16 SIMT vs WMMA at the compute-bound M band.
Timing includes the host-side activation quantization (honest: it runs
per GEMM call in serving too). Correctness vs fp32 reference with a
W-lossless/A8 tolerance.
"""
import torch
from torch.utils.cpp_extension import load

ext = load(
    name="skinny_nvfp4_v11",
    sources=["skinny_kernels.cu"],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70",
                       "--use_fast_math"],
    verbose=False,
)

dev = "cuda:0"
torch.manual_seed(0)
MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
GS = 0.01
SHAPES = [(1536, 5120), (4352, 5120), (5120, 8704), (5120, 4096),
          (5120, 2048)]
MS = [4, 5, 7, 8, 11, 16]
A8_TOL = 8e-2  # int8 activations: ~1-2% typical, tail tolerance


def pack(w):
    n, k = w.shape
    wf = w.float().view(n, k // 16, 16)
    amax = wf.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    sc = (amax / 6.0 / GS).clamp(min=2**-6)
    q = (wf / (sc * GS)).clamp(-6, 6)
    idx = (q.abs().unsqueeze(-1) - MAGS.view(1, 1, 1, 8)).abs().argmin(-1)
    sgn = q < 0
    e = sc.log2().round().clamp(-6, 8)
    scodes = ((e + 7).long() << 3).clamp(0, 255).to(torch.uint8).view(
        n, k // 16)
    eff = (2.0 ** e) * GS
    ref = (MAGS[idx] * torch.where(sgn, -1.0, 1.0) * eff).half().view(n, k)
    codes = (idx | (sgn.long() << 3)).view(n, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), scodes.contiguous(), ref


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


packed = {}
for k, n in SHAPES:
    w = torch.randn(n, k, dtype=torch.float16, device=dev) * 0.02
    packed[(k, n)] = pack(w)

print(f"{'M':>3} {'dp4a ms':>9} {'simt ms':>9} {'wmma ms':>9} "
      f"{'dp4a vs best':>13} {'max rel err':>12}")
for m in MS:
    xs = {k: torch.randn(m, k, dtype=torch.float16, device=dev) * 0.5
          for k, _ in SHAPES}
    t_d = t_s = t_w = 0.0
    worst = 0.0
    simt_ok = m in (1, 2, 3, 4, 5, 6, 7, 8)
    for k, n in SHAPES:
        codes, scales, ref = packed[(k, n)]
        x = xs[k]
        y = ext.gemm_dp4a(x, codes, scales, GS)
        y_ref = (x.float() @ ref.float().t()).half()
        rel = ((y.float() - y_ref.float()).abs() /
               y_ref.float().abs().clamp(min=1.0)).max().item()
        worst = max(worst, rel)
        assert rel < A8_TOL, f"dp4a m={m} ({k},{n}) rel={rel:.4f}"
        t_d += bench(lambda: ext.gemm_dp4a(x, codes, scales, GS))
        if simt_ok:
            t_s += bench(lambda: ext.gemm_simt(x, codes, scales, GS))
        t_w += bench(lambda: ext.gemm_wmma(x, codes, scales, GS))
    best = min([t for t in (t_s if simt_ok else 1e9, t_w)])
    print(f"{m:>3} {t_d:>9.3f} "
          f"{(t_s if simt_ok else float('nan')):>9.3f} {t_w:>9.3f} "
          f"{(best - t_d) / best * 100:>+12.1f}% {worst:>12.4f}")
