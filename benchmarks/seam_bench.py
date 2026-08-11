"""Item 2 (verify seam): SIMT-direct vs padded-WMMA at M in {4,5,8}
on the real Qwen3.6-27B TP4 per-rank verify shapes.

The MTP4 verify pass runs M=5 (4 draft + 1 bonus) through the WMMA
path, which pads rows to 16. If SIMT-direct at M=5 wins on the real
shapes, marlin_patched.py routes M<=8 to SIMT.
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

# Qwen3.6-27B TP4 per-rank verify shapes (K, N): attn o_proj, MLP down,
# MLP gate_up, GDN in_proj, attn qkv.
SHAPES = [(1536, 5120), (4352, 5120), (5120, 8704), (5120, 4096),
          (5120, 2048)]
GSCALE = 0.01
ITERS = 200
REL_TOL = 3e-2


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
    eff = (2.0 ** e) * GSCALE
    ref = (MAGS[idx] * torch.where(sgn, -1.0, 1.0) * eff).half().view(n, k)
    codes = (idx | (sgn.long() << 3)).view(n, k)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), scodes.contiguous(), ref


def bench(fn, iters=ITERS):
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


print(f"{'shape (K,N)':>14} {'M':>2} {'simt us':>8} {'wmma us':>8} "
      f"{'mma8 us':>8} {'mma8 GB/s':>9} {'winner':>6}")
totals = {}
for k, n in SHAPES:
    w = torch.randn(n, k, dtype=torch.float16, device=dev) * 0.02
    codes, scales, ref = pack_row_major(w)
    gbytes = codes.numel() + scales.numel()
    for m in (1, 4, 5, 8):
        x = torch.randn(m, k, dtype=torch.float16, device=dev) * 0.5
        y_ref = (x.float() @ ref.float().t()).half()
        checks = [("simt", ext.gemm_simt(x, codes, scales, GSCALE)),
                  ("mma8", ext.gemm_mma8(x, codes, scales, GSCALE))]
        has_wmma = k % 256 == 0
        if has_wmma:
            checks.append(("wmma", ext.gemm_wmma(x, codes, scales, GSCALE)))
        for tag, y in checks:
            rel = ((y.float() - y_ref.float()).abs() /
                   y_ref.float().abs().clamp(min=1.0)).max().item()
            assert rel < REL_TOL, f"{tag} m={m} k={k} n={n} rel={rel:.4f}"
        t_s = bench(lambda: ext.gemm_simt(x, codes, scales, GSCALE))
        t_m = bench(lambda: ext.gemm_mma8(x, codes, scales, GSCALE))
        t_w = (bench(lambda: ext.gemm_wmma(x, codes, scales, GSCALE))
               if has_wmma else float("nan"))
        bw = gbytes / (t_m * 1e-3) / 1e9
        times = {"simt": t_s, "mma8": t_m}
        if has_wmma:
            times["wmma"] = t_w
        win = min(times, key=times.get)
        totals.setdefault(m, [0.0, 0.0, 0.0])
        totals[m][0] += t_s
        totals[m][1] += t_w if has_wmma else min(t_s, t_m)
        totals[m][2] += t_m
        print(f"{f'({k},{n})':>14} {m:>2} {t_s*1000:>8.1f} {t_w*1000:>8.1f} "
              f"{t_m*1000:>8.1f} {bw:>9.0f} {win:>6}")

print("\nper-layer totals (5 shapes; GDN layer = shapes 1-4):")
for m, (ts, tw, tm) in sorted(totals.items()):
    print(f"  M={m}: simt {ts*1000:.1f}  wmma {tw*1000:.1f}  "
          f"mma8 {tm*1000:.1f} us  -> x64: simt {ts*64:.2f}  "
          f"wmma {tw*64:.2f}  mma8 {tm*64:.2f} ms  "
          f"mma8-vs-wmma {(tw-tm)*64:+.2f} ms/step")
print("CORRECTNESS PASS")
