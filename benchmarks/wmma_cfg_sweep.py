"""Expanded WMMA-band sweep: tile configs x M x carveout, plus split-K.
Run with SWEEP_CARVEOUT=0/1 (two passes; the 96KB opt-in is sticky per
process). M rows include the single-stream verify shapes (11, 16).
"""
import os
import torch
from torch.utils.cpp_extension import load

CARVE = os.environ.get("SWEEP_CARVEOUT", "0") == "1"

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
CFGS = {0: "(4,1,256)", 1: "(4,2,128)", 2: "(2,4,128)", 3: "(8,1,128)",
        4: "(8,2,128)", 5: "(4,4,128)", 6: "(2,1,256)", 7: "(2,2,128)",
        8: "(4,1,512)", 9: "(2,4,256)"}
CFG_MAX_M = {0: 16, 1: 32, 2: 64, 3: 16, 4: 32, 5: 64, 6: 16, 7: 32,
             8: 16, 9: 64}
CFG_NT = {0: 64, 1: 64, 2: 32, 3: 128, 4: 128, 5: 64, 6: 32, 7: 32,
          8: 64, 9: 32}
CFG_KC = {0: 256, 1: 128, 2: 128, 3: 128, 4: 128, 5: 128, 6: 256,
          7: 128, 8: 512, 9: 256}
CFG_NEEDS_CARVE = {8, 9}
MS = [8, 11, 16, 32, 64]
REL_TOL = 3e-2
GS_EFF = GS * 16384.0  # dequant8_tm rebias for the split-K fp32 path


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

print(f"carveout={'ON' if CARVE else 'off'}")
totals = {}
for m in MS:
    xs = {k: torch.randn(m, k, dtype=torch.float16, device=dev) * 0.5
          for k, _ in SHAPES}
    for cfg in CFGS:
        if m > CFG_MAX_M[cfg]:
            continue
        if cfg in CFG_NEEDS_CARVE and not CARVE:
            continue
        tot, ok = 0.0, True
        for k, n in SHAPES:
            if n % CFG_NT[cfg] != 0 or k % CFG_KC[cfg] != 0:
                ok = False
                break
            codes, scales, ref = packed[(k, n)]
            x = xs[k]
            y = ext.gemm_wmma_cfg(x, codes, scales, GS, cfg, CARVE)
            y_ref = (x.float() @ ref.float().t()).half()
            rel = ((y.float() - y_ref.float()).abs() /
                   y_ref.float().abs().clamp(min=1.0)).max().item()
            if rel >= REL_TOL:
                print(f"FAIL cfg{cfg} m={m} ({k},{n}) rel={rel:.4f}")
                ok = False
                break
            tot += bench(
                lambda: ext.gemm_wmma_cfg(x, codes, scales, GS, cfg, CARVE))
        if ok:
            totals[(m, cfg)] = tot

print(f"\ntile sweep (ms per 5-shape layer set)")
print(f"{'M':>4} " + " ".join(f"{CFGS[c]:>10}" for c in CFGS))
for m in MS:
    row = [f"{totals[(m, c)]:>10.3f}" if (m, c) in totals else f"{'—':>10}"
           for c in CFGS]
    print(f"{m:>4} " + " ".join(row))

print("\nsplit-K (ms per 5-shape layer set; incl. fp32->half convert)")
KS_CFGS = [0, 1, 2, 6, 7]
for m in [11, 16, 32, 64]:
    xs = {k: torch.randn(m, k, dtype=torch.float16, device=dev) * 0.5
          for k, _ in SHAPES}
    line = [f"M={m:>2}"]
    for cfg in KS_CFGS:
        if m > CFG_MAX_M[cfg]:
            continue
        for S in (2, 4):
            tot, ok = 0.0, True
            for k, n in SHAPES:
                if (n % CFG_NT[cfg] != 0 or k % CFG_KC[cfg] != 0
                        or (k // S) % CFG_KC[cfg] != 0 or (k // S) * S != k):
                    ok = False
                    break
                codes, scales, ref = packed[(k, n)]
                x = xs[k]

                def run():
                    yp = ext.gemm_wmma_splitk(x, codes, scales, cfg, S, CARVE)
                    return (yp * GS_EFF).half()

                y = run()
                y_ref = (x.float() @ ref.float().t()).half()
                rel = ((y.float() - y_ref.float()).abs() /
                       y_ref.float().abs().clamp(min=1.0)).max().item()
                if rel >= REL_TOL:
                    print(f"FAIL ks cfg{cfg} S={S} m={m} ({k},{n}) "
                          f"rel={rel:.4f}")
                    ok = False
                    break
                tot += bench(run)
            if ok:
                line.append(f"{CFGS[cfg]}xS{S}={tot:.3f}")
    print("  " + "  ".join(line))
