"""Twin-race bench: batch band M in {16, 32, 64} on the 5 real shapes.

Entrants per (shape, M):
  wmma      — production incumbent (skinny_kernels.cu dispatch)
  A_d2      — rung A, code-register depth 2, incumbent barriers
  A_d2b     — rung A, depth 2 + double-buffered smem (1 barrier/chunk)
  A_d1b     — rung A control: dbuf alone (depth 1)
  B_ring    — rung B, m8n8k4 4-deep register ring, shuffled weights

Correctness ladder before any timing: rel <= 1e-3 vs fp32 reference with
outlier activations (x[:, ::256] = 150). GPU 0, serving down. Appends
nothing to master tables — writes twin_race_20260810.csv.
Registers/thread land in the build log (-Xptxas -v); occ_report() gives
blocks/SM + warps/SM per kernel.
"""
import csv
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
tr = load(name="twin_race",
          sources=[f"{HOME}/flatness-run/twin_race.cu"],
          extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70",
                             "--use_fast_math", "-Xptxas", "-v"],
          verbose=True)

MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
SHAPES = [(1536, 5120), (4352, 5120), (5120, 8704), (5120, 4096),
          (5120, 2048)]
MS = (16, 32, 64)
GSCALE = 0.01
ITERS = 200
REL_GATE = 1e-3
OUT_CSV = f"{HOME}/flatness-run/twin_race_20260810.csv"


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


IDX_MAP = torch.tensor([(l & 3) + (4 if l & 16 else 0) for l in range(32)],
                       device=dev)
QP_MAP = torch.tensor([(l >> 2) & 3 for l in range(32)], device=dev)


def fragment_shuffle(codes, scales, n, k):
    """Bit-verified fragment-order permutation (register gate)."""
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


print("=== occupancy report (blocks/SM, warps/SM) ===")
for name, blocks, warps in tr.occ_report():
    print(f"  {name}: {blocks} blocks/SM, {warps} warps/SM")

rows_out = []
totals = {}
wins = {}
fails = []

for k, n in SHAPES:
    w = torch.randn(n, k, dtype=torch.float16, device=dev) * 0.02
    codes, scales, ref = pack_row_major(w)
    del w
    cshuf, sshuf = fragment_shuffle(codes, scales, n, k)
    gbytes = codes.numel() + scales.numel()
    for m in MS:
        x = torch.randn(m, k, dtype=torch.float16, device=dev) * 0.5
        x[:, ::256] = 150.0
        y_ref = x.float() @ ref.t().float()
        ymax = y_ref.abs().max().clamp(min=1e-6)

        entrants = [
            ("wmma", lambda: ext.gemm_wmma(x, codes, scales, GSCALE)),
            ("A_d2", lambda: tr.gemm_wmma_pipe(x, codes, scales, GSCALE,
                                               2, False)),
            ("A_d2b", lambda: tr.gemm_wmma_pipe(x, codes, scales, GSCALE,
                                                2, True)),
            ("A_d1b", lambda: tr.gemm_wmma_pipe(x, codes, scales, GSCALE,
                                                1, True)),
            ("B_ring", lambda: tr.gemm_mma8_ring(x, cshuf, sshuf, GSCALE, n)),
        ]
        # correctness ladder BEFORE any timing
        times = {}
        for tag, fn in entrants:
            rel = ((fn().float() - y_ref).abs().max() / ymax).item()
            ok = rel <= REL_GATE
            if not ok:
                fails.append((tag, k, n, m, rel))
            t = bench(fn)
            times[tag] = t
            rows_out.append({"shape_k": k, "shape_n": n, "m": m, "kernel": tag,
                             "us": round(t * 1000, 2),
                             "gbs": round(gbytes / (t * 1e-3) / 1e9, 1),
                             "rel_err": f"{rel:.2e}",
                             "ok": int(ok)})
            totals[(tag, m)] = totals.get((tag, m), 0.0) + t
        best = min(times, key=times.get)
        wins.setdefault(best, {}).setdefault(m, []).append((k, n))
        line = " ".join(f"{tag}={times[tag]*1000:.1f}" for tag, _ in entrants)
        print(f"({k},{n}) m={m}: {line}  best={best}", flush=True)
    del codes, scales, ref, cshuf, sshuf
    torch.cuda.empty_cache()

with open(OUT_CSV, "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    wcsv.writeheader()
    wcsv.writerows(rows_out)
print(f"CSV: {OUT_CSV}")

print("\n=== 5-shape totals (us) per M ===")
tags = ["wmma", "A_d2", "A_d2b", "A_d1b", "B_ring"]
print(f"{'M':>3} " + " ".join(f"{t:>8}" for t in tags))
for m in MS:
    print(f"{m:>3} " + " ".join(
        f"{totals.get((t, m), float('nan'))*1000:>8.1f}" for t in tags))

print("\n=== race verdict ===")
for tag in tags:
    for m in MS:
        wl = wins.get(tag, {}).get(m, [])
        if wl:
            print(f"  {tag} m={m}: fastest on {len(wl)}/5 -> {wl}")
if fails:
    print(f"CORRECTNESS_FAILS: {fails}")
else:
    print("CORRECTNESS_ALL_PASS")
print("TWIN_RACE_DONE")
