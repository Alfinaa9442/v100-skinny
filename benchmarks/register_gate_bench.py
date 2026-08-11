"""Register Era microbench gate (2026-08-10).

Phase 1: load-rate probe (fragment-order linear streaming, no math)
         KILL IF < 550 GB/s on the sweep shapes.
Phase 2: cure rungs on the v1 m8n8k4 path — A (fragment-order
         pre-shuffle), B2/B4 (independent accumulators), stacked.
Phase 3: gate vs the fp16 incumbents (SIMT/WMMA) on the 5-shape sweep
         set, M in {1,4,5,8}; correctness vs fp32 reference with
         outlier activations, rel <= 1e-3.

Bench-only: GPU 0, no serving, no integration. Appends nothing to master
tables — writes register_gate_20260810.csv + stdout for the notes.
"""
import csv
import os
import sys

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
rg = load(name="register_gate",
          sources=[f"{HOME}/flatness-run/register_gate.cu"],
          extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70",
                             "--use_fast_math"], verbose=False)

MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
SHAPES = [(1536, 5120), (4352, 5120), (5120, 8704), (5120, 4096),
          (5120, 2048)]  # (K, N) Qwen3.6-27B TP4 per-rank
GSCALE = 0.01
ITERS = 200
KILL_BAR_GBS = 550.0
REL_GATE = 1e-3
OUT_CSV = f"{HOME}/flatness-run/register_gate_20260810.csv"


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


# Lane roles of the mma8 kernel (see skinny_kernels.cu fragment maps).
IDX_MAP = torch.tensor([(l & 3) + (4 if l & 16 else 0) for l in range(32)],
                       device=dev)
QP_MAP = torch.tensor([(l >> 2) & 3 for l in range(32)], device=dev)


def fragment_shuffle(codes, scales, n, k):
    """Permute codes/scales into fragment order: element [g, t, lane] is
    exactly what lane reads for 8-row group g, 64-k window t — so kernel
    loads become linear (256B codes + 32B scales per warp per window)."""
    ng, nw = n // 8, k // 64
    c8 = codes.view(ng, 8, k // 2)
    s8 = scales.view(ng, 8, k // 16)
    t = torch.arange(nw, device=dev)
    boff = (t.view(nw, 1, 1) * 32 + QP_MAP.view(1, 32, 1) * 8 +
            torch.arange(8, device=dev).view(1, 1, 8))          # [nw,32,8]
    rows = IDX_MAP.view(1, 32, 1).expand(nw, 32, 8)
    cshuf = c8[:, rows, boff]                                   # [ng,nw,32,8]
    soff = t.view(nw, 1) * 4 + QP_MAP.view(1, 32)               # [nw,32]
    srows = IDX_MAP.view(1, 32).expand(nw, 32)
    sshuf = s8[:, srows, soff]                                  # [ng,nw,32]
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
    return s.elapsed_time(e) / iters  # ms


rows_out = []


def emit(shape, m, rung, us, gbs, rel):
    rows_out.append({"shape_k": shape[0], "shape_n": shape[1], "m": m,
                     "rung": rung, "us": round(us, 2),
                     "gbs": round(gbs, 1) if gbs == gbs else "",
                     "rel_err": f"{rel:.2e}" if rel == rel else ""})


# ---------------------------------------------------------------- phase 1
print("=== phase 1: load-rate probe (kill bar 550 GB/s) ===", flush=True)
probe_gbs = {}
for k, n in SHAPES:
    w = torch.randn(n, k, dtype=torch.float16, device=dev) * 0.02
    codes, scales, _ = pack_row_major(w)
    cshuf, sshuf = fragment_shuffle(codes, scales, n, k)
    sink = torch.zeros(n // 8, dtype=torch.int32, device=dev)
    gbytes = cshuf.numel() + sshuf.numel()
    t_ms = bench(lambda: rg.probe_loadrate(cshuf, sshuf, sink, n, k))
    gbs = gbytes / (t_ms * 1e-3) / 1e9
    probe_gbs[(k, n)] = gbs
    emit((k, n), 0, "probe_loadrate", t_ms * 1000, gbs, float("nan"))
    print(f"  ({k},{n}): {t_ms*1000:7.1f} us  {gbs:6.0f} GB/s", flush=True)
    del w, codes, scales, cshuf, sshuf
    torch.cuda.empty_cache()

wsum = sum(k * n for k, n in SHAPES)
agg = wsum / sum((k * n) / probe_gbs[(k, n)] for k, n in SHAPES)
n_below = sum(1 for v in probe_gbs.values() if v < KILL_BAR_GBS)
print(f"  weighted aggregate: {agg:.0f} GB/s; shapes below bar: {n_below}/5",
      flush=True)
if agg < KILL_BAR_GBS and n_below >= 3:
    print("PROBE_KILL: geometry cannot feed itself")
    with open(OUT_CSV, "w", newline="") as f:
        wcsv = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
        wcsv.writeheader()
        wcsv.writerows(rows_out)
    sys.exit(2)
print("PROBE_PASS", flush=True)

# ------------------------------------------------------------- phase 2+3
print("=== phase 2/3: cure rungs + gate ===", flush=True)
RUNGS = [("v1", False, 1), ("A_shuf", True, 1), ("B2", False, 2),
         ("B4", False, 4), ("AB2", True, 2), ("AB4", True, 4)]
wins = {}   # rung -> {m: shapes won vs fp16 incumbent}
totals = {}  # (rung, m) -> total us

for k, n in SHAPES:
    w = torch.randn(n, k, dtype=torch.float16, device=dev) * 0.02
    codes, scales, ref = pack_row_major(w)
    del w
    cshuf, sshuf = fragment_shuffle(codes, scales, n, k)
    gbytes = codes.numel() + scales.numel()
    has_wmma = k % 256 == 0
    for m in (1, 4, 5, 8):
        # outlier activations (test_tail_shapes regime), fp32 reference
        x = torch.randn(m, k, dtype=torch.float16, device=dev) * 0.5
        x[:, ::256] = 150.0
        y_ref = x.float() @ ref.t().float()
        ymax = y_ref.abs().max().clamp(min=1e-6)

        def rel_of(y):
            return ((y.float() - y_ref).abs().max() / ymax).item()

        # incumbents
        t_s = bench(lambda: ext.gemm_simt(x, codes, scales, GSCALE))
        emit((k, n), m, "simt", t_s * 1000, gbytes / (t_s * 1e-3) / 1e9,
             rel_of(ext.gemm_simt(x, codes, scales, GSCALE)))
        t_fp16 = t_s
        if has_wmma:
            t_w = bench(lambda: ext.gemm_wmma(x, codes, scales, GSCALE))
            emit((k, n), m, "wmma", t_w * 1000, gbytes / (t_w * 1e-3) / 1e9,
                 rel_of(ext.gemm_wmma(x, codes, scales, GSCALE)))
            t_fp16 = min(t_s, t_w)
        totals[("fp16", m)] = totals.get(("fp16", m), 0.0) + t_fp16

        # layout sanity: SHUF must be bit-identical to v1 (same math order)
        y_v1 = rg.gemm_mma8_v2(x, codes, scales, GSCALE, n, False, 1)
        y_sh = rg.gemm_mma8_v2(x, cshuf, sshuf, GSCALE, n, True, 1)
        assert torch.equal(y_v1, y_sh), f"shuffle layout mismatch ({k},{n}) m={m}"

        for rung, shuf, nacc in RUNGS:
            carg, sarg = (cshuf, sshuf) if shuf else (codes, scales)
            y = rg.gemm_mma8_v2(x, carg, sarg, GSCALE, n, shuf, nacc)
            rel = rel_of(y)
            t = bench(lambda: rg.gemm_mma8_v2(x, carg, sarg, GSCALE, n,
                                              shuf, nacc))
            emit((k, n), m, rung, t * 1000, gbytes / (t * 1e-3) / 1e9, rel)
            totals[(rung, m)] = totals.get((rung, m), 0.0) + t
            if t < t_fp16:
                wins.setdefault(rung, {}).setdefault(m, []).append((k, n))
        print(f"  ({k},{n}) m={m} done", flush=True)
    del codes, scales, ref, cshuf, sshuf
    torch.cuda.empty_cache()

with open(OUT_CSV, "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    wcsv.writeheader()
    wcsv.writerows(rows_out)
print(f"CSV: {OUT_CSV}", flush=True)

print("\n=== 5-shape totals (ms) per M ===")
hdr = ["fp16"] + [r for r, _, _ in RUNGS]
print(f"{'M':>3} " + " ".join(f"{h:>8}" for h in hdr))
for m in (1, 4, 5, 8):
    print(f"{m:>3} " + " ".join(
        f"{totals.get((h, m), float('nan'))*1000:>8.1f}" for h in hdr))

print("\n=== gate ===")
for rung, _, _ in RUNGS:
    for m in (1, 4, 5, 8):
        wl = wins.get(rung, {}).get(m, [])
        if wl:
            print(f"  {rung} m={m}: wins {len(wl)}/5 -> {wl}")
best_stack = min(("AB2", "AB4"), key=lambda r: totals.get((r, 8), 9e9))
w8 = len(wins.get(best_stack, {}).get(8, []))
bad_rel = [r for r in rows_out
           if r["rung"] in ("AB2", "AB4") and r["rel_err"]
           and float(r["rel_err"]) > REL_GATE]
verdict = "GREEN" if (w8 >= 3 and not bad_rel) else "KILL"
print(f"stacked best at M=8: {best_stack} wins {w8}/5 shapes; "
      f"rel>{REL_GATE:g} rows: {len(bad_rel)}")
print(f"VERDICT_{verdict}")
