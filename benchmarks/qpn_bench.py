"""QP-N m8n8k4 race — the one resurrection experiment.

Entrants on the 5 real shapes:
  M=5, 8   : simt | wmma (padded 16-row) | qpn
  M=11     : wmma | qpn(8) + simt(3) hybrid   (the chat-serving verify seam)
  M=16     : wmma | qpn x2 (two 8-row A tiles)

Correctness (rel <= 1e-3, outlier activations) before any timing.
GPU 0, serving down. New CSV qpn_race_20260810.csv; master tables
untouched.
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
qe = load(name="qpn_race",
          sources=[f"{HOME}/flatness-run/qpn_race.cu"],
          extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70",
                             "--use_fast_math", "-Xptxas", "-v"],
          verbose=True)

MAGS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], device=dev)
SHAPES = [(1536, 5120), (4352, 5120), (5120, 8704), (5120, 4096),
          (5120, 2048)]
GSCALE = 0.01
ITERS = 200
REL_GATE = 1e-3
OUT_CSV = f"{HOME}/flatness-run/qpn_race_20260810.csv"


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


# k-order that makes dequant8_tm's (j, j+4) interleave come out as the
# adjacent-k B-fragment pairs (k2j, k2j+1) with zero pack instructions.
KORDER = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7,
                       8, 10, 12, 14, 9, 11, 13, 15], device=dev)
LANE = torch.arange(32, device=dev)
COL_IN_TILE = ((LANE >> 2) & 3) * 8 + (LANE & 3) + ((LANE & 16) > 0).long() * 4


def qpn_prepack(codes, scales, n, k):
    """[ntile][group][lane] layout, B-fragment lane order, nibbles
    pre-interleaved for direct decode."""
    T, G = n // 32, k // 16
    lo = (codes & 0xF)
    hi = (codes >> 4)
    nib = torch.stack([lo, hi], dim=-1).view(n, k)          # nibble per k
    g = torch.arange(G, device=dev)
    kidx = g.view(G, 1) * 16 + KORDER.view(1, 16)           # [G,16]
    ncol = torch.arange(T, device=dev).view(T, 1) * 32 + COL_IN_TILE.view(1, 32)
    nb = nib[ncol.view(T, 1, 32, 1).expand(T, G, 32, 16),
             kidx.view(1, G, 1, 16).expand(T, G, 32, 16)]   # [T,G,32,16]
    packed = (nb[..., 0::2] | (nb[..., 1::2] << 4)).to(torch.uint8)
    sc = scales[ncol.view(T, 1, 32).expand(T, G, 32),
                g.view(1, G, 1).expand(T, G, 32)]           # [T,G,32]
    return packed.contiguous().view(-1), sc.contiguous().view(-1)


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


print(f"qpn occupancy: {qe.qpn_blocks_per_sm()} blocks/SM "
      f"(x4 warps; grid is N/32)")

rows_out = []
totals = {}
fails = []

for k, n in SHAPES:
    w = torch.randn(n, k, dtype=torch.float16, device=dev) * 0.02
    codes, scales, ref = pack_row_major(w)
    del w
    bcodes, bscales = qpn_prepack(codes, scales, n, k)
    gbytes = codes.numel() + scales.numel()

    for m in (5, 8, 11, 16):
        x = torch.randn(m, k, dtype=torch.float16, device=dev) * 0.5
        x[:, ::256] = 150.0
        y_ref = x.float() @ ref.t().float()
        ymax = y_ref.abs().max().clamp(min=1e-6)

        def qpn_call():
            out = torch.empty(m, n, dtype=torch.float16, device=dev)
            if m <= 8:
                qe.gemm_qpn(x, bcodes, bscales, GSCALE, n, out, 0)
            elif m == 11:  # hybrid: 8 rows qpn + 3 rows simt
                qe.gemm_qpn(x[:8], bcodes, bscales, GSCALE, n, out, 0)
                out[8:] = ext.gemm_simt(x[8:].contiguous(), codes, scales,
                                        GSCALE)
            else:  # 16: two 8-row A tiles
                qe.gemm_qpn(x[:8], bcodes, bscales, GSCALE, n, out, 0)
                qe.gemm_qpn(x[8:].contiguous(), bcodes, bscales, GSCALE, n,
                            out, 8)
            return out

        entrants = [("wmma", lambda: ext.gemm_wmma(x, codes, scales, GSCALE)),
                    ("qpn" if m <= 8 else
                     ("qpn+simt" if m == 11 else "qpn_x2"), qpn_call)]
        if m <= 8:
            entrants.insert(0, ("simt",
                                lambda: ext.gemm_simt(x, codes, scales,
                                                      GSCALE)))
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
                             "rel_err": f"{rel:.2e}", "ok": int(ok)})
            totals[(tag, m)] = totals.get((tag, m), 0.0) + t
        line = " ".join(f"{tag}={times[tag]*1000:.1f}" for tag in times)
        best = min(times, key=times.get)
        print(f"({k},{n}) m={m}: {line}  best={best}", flush=True)
    del codes, scales, ref, bcodes, bscales
    torch.cuda.empty_cache()

with open(OUT_CSV, "w", newline="") as f:
    wcsv = csv.DictWriter(f, fieldnames=list(rows_out[0].keys()))
    wcsv.writeheader()
    wcsv.writerows(rows_out)
print(f"CSV: {OUT_CSV}")

print("\n=== 5-shape totals (us) ===")
for m, tags in ((5, ("simt", "wmma", "qpn")), (8, ("simt", "wmma", "qpn")),
                (11, ("wmma", "qpn+simt")), (16, ("wmma", "qpn_x2"))):
    tl = " ".join(f"{t}={totals.get((t, m), float('nan'))*1000:.1f}"
                  for t in tags)
    print(f"  M={m}: {tl}")

print("CORRECTNESS_FAILS:", fails if fails else "none")
print("QPN_RACE_DONE")
