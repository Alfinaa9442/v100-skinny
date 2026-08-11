"""Phase 0 — exposure arithmetic for the batch band (no GPU, no code).

For each WMMA dispatch config, compute the weight bytes each thread /
warp / SM holds in flight per chunk (the staging burst) against the
Little's-law requirement (sustained bytes-in-flight = latency x BW), and
the grid wave count per real shape. Output = the figure for
twin_race_notes.md.

Constants: V100 HBM2 effective BW ~800 GB/s (measured stream ceiling on
this box ~790-830), DRAM latency ~300 ns (400-500 core cycles at
1.53 GHz). Little's law: 800e9 * 300e-9 = 240 KB in flight GPU-wide
/ 80 SMs = 3.0 KB/SM sustained, weights + activations combined.
"""
SHAPES = [(1536, 5120), (4352, 5120), (5120, 8704), (5120, 4096),
          (5120, 2048)]
SMS = 80
BW = 800e9
LAT = 300e-9
NEED_PER_SM = BW * LAT / SMS

# (M band, WN, WM, KC) — the settled dispatch tiles (skinny_kernels.cu)
CONFIGS = [("M<=16", 4, 1, 256), ("M17-32", 2, 2, 128), ("M33-64", 2, 4, 128)]

print(f"Little's law: {BW/1e9:.0f} GB/s x {LAT*1e9:.0f} ns = "
      f"{BW*LAT/1024:.0f} KB in flight GPU-wide -> "
      f"{NEED_PER_SM/1024:.2f} KB/SM sustained\n")

for name, wn, wm, kc in CONFIGS:
    nt, mt = wn * 16, wm * 16
    nthreads = wn * wm * 32
    cseg = nt * (kc // 16) // nthreads      # 8B code loads per thread
    xseg = mt * (kc // 8) // nthreads       # 16B x loads per thread
    w_bytes_thread = cseg * (8 + 1)         # uint2 code + scale byte
    x_bytes_thread = xseg * 16
    smem_kb = (nt + mt) * (kc + 16) * 2 / 1024
    ctas_smem96 = int(96 // smem_kb) if smem_kb <= 96 else 0
    burst_w_cta = w_bytes_thread * nthreads / 1024
    burst_x_cta = x_bytes_thread * nthreads / 1024
    print(f"{name}: tile NT={nt} MT={mt} KC={kc} threads={nthreads} "
          f"smem={smem_kb:.1f}KB (<= {ctas_smem96} CTAs/SM smem-bound; "
          f"registers bound it lower per session-18 sweep)")
    print(f"  staging depth: CSEG={cseg} code loads/thread "
          f"(+{xseg} x-loads) -> weight burst {w_bytes_thread} B/thread, "
          f"{burst_w_cta:.1f} KB/CTA; x burst {burst_x_cta:.1f} KB/CTA")
    for ctas in (1, 2):
        tot = (burst_w_cta + burst_x_cta) * ctas
        wtot = burst_w_cta * ctas
        print(f"  @{ctas} CTA/SM: total burst {tot:.1f} KB/SM "
              f"(weights {wtot:.1f}) vs need {NEED_PER_SM/1024:.1f} KB/SM "
              f"sustained -> {'OK if issued continuously' if tot >= NEED_PER_SM/1024 else 'LATENCY-EXPOSED even as a burst'}")
    # the burst is issued ONCE per chunk and must cover the whole chunk
    # period; the sustained equivalent is burst / chunk-period. Chunk
    # period at the DRAM floor would be (weights+x bytes)/BW; exposure
    # ratio = sustained-need / weight-burst says how far the pipeline
    # relies on the mma loop being long enough to hide the next refill.
    print(f"  wave counts: " + ", ".join(
        f"({k},{n}): {n // nt} CTAs = {n / nt / SMS:.1f} waves/SM-slot"
        for k, n in SHAPES))
    print()

print("Reading: the M<=16 config stages 8 independent code loads/thread "
      "(the healthy SIMT-like depth). The band configs collapse to 2 "
      "(M17-32) and 1 (M33-64) code loads/thread — a single dependent "
      "8B load per thread per chunk at M33-64, refilled only once per "
      "chunk period, is the same disease the register gate isolated at "
      "M<=8. Prediction: rung A (deepen the code prefetch to 2 chunks, "
      "cut a barrier via smem double-buffer where it fits) should "
      "recover a real fraction of the 3x-to-floor gap at M=32/64 on "
      "grid-wide shapes; N-starved shapes (N=2048: 64 CTAs < 80 SMs = "
      "0.8 waves) stay partially grid-limited regardless.")
