# QPN m8n8k4: A Volta-Native Tensor-Core Path for Skinny W4A16 GEMM

Four-quadpair `mma.sync.m8n8k4` with quadpairs split on N beats both
incumbents across M≤8 — **1.94× at M=8, at 647 GB/s effective
bandwidth** — the first tensor-core path in this stack to reach the
SIMT M=1 streaming floor. Data: `results/qpn_race_20260810.csv`;
kernel `kernels/research/qpn_race.cu`; prepack `benchmarks/qpn_bench.py`.

## Results (µs, totals over 5 (N,K) shapes; per-shape rows in the CSV)

| M | simt | wmma (padded) | qpn | qpn vs best incumbent |
|---|---|---|---|---|
| 5 | 187.9 | 251.4 | **130.7** | **1.44×** (wins 4/5; loses only N=2048 to simt) |
| 8 | 253.0 | 251.8 | **129.6** | **1.94×** (wins 5/5) |
| 11 | — | 256.8 | 333.4 (hybrid) | loses — see below |
| 16 | — | 264.7 | **253.8** (two 8-row tiles) | 1.04× (wins 4/5) |

Effective bandwidth at M=8 on (5120, 8704): 25.1 MB / 38.8 µs ≈
647 GB/s — the SIMT M=1 figure, now with tensor-core MACs; GEMM
cost is flat in M up to 8 rows. Race-harness figures are
shape-favorable (single-shape peaks, no dispatch overhead): the
production number is the dispatch-curve aggregate, **431–455 GB/s
across M=4–8** (`results/kernel_bandwidth_20260811.csv`). Kernel: 56 registers, zero spills, one
barrier (cross-warp K-reduction at output), 9 blocks/SM of headroom —
the grid (N/32), not the kernel, limits residency. Passes the 1e-3
relative-error gate on every cell, including outlier activations.

Edge cases: the M=11 qpn(8)+simt(3) hybrid loses (333.4 vs 256.8 µs)
to a second launch plus a row copy — the M 9–16 route is two 8-row
A-tiles (pad 11→16: 253.8, parity-to-+4% vs WMMA), so the win is M≤8;
(5120, 2048) at M=5 stays simt (64 CTAs = 0.8 waves, N-starved).

## Mechanism

- **Quadpairs split N, not K**: one warp instruction executes 4
  independent 8×8×4 MMAs sharing one 8×4 activation tile — activation
  traffic per weight byte drops 4×.
- **No main-loop shared memory or barriers**: activations load straight
  from global (KB-scale, L1/L2-resident; quadpair-sibling lanes hit the
  same line); weights stream global→register.
- **Direct NVFP4→B-fragment decode**: nibble pre-interleave in the
  offline prepack lands the dequant (i, i+4) output exactly on the
  adjacent-k B-fragment register pair — no inner-loop packing.
- **Scale economy**: one FP8 group-scale register serves exactly its
  group's 4 MMAs (k=4 × 4 = the group-16 granularity).

## Production integration (shipped)

QPN serves production: loader-side nibble-interleaved B-fragment
prepack at weight load, dispatch SIMT M≤3 / QPN M 4–16 / WMMA 17–64
(`VLLM_SKINNY_QPN=1`, default on). M 4–16 covers speculative-decode
verification at k≤15 and plain decode at 2–8 concurrent streams — the
measured crossover zone vs TurboMind AWQ. One duty-cycle boundary: the
M≤3 qpn1 variant is ~40% slower than SIMT at full-backbone M=1 duty
(N/32 grid starvation), so it serves only speculative profiles where
M=1 traffic is the small mtp_head drafter block
(`results/dropct_validation_20260812.csv`); plain-decode profiles keep
SIMT at M≤3.

Closed alternatives: quadpairs split on K (B_ring) — killed by 4×
activation re-reads per weight byte; runtime B-fragment packing —
killed by 8 pack instructions per decode window; the M 17–64 batch
band stays conceded to WMMA (`docs/twin_race_notes.md`).
