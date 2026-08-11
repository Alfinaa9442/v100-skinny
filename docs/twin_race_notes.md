# Batch Band M=16–64: Register-Fragment Challengers Lose, WMMA Stands

Both register-fragment challengers lose to the production WMMA kernel
across M=16–64 — load-schedule pipelining by 8–33%, the m8n8k4
register ring by 1.45–2.64× — closing the batch band as structural.
Correctness clean on every cell (rel ≤ 1e-3, outlier activations).
Data: `results/twin_race_20260810.csv`; code `kernels/research/twin_race.cu`, `benchmarks/twin_race_*.py`.

## Results (µs, totals over 5 (N,K) shapes; best-of-shape counts in CSV)

| M | wmma (incumbent) | A_d2 | A_d2b | A_d1b | B_ring |
|---|---|---|---|---|---|
| 16 | **265.4** | 352.8 | 391.1 | 297.5 | 384.8 |
| 32 | 333.9 | 429.7 | 418.0 | **332.9** | 665.5 |
| 64 | **571.2** | 595.9 | 770.2 | 618.8 | 1507.5 |

**Rung A — load-schedule pipelining.** Loses 8–33% at M=16/32: depth-2
weight-code staging costs occupancy (124–164 registers; the
double-buffered variant halves blocks/SM by doubling shared memory).
A_d1b's M=32 "win" is 0.3% on the total (3/5 shapes by 1–5%) — noise,
below the ≥20% adoption threshold.

**Rung B — m8n8k4 register ring.** Loses 1.45× (M=16) to 2.64× (M=64).
Not a resource-fit failure (82–124 registers, zero spills, 4 weight
loads in flight continuously, 8–32 warps/SM) but per-MAC economics: per
64-k window the ring issues up to 32 `mma.m8n8k4`, each fed by 2
per-lane shared-memory reads, where WMMA amortizes one `load_matrix_sync`
pair across a 16×16×16 tile; the 8-row tile re-reads activations ~4×
more per weight byte — the same LSU wall, moved to the activation side.

**Falsified exposure prediction.** Little's law (800 GB/s × 300 ns =
234 KB in flight ≈ 2.9 KB/SM) and the staging-depth collapse (8→2→1
across the M≤16 / M17–32 / M33–64 configs) predicted a rung-A win at
M=32/64. Falsified: the weight burst issues underneath the mma loop;
2–4 CTAs/SM already cover DRAM latency — depth was a red herring.

## Occupancy

| kernel | regs | blocks/SM | warps/SM |
|---|---|---|---|
| pipe (2,2,128) d2 / d2+dbuf | 124 / 100 | 4 / 2 | 16 / 8 |
| pipe (4,1,256) d2 / d2+dbuf | 164 / 136 | 2 / 1 | 8 / 4 |
| pipe (2,4,128) d2 / d2+dbuf | — | 2 / 1 | 16 / 8 |
| ring MB=4 / MB=8 | 82 / 124 | 5 / 2 | 20 / 8 |

## Batch-band closure

Three independent eliminations close the band: tile/occupancy geometry
(10-configuration sweep), m8n8k4 register fragments (register-gate at
M≤8, rung B here), and load-schedule latency cures (rung A, the
falsified prediction). The incumbent's 3×-to-floor plateau at M 17–64
is the structural price of dequant→shared-memory→fragment at V100
occupancy; no load-schedule, tile-shape, or instruction-set change on
SM70 beats it. M-heavy batch serving concedes to TurboMind AWQ; this
stack owns single-stream and low-concurrency (crossover at 4–8 streams).

## Other closed kernel paths

**int8 `dp4a` SIMT (M 4–16).** `dp4a` issues 4 MACs per slot against HFMA2's
2, and the e2m1 lattice ×2 is integer-exact, so the weight side is lossless
int8 and only activations quantize (symmetric per-16-group, fused single
launch, amortized over all N rows). The prototype was correct to 1.4–2%
relative and still ran 50–140% slower than fp16 SIMT/WMMA at every M in
4–16, across three optimization rounds (fused quant kernel recovering 3×
over the naive torch chain, direct half output, conflict-free staged
layout).

Mechanism: Volta single-dispatch. `dp4a`'s MACs land on the INT pipe
beside their own nibble unpack, so the 4-MAC/slot density pays for issue
contention with the decode. The fp16 path wins by splitting dequantization
(INT) from MACs (FP16) across both pipes — the same property, measured from
the other side. `gemm_dp4a` and `skinny_quant_a8` remain in
`kernels/skinny_kernels.cu` for reference; production dispatch never
selects them.
