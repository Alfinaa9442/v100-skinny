# A/B: NVFP4 + Skinny SM70 Kernels vs AWQ + TurboMind

NVFP4 + skinny kernels win 8 of 10 single-stream domains (+8% to +34%);
AWQ wins both code cells — **acceptance is quantization-dependent** — and
the ranking inverts at 4–8 concurrent streams.

## Setup

Same base model (Qwen3.6-27B), fork ([1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) 1.2.2), hardware (4x
V100-SXM2-16GB, TP4), limits (32768 ctx, 8192 batched tokens; single-stream
for the domain matrix), prompts, and probes; one sitting (2026-08-07). Both
stacks run MTP at k=4 (shipped default) and k=10 (deep); greedy throughout.
**A:** nvidia/Qwen3.6-27B-NVFP4 (compressed-tensors) via skinny SIMT/WMMA
kernels (M<=64) + Marlin (M>64), 4-bit lm_head. **B:**
QuantTrio/Qwen3.6-27B-AWQ via the fork's stock TurboMind SM70 AWQ route.

## Single-stream (decode tok/s, greedy)

| domain | NVFP4 k=4 | NVFP4 k=10 | AWQ k=4 | AWQ k=10 | best vs best |
|---|---|---|---|---|---|
| natural (spec_bench) | 93.7 | — | 78.6 | — | +19% |
| extraction | 173.8 | — | 149.5 | — | +16% |
| prose-think | 94.9 | — | 82.7 | — | +15% |
| prose-nothink | 82.4 | — | 76.6 | — | +8% |
| math-nothink | 152.2 | 168.5 | 134.0 | 136.9 | +23% |
| math-think | 144.0 | 160.6 | 122.4 | 113.1 | +31% |
| code-nothink | 125.3 | 118.7 | 129.3 | 127.9 | −3% (AWQ) |
| code-think | 109.3 | 101.2 | 120.7 | 113.1 | −9% (AWQ) |
| struct-json | (151.3) | 216.8 | 148.7 | 200.2 | +8% |
| struct-csv | (139.1) | 152.8 | 113.6 | 101.3 | +34% |

(Parenthesized = same-config k-sweep values, not re-run this sitting; successor dataset: results/campaign_20260811.csv.)

## Batch scaling (aggregate tok/s, natural prompts)

| config | 1 stream | 8 streams | 32 streams |
|---|---|---|---|
| NVFP4 plain (MNS=32) | 89.0 | 414.5 | 1083.0 |
| AWQ plain (MNS=32) | 70.3 | 507.3 | 1348.1 |
| NVFP4 mtp4 (MNS=8) | 85.5 | 318.6 | — |
| AWQ mtp4 (MNS=8) | 79.0 | 381.5 | — |

## Findings

1. Acceptance is a property of the quantized checkpoint pair, not the
   serving machinery. The AWQ mtp_head agrees with its target better on
   code (77.2% vs our 63.7% no-think at k=4; 71.6% vs 53.2% think) —
   winning both code cells despite ~12% slower steps (k=4: ~28.1–28.8 ms
   vs ~31.4–32.8; k=10: ~37.0–39.0 vs ~42.1–47.4).
2. The largest NVFP4 leads are bandwidth-shaped (math, struct-csv,
   extraction); deep-k (k=10) helps both stacks on math/struct and hurts
   both on code — the domain economics transfer across quantizations.
3. **Batch crossover at ~4–8 streams.** Skinny wins 1–4 streams
   (11.2 vs 14.2 ms steps at 1); TurboMind's compute-tuned AWQ GEMMs scale
   flatter into mid-M and win at 8+ (+22% plain @8, +24% @32, +20% mtp4
   @8). Guidance: NVFP4 + skinny for latency-sensitive low-concurrency
   serving (every domain except code); AWQ + TurboMind for batch farms.
4. Reproducibility: previously published NVFP4 numbers reproduced within
   ±4% same-sitting; greedy acceptance counters byte-identical.

## Reproduce

A-side launch: `scripts/launch_qwen36_ctfull_mtp.sh`. Probes (frozen prompts): `benchmarks/k_sweep_probe.py`, `benchmarks/spec_bench.py`, `benchmarks/concurrency_probe.py`.
Raw rows tagged +abours*/+abawq* in results/k_sweep_matrix.csv and results/concurrency_matrix.csv.
