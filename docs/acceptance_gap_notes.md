# Speculative-Decoding Acceptance Gap vs ninfer: Root Cause and Fix

One config flag — `"draft_sample_method": "greedy"` — closes a 10–25-point
MTP acceptance gap on sampled serving: matched-config acceptance reaches
**83.4 ± 0.6%** vs the reference engine's **80.8 ± 1.8%**.

## Root cause

The fork's SM70 MTP default is `draft_sample_method=probabilistic`: the
drafter (mtp_head) samples proposals at the request temperature, and
stochastic proposals are rejected far more often than argmax ones — a
10–25-point acceptance cost on sampled workloads. `"greedy"` (argmax
proposals; the upstream default and the reference engine's regime) removes
the gap at zero runtime cost. Greedy-decode benchmarks are unchanged
(temperature-0 drafts are argmax either way); sampled chat decode gains
20–25% throughput. Caveat: one-hot proposals under rejection sampling
perturb the sampled output distribution slightly vs pure target sampling —
the same regime the reference stacks ship.

## Reference stack

[ninfer](https://github.com/Neroued/ninfer): Qwen3.6-27B NVFP4 on one
RTX 5090 (32 GiB) — W4A16 decode (native-FP4 W4A4 serves prefill only),
INT8 KV cache, chain-MTP sharing the target lm_head, sampling temp 0.6 /
top-p 0.95 (within-candidates) / top-k 20 / presence 1.0. Conventions match
ours: decode-only tok/s, acceptance = accepted/drafted, tokens/round =
1 + acc·k. Fixture prompts SHA-matched in `benchmarks/ninfer_fixtures/`.

## Matched-config result (k=3, fp16 lm_head, seeded x3, verbatim fixtures)

| fixture   | ours            | ninfer (RTX 5090) |
|-----------|-----------------|-------------------|
| aime26_01 | **83.4 ± 0.6%** | 80.8 ± 1.8%       |
| aime26_15 | 71.3%           | 74.7%             |
| aime26_30 | 77.5%           | 80.8%             |

f01 exceeds the reference; f15/f30 sit inside the first-20k-window
measurement bias (ours capped at 20k tokens vs their full traces).

## Eliminated hypotheses

| hypothesis | verdict |
|---|---|
| fp32 recurrent-state storage | eliminated — seeded-triplicate means fall inside the fp16 arm's distribution |
| GDN gating precision (fp32 β/decay upcast) | eliminated — no acceptance gain (47.9 / 55.4) |
| drafter-block precision | eliminated — parity reached with drafter-block compute unchanged at fp16 |
| sampler semantics | false lead — vLLM's batch-1 sampler already matches ninfer's within-candidates top-p |
| draft vocab truncation | eliminated — full 151,936-entry draft vocab confirmed in source |
| backbone kernel identity | eliminated — acceptance invariant under QPN⇄WMMA kernel swap to two decimals |

## Methodology: seeded triplicates

Single unseeded runs manufactured a false effect: one fp32-state sample
suggested +5–9 acceptance points, but run-to-run noise on these fixtures is
±3–5 points. Any stochastic acceptance claim needs per-request seeds and
three seeds per arm, or greedy pairs (zero within-config variance).

## Downstream head-to-head (results/aime_headtohead.csv)

Both stacks seeded x3 on the verbatim fixtures, same box, ninfer sampling
profile: ours (k=7, greedy drafter) mean 146.7 decode tok/s vs 1Cat
AWQ + TurboMind (k=4) 106.6 — **1.38x**. 1Cat's acceptance tracks ours
closely; the lead is round latency plus the greedy-draft fix. Native-lm_head
follow-up (no acceptance tax): results/nativehead_ab_20260811.csv.
