# Full benchmark matrices

Every matrix behind the headline tables in [`README.md`](../README.md): the backstops under the 1Cat
comparison, the AIME 2026 long-reasoning fixtures, the batch crossover, the complete ninfer
head-to-head on an RTX 5090, the thinking-mode and phase matrices, and the round-latency ledger the
roadmap prices. Rows are single-stream decode-only tok/s unless the section says otherwise.

## Backstops under the 1Cat lead

The [README table](../README.md#evaluation) reports each stack at the profile it ships. Two backstops
control for profile and then for speculation itself.

**Matched k, same sitting.** The lead is not speculation depth. At matched k=7 with an fp16-class
lm_head on both sides ours wins 3, ties 2 and loses 1:

| Matched k=7, fp16-class lm_head both sides | Ours | 1Cat (AWQ) |
|---|---:|---:|
| Domains won / tied / lost | 3 / 2 / 1 | — |
| code τ (the one loss) | 3.81 | 4.23 |
| Round latency | 27.7–29.5 ms at k=7 | 29.5–31.2 ms at k=4 |

Code is the loss: their mtp_head agreement advantage cancels our round-speed edge
([`../results/campaign_20260811.csv`](../results/campaign_20260811.csv)). Our k=7 rounds run shorter
than their k=4 rounds — nearly twice the speculative depth per round for less wall time
([`../results/flagship_matrix_20260811.csv`](../results/flagship_matrix_20260811.csv)).

**Plain decode.** Strip speculation from both sides and the raw kernel shows through: **86.6 tok/s
against TurboMind's 70.7, a 1.22× lead** — single-stream, greedy, fp16 lm_head both sides, MTP off,
domain-flat ([`../results/plain_decode_compare.csv`](../results/plain_decode_compare.csv)). The
founding result the whole project rests on.

## Long reasoning: AIME 2026 fixtures

ninfer's verbatim AIME 2026 fixtures ([`../benchmarks/aime_exact.py`](../benchmarks/aime_exact.py)),
both stacks on the same box. Unlike the greedy tables in the README these rows are thinking-on and
sampled at ninfer's profile (temp 0.6, top-p 0.95, top-k 20, presence 1.0, seeded) — predominantly
reasoning trace with the boxed answer at the tail
([`../results/aime_headtohead.csv`](../results/aime_headtohead.csv)):

| Fixture | Ours (k=7 QPN, greedy drafter) | 1Cat (AWQ, k=4, shipped drafting) | Lead |
|---|---:|---:|---:|
| aime26_01 | 172.8 | 112.2 | **1.54×** |
| aime26_15 | 126.4 | 98.4 | **1.28×** |
| aime26_30 | 140.8 | 109.1 | **1.29×** |

The configs are asymmetric — ours uses `draft_sample_method=greedy` while the 1Cat arm ran its
shipped probabilistic drafting, a flag that exists untested on their stack (same fork) and is worth
roughly +20–25% acceptance under sampled serving — and this campaign ran the legacy 4-bit lm_head on
our side against fp16 on theirs, so the machinery-matched lead is smaller than shown. Their
acceptance tracks ours closely; the lead is round latency plus the greedy-drafter fix.

## The batch crossover

The ranking inverts with concurrency
([`../results/concurrency_matrix.csv`](../results/concurrency_matrix.csv), natural prompts, plain
decode, aggregate tok/s):

| Streams | Ours (NVFP4 + skinny) | 1Cat (AWQ + TurboMind) |
|---:|---:|---:|
| 8 | 414.5 | 507.3 |
| 32 | 1083.0 | 1348.1 |

TurboMind's compute-tuned AWQ GEMMs scale flatter into mid-M and win above the ~4–8-stream crossover.
The deployment split is explicit: skinny for latency and low concurrency, TurboMind for batch farms.

## Against ninfer, on a single RTX 5090

[ninfer](https://github.com/Neroued/ninfer) is a purpose-built C++ inference engine running the
**same Qwen3.6-27B NVFP4 model** on one 32 GB RTX 5090 — consumer Blackwell with native FP4 tensor
cores, hardware our V100s predate by ~8 years. Conventions are identical to ours (decode-only tok/s,
acceptance = accepted/drafted, tokens/round = 1 + acc·k). Where four V100s hold parity:

- **Plain decode**: ours 86.6 (fp16 lm_head) to 91.2 (native 4-bit lm_head, pinned launch anchor)
  against their published 86.4.
- **Structured decode**: ours 243.4 at k=15 (json) against their published 243.1. Extraction at
  366.0 is ours alone — they report no such cell.
- **Acceptance at matched config**: at k=3 with fp16 lm_head, seeded ×3 on their verbatim fixtures,
  f01 83.4 ± 0.6% against their 80.8 ± 1.8%, f15 71.3 against 74.7, f30 77.5 against 80.8.

Where they win: prefill by ~4× (native FP4 W4A4 flops — not a decode-path claim), and long reasoning,
where we run 0.79× their rate. That cell is thinking-on and sampled at ninfer's profile, not the
greedy regime of the README tables:

| aime26_01 | Ours (k=7, native lm_head) | ninfer (MTP3, n=5) |
|---|---:|---:|
| Decode tok/s | 175.5 (replicated 174.8) | 222.7 ± 3.4 |
| Tokens committed per round (1 + τ) | **5.31** | 3.43 |
| Round latency (derived) | 30.3 ms | **15.4 ms** |
| Acceptance | 61.6% at k=7 | 80.8 ± 1.8% at k=3 |
| Completion length | 8000–8073 tok | 11717 ± 477 tok |

**Our speculation commits more tokens per round than theirs (5.31 against 3.43) and we still lose,
because their engine turns a round in 15.4 ms and ours takes 30.3 ms** — the deficit is engine round
latency, host orchestration of the drafter chain, not acceptance and not matrix throughput. Only
aime26_01 is a clean comparison: their f15 run hits their own 65536-token cap, and both f15 (~65.5k)
and f30 (~46.4k) completions exceed our 24576-token serving context.

## Thinking on vs thinking off

Per-domain serving-profile matrix, native NVFP4 lm_head, decode-only, `tok/s (τ)`
([`../results/flagship_matrix_20260811.csv`](../results/flagship_matrix_20260811.csv)):

| Domain | k=7 think | k=7 nothink | k=15 think | k=15 nothink |
|---|---:|---:|---:|---:|
| prose | 96.4 (1.84) | 82.0 (1.33) | 72.1 (1.90) | 62.9 (1.48) |
| math | 183.6 (4.40) | 196.6 (4.48) | 155.0 (5.20) | 163.5 (5.54) |
| code | 122.1 (2.57) | 169.1 (3.81) | 88.6 (2.54) | 137.7 (4.45) |
| json | 157.1 (3.59) | 228.4 (5.56) | 131.6 (4.26) | 242.9 (8.74) |
| csv | 169.3 (3.93) | 200.4 (4.72) | 124.6 (3.96) | 190.1 (6.60) |
| extract | 213.9 (5.41) | 261.5 (6.92) | 225.7 (8.09) | 381.9 (14.61) |

The think cells are capped at 2048 tokens: they measure the trace-dominated phase, not a full
request. Round latency is 27.7–30.1 ms at k=7 and 39.4–40.6 ms at k=15 — depth costs ~10 ms per
round, which only pays where τ keeps climbing (json, extract).

**Run structured endpoints with thinking off.** JSON goes 157.1 → 228.4 tok/s at k=7 and
131.6 → 242.9 at k=15; code 122.1 → 169.1; csv 169.3 → 200.4. Math is the one domain whose reasoning
trace drafts as well as its answer (τ 4.40 thinking on against 4.48 off), so leaving thinking on
where it earns accuracy costs 7% (183.6 against 196.6).

This matrix is a frozen campaign and its k=15 extract cell reads 381.9 against the 366.0 of the
current shipping anchor. The campaign rode a pre-QPN-era memory envelope (`MNS=4 GMU=0.94`, CT stash
present) that no longer boots on the current kernel; a DROP_CT bisect reproduced 366 with the stash
both present and dropped, so the delta is era-level, not a config regression, and output stays
byte-identical to the k=7 lossless lineage. Freeing the CT stash is speculative-profile-only: the
plain arm boots `DROP_CT=0`, since the qpn1 M=1 path is ~40% slower than SIMT at full-backbone decode
duty ([`../results/dropct_validation_20260812.csv`](../results/dropct_validation_20260812.csv)).

### Phase split, uncapped generations

Uncapped generations (8192 budget), thinking on, k=7 native lm_head, split by phase
([`../results/phase_probe_20260811.csv`](../results/phase_probe_20260811.csv)):

| Domain | Trace τ | Payload τ | End-to-end tok/s | Generation |
|---|---:|---:|---:|---|
| prose | 2.11 | 1.47 | **91.1** | 3453 tok, natural stop |
| math | 4.40 | 5.02 | **188.7** | 2504 tok, natural stop |
| code | 2.61 | 3.69 | **142.6** | 7234 tok, natural stop |
| json | 4.56 | 6.99 | **207.3** | 7644 tok, natural stop |
| extract | 5.29 | 6.96 | **249.3** | 8192 tok, capped |

Only math traces are drafter-predictable: every other domain drafts its trace worse than its payload,
and that phase is the longest part of the request (json: 5810 of 7644 tokens). Prose inverts — its
planning trace drafts better than its creative payload, the highest-entropy text we serve.

## Roadmap: the round-latency ledger

The kernel is a 5.9× win; the serving lead over 1Cat is 1.2–2.4× depending on domain, because the
GEMM is only part of a decode step. The rest is **engine round latency**, and the ninfer comparison
prices it exactly: their k=3 round turns in 15.4 ms, ours takes 30.3 ms at k=7.

Round-latency ledger — engineering projection, not measurement:

| Item | Projected saving | Measured anchor |
|---|---:|---|
| Native single-launch speculative round (kill per-round Python orchestration) | ~5 ms | proposer wall 11.60 ms vs GPU 6.05 ms; ~2.8 ms unaccounted Python glue ([`native_round_design.md`](native_round_design.md)) |
| QPN port of the remaining decode-path GEMMs (lm_head + drafter projections onto the prepacked QPN format) | 1–2.5 ms | lm_head repack is same-values, pure speed ([`terminology_audit.md`](terminology_audit.md)) |
| mtp_head (drafter) path: graphed forwards, device-side metadata | 1.5–2.5 ms | per-iter sampling host path 2.00 ms + metadata_cpu 3.05 ms per round |
| Sampler fusion (the greedy path runs two softmaxes over a 248k vocab) | ~1 ms | sampler ~0.4 ms/token, double softmax 0.29 ms ([`decode_residual_ledger.md`](decode_residual_ledger.md)) |

Banking even ~7 ms of that 8.5–11 ms ledger puts the k=7 round near **23 ms**, which at our measured
5.31 tokens/round projects **≈230 tok/s** on aime26_01 — parity with ninfer's 222.7, as a projection
from measured round anatomy rather than a result. The native round is built, flag-gated and validated
output-safe, but inert: the captured graph does not persist the drafter's recurrent state across
rounds, so served drafts are rejected and the live stack runs the stock Python proposer. Also queued:

- **W8A16 FP8-islands acceptance experiment** (#1 post-launch): keep NVIDIA's FP8 bytes in VRAM and
  decode them in-kernel — exactly the W4A16 pattern — for the ~3.3B-parameter FP8 surface (attention
  `q/k/v/o` + GDN `out_proj`). Costs ~+0.4 GB/rank and ~2–3% plain decode; buys full bit-nativeness,
  and tests whether the original FP8 islands recover drafter agreement on prose and code
  ([`lmhead_provenance.md`](lmhead_provenance.md)).
- **Drafter self-distillation**: the per-domain τ curve shows the mtp_head is the binding constraint
  on deliberative traces (τ ≈ 2–3), where the round machinery is already efficient. Adapting the
  drafter to the served quantized backbone attacks the acceptance side of the same product.

## Reproducing these tables

| To reproduce | Run |
|---|---|
| Kernel bandwidth curve | [`../benchmarks/kernel_bw_bench.py`](../benchmarks/kernel_bw_bench.py) |
| Per-kernel M sweep | [`../benchmarks/kernel_m_sweep.py`](../benchmarks/kernel_m_sweep.py) |
| Per-domain serving matrix | [`../benchmarks/k_sweep_probe.py`](../benchmarks/k_sweep_probe.py) |
| Thinking-on phase split | [`../benchmarks/phase_probe.py`](../benchmarks/phase_probe.py) |
| AIME long-reasoning cells | [`../benchmarks/aime_exact.py`](../benchmarks/aime_exact.py) |
| Batch crossover | [`../benchmarks/concurrency_probe.py`](../benchmarks/concurrency_probe.py) |
| Serving boot (k=7 flagship) | [`../scripts/launch_qwen36_ctfull_mtp.sh`](../scripts/launch_qwen36_ctfull_mtp.sh) |

Every benchmark script writes to `results/` and prints the losslessness-diff verdict alongside
throughput. Never quote a spec-decode throughput without an output diff against plain decode at
matched configuration — acceptance metrics alone cannot distinguish speed from corruption. Step-by-
step recipes are in [`REPRODUCE.md`](REPRODUCE.md), launch envelopes in
[`DEPLOYMENT.md`](DEPLOYMENT.md).
