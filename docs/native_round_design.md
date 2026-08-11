# Native speculative round: single-CUDA-graph design

One complete k=7 speculative round (drafter ×7 → verify qlen=8 → rejection/commit)
executes behind a single `cudaGraphLaunch`. Status: built, flag-gated,
output-safe — and inert. The captured graph does not persist the drafter's
recurrent state across rounds, so its drafts are rejected and the orchestration
prize (round 29 → ~19–21 ms projected) is unrealized.

## Motivation (measured)

k=7 round ≈ 29 ms: verify GPU ~13.5 ms, allreduce ~1.7 ms (custom one-shot,
11–15 µs/op), drafter+host ~14 ms — of which GPU is only ~4.5 ms. **~9–10 ms is
Python orchestration**, the largest single line in the round; the 5090 comparison
stack pays ~3 ms here. Prize: AIME decode ~0.9× of the 5090 at measured τ.

Stage anatomy (profiler-instrumented, k=7 greedy-draft, batch=1):

| stage | ms | native-round disposition |
|---|---|---|
| drafter forwards (1+6, graphed) | 3.78 | keep (child graphs) |
| per-iter sampling host path (7 × 0.285) | 2.00 | into the graph (~0.1 total) |
| per-iter metadata rebuild (0.89 setup + 6 × 0.36) | 3.05 | device-side/pre-computed |
| unaccounted Python glue | ~2.8 | eliminated by single entry |
| **proposer** | **11.6 → ~4.5** | **−7 ms** |

Verify-side host (~3–4 ms of scheduler/output path) is second-stage scope. k=7
reference measurements: results/campaign_20260811.csv and
results/nativehead_ab_20260811.csv.

## Design

- Flag `VLLM_SM70_NATIVE_SPEC_ROUND` (default 0), read at proposer init; selects
  the native executor only when the config qualifies (method=mtp, max_num_seqs=1,
  uniform decode, k fixed at boot); anything else silently uses the Python path,
  untouched as reference and fallback. Output must match it exactly (greedy:
  byte-identical; sampled: identical under fixed seed).
- Whole-round CUDA graph: drafter loop + verify + rejection captured as one graph.
  The GPU dependency chain is graphable — drafter input i+1 = embedding(argmax_i)
  is a tensor→tensor dependency.
- Core enabling change (orchestration, not math): every accepted-length-dependent
  buffer becomes device-computed. Positions and slot indices advance by
  `num_accepted`, a GPU scalar post-rejection; a small pre-round kernel derives all
  per-round buffers from it instead of host arithmetic.
- Static per round at the launch config: batch=1, verify qlen=8, drafter qlen=1×7,
  capture size [8], KV block tables within a page, graph selection. GDN spec-state
  slot selectors are device-computable from `num_accepted`.
- Alternatives if one mega-graph resists capture: a C++ sequencer
  (`native_spec_round.cu`) issuing the same child-graph launches via
  `cudaGraphExec_t` handles (recent torch exposes `raw_cuda_graph()`); or
  native-sequencing the drafter chain only (~6–7 ms of the prize).

## Derisk conclusions

- Whole-round capture works only under `vllm.distributed.graph_capture` — TP
  collectives need the coordinated capture context.
- Replay survives with a per-round lockstep rebuild-copy of derived metadata plus
  graph-owned common buffers.
- Validated: all 4 greedy ladder cells byte-identical, native vs Python (rejection
  sampling guarantees output correctness even under draft corruption). Native round
  27.0 ms vs Python 27.3 ms — single-launch overhead is nil.

## Validation protocol

1. Greedy, fixed prompts: byte-identical outputs, identical round counts and
   per-position acceptance counters vs the Python path.
2. Seeded sampled (3 seeds × 3 AIME fixtures): identical outputs (both paths share
   the sampler kernels).
3. Closure invariants: gen == (τ+1)·rounds ± 1; τ ≤ k.
4. Only then round-latency A/B.

## Open defect: drafter state persistence

In serve mode τ collapses to 0.00–0.06 across all cells while output stays
byte-identical — every draft is rejected and repaired by rejection sampling. A
dual-path debug mode only proved the graph computes correct drafts when the Python
loop also ran and advanced the drafter's persistent state; it never tested the
graph advancing state alone.

Diagnosis: the graph's 7 chained forwards must leave the drafter's KV and GDN
conv/ssm state correct for the next round's leading forward. With only the graph
running, that state is wrong from round 2 on and drafts degrade to noise. Drafts
within a round are correct; persistence across rounds is not.

Fix classes (none chosen):
1. Bring the drafter state-cache tensors and their write-slot indices into the
   graph's owned/refreshed set; prove parity with a state-norm probe after N rounds.
2. Scope the graph to exclude the final state-commit and perform that one write in
   Python — keeps most of the ~4.5 ms proposer prize, sidesteps the hazard.
3. Leave as built: flag-gated, Python-path-preserved, byte-safe; the measured
   drafter+host ~14 ms gap stands.
