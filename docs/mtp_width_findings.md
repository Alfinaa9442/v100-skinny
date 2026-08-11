# Chain-MTP speculative depth on 4x V100: verify qlen > 16 corrupts output — k <= 15 is the serving cap

Speculative-decoding depth and width findings for chain-MTP serving of Qwen3.6-27B (NVFP4, TP4) on
4x 16GB Tesla V100 under our vLLM fork. The headline is a hard correctness wall — verify query lengths
above 16 deterministically corrupt output via the V100 flash-decode kernel's 16-query tile — with
per-domain depth economics below the wall, every quoted number gated by a byte-level output diff
against plain greedy decode.

## 1. The deep-k corruption wall

Bisect verdict: k=15 (verify qlen 16) is byte-identical to plain greedy decode on all three ladder
fixtures (CSV / math / JSON); k >= 16 (qlen 17) is deterministically corrupt — degenerate repetition
("1.1.1.1...") with identical bytes across every toggle tested. Mechanism: the V100 flash-decode
attention kernel's 16-query tile (m16 tensor-core fragment geometry) silently overruns at query
position 17. Upstream fix is a query-tile loop in the decode kernel, or an assert `qlen <= 16` at
dispatch. Serving rule: `num_speculative_tokens <= 15`, hard.

Elimination chain (layers we own):

- Skinny NVFP4 verify GEMM: EXONERATED — WMMA correctness-swept clean through M=32 on all real shapes.
- GDN recurrent-state slot provisioning: EXONERATED — allocation is fully k-derived (k+1 slots end to
  end); an instrumented deep-k boot logged zero truncation/clamp/aliasing warnings.
- Chain spec-decode fast-build: EXONERATED — A/B at k=16 with the fast path disabled: 4/4 boots degenerate.
- Also eliminated: small-q auto-raise, the rejection sampler (MAX_SPEC_LEN=128), mtp_head step indexing.

Corroboration and scope:

- A parallel-draft (DFlash-style) verify route is byte-identical to plain at query length 24: the
  16-cap is a property of the chain-MTP flash-decode path, not the hardware. That route runs
  uncaptured at 87-107 ms/step (81.9-92.5 tok/s structured) — correctness door open, performance
  needs target-graph capture engineering.
- One early k=12 boot produced degenerate output that never reproduced; any k > 10 config should pass
  a boot-time output diff before serving.
- Fixing the tile was deferred by decision: tau saturation (below) caps the prize at ~+5-6% structured
  throughput (~228-230 tok/s projected at k=16-18).

## 2. k-sweep economics

Acceptance is a text-domain property, not a stack property — publish it as a curve, not a scalar:

| domain | acceptance/draft | per-position profile |
|---|---|---|
| free-form prose | ~31% | 68/37/16/5% (collapses by position 4) |
| prose, thinking mode | 38-42% | planning trace lifts prose ~7-10 pts |
| code | 53-64% | 85/70/55/45% |
| math | 79-82% | 92/83/74/66% (23% still accepted at position 10) |
| extraction (verbatim) | 97-100% | flat to the tail |

Step-cost mechanism: verify width is nearly cost-free. Dispatch routes verify M <= 7 to SIMT-direct
(crossover to padded WMMA at M=8), and M=9-16 share one padded WMMA tile — so past the one-time
SIMT-to-WMMA crossing, marginal depth costs only the ~0.75-1 ms per sequential mtp_head drafter
iteration. Depth pays while the per-position acceptance tail clears ~10%.

Tau saturation: on structured text, accepted tokens/step plateau at ~9-10 around k=16-20 while step
cost grows linearly — the depth limit is economic near k~16 even without the tile wall. Break-even
law for deepening past k=10: k=15 pays iff (tau+1) grows >= 1.39x. Extraction clears it (tau 10.00
at k=10, saturated at exactly k+1, to 14.61 at k=15); saturated structured workloads mostly do not.

Final validated adoption (losslessness-laddered; rows in results/k_sweep_matrix.csv):

| workload | k | tok/s | vs plain |
|---|---|---|---|
| extraction | 15 | 303.9 | 3.48x |
| struct-json | 15 | 217.3 | 2.33x |
| math | 10 | 172.7 | 1.85x |
| struct-csv | 10 | 152.5 | 1.64x |
| code (algorithmic) | 4 | 125.7 | 1.35x |
| chat/prose (thinking) | 4 | 94.1 | 1.01x |

Notes: k=1 loses everywhere (speculative overhead, 1-token cap). Non-thinking prose peaks at k=3
(90.0 tok/s) and stays below the 93.2 tok/s plain baseline — the only losing cell. Code is
task-shaped: boilerplate scales with depth (SQL DDL k=10 = 171.8, +37% over the algorithmic best;
CRUD-with-thinking k=10 = 136.9) while algorithmic code prefers k~4-5. Extraction at k=15 holds
97.4% acceptance with a flat 100/100/97...97 positional profile (3.48x over its 87.4 tok/s plain
reference).

## 3. Triton backend: independent conviction, distinct corruption

- Triton at k=16 runs struct-csv/json clean and deterministic where flash degenerates — an
  independent confirmation that the flash 16-query tile overrun is the wall, not shared machinery
  above it.
- Triton has its own corruption mode: thinking-mode prose degenerates at k >= 10 (the same "1.1.1"
  collapse), and k >= 20 soft-diverges from plain decode on all prompts. The 202-251 tok/s triton
  prose rows are mirages; all triton rows in results/k_sweep_matrix.csv are unvalidated except where
  marked.
- Backend numerics shift greedy trajectories enough to flip per-workload acceptance regimes
  (non-thinking prose: 82.6 tok/s at 32.6% acceptance on flash vs 51-63 tok/s at 9-13% on triton) —
  cross-backend speculative throughput comparisons are confounded by content divergence.
- Two attention backends with distinct content-dependent corruption modes under the same MTP path
  (flash: structured text at qlen > 16; triton: thinking-mode prose at k >= 10) points to a shared
  spec-machinery interaction, not purely per-kernel bugs. Only the flash k <= 15 ladder-verified
  table above is validated.

## 4. Concurrency and batch

Plain decode, CUDA-graph capture fixed, greedy, 512 tok/request (results/concurrency_matrix.csv):

| streams | aggregate tok/s | ms/step |
|---|---|---|
| 1 | 90.2 | 11.1 |
| 8 | 412.3 | 19.3 |
| 16 | 746.0 | 21.4 |
| 32 | 1,079.9 | 29.5 |
| 64 | 1,301.5 | 49.0 |

At 32 streams that is 11.6x aggregate throughput for 2.7x step cost — the M <= 64 WMMA flat zone in
production form, with the full curve running 90 to 1,301 tok/s across 1 to 64 streams.

Capture-sizes-are-tokens: `cudagraph_capture_sizes` entries are token counts, not stream counts.
Under speculation the adjuster rounds each size to a multiple of k+1 and silently drops sizes above
the list maximum, so an under-provisioned list runs eager and masquerades as a throughput collapse.
High `max-num-seqs` without an explicit capture list drops mid-batch graphs the same way (an eager
plateau of ~235 tok/s aggregate at 40 ms steps). Both failure shapes are boot artifacts, not
speculative-decoding results.

32-wide speculation is unbootable on 16GB V100s: the corrected token-level capture list ([11..352])
fails at engine init on graph plus KV-cache memory. Wide spec on this hardware needs lower GPU memory
utilization or fewer captured sizes — unexplored, and of bounded value against plain batching.

Structured speculative concurrency plateaus at ~480 tok/s aggregate at 4 streams by two routes —
k=10 (484.0) and k=15 at the M=64 verify edge (476.2 at 71.7 ms/step): verify GEMM cost is flat from
M=16 to M=64, but the 15 sequential drafter iterations plus per-step host work grow the step.
8-stream chain-MTP4 on natural text: 339.1 tok/s aggregate (default capture sizes; a lower bound).

## 5. Retraction: the ngram tau = 10.59 result

An earlier internal result — ngram drafting at k=15 measuring tau 10.59 on CSV with a flat acceptance
tail, implying ~700 tok/s single-stream if the GDN rollback overhead were removed — is retracted. The
original measurement skipped the output diff; running it shows both ngram routes (CPU and GPU
proposers) emit identically corrupted output on the struct-csv fixture (divergence at line 2,
degenerate 5-row output vs 40 clean rows). tau = 10.59 was measured on text the bug itself created —
the ngram drafter trivially predicts the repetition the corruption induced.

Clean-text replay (CPU proposer over the chain-MTP-lossless CSV reference): tau 1.06 on CSV,
0.84 on JSON — far below chain-MTP's 4.70/7.13 at k=10. Ngram is the wrong drafter for clean
structured text; chain-MTP at k <= 15 remains optimal. Root locus of the corruption: the ngram verify
path exercises mixed speculative/non-speculative GDN state handling that chain-MTP (all-spec,
exactly-k batches, byte-verified) never touches — an open fork bug, reported.

## Validation rule

Never quote a speculative-decoding throughput number without a byte-level output diff against plain
decode on the same fixture. Four results in this document failed it (deep-k flash, triton
thinking-mode prose, triton deep-k, the ngram tau above): corrupted verification inflates both
acceptance and tok/s, because the drafter trivially predicts the degenerate text the bug produces.
Greedy determinism makes the diff cheap; run it per boot for any k > 10 configuration.
