# Deployment

Target: `c4130-local` (4× Tesla V100-SXM2-16GB), conda env `1cat-vllm-122`
([1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) 1.2.2 wheel, torch
2.10.0+cu128, tilelang/apache-tvm-ffi bumped to 0.1.10 for SM70 TileLang
compile).

| Checkpoint | Path | Role |
|---|---|---|
| Serving | `~/models/Qwen3.6-27B-NVFP4-CTfull` | the served weights |
| Source | `~/models/Qwen3.6-27B-NVFP4` (NVIDIA modelopt 0.45.0) | conversion input, **and** a runtime dependency for one shard: the [native lm_head](#native-nvfp4-lm_head-flagship) reads its codes from `model-00003-of-00003.safetensors` (1.9 GB) at load |

After conversion, only that one source shard is needed to serve — shards 1–2
(18.6 GB) are reclaimable, taking the resident footprint from ~41 GB to ~22 GB.
Point `VLLM_SKINNY_LMHEAD_NATIVE` at an explicit shard path instead of `1` if
you move or rename it. Re-running the conversion needs the full source again.

## Serving lifecycle

Drive the server through the PID-file helper — **never** ad-hoc `kill`/`pkill`:

```bash
bash ~/flatness-run/serve_ctl.sh stop     # graceful TERM of the recorded group
bash ~/flatness-run/serve_ctl.sh wait     # poll /v1/models health
```

Boot records its own PID:

```bash
setsid env <ENV VARS> bash ~/1cat-122/launch_qwen36_ctfull_mtp.sh \
  > ~/1cat-122/serve_skinny.log 2>&1 < /dev/null &
echo $! > ~/1cat-122/serve.pid
```

Teardown and boot preconditions are non-negotiable — see
[Ops hardening](#ops-hardening).

## Production launch config (k=7 flagship)

**Flagship = single-stream (MNS=1)**: k=7 chain-MTP, greedy drafter, thinking
on, **native NVFP4 lm_head**, captures `[8,16]`, `GMU=0.93`. The served config
is byte-identically the benchmarked config. Concurrent requests queue (fine
for single-user/OpenWebUI). The **multi-user variant** is `MNS=4`, captures
`[8,16,24,32]`, `GMU=0.90`: aggregate throughput up, per-request speed down,
batch band on Marlin/WMMA.

```
VLLM_SKINNY_NVFP4=1
VLLM_SKINNY_QPN=1
VLLM_SKINNY_LMHEAD=1
VLLM_SKINNY_LMHEAD_NATIVE=1
VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0
VLLM_SM70_GDN_CHAIN_SPEC_FAST_BUILD=1
MNS=1 GMU=0.93 MBT=4096   # flagship; MNS=4 multi-user variant uses GMU=0.90
EXTRA_VLLM_ARGS:
  --default-chat-template-kwargs {"enable_thinking":true}
  --reasoning-parser qwen3
  --enable-auto-tool-choice --tool-call-parser hermes
  --compilation-config {"cudagraph_capture_sizes":[8,16]}   # flagship; multi-user variant uses [8,16,24,32]
  --speculative-config {"method":"mtp","num_speculative_tokens":7,
                        "draft_sample_method":"greedy",
                        "use_local_argmax_reduction":true}
```

- Capture sizes are **tokens**, multiples of `k+1`. Changing k means changing
  the list — see [Hard rules](#hard-rules) and the
  [profiles table](#per-domain-serving-profiles).
- **NUMA pinning is part of the launch config** (2026-08-12): both launch
  scripts prefix the server with `numactl --cpunodebind=0 --membind=0`. All
  four GPUs sit on socket 0; unpinned boots re-roll thread/page placement for
  a ±3% round-time lottery (`docs/hardware_audit.md` #1–#3). `SKINNY_NUMA=0`
  disables the pin for experiments. Reboot-durable box prerequisites:
  `vllm-box-tuning.service` (C3/C6 disabled, app clocks 877/1530) and
  `/etc/sysctl.d/99-vllm-latency.conf` (`kernel.numa_balancing=0`,
  `vm.swappiness=10`). Deeper C-state parking measured and rejected
  (+0.1 ms/round). Placement is re-checked by the
  [post-boot gate](#3-verify-what-the-server-actually-served).

### Memory envelopes — GMU is per-profile, not global

| Profile | MNS | GMU | Captures | Envelope note |
|---|---|---|---|---|
| chat / multi-user | 4 | **0.90** | `[8,16,24,32]` | the native lm_head loads **lazily** on the first logits call — after the profiler commits the pool and graphs take workspace — so its ~20 MiB scale tensor OOMs rank 3 at 0.93+ (hit twice, 2026-08-11). Fix queued for v1.0.1: eager-load at weight-load time. Until then, run one generation after every boot to prove the lazy load survives |
| flagship | 1 | 0.93 | `[8,16]` | — |
| spec-bench k=7 | 1 | 0.93 + `DROP_CT=1` | `[8,16]` | the QPN prepack stash costs one extra weight-sized copy per layer (~2.7 GiB/rank). That used to mean `GMU=0.94`; since **2026-08-12** `VLLM_SKINNY_DROP_CT=1` frees the CT stash to fund the prepack at 0.93 |
| spec-bench k=15 | 1 | 0.93 + `DROP_CT=1` | `[16,32,64]` | the old CT-present envelope (`MNS=4 GMU=0.94`, the flagship-matrix sitting) **no longer boots** in the qpn1/native-head era (OOM at graph capture). Its extract cell reproduces at 366.0 vs the matrix era's 381.9 — DROP_CT bisected and exonerated, the delta is era-level |

### Native NVFP4 lm_head (flagship)

`VLLM_SKINNY_LMHEAD=1 VLLM_SKINNY_LMHEAD_NATIVE=1` loads **NVIDIA's original
lm_head codes and scales** from the modelopt source checkpoint, per rank, with
**zero requantization** — the published model's lm_head bits served through
the skinny kernel at 4-bit bandwidth.

Why it is the flagship (`results/nativehead_ab_20260811.csv`,
`docs/lmhead_provenance.md`): the old 4-bit lm_head penalty (92% worst-case
top-1 agreement, ~0.005 nats KLD, +2–3 τ pts) was a **double-quantization
artifact** — the runtime packer re-derived amax scales from a BF16 rendering
of values already NVFP4. Loading the original codes removes it: at k=7, τ
matches the BF16-rendering arm on stable-trajectory domains (code 3.81, json
5.56, extract 6.92) at full 4-bit round time (28.3–30.0 ms), byte-identical on
json and extract, with prose/math/csv diverging mid-text as numerics-only
near-tie flips.

**Path convention.** `VLLM_SKINNY_LMHEAD_NATIVE=1` resolves to the default:

```
~/models/Qwen3.6-27B-NVFP4/model-00003-of-00003.safetensors
```

i.e. the serving dir minus the `-CTfull` suffix, **last shard** — the one
holding `lm_head.weight` (U8 packed nibbles), `lm_head.weight_scale`
(F8_E4M3, group-16), `lm_head.weight_scale_2` (F32 global). Each rank slices
rows `[rank·n, (rank+1)·n)`; `VLLM_SKINNY_LMHEAD_NATIVE=<path>` selects a
different shard.

| Condition | Behaviour | Action |
|---|---|---|
| Healthy load | `Skinny lm_head: NATIVE NVFP4 codes loaded (zero requant), rows …` — one per rank | verify on every boot ([post-boot gate](#3-verify-what-the-server-actually-served)) |
| Rank row slice exceeds the source row count (padded vocab, wrong shard, wrong checkpoint) | logs `falling back to requant pack` and **silently reverts to the legacy requantizing packer** — you keep serving, at the double-quantization tax, believing you are native | treat as a boot failure: abort, fix the path, reboot |
| Source file missing or unreadable (incl. a reclaimed source checkpoint) | load raises instead of degrading; the first logits call fails loudly | fix the path |

The source checkpoint's HF repo + revision SHA is unrecovered — record repo +
commit SHA for every model artifact going forward.

**Numerical cross-check arm.** `VLLM_SKINNY_LMHEAD=0` renders the same NVIDIA
values in BF16, everything else identical — the reference against which
native-lm_head output is byte-diffed. Not a capacity fallback: use it to
validate a new build, checkpoint, or unexplained output change, never as a
silent production substitute (~5% decode, ~0.7 ms/token). The requantizing
packer (`LMHEAD=1` without `LMHEAD_NATIVE`) is **legacy** — kept only for
comparison against frozen historical rows.

## Environment flags

| Flag | Effect |
|---|---|
| `VLLM_SKINNY_NVFP4=1` | enable the custom skinny kernels (else stock Marlin) |
| `VLLM_SKINNY_QPN=1` | QPN m8n8k4 dispatch for M 4–16 (default on); costs one prepacked weight copy per layer |
| `VLLM_SKINNY_DROP_CT=1` | free the checkpoint-native weight stash after prepack (recover pre-QPN footprint) |
| `VLLM_SKINNY_MAX_M=64` | upper M bound for the skinny WMMA path |
| `VLLM_SKINNY_LMHEAD=1` | route the lm_head through the skinny 4-bit path (**required** for the native lm_head) |
| `VLLM_SKINNY_LMHEAD_NATIVE=1` | load NVIDIA's original lm_head codes/scales, zero requantization — **the flagship**. `=<path>` overrides the default source shard. Needs `VLLM_SKINNY_LMHEAD=1`. Without it, `LMHEAD=1` means the legacy requant packer |
| `VLLM_SKINNY_QPN_LMHEAD=0` | keep off: routing lm_head logits through a different kernel than the drafter's M=1 path flips near-ties (extraction acceptance 82% vs 97–100, 2/4 losslessness diffs failed). Backbone QPN is unaffected |
| `VLLM_SM70_GDN_CHAIN_SPEC_FAST_BUILD=1` | chain-MTP GDN fast metadata build (−1.4 ms/step, byte-identical) |
| `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0` | mandatory with MNS>1 (dynamic draft vocab refuses concurrency) |
| `draft_sample_method=greedy` | argmax drafter proposals — the acceptance fix; free, keep on in every profile |

## Per-domain serving profiles

From `results/flagship_matrix_20260811.csv` (native NVFP4 lm_head,
single-stream greedy, decode-only) and `results/phase_probe_20260811.csv`.
**k=7 is the general default**; only json and extract earn the deeper round.

| Domain | k | Thinking | Capture sizes | Decode tok/s | τ | Round ms |
|---|---|---|---|---|---|---|
| prose | 7 | **on** | `[8,16,24,32]` | 96.4 | 1.84 | 29.5 |
| math | 7 | either | `[8,16,24,32]` | 196.6 off / 183.6 on | 4.48 / 4.40 | 27.7 / 29.4 |
| code | 7 | **off** preferred | `[8,16,24,32]` | 169.1 off / 122.1 on | 3.81 / 2.57 | 28.4 |
| csv | 7 | **off** preferred | `[8,16,24,32]` | 200.4 off / 169.3 on | 4.72 / 3.93 | 28.4 |
| json | **10–15** | **off** | `[16,32,64]` at k=15 | 243.4 off (k=15) | 8.83 | 40.1 |
| extract | **15** | **off** | `[16,32,64]` | 366.0 off | 14.61 | 42.4 |

- **prose is the one domain thinking makes faster** (96.4 on vs 82.0 off, τ
  1.84 vs 1.33) — the planning trace is more predictable than freeform prose.
  Keep thinking on for chat.
- **math is mode-indifferent**: thinking costs ~5% (183.6 vs 196.6 at k=7;
  155.0 vs 163.5 at k=15).
- **code and csv stay at k=7**, thinking off (+39% and +18%).
- **json and extract alone clear the deeper-round break-even**: k=15 pays iff
  `τ+1` grows ≥1.39× (round 39.8 vs 28.5 ms). json-off grows 1.49×,
  extract-off 1.97×; csv 1.33×, math 1.19×, code 1.13×, prose 1.02× all fall
  short and are *slower* at k=15. k=10 is the shallower structured option when
  the k=15 memory envelope or first-token latency is inconvenient.

### Product guidance: disable thinking on structured and extraction endpoints

The thinking tax **grows with speculation depth** — the reasoning trace is
exactly the low-acceptance text a deep draft chain cannot ride:

| Domain | Thinking tax at k=7 | **at k=15** |
|---|---|---|
| json | −31% | **−46%** |
| extract | −18% | **−41%** |
| code | −28% | **−36%** |
| csv | −16% | **−34%** |

On JSON/CSV/extraction/tool-output endpoints, **turn thinking off**. Set it
per endpoint with
`--default-chat-template-kwargs {"enable_thinking":false}`, per request with
`"chat_template_kwargs": {"enable_thinking": false}`, or per chat with a
`/no_think` prefix.

### Thinking-on end-to-end rates (uncapped)

Matrix think cells are **capped at 2048 tokens** and measure the
trace-dominated phase, not a full request. For thinking-on endpoints quote
`results/phase_probe_20260811.csv` instead (k=7 native lm_head, uncapped 8192
budget, e2e = generated tokens over the full span):

| Domain | e2e tok/s | trace τ | payload τ | Finish |
|---|---|---|---|---|
| prose | 91.1 | 2.11 | 1.47 | natural |
| math | 188.7 | 4.40 | 5.02 | natural |
| code | 142.6 | 2.61 | 3.69 | natural |
| json | 207.3 | 4.56 | 6.99 | natural |
| extract | 249.3 | 5.29 | 6.96 | capped at 8192 |

The phase split is chunk-derived and biased toward the trace (round-time
-corrected json payload is ~270 tok/s): treat τ as exact, the split as
indicative.

## Ops hardening

Every rule below is the residue of an incident.

### 1. Stubborn-server teardown: TERM, wait, TERM, wait, then targeted KILL

A draining server can survive the first TERM (in-flight requests, CUDA graph
teardown, NCCL shutdown). Escalate in order, only ever against the recorded
PID:

```bash
PIDF=~/1cat-122/serve.pid
down() {  # wait up to $1 seconds for the recorded PID to exit
  for _ in $(seq 1 "$1"); do kill -0 "$(cat $PIDF)" 2>/dev/null || return 0; sleep 1; done
  return 1
}

bash ~/flatness-run/serve_ctl.sh stop   # TERM #1 to the recorded group
down 60 && exit 0
bash ~/flatness-run/serve_ctl.sh stop   # TERM #2 — expected on a draining server
down 60 && exit 0
kill -9 "$(cat $PIDF)"                  # last resort, THAT pid only
```

**Never** `pkill -9 -f "vllm[.]entrypoints"`; **never** sweep
`nvidia-smi --query-compute-apps=pid | xargs kill`. Mass termination takes out
unrelated work on the box and pattern-matches to intrusion behaviour. Two
TERMs and one targeted KILL have cleared every server so far.

### 2. Never boot over occupied GPUs — abort instead

```bash
nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader
# non-empty  -> ABORT. Do not boot. Finish the teardown first.
```

A boot onto partially-occupied GPUs does not fail cleanly: it either OOMs at
init leaving a *second* set of squatting ranks, or — the expensive case — the
new server never binds the port and **your probe silently measures the old
server**. That is the cross-labeled A/B class: a whole k-sweep attributed to
the wrong depth, recovered only via `τ / acc% = k`
(`results/aime_headtohead.csv`).

### 3. Verify what the server actually served

After **every** boot, before trusting any measurement:

```bash
# served speculative depth — must equal the intended k
grep -o 'num_speculative_tokens[^,}]*' ~/1cat-122/serve_skinny.log | head -1
# native lm_head loaded (one line per rank that logs to this stream; ≥1 required)
grep -c 'NATIVE NVFP4 codes loaded' ~/1cat-122/serve_skinny.log
# and NO silent downgrade to the requantizing packer
grep -q 'falling back to requant pack' ~/1cat-122/serve_skinny.log && echo LMHEAD_FALLBACK_ABORT
```

Record the outcome as a `K_CONFIRMED` marker in the result row (the convention
in `results/aime_headtohead.csv`). Only rows carrying it are quotable.

The lm_head line is emitted lazily and should appear during the startup
profiling forward; if it has not, warm the server with one request and
re-check rather than assuming.

Two probe-metric cross-checks catch a mislabeled boot even if the log rotated:

- `τ / acc% = k` (e.g. 4.31/0.616 = 7, 5.73/0.382 = 15).
- tokens per round ≤ `k+1`, and closure `gen == (τ+1) × steps`. One boot
  shipped corrupted spec counters and was caught only by closure.

### 4. Success-gated completion markers

A `DONE` echo must **assert its measurement row exists**, not merely that
control flow reached the end of the script — unconditional markers are how an
empty sweep gets recorded as a finished one:

```bash
done_row() {  # done_row <marker> <csv> <row-pattern>
  if grep -q "$3" "$2"; then
    echo "$1"
  else
    echo "${1}_FAILED_NO_ROW ($3 absent from $2)"
    return 1
  fi
}
done_row EXT_K15_DONE ~/flatness-run/k_sweep_matrix.csv '^15,extraction,'
```

Gate boot markers on §3, not `curl /health` alone: health returns 200 from the
*wrong* server just as happily.

### 5. Quote only compile-cache-warm boots, and watch the cache's disk

A boot whose config has no `~/.cache/vllm/torch_compile_cache` entry compiles
on first use, and **fresh-compile boots corrupt the spec-round counters**
(rounds double-count, round_ms halves; τ and wall-clock decode stay correct;
first requests carry multi-second compile TTFTs). Reproduced twice on
2026-08-12, both caught by the `CHECK` closure identity. Warm the config with
one throwaway boot-and-probe cycle before quoting, or verify `CHECK[OK]` on
every row.

The cache grows without bound (~0.6 GB per new config × 93 configs = 60 GB on
2026-08-12, which filled the root disk mid-boot). Clear it periodically:
`rm -rf ~/.cache/vllm/torch_compile_cache` costs one recompile per active
config.

### 6. Never edit a running bash script's file

Bash re-reads the script from a byte offset as it executes, and already-parsed
loop conditions live in the running shell's memory — editing does nothing at
best, corrupts the parse mid-run at worst (three deadlocks on 2026-08-11).
Only correct sequence: **stop the run → edit the file → relaunch**.

### 7. Sequential driver scripts, not pgrep-chained watchers

One driver script per pipeline, stages inline and in order. Do **not** chain
stages with background watchers spinning on `while pgrep -f "<other>.s[h]"`:
reordering a stage then requires kill+edit+relaunch of every party, and the
pattern deadlocked three times in one day. When ssh is unreliable, `scp` an
idempotent script and run it with a single short command — never a
multi-statement inline ssh for state changes.

Standing constraints: no in-place edits to installed packages (fork changes
live in `fork_patches/`, applied by copy with a retained backup), and no
boot-churn — deliberate, reviewable steps over rapid patch-boot-test loops on
the live server.

## Hard rules

- Never quote a spec-decode throughput without an output diff vs plain decode
  at matched configuration. Acceptance metrics alone can't distinguish speed
  from corruption — degenerate repetition accepts perfectly.
- Interpret the diff precisely. Spec decoding preserves target-model semantics
  by construction; byte-identity is the stronger check and holds when both
  paths use the same numerical kernels. A near-tie argmax flip across kernel
  paths (M=1 decode GEMM vs M=8+ verify GEMM), or between the native and
  BF16-rendering lm_head, is **numerical equivalence**, not corruption — track
  it as such (`docs/terminology_audit.md`, `docs/lmhead_provenance.md`).
  Degenerate/repetitive output *is* corruption; investigate immediately.
- Losslessness is claimable only against a reference at **matched lm_head
  config**. The native NVFP4 lm_head's reference arm is
  `VLLM_SKINNY_LMHEAD=0`.
- Acceptance claims need seeded triplicates or greedy pairs — single
  stochastic samples manufactured a false effect once.
- `cudagraph_capture_sizes` are **token** counts; the adjuster rounds to
  multiples of k+1 and drops entries above the list max. A wrong list starves
  graph capture silently and the run plateaus in eager mode.
- Cap `num_speculative_tokens ≤ 15` (V100 flash-decode 16-query tile; k≥16 is
  deterministically degenerate).
- Terminology: `lm_head` (final vocab projection), `mtp_head`/drafter
  (speculative proposer), `backbone` (target trunk). Bare "head" is banned in
  every table, note, and attribution.
