# Reproducing every headline number

Recipe for re-measuring the figures in [`README.md`](../README.md). Target box:
**4× Tesla V100-SXM2-16GB, TP4**, the **[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) 1.2.2** wheel, the
Qwen3.6-27B NVFP4 checkpoints of §1.

Two rules govern every number in `results/`:

1. **No speculative-decode throughput is quotable without a byte-diff of the
   output against plain decode at matched configuration** (§4f, §5). Acceptance
   metrics alone cannot distinguish speed from corruption — degenerate
   repetition accepts perfectly.
2. **Verify the served configuration from the boot log before you probe it**,
   and confirm it again from the probe's own metrics (§3.4). Trusting the
   intended config over the served one cross-labeled a published k A/B once.

Terminology is fixed: **lm_head** (final 5120 → vocab projection), **mtp_head**
(the MTP drafter module), **backbone** (target trunk). Bare "head" is not used
— [`terminology_audit.md`](terminology_audit.md).

## 1. Environment

### 1.1 Box

| Item | Value as recorded | Source |
|---|---|---|
| Host | `c4130-local` | [`DEPLOYMENT.md`](DEPLOYMENT.md) |
| GPUs | 4× Tesla V100-SXM2-16GB (SM70), NVLink | `DEPLOYMENT.md`, kernel bench |
| Measured device memcpy ceiling | **825 GB/s** | `kernel_bw_bench.py` (`COPY_CEIL`), `results/kernel_bandwidth_20260811.csv` |
| CUDA toolkit | 12.8 (`CUDA_HOME=/usr/local/cuda-12.8`) | all `scripts/launch_*.sh` |
| Arch list | `TORCH_CUDA_ARCH_LIST=7.0` | launch scripts + every kernel bench |

**Record at install time** (not recovered anywhere in this repo): NVIDIA driver
version, GPU BIOS/clock state (SM and memory clocks, power cap, persistence
mode), host CPU/RAM, OS and kernel version, NCCL version, ECC on/off.
Re-measure the 825 GB/s ceiling (`skinny_bench.py` prints
`measure_bandwidth_gbs()`); a materially different ceiling is a different box,
not a failed reproduction.

### 1.2 Conda env, wheel, torch

| Item | Value as recorded |
|---|---|
| Conda env | `1cat-vllm-122` |
| Engine | **[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM) 1.2.2** (wheel install, *no* source checkout) |
| Torch | **2.10.0+cu128** |
| Python | 3.12 (implied by the install path `…/envs/1cat-vllm-122/lib/python3.12/site-packages/`) |
| `tilelang` / `apache-tvm-ffi` | manually bumped to **0.1.10** — fixes the SM70 TileLang compile (upstream commit `0bb5cc4132`) |
| Client harness deps | [`requirements.txt`](../requirements.txt): `torch>=2.10`, `safetensors>=0.4`, `numpy>=1.26` |

**Record at install time:** the wheel's exact build/sha (the repo records only
"1.2.2"; `master_table.sh` captures the same coarse `vllm.__version__`), conda
version, and the resolved `tilelang` / `apache-tvm-ffi` / `nccl` /
`safetensors`. `master_table.sh` stamps a **kernel fingerprint**
(`md5sum kernels/skinny_kernels.cu | cut -c1-10`) into every
`results/master_table*.csv` row — the only per-run provenance the CSVs carry.
Launch scripts `cd ~` before exec'ing the API server, deliberately: run outside
any vLLM source checkout so the wheel's compiled extensions load.

### 1.3 Checkpoints and their provenance

Two artifacts, and the distinction matters for §4f.

**Source (NVIDIA, modelopt 0.45.0)** — `~/models/Qwen3.6-27B-NVFP4`. Presumed
`nvidia/Qwen3.6-27B-NVFP4`, downloaded 2026-08-06 01:15. Ships the **lm_head
already NVFP4-quantized**:

```
lm_head.weight         [248320, 2560]  U8        NVFP4-packed nibbles (lo-first)
lm_head.weight_scale   [248320, 320]   F8_E4M3   group-16 scales
lm_head.weight_scale_2 []              F32       global scale
lm_head.input_scale    []              F32       fp8-activation intent; unused in our W4A16
```

**Serving checkpoint (ours)** — `~/models/Qwen3.6-27B-NVFP4-CTfull`, ~19.2 GB,
compressed-tensors, converted 2026-08-06 03:07 by `convert_modelopt_to_ct.py`.

> ### ⚠ Provenance-SHA gap
> Per [`lmhead_provenance.md`](lmhead_provenance.md), **the exact HF repo
> + revision/SHA of the source download is not recoverable from disk** —
> no hub cache entry, no `_name_or_path`, no refs. A reproducer therefore
> cannot prove they have byte-identical source weights. Mitigations, in
> order of strength: (a) checksum your shards against the HF API and
> record repo + commit SHA before converting; (b) verify against the
> facts this repo *did* record — sampled MLP layers (`5.gate`,
> `30.down`, `60.up`) are **bit-equal** between CT `weight_packed` /
> `weight_scale` and the source tensors, with the global scale stored as
> its exact reciprocal (`source_g × ct_g = 1.0000`, a CT convention the
> serving kernels handle); the lm_head is likewise bf16-exact between the
> native codes and the CT BF16 rendering. Reproduce those checks offline
> before trusting any downstream number.

Two CT-conversion properties that are not bugs:

- **The lm_head was dequantized to BF16** (`[248320, 5120]`, 2.54 GB / 636 MB
  per rank at TP4) and listed in `quantization_config.ignore` (537 entries) —
  our artifact, not NVIDIA's recipe. The native 4-bit loader
  (`VLLM_SKINNY_LMHEAD_NATIVE`, §2.2) reads the original codes/scales from the
  **source** checkpoint — specifically `model-00003-of-00003.safetensors`
  (1.9 GB), the only source file needed once conversion is done.
- **FP8 deviation.** NVIDIA quantizes `self_attn` q/k/v/o and GDN
  `in_proj_qkv` / `in_proj_z` / `out_proj` at **FP8**; ours carries all
  non-ignored Linears at NVFP4 (V100 has no FP8 hardware), making the served
  model a *further-quantized derivative* on `self_attn.{q,k,v,o}_proj`
  (16 layers) and `linear_attn.out_proj` (~48 GDN layers). Launch wording:
  *"serves NVIDIA's published NVFP4 weights bit-exactly, with FP8 attention
  projections down-converted for pre-FP8 hardware."*

## 2. Patch installation

### 2.1 Deploy-by-copy convention

Wheel install, no source checkout: every change lives in `fork_patches/` as a
tracked file, applied by **copying it over the installed target once, keeping
the `.pre_*` / `.orig` backup**. Never hand-edit the installed file; never
hot-rewrite `site-packages` in place. Deep or experimental changes stay as
unified-diff `.patch` files and belong in a source checkout with an editable
install. Install root:
`~/miniconda3/envs/1cat-vllm-122/lib/python3.12/site-packages/`

| Tracked file | Copy to (under site-packages) | Backup name | What it adds |
|---|---|---|---|
| `fork_patches/marlin.py` | `vllm/model_executor/kernels/linear/nvfp4/marlin.py` | `marlin.py.orig` | Skinny-kernel dispatch shim, QPN dispatch + loader prepack, 4-bit / native lm_head policy |
| `fork_patches/gdn_attn.py` | `vllm/v1/attention/backends/` | `.pre_chainfast` | Chain-MTP GDN fast metadata build (−1.4 ms/step, byte-identical) + env-gated slot-debug instrumentation |
| `fork_patches/gpu_model_runner.py` | `vllm/v1/worker/` | `.pre_thinkonly` | Env-gated think-only draft gate (**off**; blocked on the async-scheduler contract) + slot-debug instrumentation |
| `kernels/skinny_kernels.cu` | `~/flatness-run/skinny_kernels.cu` | — | The CUDA source. Loaded/JIT-built at runtime via `VLLM_SKINNY_NVFP4_SRC`; also the source every kernel bench compiles |
| `benchmarks/*.py`, `scripts/*.sh` | `~/flatness-run/` and `~/1cat-122/` respectively | — | Probe harnesses and launchers (see §4 for the path caveat) |

Not installed, and contributing to no headline number:
`fork_patches/sm70_native_round.py` (native speculative-round executor — the
drafter chain as one CUDA graph, file-imported from `~/flatness-run/`;
**experimental**) and `fork_patches/llm_base_proposer.native_round.patch` (the
proposer hook selecting it — **reverted from the live env**, so
`VLLM_SM70_NATIVE_SPEC_ROUND` does nothing until re-applied; byte-identical to
the Python path but **inert**, drafts rejected, τ≈0, because the captured graph
does not persist the drafter's recurrent state across rounds).

Shim self-check: the first `_SELF_CHECK_CALLS = 3` eager calls compare skinny
against marlin at `_SELF_CHECK_TOL = 3e-2` and **permanently fall back to
marlin on mismatch**, logging `Skinny NVFP4 self-check FAILED`. Absence of
`Skinny NVFP4 self-check ok` in the boot log means you are measuring marlin.

### 2.2 Environment flags

| Flag | Default (shim) | Effect |
|---|---|---|
| `VLLM_SKINNY_NVFP4` | `0` | Enable the custom skinny kernels. `0` = stock Marlin. Launch scripts set `1`. |
| `VLLM_SKINNY_NVFP4_SRC` | `~/flatness-run/skinny_kernels.cu` | Path to the CUDA source that is JIT-built at load. |
| `VLLM_SKINNY_QPN` | `1` | QPN `m8n8k4` dispatch for M 4–16. Costs one prepacked weight copy per layer (+2.72 GiB/rank ⇒ needs `GMU=0.94`). |
| `VLLM_SKINNY_DROP_CT` | `0` | Free the checkpoint-native weight stash after the QPN prepack (the prepack is a byte-equal permutation), recovering the pre-QPN footprint. M1–3 then route to `gemm_qpn_simt`, M17+ to marlin. |
| `VLLM_SKINNY_MAX_M` | `64` | Upper M bound for the skinny WMMA path; above it marlin serves the GEMM (prefill). Tunable for dispatch experiments. |
| `VLLM_SKINNY_LMHEAD` | `0` | 4-bit **lm_head**. **Launch scripts default it to `1`** — so any boot without an explicit override ran the 4-bit lm_head. Off is the end-user serving policy for the *requant* packer. |
| `VLLM_SKINNY_LMHEAD_NATIVE` | `""` | `1` (or a path) loads **NVIDIA's original lm_head codes/scales** from the source checkpoint — zero requantization. Default path `~/models/Qwen3.6-27B-NVFP4/model-00003-of-00003.safetensors`. **Requires `VLLM_SKINNY_LMHEAD=1`.** This is the launch flagship lm_head config. |
| `VLLM_SKINNY_QPN_LMHEAD` | `0` | **Leave off.** Routing the logits projection through a different kernel than the drafter's M=1 path flips 4-bit near-ties — measured extraction acceptance 82% (vs 97–100) and 2/4 losslessness diffs failed. Backbone QPN is unaffected. |
| `VLLM_SM70_GDN_CHAIN_SPEC_FAST_BUILD` | launch scripts set `1` | Chain-MTP GDN fast metadata build. −1.4 ms/step, outputs byte-identical under greedy. A/B'd clean against the deep-k corruption (exonerated). |
| `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT` | set to `0` in every spec boot | Static draft vocab. Dynamic vocab forces `MNS=1` and is capture-unsafe. |
| `VLLM_1CAT_ENABLE_SM70_MTP_DEFAULTS` | `1` in `launch_qwen36_ctfull_mtp.sh`, unset in the plain script | Enables the fork's SM70 MTP defaults. |
| `VLLM_SM70_NVFP4_TURBOMIND` | `0` (ours) / `1` (reference) | Pins the scheme to the patched `MarlinNvFp4LinearKernel` vs the fork's TurboMind route. |
| `VLLM_SM70_QUANT_BACKEND` | `marlin` (ours) / `turbomind` | `forces_marlin()` ⇒ SM70 allowed. |
| `draft_sample_method=greedy` | — | **A `--speculative-config` JSON key, not an env var.** Argmax drafter proposals. This is *the acceptance fix* (the fork's SM70 default is `probabilistic`, costing 10–25 acceptance points on sampled serving). Free; keep it on in every spec profile. |

Profiler envs (`VLLM_SM70_MTP_PROFILE`, `VLLM_DFLASH_DDTREE_WORKER_PROFILE`)
make a boot **non-quotable**: per-call logging alone costs ~5 ms/step and can
drop wall throughput to ~20 tok/s. Never mix a profiling and a measurement boot.

### 2.3 Serving profiles as measured

`results/` rows were produced at `MNS=4 GMU=0.94 MBT=4096`, thinking on,
`--reasoning-parser qwen3`, `--enable-auto-tool-choice --tool-call-parser
hermes`, `VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0`,
`VLLM_SM70_GDN_CHAIN_SPEC_FAST_BUILD=1`; §4c has the full boot line. For
**current** production envelopes (MNS/GMU per profile, `DROP_CT`, NUMA pinning)
see [`DEPLOYMENT.md`](DEPLOYMENT.md).

| Profile | k | `cudagraph_capture_sizes` | Use |
|---|---|---|---|
| **flagship** | 7 | `[8,16,24,32]` | general chat, code — the default |
| math | 10 | *not recorded* — must be multiples of k+1 = 11 (the recorded k=10 boot passed no `--compilation-config` at all) | reasoning workloads |
| structured / extraction | 15 | `[16,32,64]` | JSON/CSV/verbatim, τ-rich tails only |

`GMU=0.94` is required by the QPN prepack stash; the historical sweep scripts in
`scripts/` use `0.93` from before the stash landed.

## 3. Serving lifecycle

**"Ops hardening" in [`DEPLOYMENT.md`](DEPLOYMENT.md) is the authority for every
teardown and boot rule.** Below is only what a reproducer runs inline.

### 3.1 The graceful PID-file pattern

Boot records its own PID (`echo $! > ~/1cat-122/serve.pid`; full boot lines in
§4). Stop and wait through the helper — **never** ad-hoc `kill`/`pkill`:

```bash
bash ~/flatness-run/serve_ctl.sh stop     # graceful TERM of the recorded group
bash ~/flatness-run/serve_ctl.sh wait     # poll /v1/models health
```

`stop` = SIGTERM the **recorded process group**, wait, escalate only against
**that PID** after a timeout. Force-kill sweeps (`pkill -9 -f`, and especially
`for p in $(nvidia-smi --query-compute-apps=pid …); do kill`) are prohibited:
indiscriminate, they take out unrelated work on a shared box, and they
pattern-match to intrusion behaviour.

> **Note on sweep drivers.** The internal sweep drivers that produced the
> historical ladder rows predate this rule (their teardown was a
> `pkill`/`nvidia-smi` kill loop) and are deliberately **not shipped** in this
> release. To reproduce a sweep, wrap the launch scripts in a sequential loop
> of `serve_ctl.sh stop` → boot with the arm's env → health-wait → probe →
> diff gate, per the boot envelopes in [`DEPLOYMENT.md`](DEPLOYMENT.md).

### 3.2 Hardened-boot rules

Full text and escalation commands: `DEPLOYMENT.md` "Ops hardening" rules 1–3.
Inline, per boot:

1. **Double-TERM for stubborn servers.** A TP4 vLLM group draining NCCL can
   ignore the first SIGTERM. Send a second TERM to the same recorded group;
   escalate to signal-9 *only against the recorded PID*.
2. **Never boot over occupied GPUs.** Confirm `nvidia-smi` reports **no compute
   apps and free memory on all four cards** first — otherwise the previous
   server keeps answering on `:8000` and your probe measures the old
   configuration under the new label.
3. **Verify the served k from the boot log before probing.** After `wait`:

   ```bash
   grep -E "num_speculative_tokens|speculative" ~/1cat-122/serve_skinny.log | head
   grep -E "route map: M=8 "                    ~/1cat-122/serve_skinny.log | head -2
   grep -E "Skinny NVFP4 self-check ok"         ~/1cat-122/serve_skinny.log | head -1
   grep -E "Skinny lm_head: NATIVE NVFP4"       ~/1cat-122/serve_skinny.log | head -1
   grep -E "Model loading took|Available KV cache|Maximum concurrency" \
        ~/1cat-122/serve_skinny.log | head -4
   ```

   `route map: M=<k+1>` confirms the verify GEMM runs at the width your k
   implies (M=8 for k=7, M=16 for k=15, M=11 for k=10).

### 3.3 Why: the cross-labeling incident class

A leftover server won the port race and swapped the labels on a published k
A/B; the rows at the foot of
[`results/aime_headtohead.csv`](../results/aime_headtohead.csv) carry the
correction note (*"corrected via `τ/acc% = k` identity: 4.31/0.616 = 7,
5.73/0.382 = 15"*) and survived only because τ and acceptance-% were both
recorded. Same family: corrupted spec counters (caught by closure, §5.2) and a
mislabeled lm_head config (caught by the flag record, `terminology_audit.md`).
**Assume nothing about what is serving; read it back.**

### 3.4 The K_CONFIRMED pattern

Acceptance-% = accepted/drafted = τ/k, so the served k falls out of any probe's
own metrics:

```
k_served = τ / acc%            (acc% expressed as a fraction)
```

Run it on the first probe output of every spec boot. On a match, tag the row
**`K_CONFIRMED`** in the CSV — that literal tag distinguishes the verified
replication row
(`ours-native-k7,verified-boot-seed-1001,174.8,"τ 4.29; … ; K_CONFIRMED"`) from
the pre-correction rows. On a mismatch the boot is void: stop, re-check §3.2,
re-boot. Rows without a confirmed k are not quotable.

## 4. Per-benchmark recipes

**Path caveat.** Several harnesses hard-code the box's absolute paths and need
editing (or a matching user account) on any other machine:

- `benchmarks/k_probe_decode.py` → `OUT = /home/user/flatness-run/k_sweep_matrix.csv`
- `benchmarks/k_sweep_probe.py` → same `OUT`
- `benchmarks/qpn_gen.py` → writes `/home/user/flatness-run/lad_<cell>_<tag>.txt`
- `benchmarks/aime_exact.py` → `FIXDIR = /home/user/flatness-run/ninfer_fixtures`
- kernel benches → `$HOME/flatness-run/skinny_kernels.cu`, `$HOME/flatness-run/qpn_race.cu`

All server-facing probes talk to `http://127.0.0.1:8000` and default to
served-model-name `qwen3.6-27b-nvfp4-skinny`; `PROBE_MODEL` overrides it (used
to point the same probe at the 1Cat reference arm).

### 4a. Kernel bandwidth curve — the 5.9× headline

**Claim:** `results/kernel_bandwidth_20260811.csv` — SIMT 565.2 GB/s at M=1
(69% of the 825 GB/s ceiling) vs stock Marlin's flat 94–96 GB/s.

Offline; no server. Synthetic weights in the five real Qwen3.6-27B TP4 per-rank
shapes `(K,N)`: `(1536,5120) (4352,5120) (5120,8704) (5120,4096) (5120,2048)`,
GPU 0, activation outliers injected (`x[:, ::256] = 150.0`) so the fp16
overflow path is exercised — only real-magnitude activations ever caught it.

```bash
conda activate 1cat-vllm-122
export TORCH_CUDA_ARCH_LIST=7.0 CUDA_HOME=/usr/local/cuda-12.8
python benchmarks/kernel_bw_bench.py
```

```
M,prod_kernel,eff_GBs,pct_of_copy_ceiling
1,simt,565.2,69
...
64,wmma,105.8,13
KERNEL_BW_DONE
```

- M sweeps `{1,2,3,4,5,8,11,16,24,32,64}`; the kernel per M is the **production
  dispatch** — SIMT M≤3, QPN M 4–16, WMMA M 17–64. Effective GB/s = packed
  weight bytes ÷ GEMM time over the five shapes, 200 timed iterations after 20
  warmups. Marlin is *not* re-measured: the flat 94–96 GB/s is recorded in
  `results/nvfp4_flatness_results.csv`.
- `kernel_bw_bench.py pershape` also races production QPN against the
  race-harness build on identical inputs (per-shape delta = integration loss,
  cross-shape spread = shape mix), ending `PER_SHAPE_DONE`. **Needs
  `~/flatness-run/qpn_race.cu`** (§6).
- Run first: `tests/test_real_weights.py`, `tests/test_qpn_real.py` (rel-err
  ≤ 1e-3 vs checkpoint tensors at M ∈ {5,8,11,16} with outliers), and
  `benchmarks/skinny_bench.py` (re-measures the copy ceiling, asserts
  rel < 1e-2 per shape before timing).

### 4b. Plain decode compare — the pure-kernel serving win

**Claim:** `results/plain_decode_compare.csv` — ours 86.6 tok/s (86.5–86.8
across domains) vs 1Cat AWQ+TurboMind 70.7, i.e. **1.22×**, both with **fp16
lm_head (lossless)**, MTP0, single-stream, greedy. The one cell where lm_head
precision must be forced off the script default, because the claim is "at zero
quality cost":

```bash
# Ours — plain, no speculation, fp16 lm_head
VLLM_SKINNY_LMHEAD=0 MNS=4 GMU=0.93 MBT=4096 \
setsid env … bash ~/1cat-122/launch_qwen36_ctfull.sh \
  > ~/1cat-122/serve_skinny.log 2>&1 < /dev/null &
echo $! > ~/1cat-122/serve.pid
bash ~/flatness-run/serve_ctl.sh wait
# §3.2 verification here, then:
python3 ~/flatness-run/k_probe_decode.py 0
```

- Use `launch_qwen36_ctfull.sh` (plain flagship), not the MTP script with
  speculation omitted: `launch_qwen36_ctfull_mtp.sh` auto-applies the fork's MTP
  defaults when no `--speculative-config` is given, and dynamic draft vocab then
  demands `MNS=1`.
- **`k=0`** disables the identity self-check gate in `k_probe_decode.py` (no
  speculative rounds to check) and the τ/steps deltas are legitimately zero.
  Decode is domain-flat here, hence one number with a range.
- Reference arm: same probe, `PROBE_MODEL` pointed at the 1Cat AWQ + TurboMind
  server (`VLLM_SM70_QUANT_BACKEND=turbomind`, `VLLM_SM70_NVFP4_TURBOMIND=1`,
  `VLLM_SKINNY_NVFP4=0`). **`launch_awq36_mtp.sh` is not committed** (§6);
  `scripts/launch_awq_tp4.sh` is the Qwen3.5-AWQ analogue and shows the env
  contract.

### 4c. Flagship matrix — think/nothink × 6 domains

**Claim:** `results/flagship_matrix_20260811.csv` — native NVFP4 lm_head,
`native_k7` and `native_k15` arms, decode-only, single-stream greedy, plus
partial `onecat_k4` rows.

```bash
# Boot the flagship (k=7), native lm_head
MNS=4 GMU=0.94 MBT=4096 \
VLLM_SKINNY_LMHEAD=1 VLLM_SKINNY_LMHEAD_NATIVE=1 \
VLLM_SM70_MTP_DYNAMIC_DRAFT_VOCAB_DEFAULT=0 \
VLLM_SM70_GDN_CHAIN_SPEC_FAST_BUILD=1 \
EXTRA_VLLM_ARGS='--default-chat-template-kwargs {"enable_thinking":true} --reasoning-parser qwen3 --compilation-config {"cudagraph_capture_sizes":[8,16,24,32]} --speculative-config {"method":"mtp","num_speculative_tokens":7,"draft_sample_method":"greedy","use_local_argmax_reduction":true}' \
setsid env … bash ~/1cat-122/launch_qwen36_ctfull_mtp.sh \
  > ~/1cat-122/serve_skinny.log 2>&1 < /dev/null &
echo $! > ~/1cat-122/serve.pid
bash ~/flatness-run/serve_ctl.sh wait
# §3.2 verification (expect "route map: M=8" and the NATIVE lm_head line)

# Warm pass, discarded; then the quotable pass
PROBE_CELLS=full python3 ~/flatness-run/k_probe_decode.py 7 > /dev/null 2>&1
PROBE_CELLS=full python3 ~/flatness-run/k_probe_decode.py 7
```

Repeat with `num_speculative_tokens=15` and
`cudagraph_capture_sizes:[16,32,64]` for the `native_k15` arm.

| `PROBE_CELLS` | Cells |
|---|---|
| *(unset)* | The frozen 6-cell campaign suite: `prose-nothink-1024`, `math-think-2048`, `code-nothink-1024`, `struct-json-1024`, `struct-csv-1024`, `extract-nothink-512` |
| `full` | **The flagship matrix**: all six domains in **both** thinking modes — think cells at 2048 tokens and nothink at 1024, except `extract-think-1024` / `extract-nothink-512` |
| `thinkext` | The thinking-mode extension pairs: `prose-think-2048`, `code-think-2048`, `struct-json-think-2048`, `math-nothink-1024` |

One line per cell, then the append confirmation:

```
k=7 json-nothink-1024: DECODE 228.4 tok/s  tau=5.56  28.6 ms/round  rounds=… \
  gen=…  ttft=… ms  (incl-prefill …)  CHECK[OK pred=… err=…%]
decode-only rows appended
```

Rows append to `k_sweep_matrix.csv` tagged `+qpnD`, `gen` holding decode-span
tokens.

> **⚠ Read the think cells correctly.** As the CSV header states, **think cells
> are capped at 2048 tokens, so they measure the trace-dominated phase, not full
> requests**: the reasoning trace is emitted first, and a 2048-token cap on a
> domain whose trace runs longer never reaches the payload phase. Do not quote a
> think cell as an end-to-end thinking-on rate — use the phase probe (§4d). The
> `onecat_k4` rows are deliberately partial (that arm was cut as redundant), so
> the matrix is not a complete head-to-head.

### 4d. Phase probe — trace vs payload

**Claim:** `results/phase_probe_20260811.csv` — end-to-end thinking-on rates the
capped think cells cannot give. Server: the same k=7 native-lm_head flagship as
§4c, already up.

```bash
python3 ~/flatness-run/phase_probe.py
```

```
PHASE json     gen= 7644 (natural)  trace:  5810 tok  tau= 4.56  … tok/s | \
  payload:  1834 tok  tau= 6.99  … tok/s | e2e  207.3 tok/s
...
PHASE_PROBE_DONE
```

One uncapped generation per domain (`BUDGET = 8192`, thinking on), five cells
(`prose`, `math`, `code`, `json`, `extract`). Spec metrics are snapshotted three
times — start, reasoning→content transition, end — so τ is metrics-exact **per
phase**. Two interpretation rules, both in the CSV header:

- `finish` is `natural` or `CAPPED`. The `extract` row is `CAPPED` at 8192 — its
  phase split is a floor, not a completed request.
- **Phase token counts are chunk-derived.** vLLM streams roughly one token per
  chunk, but multi-token chunks bias the split toward the trace;
  `usage.completion_tokens` anchors only the total. Correct a phase rate via
  round time when precision matters (json payload is ~270 tok/s corrected).

### 4e. AIME f01 — competition-math long reasoning

**Claim:** `results/aime_headtohead.csv` — ours k=7 QPN greedy-draft 172.8 tok/s
on `long_decode_aime26_01` (seeded ×3), 1Cat AWQ k=4 at 112.2, i.e. 1.54×; and
the verified-boot replication row 174.8 (τ 4.29, 8073 tok, `K_CONFIRMED`).

`benchmarks/aime_exact.py` replays **ninfer's verbatim AIME 2026 fixture
messages** — SHA-matched from their repo, stored in
`~/flatness-run/ninfer_fixtures/{long_decode_aime26_01,_15,_30}.json`. **The
fixtures are not committed here** (§6).

```bash
# Server: k=7 flagship, native lm_head (as §4c), K_CONFIRMED
AIME_FIXTURE=aime26_01 AIME_MAX_TOKENS=23000 \
  python3 ~/flatness-run/aime_exact.py 7 ninfer 1001
```

| Knob | Meaning |
|---|---|
| `argv[1]` | k — **labelling only**, printed and echoed in the done-line. It does not configure the server; that is what §3.4 is for. |
| `argv[2]` | sampling profile — `ninfer` (temp 0.6, top-p 0.95, top-k 20, presence 1.0), `greedy` (temp 0), or `tight` (a flat-phase-equivalent top-p 0.7 variant) |
| `argv[3]` | **seed** — passed through as the request's `seed`. Required for any acceptance or sampled-throughput claim (§5.3) |
| `AIME_MAX_TOKENS` | default `20000`; the recorded f01 k-A/B used **23000**. ninfer's own 65536 exceeds our context — per their method note, a budget-capped run is still a valid sustained-decode sample |
| `AIME_FIXTURE` | substring filter over the three fixture names; `aime26_01` selects f01 alone |

```
mode=ninfer k=7 maxtok=23000
               fixture  gen_tok decode tok/s   acc%  tok/round  pos1  pos2  pos3   3pos
 long_decode_aime26_01     8073        174.8   61.3       5.29    80    68    59   69.0  tail_boxed=True
AIME_EXACT_DONE k=7 mode=ninfer seed=1001
```

- `tail_boxed` = whether the last 120 characters contain `\boxed`, a cheap check
  that the trajectory terminated in an answer rather than rambling.
- **Only f01 is a clean apples-to-apples cell** against ninfer: their f15/f30 run
  to natural EOS at ~62k and ~47k tokens, beyond our serving context, so capped
  rows there measure a truncated regime. The CSV's ninfer figures
  (222.7 / 201.6 / 216.3) are their published MTP3 n=5 table and carry a
  correction-of-correction note — read the comment block before quoting.
- Run three seeds (committed protocol: `1001 2002 3003`), quote the mean, and
  structure each seeded arm as its own teardown → boot → warm → probe cycle
  (§3.1) so no arm inherits the previous arm's server state.

### 4f. Losslessness byte-diffs at matched configuration

**The gate that makes every other spec number quotable.** Generate the same
greedy cells from a **plain** (no-speculation) boot, then from the
**speculative** boot at otherwise identical configuration, and `cmp` the bytes.

```bash
# Boot A — plain reference, SAME lm_head config as the spec arm
bash ~/flatness-run/serve_ctl.sh stop        # then verify GPUs free (§3.2)
MNS=4 GMU=0.93 MBT=4096 <same VLLM_SKINNY_LMHEAD* as boot B> \
setsid env … bash ~/1cat-122/launch_qwen36_ctfull.sh > …/serve_skinny.log 2>&1 &
echo $! > ~/1cat-122/serve.pid ; bash ~/flatness-run/serve_ctl.sh wait
python3 ~/flatness-run/qpn_gen.py warmA > /dev/null 2>&1   # warm, discarded
python3 ~/flatness-run/qpn_gen.py plainq                   # the reference bytes

# Boot B — the speculative config under test (k=7 shown)
bash ~/flatness-run/serve_ctl.sh stop        # verify GPUs free
<k=7 flagship boot as §4c> ; bash ~/flatness-run/serve_ctl.sh wait
# §3.2 + §3.4 verification
python3 ~/flatness-run/qpn_gen.py k7q

# Diff
for cell in math json csv extract; do
  if cmp -s lad_${cell}_plainq.txt lad_${cell}_k7q.txt; then
    echo "LAD_${cell}_k7q_LOSSLESS"
  else
    echo "LAD_${cell}_k7q_DIFF_FAIL"
    diff <(head -c 800 lad_${cell}_plainq.txt) <(head -c 800 lad_${cell}_k7q.txt) | head -8
  fi
done
```

`qpn_gen.py <tag>` writes four cells — `math` (think, 2048), `json` (nothink,
1024), `csv` (nothink, 1024), `extract` (nothink, 512) — as
`lad_<cell>_<tag>.txt`, all `temperature: 0`, joining reasoning and content with
a `\x1d` separator so a trace/payload boundary shift cannot hide inside a
byte-identical concatenation.

**"Matched configuration" means both lm_head precision *and* kernel path.**
Byte-losslessness is claimable at either lm_head precision so long as the plain
reference uses the same one — but the plain decode GEMM runs at M=1 (SIMT) while
the k=7 verify GEMM runs at M=8 (QPN), different numerical paths.

Interpreting a diff, precisely:

- **Speculative decoding preserves target-model semantics by construction**
  (rejection sampling). Byte-identity is the *stronger* check and holds when
  both paths use the same numerical kernels.
- A **near-tie argmax flip** across kernel paths on high-entropy text is a
  *numerical-equivalence* finding, not speculation corruption — both
  continuations are valid samples of the same quantized model. Canonical
  example: the prose char-1735 flip ("cautious tread…" vs "cautious, measured
  steps…"), deterministic, reproduced 3×. Track separately; not errors.
- **Degenerate or repetitive output is corruption** — investigate immediately.
  Known signature: `"1.1.1.1"`-style collapse, which accepts perfectly on the
  acceptance metrics.

Recorded outcome for the native-lm_head flagship
(`results/nativehead_ab_20260811.csv`): **json and extract byte-IDENTICAL** to
the BF16-lm_head rendering; math, csv, prose diverge mid-text (`DIFF@2331`,
`DIFF@713`, `DIFF@807`) as numerics-only near-tie flips; code not captured. τ is
identical to the BF16 arm on stable-trajectory domains (code 3.81, json 5.56,
extract 6.92) at full 4-bit round speed (28.3–30.0 ms) — the finding that
retired the requantizing packer to legacy.

## 5. Methodology appendix

### 5.1 Decode-only rate

Every serving figure in `results/` is **decode-only, single-stream**, excluding
prefill and speculative round 1. `t_first` = first content (or reasoning) chunk
= end of prefill + round 1; `t_last` = final chunk.

```
decode tok/s   = (gen − (τ + 1)) / (t_last − t_first)
decode ms/round = (t_last − t_first) / (steps − 1)
```

`gen` is `usage.completion_tokens`; `τ` and `steps` come from **metrics deltas**
around the request:

```
steps = Δ vllm:spec_decode_num_drafts_total
acc   = Δ vllm:spec_decode_num_accepted_tokens_total
τ     = acc / steps
acc%  = acc / Δ vllm:spec_decode_num_draft_tokens_total
```

Subtracting `(τ+1)` removes exactly the tokens committed by the first round,
whose latency is folded into `t_first`.

### 5.2 The identity self-check

A round commits `(τ+1)` tokens in `round_ms`, so the decode rate must satisfy:

```
decode tok/s ≈ (τ + 1) × 1000 / round_ms
```

`k_probe_decode.py` computes this as `pred` and prints `CHECK[OK …]` at ≤ 5%
relative error, `CHECK[IDENTITY_FAIL …]` otherwise (skipped at k=0). **An
`IDENTITY_FAIL` invalidates the row**: the timing span and the metrics deltas
describe different work — a leaked concurrent request, a mid-request boot, or
corrupted spec counters. Companion checks:

- **Closure:** `gen ≈ (τ+1) × steps`. One boot shipped corrupted spec counters
  and was caught only by this.
- **Ceiling:** `tokens/round ≤ k+1`, always. A violation means the metrics are
  not describing the k you think is serving (→ §3.4).
- `pos1/pos2/pos3` and `3pos` are a **pos-counter artifact at the exact-k
  boundary** — at k=3 the `3pos` column is garbage; `acc%` is valid there.

### 5.3 Seeded-triplicate rule for acceptance claims

**Any acceptance or sampled-throughput claim requires seeded triplicates or
greedy pairs.** Single unseeded stochastic samples manufactured a false effect
once: a +9-point "fp32 recurrent-state" gain that seeded replication dissolved.
Unseeded run-to-run noise on these fixtures is ±3–5 tok/s — larger than most
effects worth measuring.

Protocol: three seeds per arm (`1001 2002 3003`), same fixture, same sampling
profile, quote mean ± spread; or greedy pairs, where determinism gives zero
within-config variance (greedy replication reproduced the domain matrix to 0.1
points, with byte-identical acceptance counters). Trace lengths on f01 are
**bimodal** (~5k or ~16k tokens) under stochastic sampling and acceptance is
phase-correlated, so f01 throughput is partly a trace-length lottery; capped
fixtures are the controlled comparison.

### 5.4 The k ≤ 15 verify-qlen wall

`num_speculative_tokens` is capped at **15**, and the boundary is exact: k=15 is
byte-identical-lossless on all ladder prompts; **k ≥ 16 is deterministically
degenerate**, with identical corrupt bytes across every toggle. The mechanism is
the **V100 flash-decode 16-query tile** (m16 fragment geometry) silently
overrunning at query position 17 — the verify batch width M = k+1 crossing 16.

Eliminated en route, do not re-litigate: our kernels (WMMA correctness-swept
clean through M=32 on real shapes), GDN slot provisioning (k-derived), the chain
fast-build (A/B'd — 4/4 degenerate with it off too), smallq auto-raise, the
rejection sampler, MTP step indexing. `TRITON_ATTN` is not an escape: it has its
own distinct corruption mode (prose-think collapses at k=10), and cross-backend
spec comparisons are confounded by per-workload acceptance regime flips.

A lone k=12-dirty boot was never reproduced — hence the standing recommendation
of a **boot-time output-diff gate for any k > 10 deploy**.

### 5.5 `cudagraph_capture_sizes` are token counts

Not sequence counts, not batch counts — **tokens**. The adjuster rounds each
entry to a multiple of **k+1** and **drops any entry above the list max**:

- A wrong list **starves graph capture silently** — the run falls back to eager
  for the shapes you meant to capture, and the rows are artifacts (the
  historical `+k10max` collapse rows are exactly this).
- The right list is `k+1` × the stream count you intend to serve: k=7 at `MNS=4`
  ⇒ `[8,16,24,32]`; k=15 ⇒ `[16,32,64]`.
- An over-large list OOMs at init and can leave GPU-squatting orphans — which is
  why §3.2 rule 2 exists.

### 5.6 Which lm_head configuration a row was measured under

The shim defaults `VLLM_SKINNY_LMHEAD` to `0`, but **the launch scripts default
it to `1`**. Any boot that did not set it explicitly ran the 4-bit lm_head —
including the seeded AIME triplicate and the ~27 ms k=7 rounds. Check the CSV
`config` column, which labels this (`k7-qpn-greedy-draft-4bit-lm_head`,
`ours_k7_fp16_lm_head_tok_s`, …).

The **mtp_head was fp16 in every boot of both stacks** — `mtp` is in the
ignore/not-convert lists of both our NVFP4-CTfull checkpoint and QuantTrio's AWQ
(15 mtp tensors, 0 quantized, both sides). No mtp_head quantization experiment
has ever been run; do not attribute any measured effect to mtp_head precision.

## 6. Gaps a reproducer will hit

Referenced by committed code or docs but **not in this repository**. Each must be
obtained or rebuilt before the corresponding recipe runs:

| Missing artifact | Referenced by | Blocks |
|---|---|---|
| `launch_awq36_mtp.sh` | `ab_report.md` "Reproduce" section | The Qwen3.6-AWQ + TurboMind reference arm (§4b, §4c `onecat_k4` rows). `scripts/launch_awq_tp4.sh` is the Qwen3.5 analogue and documents the env contract |
| `onecat_aime.sh` | `results/aime_headtohead.csv` source column | The 1Cat AIME arm |
| Source-checkpoint repo + revision SHA | `lmhead_provenance.md` | Proving weight identity (§1.3). Not recoverable from disk |

Additional friction, not missing but worth knowing:

- **Hard-coded `/home/user/flatness-run/` paths** in `k_probe_decode.py`,
  `k_sweep_probe.py`, `qpn_gen.py`, `aime_exact.py` (§4).
- **`fork_patches/README.md` names the dispatch shim `../marlin_patched.py`**,
  but the tracked file is `fork_patches/marlin.py`. Same content, stale path.
- **Sweep drivers are not shipped** (§3.1): the internal drivers that produced
  the historical ladder rows used a prohibited force-kill teardown. Rebuild a
  sweep as a sequential `serve_ctl.sh stop` → boot → warm → probe → diff-gate
  loop over the boot envelopes in [`DEPLOYMENT.md`](DEPLOYMENT.md).
- **No committed driver reproduces `flagship_matrix_20260811.csv` end-to-end** —
  §4c reconstructs it from the launch config plus `PROBE_CELLS=full`, but the
  two-arm (k=7/k=15) boot chain that produced the CSV was not committed.
- **`master_table.sh` gates on `test_real_weights.py` and
  `test_skinny_integration.py` passing** before it measures anything. Keep that
  pattern; those two tests must be runnable on the target box (they read real
  checkpoint tensors).
