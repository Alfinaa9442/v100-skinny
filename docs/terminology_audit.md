# Terminology: lm_head, mtp_head, backbone

Fixed vocabulary for the three model components, and the numerical-equivalence finding category.

## Vocabulary

```
lm_head            = target model's final vocab projection (5120 → 248320)
mtp_head / drafter = speculative MTP module proposing future tokens
backbone           = target model trunk
```

Bare "head" is banned in all tables, notes, and attributions: it has silently
aliased lm_head and mtp_head, producing causal stories built on the wrong referent.

## Numerical-equivalence findings

Cross-kernel near-tie argmax divergences are their own finding category: both
outputs are valid samples of the same weights evaluated under different GEMM
reduction orders. They are neither speculation errors nor corruption, and are
tracked separately from both. Canonical example: prose spec-vs-plain byte-diff at
matched fp16 lm_head diverges at a single near-tie argmax flip at char 1735
("cautious tread…" vs "cautious, measured steps…"), both continuations coherent —
M=8 QPN verify vs M=1 SIMT plain evaluating near-tied logits. Speculative decoding
preserves target-model semantics by construction; byte-identical equivalence holds
wherever plain and speculative execution were diffed at matched config and kernel
path.

## Ground truth

- `VLLM_SKINNY_LMHEAD` toggles the lm_head only; launch scripts default it to 1
  (4-bit lm_head).
- The mtp_head is fp16 in every configuration of both stacks: the 15 `mtp` tensors
  sit in the ignore/not-convert lists of both our NVFP4 checkpoint and QuantTrio's
  AWQ (0 quantized on either side). No mtp_head quantization has ever been run.

## Relabeled findings

| Claim | Verdict |
|---|---|
| fp16-vs-4bit lm_head round-time delta | lm_head explains ~4.8 ms of the ~6–7 ms delta by cost model (≈8 lm_head calls/round: 7 drafter samplings + 1 verify); ~1–2 ms unattributed; mtp_head contributed nothing (constant fp16). |
| 4-bit lm_head plain-decode gain | Stands: ~+5% (91.1 vs 86.6). |
| +2–3 pt acceptance tax | Stands: lm_head tax, measured via `VLLM_SKINNY_LMHEAD` A/B. |
| Taxonomy | lm_head repack (same values, QPN format) = pure speed; 4-bit lm_head = speed + small τ tax; mtp_head quantization = speed-vs-acceptance experiment (target distribution untouched by rejection sampling, τ may move) — never run. |
| Checkpoint asymmetry (their drafter fp16, ours quantized) | Refuted by inspection: both fp16. The prose τ gap is backbone-side. |

## Config labels

- Flagship-vs-flagship and plain-decode tables: fp16 lm_head on both sides.
- AIME comparison (results/aime_headtohead.csv): ours 4-bit lm_head, 1Cat fp16
  lm_head (AWQ never quantizes it). Not lm_head-matched; ours carries the small
  lm_head τ tax and its speed benefit.
- "Lossless" (speculative ≡ plain, byte-identical) is claimable at either lm_head
  precision, provided the plain reference uses the same lm_head config.
