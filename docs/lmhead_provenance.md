# lm_head provenance

Every layer NVIDIA published as NVFP4 is served bit-native: backbone MLPs bit-equal
to the source checkpoint, lm_head from NVIDIA's own 4-bit codes and scales. The sole
deviation — FP8 attention/GDN projections down-converted to NVFP4 (Volta has no FP8
hardware) — is removable by the W8A16 in-kernel decode design below.

## The discovery

The source checkpoint (Qwen3.6-27B-NVFP4, modelopt 0.45.0) stores:

```
lm_head.weight         [248320, 2560]  U8       (NVFP4-packed nibbles)
lm_head.weight_scale   [248320, 320]   F8_E4M3  (group-16 scales)
lm_head.weight_scale_2 []              F32      (global scale)
lm_head.input_scale    []              F32      (fp8-activation intent; unused in W4A16)
```

Our compressed-tensors conversion dequantized this to BF16 `lm_head.weight
[248320, 5120]`, placing lm_head in `quantization_config.ignore` — our conversion's
artifact, not NVIDIA's recipe (Qwen3.5's NVFP4 checkpoint excluded lm_head; 3.6's
does not). The runtime 4-bit lm_head then re-quantized those values with a different
packer, re-deriving amax scales: double quantization — the source of most of the
measured 4-bit lm_head penalty (92% worst-case top-1, ~0.005 nats KLD, +2–3 τ pts).
Vocab is 248320 × 5120: BF16 lm_head 2.54 GB (636 MB/rank at TP4); 4-bit ~179 MB/rank.

## A/B: native lm_head vs BF16 rendering (results/nativehead_ab_20260811.csv)

Loading NVIDIA's original codes+scales through the skinny kernel (native loader),
k=7: τ identical to the BF16 arm on stable-trajectory domains (code 3.81, json 5.56,
extract 6.92 — the requant path's τ distortions are gone); rounds at full 4-bit
speed (28.3–30.0 ms); json/extract byte-identical to the BF16 rendering;
math/csv/prose diverge mid-text as numerics-only near-tie argmax flips (same
weights, different GEMM reduction order — numerical-equivalence findings per
terminology_audit.md, not quantization error). Native 4-bit lm_head is the flagship
config; the BF16 rendering remains the cross-check arm; the requant packer is retired.

## Backbone verification

Sampled MLP layers (5.gate / 30.down / 60.up): compressed-tensors `weight_packed` and
`weight_scale` are bit-equal to the source tensors; the global scale is stored as its
exact reciprocal (source_g × ct_g = 1.0000 on all samples; handled by the serving
kernels). The sole deviation: the source quantizes self_attn q/k/v/o (16
layers) and `linear_attn.out_proj` in the ~48 GDN layers at FP8 (E4M3); our
conversion carries them at NVFP4 (GDN in_proj_a/b and conv1d protected) — a
further-quantized derivative on those projections, not a re-encoding; plausibly
relevant to the prose backbone-agreement gap vs AWQ, whose recipe also protects q/k/v.

## W8A16: removing the deviation (design)

Keep NVIDIA's FP8 bytes in VRAM and decode e4m3→fp16 in-kernel, exactly the W4A16
pattern. Source layout: `weight` F8_E4M3, `weight_scale` F32 scalar (per-tensor),
`input_scale` ignored (activations stay fp16). Kernels `skinny_fp8_{simt,wmma,qpn}`,
same skeletons as the NVFP4 family. Decoder chosen by benchmark — constant memory
serializes on divergent per-lane addresses. Candidates: (1) bitwise e4m3→fp16
(expected winner): `h = ((b&0x80)<<8) | ((b&0x7f)<<7)` reinterpreted as half =
value × 2⁻⁸, uniform for normals and subnormals; the 2⁸ folds into the writeback
scalar (`scale' = 256 × weight_scale`); NaN encodings (0x7F/0xFF) assert-checked at
load; (2) 512 B shared-memory LUT (divergent-safe; possible bank conflicts); (3)
constant-memory LUT control. Per-tensor scale applies once at accumulator writeback
(A(sW) = s(AW)); if later fused, scale before any bias/nonlinearity (moot — these
projections are bias-free in Qwen3). Loader: native-loader pattern; TP
column-parallel q/k/v, row-parallel o_proj/out_proj. Cost: ~3.3B params of FP8
surface; +~0.4 GB/rank vs NVFP4; +78% weight bytes on those layers ≈ −2–3% plain
decode. Payoff: full bit-nativeness plus the τ-recovery hypothesis on prose/code.

## Claim wording

Current: "serves NVIDIA's published NVFP4 weights bit-exactly, with FP8 attention
projections down-converted for pre-FP8 hardware." Post-W8A16: "serves the original
published weight encodings without requantization". Execution is still not native
FP8: activations run fp16 with Volta fp32 accumulation rather than the intended FP8
arithmetic — a stronger precision choice, numerically different.

## Provenance gap

The exact HF repo + revision SHA of the source download is not recoverable from disk
(presumed nvidia/Qwen3.6-27B-NVFP4; no hub cache entry, no `_name_or_path`); to be
re-derived by checksumming shards against the HF API.
