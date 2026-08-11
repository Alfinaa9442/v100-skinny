# Decode residual profile — Qwen3.5-27B-NVFP4, TP4, CUDA graphs

Per-step GPU cost breakdown for steady-state decode. Source: nsys
`--cuda-graph-trace=node`, 60 s steady-state window, single-rank trace,
2,244 decode steps (anchored by lm_head instance count; SIMT kernel
instance counts match the model's 192 R=1 + 64 R=2 projection calls per
token exactly). Kernel under test: TM-decoder default, R=2 short-K,
K%128 SIMT (kernel-source md5 recorded with the run).

| bucket | GPU-ms/token/rank | share |
|---|---:|---:|
| custom W4A16 GEMM (SIMT R1 5.22 + R2 0.78 + WMMA 0.03) | 6.04 | 51% |
| TP all-reduce (custom 1-stage, ~129 calls/token) | 1.85 | 16% |
| lm_head fp16 GEMM (CUTLASS s161616, 904 us fixed) | 0.90 | 8% |
| norms / fused elementwise (Triton) | ~0.9 | 8% |
| attention decode (Flash-V100 XQA, 16 layers) | 0.85 | 7% |
| GDN decode + causal conv (48 layers) | ~0.7 | 6% |
| sampler (2x softmax over 248k vocab, sort, argmax) | ~0.4 | 3% |
| **total GPU-busy** | **~11.8** | |

Consistency check: unprofiled TPOT ~11 ms/token, so decode is
~GPU-bound and the ledger accounts for the whole token budget.

Implications:
- In-server GEMM effective bandwidth is 596 GB/s (3.6 GB/rank /
  6.04 ms) vs 648 GB/s in the microbenchmark: the gap is the real
  layer-shape mix.
- Largest residuals: all-reduce 1.85 ms (fusion/overlap candidate),
  lm_head 0.90 ms (4-bit quantization candidate, size confirmed), double
  softmax in the sampler 0.29 ms (the greedy path should not need two).
