"""Exact-equivalence gate: fused gemm_simt_argmax vs gemm_simt+argmax.

Real lm_head geometry (62080x5120 per-rank shard), random NVFP4-packed
weights, 2000 random hidden states + 200 adversarial near-tie rows
(duplicated max columns). PASS requires exact index equality everywhere
after applying the lowest-index tie rule to both paths.
"""
import os
import torch

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
from torch.utils.cpp_extension import load  # noqa: E402

dev = "cuda:0"
torch.manual_seed(7)
HOME = os.path.expanduser("~")
ext = load(name="skinny_nvfp4_v11",
           sources=[f"{HOME}/flatness-run/skinny_kernels.cu"],
           extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                              "-gencode=arch=compute_70,code=sm_70"],
           verbose=False)
print("extension built (fused variant compiled)")

N, K = 62080, 5120
codes = torch.randint(0, 256, (N, K // 2), dtype=torch.uint8, device=dev)
scales = torch.randint(48, 90, (N, K // 16), dtype=torch.uint8, device=dev)
GS = 1e-2

def reference(x):
    y = ext.gemm_simt(x, codes, scales, GS)[0]          # half logits
    mx = y.max()
    return int((y == mx).nonzero().min().item())

def fused(x):
    bv, bi = ext.gemm_simt_argmax(x, codes, scales, GS)
    mx = bv.max()
    blk = int((bv == mx).nonzero().min().item())
    return int(bi[blk].item())

mism = 0
for t in range(2000):
    x = (torch.randn(1, K, dtype=torch.float16, device=dev) * 0.05)
    if t % 10 == 0:
        x[0, ::256] = 1.5   # outlier pattern
    a, b = reference(x), fused(x)
    if a != b:
        mism += 1
        if mism <= 3:
            print(f"MISMATCH t={t}: ref {a} fused {b}")
print(f"random rows: {2000 - mism}/2000 exact")

# adversarial ties: copy the argmax row's weights to another row so two
# columns produce identical half logits; both paths must pick the lower.
tie_mism = 0
for t in range(200):
    x = torch.randn(1, K, dtype=torch.float16, device=dev) * 0.05
    a = reference(x)
    dst = (a + 12345) % N
    codes[dst] = codes[a]
    scales[dst] = scales[a]
    ra, rb = reference(x), fused(x)
    if ra != rb:
        tie_mism += 1
        if tie_mism <= 3:
            print(f"TIE MISMATCH t={t}: ref {ra} fused {rb} (dup {a}->{dst})")
    # restore
    codes[dst] = torch.randint(0, 256, (K // 2,), dtype=torch.uint8, device=dev)
    scales[dst] = torch.randint(48, 90, (K // 16,), dtype=torch.uint8, device=dev)
print(f"tie rows: {200 - tie_mism}/200 exact")
print("EQUIV_PASS" if mism == 0 and tie_mism == 0 else "EQUIV_FAIL")
