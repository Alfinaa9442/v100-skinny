#!/usr/bin/env python3
"""
Gate 2: custom skinny NVFP4 kernel vs the marlin path's 96 GB/s.

Packs random FP16 weights into our own NVFP4 layout (e2m1 codes + fp8-e4m3
group-16 scales + fp32 global; 0.5625 bytes/weight, identical density to
the marlin measurement), validates both kernels against a dequantised
reference matmul, then runs the exact Gate 1 harness: 28 layers x four
shapes per pass, M in {1, 8, 16, 24, 32, 64}, 5 warmup + 30 timed
iterations, effective GB/s = packed bytes / pass time.

Run:  python skinny_bench.py
Out:  skinny_flatness_results.csv + summary on stdout.
"""

import csv
import os

import torch
from torch.utils.cpp_extension import load

assert torch.cuda.is_available()
dev = torch.device("cuda:0")
name = torch.cuda.get_device_name(0)
print(f"Device: {name}")

os.environ.setdefault("TORCH_CUDA_ARCH_LIST", "7.0")
ext = load(
    name="skinny_nvfp4_v9",
    sources=[os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "skinny_kernels.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo",
                       "-gencode=arch=compute_70,code=sm_70"],
    verbose=False,
)
print("Extension built.")

E2M1_MAGS = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6], device=dev)
E2M1_MIDS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=dev)
GROUP = 16


def pack_nvfp4(w):
    """w: [N, K] fp16 -> (codes u8 [N,K/2], scales u8 [N,K/16], gscale, ref)."""
    n, k = w.shape
    wf = w.float().view(n, k // GROUP, GROUP)
    scale = wf.abs().amax(-1) / 6.0
    gscale = float(scale.max().item() / 448.0) or 1e-12
    q8 = (scale / gscale).to(torch.float8_e4m3fn)
    eff = (q8.float() * gscale).clamp(min=1e-12).unsqueeze(-1)
    ratio = wf / eff
    idx = torch.bucketize(ratio.abs(), E2M1_MIDS)
    sign = ratio < 0
    ref = (E2M1_MAGS[idx] * torch.where(sign, -1.0, 1.0) * eff).half().view(n, k)
    codes = (idx | (sign.long() << 3)).view(n, k).to(torch.uint8)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).contiguous()
    scales_u8 = q8.view(torch.uint8).contiguous()
    return packed, scales_u8, gscale, ref


# ---------------------------------------------------------------
# Ceiling
# ---------------------------------------------------------------
def measure_bandwidth_gbs(gb=1.0, iters=30):
    a = torch.randn(int(gb * 2**30 / 2), dtype=torch.float16, device=dev)
    b = torch.empty_like(a)
    for _ in range(5):
        b.copy_(a)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        b.copy_(a)
    e.record(); torch.cuda.synchronize()
    return (2 * a.numel() * 2 * iters) / (s.elapsed_time(e) / 1000) / 1e9


bw = measure_bandwidth_gbs()
print(f"Measured copy bandwidth: {bw:.0f} GB/s")

# ---------------------------------------------------------------
# Build + validate
# ---------------------------------------------------------------
LAYERS, HID, FFN = 28, 3072, 8192
layer_shapes = [(HID, 3 * HID), (HID, HID), (HID, 2 * FFN), (FFN, HID)]


def kernel_for(m):
    return ext.gemm_simt if m <= 8 else ext.gemm_wmma


print("\nValidation (per shape, M in {1, 8, 16, 24, 64}):")
for k, n in layer_shapes:
    w = torch.randn(n, k, dtype=torch.float16, device=dev)
    packed, scales_u8, gscale, ref = pack_nvfp4(w)
    for m in (1, 8, 16, 24, 64):
        x = torch.randn(m, k, dtype=torch.float16, device=dev)
        y = kernel_for(m)(x, packed, scales_u8, gscale)
        y_ref = (x.float() @ ref.t().float())
        err = (y.float() - y_ref).abs().max().item()
        rel = err / y_ref.abs().max().item()
        status = "ok" if rel < 1e-2 else "FAIL"
        print(f"  {k:>5}x{n:<5} M={m:<2} rel_err={rel:.2e} [{status}]")
        assert rel < 1e-2, f"validation failed for {k}x{n} M={m}"
    del w, packed, scales_u8, ref
torch.cuda.empty_cache()

weights = []
total_packed_bytes = 0
for _ in range(LAYERS):
    per_layer = []
    for k, n in layer_shapes:
        w = torch.randn(n, k, dtype=torch.float16, device=dev)
        packed, scales_u8, gscale, ref = pack_nvfp4(w)
        del w, ref
        per_layer.append((packed, scales_u8, gscale))
        total_packed_bytes += packed.numel() + scales_u8.numel() + 4
    weights.append(per_layer)
torch.cuda.empty_cache()

print(f"\nResident packed model: {total_packed_bytes/2**30:.2f} GB")
print(f"Ideal pass time if purely memory-bound: "
      f"{total_packed_bytes/(bw*1e9)*1000:.2f} ms")

# ---------------------------------------------------------------
# Gate 1 pass structure
# ---------------------------------------------------------------
def one_pass(M, fn):
    x = torch.randn(M, HID, dtype=torch.float16, device=dev)
    for per_layer in weights:
        h = fn(x, *per_layer[0])
        h = fn(h[:, :HID].contiguous(), *per_layer[1])
        u = fn(h, *per_layer[2])
        u = torch.nn.functional.silu(u[:, :FFN]) * u[:, FFN:]
        x = fn(u.contiguous(), *per_layer[3])
    return x


def time_pass(M, fn, warmup=5, iters=30):
    for _ in range(warmup):
        one_pass(M, fn)
    torch.cuda.synchronize()
    s = torch.cuda.Event(True); e = torch.cuda.Event(True)
    s.record()
    for _ in range(iters):
        one_pass(M, fn)
    e.record(); torch.cuda.synchronize()
    return s.elapsed_time(e) / iters


MARLIN_MS = {1: 18.581, 8: 18.800, 16: 18.890, 24: 18.997, 32: 19.193,
             64: 20.703}
Ms = [1, 8, 16, 24, 32, 64]
rows = []
print(f"\n{'M':>4} | {'kernel':>6} | {'ms/pass':>8} | {'GB/s eff':>8} | "
      f"{'vs M=1':>7} | {'vs marlin':>9}")
print("-" * 62)
base = None
for M in Ms:
    candidates = {}
    if M <= 8:
        candidates["simt"] = time_pass(M, ext.gemm_simt)
    candidates["wmma"] = time_pass(M, ext.gemm_wmma)
    kname, ms = min(candidates.items(), key=lambda kv: kv[1])
    eff = total_packed_bytes / (ms / 1000) / 1e9
    if base is None:
        base = ms
    speedup = MARLIN_MS[M] / ms
    rows.append((M, kname, ms, eff, ms / base, speedup))
    print(f"{M:>4} | {kname:>6} | {ms:>8.2f} | {eff:>8.0f} | "
          f"{ms/base:>6.2f}x | {speedup:>8.2f}x")

with open("skinny_flatness_results.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["M", "ms_per_pass", "effective_GBs", "ratio_vs_M1",
                "copy_bw_GBs", "device"])
    for M, kname, ms, eff, r, _ in rows:
        w.writerow([M, f"{ms:.3f}", f"{eff:.1f}", f"{r:.3f}",
                    f"{bw:.0f}", name])
print("\nSaved skinny_flatness_results.csv")

# ---------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------
worst = min(rows, key=lambda r: r[3])
best = max(rows, key=lambda r: r[3])
print("\n=== VERDICT ===")
if worst[3] > 96:
    print(f"Beat marlin at every M: worst {worst[3]:.0f} GB/s (M={worst[0]}), "
          f"best {best[3]:.0f} GB/s (M={best[0]}); marlin was flat at ~96.")
else:
    losers = [r[0] for r in rows if r[3] <= 96]
    print(f"Best {best[3]:.0f} GB/s (M={best[0]}), but not above 96 GB/s "
          f"at M={losers}.")
r24 = next(r for r in rows if r[0] == 24)
print(f"M=24: {r24[3]:.0f} GB/s effective, {r24[4]:.2f}x of M=1, "
      f"{r24[5]:.2f}x faster than marlin.")
