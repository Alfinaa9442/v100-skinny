// Register Era microbench gate (2026-08-10) — bench-only extension.
//
// Tests the two postmortem cures for the v1 m8n8k4 register-fragment path
// (skinny_kernels.cu `skinny_nvfp4_mma8`, 200-360 GB/s):
//   (A) 8-row DRAM scatter  -> fragment-order pre-shuffled weights: the
//       Python harness permutes codes/scales so every warp load is one
//       linear 256B (codes) / 32B (scales) transaction. No converter work.
//   (B) serial mma chain    -> NACC independent accumulator fragment sets;
//       the 4 mmas of each 64-k window round-robin over them.
// mma8_v2<KC, SHUF, NACC> spans {v1-equivalent .. fully cured}; the
// load-rate probe streams the shuffled buffers at the identical warp
// geometry with no math (gate: >= 550 GB/s or the geometry cannot feed
// itself regardless of cures).
//
// Device helpers are verbatim copies from skinny_kernels.cu (TM-derived
// e2m1 decoder path only). Not wired into serving anywhere.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#define DEV_INLINE __device__ __forceinline__

DEV_INLINE half2 fp8e4m3_to_half2(unsigned char b) {
  const unsigned short hb =
      (((unsigned short)b & 0x80u) << 8) | (((unsigned short)b & 0x7Fu) << 7);
  const half hs = __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));  // *256
  return __halves2half2(hs, hs);
}

// TurboMind-derived e2m1 decoder (Apache-2.0; see skinny_kernels.cu for
// attribution). Interleaved output: out[i] holds codes (i, i+4); the 2^14
// exponent re-bias is folded into the caller's scale.
DEV_INLINE void dequant8_tm(unsigned q, half2 sc2p, half2 out[4]) {
  constexpr unsigned S = 0x80008000u, EM = 0x0E000E00u;
  unsigned v0 = ((q << 12) & S) | ((q << 9) & EM);
  unsigned v1 = ((q << 8) & S) | ((q << 5) & EM);
  unsigned v2 = ((q << 4) & S) | ((q << 1) & EM);
  unsigned v3 = (q & S) | ((q >> 3) & EM);
  out[0] = __hmul2(*reinterpret_cast<half2 *>(&v0), sc2p);
  out[1] = __hmul2(*reinterpret_cast<half2 *>(&v1), sc2p);
  out[2] = __hmul2(*reinterpret_cast<half2 *>(&v2), sc2p);
  out[3] = __hmul2(*reinterpret_cast<half2 *>(&v3), sc2p);
}

// ---------------------------------------------------------------------------
// Load-rate probe: the v2 warp geometry minus all math. Each lane reads its
// 8B code word + 1B scale per 64-k window from the pre-shuffled buffers —
// consecutive lanes are consecutive addresses (one 256B + one 32B warp
// transaction per window). Dummy checksum defeats dead-code elimination.
// ---------------------------------------------------------------------------
template <int WARPS>
__global__ void probe_loadrate_k(const uint2 *__restrict__ codes_shuf,
                                 const uint8_t *__restrict__ scales_shuf,
                                 unsigned *__restrict__ sink, int NW) {
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int g = blockIdx.x * WARPS + warp;
  const size_t base = (size_t)g * NW * 32 + lane;
  unsigned a0 = 0, a1 = 0, a2 = 0, a3 = 0;
  int t = 0;
  for (; t + 4 <= NW; t += 4) {
    const uint2 q0 = __ldcs(codes_shuf + base + (size_t)(t + 0) * 32);
    const uint2 q1 = __ldcs(codes_shuf + base + (size_t)(t + 1) * 32);
    const uint2 q2 = __ldcs(codes_shuf + base + (size_t)(t + 2) * 32);
    const uint2 q3 = __ldcs(codes_shuf + base + (size_t)(t + 3) * 32);
    a0 ^= q0.x ^ q0.y ^ __ldg(scales_shuf + base + (size_t)(t + 0) * 32);
    a1 ^= q1.x ^ q1.y ^ __ldg(scales_shuf + base + (size_t)(t + 1) * 32);
    a2 ^= q2.x ^ q2.y ^ __ldg(scales_shuf + base + (size_t)(t + 2) * 32);
    a3 ^= q3.x ^ q3.y ^ __ldg(scales_shuf + base + (size_t)(t + 3) * 32);
  }
  for (; t < NW; t++) {
    const uint2 q = __ldcs(codes_shuf + base + (size_t)t * 32);
    a0 ^= q.x ^ q.y ^ __ldg(scales_shuf + base + (size_t)t * 32);
  }
  const unsigned acc = a0 ^ a1 ^ a2 ^ a3;
  if (acc == 0xdeadbeefu) sink[g] = acc;
}

// ---------------------------------------------------------------------------
// mma8 v2: v1 body with the two cures as template switches.
//   SHUF=0, NACC=1  == v1 (scattered row loads, one accumulator chain)
//   SHUF=1          == cure A: linear fragment-order loads
//   NACC=2/4        == cure B: independent accumulator fragment sets
// Lane roles (qp, idx), activation staging, fragment maps, butterfly
// reduce and writeout are byte-identical to v1, so SHUF output must match
// v1 bit-for-bit — the harness asserts it.
// ---------------------------------------------------------------------------
// MODE (KILL-analysis stubs; ncu is admin-locked on this box so the
// mechanism is isolated causally instead): 0 = full kernel; 1 = skip
// dequant+fragment-pack (loads feed mma via reinterpret — garbage math,
// timing only); 2 = skip mma (dequant kept live via one cheap add).
template <int KC, bool SHUF, int NACC, int MODE = 0>
__global__ void mma8_v2(const uint8_t *__restrict__ codes,
                        const uint8_t *__restrict__ scales,
                        const half *__restrict__ x, half *__restrict__ y,
                        int N, int K, int M, float gscale) {
  extern __shared__ char smem_raw[];
  half2 *xs = reinterpret_cast<half2 *>(smem_raw);  // [8][PITCH] plain k order
  constexpr int PITCH = KC / 2 + 1;

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int qp = (lane >> 2) & 3;
  const int idx = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int gwarp = blockIdx.x * (int)(blockDim.x >> 5) + warp;
  const int n0 = gwarp * 8;
  const int NW = K >> 6;
  // v1 scattered row pointers (SHUF=0)
  const uint8_t *crow = codes + (size_t)(n0 + idx) * (K >> 1);
  const uint8_t *srow = scales + (size_t)(n0 + idx) * (K >> 4);
  // fragment-order linear pointers (SHUF=1)
  const uint2 *cshuf =
      reinterpret_cast<const uint2 *>(codes) + (size_t)gwarp * NW * 32 + lane;
  const uint8_t *sshuf = scales + (size_t)gwarp * NW * 32 + lane;

  const half2 gm2 = __float2half2_rn(gscale * 16384.f);

  float c[NACC][8];
#pragma unroll
  for (int a = 0; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[a][i] = 0.f;

  for (int k0 = 0; k0 < K; k0 += KC) {
    __syncthreads();
    for (int t = threadIdx.x; t < 8 * (KC / 2); t += blockDim.x) {
      const int m = t / (KC / 2), j = t % (KC / 2);
      half2 v = __float2half2_rn(0.f);
      if (m < M)
        v = *reinterpret_cast<const half2 *>(x + (size_t)m * K + k0 + 2 * j);
      xs[m * PITCH + j] = v;
    }
    __syncthreads();

#pragma unroll
    for (int s = 0; s < KC / 64; s++) {
      const int kb = k0 + s * 64 + qp * 16;
      uint2 q2;
      half2 sc2;
      if (SHUF) {
        const size_t t = (size_t)((k0 >> 6) + s) * 32;
        q2 = __ldcs(cshuf + t);
        sc2 = __hmul2(fp8e4m3_to_half2(__ldg(sshuf + t)), gm2);
      } else {
        q2 = __ldcs(reinterpret_cast<const uint2 *>(crow + (kb >> 1)));
        sc2 = __hmul2(fp8e4m3_to_half2(__ldg(srow + (kb >> 4))), gm2);
      }
      half2 af[4][2];
      if (MODE != 1) {
        half2 wa[4], wb[4];
        dequant8_tm(q2.x, sc2, wa);  // interleaved pairs (k, k+4)
        dequant8_tm(q2.y, sc2, wb);
        af[0][0] = __lows2half2(wa[0], wa[1]);
        af[0][1] = __lows2half2(wa[2], wa[3]);
        af[1][0] = __highs2half2(wa[0], wa[1]);
        af[1][1] = __highs2half2(wa[2], wa[3]);
        af[2][0] = __lows2half2(wb[0], wb[1]);
        af[2][1] = __lows2half2(wb[2], wb[3]);
        af[3][0] = __highs2half2(wb[0], wb[1]);
        af[3][1] = __highs2half2(wb[2], wb[3]);
      } else {
        // loads stay live via reinterpret; no dequant/pack instructions
        // (af[0][0]=sc2 keeps the scale load from being elided)
#pragma unroll
        for (int j = 0; j < 4; j++) {
          af[j][0] = *reinterpret_cast<const half2 *>(&q2.x);
          af[j][1] = *reinterpret_cast<const half2 *>(&q2.y);
        }
        af[0][0] = sc2;
      }
      const half2 *xrow = xs + idx * PITCH + ((kb - k0) >> 1);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const half2 b0 = xrow[2 * j], b1 = xrow[2 * j + 1];
        const unsigned *A = reinterpret_cast<const unsigned *>(af[j]);
        float *cc = c[j & (NACC - 1)];
        if (MODE == 2) {
          // keep dequant live with one cheap op; no mma issued
          cc[0] += (float)(A[0] ^ A[1]);
          continue;
        }
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
            "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
            : "+f"(cc[0]), "+f"(cc[1]), "+f"(cc[2]), "+f"(cc[3]), "+f"(cc[4]),
              "+f"(cc[5]), "+f"(cc[6]), "+f"(cc[7])
            : "r"(A[0]), "r"(A[1]),
              "r"(*reinterpret_cast<const unsigned *>(&b0)),
              "r"(*reinterpret_cast<const unsigned *>(&b1)));
      }
    }
  }

#pragma unroll
  for (int a = 1; a < NACC; a++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[0][i] += c[a][i];
#pragma unroll
  for (int i = 0; i < 8; i++) {
    c[0][i] += __shfl_xor_sync(0xffffffffu, c[0][i], 4);
    c[0][i] += __shfl_xor_sync(0xffffffffu, c[0][i], 8);
  }
  if ((lane & 12) == 0) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int r = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int cc = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      if (cc < M) y[(size_t)cc * N + n0 + r] = __float2half(c[0][i]);
    }
  }
}

// ---------------------------------------------------------------------------
// Host wrappers
// ---------------------------------------------------------------------------
constexpr int KC = 256, WARPS = 4;

void probe_loadrate(torch::Tensor codes_shuf, torch::Tensor scales_shuf,
                    torch::Tensor sink, int64_t n, int64_t k) {
  TORCH_CHECK(codes_shuf.is_cuda() && codes_shuf.dtype() == torch::kUInt8);
  TORCH_CHECK(scales_shuf.is_cuda() && scales_shuf.dtype() == torch::kUInt8);
  TORCH_CHECK(n % (WARPS * 8) == 0 && k % 256 == 0 && (k >> 6) % 4 == 0);
  TORCH_CHECK(codes_shuf.numel() == n / 8 * (k >> 6) * 32 * 8);
  TORCH_CHECK(scales_shuf.numel() == n / 8 * (k >> 6) * 32);
  TORCH_CHECK(sink.is_cuda() && sink.dtype() == torch::kInt32 &&
              sink.numel() >= n / 8);
  auto stream = at::cuda::getCurrentCUDAStream();
  probe_loadrate_k<WARPS><<<dim3((int)(n / (WARPS * 8))), dim3(WARPS * 32), 0,
                            stream>>>(
      reinterpret_cast<const uint2 *>(codes_shuf.data_ptr<uint8_t>()),
      scales_shuf.data_ptr<uint8_t>(),
      reinterpret_cast<unsigned *>(sink.data_ptr<int>()), (int)(k >> 6));
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

torch::Tensor gemm_mma8_v2(torch::Tensor x, torch::Tensor codes,
                           torch::Tensor scales, double gscale, int64_t n,
                           bool shuf, int64_t nacc, int64_t mode) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 &&
              codes.is_contiguous());
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kUInt8 &&
              scales.is_contiguous());
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(m >= 1 && m <= 8, "mma8_v2 supports M in 1..8, got ", m);
  TORCH_CHECK(k % KC == 0 && k >= KC, "K must be a multiple of ", KC);
  TORCH_CHECK(n % (WARPS * 8) == 0, "N must be a multiple of ", WARPS * 8);
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = 8 * (KC / 2 + 1) * (int)sizeof(half2);
  const dim3 grid((int)(n / (WARPS * 8))), block(WARPS * 32);

#define LAUNCH_V2(SH, NA, MO)                                               \
  mma8_v2<KC, SH, NA, MO><<<grid, block, smem, stream>>>(                   \
      codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),                \
      reinterpret_cast<const half *>(x.data_ptr<at::Half>()),               \
      reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,     \
      (int)m, (float)gscale)

  if (mode == 0) {
    if (shuf && nacc == 1) LAUNCH_V2(true, 1, 0);
    else if (shuf && nacc == 2) LAUNCH_V2(true, 2, 0);
    else if (shuf && nacc == 4) LAUNCH_V2(true, 4, 0);
    else if (!shuf && nacc == 1) LAUNCH_V2(false, 1, 0);
    else if (!shuf && nacc == 2) LAUNCH_V2(false, 2, 0);
    else if (!shuf && nacc == 4) LAUNCH_V2(false, 4, 0);
    else TORCH_CHECK(false, "nacc must be 1, 2 or 4");
  } else {
    // KILL-analysis stubs: only the v1 and AB2 configs carry modes 1/2
    TORCH_CHECK((shuf && nacc == 2) || (!shuf && nacc == 1),
                "modes 1/2 only for (shuf=0,nacc=1) or (shuf=1,nacc=2)");
    if (mode == 1) {
      if (shuf) LAUNCH_V2(true, 2, 1); else LAUNCH_V2(false, 1, 1);
    } else if (mode == 2) {
      if (shuf) LAUNCH_V2(true, 2, 2); else LAUNCH_V2(false, 1, 2);
    } else TORCH_CHECK(false, "mode must be 0, 1 or 2");
  }
#undef LAUNCH_V2
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("probe_loadrate", &probe_loadrate,
        "fragment-order load-rate probe (no math)");
  m.def("gemm_mma8_v2", &gemm_mma8_v2,
        "mma8 v2: SHUF (pre-shuffled linear loads) x NACC accumulator sets",
        py::arg("x"), py::arg("codes"), py::arg("scales"), py::arg("gscale"),
        py::arg("n"), py::arg("shuf"), py::arg("nacc"), py::arg("mode") = 0);
}
