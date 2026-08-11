// QP-N m8n8k4 (2026-08-10) — bench-only extension. One experiment:
//   8x32 QP-N mapping + A-stationary registers + direct NVFP4->B-register
//   decode. The Volta-native four-quadpair form of mma.sync.m8n8k4.
//
// Why this differs from every killed variant:
//   - The four quadpairs split the N dimension (QP q owns columns
//     q*8..q*8+7 of the warp's 32-col tile). All four consume the SAME
//     8x4 activation A tile per k-slice — the A fragment map depends
//     only on lane-position INSIDE the quadpair, so QP-sibling lanes
//     hold identical A registers. One warp mma.sync = 4 independent
//     8x8x4 MMAs = an 8x32x4 step. Activation traffic per weight byte
//     drops 4x vs the killed B_ring (its mechanism).
//   - A is loaded straight from global into registers (x is KB-scale and
//     L1/L2-resident; QP-sibling lanes hit the same cache line). No smem
//     for activations, no smem for weights, NO BARRIERS in the main
//     loop — the one __syncthreads() is the cross-warp K-reduce at
//     output (CTA's 4 warps split K to keep the grid at N/32).
//   - Weights are prepacked (Python, bench-side) into exact B-fragment
//     lane order with nibbles pre-interleaved so dequant8_tm's
//     (i, i+4)-paired output IS the adjacent-k B register pair: the 8
//     __lows2/__highs2 pack instructions per window in v1 become zero.
//   - One fp8 group scale per 16-k group is held in a register across
//     exactly the group's 4 mmas.
//
// Fragment maps are the mma8_probe.cu-derived ones already byte-verified
// in the v1 kernel (operand-position-based; roles swapped: A=activations
// row-major, B=weights col-major, C rows->M, cols->QP-local N).

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>

#define DEV_INLINE __device__ __forceinline__

DEV_INLINE half2 fp8e4m3_to_half2(unsigned char b) {
  const unsigned short hb =
      (((unsigned short)b & 0x80u) << 8) | (((unsigned short)b & 0x7Fu) << 7);
  const half hs = __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));
  return __halves2half2(hs, hs);
}

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

#define MMA_8N8K4(C, A0, A1, B0, B1)                                        \
  asm volatile(                                                             \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                     \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                        \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),         \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                                  \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

// MT = number of 8-row A tiles (M <= 8*MT): the two-tile form decodes B
// ONCE per group and feeds both A tiles — M 9..16 at ~M=8 cost, unlike
// the raced qpn_x2 which paid the whole weight stream twice.
// CTDIRECT = read the checkpoint-native row-major codes/scales (no
// prepacked third weight copy): per-lane row pointers like v1, __ldg so
// L2 amortizes the 8B-per-32B-sector scatter across the unrolled group
// loop, and v1's lows/highs repack rebuilds adjacent-k pairs (the CT
// nibble order decodes interleaved).
//   prepack form: bcodes [ntiles][G][32] x 8B nibble-interleaved.
//   ct form:      codes [N][K/2], scales [N][K/16] as served today.
template <int MT, bool CTDIRECT>
__global__ void mma8_qpn(const uint8_t *__restrict__ bcodes,
                         const uint8_t *__restrict__ bscales,
                         const half *__restrict__ x, half *__restrict__ y,
                         int N, int K, int M, float gscale) {
  constexpr int WARPS = 4;
  __shared__ float cs[WARPS][MT * 256];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);  // A row & B local col
  const int G = K >> 4, Gq = G / WARPS;
  const int g0 = warp * Gq;
  // prepack pointers
  const uint2 *cb = reinterpret_cast<const uint2 *>(bcodes) +
                    (size_t)tile * G * 32 + lane;
  const uint8_t *sb = bscales + (size_t)tile * G * 32 + lane;
  // ct-direct pointers (per-lane weight row = its B column)
  const int wcol = tile * 32 + qp * 8 + r;
  const uint8_t *crow = bcodes + (size_t)wcol * (K >> 1);
  const uint8_t *srow = bscales + (size_t)wcol * (K >> 4);

  const half2 gm2 = __float2half2_rn(gscale * 16384.f);
  float c[MT][8];
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[t][i] = 0.f;

  // Decode one 16-k group and run its 4 mma slices over all MT A tiles.
  auto consume = [&](uint2 q2, half2 sc2, int g) {
    half2 b[8];
    if (CTDIRECT) {
      // CT nibble order decodes (i, i+4)-interleaved; rebuild adjacent-k
      // pairs v1-style (8 pack ops per group).
      half2 wa[4], wb[4];
      dequant8_tm(q2.x, sc2, wa);
      dequant8_tm(q2.y, sc2, wb);
      b[0] = __lows2half2(wa[0], wa[1]);
      b[1] = __lows2half2(wa[2], wa[3]);
      b[2] = __highs2half2(wa[0], wa[1]);
      b[3] = __highs2half2(wa[2], wa[3]);
      b[4] = __lows2half2(wb[0], wb[1]);
      b[5] = __lows2half2(wb[2], wb[3]);
      b[6] = __highs2half2(wb[0], wb[1]);
      b[7] = __highs2half2(wb[2], wb[3]);
    } else {
      // prepack ordering: dequant output IS the adjacent-k pair
      dequant8_tm(q2.x, sc2, b + 0);  // slices 0,1 (k0..7)
      dequant8_tm(q2.y, sc2, b + 4);  // slices 2,3 (k8..15)
    }
    const unsigned *B = reinterpret_cast<const unsigned *>(b);
#pragma unroll
    for (int t = 0; t < MT; t++) {
      const int ar = t * 8 + r;
      uint4 a01 = make_uint4(0, 0, 0, 0), a23 = make_uint4(0, 0, 0, 0);
      if (ar < M) {
        const half *xrow = x + (size_t)ar * K;
        a01 = *reinterpret_cast<const uint4 *>(xrow + g * 16);
        a23 = *reinterpret_cast<const uint4 *>(xrow + g * 16 + 8);
      }
      const unsigned *A0 = reinterpret_cast<const unsigned *>(&a01);
      const unsigned *A1 = reinterpret_cast<const unsigned *>(&a23);
      MMA_8N8K4(c[t], A0[0], A0[1], B[0], B[1]);  // k slice 0
      MMA_8N8K4(c[t], A0[2], A0[3], B[2], B[3]);  // k slice 1
      MMA_8N8K4(c[t], A1[0], A1[1], B[4], B[5]);  // k slice 2
      MMA_8N8K4(c[t], A1[2], A1[3], B[6], B[7]);  // k slice 3
    }
  };

  if (CTDIRECT) {
    // Blocked column reads: two uint4 per 4-group block (full 32B
    // sectors — per-group 8B ldg wasted 3/4 of every sector and lost
    // 2x in the first sweep pass) + one uint of scale bytes.
    for (int gb = g0; gb < g0 + Gq; gb += 4) {
      const uint4 cq0 =
          __ldg(reinterpret_cast<const uint4 *>(crow + (size_t)gb * 8));
      const uint4 cq1 =
          __ldg(reinterpret_cast<const uint4 *>(crow + (size_t)gb * 8 + 16));
      const unsigned sq =
          __ldg(reinterpret_cast<const unsigned *>(srow + gb));
      const uint4 cq[2] = {cq0, cq1};
      const unsigned *cw = reinterpret_cast<const unsigned *>(cq);
#pragma unroll
      for (int s4 = 0; s4 < 4; s4++) {
        const half2 sc2 = __hmul2(
            fp8e4m3_to_half2((unsigned char)((sq >> (8 * s4)) & 0xFF)), gm2);
        consume(make_uint2(cw[s4 * 2], cw[s4 * 2 + 1]), sc2, gb + s4);
      }
    }
  } else {
    // Linear fragment-order stream: keep the raced form's explicit
    // 4-deep unroll (dropping it cost ~20-25% in sweep pass 2).
#pragma unroll 4
    for (int g = g0; g < g0 + Gq; g++) {
      const uint2 q2 = __ldcs(cb + (size_t)g * 32);
      const half2 sc2 =
          __hmul2(fp8e4m3_to_half2(__ldg(sb + (size_t)g * 32)), gm2);
      consume(q2, sc2, g);
    }
  }

  // C map (mma8_probe.cu, roles swapped): reg i of lane L ->
  //   A-row  (i&2)|((L&16)?4:0)|(L&1)
  //   B-col  (i&1)|(((L>>1)&1)<<1)|((i>>2)<<2)   (QP-local; +qp*8 global)
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int row = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int col = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      cs[warp][(t * 8 + row) * 32 + qp * 8 + col] = c[t][i];
    }
  __syncthreads();  // the only barrier: cross-warp K reduce
  for (int e = threadIdx.x; e < MT * 256; e += blockDim.x) {
    const float v = cs[0][e] + cs[1][e] + cs[2][e] + cs[3][e];
    const int row = e >> 5, col = e & 31;
    if (row < M) y[(size_t)row * N + (size_t)tile * 32 + col] =
        __float2half(v);
  }
}

torch::Tensor gemm_qpn(torch::Tensor x, torch::Tensor bcodes,
                       torch::Tensor bscales, double gscale, int64_t n,
                       torch::Tensor out, int64_t row0, bool ct) {
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(bcodes.is_cuda() && bcodes.dtype() == torch::kUInt8);
  TORCH_CHECK(bscales.is_cuda() && bscales.dtype() == torch::kUInt8);
  TORCH_CHECK(m >= 1 && m <= 16, "qpn supports M 1..16, got ", m);
  TORCH_CHECK(k % 64 == 0, "K % 64 (4-warp split of 16-k groups)");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  if (ct) {
    TORCH_CHECK(bcodes.numel() == n * (k / 2) &&
                bscales.numel() == n * (k / 16));
  } else {
    TORCH_CHECK(bcodes.numel() == n / 32 * (k / 16) * 32 * 8 &&
                bscales.numel() == n / 32 * (k / 16) * 32);
  }
  TORCH_CHECK(out.size(1) == n && row0 + m <= out.size(0));
  auto stream = at::cuda::getCurrentCUDAStream();
  half *yp = reinterpret_cast<half *>(out.data_ptr<at::Half>()) +
             (size_t)row0 * n;

#define LAUNCH_QPN(MTv, CTv)                                                \
  mma8_qpn<MTv, CTv><<<dim3((int)(n / 32)), dim3(128), 0, stream>>>(        \
      bcodes.data_ptr<uint8_t>(), bscales.data_ptr<uint8_t>(),              \
      reinterpret_cast<const half *>(x.data_ptr<at::Half>()), yp, (int)n,   \
      (int)k, (int)m, (float)gscale)

  if (m <= 8) {
    if (ct) LAUNCH_QPN(1, true); else LAUNCH_QPN(1, false);
  } else {
    if (ct) LAUNCH_QPN(2, true); else LAUNCH_QPN(2, false);
  }
#undef LAUNCH_QPN
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return out;
}

int64_t qpn_blocks_per_sm() {
  int blocks = -1;
  cudaOccupancyMaxActiveBlocksPerMultiprocessor(
      &blocks, reinterpret_cast<const void *>(mma8_qpn<1, false>), 128, 0);
  return blocks;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_qpn", &gemm_qpn,
        "QP-N m8n8k4: MT 8-row A tiles, prepack or CT-direct weights");
  m.def("qpn_blocks_per_sm", &qpn_blocks_per_sm, "occupancy");
}
