// Twin-race (2026-08-10) — batch band M=16..64, bench-only extension.
//
// The register gate isolated the band disease: memory-latency exposure
// (too few independent weight loads in flight between barriers). Two
// patients get the same cure:
//
//   RUNG A  wmma_pipe<WN,WM,KC,DEPTH,DBUF>: the production WMMA kernel
//           with load-schedule knobs only — DEPTH=2 keeps two chunks of
//           code words in registers (the M33-64 config had ONE 8B code
//           load/thread/chunk); DBUF ping-pongs the smem tiles so each
//           chunk pays one barrier instead of two and the dequant+store
//           of chunk i+1 issues behind the mma of chunk i.
//
//   RUNG B  mma8_ring<KC,MB>: m8n8k4 rebuilt prefetch-first — weights
//           skip smem entirely: a 4-deep register ring of fragment-order
//           pre-shuffled 8B code words (the bit-verified layout from the
//           register gate) refills one full chunk ahead; dequant sits
//           behind the load front; MB=M/8 independent accumulator sets;
//           barriers only fence the activation smem stage.
//
// Device helpers are verbatim copies from skinny_kernels.cu.
// Bench-only: nothing here is wired into serving.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

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

// ---------------------------------------------------------------------------
// RUNG A: wmma_pipe
// ---------------------------------------------------------------------------
template <int WN, int WM, int KC, int DEPTH, bool DBUF>
__global__ void wmma_pipe(const uint8_t *__restrict__ codes,
                          const uint8_t *__restrict__ scales,
                          const half *__restrict__ x, half *__restrict__ y,
                          int N, int K, int m_real, float gscale) {
  constexpr int NT = WN * 16, MT = WM * 16;
  constexpr int PW = KC + 16, PX = KC + 16;
  constexpr int NTHREADS = WN * WM * 32;
  constexpr int CSEG = NT * (KC / 16) / NTHREADS;
  constexpr int XSEG = MT * (KC / 8) / NTHREADS;
  constexpr int NBUF = DBUF ? 2 : 1;
  constexpr int TILE = NT * PW + MT * PX;  // halfs per buffer

  extern __shared__ char smem_raw[];
  half *smem = reinterpret_cast<half *>(smem_raw);

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int wn = warp % WN, wm = warp / WN;
  const int nb = blockIdx.x * NT;

  uint2 st_c[DEPTH][CSEG];
  unsigned char st_s[DEPTH][CSEG];
  uint4 st_x[XSEG];

  auto ldc = [&](int slot, int k0) {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int n = idx / (KC / 16), s = idx % (KC / 16);
      st_c[slot][i] = __ldcs(reinterpret_cast<const uint2 *>(
          codes + (size_t)(nb + n) * (K >> 1) + (k0 >> 1) + s * 8));
      st_s[slot][i] =
          __ldcs(scales + (size_t)(nb + n) * (K >> 4) + (k0 >> 4) + s);
    }
  };
  auto ldx = [&](int k0) {
#pragma unroll
    for (int i = 0; i < XSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      st_x[i] = (m < m_real)
                    ? *reinterpret_cast<const uint4 *>(x + (size_t)m * K + k0 +
                                                       j4 * 8)
                    : make_uint4(0, 0, 0, 0);
    }
  };
  auto store = [&](int buf, int slot) {
    half *ws = smem + buf * TILE;
    half *xs = ws + NT * PW;
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int n = idx / (KC / 16), s = idx % (KC / 16);
      const half2 sc2 = fp8e4m3_to_half2(st_s[slot][i]);
      half2 *wrow = reinterpret_cast<half2 *>(ws + n * PW + s * 16);
      const unsigned qs[2] = {st_c[slot][i].x, st_c[slot][i].y};
#pragma unroll
      for (int w = 0; w < 2; w++) {
        half2 t[4];
        dequant8_tm(qs[w], sc2, t);  // 2^-14 factor, undone in epilogue
        const unsigned *tr = reinterpret_cast<const unsigned *>(t);
        unsigned lin[4] = {__byte_perm(tr[0], tr[1], 0x5410),
                           __byte_perm(tr[2], tr[3], 0x5410),
                           __byte_perm(tr[0], tr[1], 0x7632),
                           __byte_perm(tr[2], tr[3], 0x7632)};
#pragma unroll
        for (int pi = 0; pi < 4; pi++)
          wrow[w * 4 + pi] = *reinterpret_cast<half2 *>(&lin[pi]);
      }
    }
#pragma unroll
    for (int i = 0; i < XSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      *reinterpret_cast<uint4 *>(xs + m * PX + j4 * 8) = st_x[i];
    }
  };
  auto mma_chunk = [&](int buf, wmma::fragment<wmma::accumulator, 16, 16, 16,
                                               float> &cfrag) {
    half *ws = smem + buf * TILE;
    half *xs = ws + NT * PW;
    wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a[2];
    wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b[2];
    wmma::load_matrix_sync(a[0], ws + wn * 16 * PW, PW);
    wmma::load_matrix_sync(b[0], xs + wm * 16 * PX, PX);
#pragma unroll
    for (int kk = 0; kk < KC / 16; kk++) {
      const int cur = kk & 1, nxt = cur ^ 1;
      if (kk + 1 < KC / 16) {
        wmma::load_matrix_sync(a[nxt], ws + wn * 16 * PW + (kk + 1) * 16, PW);
        wmma::load_matrix_sync(b[nxt], xs + wm * 16 * PX + (kk + 1) * 16, PX);
      }
      wmma::mma_sync(cfrag, a[cur], b[cur], cfrag);
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cfrag;
  wmma::fill_fragment(cfrag, 0.f);
  const int NC = K / KC;

  if (DBUF) {
    // 1 barrier/chunk: store(i+1) -> other buffer issues behind mma(i)
    ldc(0, 0);
    ldx(0);
    store(0, 0);
    if (DEPTH > 1 && NC > 1) ldc(1 & (DEPTH - 1), KC);
    __syncthreads();
    for (int i = 0; i < NC; i++) {
      const int cur = i & 1;
      if (i + 1 < NC) {
        ldx((i + 1) * KC);  // in flight under the mma
        // shallow depth refills its lone slot here: cover = one mma loop
        if (DEPTH == 1) ldc(0, (i + 1) * KC);
      }
      mma_chunk(cur, cfrag);
      if (i + 1 < NC) store(cur ^ 1, (i + 1) & (DEPTH - 1));
      if (DEPTH > 1 && i + DEPTH < NC)
        ldc((i + DEPTH) & (DEPTH - 1), (i + DEPTH) * KC);
      __syncthreads();
    }
  } else {
    // incumbent barrier structure, code-register depth DEPTH
    ldc(0, 0);
    ldx(0);
    if (DEPTH > 1 && NC > 1) ldc(1 & (DEPTH - 1), KC);
    for (int i = 0; i < NC; i++) {
      __syncthreads();
      store(0, i & (DEPTH - 1));
      __syncthreads();
      if (i + 1 < NC) ldx((i + 1) * KC);
      if (i + DEPTH < NC) ldc((i + DEPTH) & (DEPTH - 1), (i + DEPTH) * KC);
      mma_chunk(0, cfrag);
    }
  }

  __syncthreads();
  float *cs = reinterpret_cast<float *>(smem_raw) + warp * 256;
  wmma::store_matrix_sync(cs, cfrag, 16, wmma::mem_row_major);
  __syncwarp();
  const float gs_eff = gscale * 16384.f;
  for (int e = lane; e < 256; e += 32) {
    const int i = e >> 4, j = e & 15;
    const int gm = wm * 16 + j, gn = nb + wn * 16 + i;
    if (gm < m_real) y[(size_t)gm * N + gn] = __float2half(cs[e] * gs_eff);
  }
}

// ---------------------------------------------------------------------------
// RUNG B: mma8_ring — weights global->registers->fragments, 4-deep ring.
// Fragment maps identical to the v1 mma8 path (skinny_kernels.cu).
// ---------------------------------------------------------------------------
template <int KC, int MB>  // MB = M/8 blocks (2/4/8)
__global__ void mma8_ring(const uint8_t *__restrict__ codes_shuf,
                          const uint8_t *__restrict__ scales_shuf,
                          const half *__restrict__ x, half *__restrict__ y,
                          int N, int K, int M, float gscale) {
  extern __shared__ char smem_raw[];
  half2 *xs = reinterpret_cast<half2 *>(smem_raw);  // [MB*8][PITCH]
  constexpr int PITCH = KC / 2 + 1;
  constexpr int RING = 4;
  static_assert(KC / 64 == RING, "ring slot = window-in-chunk requires KC=256");

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int qp = (lane >> 2) & 3;
  const int idx = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int gwarp = blockIdx.x * (int)(blockDim.x >> 5) + warp;
  const int n0 = gwarp * 8;
  const int NW = K >> 6;
  const uint2 *cbase =
      reinterpret_cast<const uint2 *>(codes_shuf) + (size_t)gwarp * NW * 32 +
      lane;
  const uint8_t *sbase = scales_shuf + (size_t)gwarp * NW * 32 + lane;

  const half2 gm2 = __float2half2_rn(gscale * 16384.f);

  float c[MB][8];
#pragma unroll
  for (int b = 0; b < MB; b++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[b][i] = 0.f;

  uint2 q2r[RING];
  unsigned scr[RING];
#pragma unroll
  for (int r = 0; r < RING; r++) {
    q2r[r] = __ldcs(cbase + (size_t)r * 32);
    scr[r] = __ldg(sbase + (size_t)r * 32);
  }

  for (int k0 = 0, w0 = 0; k0 < K; k0 += KC, w0 += RING) {
    __syncthreads();
    for (int t = threadIdx.x; t < MB * 8 * (KC / 2); t += blockDim.x) {
      const int m = t / (KC / 2), j = t % (KC / 2);
      half2 v = __float2half2_rn(0.f);
      if (m < M)
        v = *reinterpret_cast<const half2 *>(x + (size_t)m * K + k0 + 2 * j);
      xs[m * PITCH + j] = v;
    }
    __syncthreads();

#pragma unroll
    for (int s = 0; s < RING; s++) {
      const uint2 q2 = q2r[s];
      const unsigned scb = scr[s];
      const int wnext = w0 + s + RING;
      if (wnext < NW) {  // refill a full chunk ahead: 4 loads always afloat
        q2r[s] = __ldcs(cbase + (size_t)wnext * 32);
        scr[s] = __ldg(sbase + (size_t)wnext * 32);
      }
      const half2 sc2 =
          __hmul2(fp8e4m3_to_half2((unsigned char)scb), gm2);
      half2 af[4][2];
      half2 wa[4], wb[4];
      dequant8_tm(q2.x, sc2, wa);
      dequant8_tm(q2.y, sc2, wb);
      af[0][0] = __lows2half2(wa[0], wa[1]);
      af[0][1] = __lows2half2(wa[2], wa[3]);
      af[1][0] = __highs2half2(wa[0], wa[1]);
      af[1][1] = __highs2half2(wa[2], wa[3]);
      af[2][0] = __lows2half2(wb[0], wb[1]);
      af[2][1] = __lows2half2(wb[2], wb[3]);
      af[3][0] = __highs2half2(wb[0], wb[1]);
      af[3][1] = __highs2half2(wb[2], wb[3]);
      const int koff = (s * 64 + qp * 16) >> 1;
#pragma unroll
      for (int mb = 0; mb < MB; mb++) {
        const half2 *xrow = xs + (mb * 8 + idx) * PITCH + koff;
#pragma unroll
        for (int j = 0; j < 4; j++) {
          const half2 b0 = xrow[2 * j], b1 = xrow[2 * j + 1];
          const unsigned *A = reinterpret_cast<const unsigned *>(af[j]);
          asm volatile(
              "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
              "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
              "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
              : "+f"(c[mb][0]), "+f"(c[mb][1]), "+f"(c[mb][2]),
                "+f"(c[mb][3]), "+f"(c[mb][4]), "+f"(c[mb][5]),
                "+f"(c[mb][6]), "+f"(c[mb][7])
              : "r"(A[0]), "r"(A[1]),
                "r"(*reinterpret_cast<const unsigned *>(&b0)),
                "r"(*reinterpret_cast<const unsigned *>(&b1)));
        }
      }
    }
  }

#pragma unroll
  for (int mb = 0; mb < MB; mb++) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      c[mb][i] += __shfl_xor_sync(0xffffffffu, c[mb][i], 4);
      c[mb][i] += __shfl_xor_sync(0xffffffffu, c[mb][i], 8);
    }
    if ((lane & 12) == 0) {
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const int r = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
        const int cc = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
        const int gm = mb * 8 + cc;
        if (gm < M) y[(size_t)gm * N + n0 + r] = __float2half(c[mb][i]);
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host wrappers + occupancy report
// ---------------------------------------------------------------------------
static void set_smem_attr(const void *fn, int bytes) {
  if (bytes > 48 * 1024)
    cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,
                         bytes);
}

torch::Tensor gemm_wmma_pipe(torch::Tensor x, torch::Tensor codes,
                             torch::Tensor scales, double gscale,
                             int64_t depth, bool dbuf) {
  const int64_t m = x.size(0), k = x.size(1), n = codes.size(0);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(codes.size(1) * 2 == k && scales.size(0) == n &&
              scales.size(1) * 16 == k);
  TORCH_CHECK(depth == 1 || depth == 2, "depth 1 or 2");
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_PIPE(WN, WM, KC, D, B)                                       \
  do {                                                                      \
    constexpr int NT = WN * 16, MT = WM * 16;                               \
    TORCH_CHECK(n % NT == 0, "N % ", NT);                                   \
    TORCH_CHECK(k % KC == 0, "K % ", KC);                                   \
    const int smem = (B ? 2 : 1) * (NT + MT) * (KC + 16) * (int)sizeof(half); \
    auto *fn = wmma_pipe<WN, WM, KC, D, B>;                                 \
    set_smem_attr(reinterpret_cast<const void *>(fn), smem);                \
    fn<<<dim3(n / NT), dim3(WN * WM * 32), smem, stream>>>(                 \
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),              \
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),             \
        reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
        (int)m, (float)gscale);                                             \
  } while (0)

#define PICK(WN, WM, KC)                                                    \
  do {                                                                      \
    if (depth == 2 && dbuf) LAUNCH_PIPE(WN, WM, KC, 2, true);               \
    else if (depth == 2) LAUNCH_PIPE(WN, WM, KC, 2, false);                 \
    else LAUNCH_PIPE(WN, WM, KC, 1, true);                                  \
  } while (0)

  if (m <= 16) PICK(4, 1, 256);
  else if (m <= 32) PICK(2, 2, 128);
  else if (m <= 64) PICK(2, 4, 128);
  else TORCH_CHECK(false, "M <= 64 only");
#undef PICK
#undef LAUNCH_PIPE
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor gemm_mma8_ring(torch::Tensor x, torch::Tensor codes_shuf,
                             torch::Tensor scales_shuf, double gscale,
                             int64_t n) {
  const int64_t m = x.size(0), k = x.size(1);
  constexpr int KC = 256, WARPS = 4;
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(m >= 9 && m <= 64, "ring path is for M 9..64, got ", m);
  TORCH_CHECK(k % KC == 0, "K % ", KC);
  TORCH_CHECK(n % (WARPS * 8) == 0, "N % ", WARPS * 8);
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_RING(MBv)                                                    \
  do {                                                                      \
    const int smem = MBv * 8 * (KC / 2 + 1) * (int)sizeof(half2);           \
    mma8_ring<KC, MBv>                                                      \
        <<<dim3((int)(n / (WARPS * 8))), dim3(WARPS * 32), smem, stream>>>( \
            codes_shuf.data_ptr<uint8_t>(), scales_shuf.data_ptr<uint8_t>(),\
            reinterpret_cast<const half *>(x.data_ptr<at::Half>()),         \
            reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n,       \
            (int)k, (int)m, (float)gscale);                                 \
  } while (0)

  if (m <= 16) LAUNCH_RING(2);
  else if (m <= 32) LAUNCH_RING(4);
  else LAUNCH_RING(8);
#undef LAUNCH_RING
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// blocks/SM for every raced kernel at its launch config (the goal's
// "achieved warps/SM" log; registers/thread come from ptxas -v).
std::vector<std::tuple<std::string, int64_t, int64_t>> occ_report() {
  std::vector<std::tuple<std::string, int64_t, int64_t>> out;
  auto add = [&](const char *name, const void *fn, int threads, int smem) {
    set_smem_attr(fn, smem);
    int blocks = -1;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks, fn, threads, smem);
    out.emplace_back(name, blocks, (int64_t)blocks * (threads / 32));
  };
#define ADD_PIPE(WN, WM, KC, D, B)                                          \
  add("pipe_" #WN #WM "_" #KC "_d" #D "_b" #B,                              \
      reinterpret_cast<const void *>(wmma_pipe<WN, WM, KC, D, B>),          \
      WN *WM * 32, (B ? 2 : 1) * (WN * 16 + WM * 16) * (KC + 16) * 2)
  ADD_PIPE(4, 1, 256, 2, false); ADD_PIPE(4, 1, 256, 2, true);
  ADD_PIPE(4, 1, 256, 1, true);
  ADD_PIPE(2, 2, 128, 2, false); ADD_PIPE(2, 2, 128, 2, true);
  ADD_PIPE(2, 2, 128, 1, true);
  ADD_PIPE(2, 4, 128, 2, false); ADD_PIPE(2, 4, 128, 2, true);
  ADD_PIPE(2, 4, 128, 1, true);
#undef ADD_PIPE
  add("ring_mb2", reinterpret_cast<const void *>(mma8_ring<256, 2>), 128,
      2 * 8 * 129 * 4);
  add("ring_mb4", reinterpret_cast<const void *>(mma8_ring<256, 4>), 128,
      4 * 8 * 129 * 4);
  add("ring_mb8", reinterpret_cast<const void *>(mma8_ring<256, 8>), 128,
      8 * 8 * 129 * 4);
  return out;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_wmma_pipe", &gemm_wmma_pipe,
        "rung A: pipelined WMMA (depth, dbuf knobs)");
  m.def("gemm_mma8_ring", &gemm_mma8_ring,
        "rung B: m8n8k4 register-ring, fragment-order shuffled weights");
  m.def("occ_report", &occ_report, "blocks/SM + warps/SM per kernel");
}
