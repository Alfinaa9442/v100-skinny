// SM70 skinny NVFP4 dequant-GEMM: y[M,N] = x[M,K] @ W[K,N]
//
// Packed format (0.5625 bytes/weight, same density as the marlin path):
//   codes  uint8 [N][K/2]   two e2m1 codes per byte, low nibble = even k
//   scales uint8 [N][K/16]  fp8-e4m3 per 16-k group
//   gscale float            global scale, applied in the epilogue
//
// Two kernels, split by where V100 runs out of math:
//   simt (M<=8):  warp per output row, HFMA2 inner loop, k-pairs in the
//                 half2 lanes, x chunk in smem with an XOR bank swizzle.
//   wmma (M>=9):  block dequants a weight tile to smem, tensor cores
//                 (m16n16k16, fp32 accumulate) do the arithmetic.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <mma.h>

using namespace nvcuda;

#define DEV_INLINE __device__ __forceinline__

// PRMT-LUT decoder (compile with -DSKINNY_LUT_CVT to select): loses
// to the TurboMind-derived shift+rebias decoder below by ~28% at M=1
// (504 vs 647 GB/s) and ~22% at M=16 - longer PRMT dependency chain
// and more INT-pipe ops per value. Kept for A/B reference.
// fp16 high bytes of the e2m1 magnitudes {0,.5,1,1.5} and {2,3,4,6};
// the low bytes are all 0x00, so one PRMT yields two packed halves.
constexpr unsigned LUT_LO = 0x3E3C3800u;
constexpr unsigned LUT_HI = 0x46444240u;

// Dequant one byte-pair position of an 8-code word. `q` holds 8 nibbles;
// byte `pi` gives codes (2p, 2p+1) -> returns them as one half2.
DEV_INLINE half2 dequant_pair(unsigned q, int pi, half2 sc2) {
  const unsigned mq = (q & 0x77777777u) >> (8 * pi);
  const unsigned sq = (q & 0x88888888u) >> (8 * pi);
  unsigned sel = ((mq & 0x7u) << 4) | ((mq & 0x70u) << 8);
  unsigned h = __byte_perm(LUT_LO, LUT_HI, sel);
  h |= ((sq & 0x8u) << 12) | ((sq & 0x80u) << 24);
  return __hmul2(*reinterpret_cast<half2 *>(&h), sc2);
}

DEV_INLINE half2 fp8e4m3_to_half2(unsigned char b) {
  const unsigned short hb =
      (((unsigned short)b & 0x80u) << 8) | (((unsigned short)b & 0x7Fu) << 7);
  const half hs = __hmul(__ushort_as_half(hb), __ushort_as_half(0x5C00));  // *256
  return __halves2half2(hs, hs);
}

// XOR swizzle on the low 3 bits of a k-pair index; conflict-free for the
// simt read pattern (lane-groups sharing a bank base differ in p>>5).
DEV_INLINE int swz(int p) { return (p & ~7) | ((p ^ (p >> 5)) & 7); }

#ifndef SKINNY_LUT_CVT
// Alternative e2m1 decoder derived from TurboMind's cvt_f16x8_e2m1
// (Apache-2.0; 1Cat-vLLM csrc/sm70_turbomind/lmdeploy/src/turbomind/
// kernels/attention/quantization.h). Shifts sign/EM bits into fp16
// positions; the 2^14 exponent re-bias is folded into the caller's
// scale, so no extra multiply. Output half2 pairing is INTERLEAVED:
// out[i] holds codes (i, i+4) of the 8-code word.
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
#endif

// Stage 8 contiguous activation halves as four half2 pairs. Default
// pairing is adjacent-k; the TM decoder variant needs (k, k+4) pairs to
// match dequant8_tm's interleaved output.
DEV_INLINE void stage_pairs(half2 *dst, int base_pair, const uint4 &v) {
#ifndef SKINNY_LUT_CVT
  const unsigned *r = reinterpret_cast<const unsigned *>(&v);
  unsigned o[4] = {__byte_perm(r[0], r[2], 0x5410),
                   __byte_perm(r[0], r[2], 0x7632),
                   __byte_perm(r[1], r[3], 0x5410),
                   __byte_perm(r[1], r[3], 0x7632)};
#pragma unroll
  for (int j = 0; j < 4; j++)
    dst[swz(base_pair + j)] = *reinterpret_cast<half2 *>(&o[j]);
#else
  const half2 *hv = reinterpret_cast<const half2 *>(&v);
#pragma unroll
  for (int j = 0; j < 4; j++) dst[swz(base_pair + j)] = hv[j];
#endif
}

// ---------------------------------------------------------------------------
// SIMT kernel: 8 warps/block, one output row per warp.
// ---------------------------------------------------------------------------
template <int M, int KC, int R = 1, bool ARGMAX = false>
__global__ void skinny_nvfp4_simt(const uint8_t *__restrict__ codes,
                                  const uint8_t *__restrict__ scales,
                                  const half *__restrict__ x,
                                  half *__restrict__ y, int N, int K,
                                  float gscale, half *__restrict__ bvals,
                                  int *__restrict__ bidxs) {
  extern __shared__ char smem_raw[];
  half2 *xs = reinterpret_cast<half2 *>(smem_raw);  // [M][KC/2] swizzled
  constexpr int P2 = KC / 2;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = (blockIdx.x * 8 + warp) * R;
  const uint8_t *crow[R];
  const uint8_t *srow[R];
#pragma unroll
  for (int r = 0; r < R; r++) {
    crow[r] = codes + (size_t)(n + r) * (K >> 1);
    srow[r] = scales + (size_t)(n + r) * (K >> 4);
  }
  // Fold the global scale into the group scales so in-kernel weights sit
  // at their true O(0.1) magnitudes; otherwise code*fp8scale reaches
  // ~2.7e3 and fp16 products overflow on real activation outliers.
#ifndef SKINNY_LUT_CVT
  // dequant8_tm needs a 2^14 exponent re-bias; fold it here for free.
  const half2 gm2 = __float2half2_rn(gscale * 16384.f);
#else
  const half2 gm2 = __float2half2_rn(gscale);
#endif

  float accf[R][M];
#pragma unroll
  for (int r = 0; r < R; r++)
#pragma unroll
    for (int m = 0; m < M; m++) accf[r][m] = 0.f;

  int k0 = 0;
  for (; k0 + KC <= K; k0 += KC) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * (KC / 8); idx += blockDim.x) {
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      const uint4 v =
          *reinterpret_cast<const uint4 *>(x + (size_t)m * K + k0 + j4 * 8);
      stage_pairs(xs + m * P2, j4 * 4, v);
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < KC / 512; i++) {
      const int s = lane + 32 * i;  // 16-code segment == one scale group
      uint2 q2[R];
      half2 sc2[R];
#pragma unroll
      for (int r = 0; r < R; r++) {
        q2[r] = *reinterpret_cast<const uint2 *>(crow[r] + (k0 >> 1) + s * 8);
        sc2[r] = __hmul2(fp8e4m3_to_half2(srow[r][(k0 >> 4) + s]), gm2);
      }
      // fp16 accumulation window is one 16-code segment (8 products per
      // half2 lane); flushed to fp32 so real activation outliers cannot
      // overflow half range.
      half2 acch[R][M];
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < M; m++) acch[r][m] = __float2half2_rn(0.f);
#pragma unroll
      for (int w = 0; w < 2; w++) {
        half2 w4[R][4];
#pragma unroll
        for (int r = 0; r < R; r++) {
          const unsigned qw = w == 0 ? q2[r].x : q2[r].y;
#ifndef SKINNY_LUT_CVT
          dequant8_tm(qw, sc2[r], w4[r]);
#else
#pragma unroll
          for (int pi = 0; pi < 4; pi++)
            w4[r][pi] = dequant_pair(qw, pi, sc2[r]);
#endif
        }
#pragma unroll
        for (int pi = 0; pi < 4; pi++) {
          const int psw = swz(s * 8 + w * 4 + pi);
#pragma unroll
          for (int m = 0; m < M; m++) {
            const half2 xv = xs[m * P2 + psw];
#pragma unroll
            for (int r = 0; r < R; r++)
              acch[r][m] = __hfma2(w4[r][pi], xv, acch[r][m]);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < M; m++) {
          const float2 f = __half22float2(acch[r][m]);
          accf[r][m] += f.x + f.y;
        }
    }
  }

  // Tail chunk: K % KC remainder (any multiple of 128). Same layout and
  // swizzle, runtime segment bound with idle-lane guard.
  const int tail = K - k0;
  if (tail > 0) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * (tail / 8); idx += blockDim.x) {
      const int m = idx / (tail / 8), j4 = idx % (tail / 8);
      const uint4 v =
          *reinterpret_cast<const uint4 *>(x + (size_t)m * K + k0 + j4 * 8);
      stage_pairs(xs + m * P2, j4 * 4, v);
    }
    __syncthreads();
    const int nseg = tail >> 4;
    for (int s = lane; s < nseg; s += 32) {
      uint2 q2[R];
      half2 sc2[R];
#pragma unroll
      for (int r = 0; r < R; r++) {
        q2[r] = *reinterpret_cast<const uint2 *>(crow[r] + (k0 >> 1) + s * 8);
        sc2[r] = __hmul2(fp8e4m3_to_half2(srow[r][(k0 >> 4) + s]), gm2);
      }
      half2 acch[R][M];
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < M; m++) acch[r][m] = __float2half2_rn(0.f);
#pragma unroll
      for (int w = 0; w < 2; w++) {
        half2 w4[R][4];
#pragma unroll
        for (int r = 0; r < R; r++) {
          const unsigned qw = w == 0 ? q2[r].x : q2[r].y;
#ifndef SKINNY_LUT_CVT
          dequant8_tm(qw, sc2[r], w4[r]);
#else
#pragma unroll
          for (int pi = 0; pi < 4; pi++)
            w4[r][pi] = dequant_pair(qw, pi, sc2[r]);
#endif
        }
#pragma unroll
        for (int pi = 0; pi < 4; pi++) {
          const int psw = swz(s * 8 + w * 4 + pi);
#pragma unroll
          for (int m = 0; m < M; m++) {
            const half2 xv = xs[m * P2 + psw];
#pragma unroll
            for (int r = 0; r < R; r++)
              acch[r][m] = __hfma2(w4[r][pi], xv, acch[r][m]);
          }
        }
      }
#pragma unroll
      for (int r = 0; r < R; r++)
#pragma unroll
        for (int m = 0; m < M; m++) {
          const float2 f = __half22float2(acch[r][m]);
          accf[r][m] += f.x + f.y;
        }
    }
  }

  if constexpr (!ARGMAX) {
#pragma unroll
    for (int r = 0; r < R; r++)
#pragma unroll
      for (int m = 0; m < M; m++) {
        float v = accf[r][m];
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
        if (lane == 0) y[(size_t)m * N + n + r] = __float2half(v);
      }
  } else {
    // Fused greedy argmax (M=1): identical reduce, identical
    // __float2half rounding, then compare halfs with strict > so ties
    // keep the LOWEST index — matching argmax-over-half semantics of
    // the separate path. One (val, idx) pair per block; no logits hit
    // HBM.
    __shared__ half wval[8];
    __shared__ int widx[8];
    half best_h = __float2half(-3.0e38f);
    int best_i = -1;
#pragma unroll
    for (int r = 0; r < R; r++) {
      float v = accf[r][0];
#pragma unroll
      for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(~0u, v, o);
      if (lane == 0) {
        const half h = __float2half(v);
        if (__hgt(h, best_h)) { best_h = h; best_i = n + r; }
      }
    }
    if (lane == 0) { wval[warp] = best_h; widx[warp] = best_i; }
    __syncthreads();
    if (threadIdx.x == 0) {
      half bh = wval[0];
      int bi = widx[0];
#pragma unroll
      for (int w = 1; w < 8; w++)
        if (__hgt(wval[w], bh)) { bh = wval[w]; bi = widx[w]; }
      bvals[blockIdx.x] = bh;
      bidxs[blockIdx.x] = bi;
    }
  }
}

// ---------------------------------------------------------------------------
// WMMA kernel: WN x WM warps of 16x16 output tiles, KC-deep smem staging.
// Software-pipelined: while the tensor cores chew on chunk i, each thread's
// gmem loads for chunk i+1 are already in flight into registers.
// ---------------------------------------------------------------------------
template <int WN, int WM, int KC>
__global__ void skinny_nvfp4_wmma(const uint8_t *__restrict__ codes,
                                  const uint8_t *__restrict__ scales,
                                  const half *__restrict__ x,
                                  half *__restrict__ y, int N, int K,
                                  int m_real, float gscale) {
  constexpr int NT = WN * 16, MT = WM * 16;
  constexpr int PW = KC + 16, PX = KC + 16;  // padded smem pitches (halfs)
  constexpr int NTHREADS = WN * WM * 32;
  constexpr int CSEG = NT * (KC / 16) / NTHREADS;  // code segs per thread
  constexpr int XSEG = MT * (KC / 8) / NTHREADS;   // x uint4s per thread
  static_assert(CSEG * NTHREADS == NT * (KC / 16), "code seg split");
  static_assert(XSEG * NTHREADS == MT * (KC / 8), "x seg split");

  extern __shared__ char smem_raw[];
  half *ws = reinterpret_cast<half *>(smem_raw);  // [NT][PW]
  half *xs = ws + NT * PW;                        // [MT][PX]

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int wn = warp % WN, wm = warp / WN;
  const int nb = blockIdx.x * NT;

  uint2 st_c[CSEG];
  unsigned char st_s[CSEG];
  uint4 st_x[XSEG];

  auto load_stage = [&](int k0) {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int n = idx / (KC / 16), s = idx % (KC / 16);
      st_c[i] = __ldcs(reinterpret_cast<const uint2 *>(
          codes + (size_t)(nb + n) * (K >> 1) + (k0 >> 1) + s * 8));
      st_s[i] = __ldcs(scales + (size_t)(nb + n) * (K >> 4) + (k0 >> 4) + s);
    }
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

  auto store_stage = [&]() {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int n = idx / (KC / 16), s = idx % (KC / 16);
      const half2 sc2 = fp8e4m3_to_half2(st_s[i]);
      half2 *wrow = reinterpret_cast<half2 *>(ws + n * PW + s * 16);
      const unsigned qs[2] = {st_c[i].x, st_c[i].y};
#pragma unroll
      for (int w = 0; w < 2; w++) {
#ifndef SKINNY_LUT_CVT
        half2 t[4];
        dequant8_tm(qs[w], sc2, t);  // values carry a 2^-14 factor here
        const unsigned *tr = reinterpret_cast<const unsigned *>(t);
        unsigned lin[4] = {__byte_perm(tr[0], tr[1], 0x5410),
                           __byte_perm(tr[2], tr[3], 0x5410),
                           __byte_perm(tr[0], tr[1], 0x7632),
                           __byte_perm(tr[2], tr[3], 0x7632)};
#pragma unroll
        for (int pi = 0; pi < 4; pi++)
          wrow[w * 4 + pi] = *reinterpret_cast<half2 *>(&lin[pi]);
#else
#pragma unroll
        for (int pi = 0; pi < 4; pi++)
          wrow[w * 4 + pi] = dequant_pair(qs[w], pi, sc2);
#endif
      }
    }
#pragma unroll
    for (int i = 0; i < XSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      *reinterpret_cast<uint4 *>(xs + m * PX + j4 * 8) = st_x[i];
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cfrag;
  wmma::fill_fragment(cfrag, 0.f);

  load_stage(0);
  for (int k0 = 0; k0 < K; k0 += KC) {
    __syncthreads();  // previous chunk's mma done; smem free to overwrite
    store_stage();
    __syncthreads();
    if (k0 + KC < K) load_stage(k0 + KC);  // in flight during the mma loop

    // Double-buffered fragments: kk+1's shared loads overlap kk's mma.
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
  }

  __syncthreads();  // done with ws/xs; reuse for the fp32 epilogue stage
  float *cs = reinterpret_cast<float *>(smem_raw) + warp * 256;
  wmma::store_matrix_sync(cs, cfrag, 16, wmma::mem_row_major);
  __syncwarp();
  for (int e = lane; e < 256; e += 32) {
    const int i = e >> 4, j = e & 15;  // i: n within tile, j: m within tile
    const int gm = wm * 16 + j, gn = nb + wn * 16 + i;
    #ifndef SKINNY_LUT_CVT
    const float gs_eff = gscale * 16384.f;  // undo dequant8_tm's 2^-14
#else
    const float gs_eff = gscale;
#endif
    if (gm < m_real) y[(size_t)gm * N + gn] = __float2half(cs[e] * gs_eff);
  }
}

// ---------------------------------------------------------------------------
// DP4A kernel: int8 SIMT path for the compute-bound M band (4..16).
//
// The e2m1 lattice x2 is integer-exact ({0,1,2,3,4,6,8,12}), so the
// weight side is lossless int8; only activations are quantized
// (symmetric per-16-group int8, done host-side per GEMM call and
// amortized over all N rows). dp4a delivers 4 MACs per issue slot vs
// HFMA2's 2, which is the whole bet: the M-scaling penalty of the
// fp16 SIMT path is issue-bound MAC work.
// ---------------------------------------------------------------------------
DEV_INLINE unsigned dp4a_unpack_mag(unsigned nib4) {
  // nib4 holds 4 e2m1 codes in the low nibbles of each byte
  // (0x0c0c0c0c layout). Returns 4 packed uint8 magnitudes x2 via the
  // shift identity: mag2(c) = c < 2 ? c : (2 + (c&1)) << ((c>>1) - 1).
  unsigned out = 0;
#pragma unroll
  for (int b = 0; b < 4; b++) {
    const unsigned c = (nib4 >> (8 * b)) & 0x7u;
    const unsigned m = c < 2 ? c : (2u + (c & 1u)) << ((c >> 1) - 1u);
    out |= m << (8 * b);
  }
  return out;
}

// One thread per 16-element group: absmax -> scale -> int8 quantize.
// Single launch replaces an 8-op torch chain (which cost 168us of
// launch overhead at these sizes).
__global__ void skinny_quant_a8(const half *__restrict__ x,
                                int8_t *__restrict__ xq,
                                float *__restrict__ xs, int total_groups,
                                int k) {
  const int g = blockIdx.x * blockDim.x + threadIdx.x;
  if (g >= total_groups) return;
  const half2 *xp = reinterpret_cast<const half2 *>(x) + g * 8;
  float amax = 1e-8f;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const half2 v = xp[i];
    amax = fmaxf(amax, fmaxf(fabsf(__half2float(__low2half(v))),
                             fabsf(__half2float(__high2half(v)))));
  }
  const float s = amax / 127.f, inv = 127.f / amax;
  char4 *out = reinterpret_cast<char4 *>(xq) + g * 4;
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const half2 a = xp[i * 2], b = xp[i * 2 + 1];
    char4 c;
    c.x = (signed char)__float2int_rn(__half2float(__low2half(a)) * inv);
    c.y = (signed char)__float2int_rn(__half2float(__high2half(a)) * inv);
    c.z = (signed char)__float2int_rn(__half2float(__low2half(b)) * inv);
    c.w = (signed char)__float2int_rn(__half2float(__high2half(b)) * inv);
    out[i] = c;
  }
  xs[g] = s;
  (void)k;
}

template <int M, int KC>
__global__ void skinny_nvfp4_dp4a(const uint8_t *__restrict__ codes,
                                  const uint8_t *__restrict__ scales,
                                  const int8_t *__restrict__ xq,
                                  const float *__restrict__ xs_scale,
                                  half *__restrict__ y, int N, int K,
                                  float gscale) {
  extern __shared__ char smem_raw[];
  // staged activations: int32 words [M][KC/4] then scales [M][KC/16]
  int *xw = reinterpret_cast<int *>(smem_raw);
  float *xsc = reinterpret_cast<float *>(xw + M * (KC / 4));
  constexpr int PW4 = KC / 4, PS = KC / 16;

  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int n = blockIdx.x * 8 + warp;
  const uint8_t *crow = codes + (size_t)n * (K >> 1);
  const uint8_t *srow = scales + (size_t)n * (K >> 4);

  float accf[M];
#pragma unroll
  for (int m = 0; m < M; m++) accf[m] = 0.f;

  for (int k0 = 0; k0 < K; k0 += KC) {
    __syncthreads();
    for (int idx = threadIdx.x; idx < M * (KC / 4); idx += blockDim.x) {
      const int m = idx / (KC / 4), j = idx % (KC / 4);
      const int js = (j & ~3) | ((j ^ (j >> 5)) & 3);  // bank swizzle
      xw[m * PW4 + js] = reinterpret_cast<const int *>(
          xq + (size_t)m * K + k0)[j];
    }
    for (int idx = threadIdx.x; idx < M * (KC / 16); idx += blockDim.x) {
      const int m = idx / (KC / 16), g = idx % (KC / 16);
      xsc[m * PS + g] = xs_scale[(size_t)m * (K / 16) + (k0 / 16) + g];
    }
    __syncthreads();

#pragma unroll
    for (int i = 0; i < (KC / 16 + 31) / 32; i++) {
      const int s = lane + 32 * i;  // 16-code segment == one scale group
      if (KC / 16 < 32 && s >= KC / 16) break;
      const uint2 q2 = *reinterpret_cast<const uint2 *>(
          crow + (k0 >> 1) + s * 8);
      // fp8 scale -> float, x0.5 compensates the x2 integer lattice
      const half2 sch = fp8e4m3_to_half2(srow[(k0 >> 4) + s]);
      const float wsc = __half2float(__low2half(sch)) * 0.5f;

      // unpack 16 codes -> 4 dp4a words (int8, sign applied)
      unsigned w4[4];
#pragma unroll
      for (int hw = 0; hw < 2; hw++) {
        const unsigned q = hw == 0 ? q2.x : q2.y;
        const unsigned lo = q & 0x0F0F0F0Fu;         // codes 0,2,4,6
        const unsigned hi = (q >> 4) & 0x0F0F0F0Fu;  // codes 1,3,5,7
        // interleave back to k-order nibble words: bytes of `lo` are
        // even k, bytes of `hi` odd k -> two words of 4 consecutive k
        const unsigned w01 = __byte_perm(lo, hi, 0x5140);  // k0..k3
        const unsigned w23 = __byte_perm(lo, hi, 0x7362);  // k4..k7
        const unsigned wds[2] = {w01, w23};
#pragma unroll
        for (int t = 0; t < 2; t++) {
          const unsigned mag = dp4a_unpack_mag(wds[t]);
          const unsigned sgn = (wds[t] >> 3) & 0x01010101u;  // sign bits
          const unsigned msk = sgn * 0xFFu;  // 0x00 or 0xFF per byte
          // two's complement negate where sign: (mag ^ msk) + (msk & 1)
          w4[hw * 2 + t] = __vadd4(mag ^ msk, sgn);
        }
      }

      int acc32[M];
#pragma unroll
      for (int m = 0; m < M; m++) acc32[m] = 0;
#pragma unroll
      for (int m = 0; m < M; m++) {
        const int *xrow = xw + m * PW4;
#pragma unroll
        for (int t = 0; t < 4; t++) {
          const int j = s * 4 + t;
          const int js = (j & ~3) | ((j ^ (j >> 5)) & 3);
          acc32[m] = __dp4a((int)w4[t], xrow[js], acc32[m]);
        }
      }
#pragma unroll
      for (int m = 0; m < M; m++)
        accf[m] += (float)acc32[m] * (wsc * xsc[m * PS + s]);
    }
  }

  // warp-reduce the per-lane segment partials before the single write
  const float gs = gscale;
#pragma unroll
  for (int m = 0; m < M; m++) {
    float v = accf[m];
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
      v += __shfl_down_sync(0xffffffffu, v, off);
    if (lane == 0) y[(size_t)m * N + n] = __float2half(v * gs);
  }
}

// ---------------------------------------------------------------------------
// Host dispatch
// ---------------------------------------------------------------------------
static void check_inputs(const torch::Tensor &x, const torch::Tensor &codes,
                         const torch::Tensor &scales, int64_t &m, int64_t &n,
                         int64_t &k) {
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(codes.is_cuda() && codes.dtype() == torch::kUInt8 &&
              codes.is_contiguous());
  TORCH_CHECK(scales.is_cuda() && scales.dtype() == torch::kUInt8 &&
              scales.is_contiguous());
  m = x.size(0);
  k = x.size(1);
  n = codes.size(0);
  TORCH_CHECK(codes.size(1) * 2 == k, "codes/x K mismatch");
  TORCH_CHECK(scales.size(0) == n && scales.size(1) * 16 == k);
}

// ---------------------------------------------------------------------------
// MMA8 kernel: Volta mma.sync.m8n8k4 register-fragment path for 2<=M<=8.
//
// The 16-row WMMA tile pads M=5 verify batches 3.2x; SIMT carries M FMAs
// per weight byte and goes compute-bound by M=5. This path streams weights
// through registers exactly like SIMT but hands the MACs to the tensor
// cores at their native 8-row tile, so M<=8 all run at the same cost.
//
// Fragment maps were derived empirically on V100 (mma8_probe.cu):
//   A row-major / B col-major: QP lanes {0-3,16-19} hold row/col
//   (lane&3)+4*(lane>=16), 4 contiguous k each.
//   C fp32: reg i of lane L -> row (i&2)|(L>=16?4:0)|(L&1),
//   col (i&1)|(((L>>1)&1)<<1)|((i>>2)<<2).
// Warp = 8 weight rows; the 4 QPs split K in 64-element superchunks
// (32B-sector-coalesced code reads), butterfly-reduced once at the end.
// fp32 accumulation end to end: no fp16 overflow window.
// ---------------------------------------------------------------------------
template <int KC>
__global__ void skinny_nvfp4_mma8(const uint8_t *__restrict__ codes,
                                  const uint8_t *__restrict__ scales,
                                  const half *__restrict__ x,
                                  half *__restrict__ y, int N, int K, int M,
                                  float gscale) {
  extern __shared__ char smem_raw[];
  half2 *xs = reinterpret_cast<half2 *>(smem_raw);  // [8][PITCH] plain k order
  constexpr int PITCH = KC / 2 + 1;  // odd pitch: 8 rows spread the banks

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int qp = (lane >> 2) & 3;
  const int idx = (lane & 3) + ((lane & 16) ? 4 : 0);
  const int n0 = (blockIdx.x * (int)(blockDim.x >> 5) + warp) * 8;
  const uint8_t *crow = codes + (size_t)(n0 + idx) * (K >> 1);
  const uint8_t *srow = scales + (size_t)(n0 + idx) * (K >> 4);

#ifndef SKINNY_LUT_CVT
  const half2 gm2 = __float2half2_rn(gscale * 16384.f);
#else
  const half2 gm2 = __float2half2_rn(gscale);
#endif

  float c[8];
#pragma unroll
  for (int i = 0; i < 8; i++) c[i] = 0.f;

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
      const int kb = k0 + s * 64 + qp * 16;  // this QP's 16-k window
      const uint2 q2 =
          __ldcs(reinterpret_cast<const uint2 *>(crow + (kb >> 1)));
      const half2 sc2 = __hmul2(fp8e4m3_to_half2(__ldg(srow + (kb >> 4))), gm2);
      // A fragments, k-contiguous quads for the 4 mmas of this window.
      half2 af[4][2];
#ifndef SKINNY_LUT_CVT
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
#else
#pragma unroll
      for (int j = 0; j < 4; j++) {  // adjacent pairs are already k-order
        const unsigned qw = (j < 2) ? q2.x : q2.y;
        af[j][0] = dequant_pair(qw, (j & 1) * 2, sc2);
        af[j][1] = dequant_pair(qw, (j & 1) * 2 + 1, sc2);
      }
#endif
      const half2 *xrow = xs + idx * PITCH + ((kb - k0) >> 1);
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const half2 b0 = xrow[2 * j], b1 = xrow[2 * j + 1];
        const unsigned *A = reinterpret_cast<const unsigned *>(af[j]);
        asm volatile(
            "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
            "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
            "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
            : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]),
              "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
            : "r"(A[0]), "r"(A[1]),
              "r"(*reinterpret_cast<const unsigned *>(&b0)),
              "r"(*reinterpret_cast<const unsigned *>(&b1)));
      }
    }
  }

#pragma unroll
  for (int i = 0; i < 8; i++) {
    c[i] += __shfl_xor_sync(0xffffffffu, c[i], 4);
    c[i] += __shfl_xor_sync(0xffffffffu, c[i], 8);
  }
  if ((lane & 12) == 0) {
#pragma unroll
    for (int i = 0; i < 8; i++) {
      const int r = (i & 2) | ((lane & 16) ? 4 : 0) | (lane & 1);
      const int cc = (i & 1) | (((lane >> 1) & 1) << 1) | ((i >> 2) << 2);
      if (cc < M) y[(size_t)cc * N + n0 + r] = __float2half(c[i]);
    }
  }
}

std::vector<torch::Tensor> skinny_gemm_simt_argmax(torch::Tensor x,
                                                   torch::Tensor codes,
                                                   torch::Tensor scales,
                                                   double gscale) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  constexpr int KC = 1024;
  TORCH_CHECK(m == 1, "fused argmax is M=1 only");
  TORCH_CHECK(k % 128 == 0 && k >= 128, "K must be a multiple of 128");
  TORCH_CHECK(n % 8 == 0, "N must be a multiple of 8");
  const bool two_rows = (k <= 2048) && (n % 16 == 0);
  const int nblocks = two_rows ? (int)n / 16 : (int)n / 8;
  auto bvals = torch::empty({nblocks}, x.options());
  auto bidxs = torch::empty({nblocks},
                            x.options().dtype(torch::kInt32));
  const dim3 grid(nblocks), block(256);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = (KC / 2) * sizeof(half2);
  if (two_rows)
    skinny_nvfp4_simt<1, KC, 2, true><<<grid, block, smem, stream>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()), nullptr,
        (int)n, (int)k, (float)gscale,
        reinterpret_cast<half *>(bvals.data_ptr<at::Half>()),
        bidxs.data_ptr<int>());
  else
    skinny_nvfp4_simt<1, KC, 1, true><<<grid, block, smem, stream>>>(
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()), nullptr,
        (int)n, (int)k, (float)gscale,
        reinterpret_cast<half *>(bvals.data_ptr<at::Half>()),
        bidxs.data_ptr<int>());
  return {bvals, bidxs};
}

torch::Tensor skinny_gemm_simt(torch::Tensor x, torch::Tensor codes,
                               torch::Tensor scales, double gscale) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  constexpr int KC = 1024;
  TORCH_CHECK(k % 128 == 0 && k >= 128, "K must be a multiple of 128");
  TORCH_CHECK(n % 8 == 0, "N must be a multiple of 8");
  auto y = torch::empty({m, n}, x.options());
  // Short-K rows leave <2 weight loads in flight per thread; two rows
  // per warp restores latency hiding (shape diagnostic: out_proj K=1536
  // ran at 66% of flagship bandwidth with one row per warp).
  const bool two_rows = (k <= 2048) && (n % 16 == 0);
  const dim3 grid(two_rows ? n / 16 : n / 8), block(256);
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = (int)m * (KC / 2) * sizeof(half2);

#define LAUNCH_SIMT(MM)                                                   \
  if (two_rows)                                                           \
    skinny_nvfp4_simt<MM, KC, 2><<<grid, block, smem, stream>>>(          \
        codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),            \
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),           \
        reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k, \
        (float)gscale, nullptr, nullptr);                                 \
  else                                                                    \
    skinny_nvfp4_simt<MM, KC, 1><<<grid, block, smem, stream>>>(          \
      codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),              \
      reinterpret_cast<const half *>(x.data_ptr<at::Half>()),             \
      reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
      (float)gscale, nullptr, nullptr)

  switch (m) {
    case 1: LAUNCH_SIMT(1); break;
    case 2: LAUNCH_SIMT(2); break;
    case 3: LAUNCH_SIMT(3); break;
    case 4: LAUNCH_SIMT(4); break;
    case 5: LAUNCH_SIMT(5); break;
    case 6: LAUNCH_SIMT(6); break;
    case 7: LAUNCH_SIMT(7); break;
    case 8: LAUNCH_SIMT(8); break;
    default: TORCH_CHECK(false, "simt kernel supports M in 1..8, got ", m);
  }
#undef LAUNCH_SIMT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor skinny_gemm_wmma(torch::Tensor x, torch::Tensor codes,
                               torch::Tensor scales, double gscale) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_WMMA(WN, WM, KC)                                             \
  do {                                                                      \
    constexpr int NT = WN * 16, MT = WM * 16;                               \
    TORCH_CHECK(n % NT == 0, "N must be a multiple of ", NT);               \
    TORCH_CHECK(k % KC == 0, "K must be a multiple of ", KC);               \
    const int smem = (NT + MT) * (KC + 16) * (int)sizeof(half);             \
    skinny_nvfp4_wmma<WN, WM, KC>                                           \
        <<<dim3(n / NT), dim3(WN * WM * 32), smem, stream>>>(               \
            codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),          \
            reinterpret_cast<const half *>(x.data_ptr<at::Half>()),         \
            reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n,       \
            (int)k, (int)m, (float)gscale);                                 \
  } while (0)

  if (m <= 16) LAUNCH_WMMA(4, 1, 256);       // 128B code rows, 2 CTAs/SM
  // (2,2,128) for the 17-32 band: NT=32 doubles the grid on N-starved
  // shapes (occupancy sweep 2026-08: 0.327 vs 0.350 ms/layer-set, -6.6%).
  else if (m <= 32) LAUNCH_WMMA(2, 2, 128);
  else if (m <= 64) LAUNCH_WMMA(2, 4, 128);
  else TORCH_CHECK(false, "wmma kernel supports M <= 64, got ", m);
#undef LAUNCH_WMMA
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor skinny_gemm_mma8(torch::Tensor x, torch::Tensor codes,
                               torch::Tensor scales, double gscale) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  constexpr int KC = 256, WARPS = 4;
  TORCH_CHECK(m >= 1 && m <= 8, "mma8 kernel supports M in 1..8, got ", m);
  TORCH_CHECK(k % KC == 0 && k >= KC, "K must be a multiple of ", KC);
  TORCH_CHECK(n % (WARPS * 8) == 0, "N must be a multiple of ", WARPS * 8);
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = 8 * (KC / 2 + 1) * (int)sizeof(half2);
  skinny_nvfp4_mma8<KC>
      <<<dim3(n / (WARPS * 8)), dim3(WARPS * 32), smem, stream>>>(
          codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),
          reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
          reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,
          (int)m, (float)gscale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// Split-K variant: each CTA covers one K-slice, accumulating fp32
// partials into ypart via atomicAdd (S contenders per element at most).
// Caller converts ypart * gs_eff -> half. Multiplies grid occupancy by
// the split factor on N-starved shapes.
template <int WN, int WM, int KC>
__global__ void skinny_nvfp4_wmma_ks(const uint8_t *__restrict__ codes,
                                     const uint8_t *__restrict__ scales,
                                     const half *__restrict__ x,
                                     float *__restrict__ ypart, int N, int K,
                                     int m_real, int k_slice) {
  constexpr int NT = WN * 16, MT = WM * 16;
  constexpr int PW = KC + 16, PX = KC + 16;
  constexpr int NTHREADS = WN * WM * 32;
  constexpr int CSEG = NT * (KC / 16) / NTHREADS;
  constexpr int XSEG = MT * (KC / 8) / NTHREADS;
  static_assert(CSEG * NTHREADS == NT * (KC / 16), "code seg split");
  static_assert(XSEG * NTHREADS == MT * (KC / 8), "x seg split");

  extern __shared__ char smem_raw[];
  half *ws = reinterpret_cast<half *>(smem_raw);
  half *xs = ws + NT * PW;

  const int tid = threadIdx.x;
  const int warp = tid >> 5, lane = tid & 31;
  const int wn = warp % WN, wm = warp / WN;
  const int nb = blockIdx.x * NT;
  const int kbeg = blockIdx.y * k_slice;
  const int kend = min(kbeg + k_slice, K);

  uint2 st_c[CSEG];
  unsigned char st_s[CSEG];
  uint4 st_x[XSEG];

  auto load_stage = [&](int k0) {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int n = idx / (KC / 16), s = idx % (KC / 16);
      st_c[i] = __ldcs(reinterpret_cast<const uint2 *>(
          codes + (size_t)(nb + n) * (K >> 1) + (k0 >> 1) + s * 8));
      st_s[i] = __ldcs(scales + (size_t)(nb + n) * (K >> 4) + (k0 >> 4) + s);
    }
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
  auto store_stage = [&]() {
#pragma unroll
    for (int i = 0; i < CSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int n = idx / (KC / 16), s = idx % (KC / 16);
      const half2 sc2 = fp8e4m3_to_half2(st_s[i]);
      half2 *wrow = reinterpret_cast<half2 *>(ws + n * PW + s * 16);
      const unsigned qs[2] = {st_c[i].x, st_c[i].y};
#pragma unroll
      for (int w = 0; w < 2; w++) {
#ifndef SKINNY_LUT_CVT
        half2 t[4];
        dequant8_tm(qs[w], sc2, t);
        const unsigned *tr = reinterpret_cast<const unsigned *>(t);
        unsigned lin[4] = {__byte_perm(tr[0], tr[1], 0x5410),
                           __byte_perm(tr[2], tr[3], 0x5410),
                           __byte_perm(tr[0], tr[1], 0x7632),
                           __byte_perm(tr[2], tr[3], 0x7632)};
#pragma unroll
        for (int pi = 0; pi < 4; pi++)
          wrow[w * 4 + pi] = *reinterpret_cast<half2 *>(&lin[pi]);
#else
#pragma unroll
        for (int pi = 0; pi < 4; pi++)
          wrow[w * 4 + pi] = dequant_pair(qs[w], pi, sc2);
#endif
      }
    }
#pragma unroll
    for (int i = 0; i < XSEG; i++) {
      const int idx = tid + i * NTHREADS;
      const int m = idx / (KC / 8), j4 = idx % (KC / 8);
      *reinterpret_cast<uint4 *>(xs + m * PX + j4 * 8) = st_x[i];
    }
  };

  wmma::fragment<wmma::accumulator, 16, 16, 16, float> cfrag;
  wmma::fill_fragment(cfrag, 0.f);

  load_stage(kbeg);
  for (int k0 = kbeg; k0 < kend; k0 += KC) {
    __syncthreads();
    store_stage();
    __syncthreads();
    if (k0 + KC < kend) load_stage(k0 + KC);

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
  }

  __syncthreads();
  float *cs = reinterpret_cast<float *>(smem_raw) + warp * 256;
  wmma::store_matrix_sync(cs, cfrag, 16, wmma::mem_row_major);
  __syncwarp();
  for (int e = lane; e < 256; e += 32) {
    const int i = e >> 4, j = e & 15;
    const int gm = wm * 16 + j, gn = nb + wn * 16 + i;
    if (gm < m_real) atomicAdd(&ypart[(size_t)gm * N + gn], cs[e]);
  }
}

// Config-selectable WMMA entry for tile sweeps: cfg indexes the
// (WN, WM, KC) table below. Configs 8-9 exceed the default 48KB smem
// ceiling and require carveout=true (96KB opt-in). N must divide WN*16.
static void set_smem_opt(const void *kern, int smem) {
  cudaFuncSetAttribute(kern, cudaFuncAttributeMaxDynamicSharedMemorySize,
                       96 * 1024);
  cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout,
                       100);
  (void)smem;
}

torch::Tensor skinny_gemm_wmma_cfg(torch::Tensor x, torch::Tensor codes,
                                   torch::Tensor scales, double gscale,
                                   int64_t cfg, bool carveout) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_WMMA_CFG(WN, WM, KC)                                         \
  do {                                                                      \
    constexpr int NT = WN * 16, MT = WM * 16;                               \
    TORCH_CHECK(n % NT == 0, "N must be a multiple of ", NT);               \
    TORCH_CHECK(k % KC == 0, "K must be a multiple of ", KC);               \
    TORCH_CHECK(m <= MT, "cfg supports M <= ", MT);                         \
    const int smem = (NT + MT) * (KC + 16) * (int)sizeof(half);             \
    TORCH_CHECK(smem <= 48 * 1024 || carveout,                              \
                "cfg needs 96KB smem carveout");                            \
    if (carveout)                                                           \
      set_smem_opt((const void *)&skinny_nvfp4_wmma<WN, WM, KC>, smem);     \
    skinny_nvfp4_wmma<WN, WM, KC>                                           \
        <<<dim3(n / NT), dim3(WN * WM * 32), smem, stream>>>(               \
            codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),          \
            reinterpret_cast<const half *>(x.data_ptr<at::Half>()),         \
            reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n,       \
            (int)k, (int)m, (float)gscale);                                 \
  } while (0)

  switch (cfg) {
    case 0: LAUNCH_WMMA_CFG(4, 1, 256); break;
    case 1: LAUNCH_WMMA_CFG(4, 2, 128); break;
    case 2: LAUNCH_WMMA_CFG(2, 4, 128); break;
    case 3: LAUNCH_WMMA_CFG(8, 1, 128); break;
    case 4: LAUNCH_WMMA_CFG(8, 2, 128); break;
    case 5: LAUNCH_WMMA_CFG(4, 4, 128); break;
    case 6: LAUNCH_WMMA_CFG(2, 1, 256); break;
    case 7: LAUNCH_WMMA_CFG(2, 2, 128); break;
    case 8: LAUNCH_WMMA_CFG(4, 1, 512); break;
    case 9: LAUNCH_WMMA_CFG(2, 4, 256); break;
    default: TORCH_CHECK(false, "unknown wmma cfg ", cfg);
  }
#undef LAUNCH_WMMA_CFG
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// Split-K entry: returns fp32 partial sums (caller applies gscale and
// converts). splits CTAs along K; k must divide evenly into
// KC-aligned slices.
torch::Tensor skinny_gemm_wmma_splitk(torch::Tensor x, torch::Tensor codes,
                                      torch::Tensor scales, int64_t cfg,
                                      int64_t splits, bool carveout) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  auto ypart = torch::zeros(
      {m, n}, x.options().dtype(torch::kFloat32));
  auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH_WMMA_KS(WN, WM, KC)                                         \
  do {                                                                     \
    constexpr int NT = WN * 16, MT = WM * 16;                              \
    TORCH_CHECK(n % NT == 0, "N must be a multiple of ", NT);              \
    TORCH_CHECK(m <= MT, "cfg supports M <= ", MT);                        \
    const int64_t kper = (k / splits / KC) * KC;                           \
    TORCH_CHECK(kper > 0 && kper * splits == k,                            \
                "K must split into KC-aligned slices");                    \
    const int smem = (NT + MT) * (KC + 16) * (int)sizeof(half);            \
    TORCH_CHECK(smem <= 48 * 1024 || carveout,                             \
                "cfg needs 96KB smem carveout");                           \
    if (carveout)                                                          \
      set_smem_opt((const void *)&skinny_nvfp4_wmma_ks<WN, WM, KC>, smem); \
    skinny_nvfp4_wmma_ks<WN, WM, KC>                                       \
        <<<dim3(n / NT, splits), dim3(WN * WM * 32), smem, stream>>>(      \
            codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),         \
            reinterpret_cast<const half *>(x.data_ptr<at::Half>()),        \
            ypart.data_ptr<float>(), (int)n, (int)k, (int)m, (int)kper);   \
  } while (0)

  switch (cfg) {
    case 0: LAUNCH_WMMA_KS(4, 1, 256); break;
    case 1: LAUNCH_WMMA_KS(4, 2, 128); break;
    case 2: LAUNCH_WMMA_KS(2, 4, 128); break;
    case 6: LAUNCH_WMMA_KS(2, 1, 256); break;
    case 7: LAUNCH_WMMA_KS(2, 2, 128); break;
    default: TORCH_CHECK(false, "splitk unsupported for cfg ", cfg);
  }
#undef LAUNCH_WMMA_KS
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return ypart;
}

torch::Tensor skinny_gemm_dp4a(torch::Tensor x, torch::Tensor codes,
                               torch::Tensor scales, double gscale) {
  int64_t m, n, k;
  check_inputs(x, codes, scales, m, n, k);
  TORCH_CHECK(n % 8 == 0, "N must be a multiple of 8");
  const int kc = (k % 1024 == 0) ? 1024 : 256;
  TORCH_CHECK(k % kc == 0, "K must be a multiple of 256");

  auto xq = torch::empty({m, k}, x.options().dtype(torch::kChar));
  auto xsc = torch::empty({m, k / 16}, x.options().dtype(torch::kFloat32));
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int groups = (int)(m * (k / 16));
  skinny_quant_a8<<<(groups + 255) / 256, 256, 0, stream>>>(
      reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
      xq.data_ptr<int8_t>(), xsc.data_ptr<float>(), groups, (int)k);

#define LAUNCH_DP4A(MM, KCC)                                              \
  skinny_nvfp4_dp4a<MM, KCC><<<n / 8, 256,                                \
      MM * (KCC / 4) * 4 + MM * (KCC / 16) * 4, stream>>>(                \
      codes.data_ptr<uint8_t>(), scales.data_ptr<uint8_t>(),              \
      xq.data_ptr<int8_t>(), xsc.data_ptr<float>(),                      \
      reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,   \
      (float)gscale)
#define DP4A_CASE(MM)                                                     \
  case MM:                                                                \
    if (kc == 1024) LAUNCH_DP4A(MM, 1024); else LAUNCH_DP4A(MM, 256);    \
    break

  switch (m) {
    DP4A_CASE(4);
    DP4A_CASE(5);
    DP4A_CASE(7);
    DP4A_CASE(8);
    DP4A_CASE(11);
    DP4A_CASE(16);
    default: TORCH_CHECK(false, "dp4a supports M in {4,5,7,8,11,16}");
  }
#undef DP4A_CASE
#undef LAUNCH_DP4A
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

// ---------------------------------------------------------------------------
// QPN kernel: Volta-native four-quadpair mma.m8n8k4, M 4..16 band.
//
// The quadpairs split the N dimension: one warp instruction = four
// independent 8x8x4 MMAs sharing a single 8x4 activation A tile (the A
// fragment map depends only on lane-position inside the quadpair, so
// QP-sibling lanes hold identical A registers). MT template = number of
// 8-row A tiles (MT=2 decodes B once for M 9..16). Weights arrive
// PREPACKED in fragment order ([tile N/32][group K/16][lane 32] x 8B,
// nibbles pre-interleaved so dequant8_tm's (i, i+4) output IS the
// adjacent-k B register pair) — built once at weight load by the shim's
// _qpn_prepack; the checkpoint-native layout stays for SIMT/WMMA.
// No smem in the main loop, no barriers except the cross-warp K-reduce
// at output (CTA = 4 warps splitting K to keep the grid at N/32).
// Frontier (qpn_sweep_20260810): simt M<=3, qpn 4..16, wmma 17..64 —
// 1.28x at M=5, 1.69x at M=8, 1.29x at M=11, 1.22x at M=16 vs the
// prior best incumbent on the 5-shape production set.
// ---------------------------------------------------------------------------
#define MMA_8N8K4(C, A0, A1, B0, B1)                                        \
  asm volatile(                                                             \
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "                    \
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "                     \
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"                                        \
      : "+f"(C[0]), "+f"(C[1]), "+f"(C[2]), "+f"(C[3]), "+f"(C[4]),         \
        "+f"(C[5]), "+f"(C[6]), "+f"(C[7])                                  \
      : "r"(A0), "r"(A1), "r"(B0), "r"(B1))

template <int MT>
__global__ void skinny_nvfp4_qpn(const uint8_t *__restrict__ qcodes,
                                 const uint8_t *__restrict__ qscales,
                                 const half *__restrict__ x,
                                 half *__restrict__ y, int N, int K, int M,
                                 float gscale) {
  constexpr int WARPS = 4;
  __shared__ float cs[WARPS][MT * 256];

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int qp = (lane >> 2) & 3;
  const int r = (lane & 3) + ((lane & 16) ? 4 : 0);  // A row & B local col
  const int G = K >> 4, Gq = G / WARPS;
  const int g0 = warp * Gq;
  const uint2 *cb = reinterpret_cast<const uint2 *>(qcodes) +
                    (size_t)tile * G * 32 + lane;
  const uint8_t *sb = qscales + (size_t)tile * G * 32 + lane;

  const half2 gm2 = __float2half2_rn(gscale * 16384.f);
  float c[MT][8];
#pragma unroll
  for (int t = 0; t < MT; t++)
#pragma unroll
    for (int i = 0; i < 8; i++) c[t][i] = 0.f;

#pragma unroll 4
  for (int g = g0; g < g0 + Gq; g++) {
    const uint2 q2 = __ldcs(cb + (size_t)g * 32);
    const half2 sc2 =
        __hmul2(fp8e4m3_to_half2(__ldg(sb + (size_t)g * 32)), gm2);
    half2 b[8];
    dequant8_tm(q2.x, sc2, b + 0);  // slices 0,1 (k0..7 adjacent pairs)
    dequant8_tm(q2.y, sc2, b + 4);  // slices 2,3 (k8..15)
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
      MMA_8N8K4(c[t], A0[0], A0[1], B[0], B[1]);
      MMA_8N8K4(c[t], A0[2], A0[3], B[2], B[3]);
      MMA_8N8K4(c[t], A1[0], A1[1], B[4], B[5]);
      MMA_8N8K4(c[t], A1[2], A1[3], B[6], B[7]);
    }
  }

  // C map (mma8_probe.cu, roles swapped): reg i of lane L ->
  //   A-row (i&2)|((L&16)?4:0)|(L&1); B-col (i&1)|(((L>>1)&1)<<1)|((i>>2)<<2)
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
    if (row < M)
      y[(size_t)row * N + (size_t)tile * 32 + col] = __float2half(v);
  }
}

// QPN-layout SIMT kernel (M<=3): serves the decode band from the SAME
// prepacked fragment-order buffers as gemm_qpn, so the CT-native stash
// can be dropped entirely (VLLM_SKINNY_DROP_CT=1 -> weights return to
// the pre-QPN footprint and the fp32 GDN state cache fits).
// Geometry mirrors gemm_qpn: CTA = 4 warps on one 32-column tile with K
// split across warps; lane owns one column, so every warp code load is
// one 256B line. Whole activation block (M<=3 rows) staged to smem once;
// korder nibble pairing makes dequant8_tm output the adjacent-k x pairs.
// fp32 accumulation throughout; single barrier before the cross-warp
// reduce epilogue.
template <int M>
__global__ void skinny_nvfp4_qpn_simt(const uint8_t *__restrict__ qcodes,
                                      const uint8_t *__restrict__ qscales,
                                      const half *__restrict__ x,
                                      half *__restrict__ y, int N, int K,
                                      float gscale) {
  constexpr int WARPS = 4;
  extern __shared__ char smem_raw[];
  half2 *xs = reinterpret_cast<half2 *>(smem_raw);      // [M][K/2]
  float *cs = reinterpret_cast<float *>(xs + (size_t)M * (K >> 1));
  // cs: [WARPS][32][M] fp32 partials

  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int tile = blockIdx.x;
  const int G = K >> 4, Gq = G / WARPS;
  const int g0 = warp * Gq;
  constexpr int KCH2 = 4096;  // half2 per x chunk (M=3 -> 48KB smem)
  const uint2 *cb = reinterpret_cast<const uint2 *>(qcodes) +
                    (size_t)tile * G * 32 + lane;
  const uint8_t *sb = qscales + (size_t)tile * G * 32 + lane;
  const half2 gm2 = __float2half2_rn(gscale * 16384.f);

  float acc[M];
#pragma unroll
  for (int m = 0; m < M; m++) acc[m] = 0.f;

  // Chunked x staging: whole-x doesn't fit smem at M=3 x K=17408.
  // Accumulators persist across chunks; each warp consumes only its
  // K-quarter's groups that fall inside the staged window.
  for (int c0 = 0; c0 < (K >> 1); c0 += KCH2) {
    const int clen = min(KCH2, (K >> 1) - c0);
    __syncthreads();
    for (int t = threadIdx.x; t < M * clen; t += blockDim.x) {
      const int m = t / clen, j = t % clen;
      xs[m * KCH2 + j] =
          reinterpret_cast<const half2 *>(x)[(size_t)m * (K >> 1) + c0 + j];
    }
    __syncthreads();
    const int ga = max(g0, c0 >> 3);
    const int gb = min(g0 + Gq, (c0 + clen) >> 3);
#pragma unroll 4
    for (int g = ga; g < gb; g++) {
      const uint2 q2 = __ldcs(cb + (size_t)g * 32);
      const half2 sc2 =
          __hmul2(fp8e4m3_to_half2(__ldg(sb + (size_t)g * 32)), gm2);
      half2 w[8];
      dequant8_tm(q2.x, sc2, w + 0);  // adjacent-k pairs k0..7
      dequant8_tm(q2.y, sc2, w + 4);  // k8..15
#pragma unroll
      for (int m = 0; m < M; m++) {
        const half2 *xg = xs + (size_t)m * KCH2 + (g * 8 - c0);
        float s = 0.f;
#pragma unroll
        for (int j = 0; j < 8; j++) {
          const float2 wf = __half22float2(w[j]);
          const float2 xf = __half22float2(xg[j]);
          s += wf.x * xf.x + wf.y * xf.y;
        }
        acc[m] += s;
      }
    }
  }

#pragma unroll
  for (int m = 0; m < M; m++)
    cs[(warp * 32 + lane) * M + m] = acc[m];
  __syncthreads();
  if (warp == 0) {
#pragma unroll
    for (int m = 0; m < M; m++) {
      float v = cs[lane * M + m] + cs[(32 + lane) * M + m] +
                cs[(64 + lane) * M + m] + cs[(96 + lane) * M + m];
      y[(size_t)m * N + (size_t)tile * 32 + lane] = __float2half(v);
    }
  }
}

torch::Tensor skinny_gemm_qpn_simt(torch::Tensor x, torch::Tensor qcodes,
                                   torch::Tensor qscales, double gscale,
                                   int64_t n) {
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(m >= 1 && m <= 3, "qpn_simt supports M 1..3, got ", m);
  TORCH_CHECK(k % 64 == 0, "K % 64");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  TORCH_CHECK(qcodes.numel() == n * (k >> 1) &&
              qscales.numel() == n * (k >> 4));
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  const int smem = (int)(m * 4096 * sizeof(half2)) +
                   4 * 32 * (int)m * (int)sizeof(float);  // KCH2 chunks

#define LAUNCH_QPN_SIMT(MM)                                                 \
  do {                                                                     \
    auto *fn = skinny_nvfp4_qpn_simt<MM>;                                  \
    if (smem > 48 * 1024)                                                  \
      cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize,\
                           smem);                                          \
    fn<<<dim3((int)(n / 32)), dim3(128), smem, stream>>>(                  \
        qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),           \
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),            \
        reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,  \
        (float)gscale);                                                    \
  } while (0)

  switch (m) {
    case 1: LAUNCH_QPN_SIMT(1); break;
    case 2: LAUNCH_QPN_SIMT(2); break;
    case 3: LAUNCH_QPN_SIMT(3); break;
  }
#undef LAUNCH_QPN_SIMT
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

torch::Tensor skinny_gemm_qpn(torch::Tensor x, torch::Tensor qcodes,
                              torch::Tensor qscales, double gscale,
                              int64_t n) {
  const int64_t m = x.size(0), k = x.size(1);
  TORCH_CHECK(x.is_cuda() && x.dtype() == torch::kHalf && x.is_contiguous());
  TORCH_CHECK(qcodes.is_cuda() && qcodes.dtype() == torch::kUInt8 &&
              qcodes.is_contiguous());
  TORCH_CHECK(qscales.is_cuda() && qscales.dtype() == torch::kUInt8 &&
              qscales.is_contiguous());
  TORCH_CHECK(m >= 1 && m <= 16, "qpn supports M 1..16, got ", m);
  TORCH_CHECK(k % 64 == 0, "K % 64 (4-warp split of 16-k groups)");
  TORCH_CHECK(n % 32 == 0, "N % 32");
  TORCH_CHECK(qcodes.numel() == n * (k >> 1), "qpn codes size");
  TORCH_CHECK(qscales.numel() == n * (k >> 4), "qpn scales size");
  auto y = torch::empty({m, n}, x.options());
  auto stream = at::cuda::getCurrentCUDAStream();
  if (m <= 8)
    skinny_nvfp4_qpn<1><<<dim3((int)(n / 32)), dim3(128), 0, stream>>>(
        qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
        reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,
        (int)m, (float)gscale);
  else
    skinny_nvfp4_qpn<2><<<dim3((int)(n / 32)), dim3(128), 0, stream>>>(
        qcodes.data_ptr<uint8_t>(), qscales.data_ptr<uint8_t>(),
        reinterpret_cast<const half *>(x.data_ptr<at::Half>()),
        reinterpret_cast<half *>(y.data_ptr<at::Half>()), (int)n, (int)k,
        (int)m, (float)gscale);
  C10_CUDA_KERNEL_LAUNCH_CHECK();
  return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("gemm_qpn", &skinny_gemm_qpn,
        "skinny NVFP4 GEMM (QP-N mma.m8n8k4, prepacked weights, M<=16)");
  m.def("gemm_qpn_simt", &skinny_gemm_qpn_simt,
        "skinny NVFP4 GEMM (qpn-layout SIMT, M<=3, CT-stash-free)");
  m.def("gemm_dp4a", &skinny_gemm_dp4a,
        "skinny NVFP4 GEMM (int8 dp4a SIMT, W lossless / A8 per-16)");
  m.def("gemm_simt", &skinny_gemm_simt, "skinny NVFP4 GEMM (SIMT, M<=8)");
  m.def("gemm_simt_argmax", &skinny_gemm_simt_argmax,
        "fused NVFP4 lm_head GEMM + greedy argmax (M=1; block val/idx)");
  m.def("gemm_wmma", &skinny_gemm_wmma, "skinny NVFP4 GEMM (WMMA, M<=64)");
  m.def("gemm_mma8", &skinny_gemm_mma8,
        "skinny NVFP4 GEMM (mma.m8n8k4, M<=8)");
  m.def("gemm_wmma_cfg", &skinny_gemm_wmma_cfg,
        "skinny NVFP4 GEMM (WMMA, config-selectable tile sweep)");
  m.def("gemm_wmma_splitk", &skinny_gemm_wmma_splitk,
        "skinny NVFP4 GEMM (WMMA split-K, fp32 partials)");
}
