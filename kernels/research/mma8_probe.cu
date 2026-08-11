// Empirical fragment-layout probe for sm_70 mma.sync.aligned.m8n8k4
// (.row.col.f32.f16.f16.f32). Derives which (row,col) each thread's
// a/b/c registers map to, so the real kernel doesn't guess.
//
// Method: one warp runs a single mma where A[r][k] = 16*r + k and
// B[k][c] = (k == probe_k && c == probe_c) ? 1 : 0. Then
// C[r][probe_c] = A[r][probe_k] = 16r + probe_k, other cols zero.
// Reading every thread's 8 c-registers over all (probe_k, probe_c)
// recovers the full C map and validates the assumed A/B maps.
#include <cstdio>
#include <cuda_fp16.h>

// Hypothesis under test (CUTLASS volta mma convention):
//   QP q = lane/4 mod 4?  No: QP0 = lanes {0..3, 16..19}.
//   A .row: lane l in QP: idx = (l & 3) + 4 * (l >= 16): holds row idx,
//           4 contiguous k (a0..a3 = k0..k3).
//   B .col: same lane->col idx, 4 contiguous k.
// Each QP computes an independent 8x8x4 product (we probe QP0; the
// kernel will replicate the pattern across QPs).
__global__ void probe(int probe_k, int probe_c, float *c_out) {
  const int lane = threadIdx.x & 31;
  const int in_qp0 = (lane < 4) || (lane >= 16 && lane < 20);
  const int idx = (lane & 3) + ((lane >= 16) ? 4 : 0);

  __half a[4], b[4];
  for (int i = 0; i < 4; i++) {
    a[i] = __float2half(in_qp0 ? (float)(16 * idx + i) : 0.f);
    b[i] = __float2half(
        (in_qp0 && i == probe_k && idx == probe_c) ? 1.f : 0.f);
  }
  float c[8];
  for (int i = 0; i < 8; i++) c[i] = 0.f;

  unsigned const *A = reinterpret_cast<unsigned const *>(a);
  unsigned const *B = reinterpret_cast<unsigned const *>(b);
  asm volatile(
      "mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 "
      "{%0,%1,%2,%3,%4,%5,%6,%7}, {%8,%9}, {%10,%11}, "
      "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
      : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3]), "+f"(c[4]),
        "+f"(c[5]), "+f"(c[6]), "+f"(c[7])
      : "r"(A[0]), "r"(A[1]), "r"(B[0]), "r"(B[1]));

  for (int i = 0; i < 8; i++) c_out[lane * 8 + i] = c[i];
}

int main() {
  float *d, h[32 * 8];
  cudaMalloc(&d, sizeof(h));
  // One probe per (k, c) would be exhaustive; k=0 plus varying c already
  // pins the C map (values 16r identify rows), k=1..3 confirms A order.
  for (int pk = 0; pk < 4; pk += 3) {
    for (int pc = 0; pc < 8; pc++) {
      probe<<<1, 32>>>(pk, pc, d);
      cudaMemcpy(h, d, sizeof(h), cudaMemcpyDeviceToHost);
      printf("probe_k=%d probe_c=%d:", pk, pc);
      for (int l = 0; l < 32; l++)
        for (int i = 0; i < 8; i++)
          if (h[l * 8 + i] != 0.f)
            printf(" L%d.c%d=%g", l, i, h[l * 8 + i]);
      printf("\n");
    }
  }
  cudaFree(d);
  return 0;
}
