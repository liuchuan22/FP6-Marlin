#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

// helper functions
constexpr int ceildiv(int a, int b) {
  return (a + b - 1) / b;
}

__device__ inline void cp_async4_pred(void* smem_ptr, const void* glob_ptr, bool pred = true) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .pred p;\n"
    "   setp.ne.b32 p, %0, 0;\n"
    "   @p cp.async.cg.shared.global [%1], [%2], %3;\n"
    "}\n" :: "r"((int) pred), "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}
__device__ inline void cp_async4_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 16;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .b64 p;\n"
    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
    "   cp.async.cg.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}

//!NEW: for 2 bit part, we load 8 bytes at a time.
__device__ inline void cp_async2_stream(void* smem_ptr, const void* glob_ptr) {
  const int BYTES = 8;
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "{\n"
    "   .reg .b64 p;\n"
    "   createpolicy.fractional.L2::evict_first.b64 p, 1.0;"
    "   cp.async.ca.shared.global.L2::cache_hint [%0], [%1], %2, p;\n"
    "}\n" :: "r"(smem), "l"(glob_ptr), "n"(BYTES)
  );
}


__device__ inline void cp_async_fence() {
    asm volatile("cp.async.commit_group;\n" ::);
}
template<int N>
__device__ inline void cp_async_wait() {
    asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
}
__device__ inline void cp_async_wait_all() {
    asm volatile("cp.async.wait_all;\n" ::);
}

__device__ inline void barrier_acquire(int* lock, int count) {
  if (threadIdx.x == 0) {
    int state = -1;
    do asm volatile ("ld.global.acquire.gpu.b32 %0, [%1];\n" : "=r"(state) : "l"(lock));
    while (state != count);
  }
  __syncthreads();
}
__device__ inline void barrier_release(int* lock, bool reset = false) {
  __syncthreads();
  if (threadIdx.x == 0) {
    if (reset) {
      lock[0] = 0;
      return;
    }
    int val = 1;
    asm volatile ("fence.acq_rel.gpu;\n");
    asm volatile ("red.relaxed.gpu.global.add.s32 [%0], %1;\n" : : "l"(lock), "r"(val)); 
  }
}

template <typename T, int n>
struct Vec {
  T elems[n];
  __device__ T& operator[](int i) {
    return elems[i];
  }
};

using I2 = Vec<uint32_t, 2>;
using I4 = Vec<uint32_t, 4>;
using FragA = Vec<half2, 4>;
using FragB = Vec<half2, 2>;
using FragC = Vec<float, 4>;
using FragS = Vec<half2, 1>;

__device__ inline void mma(const FragA& frag_a, const uint32_t* b, FragC& frag_c) {
  const uint32_t* a = reinterpret_cast<const uint32_t*>(&frag_a);
  // const uint32_t* b = reinterpret_cast<const uint32_t*>(&frag_b);
  float* c = reinterpret_cast<float*>(&frag_c);
  asm volatile(
    "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(c[0]), "=f"(c[1]), "=f"(c[2]), "=f"(c[3])
    :  "r"(a[0]),  "r"(a[1]),  "r"(a[2]),  "r"(a[3]),  "r"(b[0]),  "r"(b[1]),
       "f"(c[0]),  "f"(c[1]),  "f"(c[2]),  "f"(c[3])
  );
}

__device__ inline void ldsm4(FragA& frag_a, const void* smem_ptr) {
  uint32_t* a = reinterpret_cast<uint32_t*>(&frag_a);
  uint32_t smem = static_cast<uint32_t>(__cvta_generic_to_shared(smem_ptr));
  asm volatile(
    "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(smem)
  );
}

__device__ inline uint32_t mult_bias(uint32_t fp16_pair) {
    constexpr int BIAS = int(1) << 12; // 1 << 12 = 4096
    half* fp16_1 = reinterpret_cast<half*>(&fp16_pair);
    half* fp16_2 = fp16_1 + 1;
    uint32_t output;
    half* output_half_ptr = reinterpret_cast<half*>(&output);
    output_half_ptr[0] = __hmul(*fp16_1,__float2half(1.0f*BIAS));
    output_half_ptr[1] = __hmul(*fp16_2,__float2half(1.0f*BIAS));
    // printf("output: %f %f\n", __half2float(output_half_ptr[0]), __half2float(output_half_ptr[1]));
    return output;
}

__device__ inline void dequant_32fp6(
  uint32_t __restrict__ reg[][4],
  uint32_t __restrict__ *read_rptr_2bit,
  uint32_t __restrict__ *read_rptr_4bit
){
  // cast to 1 dim array
  uint32_t *output_regs = reinterpret_cast<uint32_t*>(reg);
  uint32_t *frag_ptr_2bit = read_rptr_2bit;
  uint32_t *frag_ptr_4bit = read_rptr_4bit;
  // 8 iterations, each dequantizing 4 fp16
  #pragma unroll(8)
  for (int i = 0; i < 8; i++) {
    uint32_t packed_fp6 = 0;
    uint32_t tmp = 0;
    tmp = (*frag_ptr_2bit) & 0xc0c0c0c0;
    packed_fp6 |= tmp >> 0;
    if (i % 4 == 3) {
      frag_ptr_2bit++;
    } else {
      (*frag_ptr_2bit) = (*frag_ptr_2bit) << 2;
    }
    tmp = (*frag_ptr_4bit) & 0xf0f0f0f0;
    packed_fp6 |= tmp >> 2; // 6 & 3 = 2
    if (i % 2 == 1) {
      frag_ptr_4bit++;
    } else {
      (*frag_ptr_4bit) = (*frag_ptr_4bit) << 4;
    }
    uint32_t out1, out2;
    constexpr int mask1 = 0x80000000;
    constexpr int mask2 = mask1 >> 5;
    constexpr int mask3 = mask2 & 0x7fffffff;
    constexpr int mask = mask3 | mask3 >> 16;
    out1 = packed_fp6 & 0x80008000;
    out1 |= (packed_fp6 & mask) >> 2;
    packed_fp6 = packed_fp6 << 8;
    out2 = packed_fp6 & 0x80008000;
    out2 |= (packed_fp6 & mask) >> 2;

    *output_regs = mult_bias(out1);
    output_regs++;
    *output_regs = mult_bias(out2);
    output_regs++;
  }
}

__device__ inline void dequant_8fp6(
  uint32_t __restrict__ reg[4],
  uint32_t q_2bit,
  uint32_t q_4bit
){
  uint32_t *output_regs = reinterpret_cast<uint32_t*>(reg);
  uint32_t packed_fp6 = 0;
  uint32_t tmp = 0;
  tmp = q_2bit & 0xc0c0c0c0;
  packed_fp6 |= tmp;
  tmp = q_4bit & 0xf0f0f0f0;
  packed_fp6 |= tmp >> 2;
  uint32_t out1, out2;
  constexpr int mask1 = 0x80000000;
  constexpr int mask2 = mask1 >> 5;
  constexpr int mask3 = mask2 & 0x7fffffff;
  constexpr int mask = mask3 | mask3 >> 16;
  out1 = packed_fp6 & 0x80008000;
  out1 |= (packed_fp6 & mask) >> 2;
  packed_fp6 = packed_fp6 << 8;
  out2 = packed_fp6 & 0x80008000;
  out2 |= (packed_fp6 & mask) >> 2;
  *output_regs = mult_bias(out1);
  output_regs++;
  *output_regs = mult_bias(out2);
  output_regs++;
  q_2bit = q_2bit << 2;
  q_4bit = q_4bit << 4;
  packed_fp6 = 0;
  tmp = q_2bit & 0xc0c0c0c0;
  packed_fp6 |= tmp;
  tmp = q_4bit & 0xf0f0f0f0;
  packed_fp6 |= tmp >> 2;
  out1 = packed_fp6 & 0x80008000;
  out1 |= (packed_fp6 & mask) >> 2;
  packed_fp6 = packed_fp6 << 8;
  out2 = packed_fp6 & 0x80008000;
  out2 |= (packed_fp6 & mask) >> 2;
  *output_regs = mult_bias(out1);
  output_regs++;
  *output_regs = mult_bias(out2);
  output_regs++;
}
// Lookup-table based 3-input logical operation; explicitly used for dequantization as the compiler does not seem to
// automatically recognize it in all cases. 
template <int lut>
__device__ inline int lop3(int a, int b, int c) {
  int res;
  asm volatile(
    "lop3.b32 %0, %1, %2, %3, %4;\n"
    : "=r"(res) : "r"(a), "r"(b), "r"(c), "n"(lut)
  );
  return res;
}
__device__ inline void dequant_8int4(
  uint32_t __restrict__ reg[4],
  uint32_t __restrict__ *read_rptr_4bit
) {
  half2 *reg_ptr = reinterpret_cast<half2*>(reg);
  int q = *read_rptr_4bit;
  const int LO = 0x000f000f;
  const int HI = 0x00f000f0;
  const int EX = 0x64006400;
  const int SUB = 0x64086408;
  const int MUL = 0x2c002c00;
  const int ADD = 0xd480d480;
  int lo = lop3<(0xf0 & 0xcc) | 0xaa>(q, LO, EX);
  int hi = lop3<(0xf0 & 0xcc) | 0xaa>(q, HI, EX);
  *reg_ptr = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  reg_ptr++;
  *reg_ptr = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL),
    *reinterpret_cast<const half2*>(&ADD)
  );
  reg_ptr++;
  lo = lop3<(0xf0 & 0xcc) | 0xaa>(q >> 8, LO, EX);
  hi = lop3<(0xf0 & 0xcc) | 0xaa>(q >> 8, HI, EX);
  *reg_ptr = __hsub2(
    *reinterpret_cast<half2*>(&lo),
    *reinterpret_cast<const half2*>(&SUB)
  );
  reg_ptr++;
  *reg_ptr = __hfma2(
    *reinterpret_cast<half2*>(&hi),
    *reinterpret_cast<const half2*>(&MUL),
    *reinterpret_cast<const half2*>(&ADD)
  );
  reg_ptr++;
}

template <
  const int threads,
  const int thread_m_blocks,
  const int thread_n_blocks,
  const int thread_k_blocks,
  const int stages,
  // the number of 6 bit quantization columns of B matrix
  const int quant_cols = -1
>
__global__ void fp6_kernel_marlin(
  const uint4* __restrict__ A,
  const uint4* __restrict__ B, // fp6 quantized weight matrix of shape kxn
  uint4* __restrict__ C,
  const uint4* __restrict__ s, // fp16 quantization scales of shape 1xn
  int  prob_m,
  int  prob_n,
  int  prob_k,
  int* locks // extra global storage for barrier synchronization
) {

  bool is_6bit = false;
  int bit4_sm = gridDim.x;
  if (quant_cols != -1) {
    int ratio = prob_n / quant_cols;
    int bit6_sm = gridDim.x * 3 / (ratio * 2);
    bit4_sm = gridDim.x - bit6_sm;
    if (blockIdx.x < bit4_sm) {
      prob_n = prob_n - quant_cols;
    } else {
      is_6bit = true;
      B += (prob_n - quant_cols) * prob_k * 4 / 128;
      s += (prob_n - quant_cols) * 16 / 128;
      prob_n = quant_cols;
    }
  }
  const uint2 B_2bit;
  const uint4 B_4bit;
  if (is_6bit) {
    B_2bit = reinterpret_cast<const uint2*>(B);
    B_4bit = B + prob_k * prob_n * 2 / 128;
  } else {
    B_2bit = nullptr;
    B_4bit = B;
  }

  // first get the pointer of packed 2 bit and 4 bit fp6 values
  // const uint2* B_2bit = reinterpret_cast<const uint2*>(B);
  // const uint4* B_4bit = B + prob_k * prob_n * 2 / 128;

  int parallel = 1;
  if (prob_m > 16 * thread_m_blocks) {
    parallel = prob_m / (16 * thread_m_blocks);
    prob_m = 16 * thread_m_blocks;
  }
  int k_tiles = prob_k / 16 / thread_k_blocks;
  int n_tiles = prob_n / 16 / thread_n_blocks;
  // int iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x);
  int iters;
  if (is_6bit) {
    iters = ceildiv(k_tiles * n_tiles * parallel, gridDim.x - bit4_sm);
    blockIdx.x -= bit4_sm;
  } else {
    iters = ceildiv(k_tiles * n_tiles * parallel, bit4_sm);
  }

  // a `slice` is a group of threadblocks that work on the same n-tile (of size K x 16 x thread_n_blocks)
  int slice_row = (iters * blockIdx.x) % k_tiles;
  int slice_col_par = (iters * blockIdx.x) / k_tiles;
  int slice_col = slice_col_par;
  int slice_iters; // number of tiles in current slice
  int slice_count = 0; // total number of active threadblocks in the current slice
  int slice_idx; // index of threadblock in current slice; numbered bottom to top

  if (slice_col_par >= n_tiles) {
    A += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_k / 8;
    //!TODO: C needs to be handled specially!
    C += (slice_col_par / n_tiles) * 16 * thread_m_blocks * prob_n / 8;
    locks += (slice_col_par / n_tiles) * n_tiles;
    slice_col = slice_col_par % n_tiles;
  }

  auto init_slice = [&] () {
    slice_iters = iters * (blockIdx.x + 1) - (k_tiles * slice_col_par + slice_row);
    if (slice_iters < 0 || slice_col_par >= n_tiles * parallel)
      slice_iters = 0;
    if (slice_iters == 0)
      return;
    if (slice_row + slice_iters > k_tiles) 
      slice_iters = k_tiles - slice_row;
    slice_count = 1;
    slice_idx = 0;
    int col_first = iters * ceildiv(k_tiles * slice_col_par, iters);
    if (col_first <= k_tiles * (slice_col_par + 1)) {
      int col_off = col_first - k_tiles * slice_col_par;
      slice_count = ceildiv(k_tiles - col_off, iters);
      if (col_off > 0)
        slice_count++;
      int delta_first = iters * blockIdx.x - col_first;
      if (delta_first < 0 || (col_off == 0 && delta_first == 0))
        slice_idx = slice_count - 1;
      else {
        slice_idx = slice_count - 1 - delta_first / iters;
        if (col_off > 0)
          slice_idx--;
      }
    }
    if (slice_col == n_tiles) {
      A += 16 * thread_m_blocks * prob_k / 8;
      //!TODO: C needs to be handled specially!
      C += 16 * thread_m_blocks * prob_n / 8;
      locks += n_tiles;
      slice_col = 0;
    }
  };
  init_slice();

  int a_gl_stride = prob_k / 8;
  // We typically use `constexpr` to indicate that this value is a compile-time constant
  constexpr int a_sh_stride = 16 * thread_k_blocks / 8;
  constexpr int a_gl_rd_delta_o = 16 * thread_k_blocks / 8;
  int a_gl_rd_delta_i = a_gl_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_wr_delta = a_sh_stride * (threads / a_gl_rd_delta_o);
  constexpr int a_sh_rd_delta_o = 2 * ((threads / 32) / (thread_n_blocks / 4));
  constexpr int a_sh_rd_delta_i = a_sh_stride * 16;
  constexpr int a_sh_stage = a_sh_stride * (16 * thread_m_blocks);
  constexpr int a_sh_wr_iters = ceildiv(a_sh_stage, a_sh_wr_delta);

  int b_gl_stride = 16 * prob_n / 32;
  constexpr int b_sh_stride = 32 * thread_n_blocks / 4;
  int b_gl_rd_delta_o = b_gl_stride * thread_k_blocks;
  int b_gl_rd_delta_i = b_gl_stride * (threads / b_sh_stride);
  constexpr int b_sh_wr_delta = threads;
  constexpr int b_sh_rd_delta = threads;
  constexpr int b_sh_stage = b_sh_stride * thread_k_blocks;
  constexpr int b_sh_wr_iters = b_sh_stage / b_sh_wr_delta;

  int s_gl_stride = prob_n / 8;
  constexpr int s_sh_stride = 16 * thread_n_blocks / 8;

  // Global/Shared A read/write index of current thread.
  int a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  a_gl_rd += a_gl_rd_delta_o * slice_row;
  int a_sh_wr = a_sh_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
  int a_sh_rd = a_sh_stride * ((threadIdx.x % 32) % 16) + (threadIdx.x % 32) / 16;
  a_sh_rd += 2 * ((threadIdx.x / 32) / (thread_n_blocks / 4));

  // Global/Shared B read/write index of current thread.
  int b_gl_rd = b_gl_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);
  b_gl_rd += b_sh_stride * slice_col;
  b_gl_rd += b_gl_rd_delta_o * slice_row;
  int b_sh_wr = threadIdx.x;
  int b_sh_rd = threadIdx.x;

  // only support per-channel quantization
  int s_gl_rd = s_gl_stride * ((thread_k_blocks * slice_row) / -1) + s_sh_stride * slice_col + threadIdx.x;
  int s_sh_wr = threadIdx.x;
  int s_sh_rd = 8 * ((threadIdx.x / 32) % (thread_n_blocks / 4)) + (threadIdx.x % 32) % 4;

  bool a_sh_wr_pred[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_pred[i] = a_sh_wr_delta * i + a_sh_wr < a_sh_stride * prob_m;
  bool s_sh_wr_pred = threadIdx.x < s_sh_stride;

  // When read/write A from/to shared memory, we need to transpose the data to avoid bank conflict.
  auto transform_a = [&] (int i) {
    int row = i / a_gl_rd_delta_o;
    return a_gl_rd_delta_o * row + (i % a_gl_rd_delta_o) ^ row;
  };
  int a_sh_wr_trans[a_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < a_sh_wr_iters; i++)
    a_sh_wr_trans[i] = transform_a(a_sh_wr_delta * i + a_sh_wr);
  int a_sh_rd_trans[b_sh_wr_iters][thread_m_blocks];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
    #pragma unroll
    for (int j = 0; j < thread_m_blocks; j++)
      a_sh_rd_trans[i][j] = transform_a(a_sh_rd_delta_o * i + a_sh_rd_delta_i * j + a_sh_rd);
  }

  const uint4* B_ptr_4bit[b_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
    B_ptr_4bit[i] = B_4bit + b_gl_rd_delta_i * i + b_gl_rd;
  }
  const uint2* B_ptr_2bit[b_sh_wr_iters];
  #pragma unroll
  for (int i = 0; i < b_sh_wr_iters; i++) {
    // B_ptr_2bit[i] = B_2bit + b_gl_rd_delta_i * i + b_gl_rd;
    B_ptr_2bit[i] = (!is_6bit) ? nullptr : B_2bit + b_gl_rd_delta_i * i + b_gl_rd;
  }

  // shared memory
  extern __shared__ uint4 shared_mem[];
  uint4* shared_A = shared_mem;
  // 4bit first, then 2bit. This is best suited for later reductions of marlin.
  uint4* shared_B_4bit = shared_A + a_sh_stage * stages;
  uint2* shared_B_2bit = reinterpret_cast<uint2*>(shared_B_4bit + b_sh_stage * stages);
  uint4* shared_S = (!is_6bit) ? reinterpret_cast<uint4*>(shared_B_2bit): reinterpret_cast<uint4*>(shared_B_2bit + b_sh_stage * stages);

  // registers
  FragA a_frag[2][thread_m_blocks];
  I2 b_frag_2bit[2]; // first 2 indicating double buffer.
  I4 b_frag_4bit[2];
  FragS s_frag[2][4];
  FragC c_frag[thread_m_blocks][4][2];

  auto zero_accums = [&] () {
    #pragma unroll
    for (int i = 0; i < thread_m_blocks * 4 * 2 * 4; i++)
      reinterpret_cast<float*>(c_frag)[i] = 0.0f;
  };

  auto fetch_to_shared = [&] (int pipe, int a_off, bool pred = true) {
    if (pred) {
      uint4* sh_a_stage = shared_A + a_sh_stage * pipe;
      #pragma unroll
      for (int i = 0; i < a_sh_wr_iters; i++) {
        cp_async4_pred(
          &sh_a_stage[a_sh_wr_trans[i]],
          &A[a_gl_rd_delta_i * i + a_gl_rd + a_gl_rd_delta_o * a_off],
          a_sh_wr_pred[i]
        );
      }
      //
      uint4* sh_b_stage_4bit = shared_B_4bit + b_sh_stage * pipe;
      #pragma unroll
      for (int i = 0; i < b_sh_wr_iters; i++) {
        cp_async4_stream(&sh_b_stage_4bit[b_sh_wr_delta * i + b_sh_wr], B_ptr_4bit[i]);
        B_ptr_4bit[i] += b_gl_rd_delta_o;
      }
      uint2* sh_b_stage_2bit;
      if (is_6bit) {
        sh_b_stage_2bit = shared_B_2bit + b_sh_stage * pipe;
        #pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++) {
          cp_async2_stream(&sh_b_stage_2bit[b_sh_wr_delta * i + b_sh_wr], B_ptr_2bit[i]);
          B_ptr_2bit[i] += b_gl_rd_delta_o;
        }
      } else {
        sh_b_stage_2bit = nullptr;
      }
    }
    cp_async_fence();
  };
  auto wait_for_stage = [&] () {
    // Wait until at most (stages-2) async copy stages are still pending.
    cp_async_wait<stages - 2>();
    __syncthreads();
  };

  auto fetch_to_registers = [&] (int k, int pipe) {
    uint4* sh_a_stage = shared_A + a_sh_stage * pipe;
    #pragma unroll
    for (int i = 0; i < thread_m_blocks; i++) {
      ldsm4(a_frag[k%2][i], &sh_a_stage[a_sh_rd_trans[k % b_sh_wr_iters][i]]);
    }
    uint4* sh_b_stage_4bit = shared_B_4bit + b_sh_stage * pipe;
    b_frag_4bit[k%2] = *reinterpret_cast<I4*>(&sh_b_stage_4bit[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
    uint2* sh_b_stage_2bit;
    if (is_6bit) {
      sh_b_stage_2bit = shared_B_2bit + b_sh_stage * pipe;
      b_frag_2bit[k%2] = *reinterpret_cast<I2*>(&sh_b_stage_2bit[b_sh_rd_delta * (k % b_sh_wr_iters) + b_sh_rd]);
    } else {
      sh_b_stage_2bit = nullptr;
    }
  };

  auto matmul = [&] (int k) {
    #pragma unroll
    for (int j = 0; j < 4; j++) {
      uint32_t *b_frag_2bit_ptr;
      uint32_t *b_frag_4bit_ptr = reinterpret_cast<uint32_t*>(b_frag_4bit[k % 2].elems);
      uint32_t b_dequant[4];
      if (is_6bit) {
        b_frag_2bit_ptr = reinterpret_cast<uint32_t*>(b_frag_2bit[k % 2].elems);
        dequant_8fp6(b_dequant, *b_frag_2bit_ptr, *b_frag_4bit_ptr);
        if (j % 2 == 1) {
          b_frag_2bit_ptr++;
        } else {
          *b_frag_2bit_ptr = *b_frag_2bit_ptr << 4;
        }
      } else {
        b_frag_2bit_ptr = nullptr;
        dequant_8int4(b_dequant, b_frag_4bit_ptr);
      }
      b_frag_4bit_ptr++;
      for (int i = 0; i < thread_m_blocks; i++) {
        uint32_t* b_dequant_ptr = reinterpret_cast<uint32_t*>(&b_dequant);
        mma(frag_a[k % 2][i], b_dequant_ptr, c_frag[i][j][0]);
        b_dequant_ptr += 2;
        mma(frag_a[k % 2][i], b_dequant_ptr, c_frag[i][j][1]);
      }
    }
    // first dequant the 32 fp6 values in registers
    // uint32_t b_dequant[4][4];
    // uint32_t *b_frag_2bit_ptr;
    // if (is_6bit) {
    //   b_frag_2bit_ptr = reinterpret_cast<uint32_t*>(b_frag_2bit[k % 2].elems);
    // } else {
    //   b_frag_2bit_ptr = nullptr;
    // }     
    // uint32_t *b_frag_4bit_ptr = reinterpret_cast<uint32_t*>(b_frag_4bit[k % 2].elems);
    // // scale later if we do per-channel quantization
    // //!TODO: Add the dequant function of 4bit!
    // dequant_32fp6(b_dequant, b_frag_2bit_ptr, b_frag_4bit_ptr);

    // // then do the mma
    // #pragma unroll
    // for (int j = 0; j < 4; j++) { // 4 16x16 subtiles in the row of B dimension.
    //   FragB b_frag0, b_frag1;
    //   b_frag0[0] = reinterpret_cast<half2&>(b_dequant[j][0]);
    //   b_frag0[1] = reinterpret_cast<half2&>(b_dequant[j][1]);
    //   b_frag1[0] = reinterpret_cast<half2&>(b_dequant[j][2]);
    //   b_frag1[1] = reinterpret_cast<half2&>(b_dequant[j][3]);

    //   #pragma unroll
    //   for (int i = 0; i < thread_m_blocks; i++) {
    //     mma(a_frag[k % 2][i], b_frag0, c_frag[i][j][0]);
    //     mma(a_frag[k % 2][i], b_frag1, c_frag[i][j][1]);
    //   }
    // }
  };

  // reduce over the multiple warps in smem of one threadblock
  auto thread_block_reduce = [&] () {
    constexpr int red_off = threads / b_sh_stride / 2;
    if (red_off >= 1) {
      int red_idx = threadIdx.x / b_sh_stride;
      constexpr int red_sh_stride = b_sh_stride * 4 * 2;
      constexpr int red_sh_delta = b_sh_stride; 
      int red_sh_rd = red_sh_stride * (threadIdx.x / b_sh_stride) + (threadIdx.x % b_sh_stride);

      #pragma unroll
      for (int m_block = 0; m_block < thread_m_blocks; m_block++) {
        #pragma unroll
        for (int i = red_off; i > 0; i /= 2) {
          if (i <= red_idx && red_idx < 2 * i) {
            #pragma unroll
            for (int j = 0; j < 4 * 2; j++) {
              int red_sh_wr = red_sh_delta * j + (red_sh_rd - red_sh_stride * i);
              if (i < red_off) {
                float* c_rd = reinterpret_cast<float*>(&shared_mem[red_sh_delta * j + red_sh_rd]);
                float* c_wr = reinterpret_cast<float*>(&shared_mem[red_sh_wr]);
                #pragma unroll
                for (int k = 0; k < 4; k++)
                  reinterpret_cast<FragC*>(c_frag)[4 * 2 * m_block + j][k] += c_rd[k] + c_wr[k];
              }
              shared_mem[red_sh_wr] = reinterpret_cast<uint4*>(&c_frag)[4 * 2 * m_block + j];
            }
          }
          __syncthreads();
        }
        if (red_idx == 0) {
          #pragma unroll
          for (int i = 0; i < 4 * 2; i++) {
            float* c_rd = reinterpret_cast<float*>(&shared_mem[red_sh_delta * i + red_sh_rd]);
            #pragma unroll
            for (int j = 0; j < 4; j++)
              reinterpret_cast<FragC*>(c_frag)[4 * 2 * m_block + i][j] += c_rd[j];
          }
        }
        __syncthreads();
      }
    }
  };

  //!TODO: Moidfy Global reduce and Write result for C matrix part.
  auto global_reduce = [&] (bool first = false, bool last = false) {
    constexpr int active_threads = 32 * thread_n_blocks / 4;
    if (threadIdx.x < active_threads) {
      int c_gl_stride = prob_n / 8;
      int c_gl_wr_delta_o = 8 * c_gl_stride;
      int c_gl_wr_delta_i = 4 * (active_threads / 32);
      int c_gl_wr = c_gl_stride * ((threadIdx.x % 32) / 4) + 4 * (threadIdx.x / 32) + threadIdx.x % 4;
      c_gl_wr += (2 * thread_n_blocks) * slice_col;
      constexpr int c_sh_wr_delta = active_threads;
      int c_sh_wr = threadIdx.x;

      int row = (threadIdx.x % 32) / 4;

      if (!first) {
        #pragma unroll
        for (int i = 0; i < thread_m_blocks * 4; i++) {
          cp_async4_pred(
            &shared_mem[c_sh_wr + c_sh_wr_delta * i],
            &C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)],
            i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m
          );
        }
        cp_async_fence();
        cp_async_wait<0>();
      }

      #pragma unroll
      for (int i = 0; i < thread_m_blocks * 4; i++) {
        if (i < (thread_m_blocks - 1) * 4 || 8 * (i / 2) + row < prob_m) {
          if (!first) {
            uint4 c_red = shared_mem[c_sh_wr + i * c_sh_wr_delta];
            #pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<float*>(&c_frag)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)] += __half2float(
                reinterpret_cast<__half*>(&c_red)[j]
              );
            }
          }
          if (!last) {
            uint4 c;
            #pragma unroll
            for (int j = 0; j < 2 * 4; j++) {
              reinterpret_cast<__half*>(&c)[j] = __float2half(
                reinterpret_cast<float*>(&c_frag)[4 * 2 * 4 * (i / 4) + 4 * j + (i % 4)]
              );
            }
            C[c_gl_wr + c_gl_wr_delta_o * (i / 2) + c_gl_wr_delta_i * (i % 2)] = c;
          }
        }
      }
    }
  };

  auto write_result = [&] () {
    int c_gl_stride = prob_n / 8;
    constexpr int c_sh_stride = 2 * thread_n_blocks + 1;
    int c_gl_wr_delta = c_gl_stride * (threads / (2 * thread_n_blocks));
    constexpr int c_sh_rd_delta = c_sh_stride * (threads / (2 * thread_n_blocks));

    int c_gl_wr = c_gl_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));
    c_gl_wr += (2 * thread_n_blocks) * slice_col;
    int c_sh_wr = (4 * c_sh_stride) * ((threadIdx.x % 32) / 4) + (threadIdx.x % 32) % 4;
    c_sh_wr += 32 * (threadIdx.x / 32);
    int c_sh_rd = c_sh_stride * (threadIdx.x / (2 * thread_n_blocks)) + (threadIdx.x % (2 * thread_n_blocks));

    int c_gl_wr_end = c_gl_stride * prob_m;

    // We first reorder in shared memory to guarantee the most efficient final global write patterns
    auto write = [&] (int idx, float c0, float c1, FragS& s) {
      half2 res = __halves2half2(__float2half(c0), __float2half(c1));
      // for per-channel quantization we finally apply the scale here.
      res = __hmul2(res, s[0]);
      ((half2*) shared_mem)[idx] = res;
    };
    if (threadIdx.x / 32 < thread_n_blocks / 4) {
      #pragma unroll
      for (int i = 0; i < thread_m_blocks; i++) {
        #pragma unroll
        for (int j = 0; j < 4; j++) {
          int wr = c_sh_wr + 8 * j;
          write(wr + (4 * c_sh_stride) * 0 + 0, c_frag[i][j][0][0], c_frag[i][j][0][1], s_frag[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 8 + 0, c_frag[i][j][0][2], c_frag[i][j][0][3], s_frag[j / 2][2 * (j % 2) + 0]);
          write(wr + (4 * c_sh_stride) * 0 + 4, c_frag[i][j][1][0], c_frag[i][j][1][1], s_frag[j / 2][2 * (j % 2) + 1]);
          write(wr + (4 * c_sh_stride) * 8 + 4, c_frag[i][j][1][2], c_frag[i][j][1][3], s_frag[j / 2][2 * (j % 2) + 1]);
        }
        c_sh_wr += 16 * (4 * c_sh_stride);
      }
    }
    __syncthreads();

    #pragma unroll
    for (int i = 0; i < ceildiv(16 * thread_m_blocks, threads / (2 * thread_n_blocks)); i++) {
      if (c_gl_wr < c_gl_wr_end) {
        C[c_gl_wr] = shared_mem[c_sh_rd];
        c_gl_wr += c_gl_wr_delta;
        c_sh_rd += c_sh_rd_delta;
      }
    }
  };

  auto start_pipes = [&] () {
    #pragma unroll
    for (int i = 0; i < stages - 1; i++)
      fetch_to_shared(i, i, i < slice_iters);
    zero_accums();
    wait_for_stage();
    fetch_to_registers(0, 0);
    a_gl_rd += a_gl_rd_delta_o * (stages - 1);
  };

  start_pipes();

  while (slice_iters) {
    #pragma unroll
    for (int pipe = 0; pipe < stages;) {
      #pragma unroll
      for (int k = 0; k < b_sh_wr_iters; k++) {
        fetch_to_registers(k + 1, pipe % stages);
        if (k == b_sh_wr_iters - 2) {
          fetch_to_shared((pipe + stages - 1) % stages, pipe, slice_iters >= stages);
          pipe++;
          wait_for_stage();
        }
        matmul(k);
      }
      slice_iters--;
      if (slice_iters == 0)
        break;
    }
    a_gl_rd += a_gl_rd_delta_o * stages;

    if (slice_iters == 0) {
      cp_async_wait<0>();
      bool last = slice_idx == slice_count - 1;
      // For per-channel scales, we only fetch them here in the final step before write-out
      if (last) {
        if (s_sh_wr_pred)
          cp_async4_stream(&shared_S[s_sh_wr], &s[s_gl_rd]);
        cp_async_fence();
      }
      thread_block_reduce();
      if (last) {
        cp_async_wait<0>();
        __syncthreads();
        if (threadIdx.x / 32 < thread_n_blocks / 4) {
          reinterpret_cast<uint4*>(&s_frag)[0] = shared_S[s_sh_rd + 0];
          reinterpret_cast<uint4*>(&s_frag)[1] = shared_S[s_sh_rd + 4];
        }
      }
      if (slice_count > 1) {
        barrier_acquire(&locks[slice_col], slice_idx);
        global_reduce(slice_idx == 0, last);
        barrier_release(&locks[slice_col], last);
      }
      if (last) write_result();
      slice_row = 0;
      slice_col_par++;
      slice_col++;
      init_slice();
      if (slice_iters) {
        a_gl_rd = a_gl_stride * (threadIdx.x / a_gl_rd_delta_o) + (threadIdx.x % a_gl_rd_delta_o);
        if (is_6bit) {
          #pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++)
            B_ptr_2bit[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        }
        #pragma unroll
        for (int i = 0; i < b_sh_wr_iters; i++)
          B_ptr_4bit[i] += b_sh_stride - b_gl_rd_delta_o * k_tiles;
        if (slice_col == 0) {
          if (is_6bit) {
            #pragma unroll
            for (int i = 0; i < b_sh_wr_iters; i++)
              B_ptr_2bit[i] -= b_gl_stride;
          }
          #pragma unroll
          for (int i = 0; i < b_sh_wr_iters; i++)
            B_ptr_4bit[i] -= b_gl_stride;
        }
        s_gl_rd = s_sh_stride * slice_col + threadIdx.x;
        start_pipes();
      }
    }
  }
}

const int THREADS = 256;
const int STAGES = 4;
const int SHARED_MEM = 96 * 1024; // max shared memory on compute capability 8.6 (< 8.0). What about 9.0?

#define CALL_IF(THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS) \
  else if ( \
    thread_m_blocks == THREAD_M_BLOCKS && thread_n_blocks == THREAD_N_BLOCKS && thread_k_blocks == THREAD_K_BLOCKS \
  ) { \
    cudaFuncSetAttribute( \
      fp6_kernel_marlin<THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES>, \
      cudaFuncAttributeMaxDynamicSharedMemorySize, \
      SHARED_MEM \
    ); \
    fp6_kernel_marlin< \
      THREADS, THREAD_M_BLOCKS, THREAD_N_BLOCKS, THREAD_K_BLOCKS, STAGES \
    ><<<blocks, THREADS, SHARED_MEM, stream>>>( \
      A_ptr, B_ptr, C_ptr, s_ptr, \
      prob_m, prob_n, prob_k, \
      locks \
    ); \
  }

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

int fp6_marlin_cuda(
  const void* A,
  const void* B,
        void* C,
        void* s,
  int prob_m,
  int prob_n,
  int prob_k,
  void* workspace,
  int dev = 0,
  cudaStream_t stream = 0,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 16
) {
  int tot_m = prob_m;
  int tot_m_blocks = ceildiv(tot_m, 16);
  int pad = 16 * tot_m_blocks - tot_m;

  if (sms == -1)
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev); // 114 on 4090 gpu
  // printf("sms: %d\n", sms);
  // 1 block for easy debugging
  // sms = 1;
  if (thread_k == -1 || thread_n == -1) {
    if (prob_m <= 16) {
      thread_k = 128;
      thread_n = 128;
    } else {
      thread_k = 64;
      thread_n = 256;
    }
  }

  int thread_k_blocks = thread_k / 16;
  int thread_n_blocks = thread_n / 16;
  int blocks = sms;

  if (prob_n % thread_n != 0 || prob_k % thread_k != 0)
    return ERR_PROB_SHAPE;
  if (prob_m == 0 || prob_n == 0 || prob_k == 0)
    return 0;

  const uint4* A_ptr = (const uint4*) A;
  const uint4* B_ptr = (const uint4*) B;
  uint4* C_ptr = (uint4*) C;
  const uint4* s_ptr = (const uint4*) s;

  int* locks = (int*) workspace;

  int ret = 0;
  for (int i = 0; i < tot_m_blocks; i += 4) {
    int thread_m_blocks = tot_m_blocks - i;
    prob_m = tot_m - 16 * i;
    int par = 1;
    if (thread_m_blocks > 4) {
      // Note that parallel > 1 currently only works for inputs without any padding
      par = (16 * thread_m_blocks - pad) / 64;
      if (par > max_par)
        par = max_par;
      prob_m = 64 * par;
      i += 4 * (par - 1);
      thread_m_blocks = 4;
    }
    
    // specific configurations
    if (false) {}
    CALL_IF(1,  8,  8)
    CALL_IF(1, 16,  4)
    CALL_IF(2, 16,  4)
    CALL_IF(3, 16,  4)
    CALL_IF(4, 16,  4)
    else
      ret = ERR_KERN_SHAPE;

    A_ptr += 16 * thread_m_blocks * (prob_k / 8) * par;
    C_ptr += 16 * thread_m_blocks * (prob_n / 8) * par;
  }

  return ret;
}
