#include <torch/all.h>
#include <torch/python.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>

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
  int max_par = 16,
  int quant_cols = -1
);

const int ERR_PROB_SHAPE = 1;
const int ERR_KERN_SHAPE = 2;

void fp6_marlin_mul(
  const torch::Tensor& A,
  const torch::Tensor& B,
        torch::Tensor& C,
  const torch::Tensor& s,
        torch::Tensor& workspace,
  int quant_cols = -1,
  int thread_k = -1,
  int thread_n = -1,
  int sms = -1,
  int max_par = 8
) {
  int prob_m = A.size(0);
  int prob_n = C.size(1);
  int prob_k = A.size(1);
  if (workspace.numel() < prob_n / 128 * max_par)
    AT_ERROR("workspace must be of size at least ", prob_n / 128 * max_par, ".");
  int dev = A.get_device();
  int err = fp6_marlin_cuda(
    A.data_ptr(),
    B.data_ptr(),
    C.data_ptr(),
    s.data_ptr(),
    prob_m, prob_n, prob_k,
    workspace.data_ptr(),
    dev,
    at::cuda::getCurrentCUDAStream(dev),
    thread_k,
    thread_n,
    sms,
    max_par,
    quant_cols
  );
  if (err == ERR_PROB_SHAPE) {
    AT_ERROR(
      "Problem (m=", prob_m, ", n=", prob_n, ", k=", prob_k, ")",
      " not compatible with thread_k=", thread_k, ", thread_n=", thread_n, "."
    );
  } else if (err == ERR_KERN_SHAPE) {
    AT_ERROR(
      "No kernel implementation for thread_k=", thread_k, ", thread_n=", thread_n, "."
    );
  }
  if (quant_cols == -1 || quant_cols == prob_n) {
    return;
  }
  // reshape C array layout
  C = C.flatten(0, 1);
  int tot_m = prob_m;
  int tot_m_blocks = (tot_m + 15) / 16;
  int pad = 16 * tot_m_blocks - tot_m;
  int start = 0;
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
    // C_1 contains the first prob_m * (prob_n - quant_cols) elements of C
    torch::Tensor C_1 = C.narrow(0, start, prob_m * (prob_n - quant_cols));
    // C_2 contains the last prob_m * quant_cols elements of C
    torch::Tensor C_2 = C.narrow(0, start + prob_m * (prob_n - quant_cols), prob_m * quant_cols);
    C_1 = C_1.view({prob_m, prob_n - quant_cols});
    C_2 = C_2.view({prob_m, quant_cols});
    torch::Tensor C_cat = torch::cat({C_1, C_2}, 1);
    C.narrow(0, start, prob_m * prob_n) = C_cat.flatten(0, 1);
    start += prob_m * prob_n;
  }
  C = C.view({tot_m, prob_n});
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("mul", &fp6_marlin_mul, "FP16xINT6 matmul implementation of Marlin style.");
}
