#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <cuda_fp16.h>

void dequantMatrix_fp6_to_fp16(half* A_16bit_h, unsigned char* A_x_bit_h, size_t M, size_t K, half* scale);
void weight_matrix_prepacking(int* packed_weights, int* FP6Weights, size_t K, size_t N);
void weight_prepacking_fp16_to_fp6(uint16_t* weight_16bit, uint8_t* weight_6bit_packed, size_t K, size_t N);

// Pytorch interface

/*
 * Weight prepacking (Pytorch interface).
 * [Input & Output]
 *  fp6_tensor: int tensor of shape [OC, IC // 16 * 3];   // 3 INT32 words contains 16 FP6 weights.
 * [Output]
 *  packed_tensor: int tensor of shape [IC *3//32, OC *2].
 */
torch::Tensor weight_prepacking_cpu(torch::Tensor fp6_tensor)
{
    size_t OC = fp6_tensor.size(0);
    size_t IC = fp6_tensor.size(1);
    assert (IC%3==0);   
    IC = IC*16/3;
    assert( (OC%256==0) && (IC%256==0) );
    auto packed_tensor = torch::empty({static_cast<int64_t>(IC) / 32 * 3, static_cast<int64_t>(OC) * 2}, torch::kInt32);
    packed_tensor = packed_tensor.contiguous();
    auto packed_tensor_ptr = reinterpret_cast<int*>(packed_tensor.data_ptr<int>());
    auto fp6_tensor_ptr = reinterpret_cast<int*>(fp6_tensor.data_ptr<int>());
    weight_matrix_prepacking(packed_tensor_ptr, fp6_tensor_ptr, OC, IC);
    return packed_tensor;
}

/*
 * Weight quantization (Pytorch interface).
 * [Input & Output]
 * fp16_tensor: float16 tensor of shape [OC, IC];
 * [Output]
 * fp6_tensor: int tensor of shape [IC, OC // 16 * 3]; // 3 INT32 words contains 16 FP6 weights.
 * [Note] fp16 tensor should be fake quantized.
 * [Note] fp16 tensor is transposed to match the prepacking format.
 */
torch::Tensor weight_quantization_cpu(torch::Tensor fp16_tensor)
{
    size_t OC = fp16_tensor.size(0);
    size_t IC = fp16_tensor.size(1);
    // Type check
    if (fp16_tensor.dtype() != torch::kFloat16) {
        throw std::invalid_argument("Input tensor should be of type float16.");
    }
    assert((OC % 256 == 0) && (IC % 256 == 0));
    // Transpose the tensor to match the prepacking format
    auto fp16_tensor_transposed = fp16_tensor.transpose(0, 1);
    // Unsigned int casting to int here.
    auto fp6_tensor = torch::empty({static_cast<int64_t>(IC), static_cast<int64_t>(OC) / 16 * 3}, torch::kInt32);
    auto fp6_tensor_ptr = reinterpret_cast<uint8_t*>(fp6_tensor.data_ptr<int>());
    auto fp16_tensor_ptr = reinterpret_cast<uint16_t*>(fp16_tensor_transposed.data_ptr<at::Half>());
    weight_prepacking_fp16_to_fp6(fp16_tensor_ptr, fp6_tensor_ptr, IC, OC);
    return fp6_tensor;
}

/*
 * Dequant a FP6 matrix to a equivalent FP16 matrix using CPUs.
 * A useful tool to construct input matrices for the FP16 GEMM baseline.
 * [Input]
 *  fp6_tensor:  int  tensor of shape [OC, IC // 16 * 3];   // 3 INT32 words contains 16 FP6  weights.
 *  fp16_scale:  half tensor of shape [OC]; // for row-wise quantization.
 * [Output]
 *  fp16_tensor: half tensor of shape [OC, IC].     
 */
torch::Tensor weight_dequant_cpu(torch::Tensor fp6_tensor, torch::Tensor fp16_scale) 
{
    int OC = fp6_tensor.size(0);
    assert(fp6_tensor.size(1) % 3 == 0);
    int IC = fp6_tensor.size(1) / 3 * 16;
    assert(fp16_scale.size(0)==OC);
    //
    auto fp6_tensor_ptr = reinterpret_cast<int*>(fp6_tensor.data_ptr<int>());
    auto fp16_scale_ptr = reinterpret_cast<half*>(fp16_scale.data_ptr<at::Half>());
    //
    auto options = torch::TensorOptions().dtype(fp16_scale.dtype()).device(fp16_scale.device());
    at::Tensor fp16_tensor = torch::empty({OC, IC}, options);
    auto fp16_tensor_ptr = reinterpret_cast<half*>(fp16_tensor.data_ptr<at::Half>());
    //
    dequantMatrix_fp6_to_fp16(fp16_tensor_ptr, (unsigned char*)fp6_tensor_ptr, OC, IC, fp16_scale_ptr);
    //
    return fp16_tensor;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("quantization_cpu", &weight_quantization_cpu, "Weight quantization (CPU)");
    m.def("prepacking_cpu", &weight_prepacking_cpu, "Weight prepacking (CPU)");
    m.def("dequantization_cpu", &weight_dequant_cpu, "Weight dequantization (CPU)");
}
