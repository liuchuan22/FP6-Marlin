import torch
import torch.nn as nn
import numpy as np

import fp6_marlin.cuda
import fp6_marlin.cpu

def prepacking(fp6_tensor: torch.Tensor) -> torch.Tensor:
    """
    Prepacking the fp6 weight in 2+4 format (CPU).
    Args:
        fp6_tensor (torch.Tensor): The weight tensor to be prepacked.
    Returns:
        torch.Tensor: The prepacked weight tensor.
    """
    return fp6_marlin.cpu.prepacking_cpu(fp6_tensor)

def quantize(fp16_tensor: torch.Tensor) -> torch.Tensor:
    """
    Quantize the fp16 weight into fp6 (CPU).
    Args:
        fp16_tensor (torch.Tensor): The fake quantized weight tensor. (of absolute range [0.0625, 28])
    Returns:
        torch.Tensor: The quantized fp6 weight (INT32).
    """
    return fp6_marlin.cpu.quantization_cpu(fp16_tensor)

def dequantize(fp6_tensor: torch.Tensor, fp16_scales: torch.Tensor) -> torch.Tensor:
    """
    Dequantize the fp6 weight into fp16 (CPU).
    Args:
        fp6_tensor (torch.Tensor): The quantized fp6 weight tensor. (INT32).
        fp16_scales (torch.Tensor): The scales used for dequantization. (half)
    Returns:
        torch.Tensor: The dequantized fp16 weight.
    """
    return fp6_marlin.cpu.dequantization_cpu(fp6_tensor, fp16_scales)

def mul(A, B, C, s, workspace, quant_cols = -1, thread_k = -1, thread_n = -1, sms = -1, max_par = 16):
    return fp6_marlin.cuda.mul(A, B, C, s, workspace, quant_cols, thread_k, thread_n, sms, max_par)
    