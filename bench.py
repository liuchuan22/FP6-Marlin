import sys

import numpy as np
import torch
import fp6_marlin

import time

def benchmark(f, warmup=10, iter=100):
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    for i in range(warmup):
        f()
    start_event.record()
    for i in range(iter):
        f()
    end_event.record()
    torch.cuda.synchronize()
    res = start_event.elapsed_time(end_event) / iter
    # Make sure there is enough to "cool down" the GPU in between benchmarks to avoid throttling for later runs when
    # we execute many benchmarks consecutively
    time.sleep(1.)
    return res

def get_problem(m, n, k, quant_cols=-1):
    dev = torch.device('cuda:0')
    A = torch.randn((m, k), dtype=torch.half, device=dev)
    B = torch.randint(4294967295, (k*quant_cols*3//16 + k*(n-quant_cols)//8, ), device=dev, dtype=torch.uint32).contiguous()
    B_ref = torch.randn((k, n), dtype=torch.half, device=dev)
    C = torch.zeros((m, n), dtype=torch.half, device=dev)
    s = torch.randn((n,), dtype=torch.half, device=dev)
    torch.cuda.synchronize()
    return A, B, C, B_ref, s

def benchmark_dense(A, B, C):
    res = benchmark(lambda: torch.matmul(A, B, out=C))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 2 * B.numel() + 2 * C.numel()) / res / 10 ** 9
    }

def benchmark_quant(A, B, C, s, thread_k, thread_n, quant_cols):
    workspace = torch.zeros(C.shape[1] // 128 * 16, device=torch.device('cuda:0'))
    res = benchmark(lambda: fp6_marlin.mul(A, B, C, s, workspace, quant_cols, thread_k, thread_n, -1))
    return {
        's': res,
        'TFLOP/s': 2 * A.numel() * C.shape[1] / res / 10 ** 12,
        'GB/s': (2 * A.numel() + 4 * B.numel() + 2 * C.numel() + 2 * s.numel()) / res / 10 ** 9
    }

# Pass the SM count for known GPUs to avoid the kernel having to query this information (this is very minor)
gpu = torch.cuda.get_device_name(0)
if 'A100' in gpu:
    SMS = 108
elif 'A10' in gpu:
    SMS = 72
elif '3090' in gpu:
    SMS = 82
elif '4090' in gpu:
    SMS = 114
elif 'A6000' in gpu:
    SMS = 84
else:
    SMS = -1
print('GPU: %s, SMS: %d' % (gpu, SMS))
quant_cols = 256

MODELS = {
    # 'ideal': [
    #     (3* 256 * SMS, 256 * SMS)
    # ],
    'Llama7B': [
        (4096, 3 * 4096),
        (4096, 4096),
        (4096, 2 * 10752),
        (10752, 4096)
    ],
    'Llama13B': [
        (5120, 3 * 5120),
        (5120, 5120),
        (5120, 2 * 13568),
        (13568, 5120)
    ],
    'Llama33B': [
        (6656, 3 * 6656),
        (6656, 6656),
        (6656, 2 * 17664),
        (17664, 6656)
    ],
    'Llama65B': [
        (8192, 3 * 8192),
        (8192, 8192),
        (8192, 2 * 21760),
        (21760, 8192)
    ],
    'Falcon180B': [
        # Note that parallel attention and FC allows layer fusions
        (14848, 14848 * 5 + 1024),
        (14848 * 5, 14848)
    ]
}

# Set to true in order to run a more complete benchmark sweep; the default is reproduce README experiments
ALL = True

for model, layers in MODELS.items():
        print(model)
        if ALL:
            batchsizes =  [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
        else:
            batchsizes = [1, 2, 4, 8, 16, 32, 64, 128]
        for batch in batchsizes:
            if not ALL and model != 'ideal' and batch != 16:
                continue
            tot_q = {'s': 0, 'TFLOP/s': 0, 'GB/s': 0, 'speedup': 0} 
            for layer in layers:
                A, B, C, B_ref, s = get_problem(batch, layer[1], layer[0], quant_cols)
                res_d = benchmark_dense(A, B_ref, C)
                if model == 'ideal' and batch == 16:
                    # This is a special case constructed to be optimal for a thread-shape different than the default one
                    res_q = benchmark_quant(A, B, C, s, 64, 256, quant_cols)
                else:
                    res_q = benchmark_quant(A, B, C, s, -1, -1, quant_cols)
                res_q['speedup'] = res_d['s'] / res_q['s']
                tot_q['s'] += res_q['s']
                for k in tot_q:
                    if k != 's':
                        tot_q[k] += res_q[k] * res_q['s']
            for k in tot_q:
                if k != 's':
                    tot_q[k] /= tot_q['s']
            print('batch=%04d: s=%.5f, TFLOP/s=%07.3f, GB/s=%08.3f, speedup=%.2f' % (
                batch,
                tot_q['s'],
                tot_q['TFLOP/s'],
                tot_q['GB/s'],
                tot_q['speedup']
            ))
        print()
