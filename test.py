import numpy as np
import torch
import unittest

import fp6_marlin

seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)
DEV = torch.device('cuda:0')
torch.set_printoptions(sci_mode=False, linewidth=90) 

def gen_fp6_weight(k, n):
    # 3 exponent bit, 2 mantissa bit
    maxq = 28.0
    minq = 0.0625
    w = torch.randn((k, n), dtype=torch.half, device='cpu')
    # extract scales and restrict weight to the range of fp6
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 1.0 / maxq
    s = s.reshape((-1, n)).contiguous()
    s = s.squeeze(0) # shape (1, n) -> (n,)
    assert s.shape == (n,)
    w = w / s   # no rounding here since fp6 quantization
    # now elmts have range (-28., 28.), we set all values between (-0.0625, 0.0625) to 0
    w = torch.where(torch.abs(w) < minq, torch.zeros_like(w), w)
    w = torch.clamp(w, -maxq, maxq)
    # debugging, set s as a tensor of ones
    # s = torch.ones(n, dtype=torch.half, device='cpu')

    # quantize 
    w_quant = fp6_marlin.quantize(w)    # shape (n, k * 3 / 16), type torch.kbyte
    assert w_quant.shape[0] == n
    assert w_quant.shape[1] == k * 3 // 16
    assert w_quant.dtype == torch.int32

    w_ref = fp6_marlin.dequantize(w_quant, s)
    w_ref = w_ref.transpose(0, 1) # shape (k, n)
    assert w_ref.shape[0] == k
    assert w_ref.shape[1] == n

    # pack weight
    w_packed = fp6_marlin.prepacking(w_quant) # shape (k * 3 / 32, N * 2), 2+4 format
    assert w_packed.shape[0] == k * 3 // 32
    assert w_packed.shape[1] == n * 2
    assert w_packed.dtype == torch.int32

    # permute scales
    scale_perm = []
    for i in range(4):
        scale_perm.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    s = s.reshape((-1, n)).squeeze(0)
    
    w_ref = w_ref.contiguous().to(DEV)
    w_packed = w_packed.contiguous().to(DEV)
    s = s.contiguous().to(DEV)
    return w_ref, w_packed, s

class Test(unittest.TestCase):
    def run_problem(self, m, n, k, thread_k, thread_n):
        print('% 5d % 6d % 6d % 4d % 4d' % (m, n, k, thread_k, thread_n))
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        B_ref, B, s = gen_fp6_weight(k, n)
        # print(B_ref)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        C_ref = torch.matmul(A, B_ref)
        workspace = torch.zeros(n // 128 * 16, device=DEV)
        fp6_marlin.mul(A, B, C, s, workspace, thread_k, thread_n)
        torch.cuda.synchronize()
        # self.assertLess(torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)), 0.001)
        self.assertLess(torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)), 0.01)

    def test_tiles(self):
        print()
        for m in [1, 2, 3, 4, 8, 12, 16, 24, 32, 48, 64, 118, 128, 152, 768, 1024]:
            for thread_k, thread_n in [(64, 256), (128, 128)]:
                if m > 16 and thread_k == 128:
                    continue
                # testing  'ideal' case
                self.run_problem(m, 256*2, 1024, thread_k, thread_n)
    
    def test_k_stages_divisibility(self):
        print()
        for k in [3 * 64 + 64 * 4 * 2 + 64 * i for i in range(1, 4)]:
            self.run_problem(16, 2 * 256, k, 64, 256)

    def test_very_few_stages(self):
        print()
        for k in [64, 128, 192]:
            self.run_problem(16, 2 * 256, k, 64, 256)
    
    def test_errors(self):
        print()
        m, n, k = 16, 256, 64
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        B_ref, B, s = gen_fp6_weight(k, n)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        workspace = torch.zeros(n // 128, device=DEV)
        err = False
        try:
            fp6_marlin.mul(A, B, C, s, workspace, 128, 128, -1)
        except:
            err = True 
        self.assertTrue(err)
        err = False
        try:
            fp6_marlin.mul(A, B, C, s, workspace, 256, 256, -1)
        except:
            err = True 
        self.assertTrue(err)
        s = torch.zeros((2, n), dtype=torch.half, device=DEV)
        err = False
        try:
            fp6_marlin.mul(A, B, C, s, workspace, 256, 256, -1)
        except:
            err = True 
        self.assertTrue(err)

if __name__ == '__main__':
    unittest.main()
