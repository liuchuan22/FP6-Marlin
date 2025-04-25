import numpy as np
import torch
import unittest

import fp6_marlin

seed = 0
np.random.seed(seed)
torch.random.manual_seed(seed)
DEV = torch.device('cuda:0')
torch.set_printoptions(sci_mode=False, linewidth=90) 

def gen_fp6_weight(w: torch.Tensor, k: int, n: int):
    # 3 exponent bit, 2 mantissa bit
    maxq = 28.0
    minq = 0.0625
    # w = torch.randn((k, n), dtype=torch.half, device='cpu')
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

def get_perms():
    perm = []
    for i in range(32):
        perm1 = []
        col = i // 4
        for block in [0, 1]:
            for row in [
                2 * (i % 4),
                2 * (i % 4) + 1,
                2 * (i % 4 + 4),
                2 * (i % 4 + 4) + 1
            ]:
                perm1.append(16 * row + col + 8 * block)
        for j in range(4):
            perm.extend([p + 256 * j for p in perm1])

    perm = np.array(perm)
    interleave = np.array([0, 2, 4, 6, 1, 3, 5, 7])
    perm = perm.reshape((-1, 8))[:, interleave].ravel()
    perm = torch.from_numpy(perm)
    scale_perm = []
    for i in range(4):
        scale_perm.extend([2 * i + j for j in [0, 1, 8, 9, 16, 17, 24, 25]])
    return perm, scale_perm

def gen_int4_weight(w: torch.Tensor, k: int, n: int):
    maxq = 2 ** 4 - 1
    # w = torch.randn((k, n), dtype=torch.half, device='cpu')
    s = torch.max(torch.abs(w), 0, keepdim=True)[0]
    s *= 2 / maxq
    w = torch.round(w / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    s = s.reshape((-1, n)).contiguous()

    # debugging, set s as a tensor of ones
    # s = torch.ones((1, n), dtype=torch.half, device='cpu')

    ref = (w - (maxq + 1) // 2).half() * s
    w = torch.round(ref / s).int()
    w += (maxq + 1) // 2
    w = torch.clamp(w, 0, maxq)
    perm, scale_perm = get_perms()
    s = s.reshape((-1, len(scale_perm)))[:, scale_perm]
    s = s.reshape((-1, n)).contiguous()
    s = s.squeeze(0) # shape (1, n) -> (n,)
    w = w.reshape((k // 16, 16, n // 16, 16))
    w = w.permute((0, 2, 1, 3))
    w = w.reshape((k // 16, n * 16))
    w = w.reshape((-1, perm.numel()))[:, perm].reshape(w.shape)
    w = w.numpy().astype(np.uint32)
    q = np.zeros((w.shape[0], w.shape[1] // 8), dtype=np.uint32)
    for i in range(8):
        q |= w[:, i::8] << 4 * i
    q = torch.from_numpy(q.astype(np.int32))
    q = q.contiguous().to(DEV)
    s = s.contiguous().to(DEV)
    ref = ref.contiguous().to(DEV)
    assert q.shape[0] == k // 16
    assert q.shape[1] == n * 2
    assert q.dtype == torch.int32
    return ref, q, s

def gen_mixed_weight(K: int, N: int, quant_cols: int = -1):
    w = torch.randn((K, N), dtype=torch.half, device='cpu')
    if quant_cols == -1:
        return gen_int4_weight(w, K, N)
    elif quant_cols == N:
        return gen_fp6_weight(w, K, N)
    assert quant_cols % 64 == 0
    w1, w2 = w.split([N - quant_cols, quant_cols], dim=1)
    w1_ref, w1_packed, s1 = gen_int4_weight(w1, K, N - quant_cols)
    w2_ref, w2_packed, s2 = gen_fp6_weight(w2, K, quant_cols)

    w_ref = torch.cat([w1_ref, w2_ref], dim=1)
    assert w_ref.shape[0] == K
    assert w_ref.shape[1] == N
    assert w_ref.dtype == torch.half

    s = torch.cat([s1, s2], dim=0)
    assert s.shape[0] == N
    assert s.dtype == torch.half

    w1_packed = w1_packed.flatten()
    w2_packed = w2_packed.flatten()
    w_packed = torch.cat([w1_packed, w2_packed], dim=0)
    assert w_packed.shape[0] == K * 3 // 32 * 2 * quant_cols + K // 16 * 2 * (N - quant_cols)
    assert w_packed.dtype == torch.int32

    w_ref = w_ref.contiguous().to(DEV)
    w_packed = w_packed.contiguous().to(DEV)
    s = s.contiguous().to(DEV)
    return w_ref, w_packed, s

# gen_mixed_weight(256, 256, 64)
# breakpoint()

class Test(unittest.TestCase):

    def run_problem(self, m, n, k, thread_k, thread_n, quant_cols=-1):
        print('% 5d % 6d % 6d % 4d % 4d % 4d' % (m, n, k, thread_k, thread_n, quant_cols))
        A = torch.randn((m, k), dtype=torch.half, device=DEV)
        # A = torch.randint(1, 128, (m, k), dtype=torch.half, device=DEV)
        # A = torch.full((m, k), 61, dtype=torch.half, device=DEV)
        # A[0, :16] = 0.
        B_ref, B, s = gen_mixed_weight(k, n, quant_cols)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV).contiguous()
        C_ref = torch.matmul(A, B_ref)
        workspace = torch.zeros(n // 128 * 16, device=DEV)
        fp6_marlin.mul(A, B, C, s, workspace, quant_cols, thread_k, thread_n)
        torch.cuda.synchronize()
        # C = C.flatten()
        # if quant_cols != -1:
        #     c_1 = C[:m*(n-quant_cols)].reshape(m, n-quant_cols)
        #     c_2 = C[m*(n-quant_cols):].reshape(m, quant_cols)
        #     C = torch.cat([c_1, c_2], dim=1)
        # C = C.reshape(m, n)
        # print(C)
        # print(C_ref)
        # print('C', C.shape)
        # print(C-C_ref)
        self.assertLess(torch.mean(torch.abs(C - C_ref)) / torch.mean(torch.abs(C_ref)), 0.001)

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
        B_ref, B, s = gen_mixed_weight(k, n)
        C = torch.zeros((m, n), dtype=torch.half, device=DEV)
        workspace = torch.zeros(n // 128, device=DEV)
        err = False
        try:
            fp6_marlin.mul(A, B, C, s, workspace, -1, 128, 128, -1)
        except:
            err = True 
        self.assertTrue(err)
        err = False
        try:
            fp6_marlin.mul(A, B, C, s, workspace, -1, 256, 256, -1)
        except:
            err = True 
        self.assertTrue(err)
        s = torch.zeros((2, n), dtype=torch.half, device=DEV)
        err = False
        try:
            fp6_marlin.mul(A, B, C, s, workspace, -1, 256, 256, -1)
        except:
            err = True 
        self.assertTrue(err)

    def test_mixed(self):
        print()
        for m in [1, 12, 16, 64, 65, 77, 152, 256, 1024]:
            for quant_cols in [256]:
                for n, k in [(256, 512), (256*2, 1024)]:
                    for thread_shape in [(128, 128), (64, 256)]:
                        if m > 16 and thread_shape[0] == 128:
                            continue
                        self.run_problem(m, n, k, *thread_shape, quant_cols)

if __name__ == '__main__':
    unittest.main()
