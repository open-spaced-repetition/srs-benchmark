import math
import time
import torch
from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.optests import opcheck
import unittest
import llm.rwkv
from torch import Tensor
from typing import Tuple
import torch.nn.functional as F
import numpy as np

from llm.rwkv.rwkv_model import RWKV7, RWKV7Config

DTYPE = torch.float32

def single_timestep(r_BHK, k_BHK, v_BHK, w_BHK, a_BHK, k_deformed_BHK, state_BHKK):
    r_BHK1 = r_BHK.unsqueeze(-1)
    k_BHK1 = k_BHK.unsqueeze(-1)
    v_BHK1 = v_BHK.unsqueeze(-1)
    w_BHK1 = w_BHK.unsqueeze(-1)
    a_BHK1 = a_BHK.unsqueeze(-1)
    k_deformed_BHK1 = k_deformed_BHK.unsqueeze(-1)

    # Uses broadcasting. Remember that each column in vk_skate gets its own decay.
    state_BHKK = state_BHKK * w_BHK1.mT - state_BHKK @ k_deformed_BHK1 @ (a_BHK1 * k_deformed_BHK1).mT
    state_BHKK = state_BHKK + (v_BHK1 @ k_BHK1.mT)

    # Now we have a new updated S. We evaluate it at r and return the output.
    out_BHK1 = state_BHKK @ r_BHK1
    return out_BHK1.squeeze(-1), state_BHKK

def reference_rwkv7(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT):
    r_BTHK = r_BTHK.float()
    k_BTHK = k_BTHK.float()
    v_BTHK = v_BTHK.float()
    w_BTHK = w_BTHK.float()
    a_BTHK = a_BTHK.float()
    k_deformed_BTHK = k_deformed_BTHK.float()
    skip_BT111 = skip_BT.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    B, T, H, K = r_BTHK.shape
    out_BTHK = torch.empty(B, T, H, K, dtype=torch.float32, device=r_BTHK.device)
    state_BHKK = torch.zeros(B, H, K, K, dtype=torch.float32, device=r_BTHK.device)
    for t in range(T):
        out_BTHK[:, t], next_state_BHKK = single_timestep( r_BTHK[:, t],
                                                k_BTHK[:, t],
                                                v_BTHK[:, t],
                                                w_BTHK[:, t],
                                                a_BTHK[:, t],
                                                k_deformed_BTHK[:, t],
                                                state_BHKK)
        skip_B111 = skip_BT111[:, t]
        state_BHKK = torch.where(skip_B111, state_BHKK, next_state_BHKK)
    return out_BTHK.to(DTYPE)

class TestRWKV(TestCase):
    def _test_correctness(self, name, device, B, T, H, verbose):
        with torch.no_grad():
            K = 32
            # dtype = torch.float16
            dtype = DTYPE
            r_BTHK = torch.randn(B, T, H, K, dtype=dtype, device=device) / math.sqrt(K)
            k_BTHK = torch.randn(B, T, H, K, dtype=dtype, device=device) / math.sqrt(K)
            v_BTHK = torch.randn(B, T, H, K, dtype=dtype, device=device)
            w_BTHK = torch.rand(B, T, H, K, dtype=torch.float32, device=device)
            a_BTHK = torch.rand(B, T, H, K, dtype=dtype, device=device, requires_grad=True)
            k_deformed_BTHK = torch.randn(B, T, H, K, dtype=dtype, device=device) / math.sqrt(K)
            skip_BT = torch.rand(B, T, device=device, requires_grad=False) > 0.5

            out_reference = reference_rwkv7(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT)
            time_start = time.time()
            out = llm.rwkv.rwkv_ops.RWKV7_WKV.apply(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT)
            end_time = time.time()
            print(f"Elapsed {name}: {end_time-time_start:.4f}")
            if verbose:
                print("refer")
                print(out_reference)
                print("ours")
                print(out)
                print("diff")
                print(out - out_reference)
            torch.testing.assert_close(out_reference, out)

    def test_correctness_cuda(self):
        self._test_correctness("short", "cuda", B=2100, T=80, H=1, verbose=False)

    def test_correctness_long_time_cuda(self):
        self._test_correctness("long", "cuda", B=3, T=20000, H=2, verbose=False)

    def _test_gradients(self, device, B, T, H, verbose):
        K = 32
        dtype = DTYPE
        r_BTHK = torch.randn(B, T, H, K, dtype=dtype, device=device, requires_grad=True) / math.sqrt(K)
        k_BTHK = torch.randn(B, T, H, K, dtype=dtype, device=device, requires_grad=True) / K
        v_BTHK = torch.randn(B, T, H, K, dtype=dtype, device=device, requires_grad=True) / math.sqrt(K)
        w_BTHK = torch.rand(B, T, H, K, dtype=torch.float32, device=device, requires_grad=True)
        a_BTHK = torch.rand(B, T, H, K, dtype=dtype, device=device, requires_grad=True)
        k_deformed_BTHK = torch.randn(B, T, H, K, dtype=dtype, device=device, requires_grad=True) / K
        skip_BT = torch.rand(B, T, device=device, requires_grad=False) > 0.5
        # skip_BTH = torch.zeros(B, T, H, device=device, requires_grad=False, dtype=torch.bool)
        params = [r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK]

        out_reference = reference_rwkv7(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT)
        grad_out = 1 * torch.randn_like(out_reference)
        grad_out_copy = grad_out.clone()
        grad_reference = torch.autograd.grad(out_reference, params, grad_out)

        time_start = time.time()
        out = llm.rwkv.rwkv_ops.RWKV7_WKV.apply(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT)
        grad = torch.autograd.grad(out, params, grad_out)
        # REPEAT = 1000000
        # out2 = llm.rwkv.rwkv_ops.RWKV7_WKV.apply(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK)
        # grad_reference = torch.autograd.grad(out2, params, grad_out)  # TODO remove this for checking
        # for i in range(REPEAT):
        #     out_r = llm.rwkv.rwkv_ops.RWKV7_WKV.apply(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK)
        #     grad_r = torch.autograd.grad(out_r, params, grad_out)
        #     # print("repeat", i)
        #     if not torch.allclose(torch.stack(grad), torch.stack(grad_r)):
        #         print("BAD")
        #         print(grad)
        #         print(grad_r)
        #     torch.testing.assert_close(grad, grad_r)

        grad_all = torch.stack(grad)
        end_time = time.time()
        print(f"Elapsed: {end_time-time_start:.4f}. Max abs: {grad_all.abs().max():.3f}")
        torch.testing.assert_close(grad_out, grad_out_copy)

        if verbose:
            # for i in range(6):
            #     # pass
            #     print("ref", i)
            #     print(grad_reference[i][0, 18, 1, 31])
            #     print('ours', i)
            #     print(grad[i][0, 18, 1, 31])
            # print(out_reference)
            # print(grad_reference)
            for i in range(6):
                print("Ours", i)
                print(grad[i].max())
                # print(grad[i])
                print("Answer:", i)
                print(grad_reference[i].max())
                print(i, "max error: ", (grad[i] - grad_reference[i]).abs().max())
                print(i, "max rel: ", (grad[i] / grad_reference[i]).abs().max())
                # print(grad_reference[i])
                if grad[i].mean().isnan():
                    print("NAN OURS")
                if grad_reference[i].mean().isnan():
                    print("NAN REFERENCE")
                    # print(grad_reference[i])
                assert not grad_reference[i].mean().isnan()

                # for t in range(T):
                #     diff = (grad[i][:, t] - grad_reference[i][:, t]).max().abs()
                #     print(f"{i} {t}: {diff:.5f}")
                #     if diff > 1e-4:
                #         print("correct")
                #         print(grad_reference[i][:, t])
                #         print("ours")
                #         print(grad[i][:, t])

            # print(grad_reference[1])
            # print(grad_reference[1].mean())

        torch.testing.assert_close(grad_reference, grad)

    def test_gradients_cuda(self):
        # self._test_gradients("cuda", B=1, T=2000, H=1, verbose=False)
        for t in range(1, 200, 10):
            print("testing len backwards:", t)
            self._test_gradients("cuda", B=1, T=t, H=1, verbose=False)

    def test_gradients_long_time_cuda(self):
        self._test_gradients("cuda", B=2, T=3000, H=3, verbose=False)

    # def test_timemixer(self):
    #     class Model(torch.nn.Module):
    #         def __init__(self):
    #             super().__init__()
    #             self.linear = torch.nn.Linear(1, 32*4)
    #             config = RWKV7Config(d_model=32*4, n_heads=4, n_layers=4, channel_mixer_factor=3, decay_lora=8, a_lora=8, v0_mix_amt_lora=8, gate_lora=16)
    #             self.rnn = RWKV7(config=config)
    #             # self.rnn = torch.nn.LSTM(32*4, 32*4, num_layers=2, batch_first=True)

    #         def forward(self, x):
    #             x = self.linear(x)
    #             # x, _ = self.rnn(x)
    #             x = self.rnn(x)
    #             return x

    #     # Overfit rwkv on a simple dataset
    #     L = 100
    #     C = 5
    #     def generate_sample(batch_size):
    #         assert L % C == 0
    #         N = L // C
    #         base = []
    #         labels = []
    #         mask = []
    #         for i in range(N):
    #             shape = (batch_size, C, 1)
    #             base.append(np.random.random(size=shape))
    #             if i == 0:
    #                 labels.append(np.zeros(shape))
    #                 mask.append(np.zeros(shape))
    #             else:
    #                 labels.append(base[i - 1])
    #                 mask.append(np.ones(shape))

    #         features = np.concatenate(base, axis=1)
    #         labels = np.concatenate(labels, axis=1)
    #         mask = np.concatenate(mask, axis=1)
    #         # shape = (batch_size, L // 2, 1)
    #         # base = np.random.random(size=shape)
    #         # features = np.concatenate((base, np.zeros(shape)), axis=1)
    #         # labels = np.concatenate((np.zeros(shape), base), axis=1)
    #         # mask = np.concatenate((np.zeros(shape), np.ones(shape)), axis=1)
    #         return features, labels, mask

    #     model = Model().to("cuda")
    #     optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    #     # print(generate_sample(1))
    #     # exit()

    #     for e in range(5000):
    #         features, labels, mask = generate_sample(1)
    #         features = torch.tensor(features, dtype=torch.float32, device="cuda")
    #         labels = torch.tensor(labels, dtype=torch.float32, device="cuda")
    #         mask = torch.tensor(mask, dtype=torch.float32, device="cuda")
    #         y = model(features)
    #         # print(y)
    #         loss = (((y - labels) ** 2) * mask).mean()
    #         print("loss", loss)
    #         optimizer.zero_grad()
    #         loss.backward()

    #         torch.set_printoptions(precision=8)
    #         if e == 10:
    #             for name, param in model.named_parameters():
    #                 if "rnn.blocks.1.time_mixer.lora_A_g.weight" in name:
    #                     print(name, param)
    #                     print(param.grad)

    #         optimizer.step()

    #         if e == 10:
    #             for name, param in model.named_parameters():
    #                 if "rnn.blocks.1.time_mixer.lora_A_g.weight" in name:
    #                     print("AFTER", name, param)
    #             exit()

if __name__ == "__main__":
    unittest.main()
