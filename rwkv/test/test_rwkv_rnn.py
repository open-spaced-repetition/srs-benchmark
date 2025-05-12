import unittest
from rwkv.model.rwkv_rnn_model import RWKV7RNN, RWKV7RNNChannelMixer, RWKV7RNNTimeMixer
import torch

from rwkv.model.rwkv_model import RWKV7, RWKV7ChannelMixer, RWKV7Config, RWKV7TimeMixer
from rwkv.config import *
import pandas as pd


class Test(unittest.TestCase):
    def _test_module(self, BaseModule, RNNModule, device):
        # torch.jit.ScriptModule can subtly change numerics
        # https://discuss.pytorch.org/t/second-forward-call-of-torchscripted-module-breaks-on-cuda/124291/8
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)

        # config = RWKV7Config(d_model=32*1, n_heads=1, n_layers=2, channel_mixer_factor=4, decay_lora=8, a_lora=8, v0_mix_amt_lora=8, gate_lora=16)
        config = RWKV7Config(
            d_model=32 * 1,
            n_heads=1,
            n_layers=2,
            channel_mixer_factor=3,
            layer_offset=0,
            total_layers=2,
            decay_lora=8,
            a_lora=8,
            v0_mix_amt_lora=8,
            gate_lora=16,
            dropout=0.1,
            dropout_layer=0.2,
        )
        dtype = torch.bfloat16
        B = 2
        T = 3
        C = config.d_model
        time_shift_select_BT = (
            torch.cat(
                [
                    torch.zeros(1, dtype=torch.long, device=device),
                    torch.arange(T - 1, device=device),
                ]
            )
            .unsqueeze(0)
            .repeat(B, 1)
        )
        skip_BT = torch.full((B, T), fill_value=False, dtype=torch.bool, device=device)
        if BaseModule == RWKV7:
            base = BaseModule(config).to(dtype).to(device)
        else:
            base = BaseModule(config, 0).to(dtype).to(device)
        base.eval()
        optim = torch.optim.Adam(base.parameters(), lr=1e-1)

        for _ in range(5):
            x_BTC = torch.randn(B, T, C, dtype=dtype, device=device)
            v0_BTC = torch.randn(B, T, C, dtype=dtype, device=device)
            label_BTC = torch.randn_like(x_BTC)
            if BaseModule == RWKV7TimeMixer:
                y_BTC, _ = base(x_BTC, v0_BTC, time_shift_select_BT, skip_BT)
            else:
                y_BTC = base(x_BTC, time_shift_select_BT)
            loss = ((y_BTC - label_BTC) ** 2).mean()
            optim.zero_grad()
            loss.backward()
            optim.step()

        x_BTC = torch.randn(B, T, C, dtype=dtype, device=device)
        v0_BTC = torch.randn(B, T, C, dtype=dtype, device=device)
        if BaseModule == RWKV7TimeMixer:
            label_BTC, _ = base(x_BTC, v0_BTC, time_shift_select_BT, skip_BT)
        else:
            label_BTC = base(x_BTC, time_shift_select_BT)

        # Check that sending the inputs individually gets the wrong answer
        if BaseModule == RWKV7:
            rnn = RNNModule(config).to(dtype).to(device)
        else:
            rnn = RNNModule(config, 0).to(dtype).to(device)
        rnn.eval()
        for t in range(1, T):
            rnn.load_state_dict(base.state_dict())
            if BaseModule == RWKV7TimeMixer:
                y_t, _, _ = rnn(
                    x_BTC[:, t].squeeze(1), v0_BTC[:, t].squeeze(1), state=None
                )
            else:
                y_t, _ = rnn(x_BTC[:, t].squeeze(1), state=None)
            try:
                torch.testing.assert_close(y_t, label_BTC[:, t])
                raise ValueError(
                    "Tensors are close. Either the test is wrong or the models are not utilizing state properly."
                )
            except AssertionError:
                pass

        # check that sending the inputs sequentially gets the right answer
        rnn.load_state_dict(base.state_dict())
        state = None
        for t in range(T):
            if BaseModule == RWKV7TimeMixer:
                y_t, _, state = rnn(
                    x_BTC[:, t].squeeze(1), v0_BTC[:, t].squeeze(1), state
                )
            else:
                y_t, state = rnn(x_BTC[:, t].squeeze(1), state)

            torch.testing.assert_close(y_t, label_BTC[:, t])

    def test_channel_mixer_cuda(self):
        self._test_module(RWKV7ChannelMixer, RWKV7RNNChannelMixer, torch.device("cuda"))

    def test_time_mixer_cuda(self):
        self._test_module(RWKV7TimeMixer, RWKV7RNNTimeMixer, torch.device("cuda"))

    def test_channel_mixer_cpu(self):
        self._test_module(RWKV7ChannelMixer, RWKV7RNNChannelMixer, torch.device("cpu"))

    def test_time_mixer_cpu(self):
        self._test_module(RWKV7TimeMixer, RWKV7RNNTimeMixer, torch.device("cpu"))

    # def test_rwkv_cuda(self):
    #     self._test_module(RWKV7, RWKV7RNN, torch.device("cuda"))

    # def test_rwkv_cpu(self):
    #     self._test_module(RWKV7, RWKV7RNN, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
