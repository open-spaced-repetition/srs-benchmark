
import unittest
from llm.prepare_batch import prepare
from llm.rwkv.rwkv_model import RWKV7Config
from llm.rwkv.srs_model import AnkiRWKV
from llm.data_processing import create_sample, get_rwkv_data
import torch

DTYPE = torch.float32
DEVICE = "cuda"
DROPOUT = 0.0
CARD_CONFIG = RWKV7Config(d_model=32*2, n_heads=2, n_layers=2, channel_mixer_factor=3, decay_lora=8, a_lora=8, v0_mix_amt_lora=8, gate_lora=16, k_scale_lora=4, v_scale_lora=4, dropout=DROPOUT)
NOTE_CONFIG = RWKV7Config(d_model=32*2, n_heads=2, n_layers=2, channel_mixer_factor=3, decay_lora=8, a_lora=8, v0_mix_amt_lora=8, gate_lora=16, k_scale_lora=4, v_scale_lora=4, dropout=DROPOUT)
DECK_CONFIG = RWKV7Config(d_model=32*2, n_heads=2, n_layers=2, channel_mixer_factor=3, decay_lora=8, a_lora=8, v0_mix_amt_lora=8, gate_lora=16, k_scale_lora=4, v_scale_lora=4, dropout=DROPOUT)
PRESET_CONFIG = RWKV7Config(d_model=32*2, n_heads=2, n_layers=2, channel_mixer_factor=3, decay_lora=8, a_lora=8, v0_mix_amt_lora=8, gate_lora=16, k_scale_lora=4, v_scale_lora=4, dropout=DROPOUT)
GLOBAL_CONFIG = RWKV7Config(d_model=32*2, n_heads=2, n_layers=2, channel_mixer_factor=3, decay_lora=8, a_lora=8, v0_mix_amt_lora=8, gate_lora=16, k_scale_lora=4, v_scale_lora=4, dropout=DROPOUT)

class Test(unittest.TestCase):
    def test_same_data_equivalence(self):
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)

        with torch.no_grad():
            for user_id in [1, 2, 3]:
                df = get_rwkv_data(user_id)
                # df = df[(df["review_th"] >= 16) & (df["review_th"] <= 18)]
                # print(df)
                sample = create_sample(user_id, df, dtype=DTYPE, device="cpu")

                master_model = AnkiRWKV(CARD_CONFIG, NOTE_CONFIG, DECK_CONFIG, PRESET_CONFIG, GLOBAL_CONFIG, dropout=0.0).to(DEVICE)
                for param in master_model.parameters():
                    torch.nn.init.uniform_(param, -0.20, 0.20)
                model = AnkiRWKV(card_rwkv_config=CARD_CONFIG, note_rwkv_config=NOTE_CONFIG, deck_rwkv_config=DECK_CONFIG, preset_rwkv_config=PRESET_CONFIG, global_rwkv_config=GLOBAL_CONFIG, dropout=DROPOUT).selective_cast(DTYPE).to(DEVICE)
                model.copy_downcast_(master_model, dtype=DTYPE)
                model.eval()
                prepared_batch = prepare([sample]).to(DEVICE)
                stats = model.get_loss(prepared_batch)
                p = stats.retention
                p = p.squeeze(0)

                prepared_batch2 = prepare([sample, sample]).to(DEVICE)
                stats2 = model.get_loss(prepared_batch2)
                p2 = stats2.retention
                torch.testing.assert_close(p, p2[0], equal_nan=True)
                torch.testing.assert_close(p, p2[1], equal_nan=True)

    def test_swap_equivalence(self):
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)

        with torch.no_grad():
            df1 = get_rwkv_data(1)
            # df1 = df1.head(50)
            sample1 = create_sample(1, df1, dtype=DTYPE, device="cpu")
            df2 = get_rwkv_data(3)
            # df2 = df2.head(50)
            sample2 = create_sample(3, df2, dtype=DTYPE, device="cpu")

            master_model = AnkiRWKV(CARD_CONFIG, NOTE_CONFIG, DECK_CONFIG, PRESET_CONFIG, GLOBAL_CONFIG, dropout=0.0).to(DEVICE)
            for param in master_model.parameters():
                torch.nn.init.uniform_(param, -0.20, 0.20)
            model = AnkiRWKV(card_rwkv_config=CARD_CONFIG, note_rwkv_config=NOTE_CONFIG, deck_rwkv_config=DECK_CONFIG, preset_rwkv_config=PRESET_CONFIG, global_rwkv_config=GLOBAL_CONFIG, dropout=DROPOUT).selective_cast(DTYPE).to(DEVICE)
            model.copy_downcast_(master_model, dtype=DTYPE)
            model.eval()
            prep1 = prepare([sample1, sample2]).to(DEVICE)
            stats = model.get_loss(prep1)
            prep2 = prepare([sample2, sample1]).to(DEVICE)
            stats2 = model.get_loss(prep2)
            stats3 = model.get_loss(prepare([sample1]).to(DEVICE))
            stats4 = model.get_loss(prepare([sample2]).to(DEVICE))
            l = stats.loss_tensor
            l2 = stats2.loss_tensor
            # print(p)
            # print(p2)
            torch.testing.assert_close(l[0], l2[1], equal_nan=True)
            torch.testing.assert_close(l[1], l2[0], equal_nan=True)
            torch.testing.assert_close(stats.loss_tensor[0].sum(), stats3.loss_tensor[0].sum())
            torch.testing.assert_close(stats.loss_tensor[1].sum(), stats4.loss_tensor[0].sum())

if __name__ == '__main__':
    unittest.main()