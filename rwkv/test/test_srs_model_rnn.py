from pathlib import Path
import unittest

import numpy as np
from rwkv.prepare_batch import prepare
from rwkv.model.rwkv_model import RWKV7Config
from rwkv.model.srs_model import AnkiRWKV, extract_p
from rwkv.model.srs_model_rnn import AnkiRWKVRNN, get_df_sequentially
from rwkv.data_processing import (
    add_queries,
    add_segment_features,
    create_sample,
    get_rwkv_data,
)
import torch

from rwkv.rwkv_config import DEFAULT_ANKI_RWKV_CONFIG

DATA_PATH = Path("../anki-revlogs-10k")
DEVICE = "cuda"
RNN_DEVICE = "cpu"
DTYPE = (
    torch.float32
)  # For small df sizes where there is no rwkv time-parallelism, torch.bfloat16 should work but it doesn't. There is likely a small bug somewhere.
# TODO investigate if this is still the case


class Test(unittest.TestCase):
    def test_equivalence(self):
        # torch.jit.ScriptModule can subtly change numerics
        # https://discuss.pytorch.org/t/second-forward-call-of-torchscripted-module-breaks-on-cuda/124291/8
        torch._C._jit_override_can_fuse_on_cpu(False)
        torch._C._jit_override_can_fuse_on_gpu(False)
        torch._C._jit_set_texpr_fuser_enabled(False)

        with torch.no_grad():
            base_master = AnkiRWKV(DEFAULT_ANKI_RWKV_CONFIG).to(DEVICE)
            for param in base_master.parameters():
                torch.nn.init.uniform_(param, -0.20, 0.20)

            base = AnkiRWKV(DEFAULT_ANKI_RWKV_CONFIG).selective_cast(DTYPE).to(DEVICE)
            base.copy_downcast_(base_master, dtype=DTYPE)
            base.eval()
            rnn_master = AnkiRWKVRNN(DEFAULT_ANKI_RWKV_CONFIG).to(RNN_DEVICE)
            rnn = (
                AnkiRWKVRNN(DEFAULT_ANKI_RWKV_CONFIG)
                .selective_cast(DTYPE)
                .to(RNN_DEVICE)
            )
            rnn.copy_downcast_(rnn_master, dtype=DTYPE)
            rnn.load_state_dict(base.state_dict(), strict=False)
            rnn.eval()

            for user_id in list(range(1, 2)):
                df = get_rwkv_data(DATA_PATH, user_id, equalize_review_ths=[])
                df = df.head(min(len(df), 20))

                print("Test", user_id, len(df))
                sample = create_sample(
                    user_id,
                    df.copy(),
                    equalize_review_ths=[],
                    dtype=DTYPE,
                    device="cpu",
                )
                stats = base.get_loss(prepare([sample]).to(DEVICE))
                base_stats = extract_p(stats)

                # print("base_ahead_p", base_ahead_p)
                # print("base_imm_p", base_imm_p)

                rnn_df = add_queries(add_segment_features(df), [])
                print("start rnn")
                rnn_ahead_p, rnn_imm_p = rnn.run(rnn_df, dtype=DTYPE, device=RNN_DEVICE)
                # print("rnn ahead_p", rnn_ahead_p)
                # print("rnn imm_p", rnn_imm_p)
                print(base_stats.ahead_ps)
                print(rnn_ahead_p)
                # print((base_stats.imm_ps - rnn_imm_p).abs())
                torch.testing.assert_close(base_stats.imm_ps, rnn_imm_p)
                torch.testing.assert_close(
                    base_stats.ahead_ps, rnn_ahead_p, atol=1e-5, rtol=1e-5
                )  # TODO

    # def test_sequential_df_process_identical(self):
    #     """
    #     test_equivalence shows that an srs_model and srs_model_rnn produces the same result but it relies on non-sequential df processing
    #     In this test we show that we can sequentially process the revlog, reducing the chance of possible data leakage.
    #     """
    #     for user_id in range(42, 45):
    #         df_sequential = get_df_sequentially(DATA_PATH, user_id)
    #         df_base = get_rwkv_data(DATA_PATH, user_id, equalize_review_ths=[])
    #         df_base = add_queries(add_segment_features(df_base), [])

    #         for column in df_sequential.columns:
    #             assert column in df_base.columns, f"mismatching columns {column}"
    #             print("Testing user:", user_id, "column:", column)
    #             np.testing.assert_allclose(
    #                 df_sequential[column].to_numpy(), df_base[column].to_numpy()
    #             )

    # def test_sequential_df_usable_by_rnn(self):
    #     """Show that the sequential df contains enough information to be usable by the rnn."""
    #     torch._C._jit_override_can_fuse_on_cpu(False)
    #     torch._C._jit_override_can_fuse_on_gpu(False)
    #     torch._C._jit_set_texpr_fuser_enabled(False)

    #     with torch.no_grad():
    #         base_master = AnkiRWKV(
    #             card_rwkv_config=CARD_CONFIG,
    #             note_rwkv_config=NOTE_CONFIG,
    #             deck_rwkv_config=DECK_CONFIG,
    #             preset_rwkv_config=PRESET_CONFIG,
    #             global_rwkv_config=GLOBAL_CONFIG,
    #             dropout=DROPOUT,
    #         ).to(DEVICE)
    #         for param in base_master.parameters():
    #             torch.nn.init.uniform_(param, -0.20, 0.20)

    #         base = (
    #             AnkiRWKV(
    #                 card_rwkv_config=CARD_CONFIG,
    #                 note_rwkv_config=NOTE_CONFIG,
    #                 deck_rwkv_config=DECK_CONFIG,
    #                 preset_rwkv_config=PRESET_CONFIG,
    #                 global_rwkv_config=GLOBAL_CONFIG,
    #                 dropout=DROPOUT,
    #             )
    #             .selective_cast(DTYPE)
    #             .to(DEVICE)
    #         )
    #         base.copy_downcast_(base_master, dtype=DTYPE)
    #         base.eval()

    #         rnn_master = AnkiRWKVRNN(
    #             card_rwkv_config=CARD_CONFIG,
    #             note_rwkv_config=NOTE_CONFIG,
    #             deck_rwkv_config=DECK_CONFIG,
    #             preset_rwkv_config=PRESET_CONFIG,
    #             global_rwkv_config=GLOBAL_CONFIG,
    #         ).to(DEVICE)
    #         rnn = (
    #             AnkiRWKVRNN(
    #                 card_rwkv_config=CARD_CONFIG,
    #                 note_rwkv_config=NOTE_CONFIG,
    #                 deck_rwkv_config=DECK_CONFIG,
    #                 preset_rwkv_config=PRESET_CONFIG,
    #                 global_rwkv_config=GLOBAL_CONFIG,
    #             )
    #             .selective_cast(DTYPE)
    #             .to(DEVICE)
    #         )
    #         rnn.copy_downcast_(rnn_master, dtype=DTYPE)
    #         rnn.load_state_dict(base.state_dict())

    #         for user_id in range(42, 43):
    #             print("Testing user:", user_id)
    #             df_base = get_rwkv_data(user_id, equalize_review_ths=[])
    #             print("df len", len(df_base))

    #             sample = create_sample(
    #                 user_id, df_base, equalize_review_ths=[], dtype=DTYPE, device="cpu"
    #             )
    #             stats = base.get_loss(prepare([sample]).to(DEVICE))
    #             base_ahead_p, base_imm_p = extract_p(stats)

    #             df_sequential = get_df_sequentially(user_id)
    #             rnn_ahead_p, rnn_imm_p = rnn.run(
    #                 df_sequential, dtype=DTYPE, device=DEVICE
    #             )
    #             assert len(base_ahead_p) > 0
    #             # print("base", base_ahead_p, base_imm_p)
    #             # print("rnn", rnn_ahead_p, rnn_imm_p)
    #             torch.testing.assert_close(base_ahead_p, rnn_ahead_p)
    #             torch.testing.assert_close(base_imm_p, rnn_imm_p)
    #             print("done")


if __name__ == "__main__":
    unittest.main()
