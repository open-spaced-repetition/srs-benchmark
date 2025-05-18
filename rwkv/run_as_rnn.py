"""
This script demonstrates RWKV run as an RNN.
"""

import numpy as np
import pandas as pd
from rwkv.config import (
    DAY_OFFSET_ENCODE_PERIODS,
    ID_ENCODE_DIMS,
    ID_SPLIT,
    RWKV_SUBMODULES,
)
from rwkv.data_processing import (
    CARD_FEATURE_COLUMNS,
    ID_PLACEHOLDER,
    scale_cum_new_cards_today,
    scale_cum_reviews_today,
    scale_day_offset_diff,
    scale_diff_new_cards,
    scale_diff_reviews,
    scale_duration,
    scale_elapsed_days,
    scale_elapsed_days_cumulative,
    scale_elapsed_seconds,
    scale_elapsed_seconds_cumulative,
    scale_state,
)
from rwkv.get_result import get_benchmark_info, get_stats
import torch
from rwkv.model.srs_model_rnn import AnkiRWKVRNN
from rwkv.rwkv_config import DEFAULT_ANKI_RWKV_CONFIG


class RNNProcess:
    def __init__(
        self,
        path,
        device,
        dtype,
        config=DEFAULT_ANKI_RWKV_CONFIG,
    ):
        self.rnn = AnkiRWKVRNN(config).to(device)
        if path is not None:
            self.rnn.load_state_dict(torch.load(path, weights_only=True))
            print(f"Loaded: {path}")
        else:
            print("Did not load weights.")
        self.rnn = self.rnn.selective_cast(dtype)
        self.device = device
        self.dtype = dtype

        self.card_states = {}
        self.note_states = {}
        self.deck_states = {}
        self.preset_states = {}
        self.global_state = None
        self.first_day_offset = None
        self.prev_row = None
        self.card_set = set()
        self.last_new_cards = {}
        self.i = 0
        self.last_i = {}
        self.today = -1
        self.today_reviews = 0
        self.today_new_cards = 0
        self.card2first_day_offset = {}
        self.card2elapsed_days_cumulative = {}
        self.card2elapsed_seconds_cumulative = {}
        self.id_encodings = {submodule: {} for submodule in RWKV_SUBMODULES}

    def get_tensor(self, row):
        def add_id_encoding(features):
            def generate_id_encoding(submodule):
                ENCODE_DIM = ID_ENCODE_DIMS[submodule]
                return torch.randint(
                    low=0,
                    high=ID_SPLIT,
                    size=(ENCODE_DIM,),
                    device=self.device,
                    requires_grad=False,
                ).to(self.dtype) - ((ID_SPLIT - 1) / 2)

            gather = [features]
            for submodule in RWKV_SUBMODULES:
                if submodule == "user_id":
                    continue
                if row[submodule] not in self.id_encodings[submodule]:
                    self.id_encodings[submodule][row[submodule]] = generate_id_encoding(
                        submodule
                    )

                gather.append(self.id_encodings[submodule][row[submodule]])

            return torch.cat(gather, dim=-1)

        def add_day_offset_encoding(features):
            day_offset = torch.full((1,), row["day_offset"], device=self.device)
            day_offset_first = torch.full(
                (1,), row["day_offset_first"], device=self.device
            )
            gather = [features]
            for period in DAY_OFFSET_ENCODE_PERIODS:
                f = 2 * np.pi / period
                encodings_sin = torch.sin(f * (day_offset % period)).to(self.dtype)
                encodings_cos = torch.cos(f * (day_offset % period)).to(self.dtype)
                encodings = torch.cat((encodings_sin, encodings_cos), dim=-1)
                gather.append(encodings)
                encodings_first_sin = torch.sin(f * (day_offset_first % period)).to(
                    self.dtype
                )
                encodings_first_cos = torch.cos(f * (day_offset_first % period)).to(
                    self.dtype
                )
                encodings_first = torch.cat(
                    (encodings_first_sin, encodings_first_cos), dim=-1
                )
                gather.append(encodings_first)

            return torch.cat(gather, dim=-1)

        features = torch.tensor(
            row.loc[CARD_FEATURE_COLUMNS].tolist(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        )
        features = add_id_encoding(features)
        features = add_day_offset_encoding(features)
        features = features.unsqueeze(0)
        return features

    def run(self, row, skip):
        features = self.get_tensor(row)

        with torch.no_grad():
            card_id = row["card_id"]
            note_id = row["note_id"]
            deck_id = row["deck_id"]
            preset_id = row["preset_id"]

            if card_id not in self.card_states:
                self.card_states[card_id] = None
            if note_id not in self.note_states:
                self.note_states[note_id] = None
            if deck_id not in self.deck_states:
                self.deck_states[deck_id] = None
            if preset_id not in self.preset_states:
                self.preset_states[preset_id] = None

            (
                out_ahead_logits,
                out_w,
                out_p_logits,
                next_card_state,
                next_note_state,
                next_deck_state,
                next_preset_state,
                next_global_state,
            ) = self.rnn.review(
                features,
                self.card_states[card_id],
                self.note_states[note_id],
                self.deck_states[deck_id],
                self.preset_states[preset_id],
                self.global_state,
            )
            if not skip:
                self.card_states[card_id] = next_card_state
                self.note_states[note_id] = next_note_state
                self.deck_states[deck_id] = next_deck_state
                self.preset_states[preset_id] = next_preset_state
                self.global_state = next_global_state

            out_p_probs = torch.softmax(out_p_logits, dim=-1)
            out_p_again, _, _, _ = out_p_probs.unbind(dim=-1)
            return (out_ahead_logits, out_w), 1.0 - out_p_again

    def predict_func(self, curve, elapsed_seconds):
        elapsed_seconds = torch.tensor(elapsed_seconds, device=self.device).view(1, 1)
        out_ahead_logits, out_w = curve
        curve_probs_raw = self.rnn.forgetting_curve(out_w, elapsed_seconds)
        curve_logits_raw = torch.log(
            curve_probs_raw / (1 - curve_probs_raw)
        )  # inverse sigmoid
        ahead_logit_residual = self.rnn.interp(out_ahead_logits, elapsed_seconds)
        curve_logits = curve_logits_raw + ahead_logit_residual
        return torch.sigmoid(curve_logits)

    def add_same(self, row):
        row = row.copy()
        card_id = row["card_id"]
        row["elapsed_days_cumulative"] = (
            self.card2elapsed_days_cumulative.get(card_id, 0) + row["elapsed_days"]
        )
        row["scaled_elapsed_days_cumulative"] = scale_elapsed_days_cumulative(
            row["elapsed_days_cumulative"]
        )
        row["elapsed_seconds_cumulative"] = (
            self.card2elapsed_seconds_cumulative.get(card_id, 0)
            + row["elapsed_seconds"]
        )
        row["scaled_elapsed_seconds_cumulative"] = scale_elapsed_seconds_cumulative(
            row["elapsed_seconds_cumulative"]
        )
        SECONDS_PER_DAY = 86400
        row["elapsed_seconds_sin"] = np.sin(
            (row["elapsed_seconds"] % SECONDS_PER_DAY) * 2 * np.pi / SECONDS_PER_DAY
        )
        row["elapsed_seconds_cos"] = np.cos(
            (row["elapsed_seconds"] % SECONDS_PER_DAY) * 2 * np.pi / SECONDS_PER_DAY
        )
        row["elapsed_seconds_cumulative_sin"] = np.sin(
            (row["elapsed_seconds_cumulative"] % SECONDS_PER_DAY)
            * 2
            * np.pi
            / SECONDS_PER_DAY
        )
        row["elapsed_seconds_cumulative_cos"] = np.cos(
            (row["elapsed_seconds_cumulative"] % SECONDS_PER_DAY)
            * 2
            * np.pi
            / SECONDS_PER_DAY
        )

        if self.first_day_offset is None:
            row["day_offset"] = 0
        else:
            row["day_offset"] -= self.first_day_offset

        if card_id in self.card2first_day_offset:
            row["day_offset_first"] = self.card2first_day_offset[card_id]
        else:
            row["day_offset_first"] = row["day_offset"]

        row["day_of_week"] = ((row["day_offset"] % 7) - 3) / 3

        def add_id(name):
            if np.isnan(row[name]):
                row[name] = ID_PLACEHOLDER
                row[f"{name}_is_nan"] = 1.0
            else:
                row[f"{name}_is_nan"] = 0.0

        for name in ["note_id", "deck_id", "preset_id"]:
            add_id(name)

        row["day_offset_diff"] = scale_day_offset_diff(
            row["day_offset"]
            - (0 if self.prev_row is None else self.prev_row["day_offset"])
        )
        row["day_of_week"] = ((row["day_offset"] % 7) - 3) / 3
        unscaled_diff_new_cards = (
            (len(self.card_set) - self.last_new_cards[card_id])
            if card_id in self.last_new_cards
            else 0
        )
        row["diff_new_cards"] = scale_diff_new_cards(unscaled_diff_new_cards)
        unscaled_diff_reviews = (
            (max(0, self.i - self.last_i[card_id] - 1)) if card_id in self.last_i else 0
        )
        row["diff_reviews"] = scale_diff_reviews(unscaled_diff_reviews)

        row_today_reviews = self.today_reviews
        row_today_new_cards = self.today_new_cards
        if row["day_offset"] != self.today:
            row_today_new_cards = 0
            row_today_reviews = -1

        row_today_reviews += 1
        if row["card_id"] not in self.card_set:
            row_today_new_cards += 1
        row["cum_reviews_today"] = scale_cum_reviews_today(row_today_reviews)
        row["cum_new_cards_today"] = scale_cum_new_cards_today(row_today_new_cards)

        row["scaled_elapsed_days"] = scale_elapsed_days(row["elapsed_days"].item())
        row["scaled_elapsed_seconds"] = scale_elapsed_seconds(
            row["elapsed_seconds"].item()
        )
        return row

    def imm_predict(self, row):
        row = row.copy()
        row = self.add_same(row)
        row["is_query"] = 1.0
        row["skip"] = True
        row["scaled_duration"] = 0
        row["scaled_state"] = 0
        for i in range(1, 5):
            row[f"rating_{i}"] = 0

        _, imm_probs = self.run(row, skip=True)
        return imm_probs

    def process_row(self, row):
        row = row.copy()
        row = self.add_same(row)
        row["is_query"] = 0.0
        row["skip"] = False
        row["scaled_duration"] = scale_duration(row["duration"])
        row["scaled_state"] = scale_state(row["state"])
        for i in range(1, 5):
            row[f"rating_{i}"] = 1.0 if row["rating"] == i else 0.0

        # Get prediction
        curve, _ = self.run(row, skip=False)

        # Update state
        card_id = row["card_id"]
        self.card2elapsed_days_cumulative[card_id] = (
            self.card2elapsed_days_cumulative.get(card_id, 0) + row["elapsed_days"]
        )
        self.card2elapsed_seconds_cumulative[card_id] = (
            self.card2elapsed_seconds_cumulative.get(card_id, 0)
            + row["elapsed_seconds"]
        )

        if self.first_day_offset is None:
            self.first_day_offset = row["day_offset"]

        if row["day_offset"] != self.today:
            self.today = row["day_offset"]
            self.today_new_cards = 0
            self.today_reviews = -1
        self.today_reviews += 1
        if row["card_id"] not in self.card_set:
            self.today_new_cards += 1
            self.card_set.add(card_id)
            self.card2first_day_offset[card_id] = (
                row["day_offset"] - self.first_day_offset
            )

        self.prev_row = row.copy()
        self.last_i[card_id] = self.i
        self.last_new_cards[card_id] = len(self.card_set)
        self.i += 1
        return curve


@torch.inference_mode()
def run(data_path, model_path, label_db_path, label_db_size, user_id):
    """Runs the rnn version of rwkv to explicitly show information flow. Written to guard against possible data leakage.
    The outputs will not exactly match with the parallel version. Reasons:
    - hard to match rng
    - not all computations have the right dtype (bfloat16, float)
    - JIT may fuse some floating point operations
    However, the performance should be similar.
    """
    df = pd.read_parquet(data_path / "revlogs", filters=[("user_id", "=", user_id)])
    df["review_th"] = range(1, df.shape[0] + 1)
    df_cards = pd.read_parquet(data_path / "cards", filters=[("user_id", "=", user_id)])
    df_cards.drop(columns=["user_id"], inplace=True)
    df_decks = pd.read_parquet(data_path / "decks", filters=[("user_id", "=", user_id)])
    df_decks.drop(columns=["user_id", "parent_id"], inplace=True)
    df = df.merge(df_cards, on="card_id", how="left", validate="many_to_one")
    df = df.merge(df_decks, on="deck_id", how="left", validate="many_to_one")
    df["review_th"] = range(1, df.shape[0] + 1)

    equalize_review_ths, rmse_bins = get_benchmark_info(
        label_db_path, label_db_size, user_id
    )
    rmse_bins_dict = {
        equalize_review_ths[i]: rmse_bins[i] for i in range(len(equalize_review_ths))
    }

    srs_rnn = RNNProcess(
        path=model_path, device=torch.device("cpu"), dtype=torch.float32
    )

    pred_imm = {}
    pred_ahead_curve = {}  # Map from card_ids to their latest forgetting curve
    pred_ahead = {}
    label_rating = {}
    import time

    print("revlog len:", len(df))

    time_start = time.time()
    for i, row in df.iterrows():
        if (i + 1) % 100 == 0:
            print(f"{i}/{len(df)}, rate: {(i + 1) / (time.time() - time_start):.2f}")

        card_id = row["card_id"]
        review_th = row["review_th"]

        # If this card has been seen before, get the predicted retention from the stored function
        if card_id in pred_ahead_curve:
            z = srs_rnn.predict_func(pred_ahead_curve[card_id], row["elapsed_seconds"])
            pred_ahead[review_th] = z

        # For the immediate predictions we do not send the rating and duration fields
        imm_info = row.copy()
        imm_info.drop(columns=["rating", "duration"], inplace=True)
        pred_imm[review_th] = srs_rnn.imm_predict(imm_info)

        # For ahead-of-time predictions, predict and store a prediction function
        # This function will be evaluated at the time of the next review of this card
        pred_ahead_curve[card_id] = srs_rnn.process_row(row)
        label_rating[review_th] = row["rating"] - 1

    imm_stats, _ = get_stats(
        user_id,
        equalize_review_ths,
        rmse_bins_dict,
        pred_imm,
        label_rating,
    )
    ahead_stats, _ = get_stats(
        user_id,
        equalize_review_ths,
        rmse_bins_dict,
        pred_ahead,
        label_rating,
    )

    print("RWKV-P:")
    print(imm_stats)
    print("RWKV:")
    print(ahead_stats)


if __name__ == "__main__":
    from pathlib import Path

    run(
        Path("../anki-revlogs-10k"),
        "pretrain/RWKV_trained_on_5000_10000.pth",
        # "pretrain/RWKV_trained_on_101_4999.pth",
        "label_filter_db",
        int(7e9),
        user_id=107,
    )
