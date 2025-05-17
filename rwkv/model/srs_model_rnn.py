import numpy as np
import pandas as pd
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
from rwkv.model.rwkv_rnn_model import RWKV7RNN
from rwkv.model.srs_model import is_excluded
import torch
from rwkv.rwkv_config import DEFAULT_ANKI_RWKV_CONFIG

from rwkv.rwkv_config import AnkiRWKVConfig

# An RNN implementation of srs_model.


def __nop(ob):
    return ob


ModuleType = torch.nn.Module
FunctionType = __nop

# ModuleType = torch.jit.ScriptModule
# FunctionType = torch.jit.script_method


class AnkiRWKVRNN(ModuleType):
    def __init__(self, anki_rwkv_config: AnkiRWKVConfig):
        super().__init__()
        self.card_features_dim = 92
        self.d_model = anki_rwkv_config.d_model
        self.features_fc_dim = 4 * anki_rwkv_config.d_model
        self.ahead_head_dim = 4 * self.d_model
        self.p_head_dim = 4 * self.d_model
        self.w_head_dim = 4 * self.d_model
        self.num_curves = 128

        self.features2card = torch.nn.Sequential(
            torch.nn.Linear(self.card_features_dim, self.features_fc_dim),
            torch.nn.SiLU(),
            torch.nn.LayerNorm(self.features_fc_dim),
            torch.nn.Linear(self.features_fc_dim, self.d_model),
            torch.nn.SiLU(),
        )
        self.rwkv_modules = torch.nn.ModuleList(
            [RWKV7RNN(config=config) for _, config in anki_rwkv_config.modules]
        )
        self.prehead_norm = torch.nn.LayerNorm(self.d_model)
        self.prehead_dropout = torch.nn.Dropout(p=anki_rwkv_config.dropout)
        self.head_ahead_logits = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.ahead_head_dim),
            torch.nn.ReLU(),
        )
        self.head_w = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, 1 * self.d_model),
            torch.nn.ReLU(),
            torch.nn.LayerNorm(1 * self.d_model),
            torch.nn.Dropout(p=0.1),
            torch.nn.Linear(1 * self.d_model, self.w_head_dim),
        )
        self.head_p = torch.nn.Sequential(
            torch.nn.Linear(self.d_model, self.p_head_dim),
            torch.nn.ReLU(),
        )

        self.max_e = 21
        self.point_spread = 18.5
        self.num_points = 128
        self.ahead_linear = torch.nn.Linear(self.ahead_head_dim, self.num_points)

        self.w_linear = torch.nn.Linear(self.w_head_dim, self.num_curves)

        self.s_point_spread = 18.5
        self.s_max = 22

        self.p_linear = torch.nn.Linear(self.p_head_dim, 4)

        # self.head_curve = torch.nn.Sequential(
        #     torch.nn.LayerNorm(global_rwkv_config.d_model),
        #     torch.nn.Linear(global_rwkv_config.d_model, self.head_dim),
        #     torch.nn.SiLU(),
        # )
        # self.head_p = torch.nn.Sequential(
        #     torch.nn.LayerNorm(global_rwkv_config.d_model),
        #     torch.nn.Linear(global_rwkv_config.d_model, self.head_dim),
        #     torch.nn.SiLU(),
        # )

        # self.w_linear = torch.nn.Linear(self.head_dim, self.num_curves)
        # self.s_linear = torch.nn.Linear(self.head_dim, self.num_curves)
        # self.d_linear = torch.nn.Linear(self.head_dim, self.num_curves)
        # self.d_softplus = torch.nn.Softplus()
        # self.p_linear = torch.nn.Linear(self.head_dim, 4)

    def forgetting_curve(self, w, label_elapsed_seconds):
        s_space_raw = torch.exp(
            torch.linspace(0, self.s_point_spread, self.num_curves, device=w.device)
        )
        s_space = 0.1 + (s_space_raw - 1) * (np.e ** (self.s_max - self.s_point_spread))
        label_elapsed_seconds = torch.max(torch.tensor(1.0), label_elapsed_seconds)
        return 1e-5 + (1 - 2 * 1e-5) * torch.sum(
            w * 0.9 ** (label_elapsed_seconds / s_space), dim=-1
        )

    def interp(self, out_ahead_logits, label_elapsed_seconds):
        label_elapsed_seconds = torch.clamp(label_elapsed_seconds.contiguous(), min=1)
        point_space_raw = torch.exp(
            torch.linspace(
                0, self.point_spread, self.num_points, device=out_ahead_logits.device
            )
        )
        point_space = 0.5 + (point_space_raw - 1) * (
            np.e ** (self.max_e - self.point_spread)
        )
        right_idx = torch.searchsorted(point_space, label_elapsed_seconds)
        left_idx = torch.clamp(right_idx - 1, min=0)
        xl, xr = point_space[left_idx], point_space[right_idx]
        yl = torch.gather(out_ahead_logits, dim=-1, index=left_idx)
        yr = torch.gather(out_ahead_logits, dim=-1, index=right_idx)
        res = 1e-5 + (1 - 2 * 1e-5) * (
            yl + (yr - yl) * (label_elapsed_seconds - xl) / (xr - xl)
        )
        return res.squeeze(-1)

    def review(
        self,
        card_features,
        card_state,
        note_state,
        deck_state,
        preset_state,
        global_state,
    ):
        assert len(card_features.shape) == 2
        card_features = torch.nn.functional.pad(
            card_features,
            (0, self.card_features_dim - card_features.size(1)),
            mode="constant",
            value=0,
        )  # TODO change

        card_rwkv_input = self.features2card(card_features)
        card_encoding, next_card_state = self.rwkv_modules[0].run(
            card_rwkv_input, card_state
        )
        deck_encoding, next_deck_state = self.rwkv_modules[1].run(
            card_encoding, deck_state
        )
        note_encoding, next_note_state = self.rwkv_modules[2].run(
            deck_encoding, note_state
        )
        preset_encoding, next_preset_state = self.rwkv_modules[3].run(
            note_encoding, preset_state
        )
        global_encoding, next_global_state = self.rwkv_modules[4].run(
            preset_encoding, global_state
        )

        x = self.prehead_dropout(self.prehead_norm(global_encoding))
        out_w_logits = self.w_linear(self.head_w(x).float())
        out_w = torch.nn.functional.softmax(out_w_logits, dim=-1)
        out_ahead_logits = self.ahead_linear(self.head_ahead_logits(x).float())

        x_p = self.head_p(x).float()
        out_p_logits = self.p_linear(x_p)
        return (
            out_ahead_logits,
            out_w,
            out_p_logits,
            next_card_state,
            next_note_state,
            next_deck_state,
            next_preset_state,
            next_global_state,
        )

    @torch.inference_mode()
    def run(self, df, dtype, device):
        print(
            "TODO: properly do id encode and time encode, it is just padded right now"
        )

        df = df.reset_index(drop=True)
        card_states = {}
        note_states = {}
        deck_states = {}
        preset_states = {}
        global_state = None
        ahead_ps = {}
        imm_ps = {}
        card_features_df = df[CARD_FEATURE_COLUMNS]
        card_features_all = torch.tensor(
            card_features_df.to_numpy(), dtype=dtype, device=device, requires_grad=False
        ).unsqueeze(0)
        label_elapsed_seconds_all = (
            torch.tensor(df["label_elapsed_seconds"], dtype=dtype, device=device)
            .to(torch.float32)
            .unsqueeze(0)
        )

        with torch.inference_mode():
            for i, row in df.iterrows():
                if i % 100 == 0:
                    print(i)
                card_id = row["card_id"]
                note_id = row["note_id"]
                deck_id = row["deck_id"]
                preset_id = row["preset_id"]

                if card_id not in card_states:
                    card_states[card_id] = None
                if note_id not in note_states:
                    note_states[note_id] = None
                if deck_id not in deck_states:
                    deck_states[deck_id] = None
                if preset_id not in preset_states:
                    preset_states[preset_id] = None

                card_features = card_features_all[:, i]
                (
                    out_ahead_logits,
                    out_w,
                    out_p_logits,
                    next_card_state,
                    next_note_state,
                    next_deck_state,
                    next_preset_state,
                    next_global_state,
                ) = self.review(
                    card_features,
                    card_states[card_id],
                    note_states[note_id],
                    deck_states[deck_id],
                    preset_states[preset_id],
                    global_state,
                )

                if not row["skip"]:
                    card_states[card_id] = next_card_state
                    note_states[note_id] = next_note_state
                    deck_states[deck_id] = next_deck_state
                    preset_states[preset_id] = next_preset_state
                    global_state = next_global_state

                curve_probs_raw = self.forgetting_curve(
                    out_w, label_elapsed_seconds_all[:, i].unsqueeze(0)
                )
                curve_logits_raw = torch.log(
                    curve_probs_raw / (1 - curve_probs_raw)
                )  # inverse sigmoid
                ahead_logit_residual = self.interp(
                    out_ahead_logits, label_elapsed_seconds_all[:, i].unsqueeze(0)
                )
                curve_logits = curve_logits_raw + ahead_logit_residual
                curve_p = torch.sigmoid(curve_logits)

                if row["has_label"]:
                    if row["is_query"]:
                        out_p_probs = torch.softmax(out_p_logits, dim=-1)
                        out_p_again, out_p_1, out_p_2, out_p_3 = out_p_probs.unbind(
                            dim=-1
                        )
                        out_p_binary = torch.clamp(
                            1.0 - out_p_again, min=1e-5, max=1.0 - 1e-5
                        )
                        imm_ps[row["label_review_th"]] = out_p_binary.item()
                    else:
                        ahead_ps[row["label_review_th"]] = curve_p.item()

        return ahead_ps, imm_ps

    def copy_downcast_(self, master_model, dtype):
        master_params = dict(master_model.named_parameters())
        with torch.no_grad():
            for name, param in self.named_parameters():
                # print(name, is_excluded(name))
                target_dtype = torch.float32 if is_excluded(name) else dtype
                assert param.dtype == target_dtype
                param.data.copy_(master_params[name].to(target_dtype))
                # print(param)
                assert param.dtype == target_dtype

    def selective_cast(self, dtype):
        for name, module in self.named_modules():
            if len(name) == 0:
                # Skip the root module
                continue
            if not is_excluded(name):
                if dtype == torch.bfloat16:
                    module = module.to(dtype)
                elif dtype == torch.half:
                    raise ValueError("not tested.")
                elif dtype == torch.float32:
                    pass
        return self


def get_df_sequentially(data_path, user_id, max_review_th=None):
    """Line by line process the data. Written to avoid data leakage."""
    df = pd.read_parquet(data_path / "revlogs", filters=[("user_id", "=", user_id)])
    df["review_th"] = range(1, df.shape[0] + 1)
    df_cards = pd.read_parquet(data_path / "cards", filters=[("user_id", "=", user_id)])
    df_cards.drop(columns=["user_id"], inplace=True)
    df_decks = pd.read_parquet(data_path / "decks", filters=[("user_id", "=", user_id)])
    df_decks.drop(columns=["user_id", "parent_id"], inplace=True)
    df = df.merge(df_cards, on="card_id", how="left", validate="many_to_one")
    df = df.merge(df_decks, on="deck_id", how="left", validate="many_to_one")

    card_ids = []
    note_ids = []
    deck_ids = []
    preset_ids = []
    user_ids = []
    note_id_is_nans = []
    note_id_is_nans = []
    deck_id_is_nans = []
    preset_id_is_nans = []
    scaled_elapsed_days_list = []
    scaled_elapsed_seconds_list = []
    scaled_durations_list = []
    rating_1s = []
    rating_2s = []
    rating_3s = []
    rating_4s = []
    day_offset_diffs = []
    day_of_weeks = []
    diff_new_cards_list = []
    diff_reviews_list = []
    cum_new_cards_todays = []
    cum_reviews_todays = []
    scaled_states = []
    is_querys = []
    skips = []

    # Gather the elapsed_seconds and the review_th of the next review separately.
    # For these we explicitly send information back in time
    label_elapsed_seconds = []
    label_review_ths = []
    has_label = []

    all_lists = [
        ("card_id", card_ids),
        ("note_id", note_ids),
        ("deck_id", deck_ids),
        ("preset_id", preset_ids),
        ("note_id_is_nan", note_id_is_nans),
        ("deck_id_is_nan", deck_id_is_nans),
        ("preset_id_is_nan", preset_id_is_nans),
        ("scaled_elapsed_days", scaled_elapsed_days_list),
        ("scaled_elapsed_seconds", scaled_elapsed_seconds_list),
        ("scaled_duration", scaled_durations_list),
        ("rating_1", rating_1s),
        ("rating_2", rating_2s),
        ("rating_3", rating_3s),
        ("rating_4", rating_4s),
        ("day_offset_diff", day_offset_diffs),
        ("day_of_week", day_of_weeks),
        ("diff_new_cards", diff_new_cards_list),
        ("diff_reviews", diff_reviews_list),
        ("cum_new_cards_today", cum_new_cards_todays),
        ("cum_reviews_today", cum_reviews_todays),
        ("scaled_state", scaled_states),
        ("is_query", is_querys),
        ("skip", skips),
        ("label_elapsed_seconds", label_elapsed_seconds),
        ("label_review_th", label_review_ths),
        ("has_label", has_label),
    ]

    card_set = set()
    last_new_cards = {}
    last_i = {}
    last_result_index = {}
    today_new_cards = 0
    today_reviews = 0
    today = -1

    def add_same(i, prev_row, row):
        card_id = row["card_id"]
        card_ids.append(row["card_id"])

        def add(x, values, nan_list):
            if pd.isna(x):
                values.append(ID_PLACEHOLDER)
                nan_list.append(1.0)
            else:
                values.append(x)
                nan_list.append(0.0)

        add(row["note_id"], note_ids, note_id_is_nans)
        add(row["deck_id"], deck_ids, deck_id_is_nans)
        add(row["preset_id"], preset_ids, preset_id_is_nans)
        user_ids.append(row["user_id"])

        day_offset_diffs.append(
            scale_day_offset_diff(
                row["day_offset"] - (0 if prev_row is None else prev_row["day_offset"])
            )
        )
        day_of_weeks.append(((row["day_offset"] % 7) - 3) / 3)
        diff_new_cards = (
            (len(card_set) - last_new_cards[card_id])
            if card_id in last_new_cards
            else 0
        )
        diff_new_cards_list.append(scale_diff_new_cards(diff_new_cards))
        diff_reviews = (max(0, i - last_i[card_id] - 1)) if card_id in last_i else 0
        diff_reviews_list.append(scale_diff_reviews(diff_reviews))

        cum_reviews_todays.append(scale_cum_reviews_today(today_reviews))
        cum_new_cards_todays.append(scale_cum_new_cards_today(today_new_cards))

        scaled_elapsed_days_list.append(scale_elapsed_days(row["elapsed_days"]).item())
        scaled_elapsed_seconds_list.append(
            scale_elapsed_seconds(row["elapsed_seconds"]).item()
        )

    def add_query(i, prev_row, row):
        """
        We see a review `row`. We add the information to allow us to predict the outcome for this row.
        Particularly we include attributes such as card_id, note_id, etc, but we exclude attributes such as rating, duration.
        """
        add_same(i, prev_row, row)
        is_querys.append(1.0)
        skips.append(True)

        # Values to hide
        scaled_durations_list.append(0)
        scaled_states.append(0)
        rating_1s.append(0)
        rating_2s.append(0)
        rating_3s.append(0)
        rating_4s.append(0)

    def add_predict(i, prev_row, row):
        """
        `row` is a review that just completed. We want to predict the outcome for this row.
        """
        add_same(i, prev_row, row)
        is_querys.append(0.0)
        skips.append(False)

        scaled_durations_list.append(scale_duration(row["duration"]))
        scaled_states.append(scale_state(row["state"]))
        rating_1s.append(1.0 if row["rating"] == 1 else 0.0)
        rating_2s.append(1.0 if row["rating"] == 2 else 0.0)
        rating_3s.append(1.0 if row["rating"] == 3 else 0.0)
        rating_4s.append(1.0 if row["rating"] == 4 else 0.0)

    prev_row = None
    for i, row in df.iterrows():
        if max_review_th is not None and row["review_th"] > max_review_th:
            break

        card_id = row["card_id"]
        index = len(card_ids)
        label_elapsed_seconds.append(np.pi)
        label_review_ths.append(np.nan)
        has_label.append(False)

        if row["day_offset"] != today:
            today = row["day_offset"]
            today_new_cards = 0
            today_reviews = -1

        today_reviews += 1
        if card_id not in card_set:
            today_new_cards += 1

        if card_id in card_set:
            # gather the labels
            label_elapsed_seconds.append(np.pi)
            label_review_ths.append(np.nan)
            has_label.append(False)

            label_elapsed_seconds[last_result_index[card_id]] = row["elapsed_seconds"]
            label_elapsed_seconds[index] = 0
            label_review_ths[last_result_index[card_id]] = row["review_th"]
            label_review_ths[index] = row["review_th"]
            has_label[last_result_index[card_id]] = True
            has_label[index] = True

            # Add a query row
            add_query(i, prev_row, row)
            last_result_index[card_id] = index + 1
        else:
            card_set.add(card_id)
            last_result_index[card_id] = index

        add_predict(i, prev_row, row)
        prev_row = row
        last_new_cards[card_id] = len(card_set)
        last_i[card_id] = i

    for name, x in all_lists:
        assert len(x) == len(
            card_ids
        ), f"{name} does not have the right number of values"

    return pd.DataFrame({name: values for name, values in all_lists})


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

    def run(self, row, skip):
        features = torch.tensor(
            row.loc[CARD_FEATURE_COLUMNS].tolist(),
            dtype=self.dtype,
            device=self.device,
            requires_grad=False,
        ).unsqueeze(0)
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

        # If this card has been seen before, get the predicted retention from the stored function
        card_id = row["card_id"]
        review_th = row["review_th"]
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
        "label_filter_db",
        int(7e9),
        1,
    )
    pass
