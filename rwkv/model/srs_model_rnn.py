import numpy as np
from rwkv.architecture import AnkiRWKVConfig
from rwkv.data_processing import (
    CARD_FEATURE_COLUMNS,
)
from rwkv.model.rwkv_rnn_model import RWKV7RNN
from rwkv.model.srs_model import is_excluded
import torch

# An RNN implementation of srs_model.


def __nop(ob):
    return ob


ModuleType = torch.nn.Module
FunctionType = __nop

# ModuleType = torch.jit.ScriptModule
# FunctionType = torch.jit.script_method


class SrsRWKVRnn(ModuleType):
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
                target_dtype = torch.float32 if is_excluded(name) else dtype
                assert param.dtype == target_dtype
                param.data.copy_(master_params[name].to(target_dtype))
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
