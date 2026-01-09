from dataclasses import dataclass
import math

import numpy as np
from rwkv.config import RWKV_SUBMODULES
from rwkv.data_processing import RWKVSample
from rwkv.model.rwkv_model import RWKV7
import torch
from typing import NamedTuple

from rwkv.architecture import AnkiRWKVConfig


# def __nop(ob):
#     return ob


# ModuleType = torch.nn.Module
# FunctionType = __nop

ModuleType = torch.jit.ScriptModule
FunctionType = torch.jit.script_method


class SrsRWKVIterStatistics(NamedTuple):
    average_loss: torch.Tensor
    loss_tensor: torch.Tensor
    w_loss_avg: torch.Tensor
    ahead_logits_mag_loss_avg: torch.Tensor
    ahead_logits_diff_loss_avg: torch.Tensor
    ahead_avg: torch.Tensor
    ahead_raw_avg: torch.Tensor
    ahead_n: int
    ahead_equalize_avg: torch.Tensor
    ahead_raw_equalize_avg: torch.Tensor
    ahead_equalize_n: int
    imm_avg: torch.Tensor
    imm_n: int
    imm_binary_equalize_avg: torch.Tensor
    imm_binary_equalize_n: int
    p_curve: torch.Tensor
    p_imm: torch.Tensor
    p_imm_all: torch.Tensor
    w: torch.Tensor
    label_rating: torch.Tensor
    label_elapsed_seconds: torch.Tensor
    label_review_th: torch.Tensor
    is_query: torch.Tensor
    has_label: torch.Tensor


@dataclass
class PreparedBatch:
    num_data: int
    start: torch.Tensor
    sub_gather: list[list[torch.Tensor]]
    sub_gather_lens: list[list[int]]
    time_shift_selects: list[list[torch.Tensor]]
    skips: list[list[torch.Tensor]]
    labels: torch.Tensor
    label_review_th: torch.Tensor

    def to(self, device):
        start = self.start.to(device)
        sub_gather = [[x.to(device) for x in sub] for sub in self.sub_gather]
        time_shift_selects = [
            [x.to(device) for x in sub] for sub in self.time_shift_selects
        ]
        skips = [[x.to(device) for x in sub] for sub in self.skips]
        labels = self.labels.to(device)
        label_review_th = self.label_review_th.to(device)
        return PreparedBatch(
            num_data=self.num_data,
            start=start,
            sub_gather=sub_gather,
            sub_gather_lens=self.sub_gather_lens,
            time_shift_selects=time_shift_selects,
            skips=skips,
            labels=labels,
            label_review_th=label_review_th,
        )


DTYPE_EXCLUDE = [
    "w_linear",
    "s_linear",
    "d_linear",
    "d_softplus",
    "k_linear",
    "p_linear",
    "ahead_linear",
]


def is_excluded(name):
    for query in DTYPE_EXCLUDE:
        if query in name:
            return True
    return False


class SrsRWKV(ModuleType):
    def __init__(self, anki_rwkv_config: AnkiRWKVConfig):
        super().__init__()

        self.card_features_dim = 92
        self.d_model = anki_rwkv_config.d_model
        self.features_fc_dim = 4 * self.d_model
        self.ahead_head_dim = 4 * self.d_model
        self.p_head_dim = 4 * self.d_model
        self.w_head_dim = 4 * self.d_model
        self.num_curves = 128

        with torch.no_grad():
            self.features2card = torch.nn.Sequential(
                torch.nn.Linear(self.card_features_dim, self.features_fc_dim),
                torch.nn.SiLU(),
                torch.nn.LayerNorm(self.features_fc_dim),
                torch.nn.Linear(self.features_fc_dim, self.d_model),
                torch.nn.SiLU(),
            )
            self.rwkv_modules = torch.nn.ModuleList(
                [RWKV7(config=config) for _, config in anki_rwkv_config.modules]
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
            torch.nn.init.zeros_(self.ahead_linear.weight)
            torch.nn.init.zeros_(self.ahead_linear.bias)

            self.w_linear = torch.nn.Linear(self.w_head_dim, self.num_curves)
            torch.nn.init.zeros_(self.w_linear.weight)
            torch.nn.init.zeros_(self.w_linear.bias)

            self.s_point_spread = 18.5
            self.s_max = 22

            self.p_linear = torch.nn.Linear(self.p_head_dim, 4)
            torch.nn.init.zeros_(self.p_linear.weight)
            self.p_linear.bias.copy_(torch.tensor([-0.3512, -0.0802, 0.4297, -0.2041]))

    @FunctionType
    def head_and_out(self, input):
        x = self.prehead_dropout(self.prehead_norm(input))

        out_w_logits = self.w_linear(self.head_w(x).float())
        out_w = torch.nn.functional.softmax(out_w_logits, dim=-1)
        out_w_log_p = torch.nn.functional.log_softmax(out_w_logits, dim=-1)
        out_ahead_logits = self.ahead_linear(self.head_ahead_logits(x).float())

        x_p = self.head_p(x).float()
        return out_ahead_logits, out_w, out_w_log_p, self.p_linear(x_p)

    @FunctionType
    def forgetting_curve(self, w, label_elapsed_seconds):
        s_space_raw = torch.exp(
            torch.linspace(0, self.s_point_spread, self.num_curves, device=w.device)
        )
        s_space = 0.1 + (s_space_raw - 1) * (np.e ** (self.s_max - self.s_point_spread))
        label_elapsed_seconds = torch.max(torch.tensor(1.0), label_elapsed_seconds)
        return 1e-5 + (1 - 2 * 1e-5) * torch.sum(
            w * 0.9 ** (label_elapsed_seconds / s_space), dim=-1
        )

    @FunctionType
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

    @FunctionType
    def forward_batch(
        self,
        batch_start: torch.Tensor,
        batch_sub_gather: list[list[torch.Tensor]],
        batch_sub_gather_lens: list[list[int]],
        batch_time_shift_selects: list[list[torch.Tensor]],
        batch_skips: list[list[torch.Tensor]],
        batch_num_data: int,
    ):
        x = self.features2card(batch_start)

        assert len(batch_sub_gather) == len(self.rwkv_modules)
        for i, submodule in enumerate(self.rwkv_modules):
            module_splits = batch_sub_gather[i]
            sub_lens = batch_sub_gather_lens[i]
            time_shift_selects = batch_time_shift_selects[i]
            skips = batch_skips[i]
            y = []
            for split_gather, sub_len, time_shift_select, skip in zip(
                module_splits, sub_lens, time_shift_selects, skips
            ):
                module_in = torch.index_select(
                    x, dim=0, index=torch.clamp(split_gather, min=0)
                ).view(-1, sub_len, self.d_model)
                time_shift_select_BT = time_shift_select.view(-1, sub_len)
                skip_BT = skip.view(-1, sub_len)
                assert module_in.size(0) == time_shift_select_BT.size(
                    0
                ) and module_in.size(0) == skip_BT.size(0)
                module_out = submodule(
                    module_in,
                    time_shift_select_BT=time_shift_select_BT,
                    skip_BT=skip_BT,
                )
                y.append(module_out.view(-1, self.d_model))

            x = torch.cat(y)

        x = x.view(batch_num_data, -1, self.d_model)
        return self.head_and_out(x)

    @FunctionType
    def nanmin(self, tensor):
        output = tensor.nan_to_num(1e9).min()
        return output

    @FunctionType
    def nanmax(self, tensor):
        output = tensor.nan_to_num(-1e9).max()
        return output

    @FunctionType
    def _get_loss(
        self,
        batch_start: torch.Tensor,
        batch_sub_gather: list[list[torch.Tensor]],
        batch_sub_gather_lens: list[list[int]],
        batch_time_shift_selects: list[list[torch.Tensor]],
        batch_skips: list[list[torch.Tensor]],
        batch_num_data: int,
        batch_labels: torch.Tensor,
        batch_label_review_th: torch.Tensor,
    ):
        out_ahead_logits, out_w, out_w_log_p, out_p_logits = self.forward_batch(
            batch_start,
            batch_sub_gather,
            batch_sub_gather_lens,
            batch_time_shift_selects,
            batch_skips,
            batch_num_data,
        )
        if torch.isnan(out_ahead_logits).any():
            return None

        global_labels = batch_labels.float()
        (
            label_elapsed_seconds,
            _,
            label_y,
            label_rating,
            has_label,
            label_is_equalize,
            is_query,
        ) = global_labels.unbind(-1)
        has_label = has_label.int()
        label_is_equalize = label_is_equalize.int()
        is_query = is_query.int()

        label_rating = torch.clamp(label_rating - 1, min=0)
        label_elapsed_seconds = label_elapsed_seconds.unsqueeze(-1)
        curve_probs_raw = self.forgetting_curve(out_w, label_elapsed_seconds)
        curve_logits_raw = torch.log(
            curve_probs_raw / (1 - curve_probs_raw)
        )  # inverse sigmoid
        ahead_logit_residual = self.interp(out_ahead_logits, label_elapsed_seconds)
        curve_logits = curve_logits_raw + ahead_logit_residual
        curve_probs = torch.sigmoid(curve_logits)

        out_p_probs = torch.softmax(out_p_logits, dim=-1)
        out_p_again, out_p_1, out_p_2, out_p_3 = out_p_probs.unbind(dim=-1)
        out_p_binary = torch.clamp(1.0 - out_p_again, min=1e-5, max=1.0 - 1e-5)

        if torch.isnan(curve_probs).any():
            raise Exception("nan")
        w_loss = torch.nn.functional.kl_div(
            input=out_w_log_p,
            target=torch.ones_like(out_w) / self.num_curves,
            reduction="none",
        ).mean(dim=-1)
        ahead_mask = (1 - is_query) * has_label
        immediate_mask = is_query * has_label
        assert ahead_mask.shape == label_is_equalize.shape
        ahead_equalize_mask = ahead_mask * label_is_equalize

        immediate_equalize_mask = immediate_mask * label_is_equalize
        curve_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            curve_logits, label_y, reduction="none"
        )
        curve_raw_loss = torch.nn.functional.binary_cross_entropy_with_logits(
            curve_logits_raw, label_y, reduction="none"
        )
        NUM_LABELS = 4
        B, T = label_rating.shape
        p_loss = torch.nn.functional.cross_entropy(
            out_p_logits.view(-1, NUM_LABELS),
            label_rating.long().view(-1),
            reduction="none",
        ).view(B, T)
        p_binary_loss = torch.nn.functional.binary_cross_entropy(
            out_p_binary, label_y, reduction="none"
        )
        ahead_avg = (curve_loss * ahead_mask).sum() / (1e-8 + ahead_mask.sum())
        AHEAD_SCALE = 0.5
        ahead_raw_avg = (curve_raw_loss * ahead_mask).sum() / (1e-8 + ahead_mask.sum())
        AHEAD_RAW_SCALE = 0.5
        immediate_avg = (p_loss * immediate_mask).sum() / (1e-8 + immediate_mask.sum())
        w_avg = (w_loss * ahead_mask).sum() / (1e-8 + ahead_mask.sum())
        W_LOSS_SCALE = 1e-5
        ahead_logits_mag_loss = torch.sqrt(
            1e-16 + out_ahead_logits.square().mean(dim=-1)
        )
        ahead_logits_mag_avg = (ahead_logits_mag_loss * ahead_mask).sum() / (
            1e-8 + ahead_mask.sum()
        )
        AHEAD_LOGITS_MAG_LOSS_SCALE = 1e-4
        ahead_logits_diff_loss = torch.sqrt(
            1e-16 + out_ahead_logits.diff().square().mean(dim=-1)
        )
        ahead_logits_diff_avg = (ahead_logits_diff_loss * ahead_mask).sum() / (
            1e-8 + ahead_mask.sum()
        )
        AHEAD_LOGITS_DIFF_LOSS_SCALE = 1e-3
        loss_avg = (
            AHEAD_SCALE * ahead_avg
            + immediate_avg
            + AHEAD_RAW_SCALE * ahead_raw_avg
            + W_LOSS_SCALE * w_avg
            + AHEAD_LOGITS_MAG_LOSS_SCALE * ahead_logits_mag_avg
            + AHEAD_LOGITS_DIFF_LOSS_SCALE * ahead_logits_diff_avg
        )
        loss_tensor = (
            AHEAD_SCALE * curve_loss.detach()
            + p_loss.detach()
            + AHEAD_RAW_SCALE * curve_raw_loss.detach()
            + W_LOSS_SCALE * w_loss.detach()
            + AHEAD_LOGITS_MAG_LOSS_SCALE * ahead_logits_mag_loss.detach()
            + AHEAD_LOGITS_DIFF_LOSS_SCALE * ahead_logits_diff_loss.detach()
        )

        ahead_equalize_avg = (curve_loss * ahead_equalize_mask).sum() / (
            1e-8 + ahead_equalize_mask.sum()
        )
        ahead_raw_equalize_avg = (curve_raw_loss * ahead_equalize_mask).sum() / (
            1e-8 + ahead_equalize_mask.sum()
        )
        immediate_binary_equalize_avg = (
            p_binary_loss * immediate_equalize_mask
        ).sum() / (1e-8 + immediate_equalize_mask.sum())

        return SrsRWKVIterStatistics(
            average_loss=loss_avg,
            p_curve=curve_probs.detach(),
            p_imm=out_p_binary.detach(),
            p_imm_all=out_p_probs.detach(),
            loss_tensor=loss_tensor.detach(),
            ahead_avg=ahead_avg.detach(),
            ahead_raw_avg=ahead_raw_avg.detach(),
            ahead_n=ahead_mask.sum().detach(),
            ahead_equalize_avg=ahead_equalize_avg.detach(),
            ahead_raw_equalize_avg=ahead_raw_equalize_avg.detach(),
            ahead_equalize_n=ahead_equalize_mask.sum().detach(),
            imm_avg=immediate_avg.detach(),
            imm_n=immediate_mask.sum().detach(),
            imm_binary_equalize_avg=immediate_binary_equalize_avg.detach(),
            imm_binary_equalize_n=immediate_equalize_mask.sum().detach(),
            w_loss_avg=w_avg.detach(),
            ahead_logits_mag_loss_avg=ahead_logits_mag_avg.detach(),
            ahead_logits_diff_loss_avg=ahead_logits_diff_avg.detach(),
            w=out_w.detach(),
            label_review_th=batch_label_review_th.detach(),
            label_elapsed_seconds=label_elapsed_seconds.detach(),
            label_rating=label_rating.detach(),
            is_query=is_query.detach(),
            has_label=has_label.detach(),
        )

    def get_loss(self, batch: PreparedBatch):
        return self._get_loss(
            batch.start,
            batch.sub_gather,
            batch.sub_gather_lens,
            batch.time_shift_selects,
            batch.skips,
            batch.num_data,
            batch.labels,
            batch.label_review_th,
        )

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


@dataclass
class AnkiRWKVDictStatistics:
    ahead_ps: dict[int, float]
    imm_ps: dict[int, float]
    imm_ps_all: dict
    label_ratings: dict[int, float]
    label_elapsed_seconds: dict[int, float]
    w: dict


def extract_p(stats: SrsRWKVIterStatistics):
    """Creates a nicer summary"""
    assert stats.label_review_th.size(0) == 1  # Only allow batch sizes of 1
    ahead_ps_dict = {}
    imm_ps_dict = {}
    label_ratings_dict = {}
    label_elapsed_seconds_dict = {}
    imm_ps_all_dict = {}

    label_review_ths = stats.label_review_th.squeeze(0).cpu().numpy()
    label_elapsed_seconds_list = stats.label_elapsed_seconds.squeeze(0).cpu().numpy()
    label_ratings_list = stats.label_rating.squeeze(0).cpu().numpy()
    has_labels = stats.has_label.squeeze(0).cpu().numpy()
    is_querys = stats.is_query.squeeze(0).cpu().numpy()
    p_curves = stats.p_curve.squeeze(0).cpu().numpy()
    p_imms = stats.p_imm.squeeze(0).cpu().numpy()
    p_imm_alls = stats.p_imm_all.squeeze(0).cpu().numpy()
    ws = stats.w.squeeze(0).cpu()

    for i in range(len(label_review_ths)):
        label_review_th = label_review_ths[i]
        label_elapsed_seconds_dict[label_review_th] = label_elapsed_seconds_list[i]
        label_rating = label_ratings_list[i]
        has_label = has_labels[i]
        is_query = is_querys[i]
        imm_p = p_imms[i]
        imm_p_all = p_imm_alls[i]
        ahead_p = p_curves[i]

        if has_label:
            label_ratings_dict[label_review_th] = label_rating
            if is_query:
                imm_ps_dict[label_review_th] = imm_p
                imm_ps_all_dict[label_review_th] = imm_p_all
            else:
                ahead_ps_dict[label_review_th] = ahead_p

    return AnkiRWKVDictStatistics(
        ahead_ps=ahead_ps_dict,
        imm_ps=imm_ps_dict,
        imm_ps_all=imm_ps_all_dict,
        label_ratings=label_ratings_dict,
        label_elapsed_seconds=label_elapsed_seconds_dict,
        w=ws,
    )


def greedy_splits(
    data_list: list[RWKVSample], factor, allowed_excess_in_one_step=20000
):
    """'factor' puts a limit on the memory complexity.
    'allowed_excess_in_one_step' captures the notion that at some point it is better to just separate the work into sequential calls
    example: if we are given [1, 1e6] then it would be worse to pad the 1 just to fit within the same batch.
    """
    splits_dict = {}
    for submodule in RWKV_SUBMODULES:
        if submodule == RWKV_SUBMODULES[-1]:
            longest = 0
            for data in data_list:
                module_data = data.modules[submodule]
                longest = max(longest, module_data.split_len.max().item())
            splits_dict[submodule] = [longest]
            continue

        freqs = {}
        for data in data_list:
            module_data = data.modules[submodule]
            for l, b in zip(module_data.split_len, module_data.split_B):
                if l not in freqs:
                    freqs[l] = 0
                freqs[l] += b

        lens = list(reversed(sorted(freqs.keys())))
        splits = []
        l = 0
        while l < len(lens):
            r = l
            used = lens[l] * freqs[lens[l]]
            waste = 0
            while r + 1 < len(lens):
                next_used = used + lens[r + 1] * freqs[lens[r + 1]]
                extra_waste = (lens[l] - lens[r + 1]) * freqs[lens[r + 1]]
                next_waste = waste + extra_waste
                if (
                    factor * next_used >= next_waste
                    and extra_waste <= allowed_excess_in_one_step
                ):
                    used = next_used
                    waste = next_waste
                    r += 1
                else:
                    break

            splits.append(lens[l])
            l = r + 1

        splits.reverse()
        splits_dict[submodule] = splits

    return splits_dict


def naive_splits(data_list: list[RWKVSample]):
    splits_dict = {}
    for submodule in RWKV_SUBMODULES:
        longest = 0
        for data in data_list:
            module_data = data.modules[submodule]
            longest = max(longest, module_data.split_len.max().item())

        print("longest", submodule, longest)
        if submodule == RWKV_SUBMODULES[-1]:
            splits_dict[submodule] = [longest]
            continue

        splits = []
        while longest > 0:
            splits.append(longest)
            longest = -1 + math.ceil(longest / 1.5)

        splits.reverse()
        splits_dict[submodule] = splits
    return splits_dict


if __name__ == "__main__":
    model = SrsRWKV()
    t_param = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Number of trainable parameters:", t_param)
    a_param = sum(p.numel() for p in model.parameters())
    print("Number of parameters", a_param)
