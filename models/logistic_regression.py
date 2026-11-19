import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import Config
from models.base import BaseModel


def transform_elapsed_days_real_np(x):
    return (np.log(x + 1e-5) + 1.3) / 5


def create_features(df: pd.DataFrame):
    """
    1. log(t)
    2. [Rating == 2] × log(t)
    3. [Rating == 3] × log(t)
    4. [Rating == 4] × log(t)
    5. (is_hard AND is_first_review) × log(t)
    6. (better_than_hard AND is_first_review) × log(t)
    7. log(1 + cumulative same-day fails) × log(t)
    8. log(1 + cumulative non-same-day passes * not same-day t) × log(t)
    9. log(1 + cumulative non-same-day fails * not same-day t) × log(t)
    10. log(1 + cumulative same-day passes) × log(t)
    11. 1 (constant)
    12. Rating == 2
    13. Rating == 3
    14. Rating == 4
    15. First rating == 2
    16. First rating == 3
    17. First rating == 4
    18. First rating == 2 AND is_first_review
    19. First rating == 3 AND is_first_review
    20. First rating == 4 AND is_first_review
    21. log(1 + cumulative same-day fails)
    22. log(1 + cumulative non-same-day passes * not same-day t)
    23. log(previous elapsed time)
    24. log(1 + cumulative non-same-day fails)
    25. log(1 + cumulative non-same-day passes)
    26. log(1 + cumulative same-day passes * same-day t)
    27. Has ever passed (binary)
    28. log(time since first review or last lapse)
    29. t is same-day (binary)
    30. log(cumulative time)
    31. [first_rating > 1] * log(1 + cumulative same-day fails)
    32. [first_rating > 1] * log(1 + cumulative non-same-day fails)
    33. [first_rating > 1] * log(1 + cumulative same-day count)
    34. [first_rating > 1] * log(1 + cumulative non-same-day count)
    """
    n = len(df)
    cols = [f"feat_{i}" for i in range(34)]
    if n == 0:
        return pd.DataFrame(
            np.empty((0, 34), dtype=np.float32), index=df.index, columns=cols
        )

    idx_arr = np.arange(n).astype(np.int64)
    card_ids = df["card_id"].to_numpy()
    order = np.argsort(card_ids, kind="stable")
    inv_order = np.empty(n, dtype=np.int64)
    inv_order[order] = idx_arr

    card_s = card_ids[order]
    is_start = np.empty(n, dtype=bool)
    is_start[0] = True
    is_start[1:] = card_s[1:] != card_s[:-1]
    start_idx = np.maximum.accumulate(np.where(is_start, idx_arr, 0))

    dt_secs_raw = df["delta_t_secs"].fillna(0).to_numpy(dtype=np.float64)
    dt_int_raw = df["delta_t_int"].fillna(0).to_numpy(dtype=np.float64)
    r_raw = df["feature_rating"].to_numpy(dtype=np.float64)

    dt_secs_s = dt_secs_raw[order]
    dt_int_s = dt_int_raw[order]
    r_s = r_raw[order]

    feat_elapsed_real_s = np.zeros(n, dtype=np.float64)
    feat_elapsed_real_s[1:] = np.where(is_start[1:], 0.0, dt_secs_s[:-1])

    feat_elapsed_int_s = np.zeros(n, dtype=np.float64)
    feat_elapsed_int_s[1:] = np.where(is_start[1:], 0.0, dt_int_s[:-1])

    same_day_s = (feat_elapsed_int_s == 0).astype(np.float32)
    non_same_day_s = 1.0 - same_day_s
    success_s = (r_s > 1).astype(np.float32)
    fail_s = (r_s == 1).astype(np.float32)
    is_first_s = is_start.astype(np.float32)
    not_first_s = 1.0 - is_first_s
    first_r_s = r_s[start_idx].astype(np.float32)

    def _seg_cumsum_i64(arr_01: np.ndarray) -> np.ndarray:
        cs = np.cumsum(arr_01.astype(np.int64), dtype=np.int64)
        offset = np.where(start_idx > 0, cs[start_idx - 1], 0)
        return (cs - offset).astype(np.float32)

    num_same_day_fail_s = _seg_cumsum_i64(same_day_s * not_first_s * fail_s)
    num_nsd_fail_s = _seg_cumsum_i64(non_same_day_s * not_first_s * fail_s)
    num_same_day_pass_s = _seg_cumsum_i64(same_day_s * not_first_s * success_s)
    num_nsd_pass_s = _seg_cumsum_i64(non_same_day_s * not_first_s * success_s)
    num_pass_s = _seg_cumsum_i64(success_s)
    num_same_day_s = _seg_cumsum_i64(same_day_s)
    num_non_same_day_s = _seg_cumsum_i64(non_same_day_s)

    # Per-card cumulative sum in float64 before float32 cast (avoids cross-card cancellation)
    changes = np.flatnonzero(is_start[1:]) + 1
    c_starts = np.empty(len(changes) + 1, dtype=np.int64)
    c_ends = np.empty(len(changes) + 1, dtype=np.int64)
    c_starts[0] = 0
    c_starts[1:] = changes
    c_ends[:-1] = changes
    c_ends[-1] = n
    times_f64 = np.empty(n, dtype=np.float64)
    for cs_i, ce_i in zip(c_starts.tolist(), c_ends.tolist()):
        np.cumsum(feat_elapsed_real_s[cs_i:ce_i], out=times_f64[cs_i:ce_i])
    times_s = times_f64.astype(np.float32)

    # Since times_s >= 0 is monotonically non-decreasing within each card and is_start
    # always sets fl_mask_s True, group cummax(where(fl_mask, times, 0)) equals times at latest fl_mask.
    fl_mask_s = is_start | (success_s == 0)
    last_fl_idx = np.maximum.accumulate(np.where(fl_mask_s, idx_arr, 0))
    running_max_flt_s = times_s[last_fl_idx]
    time_since_first_or_lapse_s = times_s - running_max_flt_s

    # Restore original review_th order
    r = r_raw
    feat_elapsed_real = feat_elapsed_real_s[inv_order].astype(np.float32)
    is_first = is_first_s[inv_order]
    first_r = first_r_s[inv_order]
    num_same_day_fail = num_same_day_fail_s[inv_order]
    num_nsd_fail = num_nsd_fail_s[inv_order]
    num_same_day_pass = num_same_day_pass_s[inv_order]
    num_nsd_pass = num_nsd_pass_s[inv_order]
    has_passed = (num_pass_s[inv_order] > 0).astype(np.float32)
    num_same_day = num_same_day_s[inv_order]
    num_non_same_day = num_non_same_day_s[inv_order]
    times = times_s[inv_order]
    time_since_first_or_lapse = time_since_first_or_lapse_s[inv_order]

    # --- scalars ---
    is_hard = (r == 2).astype(np.float32)
    better_than_hard = (r > 2).astype(np.float32)
    label_is_same_day = (dt_int_raw == 0).astype(np.float32)

    # --- one-hot for current rating (classes 0,1,2 = ratings 2,3,4) ---
    row_arange = np.arange(n)
    r_clipped = np.clip(r.astype(np.int32) - 2, 0, 2)
    rating_onehot = np.zeros((n, 3), dtype=np.float32)
    rating_onehot[row_arange, r_clipped] = 1.0
    rating_onehot *= (r > 1).astype(np.float32)[:, None]

    # --- first_rating per group ---
    first_r_clipped = np.clip(first_r.astype(np.int32) - 2, 0, 2)
    first_rating_onehot = np.zeros((n, 3), dtype=np.float32)
    first_rating_onehot[row_arange, first_r_clipped] = 1.0
    first_rating_onehot *= (first_r > 1).astype(np.float32)[:, None]

    # --- transformed scalars ---
    t_elapsed_real = transform_elapsed_days_real_np(feat_elapsed_real)
    t_label_real = transform_elapsed_days_real_np(dt_secs_raw.astype(np.float32))
    t_time_since_lapse = transform_elapsed_days_real_np(time_since_first_or_lapse)
    t_times = transform_elapsed_days_real_np(times)
    first_r_gt1 = (first_r > 1).astype(np.float32)

    # --- deg_1 block ---
    ones = np.ones(n, dtype=np.float32)
    not_label_sd = 1.0 - label_is_same_day
    log_sd_fail = np.log1p(num_same_day_fail)
    log_nsd_pass_nsd = np.log1p(num_nsd_pass * not_label_sd)
    log_nsd_fail = np.log1p(num_nsd_fail)

    deg1 = np.stack(
        [
            ones,  # bias
            rating_onehot[:, 0],
            rating_onehot[:, 1],
            rating_onehot[:, 2],
            is_hard * is_first,
            better_than_hard * is_first,
            log_sd_fail,
            log_nsd_pass_nsd,
            np.log1p(num_nsd_fail * not_label_sd),
            np.log1p(num_same_day_pass),
        ],
        axis=1,
    )

    # --- deg_0 block ---
    deg0 = np.stack(
        [
            ones,
            rating_onehot[:, 0],
            rating_onehot[:, 1],
            rating_onehot[:, 2],
            first_rating_onehot[:, 0],
            first_rating_onehot[:, 1],
            first_rating_onehot[:, 2],
            first_rating_onehot[:, 0] * is_first,
            first_rating_onehot[:, 1] * is_first,
            first_rating_onehot[:, 2] * is_first,
            log_sd_fail,
            log_nsd_pass_nsd,
            t_elapsed_real,
            log_nsd_fail,
            np.log1p(num_nsd_pass),
            np.log1p(num_same_day_pass * label_is_same_day),
            has_passed,
            t_time_since_lapse,
            label_is_same_day,
            t_times,
            first_r_gt1 * log_sd_fail,
            first_r_gt1 * log_nsd_fail,
            first_r_gt1 * np.log1p(num_same_day),
            first_r_gt1 * np.log1p(num_non_same_day),
        ],
        axis=1,
    )

    # --- final y_lH = [deg1 * v, deg0] ---
    v = t_label_real[:, None]
    y = np.concatenate([deg1 * v, deg0], axis=1)

    return pd.DataFrame(y, index=df.index, columns=cols)


class LogisticRegression(BaseModel):
    n_epoch = 10
    batch_size = 2**11
    lr: float = 2e-1
    betas: tuple = (0.0, 0.85)
    adam_eps = 1e-8
    wd: float = 3e-1

    def __init__(self, config, state_dict=None):
        super().__init__(config)
        mean = torch.tensor(
            [
                -0.9012,
                -0.6941,
                -0.6258,
                -0.4678,
                0.3926,
                0.1316,
                0.3508,
                -0.3259,
                -0.2125,
                -0.1402,
                1.2775,
                0.0669,
                0.3914,
                0.5828,
                0.9292,
                1.1583,
                1.4103,
                0.9388,
                0.9164,
                0.9797,
                -0.6411,
                0.4951,
                0.1038,
                -0.1564,
                0.4332,
                0.3287,
                0.3429,
                0.4068,
                -0.1779,
                -0.0585,
                -0.1062,
                -0.3521,
                -0.1613,
                -0.1758,
            ]
        )
        self.register_buffer("mean", mean)
        std = torch.tensor(
            [
                0.2869,
                0.2811,
                0.2910,
                0.2166,
                0.1740,
                0.1612,
                0.1428,
                0.1352,
                0.1284,
                0.0331,
                0.4013,
                0.2065,
                0.2022,
                0.1670,
                0.2454,
                0.2954,
                0.2896,
                0.3473,
                0.5685,
                0.2795,
                0.1298,
                0.0467,
                0.1200,
                0.1317,
                0.0785,
                0.1190,
                0.0854,
                0.1634,
                0.1892,
                0.1052,
                0.0750,
                0.1913,
                0.1372,
                0.0783,
            ]
        )
        self.register_buffer("std", std)
        self.n_features = mean.size(0)
        self.coef_res = nn.Parameter(torch.zeros(self.n_features))
        if state_dict is not None:
            self.load_state_dict(state_dict)

    @property
    def coefficients(self):
        return self.coef_res * self.std + self.mean  # type: ignore

    def optimize(self, df: pd.DataFrame):
        xrange = np.linspace(0, 1, len(df))
        df["weights"] = 0.1 + 0.9 * np.power(xrange, 4)
        x_all = df.loc[:, df.columns.str.startswith("feat_")]
        # pyrefly: ignore [missing-attribute]
        x_all = torch.tensor(np.array(x_all), dtype=torch.float)
        # pyrefly: ignore [missing-attribute]
        y_all = torch.tensor(np.array(df["y"]), dtype=torch.float)
        # pyrefly: ignore [missing-attribute]
        weights_all = torch.tensor(np.array(df["weights"]), dtype=torch.float)
        B = x_all.size(0)

        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.wd,
            betas=self.betas,
            eps=self.adam_eps,
            fused=True,
        )
        steps_per_epoch = (B + self.batch_size - 1) // self.batch_size
        total_steps = self.n_epoch * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            # pyrefly: ignore [bad-argument-type]
            optimizer,
            T_max=total_steps,
            eta_min=0,
        )

        for _ in range(self.n_epoch):
            # pyrefly: ignore [missing-attribute]
            perm = torch.randperm(B)
            for i in range(0, B, self.batch_size):
                idx = perm[i : i + self.batch_size]

                x = x_all[idx]
                y = y_all[idx]
                weights = weights_all[idx]
                logits_bl = torch.mv(x, self.coefficients)
                loss = F.binary_cross_entropy_with_logits(
                    logits_bl, y, reduction="none"
                )
                optimizer.zero_grad()
                (loss * weights).sum().backward()
                optimizer.step()
                scheduler.step()
        return self.state_dict()

    # pyrefly: ignore [missing-attribute]
    @torch.inference_mode()
    def predict(self, df: pd.DataFrame):
        df = df.copy()
        x = df.loc[:, df.columns.str.startswith("feat_")]
        # pyrefly: ignore [missing-attribute]
        x = torch.tensor(np.array(x), dtype=torch.float)
        logits_bl = torch.mv(x, self.coefficients)
        return torch.sigmoid(logits_bl)

    def log(self, x):
        params = [round(x, 4) for x in self.coefficients.tolist()]
        x["parameters"] = params
        x["state"] = [round(x, 4) for x in self.coef_res.tolist()]
