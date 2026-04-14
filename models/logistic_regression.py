import numpy as np
import torch
from torch import nn, Tensor

from config import Config
from models.base import BaseModel
import pandas as pd
import time

def transform_elapsed_days_real_np(x):
    return (np.log(x + 1e-5) + 1.3) / 5

def create_features(df):
    df = df.copy()
    g = df.groupby("card_id", sort=False)

    # --- shifted features (per group) ---
    df["feat_elapsed_real"] = g["delta_t_secs"].shift(1).fillna(0)
    df["feat_elapsed_int"]  = g["delta_t_int"].shift(1).fillna(0)
    r = df["feature_rating"].values

    # --- scalars ---
    same_day        = (df["feat_elapsed_int"].values == 0).astype(np.float32)
    success         = (r > 1).astype(np.float32)
    fail            = (r == 1).astype(np.float32)
    is_hard         = (r == 2).astype(np.float32)
    better_than_hard= (r > 2).astype(np.float32)
    label_int       = df["delta_t_int"].fillna(0).values
    label_real      = df["delta_t_secs"].fillna(0).values
    label_is_same_day = (label_int == 0).astype(np.float32)

    # --- is_first_review: first row in each group ---
    is_first = np.zeros(len(df), dtype=np.float32)
    is_first[g.cumcount().values == 0] = 1.0
    not_first = 1.0 - is_first

    # --- one-hot for current rating (classes 0,1,2 = ratings 2,3,4) ---
    r_clipped = np.clip(r.astype(np.int32) - 2, 0, 2)
    rating_onehot = np.zeros((len(df), 3), dtype=np.float32)
    rating_onehot[np.arange(len(df)), r_clipped] = 1.0
    rating_onehot *= (r > 1).astype(np.float32)[:, None]

    # --- first_rating per group ---
    df["_r"] = r
    first_r = g["_r"].transform("first").values.astype(np.float32)
    first_r_clipped = np.clip(first_r.astype(np.int32) - 2, 0, 2)
    first_rating_onehot = np.zeros((len(df), 3), dtype=np.float32)
    first_rating_onehot[np.arange(len(df)), first_r_clipped] = 1.0
    first_rating_onehot *= (first_r > 1).astype(np.float32)[:, None]

    # --- cumsum counts (group-aware) ---
    df["_sd_fail"]  = same_day * not_first * fail
    df["_nsd_fail"] = (1 - same_day) * not_first * fail
    df["_sd_pass"]  = same_day * not_first * success
    df["_nsd_pass"] = (1 - same_day) * not_first * success
    df["_pass"]     = success

    num_same_day_fail  = g["_sd_fail"].cumsum().values
    num_nsd_fail       = g["_nsd_fail"].cumsum().values
    num_same_day_pass  = g["_sd_pass"].cumsum().values
    num_nsd_pass       = g["_nsd_pass"].cumsum().values
    num_pass           = g["_pass"].cumsum().values
    has_passed         = (num_pass > 0).astype(np.float32)

    # --- time / time_since_first_or_lapse ---
    df["_elapsed_real"] = df["feat_elapsed_real"].values
    times = g["_elapsed_real"].cumsum().values.astype(np.float32)

    # first_or_lapse_time: equals `times` at first review or on a fail, else 0
    fl_mask = (is_first.astype(bool)) | (success == 0)
    first_or_lapse_time = np.where(fl_mask, times, 0.0)
    df["_flt"] = first_or_lapse_time
    # cummax per group
    running_max_flt = g["_flt"].cummax().values.astype(np.float32)
    time_since_first_or_lapse = times - running_max_flt

    # --- transformed scalars ---
    t_elapsed_real        = transform_elapsed_days_real_np(df["feat_elapsed_real"].values.astype(np.float32))
    t_label_real          = transform_elapsed_days_real_np(label_real.astype(np.float32))
    t_time_since_lapse    = transform_elapsed_days_real_np(time_since_first_or_lapse)

    # --- deg_1 block (8 cols) ---
    deg1 = np.stack([
        np.ones(len(df), dtype=np.float32),          # bias
        rating_onehot[:, 0],
        rating_onehot[:, 1],
        rating_onehot[:, 2],
        is_hard * is_first,
        better_than_hard * is_first,
        np.log1p(num_same_day_fail),
        np.log1p(num_nsd_pass * (1 - label_is_same_day)),
        np.log1p(num_nsd_fail * (1 - label_is_same_day)),
        np.log1p(num_same_day_pass),
    ], axis=1)

    # --- deg_0 block (13 cols) ---
    deg0 = np.stack([
        np.ones(len(df), dtype=np.float32),
        rating_onehot[:, 0],
        rating_onehot[:, 1],
        rating_onehot[:, 2],
        first_rating_onehot[:, 0],
        first_rating_onehot[:, 1],
        first_rating_onehot[:, 2],
        first_rating_onehot[:, 0] * is_first,
        first_rating_onehot[:, 1] * is_first,
        first_rating_onehot[:, 2] * is_first,
        np.log1p(num_same_day_fail),
        np.log1p(num_nsd_pass * (1 - label_is_same_day)),
        t_elapsed_real,
        np.log1p(num_nsd_fail),
        np.log1p(num_nsd_pass),
        np.log1p(num_same_day_pass * label_is_same_day),
        has_passed,
        t_time_since_lapse,
        label_is_same_day,
    ], axis=1)

    # --- final y_lH = [deg1 * v, deg0] ---
    v = t_label_real[:, None]
    y = np.concatenate([deg1 * v, deg0], axis=1)

    cols = [f"feat_{i}" for i in range(y.shape[1])]
    result = pd.DataFrame(y, index=df.index, columns=cols)

    # cleanup temp cols
    df.drop(columns=["_r","_sd_fail","_nsd_fail","_sd_pass","_nsd_pass","_pass","_elapsed_real","_flt","feat_elapsed_real","feat_elapsed_int"], inplace=True)
    return result

class LogisticRegression(BaseModel):
    n_epoch = 10
    batch_size = int(2 ** 11)
    lr: float = 2e-1
    betas: tuple = (0.8, 0.85)
    wd: float = 1e-2

    def __init__(self, config, state_dict=None):
        super().__init__(config)
        super().__init__(config)
        mean = torch.tensor([-0.9125, -0.7233, -0.5761, -0.4087,  0.2976,  0.0926,  0.3503, -0.3414,
            -0.3245, -0.2032,  1.5807,  0.0816,  0.3976,  0.5160,  0.2769,  0.5066,
            0.6300,  1.3532,  1.3429,  1.4083, -0.7107,  0.4998,  0.1423, -0.2108,
            0.2767,  0.2703,  0.4023,  0.4328, -0.2254])
        self.register_buffer('mean', mean)
        std = torch.tensor([0.2519, 0.2501, 0.2121, 0.1172, 0.1461, 0.1370, 0.1457, 0.1070, 0.0819,
            0.0736, 0.4088, 0.2184, 0.2105, 0.1901, 0.1622, 0.1910, 0.1691, 0.3706,
            0.6443, 0.3314, 0.1577, 0.0584, 0.1096, 0.2270, 0.0770, 0.0682, 0.0876,
            0.1757, 0.2187])
        self.register_buffer('std', std)
        self.n_features = mean.size(0)
        self.coef_res = nn.Parameter(torch.zeros(self.n_features))
        if state_dict is not None:
            self.load_state_dict(state_dict)

    def optimize(self, df):
        x_all = df.loc[:, df.columns.str.startswith("feat_")]
        x_all = torch.tensor(np.array(x_all), dtype=torch.float)
        y_all = torch.tensor(np.array(df["y"]), dtype=torch.float)
        weights_all = torch.tensor(np.array(df["weights"]), dtype=torch.float)
        B = x_all.size(0)

        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.wd, betas=self.betas, fused=True)
        steps_per_epoch = (B + self.batch_size - 1) // self.batch_size
        total_steps = self.n_epoch * steps_per_epoch
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_steps,
            eta_min=0,
        ) 
        for _ in range(self.n_epoch):
            perm = torch.randperm(B)
            for i in range(0, B, self.batch_size):
                idx = perm[i:i + self.batch_size]

                x = x_all[idx]
                y = y_all[idx]
                weights = weights_all[idx]
                logits_bl = torch.einsum('bh,h->b', x, self.coefficients)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    logits_bl, y, reduction='none'
                )
                optimizer.zero_grad()
                (loss * weights).sum().backward()
                optimizer.step()
                scheduler.step()
        return self.state_dict()

    @torch.inference_mode()
    def predict(self, df):
        df = df.copy()
        x = df.loc[:, df.columns.str.startswith("feat_")]
        x = torch.tensor(np.array(x), dtype=torch.float)
        logits_bl = torch.einsum('bh,h->b', x, self.coefficients)
        return torch.full_like(logits_bl, 0.9)
        # return torch.sigmoid(logits_bl)

    @property
    def coefficients(self):
        return self.coef_res * self.std + self.mean

    def log(self, x):
        params = [round(x, 4) for x in self.coefficients.tolist()]
        x["parameters"] = params
        x["state"] = [round(x, 4) for x in self.coef_res.tolist()]