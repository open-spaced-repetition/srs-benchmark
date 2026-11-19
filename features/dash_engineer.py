from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from .base import BaseFeatureEngineer, _card_run_bounds, _card_start_indices

_TAU_W = np.array([0.2434, 1.9739, 16.0090, 129.8426], dtype=np.float64)
_TIME_WINDOWS = np.array([1.0, 7.0, 30.0, np.inf], dtype=np.float64)


def _compute_dash_tw_all(df: pd.DataFrame, enable_decay: bool) -> list[torch.Tensor]:
    """Vectorized lag-accumulation across all cards for DASH and DASH[MCM]."""
    n = len(df)
    if n == 0:
        return []

    _, start_idx = _card_start_indices(df["card_id"].to_numpy())
    pos_in_card = np.arange(n).astype(np.int64) - start_idx
    max_lag = int(pos_in_card.max())
    if max_lag == 0:
        return list(torch.zeros((n, 8), dtype=torch.float32).unbind(0))

    t_vals = df["delta_t"].to_numpy(dtype=np.float64)
    r_bin = (df["rating"].to_numpy() > 1).astype(np.float64)

    # Accumulate in float64 to match original np.zeros(8, dtype=float64), then cast to float32
    feats = np.zeros((n, 8), dtype=np.float64)
    active = np.flatnonzero(pos_in_card >= 1)
    cum_t = np.zeros(len(active), dtype=np.float64)

    for lag in range(1, max_lag + 1):
        if lag > 1:
            keep = pos_in_card[active] >= lag
            active = active[keep]
            cum_t = cum_t[keep]
        if len(active) == 0:
            break

        # Reverse cumulative sum order: t[i] + t[i-1] + ... + t[i-lag+1]
        cum_t = cum_t + t_vals[active - (lag - 1)]
        rb = r_bin[active - lag]

        if enable_decay:
            # valid: (M, 4), decay: (M, 4)
            valid = cum_t[:, None] <= _TIME_WINDOWS
            decay = np.where(valid, np.exp(-cum_t[:, None] / _TAU_W), 0.0)
            feats[active, 0::2] += decay
            feats[active, 1::2] += decay * rb[:, None]
        else:
            valid_f = (cum_t[:, None] <= _TIME_WINDOWS).astype(np.float64)
            feats[active, 0::2] += valid_f
            feats[active, 1::2] += valid_f * rb[:, None]

    full = torch.from_numpy(feats.astype(np.float32))
    return list(full.unbind(0))


class DashFeatureEngineer(BaseFeatureEngineer):
    """
    Base feature engineer for DASH model
    DASH uses time window features without decay
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create DASH time window features
        """
        cast(Any, df)["tensor"] = _compute_dash_tw_all(df, enable_decay=False)
        return df


class DashMCMFeatureEngineer(DashFeatureEngineer):
    """
    Feature engineer for DASH[MCM] model
    Same as DASH but with exponential decay enabled
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create DASH[MCM] time window features with decay
        """
        cast(Any, df)["tensor"] = _compute_dash_tw_all(df, enable_decay=True)
        return df


class DashACTRFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for DASH[ACT-R] model
    Uses ACT-R style activation features
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create DASH[ACT-R] features
        """
        n = len(df)
        if n == 0:
            cast(Any, df)["tensor"] = []
            return df

        r_bin_f32 = torch.from_numpy((df["rating"].to_numpy() > 1).astype(np.float32))
        t_f32 = torch.from_numpy(df["delta_t"].to_numpy(dtype=np.float32, copy=True))
        empty_0_2 = torch.empty((0, 2), dtype=torch.float32)

        starts, ends = _card_run_bounds(df["card_id"].to_numpy())
        starts_l = starts.tolist()
        ends_l = ends.tolist()
        max_m = max((e - s - 1 for s, e in zip(starts_l, ends_l)), default=0)
        if max_m <= 0:
            cast(Any, df)["tensor"] = [empty_0_2] * n
            return df

        tril_r, tril_c = torch.tril_indices(max_m, max_m)
        row_starts = [idx * (idx + 1) // 2 for idx in range(max_m)]
        row_ends = [row_starts[idx] + idx + 1 for idx in range(max_m)]

        tensors: list[torch.Tensor] = []
        for s, e in zip(starts_l, ends_l):
            tensors.append(empty_0_2)
            m = e - s - 1
            if m > 0:
                t_len = row_ends[m - 1]
                r_idx = tril_r[:t_len]
                c_idx = tril_c[:t_len]
                r_card = r_bin_f32[s : e - 1]
                sp_card = t_f32[s + 1 : e]
                cs_card = torch.cumsum(sp_card, dim=0)
                diff_card = sp_card - cs_card
                packed = torch.stack(
                    (r_card[c_idx], diff_card[c_idx] + cs_card[r_idx]), dim=1
                )
                tensors.extend(
                    packed[row_starts[idx] : row_ends[idx]] for idx in range(m)
                )

        cast(Any, df)["tensor"] = tensors
        return df
