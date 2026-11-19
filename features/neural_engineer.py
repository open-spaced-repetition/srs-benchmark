from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from .base import (
    BaseFeatureEngineer,
    _card_run_bounds,
    _card_start_indices,
    _prefix_views,
    _segmented_cumsum,
)


def _history_dtype(delta_t: pd.Series) -> type[np.number]:
    """float64 for float intervals, int64 otherwise (dtype torch infers from NumPy scalars)."""
    return np.float64 if pd.api.types.is_float_dtype(delta_t) else np.int64


class HLRFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for HLR (Hierarchical Linear Regression) model
    Uses square root of success and failure counts
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create HLR features: [sqrt(successes), sqrt(failures)]
        """
        ratings = df["rating"].to_numpy()
        _, start_idx = _card_start_indices(df["card_id"].to_numpy())
        is_succ = (ratings > 1).astype(np.int64)
        is_fail = (ratings == 1).astype(np.int64)

        succ_before = _segmented_cumsum(is_succ, start_idx) - is_succ
        fail_before = _segmented_cumsum(is_fail, start_idx) - is_fail

        arr = np.empty((len(df), 2), dtype=np.float32)
        arr[:, 0] = np.sqrt(succ_before)
        arr[:, 1] = np.sqrt(fail_before)
        cast(Any, df)["tensor"] = list(torch.from_numpy(arr).unbind(0))

        return df


class ACTRFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for ACT-R model
    Uses cumulative time intervals for activation computation
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ACT-R features: cumulative time intervals
        """
        # Parent Series iteration yields numpy scalars: np.float64 promotes to torch.float64, np.int64 to torch.int64
        full_t = torch.from_numpy(
            df["delta_t"].to_numpy(dtype=_history_dtype(df["delta_t"]), copy=True)
        ).unsqueeze(1)
        starts, ends = _card_run_bounds(df["card_id"].to_numpy())
        tensors: list[torch.Tensor] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            card_cum = torch.cumsum(full_t[s:e], dim=0)
            tensors.extend(card_cum[: j + 1] for j in range(e - s))

        cast(Any, df)["tensor"] = tensors
        return df


class NN17FeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for NN-17 model
    Uses time intervals, ratings, and lapse history
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create NN-17 features: [time_intervals, ratings, lapse_history]
        """
        card_ids = df["card_id"].to_numpy()
        ratings = df["rating"].to_numpy()
        _, start_idx = _card_start_indices(card_ids)
        is_fail = (ratings == 1).astype(np.int64)
        lapses_before = _segmented_cumsum(is_fail, start_idx) - is_fail

        # Match torch.tensor((t_item[:-1], r_item[:-1], l_history)) exact dtypes:
        # empty (j == 0) defaults to float32; non-empty uses float64 for np.float64 delta_t, int64 for int delta_t.
        arr = np.empty((len(df), 3), dtype=_history_dtype(df["delta_t"]))
        arr[:, 0] = df["delta_t"].to_numpy()
        arr[:, 1] = ratings
        arr[:, 2] = lapses_before
        empty_0_3 = torch.empty((0, 3), dtype=torch.float32)
        cast(Any, df)["tensor"] = _prefix_views(
            torch.from_numpy(arr), card_ids, empty=empty_0_3
        )
        return df
