from typing import Any, cast

import numpy as np
import pandas as pd
import torch

from .base import (
    BaseFeatureEngineer,
    _card_run_bounds,
    _card_start_indices,
    _segmented_cumsum,
)


class HLRFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for HLR (Hierarchical Linear Regression) model
    Uses square root of success and failure counts
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create HLR features: [sqrt(successes), sqrt(failures)]
        """
        n = len(df)
        if n == 0:
            cast(Any, df)["tensor"] = []
            return df

        ratings = df["rating"].to_numpy()
        _, start_idx = _card_start_indices(df["card_id"].to_numpy())
        is_succ = (ratings > 1).astype(np.int64)
        is_fail = (ratings == 1).astype(np.int64)

        succ_before = _segmented_cumsum(is_succ, start_idx) - is_succ
        fail_before = _segmented_cumsum(is_fail, start_idx) - is_fail

        arr = np.empty((n, 2), dtype=np.float32)
        arr[:, 0] = np.sqrt(succ_before).astype(np.float32)
        arr[:, 1] = np.sqrt(fail_before).astype(np.float32)
        full = torch.from_numpy(arr)
        cast(Any, df)["tensor"] = list(full.unbind(0))

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
        n = len(df)
        if n == 0:
            cast(Any, df)["tensor"] = []
            return df

        # Parent Series iteration yields numpy scalars: np.float64 promotes to torch.float64, np.int64 to torch.int64
        target_dtype = (
            np.float64 if pd.api.types.is_float_dtype(df["delta_t"]) else np.int64
        )
        full_t = torch.from_numpy(
            df["delta_t"].to_numpy(dtype=target_dtype, copy=True)
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
        n = len(df)
        if n == 0:
            cast(Any, df)["tensor"] = []
            return df

        dt_np = df["delta_t"].to_numpy()
        ratings = df["rating"].to_numpy()
        _, start_idx = _card_start_indices(df["card_id"].to_numpy())
        is_fail = (ratings == 1).astype(np.int64)
        lapses_before = _segmented_cumsum(is_fail, start_idx) - is_fail

        # Match torch.tensor((t_item[:-1], r_item[:-1], l_history)) exact dtypes:
        # empty (j == 0) defaults to float32; non-empty uses float64 for np.float64 delta_t, int64 for int delta_t.
        target_dtype = (
            np.float64 if pd.api.types.is_float_dtype(df["delta_t"]) else np.int64
        )
        arr = np.empty((n, 3), dtype=target_dtype)
        arr[:, 0] = dt_np
        arr[:, 1] = ratings
        arr[:, 2] = lapses_before
        full = torch.from_numpy(arr)
        empty_0_3 = torch.empty((0, 3), dtype=torch.float32)

        starts, ends = _card_run_bounds(df["card_id"].to_numpy())
        tensors: list[torch.Tensor] = []
        for s, e in zip(starts.tolist(), ends.tolist()):
            tensors.append(empty_0_3)
            if e - s > 1:
                tensors.extend(full[s : s + j] for j in range(1, e - s))

        cast(Any, df)["tensor"] = tensors
        return df
