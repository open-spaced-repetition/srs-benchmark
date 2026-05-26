from typing import Any, cast
import pandas as pd
import torch

from .base import BaseFeatureEngineer
from models import logistic_regression


class FSRS7LREnsembleEngineer(BaseFeatureEngineer):
    """Feature engineer for FSRS-7 + LR Ensemble.

    Builds both FSRS-7 tensors and Logistic Regression features.
    Always creates ``delta_t_secs`` so that LR features work
    regardless of the ``--secs`` flag.
    """

    def _process_time_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        df = super()._process_time_intervals(df)
        # LR features require fractional-day intervals even without --secs
        if "delta_t_secs" not in df.columns and "elapsed_seconds" in df.columns:
            df["delta_t_secs"] = (df["elapsed_seconds"] / 86400).clip(lower=0)
        return df

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        t_history_list, r_history_list = self.get_history_lists(df)
        cast(Any, df)["tensor"] = [
            torch.tensor((t_item[:-1], r_item[:-1]), dtype=torch.float32).transpose(0, 1)
            for t_sublist, r_sublist in zip(t_history_list, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
        df["feature_rating"] = df.groupby("card_id")["rating"].shift(1).fillna(0)
        return df

    def _model_specific_postprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        x = logistic_regression.create_features(df)
        df = pd.concat([df, x], axis=1)
        return df

    def _compute_histories(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.use_secs_intervals:
            df["delta_t"] = df["delta_t_secs"]
        return df
