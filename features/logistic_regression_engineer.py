import numpy as np
import pandas as pd

from models import logistic_regression

from .base import BaseFeatureEngineer, _card_start_indices


class LogisticRegressionEngineer(BaseFeatureEngineer):
    def _model_specific_postprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.use_secs_intervals:
            x = logistic_regression.create_features(df)
            df = pd.concat([df, x], axis=1)
        else:
            # for --equalize_test_with_non_secs only
            pass
        return df

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        # Previous rating of the same card (0 for the first review)
        is_start, _ = _card_start_indices(df["card_id"].to_numpy())
        ratings = df["rating"].to_numpy()
        feat_r = np.zeros(len(df), dtype=np.float64)
        feat_r[1:] = np.where(is_start[1:], 0.0, ratings[:-1])
        df["feature_rating"] = feat_r
        return df

    def _compute_histories(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.use_secs_intervals:
            df["delta_t"] = df["delta_t_secs"]
        return df
