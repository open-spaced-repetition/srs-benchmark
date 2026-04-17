import pandas as pd

from models import logistic_regression
from .base import BaseFeatureEngineer


class LogisticRegressionEngineer(BaseFeatureEngineer):
    def _model_specific_postprocessing(self, df:pd.DataFrame) -> pd.DataFrame:
        if self.config.use_secs_intervals:
            x = logistic_regression.create_features(df)
            df = pd.concat([df, x], axis=1)
        else:
            # for --equalize_test_with_non_secs only
            pass
        return df

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df["feature_rating"] = df["rating"].shift(1).fillna(0)
        return df

    def _compute_histories(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.config.use_secs_intervals:
            df["delta_t"] = df["delta_t_secs"]
        return df