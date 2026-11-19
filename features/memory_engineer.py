from typing import Any, cast

import pandas as pd

from .base import BaseFeatureEngineer, _prefix_pair_tuples


class SM2FeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for SM2 model
    SM2 only needs rating history as a string sequence
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create SM2 features: rating history as string sequence
        """
        # SM2 uses the r_history field which is already created in base preprocessing
        df["sequence"] = df["r_history"]
        return df


class EbisuFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for Ebisu models
    Ebisu needs (time_interval, rating) tuples for each review
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Ebisu features: sequence of (time_interval, rating) tuples
        """
        cast(Any, df)["sequence"] = _prefix_pair_tuples(
            df["card_id"].to_numpy(),
            df["delta_t"].to_numpy(),
            df["rating"].to_numpy(),
        )
        return df
