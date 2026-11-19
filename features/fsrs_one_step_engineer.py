from typing import Any, cast

import pandas as pd

from .base import (
    BaseFeatureEngineer,
    _float32_matrix,
    _prefix_pair_tuples,
    _prefix_views,
)


class FSRSOneStepFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for FSRS models (FSRSv1, FSRSv2, FSRSv3, FSRSv4, FSRS-4.5, FSRS-5, FSRS-6)
    Also handles RNN, GRU, Transformer, SM2-trainable, Anki, and 90% models
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tensor features for FSRS-style models
        These models use (time_history, rating_history) tensors
        """
        card_ids = df["card_id"].to_numpy()
        full = _float32_matrix(df, ["delta_t", "rating"])
        cast(Any, df)["tensor"] = _prefix_views(full, card_ids)
        cast(Any, df)["inputs"] = _prefix_pair_tuples(
            card_ids, df["delta_t"].to_numpy(), df["rating"].to_numpy()
        )
        return df
