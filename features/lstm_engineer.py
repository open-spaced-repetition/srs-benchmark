from typing import Any, cast

import numpy as np
import pandas as pd

from .base import (
    BaseFeatureEngineer,
    _card_start_indices,
    _clip_non_negative,
    _float32_matrix,
    _prefix_views,
)


class LSTMFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for LSTM model
    LSTM requires additional features like new card counts and review counts
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features specific to LSTM model
        Includes cumulative statistics and tensor creation
        """
        # Auxiliary daily/card count columns are only exported when --file is set
        if self.config.save_evaluation_file:
            df = self._create_lstm_features(df)

        # Create tensor features
        df = self._create_lstm_tensors(df)

        return df

    def _create_lstm_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create LSTM-specific features:
        - Number of unique cards in the revlog
        - Number of new cards introduced today so far
        - Number of reviews done today so far
        - Number of new cards introduced since last review of this card
        - Number of reviews done since last review of this card
        """
        n = len(df)
        if n == 0:
            df["is_new_card"] = np.empty(0, dtype=np.int64)
            df["cum_new_cards"] = np.empty(0, dtype=np.int64)
            df["diff_new_cards"] = np.empty(0, dtype=np.float64)
            df["diff_reviews"] = np.empty(0, dtype=np.float64)
            if "day_offset" in df.columns:
                df["cum_new_cards_today"] = np.empty(0, dtype=np.int64)
                df["cum_reviews_today"] = np.empty(0, dtype=np.int64)
            df["delta_t_days"] = np.empty(0, dtype=np.int64)
            return df

        is_start, _ = _card_start_indices(df["card_id"].to_numpy())
        is_new_card = is_start.astype(np.int64)
        cum_new_cards = np.cumsum(is_new_card, dtype=np.int64)

        diff_new_cards = np.zeros(n, dtype=np.float64)
        diff_new_cards[1:] = np.where(
            is_start[1:], 0.0, cum_new_cards[1:] - cum_new_cards[:-1]
        )

        review_th = df["review_th"].to_numpy()
        diff_rev = np.zeros(n, dtype=np.float64)
        diff_rev[1:] = np.where(is_start[1:], 0.0, review_th[1:] - review_th[:-1])
        diff_reviews = np.maximum(0.0, -1.0 + diff_rev)

        df["is_new_card"] = is_new_card
        df["cum_new_cards"] = cum_new_cards
        df["diff_new_cards"] = diff_new_cards
        df["diff_reviews"] = diff_reviews

        if "day_offset" in df.columns:
            df["cum_new_cards_today"] = df.groupby("day_offset")["is_new_card"].cumsum()
            df["cum_reviews_today"] = df.groupby("day_offset").cumcount()

        # Time in days for forgetting curve
        df["delta_t_days"] = _clip_non_negative(df["elapsed_days"].to_numpy())

        return df

    def _create_lstm_tensors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tensor features for LSTM model
        Features include: [delta_t, optional duration, rating]
        """
        full = _float32_matrix(df, self.config.get_lstm_tensor_feature_names())
        cast(Any, df)["tensor"] = _prefix_views(full, df["card_id"].to_numpy())
        return df
