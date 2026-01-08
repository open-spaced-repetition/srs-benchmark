import pandas as pd
import torch
import numpy as np
from itertools import accumulate
from .base import BaseFeatureEngineer


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
        # Create additional features for LSTM
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
        # Mark new cards
        df["is_new_card"] = (~df["card_id"].duplicated()).astype(int)

        # Cumulative new cards
        df["cum_new_cards"] = df["is_new_card"].cumsum()

        # Difference in new cards since last review of this card
        df["diff_new_cards"] = df.groupby("card_id")["cum_new_cards"].diff().fillna(0)

        # Difference in reviews since last review of this card
        df["diff_reviews"] = np.maximum(
            0, -1 + df.groupby("card_id")["review_th"].diff().fillna(0)
        )

        # Daily statistics
        df["cum_new_cards_today"] = df.groupby("day_offset")["is_new_card"].cumsum()
        df["cum_reviews_today"] = df.groupby("day_offset").cumcount()

        # Time in days for forgetting curve
        df["delta_t_days"] = df["elapsed_days"].map(lambda x: max(0, x))

        return df

    def _create_lstm_tensors(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tensor features for LSTM model
        Features include: [delta_t, optional duration, rating]
        """
        # Define features to include in tensor
        features = self.config.get_lstm_tensor_feature_names()

        def get_history(group):
            # Create tensor for each row
            rows = group.apply(
                lambda row: torch.tensor(
                    [row[feature] for feature in features],
                    dtype=torch.float32,
                    requires_grad=False,
                ),
                axis=1,
            ).tolist()

            # Create cumulative history for each position
            cum_rows = list(
                accumulate(
                    rows,
                    lambda x, y: torch.cat((x, y.unsqueeze(0))),
                    initial=torch.empty(
                        (0, len(features)), dtype=torch.float32, requires_grad=False
                    ),
                )
            )[:-1]  # Remove the last element as it includes current review

            return pd.Series(cum_rows, index=group.index)

        # Apply history creation grouped by card
        grouped = df.groupby("card_id", group_keys=False)
        df["tensor"] = grouped[df.columns.difference(["card_id"])].apply(get_history)

        return df
