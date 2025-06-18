import pandas as pd
import torch
from .base import BaseFeatureEngineer


class FSRSFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for FSRS models (FSRSv1, FSRSv2, FSRSv3, FSRSv4, FSRS-4.5, FSRS-5, FSRS-6)
    Also handles RNN, GRU, Transformer, SM2-trainable, Anki, and 90% models
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create tensor features for FSRS-style models
        These models use (time_history, rating_history) tensors
        """
        t_history_list, r_history_list = self.get_history_lists(df)

        # Create tensor features with shape (sequence_length, 2)
        # Each row contains [time_interval, rating] for that step
        df["tensor"] = [
            torch.tensor((t_item[:-1], r_item[:-1]), dtype=torch.float32).transpose(
                0, 1
            )
            for t_sublist, r_sublist in zip(t_history_list, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]

        return df
