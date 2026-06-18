from typing import Any, cast

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

        # Create tensor features with shape (sequence_length, 2): each row is
        # [time_interval, rating] for that step. Build ONE [k, 2] float32 tensor per card
        # from its full (t, r) sequence, then take prefix views: review j's tensor is the
        # first j rows. Bit-identical to torch.tensor((t_item[:-1], r_item[:-1])) per
        # review (same float32 from the same values), but avoids one torch.tensor() call
        # per review and shrinks the intermediate column from O(k^2) to O(k) per card.
        tensors: list = []
        for t_sublist, r_sublist in zip(t_history_list, r_history_list):
            full = torch.tensor(
                (t_sublist[-1], r_sublist[-1]), dtype=torch.float32
            ).transpose(0, 1)
            tensors.extend(full[:j] for j in range(len(t_sublist)))
        cast(Any, df)["tensor"] = tensors

        return df
