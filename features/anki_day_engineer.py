import pandas as pd
import torch

from utils import cum_concat
from .base import BaseFeatureEngineer


class AnkiDayEngineer(BaseFeatureEngineer):
    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        (feature_real, feature_int, label_real, label_int, rating) tensors
        """
        (feature_real, feature_int, label_real, label_int), r_history_list = self.get_history_lists(df)
        df["tensor"] = [
            torch.tensor([x[:-1] for x in trio]).transpose(0, 1).float()
            for sublists in zip(feature_real, feature_int, label_real, label_int, r_history_list)
            for trio in zip(*sublists)
        ]
        return df

    def get_time_history_list(self, df: pd.DataFrame) -> pd.Series:
        feature_real = df.groupby("card_id", group_keys=False)["delta_t_secs"].apply(lambda x: cum_concat([[i] for i in x]))
        df["label_delta_t_secs"] = df.groupby("card_id")["delta_t_secs"].shift(-1)
        label_real = df.groupby("card_id", group_keys=False)["label_delta_t_secs"].apply(lambda x: cum_concat([[i] for i in x]))
        df.drop("label_delta_t_secs", axis=1)
        feature_int = df.groupby("card_id", group_keys=False)["delta_t_int"].apply(lambda x: cum_concat([[i] for i in x]))
        df["label_delta_t_int"] = df.groupby("card_id")["delta_t_int"].shift(-1)
        label_int = df.groupby("card_id", group_keys=False)["label_delta_t_int"].apply(lambda x: cum_concat([[i] for i in x]))
        df.drop("label_delta_t_int", axis=1)
        return (feature_real, feature_int, label_real, label_int)