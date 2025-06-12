import pandas as pd
import torch
import numpy as np
from typing import List
from .base import BaseFeatureEngineer


class GRUPFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for GRU-P model
    Uses time intervals starting from second review and ratings up to previous review
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create GRU-P features: (time_intervals[1:], ratings[:-1])
        """
        t_history_list, r_history_list = self.get_history_lists(df)

        df["tensor"] = [
            torch.tensor((t_item[1:], r_item[:-1]), dtype=torch.float32).transpose(0, 1)
            for t_sublist, r_sublist in zip(t_history_list, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]

        return df


class HLRFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for HLR (Hierarchical Linear Regression) model
    Uses square root of success and failure counts
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create HLR features: [sqrt(successes), sqrt(failures)]
        """
        _, r_history_list = self.get_history_lists(df)

        df["tensor"] = [
            torch.tensor(
                [
                    np.sqrt(
                        r_item[:-1].count(2)
                        + r_item[:-1].count(3)
                        + r_item[:-1].count(4)
                    ),  # Success count (ratings 2, 3, 4)
                    np.sqrt(r_item[:-1].count(1)),  # Failure count (rating 1)
                ],
                dtype=torch.float32,
            )
            for r_sublist in r_history_list
            for r_item in r_sublist
        ]

        return df


class ACTRFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for ACT-R model
    Uses cumulative time intervals for activation computation
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create ACT-R features: cumulative time intervals
        """
        t_history_list, _ = self.get_history_lists(df)

        df["tensor"] = [
            (torch.cumsum(torch.tensor([t_item]), dim=1)).transpose(0, 1)
            for t_sublist in t_history_list
            for t_item in t_sublist
        ]

        return df


class NN17FeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for NN-17 model
    Uses time intervals, ratings, and lapse history
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create NN-17 features: [time_intervals, ratings, lapse_history]
        """
        t_history_list, r_history_list = self.get_history_lists(df)

        df["tensor"] = [
            torch.tensor(
                (t_item[:-1], r_item[:-1], self._r_history_to_l_history(r_item[:-1]))
            ).transpose(0, 1)
            for t_sublist, r_sublist in zip(t_history_list, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]

        return df

    def _r_history_to_l_history(self, r_history: List) -> List[int]:
        """
        Convert rating history to lapse history (cumulative failure count)

        Args:
            r_history: List of ratings

        Returns:
            List of cumulative lapse counts
        """
        l_history = [0 for _ in range(len(r_history) + 1)]
        for i, r in enumerate(r_history):
            l_history[i + 1] = l_history[i] + (
                r == 1
            )  # Increment if rating is 1 (failure)
        return l_history[:-1]
