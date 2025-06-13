import pandas as pd
import torch
import numpy as np
from typing import List
from .base import BaseFeatureEngineer


class DashFeatureEngineer(BaseFeatureEngineer):
    """
    Base feature engineer for DASH model
    DASH uses time window features without decay
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create DASH time window features
        """
        t_history_list, r_history_list = self.get_history_lists(df)

        df["tensor"] = [
            torch.tensor(
                self._dash_tw_features(r_item[:-1], t_item[1:], enable_decay=False),
                dtype=torch.float32,
            )
            for t_sublist, r_sublist in zip(t_history_list, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]

        return df

    def _dash_tw_features(
        self, r_history: List, t_history: List, enable_decay: bool = False
    ) -> np.ndarray:
        """
        Create DASH time window features

        Args:
            r_history: Rating history
            t_history: Time history
            enable_decay: Whether to enable exponential decay (for MCM variant)

        Returns:
            Feature vector of length 8
        """
        features = np.zeros(8)
        r_history = np.array(r_history) > 1  # Convert to binary success/failure
        tau_w = np.array([0.2434, 1.9739, 16.0090, 129.8426])  # Decay constants
        time_windows = np.array([1, 7, 30, np.inf])  # Time windows in days

        # Compute the cumulative sum of t_history in reverse order
        cumulative_times = np.cumsum(t_history[::-1])[::-1]

        for j, time_window in enumerate(time_windows):
            # Calculate decay factors for each time window
            if enable_decay:
                decay_factors = np.exp(-cumulative_times / tau_w[j])
            else:
                decay_factors = np.ones_like(cumulative_times)

            # Identify the indices where cumulative times are within the current time window
            valid_indices = cumulative_times <= time_window

            # Update features using decay factors where valid
            features[j * 2] += np.sum(decay_factors[valid_indices])  # Total count
            features[j * 2 + 1] += np.sum(
                r_history[valid_indices] * decay_factors[valid_indices]
            )  # Success count

        return features


class DashMCMFeatureEngineer(DashFeatureEngineer):
    """
    Feature engineer for DASH[MCM] model
    Same as DASH but with exponential decay enabled
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create DASH[MCM] time window features with decay
        """
        t_history_list, r_history_list = self.get_history_lists(df)

        df["tensor"] = [
            torch.tensor(
                self._dash_tw_features(r_item[:-1], t_item[1:], enable_decay=True),
                dtype=torch.float32,
            )
            for t_sublist, r_sublist in zip(t_history_list, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]

        return df


class DashACTRFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for DASH[ACT-R] model
    Uses ACT-R style activation features
    """

    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create DASH[ACT-R] features
        """
        t_history_list, r_history_list = self.get_history_lists(df)

        df["tensor"] = [
            torch.tensor(
                self._dash_actr_features(r_item[:-1], t_item[1:]),
                dtype=torch.float32,
            )
            for t_sublist, r_sublist in zip(t_history_list, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]

        return df

    def _dash_actr_features(self, r_history: List, t_history: List) -> torch.Tensor:
        """
        Create ACT-R style features for DASH[ACT-R]

        Args:
            r_history: Rating history
            t_history: Time history

        Returns:
            Feature tensor with shape (sequence_length, 2)
        """
        r_history = torch.tensor(np.array(r_history) > 1, dtype=torch.float32)
        sp_history = torch.tensor(t_history, dtype=torch.float32)
        cumsum = torch.cumsum(sp_history, dim=0)

        # Features: [success_indicator, time_since_start]
        features = [r_history, sp_history - cumsum + cumsum[-1:]]
        return torch.stack(features, dim=1)
