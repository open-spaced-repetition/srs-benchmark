from abc import ABC, abstractmethod
from typing import Optional, List, Tuple
import pandas as pd
from config import Config
from utils import cum_concat
from fsrs_optimizer import remove_outliers, remove_non_continuous_rows  # type: ignore


class BaseFeatureEngineer(ABC):
    """
    Base abstract class for feature engineering
    Each specific model feature engineer should inherit from this class and implement the corresponding methods
    """

    def __init__(self, config: Config):
        """
        Initialize the feature engineer

        Args:
            config: Configuration object containing model parameters
        """
        self.config = config

    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Main feature creation method that includes common preprocessing steps for all models

        Args:
            df: Input raw dataframe

        Returns:
            Processed feature dataframe
        """
        # Execute common preprocessing steps
        df = self._common_preprocessing(df.copy())

        # Compute history records
        df = self._compute_histories(df)

        # Model-specific feature engineering
        df = self._model_specific_features(df)

        # Execute common postprocessing steps
        df = self._common_postprocessing(df)

        return df

    def _common_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Common preprocessing steps shared by all models
        """
        # Add review sequence number
        df["review_th"] = range(1, df.shape[0] + 1)
        df["nth_today"] = df.groupby("day_offset").cumcount() + 1
        df.sort_values(by=["card_id", "review_th"], inplace=True)

        # Filter invalid ratings
        df.drop(df[~df["rating"].isin([1, 2, 3, 4])].index, inplace=True)

        # Handle two-button mode
        if self.config.two_buttons:
            df["rating"] = df["rating"].replace({2: 3, 4: 3})

        # Calculate review count
        df["i"] = df.groupby("card_id").cumcount() + 1
        df.drop(df[df["i"] > self.config.max_seq_len * 2].index, inplace=True)

        # Process time intervals
        df = self._process_time_intervals(df)

        # Handle short-term reviews
        if not self.config.include_short_term:
            df.drop(df[df["elapsed_days"] == 0].index, inplace=True)
            df["i"] = df.groupby("card_id").cumcount() + 1

        return df

    def _process_time_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process time interval related fields
        """
        if (
            "delta_t" not in df.columns
            and "elapsed_days" in df.columns
            and "elapsed_seconds" in df.columns
        ):
            df["delta_t"] = df["elapsed_days"]
            if self.config.use_secs_intervals:
                df["delta_t_secs"] = df["elapsed_seconds"] / 86400
                df["delta_t_secs"] = df["delta_t_secs"].map(lambda x: max(0, x))

        df["delta_t"] = df["delta_t"].map(lambda x: max(0, x))
        return df

    def _compute_histories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute time and rating history records
        """
        # Calculate time history (non-seconds)
        t_history_non_secs_list = df.groupby("card_id", group_keys=False)[
            "delta_t"
        ].apply(lambda x: cum_concat([[i] for i in x]))

        # Calculate time history (seconds)
        if self.config.use_secs_intervals:
            t_history_secs_list = df.groupby("card_id", group_keys=False)[
                "delta_t_secs"
            ].apply(lambda x: cum_concat([[i] for i in x]))

        # Calculate rating history
        r_history_list = df.groupby("card_id", group_keys=False)["rating"].apply(
            lambda x: cum_concat([[i] for i in x])
        )

        # Calculate last rating
        last_rating = self._compute_last_rating(t_history_non_secs_list, r_history_list)
        df["last_rating"] = last_rating

        # Set history record strings
        df["r_history"] = [
            ",".join(map(str, item[:-1]))
            for sublist in r_history_list
            for item in sublist
        ]

        # Process time history strings
        df = self._set_time_histories(
            df,
            t_history_non_secs_list,
            t_history_secs_list if self.config.use_secs_intervals else None,
        )

        return df

    def _compute_last_rating(
        self, t_history_list: pd.Series, r_history_list: pd.Series
    ) -> List[int]:
        """
        Calculate the previous rating for each review
        """
        last_rating = []
        for t_sublist, r_sublist in zip(t_history_list, r_history_list):
            for t_history, r_history in zip(t_sublist, r_sublist):
                flag = True
                for t, r in zip(reversed(t_history[:-1]), reversed(r_history[:-1])):
                    if t > 0:
                        last_rating.append(r)
                        flag = False
                        break
                if flag:
                    last_rating.append(r_history[0])
        return last_rating

    def _set_time_histories(
        self,
        df: pd.DataFrame,
        t_history_non_secs_list: pd.Series,
        t_history_secs_list: Optional[pd.Series] = None,
    ) -> pd.DataFrame:
        """
        Set time history string fields
        """
        if t_history_secs_list:
            if self.config.equalize_test_with_non_secs:
                df["t_history"] = [
                    ",".join(map(str, item[:-1]))
                    for sublist in t_history_non_secs_list
                    for item in sublist
                ]
                df["t_history_secs"] = [
                    ",".join(map(str, item[:-1]))
                    for sublist in t_history_secs_list
                    for item in sublist
                ]
            else:
                df["t_history"] = [
                    ",".join(map(str, item[:-1]))
                    for sublist in t_history_secs_list
                    for item in sublist
                ]
            df["delta_t"] = df["delta_t_secs"]
        else:
            df["t_history"] = [
                ",".join(map(str, item[:-1]))
                for sublist in t_history_non_secs_list
                for item in sublist
            ]

        return df

    @abstractmethod
    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Model-specific feature engineering logic, needs to be implemented in subclasses

        Args:
            df: Dataframe after common preprocessing

        Returns:
            Dataframe with model-specific features added
        """
        pass

    def _common_postprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Common postprocessing steps shared by all models
        """
        # Set first rating and labels
        df["first_rating"] = df["r_history"].map(lambda x: x[0] if len(x) > 0 else "")
        df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])

        # Handle short-term reviews
        if self.config.include_short_term:
            df = df[(df["delta_t"] != 0) | (df["i"] == 1)].copy()

        # Recalculate review sequence number
        df["i"] = (
            df.groupby("card_id")
            .apply(lambda x: (x["elapsed_days"] > 0).cumsum())
            .reset_index(level=0, drop=True)
            + 1
        )

        # Handle outliers and non-continuous rows (only for non-seconds intervals)
        if not self.config.use_secs_intervals:
            df = self._handle_outliers_and_continuity(df)

        return df[df["delta_t"] > 0].sort_values(by=["review_th"])

    def _handle_outliers_and_continuity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers and non-continuous rows
        """

        filtered_dataset = (
            df[df["i"] == 2]
            .groupby(by=["first_rating"], as_index=False, group_keys=False)[df.columns]
            .apply(remove_outliers)
        )
        if filtered_dataset.empty:
            return pd.DataFrame()

        df[df["i"] == 2] = filtered_dataset
        df.dropna(inplace=True)
        df = df.groupby("card_id", as_index=False, group_keys=False)[df.columns].apply(
            remove_non_continuous_rows
        )
        return df

    def get_time_history_list(self, df: pd.DataFrame) -> pd.Series:
        """
        Get time history list for feature engineering

        Args:
            df: Input dataframe

        Returns:
            Time history list as pandas Series
        """
        if self.config.use_secs_intervals:
            t_history_list = df.groupby("card_id", group_keys=False)[
                (
                    "delta_t_secs"
                    if not self.config.equalize_test_with_non_secs
                    else "delta_t"
                )
            ].apply(lambda x: cum_concat([[i] for i in x]))
        else:
            t_history_list = df.groupby("card_id", group_keys=False)["delta_t"].apply(
                lambda x: cum_concat([[i] for i in x])
            )
        return t_history_list

    def get_rating_history_list(self, df: pd.DataFrame) -> pd.Series:
        """
        Get rating history list for feature engineering

        Args:
            df: Input dataframe

        Returns:
            Rating history list as pandas Series
        """
        r_history_list = df.groupby("card_id", group_keys=False)["rating"].apply(
            lambda x: cum_concat([[i] for i in x])
        )
        return r_history_list

    def get_nth_today_history_list(self, df: pd.DataFrame) -> pd.Series:
        """
        Get nth today history list for feature engineering
        """
        n_history_list = df.groupby("card_id", group_keys=False)["nth_today"].apply(
            lambda x: cum_concat([[i] for i in x])
        )
        return n_history_list

    def get_history_lists(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Get history record lists for feature engineering

        Returns:
            Tuple of (time_history_list, rating_history_list)
        """
        t_history_list = self.get_time_history_list(df)
        r_history_list = self.get_rating_history_list(df)
        return t_history_list, r_history_list
