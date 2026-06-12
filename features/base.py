from abc import ABC, abstractmethod
from typing import Optional, List
import pandas as pd
from config import Config
from utils import cum_concat
from fsrs_optimizer import remove_outliers, remove_non_continuous_rows  # type: ignore

# Per-card cumulative history. Nested three deep: outer = cards, middle = reviews within a
# card, inner = the cumulative prefix of (numpy) interval/rating values up to that review.
HistoryLists = list[list[list[float]]]


def _cumulative_lists_by_card(df: pd.DataFrame, col: str) -> HistoryLists:
    """Per-card list of cumulative-prefix lists for ``col``, in df row order.

    Bit-identical replacement for::

        df.groupby("card_id", group_keys=False)[col].apply(
            lambda x: cum_concat([[i] for i in x]))

    but without the pandas per-group ``apply`` overhead (the dominant create_features
    cost). Requires df sorted by ``card_id`` (guaranteed by _common_preprocessing, which
    is exactly what the groupby-apply assignment also relies on). Values are kept as numpy
    scalars (matching Series iteration), so ``str()`` and ``torch.tensor()`` downstream
    produce identical results.
    """
    card_ids = df["card_id"].values
    vals = list(df[col].values)
    out: HistoryLists = []
    n = len(vals)
    start = 0
    for i in range(1, n + 1):
        if i == n or card_ids[i] != card_ids[start]:
            block = vals[start:i]
            out.append([block[: j + 1] for j in range(len(block))])
            start = i
    return out


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
        # History lists computed once in _compute_histories and reused by
        # get_history_lists (set per user, since a fresh engineer is created per user).
        self._cached_t_history_list: Optional[HistoryLists] = None
        self._cached_r_history_list: Optional[HistoryLists] = None

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

        # Model-specific postprocessing steps
        df = self._model_specific_postprocessing(df)

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
        df["delta_t_int"] = df["elapsed_days"].map(lambda x: max(0, x))
        return df

    def _compute_histories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute time and rating history records
        """
        # Calculate time history (non-seconds)
        t_history_non_secs_list = _cumulative_lists_by_card(df, "delta_t")

        # Calculate time history (seconds)
        t_history_secs_list: Optional[HistoryLists] = None
        if self.config.use_secs_intervals:
            t_history_secs_list = _cumulative_lists_by_card(df, "delta_t_secs")

        # Calculate rating history
        r_history_list = _cumulative_lists_by_card(df, "rating")

        # Cache the per-card history lists so model-specific feature builders can reuse
        # them via get_history_lists() instead of recomputing the same groupby/cum_concat.
        # Bit-identical: get_time_history_list() groups the very column the cached list was
        # built from (delta_t_secs when --secs, else delta_t; and after _set_time_histories
        # reassigns delta_t := delta_t_secs, the --secs+equalize path groups identical values).
        self._cached_r_history_list = r_history_list
        self._cached_t_history_list = (
            t_history_secs_list
            if self.config.use_secs_intervals
            else t_history_non_secs_list
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
            t_history_secs_list,
        )

        return df

    def _compute_last_rating(
        self, t_history_list: HistoryLists, r_history_list: HistoryLists
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
        t_history_non_secs_list: HistoryLists,
        t_history_secs_list: Optional[HistoryLists] = None,
    ) -> pd.DataFrame:
        """
        Set time history string fields
        """
        if t_history_secs_list is not None:
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
        df["first_rating"] = (
            df.groupby("card_id")["rating"].transform("first").astype(str)
        )
        df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])

        # Find lapses for RMSE (bins)
        df["is_lapse"] = (
            (df["rating"] == 1) & (df["delta_t"].astype(str) != "0")
        ).astype(int)
        df["rmse_bins_lapse"] = (
            df.groupby("card_id")["is_lapse"].transform("cumsum") - df["is_lapse"]
        )
        df.drop(columns=["is_lapse"], inplace=True)

        # Handle short-term reviews
        if self.config.include_short_term:
            df = df[(df["delta_t"] != 0) | (df["i"] == 1)].copy()

        # Recalculate review sequence number
        df["i"] = df["elapsed_days"].gt(0).groupby(df["card_id"]).cumsum().add(1)

        # Handle outliers and non-continuous rows (only for non-seconds intervals)
        if not self.config.use_secs_intervals:
            df = self._handle_outliers_and_continuity(df)
            if df.empty:
                raise ValueError(
                    "No data after handling outliers and non-continuous rows"
                )

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

    def get_history_lists(self, df: pd.DataFrame):
        """
        Get history record lists for feature engineering

        Returns:
            Tuple of (time_history_list, rating_history_list)
        """
        # Reuse the lists already computed in _compute_histories (set per-user in
        # _compute_histories) instead of recomputing the same groupby/cum_concat. Falls
        # back to recomputing for engineers that don't run base _compute_histories.
        if (
            self._cached_t_history_list is not None
            and self._cached_r_history_list is not None
        ):
            return self._cached_t_history_list, self._cached_r_history_list
        return self.get_time_history_list(df), self.get_rating_history_list(df)

    def _model_specific_postprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Model-specific post processing

        Args:
            df: Dataframe after common post processing

        Returns:
            Dataframe with model-specific post processing
        """
        return df
