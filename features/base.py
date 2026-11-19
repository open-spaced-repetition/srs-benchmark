from abc import ABC, abstractmethod

import numpy as np
import pandas as pd
import torch

from config import Config

_FIRST_RATING_STR_LUT = np.array(["", "1", "2", "3", "4"], dtype=object)
# 0-d float64 array: promotes like an np.float64 scalar (NEP 50) in np.maximum
_ZERO_F64 = np.zeros((), dtype=np.float64)


def _clip_non_negative(arr: np.ndarray) -> np.ndarray:
    """Clip to >= 0 while matching pandas ``Series.map(lambda x: max(0, x))`` dtypes."""
    if np.issubdtype(arr.dtype, np.integer):
        return np.maximum(np.int64(0), arr)
    if np.issubdtype(arr.dtype, np.floating):
        return np.maximum(_ZERO_F64, arr)
    return np.maximum(0, arr)


def _card_start_indices(card_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(is_start, start_idx)`` for a 1D ``card_ids`` array sorted by card."""
    n = len(card_ids)
    is_start = np.ones(n, dtype=bool)
    is_start[1:] = card_ids[1:] != card_ids[:-1]
    start_idx = np.maximum.accumulate(
        np.where(is_start, np.arange(n).astype(np.int64), 0)
    )
    return is_start, start_idx


def _positions_in_card(card_ids: np.ndarray) -> np.ndarray:
    """0-based position of each row within its contiguous ``card_id`` run."""
    _, start_idx = _card_start_indices(card_ids)
    return np.arange(len(card_ids)).astype(np.int64) - start_idx


def _card_run_bounds(card_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(starts, ends)`` slice bounds for contiguous ``card_id`` blocks."""
    n = len(card_ids)
    if n == 0:
        empty = np.empty(0, dtype=np.int64)
        return empty, empty
    changes = np.flatnonzero(card_ids[1:] != card_ids[:-1]) + 1
    starts = np.concatenate(([0], changes)).astype(np.int64)
    ends = np.concatenate((changes, [n])).astype(np.int64)
    return starts, ends


def _segmented_cumsum(vals: np.ndarray, start_idx: np.ndarray) -> np.ndarray:
    """Segmented integer cumulative sum over contiguous runs defined by ``start_idx``."""
    cs = np.cumsum(vals, dtype=np.int64)
    offset = np.where(start_idx > 0, cs[start_idx - 1], 0)
    return cs - offset


def _float32_matrix(df: pd.DataFrame, columns: list[str]) -> torch.Tensor:
    """C-contiguous ``[len(df), len(columns)]`` float32 tensor built column by column."""
    arr = np.empty((len(df), len(columns)), dtype=np.float32)
    for col_idx, col in enumerate(columns):
        arr[:, col_idx] = df[col].to_numpy(dtype=np.float32)
    return torch.from_numpy(arr)


def _prefix_views(
    full: torch.Tensor, card_ids: np.ndarray, empty: torch.Tensor | None = None
) -> list[torch.Tensor]:
    """Zero-copy history views: the row at ``s + j`` of a card run gets ``full[s : s + j]``.

    If ``empty`` is given it replaces the zero-length view of each card's first review.
    """
    starts, ends = _card_run_bounds(card_ids)
    bounds = zip(starts.tolist(), ends.tolist())
    # Flat list comprehensions: noticeably faster than extend(generator) per card
    if empty is None:
        return [full[s : s + j] for s, e in bounds for j in range(e - s)]
    return [full[s : s + j] if j else empty for s, e in bounds for j in range(e - s)]


def _prefix_pair_tuples(
    card_ids: np.ndarray, first: np.ndarray, second: np.ndarray
) -> list[tuple]:
    """Per-row tuple of ``(first, second)`` pairs from earlier reviews of the same card.

    Elements stay NumPy scalars, matching iteration over the original Series.
    """
    pairs = list(zip(list(first), list(second)))
    starts, ends = _card_run_bounds(card_ids)
    return [
        card_pairs[:j]
        for s, e in zip(starts.tolist(), ends.tolist())
        for card_pairs in (tuple(pairs[s : e - 1]),)
        for j in range(e - s)
    ]


def _prefix_history_strings(card_ids: np.ndarray, values: np.ndarray) -> list[str]:
    """Build comma-separated prefix strings (excluding current row) per contiguous card."""
    out: list[str] = []
    acc = ""
    prev_cid = None
    for cid, s in zip(card_ids.tolist(), map(str, values.tolist())):
        if cid != prev_cid:
            out.append("")
            acc = s
            prev_cid = cid
        else:
            out.append(acc)
            acc = f"{acc},{s}"
    return out


def _compute_last_rating(
    is_start: np.ndarray, t_non_secs: np.ndarray, ratings: np.ndarray
) -> np.ndarray:
    """Rating of the latest earlier review of the same card whose own interval was
    positive, falling back to the card's first review."""
    idx_arr = np.arange(len(ratings)).astype(np.int64)
    latest_valid = np.maximum.accumulate(
        np.where(is_start | (t_non_secs > 0), idx_arr, 0)
    )
    src = np.where(is_start, idx_arr, np.roll(latest_valid, 1))
    return ratings[src]


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

        # Model-specific postprocessing steps
        df = self._model_specific_postprocessing(df)

        return df

    def create_alignment_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Lightweight non-seconds pass for ``--equalize_test_with_non_secs`` alignment.

        Computes exact ``review_th``, ``i``, ``rmse_bins_lapse``, and row filtering
        without building history strings or model-specific feature tensors.
        """
        df = self._common_preprocessing(df.copy())
        return self._common_postprocessing(df)

    def _common_preprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Common preprocessing steps shared by all models
        """
        # Add review sequence number
        df["review_th"] = np.arange(len(df)).astype(np.int64) + 1
        if self.config.save_evaluation_file and "day_offset" in df.columns:
            df["nth_today"] = df.groupby("day_offset").cumcount() + 1
        df.sort_values(by=["card_id", "review_th"], inplace=True)

        # Filter invalid ratings
        rating_vals = df["rating"].to_numpy()
        if pd.api.types.is_integer_dtype(rating_vals):
            valid_rating = (rating_vals >= 1) & (rating_vals <= 4)
        else:
            valid_rating = np.isin(rating_vals, (1, 2, 3, 4))
        if not valid_rating.all():
            df = df[valid_rating].copy()

        # Handle two-button mode
        if self.config.two_buttons and len(df) > 0:
            r_arr = df["rating"].to_numpy()
            df["rating"] = np.where((r_arr == 2) | (r_arr == 4), 3, r_arr)

        # Calculate review count on contiguous card runs
        i_vals = _positions_in_card(df["card_id"].to_numpy()) + 1
        df["i"] = i_vals
        max_i = self.config.max_seq_len * 2
        if (i_vals > max_i).any():
            df = df[i_vals <= max_i].copy()

        # Process time intervals
        df = self._process_time_intervals(df)

        # Handle short-term reviews
        if not self.config.include_short_term:
            ed_vals = df["elapsed_days"].to_numpy()
            if (ed_vals == 0).any():
                df = df[ed_vals != 0].copy()
            df["i"] = _positions_in_card(df["card_id"].to_numpy()) + 1

        return df

    def _process_time_intervals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Process time interval related fields
        """
        if len(df) == 0:
            if (
                "delta_t" not in df.columns
                and "elapsed_days" in df.columns
                and (
                    "elapsed_seconds" in df.columns
                    or not self.config.use_secs_intervals
                )
            ):
                df["delta_t"] = df["elapsed_days"]
                if self.config.use_secs_intervals:
                    df["delta_t_secs"] = df["elapsed_seconds"] / 86400
                    df["delta_t_secs"] = df["delta_t_secs"].map(lambda x: max(0, x))
            df["delta_t"] = df["delta_t"].map(lambda x: max(0, x))
            df["delta_t_int"] = df["elapsed_days"].map(lambda x: max(0, x))
            return df

        if (
            "delta_t" not in df.columns
            and "elapsed_days" in df.columns
            and ("elapsed_seconds" in df.columns or not self.config.use_secs_intervals)
        ):
            df["delta_t"] = _clip_non_negative(df["elapsed_days"].to_numpy())
            if self.config.use_secs_intervals:
                df["delta_t_secs"] = np.maximum(
                    _ZERO_F64,
                    df["elapsed_seconds"].to_numpy() / 86400,
                )
        else:
            df["delta_t"] = _clip_non_negative(df["delta_t"].to_numpy())
        df["delta_t_int"] = _clip_non_negative(df["elapsed_days"].to_numpy())
        return df

    def _compute_histories(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Compute time and rating history records on demand
        """
        config = self.config
        card_ids = df["card_id"].to_numpy()
        t_non_secs = df["delta_t"].to_numpy()
        ratings = df["rating"].to_numpy()
        secs_equalized = (
            config.use_secs_intervals and config.equalize_test_with_non_secs
        )

        if config.save_evaluation_file:
            is_start, _ = _card_start_indices(card_ids)
            df["last_rating"] = (
                _compute_last_rating(is_start, t_non_secs, ratings) if len(df) else []
            )

        if config.needs_r_history():
            df["r_history"] = _prefix_history_strings(card_ids, ratings)

        if config.needs_t_history():
            # --secs uses second-level intervals, unless test is equalized with non-secs
            t_hist_vals = (
                df["delta_t_secs"].to_numpy()
                if config.use_secs_intervals and not secs_equalized
                else t_non_secs
            )
            df["t_history"] = _prefix_history_strings(card_ids, t_hist_vals)
            if secs_equalized and config.save_evaluation_file:
                df["t_history_secs"] = _prefix_history_strings(
                    card_ids, df["delta_t_secs"].to_numpy()
                )

        if config.use_secs_intervals:
            df["delta_t"] = df["delta_t_secs"]

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

    def _common_postprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Common postprocessing steps shared by all models
        """
        if len(df) == 0:
            df["first_rating"] = (
                df.groupby("card_id")["rating"].transform("first").astype(str)
            )
            df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
            df["is_lapse"] = (
                (df["rating"] == 1) & (df["delta_t"].astype(str) != "0")
            ).astype(int)
            df["rmse_bins_lapse"] = (
                df.groupby("card_id")["is_lapse"].transform("cumsum") - df["is_lapse"]
            )
            df.drop(columns=["is_lapse"], inplace=True)
            df["i"] = df["elapsed_days"].gt(0).groupby(df["card_id"]).cumsum().add(1)
            if not self.config.use_secs_intervals:
                raise ValueError(
                    "No data after handling outliers and non-continuous rows"
                )
            return df[df["delta_t"] > 0].sort_values(by=["review_th"])

        card_ids = df["card_id"].to_numpy()
        ratings = df["rating"].to_numpy()
        delta_t_vals = df["delta_t"].to_numpy()
        _, start_idx = _card_start_indices(card_ids)

        # Set first rating and labels
        df["first_rating"] = _FIRST_RATING_STR_LUT[ratings[start_idx]]
        df["y"] = (ratings > 1).astype(np.int64)

        # Find lapses for RMSE (bins). Note: in original code, delta_t.astype(str) != "0"
        # evaluates to True for all float64 values (e.g. 0.0 -> "0.0") and delta_t != 0 for ints.
        if pd.api.types.is_float_dtype(df["delta_t"]):
            is_lapse = (ratings == 1).astype(np.int64)
        else:
            is_lapse = ((ratings == 1) & (delta_t_vals != 0)).astype(np.int64)
        df["rmse_bins_lapse"] = _segmented_cumsum(is_lapse, start_idx) - is_lapse

        # Handle short-term reviews
        if self.config.include_short_term:
            keep_short = (delta_t_vals != 0) | (df["i"].to_numpy() == 1)
            if not keep_short.all():
                df = df[keep_short].copy()
                _, start_idx = _card_start_indices(df["card_id"].to_numpy())

        # Recalculate review sequence number
        gt0 = (df["elapsed_days"].to_numpy() > 0).astype(np.int64)
        df["i"] = _segmented_cumsum(gt0, start_idx) + 1

        # Handle outliers and non-continuous rows (only for non-seconds intervals)
        if not self.config.use_secs_intervals:
            df = self._handle_outliers_and_continuity(df)
            if df.empty:
                raise ValueError(
                    "No data after handling outliers and non-continuous rows"
                )

        pos_mask = df["delta_t"].to_numpy() > 0
        return df[pos_mask].sort_values(by=["review_th"])

    def _handle_outliers_and_continuity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle outliers and non-continuous rows (bit-identical to fsrs_optimizer
        ``remove_outliers`` + ``remove_non_continuous_rows``, vectorized over contiguous cards).
        """
        n = len(df)
        if n == 0:
            return pd.DataFrame()

        i_vals = df["i"].to_numpy()
        i2_pos = np.flatnonzero(i_vals == 2)
        if len(i2_pos) == 0:
            return pd.DataFrame()

        first_ratings_i2 = df["first_rating"].to_numpy()[i2_pos]
        delta_t_i2 = df["delta_t"].to_numpy()[i2_pos]
        keep_row = np.ones(n, dtype=bool)
        any_i2_kept = False

        for fr in ("1", "2", "3", "4"):
            fr_mask = first_ratings_i2 == fr
            if not fr_mask.any():
                continue
            dt_sub = delta_t_i2[fr_mask]
            uniq_dt, counts = np.unique(dt_sub, return_counts=True)
            # Sort by (count ASC, delta_t DESC); uniq_dt is strictly unique so order is unique.
            order = np.lexsort((-uniq_dt, counts))
            total = len(dt_sub)
            threshold = max(total * 0.05, 20.0)
            max_dt = 365 if fr == "4" else 100
            removed_dts: list = []
            has_been_removed = 0
            for c, dt_val in zip(counts[order].tolist(), uniq_dt[order].tolist()):
                if has_been_removed + c >= threshold:
                    if c < 6 or dt_val > max_dt:
                        removed_dts.append(dt_val)
                        has_been_removed += c
                else:
                    removed_dts.append(dt_val)
                    has_been_removed += c

            if removed_dts:
                sub_keep = ~np.isin(dt_sub, removed_dts)
                keep_row[i2_pos[fr_mask]] = sub_keep
                if sub_keep.any():
                    any_i2_kept = True
            else:
                any_i2_kept = True

        if not any_i2_kept:
            return pd.DataFrame()

        if not keep_row.all():
            df = df[keep_row].copy()
            int_cols = [c for c in df.columns if pd.api.types.is_integer_dtype(df[c])]
            if int_cols:
                df[int_cols] = df[int_cols].astype(np.float64)

        # Vectorized remove_non_continuous_rows over contiguous card runs
        m = len(df)
        if m == 0:
            return pd.DataFrame()
        card_ids = df["card_id"].to_numpy()
        i_rem = df["i"].to_numpy()
        is_start, start_idx = _card_start_indices(card_ids)
        is_discont = np.zeros(m, dtype=bool)
        is_discont[1:] = (~is_start[1:]) & ((i_rem[1:] - i_rem[:-1]) != 1)
        if not is_discont.any():
            return df

        if df.index.is_monotonic_increasing:
            cum_discont = np.cumsum(is_discont, dtype=np.int64)
            return df[cum_discont == cum_discont[start_idx]]

        # Fallback for synthetic DataFrames with non-monotonic index labels within cards
        from fsrs_optimizer import remove_non_continuous_rows

        # pyrefly: ignore [bad-index]
        return df.groupby("card_id", as_index=False, group_keys=False)[
            df.columns
        ].apply(remove_non_continuous_rows)

    def _model_specific_postprocessing(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Model-specific post processing

        Args:
            df: Dataframe after common post processing

        Returns:
            Dataframe with model-specific post processing
        """
        return df
