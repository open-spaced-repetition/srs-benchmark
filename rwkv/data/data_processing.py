"""
Handles the processing of Anki review log data into a format suitable for training RWKV models.

This includes:
- Loading raw data from Parquet files.
- Extensive feature engineering (scaling, transformations, one-hot encoding).
- Handling missing values and special cases (e.g., NaN note_ids).
- Creating `RWKVSample` objects containing tensors for model input.
- Utilizing multiprocessing to parallelize data processing per user.
- Saving processed samples to an LMDB database.
"""
from dataclasses import dataclass
import json
import multiprocessing
from io import BytesIO
from tqdm import tqdm

import lmdb
import numpy as np
import torch
import pandas as pd
import random
from rwkv.config import RWKV_SUBMODULES
from rwkv.parse_toml import parse_toml
from rwkv.utils import save_tensor, load_tensor

# Defines the names of the columns that will be used as card features for the model.
# These features are typically numerical and represent various aspects of the review history.
CARD_FEATURE_COLUMNS = [
    "scaled_elapsed_days",  # Scaled time since last review in days
    "scaled_elapsed_days_cumulative",  # Scaled cumulative time elapsed for the card in days
    "scaled_elapsed_seconds",  # Scaled time since last review in seconds
    "elapsed_seconds_sin",  # Sinusoidal transformation of elapsed seconds (captures daily cycle)
    "elapsed_seconds_cos",  # Cosine transformation of elapsed seconds (captures daily cycle)
    "scaled_elapsed_seconds_cumulative",  # Scaled cumulative time elapsed for the card in seconds
    "elapsed_seconds_cumulative_sin",  # Sinusoidal transformation of cumulative elapsed seconds
    "elapsed_seconds_cumulative_cos",  # Cosine transformation of cumulative elapsed seconds
    "scaled_duration",  # Scaled duration of the review
    "rating_1",  # One-hot encoded: rating was 1 (e.g., "Again")
    "rating_2",  # One-hot encoded: rating was 2 (e.g., "Hard")
    "rating_3",  # One-hot encoded: rating was 3 (e.g., "Good")
    "rating_4",  # One-hot encoded: rating was 4 (e.g., "Easy")
    "note_id_is_nan",  # Boolean indicating if the original note_id was NaN
    "deck_id_is_nan",  # Boolean indicating if the original deck_id was NaN
    "preset_id_is_nan",  # Boolean indicating if the original preset_id was NaN
    "day_offset_diff",  # Difference in day offset from the previous review
    "day_of_week",  # Day of the week of the review
    "diff_new_cards",  # Difference in the count of new cards seen up to this review for this card
    "diff_reviews",  # Difference in the review count for this card
    "cum_new_cards_today",  # Cumulative new cards reviewed today by the user
    "cum_reviews_today",  # Cumulative reviews (new or old) by the user today
    "scaled_state",  # Scaled representation of the card's state (e.g., new, learning, review)
    "is_query",  # Boolean indicating if this is a query record (for inference/prediction)
]

# Precomputed statistics for scaling various features.
# These are likely derived from a larger dataset to standardize inputs.
STATISTICS = {
    "elapsed_days_mean": 1.51,
    "elapsed_days_std": 1.62,
    "elapsed_days_cumulative_mean": 2.14,
    "elapsed_days_cumulative_std": 2.25,
    "elapsed_seconds_mean": 9.96,
    "elapsed_seconds_std": 5.21,
    "elapsed_seconds_cumulative_mean": 10.86,
    "elapsed_seconds_cumulative_std": 5.8,
    "duration_mean": 8.9,
    "duration_std": 1.07,
    "diff_new_cards_mean": 2.945,
    "diff_new_cards_std": 2.011,
    "diff_reviews_mean": 4.64,
    "diff_reviews_std": 2.59,
    "cum_new_cards_today_mean": 2.55,
    "cum_new_cards_today_std": 1.41,
    "cum_reviews_today_mean": 4.59,
    "cum_reviews_today_std": 1.30,
}

# A large prime number used as a placeholder for NaN IDs (note_id, deck_id, preset_id)
# to ensure they get a unique embedding representation if treated as categorical IDs.
ID_PLACEHOLDER = 314159265358979323


def scale_elapsed_days(x):
    "scaled_elapsed_seconds",
    "elapsed_seconds_sin",
    "elapsed_seconds_cos",
    "scaled_elapsed_seconds_cumulative",
    "elapsed_seconds_cumulative_sin",
    "elapsed_seconds_cumulative_cos",
    "scaled_duration",
    "rating_1",
    "rating_2",
    "rating_3",
    "rating_4",
    "note_id_is_nan",
    "deck_id_is_nan",
    "preset_id_is_nan",
    "day_offset_diff",
    "day_of_week",
    "diff_new_cards",
    "diff_reviews",
    "cum_new_cards_today",
    "cum_reviews_today",
    "scaled_state",
    "is_query",
]

STATISTICS = {
    "elapsed_days_mean": 1.51,
    "elapsed_days_std": 1.62,
    "elapsed_days_cumulative_mean": 2.14,
    "elapsed_days_cumulative_std": 2.25,
    "elapsed_seconds_mean": 9.96,
    "elapsed_seconds_std": 5.21,
    "elapsed_seconds_cumulative_mean": 10.86,
    "elapsed_seconds_cumulative_std": 5.8,
    "duration_mean": 8.9,
    "duration_std": 1.07,
    "diff_new_cards_mean": 2.945,
    "diff_new_cards_std": 2.011,
    "diff_reviews_mean": 4.64,
    "diff_reviews_std": 2.59,
    "cum_new_cards_today_mean": 2.55,
    "cum_new_cards_today_std": 1.41,
    "cum_reviews_today_mean": 4.59,
    "cum_reviews_today_std": 1.30,
}

ID_PLACEHOLDER = 314159265358979323


# --- Generic Scaling Helpers ---

def _scale_log_epsilon_standardize(x, mean_key, std_key, epsilon=1e-5):
    """
    Generic scaling: applies log(x + epsilon) transformation and then standardizes.
    Handles x = -1 by effectively treating log(x + epsilon) as 0 for x=-1 before applying mean/std,
    preventing issues with log(-1 + epsilon) if epsilon is small.

    Args:
        x: Input array or value.
        mean_key: Key in STATISTICS dictionary for the mean.
        std_key: Key in STATISTICS dictionary for the standard deviation.
        epsilon: Small constant to avoid log(0).

    Returns:
        Scaled array or value.
    """
    return (
        np.where(x == -1, 0, np.log(x + epsilon)) - STATISTICS[mean_key]
    ) / STATISTICS[std_key]

def _scale_log_const_standardize(x, const, mean_key, std_key):
    """
    Generic scaling: applies log(x + const) transformation and then standardizes.

    Args:
        x: Input array or value.
        const: Constant added to x before log.
        mean_key: Key in STATISTICS dictionary for the mean.
        std_key: Key in STATISTICS dictionary for the standard deviation.

    Returns:
        Scaled array or value.
    """
    return (np.log(x + const) - STATISTICS[mean_key]) / STATISTICS[std_key]


# --- Feature Specific Scaling Functions ---

def scale_elapsed_days(x):
    """Scales 'elapsed_days' feature."""
    return _scale_log_epsilon_standardize(x, "elapsed_days_mean", "elapsed_days_std")


def scale_elapsed_days_cumulative(x):
    """Scales 'elapsed_days_cumulative' feature."""
    return _scale_log_epsilon_standardize(x, "elapsed_days_cumulative_mean", "elapsed_days_cumulative_std")


def scale_elapsed_seconds(x):
    """Scales 'elapsed_seconds' feature."""
    return _scale_log_epsilon_standardize(x, "elapsed_seconds_mean", "elapsed_seconds_std")


def scale_elapsed_seconds_cumulative(x):
    """Scales 'elapsed_seconds_cumulative' feature."""
    return _scale_log_epsilon_standardize(x, "elapsed_seconds_cumulative_mean", "elapsed_seconds_cumulative_std")


def scale_duration(x):
    """Scales 'duration' feature."""
    return _scale_log_const_standardize(x, 10, "duration_mean", "duration_std")


def scale_diff_new_cards(x):
    """Scales 'diff_new_cards' feature."""
    return _scale_log_const_standardize(x, 3, "diff_new_cards_mean", "diff_new_cards_std")


def scale_diff_reviews(x):
    """Scales 'diff_reviews' feature."""
    return _scale_log_const_standardize(x, 3, "diff_reviews_mean", "diff_reviews_std")


def scale_cum_new_cards_today(x):
    """Scales 'cum_new_cards_today' feature."""
    return _scale_log_const_standardize(x, 3, "cum_new_cards_today_mean", "cum_new_cards_today_std")


def scale_cum_reviews_today(x):
    """Scales 'cum_reviews_today' feature."""
    return _scale_log_const_standardize(x, 3, "cum_reviews_today_mean", "cum_reviews_today_std")


def scale_state(x):
    """
    Scales 'state' feature by subtracting 2.
    Likely to center states (e.g., 0,1,2,3 -> -2,-1,0,1).
    """
    return x - 2


def scale_day_offset_diff(x):
    """
    Scales 'day_offset_diff' feature using a double logarithm.
    This is a strong transformation for highly skewed data.
    """
    return np.log(np.log(np.e + x))


def base_transform_elapsed_days(df):
    return df.assign(scaled_elapsed_days=scale_elapsed_days(df["elapsed_days"]))


def base_transform_elapsed_days_cumulative(df):
    return df.assign(
        scaled_elapsed_days_cumulative=scale_elapsed_days_cumulative(
            df["elapsed_days_cumulative"]
        )
    )


def base_transform_elapsed_seconds(df):
    return df.assign(
        scaled_elapsed_seconds=scale_elapsed_seconds(df["elapsed_seconds"])
    )


def base_transform_elapsed_seconds_cumulative(df):
    return df.assign(
        scaled_elapsed_seconds_cumulative=scale_elapsed_seconds_cumulative(
            df["elapsed_seconds_cumulative"]
        )
    )


def base_transform_duration(df):
    return df.assign(scaled_duration=scale_duration(df["duration"]))


def base_transform_diff_new_cards(df):
    return df.assign(diff_new_cards=scale_diff_new_cards(df["diff_new_cards"]))


def base_transform_diff_reviews(df):
    return df.assign(diff_reviews=scale_diff_reviews(df["diff_reviews"]))


def base_transform_cum_new_cards_today(df):
    return df.assign(
        cum_new_cards_today=scale_cum_new_cards_today(df["cum_new_cards_today"])
    )


def base_transform_cum_reviews_today(df):
    return df.assign(cum_reviews_today=scale_cum_reviews_today(df["cum_reviews_today"]))


def base_transform_state(df):
    return df.assign(scaled_state=scale_state(df["state"]))


def base_transform_day_offset_diff(df):
    return df.assign(day_offset_diff=scale_day_offset_diff(df["day_offset_diff"]))


def add_segment_features(df, equalize_review_ths=[]):
    """
    Adds features that depend on the specific segment of data being processed.
    This typically means features relative to the start of the current segment.

    Args:
        df: Pandas DataFrame of review logs for a segment.
        equalize_review_ths: List of review thresholds used for equalization,
                             though not directly used in this function, kept for API consistency.

    Returns:
        Pandas DataFrame with added segment-specific features.
    """
    df["day_offset"] = df["day_offset"] - df["day_offset"].min()
    df["day_offset_first"] = df.groupby("card_id")["day_offset"].transform("first")
    df["day_of_week"] = ((df["day_offset"] % 7) - 3) / 3
    return df


def get_rwkv_data(data_path, user_id, equalize_review_ths=[]):
    """
    Loads and preprocesses review log data for a given user from Parquet files.

    This function performs several key steps:
    1. Loads review logs, card details, and deck details.
    2. Merges these DataFrames.
    3. Handles missing IDs (note_id, deck_id, preset_id) by filling them with placeholders.
    4. Generates numerous features:
        - Review sequence features (e.g., 'i', 'is_first_review').
        - Cumulative statistics (e.g., 'cum_new_cards', 'elapsed_days_cumulative').
        - Time-based cyclical features (e.g., 'elapsed_seconds_sin', 'elapsed_seconds_cos').
        - Label generation for supervised learning (e.g., 'label_elapsed_seconds', 'label_y').
    5. Applies scaling transformations to numerical features.
    6. Adds one-hot encoded rating features.
    7. Calls `add_segment_features` for segment-specific processing.

    Args:
        data_path: Path to the directory containing the Parquet data files ('revlogs', 'cards', 'decks').
        user_id: The ID of the user whose data is to be processed.
        equalize_review_ths: List of review thresholds used for equalization, passed to `add_queries`.

    Returns:
        A Pandas DataFrame containing the fully preprocessed review data for the user.
    """
    df = pd.read_parquet(data_path / "revlogs", filters=[("user_id", "=", user_id)])
    df_len = len(df)
    df["review_th"] = range(1, df.shape[0] + 1)
    df_cards = pd.read_parquet(data_path / "cards", filters=[("user_id", "=", user_id)])
    df_cards.drop(columns=["user_id"], inplace=True)
    df_decks = pd.read_parquet(data_path / "decks", filters=[("user_id", "=", user_id)])
    df_decks.drop(columns=["user_id", "parent_id"], inplace=True)
    df = df.merge(df_cards, on="card_id", how="left", validate="many_to_one")
    df = df.merge(df_decks, on="deck_id", how="left", validate="many_to_one")
    assert len(df) == df_len
    assert df["day_offset"].is_monotonic_increasing
    assert df["duration"].min() >= 0

    # find cards with a nan note_id and fill them with a unique value individually
    card_id_nan_mask = df["note_id"].isna()
    df["note_id_is_nan"] = card_id_nan_mask.astype(int)
    card_id_nan_card_ids = df[card_id_nan_mask]["card_id"]
    df.loc[card_id_nan_mask, "note_id"] = ID_PLACEHOLDER + card_id_nan_card_ids
    assert not df["note_id"].isna().any()

    # find cards with a nan deck_id and fill them all to a new unique value to pretend as though these cards are all in the same deck
    deck_id_nan_mask = df["deck_id"].isna()
    df["deck_id_is_nan"] = deck_id_nan_mask.astype(int)
    df.loc[deck_id_nan_mask, "deck_id"] = ID_PLACEHOLDER

    # same thing for nan preset_id
    preset_id_nan_mask = df["preset_id"].isna()
    df["preset_id_is_nan"] = preset_id_nan_mask.astype(int)
    df.loc[preset_id_nan_mask, "preset_id"] = ID_PLACEHOLDER
    df["i"] = df.groupby("card_id").cumcount() + 1
    df["y"] = df["rating"].map(lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x])
    df["is_first_review"] = (df["elapsed_days"] == -1).astype(int)
    df["cum_new_cards"] = df["is_first_review"].cumsum()
    df["diff_new_cards"] = df.groupby("card_id")["cum_new_cards"].diff().fillna(0)
    df["diff_reviews"] = np.maximum(
        0, -1 + df.groupby("card_id")["review_th"].diff().fillna(0)
    )
    df["cum_new_cards_today"] = df.groupby("day_offset")["is_first_review"].cumsum()
    df["cum_reviews_today"] = df.groupby("day_offset").cumcount()
    df["elapsed_days_cumulative"] = df.groupby("card_id")["elapsed_days"].cumsum()
    df["elapsed_seconds_cumulative"] = df.groupby("card_id")["elapsed_seconds"].cumsum()
    SECONDS_PER_DAY = 86400
    df["elapsed_seconds_sin"] = np.sin(
        (df["elapsed_seconds"] % SECONDS_PER_DAY) * 2 * np.pi / SECONDS_PER_DAY
    )
    df["elapsed_seconds_cos"] = np.cos(
        (df["elapsed_seconds"] % SECONDS_PER_DAY) * 2 * np.pi / SECONDS_PER_DAY
    )
    df["elapsed_seconds_cumulative_sin"] = np.sin(
        (df["elapsed_seconds_cumulative"] % SECONDS_PER_DAY)
        * 2
        * np.pi
        / SECONDS_PER_DAY
    )
    df["elapsed_seconds_cumulative_cos"] = np.cos(
        (df["elapsed_seconds_cumulative"] % SECONDS_PER_DAY)
        * 2
        * np.pi
        / SECONDS_PER_DAY
    )
    pd.set_option("display.max_rows", 100)  # Increase number of rows shown
    pd.set_option("display.max_columns", 50)  # Increase number of columns shown

    assert df["day_offset"].is_monotonic_increasing
    assert all(df.groupby("card_id")["i"].diff().fillna(1) == 1)
    assert all(df.groupby("card_id")["i"].first() == 1)

    assert df["day_offset"].is_monotonic_increasing
    assert df["review_th"].is_monotonic_increasing

    df["has_label"] = (df.groupby("card_id").cumcount(ascending=False) != 0).astype(int)
    df[
        [
            "label_elapsed_seconds",
            "label_elapsed_days",
            "label_y",
            "label_rating",
            "label_review_th",
        ]
    ] = df.groupby("card_id")[
        ["elapsed_seconds", "elapsed_days", "y", "rating", "review_th"]
    ].shift(-1)
    # Pi as an obvious placeholder for NaN values
    df["label_elapsed_seconds"] = df["label_elapsed_seconds"].fillna(np.pi)
    df["label_elapsed_days"] = df["label_elapsed_days"].fillna(np.pi)
    df["label_y"] = df["label_y"].fillna(0)
    df["label_rating"] = df["label_rating"].fillna(1)
    df["label_is_equalize"] = (
        df["label_review_th"].isin(set(equalize_review_ths)).astype(int)
    )

    df = base_transform_elapsed_days(df)
    df = base_transform_elapsed_days_cumulative(df)
    df = base_transform_elapsed_seconds(df)
    df = base_transform_elapsed_seconds_cumulative(df)
    df = base_transform_duration(df)
    df = base_transform_diff_new_cards(df)
    df = base_transform_diff_reviews(df)
    df = base_transform_cum_new_cards_today(df)
    df = base_transform_cum_reviews_today(df)
    df = base_transform_state(df)
    df["day_offset_diff"] = df["day_offset"].diff().fillna(0)
    df = base_transform_day_offset_diff(df)
    df["is_query"] = 0

    rating_onehot = pd.get_dummies(df["rating"], prefix="rating", dtype=int)
    rating_onehot = rating_onehot.reindex(
        columns=[f"rating_{i}" for i in [1, 2, 3, 4]], fill_value=0
    )
    df = pd.concat([df, rating_onehot], axis=1)
    assert df["day_offset"].is_monotonic_increasing
    assert df["review_th"].is_monotonic_increasing

    df = add_segment_features(df.reset_index(drop=True).copy())
    return df.reset_index(drop=True)


@dataclass
class ModuleData:
    """
    Contains data structures needed for efficient batching of sequences
    with varying lengths for a specific RWKV submodule (e.g., 'card_id', 'deck_id').

    Attributes:
        split_len: NumPy array of sequence lengths present in the batch for this module.
        split_B: NumPy array of counts for each sequence length in split_len.
        from_perm: PyTorch tensor, a permutation to sort sequences by length.
        to_perm: PyTorch tensor, the inverse of from_perm, to restore original order.
    """
    split_len: np.array
    split_B: np.array
    from_perm: torch.Tensor
    to_perm: torch.Tensor  # the inverse of from_perm


@dataclass
class RWKVSample:
    """
    Represents a processed sample for training or inference with the RWKV model.
    It contains all necessary features and labels for a sequence of reviews.

    Attributes:
        user_id: The ID of the user this sample belongs to.
        start_th: The starting review_th (review count) for this sample.
        end_th: The ending review_th for this sample.
        length: The total number of reviews in this sample.
        card_features: Tensor of continuous features for each review.
        ids: Dictionary mapping submodule names (e.g., 'card_id') to tensors of their IDs.
        modules: Dictionary mapping submodule names to ModuleData objects for batching.
        global_labels: Tensor containing labels for each review (e.g., next interval, rating).
        review_ths: Tensor of review_th for each entry in the sequence.
        label_review_ths: Tensor of review_th for the *next* review (label context).
        day_offsets: Tensor of day offsets for each review.
        day_offsets_first: Tensor of the first day offset for the card of each review.
        skips: Boolean tensor indicating if a review is a 'skip' (query record).
    """
    user_id: int
    start_th: int
    end_th: int
    length: int
    card_features: torch.Tensor
    ids: dict[str, torch.Tensor]
    modules: dict[str, ModuleData]
    global_labels: torch.Tensor
    review_ths: torch.Tensor
    label_review_ths: torch.Tensor
    day_offsets: torch.Tensor
    day_offsets_first: torch.Tensor
    skips: torch.Tensor


def add_queries(section_df, equalize_review_ths):
    """
    Augments the DataFrame with 'query' rows for inference or "what-if" scenarios.

    For each non-first review, a 'query' row is created. This query row
    represents the state *before* the actual review occurred. It has most
    outcome-related features (like rating, duration, state) zeroed out or
    set to a default, and its 'label_*' fields are populated with the
    actual outcome of the review it precedes. These query rows are marked
    with `is_query=1` and `skip=True`.

    This allows the model to be queried for a prediction at a specific point
    in time, and then compare that prediction to the actual outcome.

    Args:
        section_df: Pandas DataFrame segment of review data.
        equalize_review_ths: List of review thresholds used to mark labels for equalization.

    Returns:
        Pandas DataFrame with added query rows, sorted by review_th and then 'skip' status.
    """
    section_df["skip"] = False
    # Selectively keep certain columns, zero out the remaining columns to avoid leakage
    # The kept and rejected columns are written explicitly here so that additions and removals of columns in the future must be deliberately checked here
    keep_columns = [
        "card_id",
        "day_offset",
        "day_offset_first",
        "elapsed_days",
        "elapsed_days_cumulative",
        "elapsed_seconds",
        "elapsed_seconds_sin",
        "elapsed_seconds_cos",
        "elapsed_seconds_cumulative",
        "elapsed_seconds_cumulative_sin",
        "elapsed_seconds_cumulative_cos",
        "user_id",
        "review_th",
        "note_id",
        "deck_id",
        "preset_id",
        "note_id_is_nan",
        "deck_id_is_nan",
        "preset_id_is_nan",
        "i",
        "is_first_review",
        "cum_new_cards",
        "diff_new_cards",
        "diff_reviews",
        "cum_new_cards_today",
        "cum_reviews_today",
        "scaled_elapsed_days",
        "scaled_elapsed_days_cumulative",
        "scaled_elapsed_seconds",
        "scaled_elapsed_seconds_cumulative",
        "day_offset_diff",
        "day_of_week",
        "is_query",
    ]

    reject_columns = [
        "rating",
        "duration",
        "scaled_duration",
        "state",  # TODO state can be kept(?)
        "scaled_state",
        "y",
        "rating_1",
        "rating_2",
        "rating_3",
        "rating_4",
        "skip",
        "has_label",
        "label_elapsed_seconds",
        "label_elapsed_days",
        "label_y",
        "label_rating",
        "label_review_th",
        "label_is_equalize",
    ]
    for column in keep_columns:
        assert column not in reject_columns
        assert column in section_df.columns, f"{column} not found"

    for column in reject_columns:
        assert column not in keep_columns
        assert column in section_df.columns, f"{column} not found"

    for column in section_df.columns:
        assert column in keep_columns or column in reject_columns, f"{column} not found"
    assert len(keep_columns) + len(reject_columns) == len(
        section_df.columns
    ), "Ensure that all columns are explicitly listed"

    query_df = section_df.copy()
    query_df = query_df[query_df["is_first_review"] == False]
    if len(query_df) > 0:
        query_rating = query_df["rating"]
        query_review_ths = query_df["review_th"]
        for column in reject_columns:
            query_df[column] = 0

        query_df["skip"] = True
        query_df["label_rating"] = query_rating
        query_df["label_y"] = query_df["label_rating"].map(
            lambda x: {1: 0, 2: 1, 3: 1, 4: 1}[x]
        )
        query_df["label_review_th"] = query_review_ths
        query_df["has_label"] = True
        query_df["is_query"] = 1
        query_df["label_is_equalize"] = (
            query_df["label_review_th"].isin(set(equalize_review_ths)).astype(int)
        )

        section_df = pd.concat([section_df, query_df], ignore_index=True)
        section_df = section_df.sort_values(
            by=["review_th", "skip"], ascending=[True, False]
        ).reset_index(drop=True)

    section_df["index"] = range(0, len(section_df))
    assert (
        section_df["label_rating"].min() >= 1 and section_df["label_rating"].max() <= 4
    )
    return section_df


def create_sample(
    user_id, section_df, equalize_review_ths, dtype, device
) -> RWKVSample:
    """
    Creates an RWKVSample object from a DataFrame section.

    This involves:
    1. Applying segment features and adding query rows.
    2. Extracting card features and converting them to a tensor.
    3. Extracting ID features for each submodule and converting them to tensors.
    4. Calculating ModuleData (split_len, split_B, permutations) for each submodule
       to enable efficient batching of variable-length sequences within the submodule.
    5. Extracting global labels and other relevant tensors (review_ths, day_offsets, skips).

    Args:
        user_id: The ID of the user.
        section_df: The input DataFrame segment for this sample.
        equalize_review_ths: List of review thresholds for equalization, passed to `add_queries`.
        dtype: The PyTorch dtype for the created tensors (e.g., torch.float32).
        device: The PyTorch device for the created tensors (e.g., 'cpu').

    Returns:
        An RWKVSample object containing all processed data as tensors.
    """
    section_df = add_segment_features(section_df.copy())
    section_df = section_df.reset_index(drop=True)
    section_df = add_queries(section_df.copy(), equalize_review_ths)

    card_features = section_df[CARD_FEATURE_COLUMNS]
    card_features_tensor = torch.tensor(
        card_features.to_numpy(), dtype=dtype, device=device, requires_grad=False
    )
    module_infos = {}
    ids = {}
    for submodule in RWKV_SUBMODULES:
        ids[submodule] = torch.tensor(
            section_df[submodule].to_numpy(),
            dtype=torch.int32,
            device=device,
            requires_grad=False,
        )

    for submodule in RWKV_SUBMODULES:
        section_df_groupby = section_df.groupby(submodule, observed=True)
        names = list(set(section_df[submodule].values))
        submodule_dfs = {name: section_df_groupby.get_group(name) for name in names}

        # Need to get subgroups by length and gather them
        # keep track of their locations within section_df
        map_len_to_locs_list = {}

        for submodule_i, group_df in submodule_dfs.items():
            key = len(group_df)
            if key not in map_len_to_locs_list:
                map_len_to_locs_list[key] = []
            map_len_to_locs_list[key].append(group_df["index"].values)

        # create inverse map based on the flattened concat
        locs = []
        split_len = []
        split_B = []
        for l in sorted(map_len_to_locs_list.keys()):
            if len(map_len_to_locs_list[l]) > 0:
                split_len.append(l)
                split_B.append(len(map_len_to_locs_list[l]))
            for group_locs in map_len_to_locs_list[l]:
                locs.extend(group_locs)

        locs_dict = {locs[i]: i for i in range(len(locs))}
        inv_locs = [locs_dict[i] for i in range(len(locs))]
        split_len = np.array(split_len, dtype=np.int32)
        split_B = np.array(split_B, dtype=np.int32)
        from_perm = torch.tensor(locs, dtype=torch.int32, device=device)
        to_perm = torch.tensor(inv_locs, dtype=torch.int32, device=device)
        module_infos[submodule] = ModuleData(
            split_len=split_len, split_B=split_B, from_perm=from_perm, to_perm=to_perm
        )

    global_labels_df = section_df[
        [
            "label_elapsed_seconds",
            "label_elapsed_days",
            "label_y",
            "label_rating",
            "has_label",
            "label_is_equalize",
            "is_query",
        ]
    ]
    global_labels_tensor = torch.tensor(
        global_labels_df.to_numpy(), dtype=dtype, device=device, requires_grad=False
    )
    label_review_ths_tensor = torch.tensor(
        section_df["label_review_th"].to_numpy(),
        dtype=torch.int32,
        device=device,
        requires_grad=False,
    )
    review_ths_tensor = torch.tensor(
        section_df["review_th"].to_numpy(),
        dtype=torch.int32,
        device=device,
        requires_grad=False,
    )
    day_offsets_tensor = torch.tensor(
        section_df["day_offset"].to_numpy(),
        dtype=torch.int32,
        device=device,
        requires_grad=False,
    )
    day_offsets_first_tensor = torch.tensor(
        section_df["day_offset_first"].to_numpy(),
        dtype=torch.int32,
        device=device,
        requires_grad=False,
    )
    skips_tensor = torch.tensor(
        section_df["skip"].to_numpy(),
        dtype=torch.bool,
        device=device,
        requires_grad=False,
    )
    return RWKVSample(
        user_id=user_id,
        start_th=int(section_df["review_th"].min()),
        end_th=int(section_df["review_th"].max()),
        length=len(section_df),
        modules=module_infos,
        ids=ids,
        card_features=card_features_tensor,
        global_labels=global_labels_tensor,
        review_ths=review_ths_tensor,
        label_review_ths=label_review_ths_tensor,
        day_offsets=day_offsets_tensor,
        day_offsets_first=day_offsets_first_tensor,
        skips=skips_tensor,
    )


def job(config, user_id, max_size, done, writer_queue, progress_queue):
    """
    The main processing job for a single user, executed by a multiprocessing Pool worker.

    Steps:
    1. Skips processing if user_id is known to be problematic (e.g., 4371) or already 'done'.
    2. Loads equalization review thresholds from an LMDB database.
    3. Calls `get_rwkv_data` to get the full processed DataFrame for the user.
    4. If `max_size` is None, creates a single `RWKVSample` from the entire DataFrame.
    5. If `max_size` is specified, splits the DataFrame into chunks of `max_size // 2`
       and creates an `RWKVSample` for each chunk. This is to manage memory for very long sequences.
    6. Puts the created `RWKVSample`(s) onto the `writer_queue` to be saved by the `save_job` process.
    7. Puts an item onto `progress_queue` to signal completion for this user.

    Args:
        config: Configuration object containing paths and parameters.
        user_id: The ID of the user to process.
        max_size: Maximum sequence length for a single sample. If exceeded, data is chunked.
        done: Boolean indicating if this user's data has already been processed and saved.
        writer_queue: Multiprocessing queue to send processed RWKVSamples to the saver process.
        progress_queue: Multiprocessing queue to report progress.
    """
    if user_id == 4371:
        print("Skipping user 4371. This user has no reviews in the 10k dataset.")
        progress_queue.put(1)
        return

    if done:
        print(f"User already done: {user_id}")
        progress_queue.put(1)
        return
    random.seed(user_id)
    torch.manual_seed(user_id)
    np.random.seed(user_id)

    LABEL_FILTER_DATASET = lmdb.open(
        config.LABEL_FILTER_LMDB_PATH, map_size=config.LABEL_FILTER_LMDB_SIZE
    )
    with LABEL_FILTER_DATASET.begin(write=False) as txn:
        # Removed local load_tensor, using rwkv.utils.load_tensor
        equalize_review_ths = load_tensor(txn, f"{user_id}_review_ths", "cpu").tolist()
    LABEL_FILTER_DATASET.close()

    df = get_rwkv_data(
        config.DATA_PATH, user_id, equalize_review_ths=equalize_review_ths
    )
    if max_size is None:
        sample = create_sample(
            user_id=user_id,
            section_df=df,
            equalize_review_ths=equalize_review_ths,
            dtype=config.DTYPE,
            device=torch.device("cpu"),
        )
        writer_queue.put(sample)
    else:
        allowable_size = max_size // 2
        ranges = []
        for start_i in range(0, len(df), allowable_size):
            end_i = min(len(df), start_i + allowable_size)
            ranges.append((start_i, end_i))

        for start_i, end_i in ranges:
            section_df = df.iloc[start_i:end_i]
            sample = create_sample(
                user_id=user_id,
                section_df=section_df,
                equalize_review_ths=equalize_review_ths,
                dtype=config.DTYPE,
                device=torch.device("cpu"),
            )
            writer_queue.put(sample)
            print("New", user_id, start_i, end_i, len(section_df))

    progress_queue.put(1)


def save_job(lmdb_path, lmdb_size, writer_queue):
    """
    A dedicated process that listens on a queue for RWKVSample objects
    and writes them to an LMDB database.

    For each sample, it saves:
    - Card features tensor.
    - Global labels tensor.
    - Various ID tensors (for each submodule).
    - ModuleData components (split_len, split_B, from_perm, to_perm for each submodule).
    - Other metadata tensors (review_ths, day_offsets, skips).
    It also maintains a JSON list of [start_th, end_th, length] for each sample
    under the key f"{user_id}_batches" and marks the user as done with f"{user_id}_done".

    Args:
        lmdb_path: Path to the LMDB database.
        lmdb_size: Maximum size for the LMDB database.
        writer_queue: Multiprocessing queue from which to receive RWKVSample objects.
                      Receiving `None` signals the end of processing.
    """
    env = lmdb.open(lmdb_path, lmdb_size)
    while True:
        rwkv_sample: RWKVSample = writer_queue.get()
        if rwkv_sample is None:
            break

        with env.begin(write=True) as txn:
            batches_key = f"{rwkv_sample.user_id}_batches"
            user_keys = txn.get(batches_key.encode())
            if user_keys is None:
                user_keys = []
            else:
                user_keys = json.loads(user_keys)

            key = [rwkv_sample.start_th, rwkv_sample.end_th, rwkv_sample.length]

            prefix = f"{rwkv_sample.user_id}_{rwkv_sample.start_th}-{rwkv_sample.end_th}_{rwkv_sample.length}_"
            card_features_key = prefix + "card_features"
            global_labels_key = prefix + "global_labels"
            label_review_ths_key = prefix + "label_review_ths"
            review_ths_key = prefix + "review_ths"
            day_offsets_key = prefix + "day_offsets"
            day_offsets_first_key = prefix + "day_offsets_first"
            skips_key = prefix + "skips"

            for name, ids in rwkv_sample.ids.items():
                save_tensor(txn, prefix + name + "_id_", ids)

            for name, module in rwkv_sample.modules.items():
                module_key = prefix + name + "_"
                save_tensor(
                    txn,
                    module_key + "split_len",
                    torch.tensor(module.split_len, dtype=torch.int32),
                )
                save_tensor(
                    txn,
                    module_key + "split_B",
                    torch.tensor(module.split_B, dtype=torch.int32),
                )
                save_tensor(txn, module_key + "from_perm", module.from_perm)
                save_tensor(txn, module_key + "to_perm", module.to_perm)

            save_tensor(txn, card_features_key, rwkv_sample.card_features)
            save_tensor(txn, global_labels_key, rwkv_sample.global_labels)
            save_tensor(txn, label_review_ths_key, rwkv_sample.label_review_ths)
            save_tensor(txn, review_ths_key, rwkv_sample.review_ths)
            save_tensor(txn, day_offsets_key, rwkv_sample.day_offsets)
            save_tensor(txn, day_offsets_first_key, rwkv_sample.day_offsets_first)
            save_tensor(txn, skips_key, rwkv_sample.skips)
            txn.put(f"{rwkv_sample.user_id}_done".encode(), "true".encode())

            user_keys.append(key)
            txn.put(batches_key.encode(), json.dumps(user_keys).encode())
    env.close()


def progress_tracker(total_items, progress_queue):
    """
    Tracks and displays data generation progress using tqdm.

    Args:
        total_items: The total number of items (users) to be processed.
        progress_queue: Multiprocessing queue from which to receive progress signals.
    """
    with tqdm(total=total_items, desc="Generating Data") as pbar:
        for _ in range(total_items):
            progress_queue.get()
            pbar.update(1)


def main(config):
    """
    Main function to orchestrate the data processing workflow.

    1. Initializes LMDB environment and identifies users already processed or to be skipped.
    2. Sets up multiprocessing:
        - A `writer_queue` and a `save_job` process to handle LMDB writes.
        - A `progress_queue` and a `progress_tracker` process for displaying progress.
        - A `multiprocessing.Pool` of worker processes to execute `job` for each user.
    3. Distributes `job` tasks to the worker pool for all unprocessed users.
    4. Waits for all jobs to complete, then signals the writer and progress tracker to terminate.
    """
    LMDB_PATH = config.LMDB_PATH
    LMDB_SIZE = config.LMDB_SIZE
    USER_IDS = list(range(config.USER_START, config.USER_END + 1))
    MAX_SIZE = config.MAX_BATCH_SIZE

    done_set = set()
    user_keys_dict = {}
    env = lmdb.open(LMDB_PATH, LMDB_SIZE)
    unprocessed_users = []
    with env.begin(write=False) as txn:
        for user_id in USER_IDS:
            batches_key = f"{user_id}_batches"
            user_keys = txn.get(batches_key.encode())
            if user_keys is None:
                user_keys = []
            else:
                user_keys = json.loads(user_keys)

            user_keys_dict[user_id] = user_keys
            if txn.get(f"{user_id}_done".encode()) is not None:
                done_set.add(user_id)
            else:
                unprocessed_users.append(user_id)
    env.close()
    print("Unprocessed users:", unprocessed_users)

    with multiprocessing.Manager() as manager:
        writer_queue = manager.Queue()
        writer = multiprocessing.Process(
            target=save_job, args=(LMDB_PATH, LMDB_SIZE, writer_queue)
        )
        writer.start()

        progress_queue = manager.Queue()
        progress_process = multiprocessing.Process(
            target=progress_tracker, args=(len(unprocessed_users), progress_queue)
        )
        progress_process.start()

        with multiprocessing.Pool(processes=9) as pool:
            pool.starmap(
                job,
                [
                    (
                        config,
                        user_id,
                        MAX_SIZE,
                        user_id in done_set,
                        writer_queue,
                        progress_queue,
                    )
                    for user_id in unprocessed_users
                ],
            )

        writer_queue.put(None)
        writer.join()
        progress_process.terminate()


if __name__ == "__main__":
    config = parse_toml()
    main(config)
