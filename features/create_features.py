import copy

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from config import Config

from .factory import create_feature_engineer


def create_features(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Refactored create_features function using the new feature engineering architecture

    Args:
        df: Input dataframe with review logs
        config: Configuration object containing model name and other parameters

    Returns:
        Processed dataframe with model-specific features
    """

    # Handle special case for equalized test with non-seconds
    if config.use_secs_intervals and config.equalize_test_with_non_secs:
        return _create_features_with_equalized_test(df, config)
    else:
        return _create_features_standard(df, config)


def _create_features_standard(df: pd.DataFrame, config) -> pd.DataFrame:
    """
    Standard feature creation using the appropriate feature engineer

    Args:
        df: Input dataframe
        config: Configuration object

    Returns:
        Processed dataframe
    """
    # Create the appropriate feature engineer
    feature_engineer = create_feature_engineer(config)

    # Create features using the feature engineer
    processed_df = feature_engineer.create_features(df)

    return processed_df


def _create_features_with_equalized_test(
    df: pd.DataFrame, config: Config
) -> pd.DataFrame:
    """
    Create features with equalized test handling for seconds intervals
    This ensures test sets are consistent between seconds and non-seconds intervals

    Args:
        df: Input dataframe
        config: Configuration object

    Returns:
        Processed dataframe with split indicators
    """
    # Shallow copies are safe because only scalar attributes are rebound below. Never
    # mutate shared mutable attributes (e.g. raw_args, cuda_device_ids) in place on a
    # copy: the change would leak into the caller's config. Attributes derived in
    # Config.__init__ from use_secs_intervals (s_min, base_file_name) are not recomputed.

    # Lightweight alignment pass for non-seconds (skips history strings & model tensors)
    config_non_secs = copy.copy(config)
    config_non_secs.use_secs_intervals = False
    engineer_non_secs = create_feature_engineer(config_non_secs)
    df_non_secs = engineer_non_secs.create_alignment_features(df)

    # Create full seconds features
    config_secs = copy.copy(config)
    config_secs.use_secs_intervals = True
    engineer_secs = create_feature_engineer(config_secs)
    df_secs = engineer_secs.create_features(df)

    # Set lapses for RMSE (bins) via index alignment
    df_secs["rmse_bins_lapse"] = df_non_secs["rmse_bins_lapse"]

    # Validate that required fields match between non-seconds and seconds versions
    secs_review_th = df_secs["review_th"].to_numpy(dtype=np.int64)
    non_secs_review_th = df_non_secs["review_th"].to_numpy(dtype=np.int64)
    max_th = int(secs_review_th.max()) if len(secs_review_th) > 0 else 0

    in_non_secs = np.zeros(max_th + 1, dtype=bool)
    in_non_secs[non_secs_review_th[non_secs_review_th <= max_th]] = True
    intersect_mask = in_non_secs[secs_review_th]

    assert int(intersect_mask.sum()) == len(df_non_secs), (
        "Length mismatch between seconds and non-seconds data"
    )
    assert np.equal(
        df_secs["i"].to_numpy()[intersect_mask], df_non_secs["i"].to_numpy()
    ).all(), "Review count mismatch"

    # Create train/test split indicators for time series cross-validation
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    in_test = np.zeros(max_th + 1, dtype=bool)
    for split_i, (_, non_secs_test_index) in enumerate(tscv.split(df_non_secs)):
        test_ths = non_secs_review_th[non_secs_test_index]
        min_test_th = test_ths.min()

        # For train set: only allow reviews before the smallest review_th in test set
        df_secs[f"{split_i}_train"] = secs_review_th < min_test_th

        # For test set: only allow reviews that exist in non_secs_test_set
        valid_test_ths = test_ths[test_ths <= max_th]
        in_test[valid_test_ths] = True
        df_secs[f"{split_i}_test"] = in_test[secs_review_th]
        in_test[valid_test_ths] = False

    return df_secs
