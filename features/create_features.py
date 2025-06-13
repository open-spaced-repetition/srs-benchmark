import copy
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from .factory import create_feature_engineer
from config import Config


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


def _create_features_with_equalized_test(df: pd.DataFrame, config: Config) -> pd.DataFrame:
    """
    Create features with equalized test handling for seconds intervals
    This ensures test sets are consistent between seconds and non-seconds intervals

    Args:
        df: Input dataframe
        config: Configuration object

    Returns:
        Processed dataframe with split indicators
    """
    # Create non-seconds features
    config_non_secs = copy.deepcopy(config)
    config_non_secs.use_secs_intervals = False
    engineer_non_secs = create_feature_engineer(config_non_secs)
    df_non_secs = engineer_non_secs.create_features(df.copy())

    # Create seconds features
    config_secs = copy.deepcopy(config)
    config_secs.use_secs_intervals = True
    engineer_secs = create_feature_engineer(config_secs)
    df_secs = engineer_secs.create_features(df.copy())

    # Find intersection of processed data
    df_intersect = df_secs[df_secs["review_th"].isin(df_non_secs["review_th"])]

    # Validate that required fields match between non-seconds and seconds versions
    assert len(df_intersect) == len(df_non_secs), (
        "Length mismatch between seconds and non-seconds data"
    )
    assert np.equal(df_intersect["i"], df_non_secs["i"]).all(), "Review count mismatch"
    assert np.equal(df_intersect["t_history"], df_non_secs["t_history"]).all(), (
        "Time history mismatch"
    )
    assert np.equal(df_intersect["r_history"], df_non_secs["r_history"]).all(), (
        "Rating history mismatch"
    )

    # Create train/test split indicators for time series cross-validation
    tscv = TimeSeriesSplit(n_splits=config.n_splits)
    for split_i, (_, non_secs_test_index) in enumerate(tscv.split(df_non_secs)):
        non_secs_test_set = df_non_secs.iloc[non_secs_test_index]

        # For train set: only allow reviews before the smallest review_th in test set
        allowed_train = df_secs[
            df_secs["review_th"] < non_secs_test_set["review_th"].min()
        ]
        df_secs[f"{split_i}_train"] = df_secs["review_th"].isin(
            allowed_train["review_th"]
        )

        # For test set: only allow reviews that exist in non_secs_test_set
        df_secs[f"{split_i}_test"] = df_secs["review_th"].isin(
            non_secs_test_set["review_th"]
        )

    return df_secs
