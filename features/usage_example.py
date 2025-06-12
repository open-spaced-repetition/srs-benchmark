"""
Example usage of the new feature engineering architecture

This file demonstrates how to use the refactored feature engineering system
to replace the original create_features_helper function calls.
"""

import pandas as pd
from features import create_features, create_feature_engineer, get_supported_models
import copy


def example_basic_usage(df: pd.DataFrame, config):
    """
    Basic example of using the new feature engineering system
    """
    # Example 1: Direct usage with create_features function
    # Model name is now read from config.model_name
    processed_df = create_features(df, config=config)
    print(f"Processed {len(processed_df)} rows for {config.model_name}")

    # Example 2: Using create_feature_engineer factory
    feature_engineer = create_feature_engineer(config)
    processed_df2 = feature_engineer.create_features(df)
    print(f"Processed {len(processed_df2)} rows using feature engineer")

    # Example 3: Different models (by modifying config)
    for model in ["LSTM", "DASH", "GRU-P", "HLR"]:
        try:
            # Create a temporary config with different model name
            temp_config = copy.deepcopy(config)
            temp_config.model_name = model
            processed_df = create_features(df, config=temp_config)
            print(f"Successfully processed {len(processed_df)} rows for {model}")
        except Exception as e:
            print(f"Error processing {model}: {e}")


def example_model_specific_features(df: pd.DataFrame, config):
    """
    Examples of different model-specific feature engineering
    """
    # FSRS models - create tensor features with time and rating history
    temp_config = copy.deepcopy(config)
    temp_config.model_name = "FSRSv4"
    fsrs_engineer = create_feature_engineer(temp_config)
    fsrs_df = fsrs_engineer.create_features(df)
    print(f"FSRS tensor shape example: {fsrs_df['tensor'].iloc[0].shape}")

    # LSTM model - includes additional features like new card counts
    temp_config.model_name = "LSTM"
    lstm_engineer = create_feature_engineer(temp_config)
    lstm_df = lstm_engineer.create_features(df)
    if "is_new_card" in lstm_df.columns:
        print(
            f"LSTM includes new card features: {lstm_df['is_new_card'].sum()} new cards"
        )

    # DASH model - uses time window features
    temp_config.model_name = "DASH"
    dash_engineer = create_feature_engineer(temp_config)
    dash_df = dash_engineer.create_features(df)
    print(f"DASH feature vector length: {dash_df['tensor'].iloc[0].shape[0]}")

    # Memory models - different feature formats
    temp_config.model_name = "SM2"
    sm2_engineer = create_feature_engineer(temp_config)
    sm2_df = sm2_engineer.create_features(df)
    print(f"SM2 sequence example: {sm2_df['sequence'].iloc[0]}")


def example_seconds_intervals(df: pd.DataFrame, config):
    """
    Example of using seconds intervals vs days intervals
    The secs_ivl setting is now controlled by config.use_secs_intervals
    """
    # Days intervals (config.use_secs_intervals = False)
    config.use_secs_intervals = False
    days_df = create_features(df, config=config)

    # Seconds intervals (config.use_secs_intervals = True)
    config.use_secs_intervals = True
    secs_df = create_features(df, config=config)

    print(f"Days intervals: {len(days_df)} rows")
    print(f"Seconds intervals: {len(secs_df)} rows")

    # Compare delta_t values
    if len(days_df) > 0 and len(secs_df) > 0:
        print(f"Days delta_t example: {days_df['delta_t'].iloc[0]}")
        print(f"Seconds delta_t example: {secs_df['delta_t'].iloc[0]}")


def example_replacing_original_code(df: pd.DataFrame, config):
    """
    Example of how to replace original create_features_helper calls
    """
    # OLD CODE (original):
    # from other import create_features_helper
    # processed_df = create_features_helper(df, model_name, secs_ivl=False)

    # NEW CODE (refactored):
    from features import create_features

    processed_df = create_features(df, config=config)

    return processed_df


def list_all_supported_models():
    """
    Show all supported models
    """
    supported_models = get_supported_models()
    print("Supported models:")
    for model in supported_models:
        print(f"  - {model}")

    return supported_models


if __name__ == "__main__":
    # This would typically be called with real data and config
    print("Feature Engineering Architecture Example")
    print("=" * 50)

    # List all supported models
    models = list_all_supported_models()

    # Note: In real usage, you would have:
    # - A real DataFrame with review logs
    # - A proper config object
    # - Proper error handling

    print("\nTo use this in your code:")
    print("1. Import: from feature_engineering import create_features")
    print("2. Call: processed_df = create_features(df, config=config)")
    print("3. Or use the factory: engineer = create_feature_engineer(config)")
    print("4. Then: processed_df = engineer.create_features(df)")
