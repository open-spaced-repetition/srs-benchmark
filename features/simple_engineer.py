import pandas as pd
from .base import BaseFeatureEngineer


class AVGFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for AVG model
    AVG model only needs basic features (y labels) for computing averages
    """
    
    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        AVG model doesn't need any special features beyond the basic preprocessing
        The model just computes the average of y labels in training data
        """
        # No additional features needed for AVG model
        return df


class RMSEBinsExploitFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for RMSE-BINS-EXPLOIT model
    This model needs basic features plus the ability to create bins from r_history, t_history, and i
    """
    
    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        RMSE-BINS-EXPLOIT model needs basic features for bin creation
        The bins are created from (lapse_count, delta_t, i) tuples
        """
        # No additional tensor features needed, just the basic preprocessing
        # The bin creation logic is handled in the model's predict/adapt methods
        return df 