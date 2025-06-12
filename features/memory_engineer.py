import pandas as pd
from typing import List, Tuple
from .base import BaseFeatureEngineer


class SM2FeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for SM2 model
    SM2 only needs rating history as a string sequence
    """
    
    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create SM2 features: rating history as string sequence
        """
        # SM2 uses the r_history field which is already created in base preprocessing
        df["sequence"] = df["r_history"]
        return df


class EbisuFeatureEngineer(BaseFeatureEngineer):
    """
    Feature engineer for Ebisu models
    Ebisu needs (time_interval, rating) tuples for each review
    """
    
    def _model_specific_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create Ebisu features: sequence of (time_interval, rating) tuples
        """
        t_history_list, r_history_list = self.get_history_lists(df)
        
        # Create tuple sequences for Ebisu
        df["sequence"] = [
            tuple(zip(t_item[:-1], r_item[:-1]))
            for t_sublist, r_sublist in zip(t_history_list, r_history_list)
            for t_item, r_item in zip(t_sublist, r_sublist)
        ]
        
        return df 