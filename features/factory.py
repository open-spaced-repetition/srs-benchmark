from .base import BaseFeatureEngineer
from .fsrs_engineer import FSRSFeatureEngineer
from .fsrs_one_step_engineer import FSRSOneStepFeatureEngineer
from .lstm_engineer import LSTMFeatureEngineer
from .dash_engineer import (
    DashFeatureEngineer,
    DashMCMFeatureEngineer,
    DashACTRFeatureEngineer,
)
from .neural_engineer import (
    GRUPFeatureEngineer,
    HLRFeatureEngineer,
    ACTRFeatureEngineer,
    NN17FeatureEngineer,
)
from .memory_engineer import SM2FeatureEngineer, EbisuFeatureEngineer
from .simple_engineer import AVGFeatureEngineer, RMSEBinsExploitFeatureEngineer
from config import Config, ModelName
from typing import Type, get_args


FEATURE_ENGINEER_REGISTRY: dict[ModelName, Type[BaseFeatureEngineer]] = {
    # FSRS family and similar models that use standard tensor format
    "FSRSv1": FSRSFeatureEngineer,
    "FSRSv2": FSRSFeatureEngineer,
    "FSRSv3": FSRSFeatureEngineer,
    "FSRSv4": FSRSFeatureEngineer,
    "FSRS-4.5": FSRSFeatureEngineer,
    "FSRS-5": FSRSFeatureEngineer,
    "FSRS-6": FSRSFeatureEngineer,
    "FSRS-6-one-step": FSRSOneStepFeatureEngineer,
    "RNN": FSRSFeatureEngineer,
    "GRU": FSRSFeatureEngineer,
    "Transformer": FSRSFeatureEngineer,
    "SM2-trainable": FSRSFeatureEngineer,
    "Anki": FSRSFeatureEngineer,
    "90%": FSRSFeatureEngineer,
    # Specialized models
    "LSTM": LSTMFeatureEngineer,
    "GRU-P": GRUPFeatureEngineer,
    "HLR": HLRFeatureEngineer,
    "ACT-R": ACTRFeatureEngineer,
    "NN-17": NN17FeatureEngineer,
    # DASH variants
    "DASH": DashFeatureEngineer,
    "DASH[MCM]": DashMCMFeatureEngineer,
    "DASH[ACT-R]": DashACTRFeatureEngineer,
    # Memory models that don't use tensors
    "SM2": SM2FeatureEngineer,
    "Ebisu-v2": EbisuFeatureEngineer,
    # Simple models that only need basic features
    "AVG": AVGFeatureEngineer,
    "CONST": AVGFeatureEngineer,
    "MOVING-AVG": AVGFeatureEngineer,
    "RMSE-BINS-EXPLOIT": RMSEBinsExploitFeatureEngineer,
}


def create_feature_engineer(config: Config) -> BaseFeatureEngineer:
    """
    Factory function to create the appropriate feature engineer based on model name from config

    Args:
        config: Configuration object containing model_name and other settings

    Returns:
        Appropriate feature engineer instance

    Raises:
        ValueError: If config.model_name is not supported
    """
    model_name = config.model_name

    # Create and return the appropriate feature engineer
    feature_engineer_cls = FEATURE_ENGINEER_REGISTRY[model_name]
    return feature_engineer_cls(config)


def get_supported_models() -> tuple[str, ...]:
    """
    Get list of all supported model names

    Returns:
        List of supported model names
    """
    return get_args(ModelName)
