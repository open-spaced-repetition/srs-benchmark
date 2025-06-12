from .base import BaseFeatureEngineer
from .fsrs_engineer import FSRSFeatureEngineer
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
from config import Config


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
    # Get model name from config
    model_name = config.model_name

    # FSRS family and similar models that use standard tensor format
    if model_name.startswith("FSRS") or model_name in (
        "RNN",
        "GRU",
        "Transformer",
        "SM2-trainable",
        "Anki",
        "90%",
    ):
        return FSRSFeatureEngineer(config)

    # LSTM model with special features
    elif model_name == "LSTM":
        return LSTMFeatureEngineer(config)

    # GRU-P model with shifted time intervals
    elif model_name == "GRU-P":
        return GRUPFeatureEngineer(config)

    # HLR model with success/failure counts
    elif model_name == "HLR":
        return HLRFeatureEngineer(config)

    # ACT-R model with cumulative time features
    elif model_name == "ACT-R":
        return ACTRFeatureEngineer(config)

    # DASH variants
    elif model_name == "DASH":
        return DashFeatureEngineer(config)
    elif model_name == "DASH[MCM]":
        return DashMCMFeatureEngineer(config)
    elif model_name == "DASH[ACT-R]":
        return DashACTRFeatureEngineer(config)

    # NN-17 model with lapse history
    elif model_name == "NN-17":
        return NN17FeatureEngineer(config)

    # Memory models that don't use tensors
    elif model_name == "SM2":
        return SM2FeatureEngineer(config)
    elif model_name.startswith("Ebisu"):
        return EbisuFeatureEngineer(config)

    # Simple models that only need basic features
    elif model_name == "AVG":
        return AVGFeatureEngineer(config)
    elif model_name == "RMSE-BINS-EXPLOIT":
        return RMSEBinsExploitFeatureEngineer(config)

    else:
        raise ValueError(f"Unsupported model name: {model_name}")


def get_supported_models():
    """
    Get list of all supported model names

    Returns:
        List of supported model names
    """
    return [
        # FSRS family
        "FSRSv1",
        "FSRSv2",
        "FSRSv3",
        "FSRSv4",
        "FSRS-4.5",
        "FSRS-5",
        "FSRS-6",
        # Neural networks
        "RNN",
        "GRU",
        "GRU-P",
        "LSTM",
        "Transformer",
        "NN-17",
        # Memory models
        "SM2",
        "SM2-trainable",
        "Ebisu-v2",
        "HLR",
        "ACT-R",
        "Anki",
        # DASH variants
        "DASH",
        "DASH[MCM]",
        "DASH[ACT-R]",
        # Other models
        "AVG",
        "RMSE-BINS-EXPLOIT",
        "90%",
    ]
