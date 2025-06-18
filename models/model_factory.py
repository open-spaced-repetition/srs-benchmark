from typing import Dict, Type, Union, List, Optional
import torch
from config import Config, ModelName

from models import *
from models.base import BaseModel


MODEL_REGISTRY: dict[ModelName, Type[BaseModel]] = {
    "FSRSv1": FSRS1,
    "FSRSv2": FSRS2,
    "FSRSv3": FSRS3,
    "FSRSv4": FSRS4,
    "FSRS-4.5": FSRS4dot5,
    "FSRS-5": FSRS5,
    "FSRS-6": FSRS6,
    "HLR": HLR,
    "ACT-R": ACT_R,
    "DASH": DASH,
    "DASH[MCM]": DASH,  # DASH class handles the MCM variant via config.model_name
    "DASH[ACT-R]": DASH_ACTR,
    "SM2-trainable": SM2,
    "Anki": Anki,
    "RNN": RNN,
    "GRU": RNN,  # GRU uses the RNN class definition as per original script
    "LSTM": LSTM,
    "GRU-P": GRU_P,
    "Transformer": Transformer,
    "NN-17": NN_17,
    "90%": ConstantModel,
}

# Models NOT handled by this factory due to distinct processing flows in the original script:
# - "SM2" (the non-trainable version)
# - "Ebisu-v2"
# - "AVG"
# - "RMSE-BINS-EXPLOIT"


def create_model(
    config: Config,
    model_params: Optional[Union[List[float], Dict[str, torch.Tensor], float]] = None,
) -> BaseModel:
    """
    Creates and returns an instance of the specified model.

    Args:
        config: The application configuration object.
        model_params: Optional parameters for model initialization.
                      - List[float]: For FSRS-like models' 'w' parameter.
                      - Dict[str, Tensor]: For neural models' state_dict.
                      - float: For ConstantModel's value.
                      If None, default initialization is used.

    Returns:
        An initialized nn.Module instance, moved to the device specified in config.

    Raises:
        ValueError: If model_name is not supported by the factory.
        TypeError: If model_params are of an incorrect type for the model.
    """
    model_name = config.model_name
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Model '{model_name}' is not supported by the model factory. "
            f"Supported models: {list(MODEL_REGISTRY.keys())}"
        )

    model_cls = MODEL_REGISTRY[model_name]
    instance: BaseModel

    # Common arguments for all model constructors
    constructor_kwargs = {"config": config}

    if model_name == "90%":  # ConstantModel
        value = 0.9
        if model_params is not None:
            if not isinstance(model_params, (float, int)):
                raise TypeError(
                    f"For {model_name}, model_params must be a float/int, got {type(model_params)}"
                )
            value = float(model_params)
        instance = model_cls(value=value, **constructor_kwargs)  # type: ignore

    elif hasattr(
        model_cls, "init_w"
    ):  # FSRS-like models, HLR, ACT_R, DASH, SM2Trainable, Anki
        if model_params is not None:
            if not isinstance(model_params, list) or not all(
                isinstance(p, (float, int)) for p in model_params
            ):
                raise TypeError(
                    f"For {model_name}, model_params must be a List[float] or None, got {type(model_params)}"
                )
            constructor_kwargs["w"] = model_params  # type: ignore
        # Don't add 'w' to constructor_kwargs when model_params is None
        # This allows models to use their default parameter values
        instance = model_cls(**constructor_kwargs)  # type: ignore

    elif model_name in [
        "RNN",
        "GRU",
        "LSTM",
        "GRU-P",
        "Transformer",
        "NN-17",
    ]:  # Neural nets
        if model_params is not None:
            if not isinstance(model_params, dict):
                raise TypeError(
                    f"For {model_name}, model_params must be a state_dict (dict) or None, got {type(model_params)}"
                )
            constructor_kwargs["state_dict"] = model_params  # type: ignore
        else:
            constructor_kwargs["state_dict"] = None  # type: ignore
        instance = model_cls(**constructor_kwargs)  # type: ignore

    else:
        # This case should ideally not be reached if all registered models are handled.
        raise ValueError(f"Unhandled instantiation logic for model: {model_name}")

    return instance.to(config.device)
