"""
Parses TOML configuration files for the RWKV training and processing scripts.

This module provides a utility function to load settings from a TOML file,
perform necessary type conversions for specific keys (like PyTorch dtypes,
devices, and Path objects), and return the configuration as an argparse.Namespace.
"""
import argparse
from pathlib import Path
import tomli
from argparse import Namespace

import torch


def parse_toml():
    """
    Parses a TOML configuration file specified by the '--config' command-line argument.

    The function performs the following conversions:
    - "DTYPE": Converts string "bfloat16" to `torch.bfloat16`, "float" or "float32" to `torch.float32`.
    - "DEVICE": Converts string device name (e.g., "cuda:0") to `torch.device` object.
    - "DATA_PATH": Converts string path to `pathlib.Path` object.

    Returns:
        argparse.Namespace: An object containing all configuration settings from the TOML file,
                            with appropriate type conversions applied.

    Raises:
        ValueError: If an unsupported "DTYPE" string is provided.
        FileNotFoundError: If the specified TOML configuration file does not exist.
        tomli.TOMLDecodeError: If the TOML file is malformed.
    """
    parser = argparse.ArgumentParser(description="Parse TOML configuration file for RWKV training.")
    parser.add_argument("--config", required=True, help="Path to the TOML configuration file.")
    # Parse only the --config argument first to get the file path
    # Other arguments in the TOML file might conflict if parsed directly by argparse from cmd line
    config_args, _ = parser.parse_known_args()

    with open(config_args.config, "rb") as f:
        args_dict = tomli.load(f) # Load all TOML content into a dictionary

        # Perform type conversions for specific keys
        if "DTYPE" in args_dict:
            if args_dict["DTYPE"] == "bfloat16":
                args_dict["DTYPE"] = torch.bfloat16
            elif args_dict["DTYPE"] in ("float", "float32"):
                args_dict["DTYPE"] = torch.float32
            else:
                raise ValueError("Not currently supported DTYPE:", args_dict["DTYPE"])
        if "DEVICE" in args_dict:
            args_dict["DEVICE"] = torch.device(args_dict["DEVICE"])
        if "DATA_PATH" in args_dict:
            args_dict["DATA_PATH"] = Path(args_dict["DATA_PATH"])

        return Namespace(**args_dict) # Convert dictionary to Namespace object
    with open(args.config, "rb") as f:
        args = tomli.load(f)
        if "DTYPE" in args:
            if args["DTYPE"] == "bfloat16":
                args["DTYPE"] = torch.bfloat16
            elif args["DTYPE"] in ("float", "float32"):
                args["DTYPE"] = torch.float32
            else:
                raise ValueError("Not currently supported:", args["DTYPE"])
        if "DEVICE" in args:
            args["DEVICE"] = torch.device(args["DEVICE"])
        if "DATA_PATH" in args:
            args["DATA_PATH"] = Path(args["DATA_PATH"])
        return Namespace(**args)
