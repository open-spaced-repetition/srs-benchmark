import argparse
from pathlib import Path
try:
    import tomllib as _toml  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover
    import tomli as _toml  # type: ignore
from argparse import Namespace

import torch


def parse_toml():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Location of the toml file")
    args, _ = parser.parse_known_args()
    with open(args.config, "rb") as f:
        args = _toml.load(f)
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
