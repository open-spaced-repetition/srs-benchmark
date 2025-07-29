import argparse
import torch
from pathlib import Path
from typing import List, Optional, Literal, get_args

ModelName = Literal[
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


def create_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--processes", default=8, type=int, help="set the number of processes"
    )
    parser.add_argument("--dev", action="store_true", help="for local development")

    parser.add_argument(
        "--partitions",
        default="none",
        choices=["none", "deck", "preset"],
        help="use partitions instead of presets",
    )
    parser.add_argument(
        "--recency", action="store_true", help="enable recency weighting"
    )
    parser.add_argument(
        "--dry", action="store_true", help="evaluate default parameters"
    )

    # download revlogs from huggingface
    parser.add_argument(
        "--data",
        default="../anki-revlogs-10k",
        help="path to revlogs/*.parquet",
    )

    # short-term memory research
    parser.add_argument(
        "--secs", action="store_true", help="use elapsed_seconds as interval"
    )

    parser.add_argument(
        "--two_buttons", action="store_true", help="treat Hard and Easy as Good"
    )

    parser.add_argument(
        "--no_test_same_day",
        action="store_true",
        help="exclude reviews with elapsed_days=0 from testset",
    )
    parser.add_argument(
        "--no_train_same_day",
        action="store_true",
        help="exclude reviews with elapsed_days=0 from trainset",
    )

    parser.add_argument(
        "--equalize_test_with_non_secs",
        action="store_true",
        help="Only test with reviews that would be included in non-secs tests",
    )

    # save detailed results
    parser.add_argument("--raw", action="store_true", help="save raw predictions")
    parser.add_argument(
        "--file", action="store_true", help="save evaluation results to file"
    )
    parser.add_argument(
        "--plot", action="store_true", help="save evaluation plots to file"
    )

    # other.py only
    parser.add_argument("--algo", default="FSRSv3", help="algorithm name")
    parser.add_argument(
        "--short", action="store_true", help="include short-term reviews"
    )
    parser.add_argument(
        "--weights", action="store_true", help="save neural network weights"
    )

    # script.py only
    parser.add_argument(
        "--pretrain", action="store_true", help="FSRS-5 with only pretraining"
    )
    parser.add_argument(
        "--binary", action="store_true", help="FSRS-5 with binary ratings"
    )
    parser.add_argument("--rust", action="store_true", help="FSRS-rs")
    parser.add_argument(
        "--disable_short_term", action="store_true", help="disable short-term memory"
    )
    parser.add_argument(
        "--train_equals_test",
        action="store_true",
        help="Set train set equal to test set without splitting",
    )
    parser.add_argument(
        "--n_splits", type=int, default=5, help="Number of splits for TimeSeriesSplit"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size for training neural models",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=64,
        help="Maximum sequence length for batching inputs",
    )
    parser.add_argument(
        "--torch_num_threads",
        type=int,
        default=1,
        help="Number of threads for PyTorch intra-op parallelism",
    )
    return parser


class Config:
    """Holds all application configurations derived from command-line arguments and defaults."""

    def __init__(self, args: argparse.Namespace):
        # Store raw args for reference if needed, though direct access should be minimized
        self.raw_args: argparse.Namespace = args

        # Basic arguments from parser
        self.dev_mode: bool = args.dev
        self.dry_run: bool = args.dry
        self.model_name: ModelName = args.algo
        self.use_secs_intervals: bool = args.secs
        self.no_test_same_day: bool = args.no_test_same_day
        self.no_train_same_day: bool = args.no_train_same_day
        self.equalize_test_with_non_secs: bool = args.equalize_test_with_non_secs
        self.two_buttons: bool = args.two_buttons
        self.save_evaluation_file: bool = args.file
        self.generate_plots: bool = args.plot
        self.save_weights: bool = args.weights
        self.partitions: str = args.partitions
        self.save_raw_output: bool = args.raw
        self.num_processes: int = args.processes
        self.data_path: Path = Path(args.data)
        self.use_recency_weighting: bool = args.recency
        self.train_equals_test: bool = args.train_equals_test

        # Training/data parameters from parser (with defaults)
        self.n_splits: int = args.n_splits
        self.batch_size: int = args.batch_size
        self.max_seq_len: int = args.max_seq_len

        # Handle `include_short_term` based on model name and initial arg
        self.initial_short_term_setting: bool = args.short
        if self.model_name.startswith("FSRS-5") or self.model_name.startswith("FSRS-6"):
            self.include_short_term = True
        else:
            self.include_short_term = self.initial_short_term_setting

        # PyTorch threading settings
        self.torch_num_threads: int = args.torch_num_threads
        torch.set_num_threads(self.torch_num_threads)
        # if hasattr(torch, "set_num_interop_threads"):
        #     torch.set_num_interop_threads(args.torch_num_interop_threads)

        # Validate model name
        if self.model_name not in get_args(ModelName):
            raise ValueError(
                f"Model name '{self.model_name}' must be one of {get_args(ModelName)}"
            )

        # Path for fsrs_optimizer (used for dynamic import)
        self.fsrs_optimizer_module_path: str = "../fsrs-optimizer/src/fsrs_optimizer/"

        # Device configuration
        if torch.cuda.is_available() and self.model_name in [
            "GRU",
            "GRU-P",
            "LSTM",
            "RNN",
            "NN-17",
            "Transformer",
        ]:
            self.device: torch.device = torch.device("cuda")
        # elif torch.backends.mps.is_available(): # Support for MPS if uncommented in original
        #     self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Verbosity (can be made configurable later)
        self.verbose_logging: bool = False
        self.verbose_inadequate_data: bool = False

        # Derived file names
        _file_name_parts: list[str] = [self.model_name]
        if self.dry_run:
            _file_name_parts.append("-dry-run")
        if self.initial_short_term_setting:
            _file_name_parts.append("-short")
        if self.use_secs_intervals:
            _file_name_parts.append("-secs")
        if self.use_recency_weighting:
            _file_name_parts.append("-recency")
        if self.no_test_same_day:
            _file_name_parts.append("-no_test_same_day")
        if self.no_train_same_day:
            _file_name_parts.append("-no_train_same_day")
        if self.equalize_test_with_non_secs:
            _file_name_parts.append("-equalize_test_with_non_secs")
        if self.train_equals_test:
            _file_name_parts.append("-train_equals_test")
        if self.partitions != "none":
            _file_name_parts.append(f"-{self.partitions}")
        if self.dev_mode:
            _file_name_parts.append("-dev")

        self.base_file_name: str = "".join(_file_name_parts)
        self.optimizer_name_suffix: str = "_opt"

        # Stability (S) parameters
        _s_min_base = 0.0001 if self.use_secs_intervals else 0.01
        if self.model_name == "FSRS-6":
            self.s_min: float = 0.001 if not self.use_secs_intervals else _s_min_base
        else:
            self.s_min = _s_min_base

        self.init_s_max: float = 100.0  # Max initial stability
        self.s_max: float = 36500.0  # Max stability (e.g., 100 years)

        # Seed for reproducibility
        self.seed: int = 42

        # Apply global warning filters (can also be done in main.py)
        # warnings.filterwarnings("ignore", category=UserWarning, module="torch.optim.lr_scheduler") # Example for scheduler
        # warnings.filterwarnings("ignore", category=UserWarning) # Original broad filter

    def get_evaluation_file_name(self) -> str:
        """Returns the base name for output files (e.g., for results, plots)."""
        return self.base_file_name

    def get_optimizer_file_name(self) -> str:
        """Returns the base name for optimizer state files."""
        return self.base_file_name + self.optimizer_name_suffix

    def __repr__(self) -> str:
        """Provides a string representation of the configuration."""
        attrs = {
            k: v
            for k, v in self.__dict__.items()
            if k != "raw_args" and not k.startswith("_")
        }
        return f"Config({attrs})"


_config_instance: Optional[Config] = None


def load_config(custom_args_list: Optional[List[str]] = None) -> Config:
    """
    Parses command-line arguments (or custom arguments) and returns a singleton Config instance.

    Args:
        custom_args_list: An optional list of strings representing command-line arguments.
                          If None, sys.argv will be used.

    Returns:
        The Config instance.
    """
    global _config_instance
    if (
        _config_instance is None or custom_args_list is not None
    ):  # Re-parse if custom_args are given
        parser = create_parser()
        if custom_args_list is not None:
            args, _ = parser.parse_known_args(custom_args_list)
        else:
            args, _ = parser.parse_known_args()  # Uses sys.argv by default

        current_config = Config(args)
        if (
            custom_args_list is not None
        ):  # Don't overwrite global instance if it's a custom load for test
            return current_config
        _config_instance = current_config

    return _config_instance


# --- Example Usage (typically this would be in your main script) ---
if __name__ == "__main__":
    print("--- Loading default configuration (from command line or defaults) ---")
    config = load_config()
    print(f"Model Name: {config.model_name}")
    print(f"Device: {config.device}")
    print(f"S_MIN: {config.s_min}")
    print(f"Effective Short Term: {config.include_short_term}")
    print(f"Evaluation File Name: {config.get_evaluation_file_name()}")
    print(f"Optimizer File Name: {config.get_optimizer_file_name()}")
    print(f"Data Path: {config.data_path}")
    if config.dev_mode:
        print(f"FSRS Optimizer Module Path: {config.fsrs_optimizer_module_path}")
    print(repr(config))

    print("\n--- Loading configuration with custom arguments for testing ---")
    test_args = ["--algo", "FSRSv2", "--secs", "--dev", "--short"]
    test_config = load_config(custom_args_list=test_args)
    print(f"Test Model Name: {test_config.model_name}")
    print(f"Test Use Secs: {test_config.use_secs_intervals}")
    print(f"Test Dev Mode: {test_config.dev_mode}")
    print(f"Test S_MIN: {test_config.s_min}")  # Should be 1e-6
    print(f"Test Initial Short Term: {test_config.initial_short_term_setting}")  # True
    print(f"Test Effective Short Term: {test_config.include_short_term}")  # True

    print("\n--- Testing FSRS-6 S_MIN logic ---")
    fsrs6_no_secs_config = load_config(custom_args_list=["--algo", "FSRS-6"])
    print(f"FSRS-6 (no secs) S_MIN: {fsrs6_no_secs_config.s_min}")  # Expected: 0.001

    fsrs6_secs_config = load_config(custom_args_list=["--algo", "FSRS-6", "--secs"])
    print(f"FSRS-6 (with secs) S_MIN: {fsrs6_secs_config.s_min}")  # Expected: 1e-6

    print("\n--- Testing effective_short_term logic ---")
    fsrs5_config_no_short_arg = load_config(custom_args_list=["--algo", "FSRS-5"])
    print(
        f"FSRS-5 (no --short arg) Initial: {fsrs5_config_no_short_arg.initial_short_term_setting}, Effective: {fsrs5_config_no_short_arg.include_short_term}"
    )  # E: True

    fsrs4_config_with_short_arg = load_config(
        custom_args_list=["--algo", "FSRS-4.5", "--short"]
    )
    print(
        f"FSRS-4.5 (with --short arg) Initial: {fsrs4_config_with_short_arg.initial_short_term_setting}, Effective: {fsrs4_config_with_short_arg.include_short_term}"
    )  # E: True

    fsrs4_config_no_short_arg = load_config(custom_args_list=["--algo", "FSRS-4.5"])
    print(
        f"FSRS-4.5 (no --short arg) Initial: {fsrs4_config_no_short_arg.initial_short_term_setting}, Effective: {fsrs4_config_no_short_arg.include_short_term}"
    )  # E: False
