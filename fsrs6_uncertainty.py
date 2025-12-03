import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pyarrow.parquet as pq  # type: ignore

from config import Config, create_parser
from data_loader import UserDataLoader


def compute_mean_std(
    samples: List[List[float]],
) -> Tuple[List[float], List[float]]:
    """
    Compute mean and sample standard deviation for each parameter dimension.

    Args:
        samples: List of parameter sets, each containing 21 floats.

    Returns:
        Tuple of (mean_list, std_list), each containing 21 floats.
    """
    arr = np.array(samples, dtype=np.float64)
    mean = np.mean(arr, axis=0)
    # Use ddof=1 for an unbiased estimate of the variance
    std = np.std(arr, axis=0, ddof=1) if arr.shape[0] > 1 else np.zeros_like(mean)

    # Round to 6 decimal places as required
    mean_rounded = np.round(mean, 6)
    std_rounded = np.round(std, 6)

    return mean_rounded.tolist(), std_rounded.tolist()


def ensure_fsrs_optimizer_on_path(config: Config) -> None:
    """
    Ensure that the fsrs_optimizer module is importable by adjusting sys.path if needed.
    """
    if config.dev_mode:
        fsrs_path = os.path.abspath(config.fsrs_optimizer_module_path)
        if fsrs_path not in sys.path:
            sys.path.insert(0, fsrs_path)


def fit_fsrs6_parameters_for_user(
    user_dataset,
    n_jiggles: int,
    config: Config,
) -> List[List[float]]:
    """
    Run N_JIGGLES Monte Carlo trainings for a single user's dataset.

    Each training uses the same data but different binomial jiggle weights.
    """
    # Lazy import after sys.path has potentially been updated
    from fsrs_optimizer import Optimizer, Trainer  # type: ignore

    optimizer = Optimizer(float_delta_t=config.use_secs_intervals)

    # Make sure elapsed_days exists if we want to filter same-day rows
    if config.no_train_same_day and "elapsed_days" not in user_dataset.columns:
        raise ValueError("Column 'elapsed_days' is required but missing from dataset.")

    all_parameter_sets: List[List[float]] = []

    for _ in range(n_jiggles):
        dataset = user_dataset.copy()

        if config.no_train_same_day:
            dataset = dataset[dataset["elapsed_days"] > 0].copy()

        if len(dataset) == 0:
            # Not enough data after filtering; skip this jiggle
            continue

        # Binomial jiggle weights: 0 or 2, as described in the spec
        weights = (
            np.random.binomial(n=1, p=0.5, size=len(dataset)).astype("float32") * 2.0
        )

        # Ensure at least one non-zero weight so that training is meaningful
        if np.all(weights == 0):
            # Force a single sample to have non-zero weight
            weights[np.random.randint(0, len(weights))] = 2.0

        dataset["weights"] = weights

        # Initialize FSRS-6 model parameters for this dataset
        optimizer.define_model()
        _ = optimizer.initialize_parameters(dataset=dataset, verbose=False)

        # Optionally skip training and only use S0 initialization
        if config.only_S0:
            params = list(map(float, optimizer.init_w))
            all_parameter_sets.append(params)
            continue

        # Train FSRS-6 with the jiggle weights
        trainer = Trainer(
            dataset,
            None,
            optimizer.init_w,
            n_epoch=5,
            lr=4e-2,
            gamma=1.0,
            batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
            enable_short_term=config.include_short_term,
        )
        params = trainer.train(verbose=False)
        all_parameter_sets.append(list(map(float, params)))

    return all_parameter_sets


def generate_predictions_for_user(
    user_dataset,
    param_samples: List[List[float]],
    config: Config,
    output_dir: Path,
    user_id: int,
) -> None:
    """
    For a given user dataset and a list of parameter samples, generate predictions.

    For each review record, this will produce:
      - n_jiggles columns of stability values
      - n_jiggles columns of retrievability values
    and save the result as a TSV file.
    """
    # Import prediction utilities from fsrs_optimizer
    from fsrs_optimizer import Collection as FSRSCollection, power_forgetting_curve  # type: ignore

    df = user_dataset.copy()

    for idx, w in enumerate(param_samples):
        collection = FSRSCollection(w)
        stabilities, difficulties = collection.batch_predict(df)

        # Convert to numpy arrays for consistent downstream processing
        stabilities_arr = np.asarray(stabilities, dtype=np.float64)

        # Retrievability (predicted recall probability) using the forgetting curve
        p = power_forgetting_curve(
            df["delta_t"].to_numpy(dtype=np.float64),
            stabilities_arr,
            -w[20],
        )

        df[f"stability_{idx}"] = stabilities_arr
        df[f"retrievability_{idx}"] = p

    # Remove tensor column which is not serializable to TSV
    if "tensor" in df.columns:
        del df["tensor"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{user_id}.tsv"
    df.to_csv(output_path, sep="\t", index=False)


def main() -> None:
    """
    Run statistical uncertainty analysis for FSRS-6 using Monte Carlo jiggle weights.

    For each dataset (user), we:
      - Load data via the existing UserDataLoader
      - Train N_JIGGLES FSRS-6 models with different binomial jiggle weights
      - Output all parameter sets per user as JSON lines
    """
    parser = create_parser()
    parser.add_argument(
        "--n_jiggles",
        type=int,
        default=100,
        help="Number of Monte Carlo jiggle trainings per user.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="FSRS-6-uncertainty.jsonl",
        help="Output JSONL file path for parameter samples.",
    )
    parser.add_argument(
        "--pred_output_dir",
        type=str,
        default="evaluation/FSRS-6-uncertainty",
        help="Directory to save per-user prediction TSV files.",
    )
    args, _ = parser.parse_known_args()

    # Force algorithm to be FSRS-6 for this script
    args.algo = "FSRS-6"

    config = Config(args)
    ensure_fsrs_optimizer_on_path(config)

    n_jiggles: int = args.n_jiggles
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_output_dir = Path(args.pred_output_dir)

    data_loader = UserDataLoader(config)

    # Discover available user IDs from the parquet partitioning
    dataset = pq.ParquetDataset(config.data_path / "revlogs")
    user_ids: List[int] = []
    for user_id in dataset.partitioning.dictionaries[0]:
        uid = int(user_id.as_py())
        if config.max_user_id is not None and uid > config.max_user_id:
            continue
        user_ids.append(uid)

    user_ids.sort()

    with output_path.open("w", encoding="utf-8") as out_f:
        for user_id in user_ids:
            try:
                user_dataset = data_loader.load_user_data(user_id)
            except Exception:
                # Skip users with insufficient or invalid data
                continue

            param_samples = fit_fsrs6_parameters_for_user(
                user_dataset=user_dataset,
                n_jiggles=n_jiggles,
                config=config,
            )

            if not param_samples:
                continue

            # Compute per-user statistics
            param_mean, param_std = compute_mean_std(param_samples)

            # Save parameter samples and their statistics
            record: Dict[str, Any] = {
                "user": user_id,
                "size": int(len(user_dataset)),
                "n_jiggles": len(param_samples),
                "parameters": param_samples,
                "param_mean": param_mean,
                "param_std": param_std,
            }
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")

            # Generate and save per-review predictions for this user
            generate_predictions_for_user(
                user_dataset=user_dataset,
                param_samples=param_samples,
                config=config,
                output_dir=pred_output_dir,
                user_id=user_id,
            )


if __name__ == "__main__":
    main()
