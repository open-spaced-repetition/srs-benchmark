import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
from scipy.signal import find_peaks
from tqdm.auto import tqdm  # type: ignore

# Add parent directory to path to import from project root
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

from config import Config, create_parser
from data_loader import UserDataLoader

N_EPOCH = 5
LR = 4e-2
BATCH_SIZE = 512
GAMMA = 0


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
        # Path is relative to project root, so resolve from project root
        script_dir = Path(__file__).parent
        project_root = script_dir.parent
        fsrs_path = (project_root / config.fsrs_optimizer_module_path).resolve()
        if str(fsrs_path) not in sys.path:
            sys.path.insert(0, str(fsrs_path))


def fit_fsrs6_parameters_for_user(
    user_dataset,
    n_jiggles: int,
    config: Config,
    seed: int = 42,
) -> List[List[float]]:
    """
    Run N_JIGGLES Monte Carlo trainings for a single user's dataset.

    Each training uses card-level bootstrap resampling:
    - Sample J cards with replacement from all unique cards
    - Construct a new dataset where each card's reviews appear as many times as the card was sampled
    - Train FSRS-6 model on the resampled dataset

    Args:
        user_dataset: User's dataset (must contain 'card_id' column)
        n_jiggles: Number of bootstrap trainings
        config: Configuration object
        seed: Random seed for reproducibility (default: 42)
    """
    # Lazy import after sys.path has potentially been updated
    from fsrs_optimizer import Optimizer, Trainer  # type: ignore

    optimizer = Optimizer(float_delta_t=config.use_secs_intervals)

    # Make sure elapsed_days exists if we want to filter same-day rows
    if config.no_train_same_day and "elapsed_days" not in user_dataset.columns:
        raise ValueError("Column 'elapsed_days' is required but missing from dataset.")

    # Make sure card_id exists for card-level bootstrap
    if "card_id" not in user_dataset.columns:
        raise ValueError(
            "Column 'card_id' is required but missing from dataset for card-level bootstrap."
        )

    all_parameter_sets: List[List[float]] = []

    # Create a random number generator with the seed for this user
    # Use a deterministic seed based on user_id and global seed for reproducibility
    rng = np.random.default_rng(seed)

    # Apply filtering first to get the final dataset structure for all bootstrap samples
    dataset_filtered = user_dataset.copy()
    if config.no_train_same_day:
        dataset_filtered = dataset_filtered[dataset_filtered["elapsed_days"] > 0].copy()

    if len(dataset_filtered) == 0:
        # Not enough data after filtering
        return all_parameter_sets

    # Get unique card IDs from filtered dataset for bootstrap sampling
    unique_card_ids = dataset_filtered["card_id"].unique()
    n_cards = len(unique_card_ids)

    if n_cards == 0:
        # No cards available
        return all_parameter_sets

    # Pre-group reviews by card_id for efficient lookup
    card_groups = {
        card_id: group for card_id, group in dataset_filtered.groupby("card_id")
    }

    for jiggle_idx in range(n_jiggles):
        # Card-level bootstrap: sample n_cards cards with replacement
        # From {1, ..., J} sample J indices with replacement: j_1^(b), ..., j_J^(b)
        bootstrap_card_indices = rng.choice(n_cards, size=n_cards, replace=True)

        # Get the card IDs that were sampled (with replacement)
        # D^(b) = {h_{j_1^(b)}, ..., h_{j_J^(b)}} where each h_j contains all reviews for card j
        bootstrap_card_ids = unique_card_ids[bootstrap_card_indices]

        # Construct the bootstrap dataset by including each card's reviews as many times as it was sampled
        # If a card was sampled k times, all its reviews will appear k times in the dataset
        bootstrap_datasets = []
        for card_id in bootstrap_card_ids:
            # Get all review records for this card (using pre-grouped data)
            if card_id in card_groups:
                card_reviews = card_groups[card_id].copy()
                bootstrap_datasets.append(card_reviews)

        # Concatenate all card reviews to form the bootstrap dataset
        if not bootstrap_datasets:
            # No cards were sampled or all sampled cards have no reviews
            continue

        dataset = pd.concat(bootstrap_datasets, ignore_index=True)

        # Sort by review_th to maintain temporal order if needed
        if "review_th" in dataset.columns:
            dataset = dataset.sort_values("review_th").reset_index(drop=True)

        # Initialize FSRS-6 model parameters for this dataset
        optimizer.define_model()
        _ = optimizer.initialize_parameters(dataset=dataset, verbose=False)

        # Optionally skip training and only use S0 initialization
        if config.only_S0:
            params = list(map(float, optimizer.init_w))
            all_parameter_sets.append(params)
            continue

        # Train FSRS-6 on the bootstrap resampled dataset
        trainer = Trainer(
            dataset,
            None,
            optimizer.init_w,
            n_epoch=N_EPOCH,
            lr=LR,
            gamma=GAMMA,
            batch_size=BATCH_SIZE,
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

    # Collect all new columns to avoid DataFrame fragmentation
    new_columns = {}

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

        new_columns[f"stability_{idx}"] = stabilities_arr
        new_columns[f"retrievability_{idx}"] = p

    # Add all new columns at once using pd.concat to avoid fragmentation
    if new_columns:
        new_df = pd.DataFrame(new_columns, index=df.index)
        df = pd.concat([df, new_df], axis=1)

    # Remove tensor column which is not serializable to TSV
    if "tensor" in df.columns:
        del df["tensor"]

    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{user_id}.tsv"
    df.to_csv(output_path, sep="\t", index=False)


def visualize_stability_distributions(
    pred_file_path: Path,
    user_id: int,
    n_jiggles: int,
    output_dir: Path,
    top_n: int = 100,
    combos_per_page: int = 10,
) -> None:
    """
    Visualize stability distributions for top N (r_history, t_history) combinations.
    Generates multiple plots, each showing a subset of combinations.

    For each user, this function:
      - Reads the prediction TSV file
      - Groups by (r_history, t_history) combinations
      - Selects top N combinations by frequency
      - Visualizes the stability distribution for each combination
      - Saves multiple plots (each with combos_per_page combinations) to output_dir
    """
    # Read the prediction file
    df = pd.read_csv(pred_file_path, sep="\t")

    # Check required columns
    if "r_history" not in df.columns or "t_history" not in df.columns:
        print(
            f"Warning: User {user_id} missing r_history or t_history columns, skipping visualization."
        )
        return

    # Collect all stability columns
    stability_cols = [f"stability_{i}" for i in range(n_jiggles)]
    missing_cols = [col for col in stability_cols if col not in df.columns]
    if missing_cols:
        print(
            f"Warning: User {user_id} missing stability columns: {missing_cols}, skipping visualization."
        )
        return

    # Group by (r_history, t_history) and count frequency
    df["combo"] = df["r_history"].astype(str) + " | " + df["t_history"].astype(str)
    combo_counts = df["combo"].value_counts().head(top_n)

    if len(combo_counts) == 0:
        print(
            f"Warning: User {user_id} has no valid combinations, skipping visualization."
        )
        return

    # Filter to top N combinations
    top_combos = combo_counts.index.tolist()
    df_filtered = df[df["combo"].isin(top_combos)].copy()

    # Prepare data for plotting: collect all stability values for each combination
    all_plot_data = []
    all_combo_labels = []
    for combo in top_combos:
        combo_df = df_filtered[df_filtered["combo"] == combo].copy()
        # Collect all stability values across all jiggles for this combination
        stability_values = []
        for col in stability_cols:
            # Check if column exists and get values
            if isinstance(combo_df, pd.DataFrame) and col in combo_df.columns:
                col_values = combo_df[col]
                # Ensure we have a Series for dropna()
                if not isinstance(col_values, pd.Series):
                    col_values = pd.Series(col_values)
                valid_values = col_values.dropna()
                stability_values.extend(valid_values.tolist())
        if stability_values:
            all_plot_data.append(stability_values)
            # Create a readable label with frequency
            freq = combo_counts[combo]
            # Truncate long strings for readability
            r_hist, t_hist = combo.split(" | ", 1)
            if len(r_hist) > 30:
                r_hist = r_hist[:27] + "..."
            if len(t_hist) > 30:
                t_hist = t_hist[:27] + "..."
            all_combo_labels.append(f"{r_hist} | {t_hist}\n(freq={freq})")

    if not all_plot_data:
        print(
            f"Warning: User {user_id} has no valid stability data, skipping visualization."
        )
        return

    # Split into pages: each page has combos_per_page combinations
    output_dir.mkdir(parents=True, exist_ok=True)
    num_pages = (len(all_plot_data) + combos_per_page - 1) // combos_per_page

    for page_idx in range(num_pages):
        start_idx = page_idx * combos_per_page
        end_idx = min(start_idx + combos_per_page, len(all_plot_data))

        plot_data = all_plot_data[start_idx:end_idx]
        combo_labels = all_combo_labels[start_idx:end_idx]

        # Create the plot
        fig_height = max(8.0, float(len(plot_data) * 0.5))
        fig, ax = plt.subplots(figsize=(14, fig_height))

        # Use box plot to show distribution
        bp = ax.boxplot(
            plot_data,
            vert=True,
            patch_artist=True,
            showmeans=True,
            meanline=True,
        )

        # Set labels after creating the boxplot
        ax.set_xticklabels(combo_labels)

        # Style the boxes
        for patch in bp["boxes"]:
            patch.set_facecolor("lightblue")
            patch.set_alpha(0.7)

        ax.set_ylabel("Stability", fontsize=12)
        ax.set_xlabel(
            "(r_history, t_history) Combination (Top 100 by frequency)", fontsize=12
        )
        ax.set_title(
            f"User {user_id}: Stability Distribution by History Combination "
            f"(Page {page_idx + 1}/{num_pages}, Combos {start_idx + 1}-{end_idx})\n"
            f"(Each box shows {n_jiggles} jiggle samples)",
            fontsize=14,
            fontweight="bold",
        )
        ax.grid(True, alpha=0.3, axis="y")
        ax.tick_params(axis="x", rotation=45, labelsize=8)

        plt.tight_layout()

        # Save the plot
        output_path = output_dir / f"{user_id}_page_{page_idx + 1}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Saved stability distribution plot: {output_path}")


def visualize_parameter_distributions(
    param_samples: List[List[float]],
    user_id: int,
    output_dir: Path,
    params_per_page: int = 7,
) -> None:
    """
    Visualize FSRS-6 parameter distributions for a user across all jiggle samples.

    For each user, this function:
      - Takes the list of parameter samples (n_jiggles x 21 parameters)
      - Visualizes the distribution of each parameter across jiggle samples
      - Saves multiple plots (each with params_per_page parameters) to output_dir
    """
    if not param_samples:
        print(
            f"Warning: User {user_id} has no parameter samples, skipping parameter visualization."
        )
        return

    # Convert to numpy array for easier manipulation
    param_array = np.array(param_samples, dtype=np.float64)
    n_params = param_array.shape[1]  # Should be 21 for FSRS-6
    n_jiggles = param_array.shape[0]

    if n_params != 21:
        print(
            f"Warning: User {user_id} has {n_params} parameters (expected 21), skipping parameter visualization."
        )
        return

    # Split into pages: each page has params_per_page parameters
    output_dir.mkdir(parents=True, exist_ok=True)
    num_pages = (n_params + params_per_page - 1) // params_per_page

    for page_idx in range(num_pages):
        start_param = page_idx * params_per_page
        end_param = min(start_param + params_per_page, n_params)

        num_params_this_page = end_param - start_param

        # Create subplots: 2 columns, multiple rows
        n_rows = (num_params_this_page + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        axes = axes.flatten()

        for local_idx, param_idx in enumerate(range(start_param, end_param)):
            ax = axes[local_idx]
            param_values = param_array[:, param_idx]

            # Calculate statistics
            mean_val = np.mean(param_values)
            median_val = np.median(param_values)
            std_val = np.std(param_values, ddof=1) if len(param_values) > 1 else 0.0

            # Create histogram
            counts, bins, patches = ax.hist(
                param_values,
                bins=min(30, max(10, n_jiggles // 5)),
                alpha=0.7,
                edgecolor="black",
                linewidth=0.5,
                color="steelblue",
            )

            # Add vertical lines for statistics
            ax.axvline(
                mean_val,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Mean: {mean_val:.4f}",
            )
            ax.axvline(
                median_val,
                color="orange",
                linestyle="--",
                linewidth=2,
                label=f"Median: {median_val:.4f}",
            )
            if std_val > 0:
                ax.axvline(
                    mean_val - std_val,
                    color="gray",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.7,
                    label=f"±1 Std: {std_val:.4f}",
                )
                ax.axvline(
                    mean_val + std_val,
                    color="gray",
                    linestyle=":",
                    linewidth=1.5,
                    alpha=0.7,
                )

            # Add text box with statistics
            stats_text = f"Mean: {mean_val:.4f}\nMedian: {median_val:.4f}\nStd: {std_val:.4f}\nN: {n_jiggles}"
            ax.text(
                0.02,
                0.98,
                stats_text,
                transform=ax.transAxes,
                fontsize=9,
                verticalalignment="top",
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
            )

            ax.set_xlabel("Parameter value", fontsize=10)
            ax.set_ylabel("Frequency", fontsize=10)
            ax.set_title(f"w[{param_idx}]", fontsize=11, fontweight="bold")
            ax.legend(loc="best", fontsize=8)
            ax.grid(True, alpha=0.3)

        # Hide unused subplots
        for idx in range(num_params_this_page, len(axes)):
            axes[idx].set_visible(False)

        fig.suptitle(
            f"User {user_id}: FSRS-6 Parameter Distributions "
            f"(Page {page_idx + 1}/{num_pages}, Parameters {start_param}-{end_param - 1})\n"
            f"(Each histogram shows {n_jiggles} jiggle samples)",
            fontsize=14,
            fontweight="bold",
        )

        plt.tight_layout(rect=(0, 0, 1, 0.96))  # Leave space for suptitle

        # Save the plot
        output_path = output_dir / f"{user_id}_params_page_{page_idx + 1}.png"
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        plt.close()

        print(f"  Saved parameter distribution plot: {output_path}")


def calculate_interval_from_stability(
    stability: np.ndarray, target_retrievability: float, decay: float
) -> np.ndarray:
    """
    Calculate interval time from stability and target retrievability.

    FSRS forgetting curve: R(t) = (1 + factor * t / s)^decay
    where factor = 0.9^(1/decay) - 1, decay = -w[20]

    Solving for t: t = s * ((R^(1/decay)) - 1) / factor

    Args:
        stability: Array of stability values
        target_retrievability: Target retrievability (typically 0.9)
        decay: Decay parameter (-w[20])

    Returns:
        Array of interval times
    """
    factor = 0.9 ** (1.0 / decay) - 1.0
    r_power = target_retrievability ** (1.0 / decay)
    intervals = stability * (r_power - 1.0) / factor
    return np.maximum(intervals, 0.0)  # Ensure non-negative


def detect_peaks_in_distribution(
    normalized_diff: np.ndarray,
    n_bins: int = 100,
    min_height: Optional[float] = None,
    min_distance: Optional[int] = None,
) -> List[float]:
    """
    Automatically detect peaks in the normalized difference distribution.

    Args:
        normalized_diff: Normalized differences array
        n_bins: Number of bins for histogram
        min_height: Minimum height for a peak (as fraction of max count)
        min_distance: Minimum distance between peaks (in bins)

    Returns:
        List of detected peak values
    """
    # Create histogram
    counts, bin_edges = np.histogram(normalized_diff, bins=n_bins)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    # Set default parameters if not provided
    if min_height is None:
        min_height = np.max(counts) * 0.1  # At least 10% of max count
    if min_distance is None:
        min_distance = max(1, n_bins // 20)  # At least 5% of bins apart

    # Find peaks
    peaks, properties = find_peaks(
        counts,
        height=min_height,
        distance=min_distance,
    )

    # Get peak values (bin centers)
    peak_values = [float(bin_centers[p]) for p in peaks]

    return peak_values


def analyze_peak_data_points(
    test_set: pd.DataFrame,
    normalized_diff: np.ndarray,
    jiggle_interval_mean: np.ndarray,
    jiggle_interval_std: np.ndarray,
    standard_intervals: np.ndarray,
    peak_values: List[float],
    tolerance: float,
    output_dir: Path,
    user_id: int,
) -> None:
    """
    Analyze characteristics of data points at specific peak values in the normalized difference distribution.

    Args:
        test_set: Test dataset
        normalized_diff: Normalized differences array
        jiggle_interval_mean: Mean jiggle intervals
        jiggle_interval_std: Std of jiggle intervals
        standard_intervals: Standard FSRS intervals
        peak_values: List of peak values to analyze (if empty, will auto-detect)
        tolerance: Tolerance for matching peak values
        output_dir: Directory to save analysis results
        user_id: User ID
    """
    analysis_results = []

    # Track which data points have been assigned to avoid double counting
    assigned_mask = np.zeros(len(test_set), dtype=bool)

    for peak_val in peak_values:
        # Find data points near the peak value that haven't been assigned yet
        mask = (np.abs(normalized_diff - peak_val) < tolerance) & (~assigned_mask)
        peak_indices = np.where(mask)[0]

        if len(peak_indices) == 0:
            print(
                f"  No unassigned data points found near peak value {peak_val} (tolerance={tolerance})"
            )
            continue

        # Mark these points as assigned
        assigned_mask[peak_indices] = True

        peak_data = test_set.iloc[peak_indices].copy()
        peak_data["normalized_diff"] = normalized_diff[peak_indices]
        peak_data["jiggle_interval_mean"] = jiggle_interval_mean[peak_indices]
        peak_data["jiggle_interval_std"] = jiggle_interval_std[peak_indices]
        peak_data["standard_interval"] = standard_intervals[peak_indices]

        # Analyze common characteristics
        analysis: Dict[str, Any] = {
            "peak_value": peak_val,
            "n_points": len(peak_indices),
            "fraction": len(peak_indices) / len(test_set),
        }

        # Analyze r_history patterns
        if "r_history" in peak_data.columns:
            r_history_counts = peak_data["r_history"].value_counts().head(10)
            analysis["top_r_history"] = {
                str(k): int(v) for k, v in r_history_counts.items()
            }

        # Analyze t_history patterns
        if "t_history" in peak_data.columns:
            t_history_counts = peak_data["t_history"].value_counts().head(10)
            analysis["top_t_history"] = {
                str(k): int(v) for k, v in t_history_counts.items()
            }

        # Analyze rating distribution
        if "rating" in peak_data.columns:
            rating_counts = peak_data["rating"].value_counts()
            analysis["rating_distribution"] = {
                int(k): int(v) for k, v in rating_counts.items()
            }

        # Analyze review number (i)
        if "i" in peak_data.columns:
            i_stats = {
                "mean": float(peak_data["i"].mean()),
                "median": float(peak_data["i"].median()),
                "std": float(peak_data["i"].std()),
                "min": int(peak_data["i"].min()),
                "max": int(peak_data["i"].max()),
            }
            analysis["review_number_stats"] = i_stats

        # Analyze delta_t
        if "delta_t" in peak_data.columns:
            delta_t_stats = {
                "mean": float(peak_data["delta_t"].mean()),
                "median": float(peak_data["delta_t"].median()),
                "std": float(peak_data["delta_t"].std()),
                "min": float(peak_data["delta_t"].min()),
                "max": float(peak_data["delta_t"].max()),
            }
            analysis["delta_t_stats"] = delta_t_stats

        # Analyze interval statistics
        analysis["interval_stats"] = {
            "jiggle_mean_mean": float(peak_data["jiggle_interval_mean"].mean()),
            "jiggle_mean_std": float(peak_data["jiggle_interval_mean"].std()),
            "jiggle_std_mean": float(peak_data["jiggle_interval_std"].mean()),
            "standard_mean": float(peak_data["standard_interval"].mean()),
            "standard_std": float(peak_data["standard_interval"].std()),
        }

        # Analyze first_rating if available
        if "first_rating" in peak_data.columns:
            first_rating_counts = peak_data["first_rating"].value_counts()
            analysis["first_rating_distribution"] = {
                str(k): int(v) for k, v in first_rating_counts.items()
            }

        analysis_results.append(analysis)

        # Save detailed data for this peak
        output_dir.mkdir(parents=True, exist_ok=True)
        peak_file = output_dir / f"{user_id}_peak_{peak_val:.2f}_data.tsv"
        # Select relevant columns for output
        output_cols = [
            "normalized_diff",
            "jiggle_interval_mean",
            "jiggle_interval_std",
            "standard_interval",
        ]
        if "r_history" in peak_data.columns:
            output_cols.append("r_history")
        if "t_history" in peak_data.columns:
            output_cols.append("t_history")
        if "rating" in peak_data.columns:
            output_cols.append("rating")
        if "i" in peak_data.columns:
            output_cols.append("i")
        if "delta_t" in peak_data.columns:
            output_cols.append("delta_t")
        if "first_rating" in peak_data.columns:
            output_cols.append("first_rating")

        available_cols = [col for col in output_cols if col in peak_data.columns]
        peak_data[available_cols].to_csv(peak_file, sep="\t", index=False)
        print(
            f"  Saved peak {peak_val:.2f} data: {peak_file} ({len(peak_indices)} points)"
        )

    # Calculate coverage statistics
    total_assigned = np.sum(assigned_mask)
    total_points = len(test_set)
    coverage = total_assigned / total_points if total_points > 0 else 0.0

    # Save summary analysis
    if analysis_results:
        summary_file = output_dir / f"{user_id}_peak_analysis.json"
        summary_data = {
            "coverage": coverage,
            "total_points": total_points,
            "assigned_points": int(total_assigned),
            "unassigned_points": int(total_points - total_assigned),
            "peaks": analysis_results,
        }
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        print(f"  Saved peak analysis summary: {summary_file}")

        # Print summary to console
        print(f"\n  Peak Analysis Summary for User {user_id}:")
        print(
            f"  Coverage: {coverage * 100:.2f}% ({total_assigned}/{total_points} points assigned to peaks)"
        )
        print(
            f"  Unassigned: {(1 - coverage) * 100:.2f}% ({total_points - total_assigned} points)"
        )
        for result in analysis_results:
            print(f"\n  Peak at {result['peak_value']:.2f}:")
            print(
                f"    Number of points: {result['n_points']} ({result['fraction'] * 100:.2f}% of test set)"
            )
            if "top_r_history" in result:
                print("    Top r_history patterns:")
                for pattern, count in list(result["top_r_history"].items())[:5]:
                    print(f"      '{pattern}': {count}")
            if "top_t_history" in result:
                print("    Top t_history patterns:")
                for pattern, count in list(result["top_t_history"].items())[:5]:
                    print(f"      '{pattern}': {count}")
            if "review_number_stats" in result:
                stats = result["review_number_stats"]
                print(
                    f"    Review number (i): mean={stats['mean']:.2f}, median={stats['median']:.1f}, range=[{stats['min']}, {stats['max']}]"
                )
            if "interval_stats" in result:
                stats = result["interval_stats"]
                print(
                    f"    Jiggle interval mean: {stats['jiggle_mean_mean']:.2f} ± {stats['jiggle_mean_std']:.2f}"
                )
                print(
                    f"    Standard interval: {stats['standard_mean']:.2f} ± {stats['standard_std']:.2f}"
                )
                print(f"    Jiggle std (uncertainty): {stats['jiggle_std_mean']:.2f}")


def calculate_initial_review_interval(
    params: List[float], rating: int, target_retrievability: float = 0.9
) -> float:
    """
    Calculate the interval time for an initial review using FSRS parameters.

    Args:
        params: FSRS-6 parameter weights (21 parameters)
        target_retrievability: Target retrievability (default: 0.9)

    Returns:
        Interval time in days
    """

    # For an initial review with rating=3, the initial stability is w[2] (0-indexed, rating-1)
    # w[0] = rating 1, w[1] = rating 2, w[2] = rating 3, w[3] = rating 4
    S_MIN = 0.001
    initial_stability = max(S_MIN, params[rating - 1])

    # Now calculate interval from initial stability
    decay = -params[20]
    interval = calculate_interval_from_stability(
        np.array([initial_stability]), target_retrievability, decay
    )[0]

    return float(interval)


def validate_jiggle_method_gaussianness(
    user_dataset,
    n_jiggles: int,
    config: Config,
    seed: int,
    output_dir: Path,
    user_id: int,
    min_data_size: int = 1000,
) -> Optional[float]:
    """
    Validate jiggle method by calculating z-score for initial good review interval.

    For a single user, this function:
    - Fits jiggle parameters on a small subset of data
    - Fits standard FSRS parameters on full dataset
    - Calculates interval for initial good review (rating=3)
    - Returns z-score: (t_jiggle_mean - t_standard) / t_jiggle_std

    Args:
        user_dataset: Full user dataset
        n_jiggles: Number of bootstrap trainings
        config: Configuration object
        seed: Random seed
        output_dir: Directory to save results
        user_id: User ID
        min_data_size: Minimum data size required (default: 1000)

    Returns:
        Z-score if successful, None otherwise
    """
    if len(user_dataset) < min_data_size:
        return None

    # Fit jiggle parameters on subset
    jiggle_params = fit_fsrs6_parameters_for_user(
        user_dataset=user_dataset,
        n_jiggles=n_jiggles,
        config=config,
        seed=seed,
    )

    if not jiggle_params or len(jiggle_params) == 0:
        return None

    # Fit standard FSRS on full dataset
    from fsrs_optimizer import Optimizer, Trainer  # type: ignore

    optimizer = Optimizer(float_delta_t=config.use_secs_intervals)
    optimizer.define_model()
    _ = optimizer.initialize_parameters(dataset=user_dataset, verbose=False)

    if config.only_S0:
        standard_params = list(map(float, optimizer.init_w))
    else:
        trainer = Trainer(
            user_dataset,
            None,
            optimizer.init_w,
            n_epoch=N_EPOCH,
            lr=LR,
            gamma=GAMMA,
            batch_size=BATCH_SIZE,
            max_seq_len=config.max_seq_len,
            enable_short_term=config.include_short_term,
        )
        standard_params = list(map(float, trainer.train(verbose=False)))

    # Calculate intervals for initial good review
    target_r = 0.9
    rating = 3
    jiggle_intervals = [
        calculate_initial_review_interval(params, rating, target_r)
        for params in jiggle_params
    ]

    t_jiggle_mean = np.mean(jiggle_intervals)
    t_jiggle_std = (
        np.std(jiggle_intervals, ddof=1) if len(jiggle_intervals) > 1 else 0.0
    )
    t_standard = calculate_initial_review_interval(standard_params, rating, target_r)

    # Calculate z-score: (t_jiggle_mean - t_standard) / t_jiggle_std
    if t_jiggle_std == 0:
        return None

    z_score = (t_jiggle_mean - t_standard) / t_jiggle_std

    return float(z_score)


def collect_and_visualize_z_scores(
    data_loader: UserDataLoader,
    user_ids: List[int],
    n_jiggles: int,
    config: Config,
    seed: int,
    output_dir: Path,
    max_z_scores: int = 1000,
    min_data_size: int = 1000,
) -> None:
    """
    Collect z-scores from multiple users and visualize the distribution.

    This function validates the jiggle method by checking if z-scores follow
    a Gaussian distribution, which would indicate the bootstrap method is working correctly.

    Args:
        data_loader: UserDataLoader instance
        user_ids: List of user IDs to process
        n_jiggles: Number of bootstrap trainings
        config: Configuration object
        seed: Random seed
        output_dir: Directory to save visualization
        max_z_scores: Maximum number of z-scores to collect (default: 1000)
        min_data_size: Minimum data size per user (default: 1000)
    """
    z_scores = []
    z_scores_with_users = []  # List of (user_id, z_score) tuples

    print(f"\nCollecting z-scores for Gaussian validation (max {max_z_scores})...")

    for user_id in tqdm(user_ids):
        if len(z_scores) >= max_z_scores:
            print(f"  Collected {len(z_scores)} z-scores, stopping collection.")
            break

        try:
            user_dataset = data_loader.load_user_data(user_id)
        except Exception:
            continue

        user_seed = seed + user_id
        z_score = validate_jiggle_method_gaussianness(
            user_dataset=user_dataset,
            n_jiggles=n_jiggles,
            config=config,
            seed=user_seed,
            output_dir=output_dir,
            user_id=user_id,
            min_data_size=min_data_size,
        )

        if z_score is not None:
            print(f"  User {user_id} z-score: {z_score:.6f}")
            z_scores.append(z_score)
            z_scores_with_users.append((user_id, z_score))
            if len(z_scores) % 100 == 0:
                print(f"  Collected {len(z_scores)} z-scores...")

    if len(z_scores) == 0:
        print("  Warning: No z-scores were collected. Cannot create validation plot.")
        return

    print(f"  Total z-scores collected: {len(z_scores)}")

    # Print users with z-score < -5
    extreme_users = [(uid, z) for uid, z in z_scores_with_users if z < -5]
    if extreme_users:
        print(f"\n  Users with z-score < -5: {len(extreme_users)} users")
        extreme_users_sorted = sorted(
            extreme_users, key=lambda x: x[1]
        )  # Sort by z-score (most negative first)
        for user_id, z_score in extreme_users_sorted:
            print(f"    User {user_id}: z-score = {z_score:.6f}")
    else:
        print("\n  No users with z-score < -5 found.")

    # Visualize z-score distribution
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Histogram of z-scores
    ax1 = axes[0]
    n_bins = min(50, max(20, len(z_scores) // 20))
    counts, bins, patches = ax1.hist(
        z_scores, bins=n_bins, alpha=0.7, edgecolor="black", color="steelblue"
    )

    # Add normal distribution overlay
    from scipy.stats import norm, probplot, skew, kurtosis  # type: ignore

    mu = np.mean(z_scores)
    sigma = np.std(z_scores, ddof=1)
    x = np.linspace(np.min(z_scores), np.max(z_scores), 100)
    y = norm.pdf(x, mu, sigma) * len(z_scores) * (bins[1] - bins[0])
    ax1.plot(x, y, "r--", linewidth=2, label=f"Normal(μ={mu:.4f}, σ={sigma:.4f})")

    # Add vertical line at mean
    ax1.axvline(x=mu, color="red", linestyle="--", linewidth=2, label=f"Mean: {mu:.4f}")
    ax1.axvline(
        x=0, color="orange", linestyle=":", linewidth=1.5, alpha=0.7, label="Zero"
    )

    ax1.set_xlabel("Z-Score: (t_jiggle_mean - t_standard) / t_jiggle_std", fontsize=12)
    ax1.set_ylabel("Frequency", fontsize=12)
    ax1.set_title(
        f"Distribution of Z-Scores for Initial Review Interval\n"
        f"(N={len(z_scores)} users, {n_jiggles} jiggles per user)",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Add statistics text
    stats_text = (
        f"Statistics:\n"
        f"Mean: {mu:.6f}\n"
        f"Std: {sigma:.6f}\n"
        f"Median: {np.median(z_scores):.6f}\n"
        f"Min: {np.min(z_scores):.6f}\n"
        f"Max: {np.max(z_scores):.6f}\n"
        f"Skewness: {skew(z_scores):.4f}\n"
        f"Kurtosis: {kurtosis(z_scores):.4f}"
    )
    ax1.text(
        0.02,
        0.98,
        stats_text,
        transform=ax1.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Plot 2: Q-Q plot to check normality
    ax2 = axes[1]
    probplot(z_scores, dist="norm", plot=ax2)
    ax2.set_title("Q-Q Plot (Normal Distribution)", fontsize=12, fontweight="bold")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save the plot
    output_path = output_dir / "z_score_gaussian_validation.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved z-score validation plot: {output_path}")

    # Perform normality test
    from scipy.stats import shapiro, normaltest  # type: ignore

    if len(z_scores) >= 3:
        # Shapiro-Wilk test (works best for 3-5000 samples)
        if len(z_scores) <= 5000:
            stat_shapiro, p_shapiro = shapiro(z_scores)
            print("\n  Normality Test Results:")
            print(
                f"  Shapiro-Wilk test: statistic={stat_shapiro:.6f}, p-value={p_shapiro:.6f}"
            )
            if p_shapiro > 0.05:
                print("    ✓ Distribution appears normal (p > 0.05)")
            else:
                print("    ✗ Distribution does not appear normal (p ≤ 0.05)")

        # D'Agostino's normality test (works for any sample size)
        stat_dagostino, p_dagostino = normaltest(z_scores)
        print(
            f"  D'Agostino test: statistic={stat_dagostino:.6f}, p-value={p_dagostino:.6f}"
        )
        if p_dagostino > 0.05:
            print("    ✓ Distribution appears normal (p > 0.05)")
        else:
            print("    ✗ Distribution does not appear normal (p ≤ 0.05)")

    # Save z-scores to JSON file
    z_score_file = output_dir / "z_scores.json"
    z_score_data = {
        "n_samples": len(z_scores),
        "n_jiggles": n_jiggles,
        "mean": float(mu),
        "std": float(sigma),
        "median": float(np.median(z_scores)),
        "min": float(np.min(z_scores)),
        "max": float(np.max(z_scores)),
        "z_scores": [float(z) for z in z_scores],
        "user_z_scores": [
            {"user_id": int(uid), "z_score": float(z)} for uid, z in z_scores_with_users
        ],
        "extreme_users_z_less_than_minus5": [
            {"user_id": int(uid), "z_score": float(z)}
            for uid, z in z_scores_with_users
            if z < -5
        ],
    }
    with open(z_score_file, "w", encoding="utf-8") as f:
        json.dump(z_score_data, f, indent=2, ensure_ascii=False)
    print(f"  Saved z-scores data: {z_score_file}")


def compare_jiggle_vs_standard(
    user_dataset,
    n_jiggles: int,
    config: Config,
    seed: int,
    output_dir: Path,
    user_id: int,
) -> None:
    """
    Compare jiggle method vs standard FSRS fitting on test data.

    Splits data into 2 roughly equal sets:
    1. Training set (for both jiggle method and standard FSRS fitting)
    2. Test set (for evaluation)

    For each test data point, calculates:
    (t_jiggle,mean - t_standard) / σ_t,jiggle

    Args:
        user_dataset: Full user dataset
        n_jiggles: Number of jiggle trainings
        config: Configuration object
        seed: Random seed
        output_dir: Directory to save plots
        user_id: User ID
    """
    from fsrs_optimizer import (
        Optimizer,
        Trainer,
        Collection as FSRSCollection,
        power_forgetting_curve,
    )  # type: ignore

    # Split data into 2 roughly equal parts
    n_total = len(user_dataset)
    n_train = n_total // 2

    # Sort by review_th to maintain temporal order
    df_sorted = user_dataset.sort_values("review_th").reset_index(drop=True)

    # Create splits
    train_set = df_sorted.iloc[:n_train].copy()
    test_set = df_sorted.iloc[n_train:].copy()

    if len(test_set) == 0:
        print(
            f"Warning: User {user_id} has no test data after splitting, skipping comparison."
        )
        return

    print(
        f"User {user_id}: Split sizes - Train: {len(train_set)}, Test: {len(test_set)}"
    )

    # 1. Train jiggle method on train_set
    jiggle_params = fit_fsrs6_parameters_for_user(
        user_dataset=train_set,
        n_jiggles=n_jiggles,
        config=config,
        seed=seed,
    )

    if not jiggle_params:
        print(
            f"Warning: User {user_id} failed to generate jiggle parameters, skipping comparison."
        )
        return

    # 2. Train standard FSRS on train_set (same as jiggle)
    optimizer = Optimizer(float_delta_t=config.use_secs_intervals)
    optimizer.define_model()
    _ = optimizer.initialize_parameters(dataset=train_set, verbose=False)

    if config.only_S0:
        standard_params = list(map(float, optimizer.init_w))
    else:
        trainer = Trainer(
            train_set,
            None,
            optimizer.init_w,
            n_epoch=N_EPOCH,
            lr=LR,
            gamma=GAMMA,
            batch_size=BATCH_SIZE,
            max_seq_len=config.max_seq_len,
            enable_short_term=config.include_short_term,
        )
        standard_params = list(map(float, trainer.train(verbose=False)))

    # 3. Predict on test set with both methods
    target_r = 0.9  # Target retrievability (90%)

    # Check if test set has labels (y column)
    has_labels = "y" in test_set.columns
    if has_labels:
        y_true = test_set["y"].to_numpy(dtype=np.float64)
    else:
        y_true = None
        print(
            f"  Warning: User {user_id} test set missing 'y' column, skipping log loss calculation."
        )

    # Jiggle predictions: get stability and retrievability for each jiggle, then calculate intervals
    jiggle_intervals = []
    jiggle_retrievabilities = []
    jiggle_log_losses = []
    for idx, w in enumerate(jiggle_params):
        collection = FSRSCollection(w)
        stabilities, _ = collection.batch_predict(test_set)
        stabilities_arr = np.asarray(stabilities, dtype=np.float64)

        # Calculate retrievability (predicted recall probability)
        retrievability = power_forgetting_curve(
            test_set["delta_t"].to_numpy(dtype=np.float64),
            stabilities_arr,
            -w[20],
        )
        jiggle_retrievabilities.append(retrievability)

        # Calculate interval from stability
        intervals = calculate_interval_from_stability(stabilities_arr, target_r, -w[20])
        jiggle_intervals.append(intervals)

        # Calculate log loss if labels are available
        if has_labels and y_true is not None:
            from sklearn.metrics import log_loss  # type: ignore

            try:
                logloss = log_loss(y_true=y_true, y_pred=retrievability, labels=[0, 1])
                jiggle_log_losses.append(float(logloss))
            except Exception as e:
                print(f"  Warning: Failed to calculate log loss for jiggle {idx}: {e}")
                jiggle_log_losses.append(None)

    jiggle_intervals_arr = np.array(jiggle_intervals)  # Shape: (n_jiggles, n_test)
    jiggle_interval_mean = np.mean(jiggle_intervals_arr, axis=0)
    jiggle_interval_std = np.std(jiggle_intervals_arr, axis=0, ddof=1)

    jiggle_retrievabilities_arr = np.array(
        jiggle_retrievabilities
    )  # Shape: (n_jiggles, n_test)
    jiggle_retrievability_mean = np.mean(jiggle_retrievabilities_arr, axis=0)
    jiggle_retrievability_std = np.std(jiggle_retrievabilities_arr, axis=0, ddof=1)

    # Standard FSRS prediction
    standard_collection = FSRSCollection(standard_params)
    standard_stabilities, _ = standard_collection.batch_predict(test_set)
    standard_stabilities_arr = np.asarray(standard_stabilities, dtype=np.float64)
    standard_intervals = calculate_interval_from_stability(
        standard_stabilities_arr, target_r, -standard_params[20]
    )
    standard_retrievability = power_forgetting_curve(
        test_set["delta_t"].to_numpy(dtype=np.float64),
        standard_stabilities_arr,
        -standard_params[20],
    )

    # Calculate standard FSRS log loss if labels are available
    standard_log_loss = None
    if has_labels and y_true is not None:
        from sklearn.metrics import log_loss  # type: ignore

        try:
            standard_log_loss = float(
                log_loss(y_true=y_true, y_pred=standard_retrievability, labels=[0, 1])
            )
        except Exception as e:
            print(f"  Warning: Failed to calculate standard FSRS log loss: {e}")

    # 4. Calculate normalized difference for intervals: (t_jiggle,mean - t_standard) / σ_t,jiggle
    # Avoid division by zero
    sigma_safe = np.where(jiggle_interval_std > 1e-10, jiggle_interval_std, 1e-10)
    normalized_interval_diff = (jiggle_interval_mean - standard_intervals) / sigma_safe

    # 4.1. Calculate normalized difference for retrievability: (R_jiggle,mean - R_standard) / σ_R,jiggle
    sigma_r_safe = np.where(
        jiggle_retrievability_std > 1e-10, jiggle_retrievability_std, 1e-10
    )
    normalized_retrievability_diff = (
        jiggle_retrievability_mean - standard_retrievability
    ) / sigma_r_safe

    # 4.1. Calculate and report log loss statistics
    log_loss_stats = None
    if jiggle_log_losses:
        valid_log_losses = [ll for ll in jiggle_log_losses if ll is not None]
        if valid_log_losses:
            log_loss_mean = np.mean(valid_log_losses)
            log_loss_std = (
                np.std(valid_log_losses, ddof=1) if len(valid_log_losses) > 1 else 0.0
            )
            log_loss_min = np.min(valid_log_losses)
            log_loss_max = np.max(valid_log_losses)
            log_loss_median = np.median(valid_log_losses)

            log_loss_stats = {
                "jiggle": {
                    "mean": float(log_loss_mean),
                    "std": float(log_loss_std),
                    "median": float(log_loss_median),
                    "min": float(log_loss_min),
                    "max": float(log_loss_max),
                    "all_values": [float(ll) for ll in valid_log_losses],
                },
                "standard": (
                    float(standard_log_loss) if standard_log_loss is not None else None
                ),
                "difference": (
                    float(log_loss_mean - standard_log_loss)
                    if standard_log_loss is not None
                    else None
                ),
            }

            print(
                f"  Jiggle Log Loss: mean={log_loss_mean:.6f}, std={log_loss_std:.6f}, "
                f"median={log_loss_median:.6f}, range=[{log_loss_min:.6f}, {log_loss_max:.6f}]"
            )

            if standard_log_loss is not None:
                print(f"  Standard FSRS Log Loss: {standard_log_loss:.6f}")
                print(
                    f"  Difference (Jiggle mean - Standard): {log_loss_mean - standard_log_loss:.6f}"
                )

    # 4.5. Automatically detect and analyze peak data points for intervals
    detected_peaks_interval = detect_peaks_in_distribution(normalized_interval_diff)
    if len(detected_peaks_interval) > 0:
        print(
            f"  Detected {len(detected_peaks_interval)} peaks in interval differences: {[f'{p:.3f}' for p in detected_peaks_interval]}"
        )
        # Analyze top peaks (limit to top 5 to avoid too many analyses)
        # Sort by absolute value to prioritize significant peaks
        peaks_to_analyze = sorted(
            detected_peaks_interval, key=lambda x: abs(x), reverse=True
        )[:5]
        analyze_peak_data_points(
            test_set=test_set,
            normalized_diff=normalized_interval_diff,
            jiggle_interval_mean=jiggle_interval_mean,
            jiggle_interval_std=jiggle_interval_std,
            standard_intervals=standard_intervals,
            peak_values=peaks_to_analyze,
            tolerance=0.05,
            output_dir=output_dir,
            user_id=user_id,
        )

    # 5. Create visualization - 4 subplots: interval and retrievability comparisons
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))

    test_indices = np.arange(len(normalized_interval_diff))

    # Plot 1: Normalized interval difference over test data points
    ax1 = axes[0]
    ax1.scatter(test_indices, normalized_interval_diff, alpha=0.6, s=20)
    ax1.axhline(y=0, color="red", linestyle="--", linewidth=1, label="Zero difference")
    ax1.axhline(y=1, color="orange", linestyle=":", linewidth=1, alpha=0.7, label="±1σ")
    ax1.axhline(y=-1, color="orange", linestyle=":", linewidth=1, alpha=0.7)
    ax1.set_xlabel("Test Data Point Index", fontsize=12)
    ax1.set_ylabel("(t_jiggle,mean - t_standard) / σ_t,jiggle", fontsize=12)
    ax1.set_title(
        f"User {user_id}: Normalized Interval Difference (Jiggle vs Standard FSRS)\n"
        f"Target R={target_r}, Test set size={len(test_set)}",
        fontsize=14,
        fontweight="bold",
    )
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Histogram of normalized interval differences
    ax2 = axes[1]
    ax2.hist(
        normalized_interval_diff,
        bins=50,
        alpha=0.7,
        edgecolor="black",
        color="steelblue",
    )
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero difference")
    ax2.axvline(
        x=np.mean(normalized_interval_diff),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(normalized_interval_diff):.3f}",
    )
    ax2.set_xlabel("(t_jiggle,mean - t_standard) / σ_t,jiggle", fontsize=12)
    ax2.set_ylabel("Frequency", fontsize=12)
    ax2.set_title(
        "Distribution of Normalized Interval Differences",
        fontsize=12,
        fontweight="bold",
    )
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Normalized retrievability difference over test data points
    ax3 = axes[2]
    ax3.scatter(
        test_indices, normalized_retrievability_diff, alpha=0.6, s=20, color="green"
    )
    ax3.axhline(y=0, color="red", linestyle="--", linewidth=1, label="Zero difference")
    ax3.axhline(y=1, color="orange", linestyle=":", linewidth=1, alpha=0.7, label="±1σ")
    ax3.axhline(y=-1, color="orange", linestyle=":", linewidth=1, alpha=0.7)
    ax3.set_xlabel("Test Data Point Index", fontsize=12)
    ax3.set_ylabel("(R_jiggle,mean - R_standard) / σ_R,jiggle", fontsize=12)
    ax3.set_title(
        "Normalized Retrievability Difference (Jiggle vs Standard FSRS)",
        fontsize=14,
        fontweight="bold",
    )
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Histogram of normalized retrievability differences
    ax4 = axes[3]
    ax4.hist(
        normalized_retrievability_diff,
        bins=50,
        alpha=0.7,
        edgecolor="black",
        color="lightgreen",
    )
    ax4.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero difference")
    ax4.axvline(
        x=np.mean(normalized_retrievability_diff),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(normalized_retrievability_diff):.3f}",
    )
    ax4.set_xlabel("(R_jiggle,mean - R_standard) / σ_R,jiggle", fontsize=12)
    ax4.set_ylabel("Frequency", fontsize=12)
    ax4.set_title(
        "Distribution of Normalized Retrievability Differences",
        fontsize=12,
        fontweight="bold",
    )
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Add statistics text for intervals
    stats_text_interval = (
        f"Interval Stats:\n"
        f"Mean: {np.mean(normalized_interval_diff):.4f}\n"
        f"Std: {np.std(normalized_interval_diff):.4f}\n"
        f"Median: {np.median(normalized_interval_diff):.4f}\n"
        f"Min: {np.min(normalized_interval_diff):.4f}\n"
        f"Max: {np.max(normalized_interval_diff):.4f}"
    )
    ax2.text(
        0.02,
        0.98,
        stats_text_interval,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Add statistics text for retrievability
    stats_text_retrievability = (
        f"Retrievability Stats:\n"
        f"Mean: {np.mean(normalized_retrievability_diff):.4f}\n"
        f"Std: {np.std(normalized_retrievability_diff):.4f}\n"
        f"Median: {np.median(normalized_retrievability_diff):.4f}\n"
        f"Min: {np.min(normalized_retrievability_diff):.4f}\n"
        f"Max: {np.max(normalized_retrievability_diff):.4f}"
    )
    ax4.text(
        0.02,
        0.98,
        stats_text_retrievability,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="lightgreen", alpha=0.5),
    )

    plt.tight_layout()

    # Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{user_id}_jiggle_vs_standard.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved jiggle vs standard comparison plot: {output_path}")

    # Save log loss statistics to JSON file
    if log_loss_stats is not None:
        log_loss_file = output_dir / f"{user_id}_log_loss.json"
        with open(log_loss_file, "w", encoding="utf-8") as f:
            json.dump(log_loss_stats, f, indent=2, ensure_ascii=False)
        print(f"  Saved log loss statistics: {log_loss_file}")


def main() -> None:
    """
    Run statistical uncertainty analysis for FSRS-6 using card-level bootstrap resampling.

    For each dataset (user), we:
      - Load data via the existing UserDataLoader
      - Train N_JIGGLES FSRS-6 models with card-level bootstrap resampling
      - Output all parameter sets per user as JSON lines
    """
    parser = create_parser()
    parser.add_argument(
        "--n_jiggles",
        type=int,
        default=100,
        help="Number of Monte Carlo bootstrap resamplings per user (card-level bootstrap).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="bootstrapping/FSRS-6-uncertainty.jsonl",
        help="Output JSONL file path for parameter samples.",
    )
    parser.add_argument(
        "--pred_output_dir",
        type=str,
        default="bootstrapping/predictions",
        help="Directory to save per-user prediction TSV files.",
    )
    parser.add_argument(
        "--plot_output_dir",
        type=str,
        default="bootstrapping/plots/stability",
        help="Directory to save stability distribution plots.",
    )
    parser.add_argument(
        "--top_n_combos",
        type=int,
        default=100,
        help="Number of top (r_history, t_history) combinations to visualize.",
    )
    parser.add_argument(
        "--combos_per_page",
        type=int,
        default=10,
        help="Number of combinations to show per plot page.",
    )
    parser.add_argument(
        "--param_plot_output_dir",
        type=str,
        default="bootstrapping/plots/params",
        help="Directory to save parameter distribution plots.",
    )
    parser.add_argument(
        "--params_per_page",
        type=int,
        default=7,
        help="Number of parameters to show per plot page (arranged in 2 columns).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--compare_jiggle_vs_standard",
        action="store_true",
        help="Compare jiggle method vs standard FSRS fitting on split test data.",
    )
    parser.add_argument(
        "--comparison_output_dir",
        type=str,
        default="bootstrapping/plots/comparison",
        help="Directory to save jiggle vs standard comparison plots.",
    )
    parser.add_argument(
        "--max_users_for_comparison",
        type=int,
        default=10,
        help="Maximum number of users to run comparison on (default: 10).",
    )
    parser.add_argument(
        "--validation_output_dir",
        type=str,
        default="bootstrapping/plots/validation",
        help="Directory to save z-score validation plots.",
    )
    parser.add_argument(
        "--max_z_scores",
        type=int,
        default=1000,
        help="Maximum number of z-scores to collect for validation (default: 1000).",
    )
    parser.add_argument(
        "--min_data_size_for_validation",
        type=int,
        default=1000,
        help="Minimum data size per user for validation (default: 1000).",
    )
    parser.add_argument(
        "--validation_only",
        action="store_true",
        help="Only run validation (skip parameter fitting, predictions, and other visualizations).",
    )
    args, _ = parser.parse_known_args()

    # Force algorithm to be FSRS-6 for this script
    args.algo = "FSRS-6"

    # Set random seeds for reproducibility
    seed = args.seed
    np.random.seed(seed)
    random.seed(seed)
    # Note: torch.manual_seed would be set if using PyTorch, but we're using fsrs_optimizer here

    config = Config(args)
    ensure_fsrs_optimizer_on_path(config)

    n_jiggles: int = args.n_jiggles

    # Resolve paths relative to project root
    script_dir = Path(__file__).parent
    project_root = script_dir.parent

    output_path = (project_root / args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pred_output_dir = (project_root / args.pred_output_dir).resolve()
    plot_output_dir = (project_root / args.plot_output_dir).resolve()
    param_plot_output_dir = (project_root / args.param_plot_output_dir).resolve()
    comparison_output_dir = (project_root / args.comparison_output_dir).resolve()
    top_n_combos: int = args.top_n_combos

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

    # If validation_only mode, skip all other processing and go directly to validation
    if args.validation_only:
        validation_output_dir = (project_root / args.validation_output_dir).resolve()
        collect_and_visualize_z_scores(
            data_loader=data_loader,
            user_ids=user_ids,
            n_jiggles=n_jiggles,
            config=config,
            seed=seed,
            output_dir=validation_output_dir,
            max_z_scores=args.max_z_scores,
            min_data_size=args.min_data_size_for_validation,
        )
        return

    with output_path.open("w", encoding="utf-8") as out_f:
        for user_id in user_ids:
            try:
                user_dataset = data_loader.load_user_data(user_id)
            except Exception:
                # Skip users with insufficient or invalid data
                continue

            # Use a deterministic seed for each user based on global seed and user_id
            # This ensures reproducibility while allowing different users to have different jiggle patterns
            user_seed = seed + user_id
            param_samples = fit_fsrs6_parameters_for_user(
                user_dataset=user_dataset,
                n_jiggles=n_jiggles,
                config=config,
                seed=user_seed,
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

            # Visualize stability distributions for top N combinations
            pred_file_path = pred_output_dir / f"{user_id}.tsv"
            if pred_file_path.exists():
                visualize_stability_distributions(
                    pred_file_path=pred_file_path,
                    user_id=user_id,
                    n_jiggles=len(param_samples),
                    output_dir=plot_output_dir,
                    top_n=top_n_combos,
                    combos_per_page=args.combos_per_page,
                )

            # Visualize parameter distributions
            visualize_parameter_distributions(
                param_samples=param_samples,
                user_id=user_id,
                output_dir=param_plot_output_dir,
                params_per_page=args.params_per_page,
            )

            # Compare jiggle vs standard FSRS if requested
            if args.compare_jiggle_vs_standard:
                # Only process first N users for comparison
                user_index = user_ids.index(user_id)
                if user_index < args.max_users_for_comparison:
                    compare_jiggle_vs_standard(
                        user_dataset=user_dataset,
                        n_jiggles=n_jiggles,
                        config=config,
                        seed=user_seed,
                        output_dir=comparison_output_dir,
                        user_id=user_id,
                    )


if __name__ == "__main__":
    main()
