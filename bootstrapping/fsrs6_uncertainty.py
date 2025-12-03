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
from scipy.signal import find_peaks  # type: ignore

# Add parent directory to path to import from project root
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root))

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

    Each training uses the same data but different binomial jiggle weights.

    Args:
        user_dataset: User's dataset
        n_jiggles: Number of jiggle trainings
        config: Configuration object
        seed: Random seed for reproducibility (default: 42)
    """
    # Lazy import after sys.path has potentially been updated
    from fsrs_optimizer import Optimizer, Trainer  # type: ignore

    optimizer = Optimizer(float_delta_t=config.use_secs_intervals)

    # Make sure elapsed_days exists if we want to filter same-day rows
    if config.no_train_same_day and "elapsed_days" not in user_dataset.columns:
        raise ValueError("Column 'elapsed_days' is required but missing from dataset.")

    all_parameter_sets: List[List[float]] = []

    # Create a random number generator with the seed for this user
    # Use a deterministic seed based on user_id and global seed for reproducibility
    rng = np.random.default_rng(seed)

    for jiggle_idx in range(n_jiggles):
        dataset = user_dataset.copy()

        if config.no_train_same_day:
            dataset = dataset[dataset["elapsed_days"] > 0].copy()

        if len(dataset) == 0:
            # Not enough data after filtering; skip this jiggle
            continue

        # Binomial jiggle weights: 0 or 2, as described in the spec
        # Use the RNG to ensure reproducibility
        weights = rng.binomial(n=1, p=0.5, size=len(dataset)).astype("float32") * 2.0

        # Ensure at least one non-zero weight so that training is meaningful
        if np.all(weights == 0):
            # Force a single sample to have non-zero weight
            weights[rng.integers(0, len(weights))] = 2.0

        dataset["weights"] = weights

        # Initialize FSRS-6 model parameters for this dataset
        optimizer.define_model()
        # _ = optimizer.initialize_parameters(dataset=dataset, verbose=False)

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
            f"  Coverage: {coverage*100:.2f}% ({total_assigned}/{total_points} points assigned to peaks)"
        )
        print(
            f"  Unassigned: {(1-coverage)*100:.2f}% ({total_points - total_assigned} points)"
        )
        for result in analysis_results:
            print(f"\n  Peak at {result['peak_value']:.2f}:")
            print(
                f"    Number of points: {result['n_points']} ({result['fraction']*100:.2f}% of test set)"
            )
            if "top_r_history" in result:
                print(f"    Top r_history patterns:")
                for pattern, count in list(result["top_r_history"].items())[:5]:
                    print(f"      '{pattern}': {count}")
            if "top_t_history" in result:
                print(f"    Top t_history patterns:")
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
    # _ = optimizer.initialize_parameters(dataset=train_set, verbose=False)

    if config.only_S0:
        standard_params = list(map(float, optimizer.init_w))
    else:
        trainer = Trainer(
            train_set,
            None,
            optimizer.init_w,
            n_epoch=5,
            lr=4e-2,
            gamma=1.0,
            batch_size=config.batch_size,
            max_seq_len=config.max_seq_len,
            enable_short_term=config.include_short_term,
        )
        standard_params = list(map(float, trainer.train(verbose=False)))

    # 3. Predict on test set with both methods
    target_r = 0.9  # Target retrievability (90%)

    # Jiggle predictions: get stability for each jiggle, then calculate intervals
    jiggle_intervals = []
    for w in jiggle_params:
        collection = FSRSCollection(w)
        stabilities, _ = collection.batch_predict(test_set)
        stabilities_arr = np.asarray(stabilities, dtype=np.float64)
        intervals = calculate_interval_from_stability(stabilities_arr, target_r, -w[20])
        jiggle_intervals.append(intervals)

    jiggle_intervals_arr = np.array(jiggle_intervals)  # Shape: (n_jiggles, n_test)
    jiggle_interval_mean = np.mean(jiggle_intervals_arr, axis=0)
    jiggle_interval_std = np.std(jiggle_intervals_arr, axis=0, ddof=1)

    # Standard FSRS prediction
    standard_collection = FSRSCollection(standard_params)
    standard_stabilities, _ = standard_collection.batch_predict(test_set)
    standard_stabilities_arr = np.asarray(standard_stabilities, dtype=np.float64)
    standard_intervals = calculate_interval_from_stability(
        standard_stabilities_arr, target_r, -standard_params[20]
    )

    # 4. Calculate normalized difference: (t_jiggle,mean - t_standard) / σ_t,jiggle
    # Avoid division by zero
    sigma_safe = np.where(jiggle_interval_std > 1e-10, jiggle_interval_std, 1e-10)
    normalized_diff = (jiggle_interval_mean - standard_intervals) / sigma_safe

    # 4.5. Automatically detect and analyze peak data points
    detected_peaks = detect_peaks_in_distribution(normalized_diff)
    if len(detected_peaks) > 0:
        print(
            f"  Detected {len(detected_peaks)} peaks: {[f'{p:.3f}' for p in detected_peaks]}"
        )
        # Analyze top peaks (limit to top 5 to avoid too many analyses)
        # Sort by absolute value to prioritize significant peaks
        peaks_to_analyze = sorted(detected_peaks, key=lambda x: abs(x), reverse=True)[
            :5
        ]
        analyze_peak_data_points(
            test_set=test_set,
            normalized_diff=normalized_diff,
            jiggle_interval_mean=jiggle_interval_mean,
            jiggle_interval_std=jiggle_interval_std,
            standard_intervals=standard_intervals,
            peak_values=peaks_to_analyze,
            tolerance=0.05,
            output_dir=output_dir,
            user_id=user_id,
        )

    # 5. Create visualization
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))

    # Plot 1: Normalized difference over test data points
    ax1 = axes[0]
    test_indices = np.arange(len(normalized_diff))
    ax1.scatter(test_indices, normalized_diff, alpha=0.6, s=20)
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

    # Plot 2: Histogram of normalized differences
    ax2 = axes[1]
    ax2.hist(normalized_diff, bins=50, alpha=0.7, edgecolor="black", color="steelblue")
    ax2.axvline(x=0, color="red", linestyle="--", linewidth=2, label="Zero difference")
    ax2.axvline(
        x=np.mean(normalized_diff),
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {np.mean(normalized_diff):.3f}",
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

    # Add statistics text
    stats_text = (
        f"Mean: {np.mean(normalized_diff):.4f}\n"
        f"Std: {np.std(normalized_diff):.4f}\n"
        f"Median: {np.median(normalized_diff):.4f}\n"
        f"Min: {np.min(normalized_diff):.4f}\n"
        f"Max: {np.max(normalized_diff):.4f}"
    )
    ax2.text(
        0.02,
        0.98,
        stats_text,
        transform=ax2.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    plt.tight_layout()

    # Save the plot
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{user_id}_jiggle_vs_standard.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved jiggle vs standard comparison plot: {output_path}")


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
