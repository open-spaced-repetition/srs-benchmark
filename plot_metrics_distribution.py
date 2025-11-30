#!/usr/bin/env python3
"""
Script to visualize the distribution of metrics for each model.
Generates distribution plots for each metric and saves them to plots/<metric_name>/<model_name>.png
"""

import json
import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_metric_distribution(
    model_name: str,
    metric_name: str,
    values: list[float],
    sizes: list[int] | None = None,
    output_dir: Path | None = None,
    xlim: tuple[float, float] | None = None,
):
    """
    Plot the distribution of a metric for a model.

    Args:
        model_name: Name of the model
        metric_name: Name of the metric
        values: List of metric values
        sizes: Optional list of sample sizes for weighting
        output_dir: Output directory for the plot
    """
    # Filter out None and NaN values
    valid_indices = [
        i
        for i, v in enumerate(values)
        if v is not None and not (isinstance(v, float) and np.isnan(v))
    ]
    if len(valid_indices) == 0:
        print(f"  Warning: No valid values for {metric_name}, skipping...")
        return

    filtered_values = np.array([values[i] for i in valid_indices])
    filtered_sizes = (
        np.array([sizes[i] for i in valid_indices])
        if sizes
        else np.ones_like(filtered_values)
    )

    # Create output directory (organized by metric name)
    if output_dir is None:
        output_dir = Path("plots") / metric_name
    else:
        output_dir = Path(output_dir) / metric_name
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Calculate statistics
    mean = np.average(filtered_values, weights=filtered_sizes)
    median = np.median(filtered_values)

    # Calculate weighted standard deviation
    if len(filtered_values) > 1 and np.sum(filtered_sizes) > 0:
        # Use the same method as evaluate.py for consistency
        n_eff = (
            np.sum(filtered_sizes) ** 2 / np.sum(filtered_sizes**2)
            if np.sum(filtered_sizes**2) > 0
            else len(filtered_values)
        )
        variance = np.average((filtered_values - mean) ** 2, weights=filtered_sizes)
        if n_eff > 1:
            variance = variance * (n_eff / (n_eff - 1))
        std = np.sqrt(variance)
    else:
        std = 0.0

    # Create histogram
    counts, bins, patches = ax.hist(
        filtered_values,
        bins=50,
        weights=filtered_sizes if sizes else None,
        alpha=0.7,
        edgecolor="black",
        linewidth=0.5,
        color="steelblue",
    )

    # Add vertical lines for statistics
    ax.axvline(
        mean,
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Mean: {mean:.4f}",
    )
    ax.axvline(
        median,
        color="orange",
        linestyle="--",
        linewidth=2,
        label=f"Median: {median:.4f}",
    )
    # Add std deviation lines (only label once)
    if std > 0:
        ax.axvline(
            mean - std,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
            label=f"±1 Std: [{mean - std:.4f}, {mean + std:.4f}]",
        )
        ax.axvline(
            mean + std,
            color="gray",
            linestyle=":",
            linewidth=1.5,
            alpha=0.7,
        )

    # Add text box with statistics
    stats_text = f"Mean: {mean:.4f}\nMedian: {median:.4f}\nStd: {std:.4f}\nN: {len(filtered_values)}"
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
    )

    # Set x-axis limits if provided (skip for LogLoss to allow automatic scaling)
    if xlim is not None and metric_name != "LogLoss":
        ax.set_xlim(xlim)

    # Set labels and title
    xlabel = f"{metric_name} value"
    # Add explanation for MBE
    if metric_name == "MBE":
        xlabel += "\n(MBE > 0: overestimation, MBE < 0: underestimation)"
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("Frequency" if sizes is None else "Weighted frequency", fontsize=12)
    ax.set_title(
        f"{model_name} - {metric_name} Distribution", fontsize=14, fontweight="bold"
    )
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)

    # Save figure
    output_path = output_dir / f"{model_name}.png"
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"  Saved: {output_path}")


def collect_model_data(
    result_file: Path, metrics: list[str]
) -> tuple[str, dict[str, list[float]], list[int]]:
    """
    Collect metric data from a result file.

    Returns:
        Tuple of (model_name, values_dict, sizes)
    """
    model_name = result_file.stem
    values_dict = {metric: [] for metric in metrics}
    sizes = []

    try:
        with open(result_file, "r") as f:
            for line in f:
                data = json.loads(line.strip())
                if "metrics" not in data:
                    continue

                for metric in metrics:
                    value = data["metrics"].get(metric)
                    values_dict[metric].append(value)

                if "size" in data:
                    sizes.append(data["size"])
                else:
                    sizes.append(1)
    except Exception as e:
        print(f"Error reading {result_file}: {e}")
        return model_name, {}, []

    return model_name, values_dict, sizes


def calculate_metric_ranges(
    all_model_data: dict[str, dict[str, list[float]]], metrics: list[str]
) -> dict[str, tuple[float, float] | None]:
    """
    Calculate the global min and max for each metric across all models.

    Returns:
        Dictionary mapping metric names to (min, max) tuples
    """
    ranges = {}
    for metric in metrics:
        all_values = []
        for model_name, values_dict in all_model_data.items():
            values = values_dict.get(metric, [])
            valid_values = [
                v
                for v in values
                if v is not None and not (isinstance(v, float) and np.isnan(v))
            ]
            all_values.extend(valid_values)

        if len(all_values) > 0:
            min_val = np.min(all_values)
            max_val = np.max(all_values)
            # Add a small margin (5% on each side)
            margin = (max_val - min_val) * 0.05
            ranges[metric] = (min_val - margin, max_val + margin)
        else:
            ranges[metric] = None

    return ranges


def process_model(
    model_name: str,
    values_dict: dict[str, list[float]],
    sizes: list[int],
    metrics: list[str],
    use_weights: bool,
    metric_ranges: dict[str, tuple[float, float] | None] | None = None,
    output_dir: Path | None = None,
):
    """
    Process a single model and create distribution plots.

    Args:
        model_name: Name of the model
        values_dict: Dictionary mapping metric names to lists of values
        sizes: List of sample sizes
        metrics: List of metric names to plot
        use_weights: Whether to weight by sample size
        metric_ranges: Optional dictionary mapping metric names to (min, max) tuples
        output_dir: Output directory for plots
    """
    if len(sizes) == 0:
        print(f"No data found for {model_name}")
        return

    print(f"Processing {model_name} ({len(sizes)} users)...")

    # Plot each metric
    for metric in metrics:
        values = values_dict.get(metric, [])
        if not values:
            print(f"  Warning: No values for {metric}, skipping...")
            continue

        xlim = metric_ranges.get(metric) if metric_ranges else None

        if use_weights:
            plot_metric_distribution(
                model_name, metric, values, sizes, output_dir, xlim=xlim
            )
        else:
            plot_metric_distribution(
                model_name, metric, values, None, output_dir, xlim=xlim
            )


def main():
    parser = argparse.ArgumentParser(
        description="Visualize metric distributions for all models"
    )
    parser.add_argument(
        "--result-dir",
        type=str,
        default="result",
        help="Directory containing result JSONL files (default: result)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="plots",
        help="Output directory for plots (default: plots)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["LogLoss", "RMSE(bins)", "AUC", "MBE"],
        help="Metrics to plot (default: LogLoss RMSE(bins) AUC MBE)",
    )
    parser.add_argument(
        "--weights",
        action="store_true",
        help="Weight by sample size (default: no weighting)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Process only a specific model (filename without .jsonl extension)",
    )

    args = parser.parse_args()

    result_dir = Path(args.result_dir)
    if not result_dir.exists():
        print(f"Error: Result directory '{result_dir}' does not exist")
        return

    # Find all JSONL files
    if args.model:
        result_files = [result_dir / f"{args.model}.jsonl"]
        if not result_files[0].exists():
            print(f"Error: Model file '{result_files[0]}' does not exist")
            return
    else:
        result_files = sorted(result_dir.glob("*.jsonl"))

    if len(result_files) == 0:
        print(f"No JSONL files found in '{result_dir}'")
        return

    print(f"Found {len(result_files)} model(s) to process")
    print(f"Metrics to plot: {', '.join(args.metrics)}")
    print(f"Using weights: {args.weights}\n")

    # First pass: collect all data to calculate global ranges
    print("Collecting data from all models...")
    all_model_data = {}
    all_model_sizes = {}
    for result_file in result_files:
        try:
            model_name, values_dict, sizes = collect_model_data(
                result_file, args.metrics
            )
            if values_dict:
                all_model_data[model_name] = values_dict
                all_model_sizes[model_name] = sizes
        except Exception as e:
            print(f"Error reading {result_file.name}: {e}")
            continue

    if len(all_model_data) == 0:
        print("No valid data found in any model files")
        return

    # Calculate global ranges for each metric (except LogLoss)
    metrics_to_scale = [m for m in args.metrics if m != "LogLoss"]
    if metrics_to_scale:
        print("Calculating global ranges for consistent x-axis scaling...")
        metric_ranges = calculate_metric_ranges(all_model_data, metrics_to_scale)
        for metric, range_val in metric_ranges.items():
            if range_val:
                print(f"  {metric}: [{range_val[0]:.4f}, {range_val[1]:.4f}]")
    else:
        metric_ranges = {}

    # LogLoss uses automatic scaling (not uniform range)
    if "LogLoss" in args.metrics:
        print("  LogLoss: using automatic scaling (no uniform range)")

    print("\nGenerating plots...")

    # Second pass: generate plots with consistent x-axis ranges
    for model_name in all_model_data.keys():
        try:
            process_model(
                model_name,
                all_model_data[model_name],
                all_model_sizes[model_name],
                args.metrics,
                use_weights=args.weights,
                metric_ranges=metric_ranges,
                output_dir=Path(args.output_dir),
            )
        except Exception as e:
            print(f"Error processing {model_name}: {e}")
            continue

    print(f"\nDone! Plots saved to {args.output_dir}/")


if __name__ == "__main__":
    main()
