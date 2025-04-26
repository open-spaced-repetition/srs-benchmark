import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import root_mean_squared_error  # type: ignore
import traceback
from functools import wraps


def catch_exceptions(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs), None
        except Exception as e:
            return None, traceback.format_exc()

    return wrapper


def rmse_matrix(df):
    """
    Args:
        df (pd.DataFrame): DataFrame containing review data with columns like
                           'r_history', 't_history', 'elapsed_days', 'i',
                           'y' (true outcome, e.g., 0 or 1), 'p' (predicted recall prob),
                           and optionally 'weights'.
    Returns:
        float: The calculated root mean squared error. Returns np.nan if
               calculation is not possible (e.g., empty data after processing).
    """
    if df.empty:
        print("Warning: Input DataFrame is empty.")
        return np.nan

    # Ensure required columns exist (adjust based on actual minimum requirements)
    required_cols = ['r_history', 't_history', 'elapsed_days', 'i', 'y', 'p']
    if not all(col in df.columns for col in required_cols):
        missing_cols = [col for col in required_cols if col not in df.columns]
        raise ValueError(f"Missing required columns: {missing_cols}")

    tmp = df.copy()

    # --- Helper function to count lapses ---
    def count_lapse(r_history, t_history):
        lapse = 0
        # Defensive coding: ensure inputs are strings and handle potential errors
        r_hist_str = str(r_history)
        t_hist_str = str(t_history)
        if not r_hist_str or not t_hist_str: # Handle empty strings
            return 0
        try:
            r_parts = r_hist_str.split(",")
            t_parts = t_hist_str.split(",")
            # Iterate only up to the minimum length to avoid index errors
            min_len = min(len(r_parts), len(t_parts))
            for i in range(min_len):
                r = r_parts[i].strip() # Add strip for potential whitespace
                t = t_parts[i].strip()
                # Check if t is non-zero and r indicates a lapse (rating 1)
                if t != "0" and r == "1":
                    lapse += 1
        except Exception as e:
            # Log or handle potential errors during splitting or comparison
            print(f"Warning: Error processing history: r='{r_history}', t='{t_history}'. Error: {e}")
            # Depending on desired behavior, return 0 or propagate error
            return 0 # Defaulting to 0 lapses on error
        return lapse

    # --- Binning Logic (Identical to the original function) ---

    # Calculate 'lapse' column
    tmp["lapse"] = tmp.apply(
        lambda x: count_lapse(x["r_history"], x["t_history"]), axis=1
    )

    # Define binning functions with error handling for log arguments
    log_base_362 = np.log(3.62)
    def bin_delta_t(x):
        if pd.isna(x) or x <= 0: return 0 # Handle NaN or non-positive values
        return round(2.48 * np.power(3.62, np.floor(np.log(x) / log_base_362)), 2)

    log_base_189 = np.log(1.89)
    def bin_i(x):
        if pd.isna(x) or x <= 0: return 0 # Handle NaN or non-positive values
        return round(1.99 * np.power(1.89, np.floor(np.log(x) / log_base_189)), 0)

    log_base_173 = np.log(1.73)
    def bin_lapse(x):
        if pd.isna(x) or x <= 0: return 0 # Handle NaN or non-positive values
        # Original logic used x != 0, changed to x > 0 for log consistency
        return round(1.65 * np.power(1.73, np.floor(np.log(x) / log_base_173)), 0)

    # Apply binning functions
    tmp["delta_t"] = tmp["elapsed_days"].apply(bin_delta_t)
    tmp["i"] = tmp["i"].apply(bin_i)
    tmp["lapse"] = tmp["lapse"].apply(bin_lapse) # Apply binning to the calculated lapse counts

    # Handle weights: If 'weights' column doesn't exist, add it with value 1.
    # If it exists but has NaNs, fill them with 1.
    if "weights" not in tmp.columns:
        tmp["weights"] = 1.0
    else:
        # Ensure weights are numeric and fill NaNs
        tmp["weights"] = pd.to_numeric(tmp["weights"], errors='coerce').fillna(1.0)
        # Ensure weights are non-negative
        tmp.loc[tmp["weights"] < 0, "weights"] = 0

    # Ensure 'y' and 'p' are numeric, coercing errors to NaN
    tmp['y'] = pd.to_numeric(tmp['y'], errors='coerce')
    tmp['p'] = pd.to_numeric(tmp['p'], errors='coerce')

    # Drop rows where essential numeric columns or binning columns became NaN
    essential_cols_for_calc = ['delta_t', 'i', 'lapse', 'y', 'p', 'weights']
    tmp.dropna(subset=essential_cols_for_calc, inplace=True)

    if tmp.empty:
        print("Warning: DataFrame became empty after cleaning/binning.")
        return np.nan

    # --- New Calculation Logic ---

    # 1. Calculate the average 'y' for each bin.
    # Note: This calculates an unweighted average 'y' within each bin, matching
    # the apparent behavior of the original function's aggregation step.
    y_avg_per_bin = tmp.groupby(["delta_t", "i", "lapse"])["y"].mean().reset_index()
    y_avg_per_bin = y_avg_per_bin.rename(columns={"y": "y_avg_bin"})

    # 2. Merge the bin's average 'y' back onto the main dataframe.
    # Each row now has its original 'p' and the average 'y' of its bin.
    tmp = tmp.merge(y_avg_per_bin, on=["delta_t", "i", "lapse"], how="left")

    # Drop rows where merge might have failed (shouldn't happen with how='left'
    # if y_avg_per_bin covers all groups present in tmp, but good practice).
    tmp.dropna(subset=['y_avg_bin'], inplace=True)

    if tmp.empty:
        print("Warning: DataFrame became empty after merging average y.")
        return np.nan

    # 3. Calculate the squared difference for each review: (y_avg_bin - p_individual)^2
    tmp["sq_diff"] = (tmp["y_avg_bin"] - tmp["p"]) ** 2

    # 4. Calculate the weighted mean squared error.
    # We use the individual review weights ('weights' column).
    total_weight = tmp["weights"].sum()

    if total_weight <= 0:
        print(f"Warning: Total weight ({total_weight}) is non-positive. Cannot compute weighted average.")
        return np.nan

    weighted_mean_sq_error = np.average(tmp["sq_diff"], weights=tmp["weights"])

    # 5. Calculate the final RMSE.
    rmse_new = np.sqrt(weighted_mean_sq_error)

    return rmse_new


# def rmse_matrix(df):
#     tmp = df.copy()

#     def count_lapse(r_history, t_history):
#         lapse = 0
#         for r, t in zip(r_history.split(","), t_history.split(",")):
#             if t != "0" and r == "1":
#                 lapse += 1
#         return lapse

#     tmp["lapse"] = tmp.apply(
#         lambda x: count_lapse(x["r_history"], x["t_history"]), axis=1
#     )
#     tmp["delta_t"] = tmp["elapsed_days"].map(
#         lambda x: round(2.48 * np.power(3.62, np.floor(np.log(x) / np.log(3.62))), 2)
#     )
#     tmp["i"] = tmp["i"].map(
#         lambda x: round(1.99 * np.power(1.89, np.floor(np.log(x) / np.log(1.89))), 0)
#     )
#     tmp["lapse"] = tmp["lapse"].map(
#         lambda x: (
#             round(1.65 * np.power(1.73, np.floor(np.log(x) / np.log(1.73))), 0)
#             if x != 0
#             else 0
#         )
#     )
#     if "weights" not in tmp.columns:
#         tmp["weights"] = 1
#     tmp = (
#         tmp.groupby(["delta_t", "i", "lapse"])
#         .agg({"y": "mean", "p": "mean", "weights": "sum"})
#         .reset_index()
#     )
#     return root_mean_squared_error(tmp["y"], tmp["p"], sample_weight=tmp["weights"])


def cross_comparison(revlogs, algoA, algoB, graph=False):
    if algoA != algoB:
        cross_comparison_record = revlogs[[f"R ({algoA})", f"R ({algoB})", "y"]].copy()
        bin_algo = (
            algoA,
            algoB,
        )
        pair_algo = [(algoA, algoB), (algoB, algoA)]
    else:
        cross_comparison_record = revlogs[[f"R ({algoA})", "y"]].copy()
        bin_algo = (algoA,)
        pair_algo = [(algoA, algoA)]

    def get_bin(x, bins=20):
        return (
            np.log(np.minimum(np.floor(np.exp(np.log(bins + 1) * x) - 1), bins - 1) + 1)
            / np.log(bins)
        ).round(3)

    for algo in bin_algo:
        cross_comparison_record[f"{algo}_B-W"] = (
            cross_comparison_record[f"R ({algo})"] - cross_comparison_record["y"]
        )
        cross_comparison_record[f"{algo}_bin"] = cross_comparison_record[
            f"R ({algo})"
        ].map(get_bin)

    if graph:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.gca()
        ax.axhline(y=0.0, color="black", linestyle="-")

    universal_metric_list = []

    for algoA, algoB in pair_algo:
        cross_comparison_group = cross_comparison_record.groupby(by=f"{algoA}_bin").agg(
            {"y": ["mean"], f"{algoB}_B-W": ["mean"], f"R ({algoB})": ["mean", "count"]}
        )
        universal_metric = root_mean_squared_error(
            y_true=cross_comparison_group["y", "mean"],
            y_pred=cross_comparison_group[f"R ({algoB})", "mean"],
            sample_weight=cross_comparison_group[f"R ({algoB})", "count"],
        )
        cross_comparison_group[f"R ({algoB})", "percent"] = (
            cross_comparison_group[f"R ({algoB})", "count"]
            / cross_comparison_group[f"R ({algoB})", "count"].sum()
        )
        if graph:
            ax.scatter(
                cross_comparison_group.index,
                cross_comparison_group[f"{algoB}_B-W", "mean"],
                s=cross_comparison_group[f"R ({algoB})", "percent"] * 1024,
                alpha=0.5,
            )
            ax.plot(
                cross_comparison_group[f"{algoB}_B-W", "mean"],
                label=f"{algoB} by {algoA}, UM={universal_metric:.4f}",
            )
        universal_metric_list.append(universal_metric)
    if graph:
        ax.legend(loc="lower center")
        ax.grid(linestyle="--")
        ax.set_title(f"{algoA} vs {algoB}")
        ax.set_xlabel("Predicted R")
        ax.set_ylabel("B-W Metric")
        ax.set_xlim(0, 1)
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        fig.show()
    return universal_metric_list
