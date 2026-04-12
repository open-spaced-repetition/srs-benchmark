"""
Feature engineering and data loading for FSRS-7.

FSRS-7 always uses:
  - sub-second intervals  (delta_t = elapsed_seconds / 86400)
  - short-term reviews included
"""

from __future__ import annotations

from itertools import accumulate

import numpy as np
import pandas as pd
import torch


# ── utilities ─────────────────────────────────────────────────────────────────


def _cum_concat(x: list) -> list:
    return list(accumulate(x))


# ── core feature engineering ──────────────────────────────────────────────────


def _common_preprocess(df: pd.DataFrame, config) -> pd.DataFrame:
    df["review_th"] = range(1, df.shape[0] + 1)
    df["nth_today"] = df.groupby("day_offset").cumcount() + 1
    df.sort_values(by=["card_id", "review_th"], inplace=True)

    df.drop(df[~df["rating"].isin([1, 2, 3, 4])].index, inplace=True)

    if config.two_buttons:
        df["rating"] = df["rating"].replace({2: 3, 4: 3})

    df["i"] = df.groupby("card_id").cumcount() + 1
    df.drop(df[df["i"] > config.max_seq_len * 2].index, inplace=True)

    # Build delta_t columns
    if "delta_t" not in df.columns:
        df["delta_t"] = df["elapsed_days"]
        df["delta_t_secs"] = (df["elapsed_seconds"] / 86400).clip(lower=0)

    df["delta_t"] = df["delta_t"].clip(lower=0)
    return df


def _compute_histories(df: pd.DataFrame, config) -> pd.DataFrame:
    """Compute t_history / r_history / last_rating columns."""
    # Always use elapsed_days for last_rating (same as original)
    t_hist_days = df.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: _cum_concat([[i] for i in x])
    )
    t_hist_secs = df.groupby("card_id", group_keys=False)["delta_t_secs"].apply(
        lambda x: _cum_concat([[i] for i in x])
    )
    r_hist = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: _cum_concat([[i] for i in x])
    )

    # last_rating: most recent review with elapsed_days > 0
    last_rating = []
    for t_sub, r_sub in zip(t_hist_days, r_hist):
        for t_item, r_item in zip(t_sub, r_sub):
            found = False
            for t, r in zip(reversed(t_item[:-1]), reversed(r_item[:-1])):
                if t > 0:
                    last_rating.append(r)
                    found = True
                    break
            if not found:
                last_rating.append(r_item[0])
    df["last_rating"] = last_rating

    df["r_history"] = [
        ",".join(map(str, item[:-1]))
        for sublist in r_hist
        for item in sublist
    ]
    df["t_history"] = [
        ",".join(map(str, item[:-1]))
        for sublist in t_hist_secs
        for item in sublist
    ]

    # Switch delta_t to seconds for the rest of the pipeline
    df["delta_t"] = df["delta_t_secs"]
    return df


def _build_tensors(df: pd.DataFrame, config) -> pd.DataFrame:
    """Add the 'tensor' column used by BatchDataset.
    At this point delta_t is already delta_t_secs."""
    t_hist = df.groupby("card_id", group_keys=False)["delta_t"].apply(
        lambda x: _cum_concat([[i] for i in x])
    )
    r_hist = df.groupby("card_id", group_keys=False)["rating"].apply(
        lambda x: _cum_concat([[i] for i in x])
    )

    df["tensor"] = [
        torch.tensor((t_item[:-1], r_item[:-1]), dtype=torch.float32).transpose(0, 1)
        for t_sub, r_sub in zip(t_hist, r_hist)
        for t_item, r_item in zip(t_sub, r_sub)
    ]
    return df


def _common_postprocess(df: pd.DataFrame, config) -> pd.DataFrame:
    df["first_rating"] = df["r_history"].map(lambda x: x[0] if x else "")
    df["y"] = df["rating"].map({1: 0, 2: 1, 3: 1, 4: 1})

    # Keep same-day reviews, but always keep the very first review of each card.
    if config.include_short_term:
        df = df[(df["delta_t"] != 0) | (df["i"] == 1)].copy()

    # Recalculate i as the count of non-same-day reviews seen so far
    df["i"] = (
        df.groupby("card_id")
        .apply(lambda x: (x["elapsed_days"] > 0).cumsum(), include_groups=False)
        .reset_index(level=0, drop=True)
        + 1
    )

    return df[df["delta_t"] > 0].sort_values(by=["review_th"])


def create_features(df: pd.DataFrame, config) -> pd.DataFrame:
    """Main entry point: feature-engineer a raw review-log DataFrame for FSRS-7."""
    df = _common_preprocess(df.copy(), config)
    df = _compute_histories(df, config)
    df = _build_tensors(df, config)
    df = _common_postprocess(df, config)
    return df


# ── data loader ───────────────────────────────────────────────────────────────


def load_user_data(user_id: int, config) -> pd.DataFrame:
    """Load parquet data for one user and return a feature-engineered DataFrame."""
    df_revlogs = pd.read_parquet(config.data_path / "revlogs" / f"{user_id=}")
    dataset = create_features(df_revlogs, config)

    if dataset.shape[0] < 6:
        raise Exception(f"{user_id} does not have enough data.")

    if config.partitions != "none":
        df_cards = pd.read_parquet(
            config.data_path / "cards", filters=[("user_id", "=", user_id)]
        )
        df_cards.drop(columns=["user_id"], inplace=True)
        df_decks = pd.read_parquet(
            config.data_path / "decks", filters=[("user_id", "=", user_id)]
        )
        df_decks.drop(columns=["user_id"], inplace=True)
        dataset = dataset.merge(df_cards, on="card_id", how="left").merge(
            df_decks, on="deck_id", how="left"
        )
        dataset.fillna(-1, inplace=True)
        if config.partitions == "preset":
            dataset["partition"] = dataset["preset_id"].astype(int)
        elif config.partitions == "deck":
            dataset["partition"] = dataset["deck_id"].astype(int)
    else:
        dataset["partition"] = 0

    return dataset
