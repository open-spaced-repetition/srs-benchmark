"""
FSRS-rs model wrapper for integration with the other.py architecture.

This module provides utilities to work with the fsrs-rs (Rust-based FSRS implementation)
within the benchmark framework.
"""

from typing import List, Optional
import pandas as pd
from config import Config


def convert_to_items(df: pd.DataFrame, config: Config):
    """
    Convert a pandas DataFrame to a list of FSRSItem objects for fsrs-rs.

    Args:
        df: DataFrame with columns: card_id, review_th, t_history, r_history, delta_t, rating
        config: Configuration object

    Returns:
        list[FSRSItem]: List of FSRS items for training/evaluation
    """
    try:
        from fsrs_rs_python import FSRSItem, FSRSReview
    except ImportError:
        raise ImportError(
            "fsrs-rs-python is not installed. Please install it to use FSRS-rs models."
        )

    def accumulate(group):
        items = []
        for _, row in group.iterrows():
            t_history = [max(0, int(t)) for t in row["t_history"].split(",")] + [
                row["delta_t"]
            ]
            r_history = [int(t) for t in row["r_history"].split(",")] + [row["rating"]]
            items.append(
                (
                    row["review_th"],
                    FSRSItem(
                        reviews=[
                            FSRSReview(delta_t=int(x[0]), rating=int(x[1]))
                            for x in zip(t_history, r_history)
                        ]
                    ),
                )
            )
        return items

    result_list = sum(
        df.sort_values(by=["card_id", "review_th"])
        .groupby("card_id")[
            ["review_th", "t_history", "r_history", "delta_t", "rating"]
        ]
        .apply(accumulate)
        .tolist(),
        [],
    )
    result_list = list(map(lambda x: x[1], sorted(result_list, key=lambda x: x[0])))

    return result_list


class FSRSRsBackend:
    """Wrapper for FSRS-rs backend."""

    def __init__(self, config: Config):
        """
        Initialize FSRS-rs backend.

        Args:
            config: Configuration object
        """
        try:
            from fsrs_rs_python import FSRS
        except ImportError:
            raise ImportError(
                "fsrs-rs-python is not installed. Please install it to use FSRS-rs models."
            )

        self.config = config
        self.backend = FSRS(parameters=[])

    def train(self, train_set: pd.DataFrame) -> List[float]:
        """
        Train FSRS-rs model on training data.

        Args:
            train_set: Training dataset

        Returns:
            List[float]: Trained FSRS parameters (weights)
        """
        train_set_items = convert_to_items(train_set, self.config)
        weights = list(
            map(lambda x: round(x, 4), self.backend.benchmark(train_set_items))
        )
        return weights

    def predict(
        self, testset: pd.DataFrame, weights: List[float]
    ) -> tuple[List[float], List[float], pd.DataFrame]:
        """
        Make predictions using FSRS-rs model.

        Args:
            testset: Test dataset
            weights: FSRS parameters

        Returns:
            tuple: (predictions, labels, testset_with_predictions)
        """
        from fsrs_optimizer import Collection, power_forgetting_curve  # type: ignore

        my_collection = Collection(weights)
        testset_copy = testset.copy()

        testset_copy["stability"], testset_copy["difficulty"] = (
            my_collection.batch_predict(testset_copy)
        )
        testset_copy["p"] = power_forgetting_curve(
            testset_copy["delta_t"],
            testset_copy["stability"],
            -weights[20],
        )

        p = testset_copy["p"].tolist()
        y = testset_copy["y"].tolist()

        return p, y, testset_copy
