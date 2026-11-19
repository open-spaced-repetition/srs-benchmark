import numpy as np
import pandas as pd

from config import Config
from features import create_features


class InsufficientUserDataError(ValueError):
    """Raised when a user has too few reviews to run the benchmark."""


def read_revlogs(config: Config, user_id: int) -> pd.DataFrame:
    """Read one user's revlogs, pruned to the columns feature engineering consumes.

    All columns are kept when exporting evaluation files (``--file``).
    """
    columns = None
    if not config.save_evaluation_file:
        columns = ["card_id", "rating", "elapsed_days"]
        if config.use_secs_intervals:
            columns.append("elapsed_seconds")
        if config.lstm_use_duration and config.model_name in ("LSTM", "GRU"):
            columns.append("duration")
    return pd.read_parquet(
        config.data_path / "revlogs" / f"{user_id=}", columns=columns
    )


def _fill_numeric_na(df: pd.DataFrame) -> None:
    """In-place ``fillna(-1)`` on numeric columns that contain NaN.

    String/object/tensor columns are left untouched (a frame-wide ``fillna`` breaks on
    pandas 3 string dtypes and copies tensor columns).
    """
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]) and df[col].isna().any():
            df[col] = df[col].fillna(-1)


class UserDataLoader:
    """A class responsible for loading and preprocessing user data."""

    def __init__(self, config: Config):
        self.config = config
        self.data_path = config.data_path

    def _read_user_partition(
        self, kind: str, user_id: int, columns: list[str] | None = None
    ) -> pd.DataFrame:
        """Read one user's cards/decks from the direct hive-partition path.

        Equivalent to ``read_parquet(kind, filters=[("user_id", "=", uid)])`` followed
        by dropping the ``user_id`` column, but avoids pyarrow re-discovering all ~3000
        partitions on every call (~700 ms -> ~3 ms). The partition path encodes
        ``user_id``, so it is absent from the returned columns either way. Users with no
        partition directory (e.g. revlogs but no cards) fall back to the filtered read,
        which yields the same empty frame.
        """
        path = self.data_path / kind / f"user_id={user_id}"
        if path.exists():
            df = pd.read_parquet(path, columns=columns)
            if "user_id" in df.columns:
                df = df.drop(columns=["user_id"])
            return df
        filter_cols = (
            [*columns, "user_id"]
            if columns is not None and "user_id" not in columns
            else columns
        )
        df = pd.read_parquet(
            self.data_path / kind,
            filters=[("user_id", "=", user_id)],
            columns=filter_cols,
        )
        return df.drop(columns=["user_id"])

    def load_user_data(self, user_id: int) -> pd.DataFrame:
        """
        Load and prepare user data including revlogs, cards, and decks.

        Args:
            user_id: The ID of the user whose data to load

        Returns:
            pd.DataFrame: Processed dataset with features

        Raises:
            InsufficientUserDataError: If there is not enough data for the user
        """
        dataset = create_features(
            read_revlogs(self.config, user_id), config=self.config
        )

        if dataset.shape[0] < 6:
            raise InsufficientUserDataError(f"{user_id} does not have enough data.")

        if self.config.partitions == "none":
            dataset["partition"] = 0
        elif self.config.save_evaluation_file:
            # Full merge when exporting evaluation files so card/deck metadata columns exist
            df_cards = self._read_user_partition("cards", user_id)
            df_decks = self._read_user_partition("decks", user_id)
            dataset = dataset.merge(df_cards, on="card_id", how="left").merge(
                df_decks, on="deck_id", how="left"
            )
            _fill_numeric_na(dataset)
            partition_col = (
                "preset_id" if self.config.partitions == "preset" else "deck_id"
            )
            dataset["partition"] = dataset[partition_col].astype(int)
        else:
            # Same fill as the merge path (e.g. rmse_bins_lapse on same-day rows under
            # --equalize_test_with_non_secs), then map card_id -> partition from the small
            # metadata tables instead of merging into the full review/tensor frame.
            _fill_numeric_na(dataset)
            partition_by_card = self._partition_by_card(user_id)
            dataset["partition"] = (
                dataset["card_id"]
                .astype(np.int64)
                .map(partition_by_card)
                .fillna(-1)
                .astype(int)
            )
            dataset.reset_index(drop=True, inplace=True)

        return dataset

    def _partition_by_card(self, user_id: int) -> pd.Series:
        """``card_id``-indexed deck or preset id (NaN where the deck is unknown)."""
        df_cards = self._read_user_partition(
            "cards", user_id, columns=["card_id", "deck_id"]
        )
        partition = df_cards["deck_id"]
        if self.config.partitions == "preset":
            df_decks = self._read_user_partition(
                "decks", user_id, columns=["deck_id", "preset_id"]
            )
            partition = partition.map(df_decks.set_index("deck_id")["preset_id"])
        return pd.Series(partition.to_numpy(), index=df_cards["card_id"])
