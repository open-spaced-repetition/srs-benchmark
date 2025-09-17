import pandas as pd
from config import Config
from features import create_features


class UserDataLoader:
    """A class responsible for loading and preprocessing user data."""

    def __init__(self, config: Config):
        self.config = config
        self.data_path = config.data_path

    def load_user_data(self, user_id: int) -> pd.DataFrame:
        """
        Load and prepare user data including revlogs, cards, and decks.

        Args:
            user_id: The ID of the user whose data to load

        Returns:
            pd.DataFrame: Processed dataset with features

        Raises:
            Exception: If there is not enough data for the user
        """
        # Load revlogs
        df_revlogs = pd.read_parquet(
            self.data_path / "revlogs" / f"{user_id=}",
        )

        # Create initial features
        dataset = create_features(df_revlogs, config=self.config)

        if dataset.shape[0] < 6:
            raise Exception(f"{user_id} does not have enough data.")

        # Handle partitions if needed
        if self.config.partitions != "none":
            # Load cards and decks
            df_cards = pd.read_parquet(
                self.data_path / "cards", filters=[("user_id", "=", user_id)]
            )
            df_cards.drop(columns=["user_id"], inplace=True)

            df_decks = pd.read_parquet(
                self.data_path / "decks", filters=[("user_id", "=", user_id)]
            )
            df_decks.drop(columns=["user_id"], inplace=True)

            # Merge all data
            dataset = dataset.merge(df_cards, on="card_id", how="left").merge(
                df_decks, on="deck_id", how="left"
            )
            dataset.fillna(-1, inplace=True)

            # Set partition based on config
            if self.config.partitions == "preset":
                dataset["partition"] = dataset["preset_id"].astype(int)
            elif self.config.partitions == "deck":
                dataset["partition"] = dataset["deck_id"].astype(int)
        else:
            dataset["partition"] = 0

        return dataset
