"""
This script demonstrates RWKV run as an RNN.
"""

import numpy as np
import pandas as pd
from rwkv.config import (
    DAY_OFFSET_ENCODE_PERIODS,
    ID_ENCODE_DIMS,
    ID_SPLIT,
    RWKV_SUBMODULES,
)
from rwkv.data_processing import (
    CARD_FEATURE_COLUMNS,
    ID_PLACEHOLDER,
    scale_cum_new_cards_today,
    scale_cum_reviews_today,
    scale_day_offset_diff,
    scale_diff_new_cards,
    scale_diff_reviews,
    scale_duration,
    scale_elapsed_days,
    scale_elapsed_days_cumulative,
    scale_elapsed_seconds,
    scale_elapsed_seconds_cumulative,
    scale_state,
    # Import all specific scaling functions that are used in add_same
    scale_elapsed_days,
    scale_elapsed_days_cumulative,
    scale_elapsed_seconds,
    scale_elapsed_seconds_cumulative,
    scale_duration,
    scale_diff_new_cards,
    scale_diff_reviews,
    scale_cum_new_cards_today,
    scale_cum_reviews_today,
    scale_day_offset_diff,
    # STATS might also be needed if not encapsulated fully by scaling functions,
    # but it seems the functions from data_processing already use them.
)
from rwkv.get_result import get_benchmark_info, get_stats
import torch
from rwkv.model.srs_model_rnn import SrsRWKVRnn
from rwkv.parse_toml import parse_toml
from rwkv.architecture import DEFAULT_ANKI_RWKV_CONFIG


class RNNProcess:
    """
    Manages the state and processing for running the SrsRWKVRnn model sequentially,
    one review at a time, for a single user.

    This class keeps track of RNN states for different submodules (card, note, deck, preset)
    and global state. It also handles feature engineering for each incoming review row
    to match the expected input format of the RNN model.
    """
    def __init__(
        self,
        path,
        device,
        dtype,
        config=DEFAULT_ANKI_RWKV_CONFIG,
    ):
        """
        Initializes the RNNProcess.

        Args:
            path: Path to the trained SrsRWKVRnn model state dictionary.
                  If None, model weights are not loaded (e.g., for random init).
            device: PyTorch device to run the model on.
            dtype: PyTorch data type for the model.
            config: Model architecture configuration (default: DEFAULT_ANKI_RWKV_CONFIG).
        """
        self.rnn = SrsRWKVRnn(config).to(device)
        if path is not None:
            self.rnn.load_state_dict(torch.load(path, weights_only=True))
            print(f"Loaded: {path}")
        else:
            print("Did not load weights.")
        self.rnn = self.rnn.selective_cast(dtype) # Apply selective casting for mixed precision
        self.device = device
        self.dtype = dtype

        # State dictionaries for each submodule and global state
        self.card_states = {}
        self.note_states = {}
        self.deck_states = {}
        self.preset_states = {}
        self.global_state = None
        self.first_day_offset = None
        self.prev_row = None
        self.card_set = set()
        self.last_new_cards = {}
        self.i = 0
        self.last_i = {}
        self.today = -1
        self.today_reviews = 0
        self.today_new_cards = 0
        self.card2first_day_offset = {}
        self.card2elapsed_days_cumulative = {}
        self.card2elapsed_seconds_cumulative = {}
        self.id_encodings = {submodule: {} for submodule in RWKV_SUBMODULES} # Cache for generated ID encodings

    def get_tensor(self, row):
        """
        Converts a DataFrame row into a feature tensor suitable for the SrsRWKVRnn model.

        This involves:
        1. Extracting features listed in CARD_FEATURE_COLUMNS.
        2. Generating and adding ID encodings for each submodule (if not already cached).
        3. Generating and adding sinusoidal day offset encodings.
        4. Unsqueezing to add a batch dimension.

        Args:
            row: A Pandas Series representing a single review log entry,
                 augmented with necessary scaled features by `add_same`.

        Returns:
            A PyTorch tensor of shape (1, feature_dim).
        """
        # TODO: Investigate refactoring ID and day_offset encoding to share code
        # with rwkv.training.prepare_batch._add_encodings, if feasible.
        # The main difference is per-row processing here vs. batch processing there.

        def _generate_or_get_id_encoding(submodule_name, id_val):
            """Generates or retrieves a cached random encoding for a given ID."""
            if id_val not in self.id_encodings[submodule_name]:
                ENCODE_DIM = ID_ENCODE_DIMS[submodule_name]
                self.id_encodings[submodule_name][id_val] = (torch.randint(
                    low=0,
                    high=ID_SPLIT,
                    size=(ENCODE_DIM,),
                    device=self.device,
                    requires_grad=False,
                ).to(self.dtype) - ((ID_SPLIT - 1) / 2))
            return self.id_encodings[submodule_name][id_val]

        current_features_list = [
            torch.tensor(
                row.loc[CARD_FEATURE_COLUMNS].values.astype(np.float32), # Ensure correct initial dtype for tensor
                dtype=self.dtype,
                device=self.device,
                requires_grad=False,
            )
        ]

        for submodule_name in RWKV_SUBMODULES:
            if submodule_name == "user_id": # user_id is part of main features, not separate encoding
                continue
            id_value = row[submodule_name]
            current_features_list.append(_generate_or_get_id_encoding(submodule_name, id_value))

        features_with_ids = torch.cat(current_features_list, dim=-1)

        # Add day offset encodings
        def _add_day_offset_encodings_to_tensor(tensor_features):
            # Using tensor_features.device to ensure new tensors are on the same device
            day_offset_val = torch.full((1,), row["day_offset"], device=tensor_features.device)
            day_offset_first_val = torch.full(
                (1,), row["day_offset_first"], device=tensor_features.device
            )

            encodings_gather_list = [tensor_features] # Start with the input tensor
            for period in DAY_OFFSET_ENCODE_PERIODS:
                f = 2 * np.pi / period
                # Day offset
                sin_enc = torch.sin(f * (day_offset_val % period)).to(self.dtype)
                cos_enc = torch.cos(f * (day_offset_val % period)).to(self.dtype)
                encodings_gather_list.append(torch.cat((sin_enc, cos_enc), dim=-1))
                # First day offset
                sin_enc_first = torch.sin(f * (day_offset_first_val % period)).to(self.dtype)
                cos_enc_first = torch.cos(f * (day_offset_first_val % period)).to(self.dtype)
                encodings_gather_list.append(torch.cat((sin_enc_first, cos_enc_first), dim=-1))

            return torch.cat(encodings_gather_list, dim=-1)

        features_with_all_encodings = _add_day_offset_encodings_to_tensor(features_with_ids)
        return features_with_all_encodings.unsqueeze(0) # Add batch dimension

    def run(self, row, skip):
        """
        Processes a single review using the SrsRWKVRnn model and updates states.

        Args:
            row: Pandas Series for the review, pre-processed by `add_same` and `get_tensor`.
            skip: Boolean, if True, model states are not updated. This is used for
                  'immediate' predictions where the current review's outcome is unknown.

        Returns:
            A tuple (curve_params, imm_prediction):
            - curve_params: Tuple (out_ahead_logits, out_w) for predicting future recall.
            - imm_prediction: Immediate prediction (e.g., probability of pressing "Again").
        """
        features = self.get_tensor(row)

        with torch.no_grad():
            card_id = row["card_id"]
            note_id = row["note_id"]
            deck_id = row["deck_id"]
            preset_id = row["preset_id"]

            if card_id not in self.card_states:
                self.card_states[card_id] = None
            if note_id not in self.note_states:
                self.note_states[note_id] = None
            if deck_id not in self.deck_states:
                self.deck_states[deck_id] = None
            if preset_id not in self.preset_states:
                self.preset_states[preset_id] = None

            (
                out_ahead_logits,
                out_w,
                out_p_logits,
                next_card_state,
                next_note_state,
                next_deck_state,
                next_preset_state,
                next_global_state,
            ) = self.rnn.review( # Call the actual RNN model
                features,
                self.card_states[card_id],    # Previous state for this card_id
                self.note_states[note_id],    # Previous state for this note_id
                self.deck_states[deck_id],    # Previous state for this deck_id
                self.preset_states[preset_id],  # Previous state for this preset_id
                self.global_state,            # Previous global state
            )
            # Update states only if not skipping (i.e., this is a full review processing, not just a query)
            if not skip:
                self.card_states[card_id] = next_card_state
                self.note_states[note_id] = next_note_state
                self.deck_states[deck_id] = next_deck_state
                self.preset_states[preset_id] = next_preset_state
                self.global_state = next_global_state

            out_p_probs = torch.softmax(out_p_logits, dim=-1)
            out_p_again, _, _, _ = out_p_probs.unbind(dim=-1)
            return (out_ahead_logits, out_w), 1.0 - out_p_again

    def predict_func(self, curve, elapsed_seconds):
        """
        Predicts recall probability for a given forgetting curve and elapsed time.

        Args:
            curve: Tuple (out_ahead_logits, out_w) from a previous `run` call.
            elapsed_seconds: Time elapsed since the review that produced the curve.

        Returns:
            Predicted recall probability (sigmoid of combined logits).
        """
        elapsed_seconds = torch.tensor(elapsed_seconds, device=self.device, dtype=self.dtype).view(1, 1)
        out_ahead_logits, out_w = curve
        curve_probs_raw = self.rnn.forgetting_curve(out_w, elapsed_seconds)
        curve_logits_raw = torch.log(
            curve_probs_raw / (1 - curve_probs_raw)
        )  # inverse sigmoid
        ahead_logit_residual = self.rnn.interp(out_ahead_logits, elapsed_seconds)
        curve_logits = curve_logits_raw + ahead_logit_residual
        return torch.sigmoid(curve_logits)

    def add_same(self, row):
        """
        Performs feature engineering on a raw review log row to prepare it for `get_tensor`.
        This includes calculating cumulative stats, scaling features, and handling IDs.
        Many of these operations mirror those in `rwkv.data.data_processing.get_rwkv_data`.

        Args:
            row: A Pandas Series representing a raw review log entry.

        Returns:
            A new Pandas Series with added and scaled features.
        """
        # Feature engineering using imported scaling functions
        row = row.copy()
        card_id = row["card_id"]

        # Cumulative day/second calculations (specific to this RNN processing context)
        row["elapsed_days_cumulative"] = (
            self.card2elapsed_days_cumulative.get(card_id, 0.0) + row["elapsed_days"]
        )
        row["elapsed_seconds_cumulative"] = (
            self.card2elapsed_seconds_cumulative.get(card_id, 0.0)
            + row["elapsed_seconds"]
        )

        # Apply imported scaling functions
        row["scaled_elapsed_days"] = scale_elapsed_days(row["elapsed_days"])
        row["scaled_elapsed_days_cumulative"] = scale_elapsed_days_cumulative(row["elapsed_days_cumulative"])
        row["scaled_elapsed_seconds"] = scale_elapsed_seconds(row["elapsed_seconds"])
        row["scaled_elapsed_seconds_cumulative"] = scale_elapsed_seconds_cumulative(row["elapsed_seconds_cumulative"])
        # Duration is scaled in process_row, not here, as it's an outcome.
        # For imm_predict, scaled_duration will be 0.
        # row["scaled_duration"] = scale_duration(row["duration"])

        # Sin/Cos encodings for time (daily cycle)
        SECONDS_PER_DAY = 86400.0
        row["elapsed_seconds_sin"] = np.sin(
            (row["elapsed_seconds"] % SECONDS_PER_DAY) * 2 * np.pi / SECONDS_PER_DAY
        )
        row["elapsed_seconds_cos"] = np.cos(
            (row["elapsed_seconds"] % SECONDS_PER_DAY) * 2 * np.pi / SECONDS_PER_DAY
        )
        row["elapsed_seconds_cumulative_sin"] = np.sin(
            (row["elapsed_seconds_cumulative"] % SECONDS_PER_DAY)
            * 2
            * np.pi
            / SECONDS_PER_DAY
        )
        row["elapsed_seconds_cumulative_cos"] = np.cos(
            (row["elapsed_seconds_cumulative"] % SECONDS_PER_DAY)
            * 2
            * np.pi
            / SECONDS_PER_DAY
        )

        if self.first_day_offset is None:
            row["day_offset"] = 0
        else:
            row["day_offset"] -= self.first_day_offset

        if card_id in self.card2first_day_offset:
            row["day_offset_first"] = self.card2first_day_offset[card_id]
        else:
            row["day_offset_first"] = row["day_offset"]

        row["day_of_week"] = ((row["day_offset"] % 7) - 3) / 3

        def add_id(name):
            if np.isnan(row[name]):
                row[name] = ID_PLACEHOLDER
                row[f"{name}_is_nan"] = 1.0
            else:
                row[f"{name}_is_nan"] = 0.0

        for name in ["note_id", "deck_id", "preset_id"]:
            add_id(name)

        row["day_offset_diff"] = scale_day_offset_diff( # Uses imported function
            row["day_offset"]
            - (0 if self.prev_row is None else self.prev_row["day_offset"])
        )
        row["day_of_week"] = ((row["day_offset"] % 7) - 3) / 3 # Simple calculation, kept inline

        # Diff new cards / reviews (relative to this card's history)
        unscaled_diff_new_cards = (
            (len(self.card_set) - self.last_new_cards.get(card_id, len(self.card_set))) # if card_id not in last_new_cards, means it's a new card in this session
        )
        row["diff_new_cards"] = scale_diff_new_cards(unscaled_diff_new_cards) # Uses imported function

        unscaled_diff_reviews = (
            (max(0, self.i - self.last_i.get(card_id, self.i) - 1)) # if card_id not in last_i, it's the first review in session
        )
        row["diff_reviews"] = scale_diff_reviews(unscaled_diff_reviews) # Uses imported function

        # Cumulative new cards / reviews for the day
        current_day_reviews = self.today_reviews
        current_day_new_cards = self.today_new_cards
        if row["day_offset"] != self.today: # If new day started
            current_day_new_cards = 0
            current_day_reviews = -1 # Will be incremented to 0 for the first review of the day

        # These are for the *current* review being processed
        actual_today_reviews = current_day_reviews + 1
        actual_today_new_cards = current_day_new_cards + (1 if row["card_id"] not in self.card_set else 0)

        row["cum_reviews_today"] = scale_cum_reviews_today(actual_today_reviews) # Uses imported function
        row["cum_new_cards_today"] = scale_cum_new_cards_today(actual_today_new_cards) # Uses imported function

        # scaled_state is handled in process_row or imm_predict as it depends on context (query or actual)
        # row["scaled_state"] = scale_state(row["state"])
        return row

    def imm_predict(self, row):
        """
        Performs an "immediate" prediction for a given row.
        This means predicting the outcome of *this* review as if its rating/duration
        were unknown. The model's state is NOT updated.

        Args:
            row: A Pandas Series for the current review (raw, before state update).

        Returns:
            The immediate prediction probability (e.g., P(Again)).
        """
        row = row.copy()
        # Add features, but then zero out those that would leak current answer
        row = self.add_same(row) # Calculate all features first
        row["is_query"] = 1.0 # Mark as a query
        row["skip"] = True
        row["scaled_duration"] = 0
        row["scaled_state"] = 0
        for i in range(1, 5):
            row[f"rating_{i}"] = 0

        _, imm_probs = self.run(row, skip=True)
        return imm_probs

    def process_row(self, row):
        """
        Processes a row representing a completed review:
        1. Adds and scales features.
        2. Runs the RNN model to get outputs and update states.
        3. Updates internal tracking variables (cumulative stats, day counters, etc.).

        Args:
            row: A Pandas Series for the completed review.

        Returns:
            The curve parameters (out_ahead_logits, out_w) from the model.
        """
        row = row.copy()
        row = self.add_same(row) # Calculate all features based on this completed review
        row["is_query"] = 0.0 # Not a query, this is a learning step
        row["skip"] = False
        row["scaled_duration"] = scale_duration(row["duration"])
        row["scaled_state"] = scale_state(row["state"])
        for i in range(1, 5):
            row[f"rating_{i}"] = 1.0 if row["rating"] == i else 0.0

        # Get prediction
        curve, _ = self.run(row, skip=False)

        # Update state
        card_id = row["card_id"]
        self.card2elapsed_days_cumulative[card_id] = (
            self.card2elapsed_days_cumulative.get(card_id, 0) + row["elapsed_days"]
        )
        self.card2elapsed_seconds_cumulative[card_id] = (
            self.card2elapsed_seconds_cumulative.get(card_id, 0)
            + row["elapsed_seconds"]
        )

        if self.first_day_offset is None:
            self.first_day_offset = row["day_offset"]

        if row["day_offset"] != self.today:
            self.today = row["day_offset"]
            self.today_new_cards = 0
            self.today_reviews = -1
        self.today_reviews += 1
        if row["card_id"] not in self.card_set:
            self.today_new_cards += 1
            self.card_set.add(card_id)
            self.card2first_day_offset[card_id] = (
                row["day_offset"] - self.first_day_offset
            )

        self.prev_row = row.copy()
        self.last_i[card_id] = self.i
        self.last_new_cards[card_id] = len(self.card_set)
        self.i += 1
        return curve


@torch.inference_mode()
def run(data_path, model_path, label_db_path, label_db_size, user_id, verbose):
    """
    Runs the SrsRWKVRnn model sequentially for a single user's review history.

    This function simulates the step-by-step processing of reviews as they would occur in reality.
    It loads the user's data, initializes the RNNProcess, and then iterates through each review:
    - For each review, it first predicts the recall probability based on the previously stored
      forgetting curve for that card (if available). This is the "ahead" prediction.
    - It then predicts the immediate outcome (e.g., probability of pressing "Again") for the
      current review *before* processing its actual outcome.
    - Finally, it processes the actual outcome of the current review, updating the model's state
      and storing the new forgetting curve parameters for the card.
    - Collects all predictions and true labels to calculate and print final performance metrics.

    The primary purpose is to demonstrate and verify the information flow in the RNN model
    and to provide a way to run inference in a strictly sequential, non-leaky manner.
    The performance might differ slightly from batch processing due to RNG differences in
    encodings or slight variations in floating point operations.

    Args:
        data_path: Path to the directory containing 'revlogs.parquet', 'cards.parquet', etc.
        model_path: Path to the trained SrsRWKVRnn model state dictionary.
        label_db_path: Path to the LMDB database containing benchmark info (equalize_review_ths).
        label_db_size: Size of the label LMDB database.
        user_id: The ID of the user whose data to process.
        verbose: If True, prints detailed per-review predictions and truths.
    """
    # Load user's review logs and merge with card/deck info
    df = pd.read_parquet(data_path / "revlogs", filters=[("user_id", "=", user_id)])
    df["review_th"] = range(1, df.shape[0] + 1) # Add review_th if not present
    df_cards = pd.read_parquet(data_path / "cards", filters=[("user_id", "=", user_id)])
    df_cards.drop(columns=["user_id"], inplace=True)
    df_decks = pd.read_parquet(data_path / "decks", filters=[("user_id", "=", user_id)])
    df_decks.drop(columns=["user_id", "parent_id"], inplace=True)
    df = df.merge(df_cards, on="card_id", how="left", validate="many_to_one")
    df = df.merge(df_decks, on="deck_id", how="left", validate="many_to_one")
    # Ensure review_th is present after merges
    df["review_th"] = range(1, df.shape[0] + 1)


    equalize_review_ths, rmse_bins = get_benchmark_info(
        label_db_path, label_db_size, user_id
    )
    rmse_bins_dict = {
        equalize_review_ths[i]: rmse_bins[i] for i in range(len(equalize_review_ths))
    }

    srs_rnn = RNNProcess(
        path=model_path, device=torch.device("cpu"), dtype=torch.float32
    )

    pred_imm = {} # review_th -> immediate prediction
    pred_ahead_curve = {}  # card_id -> latest curve parameters (logits, w)
    pred_ahead = {} # review_th -> ahead prediction (recall probability)
    label_rating = {} # review_th -> actual rating - 1 (0 to 3)
    import time

    print("revlog len:", len(df))

    time_start = time.time()
    for i, row in df.iterrows():
        if (i + 1) % 100 == 0:
            print(f"{i+1}/{len(df)}, rate: {(i + 1) / (time.time() - time_start):.2f} reviews/sec")

        card_id = row["card_id"]
        review_th = row["review_th"]

        # If this card has a previously stored forgetting curve,
        # predict current recall probability using that curve and current elapsed_seconds.
        if card_id in pred_ahead_curve:
            pred_ahead[review_th] = srs_rnn.predict_func(
                pred_ahead_curve[card_id], row["elapsed_seconds"]
            ).item() # Store as float

        # Predict immediate outcome for the current review (before processing its actual result)
        # Create a copy of the row, dropping actual outcome columns (rating, duration)
        # as these would not be known at the time of immediate prediction.
        imm_info_row = row.copy()
        if 'rating' in imm_info_row: imm_info_row.drop('rating', inplace=True)
        if 'duration' in imm_info_row: imm_info_row.drop('duration', inplace=True)
        # Ensure other necessary fields for add_same are present if they were derived from rating/duration
        # For imm_predict, some features that depend on current rating/duration might be zeroed out
        # or set to defaults within imm_predict or add_same when called by imm_predict.
        pred_imm[review_th] = srs_rnn.imm_predict(imm_info_row).item() # Store as float

        # Process the actual current review, update RNN state, and get new curve parameters.
        # The `process_row` method handles feature engineering based on the actual 'row' data.
        pred_ahead_curve[card_id] = srs_rnn.process_row(row)
        label_rating[review_th] = row["rating"] - 1 # Store 0-indexed rating

        if verbose and review_th in pred_ahead:
            print(
                f"review_th: {review_th}, ahead_pred: {pred_ahead[review_th]:.3f}, imm_pred: {pred_imm[review_th]:.3f}, truth_binary: {int(row['rating'] >= 2)}"
            )
        elif verbose:
             print(
                f"review_th: {review_th}, (first_rev_no_ahead_pred), imm_pred: {pred_imm[review_th]:.3f}, truth_binary: {int(row['rating'] >= 2)}"
            )


    imm_stats, _ = get_stats(
        user_id,
        equalize_review_ths,
        rmse_bins_dict,
        pred_imm,
        label_rating,
    )
    # Filter pred_ahead to only include review_ths that are in equalize_review_ths for fair comparison
    # as some first reviews won't have an "ahead" prediction.
    filtered_pred_ahead = {th: p for th, p in pred_ahead.items() if th in equalize_review_ths}
    filtered_label_rating_for_ahead = {th: r for th, r in label_rating.items() if th in equalize_review_ths}

    # Ensure that equalize_review_ths for ahead_stats only contains keys present in filtered_pred_ahead
    valid_equalize_review_ths_for_ahead = [th for th in equalize_review_ths if th in filtered_pred_ahead]

    ahead_stats, _ = get_stats(
        user_id,
        valid_equalize_review_ths_for_ahead, # Use filtered list
        rmse_bins_dict,
        filtered_pred_ahead,
        filtered_label_rating_for_ahead,
    )

    print("--- Results for User:", user_id, "---")
    print("Immediate Predictions (RWKV-P like):")
    print(imm_stats)
    print("\nAhead-of-Time Predictions (Standard Recall Prediction):")
    print(ahead_stats)


if __name__ == "__main__":
    from pathlib import Path

    config = parse_toml()
    run(
        data_path=Path(
            config.DATA_PATH,
        ),
        model_path=config.MODEL_PATH,
        label_db_path=config.LABEL_FILTER_LMDB_PATH,
        label_db_size=config.LABEL_FILTER_LMDB_SIZE,
        user_id=config.USER,
        verbose=config.VERBOSE,
    )
    - not all computations have the right dtype (bfloat16, float)
    - JIT may fuse some floating point operations
    However, the performance should be similar.
    """
    df = pd.read_parquet(data_path / "revlogs", filters=[("user_id", "=", user_id)])
    df["review_th"] = range(1, df.shape[0] + 1)
    df_cards = pd.read_parquet(data_path / "cards", filters=[("user_id", "=", user_id)])
    df_cards.drop(columns=["user_id"], inplace=True)
    df_decks = pd.read_parquet(data_path / "decks", filters=[("user_id", "=", user_id)])
    df_decks.drop(columns=["user_id", "parent_id"], inplace=True)
    df = df.merge(df_cards, on="card_id", how="left", validate="many_to_one")
    df = df.merge(df_decks, on="deck_id", how="left", validate="many_to_one")
    df["review_th"] = range(1, df.shape[0] + 1)

    equalize_review_ths, rmse_bins = get_benchmark_info(
        label_db_path, label_db_size, user_id
    )
    rmse_bins_dict = {
        equalize_review_ths[i]: rmse_bins[i] for i in range(len(equalize_review_ths))
    }

    srs_rnn = RNNProcess(
        path=model_path, device=torch.device("cpu"), dtype=torch.float32
    )

    pred_imm = {}
    pred_ahead_curve = {}  # Map from card_ids to their latest forgetting curve
    pred_ahead = {}
    label_rating = {}
    import time

    print("revlog len:", len(df))

    time_start = time.time()
    for i, row in df.iterrows():
        if (i + 1) % 100 == 0:
            print(f"{i}/{len(df)}, rate: {(i + 1) / (time.time() - time_start):.2f}")

        card_id = row["card_id"]
        review_th = row["review_th"]

        # If this card has been seen before, get the predicted retention from the stored function
        if card_id in pred_ahead_curve:
            pred_ahead[review_th] = srs_rnn.predict_func(
                pred_ahead_curve[card_id], row["elapsed_seconds"]
            )

        # For the immediate predictions we do not send the rating and duration fields
        imm_info = row.copy()
        imm_info.drop(columns=["rating", "duration"], inplace=True)
        pred_imm[review_th] = srs_rnn.imm_predict(imm_info)

        # For ahead-of-time predictions, predict and store a prediction function
        # This function will be evaluated at the time of the next review of this card
        pred_ahead_curve[card_id] = srs_rnn.process_row(row)
        label_rating[review_th] = row["rating"] - 1

        if verbose and review_th in pred_ahead:
            print(
                f"review, {i}, ahead: {pred_ahead[review_th].item():.3f}, immediate: {pred_imm[review_th].item():.3f}, truth: {int(row['rating'] >= 2)}"
            )

    imm_stats, _ = get_stats(
        user_id,
        equalize_review_ths,
        rmse_bins_dict,
        pred_imm,
        label_rating,
    )
    ahead_stats, _ = get_stats(
        user_id,
        equalize_review_ths,
        rmse_bins_dict,
        pred_ahead,
        label_rating,
    )

    print("RWKV-P:")
    print(imm_stats)
    print("RWKV:")
    print(ahead_stats)


if __name__ == "__main__":
    from pathlib import Path

    config = parse_toml()
    run(
        data_path=Path(
            config.DATA_PATH,
        ),
        model_path=config.MODEL_PATH,
        label_db_path=config.LABEL_FILTER_LMDB_PATH,
        label_db_size=config.LABEL_FILTER_LMDB_SIZE,
        user_id=config.USER,
        verbose=config.VERBOSE,
    )
