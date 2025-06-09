"""
Global configuration settings for the RWKV model training and data processing.

This module defines constants that are used across various parts of the project,
including submodule definitions, encoding dimensions, and data processing parameters.
"""

# Defines the names of the submodules used in the RWKV architecture.
# These typically correspond to different types of IDs or features that get their own embeddings or processing paths.
RWKV_SUBMODULES = ["card_id", "note_id", "deck_id", "preset_id", "user_id"]

# Specifies the dimensionality of the random encoding vectors for different ID types.
# These encodings are added to the card features during batch preparation.
# Note: "user_id" is implicitly handled or doesn't get a separate random encoding here.
ID_ENCODE_DIMS = {
    "card_id": 12,  # Dimensionality for card_id encoding
    "note_id": 12,  # Dimensionality for note_id encoding
    "deck_id": 8,   # Dimensionality for deck_id encoding
    "preset_id": 8, # Dimensionality for preset_id encoding
}

# Defines the range [0, ID_SPLIT-1] from which random integers are drawn to create ID encodings.
# The encoding is then centered by subtracting (ID_SPLIT - 1) / 2.
ID_SPLIT = 4

# Periods (in days) used for generating sinusoidal positional encodings for day offsets.
# These help the model understand cyclical patterns in time.
DAY_OFFSET_ENCODE_PERIODS = [3, 7, 30, 100, 365, 3650, 36500]

# Maximum combined length of all sequences in a single training batch.
# This is used by `get_groups` in `train_rwkv.py` to batch together multiple
# user data segments (RWKVSamples) effectively.
MAX_TRAIN_GLOBAL_LEN = 66000 * 1
