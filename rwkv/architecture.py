"""
Defines the architecture configurations for various RWKV models,
specifically tailored for Anki flashcard review modeling.

This module specifies the parameters for different components of the RWKV model,
such as card_id, deck_id, etc., using the RWKV7Config.
"""
from dataclasses import dataclass
from rwkv.model.rwkv_model import RWKV7Config

# Number of attention heads for the RWKV models.
N_HEADS = 4
# Default dropout rate.
DROPOUT = 0.02
# Dropout rate for longer sequences or specific layers.
DROPOUT_LONG = 0.05
# Dropout rate applied within layers.
DROPOUT_LAYER = 0.01


@dataclass
class AnkiRWKVConfig:
    """
    Configuration for the Anki RWKV model.

    Attributes:
        d_model: The dimensionality of the model.
        modules: A list of module configurations (name, RWKV7Config).
        dropout: The overall dropout rate for the model.
    """
    d_model: int
    modules: list
    dropout: float


# Defines the specific configurations for different modules within the Anki RWKV model.
# Each tuple contains:
# 1. Module name (e.g., "card_id", "deck_id").
# 2. RWKV7Config object with specific parameters for that module.
_layers = [
    (
        "card_id",

DROPOUT = 0.02
DROPOUT_LONG = 0.05
DROPOUT_LAYER = 0.01


@dataclass
class AnkiRWKVConfig:
    d_model: int
    modules: list
    dropout: float


_layers = [
    (
        "card_id",
        RWKV7Config(
            d_model=32 * N_HEADS,
            n_heads=N_HEADS,
            n_layers=3,
            layer_offset=0,
            total_layers=3,
            channel_mixer_factor=1.5,
            decay_lora=16,
            a_lora=16,
            v0_mix_amt_lora=8,
            gate_lora=16,
            dropout=DROPOUT,
            dropout_layer=DROPOUT_LAYER,
        ),
    ),
    (
        "deck_id",
        RWKV7Config(
            d_model=32 * N_HEADS,
            n_heads=N_HEADS,
            n_layers=4,
            layer_offset=0,
            total_layers=4,
            channel_mixer_factor=2.0,
            decay_lora=16,
            a_lora=16,
            v0_mix_amt_lora=8,
            gate_lora=16,
            dropout=DROPOUT_LONG,
            dropout_layer=DROPOUT_LAYER,
        ),
    ),
    (
        "note_id",
        RWKV7Config(
            d_model=32 * N_HEADS,
            n_heads=N_HEADS,
            n_layers=2,
            layer_offset=0,
            total_layers=2,
            channel_mixer_factor=1.5,
            decay_lora=16,
            a_lora=16,
            v0_mix_amt_lora=8,
            gate_lora=16,
            dropout=DROPOUT,
            dropout_layer=DROPOUT_LAYER,
        ),
    ),
    (
        "preset_id",
        RWKV7Config(
            d_model=32 * N_HEADS,
            n_heads=N_HEADS,
            n_layers=3,
            layer_offset=0,
            total_layers=3,
            channel_mixer_factor=2.0,
            decay_lora=16,
            a_lora=16,
            v0_mix_amt_lora=8,
            gate_lora=16,
            dropout=DROPOUT_LONG,
            dropout_layer=DROPOUT_LAYER,
        ),
    ),
    (
        "user_id",
        RWKV7Config(
            d_model=32 * N_HEADS,
            n_heads=N_HEADS,
            n_layers=4,
            layer_offset=0,
            total_layers=4,
            channel_mixer_factor=2.0,
            decay_lora=16,
            a_lora=16,
            v0_mix_amt_lora=8,
            gate_lora=16,
            dropout=DROPOUT_LONG,
            dropout_layer=DROPOUT_LAYER,
        ),
    ),
]

DEFAULT_ANKI_RWKV_CONFIG = AnkiRWKVConfig(
    d_model=32 * N_HEADS, modules=_layers, dropout=DROPOUT
)
