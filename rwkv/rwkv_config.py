from dataclasses import dataclass
from rwkv.model.rwkv_model import RWKV7Config

RWKV_SUBMODULES = ["card_id", "note_id", "deck_id", "preset_id", "user_id"]

# CARD_ID_DIM and NOTE_ID_DIM can combine their values
ID_ENCODE_DIMS = {
    "card_id": 12,
    "note_id": 12,
    "deck_id": 8,
    "preset_id": 8,
}
ID_SPLIT = 4
DAY_OFFSET_ENCODE_PERIODS = [3, 7, 30, 100, 365, 3650, 36500]

MAX_TRAIN_GLOBAL_LEN = 66000 * 1

N_HEADS = 4

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
