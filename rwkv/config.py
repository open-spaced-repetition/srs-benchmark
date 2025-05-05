from pathlib import Path
import torch

MODEL_PATH = "pretrain/RWKV.pth"

DATA_PATH = Path("../anki-revlogs-10k")
TRAIN_DATASET_LMDB_PATH = "training_db"
TRAIN_DATASET_LMDB_SIZE = int(205 * 1e9)
ALL_DATASET_LMDB_PATH = "all_db"
ALL_DATASET_LMDB_SIZE = int(195 * 1e9)
# database for non-truncated decks, for inference

LABEL_FILTER_LMDB_PATH = "label_filter_db"

LABEL_FILTER_LMDB_SIZE = int(7 * 1e9)
DTYPE = torch.bfloat16
DEVICE = torch.device("cuda")

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


P_MOD = 1  # Tradeoff between context length and the number of samples used for probability prediction