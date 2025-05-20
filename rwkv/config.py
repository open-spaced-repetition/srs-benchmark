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
