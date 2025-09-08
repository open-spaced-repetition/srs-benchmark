# Import all models for easy access
from .fsrs_v1 import FSRS1
from .fsrs_v2 import FSRS2
from .fsrs_v3 import FSRS3
from .fsrs_v4 import FSRS4
from .fsrs_v4dot5 import FSRS4dot5
from .fsrs_v5 import FSRS5
from .fsrs_v6 import FSRS6
from .fsrs_v6_one_step import FSRS_one_step
from .rnn import RNN
from .gru_p import GRU_P
from .lstm import LSTM
from .transformer import Transformer
from .hlr import HLR
from .act_r import ACT_R
from .dash import DASH
from .dash_act_r import DASH_ACTR
from .nn_17 import NN_17
from .sm2_trainable import SM2
from .anki import Anki
from .constant import ConstantModel

# Import Protocol for type checking
from .trainable import TrainableModel

# List of all available models for easy reference
__all__ = [
    "FSRS1",
    "FSRS2",
    "FSRS3",
    "FSRS4",
    "FSRS4dot5",
    "FSRS5",
    "FSRS6",
    "FSRS_one_step",
    "RNN",
    "GRU_P",
    "LSTM",
    "Transformer",
    "HLR",
    "ACT_R",
    "DASH",
    "DASH_ACTR",
    "NN_17",
    "SM2",
    "Anki",
    "ConstantModel",
    "TrainableModel",  # Protocol for type checking
]
