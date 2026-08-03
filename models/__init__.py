# Import all models for easy access
from .act_r import ACT_R
from .anki import Anki
from .constant import ConstantModel
from .dash import DASH
from .dash_act_r import DASH_ACTR
from .fsrs_rs import FSRSRsBackend
from .fsrs_v1 import FSRS1
from .fsrs_v2 import FSRS2
from .fsrs_v3 import FSRS3
from .fsrs_v4 import FSRS4
from .fsrs_v4dot5 import FSRS4dot5
from .fsrs_v5 import FSRS5
from .fsrs_v6 import FSRS6
from .fsrs_v6_one_step import FSRS_one_step
from .fsrs_v7 import FSRS7
from .gru import GRU
from .hlr import HLR
from .logistic_regression import LogisticRegression
from .lstm import LSTM
from .nn_17 import NN_17
from .rnn import RNN
from .sm2_trainable import SM2

# Import Protocol for type checking
from .trainable import TrainableModel
from .transformer import Transformer

# List of all available models for easy reference
__all__ = [
    "ACT_R",
    "DASH",
    "DASH_ACTR",
    "FSRS1",
    "FSRS2",
    "FSRS3",
    "FSRS4",
    "FSRS5",
    "FSRS6",
    "FSRS7",
    "GRU",
    "HLR",
    "LSTM",
    "NN_17",
    "RNN",
    "SM2",
    "Anki",
    "ConstantModel",
    "FSRS4dot5",
    "FSRSRsBackend",
    "FSRS_one_step",
    "LogisticRegression",
    "TrainableModel",  # Protocol for type checking
    "Transformer",
]
