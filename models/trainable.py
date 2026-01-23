from typing import Iterator, Protocol, Union
from typing_extensions import Self
import torch
from torch import Tensor
import pandas as pd
from config import Config


class TrainableModel(Protocol):
    """
    Protocol for trainable models that depend on nn.Module.

    This Protocol defines the interface that all neural network-based trainable models must implement.
    Models including LSTM, RNN, Transformer, NN_17, GRU_P, etc. should all follow this protocol.
    """

    # Class attributes that should be available
    lr: float
    wd: float
    n_epoch: int
    config: Config

    def get_optimizer(
        self, lr: float, wd: float, betas: tuple = (0.9, 0.999)
    ) -> torch.optim.Optimizer:
        """
        Return an optimizer for training the model.

        Args:
            lr: Learning rate
            wd: Weight decay
            betas: beta1 and beta2 parameters for Adam

        Returns:
            torch.optim.Optimizer: Configured optimizer
        """
        ...

    def initialize_parameters(self, train_set: pd.DataFrame) -> None:
        """
        Initialize the model parameters on the given training dataset.

        Args:
            train_set: Training dataset as pandas DataFrame
        """
        ...

    def batch_process(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        """
        Core batch processing method for model inference.

        Args:
            sequences: Input sequences tensor
            delta_ts: Delta time tensor
            seq_lens: Sequence lengths tensor
            real_batch_size: Actual batch size

        Returns:
            dict[str, Tensor]: Dictionary containing model outputs
        """
        ...

    def filter_training_data(self, train_set: pd.DataFrame) -> pd.DataFrame:
        """
        Filter and preprocess training data.

        Args:
            train_set: Raw training dataset

        Returns:
            pd.DataFrame: Filtered training dataset
        """
        ...

    def set_hyperparameters(self, lr: float, wd: float, n_epoch: int) -> None:
        """
        Set training hyperparameters.

        Args:
            lr: Learning rate
            wd: Weight decay
            n_epoch: Number of training epochs
        """
        ...

    def apply_gradient_constraints(self) -> None:
        """Apply gradient constraints during training."""
        ...

    def apply_parameter_clipper(self) -> None:
        """Apply parameter clipping to maintain valid parameter ranges."""
        ...

    # Methods inherited from nn.Module that trainable models should have
    def parameters(self) -> Iterator[torch.nn.parameter.Parameter]:
        """Return model parameters for optimization."""
        ...

    def forward(self, *args, **kwargs) -> Union[tuple[Tensor, ...], Tensor]:
        """Forward pass of the neural network."""
        ...

    def load_state_dict(self, state_dict: dict) -> None:
        """Load model state dictionary."""
        ...

    def state_dict(self) -> dict:
        """Return model state dictionary."""
        ...

    def train(self, mode: bool = True) -> Self:
        """Set model to training mode."""
        ...

    def eval(self) -> Self:
        """Set model to evaluation mode."""
        ...

    def to(self, device: torch.device) -> Self:
        """Move model to specified device."""
        ...
