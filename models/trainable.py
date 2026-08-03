from __future__ import annotations

from collections.abc import Callable, Hashable, Iterator, Mapping
from typing import Any, Protocol, Self, Union

import pandas as pd
import torch
from shape_extensions import IntVar
from torch import Tensor

from config import Config

type ParameterList = list[float]
type TorchStateDict = Mapping[str, Any]
type ModelState = ParameterList | TorchStateDict
type PartitionedModelState = dict[Hashable, ModelState]
type TrainingState = ModelState | PartitionedModelState


class TrainableModel[InputDims: IntVar, OutputDims: IntVar](Protocol):
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

    def batch_process[
        SeqLen: IntVar,
        BatchSize: IntVar,
    ](
        self,
        sequences: Tensor[[SeqLen, BatchSize, InputDims]],
        delta_ts: Tensor[[BatchSize]],
        seq_lens: Tensor[[BatchSize]],
        real_batch_size: int,
    ) -> Mapping[
        str,
        Tensor[[]] | Tensor[[BatchSize]] | Tensor[[BatchSize, OutputDims]],
    ]:
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

    forward: Callable[..., Any]

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ) -> Any:
        """Load model state dictionary."""
        ...

    def benchmark_state(self) -> ModelState:
        """Return either serializable parameters or a torch state dict."""
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
