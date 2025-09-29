import torch
from torch import nn, Tensor
import pandas as pd
from config import Config


class BaseParameterClipper:
    def __init__(self):
        pass

    def __call__(self, module):
        pass


class BaseModel(nn.Module):
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5
    clipper: BaseParameterClipper = BaseParameterClipper()

    def __init__(self, config: Config):
        super().__init__()
        self.config = config

    def get_optimizer(self, lr: float, wd: float) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=lr)

    def initialize_parameters(self, train_set: pd.DataFrame) -> None:
        pass

    def filter_training_data(self, train_set: pd.DataFrame) -> pd.DataFrame:
        return train_set

    def set_hyperparameters(self, lr: float, wd: float, n_epoch: int) -> None:
        self.lr = lr
        self.wd = wd
        self.n_epoch = n_epoch

    def apply_gradient_constraints(self):
        pass

    def apply_parameter_clipper(self):
        self.apply(self.clipper)
