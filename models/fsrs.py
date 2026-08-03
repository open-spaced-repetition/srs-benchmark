import torch
from shape_extensions import IntVar
from torch import Tensor

from config import Config
from models.base import BaseModel, BaseParameterClipper


class FSRSParameterClipper(BaseParameterClipper):
    def __init__(self):
        super().__init__()

    def __call__(self, module):
        pass


class FSRS(BaseModel):
    clipper = FSRSParameterClipper()

    def __init__(self, config: Config):
        super().__init__(config)

    def forgetting_curve(self, t, s):
        raise NotImplementedError("Forgetting curve not implemented")

    def forward[SeqLen: IntVar, BatchSize: IntVar, StateSize: IntVar](
        self, inputs: Tensor[[SeqLen, BatchSize, 2]]
    ) -> tuple[Tensor[[SeqLen, BatchSize, StateSize]], Tensor[[BatchSize, StateSize]]]:
        raise NotImplementedError("Forward pass not implemented")

    def batch_process[SeqLen: IntVar, BatchSize: IntVar](
        self,
        sequences: Tensor[[SeqLen, BatchSize, 2]],
        delta_ts: Tensor[[BatchSize]],
        seq_lens: Tensor[[BatchSize]],
        real_batch_size: int,
    ) -> dict[str, Tensor[[BatchSize]] | Tensor[[]]]:
        outputs, _ = self.forward(sequences)
        stabilities, difficulties, *_ = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=self.config.device),
        ].transpose(0, 1)
        retentions = self.forgetting_curve(delta_ts, stabilities)
        return {
            "retentions": retentions,
            "stabilities": stabilities,
            "difficulties": difficulties,
        }

    def benchmark_state(self):
        """Override to use precision based on config.use_secs_intervals"""
        precision = 6 if self.config.use_secs_intervals else 4
        return [
            round(float(x), precision) for x in dict(self.named_parameters())["w"].data
        ]
