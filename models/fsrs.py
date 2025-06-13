import torch
from torch import Tensor
from config import Config
from models.base import BaseModel

class FSRS(BaseModel):
    def __init__(self, config: Config):
        super().__init__(config)

    def forgetting_curve(self, t, s):
        raise NotImplementedError("Forgetting curve not implemented")

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
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

    def state_dict(self):
        """Override to use precision based on config.use_secs_intervals"""
        precision = 6 if self.config.use_secs_intervals else 4
        return list(
            map(
                lambda x: round(float(x), precision),
                dict(self.named_parameters())["w"].data,
            )
        )
