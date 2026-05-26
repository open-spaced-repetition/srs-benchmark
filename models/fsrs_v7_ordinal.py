from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from config import Config
from models.fsrs_v7 import FSRS7

EPS = 1e-7
ORDINAL_PARAMETER_COUNT = 6


class FSRS7Ordinal(FSRS7):
    """
    FSRS-7 trained with an auxiliary ordinal button likelihood.

    Benchmark prediction remains the binary recall probability R. The ordinal
    probabilities are only used as an additional training signal.
    """

    uses_current_rating_targets = True
    ordinal_loss_weight = 1.0
    init_a = [0.0, 1.0, 0.0, 0.0, 0.5, 1.0]

    def __init__(self, config: Config, w: Optional[List[float]] = None):
        fsrs_w, ordinal_a = self._split_parameters(w)
        super().__init__(config, fsrs_w)
        self.a = nn.Parameter(torch.tensor(ordinal_a, dtype=torch.float32))

    @classmethod
    def _split_parameters(
        cls, parameters: Optional[List[float]]
    ) -> tuple[Optional[List[float]], List[float]]:
        if parameters is None:
            return None, cls.init_a

        fsrs_parameter_count = len(FSRS7.init_w)
        if len(parameters) == fsrs_parameter_count:
            return parameters, cls.init_a
        if len(parameters) == fsrs_parameter_count + ORDINAL_PARAMETER_COUNT:
            return parameters[:fsrs_parameter_count], parameters[fsrs_parameter_count:]

        raise ValueError(
            "FSRS-7-ordinal parameters must contain either "
            f"{fsrs_parameter_count} FSRS parameters or "
            f"{fsrs_parameter_count + ORDINAL_PARAMETER_COUNT} combined parameters."
        )

    def batch_process(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        output = super().batch_process(sequences, delta_ts, seq_lens, real_batch_size)
        output["button_probabilities"] = self.ordinal_button_probabilities(
            output["retentions"],
            output["stabilities"],
            output["difficulties"],
        )
        return output

    def ordinal_button_probabilities(
        self, retentions: Tensor, stabilities: Tensor, difficulties: Tensor
    ) -> Tensor:
        h, e = self.ordinal_gates(retentions, stabilities, difficulties)
        r = retentions.clamp(EPS, 1.0 - EPS)
        return torch.stack(
            [
                1.0 - r,
                r * (1.0 - h),
                r * (h - e),
                r * e,
            ],
            dim=1,
        )

    def ordinal_gates(
        self, retentions: Tensor, stabilities: Tensor, difficulties: Tensor
    ) -> tuple[Tensor, Tensor]:
        r = retentions.clamp(EPS, 1.0 - EPS)
        s = stabilities.clamp_min(EPS)
        a0, a1, a2, a3, a4, a5 = self.a
        raw = a0 + a1 * torch.logit(r) + a2 * torch.log(s) + a3 * difficulties
        hard_good = torch.sigmoid(raw - a4)
        good_easy_threshold = a4 + F.softplus(a5)
        easy = torch.sigmoid(raw - good_easy_threshold)
        return hard_good, easy

    def compute_extra_training_loss(self, result: dict[str, Tensor]) -> Tensor:
        ordinal_loss = self.ordinal_nll(
            result["button_probabilities"],
            result["ratings"],
            result["weights"],
        )
        return self.ordinal_loss_weight * ordinal_loss

    @staticmethod
    def ordinal_nll(
        button_probabilities: Tensor, ratings: Tensor, weights: Tensor
    ) -> Tensor:
        button_index = ratings.long().sub(1).clamp(0, 3)
        observed_probabilities = button_probabilities.gather(
            1, button_index.unsqueeze(1)
        ).squeeze(1)
        return (-observed_probabilities.clamp_min(EPS).log() * weights).sum()

    def benchmark_state(self):
        precision = 6 if self.config.use_secs_intervals else 4
        fsrs_w = [round(float(x), precision) for x in self.w.data]
        ordinal_a = [round(float(x), precision) for x in self.a.data]
        return fsrs_w + ordinal_a
