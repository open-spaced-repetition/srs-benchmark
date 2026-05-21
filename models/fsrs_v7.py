from typing import List, Union
import torch
from torch import nn, Tensor
from typing import Optional
from models.fsrs_v6 import FSRS6, FSRS6ParameterClipper
import torch.nn.functional as F
from torch.nn import Sigmoid

from config import Config

from models.fsrs_v7_interval_penalty import fsrs7_interval_growth_penalty

# scheduling penalty 1 penalizes huge interval growth for non-same-day reviews, makes log loss worse
# scheduling penalty 2 penalizes short (<10 minutes) intervals at 99% DR, makes log loss worse
# L2 penalty penalizes deviation from default parameters, improves log loss very slightly
PENALTY_W_1 = 0.5
PENALTY_W_2 = 0.0015
PENALTY_W_L2 = 0.5


class FSRS7ParameterClipper(FSRS6ParameterClipper):
    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            # Initial S
            w[0] = w[0].clamp(self.config.s_min, self.config.init_s_max / 2)
            w[1] = w[1].clamp(w[0], self.config.init_s_max)
            w[2] = w[2].clamp(w[1], self.config.init_s_max)
            w[3] = w[3].clamp(w[2], self.config.init_s_max)
            # Difficulty
            w[4] = w[4].clamp(1, 10)
            w[5] = w[5].clamp(0.001, 4)
            w[6] = w[6].clamp(0.1, 4)
            # Stability (long-term)
            w[7] = w[7].clamp(0, 4)  # subtract 1.5, see stability_after_review
            w[8] = w[8].clamp(0, 1.2)
            w[9] = w[9].clamp(0.3, 3)
            w[10] = w[10].clamp(0.01, 1.5)
            w[11] = w[11].clamp(0.001, 0.9)
            w[12] = w[12].clamp(0.1, 1)
            w[13] = w[13].clamp(0, 3.5)
            w[14] = w[14].clamp(0, 1)
            w[15] = w[15].clamp(1, 7)
            # Stability (short-term)
            w[16] = w[16].clamp(0, 4)  # subtract 1.5, see stability_after_review
            w[17] = w[17].clamp(0, 2)
            w[18] = w[18].clamp(0.5, 6)
            w[19] = w[19].clamp(0.001, 1.5)
            w[20] = w[20].clamp(0.001, 2)
            w[21] = w[21].clamp(0.001, 1)
            w[22] = w[22].clamp(0, 5)
            w[23] = w[23].clamp(0, 1)
            w[24] = w[24].clamp(1, 7)
            # Long-short term transition function
            w[25] = w[25].clamp(2.5, 15)
            w[26] = w[26].clamp(0, 1)
            # Forgetting curve
            w[27] = w[27].clamp(0.01, 0.25)  # decay 1
            w[28] = w[28].clamp(w[27], 0.95)  # decay 2
            w[29] = w[29].clamp(0.5, 0.85)  # base 1
            w[30] = w[30].clamp(w[29], 0.99)  # base 2
            w[31] = w[31].clamp(0.01, 1)  # weight 1
            w[32] = w[32].clamp(0.1, 1)  # weight 2
            w[33] = w[33].clamp(0, 0.9)  # S weight power 1
            w[34] = w[34].clamp(0.1, 1.1)  # S weight power 2
            module.w.data = w


class FSRS7(FSRS6):
    """
    README entries and the corresponding flags
    Without same-day reviews:
    FSRS-7 = python script.py --algo FSRS-7 --short --secs --equalize_test_with_non_secs --processes 15
    FSRS-7 sched. penalties = python script.py --algo FSRS-7 --sched_penalties --short --secs --equalize_test_with_non_secs --processes 15
    FSRS-7 recency = python script.py --algo FSRS-7 --recency --short --secs --equalize_test_with_non_secs --processes 15
    FSRS-7 default param. = python script.py --algo FSRS-7 --default --short --secs --equalize_test_with_non_secs --processes 15
    FSRS-7 deck = python script.py --algo FSRS-7 --partitions deck --short --secs --equalize_test_with_non_secs --processes 15
    FSRS-7 preset = python script.py --algo FSRS-7 --partitions preset --short --secs --equalize_test_with_non_secs --processes 15
    To include same-day reviews, simply remove --equalize_test_with_non_secs. FSRS-7 is intended to be always be used with --short --secs.
    Other flags that can be used with FSRS-7: --S0, --two_buttons
    """

    n_epoch: int = 8
    batch_size: int = 1024
    lr: float = 2e-2
    betas: tuple = (0.8, 0.85)  # this is for Adam, default is (0.9, 0.999)

    # Obtained via multi-user optimization (1 gradient step per user)
    init_w = [
        0.041,
        2.4175,
        4.1283,
        11.9709,  # Initial S
        5.6385,
        0.4468,
        3.262,  # Difficulty
        2.3054,
        0.1688,
        1.3325,
        0.3524,
        0.0049,
        0.7503,
        0.0896,
        0.6625,
        1.3,  # Stability (long-term)
        0.882,
        0.3072,
        3.5875,
        0.303,
        0.0107,
        0.2279,
        2.6413,
        0.5594,
        1.3,  # Stability (short-term)
        2.5,
        1.0,  # Long-short term transition function
        0.0723,
        0.1634,
        0.5,
        0.9555,
        0.2245,
        0.6232,
        0.1362,
        0.3862,
    ]

    def __init__(self, config: Config, w: Optional[List[float]] = None):
        super().__init__(config)
        if w is None:
            w = self.init_w
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.init_w_tensor = self.w.data.clone().to(self.config.device)
        self.clipper = FSRS7ParameterClipper(config)

    def batch_process(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        # start = time.perf_counter_ns()

        outputs, _ = self.forward(sequences)
        stabilities, difficulties = outputs[
            seq_lens - 1,
            torch.arange(real_batch_size, device=self.config.device),
        ].transpose(0, 1)

        retentions = self.forgetting_curve(
            delta_ts,
            stabilities,
            -self.w[-8],
            -self.w[-7],
            self.w[-6],
            self.w[-5],
            self.w[-4],
            self.w[-3],
            self.w[-2],
            self.w[-1],
        ).clamp(0.0001, 0.9999)

        output = {
            "retentions": retentions,
            "stabilities": stabilities,
            "difficulties": difficulties,
        }

        # start_2 = time.perf_counter_ns()
        if self.config.sched_penalties:
            sched_penalty_1, sched_penalty_2 = fsrs7_interval_growth_penalty(
                self.w,
                n_reviews=10,
                target_dr=0.90,
                n_newton=4,
                target_drs=[0.99],  # for the second penalty
            )
        else:
            sched_penalty_1 = torch.zeros([], device=self.config.device)
            sched_penalty_2 = torch.zeros([], device=self.config.device)
        sigma = torch.tensor(
            [
                9999.0,
                9999.0,
                9999.0,
                9999.0,
                0.523,
                0.2528,
                0.4329,
                0.2966,
                0.2139,
                0.2889,
                0.1862,
                0.0829,
                0.175,
                0.3812,
                0.3013,
                0.9104,
                0.3234,
                0.2448,
                0.3273,
                0.1842,
                0.1542,
                0.1735,
                0.4608,
                0.311,
                0.864,
                0.4053,
                0.162,
                0.0418,
                0.2596,
                0.0798,
                0.0682,
                0.1282,
                0.1397,
                0.1407,
                0.1489,
            ]
        ).to(self.config.device)
        L2_penalty = torch.sum(
            torch.square(self.w - self.init_w_tensor) / torch.square(sigma)
        )
        # sched_penalty_1 penalizes huge interval growth for non-same-day reviews
        # sched_penalty_2 penalizes short (<10 minutes) intervals at 99% DR
        # L2 penalty penalizes deviation from default parameters
        output["penalty"] = (
            PENALTY_W_1 * sched_penalty_1
            + PENALTY_W_2 * sched_penalty_2
            + PENALTY_W_L2 * L2_penalty
        ) * real_batch_size
        # end = time.perf_counter_ns()
        # total = end - start
        # penalty_only = end - start_2
        # print(f'batch_process took {total/1_000_000:.2f} ms, calculating penalty took {penalty_only/1_000_000:.2f} ms')
        return output

    # pyrefly: ignore[bad-override]
    def forgetting_curve(
        self,
        t,
        s,
        decay1=-init_w[-8],
        decay2=-init_w[-7],
        base1=init_w[-6],
        base2=init_w[-5],
        base_weight1=init_w[-4],
        base_weight2=init_w[-3],
        swp1=init_w[-2],
        swp2=init_w[-1],
    ):
        # decays must be passed into forgetting_curve with a minus sign
        t_over_s = t / s

        def power_law_retention(base, decay):
            factor = base ** (1 / decay) - 1
            return (1 + factor * t_over_s) ** decay

        R1 = power_law_retention(base1, decay1)
        R2 = power_law_retention(base2, decay2)

        # S weight power 1 should have a minus sign
        weight1 = base_weight1 * s**-swp1
        weight2 = base_weight2 * s**swp2

        return (weight1 * R1 + weight2 * R2) / (weight1 + weight2)

    def stability_after_review(
        self, state: Tensor, r: Tensor, rating: Tensor
    ) -> tuple[Tensor, Tensor]:
        """
        Calculate both long-term and short-term stability in one pass.
        Returns (new_s_long_term, new_s_short_term)
        """
        w = self.w
        batch_size = state.shape[0]

        old_s = state[:, 0]
        old_d = state[:, 1]
        success = rating > 1

        # Stack weights for both long-term (base=7) and short-term (base=16)
        # Shape: [2] for each parameter
        w_base = torch.tensor([7, 16], device=w.device)

        w_sinc_base = w[w_base]  # w[7], w[16]
        w_sinc_s_exp = w[w_base + 1]  # w[8], w[17]
        w_sinc_r_mult = w[w_base + 2]  # w[9], w[18]
        w_fail_mult = w[w_base + 3]  # w[10], w[19]
        w_fail_d_exp = w[w_base + 4]  # w[11], w[20]
        w_fail_s_exp = w[w_base + 5]  # w[12], w[21]
        w_fail_r_mult = w[w_base + 6]  # w[13], w[22]
        w_hard = w[w_base + 7]  # w[14], w[23]
        w_easy = w[w_base + 8]  # w[15], w[24]

        # Expand state to [batch_size, 2] for broadcasting with [2] weight vectors
        # Result shapes: [batch_size, 2] where dim 1 is (long-term, short-term)

        hard_penalty = torch.where(
            rating.unsqueeze(1) == 2,
            w_hard.unsqueeze(0),
            torch.ones(batch_size, 2, device=w.device),
        )
        easy_bonus = torch.where(
            rating.unsqueeze(1) == 4,
            w_easy.unsqueeze(0),
            torch.ones(batch_size, 2, device=w.device),
        )

        # Stability after failure: [batch_size, 2]
        new_s_fail = (
            w_fail_mult
            * torch.pow(old_d.unsqueeze(1), -w_fail_d_exp)
            * (torch.pow(old_s.unsqueeze(1) + 1, w_fail_s_exp) - 1)
            * torch.exp((1 - r).unsqueeze(1) * w_fail_r_mult)
        )
        pls = torch.minimum(old_s.unsqueeze(1), new_s_fail)

        # Stability increase after success: [batch_size, 2]
        SInc = (
            1
            + torch.exp(w_sinc_base - 1.5)
            * (11 - old_d).unsqueeze(1)
            * torch.pow(old_s.unsqueeze(1), -w_sinc_s_exp)
            * (torch.exp((1 - r).unsqueeze(1) * w_sinc_r_mult) - 1)
            * hard_penalty
            * easy_bonus
        )
        new_s_success = torch.maximum(pls, old_s.unsqueeze(1) * SInc)

        # Select success or failure based on rating: [batch_size, 2]
        new_s_both = torch.where(success.unsqueeze(1), new_s_success, pls)

        return new_s_both[:, 0], new_s_both[:, 1]

    def transition_function(self, delta_t: Tensor) -> Tensor:
        return 1 - self.w[26] * torch.exp(-self.w[25] * delta_t)

    def init_d(self, rating: Union[int, Tensor]) -> Tensor:
        new_d = self.w[4] - torch.exp(self.w[5] * (rating - 1)) + 1
        return new_d

    def linear_damping(self, delta_d: Tensor, old_d: Tensor) -> Tensor:
        return delta_d * (10 - old_d) / 9

    def mean_reversion(self, init: Tensor, current: Tensor) -> Tensor:
        return 0.01 * init + 0.99 * current

    def next_d(self, state: Tensor, rating: Tensor) -> Tensor:
        delta_d = -self.w[6] * (rating - 3)
        new_d = state[:, 1] + self.linear_damping(delta_d, state[:, 1])
        new_d = self.mean_reversion(self.init_d(4), new_d)
        return new_d

    def step(self, X: Tensor, state: Tensor) -> Tensor:
        """
        :param X: shape[batch_size, 2], X[:,0] is elapsed time, X[:,1] is rating
        :param state: shape[batch_size, 2], state[:,0] is stability, state[:,1] is difficulty
        :return state:
        """
        if torch.equal(state, torch.zeros_like(state)):
            keys = torch.tensor([1, 2, 3, 4], device=X.device)
            keys = keys.view(1, -1).expand(X[:, 1].long().size(0), -1)
            index = (X[:, 1].long().unsqueeze(1) == keys).nonzero(as_tuple=True)
            # first learn, init memory states
            new_s = torch.ones_like(state[:, 0], device=X.device)
            w = self.w.to(X.device)
            new_s[index[0]] = w[index[1]]
            new_d = self.init_d(X[:, 1])
            new_d = new_d.clamp(1, 10)
        else:
            r = self.forgetting_curve(
                X[:, 0],
                state[:, 0],
                -self.w[-8],
                -self.w[-7],
                self.w[-6],
                self.w[-5],
                self.w[-4],
                self.w[-3],
                self.w[-2],
                self.w[-1],
            )

            new_s_long_term, new_s_short_term = self.stability_after_review(
                state, r, X[:, 1]
            )

            # A number between 0 and 1 that represents how much of a non-same-day review this is
            # 1 = long-term
            # 0 = short-term (same-day)
            coefficient = self.transition_function(X[:, 0])
            new_s = coefficient * new_s_long_term + (1 - coefficient) * new_s_short_term

            new_d = self.next_d(state, X[:, 1])
            new_d = new_d.clamp(1, 10)

        new_s = new_s.clamp(self.config.s_min, 36500)
        return torch.stack([new_s, new_d], dim=1)
