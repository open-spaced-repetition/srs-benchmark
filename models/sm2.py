import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional # Added Optional

# TODO: Refactor to pass configuration instead of relying on globals
S_MIN = 0.01 # Placeholder
S_MAX = 36500 # Placeholder
# DEVICE is not directly used in SM2's math, but state initialization in a batch context might need it.

class SM2ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(S_MIN, S_MAX) # TODO: S_MIN, S_MAX globals. Initial interval for rep 1
            w[1] = w[1].clamp(S_MIN, S_MAX) # TODO: S_MIN, S_MAX globals. Initial interval for rep 2 (ef=6 here)
            w[2] = w[2].clamp(1.3, 10.0)   # Initial ease factor
            w[3] = w[3].clamp(0, None)     # EF modification: (5-q)*w[3] term related
            w[4] = w[4].clamp(5, None)     # EF modification: (q-w[4]) term related
            w[5] = w[5].clamp(0, None)     # EF modification: w[5] is the constant added
            module.w.data = w

class SM2(nn.Module):
    # 6 params
    init_w = [1, 6, 2.5, 0.02, 7, 0.18] # Defaults from other.py
    clipper = SM2ParameterClipper()
    lr: float = 4e-2 # Not used by SM2 itself if not training parameters
    wd: float = 1e-5  # Not used
    n_epoch: int = 5  # Not used

    def __init__(self, w: List[float] = init_w):
        super(SM2, self).__init__()
        # If SM2 parameters are to be trained, they should be nn.Parameter
        # If they are fixed as per algorithm, they can just be attributes or part of calculation
        # The original code has them as nn.Parameter, implying they might be trained by FSRS Optimizer.
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        """
        :param inputs: shape[seq_len, batch_size, 2], X[:,0] is elapsed time (not used by SM2 step), X[:,1] is rating
        :param state: shape[batch_size, 3], state[:,0] is interval, state[:,1] is ease_factor, state[:,2] is repetitions
        """
        if state is None: # Initialize state
            state = torch.zeros((inputs.shape[1], 3), device=inputs.device)
            state[:, 1] = self.w[2] # Initialize ease factor
            # Reps and interval (ivl) start at 0, handled by step logic for first review

        outputs = []
        for X_step in inputs: # Iterate over sequence length
            state = self.step(X_step, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def iter(
        self,
        sequences: Tensor, # [seq_len, batch_size, 2 (delta_t, rating)]
        delta_ts: Tensor,  # [batch_size] - delta_t for the *current* event
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs_history, _ = self.forward(sequences) # outputs_history is [seq_len, batch_size, 3 (ivl, ef, reps)]

        # Get the state *after* the last actual review in the sequence
        last_states = outputs_history[seq_lens - 1, torch.arange(real_batch_size)]
        stabilities = last_states[:, 0] # Interval is used as stability

        retentions = self.forgetting_curve(delta_ts, stabilities)
        return {"retentions": retentions, "stabilities": stabilities}

    def step(self, X_step: Tensor, state: Tensor) -> Tensor:
        """
        :param X_step: shape[batch_size, 2], X_step[:,0] is elapsed time (not directly used by SM2 logic for interval calc), X_step[:,1] is rating
        :param state: shape[batch_size, 3], state[:,0] is interval (ivl), state[:,1] is ease_factor (ef), state[:,2] is repetitions (reps)
        :return next_state: shape[batch_size, 3], [new_ivl, new_ef, new_reps]
        """
        rating = X_step[:, 1] # Current rating
        ivl = state[:, 0]
        ef = state[:, 1]
        reps = state[:, 2]

        success = rating > 1 # Original SM-2 uses rating < 3 (0-5 scale) as fail. Here rating is 1-4. So rating > 1 is pass.
                              # Let's assume rating 1=fail (q < 3), rating 2,3,4=pass (q >= 3)
                              # q_sm2_equivalent = rating + 1 approx (mapping 1->2, 2->3, 3->4, 4->5)
                              # For FSRS data, rating 1 is fail, 2,3,4 are pass.

        # For SM2 logic, q is usually on a 0-5 scale.
        # If rating is 1 (fail), q_eff = 1+1 = 2.
        # If rating is 2 (hard), q_eff = 2+1 = 3.
        # If rating is 3 (good), q_eff = 3+1 = 4.
        # If rating is 4 (easy), q_eff = 4+1 = 5.
        q_eff = rating + 1

        new_reps = torch.where(success, reps + 1, torch.zeros_like(reps)) # Reset reps on fail

        # Calculate new ease factor
        new_ef = ef - self.w[3] * (q_eff - self.w[4]) ** 2 + self.w[5]
        new_ef = new_ef.clamp(1.3, 10.0) # SM2 lower bound for EF is 1.3

        # Calculate new interval
        # If first rep (after fail or actual first):
        ivl_if_first_rep = self.w[0]
        # If second rep:
        ivl_if_second_rep = self.w[1]
        # If > second rep:
        ivl_if_gt_second_rep = ivl * new_ef # Use new_ef for current interval calculation if successful

        new_ivl = torch.where(
            new_reps == 0, # This means current was a fail, so next interval is based on rep 1 logic
            ivl_if_first_rep,
            torch.where(
                new_reps == 1,
                ivl_if_first_rep, # Interval for first successful repetition
                torch.where(
                    new_reps == 2,
                    ivl_if_second_rep, # Interval for second successful repetition
                    ivl_if_gt_second_rep # Interval for subsequent successful repetitions
                )
            )
        )
        # If rating was a fail (success=False), interval resets based on new_reps=0 (handled by new_reps logic for next pass).
        # More directly, if fail, the interval for the *next* time is typically short.
        # The SM2 algorithm resets interval to a starting value on fail.
        # The current `new_ivl` logic correctly sets interval for *next* successful review.
        # If current was a fail, new_reps becomes 0. The *next* interval will then be w[0].
        # This seems correct.

        new_ivl = new_ivl.clamp(S_MIN, S_MAX) # TODO: S_MIN, S_MAX globals

        return torch.stack([new_ivl, new_ef, new_reps], dim=1)

    def forgetting_curve(self, t: Tensor, s: Tensor) -> Tensor:
        # SM2 doesn't use a continuous forgetting curve; recall is 100% until interval `s`, then 0%.
        # For compatibility with FSRS trainer, often approximated by R = 0.9^(t/s) or similar.
        # Using the FSRS4.5 curve as a common approximation if needed by trainer.
        # However, the original `other.py` for SM2 in `process_untrainable` uses:
        # testset["p"] = np.exp(np.log(0.9) * testset["delta_t"] / testset["stability"])
        # which is 0.9**(t/s).
        return 0.9 ** (t / s)

    def state_dict(self) -> List[float]: # Override to match original format
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )
