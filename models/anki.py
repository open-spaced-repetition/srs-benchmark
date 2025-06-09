import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional

# TODO: Refactor to pass configuration instead of relying on globals
S_MIN = 0.01 # Placeholder
S_MAX = 36500 # Placeholder
# DEVICE is not directly used in Anki's math, but state initialization in a batch context might need it.

class AnkiParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            # based on limits in Anki 24.11 (these are mostly for UI, true internal limits might differ)
            # For parameters that are intervals:
            w[0] = w[0].clamp(1, 9999)    # graduating interval
            w[1] = w[1].clamp(1, 9999)    # easy interval
            # For ease factor:
            w[2] = w[2].clamp(1.30, 5.0)  # starting ease (Anki default is 2.50 or 250%)
            # For modifiers:
            w[3] = w[3].clamp(1.0, 5.0)   # easy bonus (default 1.30 or 130%)
            w[4] = w[4].clamp(0.5, 1.3)   # hard interval multiplier (default 1.20)
            w[5] = w[5].clamp(0.0, 1.0)   # new interval multiplier (after lapse, default 0.0)
            w[6] = w[6].clamp(0.5, 2.0)   # interval multiplier (master multiplier, default 1.0 or 100%)
            module.w.data = w

class Anki(nn.Module):
    # 7 params, based on typical Anki settings that could be trained
    init_w = [
        1.0,  # graduating interval (days)
        4.0,  # easy interval (days)
        2.5,  # starting ease (250%)
        1.3,  # easy bonus
        1.2,  # hard interval multiplier
        0.0,  # new interval (multiplier for current interval after lapse)
        1.0,  # interval multiplier (overall)
    ]
    clipper = AnkiParameterClipper()
    lr: float = 4e-2 # Not used if parameters are not trained
    wd: float = 1e-5  # Not used
    n_epoch: int = 5  # Not used

    def __init__(self, w: List[float] = init_w):
        super(Anki, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def forgetting_curve(self, t: Tensor, s: Tensor) -> Tensor:
        # Anki's SM-2 doesn't use a continuous forgetting curve for prediction in the same way FSRS does.
        # It schedules discrete intervals. For compatibility with evaluation:
        return 0.9 ** (t / s) # Approximation

    def _passing_early_review_intervals(self, rating: Tensor, ease: Tensor, ivl: Tensor, days_late: Tensor) -> Tensor:
        # rating: 2 (hard), 3 (good), 4 (easy)
        # days_late is negative for early reviews
        elapsed = ivl + days_late # actual time spent

        ivl_hard = torch.max(elapsed * self.w[4], ivl * self.w[4] / 2)
        # Good: Anki's actual logic for early reviews is complex.
        # It might use (ivl * ease) or (elapsed * ease) depending on how early.
        # A common simplification is to use the scheduled interval's logic.
        # For "Good", Anki often uses (ivl + days_late/2) * ease for >20% early,
        # or (ivl + days_late/4) * ease for >5 days early.
        # Here, using a simpler version: (ivl * ease), then adjusted by easy_bonus for "Easy"
        ivl_good = ivl * ease
        ivl_easy = ivl_good * self.w[3] # easy_bonus applied to good_interval

        return torch.where(
            rating == 2,
            ivl_hard,
            torch.where(
                rating == 4,
                ivl_easy,
                ivl_good,
            ),
        )

    def _passing_nonearly_review_intervals(self, rating: Tensor, ease: Tensor, ivl: Tensor, days_late: Tensor) -> Tensor:
        # rating: 2 (hard), 3 (good), 4 (easy)
        # days_late is non-negative
        ivl_hard = ivl * self.w[4]
        ivl_good = (ivl + days_late / 2) * ease # Good: Average of current ivl and elapsed time
        ivl_easy = (ivl + days_late / 4) * ease * self.w[3] # Easy: Less penalty for lateness, plus easy_bonus

        return torch.where(
            rating == 2,
            ivl_hard,
            torch.where(
                rating == 4,
                ivl_easy,
                ivl_good,
            ),
        )

    def step(self, X_step: Tensor, state: Tensor) -> Tensor:
        """
        :param X_step: shape[batch_size, 2], X_step[:,0] is delta_t (elapsed time in days), X_step[:,1] is rating (1,2,3,4)
        :param state: shape[batch_size, 2], state[:,0] is interval (ivl), state[:,1] is ease_factor (ef)
        :return next_state:
        """
        delta_t = X_step[:, 0]
        rating = X_step[:, 1]

        ivl = state[:, 0]
        ease = state[:, 1]

        # Default to current state, overwrite if first review
        new_ivl = ivl.clone()
        new_ease = ease.clone()

        # First review logic (state is zeros or some indicator)
        # Anki initializes ease to starting ease (w[2]), interval depends on rating.
        is_first_review = torch.logical_or(ivl == 0, ease == 0) # A common way to check initial state

        # Update ease factor
        new_ease = torch.where(rating == 1, ease - 0.20, new_ease)
        new_ease = torch.where(rating == 2, ease - 0.15, new_ease)
        new_ease = torch.where(rating == 4, ease + 0.15, new_ease)
        new_ease = new_ease.clamp(1.3, 5.5) # Anki's ease factor bounds (typically 130% to 550%)

        # Calculate new interval
        days_late = delta_t - ivl

        # Passing grades (2,3,4)
        pass_condition = rating > 1

        early_review_condition = pass_condition & (days_late < 0)
        non_early_review_condition = pass_condition & (days_late >= 0)

        ivl_if_pass_early = self._passing_early_review_intervals(rating, new_ease, ivl, days_late)
        ivl_if_pass_non_early = self._passing_nonearly_review_intervals(rating, new_ease, ivl, days_late)

        current_calculated_ivl = torch.zeros_like(ivl)
        current_calculated_ivl = torch.where(early_review_condition, ivl_if_pass_early, current_calculated_ivl)
        current_calculated_ivl = torch.where(non_early_review_condition, ivl_if_pass_non_early, current_calculated_ivl)

        # Apply master interval multiplier
        current_calculated_ivl = current_calculated_ivl * self.w[6]

        # Lapsed cards (rating == 1)
        ivl_if_fail = ivl * self.w[5] # new_interval multiplier

        # Set intervals based on conditions
        new_ivl = torch.where(pass_condition, current_calculated_ivl, new_ivl)
        new_ivl = torch.where(rating == 1, ivl_if_fail, new_ivl)

        # Handle first review specifically for interval
        new_ivl = torch.where(is_first_review & (rating < 4), self.w[0], new_ivl) # Graduating interval
        new_ivl = torch.where(is_first_review & (rating == 4), self.w[1], new_ivl) # Easy interval for first review
        new_ease = torch.where(is_first_review, torch.ones_like(ease) * self.w[2], new_ease) # Starting ease

        # Max interval, fuzz, etc. are not modeled here for simplicity but exist in Anki.
        # Ensure interval is at least 1 day if it's positive (or S_MIN if that's preferred)
        # Anki's "next day" is often max(current_interval + 1, calculated_interval) for reviews.
        # Here, we simplify to clamping.
        new_ivl = torch.max(new_ivl, torch.ones_like(new_ivl) * S_MIN) # TODO: S_MIN global
        new_ivl = new_ivl.clamp(S_MIN, S_MAX) # TODO: S_MIN, S_MAX globals

        return torch.stack([new_ivl, new_ease], dim=1)

    def forward(
        self, inputs: Tensor, state: Optional[Tensor] = None
    ) -> tuple[Tensor, Tensor]:
        """
        :param inputs: shape[seq_len, batch_size, 2 (delta_t, rating)]
        """
        if state is None: # Initial state: interval 0, ease 0 (or specific starting ease if known)
            state = torch.zeros((inputs.shape[1], 2), device=inputs.device)
            # state[:, 1] = self.w[2] # Initialize ease factor - step handles this via is_first_review

        outputs = []
        for X_step in inputs:
            state = self.step(X_step, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def iter(
        self,
        sequences: Tensor,
        delta_ts: Tensor,
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs_history, _ = self.forward(sequences)
        last_states = outputs_history[seq_lens - 1, torch.arange(real_batch_size)]
        intervals = last_states[:, 0] # Interval is used as stability proxy
        retentions = self.forgetting_curve(delta_ts, intervals)
        return {"retentions": retentions, "stabilities": intervals} # Changed "intervals" to "stabilities" for consistency

    def state_dict(self) -> List[float]: # Override to match original format
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )
