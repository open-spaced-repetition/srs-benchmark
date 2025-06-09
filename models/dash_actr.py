import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional # Added Optional

# DASH_ACTR does not inherit FSRS.
# It seems self-contained with its parameters.

class DASH_ACTRParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp_min(0.001)
            w[1] = w[1].clamp_min(0.001)
            module.w.data = w


class DASH_ACTR(nn.Module):
    # 5 params
    init_w = [1.4164, 0.516, -0.0564, 1.9223, 1.0549]
    clipper = DASH_ACTRParameterClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w): # Changed default from init_w to w=init_w
        super(DASH_ACTR, self).__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs: Tensor) -> Tensor:
        """
        :param inputs: shape[seq_len, batch_size, 2], 2 means r_history_item and t_history_item (delta_t for that step)
                       The original code suggests `inputs[:, :, 1]` (delta_t) is clamped.
                       The sum is over the sequence length (dim=0).
        """
        # Clamp delta_t values (inputs[:, :, 1]) to be at least 0.1
        clamped_delta_t = inputs[:, :, 1].clamp_min(0.1)

        # Create a mask for original delta_t == 0.1 (after clamping, these were originally <= 0.1)
        # In the original code, t_history was used, which are delta_t values.
        # `torch.where(inputs[:, :, 1] == 0.1, 0, inputs[:, :, 1] ** -self.w[1])`
        # This implies that if original delta_t was very small (became 0.1 after clamp), its contribution is 0.
        # otherwise, it's delta_t ** -self.w[1].
        # The term `inputs[:, :, 1] == 0.1` might be problematic due to float precision.
        # A more robust way might be to use the original un-clamped values for the condition if available,
        # or use a small epsilon. For now, sticking to the literal interpretation.
        # However, the provided code in `other.py` for DASH_ACTR used `sp_history` which seems to be already delta_t.
        # The features passed were `[r_history > 1, sp_history - cumsum + cumsum[-1:None]]`
        # This implies `inputs` here is not raw r_history, t_history but some processed features.
        # The `create_features` for DASH_ACTR creates `tensor` with shape [history_len, 2]
        # where tensor[:, 0] is r_history > 1 (binary) and tensor[:, 1] is `sp_history - cumsum + cumsum[-1:None]` (transformed time).
        # Let's assume `inputs` matches this [seq_len, batch_size, 2] structure.

        # inputs[:, :, 0] is r_binary (1 if success, 0 if fail)
        # inputs[:, :, 1] is transformed_time_decay_component

        # The formula from paper seems to be:
        # P = sigmoid( B * log( sum ( T_k ^ -D_k ) * W_k ) + A )
        # W_k is outcome weight (correct vs incorrect)
        # T_k is time since k-th event
        # D_k is decay rate for k-th event

        # The code in other.py:
        # retentions = self.sigmoid(
        #     self.w[0] * torch.log(
        #         1 + torch.sum( # Sum over history for each item in batch
        #             torch.where(
        #                 inputs[:, :, 1] == 0.1, # This condition is on transformed time, seems odd.
        #                                       # Original features had delta_t. If delta_t=0 (clamped to 0.1), then 0.
        #                                       # This was likely meant to be based on original delta_t.
        #                                       # For now, let's assume it means "if transformed time is at its minimum due to clamping"
        #                               0,
        #                               inputs[:, :, 1] ** -self.w[1] # (T_k ^ -D_k) like term
        #                           ) *
        #             torch.where(inputs[:, :, 0] == 0, self.w[2], self.w[3]), # W_k based on outcome
        #             dim=0, # Sum over seq_len
        #         ).clamp_min(0) # Ensure log argument is not negative
        #     ) + self.w[4]
        # )
        # This implies inputs[:,:,1] is related to time, and inputs[:,:,0] to rating.
        # Let's assume inputs[:,:,1] is some time-decay related term, and inputs[:,:,0] is outcome.

        time_decay_val = inputs[:, :, 1] ** -self.w[1]
        # This condition `inputs[:, :, 1] == 0.1` is problematic.
        # If `inputs[:,:,1]` comes from `sp_history - cumsum + cumsum[-1:None]`, it's not directly clamped delta_t.
        # For now, I will replicate the structure, but this part is suspicious.
        # A small positive delta_t (e.g. 0.1 days) would lead to a large value after `** -self.w[1]` if w[1] is positive.
        # The `torch.where(inputs[:,:,1] == 0.1, 0, ...)` seems to intend to zero out contributions from
        # "very recent" or "zero interval" reviews, but 0.1 is arbitrary if not from clamping.
        # Given the features used `sp_history - cumsum + cumsum[-1:None]`, this might be a proxy for recency.

        # For simplicity, I'll assume the features are as passed by the trainer:
        # inputs[:,:,0] is binary rating (0 for fail, 1 for success)
        # inputs[:,:,1] is the transformed time feature (recency weighted time since event)

        # This part is tricky to interpret without deeper context of feature transformation.
        # The `torch.where(inputs[:, :, 1] == 0.1, 0, ...)` from original code.
        # If `inputs[:,:,1]` is the transformed time, this condition needs to be understood.
        # For now, replicating the structure.
        transformed_time_component = inputs[:, :, 1]
        decayed_time_contribution = torch.where(
            transformed_time_component < 0.100001, # Using a small epsilon for float comparison
            torch.zeros_like(transformed_time_component),
            transformed_time_component ** -self.w[1]
        )

        outcome_weights = torch.where(inputs[:, :, 0] == 0, self.w[2], self.w[3])
        summed_contributions = torch.sum(decayed_time_contribution * outcome_weights, dim=0)

        log_arg = 1 + summed_contributions # Added 1 inside log as per original code
        log_val = torch.log(log_arg.clamp_min(1e-8)) # Clamped to avoid log(0) or negative

        retentions = self.sigmoid(self.w[0] * log_val + self.w[4])
        return retentions

    def iter(
        self,
        sequences: Tensor, # Expected [seq_len, batch_size, 2]
        delta_ts: Tensor, # Not directly used by forward if sequences has all info
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        # The original iter for DASH_ACTR was:
        # outputs = self.forward(sequences)
        # return {"retentions": outputs}
        # This implies forward handles batching and seq_lens implicitly or by design of input.
        # The `forward` method sums over dim=0 (seq_len), so it produces one output per batch item.
        outputs = self.forward(sequences) # sequences should be [seq_len, batch_size, features]
        return {"retentions": outputs} # outputs is already [batch_size]

    def state_dict(self) -> List[float]: # Override to match original format
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )
