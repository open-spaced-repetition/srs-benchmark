import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional # Added Optional

# ACT_R does not inherit FSRS, so FSRS base globals are not directly applicable unless used in training logic.
# It seems self-contained with its parameters.

class ACT_RParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "w"):
            w = module.w.data
            w[0] = w[0].clamp(0.001, 1)    # decay intercept (a)
            w[1] = w[1].clamp(0.001, 1)    # decay scale (c)
            w[2] = w[2].clamp(0.001, 1)    # noise (s)
            w[3] = w[3].clamp_max(-0.001)  # threshold (tau)
            w[4] = w[4].clamp(0.001, 1)    # interference scalar (h)
            module.w.data = w


class ACT_R(nn.Module):
    # 5 params
    a = 0.176786766570677  # decay intercept
    c = 0.216967308403809  # decay scale
    s = 0.254893976981164  # noise
    tau = -0.704205679427144  # threshold
    h = 0.025  # interference scalar
    init_w = [a, c, s, tau, h]
    clipper = ACT_RParameterClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, w: List[float] = init_w):
        super().__init__()
        self.w = nn.Parameter(torch.tensor(w, dtype=torch.float32))

    def forward(self, sp: Tensor) -> Tensor: # sp is expected to be sequence of delta_t for a single item
        """
        :param sp: shape[seq_len, batch_size, 1], sp contains delta_t values (elapsed time since previous review)
                   The original code implies sp is just the sequence of delta_t for one item at a time,
                   as cumsum is applied. For batch processing, this needs careful handling.
                   Assuming sp here is [seq_len, batch_size, 1] where each item in batch is independent.
        """
        # This implementation assumes batch_size = 1 or that calculations are independent per batch item.
        # If batch_size > 1, the sum in torch.log needs to be handled carefully,
        # perhaps by processing each item in the batch in a loop or using more advanced tensor ops.
        # The original code's `iter` method took `sequences` which were then passed to `forward`.
        # `sequences` in `other.py` for ACT-R was `t_history_used` then `torch.cumsum`.
        # This implies `sp` here is already cumulative time.
        # Let's assume `sp` is [seq_len, batch_size] representing cumulative times for each item.

        m = torch.zeros_like(sp, dtype=torch.float) # [seq_len, batch_size]
        if sp.ndim == 1: # If processing a single item, unsqueeze for batch compatibility
            sp = sp.unsqueeze(1)
            m = m.unsqueeze(1)

        m[0] = -torch.inf # Activation of first event is undefined or infinitely negative

        for i in range(1, sp.shape[0]): # Iterate through sequence length
            # For each item in batch, and for each time step i:
            # Calculate activation based on its own history up to i-1
            # sp[i] is current cumulative time, sp[0:i] are past cumulative times
            # This requires a loop per batch item if sum is over history of *that specific item*.
            # The original code's structure suggests it processes one item fully.
            # For simplicity here, we'll assume the sum is over the items in the batch if that's the intent,
            # or this needs to be wrapped in a per-item loop.
            # Given `torch.sum(..., dim=0)`, it sums across the sequence length dimension if sp[0:i] is used directly.
            # This seems more aligned with ACT-R's formulation where sum is over past presentations.

            # Replicating the logic from other.py: tensor `sp` here is `cumsum(delta_t)`.
            # `sp[i]` is the time of the i-th event. `sp[0:i]` are times of previous events.
            # `sp[i] - sp[0:i]` calculates time since previous events.

            # This loop is for seq_len. Batch items are processed in parallel by tensor ops.
            # act_components will be of shape [i, batch_size]
            time_since_past_events = (sp[i].unsqueeze(0) - sp[0:i]) * 86400 * self.w[4] # Add h factor here
            act_components = time_since_past_events.clamp_min(1) ** (-(self.w[1] * torch.exp(m[0:i]) + self.w[0]))
            act = torch.log(torch.sum(act_components, dim=0)) # Sum over history (dim 0)
            m[i] = act

        return self.activation(m[1:]) # Return activations from the second event onwards

    def iter(
        self,
        sequences: Tensor, # For ACT-R, this is expected to be cumulative times [seq_len, batch_size, 1]
        delta_ts: Tensor, # Not directly used by ACT-R forward, but kept for compatibility
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        # Assuming sequences has shape [seq_len, batch_size, 1] and contains cumulative times
        outputs = self.forward(sequences.squeeze(-1)) # Remove last dim if it's 1
        # The output of forward is [seq_len-1, batch_size]
        # We need to pick the activation corresponding to the actual sequence length for each item
        # This is complex because seq_lens refers to original sequence, and outputs is shorter by 1.
        # The original code in `script.py` for ACT-R processes one user at a time,
        # and its `tensor` input to `ACT_R.forward` seems to be just for that one user.
        # The `iter` method in `other.py` for ACT-R was:
        # outputs = self.forward(sequences)
        # return {"retentions": outputs[seq_lens - 2, torch.arange(real_batch_size, device=DEVICE),0]}
        # This implies `outputs` is [seq_len-1, batch_size] (if forward returns that way)
        # or [seq_len-1, batch_size, 1] and then it's squeezed.
        # Given current forward, outputs is [seq_len-1, batch_size]
        retentions = outputs[seq_lens - 2, torch.arange(real_batch_size, device=DEVICE)] # TODO: DEVICE global
        return {"retentions": retentions}


    def activation(self, m: Tensor) -> Tensor:
        return 1 / (1 + torch.exp((self.w[3] - m) / self.w[2]))

    def state_dict(self) -> List[float]: # Override to match original format
        return list(
            map(
                lambda x: round(float(x), 4),
                dict(self.named_parameters())["w"].data,
            )
        )
