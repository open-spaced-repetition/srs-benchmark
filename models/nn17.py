import torch
import torch.nn as nn
from torch import Tensor
from typing import List, Optional # Added Optional

# TODO: Refactor to pass configuration instead of relying on globals
S_MIN = 0.01       # Placeholder, ensure this is the correct S_MIN for NN-17
INIT_S_MAX = 100   # Placeholder
S_MAX = 36500      # Placeholder
DEVICE = torch.device("cpu") # Placeholder
FILE_NAME = "default_model_name" # Placeholder

def exp_activ(input: Tensor) -> Tensor:
    return torch.exp(-input).clamp(0.0001, 0.9999)

class ExpActivation(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor) -> Tensor:
        return exp_activ(input)

class NN_17ParameterClipper:
    def __init__(self, frequency: int = 1):
        self.frequency = frequency

    def __call__(self, module):
        if hasattr(module, "S0"): # Stability init params
            w = module.S0.data
            w[0] = w[0].clamp(S_MIN, INIT_S_MAX) # TODO: S_MIN, INIT_S_MAX globals
            w[1] = w[1].clamp(S_MIN, INIT_S_MAX) # TODO: S_MIN, INIT_S_MAX globals
            w[2] = w[2].clamp(S_MIN, INIT_S_MAX) # TODO: S_MIN, INIT_S_MAX globals
            w[3] = w[3].clamp(S_MIN, INIT_S_MAX) # TODO: S_MIN, INIT_S_MAX globals
            module.S0.data = w

        if hasattr(module, "D0"): # Difficulty init params
            w = module.D0.data
            w[0] = w[0].clamp(0, 1)
            w[1] = w[1].clamp(0, 1)
            w[2] = w[2].clamp(0, 1)
            w[3] = w[3].clamp(0, 1)
            module.D0.data = w

        if hasattr(module, "sinc_w"): # Parameters for Sinc (stability increase)
            w = module.sinc_w.data
            w[0] = w[0].clamp(-5, 5)
            w[1] = w[1].clamp(0, 1)
            w[2] = w[2].clamp(-5, 5)
            module.sinc_w.data = w

class NN_17(nn.Module):
    # 39 params
    init_s = [1, 2.5, 4.5, 10] # Initial stabilities for ratings 1,2,3,4
    init_d = [1, 0.72, 0.07, 0.05] # Initial difficulties for ratings 1,2,3,4
    w = [1.26, 0.0, 0.67] # Parameters for Sinc component

    clipper = NN_17ParameterClipper()
    lr: float = 4e-2
    wd: float = 1e-5
    n_epoch: int = 5

    def __init__(self, state_dict=None) -> None:
        super(NN_17, self).__init__()
        self.hidden_size = 1 # Defines hidden layer size for the small NNs inside

        self.S0 = nn.Parameter(torch.tensor(self.init_s, dtype=torch.float32))
        self.D0 = nn.Parameter(torch.tensor(self.init_d, dtype=torch.float32))
        self.sinc_w = nn.Parameter(torch.tensor(self.w, dtype=torch.float32))

        # Neural network for retention given (Difficulty, Stability, Theoretical_R)
        self.rw = nn.Sequential(
            nn.Linear(3, self.hidden_size),
            nn.Mish(),
            nn.Linear(self.hidden_size, 1),
            nn.Softplus(),
            ExpActivation(), # Outputs retention probability
        )
        # Neural network for next difficulty given (last_Difficulty, Retention_weighted, Rating)
        self.next_d_nn = nn.Sequential( # Renamed from next_d to avoid conflict with method
            nn.Linear(3, self.hidden_size),
            nn.Mish(),
            nn.Linear(self.hidden_size, 1),
            nn.Sigmoid(), # Outputs new difficulty (0-1 range, then scaled)
        )
        # Neural network for post-lapse stability given (Retention_weighted, Lapses)
        self.pls = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.Mish(),
            nn.Linear(self.hidden_size, 1),
            nn.Softplus(), # Outputs post-lapse stability
        )
        # Neural network for stability increase (part 1)
        self.sinc_nn_part = nn.Sequential( # Renamed from sinc to avoid conflict
            nn.Linear(3, self.hidden_size),
            nn.Mish(),
            nn.Linear(self.hidden_size, 1),
            nn.Softplus(),
        )
        # Neural network for stability increase (part 2, combining traditional and NN components)
        self.best_sinc = nn.Sequential(
            nn.Linear(2, self.hidden_size),
            nn.Mish(),
            nn.Linear(self.hidden_size, 1),
            nn.Softplus(),
        )

        if state_dict is not None:
            self.load_state_dict(state_dict)
        else:
            try:
                # TODO: FILE_NAME, DEVICE globals
                self.load_state_dict(
                    torch.load(
                        f"./pretrain/{FILE_NAME}_pretrain.pth", # FILE_NAME used here
                        weights_only=True,
                        map_location=DEVICE, # DEVICE used here
                    )
                )
            except FileNotFoundError:
                pass

    def forward(self, inputs: Tensor, state: Optional[Tensor] = None) -> tuple[Tensor, Tensor]:
        if state is None: # Default state if none provided
            state = torch.ones((inputs.shape[1], 2), device=inputs.device) # TODO: DEVICE? Or inputs.device
            # Initialize S0 and D0 based on first rating if possible, otherwise use a default.
            # This part is tricky as the first rating isn't directly in `inputs` for step 0.
            # The original code's `step` handles this. Here, we might need to assume a default
            # or require `inputs` to have a way to signal first rating for state init.
            # For simplicity, the step function will handle state init if state is all ones.
            # However, NN-17 in other.py initializes state with zeros for FSRS base class compatibility.
            # Let's stick to what step expects (it checks for `torch.ones_like(state)`)
            # The original `step` function in `other.py` for NN-17 initializes state in a specific way
            # if it's the first review. This needs to be replicated in the `step` method.
            # So, state should be initialized to enable detection of first review in `step`.
            # A common approach is to pass dummy values that `step` can identify.
            # Or, ensure `step` is robust to a generic zero/one initial state.
            # The current `step` expects `torch.ones_like(state)` for the first review.
            pass # State initialization is handled in the first call to step

        outputs = []
        for X_step in inputs: # Iterate over sequence length
            state = self.step(X_step, state)
            outputs.append(state)
        return torch.stack(outputs), state

    def iter(
        self,
        sequences: Tensor, # [seq_len, batch_size, 3] (delta_t, rating, lapses)
        delta_ts: Tensor,  # [batch_size] - delta_t for the *current* event, used for retention calc
        seq_lens: Tensor,
        real_batch_size: int,
    ) -> dict[str, Tensor]:
        outputs_history, _ = self.forward(sequences) # outputs_history is [seq_len, batch_size, 2 (S, D)]

        last_states = outputs_history[seq_lens - 1, torch.arange(real_batch_size, device=DEVICE)] # TODO: DEVICE global
        stabilities = last_states[:, 0]
        difficulties = last_states[:, 1]

        theoretical_r = self.forgetting_curve(delta_ts, stabilities)
        retentions = self.rw(
            torch.stack([difficulties, stabilities, theoretical_r], dim=1)
        ).squeeze(-1) # Squeeze the last dimension from [batch_size, 1] to [batch_size]
        return {"retentions": retentions, "stabilities": stabilities}

    def step(self, X_step: Tensor, state: Tensor) -> Tensor:
        """
        :param X_step: shape[batch_size, 3]
            X_step[:,0] is elapsed time (delta_t)
            X_step[:,1] is rating
            X_step[:,2] is lapses (cumulative count of fails for the item)
        :param state: shape[batch_size, 2]
            state[:,0] is stability (S)
            state[:,1] is difficulty (D, 0-1 range)
        :return next_state:
        """
        delta_t = X_step[:, 0].unsqueeze(1)
        rating = X_step[:, 1].unsqueeze(1)
        lapses = X_step[:, 2].unsqueeze(1)

        # Check if it's the first review (state might be all ones as per original logic)
        # The original code for NN-17 in other.py used state = torch.ones((inputs.shape[1], 2))
        # as the initial state passed to forward.
        if torch.all(state == 1.0): # Heuristic for first review if state comes in as all 1s
            keys = torch.tensor([1, 2, 3, 4], device=DEVICE) # TODO: DEVICE global
            # Expand keys to match batch size for vectorized comparison
            keys_expanded = keys.view(1, -1).expand(rating.size(0), -1)
            # Find indices where rating matches keys
            # rating is [batch_size, 1], so squeeze it for comparison
            index = (rating.long() == keys_expanded).nonzero(as_tuple=True)

            new_s = torch.zeros_like(state[:, 0], device=DEVICE) # TODO: DEVICE global
            new_s[index[0]] = self.S0[index[1]] # Initialize S from S0 based on first rating
            new_s = new_s.unsqueeze(1)

            new_d_init_vals = torch.zeros_like(state[:, 1], device=DEVICE) # TODO: DEVICE global
            new_d_init_vals[index[0]] = self.D0[index[1]] # Initialize D from D0
            new_d = new_d_init_vals.unsqueeze(1)
        else:
            last_s = state[:, 0].unsqueeze(1)
            last_d = state[:, 1].unsqueeze(1) # Difficulty is expected to be in [0,1] from nn

            rt = self.forgetting_curve(delta_t, last_s) # Theoretical Retention
            rt = rt.clamp(0.0001, 0.9999)

            rw_input = torch.cat([last_d, last_s, rt], dim=1)
            rw = self.rw(rw_input) # Weighted Retention (model's belief)
            rw = rw.clamp(0.0001, 0.9999)

            sr = self.inverse_forgetting_curve(rw, delta_t) # Stability that would lead to rw
            sr = sr.clamp(S_MIN, S_MAX) # TODO: S_MIN, S_MAX globals

            next_d_input = torch.cat([last_d, rw, rating], dim=1)
            new_d = self.next_d_nn(next_d_input) # New difficulty from NN (0-1 range)

            pls_input = torch.cat([rw, lapses], dim=1)
            pls = self.pls(pls_input) # Post-lapse stability
            pls = pls.clamp(S_MIN, S_MAX) # TODO: S_MIN, S_MAX globals

            # Stability Increase Component (SInc)
            # Traditional calculation part
            sinc_t = 1 + torch.exp(self.sinc_w[0]) * (5 * (1 - new_d) + 1) * torch.pow(
                sr, -self.sinc_w[1]
            ) * torch.exp(-rw * self.sinc_w[2])

            # NN based calculation part
            sinc_nn_input = torch.cat([new_d, sr, rw], dim=1)
            sinc_nn = 1 + self.sinc_nn_part(sinc_nn_input) # Renamed self.sinc to self.sinc_nn_part

            # Combine traditional and NN SInc components
            best_sinc_input = torch.cat([sinc_t, sinc_nn], dim=1)
            best_sinc = 1 + self.best_sinc(best_sinc_input)
            best_sinc = best_sinc.clamp(1, 100) # Clamp the multiplier

            new_s = torch.where(
                rating > 1, # If success
                sr * best_sinc,
                pls, # If fail
            )

        new_s = new_s.clamp(S_MIN, S_MAX) # TODO: S_MIN, S_MAX globals
        new_d = new_d.clamp(0.0001, 1.0) # Ensure D stays in [0,1] after NN
        next_state = torch.cat([new_s, new_d], dim=1)
        return next_state

    def forgetting_curve(self, t: Tensor, s: Tensor) -> Tensor:
        return 0.9 ** (t / s)

    def inverse_forgetting_curve(self, r: Tensor, t: Tensor) -> Tensor:
        log_09 = -0.10536051565782628 # log(0.9)
        return log_09 / torch.log(r) * t

    # state_dict method is not overridden, will use default nn.Module.state_dict()
