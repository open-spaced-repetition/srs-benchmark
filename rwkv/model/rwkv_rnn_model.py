from typing import List, Optional, Tuple
from rwkv.model.rwkv_model import LoraMLP, LoraSimple, RWKV7ChannelMixer, RWKV7Config
from rwkv.model.rwkv_ops import RWKV7_WKV, reference_rwkv7, single_timestep
import torch
import copy

# RWKV-7 formulated as an RNN for inference.
# For the first iteration just pass a None for the state parameter.


def __nop(ob):
    return ob


ModuleType = torch.nn.Module
FunctionType = __nop

# ModuleType = torch.jit.ScriptModule
# FunctionType = torch.jit.script_method


class RWKV7RNN(torch.nn.Module):
    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.blocks = torch.nn.ModuleList(
            [RWKV7RNNLayer(config, layer_id) for layer_id in range(config.n_layers)]
        )

    def forward(self, in_BC, state):
        if state is None:
            state = {}
            for i in range(len(self.blocks)):
                state[i] = None

        x_BC, v0_BC = in_BC, torch.empty_like(in_BC)
        for i, block in enumerate(self.blocks):
            x_BC, v0_BC, block_state = block(in_BC=x_BC, v0_BC=v0_BC, state=state[i])
            state[i] = block_state
        return x_BC, state

    def run(self, in_BC, state):
        state = copy.deepcopy(state)
        return self.forward(in_BC, state)


class RWKV7RNNLayer(ModuleType):
    def __init__(self, config: RWKV7Config, layer_id):
        super().__init__()
        self.time_mixer = RWKV7RNNTimeMixer(config, layer_id)
        self.channel_mixer = RWKV7RNNChannelMixer(config, layer_id)

    def forward(
        self,
        in_BC,
        v0_BC,
        state: Optional[
            Tuple[Optional[Tuple[torch.Tensor, torch.Tensor]], Optional[torch.Tensor]]
        ],
    ):
        if state is None:
            state = None, None
        time_state, channel_state = state
        x_BC, v0_BC, time_state = self.time_mixer(
            in_BC=in_BC, v0_BC=v0_BC, state=time_state
        )
        x_BC, channel_state = self.channel_mixer(x_BC, state=channel_state)
        return x_BC, v0_BC, (time_state, channel_state)


class RWKV7RNNChannelMixer(ModuleType):
    def __init__(self, config: RWKV7Config, layer_id):
        super().__init__()
        ratio_1_to_almost_0 = 1.0 - (layer_id / config.n_layers)
        self.layer_norm = torch.nn.LayerNorm(config.d_model)

        channel_ratio = torch.ones(1, 1, config.d_model)
        for i in range(config.d_model):
            channel_ratio[0, 0, i] = i / config.d_model

        self.lerp_k = torch.nn.Parameter(
            1 - torch.pow(channel_ratio, ratio_1_to_almost_0**4)
        )

        k_dim = int(config.channel_mixer_factor * config.d_model)
        self.W_k = torch.nn.Linear(config.d_model, k_dim, bias=False)
        self.W_v = torch.nn.Linear(k_dim, config.d_model, bias=False)

    def forward(self, in_BC, state: Optional[torch.Tensor]):
        x_shift_B1C = state
        assert len(in_BC.shape) == 2
        assert self.lerp_k.dtype == in_BC.dtype

        in_B1C = in_BC.unsqueeze(1)
        x_B1C = self.layer_norm(in_B1C)
        if x_shift_B1C is None:
            x_shift_B1C = x_B1C

        x_layer_norm_B1C = x_B1C
        k_B1K = self.W_k(torch.lerp(x_B1C, x_shift_B1C, self.lerp_k))
        o_B1C = self.W_v(torch.square(torch.nn.functional.relu(k_B1K)))

        return (in_B1C + o_B1C).squeeze(1), x_layer_norm_B1C


class RWKV7RNNTimeMixer(ModuleType):
    def __init__(self, config: RWKV7Config, layer_id):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.layer_id = layer_id
        C = config.d_model
        self.H = config.n_heads
        self.K = C // config.n_heads

        self.layer_norm = torch.nn.LayerNorm(config.d_model)
        self.rkvdag_lerp = torch.nn.Parameter(torch.empty(8, 1, 1, config.d_model))
        self.bonus = torch.nn.Parameter(
            torch.zeros(1, 1, config.n_heads, config.d_model // config.n_heads)
        )  # r_k

        self.W_r = torch.nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_k = torch.nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_v = torch.nn.Linear(config.d_model, config.d_model, bias=False)
        self.W_o = torch.nn.Linear(config.d_model, config.d_model, bias=False)

        self.k_scale_linear = torch.nn.Linear(config.d_model, self.H, bias=True)
        self.v_scale_linear = torch.nn.Linear(config.d_model, self.H, bias=True)
        self.v_lora_simple = LoraSimple(
            name="v",
            d_model=config.d_model,
            d_lora=config.v0_mix_amt_lora,
            layer_id=layer_id,
        )
        self.a_lora_simple = LoraSimple(
            name="a", d_model=config.d_model, d_lora=config.a_lora, layer_id=layer_id
        )
        self.d_lora_mlp = LoraMLP(
            name="d",
            config=config,
            d_lora=config.decay_lora,
            out_dim=config.d_model,
            layer_id=layer_id,
        )

        self.lora_A_g = torch.nn.Linear(config.d_model, config.gate_lora, bias=False)
        self.lora_B_g = torch.nn.Linear(config.gate_lora, config.d_model, bias=False)

        self.out_group_norm = torch.nn.GroupNorm(
            config.n_heads, config.d_model, eps=64e-5
        )

    def forward(self, in_BC, v0_BC, state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        B, C = in_BC.shape
        H, K = self.H, self.K

        assert len(in_BC.shape) == 2
        assert self.bonus.dtype == in_BC.dtype
        in_B1C = in_BC.unsqueeze(1)

        x_B1C = self.layer_norm(in_B1C)
        x_layer_norm_B1C = x_B1C
        if state is None:
            x_shift_B1C = x_B1C
            state_B1HKK = torch.zeros(
                B, 1, H, K, K, dtype=torch.float32, device=in_BC.device
            )
        else:
            x_shift_B1C, state_B1HKK = state

        rkvdag_6B1C = torch.lerp(
            x_B1C.unsqueeze(0), x_shift_B1C.unsqueeze(0), self.rkvdag_lerp
        )
        r_B1C, k_B1C, v_B1C, d_B1C, a_B1C, g_B1C, k_scale_B1C, v_scale_B1C = (
            rkvdag_6B1C.unbind(dim=0)
        )
        r_B1C = self.W_r(r_B1C)
        k_B1C = self.W_k(k_B1C)
        k_scale_B1H = torch.nn.functional.sigmoid(self.k_scale_linear(k_scale_B1C))
        v_scale_B1H = torch.nn.functional.sigmoid(self.v_scale_linear(v_scale_B1C))

        if self.layer_id == 0:
            v_B1C = self.W_v(v_B1C)
            v0_BC = v_B1C.squeeze(1)
        else:
            v_lerp_B1C = torch.nn.functional.sigmoid(self.v_lora_simple(v_B1C))
            v_B1C = torch.lerp(self.W_v(v_B1C), v0_BC.unsqueeze(1), v_lerp_B1C)

        a_B1C = torch.nn.functional.sigmoid(self.a_lora_simple(a_B1C))
        g_B1C = self.lora_B_g(torch.nn.functional.sigmoid(self.lora_A_g(g_B1C)))

        _d_B1C = -0.5 - torch.nn.functional.softplus(-self.d_lora_mlp(d_B1C))
        w_B1C = torch.exp(-torch.exp(_d_B1C.float()))

        k_B1HK = k_scale_B1H.unsqueeze(-1) * torch.nn.functional.normalize(
            k_B1C.view(B, 1, H, K), dim=-1, p=2.0
        )
        r_B1HK = r_B1C.view(B, 1, H, K)
        v_B1HK = v_scale_B1H.unsqueeze(-1) * torch.nn.functional.normalize(
            v_B1C.view(B, 1, H, K), dim=-1, p=2.0
        )
        w_B1HK = w_B1C.view(B, 1, H, K)
        a_B1HK = a_B1C.view(B, 1, H, K)
        k_deformed_B1HK = k_B1HK
        k_B1HK = k_B1HK * a_B1HK

        out_BHK, next_state_BHKK = single_timestep(
            r_B1HK.float().squeeze(1),
            k_B1HK.float().squeeze(1),
            v_B1HK.float().squeeze(1),
            w_B1HK.float().squeeze(1),
            a_B1HK.float().squeeze(1),
            k_deformed_B1HK.float().squeeze(1),
            state_B1HKK.float().squeeze(1),
        )

        out_B1HK = out_BHK.to(in_B1C.dtype).unsqueeze(1)

        out_B1C = self.out_group_norm(out_B1HK.view(B, C)).view(B, 1, C)
        bonus_B1C = (
            (r_B1HK * self.bonus * k_B1HK).sum(dim=-1, keepdim=True) * v_B1HK
        ).view(B, 1, C)
        out_B1C = self.W_o(g_B1C * (out_B1C + bonus_B1C))
        return (
            (in_B1C + out_B1C).squeeze(1),
            v0_BC,
            (x_layer_norm_B1C, next_state_BHKK.unsqueeze(1)),
        )


if __name__ == "__main__":
    pass
