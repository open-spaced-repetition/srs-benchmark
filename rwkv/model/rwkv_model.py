from dataclasses import dataclass
import math
import torch

from rwkv.model.rwkv_ops import RWKV7_WKV, reference_rwkv7

"""
IMPORTANT: the CUDA implementation in this repository only supports head dimensions of 32. d_model // n_heads == 32.

Sources:
https://github.com/BlinkDL/RWKV-LM/blob/main/RWKV-v5/src/model.py#L766
https://github.com/SmerkyG/RWKV_Explained/blob/main/rwkv7.py
"""

torch.manual_seed(2025)

# def __nop(ob):
#     return ob
# ModuleType = torch.nn.Module
# FunctionType = __nop

ModuleType = torch.jit.ScriptModule
FunctionType = torch.jit.script_method

@dataclass
class RWKV7Config:
    d_model: int  # The model dimension. d_model / n_heads is the dimension for each head.
    n_heads: int
    n_layers: int
    channel_mixer_factor: int

    # For stacking RWKV7 on top of one-another. We allow sending in the total number of layers and a layer offset so that we can achieve better initialization
    layer_offset: int  
    total_layers: int

    decay_lora: int
    a_lora: int  # a = in-context learning rate
    v0_mix_amt_lora: int
    gate_lora: int

    dropout: float
    dropout_layer: float

class RWKV7(ModuleType):
    def __init__(self, config: RWKV7Config):
        super().__init__()
        self.blocks = torch.nn.ModuleList([RWKV7Layer(config, layer_id) for layer_id in range(config.layer_offset, config.layer_offset + config.n_layers)])

    @FunctionType
    def forward(self, in_BTC, time_shift_select_BT, skip_BT):
        x_BTC, v0_BTC = in_BTC, torch.empty_like(in_BTC)
        for _, block in enumerate(self.blocks):
            x_BTC, v0_BTC = block(in_BTC=x_BTC, v0_BTC=v0_BTC, time_shift_select_BT=time_shift_select_BT, skip_BT=skip_BT)
        return x_BTC

class RWKV7Layer(ModuleType):
    def __init__(self, config: RWKV7Config, layer_id):
        super().__init__()
        self.time_mixer = RWKV7TimeMixer(config, layer_id)
        self.channel_mixer = RWKV7ChannelMixer(config, layer_id)
        self.dropout = torch.nn.Dropout(p=config.dropout_layer)

    @FunctionType
    def forward(self, in_BTC, v0_BTC, time_shift_select_BT, skip_BT):
        x_BTC, v0_BTC = self.time_mixer(in_BTC=in_BTC, v0_BTC=v0_BTC, time_shift_select_BT=time_shift_select_BT, skip_BT=skip_BT)
        return self.dropout(self.channel_mixer(x_BTC, time_shift_select_BT=time_shift_select_BT)), v0_BTC

class RWKV7ChannelMixer(ModuleType):
    # Also the same as for RWKV-5
    def __init__(self, config: RWKV7Config, layer_id):
        super().__init__()
        assert config.d_model // config.n_heads == 32
        self.d_model = config.d_model
        with torch.no_grad():
            ratio_1_to_almost_0 = 1.0 - (layer_id / config.total_layers)
            self.layer_norm = torch.nn.LayerNorm(config.d_model)
            self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))
            
            channel_ratio = torch.ones(1, 1, config.d_model)
            for i in range(config.d_model):
                channel_ratio[0, 0, i] = i / config.d_model

            self.lerp_k = torch.nn.Parameter(1 - torch.pow(channel_ratio, ratio_1_to_almost_0 ** 4))

            k_dim = int(config.channel_mixer_factor * config.d_model)
            self.W_k = torch.nn.Linear(config.d_model, k_dim, bias=False)
            self.W_v = torch.nn.Linear(k_dim, config.d_model, bias=False)

            self.W_k.weight.data.uniform_(-0.5/(config.d_model**0.5), 0.5/(config.d_model**0.5))
            self.W_v.weight.data.zero_()

            self.dropout = torch.nn.Dropout(p=config.dropout)

    @FunctionType
    def forward(self, in_BTC, time_shift_select_BT):
        x_BTC = self.layer_norm(in_BTC)
        x_shift_BTC = torch.gather(x_BTC, dim=1, index=time_shift_select_BT.unsqueeze(-1).expand(-1, -1, self.d_model))
        k_BTK = self.W_k(torch.lerp(x_BTC, x_shift_BTC, self.lerp_k))
        o_BTC = self.W_v(torch.square(torch.nn.functional.relu(k_BTK)))
        return in_BTC + self.dropout(o_BTC)

def ortho_init(x, scale):
    with torch.no_grad():
        shape = x.shape
        if len(shape) == 2:
            gain = math.sqrt(shape[0] / shape[1]) if shape[0] > shape[1] else 1
            torch.nn.init.orthogonal_(x, gain=gain * scale)
        elif len(shape) == 3:
            gain = math.sqrt(shape[1] / shape[2]) if shape[1] > shape[2] else 1
            for i in range(shape[0]):
                torch.nn.init.orthogonal_(x[i], gain=gain * scale)
        else:
            assert False
        return x

class LoraSimple(ModuleType):
    def __init__(self, name, d_model, d_lora, layer_id):
        super().__init__()
        with torch.no_grad():
            # The lambda term can be written out as a linear layer that includes a bias
            self.A = torch.nn.Linear(d_model, d_lora, bias=False)
            torch.nn.init.zeros_(self.A.weight)
            self.B_and_lamb = torch.nn.Linear(d_lora, d_model, bias=True)
            ortho_init(self.B_and_lamb.weight, scale=0.1)
            if name == "v":
                # Bias with ones to let the first layer's value flow directly
                torch.nn.init.ones_(self.B_and_lamb.bias)
            else:
                torch.nn.init.zeros_(self.B_and_lamb.bias)
            

    @FunctionType
    def forward(self, in_BTC):
        return self.B_and_lamb(self.A(in_BTC))

class LoraMLP(ModuleType):
    def __init__(self, name, config: RWKV7Config, d_lora, out_dim, layer_id):
        super().__init__()
        C = out_dim
        ratio_0_to_1 = layer_id / (config.total_layers - 1)

        with torch.no_grad():
            self.A = torch.nn.Linear(config.d_model, d_lora, bias=False)
            torch.nn.init.zeros_(self.A.weight)
            self.B_and_lamb = torch.nn.Linear(d_lora, out_dim, bias=True)
            ortho_init(self.B_and_lamb.weight, scale=0.1)
            if name == "d":
                decay_speed = torch.ones(C)
                for i in range(C):
                    decay_speed[i] = -7 + 5 * (i / (C - 1)) ** (0.85 + 1.0 * ratio_0_to_1 ** 0.5)
                self.B_and_lamb.bias.copy_(decay_speed + 0.5)
            else:
                torch.nn.init.zeros_(self.B_and_lamb.bias)

    @FunctionType
    def forward(self, in_BTC):
        return self.B_and_lamb(torch.nn.functional.tanh(self.A(in_BTC)))

class RWKV7TimeMixer(ModuleType):
    def __init__(self, config: RWKV7Config, layer_id):
        super().__init__()
        assert config.d_model % config.n_heads == 0
        self.layer_id = layer_id
        C = config.d_model
        self.d_model = C
        self.H = config.n_heads
        self.K = C // config.n_heads

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (config.n_layers - 1)
            ratio_1_to_almost_0 = 1.0 - (layer_id / config.n_layers)
            channel_ratio = torch.ones(1, 1, C)
            for i in range(C):
                channel_ratio[0, 0, i] = i / C

            self.layer_norm = torch.nn.LayerNorm(config.d_model)
            self.time_shift = torch.nn.ZeroPad2d((0, 0, 1, -1))

            self.rkvdag_lerp = torch.nn.Parameter(torch.empty(8, 1, 1, config.d_model))

            # Overall, the earlier the layer the more that we care about the shifted input.
            self.rkvdag_lerp[0] = 1.0 - torch.pow(channel_ratio, 0.2 * ratio_1_to_almost_0)  # r
            # The weight for k, v, can become negative and are roughly centered around 0 for the later layers.
            self.rkvdag_lerp[1] = 1.0 - (torch.pow(channel_ratio, 0.9 * ratio_1_to_almost_0) + 0.4 * ratio_0_to_1)  # k
            self.rkvdag_lerp[2] = 1.0 - (torch.pow(channel_ratio, 0.2 * ratio_1_to_almost_0) + 0.6 * ratio_0_to_1) # v
            self.rkvdag_lerp[3] = 1.0 - torch.pow(channel_ratio, 0.9 * ratio_1_to_almost_0)  # d (aka w)
            self.rkvdag_lerp[4] = 1.0 - torch.pow(channel_ratio, 0.9 * ratio_1_to_almost_0)  # a
            self.rkvdag_lerp[5] = 1.0 - torch.pow(channel_ratio, 0.2 * ratio_1_to_almost_0)  # g
            self.rkvdag_lerp[6] = 1.0 - torch.pow(channel_ratio, 0.9 * ratio_1_to_almost_0)
            self.rkvdag_lerp[7] = 1.0 - torch.pow(channel_ratio, 0.9 * ratio_1_to_almost_0)

            self.bonus = torch.nn.Parameter(torch.zeros(1, 1, config.n_heads, config.d_model // config.n_heads))  # r_k

            self.W_r = torch.nn.Linear(config.d_model, config.d_model, bias=False)
            self.W_k = torch.nn.Linear(config.d_model, config.d_model, bias=False)
            self.W_v = torch.nn.Linear(config.d_model, config.d_model, bias=False)
            self.W_o = torch.nn.Linear(config.d_model, config.d_model, bias=False)

            self.W_r.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.W_k.weight.data.uniform_(-0.05/(C**0.5), 0.05/(C**0.5))
            self.W_v.weight.data.uniform_(-0.5/(C**0.5), 0.5/(C**0.5))
            self.W_o.weight.data.zero_()

            self.k_scale_linear = torch.nn.Linear(config.d_model, self.H, bias=True)
            self.v_scale_linear = torch.nn.Linear(config.d_model, self.H, bias=True)
            torch.nn.init.zeros_(self.k_scale_linear.weight)
            torch.nn.init.zeros_(self.k_scale_linear.bias)
            torch.nn.init.zeros_(self.v_scale_linear.weight)
            torch.nn.init.zeros_(self.v_scale_linear.bias)

            self.v_lora_simple = LoraSimple(name="v", d_model=config.d_model, d_lora=config.v0_mix_amt_lora, layer_id=layer_id)
            self.a_lora_simple = LoraSimple(name="a", d_model=config.d_model, d_lora=config.a_lora, layer_id=layer_id)
            self.d_lora_mlp = LoraMLP(name="d", config=config, d_lora=config.decay_lora, out_dim = config.d_model, layer_id=layer_id)

            self.lora_A_g = torch.nn.Linear(config.d_model, config.gate_lora, bias=False)
            torch.nn.init.zeros_(self.lora_A_g.weight)
            self.lora_B_g = torch.nn.Linear(config.gate_lora, config.d_model, bias=False)
            ortho_init(self.lora_B_g.weight, 0.1)

            self.out_group_norm = torch.nn.GroupNorm(config.n_heads, config.d_model, eps=64e-5)
            self.dropout = torch.nn.Dropout(p=config.dropout)

    @FunctionType
    def forward(self, in_BTC, v0_BTC, time_shift_select_BT, skip_BT):
        B, T, C = in_BTC.shape
        H, K = self.H, self.K

        x_BTC = self.layer_norm(in_BTC)
        x_shift_BTC = torch.gather(x_BTC, dim=1, index=time_shift_select_BT.unsqueeze(-1).expand(-1, -1, self.d_model))

        rkvdag_8BTC = torch.lerp(x_BTC.unsqueeze(0), x_shift_BTC.unsqueeze(0), self.rkvdag_lerp)
        r_BTC, k_BTC, v_BTC, d_BTC, a_BTC, g_BTC, k_scale_BTC, v_scale_BTC = rkvdag_8BTC.unbind(dim=0)
        r_BTC = self.W_r(r_BTC)
        k_BTC = self.W_k(k_BTC)
        k_scale_BTH = torch.nn.functional.sigmoid(self.k_scale_linear(k_scale_BTC))
        v_scale_BTH = torch.nn.functional.sigmoid(self.v_scale_linear(v_scale_BTC))

        if self.layer_id == 0:
            v_BTC = self.W_v(v_BTC)
            v0_BTC = v_BTC
        else:
            v_lerp_BTC = torch.nn.functional.sigmoid(self.v_lora_simple(v_BTC))
            v_BTC = torch.lerp(self.W_v(v_BTC), v0_BTC, v_lerp_BTC)

        a_BTC = torch.nn.functional.sigmoid(self.a_lora_simple(a_BTC))
        g_BTC = self.lora_B_g(torch.nn.functional.sigmoid(self.lora_A_g(g_BTC)))

        _d_BTC = -0.5 - torch.nn.functional.softplus(-self.d_lora_mlp(d_BTC))
        w_BTC = torch.exp(-torch.exp(_d_BTC.float()))

        k_BTHK = k_scale_BTH.unsqueeze(-1) * torch.nn.functional.normalize(k_BTC.view(B, T, H, K), dim=-1, p=2.0)
        r_BTHK = r_BTC.view(B, T, H, K)
        v_BTHK = v_scale_BTH.unsqueeze(-1) * torch.nn.functional.normalize(v_BTC.view(B, T, H, K), dim=-1, p=2.0)
        w_BTHK = w_BTC.view(B, T, H, K)
        a_BTHK = a_BTC.view(B, T, H, K)
        k_deformed_BTHK = k_BTHK
        k_BTHK = k_BTHK * a_BTHK

        if r_BTHK.is_cuda:
            out_BTHK = RWKV7_WKV.apply(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT)
        else:
            out_BTHK = reference_rwkv7(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT)

        out_BTC = self.out_group_norm(out_BTHK.view(B*T, C)).view(B, T, C)
        bonus_BTC = ((r_BTHK * self.bonus * k_BTHK).sum(dim=-1, keepdim=True) * v_BTHK).view(B, T, C)
        out_BTC = self.W_o(g_BTC * (out_BTC + bonus_BTC))
        return in_BTC + self.dropout(out_BTC), v0_BTC