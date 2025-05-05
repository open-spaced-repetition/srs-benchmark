import torch
from torch import Tensor

class RWKV7_WKV(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *inputs):
        r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT = inputs
        assert all(i.is_contiguous() for i in [r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT])
        # assert all(not torch.isnan(i).any() for i in [r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT])
        assert w_BTHK.dtype == torch.float32
        assert skip_BT.dtype == torch.bool
        dtype = r_BTHK.dtype
        assert all(i.dtype == dtype for i in [r_BTHK, k_BTHK, v_BTHK, a_BTHK, k_deformed_BTHK])
        if r_BTHK.is_cuda:
            if r_BTHK.dtype == torch.bfloat16:
                out, state_checkpoints = torch.ops.rwkv.rwkv7_wkv_forward_bfloat16.default(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT)
            elif r_BTHK.dtype == torch.float:
                out, state_checkpoints = torch.ops.rwkv.rwkv7_wkv_forward_float.default(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT)
            elif r_BTHK.dtype == torch.half:
                out, state_checkpoints = torch.ops.rwkv.rwkv7_wkv_forward_half.default(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT)
            
            ctx.save_for_backward(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, state_checkpoints)
            return out
        else:
            raise ValueError("Not supported. TODO")
            # return reference_rwkv7(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK)

    @staticmethod
    def backward(ctx, grad_BTHK):
        r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, state_checkpoints = ctx.saved_tensors
        if r_BTHK.dtype == torch.bfloat16:
            r_grad, k_grad, v_grad, w_grad, a_grad, k_deformed_grad = torch.ops.rwkv.rwkv7_wkv_backward_bfloat16.default(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, state_checkpoints, grad_BTHK)
        elif r_BTHK.dtype == torch.float:
            r_grad, k_grad, v_grad, w_grad, a_grad, k_deformed_grad = torch.ops.rwkv.rwkv7_wkv_backward_float.default(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, state_checkpoints, grad_BTHK)
        elif r_BTHK.dtype == torch.half:
            r_grad, k_grad, v_grad, w_grad, a_grad, k_deformed_grad = torch.ops.rwkv.rwkv7_wkv_backward_half.default(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT, state_checkpoints, grad_BTHK)
        return r_grad, k_grad, v_grad, w_grad, a_grad, k_deformed_grad, None

# Unused reference code for backpropagation for RWKV-7 wkv.
def reference_backward(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, state_checkpoints, grad_BTHK):
    B, T, H, K = r_BTHK.shape

    with torch.no_grad():
        # compute all the states. For this example we don't need to use the checkpoints since we don't care about memory usage.
        states_BTHKK = torch.zeros(B, T, H, K, K, dtype=r_BTHK.dtype, device=r_BTHK.device)
        state_BHKK = torch.zeros(B, H, K, K, dtype=r_BTHK.dtype, device=r_BTHK.device)
        for t in range(T):
            _, state_BHKK = single_timestep(r_BTHK[:, t],
                                            k_BTHK[:, t],
                                            v_BTHK[:, t],
                                            w_BTHK[:, t],
                                            a_BTHK[:, t],
                                            k_deformed_BTHK[:, t],
                                            state_BHKK)
            states_BTHKK[:, t] = state_BHKK.detach()

        # r_grad_BTHK = torch.empty(B, T, H, K, dtype=r_BTHK.dtype, device=r_BTHK.device)
        grad_BTHK1 = grad_BTHK.unsqueeze(-1)
        r_BTHK1 = r_BTHK.unsqueeze(-1)
        v_BTHK1 = v_BTHK.unsqueeze(-1)
        k_BTHK1 = k_BTHK.unsqueeze(-1)
        # w_BTHK1 = w_BTHK.unsqueeze(-1)
        w_diag_BTHKK = w_BTHK.diag_embed()
        a_BTHK1 = a_BTHK.unsqueeze(-1)
        k_deformed_BTHK1 = k_deformed_BTHK.unsqueeze(-1)
        # r_grad_BTHK1 = k_BTHK1 @ v_BTHK1.mT @ grad_BTHK1
        r_grad_BTHK1 = states_BTHKK.mT @ grad_BTHK1
        r_grad_BTHK = r_grad_BTHK1.squeeze(-1)

        dS_BTHKK = grad_BTHK1 @ r_BTHK1.mT
        scale_BTHKK = w_diag_BTHKK - k_deformed_BTHK1 @ (a_BTHK1 * k_deformed_BTHK1).mT
        v_grad_BTHK = torch.empty_like(r_grad_BTHK)
        k_grad_BTHK = torch.empty_like(r_grad_BTHK)
        w_grad_BTHK = torch.empty_like(r_grad_BTHK)
        a_grad_BTHK = torch.empty_like(r_grad_BTHK)
        k_deformed_grad_BTHK = torch.empty_like(r_grad_BTHK)
        for t in reversed(range(T)):
            v_grad_BTHK[:, t] = (dS_BTHKK[:, t] @ k_BTHK1[:, t]).squeeze(-1)
            k_grad_BTHK[:, t] = (dS_BTHKK[:, t].mT @ v_BTHK1[:, t]).squeeze(-1)

            # derivative wrt diag(w) - k_def a k_def^T
            if t > 0:
                # We can avoid a full matrix multiply by going back to the definition
                grad_decay_remove_BHKK = states_BTHKK[:, t-1].mT @ dS_BTHKK[:, t]

                a_grad_BTHK[:, t] = -((grad_decay_remove_BHKK.mT @ k_deformed_BTHK1[:, t]) * k_deformed_BTHK1[:, t]).squeeze(-1)
                k_deformed_grad_BTHK[:, t] = -(grad_decay_remove_BHKK @ (a_BTHK1[:, t] * k_deformed_BTHK1[:, t])).squeeze(-1)
                # for the dot product, do it directly with grad_decay (broadcast into rows)
                k_deformed_grad_BTHK[:, t] -= (a_BTHK1[:, t] * (grad_decay_remove_BHKK.mT @ k_deformed_BTHK1[:, t])).squeeze(-1)
                w_grad_BTHK[:, t] = grad_decay_remove_BHKK.diagonal(dim1=-2, dim2=-1)
            else:
                w_grad_BTHK[:, t] = 0
                a_grad_BTHK[:, t] = 0
                k_deformed_grad_BTHK[:, t] = 0

            # find the contribution to the t-1's S gradient
            dS_t_BHKK = dS_BTHKK[:, t]
            if t > 0:
                bonus_dS_BHKK = dS_t_BHKK @ scale_BTHKK[:, t].mT
                dS_BTHKK[:, t-1] += bonus_dS_BHKK

    return r_grad_BTHK, k_grad_BTHK, v_grad_BTHK, w_grad_BTHK, a_grad_BTHK, k_deformed_grad_BTHK

def single_timestep(r_BHK, k_BHK, v_BHK, w_BHK, a_BHK, k_deformed_BHK, state_BHKK):
    r_BHK1 = r_BHK.unsqueeze(-1)
    k_BHK1 = k_BHK.unsqueeze(-1)
    v_BHK1 = v_BHK.unsqueeze(-1)
    w_BHK1 = w_BHK.unsqueeze(-1)
    a_BHK1 = a_BHK.unsqueeze(-1)
    k_deformed_BHK1 = k_deformed_BHK.unsqueeze(-1)

    # Uses broadcasting. Remember that each column in vk_skate gets its own decay.
    state_BHKK = state_BHKK * w_BHK1.mT - state_BHKK @ k_deformed_BHK1 @ (a_BHK1 * k_deformed_BHK1).mT
    state_BHKK = state_BHKK + (v_BHK1 @ k_BHK1.mT)

    # Now we have a new updated S. We evaluate it at r and return the output.
    out_BHK1 = state_BHKK @ r_BHK1
    return out_BHK1.squeeze(-1), state_BHKK

def reference_rwkv7(r_BTHK, k_BTHK, v_BTHK, w_BTHK, a_BTHK, k_deformed_BTHK, skip_BT):
    r_BTHK = r_BTHK.float()
    k_BTHK = k_BTHK.float()
    v_BTHK = v_BTHK.float()
    w_BTHK = w_BTHK.float()
    a_BTHK = a_BTHK.float()
    k_deformed_BTHK = k_deformed_BTHK.float()
    skip_BT111 = skip_BT.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    B, T, H, K = r_BTHK.shape
    out_BTHK = torch.empty(B, T, H, K, dtype=torch.float32, device=r_BTHK.device)
    state_BHKK = torch.zeros(B, H, K, K, dtype=torch.float32, device=r_BTHK.device)
    for t in range(T):
        out_BTHK[:, t], next_state_BHKK = single_timestep( r_BTHK[:, t],
                                                k_BTHK[:, t],
                                                v_BTHK[:, t],
                                                w_BTHK[:, t],
                                                a_BTHK[:, t],
                                                k_deformed_BTHK[:, t],
                                                state_BHKK)
        skip_B111 = skip_BT111[:, t]
        state_BHKK = torch.where(skip_B111, state_BHKK, next_state_BHKK)
    return out_BTHK.to(k_BTHK.dtype)