from dataclasses import dataclass
from typing import Optional, Dict

import torch
import torch.nn as nn


@dataclass
class MaskedOVState:
    """
    Stores SVD factors for all layers/heads, plus learnable theta.
    """
    U: torch.Tensor    # [L, H, d_head+1, r]
    S: torch.Tensor    # [L, H, r]
    Vh: torch.Tensor   # [L, H, r, d_model]
    theta: nn.Parameter  # [L, H, r]


class MaskedOVHook(nn.Module):
    """
    Implements the masked OV projection using TransformerLens hooks:
      - store z at blocks.{l}.attn.hook_z
      - replace blocks.{l}.hook_attn_out

    Per spec:
      A_aug = [W_O; b_O/n_heads]
      SVD(A_aug) = U S Vh
      Mask: S_mask = sigmoid(theta) * S
      And compute:
        out_head = z_aug @ U @ diag(S_mask) @ Vh
      where z_aug = concat(z, ones)

    We do it without forming A_aug_mask explicitly:
      t = (z_aug @ U) * S_mask
      out_head = t @ Vh
    """

    def __init__(self, state: MaskedOVState):
        super().__init__()
        self.state = state
        self._z_cache: Dict[int, torch.Tensor] = {}  # layer -> z tensor

    def mask_values(self) -> torch.Tensor:
        return torch.sigmoid(self.state.theta)

    def l1_mean(self) -> torch.Tensor:
        return self.mask_values().mean()

    def scores_m_times_s(self) -> torch.Tensor:
        """
        For selected.json ranking: m_i * S_i
        """
        return self.mask_values() * self.state.S

    def hook_store_z(self, z: torch.Tensor, hook):
        # hook.name like "blocks.0.attn.hook_z"
        # layer index is after "blocks."
        name = hook.name
        layer = int(name.split(".")[1])
        self._z_cache[layer] = z
        return z

    def hook_replace_attn_out(self, attn_out: torch.Tensor, hook):
        # hook.name like "blocks.0.hook_attn_out"
        layer = int(hook.name.split(".")[1])
        if layer not in self._z_cache:
            # Should not happen if hook_z ran first
            return attn_out

        z = self._z_cache.pop(layer)  # [b, pos, H, d_head]
        U = self.state.U[layer]       # [H, d_head+1, r]
        S = self.state.S[layer]       # [H, r]
        Vh = self.state.Vh[layer]     # [H, r, d_model]
        theta = self.state.theta[layer]  # [H, r]

        bsz, seq, n_heads, d_head = z.shape

        # z_aug = concat(z, ones) in last dim
        ones = torch.ones((bsz, seq, n_heads, 1), device=z.device, dtype=z.dtype)
        z_aug = torch.cat([z, ones], dim=-1)  # [b, pos, H, d_head+1]

        # compute in float32 for stability; then cast back
        z_aug_f = z_aug.float()
        U_f = U.float()
        S_f = S.float()
        Vh_f = Vh.float()

        m = torch.sigmoid(theta.float())     # [H, r]
        S_mask = m * S_f                     # [H, r]

        # t = einsum('bphe,her->bphr', z_aug, U)
        t = torch.einsum("bphe,her->bphr", z_aug_f, U_f)  # [b, pos, H, r]
        t = t * S_mask[None, None, :, :]                 # scale each head/direction
        # out_head = einsum('bphr,hrm->bphm', t, Vh)
        out_heads = torch.einsum("bphr,hrm->bphm", t, Vh_f)  # [b, pos, H, d_model]
        out = out_heads.sum(dim=2)  # sum heads -> [b, pos, d_model]

        return out.to(dtype=attn_out.dtype)


def freeze_model_params(model) -> None:
    for p in model.parameters():
        p.requires_grad_(False)


def build_mask_state(U: torch.Tensor, S: torch.Tensor, Vh: torch.Tensor, init_theta: float, device: torch.device) -> MaskedOVState:
    """
    theta initialized to +4.0 everywhere (sigmoid ~ 0.982).
    """
    L, H, _, r = U.shape
    theta = nn.Parameter(torch.full((L, H, r), float(init_theta), device=device, dtype=torch.float32))
    return MaskedOVState(U=U, S=S, Vh=Vh, theta=theta)
