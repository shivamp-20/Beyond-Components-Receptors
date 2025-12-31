"""OV mask parameters and masked forward hooks.

We implement the paired clean/corrupt + complementary mixing on OV SVD directions:

alpha_clean = z_clean_aug @ U
alpha_corr  = z_corr_aug  @ U

alpha_mix = alpha_clean * (S * m) + alpha_corr * (S * (1 - m))

result_masked = alpha_mix @ Vh

We override `blocks.<l>.attn.hook_result` to return result_masked.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from ov_svd import SVDBank


class MaskParams(nn.Module):
    """Trainable mask logits for all (layer, head, direction)."""

    def __init__(self, n_layers: int, n_heads: int, r_max: int, valid: torch.Tensor, init_m: float):
        super().__init__()
        # logits so that sigmoid(logits) == init_m
        init_m = float(init_m)
        init_m = min(max(init_m, 1e-4), 1 - 1e-4)
        init_logit = torch.log(torch.tensor(init_m) / (1 - torch.tensor(init_m))).item()

        self.mask_logits = nn.Parameter(torch.full((n_layers, n_heads, r_max), init_logit, dtype=torch.float32))
        self.register_buffer("valid", valid.bool())

        # For invalid dims, push logits very negative so sigmoid ~ 0
        with torch.no_grad():
            self.mask_logits[~self.valid] = -20.0

    def m(self) -> torch.Tensor:
        """Return masks in (0,1), shape [n_layers, n_heads, r_max]."""
        m = torch.sigmoid(self.mask_logits)
        # ensure invalid dims are exactly 0
        return m * self.valid.float()

    def sparsity(self) -> torch.Tensor:
        """Mean mask value over valid dims."""
        m = self.m()
        denom = self.valid.float().sum().clamp_min(1.0)
        return m.sum() / denom


def make_masked_forward_hooks(
    *,
    svd: SVDBank,
    mask_params: MaskParams,
    corrupt_cache: Dict[str, torch.Tensor],
    device: torch.device,
) -> List[Tuple[str, callable]]:
    """Create forward hooks for a single masked clean pass.

    We need:
      - hook_z: store clean z per layer
      - hook_result: replace per-head result using mixed OV directions
    """
    U = svd.U.to(device)
    S = svd.S.to(device)
    Vh = svd.Vh.to(device)
    valid = svd.valid.to(device)

    z_clean_by_layer: Dict[int, torch.Tensor] = {}

    def save_z(z: torch.Tensor, hook, layer: int):
        # z: [batch, pos, head, d_head]
        z_clean_by_layer[layer] = z
        return z

    def replace_result(result: torch.Tensor, hook, layer: int):
        # result: [batch, pos, head, d_model]
        z_clean = z_clean_by_layer[layer]  # [b,p,h,d_head]
        z_corr = corrupt_cache[f"blocks.{layer}.attn.hook_z"]  # [b,p,h,d_head], already aligned via padding

        # Augment if needed
        if svd.use_bias_aug:
            ones = torch.ones(z_clean.shape[:-1] + (1,), device=device, dtype=z_clean.dtype)
            z_clean_aug = torch.cat([z_clean, ones], dim=-1)  # [b,p,h,d_head+1]
            z_corr_aug = torch.cat([z_corr, ones], dim=-1)
        else:
            z_clean_aug = z_clean
            z_corr_aug = z_corr

        # alpha_* : [b,p,h,r_max]
        alpha_clean = torch.einsum("b p h d, h d r -> b p h r", z_clean_aug, U[layer])
        alpha_corr = torch.einsum("b p h d, h d r -> b p h r", z_corr_aug, U[layer])

        m = mask_params.m()[layer].to(device)  # [h,r]
        Sm = S[layer] * m
        Scomp = S[layer] * (1.0 - m)

        alpha_mix = alpha_clean * Sm + alpha_corr * Scomp

        # result_masked: [b,p,h,d_model]
        result_masked = torch.einsum("b p h r, h r m -> b p h m", alpha_mix, Vh[layer])

        # In case of numerical weirdness on invalid dims, enforce valid mask by zeroing unused directions.
        # (Normally already handled because U/S/Vh are zeros there.)
        return result_masked

    hooks: List[Tuple[str, callable]] = []
    for l in range(U.shape[0]):
        hooks.append((f"blocks.{l}.attn.hook_z", lambda z, hook, layer=l: save_z(z, hook, layer)))
        hooks.append((f"blocks.{l}.attn.hook_result", lambda r, hook, layer=l: replace_result(r, hook, layer)))
    return hooks
