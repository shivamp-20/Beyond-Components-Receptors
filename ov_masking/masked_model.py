from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


@dataclass
class MaskConfig:
    init_theta: float = 4.0  # spec default


class OVMaskParams(nn.Module):
    """
    Learnable mask logits theta[l,h,i] over singular directions i (r = d_head+1).
    m = sigmoid(theta) in (0,1)
    """
    def __init__(self, n_layers: int, n_heads: int, r: int, init_theta: float):
        super().__init__()
        theta = torch.full((n_layers, n_heads, r), float(init_theta))
        self.theta = nn.Parameter(theta)

    def mask_values(self) -> torch.Tensor:
        return torch.sigmoid(self.theta)  # [L,H,R]

    def l1_mean(self) -> torch.Tensor:
        return self.mask_values().mean()


class OVMaskedRunner:
    """
    Runs a HookedTransformer forward with OV head projections replaced by
    U diag(m*S) Vh where (U,S,Vh) are SVD of augmented A_aug=[W_O; b_O/n_heads].
    We do this by:
      - capturing hook_z (z per head)
      - replacing hook_result with our masked result
    IMPORTANT:
      - student model must have attn.b_O = 0 to avoid double-counting bias,
        since bias is folded into A_aug via the appended ones dimension.
    """
    def __init__(
        self,
        *,
        model,
        svd_tensors: Dict[str, torch.Tensor],  # U,S,Vh on device
        mask_params: OVMaskParams,
    ):
        self.model = model
        self.U = svd_tensors["U"]   # [L,H,d_aug,R]
        self.S = svd_tensors["S"]   # [L,H,R]
        self.Vh = svd_tensors["Vh"] # [L,H,R,d_model]
        self.mask_params = mask_params

        self.n_layers = self.U.shape[0]
        self.n_heads = self.U.shape[1]
        self.d_aug = self.U.shape[2]
        self.r = self.U.shape[3]

    def _build_hooks(self):
        """
        Build a stable list of hooks. We use a per-forward dict `z_store`.
        """
        z_store: Dict[int, torch.Tensor] = {}

        def make_hook_z(layer: int):
            def hook_z(z, hook):
                # z: [B, P, H, d_head]
                z_store[layer] = z
                return z
            return hook_z

        # def make_hook_result(layer: int):
        #     def hook_result(result, hook):
        #         # result: [B, P, H, d_model] (we will replace it)
        #         z = z_store.pop(layer)  # [B,P,H,d_head]
        #         B, P, H, d_head = z.shape
        #         device = z.device
        #         dtype = z.dtype

        #         ones = torch.ones((B, P, H, 1), device=device, dtype=dtype)
        #         z_aug = torch.cat([z, ones], dim=-1)  # [B,P,H,d_aug]

        #         # Get per-layer tensors
        #         U = self.U[layer]    # [H,d_aug,R]
        #         S = self.S[layer]    # [H,R]
        #         Vh = self.Vh[layer]  # [H,R,d_model]

        #         m = torch.sigmoid(self.mask_params.theta[layer])  # [H,R]
        #         S_mask = m * S  # [H,R]

        #         # Efficient: (z_aug @ U) * S_mask @ Vh
        #         # t: [B,P,H,R]
        #         t = torch.einsum("bphd,hdr->bphr", z_aug, U)
        #         t = t * S_mask.unsqueeze(0).unsqueeze(0)  # broadcast over B,P
        #         out = torch.einsum("bphr,hrm->bphm", t, Vh)
        #         return out
        #     return hook_result

        def make_hook_result(layer: int):
            def hook_result(result, hook):
                # result: [B, P, H, d_model]
                target_dtype = result.dtype
                device = result.device

                z = z_store.pop(layer)  # [B,P,H,d_head]
                z = z.to(device=device, dtype=target_dtype)

                B, P, H, d_head = z.shape

                ones = torch.ones((B, P, H, 1), device=device, dtype=target_dtype)
                z_aug = torch.cat([z, ones], dim=-1)  # [B,P,H,d_aug]

                # Per-layer tensors (cast to match result dtype)
                U = self.U[layer].to(dtype=target_dtype)
                S = self.S[layer].to(dtype=target_dtype)
                Vh = self.Vh[layer].to(dtype=target_dtype)

                # Mask values in same dtype
                m = torch.sigmoid(self.mask_params.theta[layer]).to(dtype=target_dtype)  # [H,R]
                S_mask = m * S  # [H,R]

                # (z_aug @ U) * S_mask @ Vh
                t = torch.einsum("bphd,hdr->bphr", z_aug, U)
                t = t * S_mask.unsqueeze(0).unsqueeze(0)
                out = torch.einsum("bphr,hrm->bphm", t, Vh)

                # Return same dtype as original result to avoid downstream dtype issues
                return out.to(dtype=target_dtype)
            return hook_result

        hooks = []
        for l in range(self.n_layers):
            hooks.append((f"blocks.{l}.attn.hook_z", make_hook_z(l)))
            hooks.append((f"blocks.{l}.attn.hook_result", make_hook_result(l)))
        return hooks

    def __call__(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns logits: [B, T, V]
        """
        hooks = self._build_hooks()
        logits = self.model.run_with_hooks(
            tokens,
            attention_mask=attention_mask,
            fwd_hooks=hooks,
            reset_hooks_end=True,
        )
        return logits
