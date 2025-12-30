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
      - capturing hook_z (z per head) at blocks.{l}.attn.hook_z
      - replacing hook_result (per-head contribution after W_O) at blocks.{l}.attn.hook_result

    IMPORTANT:
      - student model must have attn.b_O = 0 to avoid double-counting bias,
        since bias is folded into A_aug via the appended ones dimension.
    """

    def __init__(
        self,
        *,
        model,
        svd_tensors: Dict[str, torch.Tensor],  # expects keys: "U","S","Vh"
        mask_params: "OVMaskParams",
    ):
        self.model = model

        # Expected shapes:
        # U:  [L, H, r, r]
        # S:  [L, H, r]
        # Vh: [L, H, r, d_model]
        self.U = svd_tensors["U"]
        self.S = svd_tensors["S"]
        self.Vh = svd_tensors["Vh"]

        self.mask_params = mask_params

        self.n_layers = self.U.shape[0]
        self.n_heads = self.U.shape[1]
        self.r = self.S.shape[2]

        # Per-forward scratch (store z by layer)
        self._z_store: Dict[int, torch.Tensor] = {}

    def _build_hooks(self):
        hooks = []

        def make_hook_z(layer: int):
            def hook_z(z, hook):
                # z: [B, P, H, d_head]
                self._z_store[layer] = z
                return z
            return hook_z

        def make_hook_result(layer: int):
            def hook_result(result, hook):
                # result: [B, P, H, d_model]
                if layer not in self._z_store:
                    raise RuntimeError(
                        f"Missing stored z for layer {layer}. Did hook_z run?"
                    )

                out_dtype = result.dtype
                device = result.device

                # Fetch and clear stored z for this layer
                z = self._z_store.pop(layer)  # [B,P,H,d_head]

                # Do math in float32 for stability, cast back at end
                z = z.to(device=device, dtype=torch.float32)
                B, P, H, d_head = z.shape
                if H != self.n_heads:
                    raise RuntimeError(f"Head mismatch: got H={H}, expected {self.n_heads}")

                ones = torch.ones((B, P, H, 1), device=device, dtype=torch.float32)
                z_aug = torch.cat([z, ones], dim=-1)  # [B,P,H,r] where r=d_head+1

                # Per-layer SVD tensors
                U = self.U[layer].to(device=device, dtype=torch.float32)    # [H,r,r]
                S = self.S[layer].to(device=device, dtype=torch.float32)    # [H,r]
                Vh = self.Vh[layer].to(device=device, dtype=torch.float32)  # [H,r,d_model]

                # Mask values m in (0,1): [L,H,r]
                m = self.mask_params.mask_values().to(device=device, dtype=torch.float32)
                m_l = m[layer]               # [H,r]
                S_mask = S * m_l             # [H,r]

                # Compute: z_aug @ (U @ diag(S_mask) @ Vh)
                # Step 1: t = z_aug @ U  -> [B,P,H,r]
                t = torch.einsum("bphr,hru->bphu", z_aug, U)
                # Step 2: t *= S_mask
                t = t * S_mask.unsqueeze(0).unsqueeze(0)
                # Step 3: out = t @ Vh -> [B,P,H,d_model]
                out = torch.einsum("bphu,hum->bphm", t, Vh)

                return out.to(dtype=out_dtype)

            return hook_result

        for l in range(self.n_layers):
            hooks.append((f"blocks.{l}.attn.hook_z", make_hook_z(l)))
            hooks.append((f"blocks.{l}.attn.hook_result", make_hook_result(l)))

        return hooks

    def __call__(self, tokens: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Returns logits: [B, T, V]
        """
        self._z_store.clear()
        hooks = self._build_hooks()

        logits = self.model.run_with_hooks(
            tokens,
            attention_mask=attention_mask,
            return_type="logits",
            fwd_hooks=hooks,
            reset_hooks_end=True,
        )
        return logits
