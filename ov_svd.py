"""Per-head OV SVD factors with (optional) bias augmentation.

We compute an SVD per (layer, head) of an augmented W_O:

  W_O_aug has shape (d_head + 1, d_model)
    - rows 0..d_head-1: W_O[head]  (d_head, d_model)
    - last row: b_O_head (d_model,)   (optional, depending on bias_handling)

Then thin SVD:
  W_O_aug = U @ diag(S) @ Vh
We keep only singular values > svd_eps.

We save:
  U: (d_head+1, r), S: (r,), Vh: (r, d_model)
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Tuple

import torch


BiasHandling = Literal["auto", "none", "force_per_head"]


@dataclass
class SVDHead:
    U: torch.Tensor   # (d_head+1, r) or (d_head, r) if no augmentation
    S: torch.Tensor   # (r,)
    Vh: torch.Tensor  # (r, d_model)


@dataclass
class SVDBank:
    """All SVD factors for the model, padded to a common r_max."""
    U: torch.Tensor       # [n_layers, n_heads, d_head+1, r_max]  (or d_head if no aug)
    S: torch.Tensor       # [n_layers, n_heads, r_max]
    Vh: torch.Tensor      # [n_layers, n_heads, r_max, d_model]
    valid: torch.Tensor   # [n_layers, n_heads, r_max] bool
    use_bias_aug: bool
    bias_mode: str        # human-readable detected mode


def _get_attn_module(model, layer: int):
    return model.blocks[layer].attn


def _get_WO(model, layer: int) -> torch.Tensor:
    attn = _get_attn_module(model, layer)
    if hasattr(attn, "W_O"):
        return attn.W_O
    # fallbacks (older)
    if hasattr(model, "W_O"):
        return model.W_O[layer]
    raise AttributeError("Could not find W_O in model; check TransformerLens version.")


def _get_bO(model, layer: int) -> torch.Tensor:
    attn = _get_attn_module(model, layer)
    if hasattr(attn, "b_O") and attn.b_O is not None:
        return attn.b_O
    if hasattr(model, "b_O"):
        return model.b_O[layer]
    # Some models may have no b_O; return zeros.
    return torch.zeros(model.cfg.d_model, device=next(model.parameters()).device, dtype=next(model.parameters()).dtype)


@torch.no_grad()
def detect_hook_result_bias_mode(model, layer: int = 0) -> str:
    """Detect whether hook_result already includes an output bias share.

    Returns one of:
      - "no_bias_in_hook_result"
      - "per_head_bias_in_hook_result"
      - "unknown"

    This is a best-effort check (small numerical tolerances).
    """
    device = next(model.parameters()).device

    # Ensure hook_result exists
    if hasattr(model, "set_use_attn_result"):
        model.set_use_attn_result(True)
    model.cfg.use_attn_result = True

    try:
        tokens = model.to_tokens("Hello world", prepend_bos=model.cfg.default_prepend_bos).to(device)
    except TypeError:
        # Older TransformerLens versions may not accept prepend_bos kwarg here.
        tokens = model.to_tokens("Hello world").to(device)

    _, cache = model.run_with_cache(tokens, names_filter=lambda n: n.endswith("attn.hook_z") or n.endswith("attn.hook_result"))

    z = cache[f"blocks.{layer}.attn.hook_z"]              # [b, pos, head, d_head]
    result = cache[f"blocks.{layer}.attn.hook_result"]    # [b, pos, head, d_model]

    W_O = _get_WO(model, layer)                           # [head, d_head, d_model]
    pred = torch.einsum("b p h d, h d m -> b p h m", z, W_O)

    err_no_bias = (result - pred).abs().mean().item()
    if err_no_bias < 1e-4:
        return "no_bias_in_hook_result"

    b_O = _get_bO(model, layer)                           # [d_model]
    pred_per_head_bias = pred + (b_O / model.cfg.n_heads).view(1, 1, 1, -1)
    err_per_head = (result - pred_per_head_bias).abs().mean().item()
    if err_per_head < 1e-4:
        return "per_head_bias_in_hook_result"

    return "unknown"


@torch.no_grad()
def compute_or_load_svd_bank(
    *,
    model,
    svd_dir: Path,
    svd_eps: float,
    bias_handling: BiasHandling,
    logger,
) -> SVDBank:
    svd_dir.mkdir(parents=True, exist_ok=True)
    meta_path = svd_dir / "metadata.json"

    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("model_name") == model.cfg.model_name and float(meta.get("svd_eps")) == float(svd_eps):
            logger.info(f"[SVD] Found existing SVD bank at {svd_dir} (eps={svd_eps}). Loading...")
            U = torch.load(svd_dir / "U.pt", map_location="cpu")
            S = torch.load(svd_dir / "S.pt", map_location="cpu")
            Vh = torch.load(svd_dir / "Vh.pt", map_location="cpu")
            valid = torch.load(svd_dir / "valid.pt", map_location="cpu")
            return SVDBank(U=U, S=S, Vh=Vh, valid=valid, use_bias_aug=bool(meta["use_bias_aug"]), bias_mode=str(meta.get("bias_mode", "unknown")))

    # Decide bias augmentation
    bias_mode = detect_hook_result_bias_mode(model, layer=0)
    if bias_handling == "none":
        use_bias_aug = False
    elif bias_handling == "force_per_head":
        use_bias_aug = True
    else:
        # auto: use bias augmentation if hook_result appears to already be per-head biased
        # (else, using it might double-count downstream depending on implementation)
        use_bias_aug = (bias_mode == "per_head_bias_in_hook_result")

    logger.info(f"[SVD] bias_handling={bias_handling} -> use_bias_aug={use_bias_aug} (detected bias_mode={bias_mode})")

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head

    # r_max is at most d_head+1 (with augmentation) or d_head (without)
    d_in = d_head + 1 if use_bias_aug else d_head
    r_max = min(d_in, d_model)

    U_all = torch.zeros((n_layers, n_heads, d_in, r_max), dtype=torch.float32)
    S_all = torch.zeros((n_layers, n_heads, r_max), dtype=torch.float32)
    Vh_all = torch.zeros((n_layers, n_heads, r_max, d_model), dtype=torch.float32)
    valid = torch.zeros((n_layers, n_heads, r_max), dtype=torch.bool)

    for l in range(n_layers):
        W_O = _get_WO(model, l).detach().float().cpu()  # [head, d_head, d_model]
        b_O = _get_bO(model, l).detach().float().cpu()  # [d_model]
        for h in range(n_heads):
            W = W_O[h]  # (d_head, d_model)
            if use_bias_aug:
                W_aug = torch.cat([W, (b_O / n_heads).view(1, -1)], dim=0)  # (d_head+1, d_model)
            else:
                W_aug = W  # (d_head, d_model)

            # Thin SVD
            U, S, Vh = torch.linalg.svd(W_aug, full_matrices=False)  # U(d_in,k), S(k), Vh(k,d_model)
            keep = (S > svd_eps)
            r = int(keep.sum().item())
            if r == 0:
                # Extremely unlikely; keep the top singular vector
                r = 1
                keep = torch.zeros_like(S, dtype=torch.bool)
                keep[0] = True

            U = U[:, keep]
            S = S[keep]
            Vh = Vh[keep, :]

            U_all[l, h, :, :r] = U
            S_all[l, h, :r] = S
            Vh_all[l, h, :r, :] = Vh
            valid[l, h, :r] = True

    # Save
    torch.save(U_all, svd_dir / "U.pt")
    torch.save(S_all, svd_dir / "S.pt")
    torch.save(Vh_all, svd_dir / "Vh.pt")
    torch.save(valid, svd_dir / "valid.pt")
    meta = {
        "model_name": model.cfg.model_name,
        "svd_eps": float(svd_eps),
        "use_bias_aug": bool(use_bias_aug),
        "bias_mode": bias_mode,
        "note": "SVD of (possibly augmented) W_O per head, padded to r_max.",
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info(f"[SVD] Saved SVD bank to {svd_dir}")

    return SVDBank(U=U_all, S=S_all, Vh=Vh_all, valid=valid, use_bias_aug=use_bias_aug, bias_mode=bias_mode)
