import json
import os
from dataclasses import dataclass
from typing import Tuple

import torch
from tqdm import tqdm


@dataclass
class SVDSpec:
    model_name: str
    n_layers: int
    n_heads: int
    d_head: int
    d_model: int
    r: int  # should be d_head+1


def _svd_filepaths(cache_dir: str, l: int, h: int):
    base = os.path.join(cache_dir, "svd")
    os.makedirs(base, exist_ok=True)
    U_path = os.path.join(base, f"l{l}_h{h}_U.pt")
    S_path = os.path.join(base, f"l{l}_h{h}_S.pt")
    Vh_path = os.path.join(base, f"l{l}_h{h}_Vh.pt")
    return U_path, S_path, Vh_path


def svd_exists(cache_dir: str, spec: SVDSpec) -> bool:
    meta_path = os.path.join(cache_dir, "svd_meta.json")
    if not os.path.exists(meta_path):
        return False
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return (
            meta.get("model_name") == spec.model_name
            and meta.get("n_layers") == spec.n_layers
            and meta.get("n_heads") == spec.n_heads
            and meta.get("d_head") == spec.d_head
            and meta.get("d_model") == spec.d_model
            and meta.get("r") == spec.r
        )
    except Exception:
        return False


@torch.no_grad()
def precompute_svd(cache_dir: str, teacher_model, force: bool = False) -> SVDSpec:
    """
    For each (layer, head), build A_aug by stacking W_O and (b_O/n_heads) as final row, then SVD.

    A_aug[l,h] shape: [d_head+1, d_model]
    U:  [d_head+1, r]
    S:  [r]
    Vh: [r, d_model]
    where r=d_head+1 (full_matrices=False).
    """
    os.makedirs(cache_dir, exist_ok=True)

    cfg = teacher_model.cfg
    spec = SVDSpec(
        model_name="gpt2",
        n_layers=int(cfg.n_layers),
        n_heads=int(cfg.n_heads),
        d_head=int(cfg.d_head),
        d_model=int(cfg.d_model),
        r=int(cfg.d_head + 1),
    )

    meta_path = os.path.join(cache_dir, "svd_meta.json")
    if (not force) and svd_exists(cache_dir, spec):
        return spec

    # Compute on CPU for simplicity.
    for l in tqdm(range(spec.n_layers), desc="SVD layers"):
        W_O = teacher_model.blocks[l].attn.W_O.detach().cpu().float()  # [h, d_head, d_model]
        b_O = teacher_model.blocks[l].attn.b_O.detach().cpu().float()  # [d_model]
        bias_per_head = b_O / spec.n_heads  # [d_model]

        for h in range(spec.n_heads):
            A_base = W_O[h]  # [d_head, d_model]
            A_aug = torch.cat([A_base, bias_per_head[None, :]], dim=0)  # [d_head+1, d_model]

            U, S, Vh = torch.linalg.svd(A_aug, full_matrices=False)

            U_path, S_path, Vh_path = _svd_filepaths(cache_dir, l, h)
            torch.save(U.contiguous(), U_path)
            torch.save(S.contiguous(), S_path)
            torch.save(Vh.contiguous(), Vh_path)

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(spec.__dict__, f, indent=2)

    return spec


def load_all_svd(cache_dir: str, spec: SVDSpec, device: torch.device, dtype: torch.dtype):
    """
    Loads all U,S,Vh into big tensors:

      U  : [n_layers, n_heads, d_head+1, r]
      S  : [n_layers, n_heads, r]
      Vh : [n_layers, n_heads, r, d_model]
    """
    U_all = torch.empty((spec.n_layers, spec.n_heads, spec.d_head + 1, spec.r), dtype=torch.float32, device="cpu")
    S_all = torch.empty((spec.n_layers, spec.n_heads, spec.r), dtype=torch.float32, device="cpu")
    Vh_all = torch.empty((spec.n_layers, spec.n_heads, spec.r, spec.d_model), dtype=torch.float32, device="cpu")

    for l in range(spec.n_layers):
        for h in range(spec.n_heads):
            U_path, S_path, Vh_path = _svd_filepaths(cache_dir, l, h)
            U = torch.load(U_path, map_location="cpu").float()
            S = torch.load(S_path, map_location="cpu").float()
            Vh = torch.load(Vh_path, map_location="cpu").float()

            # Safety checks
            if U.shape != (spec.d_head + 1, spec.r):
                raise ValueError(f"Bad U shape {U.shape} at l={l},h={h}")
            if S.shape != (spec.r,):
                raise ValueError(f"Bad S shape {S.shape} at l={l},h={h}")
            if Vh.shape != (spec.r, spec.d_model):
                raise ValueError(f"Bad Vh shape {Vh.shape} at l={l},h={h}")

            U_all[l, h] = U
            S_all[l, h] = S
            Vh_all[l, h] = Vh

    # Move to device/dtype for compute
    return (
        U_all.to(device=device, dtype=dtype),
        S_all.to(device=device, dtype=dtype),
        Vh_all.to(device=device, dtype=dtype),
    )
