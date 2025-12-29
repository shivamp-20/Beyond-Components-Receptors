from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch

from .utils import ensure_dir


@dataclass(frozen=True)
class SVDMeta:
    model_name: str
    n_layers: int
    n_heads: int
    d_model: int
    d_head: int
    r: int  # expected rank (= d_head+1)
    dtype: str


def _meta_path(cache_dir: Path) -> Path:
    return cache_dir / "svd_meta.json"


def build_model_id(model_name: str, n_layers: int, n_heads: int, d_model: int, d_head: int) -> str:
    # Simple stable ID; you can add hashing if you want.
    return f"{model_name}_L{n_layers}_H{n_heads}_D{d_model}_Dh{d_head}"


def get_svd_dir(cache_root: Path, model_id: str) -> Path:
    return cache_root / "svd" / model_id


def svd_file(cache_dir: Path, l: int, h: int, comp: str) -> Path:
    return cache_dir / f"l{l}_h{h}_{comp}.pt"


@torch.no_grad()
def maybe_compute_and_save_svd_cache(
    *,
    model,
    model_name: str,
    cache_root: Path,
    force_recompute: bool = False,
) -> Tuple[Path, SVDMeta]:
    """
    Precompute U,S,Vh for A_aug[l,h] = stack_rows(W_O[l,h], b_O[l]/n_heads).
    Saves:
      svd/l{l}_h{h}_U.pt
      svd/l{l}_h{h}_S.pt
      svd/l{l}_h{h}_Vh.pt
      svd_meta.json
    """
    cfg = model.cfg
    n_layers = cfg.n_layers
    n_heads = cfg.n_heads
    d_model = cfg.d_model
    d_head = cfg.d_head
    r = d_head + 1

    model_id = build_model_id(model_name, n_layers, n_heads, d_model, d_head)
    cache_dir = get_svd_dir(cache_root, model_id)
    ensure_dir(cache_dir)

    meta_fp = _meta_path(cache_dir)
    if meta_fp.exists() and not force_recompute:
        meta = json.loads(meta_fp.read_text(encoding="utf-8"))
        svd_meta = SVDMeta(**meta)
        # Basic sanity:
        if svd_meta.r != r:
            raise ValueError(f"SVD cache rank mismatch: expected r={r} got {svd_meta.r}")
        return cache_dir, svd_meta

    # Compute from model weights on CPU float32 for numerical stability.
    for l in range(n_layers):
        attn = model.blocks[l].attn
        W_O = attn.W_O.detach().cpu().float()  # [n_heads, d_head, d_model]
        b_O = attn.b_O.detach().cpu().float()  # [d_model]
        bias_per_head = (b_O / n_heads).unsqueeze(0)  # [1, d_model]

        for h in range(n_heads):
            A_base = W_O[h]  # [d_head, d_model]
            A_aug = torch.cat([A_base, bias_per_head], dim=0)  # [d_head+1, d_model]

            # full_matrices=False => U:[d_aug,r], S:[r], Vh:[r,d_model] where r=d_aug
            U, S, Vh = torch.linalg.svd(A_aug, full_matrices=False)

            torch.save(U.contiguous(), svd_file(cache_dir, l, h, "U"))
            torch.save(S.contiguous(), svd_file(cache_dir, l, h, "S"))
            torch.save(Vh.contiguous(), svd_file(cache_dir, l, h, "Vh"))

    svd_meta = SVDMeta(
        model_name=model_name,
        n_layers=n_layers,
        n_heads=n_heads,
        d_model=d_model,
        d_head=d_head,
        r=r,
        dtype="float32",
    )
    meta_fp.write_text(json.dumps(svd_meta.__dict__, indent=2, sort_keys=True), encoding="utf-8")
    return cache_dir, svd_meta


@torch.no_grad()
def load_svd_tensors(
    cache_dir: Path,
    n_layers: int,
    n_heads: int,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, torch.Tensor]:
    """
    Load cached U,S,Vh into contiguous tensors:
      U:  [n_layers, n_heads, d_aug, r]
      S:  [n_layers, n_heads, r]
      Vh: [n_layers, n_heads, r, d_model]
    """
    meta = json.loads(_meta_path(cache_dir).read_text(encoding="utf-8"))
    r = int(meta["r"])
    d_head = int(meta["d_head"])
    d_model = int(meta["d_model"])
    d_aug = d_head + 1

    U_all = torch.empty((n_layers, n_heads, d_aug, r), dtype=dtype, device=device)
    S_all = torch.empty((n_layers, n_heads, r), dtype=dtype, device=device)
    Vh_all = torch.empty((n_layers, n_heads, r, d_model), dtype=dtype, device=device)

    for l in range(n_layers):
        for h in range(n_heads):
            U = torch.load(svd_file(cache_dir, l, h, "U"), map_location="cpu").to(dtype=dtype, device=device)
            S = torch.load(svd_file(cache_dir, l, h, "S"), map_location="cpu").to(dtype=dtype, device=device)
            Vh = torch.load(svd_file(cache_dir, l, h, "Vh"), map_location="cpu").to(dtype=dtype, device=device)
            U_all[l, h].copy_(U)
            S_all[l, h].copy_(S)
            Vh_all[l, h].copy_(Vh)

    return {"U": U_all, "S": S_all, "Vh": Vh_all}
