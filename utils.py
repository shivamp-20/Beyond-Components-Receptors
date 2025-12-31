"""Small utilities: seeding, padding, KL loss, metrics."""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def left_pad(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """Left-pad a batch of variable-length sequences.

    Returns:
      tokens: [batch, max_len]
      attn_mask: [batch, max_len] (1 where real tokens, 0 where pad)
    """
    max_len = max(len(s) for s in seqs)
    batch = len(seqs)
    tokens = torch.full((batch, max_len), pad_id, dtype=torch.long)
    mask = torch.zeros((batch, max_len), dtype=torch.long)
    for i, s in enumerate(seqs):
        L = len(s)
        tokens[i, max_len - L :] = torch.tensor(s, dtype=torch.long)
        mask[i, max_len - L :] = 1
    return tokens, mask


def gather_logits_at_positions(logits: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    """Gather logits[b, positions[b]] -> [batch, vocab]."""
    bsz = logits.shape[0]
    return logits[torch.arange(bsz, device=logits.device), positions]


def kl_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    """KL( softmax(p/T) || softmax(q/T) ) averaged over batch."""
    T = float(temperature)
    p = torch.softmax(p_logits / T, dim=-1)
    log_p = torch.log(p.clamp_min(1e-12))
    log_q = torch.log_softmax(q_logits / T, dim=-1)
    kl = torch.sum(p * (log_p - log_q), dim=-1).mean()
    return kl


def accuracy_from_logits(logits: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
    preds = torch.argmax(logits, dim=-1)
    return (preds == label_ids).float().mean()


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def _fmt_thresh(t: float) -> str:
    # 0.1 -> "0p1" (safe for JSON keys)
    s = f"{t}".replace(".", "p")
    return s


@torch.no_grad()
def mask_value_stats(
    m: torch.Tensor,
    valid: torch.Tensor,
    *,
    thresholds=(0.01, 0.1, 0.5, 0.9),
    quantiles=(0.01, 0.1, 0.5, 0.9, 0.99),
) -> Dict[str, float]:
    """
    Summarize the *distribution* of masks over valid singular directions.

    - frac_gt_0p9 ~ "kept directions" (almost fully clean)
    - frac_lt_0p1 ~ "mostly replaced by corrupt" (almost off)
    - quantiles show bimodality / collapse
    """
    m = m.detach()
    valid = valid.detach().bool()

    flat = m[valid].float().flatten()
    if flat.numel() == 0:
        return {"mask_n_valid": 0}

    stats: Dict[str, float] = {
        "mask_n_valid": float(flat.numel()),
        "mask_mean": float(flat.mean().item()),
        "mask_std": float(flat.std(unbiased=False).item()),
        "mask_min": float(flat.min().item()),
        "mask_max": float(flat.max().item()),
    }

    for t in thresholds:
        k = _fmt_thresh(t)
        stats[f"mask_frac_gt_{k}"] = float((flat > t).float().mean().item())
        stats[f"mask_frac_lt_{k}"] = float((flat < t).float().mean().item())

        # also store counts for the most important thresholds
        if t in (0.5, 0.9):
            stats[f"mask_count_gt_{k}"] = float((flat > t).sum().item())

    for q in quantiles:
        qq = int(round(q * 100))
        stats[f"mask_q{qq:02d}"] = float(torch.quantile(flat, q).item())

    return stats

@torch.no_grad()
def paper_full_sparsity(
    m: torch.Tensor,
    valid: torch.Tensor,
    *,
    total_directions: int,
    threshold: float = 1e-2,
) -> Dict[str, float]:
    """
    Simple 'full sparsity' version:
      S_full = 1 - n_active / N_total
    """
    rel = paper_relative_sparsity(m, valid, threshold=threshold)
    N_total = float(max(int(total_directions), 1))
    n_active = float(rel["n_active"])
    return {
        "thr": float(threshold),
        "N_total": float(total_directions),
        "n_active": n_active,
        "active_frac_full": float(n_active / N_total),
        "S_full": float(1.0 - (n_active / N_total)),
    }


@torch.no_grad()
def sparsity_measures_multi_threshold(
    m: torch.Tensor,
    valid: torch.Tensor,
    *,
    thresholds=(1e-2, 1e-1, 0.5, 0.9),
    total_directions: int | None = None,
) -> Dict[str, float]:
    """
    Convenience: compute S_rel (+ optionally S_full) at multiple thresholds.
    Keys use the string form of the threshold, e.g. '0.01', '0.1', '0.5', '0.9'.
    """
    out: Dict[str, float] = {}
    for thr in thresholds:
        rel = paper_relative_sparsity(m, valid, threshold=float(thr))
        k = str(float(thr))
        out[f"active_frac@{k}"] = float(rel["active_frac"])
        out[f"S_rel@{k}"] = float(rel["S_rel"])
        out[f"n_active@{k}"] = float(rel["n_active"])
        out["N_learnable"] = float(rel["N_learnable"])  # same across thresholds

        if total_directions is not None:
            full = paper_full_sparsity(m, valid, total_directions=total_directions, threshold=float(thr))
            out[f"active_frac_full@{k}"] = float(full["active_frac_full"])
            out[f"S_full@{k}"] = float(full["S_full"])
            out["N_total"] = float(full["N_total"])
    return out
