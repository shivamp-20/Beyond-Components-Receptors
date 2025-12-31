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
def paper_relative_sparsity(
    m: torch.Tensor,
    valid: torch.Tensor,
    threshold: float = 1e-2,
) -> Dict[str, float]:
    """
    Paper App. B.6 'Relative Sparsity':
      n_active = count(m > 1e-2) over learnable directions
      S_rel = 1 - n_active / N_learnable
    """
    valid = valid.bool()
    flat = m[valid].float().flatten()
    if flat.numel() == 0:
        return {"N_learnable": 0.0, "n_active": 0.0, "active_frac": 0.0, "S_rel": 0.0, "thr": float(threshold)}

    n_learnable = float(flat.numel())
    n_active = float((flat > threshold).sum().item())
    active_frac = n_active / n_learnable
    s_rel = 1.0 - active_frac
    return {
        "thr": float(threshold),
        "N_learnable": n_learnable,
        "n_active": n_active,
        "active_frac": float(active_frac),
        "S_rel": float(s_rel),
    }

