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
