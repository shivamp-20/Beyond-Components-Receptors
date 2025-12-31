"""Optional: save 'logit receptors' for OV directions (project Vh rows through unembedding).

Receptor direction:
  v = Vh[row]  (d_model,)
Project to vocab:
  scores = v @ W_U  (d_vocab,)
Save top-k and bottom-k tokens as JSON.

This is optional and controlled by config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import torch

from ov_svd import SVDBank
from utils import save_json


def _get_WU(model) -> torch.Tensor:
    if hasattr(model, "W_U"):
        return model.W_U
    if hasattr(model, "unembed") and hasattr(model.unembed, "W_U"):
        return model.unembed.W_U
    raise AttributeError("Could not find unembedding matrix W_U in model.")


@torch.no_grad()
def save_receptors(
    *,
    model,
    svd: SVDBank,
    masks: torch.Tensor,   # [n_layers, n_heads, r_max]
    out_dir: Path,
    topk: int = 20,
    mask_threshold: float = 0.5,
    logger,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    W_U = _get_WU(model).detach().float().cpu()  # [d_model, d_vocab]
    tokenizer = model.tokenizer

    n_layers, n_heads, r_max = masks.shape
    Vh = svd.Vh.detach().float().cpu()  # [n_layers, n_heads, r_max, d_model]
    valid = svd.valid.detach().cpu()

    saved = 0
    for l in range(n_layers):
        for h in range(n_heads):
            for i in range(r_max):
                if not bool(valid[l, h, i]):
                    continue
                if float(masks[l, h, i].item()) < float(mask_threshold):
                    continue

                v = Vh[l, h, i, :]  # [d_model]
                scores = torch.matmul(v, W_U)  # [d_vocab]
                top_vals, top_idx = torch.topk(scores, k=topk)
                bot_vals, bot_idx = torch.topk(-scores, k=topk)
                bot_vals = -bot_vals

                def tok_str(tid: int) -> str:
                    return tokenizer.decode([int(tid)])

                obj = {
                    "layer": int(l),
                    "head": int(h),
                    "direction": int(i),
                    "mask": float(masks[l, h, i].item()),
                    "top": [{"token_id": int(t), "token": tok_str(t), "score": float(s)} for t, s in zip(top_idx.tolist(), top_vals.tolist())],
                    "bottom": [{"token_id": int(t), "token": tok_str(t), "score": float(s)} for t, s in zip(bot_idx.tolist(), bot_vals.tolist())],
                }
                save_json(out_dir / f"l{l}_h{h}_i{i}.json", obj)
                saved += 1

    logger.info(f"[Receptors] Saved {saved} receptor JSON files to {out_dir}")
