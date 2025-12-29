from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .utils import ensure_dir, json_dump


def teacher_cache_paths(cache_dir: Path, task: str, split: str) -> Dict[str, Path]:
    """
    Spec filenames:
      teacher/{task}_{split}_clean_logp.pt
      teacher/{task}_{split}_corr_logp.pt
    """
    ensure_dir(cache_dir)
    return {
        "clean": cache_dir / f"{task}_{split}_clean_logp.pt",
        "corr": cache_dir / f"{task}_{split}_corr_logp.pt",
        "meta": cache_dir / f"{task}_{split}_meta.json",
    }


def _ensure_pad_token(tokenizer) -> None:
    # GPT-2 often has no pad token; safest is to use eos for padding.
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token


def tokenize_prompts(tokenizer, prompts: List[str], device: torch.device):
    """
    Returns:
      tokens: [B, T]
      attention_mask: [B, T] (1 for real tokens, 0 for padding)
      last_pos: [B] index of final non-padding token
    """
    _ensure_pad_token(tokenizer)
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    tokens = enc["input_ids"].to(device)
    attention_mask = enc["attention_mask"].to(device)
    last_pos = attention_mask.sum(dim=1) - 1
    if (last_pos < 0).any():
        raise ValueError("Found an empty prompt after tokenization.")
    return tokens, attention_mask, last_pos


@torch.no_grad()
def maybe_build_teacher_cache(
    *,
    model,
    tokenizer,
    dataloader: DataLoader,
    n_items: int,
    vocab_size: int,
    cache_paths: Dict[str, Path],
    device: torch.device,
    force_recompute: bool,
    desc_prefix: str,
) -> None:
    """
    Build teacher logp tensors (float16) for clean and corr prompts.
    Stored as shape [N, vocab].
    """
    if cache_paths["clean"].exists() and cache_paths["corr"].exists() and cache_paths["meta"].exists() and not force_recompute:
        return

    ensure_dir(cache_paths["clean"].parent)

    logp_clean = torch.empty((n_items, vocab_size), dtype=torch.float16, device="cpu")
    logp_corr = torch.empty((n_items, vocab_size), dtype=torch.float16, device="cpu")

    model.eval()

    for batch in tqdm(dataloader, desc=f"{desc_prefix}: teacher_cache", leave=False):
        idxs, prompts_clean, prompts_corr, *_ = batch
        idxs = idxs.tolist()

        # Clean
        tokens, attn_mask, last_pos = tokenize_prompts(tokenizer, list(prompts_clean), device=device)
        logits = model(tokens, attention_mask=attn_mask)  # [B,T,V]
        bsz = logits.size(0)
        logits_last = logits[torch.arange(bsz, device=device), last_pos]  # [B,V]
        logp = torch.log_softmax(logits_last.float(), dim=-1).to(dtype=torch.float16).cpu()
        logp_clean[idxs] = logp

        # Corrupt
        tokens, attn_mask, last_pos = tokenize_prompts(tokenizer, list(prompts_corr), device=device)
        logits = model(tokens, attention_mask=attn_mask)
        bsz = logits.size(0)
        logits_last = logits[torch.arange(bsz, device=device), last_pos]
        logp = torch.log_softmax(logits_last.float(), dim=-1).to(dtype=torch.float16).cpu()
        logp_corr[idxs] = logp

    torch.save(logp_clean.contiguous(), cache_paths["clean"])
    torch.save(logp_corr.contiguous(), cache_paths["corr"])

    meta = {
        "n_items": n_items,
        "vocab_size": vocab_size,
        "dtype": "float16",
    }
    cache_paths["meta"].write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")


def load_teacher_cache(cache_paths: Dict[str, Path]) -> Dict[str, torch.Tensor]:
    """
    Loads teacher cache to CPU tensors:
      clean: [N,V] float16
      corr:  [N,V] float16
    """
    clean = torch.load(cache_paths["clean"], map_location="cpu")
    corr = torch.load(cache_paths["corr"], map_location="cpu")
    return {"clean": clean, "corr": corr}
