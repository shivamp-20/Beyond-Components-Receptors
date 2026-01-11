#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
inspect_gp_next_token.py

Inspect next-token predictions for the GPTask:
- Shows top-k next-token predictions for ORIGINAL model
- Optionally compares to MASKED forward (requires outputs/gp/svd_cache.pt and outputs/gp/masks.pt)
- Probes Regime-A pronoun tokens: " he" and " she" (leading space tokens)

Usage:
  python inspect_gp_next_token.py --split test --n 30 --topk 10
  python inspect_gp_next_token.py --split test --n 30 --topk 10 --masked on
"""

from __future__ import annotations

import argparse
import os
import sys
import random
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import torch

# --- Make it easy to run from repo root or scripts/ ---
THIS_DIR = Path(__file__).resolve().parent
for p in [THIS_DIR, THIS_DIR.parent]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

try:
    import train_gp_masks_and_dump_ov_logit_receptors as gp
except Exception as e:
    raise RuntimeError(
        "Failed to import train_gp_masks_and_dump_ov_logit_receptors.py. "
        "Run this script from the repo root or ensure it is on PYTHONPATH."
    ) from e

from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer


def _decode_tok(tokenizer: GPT2TokenizerFast, tok_id: int) -> str:
    # Keep leading spaces visible
    return tokenizer.decode([int(tok_id)], clean_up_tokenization_spaces=False)


def _topk_list(tokenizer: GPT2TokenizerFast, probs_row: torch.Tensor, k: int):
    vals, idx = torch.topk(probs_row, k=k, largest=True)
    out = []
    for v, i in zip(vals, idx):
        out.append((_decode_tok(tokenizer, int(i)), int(i), float(v)))
    return out


def _inv_sigmoid(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    x = torch.clamp(x, eps, 1 - eps)
    return torch.log(x / (1 - x))


def _resolve_csv(data_dir: Path, split: str, csv_override: Optional[str]) -> Path:
    if csv_override is not None:
        p = Path(csv_override)
        if not p.exists():
            raise FileNotFoundError(f"--csv path not found: {p}")
        return p

    split_to_csv = {
        "train": "train_1k_gp.csv",
        "val": "val_gp.csv",
        "test": "test_gp.csv",
    }
    if split not in split_to_csv:
        raise ValueError(f"Unknown split={split}. Use train/val/test or pass --csv.")
    p = data_dir / split_to_csv[split]
    if not p.exists():
        raise FileNotFoundError(f"CSV for split='{split}' not found at: {p}")
    return p


@torch.no_grad()
def _compute_orig_probs(
    model: HookedTransformer,
    tokens_clean: torch.Tensor,
    last_idx_clean: torch.Tensor,
) -> torch.Tensor:
    logits_full = model(tokens_clean)  # (B,S,V)
    logits_last = logits_full[torch.arange(tokens_clean.size(0), device=tokens_clean.device), last_idx_clean, :]
    return torch.softmax(logits_last, dim=-1)


@torch.no_grad()
def _compute_masked_probs(
    model: HookedTransformer,
    tokens_clean: torch.Tensor,
    tokens_corr: torch.Tensor,
    last_idx_clean: torch.Tensor,
    svd_cache_path: Path,
    masks_path: Path,
    dtype_cache: torch.dtype = torch.float16,
) -> torch.Tensor:
    # Load SVD cache (qk, ov, mlp_in, mlp_out, rank_total_ov)
    qk, ov, mlp_in, mlp_out, _rank_total_ov = gp.load_svd_cache(str(svd_cache_path), device=str(tokens_clean.device))

    # Build mask bank
    masks_bank = gp.MaskBank(qk=qk, ov=ov, mlp_in=mlp_in, mlp_out=mlp_out).to(tokens_clean.device)
    masks_bank.eval()

    # Load saved masks (these are sigmoid(mask_raw) tensors)
    saved = torch.load(str(masks_path), map_location="cpu")
    with torch.no_grad():
        n_layers = len(saved["ov"])
        n_heads = len(saved["ov"][0])
        for l in range(n_layers):
            for h in range(n_heads):
                masks_bank.qk[l][h].copy_(_inv_sigmoid(saved["qk"][l][h]).to(tokens_clean.device))
                masks_bank.ov[l][h].copy_(_inv_sigmoid(saved["ov"][l][h]).to(tokens_clean.device))
            masks_bank.mlp_in[l].copy_(_inv_sigmoid(saved["mlp_in"][l]).to(tokens_clean.device))
            masks_bank.mlp_out[l].copy_(_inv_sigmoid(saved["mlp_out"][l]).to(tokens_clean.device))

    # Corrupted projections for complement terms
    zc_ov, zc_mlp_in, zc_mlp_out = gp.corrupt_projections(
        model=model,
        tokens=tokens_corr,
        qk=qk,
        ov=ov,
        mlp_in=mlp_in,
        mlp_out=mlp_out,
        dtype_cache=dtype_cache,
    )

    # Masked forward logits at last index
    logits_masked_last = gp.masked_forward_logits_last(
        model=model,
        tokens=tokens_clean,
        qk=qk,
        ov=ov,
        mlp_in=mlp_in,
        mlp_out=mlp_out,
        masks=masks_bank,
        z_corr_ov=zc_ov,
        z_corr_mlp_in=zc_mlp_in,
        z_corr_mlp_out=zc_mlp_out,
        last_indices=last_idx_clean,
    )
    return torch.softmax(logits_masked_last, dim=-1)


def inspect_example(
    tokenizer: GPT2TokenizerFast,
    rows_s: List[Dict[str, str]],
    tokens_clean: torch.Tensor,
    last_clean: torch.Tensor,
    probs_orig: torch.Tensor,
    he_id: int,
    she_id: int,
    i: int,
    topk: int,
    probs_masked: Optional[torch.Tensor] = None,
    print_full_prefix: bool = False,
) -> None:
    r = rows_s[i]
    print("=" * 110)
    print(f"i={i} | gold_pronoun={r.get('pronoun','')} | template={r.get('template','')} | name={r.get('name','')}")
    prefix = r["prefix"]
    corr_prefix = r["corr_prefix"]
    if not print_full_prefix:
        prefix_show = prefix if len(prefix) <= 160 else prefix[:160] + "…"
        corr_show = corr_prefix if len(corr_prefix) <= 160 else corr_prefix[:160] + "…"
    else:
        prefix_show = prefix
        corr_show = corr_prefix

    print("- prefix repr     :", repr(prefix_show))
    print("- corr_prefix repr:", repr(corr_show))

    li = int(last_clean[i].item())
    last_tok_id = int(tokens_clean[i, li].item())
    print(f"- last_idx_clean={li} | last_token={_decode_tok(tokenizer, last_tok_id)!r} | last_token_id={last_tok_id}")
    print()

    p = probs_orig[i]
    top1_id = int(torch.argmax(p).item())
    print(f"[ORIG] top1={_decode_tok(tokenizer, top1_id)!r} (id={top1_id}) p={float(p[top1_id]):.6f}")
    print(f"[ORIG] p(' he')={float(p[he_id]):.6f}  p(' she')={float(p[she_id]):.6f}  diff={float(p[he_id]-p[she_id]):.6f}")
    print("[ORIG] top-k next tokens:")
    for tok, tid, prob in _topk_list(tokenizer, p, topk):
        print(f"  {tok!r:>14}  id={tid:<6}  p={prob:.6f}")

    if probs_masked is not None:
        pm = probs_masked[i]
        top1m_id = int(torch.argmax(pm).item())
        print()
        print(f"[MASKED] top1={_decode_tok(tokenizer, top1m_id)!r} (id={top1m_id}) p={float(pm[top1m_id]):.6f}")
        print(f"[MASKED] p(' he')={float(pm[he_id]):.6f}  p(' she')={float(pm[she_id]):.6f}  diff={float(pm[he_id]-pm[she_id]):.6f}")
        print("[MASKED] top-k next tokens:")
        for tok, tid, prob in _topk_list(tokenizer, pm, topk):
            print(f"  {tok!r:>14}  id={tid:<6}  p={prob:.6f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data_main")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--csv", type=str, default=None, help="Optional path to a CSV to inspect.")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--out_dir", type=str, default="outputs/gp")
    parser.add_argument("--svd_cache", type=str, default=None)
    parser.add_argument("--masks", type=str, default=None)

    parser.add_argument("--masked", type=str, default="auto", choices=["auto", "on", "off"])
    parser.add_argument("--dtype_cache", type=str, default="float16", choices=["float16", "float32"])

    parser.add_argument("--device", type=str, default=None, help="cuda/cpu. Default: auto.")
    parser.add_argument("--verbose", action="store_true", help="Print detailed blocks for every sampled example.")
    parser.add_argument("--print_full_prefix", action="store_true", help="Print full prefix strings (not truncated).")

    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype_cache = torch.float16 if args.dtype_cache == "float16" else torch.float32

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    svd_cache_path = Path(args.svd_cache) if args.svd_cache else (out_dir / "svd_cache.pt")
    masks_path = Path(args.masks) if args.masks else (out_dir / "masks.pt")

    # Tokenizer/model
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()

    # Regime A: leading-space tokens for pronouns
    he_ids = tokenizer.encode(" he", add_special_tokens=False)
    she_ids = tokenizer.encode(" she", add_special_tokens=False)
    if len(he_ids) != 1 or len(she_ids) != 1:
        raise ValueError(f'" he"/" she" are not single tokens: he_ids={he_ids}, she_ids={she_ids}')
    he_id, she_id = he_ids[0], she_ids[0]
    print(f"[TOKENS] ' he' -> id={he_id}, decoded={_decode_tok(tokenizer, he_id)!r}")
    print(f"[TOKENS] ' she' -> id={she_id}, decoded={_decode_tok(tokenizer, she_id)!r}")
    print()

    # Load rows
    csv_path = _resolve_csv(data_dir, args.split, args.csv)
    rows = gp.load_gp_csv(str(csv_path))
    if not rows:
        raise RuntimeError(f"No rows loaded from {csv_path}")
    rng = random.Random(args.seed)
    n = min(args.n, len(rows))
    rows_s = rng.sample(rows, k=n)

    clean_prompts = [r["prefix"] for r in rows_s]
    corr_prompts = [r["corr_prefix"] for r in rows_s]

    # Tokenize
    tokens_clean, last_clean, tokens_corr, _last_corr = gp.tokenize_pair_batch(
        tokenizer, clean_prompts, corr_prompts, device=device
    )

    # Original probs
    probs_orig = _compute_orig_probs(model, tokens_clean, last_clean)

    # Masked probs (optional)
    probs_masked: Optional[torch.Tensor] = None
    want_masked = args.masked
    if want_masked == "on" or (want_masked == "auto" and svd_cache_path.exists() and masks_path.exists()):
        if not svd_cache_path.exists() or not masks_path.exists():
            raise FileNotFoundError(
                f"Masked requested but missing files: svd_cache={svd_cache_path} exists={svd_cache_path.exists()}, "
                f"masks={masks_path} exists={masks_path.exists()}"
            )
        print(f"[MASKED] Using svd_cache={svd_cache_path}")
        print(f"[MASKED] Using masks    ={masks_path}")
        probs_masked = _compute_masked_probs(
            model=model,
            tokens_clean=tokens_clean,
            tokens_corr=tokens_corr,
            last_idx_clean=last_clean,
            svd_cache_path=svd_cache_path,
            masks_path=masks_path,
            dtype_cache=dtype_cache,
        )
        print()

    # Summary table (compact)
    print("=== SUMMARY (compact) ===")
    header = [
        "i",
        "gold",
        "top1_orig",
        "p_top1",
        "p_he",
        "p_she",
        "he-she",
        "top1_masked",
        "p_he_m",
        "p_she_m",
        "he-she_m",
    ]
    print("\t".join(header))
    for i in range(n):
        p = probs_orig[i]
        top1_id = int(torch.argmax(p).item())
        row = [
            str(i),
            str(rows_s[i].get("pronoun", "")),
            _decode_tok(tokenizer, top1_id).replace("\t", " "),
            f"{float(p[top1_id]):.4f}",
            f"{float(p[he_id]):.4f}",
            f"{float(p[she_id]):.4f}",
            f"{float(p[he_id]-p[she_id]):+.4f}",
        ]
        if probs_masked is not None:
            pm = probs_masked[i]
            top1m_id = int(torch.argmax(pm).item())
            row += [
                _decode_tok(tokenizer, top1m_id).replace("\t", " "),
                f"{float(pm[he_id]):.4f}",
                f"{float(pm[she_id]):.4f}",
                f"{float(pm[he_id]-pm[she_id]):+.4f}",
            ]
        else:
            row += ["", "", "", ""]
        print("\t".join(row))

    # Detailed blocks
    print()
    if args.verbose:
        for i in range(n):
            inspect_example(
                tokenizer=tokenizer,
                rows_s=rows_s,
                tokens_clean=tokens_clean,
                last_clean=last_clean,
                probs_orig=probs_orig,
                he_id=he_id,
                she_id=she_id,
                i=i,
                topk=args.topk,
                probs_masked=probs_masked,
                print_full_prefix=args.print_full_prefix,
            )
    else:
        # Show just first 3 detailed examples by default
        for i in range(min(3, n)):
            inspect_example(
                tokenizer=tokenizer,
                rows_s=rows_s,
                tokens_clean=tokens_clean,
                last_clean=last_clean,
                probs_orig=probs_orig,
                he_id=he_id,
                she_id=she_id,
                i=i,
                topk=args.topk,
                probs_masked=probs_masked,
                print_full_prefix=args.print_full_prefix,
            )


if __name__ == "__main__":
    main()
