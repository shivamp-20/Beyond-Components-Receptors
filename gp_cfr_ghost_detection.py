#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gp_cfr_ghost_detection.py

Causal Fidelity Ratio (CFR) ghost detection for logit receptors.

For each (layer, head, sv_idx) with trained mask weight > threshold:
  1. Project out its write direction from the final residual stream
  2. Recompute ln_final + unembed to get ablated logits
  3. Measure accuracy drop (AccDrop)
  4. Normalize: CFR = AccDrop / max(AccDrop)

Ghosts: CFR < 0.05.  Real: CFR >= 0.05.

Inputs:
  data_main/test_gp.csv            (held-out test set)
  outputs/gp/svd_cache.pt          (OV SVD cache from mask training)
  outputs/gp/masks.pt              (trained sigmoid mask weights)

Run:
  python gp_cfr_ghost_detection.py \
    --data_dir data_main \
    --csv test_gp.csv \
    --out_dir outputs/gp \
    --mask_threshold 0.90 \
    --batch_size 64 \
    --device cuda \
    --use_both 1 \
    --known_receptors "10,9,0,+1;11,8,6,+1;9,7,1,-1"
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer


# =====================================================================
# Utility helpers (self-contained — copied/adapted from training script)
# =====================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def normalize_prefix(s: str) -> str:
    return s.rstrip()


def detect_delimiter(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "|", "\t", ";"])
        return dialect.delimiter
    except Exception:
        return "|"


def load_gp_csv(path: str) -> List[Dict[str, str]]:
    delim = detect_delimiter(path)
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for r in reader:
            r["prefix"] = normalize_prefix(r["prefix"])
            r["corr_prefix"] = normalize_prefix(r["corr_prefix"])
            rows.append(r)
    return rows


@dataclass
class SVDMat:
    U: torch.Tensor    # (in_dim, r)
    S: torch.Tensor    # (r,)
    Vh: torch.Tensor   # (r, out_dim)

    @property
    def r(self) -> int:
        return int(self.S.shape[0])


def load_svd_cache_ov_only(path: str, device: str) -> List[List[SVDMat]]:
    """Load only the OV SVDs from the cache (we don't need QK/MLP for this experiment)."""
    cache = torch.load(path, map_location="cpu")
    n_layers = cache["meta"]["n_layers"]
    n_heads = cache["meta"]["n_heads"]

    ov: List[List[SVDMat]] = [[None for _ in range(n_heads)] for _ in range(n_layers)]
    for l in range(n_layers):
        for h in range(n_heads):
            ov[l][h] = SVDMat(
                U=cache["ov"][l][h]["U"].to(device),
                S=cache["ov"][l][h]["S"].to(device),
                Vh=cache["ov"][l][h]["Vh"].to(device),
            )
    return ov


def expand_rows_to_examples(rows: List[Dict[str, str]], use_both: bool = True) -> List[Dict[str, str]]:
    """Expand CSV rows into labelled examples. use_both=True adds corrupted prefixes too."""
    out: List[Dict[str, str]] = []
    for r in rows:
        p = (r.get("pronoun") or "").strip().lower()
        cp = (r.get("corr_pronoun") or "").strip().lower()
        if p in ("he", "she"):
            out.append({"text": r["prefix"], "label": p})
        if use_both and cp in ("he", "she"):
            out.append({"text": r["corr_prefix"], "label": cp})
    return out


def encode_texts(tokenizer: GPT2TokenizerFast, texts: List[str]) -> List[List[int]]:
    return [tokenizer.encode(t, add_special_tokens=False) for t in texts]


def pad_to_length(
    ids_list: List[List[int]], pad_len: int, pad_id: int, device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    B = len(ids_list)
    tokens = torch.full((B, pad_len), pad_id, dtype=torch.long)
    last_idx = torch.empty((B,), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        L = len(ids)
        tokens[i, :L] = torch.tensor(ids, dtype=torch.long)
        last_idx[i] = L - 1
    return tokens.to(device), last_idx.to(device)


def tokenize_texts(
    tokenizer: GPT2TokenizerFast, texts: List[str], device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    ids_list = encode_texts(tokenizer, texts)
    pad_len = max(len(x) for x in ids_list)
    pad_id = tokenizer.pad_token_id
    return pad_to_length(ids_list, pad_len, pad_id, device)


def batches(items, batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# =====================================================================
# Masks.pt structure inspector
# =====================================================================

def inspect_masks(masks_dict: dict) -> None:
    """Print structure of masks.pt so we understand the format."""
    print("\n=== MASKS.PT STRUCTURE ===")
    for key in masks_dict:
        val = masks_dict[key]
        if isinstance(val, list):
            print(f"  key='{key}' -> list of length {len(val)}")
            if len(val) > 0:
                first = val[0]
                if isinstance(first, list):
                    print(f"    [0] -> list of length {len(first)}")
                    if len(first) > 0 and isinstance(first[0], torch.Tensor):
                        print(f"      [0][0] -> Tensor shape={first[0].shape} dtype={first[0].dtype}")
                elif isinstance(first, torch.Tensor):
                    print(f"    [0] -> Tensor shape={first.shape} dtype={first.dtype}")
                else:
                    print(f"    [0] -> type={type(first)}")
        elif isinstance(val, torch.Tensor):
            print(f"  key='{key}' -> Tensor shape={val.shape}")
        else:
            print(f"  key='{key}' -> type={type(val)}")
    print()


# =====================================================================
# Known receptors parser
# =====================================================================

def parse_known_receptors(s: str) -> Dict[Tuple[int, int, int], Tuple[str, int]]:
    """
    Parse --known_receptors "10,9,0,+1;11,8,6,+1;9,7,1,-1"
    Returns {(layer, head, sv_idx): (name, expected_polarity)}
    """
    if not s or not s.strip():
        return {}
    names = ["R1", "R2", "R3", "R4", "R5", "R6", "R7", "R8", "R9", "R10"]
    out = {}
    for i, part in enumerate(s.split(";")):
        part = part.strip()
        if not part:
            continue
        fields = [f.strip() for f in part.split(",")]
        l, h, sv = int(fields[0]), int(fields[1]), int(fields[2])
        pol = int(fields[3]) if len(fields) > 3 else 0
        name = names[i] if i < len(names) else f"R{i+1}"
        out[(l, h, sv)] = (name, pol)
    return out


# =====================================================================
# Core CFR computation
# =====================================================================

@torch.no_grad()
def collect_residuals_and_labels(
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    examples: List[Dict[str, str]],
    he_id: int,
    she_id: int,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    """
    Run ONE forward pass over all examples.

    Returns:
        resid_clean: (N, d_model) — final residual at decision position (after block 11, before ln_final)
        labels:      (N,) — +1 for he, -1 for she
        logit_diff_clean: (N,) — he_logit - she_logit (for coverage check)
        baseline_acc: float
    """
    all_resid = []
    all_labels = []
    all_logit_diff = []
    all_correct = []

    hook_name = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"

    for batch in batches(examples, batch_size):
        texts = [e["text"] for e in batch]
        labs = [+1 if e["label"] == "he" else -1 for e in batch]

        tokens, last_idx = tokenize_texts(tokenizer, texts, device)
        B = tokens.shape[0]

        # Run with cache to get residuals
        logits, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n == hook_name
        )
        resid = cache[hook_name]  # (B, S, d_model)

        # Extract at decision position
        resid_at_pos = resid[torch.arange(B, device=device), last_idx, :]  # (B, d_model)

        # Clean logits at decision position
        logits_at_pos = logits[torch.arange(B, device=device), last_idx, :]  # (B, vocab)
        he_logits = logits_at_pos[:, he_id]
        she_logits = logits_at_pos[:, she_id]
        logit_diff = he_logits - she_logits  # (B,)

        # Labels
        lab_tensor = torch.tensor(labs, device=device, dtype=torch.float32)

        # Correctness: model predicts he if he_logit > she_logit
        pred_he = (he_logits > she_logits).float()  # 1 if pred=he, 0 if pred=she
        correct = ((pred_he == 1) & (lab_tensor == 1)) | ((pred_he == 0) & (lab_tensor == -1))

        all_resid.append(resid_at_pos.cpu())
        all_labels.append(lab_tensor.cpu())
        all_logit_diff.append(logit_diff.cpu())
        all_correct.append(correct.float().cpu())

        del cache  # free memory

    resid_clean = torch.cat(all_resid, dim=0)            # (N, d_model)
    labels = torch.cat(all_labels, dim=0)                 # (N,)
    logit_diff_clean = torch.cat(all_logit_diff, dim=0)   # (N,)
    correct_all = torch.cat(all_correct, dim=0)           # (N,)
    baseline_acc = float(correct_all.mean().item())

    return resid_clean, labels, logit_diff_clean, baseline_acc


@torch.no_grad()
def compute_ablated_accuracy(
    model: HookedTransformer,
    resid_clean: torch.Tensor,   # (N, d_model) on CPU
    labels: torch.Tensor,        # (N,) on CPU (+1/-1)
    r_k: torch.Tensor,           # (d_model,) unit vector, on CPU
    he_id: int,
    she_id: int,
    device: str,
    batch_size: int = 512,       # batched ln_final+unembed to avoid OOM
) -> float:
    """
    Ablate receptor direction r_k from residuals, recompute logits, return accuracy.
    """
    N = resid_clean.shape[0]
    correct = 0

    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        resid_batch = resid_clean[start:end].to(device)  # (B, d_model)
        lab_batch = labels[start:end]                      # (B,)

        r_k_dev = r_k.to(device)

        # Project out r_k
        proj = (resid_batch @ r_k_dev).unsqueeze(-1) * r_k_dev.unsqueeze(0)  # (B, d_model)
        resid_abl = resid_batch - proj  # (B, d_model)

        # Recompute logits
        ln_out = model.ln_final(resid_abl)                     # (B, d_model)
        logits = ln_out @ model.W_U + model.b_U                # (B, vocab)

        he_logits = logits[:, he_id].cpu()
        she_logits = logits[:, she_id].cpu()

        pred_he = (he_logits > she_logits).float()
        lab_he = (lab_batch == 1).float()
        batch_correct = ((pred_he == 1) & (lab_he == 1)) | ((pred_he == 0) & (lab_he == 0))
        correct += int(batch_correct.sum().item())

    return correct / N


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="CFR Ghost Detection for logit receptors")
    ap.add_argument("--data_dir", type=str, default="data_main")
    ap.add_argument("--csv", type=str, default="test_gp.csv")
    ap.add_argument("--out_dir", type=str, default="outputs/gp")
    ap.add_argument("--svd_cache", type=str, default=None, help="Path to svd_cache.pt (default: out_dir/svd_cache.pt)")
    ap.add_argument("--masks_file", type=str, default=None, help="Path to masks.pt (default: out_dir/masks.pt)")
    ap.add_argument("--mask_threshold", type=float, default=0.90)
    ap.add_argument("--ghost_threshold", type=float, default=0.05, help="CFR below this => ghost")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--abl_batch_size", type=int, default=512, help="Batch size for ln_final+unembed during ablation (memory only)")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_both", type=int, default=1, help="1: use clean+corrupt examples, 0: clean only")
    ap.add_argument("--known_receptors", type=str, default="", help="Semicolon-separated known receptors: L,H,SV,pol;...")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA not available, falling back to CPU.")
        device = "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)

    svd_path = Path(args.svd_cache) if args.svd_cache else out_dir / "svd_cache.pt"
    masks_path = Path(args.masks_file) if args.masks_file else out_dir / "masks.pt"

    if not svd_path.exists():
        raise FileNotFoundError(f"Missing SVD cache: {svd_path}")
    if not masks_path.exists():
        raise FileNotFoundError(f"Missing masks: {masks_path}")

    known = parse_known_receptors(args.known_receptors)
    known_inv = {v[0]: k for k, v in known.items()}  # name -> (l,h,sv)

    # ----------------------------------------------------------------
    # Step 0: Inspect masks.pt structure
    # ----------------------------------------------------------------
    print("=" * 70)
    print("CFR GHOST DETECTION EXPERIMENT")
    print("=" * 70)

    masks_dict = torch.load(str(masks_path), map_location="cpu")
    inspect_masks(masks_dict)

    # ----------------------------------------------------------------
    # Step 1: Load model, SVD cache, masks
    # ----------------------------------------------------------------
    print("[LOAD] Tokenizer ...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    he_ids = tokenizer.encode(" he", add_special_tokens=False)
    she_ids = tokenizer.encode(" she", add_special_tokens=False)
    assert len(he_ids) == 1 and len(she_ids) == 1, f"Pronoun tokenization issue: he={he_ids}, she={she_ids}"
    he_id, she_id = he_ids[0], she_ids[0]
    print(f"[TOKENS] he_id={he_id} ('{tokenizer.decode([he_id])}'), she_id={she_id} ('{tokenizer.decode([she_id])}')")

    print("[LOAD] Model (gpt2-small) ...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("[LOAD] SVD cache ...")
    ov = load_svd_cache_ov_only(str(svd_path), device="cpu")  # keep on CPU; we'll move per-receptor

    print("[LOAD] Masks ...")
    # masks_dict["ov"][layer][head] -> tensor of shape (r,) with sigmoid values

    # ----------------------------------------------------------------
    # Step 2: Enumerate receptors with mask > threshold
    # ----------------------------------------------------------------
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model

    receptor_list: List[Dict[str, Any]] = []
    for l in range(n_layers):
        for h in range(n_heads):
            mask_vals = masks_dict["ov"][l][h]  # tensor (r,)
            svd = ov[l][h]
            for k in range(svd.r):
                mw = float(mask_vals[k].item())
                if mw > args.mask_threshold:
                    receptor_list.append({
                        "layer": l,
                        "head": h,
                        "sv_idx": k,
                        "mask_weight": mw,
                        "r_k": svd.Vh[k, :].clone(),  # (d_model,) — unit vector (right singular vector)
                    })

    n_receptors = len(receptor_list)
    print(f"\n[RECEPTORS] Found {n_receptors} with mask > {args.mask_threshold}")
    if n_receptors == 0:
        print("[ERROR] No receptors found! Check mask_threshold or masks.pt.")
        return

    # ----------------------------------------------------------------
    # Step 3: Load test data
    # ----------------------------------------------------------------
    csv_path = str(data_dir / args.csv)
    rows = load_gp_csv(csv_path)
    examples = expand_rows_to_examples(rows, use_both=bool(args.use_both))
    print(f"[DATA] Loaded {len(rows)} rows -> {len(examples)} examples (use_both={bool(args.use_both)})")
    n_he = sum(1 for e in examples if e["label"] == "he")
    n_she = sum(1 for e in examples if e["label"] == "she")
    print(f"[DATA] he: {n_he}, she: {n_she}")

    # ----------------------------------------------------------------
    # Step 4: ONE forward pass to collect clean residuals
    # ----------------------------------------------------------------
    print("\n[FORWARD] Collecting clean residuals (one forward pass) ...")
    resid_clean, labels, logit_diff_clean, baseline_acc = collect_residuals_and_labels(
        model, tokenizer, examples, he_id, she_id, args.batch_size, device
    )
    N = resid_clean.shape[0]
    print(f"[BASELINE] Clean accuracy: {baseline_acc * 100:.2f}% ({int(baseline_acc * N)}/{N})")

    # ----------------------------------------------------------------
    # Step 5: For each receptor — ablate, measure accuracy, compute AUC
    # ----------------------------------------------------------------
    print(f"\n[ABLATION] Testing {n_receptors} receptors ...")
    for idx, rec in enumerate(receptor_list):
        r_k = rec["r_k"]  # (d_model,) on CPU

        # --- Ablated accuracy ---
        acc_abl = compute_ablated_accuracy(
            model, resid_clean, labels, r_k, he_id, she_id, device, args.abl_batch_size
        )
        rec["acc_ablated"] = acc_abl
        rec["acc_drop"] = baseline_acc - acc_abl

        # --- AUC + polarity ---
        # Receptor activation: projection of residual onto r_k
        g_k = (resid_clean @ r_k).numpy()  # (N,)
        lab_np = labels.numpy()             # (N,) +1/-1

        # Determine polarity from correlation
        corr = np.corrcoef(g_k, lab_np)[0, 1]
        polarity = +1 if corr > 0 else -1
        rec["polarity"] = polarity
        rec["corr_with_label"] = float(corr)

        # AUC: labels need to be {0, 1}
        lab_binary = ((lab_np + 1) / 2).astype(int)  # he=1, she=0
        try:
            auc = roc_auc_score(lab_binary, polarity * g_k)
        except ValueError:
            auc = 0.5
        rec["auc"] = auc

        if (idx + 1) % 25 == 0 or idx == 0 or idx == n_receptors - 1:
            l, h, sv = rec["layer"], rec["head"], rec["sv_idx"]
            print(f"  [{idx+1}/{n_receptors}] L{l}H{h}sv{sv}: AccDrop={rec['acc_drop']*100:+.2f}%, AUC={auc:.3f}, pol={polarity:+d}")

    # ----------------------------------------------------------------
    # Step 6: Compute CFR
    # ----------------------------------------------------------------
    acc_drops = np.array([r["acc_drop"] for r in receptor_list])
    max_drop = acc_drops.max()
    print(f"\n[CFR] max(AccDrop) = {max_drop * 100:.3f}%")

    if max_drop <= 0:
        print("[WARN] max(AccDrop) <= 0 — no receptor causes accuracy loss. Setting CFR=0 for all.")
        for r in receptor_list:
            r["cfr"] = 0.0
            r["is_ghost"] = True
    else:
        for r in receptor_list:
            cfr = max(0.0, r["acc_drop"] / max_drop)  # clamp negatives to 0
            r["cfr"] = cfr
            r["is_ghost"] = cfr < args.ghost_threshold

    # Sort by CFR descending
    receptor_list.sort(key=lambda r: r["cfr"], reverse=True)

    # ----------------------------------------------------------------
    # Step 7: Cosine similarities
    # ----------------------------------------------------------------
    print("[COSINE] Computing pairwise cosine similarities ...")
    directions = torch.stack([r["r_k"] for r in receptor_list], dim=0)  # (K, d_model)
    # Normalize (should already be unit, but be safe)
    norms = directions.norm(dim=1, keepdim=True).clamp(min=1e-8)
    directions_normed = directions / norms
    cos_matrix = (directions_normed @ directions_normed.T).numpy()  # (K, K)

    # For each receptor: max |cos| to any REAL receptor (excluding self)
    real_indices = [i for i, r in enumerate(receptor_list) if not r["is_ghost"]]
    for i, r in enumerate(receptor_list):
        if len(real_indices) == 0:
            r["max_cos_to_real"] = 0.0
        elif not r["is_ghost"]:
            # For real receptors: max |cos| to OTHER real receptors
            cos_to_others = [abs(cos_matrix[i, j]) for j in real_indices if j != i]
            r["max_cos_to_real"] = max(cos_to_others) if cos_to_others else 0.0
        else:
            # For ghosts: max |cos| to any real receptor
            cos_to_reals = [abs(cos_matrix[i, j]) for j in real_indices]
            r["max_cos_to_real"] = max(cos_to_reals) if cos_to_reals else 0.0

    # ----------------------------------------------------------------
    # Step 8: Coverage check (R² with real receptors)
    # ----------------------------------------------------------------
    print("[COVERAGE] Computing R² of real receptors vs model logit diff ...")
    logit_diff_np = logit_diff_clean.numpy()

    if len(real_indices) > 0:
        # Activations on real receptors
        real_dirs = torch.stack([receptor_list[i]["r_k"] for i in real_indices], dim=0)  # (K_real, d_model)
        g_real = (resid_clean @ real_dirs.T).numpy()  # (N, K_real)
        reg_real = LinearRegression().fit(g_real, logit_diff_np)
        r2_real = r2_score(logit_diff_np, reg_real.predict(g_real))
    else:
        r2_real = 0.0

    # All receptors
    all_dirs = torch.stack([r["r_k"] for r in receptor_list], dim=0)  # (K, d_model)
    g_all = (resid_clean @ all_dirs.T).numpy()  # (N, K)
    reg_all = LinearRegression().fit(g_all, logit_diff_np)
    r2_all = r2_score(logit_diff_np, reg_all.predict(g_all))

    print(f"  R² (real receptors only, K={len(real_indices)}): {r2_real:.4f}")
    print(f"  R² (all receptors, K={n_receptors}):  {r2_all:.4f}")

    # ----------------------------------------------------------------
    # Step 9: Print results
    # ----------------------------------------------------------------
    n_real = sum(1 for r in receptor_list if not r["is_ghost"])
    n_ghost = n_receptors - n_real
    ghost_frac = n_ghost / n_receptors if n_receptors > 0 else 0

    print("\n" + "=" * 110)
    print("=== CFR GHOST DETECTION RESULTS ===")
    print("=" * 110)
    print(f"Clean baseline accuracy: {baseline_acc * 100:.2f}%")
    print(f"Number of receptors tested: {n_receptors} (mask threshold > {args.mask_threshold})")
    print(f"Ghost threshold: CFR < {args.ghost_threshold}")
    print()

    # Full table
    header = f"{'Rank':>4} | {'Layer':>5} | {'Head':>4} | {'SV':>3} | {'MaskWt':>7} | {'AUC':>6} | {'AccDrop%':>9} | {'CFR':>7} | {'Pol':>4} | {'Ghost?':>6} | {'MaxCosReal':>10} | {'Name':>5}"
    print(header)
    print("-" * len(header))
    for rank_i, r in enumerate(receptor_list):
        key = (r["layer"], r["head"], r["sv_idx"])
        name = known.get(key, ("", 0))[0]
        ghost_str = "GHOST" if r["is_ghost"] else "REAL"
        print(
            f"{rank_i+1:>4} | {r['layer']:>5} | {r['head']:>4} | {r['sv_idx']:>3} | "
            f"{r['mask_weight']:>7.4f} | {r['auc']:>6.3f} | {r['acc_drop']*100:>+9.3f} | "
            f"{r['cfr']:>7.4f} | {r['polarity']:>+4d} | {ghost_str:>6} | "
            f"{r['max_cos_to_real']:>10.4f} | {name:>5}"
        )

    # Summary
    print("\n" + "=" * 60)
    print("[SUMMARY]")
    print(f"  Receptors with CFR >= {args.ghost_threshold}: {n_real} (REAL)")
    print(f"  Receptors with CFR <  {args.ghost_threshold}: {n_ghost} (GHOSTS)")
    print(f"  Ghost fraction: {n_ghost}/{n_receptors} = {ghost_frac*100:.1f}%")

    # Known receptors
    if known:
        print(f"\n[KNOWN RECEPTORS]")
        for key, (name, _) in known.items():
            found = [r for r in receptor_list if (r["layer"], r["head"], r["sv_idx"]) == key]
            if found:
                r = found[0]
                rank_i = receptor_list.index(r) + 1
                ghost_str = "GHOST" if r["is_ghost"] else "REAL"
                print(f"  {name} (L{r['layer']}H{r['head']}sv{r['sv_idx']}): CFR={r['cfr']:.4f}, AccDrop={r['acc_drop']*100:+.3f}%, AUC={r['auc']:.3f}, Rank={rank_i} [{ghost_str}]")
            else:
                print(f"  {name} (L{key[0]}H{key[1]}sv{key[2]}): NOT FOUND above mask threshold {args.mask_threshold}")

    # Coverage
    print(f"\n[COVERAGE CHECK]")
    print(f"  R² of {n_real} real receptors vs model logit diff:  {r2_real:.4f}")
    print(f"  R² of all {n_receptors} receptors vs model logit diff: {r2_all:.4f}")

    # Cosine analysis
    if n_real > 0 and n_ghost > 0:
        ghost_cos = [r["max_cos_to_real"] for r in receptor_list if r["is_ghost"]]
        real_cos = [r["max_cos_to_real"] for r in receptor_list if not r["is_ghost"]]
        print(f"\n[COSINE ANALYSIS]")
        print(f"  Ghost receptors — max |cos| to nearest real receptor:")
        print(f"    Mean:   {np.mean(ghost_cos):.4f}")
        print(f"    Median: {np.median(ghost_cos):.4f}")
        print(f"    Min:    {np.min(ghost_cos):.4f}")
        print(f"    Max:    {np.max(ghost_cos):.4f}")
        if len(real_cos) > 1:
            print(f"  Real receptors — max |cos| to nearest OTHER real receptor:")
            print(f"    Mean:   {np.mean(real_cos):.4f}")
            print(f"    Median: {np.median(real_cos):.4f}")

    # ----------------------------------------------------------------
    # Step 10: Save plots
    # ----------------------------------------------------------------
    print(f"\n[PLOTS] Saving plots to {out_dir} ...")

    cfr_vals = np.array([r["cfr"] for r in receptor_list])
    auc_vals = np.array([r["auc"] for r in receptor_list])
    acc_drop_vals = np.array([r["acc_drop"] for r in receptor_list])
    mask_weights = np.array([r["mask_weight"] for r in receptor_list])
    is_ghost = np.array([r["is_ghost"] for r in receptor_list])
    max_cos_vals = np.array([r["max_cos_to_real"] for r in receptor_list])

    # --- Plot 1: CFR Distribution Histogram ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.arange(0, 1.02, 0.02)
    ghost_cfrs = cfr_vals[is_ghost]
    real_cfrs = cfr_vals[~is_ghost]
    ax.hist(ghost_cfrs, bins=bins, color="red", alpha=0.7, label=f"Ghost (CFR<{args.ghost_threshold}, n={n_ghost})", edgecolor="darkred")
    ax.hist(real_cfrs, bins=bins, color="blue", alpha=0.7, label=f"Real (CFR≥{args.ghost_threshold}, n={n_real})", edgecolor="darkblue")
    ax.axvline(x=args.ghost_threshold, color="black", linestyle="--", linewidth=1.5, label=f"Threshold={args.ghost_threshold}")
    ax.set_xlabel("Causal Fidelity Ratio (CFR)", fontsize=13)
    ax.set_ylabel("Count of Receptors", fontsize=13)
    ax.set_title(f"CFR Distribution (N={n_receptors} receptors, mask > {args.mask_threshold})", fontsize=14)
    ax.legend(fontsize=11)
    ax.set_xlim(-0.02, 1.05)
    plt.tight_layout()
    plot1_path = str(out_dir / "plot1_cfr_histogram.png")
    fig.savefig(plot1_path, dpi=150)
    plt.close(fig)
    print(f"  [1] {plot1_path}")

    # --- Plot 2: AUC vs CFR Scatter ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(auc_vals[is_ghost], cfr_vals[is_ghost], c="red", alpha=0.5, s=30, label=f"Ghost (n={n_ghost})", zorder=2)
    ax.scatter(auc_vals[~is_ghost], cfr_vals[~is_ghost], c="blue", alpha=0.7, s=60, label=f"Real (n={n_real})", zorder=3)
    # Mark known receptors
    for key, (name, _) in known.items():
        found = [r for r in receptor_list if (r["layer"], r["head"], r["sv_idx"]) == key]
        if found:
            r = found[0]
            ax.scatter(r["auc"], r["cfr"], marker="*", s=250, c="gold", edgecolors="black", zorder=5)
            ax.annotate(name, (r["auc"], r["cfr"]), textcoords="offset points",
                       xytext=(8, 8), fontsize=11, fontweight="bold")
    ax.axhline(y=args.ghost_threshold, color="black", linestyle="--", linewidth=1.5, label=f"Ghost threshold (CFR={args.ghost_threshold})")
    ax.set_xlabel("AUC (Correlation)", fontsize=13)
    ax.set_ylabel("CFR (Causation)", fontsize=13)
    ax.set_title("Correlation (AUC) vs Causation (CFR)", fontsize=14)
    ax.legend(fontsize=10)
    ax.set_xlim(0.48, 1.02)
    ax.set_ylim(-0.05, 1.1)
    plt.tight_layout()
    plot2_path = str(out_dir / "plot2_auc_vs_cfr.png")
    fig.savefig(plot2_path, dpi=150)
    plt.close(fig)
    print(f"  [2] {plot2_path}")

    # --- Plot 3: Cosine-to-Nearest-Real vs CFR ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(max_cos_vals[is_ghost], cfr_vals[is_ghost], c="red", alpha=0.5, s=30, label=f"Ghost (n={n_ghost})", zorder=2)
    ax.scatter(max_cos_vals[~is_ghost], cfr_vals[~is_ghost], c="blue", alpha=0.7, s=60, label=f"Real (n={n_real})", zorder=3)
    # Trend line for ghosts
    if n_ghost > 2:
        ghost_mask = is_ghost
        x_trend = max_cos_vals[ghost_mask]
        y_trend = cfr_vals[ghost_mask]
        if x_trend.std() > 1e-8:
            z = np.polyfit(x_trend, y_trend, 1)
            x_line = np.linspace(x_trend.min(), x_trend.max(), 50)
            ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.6, linewidth=2,
                   label=f"Ghost trend (slope={z[0]:.3f})")
    ax.set_xlabel("Max |cos(r_k, r_real)| to Nearest Real Receptor", fontsize=13)
    ax.set_ylabel("CFR", fontsize=13)
    ax.set_title("Geometric Overlap Predicts Ghost Status", fontsize=14)
    ax.legend(fontsize=10)
    plt.tight_layout()
    plot3_path = str(out_dir / "plot3_cosine_vs_cfr.png")
    fig.savefig(plot3_path, dpi=150)
    plt.close(fig)
    print(f"  [3] {plot3_path}")

    # --- Plot 4: AccDrop Bar Chart (Top 20) ---
    top_n = min(20, n_receptors)
    fig, ax = plt.subplots(figsize=(14, 6))
    top_recs = receptor_list[:top_n]
    labels_bar = [f"L{r['layer']}H{r['head']}sv{r['sv_idx']}" for r in top_recs]
    drops_bar = [r["acc_drop"] * 100 for r in top_recs]
    colors_bar = ["blue" if not r["is_ghost"] else "red" for r in top_recs]
    bars = ax.bar(range(top_n), drops_bar, color=colors_bar, edgecolor="black", alpha=0.8)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(labels_bar, rotation=60, ha="right", fontsize=9)
    ax.set_ylabel("AccDrop (%)", fontsize=13)
    ax.set_title(f"Single-Ablation Accuracy Drop (Top {top_n})", fontsize=14)
    # Annotate known receptors
    for i, r in enumerate(top_recs):
        key = (r["layer"], r["head"], r["sv_idx"])
        if key in known:
            name = known[key][0]
            ax.annotate(name, (i, drops_bar[i]), textcoords="offset points",
                       xytext=(0, 5), ha="center", fontsize=10, fontweight="bold", color="darkblue")
    # Legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='blue', edgecolor='black', label='Real'),
                       Patch(facecolor='red', edgecolor='black', label='Ghost')]
    ax.legend(handles=legend_elements, fontsize=11)
    plt.tight_layout()
    plot4_path = str(out_dir / "plot4_accdrop_top20.png")
    fig.savefig(plot4_path, dpi=150)
    plt.close(fig)
    print(f"  [4] {plot4_path}")

    # --- Plot 5: CFR at Different Mask Thresholds ---
    thresholds = [0.90, 0.95, 0.98, 0.99]
    fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=False)
    for ax_i, thresh in enumerate(thresholds):
        recs_at_thresh = [r for r in receptor_list if r["mask_weight"] > thresh]
        n_real_t = sum(1 for r in recs_at_thresh if not r["is_ghost"])
        n_ghost_t = sum(1 for r in recs_at_thresh if r["is_ghost"])
        ax = axes[ax_i]
        ax.bar(["Real", "Ghost"], [n_real_t, n_ghost_t], color=["blue", "red"], edgecolor="black", alpha=0.8)
        ax.set_title(f"mask > {thresh}", fontsize=12)
        ax.set_ylabel("Count" if ax_i == 0 else "", fontsize=11)
        # Annotate counts
        ax.text(0, n_real_t + 0.5, str(n_real_t), ha="center", fontsize=12, fontweight="bold", color="blue")
        ax.text(1, n_ghost_t + 0.5, str(n_ghost_t), ha="center", fontsize=12, fontweight="bold", color="red")
    fig.suptitle("Ghost Detection at Different Mask Thresholds", fontsize=14, y=1.02)
    plt.tight_layout()
    plot5_path = str(out_dir / "plot5_thresholds.png")
    fig.savefig(plot5_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  [5] {plot5_path}")

    # ----------------------------------------------------------------
    # Step 11: Save JSON results
    # ----------------------------------------------------------------
    results_json = {
        "config": {
            "csv": args.csv,
            "mask_threshold": args.mask_threshold,
            "ghost_threshold": args.ghost_threshold,
            "use_both": bool(args.use_both),
            "n_examples": N,
        },
        "baseline_acc": baseline_acc,
        "n_receptors": n_receptors,
        "n_real": n_real,
        "n_ghost": n_ghost,
        "ghost_fraction": ghost_frac,
        "max_acc_drop": float(max_drop),
        "r2_real_only": r2_real,
        "r2_all": r2_all,
        "receptors": [
            {
                "rank": i + 1,
                "layer": r["layer"],
                "head": r["head"],
                "sv_idx": r["sv_idx"],
                "mask_weight": r["mask_weight"],
                "auc": r["auc"],
                "acc_drop": r["acc_drop"],
                "cfr": r["cfr"],
                "polarity": r["polarity"],
                "is_ghost": r["is_ghost"],
                "max_cos_to_real": r["max_cos_to_real"],
                "known_name": known.get((r["layer"], r["head"], r["sv_idx"]), ("", 0))[0],
            }
            for i, r in enumerate(receptor_list)
        ],
    }

    json_path = str(out_dir / "cfr_ghost_detection_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[SAVE] Results JSON -> {json_path}")

    # ----------------------------------------------------------------
    # Sanity checks
    # ----------------------------------------------------------------
    print("\n" + "=" * 60)
    print("[SANITY CHECKS]")
    print(f"  Baseline accuracy: {baseline_acc*100:.2f}% (expected ~89-90%)")

    if known:
        r1_key = known_inv.get("R1")
        r2_key = known_inv.get("R2")
        r3_key = known_inv.get("R3")

        if r1_key:
            r1 = [r for r in receptor_list if (r["layer"], r["head"], r["sv_idx"]) == r1_key]
            if r1:
                rank1 = receptor_list.index(r1[0]) + 1
                print(f"  R1 (L{r1_key[0]}H{r1_key[1]}sv{r1_key[2]}): CFR={r1[0]['cfr']:.4f}, Rank={rank1} — {'PASS' if rank1 <= 3 else 'CHECK: expected top-3'}")

        if r2_key:
            r2 = [r for r in receptor_list if (r["layer"], r["head"], r["sv_idx"]) == r2_key]
            if r2:
                print(f"  R2 (L{r2_key[0]}H{r2_key[1]}sv{r2_key[2]}): CFR={r2[0]['cfr']:.4f}, Ghost={r2[0]['is_ghost']} — {'PASS' if r2[0]['is_ghost'] else 'CHECK: expected ghost'}")

        if r3_key:
            r3 = [r for r in receptor_list if (r["layer"], r["head"], r["sv_idx"]) == r3_key]
            if r3:
                print(f"  R3 (L{r3_key[0]}H{r3_key[1]}sv{r3_key[2]}): CFR={r3[0]['cfr']:.4f}, Ghost={r3[0]['is_ghost']} — {'PASS' if not r3[0]['is_ghost'] else 'CHECK: expected real'}")

    neg_drops = [r for r in receptor_list if r["acc_drop"] < -0.005]
    if neg_drops:
        print(f"  [WARN] {len(neg_drops)} receptors have AccDrop < -0.5% (ablation helps). Investigate.")
    else:
        print(f"  No significant negative AccDrops. PASS.")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
