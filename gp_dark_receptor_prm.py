#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gp_dark_receptor_prm.py

Dark Receptor Characterization + Predictive Receptor Margin (PRM).

Characterizes L11H1sv1 ("dark receptor") discovered by OCA:
  - AUC=0.526 (near chance for gender) but ablation drops accuracy by 8.17%.
  - Causally critical, correlatively invisible.

Tests whether it adds error-predictive power beyond gender-aligned receptors (R1, R3).

Run:
  python gp_dark_receptor_prm.py \
    --data_dir data_main \
    --csv test_gp.csv \
    --out_dir outputs/gp \
    --receptors "10,9,0,+1;9,7,1,-1" \
    --dark_receptor "11,1,1,+1" \
    --batch_size 64 \
    --device cuda \
    --use_both 1
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer


# =====================================================================
# Utility helpers (self-contained, same as CFR/OCA scripts)
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
    U: torch.Tensor
    S: torch.Tensor
    Vh: torch.Tensor

    @property
    def r(self) -> int:
        return int(self.S.shape[0])


def load_svd_cache_ov_only(path: str, device: str) -> List[List[SVDMat]]:
    cache = torch.load(path, map_location="cpu")
    n_layers = cache["meta"]["n_layers"]
    n_heads = cache["meta"]["n_heads"]
    ov = [[None for _ in range(n_heads)] for _ in range(n_layers)]
    for l in range(n_layers):
        for h in range(n_heads):
            ov[l][h] = SVDMat(
                U=cache["ov"][l][h]["U"].to(device),
                S=cache["ov"][l][h]["S"].to(device),
                Vh=cache["ov"][l][h]["Vh"].to(device),
            )
    return ov


def expand_rows_to_examples(rows: List[Dict[str, str]], use_both: bool = True) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        p = (r.get("pronoun") or "").strip().lower()
        cp = (r.get("corr_pronoun") or "").strip().lower()
        if p in ("he", "she"):
            out.append({"text": r["prefix"], "label": p})
        if use_both and cp in ("he", "she"):
            out.append({"text": r["corr_prefix"], "label": cp})
    return out


def encode_texts(tokenizer, texts):
    return [tokenizer.encode(t, add_special_tokens=False) for t in texts]


def pad_to_length(ids_list, pad_len, pad_id, device):
    B = len(ids_list)
    tokens = torch.full((B, pad_len), pad_id, dtype=torch.long)
    last_idx = torch.empty((B,), dtype=torch.long)
    for i, ids in enumerate(ids_list):
        L = len(ids)
        tokens[i, :L] = torch.tensor(ids, dtype=torch.long)
        last_idx[i] = L - 1
    return tokens.to(device), last_idx.to(device)


def batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def parse_receptor_spec(spec: str) -> Tuple[int, int, int, int]:
    """Parse 'layer,head,sv_idx,polarity' string."""
    parts = spec.strip().split(",")
    l, h, sv = int(parts[0]), int(parts[1]), int(parts[2])
    pol = int(parts[3])
    return l, h, sv, pol


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Dark Receptor Characterization + PRM")
    ap.add_argument("--data_dir", type=str, default="data_main")
    ap.add_argument("--csv", type=str, default="test_gp.csv")
    ap.add_argument("--out_dir", type=str, default="outputs/gp")
    ap.add_argument("--svd_cache", type=str, default=None)
    ap.add_argument("--receptors", type=str, default="10,9,0,+1;9,7,1,-1",
                    help="R1;R3 specs as layer,head,sv,polarity")
    ap.add_argument("--dark_receptor", type=str, default="11,1,1,+1",
                    help="Dark receptor spec")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_both", type=int, default=1)
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

    # ==================================================================
    # Step 0: Setup
    # ==================================================================
    print("=" * 70)
    print("DARK RECEPTOR CHARACTERIZATION + PRM")
    print("=" * 70)

    # Parse receptor specs
    rec_specs = [parse_receptor_spec(s) for s in args.receptors.split(";")]
    dark_spec = parse_receptor_spec(args.dark_receptor)

    r1_l, r1_h, r1_sv, r1_pol = rec_specs[0]
    r3_l, r3_h, r3_sv, r3_pol = rec_specs[1]
    dk_l, dk_h, dk_sv, dk_pol = dark_spec

    print(f"R1: L{r1_l}H{r1_h}sv{r1_sv} pol={r1_pol:+d}")
    print(f"R3: L{r3_l}H{r3_h}sv{r3_sv} pol={r3_pol:+d}")
    print(f"Dark: L{dk_l}H{dk_h}sv{dk_sv} pol={dk_pol:+d}")

    # Load tokenizer
    print("\n[LOAD] Tokenizer ...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    he_ids = tokenizer.encode(" he", add_special_tokens=False)
    she_ids = tokenizer.encode(" she", add_special_tokens=False)
    assert len(he_ids) == 1 and len(she_ids) == 1
    he_id, she_id = he_ids[0], she_ids[0]

    # Load model
    print("[LOAD] Model (gpt2-small) ...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load SVD cache and extract directions
    print("[LOAD] SVD cache ...")
    ov = load_svd_cache_ov_only(str(svd_path), device="cpu")
    r1 = ov[r1_l][r1_h].Vh[r1_sv, :].clone().cpu().float()   # (768,)
    r3 = ov[r3_l][r3_h].Vh[r3_sv, :].clone().cpu().float()
    r_dark = ov[dk_l][dk_h].Vh[dk_sv, :].clone().cpu().float()

    # Load test data
    print("[LOAD] Test data ...")
    csv_path = str(data_dir / args.csv)
    rows = load_gp_csv(csv_path)
    examples = expand_rows_to_examples(rows, use_both=bool(args.use_both))
    N = len(examples)
    print(f"  {len(rows)} rows -> {N} examples (use_both={bool(args.use_both)})")

    # ==================================================================
    # Step 1: Collect residuals + attention (ONE forward pass)
    # ==================================================================
    print("\n[FORWARD] Collecting residuals + L11H1 attention ...")

    hook_resid = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    hook_attn = f"blocks.{dk_l}.attn.hook_pattern"
    hook_names = {hook_resid, hook_attn}

    all_resid = []          # (N, d_model)
    all_labels = []         # (N,) signed: +1=he, -1=she
    all_seq_lens = []       # (N,) actual token count per example
    all_attn_rows = []      # (N, max_seq_len) — attention at decision pos
    all_token_ids = []      # list of (seq_len,) int lists — for token decoding
    all_logit_diff = []     # (N,) he_logit - she_logit

    max_seq_len_global = 0  # track for padding attention later

    for batch_items in batches(examples, args.batch_size):
        texts = [e["text"] for e in batch_items]
        labs = [+1 if e["label"] == "he" else -1 for e in batch_items]

        ids_list = encode_texts(tokenizer, texts)
        pad_len = max(len(x) for x in ids_list)
        max_seq_len_global = max(max_seq_len_global, pad_len)
        pad_id = tokenizer.pad_token_id
        tokens, last_idx = pad_to_length(ids_list, pad_len, pad_id, device)
        B = tokens.shape[0]

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n in hook_names
            )

        # Residuals at decision position
        resid = cache[hook_resid]  # (B, S, d_model)
        resid_at_pos = resid[torch.arange(B, device=device), last_idx, :]  # (B, d_model)

        # Attention: (B, n_heads, S, S) -> head dk_h, decision row
        attn_full = cache[hook_attn]  # (B, n_heads, S, S)
        # Extract head dk_h (=1), row at decision position for each example
        attn_head = attn_full[:, dk_h, :, :]  # (B, S, S)
        # For each example i, get row last_idx[i] -> (S,)
        attn_rows = attn_head[torch.arange(B, device=device), last_idx, :]  # (B, S)

        # Clean logits at decision position
        logits_at_pos = logits[torch.arange(B, device=device), last_idx, :]
        he_logits = logits_at_pos[:, he_id]
        she_logits = logits_at_pos[:, she_id]
        logit_diff = he_logits - she_logits

        lab_tensor = torch.tensor(labs, dtype=torch.float32)

        all_resid.append(resid_at_pos.cpu())
        all_labels.append(lab_tensor)
        all_logit_diff.append(logit_diff.cpu())
        all_attn_rows.append(attn_rows.cpu())

        for ids in ids_list:
            all_seq_lens.append(len(ids))
            all_token_ids.append(ids)

        del cache

    resid_final = torch.cat(all_resid, dim=0)         # (N, d_model)
    y_signed = torch.cat(all_labels, dim=0)            # (N,) +1/-1
    logit_diff_all = torch.cat(all_logit_diff, dim=0)  # (N,)
    seq_lens = np.array(all_seq_lens)                   # (N,)

    # Pad attention rows to global max seq len
    # Each batch may have different pad_len, so re-pad to max_seq_len_global
    attn_padded = torch.zeros(N, max_seq_len_global)
    idx = 0
    for attn_batch in all_attn_rows:
        B_batch, S_batch = attn_batch.shape
        attn_padded[idx:idx + B_batch, :S_batch] = attn_batch
        idx += B_batch

    # Compute predictions and accuracy
    y_binary = (y_signed == 1).long()          # 1=he, 0=she
    pred = (logit_diff_all > 0).long()         # 1=he, 0=she
    correct = (pred == y_binary)
    error = ~correct
    baseline_acc = float(correct.float().mean().item())
    n_errors = int(error.sum().item())

    print(f"[BASELINE] Accuracy: {baseline_acc*100:.2f}% ({N - n_errors}/{N} correct, {n_errors} errors)")

    # ==================================================================
    # Step 2: Compute receptor activations
    # ==================================================================
    print("\n[STEP 2] Receptor activations ...")
    g1 = (resid_final @ r1).numpy()          # (N,)
    g3 = (resid_final @ r3).numpy()          # (N,)
    g_dark = (resid_final @ r_dark).numpy()  # (N,)

    print(f"  g1 (R1): mean={g1.mean():.4f}, std={g1.std():.4f}")
    print(f"  g3 (R3): mean={g3.mean():.4f}, std={g3.std():.4f}")
    print(f"  g_dark:  mean={g_dark.mean():.4f}, std={g_dark.std():.4f}")

    # ==================================================================
    # Step 3: Token profiles via W_U
    # ==================================================================
    print("\n[STEP 3] Token profiles ...")
    W_U = model.W_U.cpu().float()  # (d_model, d_vocab)

    logit_dark = (r_dark @ W_U).numpy()  # (vocab,)
    logit_r1 = (r1 @ W_U).numpy()
    logit_r3 = (r3 @ W_U).numpy()

    vocab_size = logit_dark.shape[0]

    def top_bottom_tokens(logit_profile, tokenizer, n=30):
        top_idx = np.argsort(logit_profile)[::-1][:n]
        bot_idx = np.argsort(logit_profile)[:n]
        top = [(int(i), tokenizer.decode([i]), float(logit_profile[i])) for i in top_idx]
        bot = [(int(i), tokenizer.decode([i]), float(logit_profile[i])) for i in bot_idx]
        return top, bot

    dark_top, dark_bot = top_bottom_tokens(logit_dark, tokenizer, 30)
    r1_top, _ = top_bottom_tokens(logit_r1, tokenizer, 10)
    r3_top, _ = top_bottom_tokens(logit_r3, tokenizer, 10)

    # Find ranks of gendered tokens in dark profile
    he_rank = int(np.where(np.argsort(logit_dark)[::-1] == he_id)[0][0]) + 1
    she_rank = int(np.where(np.argsort(logit_dark)[::-1] == she_id)[0][0]) + 1
    him_id = tokenizer.encode(" him", add_special_tokens=False)[0]
    her_id = tokenizer.encode(" her", add_special_tokens=False)[0]
    him_rank = int(np.where(np.argsort(logit_dark)[::-1] == him_id)[0][0]) + 1
    her_rank = int(np.where(np.argsort(logit_dark)[::-1] == her_id)[0][0]) + 1

    print(f"\n  Dark receptor — top-10 tokens:")
    for i, (tid, tok, val) in enumerate(dark_top[:10]):
        print(f"    {i+1:>3}. {repr(tok):<15} logit={val:+.4f}")
    print(f"  Dark receptor — bottom-10 tokens:")
    for i, (tid, tok, val) in enumerate(dark_bot[:10]):
        print(f"    {i+1:>3}. {repr(tok):<15} logit={val:+.4f}")

    print(f"\n  Gender token ranks in dark profile (out of {vocab_size}):")
    print(f"    ' he' = rank {he_rank}, ' she' = rank {she_rank}")
    print(f"    ' him' = rank {him_rank}, ' her' = rank {her_rank}")

    print(f"\n  R1 — top-10 tokens:")
    for i, (tid, tok, val) in enumerate(r1_top):
        print(f"    {i+1:>3}. {repr(tok):<15} logit={val:+.4f}")
    print(f"  R3 — top-10 tokens:")
    for i, (tid, tok, val) in enumerate(r3_top):
        print(f"    {i+1:>3}. {repr(tok):<15} logit={val:+.4f}")

    # ==================================================================
    # Step 4: Dark receptor activation analysis
    # ==================================================================
    print("\n[STEP 4] Dark receptor activation analysis ...")
    y_np = y_signed.numpy()
    correct_np = correct.numpy()
    error_np = error.numpy()

    he_mask = (y_np == 1)
    she_mask = (y_np == -1)

    print(f"  g_dark — he-context:  mean={g_dark[he_mask].mean():.4f}, std={g_dark[he_mask].std():.4f}")
    print(f"  g_dark — she-context: mean={g_dark[she_mask].mean():.4f}, std={g_dark[she_mask].std():.4f}")
    print(f"  g_dark — correct:     mean={g_dark[correct_np].mean():.4f}, std={g_dark[correct_np].std():.4f}")
    print(f"  g_dark — error:       mean={g_dark[error_np].mean():.4f}, std={g_dark[error_np].std():.4f}")

    corr_length = np.corrcoef(np.abs(g_dark), seq_lens)[0, 1]
    print(f"  Correlation |g_dark| vs sentence length: {corr_length:.4f}")

    # ==================================================================
    # Step 5: Attention pattern analysis for L11H1
    # ==================================================================
    print("\n[STEP 5] L11H1 attention pattern analysis ...")

    # Find most common sequence length for clean attention plot
    unique_lens, len_counts = np.unique(seq_lens, return_counts=True)
    most_common_len = int(unique_lens[np.argmax(len_counts)])
    most_common_count = int(len_counts[np.argmax(len_counts)])
    print(f"  Most common seq length: {most_common_len} ({most_common_count} examples)")

    # Filter to examples with this length
    len_mask = (seq_lens == most_common_len)
    attn_common = attn_padded[len_mask, :most_common_len]  # (n_common, most_common_len)
    correct_common = correct_np[len_mask]
    error_common = error_np[len_mask]

    avg_attn = attn_common.numpy().mean(axis=0)
    avg_attn_correct = attn_common[correct_common].numpy().mean(axis=0) if correct_common.any() else np.zeros(most_common_len)
    avg_attn_error = attn_common[error_common].numpy().mean(axis=0) if error_common.any() else np.zeros(most_common_len)

    # Get a representative example for token labels
    rep_idx = np.where(len_mask)[0][0]
    rep_tokens = all_token_ids[rep_idx]
    rep_token_strs = [tokenizer.decode([t]) for t in rep_tokens]
    print(f"\n  Representative tokens (len={most_common_len}):")
    for i, ts in enumerate(rep_token_strs):
        print(f"    pos {i}: {repr(ts)}")

    n_common_errors = int(error_common.sum())
    print(f"  Filtered examples: {len(attn_common)} (correct={int(correct_common.sum())}, error={n_common_errors})")

    # ==================================================================
    # Step 6: Predictive Receptor Margin
    # ==================================================================
    print("\n[STEP 6] Predictive Receptor Margin ...")

    # Gender score: pol1*g1 + pol3*g3 = (+1)*g1 + (-1)*g3 = g1 - g3
    gender_score = float(r1_pol) * g1 + float(r3_pol) * g3  # (N,)

    # PRM2: confidence from gender receptors
    prm2 = np.abs(gender_score)

    # PRM_dark: dark receptor alone
    prm_dark = np.abs(g_dark)

    # PRM3: gender x dark
    prm3 = prm2 * prm_dark

    error_binary = error_np.astype(int)

    # Dark receptor sign exploration
    auc_dark_neg_g = roc_auc_score(error_binary, -g_dark)       # low g_dark → error
    auc_dark_pos_g = roc_auc_score(error_binary, g_dark)        # high g_dark → error
    auc_dark_abs = roc_auc_score(error_binary, -np.abs(g_dark)) # low |g_dark| → error
    print(f"  Dark receptor error AUC: -g_dark={auc_dark_neg_g:.4f}, +g_dark={auc_dark_pos_g:.4f}, -|g_dark|={auc_dark_abs:.4f}")

    best_dark_auc = max(auc_dark_neg_g, auc_dark_pos_g, auc_dark_abs)
    best_dark_name = ["-g_dark", "+g_dark", "-|g_dark|"][[auc_dark_neg_g, auc_dark_pos_g, auc_dark_abs].index(best_dark_auc)]
    print(f"  Best dark predictor: {best_dark_name} (AUC={best_dark_auc:.4f})")

    # Main AUCs
    auc_prm2 = roc_auc_score(error_binary, -prm2)
    auc_prm3 = roc_auc_score(error_binary, -prm3)
    auc_dark_only = best_dark_auc

    print(f"\n  Error Prediction AUC-ROC:")
    print(f"    PRM2 (gender only):        {auc_prm2:.4f}")
    print(f"    PRM3 (gender x dark):      {auc_prm3:.4f}")
    print(f"    PRM_dark (dark only):       {auc_dark_only:.4f}")
    print(f"    Improvement (PRM3 - PRM2): {auc_prm3 - auc_prm2:+.4f}")

    # ==================================================================
    # Step 7: Precision@k curves
    # ==================================================================
    print("\n[STEP 7] Precision@k curves ...")
    prec_curves = {}
    for name, scores in [("PRM2", prm2), ("PRM3", prm3), ("PRM_dark", prm_dark)]:
        order = np.argsort(scores)  # ascending = lowest confidence first = most error-prone
        cum_errors = np.cumsum(error_binary[order])
        precision_at_k = cum_errors / np.arange(1, N + 1)
        prec_curves[name] = precision_at_k
        # Report precision at k=n_errors (ideal = 100%)
        print(f"  {name}: P@{n_errors}={precision_at_k[n_errors-1]*100:.1f}%")

    # ==================================================================
    # Step 8: Error type decomposition
    # ==================================================================
    print("\n[STEP 8] Error type decomposition ...")
    gender_thr = float(np.percentile(prm2, 25))
    dark_thr = float(np.percentile(prm_dark, 25))

    type_G  = (prm2 < gender_thr) & (prm_dark >= dark_thr)
    type_D  = (prm2 >= gender_thr) & (prm_dark < dark_thr)
    type_GD = (prm2 < gender_thr) & (prm_dark < dark_thr)
    type_OK = (prm2 >= gender_thr) & (prm_dark >= dark_thr)

    print(f"  Gender threshold (25th pctile of |g1-g3|): {gender_thr:.4f}")
    print(f"  Dark threshold (25th pctile of |g_dark|):  {dark_thr:.4f}")

    error_types_data = {}
    for tname, mask in [("type_G", type_G), ("type_D", type_D),
                        ("type_GD", type_GD), ("type_OK", type_OK)]:
        n_tot = int(mask.sum())
        n_err = int((mask & error_np).sum())
        rate = n_err / n_tot * 100 if n_tot > 0 else 0.0
        error_types_data[tname] = {"n_total": n_tot, "n_errors": n_err, "error_rate": float(rate)}

    label_map = {"type_G": "Type G (gender fail)", "type_D": "Type D (dark fail)",
                 "type_GD": "Type GD (both fail)", "type_OK": "Type OK (neither)"}
    for tname, d in error_types_data.items():
        print(f"  {label_map[tname]}: {d['n_total']} examples, {d['n_errors']} errors ({d['error_rate']:.1f}% error rate)")

    print(f"\n  Among {n_errors} total errors:")
    for tname, d in error_types_data.items():
        frac = d["n_errors"] / n_errors * 100 if n_errors > 0 else 0
        print(f"    {tname}: {d['n_errors']} ({frac:.1f}%)")

    # ==================================================================
    # Step 9: Cosine checks
    # ==================================================================
    print(f"\n{'='*60}")
    print("[STEP 9] RECEPTOR GEOMETRY")
    cos_r1_r3 = float((r1 @ r3).item())
    cos_r1_dark = float((r1 @ r_dark).item())
    cos_r3_dark = float((r3 @ r_dark).item())
    print(f"  cos(R1, R3)   = {cos_r1_r3:.4f}")
    print(f"  cos(R1, dark) = {cos_r1_dark:.4f}")
    print(f"  cos(R3, dark) = {cos_r3_dark:.4f}")

    # ==================================================================
    # PLOTS
    # ==================================================================
    print(f"\n[PLOTS] Saving to {out_dir} ...")

    # --- Plot 1: Dark receptor token profile ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))

    # Top-20
    top20_tokens = [repr(tok) for _, tok, _ in dark_top[:20]]
    top20_vals = [val for _, _, val in dark_top[:20]]
    ax1.barh(range(19, -1, -1), top20_vals, color="steelblue", edgecolor="black", alpha=0.8)
    ax1.set_yticks(range(19, -1, -1))
    ax1.set_yticklabels(top20_tokens, fontsize=8)
    ax1.set_xlabel("Logit value", fontsize=11)
    ax1.set_title("Top-20 promoted tokens", fontsize=12)

    # Bottom-20
    bot20_tokens = [repr(tok) for _, tok, _ in dark_bot[:20]]
    bot20_vals = [val for _, _, val in dark_bot[:20]]
    ax2.barh(range(19, -1, -1), bot20_vals, color="indianred", edgecolor="black", alpha=0.8)
    ax2.set_yticks(range(19, -1, -1))
    ax2.set_yticklabels(bot20_tokens, fontsize=8)
    ax2.set_xlabel("Logit value", fontsize=11)
    ax2.set_title("Top-20 suppressed tokens", fontsize=12)

    fig.suptitle("L11H1sv1 (Dark Receptor) — Logit Space Profile\n"
                 f"Gender token ranks: ' he'={he_rank}, ' she'={she_rank}, "
                 f"' him'={him_rank}, ' her'={her_rank} (out of {vocab_size})",
                 fontsize=13)
    plt.tight_layout()
    p1 = str(out_dir / "dark_plot1_token_profile.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  [1] {p1}")

    # --- Plot 2: 2D Error Landscape (THE MONEY FIGURE) ---
    fig, ax = plt.subplots(figsize=(10, 8))
    # Correct examples: blue, small
    ax.scatter(prm2[correct_np], prm_dark[correct_np],
               c="royalblue", s=15, alpha=0.3, label="Correct", zorder=2)
    # Error examples: red, larger
    ax.scatter(prm2[error_np], prm_dark[error_np],
               c="red", s=60, alpha=0.85, edgecolors="darkred", linewidths=0.5,
               label=f"Error (n={n_errors})", zorder=3)
    # Threshold lines
    ax.axvline(x=gender_thr, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    ax.axhline(y=dark_thr, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
    # Quadrant labels
    xmax = ax.get_xlim()[1]
    ymax = ax.get_ylim()[1]
    ax.text(gender_thr * 0.3, ymax * 0.9, "G fail", fontsize=12, fontweight="bold",
            color="gray", ha="center", va="top")
    ax.text(xmax * 0.7, dark_thr * 0.3, "D fail", fontsize=12, fontweight="bold",
            color="gray", ha="center", va="center")
    ax.text(gender_thr * 0.3, dark_thr * 0.3, "GD fail", fontsize=12, fontweight="bold",
            color="gray", ha="center", va="center")
    ax.text(xmax * 0.7, ymax * 0.9, "OK", fontsize=12, fontweight="bold",
            color="gray", ha="center", va="top")
    ax.set_xlabel("|g₁ − g₃| (Gender Margin)", fontsize=12)
    ax.set_ylabel("|g_dark| (Dark Receptor Activation)", fontsize=12)
    ax.set_title("Error Landscape: Gender Margin vs Dark Receptor", fontsize=14)
    ax.legend(fontsize=11, loc="upper right")
    plt.tight_layout()
    p2 = str(out_dir / "dark_plot2_error_landscape.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  [2] {p2}")

    # --- Plot 3: Precision@k Curves ---
    fig, ax = plt.subplots(figsize=(10, 7))
    ks = np.arange(1, N + 1)
    base_rate = n_errors / N

    ax.plot(ks, prec_curves["PRM2"], color="red", linewidth=1.5,
            label=f"PRM₂ (gender only) AUC={auc_prm2:.3f}", alpha=0.8)
    ax.plot(ks, prec_curves["PRM3"], color="blue", linewidth=1.5,
            label=f"PRM₃ (gender × dark) AUC={auc_prm3:.3f}", alpha=0.8)
    ax.plot(ks, prec_curves["PRM_dark"], color="green", linewidth=1.5,
            label=f"PRM_dark (dark only) AUC={auc_dark_abs:.3f}", alpha=0.8)
    ax.axhline(y=base_rate, color="gray", linestyle=":", linewidth=1.5,
               label=f"Base error rate ({base_rate*100:.1f}%)")
    ax.axvline(x=n_errors, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.text(n_errors + 5, 0.9, f"n_errors={n_errors}", fontsize=9, color="gray")
    ax.set_xlabel("k (examples flagged)", fontsize=12)
    ax.set_ylabel("Precision@k", fontsize=12)
    ax.set_title("Error Prediction: Precision@k", fontsize=14)
    ax.legend(fontsize=10, loc="upper right")
    ax.set_xlim(0, N)
    ax.set_ylim(0, 1.02)
    plt.tight_layout()
    p3 = str(out_dir / "dark_plot3_precision_at_k.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"  [3] {p3}")

    # --- Plot 4: Error Type Distribution ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    type_names_short = ["G fail", "D fail", "GD fail", "OK"]
    type_keys = ["type_G", "type_D", "type_GD", "type_OK"]
    totals = [error_types_data[k]["n_total"] for k in type_keys]
    n_errs = [error_types_data[k]["n_errors"] for k in type_keys]
    n_corrects = [t - e for t, e in zip(totals, n_errs)]
    rates = [error_types_data[k]["error_rate"] for k in type_keys]
    colors_bar = ["#ff9999", "#9999ff", "#cc66cc", "#99cc99"]

    # Left: stacked bar — total examples with error overlay
    x_pos = np.arange(len(type_names_short))
    bars_c = ax1.bar(x_pos, n_corrects, color=colors_bar, edgecolor="black", alpha=0.6, label="Correct")
    bars_e = ax1.bar(x_pos, n_errs, bottom=n_corrects, color="red", edgecolor="black", alpha=0.8, label="Error")
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(type_names_short, fontsize=11)
    ax1.set_ylabel("Number of examples", fontsize=11)
    ax1.set_title("Examples by Failure Type", fontsize=12)
    ax1.legend(fontsize=10)
    for i, (t, e) in enumerate(zip(totals, n_errs)):
        ax1.text(i, t + 2, f"{t}\n({e} err)", ha="center", fontsize=9)

    # Right: error rate per type
    bar_colors = ["indianred" if r > base_rate * 100 else "steelblue" for r in rates]
    ax2.bar(x_pos, rates, color=bar_colors, edgecolor="black", alpha=0.8)
    ax2.axhline(y=base_rate * 100, color="gray", linestyle=":", linewidth=1.5,
                label=f"Overall ({base_rate*100:.1f}%)")
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(type_names_short, fontsize=11)
    ax2.set_ylabel("Error Rate (%)", fontsize=11)
    ax2.set_title("Error Rate by Failure Type", fontsize=12)
    ax2.legend(fontsize=10)
    for i, r in enumerate(rates):
        ax2.text(i, r + 0.5, f"{r:.1f}%", ha="center", fontsize=10, fontweight="bold")

    fig.suptitle("Error Types by Receptor Failure Mode", fontsize=14)
    plt.tight_layout()
    p4 = str(out_dir / "dark_plot4_error_types.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"  [4] {p4}")

    # --- Plot 5: L11H1 Attention Pattern ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    positions = np.arange(most_common_len)
    # Truncate token labels to fit
    tok_labels = [s.strip() if len(s.strip()) <= 8 else s.strip()[:7] + "…"
                  for s in rep_token_strs]

    ax1.bar(positions, avg_attn, color="steelblue", edgecolor="black", alpha=0.8)
    ax1.set_xticks(positions)
    ax1.set_xticklabels(tok_labels, rotation=60, ha="right", fontsize=7)
    ax1.set_ylabel("Attention weight", fontsize=11)
    ax1.set_title(f"Average attention (n={most_common_count})", fontsize=12)
    # Mark last position (decision pos)
    ax1.axvline(x=most_common_len - 1, color="red", linestyle="--", alpha=0.5, linewidth=1)
    ax1.text(most_common_len - 1.5, ax1.get_ylim()[1] * 0.95, "decision\npos", fontsize=8, color="red", ha="right")

    # Split by correct/error
    width = 0.35
    n_err_common = int(error_common.sum())
    if n_err_common > 0:
        ax2.bar(positions - width / 2, avg_attn_correct, width, color="royalblue",
                edgecolor="black", alpha=0.7, label=f"Correct (n={int(correct_common.sum())})")
        ax2.bar(positions + width / 2, avg_attn_error, width, color="red",
                edgecolor="black", alpha=0.7, label=f"Error (n={n_err_common})")
    else:
        ax2.bar(positions, avg_attn_correct, color="royalblue", edgecolor="black", alpha=0.7,
                label=f"Correct (n={int(correct_common.sum())})")
        ax2.text(0.5, 0.5, "No errors at this seq length", transform=ax2.transAxes,
                 ha="center", fontsize=12, color="gray")
    ax2.set_xticks(positions)
    ax2.set_xticklabels(tok_labels, rotation=60, ha="right", fontsize=7)
    ax2.set_ylabel("Attention weight", fontsize=11)
    ax2.set_title("Correct vs Error", fontsize=12)
    ax2.legend(fontsize=9)

    fig.suptitle(f"L11H1 Attention at Decision Position (seq_len={most_common_len})", fontsize=14)
    plt.tight_layout()
    p5 = str(out_dir / "dark_plot5_attention.png")
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    print(f"  [5] {p5}")

    # --- Plot 6: Activation distributions ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    alpha_h = 0.6
    bins = 40

    ax1.hist(g1[he_mask], bins=bins, color="blue", alpha=alpha_h, label="he-context", density=True)
    ax1.hist(g1[she_mask], bins=bins, color="red", alpha=alpha_h, label="she-context", density=True)
    ax1.set_xlabel("g₁ (R1 activation)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title("R1 (gender-aligned, AUC=0.964)", fontsize=11)
    ax1.legend(fontsize=9)

    ax2.hist(g3[he_mask], bins=bins, color="blue", alpha=alpha_h, label="he-context", density=True)
    ax2.hist(g3[she_mask], bins=bins, color="red", alpha=alpha_h, label="she-context", density=True)
    ax2.set_xlabel("g₃ (R3 activation)", fontsize=11)
    ax2.set_title("R3 (gender-aligned, AUC=0.958)", fontsize=11)
    ax2.legend(fontsize=9)

    ax3.hist(g_dark[he_mask], bins=bins, color="blue", alpha=alpha_h, label="he-context", density=True)
    ax3.hist(g_dark[she_mask], bins=bins, color="red", alpha=alpha_h, label="she-context", density=True)
    ax3.set_xlabel("g_dark (dark receptor activation)", fontsize=11)
    ax3.set_title("Dark (NOT gender-aligned, AUC=0.526)", fontsize=11)
    ax3.legend(fontsize=9)

    fig.suptitle("Receptor Activation Distributions by Gender Context", fontsize=14)
    plt.tight_layout()
    p6 = str(out_dir / "dark_plot6_distributions.png")
    fig.savefig(p6, dpi=150)
    plt.close(fig)
    print(f"  [6] {p6}")

    # ==================================================================
    # Save JSON
    # ==================================================================
    results_json = {
        "baseline_accuracy": float(baseline_acc),
        "n_examples": int(N),
        "n_errors": int(n_errors),
        "receptor_cosines": {
            "r1_r3": float(cos_r1_r3),
            "r1_dark": float(cos_r1_dark),
            "r3_dark": float(cos_r3_dark),
        },
        "dark_token_profile": {
            "top20": [{"token": tok, "logit": float(val)} for _, tok, val in dark_top[:20]],
            "bottom20": [{"token": tok, "logit": float(val)} for _, tok, val in dark_bot[:20]],
            "gender_ranks": {
                "he": int(he_rank), "she": int(she_rank),
                "him": int(him_rank), "her": int(her_rank),
            },
        },
        "error_prediction_auc": {
            "prm2": float(auc_prm2),
            "prm3": float(auc_prm3),
            "dark_only": float(auc_dark_only),
            "dark_neg_g": float(auc_dark_neg_g),
            "dark_pos_g": float(auc_dark_pos_g),
            "dark_abs": float(auc_dark_abs),
            "best_dark_predictor": best_dark_name,
            "improvement_prm3_vs_prm2": float(auc_prm3 - auc_prm2),
        },
        "error_types": error_types_data,
        "dark_activation_stats": {
            "he_context": {"mean": float(g_dark[he_mask].mean()), "std": float(g_dark[he_mask].std())},
            "she_context": {"mean": float(g_dark[she_mask].mean()), "std": float(g_dark[she_mask].std())},
            "correct": {"mean": float(g_dark[correct_np].mean()), "std": float(g_dark[correct_np].std())},
            "error": {"mean": float(g_dark[error_np].mean()), "std": float(g_dark[error_np].std())},
        },
        "correlation_dark_abs_vs_seq_length": float(corr_length),
    }

    json_path = str(out_dir / "dark_receptor_prm_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[SAVE] {json_path}")

    # ==================================================================
    # Sanity Checks
    # ==================================================================
    print(f"\n{'='*60}")
    print("[SANITY CHECKS]")
    print(f"  1. Baseline accuracy: {baseline_acc*100:.2f}% (expected ~91.34%), errors={n_errors} (expected ~54)")

    ortho_pass = abs(cos_r1_dark) < 0.1 and abs(cos_r3_dark) < 0.1
    print(f"  2. cos(R1,dark)={cos_r1_dark:.4f}, cos(R3,dark)={cos_r3_dark:.4f} "
          f"— {'PASS' if ortho_pass else 'CHECK'} (expected < 0.1)")

    # Check R1/R3 top tokens are gendered
    r1_top_strs = {tok.strip().lower() for _, tok, _ in r1_top}
    r3_top_strs = {tok.strip().lower() for _, tok, _ in r3_top}
    r1_gendered = any(g in r1_top_strs for g in ["he", "she", "him", "her", "his", "himself", "herself"])
    r3_gendered = any(g in r3_top_strs for g in ["he", "she", "him", "her", "his", "himself", "herself"])
    print(f"  3. R1 top-10 has gendered tokens: {'PASS' if r1_gendered else 'CHECK'}")
    print(f"     R3 top-10 has gendered tokens: {'PASS' if r3_gendered else 'CHECK'}")

    # Dark NOT gendered
    dark_top_strs = {tok.strip().lower() for _, tok, _ in dark_top[:10]}
    dark_gendered = any(g in dark_top_strs for g in ["he", "she", "him", "her"])
    print(f"  4. Dark top-10 NOT gendered: {'PASS' if not dark_gendered else 'CHECK'}")

    print(f"  5. PRM2 AUC={auc_prm2:.4f} — {'PASS' if auc_prm2 > 0.5 else 'CHECK'} (expected > 0.5)")

    ok_rate = error_types_data["type_OK"]["error_rate"]
    min_other = min(error_types_data[k]["error_rate"] for k in ["type_G", "type_D", "type_GD"])
    print(f"  6. Type OK error rate={ok_rate:.1f}% — {'PASS' if ok_rate <= min_other + 5 else 'CHECK'} (expected lowest)")

    dark_he = g_dark[he_mask].mean()
    dark_she = g_dark[she_mask].mean()
    dark_overlap = abs(dark_he - dark_she) / max(g_dark[he_mask].std(), g_dark[she_mask].std(), 1e-6)
    print(f"  7. Dark gender separation: d'={dark_overlap:.4f} — {'PASS' if dark_overlap < 0.5 else 'CHECK'} (expected small)")

    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
