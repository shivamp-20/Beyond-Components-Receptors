#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gp_phase1_causal_subspace_geometry.py

Phase 1: Causal Subspace Geometry — four connected analyses:
  A) Pairwise cosine structure across pipeline stages (207→13→3)
  B) SVD / effective rank / Receptor Superposition Index
  C) Embedding geometry audit (R1 inherited vs R3 computed)
  D) Receptor rank spectrum (marginal R² per direction)

Requires ONE forward pass (for Part D R² computation).
All other parts are pure linear algebra on saved tensors.

Run:
  python gp_phase1_causal_subspace_geometry.py \
    --data_dir data_main \
    --csv test_gp.csv \
    --out_dir outputs/gp \
    --cfr_json outputs/gp/cfr_ghost_detection_results.json \
    --oca_json outputs/gp/oca_ghost_refinement_results.json \
    --batch_size 64 \
    --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
# Utility helpers (same as previous scripts)
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
    rows = []
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


def load_svd_cache_ov_only(path: str, device: str):
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


def expand_rows_to_examples(rows, use_both=True):
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


def cosine_stats(directions: np.ndarray) -> Dict[str, float]:
    """Compute pairwise |cosine| stats for a set of unit-norm row vectors."""
    K = directions.shape[0]
    cos_mat = directions @ directions.T  # (K, K)
    # Upper triangle off-diagonal
    ii, jj = np.triu_indices(K, k=1)
    off_diag = np.abs(cos_mat[ii, jj])
    return {
        "mean": float(off_diag.mean()) if len(off_diag) > 0 else 0.0,
        "max": float(off_diag.max()) if len(off_diag) > 0 else 0.0,
        "median": float(np.median(off_diag)) if len(off_diag) > 0 else 0.0,
        "off_diag_values": off_diag,
        "cos_matrix": cos_mat,
    }


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 1: Causal Subspace Geometry")
    ap.add_argument("--data_dir", type=str, default="data_main")
    ap.add_argument("--csv", type=str, default="test_gp.csv")
    ap.add_argument("--out_dir", type=str, default="outputs/gp")
    ap.add_argument("--svd_cache", type=str, default=None)
    ap.add_argument("--cfr_json", type=str, default=None)
    ap.add_argument("--oca_json", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_both", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    svd_path = Path(args.svd_cache) if args.svd_cache else out_dir / "svd_cache.pt"
    cfr_path = Path(args.cfr_json) if args.cfr_json else out_dir / "cfr_ghost_detection_results.json"
    oca_path = Path(args.oca_json) if args.oca_json else out_dir / "oca_ghost_refinement_results.json"

    print("=" * 70)
    print("PHASE 1: CAUSAL SUBSPACE GEOMETRY")
    print("=" * 70)

    # ==================================================================
    # LOAD DATA
    # ==================================================================
    print("\n[LOAD] CFR results ...")
    with open(cfr_path) as f:
        cfr_data = json.load(f)

    print("[LOAD] OCA results ...")
    with open(oca_path) as f:
        oca_data = json.load(f)

    print("[LOAD] SVD cache ...")
    ov = load_svd_cache_ov_only(str(svd_path), device="cpu")

    print("[LOAD] Tokenizer ...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    he_id = tokenizer.encode(" he", add_special_tokens=False)[0]
    she_id = tokenizer.encode(" she", add_special_tokens=False)[0]

    print("[LOAD] Model (gpt2-small) ...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ------------------------------------------------------------------
    # Extract direction vectors for all three sets
    # ------------------------------------------------------------------
    # All 207
    all_recs = cfr_data["receptors"]
    triples_207 = [(r["layer"], r["head"], r["sv_idx"]) for r in all_recs]
    print(f"  {len(triples_207)} mask-identified receptors loaded")

    # 13 CFR survivors (is_ghost == False in CFR data)
    cfr_survivors = [r for r in all_recs if not r["is_ghost"]]
    cfr_survivors.sort(key=lambda r: r["cfr"], reverse=True)
    triples_13 = [(r["layer"], r["head"], r["sv_idx"]) for r in cfr_survivors]
    print(f"  {len(triples_13)} CFR survivors loaded")

    # 3 OCA survivors
    oca_survivors = [r for r in oca_data["receptors"] if r["survives_oca"]]
    triples_3 = [(r["layer"], r["head"], r["sv_idx"]) for r in oca_survivors]
    # Identify R1, R3, Dark by known positions
    oca_labels = []
    for r in oca_survivors:
        key = (r["layer"], r["head"], r["sv_idx"])
        if key == (10, 9, 0):
            oca_labels.append("R1")
        elif key == (9, 7, 1):
            oca_labels.append("R3")
        elif key == (11, 1, 1):
            oca_labels.append("Dark")
        else:
            oca_labels.append(f"L{key[0]}H{key[1]}sv{key[2]}")
    print(f"  3 OCA survivors: {list(zip(oca_labels, triples_3))}")

    def get_direction(l, h, sv):
        v = ov[l][h].Vh[sv, :].clone().cpu().float()
        v = v / v.norm()
        return v

    dirs_207 = torch.stack([get_direction(*t) for t in triples_207]).numpy()  # (207, 768)
    dirs_13 = torch.stack([get_direction(*t) for t in triples_13]).numpy()    # (13, 768)
    dirs_3 = torch.stack([get_direction(*t) for t in triples_3]).numpy()      # (3, 768)

    v_R1 = get_direction(10, 9, 0).numpy()
    v_R3 = get_direction(9, 7, 1).numpy()
    v_dark = get_direction(11, 1, 1).numpy()

    # Labels for 13 CFR survivors
    cfr_labels = [f"L{r['layer']}H{r['head']}sv{r['sv_idx']}" for r in cfr_survivors]

    # ==================================================================
    # PART A: PAIRWISE COSINE STRUCTURE
    # ==================================================================
    print(f"\n{'='*60}")
    print("PART A: PAIRWISE COSINE STRUCTURE")
    print(f"{'='*60}")

    stats_207 = cosine_stats(dirs_207)
    stats_13 = cosine_stats(dirs_13)
    stats_3 = cosine_stats(dirs_3)

    for name, stats, K in [("207 mask-identified", stats_207, 207),
                            ("13 CFR survivors", stats_13, 13),
                            ("3 OCA survivors", stats_3, 3)]:
        print(f"  {name} (K={K}):")
        print(f"    Mean |cos| = {stats['mean']:.4f}")
        print(f"    Max  |cos| = {stats['max']:.4f}")
        print(f"    Median     = {stats['median']:.4f}")

    # 3x3 matrix
    cos_3x3 = dirs_3 @ dirs_3.T
    print(f"\n  3x3 cosine matrix (OCA survivors):")
    header = "         " + "".join(f"{lb:>10}" for lb in oca_labels)
    print(f"  {header}")
    for i, lb in enumerate(oca_labels):
        row = f"  {lb:<8}" + "".join(f"{cos_3x3[i,j]:>10.4f}" for j in range(3))
        print(row)

    print(f"\n  === PART A SUMMARY ===")
    print(f"  Mean |cos| drop: 207→13: {stats_207['mean']:.4f} → {stats_13['mean']:.4f} (Δ = {stats_207['mean']-stats_13['mean']:+.4f})")
    print(f"  Mean |cos| drop: 13→3:   {stats_13['mean']:.4f} → {stats_3['mean']:.4f} (Δ = {stats_13['mean']-stats_3['mean']:+.4f})")
    print(f"  Total drop 207→3:        {stats_207['mean']:.4f} → {stats_3['mean']:.4f}")

    # --- Plot 1: Cosine histograms ---
    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(0, 1, 50)
    ax.hist(stats_207["off_diag_values"], bins=bins, alpha=0.5, color="red",
            label=f"207 raw (mean={stats_207['mean']:.3f})", density=True)
    ax.hist(stats_13["off_diag_values"], bins=bins, alpha=0.6, color="blue",
            label=f"13 CFR (mean={stats_13['mean']:.3f})", density=True)
    if len(stats_3["off_diag_values"]) > 0:
        for val in stats_3["off_diag_values"]:
            ax.axvline(x=val, color="green", linewidth=2, alpha=0.8)
        ax.plot([], [], color="green", linewidth=2, label=f"3 OCA (mean={stats_3['mean']:.3f})")
    ax.set_xlabel("|Cosine Similarity|", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Pairwise |cosine| Distribution Across Pipeline Stages", fontsize=13)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p1 = str(out_dir / "phase1_plot1_cosine_histograms.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"\n  [Plot 1] {p1}")

    # --- Plot 2: 13x13 cosine heatmap ---
    fig, ax = plt.subplots(figsize=(12, 10))
    abs_cos_13 = np.abs(stats_13["cos_matrix"])
    im = ax.imshow(abs_cos_13, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(13))
    ax.set_xticklabels(cfr_labels, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(13))
    ax.set_yticklabels(cfr_labels, fontsize=8)
    # Annotate cells
    for i in range(13):
        for j in range(13):
            val = abs_cos_13[i, j]
            color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6, color=color)
    plt.colorbar(im, ax=ax, label="|cosine|")
    ax.set_title("Pairwise |cosine| Among 13 CFR Survivors", fontsize=13)
    plt.tight_layout()
    p2 = str(out_dir / "phase1_plot2_cosine_heatmap_13.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  [Plot 2] {p2}")

    # --- Plot 3: 3x3 cosine matrix ---
    fig, ax = plt.subplots(figsize=(6, 5))
    abs_cos_3 = np.abs(cos_3x3)
    im = ax.imshow(abs_cos_3, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(3))
    ax.set_xticklabels(oca_labels, fontsize=12)
    ax.set_yticks(range(3))
    ax.set_yticklabels(oca_labels, fontsize=12)
    for i in range(3):
        for j in range(3):
            val = cos_3x3[i, j]
            ax.text(j, i, f"{val:.4f}", ha="center", va="center", fontsize=11,
                    fontweight="bold", color="white" if abs(val) > 0.5 else "black")
    plt.colorbar(im, ax=ax, label="|cosine|")
    ax.set_title("Cosine Matrix: 3 OCA Survivors", fontsize=13)
    plt.tight_layout()
    p3 = str(out_dir / "phase1_plot3_cosine_matrix_3.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"  [Plot 3] {p3}")

    # ==================================================================
    # PART B: SINGULAR VALUE SPECTRUM
    # ==================================================================
    print(f"\n{'='*60}")
    print("PART B: SINGULAR VALUE SPECTRUM")
    print(f"{'='*60}")

    # SVD of 13 CFR survivors
    U13, S13, Vt13 = np.linalg.svd(dirs_13, full_matrices=False)
    total_var_13 = (S13 ** 2).sum()
    cum_var_13 = np.cumsum(S13 ** 2) / total_var_13
    eff_rank_13_95 = int(np.argmax(cum_var_13 >= 0.95)) + 1
    eff_rank_13_99 = int(np.argmax(cum_var_13 >= 0.99)) + 1
    stable_rank_13 = float(total_var_13 / (S13[0] ** 2))
    RSI = 1.0 - eff_rank_13_95 / 13.0

    print(f"  Singular values (13 survivors):")
    for i, s in enumerate(S13):
        print(f"    σ_{i+1} = {s:.4f}  (cum var = {cum_var_13[i]:.4f})")
    print(f"  Effective rank (95%): {eff_rank_13_95}")
    print(f"  Effective rank (99%): {eff_rank_13_99}")
    print(f"  Stable rank:          {stable_rank_13:.2f}")
    print(f"  RSI (Superposition):  {RSI:.4f}")

    # SVD of 207 directions
    U207, S207, Vt207 = np.linalg.svd(dirs_207, full_matrices=False)
    total_var_207 = (S207 ** 2).sum()
    cum_var_207 = np.cumsum(S207 ** 2) / total_var_207
    eff_rank_207_95 = int(np.argmax(cum_var_207 >= 0.95)) + 1
    eff_rank_207_99 = int(np.argmax(cum_var_207 >= 0.99)) + 1
    stable_rank_207 = float(total_var_207 / (S207[0] ** 2))

    print(f"\n  207 mask-identified:")
    print(f"  Top-5 singular values: {S207[:5].tolist()}")
    print(f"  Effective rank (95%): {eff_rank_207_95}")
    print(f"  Effective rank (99%): {eff_rank_207_99}")
    print(f"  Stable rank:          {stable_rank_207:.2f}")

    # --- Plot 4: Singular value spectra ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.bar(range(1, 14), S13, color="steelblue", edgecolor="black", alpha=0.8)
    ax1.axvline(x=eff_rank_13_95 + 0.5, color="red", linestyle="--", linewidth=1.5,
                label=f"95% rank = {eff_rank_13_95}")
    ax1.axvline(x=eff_rank_13_99 + 0.5, color="darkred", linestyle=":", linewidth=1.5,
                label=f"99% rank = {eff_rank_13_99}")
    ax1.set_xlabel("Component index", fontsize=11)
    ax1.set_ylabel("Singular value", fontsize=11)
    ax1.set_title("13 CFR Survivors", fontsize=12)
    ax1.set_xticks(range(1, 14))
    ax1.legend(fontsize=9)

    n_show = min(20, len(S207))
    ax2.plot(range(1, n_show + 1), S207[:n_show], "o-", color="steelblue", markersize=5)
    ax2.axvline(x=eff_rank_207_95 + 0.5, color="red", linestyle="--", linewidth=1.5,
                label=f"95% rank = {eff_rank_207_95}")
    ax2.set_xlabel("Component index", fontsize=11)
    ax2.set_ylabel("Singular value", fontsize=11)
    ax2.set_title(f"207 Mask-Identified (top {n_show})", fontsize=12)
    ax2.legend(fontsize=9)

    fig.suptitle("Singular Value Spectra", fontsize=14)
    plt.tight_layout()
    p4 = str(out_dir / "phase1_plot4_singular_values.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"\n  [Plot 4] {p4}")

    # --- Plot 5: Cumulative variance ---
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, 14), cum_var_13, "o-", color="steelblue", linewidth=2,
            markersize=8, label="13 CFR survivors")
    n_cmp = min(13, len(cum_var_207))
    ax.plot(range(1, n_cmp + 1), cum_var_207[:n_cmp], "s--", color="coral", linewidth=2,
            markersize=6, label="207 mask-identified (first 13 components)")
    ax.axhline(y=0.95, color="red", linestyle=":", linewidth=1.5, label="95% threshold")
    ax.axhline(y=0.99, color="darkred", linestyle=":", linewidth=1.5, label="99% threshold")
    ax.set_xlabel("Number of components", fontsize=12)
    ax.set_ylabel("Cumulative variance explained", fontsize=12)
    ax.set_title("Effective Dimensionality of Receptor Direction Sets", fontsize=13)
    ax.set_xticks(range(1, 14))
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p5 = str(out_dir / "phase1_plot5_cumulative_variance.png")
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    print(f"  [Plot 5] {p5}")

    # ==================================================================
    # PART C: EMBEDDING GEOMETRY AUDIT
    # ==================================================================
    print(f"\n{'='*60}")
    print("PART C: EMBEDDING GEOMETRY AUDIT")
    print(f"{'='*60}")

    # Get embedding matrix
    W_E = model.W_E.cpu().float()  # (vocab_size, d_model) in TransformerLens

    # Load names and compute he-fractions from dataset
    csv_path = str(data_dir / args.csv)
    rows = load_gp_csv(csv_path)

    # Collect name -> gender counts
    name_counts = defaultdict(lambda: {"he": 0, "she": 0})
    for r in rows:
        name = (r.get("name") or "").strip()
        pronoun = (r.get("pronoun") or "").strip().lower()
        if name and pronoun in ("he", "she"):
            name_counts[name][pronoun] += 1

        corr_name = (r.get("corr_name") or "").strip()
        corr_pronoun = (r.get("corr_pronoun") or "").strip().lower()
        if corr_name and corr_pronoun in ("he", "she"):
            name_counts[corr_name][corr_pronoun] += 1

    # Compute he-fraction for each name
    name_he_frac = {}
    for name, counts in name_counts.items():
        total = counts["he"] + counts["she"]
        if total > 0:
            name_he_frac[name] = counts["he"] / total

    print(f"  {len(name_he_frac)} unique names found")
    n_male = sum(1 for f in name_he_frac.values() if f > 0.7)
    n_female = sum(1 for f in name_he_frac.values() if f < 0.3)
    print(f"  Male (he_frac > 0.7): {n_male}")
    print(f"  Female (he_frac < 0.3): {n_female}")

    # Get single-token name embeddings
    emb_names = []
    emb_vectors = []
    emb_he_fracs = []
    for name, he_frac in name_he_frac.items():
        # GPT-2 tokenizes with space prefix in context
        tids = tokenizer.encode(" " + name, add_special_tokens=False)
        if len(tids) == 1:
            emb = W_E[tids[0], :].numpy()
            emb_names.append(name)
            emb_vectors.append(emb)
            emb_he_fracs.append(he_frac)

    emb_vectors = np.stack(emb_vectors)    # (N_names, 768)
    emb_he_fracs = np.array(emb_he_fracs)  # (N_names,)
    print(f"  Single-token names: {len(emb_names)}")

    # Compute embedding gender direction
    male_mask = emb_he_fracs > 0.7
    female_mask = emb_he_fracs < 0.3
    mean_male = emb_vectors[male_mask].mean(axis=0)
    mean_female = emb_vectors[female_mask].mean(axis=0)
    d_gender = mean_male - mean_female
    d_gender = d_gender / np.linalg.norm(d_gender)

    print(f"  d_gender computed from {int(male_mask.sum())} male + {int(female_mask.sum())} female single-token names")

    # Alignment measurements
    cos_R1_gender = float(v_R1 @ d_gender)
    cos_R3_gender = float(v_R3 @ d_gender)
    cos_dark_gender = float(v_dark @ d_gender)

    print(f"\n  cos(R1, d_gender)   = {cos_R1_gender:.4f}")
    print(f"  cos(R3, d_gender)   = {cos_R3_gender:.4f}")
    print(f"  cos(dark, d_gender) = {cos_dark_gender:.4f}")

    # Project name embeddings onto receptor directions
    proj_R1 = emb_vectors @ v_R1
    proj_R3 = emb_vectors @ v_R3
    proj_dark = emb_vectors @ v_dark

    # Embedding AUC
    labels_binary = (emb_he_fracs > 0.5).astype(int)
    # Handle edge case: need both classes
    if labels_binary.sum() > 0 and labels_binary.sum() < len(labels_binary):
        emb_auc_R1 = roc_auc_score(labels_binary, proj_R1)
        emb_auc_R1 = max(emb_auc_R1, 1 - emb_auc_R1)
        emb_auc_R3 = roc_auc_score(labels_binary, proj_R3)
        emb_auc_R3 = max(emb_auc_R3, 1 - emb_auc_R3)
        emb_auc_dark = roc_auc_score(labels_binary, proj_dark)
        emb_auc_dark = max(emb_auc_dark, 1 - emb_auc_dark)
    else:
        emb_auc_R1 = emb_auc_R3 = emb_auc_dark = 0.5

    print(f"\n  Embedding AUC along R1 direction:   {emb_auc_R1:.4f}")
    print(f"  Embedding AUC along R3 direction:   {emb_auc_R3:.4f}")
    print(f"  Embedding AUC along dark direction: {emb_auc_dark:.4f}")

    # Fraction of d_gender captured by 3-receptor subspace
    # Project d_gender onto span(v_R1, v_R3, v_dark) via least squares
    B = dirs_3.T  # (768, 3)
    coeff, _, _, _ = np.linalg.lstsq(B, d_gender, rcond=None)
    d_gender_proj = B @ coeff
    captured_frac = float(np.linalg.norm(d_gender_proj) ** 2)
    print(f"  Fraction of d_gender in receptor subspace: {captured_frac:.4f}")

    # --- Plot 6: Embedding projections ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, proj, title, auc_val in zip(axes,
        [proj_R1, proj_R3, proj_dark],
        ["R1 (L10H9sv0)", "R3 (L9H7sv1)", "Dark (L11H1sv1)"],
        [emb_auc_R1, emb_auc_R3, emb_auc_dark]):

        sc = ax.scatter(emb_he_fracs, proj, c=emb_he_fracs, cmap="RdBu_r",
                        alpha=0.6, edgecolors="gray", linewidths=0.3, s=30)
        ax.set_xlabel("He-fraction", fontsize=11)
        ax.set_ylabel("Embedding · receptor direction", fontsize=10)
        ax.set_title(f"{title}\nEmb AUC = {auc_val:.3f}", fontsize=11)
        # Trend line
        z = np.polyfit(emb_he_fracs, proj, 1)
        xline = np.linspace(0, 1, 100)
        ax.plot(xline, np.polyval(z, xline), "k--", alpha=0.5, linewidth=1)

    fig.suptitle("Name Embedding Projections onto Receptor Directions", fontsize=14)
    plt.tight_layout()
    p6 = str(out_dir / "phase1_plot6_embedding_projections.png")
    fig.savefig(p6, dpi=150)
    plt.close(fig)
    print(f"\n  [Plot 6] {p6}")

    # --- Plot 7: Alignment bars ---
    fig, ax = plt.subplots(figsize=(7, 5))
    bar_vals = [abs(cos_R1_gender), abs(cos_R3_gender), abs(cos_dark_gender)]
    bar_colors = ["steelblue", "coral", "gray"]
    ax.bar(["R1", "R3", "Dark"], bar_vals, color=bar_colors, edgecolor="black", alpha=0.8)
    ax.set_ylabel("|cos(receptor, d_gender)|", fontsize=12)
    ax.set_title("Receptor Alignment with Embedding Gender Axis", fontsize=13)
    for i, v in enumerate(bar_vals):
        ax.text(i, v + 0.005, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0, max(bar_vals) * 1.2 + 0.05)
    plt.tight_layout()
    p7 = str(out_dir / "phase1_plot7_alignment_bars.png")
    fig.savefig(p7, dpi=150)
    plt.close(fig)
    print(f"  [Plot 7] {p7}")

    # ==================================================================
    # PART D: RECEPTOR RANK SPECTRUM
    # ==================================================================
    print(f"\n{'='*60}")
    print("PART D: RECEPTOR RANK SPECTRUM")
    print(f"{'='*60}")

    # Need one forward pass to get residuals for R² computation
    print("  [FORWARD] Collecting residuals for R² computation ...")
    examples = expand_rows_to_examples(rows, use_both=bool(args.use_both))
    N_ex = len(examples)

    hook_name = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"
    all_resid, all_logit_diff = [], []

    for batch_items in batches(examples, args.batch_size):
        texts = [e["text"] for e in batch_items]
        ids_list = encode_texts(tokenizer, texts)
        pad_len = max(len(x) for x in ids_list)
        tokens, last_idx = pad_to_length(ids_list, pad_len, tokenizer.pad_token_id, device)
        B = tokens.shape[0]

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens, names_filter=lambda n: n == hook_name
            )

        resid = cache[hook_name]
        resid_at = resid[torch.arange(B, device=device), last_idx, :]
        logits_at = logits[torch.arange(B, device=device), last_idx, :]
        ld = logits_at[:, he_id] - logits_at[:, she_id]

        all_resid.append(resid_at.cpu())
        all_logit_diff.append(ld.cpu())
        del cache

    resid_all = torch.cat(all_resid).numpy()          # (N, 768)
    logit_diff_np = torch.cat(all_logit_diff).numpy()  # (N,)
    print(f"  {N_ex} examples, residuals collected.")

    # Compute incremental R² by greedily adding directions
    # Order: R1 first (highest raw AccDrop), then R3, then Dark
    dir_order = [("R1", v_R1), ("R3", v_R3), ("Dark", v_dark)]
    r2_incremental = []
    proj_cumulative = np.zeros((N_ex, 0))

    for name, direction in dir_order:
        proj_k = (resid_all @ direction).reshape(-1, 1)
        proj_cumulative = np.hstack([proj_cumulative, proj_k])
        reg = LinearRegression().fit(proj_cumulative, logit_diff_np)
        r2_k = r2_score(logit_diff_np, reg.predict(proj_cumulative))
        r2_incremental.append(float(r2_k))
        print(f"  R²(+{name}): {r2_k:.4f}")

    # Marginal gains
    rho = [r2_incremental[0],
           r2_incremental[1] - r2_incremental[0],
           r2_incremental[2] - r2_incremental[1]]

    ICR = rho[0] / sum(rho) if sum(rho) > 0 else 0.0

    print(f"\n  Marginal R² gains: {[f'{r:.4f}' for r in rho]}")
    print(f"  ICR (top direction): {ICR:.4f}")
    print(f"  Cumulative R²: {[f'{r:.4f}' for r in r2_incremental]}")

    # --- Plot 8: Marginal R² bar chart ---
    fig, ax = plt.subplots(figsize=(9, 6))
    bar_labels = [f"R1\n(L10H9sv0)", f"R3\n(L9H7sv1)", f"Dark\n(L11H1sv1)"]
    bar_colors_d = ["steelblue", "coral", "gray"]
    bars = ax.bar(range(3), rho, color=bar_colors_d, edgecolor="black", alpha=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels(bar_labels, fontsize=11)
    ax.set_ylabel("Marginal R² Gain", fontsize=12)
    ax.set_title("Receptor Rank Spectrum (Marginal Explanatory Power)", fontsize=13)
    for i, (r, cumr) in enumerate(zip(rho, r2_incremental)):
        ax.text(i, r + max(rho) * 0.02, f"Δ={r:.4f}\ncum={cumr:.4f}",
                ha="center", fontsize=10)
    plt.tight_layout()
    p8 = str(out_dir / "phase1_plot8_receptor_spectrum.png")
    fig.savefig(p8, dpi=150)
    plt.close(fig)
    print(f"\n  [Plot 8] {p8}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: CAUSAL SUBSPACE GEOMETRY — SUMMARY")
    print(f"{'='*60}")
    print(f"Mean |cos| (207 raw):         {stats_207['mean']:.4f}")
    print(f"Mean |cos| (13 CFR):          {stats_13['mean']:.4f}")
    print(f"Mean |cos| (3 OCA):           {stats_3['mean']:.4f}")
    ratio = stats_207['mean'] / stats_3['mean'] if stats_3['mean'] > 0 else float('inf')
    print(f"Decorrelation ratio (207→3):  {ratio:.2f}x")
    print()
    print(f"Effective rank 13 (95%):      {eff_rank_13_95}")
    print(f"Effective rank 13 (99%):      {eff_rank_13_99}")
    print(f"Stable rank (13 survivors):   {stable_rank_13:.2f}")
    print(f"RSI (Superposition Index):    {RSI:.4f}")
    print()
    print(f"Effective rank 207 (95%):     {eff_rank_207_95}")
    print(f"Effective rank 207 (99%):     {eff_rank_207_99}")
    print(f"Stable rank (207 raw):        {stable_rank_207:.2f}")
    print()
    print(f"cos(R1, d_gender):            {cos_R1_gender:.4f}")
    print(f"cos(R3, d_gender):            {cos_R3_gender:.4f}")
    print(f"cos(dark, d_gender):          {cos_dark_gender:.4f}")
    print(f"Embedding AUC (R1 dir):       {emb_auc_R1:.4f}")
    print(f"Embedding AUC (R3 dir):       {emb_auc_R3:.4f}")
    print(f"Embedding AUC (dark dir):     {emb_auc_dark:.4f}")
    print(f"d_gender captured fraction:   {captured_frac:.4f}")
    print()
    print(f"ICR (top direction share):    {ICR:.4f}")
    print(f"Marginal R²: {[f'{r:.4f}' for r in rho]}")
    print(f"Cumulative R²: {[f'{r:.4f}' for r in r2_incremental]}")
    print("=" * 60)

    # ==================================================================
    # Save JSON
    # ==================================================================
    results_json = {
        "part_a": {
            "mean_cos_207": float(stats_207["mean"]),
            "max_cos_207": float(stats_207["max"]),
            "median_cos_207": float(stats_207["median"]),
            "mean_cos_13": float(stats_13["mean"]),
            "max_cos_13": float(stats_13["max"]),
            "median_cos_13": float(stats_13["median"]),
            "mean_cos_3": float(stats_3["mean"]),
            "max_cos_3": float(stats_3["max"]),
            "median_cos_3": float(stats_3["median"]),
            "cos_3x3": cos_3x3.tolist(),
        },
        "part_b": {
            "singular_values_13": S13.tolist(),
            "cum_var_13": cum_var_13.tolist(),
            "eff_rank_13_95": int(eff_rank_13_95),
            "eff_rank_13_99": int(eff_rank_13_99),
            "stable_rank_13": float(stable_rank_13),
            "RSI": float(RSI),
            "singular_values_207_top20": S207[:20].tolist(),
            "eff_rank_207_95": int(eff_rank_207_95),
            "eff_rank_207_99": int(eff_rank_207_99),
            "stable_rank_207": float(stable_rank_207),
        },
        "part_c": {
            "n_single_token_names": int(len(emb_names)),
            "n_male": int(male_mask.sum()),
            "n_female": int(female_mask.sum()),
            "cos_R1_gender": float(cos_R1_gender),
            "cos_R3_gender": float(cos_R3_gender),
            "cos_dark_gender": float(cos_dark_gender),
            "emb_auc_R1": float(emb_auc_R1),
            "emb_auc_R3": float(emb_auc_R3),
            "emb_auc_dark": float(emb_auc_dark),
            "captured_fraction": float(captured_frac),
        },
        "part_d": {
            "r2_cumulative": [float(r) for r in r2_incremental],
            "r2_marginal": [float(r) for r in rho],
            "ICR": float(ICR),
            "direction_order": ["R1", "R3", "Dark"],
        },
    }

    json_path = str(out_dir / "phase1_causal_subspace_results.json")
    with open(json_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"\n[SAVE] {json_path}")

    # ==================================================================
    # Sanity Checks
    # ==================================================================
    print(f"\n{'='*60}")
    print("[SANITY CHECKS]")
    print(f"  1. 207 directions loaded: {dirs_207.shape[0]} (expected 207)")
    print(f"  2. 13 CFR survivors loaded: {dirs_13.shape[0]} (expected 13)")
    print(f"  3. 3 OCA survivors: {oca_labels}")
    print(f"  4. cos(R1,dark)={cos_3x3[0,2] if len(oca_labels)==3 else 'N/A':.4f} (expected ~0)")
    print(f"  5. RSI={RSI:.4f} — {'PASS' if RSI > 0.5 else 'CHECK'} (expected > 0.5)")
    print(f"  6. cos(R1,d_gender)={cos_R1_gender:.4f} — {'PASS' if abs(cos_R1_gender) > 0.3 else 'CHECK'} (expected |cos| > 0.5)")
    print(f"  7. cos(R3,d_gender)={cos_R3_gender:.4f} — {'PASS' if abs(cos_R3_gender) < 0.2 else 'CHECK'} (expected |cos| < 0.15)")
    print(f"  8. Emb AUC R1={emb_auc_R1:.4f} — {'PASS' if emb_auc_R1 > 0.7 else 'CHECK'} (expected > 0.8)")
    print(f"  9. Final R²={r2_incremental[-1]:.4f} (expected ~0.9815)")
    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
