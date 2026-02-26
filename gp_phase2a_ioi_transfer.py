#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gp_phase2a_ioi_transfer.py

Phase 2A: IOI Zero-Shot Transfer Experiment.

Tests whether GP receptor directions (R1, R3, Dark) carry any information
relevant to the IOI (Indirect Object Identification) task.

Run:
  python gp_phase2a_ioi_transfer.py \
    --data_dir data_main \
    --ioi_csv test_1k_ioi.csv \
    --out_dir outputs/gp \
    --batch_size 16 \
    --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, roc_auc_score
from transformer_lens import HookedTransformer


# =====================================================================
# Utility helpers
# =====================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def safe_auc(labels, scores):
    """Polarity-agnostic AUC: returns max(AUC, 1 - AUC)."""
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    if len(np.unique(labels)) < 2:
        return 0.5
    try:
        auc = roc_auc_score(labels, scores)
        return max(auc, 1 - auc)
    except Exception:
        return 0.5


def batches(n, batch_size):
    """Yield (start, end) index pairs."""
    for i in range(0, n, batch_size):
        yield i, min(i + batch_size, n)


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2A: IOI Zero-Shot Transfer")
    ap.add_argument("--data_dir", type=str, default="data_main")
    ap.add_argument("--ioi_csv", type=str, default="test_1k_ioi.csv")
    ap.add_argument("--out_dir", type=str, default="outputs/gp")
    ap.add_argument("--svd_cache", type=str, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--device", type=str, default="cuda")
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

    print("=" * 70)
    print("PHASE 2A: IOI ZERO-SHOT TRANSFER")
    print("=" * 70)

    # ==================================================================
    # SETUP
    # ==================================================================
    print("\n[LOAD] Model (gpt2-small) ...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    n_layers = model.cfg.n_layers  # 12

    print("[LOAD] SVD cache ...")
    ov = load_svd_cache_ov_only(str(svd_path), device="cpu")

    def get_direction(l, h, sv):
        v = ov[l][h].Vh[sv, :].clone().cpu().float()
        return v / v.norm()

    v_R1 = get_direction(10, 9, 0)    # (768,)
    v_R3 = get_direction(9, 7, 1)
    v_dark = get_direction(11, 1, 1)

    # ==================================================================
    # LOAD IOI DATASET
    # ==================================================================
    print("[LOAD] IOI dataset ...")
    ioi_path = str(data_dir / args.ioi_csv)
    ioi_rows = []
    with open(ioi_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ioi_rows.append(r)

    n_examples = len(ioi_rows)
    print(f"  {n_examples} IOI examples loaded")

    # Tokenize IO and S names
    io_token_ids = []
    s_token_ids = []
    valid_mask = []

    for r in ioi_rows:
        io_name = r["ioi_sentences_labels"].strip()
        s_name = r["ioi_sentences_labels_wrong"].strip()

        io_toks = tokenizer.encode(" " + io_name, add_special_tokens=False)
        s_toks = tokenizer.encode(" " + s_name, add_special_tokens=False)

        io_token_ids.append(io_toks[0])
        s_token_ids.append(s_toks[0])
        valid_mask.append(len(io_toks) == 1 and len(s_toks) == 1)

    valid_count = sum(valid_mask)
    print(f"  Single-token name pairs: {valid_count}/{n_examples}")

    # ==================================================================
    # FORWARD PASSES — collect residuals at all layers
    # ==================================================================
    print("\n[FORWARD] Collecting residuals at all layers + logits ...")

    # Hook names for all layers
    resid_hooks = set()
    for l in range(n_layers):
        resid_hooks.add(f"blocks.{l}.hook_resid_post")

    # Storage
    all_residuals = torch.zeros(n_examples, n_layers, 768)  # layers 0..11
    all_logits_io = torch.zeros(n_examples)
    all_logits_s = torch.zeros(n_examples)
    all_top1_correct = torch.zeros(n_examples, dtype=torch.bool)
    all_io_gt_s = torch.zeros(n_examples, dtype=torch.bool)

    batch_count = 0
    for start, end in batches(n_examples, args.batch_size):
        batch_texts = [ioi_rows[i]["ioi_sentences_input"] for i in range(start, end)]
        B = len(batch_texts)

        # Tokenize
        ids_list = [tokenizer.encode(t, add_special_tokens=False) for t in batch_texts]
        pad_len = max(len(x) for x in ids_list)
        pad_id = tokenizer.eos_token_id

        tokens_t = torch.full((B, pad_len), pad_id, dtype=torch.long, device=device)
        last_idx = torch.empty(B, dtype=torch.long, device=device)
        for i, ids in enumerate(ids_list):
            L = len(ids)
            tokens_t[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
            last_idx[i] = L - 1

        with torch.no_grad():
            logits, cache = model.run_with_cache(
                tokens_t, names_filter=lambda n: n in resid_hooks
            )

        for b in range(B):
            gi = start + b
            pred_pos = last_idx[b]

            # Residuals at each layer
            for l in range(n_layers):
                all_residuals[gi, l] = cache[f"blocks.{l}.hook_resid_post"][b, pred_pos].cpu()

            # Logits at prediction position
            final_logits = logits[b, pred_pos]
            io_tok = io_token_ids[gi]
            s_tok = s_token_ids[gi]

            all_logits_io[gi] = final_logits[io_tok].cpu()
            all_logits_s[gi] = final_logits[s_tok].cpu()
            all_top1_correct[gi] = (final_logits.argmax().item() == io_tok)
            all_io_gt_s[gi] = (final_logits[io_tok] > final_logits[s_tok])

        del logits, cache
        torch.cuda.empty_cache()

        batch_count += 1
        if batch_count % 10 == 0:
            print(f"  Batch {batch_count}/{(n_examples + args.batch_size - 1) // args.batch_size}")

    print(f"  Done. {n_examples} examples processed.")

    # ==================================================================
    # RECEPTOR PROJECTIONS
    # ==================================================================
    print("\n[PROJECTIONS] Computing receptor activations at all layers ...")

    # Shape: (n_examples, n_layers) — dot product at each layer
    g_R1 = (all_residuals @ v_R1).numpy()       # (N, 12)
    g_R3 = (all_residuals @ v_R3).numpy()
    g_dark = (all_residuals @ v_dark).numpy()

    # Final layer = layer 11 (index -1)
    g_R1_final = g_R1[:, -1]
    g_R3_final = g_R3[:, -1]
    g_dark_final = g_dark[:, -1]

    # ==================================================================
    # IOI MODEL PERFORMANCE
    # ==================================================================
    ioi_logit_diff = (all_logits_io - all_logits_s).numpy()
    top1_acc = float(all_top1_correct.float().mean().item())
    ioi_acc = float(all_io_gt_s.float().mean().item())
    mean_ld = float(ioi_logit_diff.mean())

    print(f"\n{'='*60}")
    print("IOI MODEL PERFORMANCE")
    print(f"{'='*60}")
    print(f"  Top-1 accuracy:        {top1_acc*100:.2f}%")
    print(f"  IOI accuracy (IO > S): {ioi_acc*100:.2f}%")
    print(f"  Mean IOI logit diff:   {mean_ld:.4f}")

    # ==================================================================
    # RECEPTOR ACTIVATION STATISTICS
    # ==================================================================
    print(f"\n{'='*60}")
    print("RECEPTOR ACTIVATIONS ON IOI (final layer)")
    print(f"{'='*60}")
    print(f"  R1:   mean={g_R1_final.mean():.4f}, std={g_R1_final.std():.4f}")
    print(f"  R3:   mean={g_R3_final.mean():.4f}, std={g_R3_final.std():.4f}")
    print(f"  Dark: mean={g_dark_final.mean():.4f}, std={g_dark_final.std():.4f}")

    # ==================================================================
    # TRANSFER AUC
    # ==================================================================
    ioi_correct = all_io_gt_s.numpy().astype(int)

    transfer_auc_R1 = safe_auc(ioi_correct, g_R1_final)
    transfer_auc_R3 = safe_auc(ioi_correct, g_R3_final)
    transfer_auc_dark = safe_auc(ioi_correct, g_dark_final)

    print(f"\n{'='*60}")
    print("TRANSFER AUC (g_k predicts IOI correctness)")
    print(f"{'='*60}")
    print(f"  R1:   {transfer_auc_R1:.4f}")
    print(f"  R3:   {transfer_auc_R3:.4f}")
    print(f"  Dark: {transfer_auc_dark:.4f}")

    # ==================================================================
    # CROSS-TASK CORRELATION AND R²
    # ==================================================================
    corr_R1 = float(np.corrcoef(g_R1_final, ioi_logit_diff)[0, 1])
    corr_R3 = float(np.corrcoef(g_R3_final, ioi_logit_diff)[0, 1])
    corr_dark = float(np.corrcoef(g_dark_final, ioi_logit_diff)[0, 1])

    print(f"\n{'='*60}")
    print("CROSS-TASK CORRELATION (g_k vs IOI logit diff)")
    print(f"{'='*60}")
    print(f"  corr(g1, Δlogit_IOI):     {corr_R1:.4f}")
    print(f"  corr(g3, Δlogit_IOI):     {corr_R3:.4f}")
    print(f"  corr(g_dark, Δlogit_IOI): {corr_dark:.4f}")

    # Cross-task R²
    X_all = np.column_stack([g_R1_final, g_R3_final, g_dark_final])
    reg_all = LinearRegression().fit(X_all, ioi_logit_diff)
    ioi_r2 = float(reg_all.score(X_all, ioi_logit_diff))

    # Single-receptor R²
    r2_R1 = float(LinearRegression().fit(g_R1_final.reshape(-1, 1), ioi_logit_diff).score(
        g_R1_final.reshape(-1, 1), ioi_logit_diff))
    r2_R3 = float(LinearRegression().fit(g_R3_final.reshape(-1, 1), ioi_logit_diff).score(
        g_R3_final.reshape(-1, 1), ioi_logit_diff))
    r2_dark = float(LinearRegression().fit(g_dark_final.reshape(-1, 1), ioi_logit_diff).score(
        g_dark_final.reshape(-1, 1), ioi_logit_diff))

    gp_r2 = 0.9815

    print(f"\n{'='*60}")
    print("CROSS-TASK R²")
    print(f"{'='*60}")
    print(f"  GP R²:                  {gp_r2:.4f}")
    print(f"  IOI R² (all 3 recptrs): {ioi_r2:.4f}  (ratio: {ioi_r2/gp_r2:.4f}x)")
    print(f"  IOI R² (R1 only):       {r2_R1:.4f}")
    print(f"  IOI R² (R3 only):       {r2_R3:.4f}")
    print(f"  IOI R² (Dark only):     {r2_dark:.4f}")

    # ==================================================================
    # EMBEDDING PROJECTION CONTROL (GENDER CONFOUND)
    # ==================================================================
    print(f"\n{'='*60}")
    print("EMBEDDING PROJECTION CONTROL (GENDER CONFOUND)")
    print(f"{'='*60}")

    W_E = model.W_E.cpu().float()  # (vocab_size, d_model)
    v_R1_np = v_R1.numpy()
    v_R3_np = v_R3.numpy()
    v_dark_np = v_dark.numpy()

    emb_io = np.stack([W_E[io_token_ids[i]].numpy() for i in range(n_examples)])  # (N, 768)
    emb_s = np.stack([W_E[s_token_ids[i]].numpy() for i in range(n_examples)])

    proj_io_R1 = emb_io @ v_R1_np
    proj_s_R1 = emb_s @ v_R1_np
    proj_io_R3 = emb_io @ v_R3_np
    proj_s_R3 = emb_s @ v_R3_np
    proj_io_dark = emb_io @ v_dark_np
    proj_s_dark = emb_s @ v_dark_np

    # Embedding AUC: can embedding projection separate IO from S names?
    emb_labels = np.concatenate([np.ones(n_examples), np.zeros(n_examples)])
    emb_auc_R1_ioi = safe_auc(emb_labels, np.concatenate([proj_io_R1, proj_s_R1]))
    emb_auc_R3_ioi = safe_auc(emb_labels, np.concatenate([proj_io_R3, proj_s_R3]))
    emb_auc_dark_ioi = safe_auc(emb_labels, np.concatenate([proj_io_dark, proj_s_dark]))

    print(f"  Emb AUC (R1 dir, IO vs S):   {emb_auc_R1_ioi:.4f}")
    print(f"  Emb AUC (R3 dir, IO vs S):   {emb_auc_R3_ioi:.4f}")
    print(f"  Emb AUC (dark dir, IO vs S): {emb_auc_dark_ioi:.4f}")

    # Difference projection correlation
    diff_R1 = proj_io_R1 - proj_s_R1
    diff_R3 = proj_io_R3 - proj_s_R3
    diff_dark = proj_io_dark - proj_s_dark
    corr_emb_R1 = float(np.corrcoef(diff_R1, ioi_logit_diff)[0, 1])
    corr_emb_R3 = float(np.corrcoef(diff_R3, ioi_logit_diff)[0, 1])
    print(f"  corr(emb_proj_diff_R1, IOI logit diff): {corr_emb_R1:.4f}")
    print(f"  corr(emb_proj_diff_R3, IOI logit diff): {corr_emb_R3:.4f}")

    # ==================================================================
    # GENDER CONFOUND CONTROL
    # ==================================================================
    print(f"\n{'='*60}")
    print("GENDER CONFOUND CONTROL")
    print(f"{'='*60}")

    # Use R1 embedding projection as gender proxy (R1 pol=+1 for male)
    all_proj = np.concatenate([proj_io_R1, proj_s_R1])
    median_proj = float(np.median(all_proj))
    io_is_male = proj_io_R1 > median_proj
    s_is_male = proj_s_R1 > median_proj

    same_gender = (io_is_male == s_is_male)
    mixed_gender = ~same_gender

    print(f"  Same-gender pairs:  {int(same_gender.sum())}")
    print(f"  Mixed-gender pairs: {int(mixed_gender.sum())}")

    auc_same = safe_auc(ioi_correct[same_gender], g_R1_final[same_gender]) if same_gender.sum() > 50 else 0.5
    auc_mixed = safe_auc(ioi_correct[mixed_gender], g_R1_final[mixed_gender]) if mixed_gender.sum() > 50 else 0.5

    print(f"  R1 transfer AUC (all):          {transfer_auc_R1:.4f}")
    print(f"  R1 transfer AUC (same-gender):  {auc_same:.4f}")
    print(f"  R1 transfer AUC (mixed-gender): {auc_mixed:.4f}")

    # ==================================================================
    # PER-LAYER COMMITMENT CURVES
    # ==================================================================
    print(f"\n{'='*60}")
    print("PER-LAYER COMMITMENT CURVES (AUC on IOI)")
    print(f"{'='*60}")

    layer_auc_R1 = []
    layer_auc_R3 = []
    layer_auc_dark = []

    for l in range(n_layers):
        layer_auc_R1.append(safe_auc(ioi_correct, g_R1[:, l]))
        layer_auc_R3.append(safe_auc(ioi_correct, g_R3[:, l]))
        layer_auc_dark.append(safe_auc(ioi_correct, g_dark[:, l]))

    for l in range(n_layers):
        print(f"  Layer {l:>2}: R1={layer_auc_R1[l]:.4f}, R3={layer_auc_R3[l]:.4f}, Dark={layer_auc_dark[l]:.4f}")

    # ==================================================================
    # PLOTS
    # ==================================================================
    print(f"\n[PLOTS] Saving to {out_dir} ...")

    # --- Plot 1: Per-layer AUC curves ---
    fig, ax = plt.subplots(figsize=(10, 6))
    layers = list(range(n_layers))
    ax.plot(layers, layer_auc_R1, "o-", color="steelblue", label="R1 (L10H9sv0)", linewidth=2, markersize=6)
    ax.plot(layers, layer_auc_R3, "s-", color="coral", label="R3 (L9H7sv1)", linewidth=2, markersize=6)
    ax.plot(layers, layer_auc_dark, "^-", color="gray", label="Dark (L11H1sv1)", linewidth=2, markersize=6)
    ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xlabel("Layer", fontsize=12)
    ax.set_ylabel("AUC (predicting IOI correctness)", fontsize=12)
    ax.set_title("GP Receptor AUC on IOI Task Across Layers", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.4, max(max(layer_auc_R1), max(layer_auc_R3), max(layer_auc_dark), 0.7) + 0.05)
    ax.set_xticks(layers)
    plt.tight_layout()
    p1 = str(out_dir / "phase2a_plot1_layer_auc.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  [1] {p1}")

    # --- Plot 2: Cross-task R² scatter ---
    fig, ax = plt.subplots(figsize=(9, 7))
    receptor_score = g_R1_final - g_R3_final  # pol1*g1 + pol3*g3
    correct_mask = ioi_correct.astype(bool)
    ax.scatter(receptor_score[correct_mask], ioi_logit_diff[correct_mask],
               c="royalblue", alpha=0.3, s=12, label="IOI correct", zorder=2)
    ax.scatter(receptor_score[~correct_mask], ioi_logit_diff[~correct_mask],
               c="red", alpha=0.5, s=25, label="IOI error", zorder=3)
    ax.set_xlabel("GP receptor score (g₁ − g₃)", fontsize=12)
    ax.set_ylabel("IOI logit difference", fontsize=12)
    ax.set_title(f"GP Receptors vs IOI Logit Diff (R² = {ioi_r2:.4f})", fontsize=13)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)
    ax.legend(fontsize=10)
    plt.tight_layout()
    p2 = str(out_dir / "phase2a_plot2_cross_task_scatter.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  [2] {p2}")

    # --- Plot 3: Transfer AUC bars (GP vs IOI) ---
    fig, ax = plt.subplots(figsize=(9, 6))
    x = np.arange(3)
    width = 0.35
    gp_aucs = [0.964, 0.958, 0.526]  # known from Exp 0c
    ioi_aucs = [transfer_auc_R1, transfer_auc_R3, transfer_auc_dark]
    bars1 = ax.bar(x - width / 2, gp_aucs, width, label="GP task", color="steelblue",
                   edgecolor="black", alpha=0.8)
    bars2 = ax.bar(x + width / 2, ioi_aucs, width, label="IOI task", color="coral",
                   edgecolor="black", alpha=0.8)
    ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.5, label="Chance")
    ax.set_xticks(x)
    ax.set_xticklabels(["R1", "R3", "Dark"], fontsize=12)
    ax.set_ylabel("AUC", fontsize=12)
    ax.set_title("Receptor AUC: GP vs IOI Task", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(0.3, 1.05)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{bar.get_height():.3f}", ha="center", fontsize=9)
    plt.tight_layout()
    p3 = str(out_dir / "phase2a_plot3_transfer_auc_bars.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"  [3] {p3}")

    # --- Plot 4: Embedding projection control ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, pio, pso, name, auc_val in zip(axes,
        [proj_io_R1, proj_io_R3, proj_io_dark],
        [proj_s_R1, proj_s_R3, proj_s_dark],
        ["R1", "R3", "Dark"],
        [emb_auc_R1_ioi, emb_auc_R3_ioi, emb_auc_dark_ioi]):

        ax.hist(pio, bins=30, alpha=0.6, color="steelblue", label="IO names", density=True)
        ax.hist(pso, bins=30, alpha=0.6, color="coral", label="S names", density=True)
        ax.set_xlabel(f"Embedding · v_{name}", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{name}: IO vs S embeddings\nAUC = {auc_val:.3f}", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("Embedding Projection of IOI Names onto GP Receptor Directions", fontsize=13)
    plt.tight_layout()
    p4 = str(out_dir / "phase2a_plot4_embedding_control.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"  [4] {p4}")

    # --- Plot 5: Activation distributions by IOI correctness ---
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, gf, name in zip(axes,
        [g_R1_final, g_R3_final, g_dark_final],
        ["R1", "R3", "Dark"]):

        ax.hist(gf[correct_mask], bins=30, alpha=0.6, color="steelblue",
                label="IOI correct", density=True)
        ax.hist(gf[~correct_mask], bins=30, alpha=0.6, color="coral",
                label="IOI error", density=True)
        ax.set_xlabel(f"g_{name} (final layer)", fontsize=10)
        ax.set_ylabel("Density", fontsize=10)
        ax.set_title(f"{name} activations by IOI correctness", fontsize=11)
        ax.legend(fontsize=9)

    fig.suptitle("Receptor Activation Distributions by IOI Correctness", fontsize=13)
    plt.tight_layout()
    p5 = str(out_dir / "phase2a_plot5_activation_distributions.png")
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    print(f"  [5] {p5}")

    # --- Plot 6: Gender confound control ---
    fig, ax = plt.subplots(figsize=(7, 5))
    aucs_gc = [transfer_auc_R1]
    labels_gc = [f"All\n(N={n_examples})"]
    colors_gc = ["steelblue"]
    if same_gender.sum() > 50:
        aucs_gc.append(auc_same)
        labels_gc.append(f"Same-gender\n(N={int(same_gender.sum())})")
        colors_gc.append("coral")
    if mixed_gender.sum() > 50:
        aucs_gc.append(auc_mixed)
        labels_gc.append(f"Mixed-gender\n(N={int(mixed_gender.sum())})")
        colors_gc.append("gray")
    ax.bar(range(len(aucs_gc)), aucs_gc, color=colors_gc, edgecolor="black", alpha=0.8)
    ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_xticks(range(len(aucs_gc)))
    ax.set_xticklabels(labels_gc, fontsize=10)
    ax.set_ylabel("R1 Transfer AUC", fontsize=12)
    ax.set_title("R1 Transfer AUC: Gender Confound Control", fontsize=13)
    for i, v in enumerate(aucs_gc):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_ylim(0.4, max(aucs_gc) + 0.08)
    plt.tight_layout()
    p6 = str(out_dir / "phase2a_plot6_gender_control.png")
    fig.savefig(p6, dpi=150)
    plt.close(fig)
    print(f"  [6] {p6}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print(f"\n{'='*60}")
    print("PHASE 2A: IOI ZERO-SHOT TRANSFER — SUMMARY")
    print(f"{'='*60}")
    print(f"IOI accuracy (IO > S):       {ioi_acc:.4f}")
    print(f"Transfer AUC R1:             {transfer_auc_R1:.4f}")
    print(f"Transfer AUC R3:             {transfer_auc_R3:.4f}")
    print(f"Transfer AUC dark:           {transfer_auc_dark:.4f}")
    print(f"Cross-task R²:               {ioi_r2:.4f} (GP: {gp_r2})")
    print(f"corr(g1, IOI logit diff):    {corr_R1:.4f}")
    print(f"corr(g3, IOI logit diff):    {corr_R3:.4f}")
    print(f"Emb AUC (R1, IO vs S):       {emb_auc_R1_ioi:.4f}")
    print(f"Emb AUC (R3, IO vs S):       {emb_auc_R3_ioi:.4f}")
    print(f"Gender ctrl (same):          {auc_same:.4f}")
    print(f"Gender ctrl (mixed):         {auc_mixed:.4f}")
    print("=" * 60)

    # ==================================================================
    # SAVE JSON
    # ==================================================================
    results = {
        "ioi_performance": {
            "top1_accuracy": float(top1_acc),
            "ioi_accuracy": float(ioi_acc),
            "mean_logit_diff": float(mean_ld),
            "n_examples": int(n_examples),
            "valid_single_token": int(valid_count),
        },
        "receptor_activations_ioi": {
            "R1": {"mean": float(g_R1_final.mean()), "std": float(g_R1_final.std())},
            "R3": {"mean": float(g_R3_final.mean()), "std": float(g_R3_final.std())},
            "Dark": {"mean": float(g_dark_final.mean()), "std": float(g_dark_final.std())},
        },
        "transfer_auc": {
            "R1": float(transfer_auc_R1),
            "R3": float(transfer_auc_R3),
            "Dark": float(transfer_auc_dark),
        },
        "cross_task_correlation": {
            "R1": float(corr_R1),
            "R3": float(corr_R3),
            "Dark": float(corr_dark),
        },
        "cross_task_r2": {
            "ioi_r2_all3": float(ioi_r2),
            "ioi_r2_R1": float(r2_R1),
            "ioi_r2_R3": float(r2_R3),
            "ioi_r2_dark": float(r2_dark),
            "gp_r2": float(gp_r2),
        },
        "embedding_control": {
            "emb_auc_R1_ioi": float(emb_auc_R1_ioi),
            "emb_auc_R3_ioi": float(emb_auc_R3_ioi),
            "emb_auc_dark_ioi": float(emb_auc_dark_ioi),
            "corr_emb_diff_R1": float(corr_emb_R1),
            "corr_emb_diff_R3": float(corr_emb_R3),
        },
        "gender_confound": {
            "n_same_gender": int(same_gender.sum()),
            "n_mixed_gender": int(mixed_gender.sum()),
            "auc_all": float(transfer_auc_R1),
            "auc_same_gender": float(auc_same),
            "auc_mixed_gender": float(auc_mixed),
        },
        "layer_auc": {
            "R1": [float(a) for a in layer_auc_R1],
            "R3": [float(a) for a in layer_auc_R3],
            "Dark": [float(a) for a in layer_auc_dark],
        },
    }

    json_path = str(out_dir / "phase2a_ioi_transfer_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] {json_path}")

    # ==================================================================
    # SANITY CHECKS
    # ==================================================================
    print(f"\n{'='*60}")
    print("[SANITY CHECKS]")
    n_errors = int((~all_io_gt_s).sum().item())
    print(f"  1. IOI examples: {n_examples} (expected 1000)")
    print(f"  2. IOI accuracy: {ioi_acc*100:.2f}% — {'PASS' if ioi_acc > 0.7 else 'CHECK'} (GPT-2 should be >80% on IOI)")
    print(f"  3. Mean IOI logit diff: {mean_ld:.2f} — {'PASS' if mean_ld > 0 else 'CHECK'} (expected positive)")
    print(f"  4. Transfer AUC R1={transfer_auc_R1:.4f} — near chance (0.5) expected")
    print(f"  5. Transfer AUC R3={transfer_auc_R3:.4f} — near chance (0.5) expected")
    print(f"  6. Cross-task R²={ioi_r2:.4f} — {'PASS' if ioi_r2 < 0.1 else 'CHECK'} (expected near 0)")
    print(f"  7. Single-token names: {valid_count}/{n_examples} — {'PASS' if valid_count == n_examples else 'CHECK'}")
    print(f"  8. N errors (IO < S): {n_errors}")
    print(f"  9. Emb AUC R1 IO vs S: {emb_auc_R1_ioi:.4f} — checks for gender confound")
    print(f"  10. Gender split: same={int(same_gender.sum())}, mixed={int(mixed_gender.sum())}")
    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
