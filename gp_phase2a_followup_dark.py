#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gp_phase2a_followup_dark.py

Phase 2A Follow-Up: Dark Receptor IOI Characterization.

Tests whether the dark receptor's (L11H1sv1) correlation with IOI
is real or confounded, whether it is causal, and where the signal
originates (layer delta analysis).

Four parts:
  1. Confound controls (sentence length, template, IO position, pred position)
  2. Causal ablation (project out each receptor direction, recompute IOI accuracy)
  3. Layer delta analysis (layer 10→11 jump in dark activation)
  4. Error analysis (characterize the ~28 IOI errors)

Run on Kaggle (GPU T4/P100):
  pip install transformer_lens --quiet
  python gp_phase2a_followup_dark.py \
    --data_dir data_main \
    --ioi_csv test_1k_ioi.csv \
    --out_dir outputs/gp \
    --svd_cache outputs/gp/svd_cache.pt \
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
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import roc_auc_score
from transformer_lens import HookedTransformer


# =====================================================================
# Utility helpers
# =====================================================================

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
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
        return max(auc, 1.0 - auc)
    except Exception:
        return 0.5


def batches(n, batch_size):
    """Yield (start, end) index pairs."""
    for i in range(0, n, batch_size):
        yield i, min(i + batch_size, n)


def manual_ln(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
              eps: float = 1e-5) -> torch.Tensor:
    """
    Manual LayerNorm matching TransformerLens ln_final.
    x: (..., d_model)
    weight, bias: (d_model,)
    """
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / (var + eps).sqrt()
    return x_norm * weight + bias


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Phase 2A Follow-Up: Dark Receptor IOI Characterization")
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
    print("PHASE 2A FOLLOW-UP: DARK RECEPTOR IOI CHARACTERIZATION")
    print("=" * 70)

    # ==================================================================
    # SETUP: Load model
    # ==================================================================
    print("\n[LOAD] Model (gpt2-small) ...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tokenizer = model.tokenizer
    tokenizer.pad_token = tokenizer.eos_token
    n_layers = model.cfg.n_layers  # 12
    d_model = model.cfg.d_model    # 768

    # Verify model component shapes
    print(f"  model.ln_final.w.shape = {model.ln_final.w.shape}  (expect (768,))")
    print(f"  model.ln_final.b.shape = {model.ln_final.b.shape}  (expect (768,))")
    print(f"  model.W_U.shape        = {model.W_U.shape}  (expect (768, 50257))")
    print(f"  model.b_U.shape        = {model.b_U.shape}  (expect (50257,))")

    # ==================================================================
    # SETUP: Load SVD cache and extract receptor directions
    # ==================================================================
    print("[LOAD] SVD cache ...")
    ov = load_svd_cache_ov_only(str(svd_path), device="cpu")

    def get_direction(l, h, sv):
        v = ov[l][h].Vh[sv, :].clone().cpu().float()
        return v / v.norm()

    v_R1 = get_direction(10, 9, 0)     # (768,)
    v_R3 = get_direction(9, 7, 1)      # (768,)
    v_dark = get_direction(11, 1, 1)   # (768,)

    print(f"  v_R1 norm:   {v_R1.norm().item():.6f} (should be 1.0)")
    print(f"  v_R3 norm:   {v_R3.norm().item():.6f} (should be 1.0)")
    print(f"  v_dark norm: {v_dark.norm().item():.6f} (should be 1.0)")

    # ==================================================================
    # LOAD IOI DATASET
    # ==================================================================
    print("[LOAD] IOI dataset ...")
    ioi_path = str(data_dir / args.ioi_csv)
    ioi_rows: List[dict] = []
    with open(ioi_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            ioi_rows.append(r)

    n_examples = len(ioi_rows)
    print(f"  {n_examples} IOI examples loaded")

    # Tokenize IO and S names
    io_token_ids: List[int] = []
    s_token_ids: List[int] = []
    valid_mask: List[bool] = []

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
    # EXTRACT SURFACE FEATURES for confound analysis
    # ==================================================================
    print("[FEATURES] Extracting surface features ...")

    sentence_lengths = np.array(
        [len(r["ioi_sentences_input"].split()) for r in ioi_rows], dtype=np.float64
    )
    pred_positions = np.array(
        [len(tokenizer.encode(r["ioi_sentences_input"], add_special_tokens=False))
         for r in ioi_rows], dtype=np.float64
    )

    # Template type (check multi-word templates FIRST to avoid false matches)
    templates = []
    for r in ioi_rows:
        inp = r["ioi_sentences_input"]
        if "decided to give" in inp:
            templates.append("decided")
        elif "wanted to give" in inp:
            templates.append("wanted")
        elif "gave" in inp:
            templates.append("gave")
        else:
            templates.append("other")
    templates = np.array(templates)

    # IO position: does IO name appear before S name?
    io_is_first = []
    for r in ioi_rows:
        inp = r["ioi_sentences_input"]
        io_name = r["ioi_sentences_labels"].strip()
        s_name = r["ioi_sentences_labels_wrong"].strip()
        io_is_first.append(inp.find(io_name) < inp.find(s_name))
    io_is_first = np.array(io_is_first)

    # Print feature summary
    for tmpl in ["gave", "decided", "wanted", "other"]:
        print(f"  Template '{tmpl}': N={(templates == tmpl).sum()}")
    print(f"  IO first: {io_is_first.sum()}, IO second: {(~io_is_first).sum()}")
    print(f"  Sentence length: mean={sentence_lengths.mean():.1f}, "
          f"std={sentence_lengths.std():.1f}, "
          f"range=[{sentence_lengths.min():.0f}, {sentence_lengths.max():.0f}]")
    print(f"  Pred position:   mean={pred_positions.mean():.1f}, "
          f"std={pred_positions.std():.1f}, "
          f"range=[{pred_positions.min():.0f}, {pred_positions.max():.0f}]")

    # ==================================================================
    # FORWARD PASSES — collect residuals at all layers + logits
    # ==================================================================
    print("\n[FORWARD] Collecting residuals at all layers + logits ...")

    resid_hooks = set()
    for l in range(n_layers):
        resid_hooks.add(f"blocks.{l}.hook_resid_post")

    # Storage: n_layers=12 layers (0..11)
    all_residuals = torch.zeros(n_examples, n_layers, d_model)
    all_logits_io = torch.zeros(n_examples)
    all_logits_s = torch.zeros(n_examples)
    all_io_gt_s = torch.zeros(n_examples, dtype=torch.bool)

    batch_count = 0
    for start, end in batches(n_examples, args.batch_size):
        batch_texts = [ioi_rows[i]["ioi_sentences_input"] for i in range(start, end)]
        B = len(batch_texts)

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

            for l in range(n_layers):
                all_residuals[gi, l] = cache[f"blocks.{l}.hook_resid_post"][b, pred_pos].cpu()

            final_logits = logits[b, pred_pos]
            io_tok = io_token_ids[gi]
            s_tok = s_token_ids[gi]
            all_logits_io[gi] = final_logits[io_tok].cpu()
            all_logits_s[gi] = final_logits[s_tok].cpu()
            all_io_gt_s[gi] = (final_logits[io_tok] > final_logits[s_tok])

        del logits, cache
        torch.cuda.empty_cache()

        batch_count += 1
        if batch_count % 10 == 0:
            print(f"  Batch {batch_count}/{(n_examples + args.batch_size - 1) // args.batch_size}")

    print(f"  Done. {n_examples} examples processed.")

    # ==================================================================
    # COMPUTE DERIVED QUANTITIES
    # ==================================================================
    ioi_logit_diff = (all_logits_io - all_logits_s).numpy()
    ioi_correct = all_io_gt_s.numpy().astype(int)
    ioi_acc_original = float(ioi_correct.mean())
    mean_ld = float(ioi_logit_diff.mean())

    print(f"\n{'='*60}")
    print("IOI MODEL PERFORMANCE (from forward passes)")
    print(f"{'='*60}")
    print(f"  IOI accuracy (IO > S): {ioi_acc_original*100:.2f}%")
    print(f"  Mean IOI logit diff:   {mean_ld:.4f}")

    # Receptor projections at all layers: (N, n_layers)
    g_R1_all = (all_residuals @ v_R1).numpy()
    g_R3_all = (all_residuals @ v_R3).numpy()
    g_dark_all = (all_residuals @ v_dark).numpy()

    # Final layer (layer 11 = index -1)
    g_R1_final = g_R1_all[:, -1]
    g_R3_final = g_R3_all[:, -1]
    g_dark_final = g_dark_all[:, -1]

    print(f"\nReceptor activations on IOI (final layer):")
    print(f"  R1:   mean={g_R1_final.mean():.4f}, std={g_R1_final.std():.4f}")
    print(f"  R3:   mean={g_R3_final.mean():.4f}, std={g_R3_final.std():.4f}")
    print(f"  Dark: mean={g_dark_final.mean():.4f}, std={g_dark_final.std():.4f}")

    # Transfer AUC (for reference / cross-check with Phase 2A)
    transfer_auc_R1 = safe_auc(ioi_correct, g_R1_final)
    transfer_auc_R3 = safe_auc(ioi_correct, g_R3_final)
    transfer_auc_dark = safe_auc(ioi_correct, g_dark_final)
    print(f"\nTransfer AUC (cross-check with Phase 2A):")
    print(f"  R1={transfer_auc_R1:.4f}, R3={transfer_auc_R3:.4f}, Dark={transfer_auc_dark:.4f}")

    # ==================================================================
    # PART 1: CONFOUND CONTROLS
    # ==================================================================
    print(f"\n{'='*60}")
    print("=== PART 1: CONFOUND CONTROLS ===")
    print(f"{'='*60}")

    # --- Sentence length ---
    r_dark_len = float(np.corrcoef(g_dark_final, sentence_lengths)[0, 1])
    r_logit_len = float(np.corrcoef(ioi_logit_diff, sentence_lengths)[0, 1])
    print(f"\nSentence length:")
    print(f"  corr(g_dark, sentence_length):     {r_dark_len:.4f}")
    print(f"  corr(IOI_logit_diff, sent_length): {r_logit_len:.4f}")

    # Partial correlation: dark vs IOI logit diff, controlling for sentence length
    reg_dark_len = LinearRegression().fit(sentence_lengths.reshape(-1, 1), g_dark_final)
    resid_dark_len = g_dark_final - reg_dark_len.predict(sentence_lengths.reshape(-1, 1))
    reg_logit_len = LinearRegression().fit(sentence_lengths.reshape(-1, 1), ioi_logit_diff)
    resid_logit_len = ioi_logit_diff - reg_logit_len.predict(sentence_lengths.reshape(-1, 1))
    partial_corr_len = float(np.corrcoef(resid_dark_len, resid_logit_len)[0, 1])
    print(f"  Partial corr (dark, IOI | length):  {partial_corr_len:.4f}  (raw: -0.3389)")

    # --- Prediction position ---
    r_dark_pos = float(np.corrcoef(g_dark_final, pred_positions)[0, 1])
    r_logit_pos = float(np.corrcoef(ioi_logit_diff, pred_positions)[0, 1])
    print(f"\nPrediction position:")
    print(f"  corr(g_dark, pred_position):        {r_dark_pos:.4f}")
    print(f"  corr(IOI_logit_diff, pred_position): {r_logit_pos:.4f}")

    # Partial correlation controlling for BOTH length and position
    X_confounds = np.column_stack([sentence_lengths, pred_positions])
    reg_dark_both = LinearRegression().fit(X_confounds, g_dark_final)
    resid_dark_both = g_dark_final - reg_dark_both.predict(X_confounds)
    reg_logit_both = LinearRegression().fit(X_confounds, ioi_logit_diff)
    resid_logit_both = ioi_logit_diff - reg_logit_both.predict(X_confounds)
    partial_corr_both = float(np.corrcoef(resid_dark_both, resid_logit_both)[0, 1])
    print(f"  Partial corr (dark, IOI | length+pos): {partial_corr_both:.4f}")

    # --- Template type split ---
    tmpl_names = ["gave", "decided", "wanted"]
    tmpl_aucs = []
    tmpl_corrs = []
    tmpl_ns = []
    print(f"\nTemplate split:")
    for tmpl in tmpl_names:
        mask = templates == tmpl
        n = int(mask.sum())
        tmpl_ns.append(n)
        if n > 20:
            auc = safe_auc(ioi_correct[mask], g_dark_final[mask])
            corr = float(np.corrcoef(g_dark_final[mask], ioi_logit_diff[mask])[0, 1])
        else:
            auc, corr = 0.5, 0.0
        tmpl_aucs.append(auc)
        tmpl_corrs.append(corr)
        print(f"  Template '{tmpl}' (N={n}): dark AUC={auc:.4f}, corr={corr:.4f}")

    # --- IO position split ---
    print(f"\nIO position:")
    n_first = int(io_is_first.sum())
    n_second = int((~io_is_first).sum())
    auc_first = safe_auc(ioi_correct[io_is_first], g_dark_final[io_is_first])
    auc_second = safe_auc(ioi_correct[~io_is_first], g_dark_final[~io_is_first])
    corr_first = float(np.corrcoef(g_dark_final[io_is_first], ioi_logit_diff[io_is_first])[0, 1])
    corr_second = float(np.corrcoef(g_dark_final[~io_is_first], ioi_logit_diff[~io_is_first])[0, 1])
    print(f"  IO first  (N={n_first}): dark AUC={auc_first:.4f}, corr={corr_first:.4f}")
    print(f"  IO second (N={n_second}): dark AUC={auc_second:.4f}, corr={corr_second:.4f}")

    # ==================================================================
    # PART 2: CAUSAL ABLATION ON IOI
    # ==================================================================
    print(f"\n{'='*60}")
    print("=== PART 2: CAUSAL ABLATION ON IOI ===")
    print(f"{'='*60}")

    # Get model components for manual logit computation (move to CPU for safety)
    ln_final_weight = model.ln_final.w.detach().cpu().float()  # (768,)
    ln_final_bias = model.ln_final.b.detach().cpu().float()    # (768,)
    W_U = model.W_U.detach().cpu().float()                     # (768, 50257)
    b_U = model.b_U.detach().cpu().float()                     # (50257,)

    # Final residual streams: post-block-11, BEFORE ln_final
    # In TransformerLens, blocks.11.hook_resid_post is after block 11, before ln_final
    final_resid = all_residuals[:, -1, :].clone().float()  # (N, 768)

    # --- Sanity check: reconstruct IOI accuracy from manual computation ---
    print("\n[SANITY CHECK] Reconstructing IOI accuracy manually ...")
    recon_logits = manual_ln(final_resid, ln_final_weight, ln_final_bias) @ W_U + b_U
    # (N, 50257)

    recon_io_logits = torch.tensor(
        [recon_logits[i, io_token_ids[i]].item() for i in range(n_examples)]
    )
    recon_s_logits = torch.tensor(
        [recon_logits[i, s_token_ids[i]].item() for i in range(n_examples)]
    )
    recon_acc = float((recon_io_logits > recon_s_logits).float().mean().item())
    recon_ld = float((recon_io_logits - recon_s_logits).mean().item())

    print(f"  Reconstructed IOI accuracy: {recon_acc*100:.2f}% "
          f"(original: {ioi_acc_original*100:.2f}%)")
    print(f"  Reconstructed mean logit diff: {recon_ld:.4f} (original: {mean_ld:.4f})")

    # Check agreement
    recon_correct = (recon_io_logits > recon_s_logits).numpy().astype(int)
    agreement = float((recon_correct == ioi_correct).mean())
    print(f"  Agreement with forward-pass labels: {agreement*100:.2f}%")

    if abs(recon_acc - ioi_acc_original) > 0.02:
        print("  ⚠ WARNING: Reconstructed accuracy differs by >2%. "
              "Check LayerNorm implementation!")
    else:
        print("  ✓ PASS: Reconstruction matches forward-pass accuracy.")

    # Use reconstructed accuracy as the baseline for ablation
    # (ensures apples-to-apples comparison)
    baseline_acc = recon_acc

    # --- Causal ablation for each receptor ---
    print(f"\nOriginal IOI accuracy (reconstructed): {baseline_acc*100:.2f}%")

    ablation_results = {}
    for name, v_k in [("R1", v_R1), ("R3", v_R3), ("Dark", v_dark)]:
        v_k_cpu = v_k.cpu().float()

        # Project out v_k: x_ablated = x - (x·v_k) * v_k
        projections = (final_resid @ v_k_cpu).unsqueeze(1)  # (N, 1)
        ablated_resid = final_resid - projections * v_k_cpu.unsqueeze(0)  # (N, 768)

        # Recompute logits through ln_final + W_U + b_U
        abl_logits = manual_ln(ablated_resid, ln_final_weight, ln_final_bias) @ W_U + b_U

        abl_io = torch.tensor(
            [abl_logits[i, io_token_ids[i]].item() for i in range(n_examples)]
        )
        abl_s = torch.tensor(
            [abl_logits[i, s_token_ids[i]].item() for i in range(n_examples)]
        )

        abl_acc = float((abl_io > abl_s).float().mean().item())
        acc_drop = baseline_acc - abl_acc
        abl_logit_diff = (abl_io - abl_s).numpy()
        mean_ld_drop = float(ioi_logit_diff.mean()) - float(abl_logit_diff.mean())

        ablation_results[name] = {
            "acc_after": abl_acc,
            "acc_drop": acc_drop,
            "mean_logit_diff_after": float(abl_logit_diff.mean()),
            "mean_logit_diff_drop": mean_ld_drop,
        }

        print(f"\n  {name}: IOI Acc after ablation = {abl_acc*100:.2f}%  "
              f"(AccDrop = {acc_drop*100:.2f}%)")
        print(f"    Mean logit diff: {abl_logit_diff.mean():.4f} "
              f"(original: {ioi_logit_diff.mean():.4f}, drop: {mean_ld_drop:.4f})")

    # Extract individual AccDrops for later use
    ioi_accdrop_R1 = ablation_results["R1"]["acc_drop"]
    ioi_accdrop_R3 = ablation_results["R3"]["acc_drop"]
    ioi_accdrop_dark = ablation_results["Dark"]["acc_drop"]

    # Cross-task CFR for dark receptor
    gp_cfr_dark = 0.0817 / (0.526 - 0.5)  # = 3.14
    ioi_cfr_dark = (ioi_accdrop_dark / (transfer_auc_dark - 0.5)
                    if transfer_auc_dark > 0.5 and ioi_accdrop_dark > 0 else 0.0)

    print(f"\nCross-task CFR:")
    print(f"  Dark GP CFR:  {gp_cfr_dark:.2f}")
    print(f"  Dark IOI CFR: {ioi_cfr_dark:.2f}")

    # ==================================================================
    # PART 3: LAYER DELTA ANALYSIS
    # ==================================================================
    print(f"\n{'='*60}")
    print("=== PART 3: LAYER DELTA ANALYSIS ===")
    print(f"{'='*60}")

    # Delta = dark activation at layer 11 minus layer 10
    # all_residuals[:, l, :] = residual after block l (0-indexed)
    # Layer index 10 = post-block-10, layer index 11 = post-block-11
    # L11H1 is in block 11, so delta isolates block 11's contribution
    delta_dark = g_dark_all[:, 11] - g_dark_all[:, 10]

    print(f"\nLayer 10→11 delta stats:")
    print(f"  mean(δ):  {delta_dark.mean():.4f}")
    print(f"  std(δ):   {delta_dark.std():.4f}")

    # Correlation of delta with IOI logit diff
    r_delta = float(np.corrcoef(delta_dark, ioi_logit_diff)[0, 1])
    print(f"  corr(δ, IOI_logit_diff):  {r_delta:.4f}")

    # AUC of delta for predicting IOI correctness
    delta_auc = safe_auc(ioi_correct, delta_dark)
    print(f"  AUC(δ, IOI_correct):      {delta_auc:.4f}")

    # L11 fraction: r²_delta / r²_total
    r_total = float(np.corrcoef(g_dark_final, ioi_logit_diff)[0, 1])
    l11_fraction = (r_delta ** 2) / (r_total ** 2) if r_total != 0 else 0.0
    print(f"  L11 fraction (r²_δ / r²_total): {l11_fraction:.4f}")

    # Also: correlation at layer 10
    r_layer10 = float(np.corrcoef(g_dark_all[:, 10], ioi_logit_diff)[0, 1])
    print(f"\n  corr(g_dark_layer10, IOI_logit_diff): {r_layer10:.4f}")
    print(f"  corr(g_dark_layer11, IOI_logit_diff): {r_total:.4f}")
    print(f"  Signal gain from L11: {abs(r_total) - abs(r_layer10):.4f}")

    # ==================================================================
    # PART 4: ERROR ANALYSIS
    # ==================================================================
    print(f"\n{'='*60}")
    print("=== PART 4: ERROR ANALYSIS ===")
    print(f"{'='*60}")

    correct_mask = ioi_correct.astype(bool)
    error_mask = ~correct_mask
    n_errors = int(error_mask.sum())
    n_correct = int(correct_mask.sum())

    print(f"\nN errors: {n_errors}, N correct: {n_correct}")
    print(f"Dark mean (correct): {g_dark_final[correct_mask].mean():.4f} "
          f"(std={g_dark_final[correct_mask].std():.4f})")
    if n_errors > 0:
        print(f"Dark mean (error):   {g_dark_final[error_mask].mean():.4f} "
              f"(std={g_dark_final[error_mask].std():.4f})")
    else:
        print("Dark mean (error):   N/A (no errors)")

    # Cohen's d
    if n_errors > 0:
        d_prime = float(
            (g_dark_final[correct_mask].mean() - g_dark_final[error_mask].mean())
            / g_dark_final.std()
        )
    else:
        d_prime = 0.0
    print(f"Cohen's d (correct vs error): {d_prime:.4f}")

    # Errors by template
    if n_errors > 0:
        error_templates = templates[error_mask]
        print("\nErrors by template:")
        for tmpl in ["gave", "decided", "wanted"]:
            n_err_tmpl = int((error_templates == tmpl).sum())
            n_total_tmpl = int((templates == tmpl).sum())
            rate = n_err_tmpl / n_total_tmpl * 100 if n_total_tmpl > 0 else 0
            print(f"  Template '{tmpl}': {n_err_tmpl}/{n_total_tmpl} errors ({rate:.1f}%)")

    # Errors by IO position
    if n_errors > 0:
        error_io_first = io_is_first[error_mask]
        n_err_first = int(error_io_first.sum())
        n_err_second = n_errors - n_err_first
        print(f"\nErrors by IO position:")
        print(f"  IO first:  {n_err_first}/{n_first} ({n_err_first/n_first*100:.1f}%)")
        print(f"  IO second: {n_err_second}/{n_second} ({n_err_second/n_second*100:.1f}%)")

    # ==================================================================
    # PLOTS
    # ==================================================================
    print(f"\n[PLOTS] Saving to {out_dir} ...")

    # --- Plot 1: Confound controls (2×2 panel) ---
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Panel A: dark vs sentence length
    ax = axes[0, 0]
    jitter = np.random.uniform(-0.2, 0.2, n_examples)
    ax.scatter(sentence_lengths + jitter, g_dark_final, alpha=0.3, s=10, c="steelblue")
    ax.set_xlabel("Sentence length (words)")
    ax.set_ylabel("g_dark (final layer)")
    ax.set_title(f"Dark vs Length (r={r_dark_len:.3f})")

    # Panel B: dark AUC by template
    ax = axes[0, 1]
    bars = ax.bar(range(3), tmpl_aucs, color=["steelblue", "coral", "gray"],
                  edgecolor="black", alpha=0.8)
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{t}\n(N={n})" for t, n in zip(tmpl_names, tmpl_ns)])
    ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("Dark Transfer AUC")
    ax.set_title("Dark AUC by Template Type")
    for i, v in enumerate(tmpl_aucs):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0.4, max(tmpl_aucs) + 0.06)

    # Panel C: dark AUC by IO position
    ax = axes[1, 0]
    bars = ax.bar([0, 1], [auc_first, auc_second], color=["steelblue", "coral"],
                  edgecolor="black", alpha=0.8)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([f"IO first\n(N={n_first})", f"IO second\n(N={n_second})"])
    ax.axhline(y=0.5, color="black", linestyle="--", alpha=0.5)
    ax.set_ylabel("Dark Transfer AUC")
    ax.set_title("Dark AUC by IO Position")
    for i, v in enumerate([auc_first, auc_second]):
        ax.text(i, v + 0.005, f"{v:.3f}", ha="center", fontsize=10, fontweight="bold")
    ax.set_ylim(0.4, max(auc_first, auc_second) + 0.06)

    # Panel D: dark vs prediction position
    ax = axes[1, 1]
    jitter2 = np.random.uniform(-0.2, 0.2, n_examples)
    ax.scatter(pred_positions + jitter2, g_dark_final, alpha=0.3, s=10, c="steelblue")
    ax.set_xlabel("Prediction position (tokens)")
    ax.set_ylabel("g_dark (final layer)")
    ax.set_title(f"Dark vs Pred Position (r={r_dark_pos:.3f})")

    fig.suptitle("Dark Receptor IOI Signal: Confound Controls",
                 fontsize=14, fontweight="bold")
    plt.tight_layout()
    p1 = str(out_dir / "phase2a_followup_plot1_confounds.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  [1] {p1}")

    # --- Plot 2: Causal ablation comparison ---
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    x_pos = np.arange(3)
    width = 0.35
    gp_accdrops = [3.59, 1.31, 8.17]
    ioi_accdrops = [ioi_accdrop_R1 * 100, ioi_accdrop_R3 * 100, ioi_accdrop_dark * 100]

    bars1 = ax.bar(x_pos - width / 2, gp_accdrops, width, label="GP AccDrop",
                   color="steelblue", edgecolor="black", alpha=0.8)
    bars2 = ax.bar(x_pos + width / 2, ioi_accdrops, width, label="IOI AccDrop",
                   color="coral", edgecolor="black", alpha=0.8)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["R1\n(L10H9sv0)", "R3\n(L9H7sv1)", "Dark\n(L11H1sv1)"])
    ax.set_ylabel("AccDrop (%)")
    ax.set_title("Cross-Task Causal Ablation: GP vs IOI", fontsize=13)
    ax.legend(fontsize=11)
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.1,
                f"{bar.get_height():.2f}%", ha="center", fontsize=9)
    for bar in bars2:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.1,
                f"{h:.2f}%", ha="center", fontsize=9)
    plt.tight_layout()
    p2 = str(out_dir / "phase2a_followup_plot2_causal_ablation.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  [2] {p2}")

    # --- Plot 3: Layer delta analysis ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: delta vs IOI logit diff
    ax = axes[0]
    colors_scatter = ["steelblue" if c else "coral" for c in ioi_correct]
    ax.scatter(delta_dark, ioi_logit_diff, c=colors_scatter, alpha=0.3, s=10)
    ax.set_xlabel("δ_dark (layer 10→11)")
    ax.set_ylabel("IOI logit difference")
    ax.set_title(f"L11H1 Contribution (corr={r_delta:.3f})")
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.3)
    ax.axvline(x=0, color="gray", linestyle="--", alpha=0.3)

    # Right: delta distribution correct vs error
    ax = axes[1]
    ax.hist(delta_dark[correct_mask], bins=30, alpha=0.6, color="steelblue",
            label="IOI correct", density=True)
    if error_mask.sum() > 0:
        ax.hist(delta_dark[error_mask], bins=15, alpha=0.6, color="coral",
                label="IOI error", density=True)
    ax.set_xlabel("δ_dark (layer 10→11)")
    ax.set_ylabel("Density")
    ax.set_title(f"L11 Delta by IOI Correctness (AUC={delta_auc:.3f})")
    ax.legend()

    fig.suptitle("Layer 11 Contribution to Dark Receptor IOI Signal",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p3 = str(out_dir / "phase2a_followup_plot3_layer_delta.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"  [3] {p3}")

    # --- Plot 4: Cross-task causal profile (the money figure) ---
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    receptors = ["R1", "R3", "Dark"]
    gp_aucs_ref = [0.964, 0.958, 0.526]
    ioi_aucs_ref = [transfer_auc_R1, transfer_auc_R3, transfer_auc_dark]
    gp_ads = [3.59, 1.31, 8.17]
    ioi_ads = [ioi_accdrop_R1 * 100, ioi_accdrop_R3 * 100, ioi_accdrop_dark * 100]

    for idx, (name, ax) in enumerate(zip(receptors, axes)):
        x_pts = np.arange(2)
        w = 0.35

        bars_auc = ax.bar(x_pts - w / 2, [gp_aucs_ref[idx], ioi_aucs_ref[idx]], w,
                          color="steelblue", alpha=0.6, label="AUC", edgecolor="black")
        ax2 = ax.twinx()
        bars_ad = ax2.bar(x_pts + w / 2, [gp_ads[idx], ioi_ads[idx]], w,
                          color="coral", alpha=0.6, label="AccDrop %", edgecolor="black")

        ax.set_xticks(x_pts)
        ax.set_xticklabels(["GP", "IOI"])
        ax.set_ylabel("AUC")
        ax2.set_ylabel("AccDrop (%)")
        ax.set_title(f"{name}", fontsize=13, fontweight="bold")
        ax.axhline(y=0.5, color="gray", linestyle=":", alpha=0.3)
        ax.set_ylim(0.3, 1.05)

        # Annotate AUC bars
        for b in bars_auc:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                    f"{b.get_height():.3f}", ha="center", fontsize=8)
        # Annotate AccDrop bars
        for b in bars_ad:
            h = b.get_height()
            ax2.text(b.get_x() + b.get_width() / 2, h + 0.1,
                     f"{h:.2f}%", ha="center", fontsize=8)

        # Combined legend
        lines_a, labels_a = ax.get_legend_handles_labels()
        lines_b, labels_b = ax2.get_legend_handles_labels()
        if idx == 2:  # legend only on last panel
            ax.legend(lines_a + lines_b, labels_a + labels_b, fontsize=8, loc="upper left")

    fig.suptitle("Cross-Task Causal Profile: Three Circuit Channels",
                 fontsize=13, fontweight="bold")
    plt.tight_layout()
    p4 = str(out_dir / "phase2a_followup_plot4_cross_task_profile.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"  [4] {p4}")

    # --- Plot 5: Dark activation by IOI error status ---
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    data_correct = g_dark_final[correct_mask]
    data_error = g_dark_final[error_mask] if n_errors > 0 else np.array([])

    violin_data = [data_correct]
    positions = [0]
    tick_labels = [f"IOI Correct\n(N={n_correct})"]
    if n_errors > 0:
        violin_data.append(data_error)
        positions.append(1)
        tick_labels.append(f"IOI Error\n(N={n_errors})")

    parts = ax.violinplot(violin_data, positions=positions, showmeans=True, showmedians=True)

    if n_errors > 0:
        # Overlay individual error points
        ax.scatter(
            np.ones(n_errors) + np.random.uniform(-0.05, 0.05, n_errors),
            data_error,
            c="red", s=30, zorder=5, alpha=0.7, label="Individual errors"
        )
        ax.legend(fontsize=10)

    ax.set_xticks(positions)
    ax.set_xticklabels(tick_labels, fontsize=11)
    ax.set_ylabel("g_dark (final layer)", fontsize=12)
    ax.set_title(f"Dark Receptor: IOI Correct vs Error (d'={d_prime:.3f})", fontsize=13)
    plt.tight_layout()
    p5 = str(out_dir / "phase2a_followup_plot5_error_analysis.png")
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    print(f"  [5] {p5}")

    # ==================================================================
    # FINAL SUMMARY
    # ==================================================================
    print(f"\n{'='*60}")
    print("PHASE 2A FOLLOW-UP: DARK RECEPTOR IOI — SUMMARY")
    print(f"{'='*60}")
    print("CONFOUND CONTROLS:")
    print(f"  corr(dark, sent_length):        {r_dark_len:.4f}")
    print(f"  corr(dark, pred_position):      {r_dark_pos:.4f}")
    print(f"  Partial corr (controlling all): {partial_corr_both:.4f}  (raw: -0.3389)")
    print(f"  Template AUCs: gave={tmpl_aucs[0]:.3f}, "
          f"decided={tmpl_aucs[1]:.3f}, wanted={tmpl_aucs[2]:.3f}")
    print(f"  IO position: first={auc_first:.3f}, second={auc_second:.3f}")
    print()
    print("CAUSAL ABLATION (AccDrop %):")
    print(f"  R1:   GP={3.59:.2f}%  IOI={ioi_accdrop_R1*100:.2f}%")
    print(f"  R3:   GP={1.31:.2f}%  IOI={ioi_accdrop_R3*100:.2f}%")
    print(f"  Dark: GP={8.17:.2f}%  IOI={ioi_accdrop_dark*100:.2f}%")
    print()
    print("LAYER DELTA:")
    print(f"  corr(delta_L11, IOI_logit): {r_delta:.4f}")
    print(f"  AUC(delta, IOI_correct):    {delta_auc:.4f}")
    print(f"  L11 fraction:               {l11_fraction:.4f}")
    print()
    print("ERROR ANALYSIS:")
    print(f"  Dark mean correct: {g_dark_final[correct_mask].mean():.4f}")
    if n_errors > 0:
        print(f"  Dark mean error:   {g_dark_final[error_mask].mean():.4f}")
    print(f"  Cohen's d:         {d_prime:.3f}")
    print("=" * 60)

    # ==================================================================
    # SAVE RESULTS JSON
    # ==================================================================
    results = {
        "confound_controls": {
            "corr_dark_length": float(r_dark_len),
            "corr_logit_length": float(r_logit_len),
            "partial_corr_controlling_length": float(partial_corr_len),
            "corr_dark_position": float(r_dark_pos),
            "corr_logit_position": float(r_logit_pos),
            "partial_corr_controlling_both": float(partial_corr_both),
            "template_aucs": {t: float(a) for t, a in zip(tmpl_names, tmpl_aucs)},
            "template_corrs": {t: float(c) for t, c in zip(tmpl_names, tmpl_corrs)},
            "template_ns": {t: int(n) for t, n in zip(tmpl_names, tmpl_ns)},
            "io_position_aucs": {
                "first": float(auc_first),
                "second": float(auc_second),
            },
            "io_position_corrs": {
                "first": float(corr_first),
                "second": float(corr_second),
            },
            "io_position_ns": {
                "first": int(n_first),
                "second": int(n_second),
            },
        },
        "causal_ablation": {
            "baseline_acc": float(baseline_acc),
            "R1_ioi_acc_after": float(ablation_results["R1"]["acc_after"]),
            "R1_ioi_accdrop": float(ioi_accdrop_R1),
            "R3_ioi_acc_after": float(ablation_results["R3"]["acc_after"]),
            "R3_ioi_accdrop": float(ioi_accdrop_R3),
            "dark_ioi_acc_after": float(ablation_results["Dark"]["acc_after"]),
            "dark_ioi_accdrop": float(ioi_accdrop_dark),
            "R1_gp_accdrop": 0.0359,
            "R3_gp_accdrop": 0.0131,
            "dark_gp_accdrop": 0.0817,
            "dark_gp_cfr": float(gp_cfr_dark),
            "dark_ioi_cfr": float(ioi_cfr_dark),
            "R1_mean_ld_drop": float(ablation_results["R1"]["mean_logit_diff_drop"]),
            "R3_mean_ld_drop": float(ablation_results["R3"]["mean_logit_diff_drop"]),
            "dark_mean_ld_drop": float(ablation_results["Dark"]["mean_logit_diff_drop"]),
        },
        "layer_delta": {
            "delta_mean": float(delta_dark.mean()),
            "delta_std": float(delta_dark.std()),
            "corr_delta_ioi": float(r_delta),
            "auc_delta": float(delta_auc),
            "l11_fraction": float(l11_fraction),
            "corr_layer10": float(r_layer10),
            "corr_layer11": float(r_total),
            "signal_gain": float(abs(r_total) - abs(r_layer10)),
        },
        "error_analysis": {
            "n_errors": int(n_errors),
            "n_correct": int(n_correct),
            "dark_mean_correct": float(g_dark_final[correct_mask].mean()),
            "dark_std_correct": float(g_dark_final[correct_mask].std()),
            "dark_mean_error": float(g_dark_final[error_mask].mean()) if n_errors > 0 else None,
            "dark_std_error": float(g_dark_final[error_mask].std()) if n_errors > 0 else None,
            "cohens_d": float(d_prime),
        },
        "transfer_aucs_crosscheck": {
            "R1": float(transfer_auc_R1),
            "R3": float(transfer_auc_R3),
            "Dark": float(transfer_auc_dark),
        },
    }

    json_path = str(out_dir / "phase2a_followup_results.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[SAVE] {json_path}")

    # ==================================================================
    # SANITY CHECKS
    # ==================================================================
    print(f"\n{'='*60}")
    print("[SANITY CHECKS]")
    print(f"{'='*60}")
    print(f"  1. N examples: {n_examples} (expected 1000)")
    print(f"  2. IOI accuracy: {ioi_acc_original*100:.2f}% — "
          f"{'PASS' if ioi_acc_original > 0.7 else 'CHECK'}")
    print(f"  3. Reconstruction agreement: {agreement*100:.2f}% — "
          f"{'PASS' if agreement > 0.98 else 'CHECK'}")
    print(f"  4. Dark transfer AUC: {transfer_auc_dark:.4f} — "
          f"{'PASS' if abs(transfer_auc_dark - 0.776) < 0.05 else 'CHECK'} (expected ~0.776)")
    print(f"  5. Partial corr survived: {partial_corr_both:.4f} — "
          f"{'PASS (signal survives)' if abs(partial_corr_both) > 0.2 else 'CHECK'}")
    print(f"  6. R1 IOI AccDrop: {ioi_accdrop_R1*100:.2f}% — "
          f"{'PASS (near 0)' if abs(ioi_accdrop_R1) < 0.03 else 'CHECK'}")
    print(f"  7. R3 IOI AccDrop: {ioi_accdrop_R3*100:.2f}% — "
          f"{'PASS (near 0)' if abs(ioi_accdrop_R3) < 0.03 else 'CHECK'}")
    print(f"  8. Dark IOI AccDrop: {ioi_accdrop_dark*100:.2f}% — "
          f"{'PASS (causal)' if ioi_accdrop_dark > 0.005 else 'CHECK (not causal?)'}")
    print(f"  9. Template AUCs similar: "
          f"{'PASS' if max(tmpl_aucs) - min(tmpl_aucs) < 0.15 else 'CHECK'} "
          f"(range={max(tmpl_aucs) - min(tmpl_aucs):.3f})")
    print(f"  10. IO position AUCs similar: "
          f"{'PASS' if abs(auc_first - auc_second) < 0.1 else 'CHECK'} "
          f"(diff={abs(auc_first - auc_second):.3f})")
    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
