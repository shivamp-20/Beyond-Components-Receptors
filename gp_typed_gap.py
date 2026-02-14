#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Type D Gap Analysis — Decomposing the Circuit-Model Disagreement

Loads exp0c cached tensors. Loads model ONLY for W_U and ln_final
(no forward passes through transformer blocks).

Answers: why does the model disagree with the receptor circuit on 52%
of errors?

Run on Kaggle:
  !python gp_typed_gap.py \
      --tensors outputs/gp/exp0c_tensors.pt \
      --out_dir outputs/gp \
      --receptors "10,9,0,+1;11,8,6,+1;9,7,1,-1" \
      --device cuda

View results:
  from IPython.display import display, Image
  display(Image("outputs/gp/typed_scatter.png"))
  display(Image("outputs/gp/typed_margin_dist.png"))
  display(Image("outputs/gp/typed_variance_pie.png"))
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def load_receptors(svd_path: str, spec: str):
    path = Path(svd_path)
    if not path.exists():
        for alt_name in ["svd_cache.pt", "ov_svd_cache.pt"]:
            alt = path.parent / alt_name
            if alt.exists():
                path = alt
                break
    ckpt = torch.load(str(path), map_location="cpu")
    ov = ckpt.get("ov", ckpt.get("svd", {}).get("ov", ckpt))
    dirs, pols, meta = [], [], []
    for idx, s in enumerate(spec.split(";"), 1):
        parts = [x.strip() for x in s.strip().split(",")]
        layer, head, sv, pol = int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        cell = ov[layer][head]
        Vh = cell["Vh"] if isinstance(cell, dict) else cell.Vh
        v = Vh[sv].detach().float().cpu()
        v = v / (v.norm() + 1e-12)
        dirs.append(v)
        pols.append(pol)
        meta.append(dict(name=f"R{idx}", layer=layer, head=head, sv=sv, pol=pol))
    return torch.stack(dirs), torch.tensor(pols, dtype=torch.float32), meta


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors", default="outputs/gp/exp0c_tensors.pt")
    ap.add_argument("--svd_cache", default="")
    ap.add_argument("--out_dir", default="outputs/gp")
    ap.add_argument("--receptors", default="10,9,0,+1;11,8,6,+1;9,7,1,-1")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TYPE D GAP ANALYSIS")
    print("=" * 70)

    # ── Load cached tensors ──
    print("\n[LOAD] exp0c tensors ...")
    ckpt = torch.load(args.tensors, map_location="cpu")
    g = ckpt["g"]                    # (3, 13, N)
    polarity = ckpt["polarity"]      # (3,)
    ys = ckpt["ys"]                  # (N,)  +1 or -1
    he_logits = ckpt["he_logits"]    # (N,)
    she_logits = ckpt["she_logits"]  # (N,)
    resid_all = ckpt["resid_all"]    # (13, N, 768)
    layer_labels = ckpt["layer_labels"]

    K, L, N = g.shape
    final_idx = L - 1
    d_model = resid_all.shape[2]

    print(f"  N={N}, K={K}, d_model={d_model}")

    # ── Load model for W_U and ln_final (no forward passes) ──
    print("\n[MODEL] Loading gpt2-small for W_U and ln_final ...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=str(args.device))
    tok = model.tokenizer
    he_id = tok.encode(" he", add_special_tokens=False)[0]
    she_id = tok.encode(" she", add_special_tokens=False)[0]

    W_U = model.W_U.detach().cpu().float()          # (d_model, vocab)
    b_U = model.b_U.detach().cpu().float()           # (vocab,)
    # ln_final parameters
    ln_w = model.ln_final.w.detach().cpu().float()   # (d_model,)
    ln_b = model.ln_final.b.detach().cpu().float()   # (d_model,)

    # w_diff: unembedding direction for he−she
    w_diff = W_U[:, he_id] - W_U[:, she_id]         # (d_model,)
    b_diff = float(b_U[he_id] - b_U[she_id])

    print(f"  ||w_diff|| = {w_diff.norm():.4f}")
    print(f"  b_diff = {b_diff:.6f}")

    # ── Load receptor directions ──
    svd_path = args.svd_cache or str(out_dir / "ov_svd_cache.pt")
    rec_dirs, pol_tensor, rec_meta = load_receptors(svd_path, args.receptors)
    pol = pol_tensor.numpy()

    # ═══════════════════════════════════════════════════════
    # STEP 1: Basic reconstruction — receptor score vs model logit
    # ═══════════════════════════════════════════════════════

    # Final-layer quantities
    g_final = g[:, final_idx, :].numpy()          # (3, N)
    pol_g = np.array([pol[k] * g_final[k] for k in range(K)])  # (3, N)
    receptor_score = pol_g[0] + pol_g[2]          # R1+R3 combined (N,)
    receptor_score_all3 = pol_g.sum(axis=0)       # R1+R2+R3

    ys_np = ys.numpy()
    model_diff = (he_logits - she_logits).numpy()  # positive = he

    # Signed model margin: ys * model_diff (positive = model correct)
    model_margin = ys_np * model_diff
    # Signed receptor margin: ys * receptor_score
    receptor_margin = ys_np * receptor_score

    # Model correct/error
    model_correct = model_margin > 0
    model_error = ~model_correct

    # Error types (same as autopsy)
    m1 = ys_np * pol[0] * g_final[0]   # R1 margin
    m3 = ys_np * pol[2] * g_final[2]   # R3 margin
    err = model_error

    type_D = err & (m1 >= 0) & (m3 >= 0)
    type_C = err & (m1 < 0) & (m3 < 0)
    type_B = err & (m1 >= 0) & (m3 < 0)
    type_A = err & (m1 < 0) & (m3 >= 0)

    n_err = int(err.sum())
    nD = int(type_D.sum())

    print(f"\n  Model errors: {n_err}/{N} = {100*n_err/N:.1f}%")
    print(f"  Type D: {nD} ({100*nD/max(1,n_err):.1f}% of errors)")

    # ═══════════════════════════════════════════════════════
    # STEP 2: Correlation — R² between receptor and model
    # ═══════════════════════════════════════════════════════

    # R² of model_diff vs receptor_score (all examples)
    def r_squared(x, y):
        xm = x - x.mean()
        ym = y - y.mean()
        ss_res = np.sum((y - (np.polyfit(x, y, 1)[0] * x + np.polyfit(x, y, 1)[1]))**2)
        ss_tot = np.sum(ym**2)
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # Simpler: Pearson r², then R² = r²
    corr_13 = np.corrcoef(receptor_score, model_diff)[0, 1]
    corr_all = np.corrcoef(receptor_score_all3, model_diff)[0, 1]
    r2_13 = corr_13 ** 2
    r2_all = corr_all ** 2

    print(f"\n{'=' * 70}")
    print("RECEPTOR-MODEL CORRELATION")
    print("=" * 70)
    print(f"  Pearson r (R1+R3 vs logit_diff):   {corr_13:+.4f}  → R² = {r2_13:.4f}")
    print(f"  Pearson r (R1+R2+R3 vs logit_diff): {corr_all:+.4f}  → R² = {r2_all:.4f}")

    # Linear regression: model_diff ≈ a * receptor_score + b
    a_13, b_13 = np.polyfit(receptor_score, model_diff, 1)
    predicted_diff = a_13 * receptor_score + b_13
    residual = model_diff - predicted_diff

    print(f"\n  Linear fit: logit_diff ≈ {a_13:.4f} × receptor_score + {b_13:.4f}")
    print(f"  Residual std: {residual.std():.4f}")
    print(f"  Model logit_diff std: {model_diff.std():.4f}")
    print(f"  → Receptor explains {r2_13:.1%} of logit variance")

    # ═══════════════════════════════════════════════════════
    # STEP 3: Type D margin analysis
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("TYPE D MARGIN ANALYSIS")
    print("=" * 70)

    D_model_margin = model_margin[type_D]      # negative (model wrong)
    D_receptor_margin = receptor_margin[type_D]  # positive (circuit right)
    D_model_diff = model_diff[type_D]
    D_ys = ys_np[type_D]

    print(f"\n  Type D errors (N={nD}):")
    print(f"    Model margin (ys × logit_diff):  "
          f"mean={D_model_margin.mean():+.2f}  "
          f"median={np.median(D_model_margin):+.2f}  "
          f"std={D_model_margin.std():.2f}")
    print(f"    Receptor margin (ys × R1+R3):    "
          f"mean={D_receptor_margin.mean():+.2f}  "
          f"median={np.median(D_receptor_margin):+.2f}  "
          f"std={D_receptor_margin.std():.2f}")

    # Compare to correct examples
    C_model_margin = model_margin[model_correct]
    C_receptor_margin = receptor_margin[model_correct]

    print(f"\n  Correct examples (N={int(model_correct.sum())}):")
    print(f"    Model margin:    mean={C_model_margin.mean():+.2f}  "
          f"median={np.median(C_model_margin):+.2f}")
    print(f"    Receptor margin: mean={C_receptor_margin.mean():+.2f}  "
          f"median={np.median(C_receptor_margin):+.2f}")

    # How small are Type D model margins?
    D_abs_model = np.abs(D_model_margin)
    C_abs_model = np.abs(C_model_margin)

    print(f"\n  |Model margin| distribution:")
    print(f"    Type D: mean={D_abs_model.mean():.2f}  "
          f"median={np.median(D_abs_model):.2f}  "
          f"max={D_abs_model.max():.2f}")
    print(f"    Correct: mean={C_abs_model.mean():.2f}  "
          f"median={np.median(C_abs_model):.2f}")
    print(f"    Type D model confidence is {D_abs_model.mean()/C_abs_model.mean():.1%}"
          f" of correct-example confidence")

    # How many Type D are small-margin?
    for threshold in [0.5, 1.0, 2.0, 5.0]:
        n_small = int((D_abs_model < threshold).sum())
        print(f"    |model margin| < {threshold}: {n_small}/{nD}"
              f" = {100*n_small/max(1,nD):.0f}%")

    # ═══════════════════════════════════════════════════════
    # STEP 4: Residual analysis — what's the "anti-circuit" signal?
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("RESIDUAL SIGNAL (model − receptor prediction)")
    print("=" * 70)

    resid_signal = model_diff - predicted_diff  # (N,)

    print(f"\n  Residual = actual logit_diff − (a × receptor_score + b)")
    print(f"  Overall: mean={resid_signal.mean():.4f}  std={resid_signal.std():.4f}")
    print(f"  Type D:  mean={resid_signal[type_D].mean():+.4f}"
          f"  std={resid_signal[type_D].std():.4f}")
    print(f"  Correct: mean={resid_signal[model_correct].mean():+.4f}"
          f"  std={resid_signal[model_correct].std():.4f}")

    # For Type D: is the residual consistently anti-correct?
    D_resid_signed = D_ys * resid_signal[type_D]  # negative = anti-correct
    print(f"\n  Type D signed residual (ys × residual):")
    print(f"    mean={D_resid_signed.mean():+.4f}  "
          f"fraction < 0: {(D_resid_signed < 0).mean():.1%}")
    print(f"    → {'Residual is consistently anti-correct' if (D_resid_signed < 0).mean() > 0.5 else 'Mixed signal'}")

    # ═══════════════════════════════════════════════════════
    # STEP 5: Subspace decomposition (using ln_final + W_U)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("LOGIT VARIANCE DECOMPOSITION (receptor subspace)")
    print("=" * 70)

    # Get final-layer residuals
    x_final = resid_all[final_idx]   # (N, d_model) pre-ln_final

    # Apply ln_final manually: ln(x) = (x - mean) / std * w + b
    x_mean = x_final.mean(dim=-1, keepdim=True)
    x_std = x_final.std(dim=-1, keepdim=True, unbiased=False)
    x_ln = (x_final - x_mean) / (x_std + 1e-5) * ln_w.unsqueeze(0) + ln_b.unsqueeze(0)

    # Verify: our ln_final matches cached logits
    recon_he = (x_ln @ W_U[:, he_id] + b_U[he_id]).numpy()
    recon_she = (x_ln @ W_U[:, she_id] + b_U[she_id]).numpy()
    recon_diff = recon_he - recon_she
    recon_error = np.abs(recon_diff - model_diff).mean()
    print(f"\n  [SANITY] ln_final reconstruction error: {recon_error:.6f}"
          f"  (should be ~0)")

    # Project x_ln onto receptor subspace (handling non-orthogonality)
    V = rec_dirs.float()              # (3, d_model)
    G = V @ V.T                       # (3, 3) Gram matrix
    G_inv = torch.linalg.inv(G)       # (3, 3)

    # Receptor activations in post-ln space
    g_ln = (x_ln @ V.T)              # (N, 3) — projections onto receptor dirs
    # Oblique projection coefficients
    coeffs = (g_ln @ G_inv.T)        # (N, 3)
    # Reconstruct in receptor subspace
    x_parallel = coeffs @ V           # (N, d_model)
    x_perp = x_ln - x_parallel        # (N, d_model)

    # Logit contributions
    logit_parallel = (x_parallel @ w_diff).numpy() + b_diff  # receptor subspace
    logit_perp = (x_perp @ w_diff).numpy()                    # orthogonal complement
    logit_total = logit_parallel + logit_perp

    # Verify decomposition
    decomp_error = np.abs(logit_total - recon_diff).mean()
    print(f"  [SANITY] Decomposition error: {decomp_error:.6f}  (should be ~0)")

    # Variance decomposition
    var_total = np.var(recon_diff)
    var_parallel = np.var(logit_parallel)
    var_perp = np.var(logit_perp)
    cov_term = 2 * np.cov(logit_parallel, logit_perp)[0, 1]

    print(f"\n  Var(logit_diff) = Var(parallel) + Var(perp) + 2*Cov")
    print(f"    Var(total):    {var_total:.4f}")
    print(f"    Var(parallel): {var_parallel:.4f}  ({100*var_parallel/var_total:.1f}%)")
    print(f"    Var(perp):     {var_perp:.4f}  ({100*var_perp/var_total:.1f}%)")
    print(f"    2*Cov:         {cov_term:.4f}  ({100*cov_term/var_total:.1f}%)")

    # Correlation between parallel and perp
    corr_pp = np.corrcoef(logit_parallel, logit_perp)[0, 1]
    print(f"    corr(parallel, perp) = {corr_pp:+.4f}")

    # For Type D: which component dominates?
    print(f"\n  Type D breakdown:")
    print(f"    logit_parallel (circuit):  mean={logit_parallel[type_D].mean():+.2f}"
          f"  (signed: ys × = {(D_ys * logit_parallel[type_D]).mean():+.2f})")
    print(f"    logit_perp (other):        mean={logit_perp[type_D].mean():+.2f}"
          f"  (signed: ys × = {(D_ys * logit_perp[type_D]).mean():+.2f})")
    print(f"    For Type D: circuit says {'correct' if (D_ys * logit_parallel[type_D]).mean() > 0 else 'wrong'},"
          f" other says {'correct' if (D_ys * logit_perp[type_D]).mean() > 0 else 'wrong'}")

    # ═══════════════════════════════════════════════════════
    # STEP 6: Per error type subspace analysis
    # ═══════════════════════════════════════════════════════
    print(f"\n  Per error type (signed logit contributions, ys × component):")
    print(f"  {'Type':15s}  {'N':>4s}  {'parallel':>10s}  {'perp':>10s}  {'total':>10s}")
    for label, mask in [("Correct", model_correct),
                        ("Type B (inhib)", type_B),
                        ("Type C (both)", type_C),
                        ("Type D (margin)", type_D)]:
        if mask.sum() == 0:
            continue
        y_m = ys_np[mask]
        par_m = (y_m * logit_parallel[mask]).mean()
        perp_m = (y_m * logit_perp[mask]).mean()
        tot_m = (y_m * recon_diff[mask]).mean()
        print(f"  {label:15s}  {int(mask.sum()):4d}  {par_m:+10.2f}"
              f"  {perp_m:+10.2f}  {tot_m:+10.2f}")

    # ═══════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════

    # ── Plot 1: Receptor score vs Model logit diff ──
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(receptor_score[model_correct], model_diff[model_correct],
               s=6, alpha=0.15, c="#4a90d9", label="Correct", rasterized=True)
    ax.scatter(receptor_score[type_C], model_diff[type_C],
               s=30, alpha=0.8, c="#e74c3c", marker="x", lw=1.2,
               label=f"Type C: both fail ({int(type_C.sum())})")
    ax.scatter(receptor_score[type_B], model_diff[type_B],
               s=30, alpha=0.8, c="#e67e22", marker="x", lw=1.2,
               label=f"Type B: inhib fail ({int(type_B.sum())})")
    ax.scatter(receptor_score[type_D], model_diff[type_D],
               s=30, alpha=0.8, c="#95a5a6", marker="x", lw=1.5,
               label=f"Type D: margin ({nD})")

    # Regression line
    xx = np.linspace(receptor_score.min(), receptor_score.max(), 100)
    ax.plot(xx, a_13 * xx + b_13, "k--", alpha=0.5, lw=1,
            label=f"Fit: R²={r2_13:.3f}")

    ax.axhline(0, color="gray", alpha=0.3, lw=0.8)
    ax.axvline(0, color="gray", alpha=0.3, lw=0.8)
    ax.set_xlabel("Receptor score (pol₁g₁ + pol₃g₃)", fontsize=11)
    ax.set_ylabel("Model logit difference (he − she)", fontsize=11)
    ax.set_title("Receptor Score vs Model Decision", fontsize=13)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, alpha=0.12)
    plt.tight_layout()
    p1 = out_dir / "typed_scatter.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"\n[PLOT] {p1}")

    # ── Plot 2: Model margin distributions by error type ──
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bins = np.linspace(-15, 15, 60)
    ax.hist(model_margin[model_correct], bins=bins, alpha=0.4,
            color="#4a90d9", label="Correct", density=True)
    if type_D.sum() > 0:
        ax.hist(model_margin[type_D], bins=bins, alpha=0.7,
                color="#95a5a6", label="Type D", density=True)
    if type_C.sum() > 0:
        ax.hist(model_margin[type_C], bins=bins, alpha=0.7,
                color="#e74c3c", label="Type C", density=True)
    ax.axvline(0, color="black", ls="--", alpha=0.5)
    ax.set_xlabel("Model margin (ys × logit_diff)", fontsize=10)
    ax.set_ylabel("Density", fontsize=10)
    ax.set_title("Model Margin by Error Type", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    ax = axes[1]
    ax.hist(np.abs(model_margin[type_D]), bins=30, alpha=0.7,
            color="#95a5a6", label="Type D |margin|")
    ax.axvline(np.median(np.abs(model_margin[type_D])), color="black",
               ls="--", alpha=0.6, label=f"median={np.median(np.abs(model_margin[type_D])):.2f}")
    ax.set_xlabel("|Model margin| for Type D errors", fontsize=10)
    ax.set_ylabel("Count", fontsize=10)
    ax.set_title(f"Type D: How Wrong Is the Model? (N={nD})", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()
    p2 = out_dir / "typed_margin_dist.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p2}")

    # ── Plot 3: Variance pie chart ──
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Pie: variance decomposition
    ax = axes[0]
    sizes = [max(0, var_parallel), max(0, var_perp)]
    labels_pie = [
        f"Receptor subspace\n({100*var_parallel/var_total:.1f}%)",
        f"Other components\n({100*var_perp/var_total:.1f}%)",
    ]
    colors_pie = ["#3498db", "#e74c3c"]
    ax.pie(sizes, labels=labels_pie, colors=colors_pie, autopct="%1.0f%%",
           startangle=90, textprops={"fontsize": 10})
    ax.set_title(f"Logit Variance Decomposition\n(covariance term: {100*cov_term/var_total:.1f}%)",
                 fontsize=11)

    # Bar: signed contributions by error type
    ax = axes[1]
    types = ["Correct", "Type B", "Type C", "Type D"]
    masks = [model_correct, type_B, type_C, type_D]
    par_vals, perp_vals = [], []
    for mask in masks:
        if mask.sum() == 0:
            par_vals.append(0)
            perp_vals.append(0)
        else:
            par_vals.append(float((ys_np[mask] * logit_parallel[mask]).mean()))
            perp_vals.append(float((ys_np[mask] * logit_perp[mask]).mean()))

    x = np.arange(len(types))
    w = 0.35
    ax.bar(x - w/2, par_vals, w, label="Receptor subspace",
           color="#3498db", edgecolor="white")
    ax.bar(x + w/2, perp_vals, w, label="Other components",
           color="#e74c3c", edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(types, fontsize=9)
    ax.set_ylabel("Mean signed logit contribution\n(ys × component)", fontsize=10)
    ax.set_title("Logit Decomposition by Error Type", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    p3 = out_dir / "typed_variance_pie.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p3}")

    # ═══════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════
    jp = out_dir / "typed_gap_results.json"
    with open(jp, "w") as f:
        json.dump({
            "R2_receptor_vs_model": round(r2_13, 4),
            "R2_all3_vs_model": round(r2_all, 4),
            "var_parallel_frac": round(float(var_parallel / var_total), 4),
            "var_perp_frac": round(float(var_perp / var_total), 4),
            "cov_frac": round(float(cov_term / var_total), 4),
            "typeD_mean_model_margin": round(float(D_model_margin.mean()), 4),
            "typeD_median_abs_model_margin": round(float(np.median(D_abs_model)), 4),
            "typeD_mean_receptor_margin": round(float(D_receptor_margin.mean()), 4),
            "typeD_resid_frac_anticorrect": round(float((D_resid_signed < 0).mean()), 4),
        }, f, indent=2)
    print(f"\n[SAVE] {jp}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
