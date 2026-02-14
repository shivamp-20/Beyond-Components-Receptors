#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gp_phase_and_autopsy.py — Phase Diagram + Error Autopsy

Loads cached tensors from exp0c.  No new forward passes needed.

Free Win 1 — Phase Diagram:
  2D scatter of R1 vs R3 margin (how much each helps the correct answer).
  4-panel evolution across layers showing clusters separate.

Free Win 2 — Error Autopsy:
  Classify every model error by which functional channel failed.
  Type A = promotion failure, B = inhibition failure, C = both, D = margin.

Run on Kaggle:
  !python gp_phase_and_autopsy.py \\
      --tensors outputs/gp/exp0c_tensors.pt \\
      --data_dir data_main \\
      --csv train_1k_gp.csv \\
      --out_dir outputs/gp
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ═══════════════════════════════════════════════════════════════════
# CSV loader — reconstruct example list in exact exp0c order
# ═══════════════════════════════════════════════════════════════════

def sniff_delim(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"]).delimiter
    except Exception:
        return ","


def load_examples_with_meta(csv_path: str) -> List[Dict[str, str]]:
    """Reconstruct example list matching exp0c ordering, with name metadata."""
    txt = Path(csv_path).read_text(encoding="utf-8", errors="ignore")
    delim = sniff_delim(txt[:4000])
    reader = csv.DictReader(txt.splitlines(), delimiter=delim)

    examples: List[Dict[str, str]] = []
    for r in reader:
        p = (r.get("pronoun") or "").strip()
        if p not in ("he", "she"):
            continue
        examples.append({
            "text": (r.get("prefix") or "").rstrip(),
            "label": p,
            "name": (r.get("name") or "").strip(),
            "side": "clean",
        })
        cp = (r.get("corr_pronoun") or "").strip()
        if cp in ("he", "she"):
            examples.append({
                "text": (r.get("corr_prefix") or "").rstrip(),
                "label": cp,
                "name": (r.get("corr_name") or "").strip(),
                "side": "corr",
            })
    return examples


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tensors", default="outputs/gp/exp0c_tensors.pt")
    ap.add_argument("--data_dir", default="data_main")
    ap.add_argument("--csv", default="train_1k_gp.csv")
    ap.add_argument("--out_dir", default="outputs/gp")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── Load cached tensors ──
    print("[LOAD] Loading exp0c tensors...")
    ckpt = torch.load(args.tensors, map_location="cpu")
    g = ckpt["g"]                   # (3, 13, N)
    polarity = ckpt["polarity"]     # (3,)
    ys = ckpt["ys"]                 # (N,)
    he_logits = ckpt["he_logits"]   # (N,)
    she_logits = ckpt["she_logits"] # (N,)
    layer_labels = ckpt["layer_labels"]  # [-1, 0, ..., 11]

    K, L, N = g.shape
    final_idx = L - 1               # index 12 → layer 11

    print(f"  N={N}, K={K} receptors, L={L} layer snapshots")
    print(f"  polarity: {polarity.tolist()}")

    # ── Load CSV for name analysis ──
    csv_path = Path(args.data_dir) / args.csv
    meta = None
    if csv_path.exists():
        meta = load_examples_with_meta(str(csv_path))
        if len(meta) != N:
            print(f"[WARN] CSV examples ({len(meta)}) != tensor N ({N}). "
                  "Name analysis disabled.")
            meta = None
        else:
            print(f"  Loaded {len(meta)} examples with name metadata")

    # ── Derived quantities ──
    pol_g = polarity.view(K, 1, 1) * g           # (3, 13, N) polarity-adjusted

    # Margin: how much each receptor helps the CORRECT answer
    # margin[k, l, i] = ys[i] * polarity[k] * g[k, l, i]
    margin = ys.view(1, 1, N) * pol_g             # (3, 13, N)

    # Combined receptor scores at each layer
    s_all = pol_g.sum(dim=0)                      # (13, N) R1+R2+R3
    s_R13 = pol_g[0] + pol_g[2]                   # (13, N) R1+R3 only

    # Model predictions from cached logits
    model_pred = torch.where(he_logits > she_logits,
                             torch.ones(N), -torch.ones(N))
    model_correct = (model_pred == ys)
    model_error = ~model_correct

    n_correct = int(model_correct.sum().item())
    n_error = int(model_error.sum().item())
    acc = n_correct / N

    # Receptor-based predictions at final layer
    rec_pred_all = torch.where(s_all[final_idx] > 0, torch.ones(N), -torch.ones(N))
    rec_pred_R13 = torch.where(s_R13[final_idx] > 0, torch.ones(N), -torch.ones(N))
    rec_correct_all = (rec_pred_all == ys)
    rec_correct_R13 = (rec_pred_R13 == ys)
    agree_all = (model_pred == rec_pred_all)
    agree_R13 = (model_pred == rec_pred_R13)

    # ═══════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE DIAGRAM + ERROR AUTOPSY")
    print("=" * 70)

    print(f"\nACCURACY COMPARISON:")
    print(f"  Model (he vs she logit):  {n_correct}/{N} = {acc:.1%}")
    print(f"  Receptor (R1+R2+R3):      {int(rec_correct_all.sum())}/{N}"
          f" = {rec_correct_all.float().mean():.1%}")
    print(f"  Receptor (R1+R3 only):    {int(rec_correct_R13.sum())}/{N}"
          f" = {rec_correct_R13.float().mean():.1%}")

    print(f"\nMODEL-RECEPTOR AGREEMENT:")
    print(f"  Model vs R1+R2+R3: {int(agree_all.sum())}/{N}"
          f" = {agree_all.float().mean():.1%}")
    print(f"  Model vs R1+R3:    {int(agree_R13.sum())}/{N}"
          f" = {agree_R13.float().mean():.1%}")

    # ═══════════════════════════════════════════════════════════
    # ERROR AUTOPSY
    # ═══════════════════════════════════════════════════════════

    # Final-layer margins (numpy for convenience)
    m1 = margin[0, final_idx, :].numpy()       # R1 margin
    m3 = margin[2, final_idx, :].numpy()       # R3 margin
    err = model_error.numpy().astype(bool)
    cor = model_correct.numpy().astype(bool)

    # Error type classification
    type_A = err & (m1 < 0) & (m3 >= 0)    # promotion failure
    type_B = err & (m1 >= 0) & (m3 < 0)    # inhibition failure
    type_C = err & (m1 < 0) & (m3 < 0)     # both fail
    type_D = err & (m1 >= 0) & (m3 >= 0)   # margin / other-component failure

    nA = int(type_A.sum())
    nB = int(type_B.sum())
    nC = int(type_C.sum())
    nD = int(type_D.sum())
    ne = max(1, n_error)

    print(f"\n{'=' * 70}")
    print(f"ERROR AUTOPSY  ({n_error} errors / {N} examples = {100*n_error/N:.1f}%)")
    print(f"{'=' * 70}")
    print(f"\n  Type A (promotion failure:  R1 wrong, R3 right):   "
          f"{nA:3d}  ({100*nA/ne:.1f}% of errors)")
    print(f"  Type B (inhibition failure: R1 right, R3 wrong):   "
          f"{nB:3d}  ({100*nB/ne:.1f}% of errors)")
    print(f"  Type C (both fail:          R1 wrong, R3 wrong):   "
          f"{nC:3d}  ({100*nC/ne:.1f}% of errors)")
    print(f"  Type D (margin failure:     both right, model wrong): "
          f"{nD:3d}  ({100*nD/ne:.1f}% of errors)")
    print(f"  Sum check: {nA+nB+nC+nD} == {n_error}")

    # Circuit coverage: errors explained by R1+R3
    explained = nA + nB + nC
    print(f"\n  Circuit coverage: {explained}/{n_error}"
          f" = {100*explained/ne:.1f}% of errors explained by R1+R3 failures")
    print(f"  Unexplained (Type D): {nD}/{n_error}"
          f" = {100*nD/ne:.1f}% — circuit limitation")

    # Per-group receptor margins
    print(f"\nMEAN RECEPTOR MARGINS:")
    print(f"  {'Group':22s}  {'margin_R1':>10s}  {'margin_R3':>10s}"
          f"  {'combined':>10s}  {'n':>5s}")
    for label, mask in [("Correct", cor), ("All errors", err),
                        ("Type A (prom fail)", type_A),
                        ("Type B (inhib fail)", type_B),
                        ("Type C (both fail)", type_C),
                        ("Type D (margin fail)", type_D)]:
        if mask.sum() == 0:
            continue
        print(f"  {label:22s}  {m1[mask].mean():+10.2f}  {m3[mask].mean():+10.2f}"
              f"  {(m1[mask]+m3[mask]).mean():+10.2f}  {mask.sum():5d}")

    # Quadrant analysis (ALL examples)
    q_pp = (m1 >= 0) & (m3 >= 0)   # both help
    q_pn = (m1 >= 0) & (m3 < 0)    # R1 helps, R3 hurts
    q_np = (m1 < 0) & (m3 >= 0)    # R1 hurts, R3 helps
    q_nn = (m1 < 0) & (m3 < 0)     # both hurt

    print(f"\nQUADRANT POPULATIONS:")
    print(f"  {'Quadrant':28s}  {'n':>5s}  {'(%)':>6s}  {'errors':>6s}  {'acc':>6s}")
    for lab, mask in [("R1+/R3+ (both help)", q_pp),
                      ("R1+/R3- (inhib fails)", q_pn),
                      ("R1-/R3+ (prom fails)", q_np),
                      ("R1-/R3- (both fail)", q_nn)]:
        nq = int(mask.sum())
        neq = int((mask & err).sum())
        aq = 1.0 - neq / max(1, nq)
        print(f"  {lab:28s}  {nq:5d}  {100*nq/N:5.1f}%  {neq:6d}  {aq:5.1%}")

    # R1-R3 disagreement
    disagree = ((m1 > 0) & (m3 < 0)) | ((m1 < 0) & (m3 > 0))
    print(f"\nR1-R3 DISAGREEMENT RATE:")
    print(f"  Overall:       {disagree.sum()}/{N} = {disagree.mean():.1%}")
    print(f"  Among correct: {(disagree & cor).sum()}/{cor.sum()}"
          f" = {(disagree & cor).sum()/max(1,cor.sum()):.1%}")
    print(f"  Among errors:  {(disagree & err).sum()}/{err.sum()}"
          f" = {(disagree & err).sum()/max(1,err.sum()):.1%}")

    # ── Name analysis ──
    if meta is not None:
        error_names: Counter = Counter()
        total_names: Counter = Counter()
        for i in range(N):
            name = meta[i]["name"]
            total_names[name] += 1
            if err[i]:
                error_names[name] += 1

        print(f"\nTOP ERROR-PRONE NAMES (≥2 appearances, sorted by error count):")
        print(f"  {'Name':15s}  {'Errors':>6s}  {'Total':>5s}  {'Rate':>6s}")
        scored = [(name, error_names.get(name, 0), cnt)
                  for name, cnt in total_names.items() if cnt >= 2]
        scored.sort(key=lambda x: (-x[1], -x[2]))
        for name, errs, total in scored[:20]:
            if errs > 0:
                print(f"  {name:15s}  {errs:6d}  {total:5d}  {100*errs/total:5.1f}%")

        # Sample error sentences
        print(f"\nSAMPLE ERROR SENTENCES (up to 5 per type):")
        for label, mask in [("Type A", type_A), ("Type B", type_B),
                            ("Type C", type_C), ("Type D", type_D)]:
            idxs = np.where(mask)[0][:5]
            if len(idxs) == 0:
                continue
            print(f"\n  --- {label} ---")
            for i in idxs:
                ex = meta[i]
                print(f"    [{i}] name={ex['name']:12s} label={ex['label']}"
                      f"  m_R1={m1[i]:+.2f}  m_R3={m3[i]:+.2f}"
                      f"  | {ex['text'][:65]}...")

    # ═══════════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════════

    # ── Plot 1: Main Phase Diagram ──
    fig, ax = plt.subplots(figsize=(7, 6))

    ax.scatter(m1[cor], m3[cor], s=8, alpha=0.2, c="#4a90d9",
               label=f"Correct (n={n_correct})", zorder=2, rasterized=True)
    ax.scatter(m1[err], m3[err], s=35, alpha=0.85, c="#e74c3c",
               marker="x", linewidths=1.5,
               label=f"Error (n={n_error})", zorder=3)

    # Decision boundary: m_R1 + m_R3 = 0
    lim = max(np.abs(m1).max(), np.abs(m3).max()) * 1.1
    ax.plot([-lim, lim], [lim, -lim], "k--", alpha=0.4, lw=1, label="R1+R3=0")
    ax.axhline(0, color="gray", alpha=0.25, lw=0.5)
    ax.axvline(0, color="gray", alpha=0.25, lw=0.5)

    # Quadrant annotations
    off = lim * 0.82
    ax.text(off, off, f"Both help\nn={int(q_pp.sum())}",
            ha="center", va="center", fontsize=8, alpha=0.45)
    ax.text(-off, off, f"R1 fails\nn={int(q_np.sum())}",
            ha="center", va="center", fontsize=8, alpha=0.45)
    ax.text(off, -off, f"R3 fails\nn={int(q_pn.sum())}",
            ha="center", va="center", fontsize=8, alpha=0.45)
    ax.text(-off, -off, f"Both fail\nn={int(q_nn.sum())}",
            ha="center", va="center", fontsize=8, alpha=0.45)

    ax.set_xlabel("R1 margin  (y · v₁ · g₁)  ←  promotion channel", fontsize=11)
    ax.set_ylabel("R3 margin  (y · v₃ · g₃)  ←  inhibition channel", fontsize=11)
    ax.set_title("Phase Diagram: Decision Landscape (Layer 11)", fontsize=13)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.12)
    plt.tight_layout()
    p1 = out_dir / "phase_diagram.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[PLOT] {p1}")

    # ── Plot 2: 4-Panel Phase Evolution ──
    # Use "he-direction" coordinates, colored by label
    snap_indices = [0, 6, 10, 12]
    snap_labels = ["Layer −1\n(embedding)", "Layer 5",
                   "Layer 9\n(R3 commits)", "Layer 11\n(final)"]

    x_he = pol_g[0].numpy()    # (13, N) R1 push toward "he"
    y_he = pol_g[2].numpy()    # (13, N) R3 push toward "he"

    male = (ys.numpy() == 1)
    female = (ys.numpy() == -1)

    # Use final-layer range for all panels to show absolute growth
    global_lim = max(np.abs(x_he[final_idx]).max(),
                     np.abs(y_he[final_idx]).max()) * 1.1

    fig, axes = plt.subplots(1, 4, figsize=(20, 4.8))
    for panel, (si, slab) in enumerate(zip(snap_indices, snap_labels)):
        ax = axes[panel]
        ax.scatter(x_he[si, female], y_he[si, female],
                   s=5, alpha=0.2, c="#e74c3c", rasterized=True)
        ax.scatter(x_he[si, male], y_he[si, male],
                   s=5, alpha=0.2, c="#4a90d9", rasterized=True)

        ax.plot([-global_lim, global_lim], [global_lim, -global_lim],
                "k--", alpha=0.25, lw=0.8)
        ax.axhline(0, color="gray", alpha=0.2, lw=0.5)
        ax.axvline(0, color="gray", alpha=0.2, lw=0.5)

        ax.set_title(slab, fontsize=10)
        ax.set_xlim(-global_lim, global_lim)
        ax.set_ylim(-global_lim, global_lim)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.1)
        if panel == 0:
            ax.set_ylabel("R3 → he", fontsize=10)
            from matplotlib.lines import Line2D
            legend_elements = [Line2D([0], [0], marker="o", color="w",
                                      markerfacecolor="#4a90d9", markersize=6,
                                      label="he"),
                               Line2D([0], [0], marker="o", color="w",
                                      markerfacecolor="#e74c3c", markersize=6,
                                      label="she")]
            ax.legend(handles=legend_elements, fontsize=8, loc="lower right")
        ax.set_xlabel("R1 → he", fontsize=10)

    fig.suptitle("Phase Evolution: Cluster Separation Across Layers",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    p2 = out_dir / "phase_evolution.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {p2}")

    # ── Plot 3: Error Type Bar Chart ──
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    types_lab = ["Type A\nPromotion\nfailure",
                 "Type B\nInhibition\nfailure",
                 "Type C\nBoth\nfail",
                 "Type D\nMargin\nfailure"]
    counts = [nA, nB, nC, nD]
    colors = ["#3498db", "#e67e22", "#e74c3c", "#95a5a6"]
    bars = ax.bar(types_lab, counts, color=colors, edgecolor="white", lw=1.2)
    for bar, cnt in zip(bars, counts):
        if cnt > 0:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(counts) * 0.02,
                    f"{cnt}\n({100*cnt/ne:.0f}%)",
                    ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of errors", fontsize=11)
    ax.set_title(f"Error Classification ({n_error} model errors)", fontsize=12)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    p3 = out_dir / "error_types.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {p3}")

    # ── Plot 4: Margin Histograms ──
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (data, label) in zip(axes, [
            (m1, "R1 margin (promotion)"),
            (m3, "R3 margin (inhibition)"),
            (m1 + m3, "R1+R3 combined margin")]):
        ax.hist(data[cor], bins=50, alpha=0.5, color="#4a90d9",
                label="Correct", density=True)
        ax.hist(data[err], bins=30, alpha=0.7, color="#e74c3c",
                label="Error", density=True)
        ax.axvline(0, color="black", ls="--", alpha=0.5, lw=1)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel("Density")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.15)

    fig.suptitle("Margin Distributions: Correct vs Error",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    p4 = out_dir / "margin_histograms.png"
    fig.savefig(p4, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {p4}")

    # ── Plot 5: Boxplots by Error Type ──
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    groups = []
    labels_box = []
    box_colors = []
    for mask, lab, col in [(cor, "Correct", "#4a90d9"),
                           (type_A, "A:Prom", "#3498db"),
                           (type_B, "B:Inhib", "#e67e22"),
                           (type_C, "C:Both", "#e74c3c"),
                           (type_D, "D:Margin", "#95a5a6")]:
        if mask.sum() > 0:
            groups.append(mask)
            labels_box.append(lab)
            box_colors.append(col)

    for ax, (data, title) in zip(axes, [(m1, "R1 (promotion) margin"),
                                        (m3, "R3 (inhibition) margin")]):
        bp = ax.boxplot([data[mask] for mask in groups],
                        labels=labels_box, patch_artist=True,
                        showfliers=True, flierprops=dict(markersize=2, alpha=0.3))
        for patch, col in zip(bp["boxes"], box_colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.6)
        ax.axhline(0, color="red", ls="--", alpha=0.5)
        ax.set_ylabel("Margin")
        ax.set_title(title, fontsize=11)
        ax.grid(True, axis="y", alpha=0.2)

    plt.tight_layout()
    p5 = out_dir / "margin_by_error_type.png"
    fig.savefig(p5, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {p5}")

    # ── Plot 6: Receptor explanation coverage across layers ──
    # At each layer: what fraction of model errors have negative R1+R3 margin?
    coverage = []
    for li in range(L):
        m_combined_l = margin[0, li, :].numpy() + margin[2, li, :].numpy()
        explained_l = err & (m_combined_l < 0)
        coverage.append(int(explained_l.sum()) / ne)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(layer_labels, coverage, "o-", color="#2c3e50", markersize=5)
    ax.set_xlabel("Layer (−1 = embedding, 0..11 = post-block)", fontsize=11)
    ax.set_ylabel("Fraction of model errors with R1+R3 margin < 0", fontsize=10)
    ax.set_title("Circuit Explanation Coverage Across Layers", fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.axhline(1.0, color="gray", ls=":", alpha=0.3)
    ax.grid(True, alpha=0.2)
    plt.tight_layout()
    p6 = out_dir / "circuit_coverage.png"
    fig.savefig(p6, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {p6}")

    # ═══════════════ JSON summary ═══════════════
    summary: Dict[str, Any] = {
        "experiment": "phase_diagram_and_error_autopsy",
        "N": N,
        "model_accuracy": acc,
        "receptor_accuracy_all3": float(rec_correct_all.float().mean()),
        "receptor_accuracy_R13": float(rec_correct_R13.float().mean()),
        "agreement_all3": float(agree_all.float().mean()),
        "agreement_R13": float(agree_R13.float().mean()),
        "n_errors": n_error,
        "error_types": {
            "A_promotion_failure": nA,
            "B_inhibition_failure": nB,
            "C_both_fail": nC,
            "D_margin_failure": nD,
        },
        "error_type_fractions": {
            "A": round(nA / ne, 3),
            "B": round(nB / ne, 3),
            "C": round(nC / ne, 3),
            "D": round(nD / ne, 3),
        },
        "circuit_coverage": round(explained / ne, 3),
        "quadrants": {
            "both_help": int(q_pp.sum()),
            "inhib_fails": int(q_pn.sum()),
            "prom_fails": int(q_np.sum()),
            "both_fail": int(q_nn.sum()),
        },
        "disagreement_rate_overall": round(float(disagree.mean()), 3),
        "coverage_by_layer": {str(ll): round(c, 3)
                              for ll, c in zip(layer_labels, coverage)},
    }
    jp = out_dir / "phase_autopsy_results.json"
    with open(jp, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[SAVE] {jp}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
