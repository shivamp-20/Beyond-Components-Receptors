#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp 5b: Ambiguous Name Analysis

For 128 names that appear as both "he" and "she" in the dataset,
we run the SAME sentence through the model and measure what gender
the receptors assign based purely on name-embedding geometry.

Also runs unambiguous names through the same template as a control,
and compares AUC between the two pools.

Run on Kaggle:
  !python gp_exp5b_ambiguous.py \
      --svd_cache outputs/gp/ov_svd_cache.pt \
      --csv_dir data_main \
      --out_dir outputs/gp \
      --receptors "10,9,0,+1;11,8,6,+1;9,7,1,-1" \
      --device cuda

View results:
  from IPython.display import display, Image
  display(Image("outputs/gp/exp5b_ambig_scatter.png"))
  display(Image("outputs/gp/exp5b_auc_comparison.png"))
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from transformer_lens import HookedTransformer


# ═══════════════════════════════════════════════════════════════
# UTILITIES (same as exp5 main)
# ═══════════════════════════════════════════════════════════════

def try_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, s))
    except Exception:
        order = np.argsort(s)
        y_sorted = y[order]
        s_sorted = s[order]
        ranks = np.empty(len(y), dtype=np.float64)
        i, rank = 0, 1.0
        while i < len(y):
            j = i + 1
            while j < len(y) and s_sorted[j] == s_sorted[i]:
                j += 1
            ranks[i:j] = (rank + rank + j - i - 1.0) / 2.0
            rank += j - i
            i = j
        u = ranks[y_sorted == 1].sum() - n_pos * (n_pos + 1) / 2.0
        return float(u / (n_pos * n_neg))


def load_receptors(svd_path: str, spec: str):
    path = Path(svd_path)
    if not path.exists():
        alt = path.parent / "svd_cache.pt"
        if alt.exists():
            path = alt
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
# NAME EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_names_with_counts(csv_dir: str):
    """Returns male_only, female_only, ambiguous lists + usage counts for ambiguous."""
    male_set, female_set = set(), set()
    # Track per-name gender usage counts
    name_he_count: Counter = Counter()
    name_she_count: Counter = Counter()

    for fname in ["train_1k_gp.csv", "test_gp.csv", "val_gp.csv"]:
        fpath = Path(csv_dir) / fname
        if not fpath.exists():
            continue
        with open(fpath, encoding="utf-8", errors="ignore") as f:
            for r in csv.DictReader(f):
                pron = (r.get("pronoun") or "").strip().lower()
                name = (r.get("name") or "").strip()
                cpron = (r.get("corr_pronoun") or "").strip().lower()
                cname = (r.get("corr_name") or "").strip()

                if pron == "he" and name:
                    male_set.add(name)
                    name_he_count[name] += 1
                elif pron == "she" and name:
                    female_set.add(name)
                    name_she_count[name] += 1
                if cpron == "he" and cname:
                    male_set.add(cname)
                    name_he_count[cname] += 1
                elif cpron == "she" and cname:
                    female_set.add(cname)
                    name_she_count[cname] += 1

    ambiguous = sorted(male_set & female_set)
    male_only = sorted(male_set - set(ambiguous))
    female_only = sorted(female_set - set(ambiguous))

    ambig_info = {}
    for name in ambiguous:
        he_n = name_he_count[name]
        she_n = name_she_count[name]
        total = he_n + she_n
        ambig_info[name] = {
            "he_count": he_n, "she_count": she_n,
            "he_frac": he_n / total if total > 0 else 0.5,
            "lean": "male" if he_n > she_n else ("female" if she_n > he_n else "neutral"),
        }

    return male_only, female_only, ambiguous, ambig_info


# ═══════════════════════════════════════════════════════════════
# FORWARD PASS
# ═══════════════════════════════════════════════════════════════

TEMPLATE = "So {name} is a great listener, isn't"


@torch.no_grad()
def run_names(model, tok, names: List[str], rec_dirs: torch.Tensor,
              he_id: int, she_id: int, batch_size: int, device):
    """Run one template with a list of names. Returns activations."""
    texts = [TEMPLATE.format(name=n) for n in names]
    N = len(texts)
    K = rec_dirs.shape[0]

    enc = tok(texts, add_special_tokens=False, padding=True, return_tensors="pt")
    tokens = enc["input_ids"]
    attn_mask = enc.get("attention_mask", torch.ones_like(tokens))
    last_idx = (attn_mask.sum(dim=1) - 1).to(torch.long)

    g_all = np.zeros((N, K), dtype=np.float64)
    he_logits = np.zeros(N, dtype=np.float64)
    she_logits = np.zeros(N, dtype=np.float64)

    rec_dev = rec_dirs.to(device)
    hook = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"

    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        btok = tokens[start:end].to(device)
        blidx = last_idx[start:end].to(device)
        logits, cache = model.run_with_cache(btok, names_filter=[hook])
        B = end - start
        b_idx = torch.arange(B, device=device)
        resid = cache[hook][b_idx, blidx, :]
        g_all[start:end] = (resid @ rec_dev.T).cpu().numpy()
        dec_logits = logits[b_idx, blidx, :]
        he_logits[start:end] = dec_logits[:, he_id].cpu().numpy()
        she_logits[start:end] = dec_logits[:, she_id].cpu().numpy()

    return g_all, he_logits, she_logits


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svd_cache", required=True)
    ap.add_argument("--csv_dir", default="data_main")
    ap.add_argument("--out_dir", default="outputs/gp")
    ap.add_argument("--receptors", default="10,9,0,+1;11,8,6,+1;9,7,1,-1")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    print("=" * 70)
    print("EXP 5b: AMBIGUOUS NAME ANALYSIS")
    print("=" * 70)

    # ── Load model ──
    print("\n[MODEL] Loading gpt2-small ...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=str(device))
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    he_id = tok.encode(" he", add_special_tokens=False)[0]
    she_id = tok.encode(" she", add_special_tokens=False)[0]

    # ── Load receptors ──
    rec_dirs, polarity, rec_meta = load_receptors(args.svd_cache, args.receptors)
    pol = polarity.numpy()
    K = rec_dirs.shape[0]
    print(f"  Receptors: {[m['name'] for m in rec_meta]}")

    # ── Extract names ──
    male_only, female_only, ambiguous, ambig_info = extract_names_with_counts(args.csv_dir)
    print(f"\n[NAMES] male-only={len(male_only)}  female-only={len(female_only)}"
          f"  ambiguous={len(ambiguous)}")

    # Count how many ambiguous names lean male vs female
    lean_male = [n for n in ambiguous if ambig_info[n]["lean"] == "male"]
    lean_female = [n for n in ambiguous if ambig_info[n]["lean"] == "female"]
    lean_neutral = [n for n in ambiguous if ambig_info[n]["lean"] == "neutral"]
    print(f"  Ambiguous lean: male={len(lean_male)}, female={len(lean_female)},"
          f" neutral={len(lean_neutral)}")

    # ═══════════════════════════════════════════════════════
    # RUN 1: Unambiguous names (control)
    # ═══════════════════════════════════════════════════════
    print(f"\n[RUN] Unambiguous names through template: \"{TEMPLATE}\"")
    unamb_names = male_only + female_only
    unamb_labels = np.array([1] * len(male_only) + [0] * len(female_only))

    g_unamb, he_log_u, she_log_u = run_names(
        model, tok, unamb_names, rec_dirs, he_id, she_id,
        args.batch_size, device)

    print(f"  N={len(unamb_names)} (male={len(male_only)}, female={len(female_only)})")

    # AUCs for unambiguous
    unamb_aucs = {}
    for k in range(K):
        unamb_aucs[f"R{k+1}"] = try_roc_auc(unamb_labels, pol[k] * g_unamb[:, k])
    unamb_aucs["model"] = try_roc_auc(unamb_labels, he_log_u - she_log_u)
    comb = pol[0] * g_unamb[:, 0] + pol[2] * g_unamb[:, 2]
    unamb_aucs["R1R3"] = try_roc_auc(unamb_labels, comb)

    print(f"  Model AUC: {unamb_aucs['model']:.3f}")
    print(f"  R1 AUC: {unamb_aucs['R1']:.3f}   R3 AUC: {unamb_aucs['R3']:.3f}"
          f"   R1+R3 AUC: {unamb_aucs['R1R3']:.3f}")

    # ═══════════════════════════════════════════════════════
    # RUN 2: Ambiguous names
    # ═══════════════════════════════════════════════════════
    print(f"\n[RUN] Ambiguous names (N={len(ambiguous)}) ...")

    g_amb, he_log_a, she_log_a = run_names(
        model, tok, ambiguous, rec_dirs, he_id, she_id,
        args.batch_size, device)

    # ── Receptor "vote" for each ambiguous name ──
    # pol_g > 0 → receptor votes "male"
    r1_vote = pol[0] * g_amb[:, 0]     # positive = male
    r3_vote = pol[2] * g_amb[:, 2]     # positive = male
    model_vote = he_log_a - she_log_a  # positive = male
    combined_vote = r1_vote + r3_vote

    # ── Compare receptor vote to dataset lean ──
    # "Dataset lean" = which gender this name is used as MORE often
    # he_frac > 0.5 → leans male
    he_fracs = np.array([ambig_info[n]["he_frac"] for n in ambiguous])

    # Binary: does the receptor agree with the dataset's majority label?
    dataset_majority = (he_fracs > 0.5).astype(int)  # 1=male lean, 0=female lean
    # Exclude perfectly neutral names (he_frac == 0.5)
    non_neutral = he_fracs != 0.5
    if non_neutral.sum() > 10:
        r1_vs_lean = try_roc_auc(dataset_majority[non_neutral], r1_vote[non_neutral])
        r3_vs_lean = try_roc_auc(dataset_majority[non_neutral], r3_vote[non_neutral])
        model_vs_lean = try_roc_auc(dataset_majority[non_neutral], model_vote[non_neutral])
    else:
        r1_vs_lean = r3_vs_lean = model_vs_lean = float("nan")

    # ── AUC with random label assignment (each ambiguous name labeled by majority) ──
    amb_aucs = {}
    for k in range(K):
        amb_aucs[f"R{k+1}"] = try_roc_auc(dataset_majority, pol[k] * g_amb[:, k])
    amb_aucs["model"] = try_roc_auc(dataset_majority, model_vote)
    amb_aucs["R1R3"] = try_roc_auc(dataset_majority, combined_vote)

    print(f"\n  AUC (predicting dataset majority label):")
    print(f"    Model: {amb_aucs['model']:.3f}")
    print(f"    R1: {amb_aucs['R1']:.3f}   R3: {amb_aucs['R3']:.3f}"
          f"   R1+R3: {amb_aucs['R1R3']:.3f}")

    print(f"\n  AUC (receptor vote vs dataset lean, excluding neutrals):")
    print(f"    R1: {r1_vs_lean:.3f}   R3: {r3_vs_lean:.3f}"
          f"   Model: {model_vs_lean:.3f}")

    # ═══════════════════════════════════════════════════════
    # PER-NAME TABLE
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("PER-NAME ANALYSIS (ambiguous names)")
    print("=" * 70)

    # Sort by R1 vote (most "male" first)
    order = np.argsort(-r1_vote)

    print(f"\n  {'Name':15s}  {'he_frac':>7s}  {'lean':>7s}  {'R1_vote':>8s}"
          f"  {'R3_vote':>8s}  {'Model':>8s}  {'R_gender':>9s}  {'Match':>5s}")
    print(f"  {'-' * 85}")

    n_match = 0
    n_total_nonneut = 0
    for i in order:
        name = ambiguous[i]
        info = ambig_info[name]
        r_gender = "male" if combined_vote[i] > 0 else "female"
        m_gender = "male" if model_vote[i] > 0 else "female"
        match = "✓" if r_gender == info["lean"] else ("✗" if info["lean"] != "neutral" else "~")

        if info["lean"] != "neutral":
            n_total_nonneut += 1
            if r_gender == info["lean"]:
                n_match += 1

        print(f"  {name:15s}  {info['he_frac']:7.2f}  {info['lean']:>7s}"
              f"  {r1_vote[i]:+8.2f}  {r3_vote[i]:+8.2f}"
              f"  {model_vote[i]:+8.2f}  {r_gender:>9s}  {match:>5s}")

    match_rate = n_match / max(1, n_total_nonneut)
    print(f"\n  Receptor-lean agreement: {n_match}/{n_total_nonneut} = {match_rate:.1%}")
    print(f"  (How often R1+R3 combined vote matches dataset majority gender)")

    # ═══════════════════════════════════════════════════════
    # SUMMARY COMPARISON
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("AUC COMPARISON: UNAMBIGUOUS vs AMBIGUOUS")
    print("=" * 70)

    print(f"\n  {'':15s}  {'Model':>7s}  {'R1':>7s}  {'R3':>7s}"
          f"  {'R2':>7s}  {'R1+R3':>7s}  {'N':>5s}")
    print(f"  {'Unambiguous':15s}  {unamb_aucs['model']:7.3f}"
          f"  {unamb_aucs['R1']:7.3f}  {unamb_aucs['R3']:7.3f}"
          f"  {unamb_aucs['R2']:7.3f}  {unamb_aucs['R1R3']:7.3f}"
          f"  {len(unamb_names):5d}")
    print(f"  {'Ambiguous':15s}  {amb_aucs['model']:7.3f}"
          f"  {amb_aucs['R1']:7.3f}  {amb_aucs['R3']:7.3f}"
          f"  {amb_aucs['R2']:7.3f}  {amb_aucs['R1R3']:7.3f}"
          f"  {len(ambiguous):5d}")
    print(f"  {'Drop':15s}"
          f"  {amb_aucs['model'] - unamb_aucs['model']:+7.3f}"
          f"  {amb_aucs['R1'] - unamb_aucs['R1']:+7.3f}"
          f"  {amb_aucs['R3'] - unamb_aucs['R3']:+7.3f}"
          f"  {amb_aucs['R2'] - unamb_aucs['R2']:+7.3f}"
          f"  {amb_aucs['R1R3'] - unamb_aucs['R1R3']:+7.3f}")

    # ═══════════════════════════════════════════════════════
    # CROSS-REF WITH ERROR AUTOPSY NAMES
    # ═══════════════════════════════════════════════════════
    error_names = ["Christian", "Angel", "Hunter", "Tyler", "Haven",
                   "Morgan", "Kennedy", "Wren", "Juniper", "Kayden",
                   "Zion", "Luca", "Sage", "Genesis", "Finley", "Reilly"]

    print(f"\n{'=' * 70}")
    print("CROSS-REFERENCE: ERROR-PRONE NAMES FROM AUTOPSY")
    print("=" * 70)
    print(f"\n  {'Name':15s}  {'he_frac':>7s}  {'lean':>7s}  {'R1_vote':>8s}"
          f"  {'R3_vote':>8s}  {'R_gender':>9s}  {'Match':>5s}")
    for ename in error_names:
        if ename in ambiguous:
            i = ambiguous.index(ename)
            info = ambig_info[ename]
            r_gender = "male" if combined_vote[i] > 0 else "female"
            match = "✓" if r_gender == info["lean"] else ("✗" if info["lean"] != "neutral" else "~")
            print(f"  {ename:15s}  {info['he_frac']:7.2f}  {info['lean']:>7s}"
                  f"  {r1_vote[i]:+8.2f}  {r3_vote[i]:+8.2f}"
                  f"  {r_gender:>9s}  {match:>5s}")
        else:
            print(f"  {ename:15s}  (not ambiguous — single-gender in dataset)")

    # ═══════════════════════════════════════════════════════
    # PLOT 1: Scatter — R1 vote vs he_frac
    # ═══════════════════════════════════════════════════════
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, (vote, label) in zip(axes, [
        (r1_vote, "R1 (promotion) vote"),
        (r3_vote, "R3 (inhibition) vote"),
        (model_vote, "Model (logit_he − logit_she)"),
    ]):
        colors = ["#3498db" if h > 0.5 else "#e74c3c" if h < 0.5 else "#95a5a6"
                  for h in he_fracs]
        ax.scatter(he_fracs, vote, c=colors, s=25, alpha=0.7, edgecolors="white",
                   linewidth=0.5)
        ax.axhline(0, color="black", ls="--", alpha=0.4, lw=1)
        ax.axvline(0.5, color="black", ls="--", alpha=0.4, lw=1)
        ax.set_xlabel("Dataset he_fraction (>0.5 = leans male)", fontsize=10)
        ax.set_ylabel(label, fontsize=10)
        ax.grid(True, alpha=0.15)

        # Annotate error-prone names
        for ename in ["Christian", "Angel", "Tyler", "Haven", "Kennedy"]:
            if ename in ambiguous:
                i = ambiguous.index(ename)
                ax.annotate(ename, (he_fracs[i], vote[i]),
                            fontsize=7, alpha=0.7,
                            xytext=(5, 5), textcoords="offset points")

    fig.suptitle("Exp 5b: Receptor Vote vs Dataset Gender Lean (Ambiguous Names)",
                 fontsize=13, y=1.02)
    plt.tight_layout()
    p1 = out_dir / "exp5b_ambig_scatter.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[PLOT] {p1}")

    # ═══════════════════════════════════════════════════════
    # PLOT 2: AUC Comparison Bar Chart
    # ═══════════════════════════════════════════════════════
    fig, ax = plt.subplots(figsize=(8, 5))

    metrics = ["Model", "R1", "R3", "R2", "R1+R3"]
    unamb_vals = [unamb_aucs["model"], unamb_aucs["R1"], unamb_aucs["R3"],
                  unamb_aucs["R2"], unamb_aucs["R1R3"]]
    amb_vals = [amb_aucs["model"], amb_aucs["R1"], amb_aucs["R3"],
                amb_aucs["R2"], amb_aucs["R1R3"]]

    x = np.arange(len(metrics))
    w = 0.35
    ax.bar(x - w / 2, unamb_vals, w, label=f"Unambiguous (N={len(unamb_names)})",
           color="#3498db", edgecolor="white")
    ax.bar(x + w / 2, amb_vals, w, label=f"Ambiguous (N={len(ambiguous)})",
           color="#e67e22", edgecolor="white")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=10)
    ax.set_ylabel("ROC AUC", fontsize=11)
    ax.set_title("AUC: Unambiguous vs Ambiguous Names", fontsize=13)
    ax.set_ylim(0.4, 1.02)
    ax.legend(fontsize=10)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    p2 = out_dir / "exp5b_auc_comparison.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p2}")

    # ═══════════════════════════════════════════════════════
    # JSON
    # ═══════════════════════════════════════════════════════
    jp = out_dir / "exp5b_results.json"
    with open(jp, "w") as f:
        json.dump({
            "unambiguous_aucs": unamb_aucs,
            "ambiguous_aucs": amb_aucs,
            "receptor_lean_agreement": round(match_rate, 3),
            "n_ambiguous": len(ambiguous),
            "n_unambiguous": len(unamb_names),
        }, f, indent=2)
    print(f"\n[SAVE] {jp}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
