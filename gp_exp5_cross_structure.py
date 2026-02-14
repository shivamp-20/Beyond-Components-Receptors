#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Exp 5: Cross-Structure Generalization of Receptor Directions

Tests whether receptor directions (found via mask training on tag questions)
generalize to 5 different syntactic structures.

The SAME receptor directions are used — no retraining.
New sentences are generated programmatically from name lists extracted
from the existing GP CSVs.

Measures per structure:
  - Per-receptor AUC  (can pol_k * g_k separate male/female?)
  - Combined R1+R3 AUC
  - Model AUC  (can logit_he − logit_she separate?)
  - Sanity: top predictions, he/she rates, sign agreement

Run on Kaggle:
  !python gp_exp5_cross_structure.py \\
      --svd_cache outputs/gp/ov_svd_cache.pt \\
      --csv_dir data_main \\
      --out_dir outputs/gp \\
      --receptors "10,9,0,+1;11,8,6,+1;9,7,1,-1" \\
      --batch_size 64 \\
      --device cuda

View results:
  from IPython.display import display, Image
  display(Image("outputs/gp/exp5_auc_bars.png"))
  display(Image("outputs/gp/exp5_mini_phase.png"))
  display(Image("outputs/gp/exp5_gen_drop.png"))
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
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
# STRUCTURES — 5 syntactic patterns, 3 templates each
# ═══════════════════════════════════════════════════════════════
# Design: every template ends at a position where he/she is a
# plausible next token, so Model AUC is meaningful.

STRUCTURES = {
    "A_tag": {
        "label": "A: Tag question (baseline)",
        "short": "A: Tag Q",
        "templates": [
            "So {name} is a great listener, isn't",
            "So {name} always keeps their promises, doesn't",
            "So {name} has a knack for fixing things, doesn't",
        ],
    },
    "B_because": {
        "label": "B: Because clause",
        "short": "B: Because",
        "templates": [
            "{name} was really tired because",
            "{name} went to the store because",
            "{name} called the doctor because",
        ],
    },
    "C_temporal": {
        "label": "C: Temporal clause",
        "short": "C: Temporal",
        "templates": [
            "After {name} left,",
            "When {name} arrived,",
            "Before {name} spoke,",
        ],
    },
    "D_prep": {
        "label": "D: Prepositional",
        "short": "D: Prep-of",
        "templates": [
            "Friends of {name} said that",
            "The boss of {name} mentioned that",
            "Neighbors of {name} think that",
        ],
    },
    "E_cross": {
        "label": "E: Cross-sentence",
        "short": "E: Cross-sent",
        "templates": [
            "{name} finished the project. Then",
            "{name} left the building. Later",
            "{name} read the letter. Then",
        ],
    },
}


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def try_roc_auc(y_true: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC with sklearn or rank-based fallback."""
    y = np.asarray(y_true, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos = int(y.sum())
    n = len(y)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    try:
        from sklearn.metrics import roc_auc_score
        return float(roc_auc_score(y, s))
    except Exception:
        order = np.argsort(s)
        y_sorted = y[order]
        s_sorted = s[order]
        ranks = np.empty(n, dtype=np.float64)
        i = 0
        rank = 1.0
        while i < n:
            j = i + 1
            while j < n and s_sorted[j] == s_sorted[i]:
                j += 1
            avg_rank = (rank + rank + j - i - 1.0) / 2.0
            ranks[i:j] = avg_rank
            rank += j - i
            i = j
        r_pos = ranks[y_sorted == 1].sum()
        u = r_pos - n_pos * (n_pos + 1) / 2.0
        return float(u / (n_pos * n_neg))


# ═══════════════════════════════════════════════════════════════
# NAME EXTRACTION
# ═══════════════════════════════════════════════════════════════

def extract_names(csv_dir: str) -> Tuple[List[str], List[str], List[str]]:
    """Extract male-only, female-only, ambiguous names from GP CSVs."""
    male_set: set = set()
    female_set: set = set()

    csv_dir_p = Path(csv_dir)
    for fname in ["train_1k_gp.csv", "test_gp.csv", "val_gp.csv"]:
        fpath = csv_dir_p / fname
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
                elif pron == "she" and name:
                    female_set.add(name)
                if cpron == "he" and cname:
                    male_set.add(cname)
                elif cpron == "she" and cname:
                    female_set.add(cname)

    ambiguous = male_set & female_set
    male_only = sorted(male_set - ambiguous)
    female_only = sorted(female_set - ambiguous)
    return male_only, female_only, sorted(ambiguous)


# ═══════════════════════════════════════════════════════════════
# SENTENCE GENERATION
# ═══════════════════════════════════════════════════════════════

def generate_examples(
    male_names: List[str],
    female_names: List[str],
) -> Dict[str, List[Dict[str, Any]]]:
    """Generate examples for every structure.  Returns {struct_key: [examples]}."""
    all_examples: Dict[str, List[Dict[str, Any]]] = {}

    for sk, info in STRUCTURES.items():
        exs: List[Dict[str, Any]] = []
        for gender_label, y_val, name_pool in [
            ("he", 1, male_names),
            ("she", 0, female_names),
        ]:
            for name in name_pool:
                for ti, tmpl in enumerate(info["templates"]):
                    exs.append({
                        "text": tmpl.format(name=name),
                        "name": name,
                        "gender": gender_label,
                        "y": y_val,
                        "template_idx": ti,
                    })
        all_examples[sk] = exs
    return all_examples


# ═══════════════════════════════════════════════════════════════
# RECEPTOR LOADING
# ═══════════════════════════════════════════════════════════════

def load_receptors(
    svd_path: str, receptors_spec: str,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """Load receptor directions from SVD cache.
    Returns (rec_dirs (K, d_model), polarity (K,), meta list).
    """
    path = Path(svd_path)
    if not path.exists():
        # Try common alternative names in same directory
        alt = path.parent / "svd_cache.pt"
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"SVD cache not found: {svd_path}")

    ckpt = torch.load(str(path), map_location="cpu")
    if "ov" in ckpt:
        ov = ckpt["ov"]
    elif "svd" in ckpt and "ov" in ckpt["svd"]:
        ov = ckpt["svd"]["ov"]
    else:
        ov = ckpt

    specs = [s.strip() for s in receptors_spec.split(";") if s.strip()]
    dirs, pols, meta = [], [], []

    for idx, spec in enumerate(specs, start=1):
        parts = [x.strip() for x in spec.split(",")]
        layer, head, sv_idx, pol = (
            int(parts[0]), int(parts[1]), int(parts[2]), int(parts[3])
        )
        cell = ov[layer][head]
        Vh = cell["Vh"] if isinstance(cell, dict) else cell.Vh
        v = Vh[sv_idx].detach().to(torch.float32).cpu()
        v = v / (v.norm() + 1e-12)

        dirs.append(v)
        pols.append(pol)
        meta.append(dict(name=f"R{idx}", layer=layer, head=head,
                         sv_idx=sv_idx, polarity=pol))

    return (torch.stack(dirs, dim=0),
            torch.tensor(pols, dtype=torch.float32),
            meta)


# ═══════════════════════════════════════════════════════════════
# FORWARD PASS — one structure at a time
# ═══════════════════════════════════════════════════════════════

@torch.no_grad()
def run_structure(
    model, tok, examples: List[Dict], rec_dirs: torch.Tensor,
    he_id: int, she_id: int, batch_size: int, device: torch.device,
) -> Dict[str, Any]:
    """Forward pass for one structure.  Returns numpy arrays."""
    texts = [ex["text"] for ex in examples]
    labels = np.array([ex["y"] for ex in examples], dtype=np.int64)
    N = len(examples)
    K = rec_dirs.shape[0]

    enc = tok(texts, add_special_tokens=False, padding=True,
              return_tensors="pt")
    tokens = enc["input_ids"]
    attn_mask = enc.get("attention_mask", torch.ones_like(tokens))
    last_idx = (attn_mask.sum(dim=1) - 1).to(torch.long)

    g_all = np.zeros((N, K), dtype=np.float64)
    he_logits = np.zeros(N, dtype=np.float64)
    she_logits = np.zeros(N, dtype=np.float64)
    top5_ids = np.zeros((N, 5), dtype=np.int64)

    rec_dirs_dev = rec_dirs.to(device)
    hook_name = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"

    for start in range(0, N, batch_size):
        end = min(N, start + batch_size)
        btok = tokens[start:end].to(device)
        blidx = last_idx[start:end].to(device)

        logits, cache = model.run_with_cache(
            btok, names_filter=[hook_name])

        B = end - start
        b_idx = torch.arange(B, device=device)

        resid = cache[hook_name][b_idx, blidx, :]          # (B, d_model)
        g_all[start:end] = (resid @ rec_dirs_dev.T).cpu().numpy()

        dec_logits = logits[b_idx, blidx, :]                # (B, vocab)
        he_logits[start:end] = dec_logits[:, he_id].cpu().numpy()
        she_logits[start:end] = dec_logits[:, she_id].cpu().numpy()
        top5_ids[start:end] = dec_logits.topk(5, dim=-1).indices.cpu().numpy()

    return dict(
        g=g_all, labels=labels,
        he_logits=he_logits, she_logits=she_logits,
        top5_ids=top5_ids,
        last_idx=last_idx.numpy(),
        seq_len=tokens.shape[1],
    )


# ═══════════════════════════════════════════════════════════════
# METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════

def compute_metrics(
    results: Dict, polarity: torch.Tensor,
    he_id: int, she_id: int, tok,
) -> Dict[str, Any]:
    """Compute AUCs, sign agreement, sanity checks."""
    g = results["g"]                  # (N, K)
    labels = results["labels"]        # (N,) 1=male, 0=female
    pol = polarity.numpy()
    N, K = g.shape

    m: Dict[str, Any] = {}

    # ── Per-receptor AUC ──
    for k in range(K):
        pol_g = pol[k] * g[:, k]
        m[f"R{k+1}_AUC"] = try_roc_auc(labels, pol_g)

    # ── Combined AUCs ──
    comb_13 = pol[0] * g[:, 0] + pol[2] * g[:, 2]
    comb_all = sum(pol[k] * g[:, k] for k in range(K))
    m["R1R3_AUC"] = try_roc_auc(labels, comb_13)
    m["R123_AUC"] = try_roc_auc(labels, comb_all)

    # ── Model AUC ──
    logit_diff = results["he_logits"] - results["she_logits"]
    m["model_AUC"] = try_roc_auc(labels, logit_diff)

    # ── Sign agreement ──
    for k in range(K):
        pol_g = pol[k] * g[:, k]
        correct = ((pol_g > 0) & (labels == 1)) | ((pol_g < 0) & (labels == 0))
        m[f"R{k+1}_sign_agree"] = float(correct.mean())

    # ── He/she prediction rates ──
    top1 = results["top5_ids"][:, 0]
    pred_he = top1 == he_id
    pred_she = top1 == she_id
    pred_gender = pred_he | pred_she
    m["heshe_pred_rate"] = float(pred_gender.mean())

    if pred_gender.sum() > 0:
        correct_gen = (pred_he & (labels == 1)) | (pred_she & (labels == 0))
        m["model_acc_heshe"] = float(correct_gen[pred_gender].mean())
    else:
        m["model_acc_heshe"] = float("nan")

    # ── He/she in top-5 ──
    he_top5 = (results["top5_ids"] == he_id).any(axis=1)
    she_top5 = (results["top5_ids"] == she_id).any(axis=1)
    m["heshe_top5_rate"] = float((he_top5 | she_top5).mean())

    # ── Top predicted tokens ──
    counter = Counter(top1.tolist())
    m["top_predictions"] = [
        (tok.decode([tid]), cnt, round(cnt / N, 3))
        for tid, cnt in counter.most_common(8)
    ]

    # ── Mean activation by gender ──
    male = labels == 1
    female = labels == 0
    for k in range(K):
        m[f"R{k+1}_mean_male"] = float(g[male, k].mean()) if male.any() else float("nan")
        m[f"R{k+1}_mean_female"] = float(g[female, k].mean()) if female.any() else float("nan")

    m["N"] = N
    m["n_male"] = int(male.sum())
    m["n_female"] = int(female.sum())

    return m


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--svd_cache", required=True,
                    help="Path to ov_svd_cache.pt (or svd_cache.pt)")
    ap.add_argument("--csv_dir", default="data_main",
                    help="Directory containing train/test/val GP CSVs")
    ap.add_argument("--out_dir", default="outputs/gp")
    ap.add_argument("--receptors", default="10,9,0,+1;11,8,6,+1;9,7,1,-1")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # ── Load model ──
    print("=" * 70)
    print("EXPERIMENT 5: CROSS-STRUCTURE GENERALIZATION")
    print("=" * 70)

    print("\n[MODEL] Loading gpt2-small ...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=str(device))
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    he_id = tok.encode(" he", add_special_tokens=False)[0]
    she_id = tok.encode(" she", add_special_tokens=False)[0]
    print(f"  he_id={he_id}  she_id={she_id}")

    # ── Extract names ──
    print("\n[NAMES] Extracting from CSVs ...")
    male_only, female_only, ambiguous = extract_names(args.csv_dir)
    print(f"  male-only: {len(male_only)}   female-only: {len(female_only)}"
          f"   ambiguous: {len(ambiguous)}")
    print(f"  Sample male:   {male_only[:5]}")
    print(f"  Sample female: {female_only[:5]}")
    print(f"  Sample ambig:  {ambiguous[:5]}")

    # ── Load receptors ──
    print("\n[RECEPTORS] Loading from SVD cache ...")
    rec_dirs, polarity, rec_meta = load_receptors(args.svd_cache, args.receptors)
    K = rec_dirs.shape[0]
    for rm in rec_meta:
        print(f"  {rm['name']}: L{rm['layer']}H{rm['head']} sv{rm['sv_idx']}"
              f"  pol={rm['polarity']:+d}")
    print(f"  cos(R1,R3) = {float(torch.dot(rec_dirs[0], rec_dirs[2])):.4f}")

    # ── Generate sentences ──
    print("\n[GEN] Generating sentences ...")
    all_examples = generate_examples(male_only, female_only)
    for sk in STRUCTURES:
        print(f"  {STRUCTURES[sk]['short']}: {len(all_examples[sk])} examples")

    # ════════════════════════════════════════════════════════
    # SANITY: Token inspection — 2 samples per structure
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("TOKENIZATION SANITY CHECK")
    print("=" * 70)

    for sk, info in STRUCTURES.items():
        exs = all_examples[sk]
        print(f"\n  [{info['short']}]")
        for sample_i in [0, len(exs) // 2]:  # one male, one female
            text = exs[sample_i]["text"]
            tids = tok.encode(text, add_special_tokens=False)
            decoded = [tok.decode([t]) for t in tids]
            print(f"    \"{text}\"")
            print(f"    tokens({len(tids)}): {decoded}")
            print(f"    decision pos: {len(tids)-1}"
                  f"  token: '{decoded[-1]}'")
            print(f"    name: '{exs[sample_i]['name']}'"
                  f"  gender: {exs[sample_i]['gender']}")

    # ════════════════════════════════════════════════════════
    # RUN FORWARD PASSES — one structure at a time
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("FORWARD PASSES")
    print("=" * 70)

    all_results: Dict[str, Dict] = {}
    all_metrics: Dict[str, Dict] = {}

    for sk, info in STRUCTURES.items():
        exs = all_examples[sk]
        print(f"\n[RUN] {info['label']}  (N={len(exs)}) ...")

        results = run_structure(
            model, tok, exs, rec_dirs,
            he_id, she_id, args.batch_size, device,
        )
        all_results[sk] = results

        metrics = compute_metrics(results, polarity, he_id, she_id, tok)
        all_metrics[sk] = metrics

        print(f"  seq_len={results['seq_len']}  N={metrics['N']}"
              f"  (male={metrics['n_male']}, female={metrics['n_female']})")
        print(f"  Model AUC: {metrics['model_AUC']:.3f}"
              f"   he/she pred rate: {metrics['heshe_pred_rate']:.1%}"
              f"   he/she top-5 rate: {metrics['heshe_top5_rate']:.1%}")
        print(f"  R1 AUC: {metrics['R1_AUC']:.3f}"
              f"   R2 AUC: {metrics['R2_AUC']:.3f}"
              f"   R3 AUC: {metrics['R3_AUC']:.3f}"
              f"   R1+R3 AUC: {metrics['R1R3_AUC']:.3f}")
        print(f"  Top predictions: "
              + ", ".join(f"'{t}' ({p:.0%})" for t, c, p
                         in metrics["top_predictions"][:5]))
        print(f"  done.")

    # ════════════════════════════════════════════════════════
    # SUMMARY TABLE
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("SUMMARY: AUC TABLE")
    print("=" * 70)

    header = (f"  {'Structure':24s}  {'Model':>7s}  {'R1':>7s}  {'R2':>7s}"
              f"  {'R3':>7s}  {'R1+R3':>7s}  {'he/she%':>7s}  {'N':>5s}")
    print(header)
    print("  " + "-" * (len(header) - 2))

    baseline_key = "A_tag"
    for sk in STRUCTURES:
        m = all_metrics[sk]
        s = STRUCTURES[sk]["short"]
        print(f"  {s:24s}  {m['model_AUC']:7.3f}  {m['R1_AUC']:7.3f}"
              f"  {m['R2_AUC']:7.3f}  {m['R3_AUC']:7.3f}"
              f"  {m['R1R3_AUC']:7.3f}  {m['heshe_pred_rate']:6.1%}"
              f"  {m['N']:5d}")

    # ════════════════════════════════════════════════════════
    # GENERALIZATION GAP ANALYSIS
    # ════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("GENERALIZATION GAP  (relative to baseline A)")
    print("=" * 70)

    base_m = all_metrics[baseline_key]
    print(f"\n  Baseline (A): R1={base_m['R1_AUC']:.3f}"
          f"  R3={base_m['R3_AUC']:.3f}  R2={base_m['R2_AUC']:.3f}"
          f"  Model={base_m['model_AUC']:.3f}")

    r1_drops = []
    r3_drops = []

    print(f"\n  {'Structure':24s}  {'ΔR1':>7s}  {'ΔR3':>7s}  {'ΔR2':>7s}"
          f"  {'ΔModel':>7s}")
    for sk in STRUCTURES:
        if sk == baseline_key:
            continue
        m = all_metrics[sk]
        s = STRUCTURES[sk]["short"]
        dr1 = m["R1_AUC"] - base_m["R1_AUC"]
        dr3 = m["R3_AUC"] - base_m["R3_AUC"]
        dr2 = m["R2_AUC"] - base_m["R2_AUC"]
        dm = m["model_AUC"] - base_m["model_AUC"]
        print(f"  {s:24s}  {dr1:+7.3f}  {dr3:+7.3f}  {dr2:+7.3f}  {dm:+7.3f}")
        r1_drops.append(dr1)
        r3_drops.append(dr3)

    mean_r1_drop = np.mean(r1_drops)
    mean_r3_drop = np.mean(r3_drops)
    min_r1 = min(all_metrics[sk]["R1_AUC"] for sk in STRUCTURES)
    min_r3 = min(all_metrics[sk]["R3_AUC"] for sk in STRUCTURES)

    print(f"\n  Mean AUC drop:  R1={mean_r1_drop:+.3f}  R3={mean_r3_drop:+.3f}")
    print(f"  Worst-case AUC: R1={min_r1:.3f}  R3={min_r3:.3f}")
    print(f"\n  PREDICTION (R1 generalizes better than R3, "
          f"i.e. |ΔR1| < |ΔR3|):")
    r1_abs_mean = np.mean(np.abs(r1_drops))
    r3_abs_mean = np.mean(np.abs(r3_drops))
    confirmed = r1_abs_mean < r3_abs_mean
    print(f"    mean|ΔR1|={r1_abs_mean:.3f}  mean|ΔR3|={r3_abs_mean:.3f}"
          f"  → {'CONFIRMED' if confirmed else 'NOT CONFIRMED'}")

    # ════════════════════════════════════════════════════════
    # SIGN AGREEMENT TABLE
    # ════════════════════════════════════════════════════════
    print(f"\n  SIGN AGREEMENT:")
    print(f"  {'Structure':24s}  {'R1':>7s}  {'R3':>7s}")
    for sk in STRUCTURES:
        m = all_metrics[sk]
        s = STRUCTURES[sk]["short"]
        print(f"  {s:24s}  {m['R1_sign_agree']:6.1%}"
              f"  {m['R3_sign_agree']:6.1%}")

    # ════════════════════════════════════════════════════════
    # SANITY: MODEL BEHAVIOR PER STRUCTURE
    # ════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("SANITY: MODEL BEHAVIOR AT DECISION POSITION")
    print("=" * 70)

    for sk in STRUCTURES:
        m = all_metrics[sk]
        s = STRUCTURES[sk]["short"]
        print(f"\n  [{s}]")
        print(f"    he/she as top-1: {m['heshe_pred_rate']:.1%}"
              f"    he/she in top-5: {m['heshe_top5_rate']:.1%}")
        if m["heshe_pred_rate"] > 0.01:
            print(f"    Accuracy (among he/she preds): {m['model_acc_heshe']:.1%}")
        else:
            print(f"    (Model rarely predicts he/she → Model AUC is logit-based, "
                  f"not prediction-based)")
        print(f"    Top tokens: "
              + ", ".join(f"'{t}'({p:.0%})" for t, c, p
                         in m["top_predictions"][:5]))

    # ════════════════════════════════════════════════════════
    # MEAN ACTIVATION SEPARATION
    # ════════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("MEAN ACTIVATION BY GENDER  (raw g, not polarity-adjusted)")
    print("=" * 70)
    print(f"\n  {'Structure':24s}  {'R1_male':>8s}  {'R1_fem':>8s}"
          f"  {'R3_male':>8s}  {'R3_fem':>8s}")
    for sk in STRUCTURES:
        m = all_metrics[sk]
        s = STRUCTURES[sk]["short"]
        print(f"  {s:24s}  {m['R1_mean_male']:+8.2f}  {m['R1_mean_female']:+8.2f}"
              f"  {m['R3_mean_male']:+8.2f}  {m['R3_mean_female']:+8.2f}")

    # ════════════════════════════════════════════════════════
    # PLOT 1: AUC Grouped Bar Chart
    # ════════════════════════════════════════════════════════
    struct_keys = list(STRUCTURES.keys())
    struct_labels = [STRUCTURES[sk]["short"] for sk in struct_keys]
    x_pos = np.arange(len(struct_keys))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, (metric_key, label, color) in enumerate([
        ("model_AUC", "Model", "#2c3e50"),
        ("R1_AUC", "R1 (promotion)", "#3498db"),
        ("R3_AUC", "R3 (inhibition)", "#e67e22"),
        ("R2_AUC", "R2 (ghost)", "#95a5a6"),
    ]):
        vals = [all_metrics[sk][metric_key] for sk in struct_keys]
        bars = ax.bar(x_pos + i * width, vals, width, label=label,
                      color=color, edgecolor="white", lw=0.8)

    ax.set_xticks(x_pos + 1.5 * width)
    ax.set_xticklabels(struct_labels, fontsize=9)
    ax.set_ylabel("ROC AUC", fontsize=11)
    ax.set_title("Exp 5: Receptor AUC Across Syntactic Structures", fontsize=13)
    ax.set_ylim(0.4, 1.02)
    ax.axhline(0.5, color="gray", ls=":", alpha=0.4, lw=1)
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(True, axis="y", alpha=0.15)
    plt.tight_layout()
    p1 = out_dir / "exp5_auc_bars.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"\n[PLOT] {p1}")

    # ════════════════════════════════════════════════════════
    # PLOT 2: Mini Phase Diagrams (5 panels)
    # ════════════════════════════════════════════════════════
    pol_np = polarity.numpy()

    fig, axes = plt.subplots(1, 5, figsize=(22, 4))
    for panel, sk in enumerate(struct_keys):
        ax = axes[panel]
        r = all_results[sk]
        g = r["g"]
        labels = r["labels"]

        x_vals = pol_np[0] * g[:, 0]    # R1 toward he
        y_vals = pol_np[2] * g[:, 2]    # R3 toward he

        male = labels == 1
        female = labels == 0

        ax.scatter(x_vals[female], y_vals[female],
                   s=3, alpha=0.15, c="#e74c3c", rasterized=True)
        ax.scatter(x_vals[male], y_vals[male],
                   s=3, alpha=0.15, c="#4a90d9", rasterized=True)

        # Shared axis limits across panels
        lim = max(np.abs(x_vals).max(), np.abs(y_vals).max(), 1) * 1.05
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.plot([-lim, lim], [lim, -lim], "k--", alpha=0.2, lw=0.7)
        ax.axhline(0, color="gray", alpha=0.15, lw=0.5)
        ax.axvline(0, color="gray", alpha=0.15, lw=0.5)
        ax.set_aspect("equal")
        ax.set_title(STRUCTURES[sk]["short"], fontsize=9)
        ax.grid(True, alpha=0.1)

        if panel == 0:
            ax.set_ylabel("R3 → he", fontsize=9)
            legend_elements = [
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#4a90d9", markersize=5, label="male"),
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#e74c3c", markersize=5, label="female"),
            ]
            ax.legend(handles=legend_elements, fontsize=7, loc="lower right")
        ax.set_xlabel("R1 → he", fontsize=9)

    fig.suptitle("Exp 5: Phase Diagrams Across Structures", fontsize=13, y=1.02)
    plt.tight_layout()
    p2 = out_dir / "exp5_mini_phase.png"
    fig.savefig(p2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {p2}")

    # ════════════════════════════════════════════════════════
    # PLOT 3: Generalization Drop
    # ════════════════════════════════════════════════════════
    non_baseline = [sk for sk in struct_keys if sk != baseline_key]
    nb_labels = [STRUCTURES[sk]["short"] for sk in non_baseline]
    x_nb = np.arange(len(non_baseline))
    w = 0.30

    fig, ax = plt.subplots(figsize=(8, 4.5))
    r1_vals = [all_metrics[sk]["R1_AUC"] - base_m["R1_AUC"]
               for sk in non_baseline]
    r3_vals = [all_metrics[sk]["R3_AUC"] - base_m["R3_AUC"]
               for sk in non_baseline]

    ax.bar(x_nb - w / 2, r1_vals, w, label="R1 (promotion)",
           color="#3498db", edgecolor="white")
    ax.bar(x_nb + w / 2, r3_vals, w, label="R3 (inhibition)",
           color="#e67e22", edgecolor="white")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x_nb)
    ax.set_xticklabels(nb_labels, fontsize=9)
    ax.set_ylabel("ΔAUC from baseline (A)", fontsize=11)
    ax.set_title("Generalization Gap: AUC Change Relative to Baseline",
                 fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.2)
    plt.tight_layout()
    p3 = out_dir / "exp5_gen_drop.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p3}")

    # ════════════════════════════════════════════════════════
    # JSON OUTPUT
    # ════════════════════════════════════════════════════════
    json_payload: Dict[str, Any] = {
        "experiment": "exp5_cross_structure_generalization",
        "names": {"male_only": len(male_only), "female_only": len(female_only),
                  "ambiguous": len(ambiguous)},
        "receptors": rec_meta,
        "structures": {},
    }
    for sk in STRUCTURES:
        m = all_metrics[sk]
        json_payload["structures"][sk] = {
            "label": STRUCTURES[sk]["label"],
            "N": m["N"],
            "model_AUC": round(m["model_AUC"], 4),
            "R1_AUC": round(m["R1_AUC"], 4),
            "R2_AUC": round(m["R2_AUC"], 4),
            "R3_AUC": round(m["R3_AUC"], 4),
            "R1R3_AUC": round(m["R1R3_AUC"], 4),
            "R1_sign_agree": round(m["R1_sign_agree"], 4),
            "R3_sign_agree": round(m["R3_sign_agree"], 4),
            "heshe_pred_rate": round(m["heshe_pred_rate"], 4),
            "heshe_top5_rate": round(m["heshe_top5_rate"], 4),
        }
    json_payload["generalization"] = {
        "mean_R1_drop": round(float(mean_r1_drop), 4),
        "mean_R3_drop": round(float(mean_r3_drop), 4),
        "worst_R1": round(float(min_r1), 4),
        "worst_R3": round(float(min_r3), 4),
        "R1_more_robust_than_R3": bool(confirmed),
    }

    jp = out_dir / "exp5_results.json"
    with open(jp, "w") as f:
        json.dump(json_payload, f, indent=2)
    print(f"\n[SAVE] {jp}")
    print("\n[DONE] Exp 5 complete.")


if __name__ == "__main__":
    main()
