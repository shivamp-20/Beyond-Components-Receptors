#!/usr/bin/env python3
"""
Exp 0c: Receptor Commitment & Per-Layer Diagnostics
====================================================
Caches residual stream at 13 layer snapshots (embedding + 12 post-block)
at the decision position for every example.

Computes per-layer:
  1. Sign-agreement curves + commitment layers (3 thresholds)
  2. AUC curves (polarity-adjusted, per receptor + combined)
  3. Mean activation by gender (polarity-adjusted)
  4. Delta_g per layer transition (where signal is written)
  5. Deliberation index (within-class variance, normalized)
  6. RCS — Receptor Concentration Score

Saves:
  - exp0c_tensors.pt  (resid_all, g, polarity, ys, logits, names)
  - exp0c_results.json
  - 6 plots

Run on Kaggle:
  !python gp_exp0c_commitment.py \
      --data_dir data_main \
      --csv train_1k_gp.csv \
      --out_dir outputs/gp \
      --receptors "10,9,0,+1;11,8,6,+1;9,7,1,-1" \
      --batch_size 64 \
      --device cuda

View results:
  from IPython.display import display, Image
  for name in ['commitment_plot', 'auc_plot', 'activation_by_gender',
               'delta_by_layer', 'deliberation', 'rcs']:
      display(Image(f"outputs/gp/exp0c_{name}.png"))
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer


# ═══════════════════════════════════════════════════════════════
# UTILITIES
# ═══════════════════════════════════════════════════════════════

def try_roc_auc(y01: np.ndarray, scores: np.ndarray) -> float:
    """ROC AUC with sklearn or rank-based fallback."""
    y = np.asarray(y01, dtype=np.int64)
    s = np.asarray(scores, dtype=np.float64)
    n_pos, n = int(y.sum()), len(y)
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
        i, rank = 0, 1.0
        while i < n:
            j = i + 1
            while j < n and s_sorted[j] == s_sorted[i]:
                j += 1
            ranks[i:j] = (rank + rank + j - i - 1.0) / 2.0
            rank += j - i
            i = j
        u = ranks[y_sorted == 1].sum() - n_pos * (n_pos + 1) / 2.0
        return float(u / (n_pos * n_neg))


def sign_eps(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """Sign with epsilon dead-zone: +1, -1, or 0."""
    return np.where(x > eps, 1, np.where(x < -eps, -1, 0)).astype(np.int8)


def entropy(p: np.ndarray) -> float:
    """Shannon entropy of a probability vector."""
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-12, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


def commit_layer(curve: List[float], thr: float, labels: List[int]) -> int:
    """Smallest layer label l such that curve[l:] stays >= thr."""
    last_below = -1
    for i, v in enumerate(curve):
        if v < thr:
            last_below = i
    idx = min(last_below + 1, len(curve) - 1)
    return int(labels[idx])


# ═══════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════

def load_examples(csv_path: Path, use_both: bool = True):
    """Load GP examples. Returns (prefixes, ys, names).
    ys: +1 for he, -1 for she.
    If use_both, includes corr_prefix/corr_pronoun rows too."""
    prefixes, ys_list, names = [], [], []

    with open(csv_path, encoding="utf-8", errors="ignore") as f:
        reader = csv.DictReader(f)
        for r in reader:
            pron = (r.get("pronoun") or "").strip().lower()
            prefix = (r.get("prefix") or "").strip()
            name = (r.get("name") or "").strip()
            if pron in ("he", "she") and prefix:
                prefixes.append(prefix)
                ys_list.append(+1 if pron == "he" else -1)
                names.append(name)

            if use_both:
                cpron = (r.get("corr_pronoun") or "").strip().lower()
                cprefix = (r.get("corr_prefix") or "").strip()
                cname = (r.get("corr_name") or "").strip()
                if cpron in ("he", "she") and cprefix:
                    prefixes.append(cprefix)
                    ys_list.append(+1 if cpron == "he" else -1)
                    names.append(cname)

    return prefixes, np.array(ys_list, dtype=np.float32), names


def load_receptors(out_dir: Path, spec: str):
    """Load receptor directions from SVD cache."""
    for fname in ["ov_svd_cache.pt", "svd_cache.pt"]:
        path = out_dir / fname
        if path.exists():
            break
    else:
        raise FileNotFoundError(f"No SVD cache in {out_dir}")

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
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out_dir", default="outputs/gp")
    ap.add_argument("--receptors", default="10,9,0,+1;11,8,6,+1;9,7,1,-1")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--use_both", type=int, default=1,
                    help="1=include corr rows, 0=clean only")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = Path(args.data_dir) / args.csv
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    THRESHOLDS = [0.80, 0.85, 0.90]
    PRIMARY_THR = 0.85
    AUC_THR = 0.90

    print("=" * 70)
    print("EXP 0c: RECEPTOR COMMITMENT & PER-LAYER DIAGNOSTICS")
    print("=" * 70)

    # ── Load model ──
    print("\n[MODEL] Loading gpt2-small ...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=str(device))
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    he_id = tok.encode(" he", add_special_tokens=False)[0]
    she_id = tok.encode(" she", add_special_tokens=False)[0]
    n_layers = model.cfg.n_layers
    d_model = model.cfg.d_model
    print(f"  he_id={he_id}  she_id={she_id}  n_layers={n_layers}  d_model={d_model}")

    # ── Load data ──
    print(f"\n[DATA] Loading from {csv_path} ...")
    prefixes, ys, names = load_examples(csv_path, use_both=bool(args.use_both))
    N = len(prefixes)
    n_male = int((ys == 1).sum())
    n_female = int((ys == -1).sum())
    print(f"  N={N}  (male={n_male}, female={n_female})  use_both={args.use_both}")

    # Tokenize
    enc = tok(prefixes, add_special_tokens=False, padding=True, return_tensors="pt")
    tokens = enc["input_ids"].to(device)
    attn_mask = enc.get("attention_mask", torch.ones_like(tokens)).to(device)
    last_idx = (attn_mask.sum(dim=1) - 1).to(torch.long)
    seq_len = tokens.shape[1]
    print(f"  seq_len={seq_len} (padded)")

    # ── Load receptors ──
    print(f"\n[RECEPTORS] Loading from SVD cache ...")
    rec_dirs, polarity, rec_meta = load_receptors(out_dir, args.receptors)
    K = rec_dirs.shape[0]
    pol = polarity.numpy()
    for rm in rec_meta:
        print(f"  {rm['name']}: L{rm['layer']}H{rm['head']} sv{rm['sv']} pol={rm['pol']:+d}")
    for i in range(K):
        for j in range(i + 1, K):
            c = float(torch.dot(rec_dirs[i], rec_dirs[j]))
            print(f"  cos(R{i+1},R{j+1}) = {c:+.4f}")

    # ── Hook names: 13 layer snapshots ──
    # Layer -1 = embedding (blocks.0.hook_resid_pre)
    # Layer 0..11 = post-block (blocks.l.hook_resid_post)
    hook_names = ["blocks.0.hook_resid_pre"] + \
                 [f"blocks.{l}.hook_resid_post" for l in range(n_layers)]
    layer_labels = [-1] + list(range(n_layers))  # 13 labels
    L = len(layer_labels)

    # ── Forward passes: cache residuals at decision position ──
    print(f"\n[RUN] Caching {L} layer snapshots, batch_size={args.batch_size} ...")

    resid_all = torch.empty((L, N, d_model), dtype=torch.float32, device="cpu")
    he_logits = np.zeros(N, dtype=np.float64)
    she_logits = np.zeros(N, dtype=np.float64)

    for start in range(0, N, args.batch_size):
        end = min(N, start + args.batch_size)
        B = end - start
        btok = tokens[start:end]
        blidx = last_idx[start:end]
        b_idx = torch.arange(B, device=device)

        logits, cache = model.run_with_cache(btok, names_filter=hook_names)

        # Logits at decision position
        dec_logits = logits[b_idx, blidx, :]
        he_logits[start:end] = dec_logits[:, he_id].cpu().numpy()
        she_logits[start:end] = dec_logits[:, she_id].cpu().numpy()

        # Residual snapshots at decision position
        for li, hname in enumerate(hook_names):
            resid_all[li, start:end, :] = cache[hname][b_idx, blidx, :].cpu()

        if end == N or (start // args.batch_size) % 10 == 0:
            print(f"  {end}/{N}")

    # ── Baseline accuracy ──
    diff = he_logits - she_logits
    pred = np.where(diff > 0, 1, -1).astype(np.float32)
    acc = float((pred == ys).mean())
    print(f"\n[BASELINE] Model accuracy (he vs she logit): {acc:.3f}")

    # ═══════════════════════════════════════════════════════
    # COMPUTE RECEPTOR ACTIVATIONS
    # ═══════════════════════════════════════════════════════

    # g[k,l,i] = receptor_k · resid[l,i]
    g = torch.einsum("kd,lnd->kln", rec_dirs.float(), resid_all.float())  # (K, L, N)
    g_np = g.numpy()

    # polarity-adjusted: pol_g[k,l,i] = polarity[k] * g[k,l,i]
    pol_g = polarity.view(K, 1, 1) * g  # (K, L, N)
    pol_g_np = pol_g.numpy()

    # combined score: s[l,i] = sum_k pol_g[k,l,i]
    s = pol_g.sum(dim=0)  # (L, N)

    # signed margin: ys * pol * g (positive = pushing correct direction)
    ys_t = torch.tensor(ys, dtype=torch.float32)
    signed_margin = (ys_t.view(1, 1, -1) * pol_g).numpy()  # (K, L, N)

    # Binary labels for AUC
    y01 = ((ys + 1) / 2).astype(np.int64)  # 1=male, 0=female
    male = ys == 1
    female = ys == -1

    # ═══════════════════════════════════════════════════════
    # METRIC 1: Sign Agreement + Commitment Layers
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("SIGN AGREEMENT & COMMITMENT")
    print("=" * 70)

    def agreement_curve(score_ln: np.ndarray) -> List[float]:
        """Agreement with FINAL sign, per layer."""
        final_sign = sign_eps(score_ln[-1])
        return [float((sign_eps(score_ln[li]) == final_sign).mean())
                for li in range(score_ln.shape[0])]

    agreement: Dict[str, List[float]] = {}
    for k in range(K):
        agreement[f"R{k+1}"] = agreement_curve(signed_margin[k])

    combined_signed = (ys_t.view(1, -1) * s).numpy()  # (L, N)
    agreement["combined"] = agreement_curve(combined_signed)

    # Commitment layers at multiple thresholds
    commit_sign: Dict[str, Dict[str, int]] = {}
    for thr in THRESHOLDS:
        key = f"{thr:.2f}"
        commit_sign[key] = {}
        for name, curve in agreement.items():
            commit_sign[key][name] = commit_layer(curve, thr, layer_labels)

    primary = commit_sign[f"{PRIMARY_THR:.2f}"]

    for thr in THRESHOLDS:
        key = f"{thr:.2f}"
        cl = commit_sign[key]
        print(f"  thr={thr:.2f}:  R1=layer {cl['R1']}  R2=layer {cl['R2']}"
              f"  R3=layer {cl['R3']}  combined=layer {cl['combined']}")

    # ═══════════════════════════════════════════════════════
    # METRIC 2: AUC Curves + AUC Commitment
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("AUC CURVES")
    print("=" * 70)

    auc: Dict[str, List[float]] = {}
    for k in range(K):
        auc[f"R{k+1}"] = [try_roc_auc(y01, pol_g_np[k, li])
                           for li in range(L)]
    auc["combined"] = [try_roc_auc(y01, s[li].numpy()) for li in range(L)]

    # AUC commitment
    commit_auc: Dict[str, int] = {}
    for name in [f"R{k+1}" for k in range(K)] + ["combined"]:
        commit_auc[name] = commit_layer(auc[name], AUC_THR, layer_labels)

    print(f"  AUC commitment (>= {AUC_THR:.2f}):")
    for name in ["R1", "R2", "R3", "combined"]:
        print(f"    {name}: layer {commit_auc[name]}")

    # Final layer AUCs
    fi = L - 1
    print(f"\n  Final-layer AUCs (layer {layer_labels[fi]}):")
    for name in ["R1", "R2", "R3", "combined"]:
        print(f"    {name}: {auc[name][fi]:.4f}")

    # ═══════════════════════════════════════════════════════
    # METRIC 3: Mean Activation by Gender
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("MEAN ACTIVATION BY GENDER (polarity-adjusted)")
    print("=" * 70)

    mean_male: Dict[str, List[float]] = {}
    mean_female: Dict[str, List[float]] = {}

    for k in range(K):
        name = f"R{k+1}"
        mean_male[name] = [float(pol_g_np[k, li, male].mean()) for li in range(L)]
        mean_female[name] = [float(pol_g_np[k, li, female].mean()) for li in range(L)]
    mean_male["combined"] = [float(s[li].numpy()[male].mean()) for li in range(L)]
    mean_female["combined"] = [float(s[li].numpy()[female].mean()) for li in range(L)]

    print(f"\n  {'Layer':>6s}", end="")
    for name in ["R1", "R2", "R3"]:
        print(f"  {name+'_M':>8s} {name+'_F':>8s}", end="")
    print()
    for li in range(L):
        print(f"  {layer_labels[li]:6d}", end="")
        for name in ["R1", "R2", "R3"]:
            print(f"  {mean_male[name][li]:+8.2f} {mean_female[name][li]:+8.2f}", end="")
        print()

    # ═══════════════════════════════════════════════════════
    # METRIC 4: Delta_g (where signal is written)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("DELTA_G (layer-to-layer change)")
    print("=" * 70)

    delta_unsigned: Dict[str, List[float]] = {}
    delta_signed_male: Dict[str, List[float]] = {}
    delta_signed_female: Dict[str, List[float]] = {}

    for k in range(K):
        name = f"R{k+1}"
        du, dsm, dsf = [], [], []
        for li in range(L - 1):
            d = g_np[k, li + 1] - g_np[k, li]  # (N,)
            du.append(float(np.abs(d).mean()))
            sd = pol[k] * d
            dsm.append(float(sd[male].mean()))
            dsf.append(float(sd[female].mean()))
        delta_unsigned[name] = du
        delta_signed_male[name] = dsm
        delta_signed_female[name] = dsf

    # Find peak delta layers
    for name in ["R1", "R2", "R3"]:
        peak = np.argmax(delta_unsigned[name])
        print(f"  {name}: peak |Δg| at transition {layer_labels[peak]}→{layer_labels[peak+1]}"
              f"  (|Δg|={delta_unsigned[name][peak]:.2f})")

    # ═══════════════════════════════════════════════════════
    # METRIC 5: Deliberation Index
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("DELIBERATION INDEX (within-class variance)")
    print("=" * 70)

    di_raw: Dict[str, List[float]] = {}
    di_norm: Dict[str, List[float]] = {}

    for k in range(K):
        name = f"R{k+1}"
        raw = []
        for li in range(L):
            vm = float(pol_g_np[k, li, male].var()) if male.sum() > 1 else 0.0
            vf = float(pol_g_np[k, li, female].var()) if female.sum() > 1 else 0.0
            raw.append(0.5 * (vm + vf))
        di_raw[name] = raw
        base = raw[0] if raw[0] > 1e-12 else 1.0
        di_norm[name] = [r / base for r in raw]

    for name in ["R1", "R2", "R3"]:
        print(f"  {name}: DI_embed={di_raw[name][0]:.2f}"
              f"  DI_final={di_raw[name][-1]:.2f}"
              f"  ratio={di_norm[name][-1]:.3f}")

    # ═══════════════════════════════════════════════════════
    # METRIC 6: RCS (Receptor Concentration Score)
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("RCS (Receptor Concentration Score)")
    print("=" * 70)

    g_mean_abs = np.abs(g_np).mean(axis=2)  # (K, L)
    rcs = []
    for li in range(L):
        w = g_mean_abs[:, li]
        total = w.sum()
        if total < 1e-12:
            rcs.append(float("nan"))
            continue
        p = w / total
        H = entropy(p)
        rcs.append(1.0 - H / math.log(K))

    print(f"  RCS_embed={rcs[0]:.4f}  RCS_final={rcs[-1]:.4f}")
    print(f"  RCS curve: {['%.3f' % r for r in rcs]}")

    # ═══════════════════════════════════════════════════════
    # KEY RESULT: Commitment Ordering
    # ═══════════════════════════════════════════════════════
    print(f"\n{'=' * 70}")
    print("KEY RESULT: COMMITMENT ORDERING")
    print("=" * 70)

    r3_before_r1_sign = primary["R3"] < primary["R1"]
    r3_before_r1_auc = commit_auc["R3"] < commit_auc["R1"]

    print(f"\n  Sign agreement (thr={PRIMARY_THR}):")
    print(f"    R1 commits at layer {primary['R1']}")
    print(f"    R2 commits at layer {primary['R2']}")
    print(f"    R3 commits at layer {primary['R3']}")
    print(f"    Combined commits at layer {primary['combined']}")

    print(f"\n  AUC commitment (thr={AUC_THR}):")
    print(f"    R1 commits at layer {commit_auc['R1']}")
    print(f"    R3 commits at layer {commit_auc['R3']}")

    print(f"\n  Prediction: R3 commits before R1 (inhibition before promotion)")
    print(f"    Sign agreement: {'CONFIRMED' if r3_before_r1_sign else 'NOT CONFIRMED'}"
          f"  (R3={primary['R3']} vs R1={primary['R1']})")
    print(f"    AUC: {'CONFIRMED' if r3_before_r1_auc else 'NOT CONFIRMED'}"
          f"  (R3={commit_auc['R3']} vs R1={commit_auc['R1']})")

    # ═══════════════════════════════════════════════════════
    # SAVE TENSORS (for downstream: phase diagram, error autopsy, typed gap)
    # ═══════════════════════════════════════════════════════
    tensor_path = out_dir / "exp0c_tensors.pt"
    torch.save({
        "layer_labels": layer_labels,
        "resid_all": resid_all,       # (13, N, 768)
        "g": g,                       # (K, 13, N)
        "polarity": polarity,         # (K,)
        "ys": torch.tensor(ys),       # (N,)
        "he_logits": torch.tensor(he_logits),
        "she_logits": torch.tensor(she_logits),
        "names": names,               # list of N strings
    }, str(tensor_path))
    print(f"\n[SAVE] {tensor_path}  (resid_all: {resid_all.shape})")

    # ═══════════════════════════════════════════════════════
    # SAVE JSON
    # ═══════════════════════════════════════════════════════
    json_path = out_dir / "exp0c_results.json"
    payload = {
        "meta": {
            "csv": str(csv_path), "N": N,
            "n_male": n_male, "n_female": n_female,
            "seq_len": int(seq_len),
            "receptors": rec_meta,
            "baseline_acc": round(acc, 4),
        },
        "layer_labels": layer_labels,
        "agreement_curves": agreement,
        "commitment_sign": commit_sign,
        "commitment_auc": commit_auc,
        "auc_curves": auc,
        "mean_male": mean_male,
        "mean_female": mean_female,
        "delta_unsigned": delta_unsigned,
        "delta_signed_male": delta_signed_male,
        "delta_signed_female": delta_signed_female,
        "di_norm": di_norm,
        "rcs": rcs,
        "r3_before_r1_sign": bool(r3_before_r1_sign),
        "r3_before_r1_auc": bool(r3_before_r1_auc),
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"[SAVE] {json_path}")

    # ═══════════════════════════════════════════════════════
    # PLOTS
    # ═══════════════════════════════════════════════════════
    xs = layer_labels
    colors = {"R1": "#3498db", "R2": "#95a5a6", "R3": "#e67e22", "combined": "#2c3e50"}

    # ── Plot 1: Sign Agreement + Commitment ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["R1", "R2", "R3", "combined"]:
        ax.plot(xs, agreement[name], "o-", label=name, color=colors[name],
                markersize=4, lw=1.5)
    ax.axhline(PRIMARY_THR, ls="--", color="gray", alpha=0.5,
               label=f"threshold={PRIMARY_THR}")
    for name in ["R1", "R3"]:
        ax.axvline(primary[name], ls=":", color=colors[name], alpha=0.4)
        ax.text(primary[name] + 0.15, 0.52, f"{name}={primary[name]}",
                fontsize=8, color=colors[name])
    ax.set_xlabel("Layer (-1=embedding, 0..11=post-block)", fontsize=10)
    ax.set_ylabel("Agreement with final sign", fontsize=10)
    ax.set_title(f"Sign Agreement Curves (commitment at {PRIMARY_THR})", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    p1 = out_dir / "exp0c_commitment_plot.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p1}")

    # ── Plot 2: AUC Curves ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for name in ["R1", "R2", "R3", "combined"]:
        ax.plot(xs, auc[name], "o-", label=name, color=colors[name],
                markersize=4, lw=1.5)
    for name in ["R1", "R3"]:
        ax.axvline(commit_auc[name], ls=":", color=colors[name], alpha=0.4)
    ax.axhline(AUC_THR, ls="--", color="gray", alpha=0.5,
               label=f"AUC threshold={AUC_THR}")
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("ROC AUC (polarity-adjusted)", fontsize=10)
    ax.set_title("Per-Layer AUC of Each Receptor", fontsize=12)
    ax.legend(fontsize=9)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    p2 = out_dir / "exp0c_auc_plot.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p2}")

    # ── Plot 3: Mean Activation by Gender (3 panels) ──
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for k, ax in enumerate(axes):
        name = f"R{k+1}"
        ax.plot(xs, mean_male[name], "o-", label="male", color="#3498db",
                markersize=3, lw=1.5)
        ax.plot(xs, mean_female[name], "o-", label="female", color="#e74c3c",
                markersize=3, lw=1.5)
        ax.axvline(primary[name], ls=":", color="gray", alpha=0.4)
        ax.axhline(0, ls="-", color="gray", alpha=0.2)
        ax.set_title(f"{name} (pol={pol[k]:+.0f})", fontsize=11)
        ax.set_xlabel("Layer", fontsize=9)
        if k == 0:
            ax.set_ylabel("Mean polarity-adjusted activation", fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.15)
    fig.suptitle("Mean Activation by Gender (polarity-adjusted)", fontsize=12, y=1.01)
    plt.tight_layout()
    p3 = out_dir / "exp0c_activation_by_gender.png"
    fig.savefig(p3, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[PLOT] {p3}")

    # ── Plot 4: Delta by Layer ──
    dx = layer_labels[:-1]  # 12 transition source layers
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(K):
        name = f"R{k+1}"
        ax.plot(dx, delta_unsigned[name], "o-", label=name, color=colors[name],
                markersize=4, lw=1.5)
    ax.set_xlabel("Layer transition source (l → l+1)", fontsize=10)
    ax.set_ylabel("Mean |Δg|", fontsize=10)
    ax.set_title("Where Signal Is Written (layer-wise activation change)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    p4 = out_dir / "exp0c_delta_by_layer.png"
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p4}")

    # ── Plot 5: Deliberation Index ──
    fig, ax = plt.subplots(figsize=(8, 5))
    for k in range(K):
        name = f"R{k+1}"
        ax.plot(xs, di_norm[name], "o-", label=name, color=colors[name],
                markersize=4, lw=1.5)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("DI / DI(embedding)", fontsize=10)
    ax.set_title("Deliberation Index (within-class variance, normalized)", fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    p5 = out_dir / "exp0c_deliberation.png"
    fig.savefig(p5, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p5}")

    # ── Plot 6: RCS ──
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(xs, rcs, "o-", color="#2c3e50", markersize=5, lw=2)
    ax.set_xlabel("Layer", fontsize=10)
    ax.set_ylabel("RCS (0=spread, 1=dominant)", fontsize=10)
    ax.set_title("Receptor Concentration Score per Layer", fontsize=12)
    ax.set_ylim(0, 1.02)
    ax.grid(True, alpha=0.15)
    plt.tight_layout()
    p6 = out_dir / "exp0c_rcs.png"
    fig.savefig(p6, dpi=150)
    plt.close(fig)
    print(f"[PLOT] {p6}")

    print("\n[DONE] Exp 0c complete.")


if __name__ == "__main__":
    main()
