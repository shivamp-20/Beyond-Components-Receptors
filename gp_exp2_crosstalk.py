#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
gp_exp2_crosstalk.py  —  Experiment 2: Receptor Cross-Talk Under Scaling

Scales each receptor's OV singular component INSIDE the forward pass
(not at the final residual) and measures all receptors' activations at
the final layer.  Separates geometric coupling (direction cosine overlap)
from computational coupling (downstream layer processing).

Key outputs
-----------
- Cross-talk ratio matrix:   how much each receptor's activation changes
                              when a different receptor is scaled.
- Computational fraction:     what fraction of cross-talk is NOT explained
                              by simple direction cosine (i.e., caused by
                              downstream computation).
- Linearity check:            R² of |Δg| vs |scale − 1|.
- Asymmetry check:            CompFrac(R3→R1) vs CompFrac(R1→R3).

Run on Kaggle
-------------
  !python gp_exp2_crosstalk.py \\
      --data_dir data_main \\
      --test_csv test_gp.csv \\
      --out_dir outputs/gp \\
      --receptors "10,9,0,+1;11,8,6,+1;9,7,1,-1" \\
      --scales "0.0,0.5,2.0,5.0,10.0,20.0" \\
      --ref_scale 5.0 \\
      --batch_size 64 \\
      --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer

# ── Import training-script helpers ──
# Rename if your file is called something else on Kaggle.
try:
    import train_gp_masks_and_dump_ov_logit_receptors_ddp as gp
except ImportError:
    import train_gp_masks_and_dump_ov_logit_receptors as gp


# ═══════════════════════════════════════════════════════════════════
# Receptor data-class
# ═══════════════════════════════════════════════════════════════════

@dataclass
class Receptor:
    name: str
    layer: int
    head: int
    sv_idx: int
    polarity: int           # +1 or −1
    u_vec: torch.Tensor     # (d_model+1,)  left singular vector
    sigma: float            # singular value
    v_row: torch.Tensor     # (d_model,)    right singular vector = direction


def parse_receptors_arg(s: str) -> List[Tuple[int, int, int, int]]:
    """Parse "L,H,SV,pol; …" into list of (layer, head, sv_idx, polarity)."""
    out = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        vals = [x.strip() for x in part.split(",")]
        if len(vals) != 4:
            raise ValueError(f"Expected 4 values per receptor, got: {part}")
        out.append((int(vals[0]), int(vals[1]), int(vals[2]), int(vals[3])))
    return out


def load_receptors(specs, ov, device) -> List[Receptor]:
    recs = []
    for i, (l, h, sv, pol) in enumerate(specs):
        svd = ov[l][h]
        assert 0 <= sv < svd.r, f"sv_idx {sv} out of range for ov[{l}][{h}].r={svd.r}"
        u = svd.U[:, sv].contiguous().to(device)
        s = float(svd.S[sv].item())
        v = svd.Vh[sv, :].contiguous().to(device)
        recs.append(Receptor(
            name=f"R{i+1}", layer=l, head=h, sv_idx=sv,
            polarity=pol, u_vec=u, sigma=s, v_row=v,
        ))
    return recs


# ═══════════════════════════════════════════════════════════════════
# Data helpers
# ═══════════════════════════════════════════════════════════════════

def expand_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Expand each CSV row into labelled examples (clean + corrupt)."""
    out: List[Dict[str, str]] = []
    for r in rows:
        p = (r.get("pronoun") or "").strip().lower()
        cp = (r.get("corr_pronoun") or "").strip().lower()
        if p in ("he", "she"):
            out.append({"text": r["prefix"], "label": p})
        if cp in ("he", "she"):
            out.append({"text": r["corr_prefix"], "label": cp})
    return out


def tokenize_batch(tokenizer, texts, device):
    ids_list = gp._encode_texts(tokenizer, texts)
    pad_len = max(len(x) for x in ids_list)
    pad_id = tokenizer.pad_token_id
    tokens, _attn, last_idx = gp._pad_to_length(ids_list, pad_len, pad_id, device)
    return tokens, last_idx


# ═══════════════════════════════════════════════════════════════════
# Forward pass with in-forward-pass OV intervention
# ═══════════════════════════════════════════════════════════════════

@torch.no_grad()
def forward_with_intervention(
    model: HookedTransformer,
    tokens: torch.Tensor,           # (B, S)
    last_idx: torch.Tensor,         # (B,)
    receptors: List[Receptor],
    src_idx: int = -1,              # which receptor to scale; -1 = clean
    scale: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Full forward pass.  If src_idx >= 0, the src receptor's OV singular
    component is multiplied by `scale` at ALL positions inside its host
    head.  Downstream layers then process the modified residual stream.

    Returns
    -------
    g      : (B, K)  receptor activations  g_j = r_j · x_final(t*)
    a_src  : (B,)    scalar activation a_k(t*) of source receptor
             (clean value — computed before delta is applied)
             Zeros when src_idx < 0.
    """
    device = tokens.device
    dtype = torch.float32
    act_fn = gp.get_act_fn(model)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head
    K = len(receptors)
    B, S = tokens.shape

    causal = gp.make_causal_mask(S, device=str(device))
    x = model.embed(tokens) + model.pos_embed(tokens)

    a_src = torch.zeros(B, device=device, dtype=dtype)

    for l in range(n_layers):
        block = model.blocks[l]
        attn = block.attn
        mlp = block.mlp

        x_ln1 = block.ln1(x)
        head_outs = []

        for h in range(n_heads):
            # Attention pattern (standard, unmasked)
            pat = gp.attention_pattern_original(
                x_ln1, attn.W_Q[h], attn.b_Q[h],
                attn.W_K[h], attn.b_K[h], causal, d_head,
            )                                           # (B, S, S)
            ctx = pat @ x_ln1                           # (B, S, D)

            # Standard head output
            v = (ctx @ attn.W_V[h]) + attn.b_V[h]      # (B, S, Dh)
            head_out = v @ attn.W_O[h]                  # (B, S, D)

            # ── Intervention ──────────────────────────
            # If this head hosts the source receptor, scale its singular
            # component.  The delta is added to head_out at ALL positions
            # so downstream layers (including other receptors' host heads)
            # see the modification.
            if src_idx >= 0:
                rec = receptors[src_idx]
                if l == rec.layer and h == rec.head:
                    # Scalar activation at every position (clean, before delta)
                    #   a_k(t) = u_k^T · [ctx(t); 1]
                    ones = torch.ones(B, S, 1, device=device, dtype=ctx.dtype)
                    ctx_aug = torch.cat([ctx, ones], dim=-1)    # (B, S, D+1)
                    a_k = ctx_aug @ rec.u_vec                   # (B, S)

                    # Record at decision position
                    a_src = a_k[torch.arange(B, device=device), last_idx].to(dtype)

                    # Delta = (scale − 1) · σ_k · a_k(t) · v_k  at every position
                    coeff = (scale - 1.0) * rec.sigma           # scalar
                    delta = coeff * a_k.unsqueeze(-1) * rec.v_row   # (B, S, D)
                    head_out = head_out + delta

            head_outs.append(head_out)

        attn_out = torch.stack(head_outs, dim=0).sum(dim=0) + attn.b_O
        x = x + attn_out

        # MLP (standard)
        x_ln2 = block.ln2(x)
        pre = (x_ln2 @ mlp.W_in) + mlp.b_in
        h_act = act_fn(pre)
        x = x + ((h_act @ mlp.W_out) + mlp.b_out)

    # ── Receptor readout at decision position ──
    resid = x[torch.arange(B, device=device), last_idx, :]     # (B, D)
    g = torch.zeros(B, K, device=device, dtype=dtype)
    for j, rec in enumerate(receptors):
        g[:, j] = (resid * rec.v_row).sum(dim=-1).to(dtype)

    return g, a_src


# ═══════════════════════════════════════════════════════════════════
# Batch runner
# ═══════════════════════════════════════════════════════════════════

def run_all_examples(model, tokenizer, examples, receptors,
                     src_idx, scale, batch_size, device):
    """Run forward pass on all examples, return (g, a_src) tensors."""
    g_parts, a_parts = [], []
    for i in range(0, len(examples), batch_size):
        batch = examples[i : i + batch_size]
        tokens, last_idx = tokenize_batch(
            tokenizer, [e["text"] for e in batch], device,
        )
        g, a_src = forward_with_intervention(
            model, tokens, last_idx, receptors, src_idx, scale,
        )
        g_parts.append(g.cpu())
        a_parts.append(a_src.cpu())
    return torch.cat(g_parts, 0), torch.cat(a_parts, 0)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir",   default="data_main")
    ap.add_argument("--test_csv",   default="test_gp.csv")
    ap.add_argument("--out_dir",    default="outputs/gp")
    ap.add_argument("--receptors",  default="10,9,0,+1;11,8,6,+1;9,7,1,-1")
    ap.add_argument("--scales",     default="0.0,0.5,2.0,5.0,10.0,20.0")
    ap.add_argument("--ref_scale",  type=float, default=5.0,
                    help="Reference scale for summary tables / plots")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device",     default="cuda")
    ap.add_argument("--seed",       type=int, default=0)
    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    scales = [float(x) for x in args.scales.split(",")]
    ref = args.ref_scale

    # ── Load model ──
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # ── Load SVD cache & receptors ──
    svd_path = out_dir / "svd_cache.pt"
    if not svd_path.exists():
        raise FileNotFoundError(f"Missing: {svd_path}")
    _qk, ov, _mlp_in, _mlp_out, _rt = gp.load_svd_cache(str(svd_path), device=device)

    specs = parse_receptors_arg(args.receptors)
    receptors = load_receptors(specs, ov, device)
    K = len(receptors)

    # ── Load data ──
    test_rows = gp.load_gp_csv(str(Path(args.data_dir) / args.test_csv))
    examples = expand_rows(test_rows)
    N = len(examples)

    # ════════════════════════════════════════════════════════════════
    print("=" * 70)
    print("EXPERIMENT 2: RECEPTOR CROSS-TALK UNDER SCALING")
    print("=" * 70)

    print(f"\nN = {N} examples")
    print(f"Scales: {scales}")
    print(f"Reference scale: {ref}")

    # ── Receptor info ──
    print("\nRECEPTOR DIRECTIONS:")
    for r in receptors:
        print(f"  {r.name}: L{r.layer}H{r.head} sv{r.sv_idx}  pol={r.polarity:+d}"
              f"  sigma={r.sigma:.4f}  ||v||={r.v_row.norm().item():.4f}")

    # ── Cosine similarities ──
    cos_mat = torch.zeros(K, K)
    for i in range(K):
        for j in range(K):
            cos_mat[i, j] = torch.dot(receptors[i].v_row, receptors[j].v_row).item()

    print("\nCOSINE SIMILARITIES:")
    for i in range(K):
        for j in range(i + 1, K):
            print(f"  cos({receptors[i].name},{receptors[j].name}) = {cos_mat[i,j]:+.4f}")

    # ═══════════════  CLEAN PASS  ═══════════════
    print("\n[RUN] Clean forward pass ...")
    g_clean, _ = run_all_examples(
        model, tokenizer, examples, receptors,
        src_idx=-1, scale=1.0, batch_size=args.batch_size, device=device,
    )
    print("CLEAN BASELINE (g_j at final layer):")
    for j in range(K):
        gc = g_clean[:, j]
        print(f"  g_{receptors[j].name}:  mean={gc.mean():.4f}  std={gc.std():.4f}"
              f"  min={gc.min():.4f}  max={gc.max():.4f}")

    # ═══════════════  INTERVENTION PASSES  ═══════════════
    # results[src_idx][scale] = {"g": (N,K), "a_src": (N,)}
    results: Dict[int, Dict[float, Dict[str, torch.Tensor]]] = {}

    for si in range(K):
        results[si] = {}
        for sc in scales:
            tag = f"{receptors[si].name} × {sc}"
            print(f"[RUN] Scaling {tag} ...", end="", flush=True)
            g_int, a_src = run_all_examples(
                model, tokenizer, examples, receptors,
                src_idx=si, scale=sc, batch_size=args.batch_size, device=device,
            )
            results[si][sc] = {"g": g_int, "a_src": a_src}
            print("  done")

    # ═══════════════  SANITY CHECK  ═══════════════
    # At scale=0, self-delta should be ≈ −sigma * mean(|a_k|)
    print("\n" + "=" * 70)
    print("SANITY CHECKS")
    print("=" * 70)

    for si in range(K):
        if 0.0 not in results[si]:
            continue
        g0 = results[si][0.0]["g"]
        a0 = results[si][0.0]["a_src"]
        dg_self = (g0[:, si] - g_clean[:, si])
        expected = -receptors[si].sigma * a0          # geometric prediction for self
        err = (dg_self - expected).abs().mean().item()
        print(f"  scale=0, {receptors[si].name}: mean|Δg_self − (−σ·a_k)| = {err:.6f}"
              f"  (should be small if downstream processing is modest)")

    # ═══════════════  COMPUTE METRICS  ═══════════════
    # metrics[si][sc][tj] = dict of floats
    metrics: Dict[int, Dict[float, Dict[int, Dict[str, float]]]] = {}

    for si in range(K):
        metrics[si] = {}
        src = receptors[si]
        for sc in scales:
            metrics[si][sc] = {}
            g_int = results[si][sc]["g"]
            a_src = results[si][sc]["a_src"]
            dg = g_int - g_clean                        # (N, K)

            for tj in range(K):
                actual = dg[:, tj]                      # (N,)
                cos_ij = cos_mat[si, tj].item()
                geo = (sc - 1.0) * src.sigma * a_src * cos_ij   # (N,)
                comp = actual - geo                     # (N,)

                m_actual     = actual.mean().item()
                m_abs_actual = actual.abs().mean().item()
                m_geo        = geo.mean().item()
                m_abs_geo    = geo.abs().mean().item()
                m_abs_comp   = comp.abs().mean().item()
                comp_frac    = m_abs_comp / (m_abs_actual + 1e-12)

                # Pearson correlation (actual vs geometric, per-example)
                a_np, g_np = actual.numpy(), geo.numpy()
                if actual.std() > 1e-9 and geo.std() > 1e-9:
                    corr = float(np.corrcoef(a_np, g_np)[0, 1])
                else:
                    corr = float("nan")

                metrics[si][sc][tj] = {
                    "mean_actual":     m_actual,
                    "mean_abs_actual": m_abs_actual,
                    "mean_geo":        m_geo,
                    "mean_abs_geo":    m_abs_geo,
                    "mean_abs_comp":   m_abs_comp,
                    "comp_frac":       comp_frac,
                    "corr":            corr,
                }

    # ═══════════════  PRINT TABLES  ═══════════════
    for si in range(K):
        src = receptors[si]
        print(f"\n{'='*70}")
        print(f"SCALING {src.name}  (L{src.layer}H{src.head} sv{src.sv_idx})")
        print(f"{'='*70}")

        # Column header
        parts = [f"{'scale':>6}"]
        for tj in range(K):
            parts.append(f"  |Δg_{receptors[tj].name}|(act)")
            parts.append(f"  |Δg_{receptors[tj].name}|(geo)")
        hdr = "".join(parts)
        print(hdr)
        print("-" * len(hdr))

        for sc in scales:
            row = f"{sc:6.1f}"
            for tj in range(K):
                m = metrics[si][sc][tj]
                row += f"  {m['mean_abs_actual']:12.4f}"
                row += f"  {m['mean_abs_geo']:12.4f}"
            print(row)

        # Cross-talk ratio at ref scale
        if ref in scales:
            self_abs = metrics[si][ref][si]["mean_abs_actual"]
            print(f"\n  [CROSS-TALK RATIO]  scale={ref}")
            for tj in range(K):
                m = metrics[si][ref][tj]
                ctr = m["mean_abs_actual"] / (self_abs + 1e-12)
                tag = " (self)" if tj == si else ""
                print(f"    {src.name}→{receptors[tj].name}: {ctr:.4f}{tag}")

            print(f"\n  [COMPUTATIONAL FRACTION]  scale={ref}")
            for tj in range(K):
                m = metrics[si][ref][tj]
                print(f"    {src.name}→{receptors[tj].name}:"
                      f"  CompFrac={m['comp_frac']*100:5.1f}%"
                      f"  corr(act,geo)={m['corr']:.3f}")

    # ═══════════════  SUMMARY  ═══════════════
    if ref in scales:
        print(f"\n{'='*70}")
        print(f"SUMMARY  (reference scale = {ref})")
        print(f"{'='*70}")

        # Cross-talk ratio matrix
        print("\nCROSS-TALK RATIO MATRIX  (mean|Δg_target| / mean|Δg_self|):")
        print(f"  src\\tgt", end="")
        for j in range(K):
            print(f"  {receptors[j].name:>8}", end="")
        print()
        for i in range(K):
            self_abs = metrics[i][ref][i]["mean_abs_actual"]
            print(f"  {receptors[i].name}→  ", end="")
            for j in range(K):
                ctr = metrics[i][ref][j]["mean_abs_actual"] / (self_abs + 1e-12)
                print(f"  {ctr:8.4f}", end="")
            print()

        # Computational fraction matrix
        print("\nCOMPUTATIONAL FRACTION MATRIX  (% of |Δg| not explained by cosine):")
        print(f"  src\\tgt", end="")
        for j in range(K):
            print(f"  {receptors[j].name:>8}", end="")
        print()
        for i in range(K):
            print(f"  {receptors[i].name}→  ", end="")
            for j in range(K):
                cf = metrics[i][ref][j]["comp_frac"] * 100
                print(f"  {cf:7.1f}%", end="")
            print()

        # Asymmetry check
        print("\nASYMMETRY CHECK  (R3→R1 should have higher CompFrac than R1→R3):")
        cf_31 = metrics[2][ref][0]["comp_frac"] * 100
        cf_13 = metrics[0][ref][2]["comp_frac"] * 100
        pred = "CONFIRMED" if cf_31 > cf_13 else "NOT CONFIRMED"
        print(f"  CompFrac(R3→R1) = {cf_31:.1f}%")
        print(f"  CompFrac(R1→R3) = {cf_13:.1f}%")
        print(f"  Prediction: {pred}")

    # Linearity check
    print("\nLINEARITY CHECK  (R² of mean|Δg| vs |scale − 1|):")
    for i in range(K):
        for j in range(K):
            if i == j:
                continue
            x_vals = np.array([abs(s - 1.0) for s in scales])
            y_vals = np.array([metrics[i][s][j]["mean_abs_actual"] for s in scales])
            if x_vals.std() > 0 and y_vals.std() > 0:
                r2 = float(np.corrcoef(x_vals, y_vals)[0, 1]) ** 2
            else:
                r2 = 0.0
            print(f"  {receptors[i].name}→{receptors[j].name}: R² = {r2:.4f}")

    # ═══════════════  PLOTS  ═══════════════

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    # ── Plot 1: Cross-talk vs scale ──
    fig, axes = plt.subplots(1, K, figsize=(5 * K, 4.5))
    if K == 1:
        axes = [axes]

    for si in range(K):
        ax = axes[si]
        for tj in range(K):
            act = [metrics[si][s][tj]["mean_abs_actual"] for s in scales]
            geo = [metrics[si][s][tj]["mean_abs_geo"] for s in scales]
            ax.plot(scales, act, "o-",  color=colors[tj], markersize=4,
                    label=f"→{receptors[tj].name} actual")
            ax.plot(scales, geo, "s--", color=colors[tj], markersize=3,
                    alpha=0.55, label=f"→{receptors[tj].name} geometric")
        ax.set_xlabel("Scale factor")
        ax.set_ylabel("Mean |Δg|")
        ax.set_title(f"Scaling {receptors[si].name} (L{receptors[si].layer}H{receptors[si].head})")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(True, alpha=0.3)

    fig.suptitle("Exp 2: Cross-Talk vs Scale", fontsize=13, y=1.02)
    plt.tight_layout()
    p1 = out_dir / "exp2_crosstalk_vs_scale.png"
    fig.savefig(p1, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[PLOT] {p1}")

    # ── Plot 2: Computational-fraction heatmap ──
    if ref in scales:
        fig, ax = plt.subplots(figsize=(4.5, 3.8))
        cf_arr = np.array([
            [metrics[i][ref][j]["comp_frac"] * 100 for j in range(K)]
            for i in range(K)
        ])
        im = ax.imshow(cf_arr, cmap="YlOrRd", vmin=0, vmax=max(100, cf_arr.max()),
                        aspect="auto")
        ax.set_xticks(range(K))
        ax.set_yticks(range(K))
        ax.set_xticklabels([r.name for r in receptors])
        ax.set_yticklabels([r.name for r in receptors])
        ax.set_xlabel("Target receptor")
        ax.set_ylabel("Source receptor (scaled)")
        ax.set_title(f"Computational Fraction (%)  scale={ref}")
        for i in range(K):
            for j in range(K):
                clr = "white" if cf_arr[i, j] > 50 else "black"
                ax.text(j, i, f"{cf_arr[i,j]:.1f}%", ha="center", va="center",
                        color=clr, fontsize=11, fontweight="bold")
        plt.colorbar(im, ax=ax, label="%")
        plt.tight_layout()
        p2 = out_dir / "exp2_comp_fraction.png"
        fig.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] {p2}")

    # ── Plot 3: Actual vs Geometric bar chart ──
    if ref in scales:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        pairs, act_vals, geo_vals = [], [], []
        for i in range(K):
            for j in range(K):
                if i == j:
                    continue
                pairs.append(f"{receptors[i].name}→{receptors[j].name}")
                act_vals.append(metrics[i][ref][j]["mean_abs_actual"])
                geo_vals.append(metrics[i][ref][j]["mean_abs_geo"])

        x = np.arange(len(pairs))
        w = 0.35
        ax.bar(x - w / 2, act_vals, w, label="Actual |Δg|", color="#1f77b4")
        ax.bar(x + w / 2, geo_vals, w, label="Geometric |Δg|", color="#ff7f0e",
               alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels(pairs, rotation=30, ha="right")
        ax.set_ylabel("Mean |Δg|")
        ax.set_title(f"Actual vs Geometric Cross-Talk  (scale={ref})")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.3)
        plt.tight_layout()
        p3 = out_dir / "exp2_actual_vs_geometric.png"
        fig.savefig(p3, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"[PLOT] {p3}")

    # ═══════════════  SAVE JSON  ═══════════════
    json_out: Dict[str, Any] = {
        "experiment": "exp2_crosstalk",
        "N": N,
        "scales": scales,
        "ref_scale": ref,
        "receptors": [
            {"name": r.name, "layer": r.layer, "head": r.head,
             "sv_idx": r.sv_idx, "polarity": r.polarity, "sigma": r.sigma}
            for r in receptors
        ],
        "cosine_similarities": {
            f"{receptors[i].name},{receptors[j].name}": round(cos_mat[i, j].item(), 6)
            for i in range(K) for j in range(i + 1, K)
        },
        "metrics": {},
    }
    for si in range(K):
        sk = receptors[si].name
        json_out["metrics"][sk] = {}
        for sc in scales:
            json_out["metrics"][sk][str(sc)] = {}
            for tj in range(K):
                tk = receptors[tj].name
                json_out["metrics"][sk][str(sc)][tk] = metrics[si][sc][tj]

    jp = out_dir / "exp2_crosstalk_results.json"
    with open(jp, "w") as f:
        json.dump(json_out, f, indent=2)
    print(f"\n[SAVE] {jp}")
    print("\n[DONE]")


if __name__ == "__main__":
    main()
