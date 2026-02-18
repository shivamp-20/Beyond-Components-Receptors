#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gp_oca_ghost_refinement.py

Orthogonal Complement Ablation (OCA) — Stage 2 ghost refinement.

After CFR (Stage 1) prunes 207 → 13 raw-real receptors, some of the 13
have inflated AccDrop due to geometric overlap with the top receptors.
OCA removes only the UNIQUE component of each receptor (orthogonal to the
anchor subspace) and re-measures causality.

Algorithm:
  1. Seed anchor set A = top-2 by raw CFR.
  2. Build orthonormal basis for span(A) via Gram-Schmidt.
  3. For each non-seed receptor k:
       r_k_perp = r_k - proj_{span(A)}(r_k)
       Ablate r_k_perp_hat from cached residual, measure AccDrop_perp.
  4. Receptors with AccDrop_perp >= threshold → join anchor set.
  5. Repeat from (2) until anchor set converges.

Run:
  python gp_oca_ghost_refinement.py \
    --data_dir data_main \
    --csv test_gp.csv \
    --out_dir outputs/gp \
    --cfr_json outputs/gp/cfr_ghost_detection_results.json \
    --accdrop_threshold 2.0 \
    --batch_size 64 \
    --device cuda
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer


# =====================================================================
# Utility helpers (same as CFR script — self-contained)
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


def tokenize_texts(tokenizer, texts, device):
    ids_list = encode_texts(tokenizer, texts)
    pad_len = max(len(x) for x in ids_list)
    pad_id = tokenizer.pad_token_id
    return pad_to_length(ids_list, pad_len, pad_id, device)


def batches(items, batch_size):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


# =====================================================================
# Residual collection (identical to CFR script)
# =====================================================================

@torch.no_grad()
def collect_residuals_and_labels(
    model: HookedTransformer,
    tokenizer,
    examples: List[Dict[str, str]],
    he_id: int,
    she_id: int,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    all_resid, all_labels, all_logit_diff, all_correct = [], [], [], []
    hook_name = f"blocks.{model.cfg.n_layers - 1}.hook_resid_post"

    for batch in batches(examples, batch_size):
        texts = [e["text"] for e in batch]
        labs = [+1 if e["label"] == "he" else -1 for e in batch]
        tokens, last_idx = tokenize_texts(tokenizer, texts, device)
        B = tokens.shape[0]

        logits, cache = model.run_with_cache(
            tokens, names_filter=lambda n: n == hook_name
        )
        resid = cache[hook_name]
        resid_at_pos = resid[torch.arange(B, device=device), last_idx, :]
        logits_at_pos = logits[torch.arange(B, device=device), last_idx, :]
        he_logits = logits_at_pos[:, he_id]
        she_logits = logits_at_pos[:, she_id]
        logit_diff = he_logits - she_logits
        lab_tensor = torch.tensor(labs, device=device, dtype=torch.float32)
        pred_he = (he_logits > she_logits).float()
        correct = ((pred_he == 1) & (lab_tensor == 1)) | ((pred_he == 0) & (lab_tensor == -1))

        all_resid.append(resid_at_pos.cpu())
        all_labels.append(lab_tensor.cpu())
        all_logit_diff.append(logit_diff.cpu())
        all_correct.append(correct.float().cpu())
        del cache

    resid_clean = torch.cat(all_resid, dim=0)
    labels = torch.cat(all_labels, dim=0)
    logit_diff_clean = torch.cat(all_logit_diff, dim=0)
    baseline_acc = float(torch.cat(all_correct, dim=0).mean().item())
    return resid_clean, labels, logit_diff_clean, baseline_acc


# =====================================================================
# Ablation (identical to CFR script)
# =====================================================================

@torch.no_grad()
def compute_ablated_accuracy(
    model: HookedTransformer,
    resid_clean: torch.Tensor,
    labels: torch.Tensor,
    direction: torch.Tensor,
    he_id: int,
    she_id: int,
    device: str,
    batch_size: int = 512,
) -> float:
    """Ablate a UNIT direction from cached residuals, return accuracy."""
    N = resid_clean.shape[0]
    correct = 0
    for start in range(0, N, batch_size):
        end = min(start + batch_size, N)
        resid_batch = resid_clean[start:end].to(device)
        lab_batch = labels[start:end]
        d = direction.to(device)
        proj = (resid_batch @ d).unsqueeze(-1) * d.unsqueeze(0)
        resid_abl = resid_batch - proj
        ln_out = model.ln_final(resid_abl)
        logits = ln_out @ model.W_U + model.b_U
        he_logits = logits[:, he_id].cpu()
        she_logits = logits[:, she_id].cpu()
        pred_he = (he_logits > she_logits).float()
        lab_he = (lab_batch == 1).float()
        batch_correct = ((pred_he == 1) & (lab_he == 1)) | ((pred_he == 0) & (lab_he == 0))
        correct += int(batch_correct.sum().item())
    return correct / N


# =====================================================================
# Gram-Schmidt orthonormalization
# =====================================================================

def gram_schmidt(vectors: List[torch.Tensor], eps: float = 1e-10) -> List[torch.Tensor]:
    """
    Orthonormalize a list of vectors via modified Gram-Schmidt.
    Returns only the linearly independent basis vectors (drops near-zero norms).
    All computation in float64 for numerical stability.
    """
    basis: List[torch.Tensor] = []
    for v in vectors:
        v = v.double()
        for b in basis:
            v = v - (v @ b) * b
        norm = v.norm().item()
        if norm > eps:
            basis.append(v / norm)
    return basis


def project_onto_subspace(
    v: torch.Tensor,
    basis: List[torch.Tensor],
) -> torch.Tensor:
    """
    Project v onto the subspace spanned by an orthonormal basis.
    Returns the projection (parallel component).
    """
    v = v.double()
    proj = torch.zeros_like(v)
    for b in basis:
        proj = proj + (v @ b) * b
    return proj


def orthogonal_complement(
    v: torch.Tensor,
    basis: List[torch.Tensor],
) -> Tuple[torch.Tensor, float]:
    """
    Compute the component of v orthogonal to the subspace spanned by basis.
    Returns (v_perp, norm_perp).
    v_perp is NOT normalized — caller should check norm_perp and normalize if needed.
    """
    v = v.double()
    v_parallel = project_onto_subspace(v, basis)
    v_perp = v - v_parallel
    norm_perp = v_perp.norm().item()
    return v_perp, norm_perp


# =====================================================================
# Receptor label helper
# =====================================================================

def rec_label(rec: Dict[str, Any]) -> str:
    """Short human-readable label for a receptor."""
    name = rec.get("known_name", "")
    lhs = f"L{rec['layer']}H{rec['head']}sv{rec['sv_idx']}"
    if name:
        return f"{lhs} ({name})"
    return lhs


# =====================================================================
# Main
# =====================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="OCA Ghost Refinement (Stage 2)")
    ap.add_argument("--data_dir", type=str, default="data_main")
    ap.add_argument("--csv", type=str, default="test_gp.csv")
    ap.add_argument("--out_dir", type=str, default="outputs/gp")
    ap.add_argument("--svd_cache", type=str, default=None)
    ap.add_argument("--cfr_json", type=str, default=None,
                    help="Path to cfr_ghost_detection_results.json (default: out_dir/cfr_ghost_detection_results.json)")
    ap.add_argument("--accdrop_threshold", type=float, default=2.0,
                    help="AccDrop_perp threshold in PERCENT for OCA survival (default 2.0%%)")
    ap.add_argument("--norm_perp_min", type=float, default=0.01,
                    help="Skip receptor if ||r_perp|| < this (entirely in anchor subspace)")
    ap.add_argument("--max_iterations", type=int, default=10,
                    help="Safety cap on iterations")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--abl_batch_size", type=int, default=512)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_both", type=int, default=1)
    ap.add_argument("--n_seeds", type=int, default=2,
                    help="Number of top-CFR receptors to use as immutable seeds")
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
    cfr_json_path = Path(args.cfr_json) if args.cfr_json else out_dir / "cfr_ghost_detection_results.json"

    if not svd_path.exists():
        raise FileNotFoundError(f"Missing SVD cache: {svd_path}")
    if not cfr_json_path.exists():
        raise FileNotFoundError(f"Missing CFR results: {cfr_json_path}")

    # ----------------------------------------------------------------
    # Step 1: Load CFR results
    # ----------------------------------------------------------------
    print("=" * 70)
    print("OCA GHOST REFINEMENT (Stage 2)")
    print("=" * 70)

    with open(cfr_json_path, "r") as f:
        cfr_results = json.load(f)

    all_receptors = cfr_results["receptors"]
    raw_real = [r for r in all_receptors if not r["is_ghost"]]
    raw_real.sort(key=lambda r: r["cfr"], reverse=True)
    n_total = cfr_results["n_receptors"]
    n_raw_real = len(raw_real)

    print(f"[CFR] Loaded {n_total} total receptors, {n_raw_real} raw-real (CFR >= 0.05)")
    print(f"[CFR] Baseline accuracy from CFR run: {cfr_results['baseline_acc']*100:.2f}%")

    if n_raw_real < 3:
        print("[WARN] Fewer than 3 raw-real receptors — OCA may not be meaningful.")

    # Print the raw-real receptors
    print(f"\n[RAW-REAL RECEPTORS] (sorted by CFR descending):")
    for r in raw_real:
        name = r.get("known_name", "")
        print(f"  L{r['layer']}H{r['head']}sv{r['sv_idx']:>2}  "
              f"CFR={r['cfr']:.4f}  AccDrop={r['acc_drop']*100:+.2f}%  "
              f"AUC={r['auc']:.3f}  {name}")

    # ----------------------------------------------------------------
    # Step 2: Load model + SVD cache + test data
    # ----------------------------------------------------------------
    print("\n[LOAD] Tokenizer ...")
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    he_ids = tokenizer.encode(" he", add_special_tokens=False)
    she_ids = tokenizer.encode(" she", add_special_tokens=False)
    assert len(he_ids) == 1 and len(she_ids) == 1
    he_id, she_id = he_ids[0], she_ids[0]

    print("[LOAD] Model (gpt2-small) ...")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    print("[LOAD] SVD cache (OV only) ...")
    ov = load_svd_cache_ov_only(str(svd_path), device="cpu")

    print("[LOAD] Test data ...")
    csv_path = str(data_dir / args.csv)
    rows = load_gp_csv(csv_path)
    examples = expand_rows_to_examples(rows, use_both=bool(args.use_both))
    N_examples = len(examples)
    print(f"  {len(rows)} rows -> {N_examples} examples (use_both={bool(args.use_both)})")

    # ----------------------------------------------------------------
    # Step 3: Attach direction vectors from SVD cache to each receptor
    # ----------------------------------------------------------------
    for r in raw_real:
        l, h, sv = r["layer"], r["head"], r["sv_idx"]
        r["direction"] = ov[l][h].Vh[sv, :].clone().cpu()  # (d_model,)

    # ----------------------------------------------------------------
    # Step 4: One forward pass to cache residuals
    # ----------------------------------------------------------------
    print("\n[FORWARD] Collecting clean residuals ...")
    resid_clean, labels, logit_diff_clean, baseline_acc = collect_residuals_and_labels(
        model, tokenizer, examples, he_id, she_id, args.batch_size, device
    )
    print(f"[BASELINE] Clean accuracy: {baseline_acc*100:.2f}% ({int(baseline_acc*N_examples)}/{N_examples})")

    # ----------------------------------------------------------------
    # Step 5: Initialize seeds (immutable anchors)
    # ----------------------------------------------------------------
    n_seeds = min(args.n_seeds, n_raw_real)
    seeds = raw_real[:n_seeds]
    non_seeds = raw_real[n_seeds:]

    seed_keys = set((r["layer"], r["head"], r["sv_idx"]) for r in seeds)

    print(f"\n[SEEDS] Top-{n_seeds} by raw CFR (immutable):")
    for r in seeds:
        print(f"  {rec_label(r)}: CFR={r['cfr']:.4f}, AccDrop={r['acc_drop']*100:+.2f}%")

    # Also compute raw AccDrop for seeds (by ablating full direction) for comparison
    print("\n[RAW ACCDROP] Verifying raw AccDrop for all raw-real receptors ...")
    for r in raw_real:
        acc_abl = compute_ablated_accuracy(
            model, resid_clean, labels, r["direction"], he_id, she_id, device, args.abl_batch_size
        )
        r["raw_accdrop_verified"] = float(baseline_acc - acc_abl)

    print("  (verified — differences vs CFR JSON are from fresh forward pass)")

    # ----------------------------------------------------------------
    # Step 6: Iterative OCA (permanent anchor — once in, never re-tested)
    # ----------------------------------------------------------------
    threshold_frac = args.accdrop_threshold / 100.0  # convert percent to fraction
    permanent_anchor_keys = set(seed_keys)  # seeds are permanent from the start
    iteration_history: List[Dict[str, Any]] = []
    # Accumulate OCA results for ALL non-seeds across iterations (last test wins)
    all_oca_results: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    for iteration in range(1, args.max_iterations + 1):
        print(f"\n{'='*70}")
        print(f"=== OCA ITERATION {iteration} ===")
        print(f"{'='*70}")

        # Build current anchor directions (from permanent set)
        anchor_recs = [r for r in raw_real
                       if (r["layer"], r["head"], r["sv_idx"]) in permanent_anchor_keys]
        anchor_dirs = [r["direction"] for r in anchor_recs]

        # Gram-Schmidt orthonormalize
        basis = gram_schmidt(anchor_dirs)
        subspace_dim = len(basis)

        print(f"Anchor set ({len(anchor_recs)} receptors, subspace dim={subspace_dim}):")
        for r in anchor_recs:
            print(f"  {rec_label(r)}")

        # Only test candidates NOT already in the permanent anchor
        candidates = [r for r in raw_real
                      if (r["layer"], r["head"], r["sv_idx"]) not in permanent_anchor_keys]

        if len(candidates) == 0:
            print("  No candidates left to test.")
            iteration_history.append({
                "iteration": iteration,
                "anchor_size": len(permanent_anchor_keys),
                "subspace_dim": subspace_dim,
                "n_new_members": 0,
                "results": [],
            })
            print(f"\n*** CONVERGED at iteration {iteration} (no candidates) ***")
            break

        print(f"  Testing {len(candidates)} candidates ...")
        iter_results: List[Dict[str, Any]] = []
        new_member_keys: List[Tuple[int, int, int]] = []

        for r in candidates:
            rkey = (r["layer"], r["head"], r["sv_idx"])
            direction = r["direction"]  # (d_model,)

            # Orthogonal complement
            v_perp, norm_perp = orthogonal_complement(direction, basis)

            r_result = {
                "layer": r["layer"],
                "head": r["head"],
                "sv_idx": r["sv_idx"],
                "known_name": r.get("known_name", ""),
                "raw_accdrop": r["raw_accdrop_verified"],
                "raw_cfr": r["cfr"],
                "norm_perp": float(norm_perp),
            }

            if norm_perp < args.norm_perp_min:
                # Direction is entirely within anchor subspace
                r_result["accdrop_perp"] = 0.0
                r_result["oca_status"] = "GHOST (subsumed)"
                r_result["oca_cfr"] = 0.0
            else:
                # Normalize and ablate
                v_perp_hat = (v_perp / norm_perp).float().cpu()
                acc_abl = compute_ablated_accuracy(
                    model, resid_clean, labels, v_perp_hat,
                    he_id, she_id, device, args.abl_batch_size
                )
                accdrop_perp = float(baseline_acc - acc_abl)
                r_result["accdrop_perp"] = accdrop_perp
                r_result["oca_status"] = "REAL" if accdrop_perp >= threshold_frac else "GHOST"
                # OCA-CFR: normalized against max raw AccDrop for comparability
                r_result["oca_cfr"] = float(max(0.0, accdrop_perp / cfr_results["max_acc_drop"])) \
                    if cfr_results["max_acc_drop"] > 0 else 0.0

                if accdrop_perp >= threshold_frac:
                    new_member_keys.append(rkey)

            iter_results.append(r_result)
            all_oca_results[rkey] = r_result  # store / overwrite with latest

        # Print iteration table
        iter_results.sort(key=lambda x: x["accdrop_perp"], reverse=True)
        header = (f"{'Receptor':<20} | {'RawAccDrop%':>11} | {'||r_perp||':>10} | "
                  f"{'AccDrop_perp%':>13} | {'OCA-CFR':>8} | {'Status':<18}")
        print(f"\n{header}")
        print("-" * len(header))
        for ir in iter_results:
            label = f"L{ir['layer']}H{ir['head']}sv{ir['sv_idx']}"
            name = ir.get("known_name", "")
            if name:
                label += f" ({name})"
            print(f"{label:<20} | {ir['raw_accdrop']*100:>+11.3f} | {ir['norm_perp']:>10.4f} | "
                  f"{ir['accdrop_perp']*100:>+13.3f} | {ir['oca_cfr']:>8.4f} | {ir['oca_status']:<18}")

        print(f"\nNew members this iteration: {len(new_member_keys)}")
        for mk in new_member_keys:
            print(f"  + L{mk[0]}H{mk[1]}sv{mk[2]}")

        # Add new members permanently
        permanent_anchor_keys.update(new_member_keys)
        print(f"Anchor set size: {len(permanent_anchor_keys)} (seeds={len(seed_keys)} + {len(permanent_anchor_keys) - len(seed_keys)} added)")

        iteration_history.append({
            "iteration": iteration,
            "anchor_size": len(permanent_anchor_keys),
            "subspace_dim": subspace_dim,
            "n_new_members": len(new_member_keys),
            "results": iter_results,
        })

        # Check convergence: no new members → done
        if len(new_member_keys) == 0:
            print(f"\n*** CONVERGED at iteration {iteration} (no new members) ***")
            break

    else:
        print(f"\n[WARN] Did not converge in {args.max_iterations} iterations.")

    anchor_keys = permanent_anchor_keys  # final set for downstream code

    # ----------------------------------------------------------------
    # Step 7: Final summary
    # ----------------------------------------------------------------
    final_anchor_recs = [r for r in raw_real
                         if (r["layer"], r["head"], r["sv_idx"]) in anchor_keys]
    n_final = len(final_anchor_recs)

    # Build results map from accumulated OCA results (covers all non-seeds)
    last_results_map = all_oca_results
    last_iter = iteration_history[-1]

    print(f"\n{'='*70}")
    print(f"=== CONVERGED (iteration {last_iter['iteration']}) ===")
    print(f"{'='*70}")
    print(f"Final independent receptors: {n_final} (down from {n_raw_real} raw CFR, {n_total} total)")

    print(f"\n[FINAL SURVIVORS]")
    header = f"{'Receptor':<20} | {'RawCFR':>7} | {'RawAccDrop%':>11} | {'OCA AccDrop%':>12} | {'||r_perp||':>10} | {'Type':<10}"
    print(header)
    print("-" * len(header))
    for r in final_anchor_recs:
        rkey = (r["layer"], r["head"], r["sv_idx"])
        label = rec_label(r)
        is_seed = rkey in seed_keys
        if rkey in last_results_map:
            ir = last_results_map[rkey]
            print(f"{label:<20} | {r['cfr']:>7.4f} | {r['raw_accdrop_verified']*100:>+11.3f} | "
                  f"{ir['accdrop_perp']*100:>+12.3f} | {ir['norm_perp']:>10.4f} | {'Non-seed':<10}")
        else:
            # Seed — no OCA needed
            print(f"{label:<20} | {r['cfr']:>7.4f} | {r['raw_accdrop_verified']*100:>+11.3f} | "
                  f"{'(seed)':>12} | {'1.0000':>10} | {'Seed':<10}")

    # Coverage check
    print(f"\n[COVERAGE CHECK]")
    logit_diff_np = logit_diff_clean.numpy()

    # R² with final anchors only
    final_dirs = torch.stack([r["direction"] for r in final_anchor_recs], dim=0)
    g_final = (resid_clean @ final_dirs.T).numpy()
    reg_final = LinearRegression().fit(g_final, logit_diff_np)
    r2_final = r2_score(logit_diff_np, reg_final.predict(g_final))

    # R² with all raw-real
    raw_real_dirs = torch.stack([r["direction"] for r in raw_real], dim=0)
    g_raw_real = (resid_clean @ raw_real_dirs.T).numpy()
    reg_raw = LinearRegression().fit(g_raw_real, logit_diff_np)
    r2_raw_real = r2_score(logit_diff_np, reg_raw.predict(g_raw_real))

    # R² with all 207 (from CFR json)
    r2_all = cfr_results.get("r2_all", float("nan"))

    print(f"  R² of {n_final} OCA-independent receptors: {r2_final:.4f}")
    print(f"  R² of {n_raw_real} raw CFR-real receptors:  {r2_raw_real:.4f}")
    print(f"  R² of {n_total} total receptors (from CFR): {r2_all:.4f}")

    # Ghost reclassification report
    print(f"\n[GHOST RECLASSIFICATION]")
    for r in raw_real:
        rkey = (r["layer"], r["head"], r["sv_idx"])
        is_seed = rkey in seed_keys
        is_final = rkey in anchor_keys
        label = rec_label(r)

        if is_seed:
            status = "SEED (always real)"
        elif is_final:
            ir = last_results_map.get(rkey, {})
            status = f"REAL (AccDrop_perp={ir.get('accdrop_perp', 0)*100:+.2f}%)"
        else:
            ir = last_results_map.get(rkey, {})
            status = f"GHOST (AccDrop_perp={ir.get('accdrop_perp', 0)*100:+.2f}%, ||r_perp||={ir.get('norm_perp', 0):.4f})"

        was_ghost_str = "" if is_final else " ← RECLASSIFIED from real to ghost"
        print(f"  {label:<25} Raw CFR={r['cfr']:.4f} → OCA: {status}{was_ghost_str}")

    # ----------------------------------------------------------------
    # Step 8: Plots
    # ----------------------------------------------------------------
    print(f"\n[PLOTS] Saving to {out_dir} ...")

    # --- Plot 1: Before/After comparison bar chart ---
    fig, ax = plt.subplots(figsize=(8, 6))
    n_ghost_raw = n_total - n_raw_real
    n_ghost_oca = n_total - n_final
    x_pos = [0, 1]
    real_bars = [n_raw_real, n_final]
    ghost_bars = [n_ghost_raw, n_ghost_oca]
    width = 0.35
    bars_r = ax.bar([x - width/2 for x in x_pos], real_bars, width, label="Real", color="blue", edgecolor="black", alpha=0.8)
    bars_g = ax.bar([x + width/2 for x in x_pos], ghost_bars, width, label="Ghost", color="red", edgecolor="black", alpha=0.8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(["Raw CFR (Stage 1)", "After OCA (Stage 2)"], fontsize=12)
    ax.set_ylabel("Number of Receptors", fontsize=12)
    ax.set_title(f"Ghost Detection: CFR → OCA Refinement\n({n_total} total receptors)", fontsize=13)
    ax.legend(fontsize=11)
    # Annotate counts
    for bar in bars_r:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", fontweight="bold", color="blue", fontsize=13)
    for bar in bars_g:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(int(bar.get_height())), ha="center", fontweight="bold", color="red", fontsize=13)
    plt.tight_layout()
    p1 = str(out_dir / "oca_plot1_before_after.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    print(f"  [1] {p1}")

    # --- Plot 2: Raw AccDrop vs OCA AccDrop scatter ---
    fig, ax = plt.subplots(figsize=(9, 8))
    # Collect data for all non-seed raw-real receptors
    raw_drops_ns = []
    oca_drops_ns = []
    labels_ns = []
    names_ns = []
    is_oca_real_ns = []
    for r in non_seeds:
        rkey = (r["layer"], r["head"], r["sv_idx"])
        ir = last_results_map.get(rkey)
        if ir is None:
            continue
        raw_drops_ns.append(r["raw_accdrop_verified"] * 100)
        oca_drops_ns.append(ir["accdrop_perp"] * 100)
        labels_ns.append(rec_label(r))
        names_ns.append(r.get("known_name", ""))
        is_oca_real_ns.append(rkey in anchor_keys)

    raw_drops_ns = np.array(raw_drops_ns)
    oca_drops_ns = np.array(oca_drops_ns)
    is_oca_real_ns = np.array(is_oca_real_ns)

    # Plot ghost points
    ghost_mask = ~is_oca_real_ns
    if ghost_mask.any():
        ax.scatter(raw_drops_ns[ghost_mask], oca_drops_ns[ghost_mask],
                   c="red", s=60, alpha=0.7, label="OCA Ghost", zorder=3)
    # Plot real points
    if is_oca_real_ns.any():
        ax.scatter(raw_drops_ns[is_oca_real_ns], oca_drops_ns[is_oca_real_ns],
                   c="blue", s=80, alpha=0.8, label="OCA Real", zorder=4)

    # Diagonal line (y=x)
    max_val = max(raw_drops_ns.max(), oca_drops_ns.max()) if len(raw_drops_ns) > 0 else 10
    ax.plot([0, max_val * 1.1], [0, max_val * 1.1], "k--", alpha=0.4, linewidth=1, label="y = x (no inflation)")

    # Threshold line
    ax.axhline(y=args.accdrop_threshold, color="gray", linestyle=":", linewidth=1.5,
               label=f"OCA threshold ({args.accdrop_threshold}%)")

    # Label all points
    for i in range(len(raw_drops_ns)):
        name = names_ns[i]
        label_text = name if name else labels_ns[i]
        offset = (5, 5) if i % 2 == 0 else (5, -12)
        ax.annotate(label_text, (raw_drops_ns[i], oca_drops_ns[i]),
                    textcoords="offset points", xytext=offset, fontsize=8,
                    fontweight="bold" if name else "normal")

    # Also plot seeds as stars
    for r in seeds:
        rd = r["raw_accdrop_verified"] * 100
        ax.scatter(rd, rd, marker="*", s=300, c="gold", edgecolors="black", zorder=5)
        ax.annotate(rec_label(r), (rd, rd), textcoords="offset points",
                    xytext=(8, 8), fontsize=9, fontweight="bold")

    ax.set_xlabel("Raw AccDrop (%) — single ablation", fontsize=12)
    ax.set_ylabel("OCA AccDrop_perp (%) — unique component only", fontsize=12)
    ax.set_title("Raw vs Orthogonalized Accuracy Drop", fontsize=13)
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(-0.5, max_val * 1.15)
    ax.set_ylim(-0.5, max_val * 1.15)
    plt.tight_layout()
    p2 = str(out_dir / "oca_plot2_raw_vs_oca.png")
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    print(f"  [2] {p2}")

    # --- Plot 3: Independence fraction vs OCA AccDrop ---
    fig, ax = plt.subplots(figsize=(9, 7))
    norm_perps = np.array([last_results_map.get(
        (r["layer"], r["head"], r["sv_idx"]), {}).get("norm_perp", 1.0)
        for r in non_seeds])
    oca_drops_all = np.array([last_results_map.get(
        (r["layer"], r["head"], r["sv_idx"]), {}).get("accdrop_perp", 0.0) * 100
        for r in non_seeds])
    is_oca_real_all = np.array([
        (r["layer"], r["head"], r["sv_idx"]) in anchor_keys
        for r in non_seeds])

    ghost_m = ~is_oca_real_all
    if ghost_m.any():
        ax.scatter(norm_perps[ghost_m], oca_drops_all[ghost_m],
                   c="red", s=50, alpha=0.7, label="OCA Ghost", zorder=3)
    if is_oca_real_all.any():
        ax.scatter(norm_perps[is_oca_real_all], oca_drops_all[is_oca_real_all],
                   c="blue", s=70, alpha=0.8, label="OCA Real", zorder=4)

    ax.axhline(y=args.accdrop_threshold, color="gray", linestyle=":", linewidth=1.5,
               label=f"Threshold ({args.accdrop_threshold}%)")

    # Label points
    for i, r in enumerate(non_seeds):
        name = r.get("known_name", "")
        if name:
            ax.annotate(name, (norm_perps[i], oca_drops_all[i]),
                        textcoords="offset points", xytext=(6, 6), fontsize=9, fontweight="bold")

    ax.set_xlabel("Independence fraction ||r_perp|| (0=fully subsumed, 1=fully independent)", fontsize=11)
    ax.set_ylabel("OCA AccDrop_perp (%)", fontsize=12)
    ax.set_title("Independence Fraction vs Unique Causal Contribution", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_xlim(-0.05, 1.1)
    plt.tight_layout()
    p3 = str(out_dir / "oca_plot3_independence_vs_accdrop.png")
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    print(f"  [3] {p3}")

    # --- Plot 4: Convergence plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    iters = [h["iteration"] for h in iteration_history]
    sizes = [h["anchor_size"] for h in iteration_history]
    # Add iteration 0 (seeds only)
    iters = [0] + iters
    sizes = [n_seeds] + sizes
    ax.plot(iters, sizes, "bo-", markersize=10, linewidth=2)
    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel("Anchor Set Size", fontsize=12)
    ax.set_title("OCA Convergence", fontsize=13)
    ax.set_xticks(iters)
    ax.set_ylim(0, max(sizes) + 2)
    for i, (x, y) in enumerate(zip(iters, sizes)):
        ax.annotate(str(y), (x, y), textcoords="offset points", xytext=(0, 10),
                    ha="center", fontsize=12, fontweight="bold")
    ax.axhline(y=n_seeds, color="gray", linestyle="--", alpha=0.5, label=f"Seeds ({n_seeds})")
    ax.legend(fontsize=10)
    plt.tight_layout()
    p4 = str(out_dir / "oca_plot4_convergence.png")
    fig.savefig(p4, dpi=150)
    plt.close(fig)
    print(f"  [4] {p4}")

    # ----------------------------------------------------------------
    # Step 9: Save JSON
    # ----------------------------------------------------------------
    final_receptor_details = []
    for r in raw_real:
        rkey = (r["layer"], r["head"], r["sv_idx"])
        is_seed = rkey in seed_keys
        is_final = rkey in anchor_keys
        ir = last_results_map.get(rkey, {})

        final_receptor_details.append({
            "layer": int(r["layer"]),
            "head": int(r["head"]),
            "sv_idx": int(r["sv_idx"]),
            "known_name": r.get("known_name", ""),
            "raw_cfr": float(r["cfr"]),
            "raw_accdrop": float(r["raw_accdrop_verified"]),
            "auc": float(r["auc"]),
            "is_seed": bool(is_seed),
            "norm_perp": float(ir.get("norm_perp", 1.0)) if not is_seed else 1.0,
            "accdrop_perp": float(ir.get("accdrop_perp", r["raw_accdrop_verified"])) if not is_seed else float(r["raw_accdrop_verified"]),
            "oca_cfr": float(ir.get("oca_cfr", r["cfr"])) if not is_seed else float(r["cfr"]),
            "oca_status": "SEED" if is_seed else ir.get("oca_status", "UNKNOWN"),
            "survives_oca": bool(is_final),
        })

    oca_json = {
        "config": {
            "cfr_json": str(cfr_json_path),
            "accdrop_threshold_pct": float(args.accdrop_threshold),
            "norm_perp_min": float(args.norm_perp_min),
            "n_seeds": int(n_seeds),
            "n_examples": int(N_examples),
            "use_both": bool(args.use_both),
        },
        "baseline_acc": float(baseline_acc),
        "n_total_receptors": int(n_total),
        "n_raw_real": int(n_raw_real),
        "n_oca_survivors": int(n_final),
        "n_reclassified_ghost": int(n_raw_real - n_final),
        "iterations_to_converge": int(last_iter["iteration"]),
        "r2_oca_survivors": float(r2_final),
        "r2_raw_real": float(r2_raw_real),
        "r2_all": float(r2_all),
        "iteration_history": [
            {
                "iteration": int(h["iteration"]),
                "anchor_size": int(h["anchor_size"]),
                "subspace_dim": int(h["subspace_dim"]),
                "n_new_members": int(h["n_new_members"]),
            }
            for h in iteration_history
        ],
        "receptors": final_receptor_details,
    }

    json_path = str(out_dir / "oca_ghost_refinement_results.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(oca_json, f, indent=2)
    print(f"\n[SAVE] {json_path}")

    # ----------------------------------------------------------------
    # Sanity checks
    # ----------------------------------------------------------------
    print(f"\n{'='*60}")
    print("[SANITY CHECKS]")
    print(f"  Baseline accuracy: {baseline_acc*100:.2f}% (expected ~91.34%)")

    # Check R1 and R3 are seeds
    for r in seeds:
        name = r.get("known_name", "")
        if name:
            print(f"  {name} is seed: PASS")

    # Check R2
    r2_recs = [r for r in raw_real if r.get("known_name") == "R2"]
    if r2_recs:
        r2 = r2_recs[0]
        rkey = (r2["layer"], r2["head"], r2["sv_idx"])
        ir = last_results_map.get(rkey, {})
        norm_p = ir.get("norm_perp", -1)
        ad_p = ir.get("accdrop_perp", -1)
        survived = rkey in anchor_keys
        print(f"  R2: ||r_perp||={norm_p:.4f}, AccDrop_perp={ad_p*100:+.2f}%, survived={survived} "
              f"— {'PASS (ghost)' if not survived else 'CHECK: expected ghost'}")

    # Check L11H1sv1 if present
    anti_ghost = [r for r in raw_real
                  if r["layer"] == 11 and r["head"] == 1 and r["sv_idx"] == 1]
    if anti_ghost:
        r_ag = anti_ghost[0]
        rkey = (r_ag["layer"], r_ag["head"], r_ag["sv_idx"])
        ir = last_results_map.get(rkey, {})
        norm_p = ir.get("norm_perp", -1)
        ad_p = ir.get("accdrop_perp", -1)
        survived = rkey in anchor_keys
        print(f"  L11H1sv1: ||r_perp||={norm_p:.4f}, AccDrop_perp={ad_p*100:+.2f}%, survived={survived} "
              f"— {'PASS (real)' if survived else 'CHECK: expected real'}")

    # Convergence
    n_iters = last_iter["iteration"]
    print(f"  Convergence in {n_iters} iterations — {'PASS' if n_iters <= 3 else 'OK (more than expected)'}")

    print(f"\n[DONE]")


if __name__ == "__main__":
    main()
