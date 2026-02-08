#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gp_exp4_receptor_shapley_ablation.py

Experiment 4 (GP): Receptor ablation via residual-stream projection + Shapley values.

Each receptor is defined by the OV right-singular vector Vh[sv_idx] from svd_cache.pt.
This is the WRITE direction of that head's component: the head writes σ_k * a_k * Vh[k,:]
into the residual stream.

"Ablate receptor k" = project out Vh[k,:] from the FINAL residual stream at the decision
position, then recompute logits via ln_final + W_U.  This removes ALL signal along that
direction regardless of source (host head + any other component that writes there).

For K=3 receptors we sweep all 2^3=8 coalitions and compute:
  - accuracy / margin for each coalition
  - necessity, sufficiency
  - Shapley values (should sum to v(R123) - v(NONE))
  - pairwise interaction indices

Run:
  python gp_exp4_receptor_shapley_ablation.py \
    --data_dir data_main --csv test_gp.csv --out_dir outputs/gp \
    --receptors "10,9,0,+1;11,8,6,+1;9,7,1,-1" \
    --batch_size 64 --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

import numpy as np
import torch
from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer

try:
    import train_gp_masks_and_dump_ov_logit_receptors_ddp as gp
except Exception:
    import train_gp_masks_and_dump_ov_logit_receptors as gp


# ---------------------------------------------------------------------------
# Dataclass & parsing
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReceptorSpec:
    layer: int
    head: int
    sv_idx: int
    polarity: int  # +1 male-on-top, -1 female-on-top


def parse_receptors(spec: str) -> List[ReceptorSpec]:
    out: List[ReceptorSpec] = []
    for part in spec.strip().split(";"):
        part = part.strip()
        if not part:
            continue
        fields = [x.strip() for x in part.split(",")]
        if len(fields) != 4:
            raise ValueError(f"Bad receptor spec: {part!r}")
        L, H, K, pol = int(fields[0]), int(fields[1]), int(fields[2]), int(fields[3])
        if pol not in (-1, +1):
            raise ValueError(f"Polarity must be ±1, got {pol}")
        out.append(ReceptorSpec(layer=L, head=H, sv_idx=K, polarity=pol))
    if len(out) != 3:
        raise ValueError(f"Expected 3 receptors, got {len(out)}")
    return out


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def expand_rows(rows: List[Dict[str, str]], use_both: bool) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        p = (r.get("pronoun") or "").strip().lower()
        cp = (r.get("corr_pronoun") or "").strip().lower()
        if p in ("he", "she"):
            out.append({"text": r["prefix"], "label": p})
        if use_both and cp in ("he", "she"):
            out.append({"text": r["corr_prefix"], "label": cp})
    return out


def tokenize(tokenizer, texts: List[str], device: str):
    ids_list = gp._encode_texts(tokenizer, texts)
    pad_len = max(len(x) for x in ids_list)
    pad_id = tokenizer.pad_token_id
    tokens, _attn, last_idx = gp._pad_to_length(ids_list, pad_len, pad_id, device)
    return tokens, last_idx


def coalition_label(keep: Set[int]) -> str:
    if not keep:
        return "NONE"
    return "R" + "".join(str(i + 1) for i in sorted(keep))


# ---------------------------------------------------------------------------
# Core: get final residual, then ablate + recompute logits
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_final_residuals(
    model: HookedTransformer,
    tokens_all: torch.Tensor,
    last_idx_all: torch.Tensor,
    batch_size: int,
) -> torch.Tensor:
    """Run the model and collect the final residual stream at the decision position.
    Returns: (N, d_model)"""
    N = tokens_all.shape[0]
    device = tokens_all.device
    d_model = model.cfg.d_model
    last_layer = model.cfg.n_layers - 1
    hook_name = f"blocks.{last_layer}.hook_resid_post"

    resids = []
    for i in range(0, N, batch_size):
        j = min(i + batch_size, N)
        toks = tokens_all[i:j]
        lidx = last_idx_all[i:j]
        B = toks.shape[0]

        _, cache = model.run_with_cache(toks, names_filter=[hook_name])
        resid = cache[hook_name]  # (B, S, d_model)

        ar = torch.arange(B, device=device)
        resids.append(resid[ar, lidx, :].clone())  # (B, d_model)

    return torch.cat(resids, dim=0)  # (N, d_model)


@torch.no_grad()
def logits_from_residual(
    model: HookedTransformer,
    resid: torch.Tensor,  # (N, d_model)
    he_id: int,
    she_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Apply ln_final + unembed, return (logit_he, logit_she) each (N,)."""
    normed = model.ln_final(resid)           # (N, d_model)
    logits = normed @ model.W_U + model.b_U  # (N, vocab)
    return logits[:, he_id], logits[:, she_id]


def project_out(resid: torch.Tensor, directions: List[torch.Tensor]) -> torch.Tensor:
    """Project out a set of directions from residual vectors.
    Uses sequential projection (Gram-Schmidt-like) for robustness when
    directions from different heads are non-orthogonal.
    
    resid: (N, d_model)
    directions: list of (d_model,) unit vectors
    """
    if not directions:
        return resid
    # Stack and orthonormalize via QR
    D = torch.stack(directions, dim=0)  # (m, d_model)
    Q, _ = torch.linalg.qr(D.T, mode="reduced")  # (d_model, m) orthonormal cols
    # Project out the subspace
    proj = (resid @ Q) @ Q.T  # (N, d_model)
    return resid - proj


def eval_coalition(
    model: HookedTransformer,
    resid_final: torch.Tensor,  # (N, d_model) — clean residual
    y: torch.Tensor,            # (N,) in {+1, -1}
    he_id: int,
    she_id: int,
    directions_to_ablate: List[torch.Tensor],
) -> Dict[str, Any]:
    """Evaluate accuracy/margin after projecting out given directions."""
    resid = project_out(resid_final, directions_to_ablate)
    he_logit, she_logit = logits_from_residual(model, resid, he_id, she_id)
    diff = he_logit - she_logit                         # (N,)
    pred = torch.where(diff > 0, 1, -1).to(y.device)
    margin = (y.float() * diff.float()).cpu().numpy()

    acc = float((pred == y).float().mean().item())
    return {
        "accuracy": acc,
        "mean_margin": float(margin.mean()),
        "median_margin": float(np.median(margin)),
        "frac_positive_margin": float((margin > 0).mean()),
    }


# ---------------------------------------------------------------------------
# Shapley & interaction
# ---------------------------------------------------------------------------

def shapley_values(v_map: Dict[frozenset, float], K: int) -> List[float]:
    fact = math.factorial
    N_set = set(range(K))
    phis = []
    for k in range(K):
        total = 0.0
        for s_size in range(K):
            for S in combinations(N_set - {k}, s_size):
                S = frozenset(S)
                Sk = S | {k}
                w = fact(len(S)) * fact(K - len(S) - 1) / fact(K)
                total += w * (v_map[Sk] - v_map[S])
        phis.append(total)
    return phis


def interaction_pair(v_map: Dict[frozenset, float], i: int, j: int) -> float:
    return 0.5 * (
        v_map[frozenset({i, j})] - v_map[frozenset({i})]
        - v_map[frozenset({j})] + v_map[frozenset()]
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data_main")
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="outputs/gp")
    ap.add_argument("--receptors", type=str, required=True)
    ap.add_argument("--use_both", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_json", type=int, default=1)
    ap.add_argument("--save_name", type=str, default="exp4_shapley_results.json")
    args = ap.parse_args()

    set_seed(args.seed)
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda not available, using cpu.")
        device = "cpu"

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    he_ids = tokenizer.encode(" he", add_special_tokens=False)
    she_ids = tokenizer.encode(" she", add_special_tokens=False)
    assert len(he_ids) == 1 and len(she_ids) == 1
    he_id, she_id = he_ids[0], she_ids[0]
    print(f"[TOKENS] he: {he_id} -> {tokenizer.decode([he_id])!r}")
    print(f"[TOKENS] she: {she_id} -> {tokenizer.decode([she_id])!r}")

    # Data
    rows = gp.load_gp_csv(str(data_dir / args.csv))
    examples = expand_rows(rows, use_both=bool(args.use_both))
    if args.max_examples > 0:
        examples = examples[:args.max_examples]
    assert len(examples) > 0

    y_list = [+1 if e["label"] == "he" else -1 for e in examples]
    y = torch.tensor(y_list, dtype=torch.long, device=device)
    print(f"label counts: {{+1: {(y==1).sum().item()}, -1: {(y==-1).sum().item()}}}")

    tokens_all, last_idx_all = tokenize(tokenizer, [e["text"] for e in examples], device)
    N, S = tokens_all.shape
    print(f"\n[RUN] N={N} | seq_len={S} | batch={args.batch_size} | device={device}")

    # Model
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load SVD cache
    svd_path = out_dir / "svd_cache.pt"
    assert svd_path.exists(), f"Missing {svd_path}"
    _qk, ov, _mlp_in, _mlp_out, _rank_total_ov = gp.load_svd_cache(str(svd_path), device=device)

    # Parse receptors and extract Vh directions
    rec_specs = parse_receptors(args.receptors)
    rec_dirs: List[torch.Tensor] = []

    print(f"\n[RECEPTORS] OV right-singular vectors Vh[sv_idx] (write direction, d_model).")
    for idx, rs in enumerate(rec_specs):
        svd = ov[rs.layer][rs.head]
        assert 0 <= rs.sv_idx < svd.r, f"R{idx+1}: sv_idx={rs.sv_idx} >= rank={svd.r}"
        v = svd.Vh[rs.sv_idx, :].clone().to(device)
        v = v / (v.norm() + 1e-12)
        rec_dirs.append(v)
        print(f"  R{idx+1}: L{rs.layer}H{rs.head} sv={rs.sv_idx} pol={rs.polarity:+d}  ||Vh||={float(svd.Vh[rs.sv_idx].norm()):.4f}")

    # Cosine similarities between receptor directions (non-orthogonal across heads)
    print("\n[COSINES] (different heads → not necessarily orthogonal)")
    for i in range(3):
        for j in range(i + 1, 3):
            c = float((rec_dirs[i] @ rec_dirs[j]).item())
            print(f"  cos(R{i+1}, R{j+1}) = {c:+.4f}")

    # -----------------------------------------------------------------------
    # Step 1: collect final residual (one forward pass, cached)
    # -----------------------------------------------------------------------
    print("\n[CACHE] collecting final residual stream at decision positions ...")
    resid_final = collect_final_residuals(model, tokens_all, last_idx_all, args.batch_size)
    print(f"  resid shape: {tuple(resid_final.shape)}, norm: {resid_final.norm(dim=-1).mean():.2f}")

    # Sanity: verify baseline accuracy from residual matches direct model output
    baseline = eval_coalition(model, resid_final, y, he_id, she_id, [])
    print(f"  baseline acc (from cached resid): {baseline['accuracy']:.4f}")

    # Diagnostic: how much of the residual lies along each receptor direction?
    print("\n[DIAG] projection magnitudes (mean |resid · r|) per receptor:")
    for idx, r in enumerate(rec_dirs):
        proj_mag = (resid_final @ r).abs().mean().item()
        resid_norm = resid_final.norm(dim=-1).mean().item()
        print(f"  R{idx+1}: mean|proj| = {proj_mag:.4f}  (resid_norm = {resid_norm:.2f}, ratio = {proj_mag/resid_norm:.4f})")

    # -----------------------------------------------------------------------
    # Step 2: sweep all 2^3 coalitions
    # -----------------------------------------------------------------------
    K = 3
    full_set = set(range(K))
    all_keeps: List[Set[int]] = []
    for size in range(K + 1):
        for combo in combinations(range(K), size):
            all_keeps.append(set(combo))

    results: Dict[str, Any] = {}
    results["BASELINE"] = {**baseline, "keep_set": [0, 1, 2], "ablate_set": []}

    print("\n[SWEEP] evaluating 8 coalitions ...")
    for keep in all_keeps:
        ablate = sorted(full_set - keep)
        label = coalition_label(keep)
        dirs_to_remove = [rec_dirs[i] for i in ablate]

        out = eval_coalition(model, resid_final, y, he_id, she_id, dirs_to_remove)
        results[label] = {**out, "keep_set": sorted(keep), "ablate_set": ablate}
        print(f"  {label:<5}  ablate={str(ablate):<12}  acc={out['accuracy']:.4f}  margin={out['mean_margin']:+.4f}")

    # -----------------------------------------------------------------------
    # Step 3: compute Shapley values, necessity, sufficiency, interactions
    # -----------------------------------------------------------------------
    v_map: Dict[frozenset, float] = {}
    for keep in all_keeps:
        v_map[frozenset(keep)] = results[coalition_label(keep)]["accuracy"]

    v_none = v_map[frozenset()]
    v_full = v_map[frozenset({0, 1, 2})]
    phis = shapley_values(v_map, K)
    I12 = interaction_pair(v_map, 0, 1)
    I13 = interaction_pair(v_map, 0, 2)
    I23 = interaction_pair(v_map, 1, 2)

    necessity = {
        f"remove_R{k+1}": v_full - v_map[frozenset(full_set - {k})]
        for k in range(K)
    }
    sufficiency = {
        f"only_R{k+1}_gain": v_map[frozenset({k})] - v_none
        for k in range(K)
    }

    # -----------------------------------------------------------------------
    # Print tables
    # -----------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: PER-RECEPTOR ABLATION + SHAPLEY VALUES (he vs she)")
    print("=" * 80)

    print("\n[TABLE 1] All coalitions (keep set) + BASELINE")
    hdr = f"{'Keep':<8} {'Ablated':<12} {'Acc':>8} {'MeanMar':>10} {'MedMar':>10} {'%PosMar':>8}"
    print(hdr)
    print("-" * len(hdr))
    for label in ["NONE", "R1", "R2", "R3", "R12", "R13", "R23", "R123", "BASELINE"]:
        r = results[label]
        print(f"{label:<8} {str(r['ablate_set']):<12} {r['accuracy']:>8.4f} "
              f"{r['mean_margin']:>+10.4f} {r['median_margin']:>+10.4f} {r['frac_positive_margin']:>8.4f}")

    r123 = results["R123"]["accuracy"]
    rbase = results["BASELINE"]["accuracy"]
    print(f"\n[SANITY] R123 acc={r123:.4f}  BASELINE acc={rbase:.4f}  diff={abs(r123-rbase):.6f}")

    print("\n[TABLE 2] Necessity (accuracy drop when removing each from full set)")
    for k in range(K):
        kept = frozenset(full_set - {k})
        print(f"  Remove R{k+1}: {v_full:.4f} -> {v_map[kept]:.4f}  (drop={necessity[f'remove_R{k+1}']:+.4f})")

    print("\n[TABLE 3] Sufficiency (accuracy with only one receptor)")
    for k in range(K):
        print(f"  Only R{k+1}: {v_map[frozenset({k})]:.4f}  (gain over NONE: {sufficiency[f'only_R{k+1}_gain']:+.4f})")

    print("\n[TABLE 4] Shapley values")
    for k in range(K):
        print(f"  phi(R{k+1}) = {phis[k]:+.6f}")
    print(f"  sum = {sum(phis):+.6f}  vs  v(R123)-v(NONE) = {v_full - v_none:+.6f}")

    print("\n[TABLE 5] Interaction indices (>0 synergy, <0 redundancy)")
    print(f"  I(R1,R2) = {I12:+.6f}")
    print(f"  I(R1,R3) = {I13:+.6f}")
    print(f"  I(R2,R3) = {I23:+.6f}")

    print("\n[TABLE 6] Redundancy check")
    rhs = sum(v_map[frozenset({k})] for k in range(K)) - 2 * v_none
    print(f"  v(R123) = {v_full:.4f}")
    print(f"  sum_k v({{k}}) - 2*v(NONE) = {rhs:.4f}")
    print(f"  diff = {v_full - rhs:+.4f}  (>0: synergy, <0: redundancy)")

    # -----------------------------------------------------------------------
    # Also compute Shapley on mean_margin (more sensitive than accuracy)
    # -----------------------------------------------------------------------
    m_map: Dict[frozenset, float] = {}
    for keep in all_keeps:
        m_map[frozenset(keep)] = results[coalition_label(keep)]["mean_margin"]
    m_phis = shapley_values(m_map, K)
    mI12 = interaction_pair(m_map, 0, 1)
    mI13 = interaction_pair(m_map, 0, 2)
    mI23 = interaction_pair(m_map, 1, 2)

    print("\n[TABLE 7] Shapley on MEAN MARGIN (more sensitive)")
    for k in range(K):
        print(f"  phi_margin(R{k+1}) = {m_phis[k]:+.6f}")
    print(f"  sum = {sum(m_phis):+.6f}  vs  m(R123)-m(NONE) = {m_map[frozenset({0,1,2})] - m_map[frozenset()]:+.6f}")
    print(f"\n  I_margin(R1,R2) = {mI12:+.6f}")
    print(f"  I_margin(R1,R3) = {mI13:+.6f}")
    print(f"  I_margin(R2,R3) = {mI23:+.6f}")

    # -----------------------------------------------------------------------
    # Save
    # -----------------------------------------------------------------------
    if args.save_json:
        payload = {
            "meta": {"csv": args.csv, "N": N, "batch_size": args.batch_size,
                     "device": device, "seed": args.seed},
            "tokens": {"he_id": he_id, "she_id": she_id},
            "receptors": [
                {"idx": i + 1, "layer": rs.layer, "head": rs.head,
                 "sv_idx": rs.sv_idx, "polarity": rs.polarity}
                for i, rs in enumerate(rec_specs)
            ],
            "cosines": {
                f"R{i+1}_R{j+1}": float((rec_dirs[i] @ rec_dirs[j]).item())
                for i in range(3) for j in range(i + 1, 3)
            },
            "results": results,
            "shapley_accuracy": {f"phi_R{k+1}": float(phis[k]) for k in range(K)},
            "shapley_margin": {f"phi_R{k+1}": float(m_phis[k]) for k in range(K)},
            "interaction_accuracy": {"I_R1_R2": I12, "I_R1_R3": I13, "I_R2_R3": I23},
            "interaction_margin": {"I_R1_R2": mI12, "I_R1_R3": mI13, "I_R2_R3": mI23},
            "necessity": necessity,
            "sufficiency": sufficiency,
            "sanity": {"r123_acc": r123, "baseline_acc": rbase, "diff": abs(r123 - rbase)},
        }
        out_path = out_dir / args.save_name
        with out_path.open("w") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[SAVED] {out_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
