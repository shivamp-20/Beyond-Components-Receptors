#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gp_exp4_receptor_shapley_ablation.py

Experiment 4 (GP): Receptor ablation at host head output + coalition sweep (2^3) + Shapley values.

You provide 3 receptors as (layer, head, sv_idx, polarity), where the receptor direction is the OV
right-singular vector Vh[sv_idx] from outputs/gp/svd_cache.pt.

"Ablate receptor k" = at its host head output (per-head result after W_O, shape d_model), project out
the receptor direction at the *decision position only*:
    out <- out - (out · r_hat) r_hat
If multiple receptors live in the same (layer,head), we remove the span of their directions via QR.

Metrics are computed on the GP dataset using the clean binary he-vs-she decision:
    pred = +1 if logit(" he") > logit(" she") else -1
    y_true = +1 for he-label, -1 for she-label
    margin = y_true * (logit_he - logit_she)

Outputs:
- Prints a table for all coalitions + BASELINE.
- Prints necessity/sufficiency/Shapley/interaction indices.
- Saves JSON: <out_dir>/exp4_shapley_results.json

Run (Kaggle):
!python gp_exp4_receptor_shapley_ablation.py \
  --data_dir data_main \
  --csv test_gp.csv \
  --out_dir outputs/gp \
  --receptors "10,9,0,+1;11,8,6,+1;9,7,1,-1" \
  --batch_size 64 \
  --device cuda
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Dict, List, Sequence, Set, Tuple

import numpy as np
import torch
from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer


# -----------------------------------------------------------------------------
# Import your training helpers (ddp or non-ddp)
# -----------------------------------------------------------------------------
try:
    import train_gp_masks_and_dump_ov_logit_receptors_ddp as gp  # type: ignore
except Exception:
    import train_gp_masks_and_dump_ov_logit_receptors as gp  # type: ignore


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batches(n: int, batch_size: int) -> List[Tuple[int, int]]:
    return [(i, min(i + batch_size, n)) for i in range(0, n, batch_size)]


def expand_rows_to_examples(rows: List[Dict[str, str]], use_both: bool) -> List[Dict[str, str]]:
    """
    Each CSV row can yield:
      - clean example:   (prefix, pronoun)
      - corrupt example: (corr_prefix, corr_pronoun)
    """
    out: List[Dict[str, str]] = []
    for r in rows:
        p = (r.get("pronoun") or "").strip().lower()
        cp = (r.get("corr_pronoun") or "").strip().lower()
        if p in ("he", "she"):
            out.append({"text": r["prefix"], "label": p})
        if use_both and (cp in ("he", "she")):
            out.append({"text": r["corr_prefix"], "label": cp})
    return out


def tokenize_texts(
    tokenizer: GPT2TokenizerFast, texts: List[str], device: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Use your exact low-level tokenizer + padding helpers.
    Returns:
      tokens:  (B,S)
      last_idx:(B,) index of last non-pad token
    """
    ids_list = gp._encode_texts(tokenizer, texts)
    pad_len = max(len(x) for x in ids_list)
    pad_id = tokenizer.pad_token_id
    tokens, _attn, last_idx = gp._pad_to_length(ids_list, pad_len, pad_id, device)
    return tokens, last_idx


def coalition_label(keep_set: Set[int]) -> str:
    if len(keep_set) == 0:
        return "NONE"
    return "R" + "".join(str(i + 1) for i in sorted(keep_set))


def ensure_hookpoint_exists(model: HookedTransformer, name: str) -> None:
    if not hasattr(model, "hook_dict") or name not in model.hook_dict:
        candidates = []
        if hasattr(model, "hook_dict"):
            for k in model.hook_dict.keys():
                if "hook_result" in k and ".attn." in k:
                    candidates.append(k)
        msg = f"Missing hookpoint: {name}."
        if candidates:
            msg += f" Found hook_result candidates (first 8): {candidates[:8]}"
        raise RuntimeError(msg)


@dataclass(frozen=True)
class ReceptorSpec:
    layer: int
    head: int
    sv_idx: int
    polarity: int  # +1 if male tokens on top, -1 if female tokens on top


def parse_receptors(spec: str) -> List[ReceptorSpec]:
    """
    Format:
      "L,H,sv,+1;L,H,sv,-1;L,H,sv,+1"
    """
    out: List[ReceptorSpec] = []
    spec = (spec or "").strip()
    if not spec:
        raise ValueError("--receptors is required.")
    for part in spec.split(";"):
        part = part.strip()
        if not part:
            continue
        fields = [x.strip() for x in part.split(",")]
        if len(fields) != 4:
            raise ValueError(f"Bad receptor spec chunk: {part!r}. Expected 'L,H,sv,polarity'.")
        L, H, K = int(fields[0]), int(fields[1]), int(fields[2])
        pol = int(fields[3])
        if pol not in (-1, +1):
            raise ValueError(f"Polarity must be +1 or -1, got {pol} in {part!r}.")
        out.append(ReceptorSpec(layer=L, head=H, sv_idx=K, polarity=pol))
    if len(out) != 3:
        raise ValueError(f"Expected exactly 3 receptors for Exp4, got {len(out)}: {out}")
    return out


def orthonormal_basis_from_rows(rows: torch.Tensor) -> torch.Tensor:
    """
    rows: (m, d_model) stacked receptor directions
    Returns Q: (d_model, r) with orthonormal columns spanning rows' span.
    """
    Q, _ = torch.linalg.qr(rows.T, mode="reduced")
    return Q


@torch.no_grad()
def eval_condition(
    model: HookedTransformer,
    tokens_all: torch.Tensor,          # (N,S)
    last_idx_all: torch.Tensor,        # (N,)
    y_all: torch.Tensor,               # (N,) in {+1,-1}
    he_id: int,
    she_id: int,
    ablate_specs: List[Tuple[int, Dict[int, torch.Tensor]]],  # per layer: {head -> Q (d_model,r)}
    batch_size: int,
) -> Dict[str, Any]:
    """
    Evaluate a single ablation condition (ablate set fixed).
    ablate_specs is a list over layers that need hooking:
      [(layer, {head: Q_basis}), ...]
    where Q_basis has orthonormal columns spanning the receptor directions to remove.
    """

    N, _S = tokens_all.shape
    device = tokens_all.device
    arange_cache: torch.Tensor | None = None

    # Pre-build hook list (one hook per layer where we ablate something)
    fwd_hooks = []
    state: Dict[str, Any] = {}

    for layer, head_to_Q in ablate_specs:
        hook_name = f"blocks.{layer}.attn.hook_result"
        ensure_hookpoint_exists(model, hook_name)

        def make_hook(head_to_Q_local: Dict[int, torch.Tensor]):
            def hook_fn(result: torch.Tensor, hook) -> torch.Tensor:
                # result: (B,S,H,d_model)
                last_idx = state["last_idx"]  # (B,)
                B = result.shape[0]
                nonlocal arange_cache
                if arange_cache is None or arange_cache.numel() != B:
                    arange_cache = torch.arange(B, device=result.device)

                out = result.clone()
                for h, Q in head_to_Q_local.items():
                    vec = out[arange_cache, last_idx, h, :]  # (B,D)
                    vecQ = vec @ Q           # (B,r)
                    proj = vecQ @ Q.T        # (B,D)
                    out[arange_cache, last_idx, h, :] = vec - proj
                return out
            return hook_fn

        fwd_hooks.append((hook_name, make_hook(head_to_Q)))

    margins: List[float] = []
    correct = 0
    total = 0

    for i0, i1 in batches(N, batch_size):
        tokens = tokens_all[i0:i1]
        last_idx = last_idx_all[i0:i1]
        y = y_all[i0:i1]
        state["last_idx"] = last_idx

        if len(fwd_hooks) == 0:
            logits = model(tokens)  # (B,S,V)
        else:
            logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks)  # (B,S,V)

        B = tokens.shape[0]
        ar = torch.arange(B, device=device)

        he_logit = logits[ar, last_idx, he_id]
        she_logit = logits[ar, last_idx, she_id]
        diff = he_logit - she_logit

        pred = torch.where(diff > 0, torch.ones_like(y), -torch.ones_like(y))
        correct += int((pred == y).sum().item())
        total += int(B)

        margin = (y * diff).detach().float().cpu().numpy().tolist()
        margins.extend(margin)

    margins_np = np.asarray(margins, dtype=np.float64)
    acc = correct / max(1, total)
    return {
        "n": int(total),
        "accuracy": float(acc),
        "mean_margin": float(margins_np.mean()) if margins_np.size else float("nan"),
        "median_margin": float(np.median(margins_np)) if margins_np.size else float("nan"),
        "frac_positive_margin": float((margins_np > 0).mean()) if margins_np.size else float("nan"),
    }


def shapley_values(v_map: Dict[frozenset, float], K: int) -> List[float]:
    """Generic Shapley formula."""
    fact = math.factorial
    denom = fact(K)
    Nset = set(range(K))
    phis = []
    for k in range(K):
        total = 0.0
        for s_size in range(K):
            for S in combinations(Nset - {k}, s_size):
                S = frozenset(S)
                Sk = frozenset(set(S) | {k})
                w = fact(len(S)) * fact(K - len(S) - 1) / denom
                total += w * (v_map[Sk] - v_map[S])
        phis.append(total)
    return phis


def interaction_pair(v_map: Dict[frozenset, float], i: int, j: int) -> float:
    empty = frozenset()
    return 0.5 * (v_map[frozenset({i, j})] - v_map[frozenset({i})] - v_map[frozenset({j})] + v_map[empty])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data_main")
    ap.add_argument("--csv", type=str, required=True, help="GP csv (e.g., test_gp.csv)")
    ap.add_argument("--out_dir", type=str, default="outputs/gp")
    ap.add_argument("--receptors", type=str, required=True, help='e.g. "10,9,0,+1;11,8,6,+1;9,7,1,-1"')

    ap.add_argument("--use_both", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_json", type=int, default=1)
    ap.add_argument("--save_name", type=str, default="exp4_shapley_results.json")

    args = ap.parse_args()
    set_seed(int(args.seed))

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda requested but not available; falling back to cpu.")
        device = "cpu"

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    he_ids = tokenizer.encode(" he", add_special_tokens=False)
    she_ids = tokenizer.encode(" she", add_special_tokens=False)
    if len(he_ids) != 1 or len(she_ids) != 1:
        raise ValueError(f'" he"/" she" not single tokens: he_ids={he_ids}, she_ids={she_ids}')
    he_id, she_id = int(he_ids[0]), int(she_ids[0])
    print("[TOKENS] he:", he_id, "->", repr(tokenizer.decode([he_id], clean_up_tokenization_spaces=False)))
    print("[TOKENS] she:", she_id, "->", repr(tokenizer.decode([she_id], clean_up_tokenization_spaces=False)))

    rows = gp.load_gp_csv(str(data_dir / args.csv))
    examples = expand_rows_to_examples(rows, use_both=bool(int(args.use_both)))
    if int(args.max_examples) > 0:
        examples = examples[: int(args.max_examples)]
    if len(examples) == 0:
        raise ValueError("No examples after expand.")

    y_list: List[int] = []
    for e in examples:
        lab = e["label"].strip().lower()
        if lab == "he":
            y_list.append(+1)
        elif lab == "she":
            y_list.append(-1)
        else:
            raise ValueError(f"Unexpected label: {lab!r}")
    y = torch.tensor(y_list, dtype=torch.int64, device=device)
    print("label counts:", {1: int((y==1).sum().item()), -1: int((y==-1).sum().item())})
    print("unique pronoun strings:", sorted(set(e["label"].strip().lower() for e in examples)))

    texts = [e["text"] for e in examples]
    tokens_all, last_idx_all = tokenize_texts(tokenizer, texts, device=device)
    N, S = tokens_all.shape
    print(f"\n[RUN] N={N} examples | seq_len={S} | batch_size={int(args.batch_size)} | device={device}")

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    svd_path = out_dir / "svd_cache.pt"
    if not svd_path.exists():
        raise FileNotFoundError(f"Missing {svd_path} (run mask training first)")
    _qk, ov, _mlp_in, _mlp_out, _rank_total_ov = gp.load_svd_cache(str(svd_path), device=device)

    rec_specs = parse_receptors(args.receptors)

    rec_dirs: List[torch.Tensor] = []
    print("\n[RECEPTORS] Using OV right singular vectors v_row = Vh[sv_idx] (d_model).")

    for idx, rs in enumerate(rec_specs, start=1):
        svd = ov[rs.layer][rs.head]
        if not (0 <= rs.sv_idx < svd.r):
            raise ValueError(f"R{idx}: sv_idx={rs.sv_idx} out of range for ov[{rs.layer}][{rs.head}].r={svd.r}")
        v_row = svd.Vh[rs.sv_idx, :].contiguous()
        norm = float(v_row.norm().detach().cpu().item())
        if norm == 0.0 or not math.isfinite(norm):
            raise ValueError(f"R{idx}: bad norm {norm}")
        v_hat = v_row / (v_row.norm() + 1e-12)
        rec_dirs.append(v_hat)
        print(f"  R{idx}: (layer={rs.layer}, head={rs.head}, sv_idx={rs.sv_idx}) polarity={rs.polarity:+d} | ||v||={norm:.6g}")

    print("\n[SANITY] receptor cosine similarities:")
    for i in range(3):
        for j in range(i+1, 3):
            c = float((rec_dirs[i] @ rec_dirs[j]).detach().cpu().item())
            print(f"  cos(R{i+1}, R{j+1}) = {c:+.4f}")

    K = 3
    full_set = set(range(K))
    all_keep_sets: List[Set[int]] = []
    for size in range(K + 1):
        for combo in combinations(range(K), size):
            all_keep_sets.append(set(combo))

    # BASELINE
    baseline = eval_condition(
        model=model,
        tokens_all=tokens_all,
        last_idx_all=last_idx_all,
        y_all=y,
        he_id=he_id,
        she_id=she_id,
        ablate_specs=[],
        batch_size=int(args.batch_size),
    )

    results: Dict[str, Any] = {}
    results["BASELINE"] = {**baseline, "keep_set": [0,1,2], "ablate_set": []}

    for keep in all_keep_sets:
        ablate = sorted(list(full_set - keep))
        label = coalition_label(keep)

        layer_to_head_rows: Dict[int, Dict[int, List[torch.Tensor]]] = {}
        for ridx in ablate:
            rs = rec_specs[ridx]
            layer_to_head_rows.setdefault(rs.layer, {}).setdefault(rs.head, []).append(rec_dirs[ridx])

        ablate_specs: List[Tuple[int, Dict[int, torch.Tensor]]] = []
        for L, head_map in sorted(layer_to_head_rows.items(), key=lambda x: x[0]):
            head_to_Q: Dict[int, torch.Tensor] = {}
            for H, vecs in head_map.items():
                rows_t = torch.stack(vecs, dim=0)
                Q = orthonormal_basis_from_rows(rows_t)
                head_to_Q[H] = Q
            ablate_specs.append((L, head_to_Q))

        out = eval_condition(
            model=model,
            tokens_all=tokens_all,
            last_idx_all=last_idx_all,
            y_all=y,
            he_id=he_id,
            she_id=she_id,
            ablate_specs=ablate_specs,
            batch_size=int(args.batch_size),
        )
        results[label] = {**out, "keep_set": sorted(list(keep)), "ablate_set": ablate}
        print(f"  done {label:<4}  acc={out['accuracy']:.3f}  mean_margin={out['mean_margin']:+.3f}")

    expected_labels = ["NONE","R1","R2","R3","R12","R13","R23","R123"]
    for lab in expected_labels:
        if lab not in results:
            raise RuntimeError(f"Missing coalition result: {lab}")

    v_map: Dict[frozenset, float] = {frozenset(keep): float(results[coalition_label(keep)]["accuracy"]) for keep in all_keep_sets}
    v_none = v_map[frozenset()]
    v_full = v_map[frozenset({0,1,2})]

    phis = shapley_values(v_map, K=3)

    I12 = interaction_pair(v_map, 0, 1)
    I13 = interaction_pair(v_map, 0, 2)
    I23 = interaction_pair(v_map, 1, 2)

    necessity = {
        "remove_R1": float(v_full - v_map[frozenset({1,2})]),
        "remove_R2": float(v_full - v_map[frozenset({0,2})]),
        "remove_R3": float(v_full - v_map[frozenset({0,1})]),
    }
    sufficiency = {
        "only_R1_gain": float(v_map[frozenset({0})] - v_none),
        "only_R2_gain": float(v_map[frozenset({1})] - v_none),
        "only_R3_gain": float(v_map[frozenset({2})] - v_none),
    }

    print("\n" + "="*80)
    print("EXPERIMENT 4: PER-RECEPTOR ABLATION + SHAPLEY VALUES (he vs she)")
    print("="*80)

    print("\n[TABLE 1] All coalitions (keep set) + BASELINE")
    hdr = f"{'Keep':<10} {'Ablated':<12} {'Acc':>7} {'MeanMar':>10} {'MedMar':>10} {'%PosMar':>9}"
    print(hdr)
    print("-"*len(hdr))
    order = ["NONE","R1","R2","R3","R12","R13","R23","R123"]
    for lab in order:
        r = results[lab]
        print(f"{lab:<10} {str(r['ablate_set']):<12} {r['accuracy']:>7.3f} {r['mean_margin']:>+10.3f} {r['median_margin']:>+10.3f} {r['frac_positive_margin']:>9.3f}")
    r = results["BASELINE"]
    print(f"{'BASELINE':<10} {str(r['ablate_set']):<12} {r['accuracy']:>7.3f} {r['mean_margin']:>+10.3f} {r['median_margin']:>+10.3f} {r['frac_positive_margin']:>9.3f}")

    print("\n[SANITY] R123 vs BASELINE accuracy:")
    r123 = results["R123"]["accuracy"]
    rbase = results["BASELINE"]["accuracy"]
    print(f"  R123:     {r123:.3f}")
    print(f"  BASELINE: {rbase:.3f}")
    print(f"  abs diff: {abs(r123 - rbase):.4f}  (want <~0.005)")

    print("\n[TABLE 2] Necessity: accuracy drop from removing each receptor from full set")
    print(f"  Remove R1: {v_full:.3f} -> {v_map[frozenset({1,2})]:.3f}  (drop={necessity['remove_R1']:+.3f})")
    print(f"  Remove R2: {v_full:.3f} -> {v_map[frozenset({0,2})]:.3f}  (drop={necessity['remove_R2']:+.3f})")
    print(f"  Remove R3: {v_full:.3f} -> {v_map[frozenset({0,1})]:.3f}  (drop={necessity['remove_R3']:+.3f})")

    print("\n[TABLE 3] Sufficiency: gain over NONE with only one receptor kept")
    print(f"  Only R1: {v_map[frozenset({0})]:.3f}  (gain={sufficiency['only_R1_gain']:+.3f})")
    print(f"  Only R2: {v_map[frozenset({1})]:.3f}  (gain={sufficiency['only_R2_gain']:+.3f})")
    print(f"  Only R3: {v_map[frozenset({2})]:.3f}  (gain={sufficiency['only_R3_gain']:+.3f})")

    print("\n[TABLE 4] Shapley values (sum ≈ v(R123)-v(NONE))")
    print(f"  phi(R1) = {phis[0]:+.4f}")
    print(f"  phi(R2) = {phis[1]:+.4f}")
    print(f"  phi(R3) = {phis[2]:+.4f}")
    print(f"  sum     = {sum(phis):+.4f}  vs  v(R123)-v(NONE) = {(v_full - v_none):+.4f}")

    print("\n[TABLE 5] Interaction indices (positive=synergy, negative=redundancy)")
    print(f"  I(R1,R2) = {I12:+.4f}")
    print(f"  I(R1,R3) = {I13:+.4f}")
    print(f"  I(R2,R3) = {I23:+.4f}")

    print("\n[TABLE 6] Quick redundancy check")
    lhs = v_full
    rhs = v_map[frozenset({0})] + v_map[frozenset({1})] + v_map[frozenset({2})] - 2*v_none
    print(f"  v(R123) = {lhs:.3f}")
    print(f"  v(R1)+v(R2)+v(R3)-2*v(NONE) = {rhs:.3f}")
    print("  lhs >> rhs -> synergy; lhs << rhs -> redundancy; close -> roughly additive.")

    payload = {
        "meta": {
            "csv": str(args.csv),
            "N": int(N),
            "batch_size": int(args.batch_size),
            "device": device,
            "use_both": bool(int(args.use_both)),
            "seed": int(args.seed),
        },
        "tokens": {
            "he_id": int(he_id),
            "she_id": int(she_id),
            "he_str": tokenizer.decode([he_id], clean_up_tokenization_spaces=False),
            "she_str": tokenizer.decode([she_id], clean_up_tokenization_spaces=False),
        },
        "receptors": [{"idx": i+1, "layer": rs.layer, "head": rs.head, "sv_idx": rs.sv_idx, "polarity": rs.polarity} for i, rs in enumerate(rec_specs)],
        "results": results,
        "necessity": necessity,
        "sufficiency": sufficiency,
        "shapley": {"phi_R1": float(phis[0]), "phi_R2": float(phis[1]), "phi_R3": float(phis[2])},
        "interaction": {"I_R1_R2": float(I12), "I_R1_R3": float(I13), "I_R2_R3": float(I23)},
        "sanity": {"r123_acc": float(r123), "baseline_acc": float(rbase), "abs_diff": float(abs(r123 - rbase))},
    }

    if int(args.save_json) == 1:
        out_path = out_dir / str(args.save_name)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[SAVED] {out_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
