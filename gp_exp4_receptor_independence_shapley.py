#!/usr/bin/env python3
"""
Experiment 4 (GP): Per-receptor ablation + independence + Shapley.

You provide 3 receptors as (layer, head, sv_idx, polarity). Each receptor direction is taken
from the OV SVD cache (Vh[sv_idx], which lives in d_model / residual-stream space).

"Ablate receptor k" = at its host head, project OUT that direction from that head's *per-head*
output (TransformerLens hookpoint: blocks.{layer}.attn.hook_result) at the decision position.

We measure:
- he-vs-she accuracy + margin for all 8 receptor coalitions (which receptors you KEEP)
- "top1 expected" accuracy + other_rate (top1 not he/s) as a stricter metric
- change in other receptors' activations g_j = r_j^T x_final when ablating i (independence)
- Shapley values + pairwise interaction indices (on accuracy and on mean margin)
- debug sanity that hooks actually remove a nontrivial projection

Outputs:
- JSON:   {out_dir}/{save_prefix}_results.json
- PNGs:   {out_dir}/{save_prefix}_acc.png, {save_prefix}_shapley.png, {save_prefix}_delta_acts.png
"""

from __future__ import annotations
import argparse
import csv
import json
import math
from dataclasses import dataclass
from itertools import combinations, permutations
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import torch

# matplotlib is used only for saving plots (works in Kaggle headless)
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer


# ------------------------- helpers -------------------------

@dataclass
class ReceptorSpec:
    layer: int
    head: int
    sv_idx: int
    polarity: int  # +1 male-on-top, -1 female-on-top


def parse_receptors(s: str) -> List[ReceptorSpec]:
    """
    Format:
      "layer,head,sv_idx,polarity;layer,head,sv_idx,polarity;layer,head,sv_idx,polarity"
    Example:
      "10,9,0,+1;11,8,6,+1;9,7,1,-1"
    """
    specs: List[ReceptorSpec] = []
    for part in s.split(";"):
        part = part.strip()
        if not part:
            continue
        a = [x.strip() for x in part.split(",")]
        if len(a) != 4:
            raise ValueError(f"Bad receptor spec chunk: {part!r} (want 4 comma-separated fields)")
        layer, head, sv_idx = int(a[0]), int(a[1]), int(a[2])
        pol = int(a[3])
        if pol not in (-1, +1):
            raise ValueError(f"polarity must be ±1, got {pol}")
        specs.append(ReceptorSpec(layer=layer, head=head, sv_idx=sv_idx, polarity=pol))
    if len(specs) != 3:
        raise ValueError(f"Expected exactly 3 receptors; got {len(specs)}")
    return specs


def sniff_delimiter(csv_path: Path) -> str:
    # match train script behavior closely
    with csv_path.open("r", encoding="utf-8") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"])
        return dialect.delimiter
    except Exception:
        return ","


def load_gp_rows(data_dir: Path, csv_name: str) -> List[dict]:
    csv_path = data_dir / csv_name
    assert csv_path.exists(), f"Missing {csv_path}"
    delim = sniff_delimiter(csv_path)
    rows: List[dict] = []
    with csv_path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        for r in reader:
            # expected: prefix, pronoun
            rows.append(r)
    # basic sanity
    assert "prefix" in rows[0] and "pronoun" in rows[0], f"CSV must have columns prefix, pronoun. Got {list(rows[0].keys())}"
    return rows


def make_batches(rows: List[dict], tokenizer, max_examples: int, batch_size: int):
    # labels: +1 for male (he), -1 for female (she)
    kept: List[dict] = []
    for r in rows:
        p = (r["pronoun"] or "").strip().lower()
        if p not in ("he", "she"):
            continue
        kept.append(r)
        if max_examples > 0 and len(kept) >= max_examples:
            break

    labels = torch.tensor([+1 if r["pronoun"].strip().lower() == "he" else -1 for r in kept], dtype=torch.int64)
    prefixes = [r["prefix"] for r in kept]

    toks = tokenizer(
        prefixes,
        add_special_tokens=False,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    input_ids = toks["input_ids"]
    attn_mask = toks["attention_mask"]
    # decision position is the last non-pad token index
    last_idx = attn_mask.sum(dim=1) - 1  # (B,)
    assert (last_idx >= 0).all()

    n = input_ids.size(0)
    for i in range(0, n, batch_size):
        yield (
            input_ids[i:i+batch_size],
            attn_mask[i:i+batch_size],
            last_idx[i:i+batch_size],
            labels[i:i+batch_size],
        )


def coalition_label(keep: Tuple[int, ...]) -> str:
    # receptors are indexed 0,1,2; labels are R1,R2,R3
    if len(keep) == 0:
        return "NONE"
    return "R" + "".join(str(i+1) for i in keep)


def mean_entropy(p: np.ndarray) -> float:
    # p >= 0, sum=1
    p = np.clip(p, 1e-12, 1.0)
    return float(-(p * np.log(p)).sum())


def rcs_from_contrib(contrib: np.ndarray) -> float:
    # contrib: shape (3,) absolute contributions (>=0)
    tot = float(contrib.sum())
    if not math.isfinite(tot) or tot <= 0:
        return float("nan")
    p = (contrib / tot).astype(np.float64)
    H = mean_entropy(p)
    return float(1.0 - H / math.log(3.0))


# ------------------------- main -------------------------

@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", required=True, type=str)
    ap.add_argument("--csv", required=True, type=str)
    ap.add_argument("--out_dir", required=True, type=str)
    ap.add_argument("--receptors", required=True, type=str)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--use_ln_final", type=int, default=0, help="If 1, apply ln_final before computing g_k = r_k^T x_final.")
    ap.add_argument("--save_prefix", type=str, default="exp4")
    ap.add_argument("--debug_first_batch", type=int, default=1)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    specs = parse_receptors(args.receptors)

    # Load model
    device = args.device
    print("[MODEL] Loading gpt2-small into HookedTransformer")
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)

    # IMPORTANT: hook_result is only computed when this flag is on in TransformerLens
    model.cfg.use_attn_result = True

    tok = model.tokenizer
    # Ensure pad token exists for batching
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    he_id = tok.encode(" he", add_special_tokens=False)[0]
    she_id = tok.encode(" she", add_special_tokens=False)[0]
    print(f"[TOKENS] he:  {he_id} -> {tok.decode([he_id])!r}")
    print(f"[TOKENS] she: {she_id} -> {tok.decode([she_id])!r}")

    # Load dataset
    rows = load_gp_rows(data_dir, args.csv)
    labels = [r["pronoun"].strip().lower() for r in rows if (r.get("pronoun","").strip().lower() in ("he","she"))]
    print(f"label counts: {{1: {labels.count('he')}, -1: {labels.count('she')}}}")
    print(f"unique pronoun strings: {sorted(set(labels))}")

    # Load OV SVD cache from your out_dir (same as training script)
    svd_path = out_dir / "svd_cache.pt"
    assert svd_path.exists(), f"Missing {svd_path}. Run mask training first (it creates svd_cache.pt)."
    # We reuse the repo's loader for correctness.
    import train_gp_masks_and_dump_ov_logit_receptors_ddp as gp
    _, ov, _, _, _ = gp.load_svd_cache(str(svd_path), device="cpu")

    # Extract receptor directions (Vh[sv_idx]) in d_model space
    dirs: List[torch.Tensor] = []
    pols: List[int] = []
    for k, sp in enumerate(specs):
        Vh = ov[sp.layer][sp.head].Vh  # (rank, d_model)
        v = Vh[sp.sv_idx].to(torch.float32)
        v = v / (v.norm() + 1e-12)
        dirs.append(v.to(device))
        pols.append(sp.polarity)

    print("\n[RECEPTORS] Using OV right singular vectors v_row = Vh[sv_idx] (d_model).")
    for k, sp in enumerate(specs):
        print(f"  R{k+1}: (layer={sp.layer}, head={sp.head}, sv_idx={sp.sv_idx}) polarity={sp.polarity:+d} | ||v||={dirs[k].norm().item():.4g}")

    # receptor cosine sanity (helpful for "redundancy")
    print("\n[SANITY] receptor cosine similarities:")
    for i in range(3):
        for j in range(i+1, 3):
            ci = torch.dot(dirs[i], dirs[j]).item()
            print(f"  cos(R{i+1}, R{j+1}) = {ci:+.4f}")

    # Value function will be computed on 8 coalitions + baseline
    full = (0, 1, 2)
    all_keeps: List[Tuple[int, ...]] = []
    for size in range(0, 4):
        for combo in combinations(full, size):
            all_keeps.append(tuple(sorted(combo)))

    # We'll store results keyed by coalition label
    results: Dict[str, dict] = {}

    # Precompute a fixed dataset order (tokenize once)
    # We'll iterate batches from this fixed order for every coalition so comparisons are clean.
    rows_filtered = [r for r in rows if (r.get("pronoun","").strip().lower() in ("he","she"))]
    if args.max_examples > 0:
        rows_filtered = rows_filtered[:args.max_examples]

    # Tokenize once for speed
    prefixes = [r["prefix"] for r in rows_filtered]
    y = torch.tensor([+1 if r["pronoun"].strip().lower() == "he" else -1 for r in rows_filtered], dtype=torch.int64)

    toks = tok(prefixes, add_special_tokens=False, padding=True, return_tensors="pt")
    input_ids_all = toks["input_ids"]
    attn_mask_all = toks["attention_mask"]
    last_idx_all = attn_mask_all.sum(dim=1) - 1

    N = input_ids_all.size(0)
    S = input_ids_all.size(1)
    n_layers = model.cfg.n_layers
    print(f"\n[RUN] N={N} examples | seq_len={S} | batch_size={args.batch_size} | device={device}")

    # ---- hooks ----
    def build_ablation_hooks(ablate_set: Tuple[int, ...], last_idx_batch: torch.Tensor):
        """
        Return a list of fwd_hooks for this batch.
        We create per-layer hooks on blocks.{l}.attn.hook_result and one hook on
        blocks.{L-1}.hook_resid_post to capture x_final activations.
        """
        # layer -> head -> Q(d_model, m)
        layer_map: Dict[int, Dict[int, torch.Tensor]] = {}
        for k in ablate_set:
            sp = specs[k]
            layer_map.setdefault(sp.layer, {}).setdefault(sp.head, [])
        # fill direction lists
        for k in ablate_set:
            sp = specs[k]
            layer_map[sp.layer][sp.head].append(dirs[k])

        # convert lists -> orthonormal bases Q
        for l in list(layer_map.keys()):
            for h in list(layer_map[l].keys()):
                D = torch.stack(layer_map[l][h], dim=0)  # (m, d_model)
                # Orthonormal basis of span(D): QR on D^T
                Q, _ = torch.linalg.qr(D.T, mode="reduced")  # (d_model, m)
                layer_map[l][h] = Q

        debug = {"removed_mean": {}}  # filled in hook if debug_first_batch

        def make_layer_hook(l: int, head_to_Q: Dict[int, torch.Tensor]):
            def hook_fn(act: torch.Tensor, hook):
                # act: (B, S, H, d_model)
                B = act.size(0)
                idx = torch.arange(B, device=act.device)
                for h, Q in head_to_Q.items():
                    vecs = act[idx, last_idx_batch, h, :]  # (B, d_model)
                    proj = (vecs @ Q) @ Q.T               # (B, d_model)
                    act[idx, last_idx_batch, h, :] = vecs - proj
                    if args.debug_first_batch and (l, h) not in debug["removed_mean"]:
                        debug["removed_mean"][(l, h)] = float(proj.norm(dim=-1).mean().detach().cpu())
                return act
            return hook_fn

        hooks = []
        for l, head_to_Q in layer_map.items():
            hooks.append((f"blocks.{l}.attn.hook_result", make_layer_hook(l, head_to_Q)))
        return hooks, debug

    # We capture final activations by a separate hook and accumulate sums on CPU.
    def run_coalition(keep: Tuple[int, ...]):
        label = coalition_label(keep)
        ablate = tuple(sorted(set(full) - set(keep)))

        # accumulators
        n_total = 0
        correct_hvs = 0
        correct_top1_expected = 0
        other_top1 = 0
        margins: List[float] = []

        # activation sums: overall and by label
        g_sum = np.zeros((3,), dtype=np.float64)
        g_sum_m = np.zeros((3,), dtype=np.float64)
        g_sum_f = np.zeros((3,), dtype=np.float64)
        n_m = 0
        n_f = 0

        # RCS at final layer based on |v_k * g_k| (dataset mean)
        abs_contrib_sum = np.zeros((3,), dtype=np.float64)

        first_batch_debug = None
        first_batch_logits_pair = None  # for sanity diffs

        for start in range(0, N, args.batch_size):
            end = min(N, start + args.batch_size)
            input_ids = input_ids_all[start:end].to(device)
            attn_mask = attn_mask_all[start:end].to(device)
            last_idx = last_idx_all[start:end].to(device)
            yb = y[start:end].to(device)

            # hook to capture x_final at decision pos and compute g_k
            g_batch_cpu = None

            def capture_final_resid(resid: torch.Tensor, hook):
                # resid: (B,S,d_model) at last block resid_post
                nonlocal g_batch_cpu
                B = resid.size(0)
                idx = torch.arange(B, device=resid.device)
                x = resid[idx, last_idx, :]  # (B,d_model)
                if args.use_ln_final:
                    x = model.ln_final(x)
                g = torch.stack([torch.sum(x * dirs[k], dim=-1) for k in range(3)], dim=-1)  # (B,3)
                g_batch_cpu = g.detach().cpu().numpy()
                return resid

            fwd_hooks = [(f"blocks.{n_layers-1}.hook_resid_post", capture_final_resid)]

            ablation_hooks, dbg = build_ablation_hooks(ablate, last_idx)
            fwd_hooks.extend(ablation_hooks)

            with model.hooks(fwd_hooks=fwd_hooks):
                logits = model(input_ids, attention_mask=attn_mask)  # (B,S,V)

            # sanity: did we actually capture g?
            assert g_batch_cpu is not None, "Failed to capture final residual activations. Check hook name."

            # logits at decision position
            B = logits.size(0)
            idx = torch.arange(B, device=logits.device)
            he_log = logits[idx, last_idx, he_id]
            she_log = logits[idx, last_idx, she_id]

            # he-vs-she prediction
            pred = torch.where(he_log > she_log, torch.ones_like(yb), -torch.ones_like(yb))
            correct = (pred == yb).to(torch.int64)
            correct_hvs += int(correct.sum().item())

            # top1 expected + other rate
            top1 = torch.argmax(logits[idx, last_idx, :], dim=-1)
            is_other = (top1 != he_id) & (top1 != she_id)
            other_top1 += int(is_other.sum().item())
            exp = torch.where(yb == 1, torch.tensor(he_id, device=logits.device), torch.tensor(she_id, device=logits.device))
            correct_top1_expected += int((top1 == exp).sum().item())

            # margins
            margin = (yb.to(torch.float32) * (he_log - she_log)).detach().cpu().numpy()
            margins.extend(margin.tolist())

            # receptor activation summaries
            g_sum += g_batch_cpu.sum(axis=0)
            abs_contrib_sum += np.abs((np.array(pols, dtype=np.float64)[None, :] * g_batch_cpu)).sum(axis=0)

            # by label
            y_cpu = yb.detach().cpu().numpy()
            for i_ex in range(len(y_cpu)):
                if y_cpu[i_ex] == 1:
                    g_sum_m += g_batch_cpu[i_ex]
                    n_m += 1
                else:
                    g_sum_f += g_batch_cpu[i_ex]
                    n_f += 1

            n_total += B

            # debug
            if start == 0 and args.debug_first_batch:
                first_batch_debug = dbg["removed_mean"]
                first_batch_logits_pair = (
                    float(he_log[0].detach().cpu()),
                    float(she_log[0].detach().cpu()),
                )

        acc_hvs = correct_hvs / max(1, n_total)
        acc_top1 = correct_top1_expected / max(1, n_total)
        other_rate = other_top1 / max(1, n_total)

        margins_np = np.array(margins, dtype=np.float64)
        mean_margin = float(np.mean(margins_np))
        med_margin = float(np.median(margins_np))
        frac_pos = float(np.mean(margins_np > 0))

        g_mean = (g_sum / max(1, n_total)).tolist()
        g_mean_m = (g_sum_m / max(1, n_m)).tolist()
        g_mean_f = (g_sum_f / max(1, n_f)).tolist()

        # mean RCS based on dataset-average absolute contributions
        rcs = rcs_from_contrib(abs_contrib_sum / max(1, n_total))

        out = {
            "keep": list(keep),
            "ablate": list(ablate),
            "acc_he_vs_she": acc_hvs,
            "acc_top1_expected": acc_top1,
            "other_rate_top1": other_rate,
            "mean_margin": mean_margin,
            "median_margin": med_margin,
            "frac_positive_margin": frac_pos,
            "g_mean": g_mean,
            "g_mean_male": g_mean_m,
            "g_mean_female": g_mean_f,
            "rcs_final_mean": rcs,
            "n": n_total,
        }

        if args.debug_first_batch:
            out["debug_removed_mean_proj_norm_first_batch"] = {f"L{l}H{h}": v for (l, h), v in (first_batch_debug or {}).items()}
            out["debug_first_example_he_she_logits"] = {"he": first_batch_logits_pair[0], "she": first_batch_logits_pair[1]}

        return label, out

    # ---- run baseline first (no ablation hooks) ----
    # We'll compute it by calling run_coalition(R123) but also a true BASELINE with no hooks
    # (and compare)
    print("\n[RUN] Running all 8 coalitions...")
    for keep in all_keeps:
        label, out = run_coalition(keep)
        results[label] = out
        print(f"  done {label:<4} acc(hvs)={out['acc_he_vs_she']:.3f}  mean_margin={out['mean_margin']:+.3f}  other_rate={out['other_rate_top1']:.3f}")

    # true baseline: no hooks at all
    def run_baseline_nohooks():
        correct_hvs = 0
        margins = []
        other_top1 = 0
        correct_top1_expected = 0
        n_total = 0

        for start in range(0, N, args.batch_size):
            end = min(N, start + args.batch_size)
            input_ids = input_ids_all[start:end].to(device)
            attn_mask = attn_mask_all[start:end].to(device)
            last_idx = last_idx_all[start:end].to(device)
            yb = y[start:end].to(device)

            logits = model(input_ids, attention_mask=attn_mask)
            B = logits.size(0)
            idx = torch.arange(B, device=logits.device)
            he_log = logits[idx, last_idx, he_id]
            she_log = logits[idx, last_idx, she_id]

            pred = torch.where(he_log > she_log, torch.ones_like(yb), -torch.ones_like(yb))
            correct_hvs += int((pred == yb).sum().item())

            top1 = torch.argmax(logits[idx, last_idx, :], dim=-1)
            is_other = (top1 != he_id) & (top1 != she_id)
            other_top1 += int(is_other.sum().item())
            exp = torch.where(yb == 1, torch.tensor(he_id, device=logits.device), torch.tensor(she_id, device=logits.device))
            correct_top1_expected += int((top1 == exp).sum().item())

            margin = (yb.to(torch.float32) * (he_log - she_log)).detach().cpu().numpy()
            margins.extend(margin.tolist())
            n_total += B

        margins_np = np.array(margins, dtype=np.float64)
        return {
            "keep": [0,1,2],
            "ablate": [],
            "acc_he_vs_she": correct_hvs / n_total,
            "acc_top1_expected": correct_top1_expected / n_total,
            "other_rate_top1": other_top1 / n_total,
            "mean_margin": float(np.mean(margins_np)),
            "median_margin": float(np.median(margins_np)),
            "frac_positive_margin": float(np.mean(margins_np > 0)),
            "n": n_total,
        }

    results["BASELINE"] = run_baseline_nohooks()

    # ---- Shapley on accuracy + mean margin ----
    def v_acc(keep_set: Tuple[int, ...]) -> float:
        return float(results[coalition_label(tuple(sorted(keep_set)))]["acc_he_vs_she"])

    def v_margin(keep_set: Tuple[int, ...]) -> float:
        return float(results[coalition_label(tuple(sorted(keep_set)))]["mean_margin"])

    def shapley_values(value_fn):
        phi = {0: 0.0, 1: 0.0, 2: 0.0}
        perms = list(permutations([0, 1, 2]))
        for perm in perms:
            S = tuple()
            for k in perm:
                v_before = value_fn(S)
                S_after = tuple(sorted(S + (k,)))
                v_after = value_fn(S_after)
                phi[k] += (v_after - v_before) / len(perms)
                S = S_after
        return phi

    phi_acc = shapley_values(v_acc)
    phi_margin = shapley_values(v_margin)

    def interaction_ij(value_fn, i: int, j: int) -> float:
        # I_ij = 0.5 [v({i,j}) - v({i}) - v({j}) + v({})]  (K=3)
        return 0.5 * (
            value_fn(tuple(sorted((i, j))))
            - value_fn((i,))
            - value_fn((j,))
            + value_fn(tuple())
        )

    inter_acc = {
        "I12": interaction_ij(v_acc, 0, 1),
        "I13": interaction_ij(v_acc, 0, 2),
        "I23": interaction_ij(v_acc, 1, 2),
    }
    inter_margin = {
        "I12": interaction_ij(v_margin, 0, 1),
        "I13": interaction_ij(v_margin, 0, 2),
        "I23": interaction_ij(v_margin, 1, 2),
    }

    # ---- Independence deltas (single ablation from full set) ----
    # Compare coalition "R123" vs "Rjk" (remove i)
    def g_mean(label: str) -> np.ndarray:
        return np.array(results[label]["g_mean"], dtype=np.float64)

    indep = {}
    for i in [0, 1, 2]:
        keep_others = tuple(sorted(set(full) - {i}))
        lab = coalition_label(keep_others)
        indep[f"remove_R{i+1}"] = {
            "acc_drop": v_acc(full) - v_acc(keep_others),
            "mean_margin_drop": v_margin(full) - v_margin(keep_others),
            "delta_g_mean": (g_mean(lab) - g_mean("R123")).tolist(),  # how x_final receptor activations shift
        }

    # ---- Print tables ----
    print("\n" + "=" * 80)
    print("EXPERIMENT 4: PER-RECEPTOR ABLATION + INDEPENDENCE + SHAPLEY (GP)")
    print("=" * 80)

    order = ["NONE", "R1", "R2", "R3", "R12", "R13", "R23", "R123", "BASELINE"]
    print("\n[TABLE 1] Coalitions")
    print(f"{'Keep':<8} {'Ablated':<12} {'Acc(hvs)':>9} {'AccTop1':>9} {'Other':>7} {'MeanMar':>9} {'RCS':>6}")
    print("-" * 72)
    for lab in order:
        if lab not in results:
            continue
        r = results[lab]
        print(f"{lab:<8} {str(r.get('ablate', [])):<12} {r['acc_he_vs_she']:>9.3f} {r['acc_top1_expected']:>9.3f} {r['other_rate_top1']:>7.3f} {r['mean_margin']:>+9.3f} {r.get('rcs_final_mean', float('nan')):>6.3f}")

    print("\n[TABLE 2] Necessity (drop from removing each receptor from full set, on he-vs-she acc)")
    print(f"  Remove R1: {v_acc(full):.3f} -> {v_acc((1,2)):.3f} (drop={v_acc(full)-v_acc((1,2)):+.3f})")
    print(f"  Remove R2: {v_acc(full):.3f} -> {v_acc((0,2)):.3f} (drop={v_acc(full)-v_acc((0,2)):+.3f})")
    print(f"  Remove R3: {v_acc(full):.3f} -> {v_acc((0,1)):.3f} (drop={v_acc(full)-v_acc((0,1)):+.3f})")

    print("\n[TABLE 3] Sufficiency (gain over NONE)")
    print(f"  Only R1: {v_acc((0,)):.3f} (gain={v_acc((0,))-v_acc(tuple()):+.3f})")
    print(f"  Only R2: {v_acc((1,)):.3f} (gain={v_acc((1,))-v_acc(tuple()):+.3f})")
    print(f"  Only R3: {v_acc((2,)):.3f} (gain={v_acc((2,))-v_acc(tuple()):+.3f})")

    print("\n[TABLE 4] Shapley values (accuracy)")
    for k in [0,1,2]:
        print(f"  phi_acc(R{k+1}) = {phi_acc[k]:+0.4f}")
    print(f"  sum = {sum(phi_acc.values()):+0.4f}  vs  v(R123)-v(NONE) = {v_acc(full)-v_acc(tuple()):+0.4f}")

    print("\n[TABLE 5] Shapley values (mean margin)")
    for k in [0,1,2]:
        print(f"  phi_margin(R{k+1}) = {phi_margin[k]:+0.4f}")
    print(f"  sum = {sum(phi_margin.values()):+0.4f}  vs  v(R123)-v(NONE) = {v_margin(full)-v_margin(tuple()):+0.4f}")

    print("\n[TABLE 6] Interaction indices (accuracy | margin)")
    for key in ["I12","I13","I23"]:
        print(f"  {key}: {inter_acc[key]:+0.4f} | {inter_margin[key]:+0.4f}")

    print("\n[TABLE 7] Independence deltas (single ablation)")
    for k, d in indep.items():
        dg = d["delta_g_mean"]
        print(f"  {k}: acc_drop={d['acc_drop']:+.3f}  margin_drop={d['mean_margin_drop']:+.3f}  delta_g={['%+.3f'%x for x in dg]}")

    print("\n[SANITY] R123 vs BASELINE (no hooks)")
    print(f"  R123 acc(hvs)     : {results['R123']['acc_he_vs_she']:.3f}")
    print(f"  BASELINE acc(hvs) : {results['BASELINE']['acc_he_vs_she']:.3f}")
    print(f"  abs diff          : {abs(results['R123']['acc_he_vs_she']-results['BASELINE']['acc_he_vs_she']):.4f}  (want <~0.005)")

    if args.debug_first_batch:
        print("\n[DEBUG] mean projection norm removed (first batch only) per ablated head:")
        for lab in ["NONE", "R23", "R13", "R12"]:  # ablate all / ablate single receptor cases
            if lab in results and "debug_removed_mean_proj_norm_first_batch" in results[lab]:
                d = results[lab]["debug_removed_mean_proj_norm_first_batch"]
                print(f"  {lab}: {d}")

    # ---- Save JSON ----
    payload = {
        "args": vars(args),
        "receptors": [sp.__dict__ for sp in specs],
        "polarity": pols,
        "results": results,
        "shapley": {"accuracy": {f"R{k+1}": phi_acc[k] for k in phi_acc},
                    "margin": {f"R{k+1}": phi_margin[k] for k in phi_margin}},
        "interaction": {"accuracy": inter_acc, "margin": inter_margin},
        "independence": indep,
    }
    json_path = out_dir / f"{args.save_prefix}_results.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[SAVED] {json_path}")

    # ---- Plots ----
    # 1) Acc bars
    labs = ["NONE", "R1", "R2", "R3", "R12", "R13", "R23", "R123"]
    accs = [results[lab]["acc_he_vs_she"] for lab in labs]
    acc_top1 = [results[lab]["acc_top1_expected"] for lab in labs]
    plt.figure()
    xs = np.arange(len(labs))
    plt.bar(xs - 0.2, accs, width=0.4, label="he-vs-she acc")
    plt.bar(xs + 0.2, acc_top1, width=0.4, label="top1 expected acc")
    plt.xticks(xs, labs)
    plt.ylim(0, 1.0)
    plt.title("Exp4: accuracy by coalition")
    plt.legend()
    acc_png = out_dir / f"{args.save_prefix}_acc.png"
    plt.tight_layout()
    plt.savefig(acc_png, dpi=200)
    plt.close()
    print(f"[SAVED] {acc_png}")

    # 2) Shapley (accuracy + margin)
    plt.figure()
    xs = np.arange(3)
    plt.bar(xs - 0.2, [phi_acc[k] for k in [0,1,2]], width=0.4, label="phi(acc)")
    plt.bar(xs + 0.2, [phi_margin[k] for k in [0,1,2]], width=0.4, label="phi(mean margin)")
    plt.xticks(xs, ["R1", "R2", "R3"])
    plt.title("Exp4: Shapley values")
    plt.legend()
    shp_png = out_dir / f"{args.save_prefix}_shapley.png"
    plt.tight_layout()
    plt.savefig(shp_png, dpi=200)
    plt.close()
    print(f"[SAVED] {shp_png}")

    # 3) Delta activations heatmap: rows=removed receptor, cols=activation of R1/R2/R3
    delta = np.stack([
        np.array(indep["remove_R1"]["delta_g_mean"], dtype=np.float64),
        np.array(indep["remove_R2"]["delta_g_mean"], dtype=np.float64),
        np.array(indep["remove_R3"]["delta_g_mean"], dtype=np.float64),
    ], axis=0)
    plt.figure()
    plt.imshow(delta, aspect="auto")
    plt.colorbar()
    plt.xticks([0,1,2], ["g1", "g2", "g3"])
    plt.yticks([0,1,2], ["remove R1", "remove R2", "remove R3"])
    plt.title("Exp4: Δ receptor activations (x_final) vs R123 baseline")
    d_png = out_dir / f"{args.save_prefix}_delta_acts.png"
    plt.tight_layout()
    plt.savefig(d_png, dpi=200)
    plt.close()
    print(f"[SAVED] {d_png}")

    print("\n[DONE] Exp4 complete.")


if __name__ == "__main__":
    main()
