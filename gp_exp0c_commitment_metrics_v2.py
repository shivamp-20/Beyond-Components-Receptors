#!/usr/bin/env python3
"""
GP Exp0c: Commitment + per-layer receptor diagnostics.

Implements:
- Cache residual stream at decision position for 13 snapshots: layer=-1 (blocks.0.hook_resid_pre) and layers 0..11 (blocks.l.hook_resid_post).
- Compute per-receptor activations g[k,l,i] = r_k @ resid[l,i]
- Metrics: commitment layers (agreement >= threshold and stays), AUC curves, mean activations by gender, deliberation index, delta_g, RCS.
- Plots + JSON output.

Assumptions (consistent with your prior Exp0/Exp4 scripts):
- GP CSV has columns: prefix, pronoun (and maybe corr_prefix, corr_pronoun). We only use rows where pronoun in {"he","she"}.
- Decision position = last non-pad token in the prefix (i.e., position whose next-token logits predict pronoun).
- Receptors are OV right singular vectors Vh[sv_idx] in d_model space (as in your printing: receptor_vocab = v_row @ W_U).
"""

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer


# -----------------------------
# Small utilities
# -----------------------------
def sign_pm1(x: torch.Tensor) -> torch.Tensor:
    """Return sign in {-1, +1} with 0 mapped to +1."""
    return torch.where(x >= 0, torch.ones_like(x), -torch.ones_like(x))


def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")


def try_roc_auc(y01: np.ndarray, scores: np.ndarray) -> float:
    """Compute ROC AUC with sklearn if available, else a tie-aware rank-based fallback."""
    try:
        from sklearn.metrics import roc_auc_score  # type: ignore
        return float(roc_auc_score(y01, scores))
    except Exception:
        # Fallback: Mann–Whitney U / rank statistic with average ranks for ties.
        y = y01.astype(np.int64)
        n_pos = int(y.sum())
        n = int(y.shape[0])
        n_neg = n - n_pos
        if n_pos == 0 or n_neg == 0:
            return float("nan")

        s = scores.astype(np.float64)
        order = np.argsort(s)
        s_sorted = s[order]
        y_sorted = y[order]

        ranks = np.empty(n, dtype=np.float64)
        i = 0
        rank = 1.0
        while i < n:
            j = i + 1
            while j < n and s_sorted[j] == s_sorted[i]:
                j += 1
            # average rank for tie block [i, j)
            avg_rank = (rank + (rank + (j - i) - 1.0)) / 2.0
            ranks[i:j] = avg_rank
            rank += (j - i)
            i = j

        # Sum of ranks for positives
        r_pos = ranks[y_sorted == 1].sum()
        # U statistic
        u = r_pos - n_pos * (n_pos + 1) / 2.0
        auc = u / (n_pos * n_neg)
        return float(auc)


def entropy(p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p, eps, 1.0)
    p = p / p.sum()
    return float(-(p * np.log(p)).sum())


@dataclass
class Example:
    prefix: str
    y: int  # +1 (male), -1 (female)


def sniff_delimiter(sample: str) -> str:
    try:
        return csv.Sniffer().sniff(sample, delimiters=[",", "\t", ";", "|"]).delimiter
    except Exception:
        return ","


def load_gp_examples(csv_path: Path, use_both: int = 1) -> List[Example]:
    """
    Loads GP rows. Keeps only pronoun in {"he","she"}.
    If use_both=1 and corr_prefix/corr_pronoun exist, we also include the corrupt version as another example.
    """
    txt = csv_path.read_text(encoding="utf-8", errors="ignore")
    delim = sniff_delimiter(txt[:4000])
    reader = csv.DictReader(txt.splitlines(), delimiter=delim)

    exs: List[Example] = []
    for r in reader:
        if "prefix" not in r or "pronoun" not in r:
            continue

        p = (r["pronoun"] or "").strip()
        if p not in ("he", "she"):
            continue
        y = +1 if p == "he" else -1
        exs.append(Example(prefix=(r["prefix"] or ""), y=y))

        if use_both and ("corr_prefix" in r) and ("corr_pronoun" in r):
            cp = (r["corr_pronoun"] or "").strip()
            if cp in ("he", "she"):
                cy = +1 if cp == "he" else -1
                exs.append(Example(prefix=(r["corr_prefix"] or ""), y=cy))
    return exs


# -----------------------------
# Receptor loading (OV Vh row)
# -----------------------------
def load_receptor_dirs(
    out_dir: Path,
    receptors_spec: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, List[Dict]]:
    """
    receptors_spec: "L,H,SV,pol;L,H,SV,pol;L,H,SV,pol"
    Returns:
      rec_dirs: (3, d_model) unit vectors on CPU
      polarity: (3,) tensor on CPU (values in {-1,+1})
      meta: list with layer/head/sv/pol entries
    """
    # This file is produced by your mask training script.
    ov_path = out_dir / "ov_svd_cache.pt"
    if not ov_path.exists():
        # Some repos use a different name.
        alt = out_dir / "svd_cache.pt"
        if alt.exists():
            ov_path = alt
        else:
            raise FileNotFoundError(f"Could not find OV SVD cache at {out_dir}/ov_svd_cache.pt (or svd_cache.pt).")

    ckpt = torch.load(str(ov_path), map_location="cpu")
    # Expected structure (based on your earlier scripts): ckpt["ov"] is list[list[dict]]
    if "ov" in ckpt:
        ov = ckpt["ov"]
    elif "svd" in ckpt and "ov" in ckpt["svd"]:
        ov = ckpt["svd"]["ov"]
    else:
        # Best-effort: try direct.
        ov = ckpt

    specs = [s.strip() for s in receptors_spec.split(";") if s.strip()]
    if len(specs) != 3:
        raise ValueError("Expected exactly 3 receptors in --receptors, e.g. '10,9,0,+1;11,8,6,+1;9,7,1,-1'")

    dirs: List[torch.Tensor] = []
    pols: List[int] = []
    meta: List[Dict] = []

    for idx, spec in enumerate(specs, start=1):
        parts = [x.strip() for x in spec.split(",")]
        if len(parts) != 4:
            raise ValueError(f"Bad receptor spec '{spec}'. Expected 'layer,head,sv_idx,polarity'.")
        layer = int(parts[0]); head = int(parts[1]); sv_idx = int(parts[2]); pol = int(parts[3])
        if pol not in (-1, +1):
            raise ValueError(f"Polarity must be +1 or -1, got {pol} in '{spec}'.")

        cell = ov[layer][head]
        if isinstance(cell, dict) and "Vh" in cell:
            Vh = cell["Vh"]
        elif hasattr(cell, "Vh"):
            Vh = cell.Vh
        else:
            raise KeyError(f"Could not find Vh for ov[{layer}][{head}]. Keys={list(cell.keys()) if isinstance(cell, dict) else type(cell)}")

        v = Vh[sv_idx].detach().to(torch.float32).cpu()
        v = v / (v.norm() + 1e-12)

        dirs.append(v)
        pols.append(pol)
        meta.append({"name": f"R{idx}", "layer": layer, "head": head, "sv_idx": sv_idx, "polarity": pol})

    rec_dirs = torch.stack(dirs, dim=0)       # (3, d_model) on CPU
    polarity = torch.tensor(pols, dtype=torch.float32)  # (3,) on CPU
    return rec_dirs, polarity, meta


# -----------------------------
# Core computation
# -----------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--receptors", type=str, required=True,
                    help="Three receptors: 'L,H,SV,pol;L,H,SV,pol;L,H,SV,pol' e.g. '10,9,0,+1;11,8,6,+1;9,7,1,-1'")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_examples", type=int, default=0)
    ap.add_argument("--use_both", type=int, default=1)
        # Commitment thresholds for sign-agreement (sensitivity table)
    ap.add_argument("--commit_thresholds", type=str, default="0.8,0.85,0.9",
                    help="Comma-separated thresholds for sign-agreement commitment (e.g. \"0.8,0.85,0.9\").")
    ap.add_argument("--primary_commit_thr", type=float, default=0.85,
                    help="Primary sign-agreement threshold used for plots / quick checks.")
    ap.add_argument("--auc_commit_thr", type=float, default=0.90,
                    help="AUC commitment threshold (AUC>=thr and stays >= thr).")
    # Back-compat: old single-threshold flag (used only if --commit_thresholds is empty)
    ap.add_argument("--commit_thr", type=float, default=0.95)
    ap.add_argument("--save_prefix", type=str, default="exp0c")
    args = ap.parse_args()

    def _parse_thresholds_csv(s: str) -> list[float]:
        vals: list[float] = []
        for part in (s or "").split(","):
            part = part.strip()
            if not part:
                continue
            try:
                vals.append(float(part))
            except Exception as e:
                raise ValueError(f"Bad threshold '{part}' in --commit_thresholds: {e}")
        # stable, unique, sorted
        vals = sorted(set(vals))
        return vals

    commit_thresholds = _parse_thresholds_csv(args.commit_thresholds) if (args.commit_thresholds and args.commit_thresholds.strip()) else [float(args.commit_thr)]
    primary_thr = float(args.primary_commit_thr)
    auc_commit_thr = float(args.auc_commit_thr)
    # Ensure primary_thr is included in the sensitivity list for consistent reporting/plotting
    if primary_thr not in commit_thresholds:
        commit_thresholds = sorted(set(commit_thresholds + [primary_thr]))

    data_dir = Path(args.data_dir)
    csv_path = data_dir / args.csv
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Load model
    print("[MODEL] Loading gpt2-small into HookedTransformer")
    model = HookedTransformer.from_pretrained("gpt2-small", device=str(device))
    tok = model.tokenizer
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    he_id = tok.encode(" he", add_special_tokens=False)[0]
    she_id = tok.encode(" she", add_special_tokens=False)[0]
    print(f"[TOKENS] he:  {he_id} -> {tok.decode([he_id])!r}")
    print(f"[TOKENS] she: {she_id} -> {tok.decode([she_id])!r}")

    # Load dataset
    exs = load_gp_examples(csv_path, use_both=int(args.use_both))
    if args.max_examples and args.max_examples > 0:
        exs = exs[: args.max_examples]

    ys = torch.tensor([e.y for e in exs], dtype=torch.float32)  # (N,)
    N = len(exs)
    counts = {+1: int((ys == 1).sum().item()), -1: int((ys == -1).sum().item())}
    print(f"label counts: {counts}")
    print(f"[DATA] N={N} | use_both={args.use_both} | csv={csv_path}")

    prefixes = [e.prefix for e in exs]
    enc = tok(prefixes, add_special_tokens=False, padding=True, return_tensors="pt")
    tokens = enc["input_ids"]  # (N, S)
    attn = enc.get("attention_mask", torch.ones_like(tokens))
    last_idx = (attn.sum(dim=1) - 1).to(torch.long)  # (N,)
    seq_len = tokens.shape[1]
    print(f"[TOKENS] seq_len={seq_len} (padded)")

    # Load receptors (OV Vh rows in d_model space)
    rec_dirs, polarity, meta = load_receptor_dirs(out_dir, args.receptors, device=device)
    # Sanity: cosine similarities
    print("\n[RECEPTORS] Using OV right singular vectors v_row = Vh[sv_idx] (d_model).")
    for m in meta:
        print(f"  {m['name']}: (layer={m['layer']}, head={m['head']}, sv_idx={m['sv_idx']}) polarity={m['polarity']:+d} | ||v||={rec_dirs[int(m['name'][-1])-1].norm():.6g}")
    print("\n[SANITY] receptor cosine similarities:")
    for i in range(3):
        for j in range(i+1, 3):
            c = float(torch.dot(rec_dirs[i], rec_dirs[j]).item())
            print(f"  cos(R{i+1}, R{j+1}) = {c:+.4f}")

    # Hook names: include embedding snapshot and resid_post for all 12 blocks.
    hook_embed_name = "blocks.0.hook_resid_pre"  # embedding+pos before block 0
    hook_post_names = [f"blocks.{l}.hook_resid_post" for l in range(model.cfg.n_layers)]
    hook_names = [hook_embed_name] + hook_post_names  # length 13

    # Allocate residual snapshots: (13, N, d_model) on CPU
    d_model = model.cfg.d_model
    resid_all = torch.empty((len(hook_names), N, d_model), dtype=torch.float32, device="cpu")

    # Also collect baseline logits diagnostics
    he_logits = torch.empty((N,), dtype=torch.float32, device="cpu")
    she_logits = torch.empty((N,), dtype=torch.float32, device="cpu")
    top1_ids = torch.empty((N,), dtype=torch.long, device="cpu")

    # Run with cache in batches
    print(f"\n[RUN] Caching resid_post at decision position: 13 snapshots (layer -1..11). batch_size={args.batch_size} device={device}")
    tokens = tokens.to(device)
    last_idx = last_idx.to(device)

    for start in range(0, N, args.batch_size):
        end = min(N, start + args.batch_size)
        btok = tokens[start:end]
        blidx = last_idx[start:end]  # (B,)
        logits, cache = model.run_with_cache(btok, names_filter=hook_names)

        # Sanity: cache contains keys we asked for
        if start == 0:
            missing = [k for k in hook_names if k not in cache]
            if missing:
                print("[WARN] Some requested cache keys missing:", missing)
                print("[WARN] Available keys sample:", list(cache.keys())[:20])

        # Baseline logits at decision pos
        b_idx = torch.arange(end - start, device=device)
        dec_logits = logits[b_idx, blidx, :]  # (B, vocab)
        he_logits[start:end] = dec_logits[:, he_id].detach().to("cpu")
        she_logits[start:end] = dec_logits[:, she_id].detach().to("cpu")
        top1_ids[start:end] = dec_logits.argmax(dim=-1).detach().to("cpu")

        # Resid snapshots at decision pos
        for li, hname in enumerate(hook_names):
            x = cache[hname]  # (B, S, d_model)
            x_dec = x[b_idx, blidx, :]  # (B, d_model)
            resid_all[li, start:end, :] = x_dec.detach().to("cpu")

        if (end == N) or ((start // args.batch_size) % 10 == 0):
            print(f"  processed {end}/{N}")

    # Baseline diagnostics: he-vs-she vs top1-other
    ys_cpu = ys.cpu()
    he_logits_np = he_logits.numpy()
    she_logits_np = she_logits.numpy()
    diff = he_logits_np - she_logits_np
    pred_hvs = np.where(diff > 0, +1, -1)
    acc_hvs = float((pred_hvs == ys_cpu.numpy()).mean())

    pred_top1 = top1_ids.numpy()
    other_rate = float(((pred_top1 != he_id) & (pred_top1 != she_id)).mean())

    # Top1 expected (counts correct only when top1 is the correct pronoun token)
    exp_tok = np.where(ys_cpu.numpy() == 1, he_id, she_id)
    acc_top1_expected = float((pred_top1 == exp_tok).mean())

    print("\n[BASELINE] Decision-position logits (final):")
    print(f"  acc_he_vs_she_only = {acc_hvs:.3f}")
    print(f"  acc_top1_expected  = {acc_top1_expected:.3f}")
    print(f"  other_rate_top1    = {other_rate:.3f}")

    # -----------------------------
    # Compute g, signed_g, s
    # -----------------------------
    # resid_all: (13, N, d_model)
    # rec_dirs: (3, d_model)
    rec_dirs_f = rec_dirs.to(torch.float32)
    resid_all_f = resid_all.to(torch.float32)

    # g[k,l,i] = rec[k] @ resid[l,i]
    g = torch.einsum("kd,lnd->kln", rec_dirs_f, resid_all_f)  # (3, 13, N)

    v = polarity.view(3, 1, 1)  # (3,1,1)
    pol_g = v * g               # polarity-adjusted activation
    s = pol_g.sum(dim=0)        # combined score: (13, N)

    # -----------------------------
    # Metric 1: Sign-agreement curves + commitment layers (sensitivity across thresholds)
    # signed_margin_g[k, l, i] = y[i] * polarity[k] * g[k, l, i]
    signed_margin_g = (
        y_t.view(1, 1, -1) * (polarity_t.view(3, 1, 1) * g)
    ).cpu().numpy()  # (3, 13, N)

    def _thr_key(t: float) -> str:
        # Fixed 2dp keys for JSON + printing
        return f"{t:.2f}"

    def _commit_layer_from_curve(curve: list[float], thr: float) -> int:
        """Return the smallest layer label l such that curve[l:] stays >= thr."""
        last_below = -1
        for i, v in enumerate(curve):
            if v < thr:
                last_below = i
        idx = min(last_below + 1, len(curve) - 1)
        return int(layer_labels[idx])

    def _agreement_curve(score_lxn: np.ndarray) -> list[float]:
        """Agreement with the FINAL sign per example, per layer."""
        # score_lxn: (L, N)
        final = score_lxn[-1]
        final_s = sign_eps(final, eps=1e-12)
        curve: list[float] = []
        for li in range(score_lxn.shape[0]):
            s = sign_eps(score_lxn[li], eps=1e-12)
            curve.append(float((s == final_s).mean()))
        return curve

    agreement: Dict[str, list[float]] = {}
    for k in range(3):
        agreement[f"R{k+1}"] = _agreement_curve(signed_margin_g[k])
    # combined signed margin: y * sum_k polarity[k]*g[k]
    combined_score = (polarity_t.view(3, 1, 1) * g).sum(dim=0)  # (13, N)
    combined_signed = (y_t.view(1, -1) * combined_score).cpu().numpy()  # (13, N)
    agreement["combined"] = _agreement_curve(combined_signed)

    # Commitment layers across thresholds (sensitivity table)
    commit_layers_sign: Dict[str, Dict[str, int]] = { _thr_key(t): {} for t in commit_thresholds }
    for t in commit_thresholds:
        tk = _thr_key(t)
        for name, curve in agreement.items():
            commit_layers_sign[tk][name] = _commit_layer_from_curve(curve, t)

    primary_key = _thr_key(primary_thr)
    commit_layers_primary = commit_layers_sign[primary_key]

    # Metric 2: AUC per layer (per receptor + combined)
    # -----------------------------
    y01 = ((ys_cpu.numpy() + 1) / 2).astype(np.int64)
    auc: Dict[str, List[float]] = {}
    for k in range(3):
        vals = []
        for li in range(len(layer_labels)):
            vals.append(try_roc_auc(y01, pol_g[k, li].numpy()))
        auc[f"R{k+1}"] = vals
    auc["combined"] = [try_roc_auc(y01, s[li].numpy()) for li in range(len(layer_labels))]
    # Metric 2b: AUC-based commitment layers (AUC >= auc_commit_thr and stays >= thr)
    commit_layers_auc: Dict[str, int] = {}
    for name in ["R1", "R2", "R3", "combined"]:
        commit_layers_auc[name] = _commit_layer_from_curve(auc[name], auc_commit_thr)

    # -----------------------------
    # Metric 3: mean activation by group
    # -----------------------------
    male = (ys_cpu == 1)
    fem = (ys_cpu == -1)

    mean_male: Dict[str, List[float]] = {}
    mean_female: Dict[str, List[float]] = {}

    for k in range(3):
        mm = []
        mf = []
        for li in range(len(layer_labels)):
            mm.append(float(pol_g[k, li, male].mean().item()) if male.any() else float("nan"))
            mf.append(float(pol_g[k, li, fem].mean().item()) if fem.any() else float("nan"))
        mean_male[f"R{k+1}"] = mm
        mean_female[f"R{k+1}"] = mf

    # combined means (optional, but useful)
    mean_male["combined"] = [float(s[li, male].mean().item()) if male.any() else float("nan") for li in range(len(layer_labels))]
    mean_female["combined"] = [float(s[li, fem].mean().item()) if fem.any() else float("nan") for li in range(len(layer_labels))]

    # -----------------------------
    # Metric 4: Deliberation Index
    # -----------------------------
    di: Dict[str, List[float]] = {}
    di_norm: Dict[str, List[float]] = {}
    for k in range(3):
        curve = []
        for li in range(len(layer_labels)):
            vm = float(pol_g[k, li, male].var(unbiased=False).item()) if male.sum() > 1 else float("nan")
            vf = float(pol_g[k, li, fem].var(unbiased=False).item()) if fem.sum() > 1 else float("nan")
            curve.append(0.5 * (vm + vf))
        di[f"R{k+1}"] = [safe_float(x) for x in curve]
        base = curve[0] if (curve[0] is not None and np.isfinite(curve[0]) and curve[0] > 1e-12) else (curve[1] if len(curve) > 1 else 1.0)
        di_norm[f"R{k+1}"] = [safe_float(x / base) for x in curve]

    # -----------------------------
    # Metric 5: delta_g per layer (where signal is written)
    # -----------------------------
    delta_unsigned: Dict[str, List[float]] = {}
    delta_signed_male: Dict[str, List[float]] = {}
    delta_signed_female: Dict[str, List[float]] = {}

    for k in range(3):
        du = []
        dsm = []
        dsf = []
        for li in range(len(layer_labels) - 1):  # 12 deltas
            d = (g[k, li + 1] - g[k, li])  # (N,)
            du.append(float(d.abs().mean().item()))
            # signed per group: polarity * delta
            sd = polarity[k] * d
            dsm.append(float(sd[male].mean().item()) if male.any() else float("nan"))
            dsf.append(float(sd[fem].mean().item()) if fem.any() else float("nan"))
        delta_unsigned[f"R{k+1}"] = du
        delta_signed_male[f"R{k+1}"] = dsm
        delta_signed_female[f"R{k+1}"] = dsf

    # -----------------------------
    # Metric 6: RCS per layer
    # -----------------------------
    g_mean_abs = g.abs().mean(dim=2).numpy()  # (3, 13)
    rcs_per_layer = []
    for li in range(len(layer_labels)):
        w = g_mean_abs[:, li]
        ssum = float(w.sum())
        if ssum <= 1e-12:
            rcs_per_layer.append(float("nan"))
            continue
        p = w / ssum
        H = entropy(p)
        rcs = 1.0 - H / math.log(3.0)
        rcs_per_layer.append(float(rcs))

    # Prediction check (optional narrative)
    pred_text = "R3 commits before R1 (inhibition before promotion)"
    confirmed_sign = (commit_layers_primary["R3"] < commit_layers_primary["R1"])
    confirmed_auc = (commit_layers_auc["R3"] < commit_layers_auc["R1"]) if ("R3" in commit_layers_auc and "R1" in commit_layers_auc) else False

    # -----------------------------
    # Save JSON + tensors
    # -----------------------------
    out_json = out_dir / f"{args.save_prefix}_commitment_results.json"
    payload = {
        "meta": {
            "csv": str(csv_path),
            "N": N,
            "seq_len": int(seq_len),
            "layer_labels": layer_labels,
            "receptors": meta,
            "baseline": {
                "acc_he_vs_she_only": acc_hvs,
                "acc_top1_expected": acc_top1_expected,
                "other_rate_top1": other_rate,
            },
        },
        "commitment_sign": {
            "thresholds": [float(_thr_key(t)) for t in commit_thresholds],
            "primary_threshold": float(_thr_key(primary_thr)),
            "layers": commit_layers_sign,  # keys like "0.85" -> {R1,R2,R3,combined}
        },
        "commitment_auc": {
            "threshold": float(_thr_key(auc_commit_thr)),
            "layers": commit_layers_auc,
        },
        "agreement_curves": agreement,
        "auc_curves": auc,
        "mean_activation_male": mean_male,
        "mean_activation_female": mean_female,
        "deliberation_index": di_norm,
        "delta_g_unsigned": delta_abs,
        "delta_g_signed_male": delta_signed_m,
        "delta_g_signed_female": delta_signed_f,
        "rcs_per_layer": rcs_per_layer,
        "prediction": pred_text,
        "confirmed_sign": bool(confirmed_sign),
        "confirmed_auc": bool(confirmed_auc),    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    torch.save(
        {
            "layer_labels": layer_labels,
            "resid_all": resid_all,  # (13,N,768)
            "g": g,                  # (3,13,N)
            "polarity": polarity,
            "ys": ys_cpu,
            "he_logits": he_logits,
            "she_logits": she_logits,
        },
        str(out_dir / f"{args.save_prefix}_tensors.pt"),
    )

    print(f"\n[SAVED] {out_json}")
    print(f"[SAVED] {out_dir / f'{args.save_prefix}_tensors.pt'}")

    # -----------------------------
    # Plots
    # -----------------------------
    xs = layer_labels

    # Plot 1: agreement curves
    plt.figure(figsize=(8, 5))
    for name in ["R1", "R2", "R3", "combined"]:
        plt.plot(xs, agreement[name], label=name)
    plt.axhline(primary_thr, linestyle="--", label=f"thr={primary_thr:.2f}")
    for name in ["R1", "R2", "R3", "combined"]:
        plt.axvline(commit_layers_primary[name], linestyle=":", alpha=0.5)
    plt.xlabel("Layer (−1=embed, 0..11=post-block)")
    plt.ylabel("Agreement w/ final sign")
    plt.title(f"Per-Receptor Commitment (agreement ≥ {primary_thr:.2f} and stays)")
    plt.legend()
    p1 = out_dir / f"{args.save_prefix}_commitment.png"
    plt.tight_layout()
    plt.savefig(p1, dpi=160)
    plt.close()

    # Plot 2: AUC curves
    plt.figure(figsize=(8, 5))
    for name in ["R1", "R2", "R3", "combined"]:
        plt.plot(xs, auc[name], label=name)
    for name in ["R1", "R2", "R3", "combined"]:
        plt.axvline(commit_layers_primary[name], linestyle=":", alpha=0.35)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Layer (−1=embed, 0..11=post-block)")
    plt.ylabel("ROC AUC (polarity-adjusted)")
    plt.title("Per-layer AUC of each receptor (and combined)")
    plt.legend()
    p2 = out_dir / f"{args.save_prefix}_auc.png"
    plt.tight_layout()
    plt.savefig(p2, dpi=160)
    plt.close()

    # Plot 3: mean activation by gender (polarity-adjusted)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), sharey=False)
    for k, ax in enumerate(axes):
        name = f"R{k+1}"
        ax.plot(xs, mean_male[name], label="male")
        ax.plot(xs, mean_female[name], label="female")
        ax.axvline(commit_layers_primary[name], linestyle=":", alpha=0.35)
        ax.set_title(name)
        ax.set_xlabel("Layer")
        if k == 0:
            ax.set_ylabel("Mean polarity-adjusted activation")
        ax.legend()
    fig.suptitle("Mean activation by gender group (polarity-adjusted)")
    p3 = out_dir / f"{args.save_prefix}_activation_by_gender.png"
    fig.tight_layout()
    fig.savefig(p3, dpi=160)
    plt.close(fig)

    # Plot 4: delta per layer (unsigned)
    plt.figure(figsize=(8, 5))
    dx = list(range(-1, 11))  # 12 transitions: (-1->0) ... (10->11) ; label by source layer
    for k in range(3):
        plt.plot(dx, delta_unsigned[f"R{k+1}"], label=f"R{k+1}")
    plt.xlabel("Layer transition source (l → l+1)")
    plt.ylabel("mean |Δg|")
    plt.title("Layer-wise change in receptor activation (where signal is written)")
    plt.legend()
    p4 = out_dir / f"{args.save_prefix}_delta_per_layer.png"
    plt.tight_layout()
    plt.savefig(p4, dpi=160)
    plt.close()

    # Plot 5: deliberation index (normalized)
    plt.figure(figsize=(8, 5))
    for k in range(3):
        plt.plot(xs, di_norm[f"R{k+1}"], label=f"R{k+1}")
    plt.xlabel("Layer (−1=embed, 0..11=post-block)")
    plt.ylabel("DI / DI(layer=-1)")
    plt.title("Deliberation Index (within-class variance, normalized)")
    plt.legend()
    p5 = out_dir / f"{args.save_prefix}_deliberation.png"
    plt.tight_layout()
    plt.savefig(p5, dpi=160)
    plt.close()

    # Plot 6: RCS per layer
    plt.figure(figsize=(8, 5))
    plt.plot(xs, rcs_per_layer, label="RCS")
    plt.ylim(0.0, 1.0)
    plt.xlabel("Layer (−1=embed, 0..11=post-block)")
    plt.ylabel("RCS")
    plt.title("Receptor Concentration Score per layer (0=spread, 1=dominant)")
    plt.legend()
    p6 = out_dir / f"{args.save_prefix}_rcs.png"
    plt.tight_layout()
    plt.savefig(p6, dpi=160)
    plt.close()

    print(f"[SAVED] {p1}")
    print(f"[SAVED] {p2}")
    print(f"[SAVED] {p3}")
    print(f"[SAVED] {p4}")
    print(f"[SAVED] {p5}")
    print(f"[SAVED] {p6}")

    # Key printed summary
    print("\n[COMMITMENT - sign agreement]")
    for t in [0.80, 0.85, 0.90]:
        tk = _thr_key(t)
        if tk in commit_layers_sign:
            cl = commit_layers_sign[tk]
            print(f"  threshold={float(tk):.2f}:  R1={cl['R1']}  R2={cl['R2']}  R3={cl['R3']}  combined={cl['combined']}")
    # Also print any extra thresholds the user passed
    extra = [t for t in commit_thresholds if _thr_key(t) not in {_thr_key(0.80), _thr_key(0.85), _thr_key(0.90)}]
    for t in extra:
        tk = _thr_key(t)
        cl = commit_layers_sign[tk]
        print(f"  threshold={float(tk):.2f}:  R1={cl['R1']}  R2={cl['R2']}  R3={cl['R3']}  combined={cl['combined']}")

    print(f"\n[COMMITMENT - AUC >= {auc_commit_thr:.2f}]")
    print(f"  R1={commit_layers_auc['R1']}  R2={commit_layers_auc['R2']}  R3={commit_layers_auc['R3']}  combined={commit_layers_auc['combined']}")

    print("\n[RESULT] Final-layer AUCs (layer=11):")
    print(f"  R1: {auc['R1'][final_idx]:.3f}  R2: {auc['R2'][final_idx]:.3f}  R3: {auc['R3'][final_idx]:.3f}  combined: {auc['combined'][final_idx]:.3f}")
    print(f"[RESULT] confirmed_sign={confirmed_sign}  ({pred_text}; primary_thr={primary_thr:.2f})")
    print(f"[RESULT] confirmed_auc={confirmed_auc}  ({pred_text}; auc_thr={auc_commit_thr:.2f})")
    print("\n[DONE] Exp0c complete.")


if __name__ == "__main__":
    main()
