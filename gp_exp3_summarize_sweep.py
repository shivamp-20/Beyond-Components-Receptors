#!/usr/bin/env python3
"""
Experiment 3: Summarize Exp2 sweep runs.

Scans for exp2_results.json produced by gp_exp2_rotate_subspace_v5_sweep.py and computes:
- KL needed to reach flip_target% flips (he->she and she->he), for ROTATED and STAR
- Flip rate at a fixed KL budget (kl_budget), for ROTATED and STAR
- other% diagnostic at the chosen KL budget
- w concentration: max |w_j| and top-3 |w_j|
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _normalize_sweep(sweep_obj: Any) -> List[Dict[str, Any]]:
    """
    Exp2 writes sweeps as dict: { "0.0": {...}, "1.0": {...}, ... }
    Normalize to list of row dicts sorted by sigma_scale.
    """
    if sweep_obj is None:
        return []

    if isinstance(sweep_obj, dict):
        # IMPORTANT: dict iteration gives KEYS; we want VALUES (row dicts)
        rows = list(sweep_obj.values())
        rows = [r for r in rows if isinstance(r, dict)]
    elif isinstance(sweep_obj, list):
        rows = [r for r in sweep_obj if isinstance(r, dict)]
    else:
        return []

    rows.sort(key=lambda r: _safe_float(r.get("sigma_scale")))
    return rows


def _kl_at_flip_target(rows: List[Dict[str, Any]], flip_key: str, flip_target: float) -> float:
    best = float("inf")
    for row in rows:
        flip = _safe_float(row.get(flip_key))
        kl = _safe_float(row.get("kl_mean"))
        if math.isnan(flip) or math.isnan(kl):
            continue
        if flip >= flip_target and kl < best:
            best = kl
    return best if best != float("inf") else float("nan")


def _flip_at_kl_budget(rows: List[Dict[str, Any]], flip_key: str, kl_budget: float) -> Tuple[float, float, float]:
    """
    Returns (flip_at_budget, other_at_budget, kl_used).
    Rule: among rows with kl_mean <= kl_budget pick MAX flip. Tie-break: smaller KL.
    Fallback: if none under budget, pick minimum KL row.
    """
    if not rows:
        return (float("nan"), float("nan"), float("nan"))

    best_row = None
    best_flip = -float("inf")
    best_kl = float("inf")

    for row in rows:
        kl = _safe_float(row.get("kl_mean"))
        flip = _safe_float(row.get(flip_key))
        if math.isnan(kl) or math.isnan(flip):
            continue
        if kl <= kl_budget:
            if (flip > best_flip) or (flip == best_flip and kl < best_kl):
                best_flip = flip
                best_kl = kl
                best_row = row

    if best_row is None:
        for row in rows:
            kl = _safe_float(row.get("kl_mean"))
            flip = _safe_float(row.get(flip_key))
            if math.isnan(kl) or math.isnan(flip):
                continue
            if kl < best_kl:
                best_kl = kl
                best_flip = flip
                best_row = row

    other = _safe_float(best_row.get("other_pct_all")) if isinstance(best_row, dict) else float("nan")
    return (best_flip, other, best_kl)


def _w_concentration(w: Any) -> Tuple[float, str]:
    if not isinstance(w, list) or len(w) == 0:
        return (float("nan"), "")
    vals = [abs(_safe_float(x)) for x in w]
    max_abs = max(vals) if vals else float("nan")
    idxs = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)[:3]
    top3 = ",".join([f"t{i}:{vals[i]:.4g}" for i in idxs])
    return (float(max_abs), top3)


@dataclass
class Row:
    run_name: str
    json_path: str
    k: Optional[int]
    w_mode: str
    lda_reg: Optional[float]
    max_abs_w: float
    top3_abs_w: str

    rot_kl95_he: float
    star_kl95_he: float
    d_kl95_he: float

    rot_flip_he_at_budget: float
    star_flip_he_at_budget: float

    rot_other_at_budget: float
    star_other_at_budget: float


def _find_exp2_jsons(search_root: Path) -> List[Path]:
    out: List[Path] = []
    for root, _, files in os.walk(search_root):
        if "exp2_results.json" in files:
            out.append(Path(root) / "exp2_results.json")
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True, help="Base output dir (e.g., outputs/gp). Used as default search_root.")
    ap.add_argument("--search_root", default=None, help="Root dir to scan (defaults to out_dir).")
    ap.add_argument("--flip_target", type=float, default=95.0, help="Flip target percent (default 95).")
    ap.add_argument("--kl_budget", type=float, default=2.0, help="KL budget for flip@KL (default 2.0).")
    ap.add_argument("--topn", type=int, default=20, help="How many rows to print.")
    ap.add_argument("--save_csv", default=None, help="If set, write CSV to this path.")
    args = ap.parse_args()

    search_root = Path(args.search_root) if args.search_root else Path(args.out_dir)
    json_paths = _find_exp2_jsons(search_root)
    if not json_paths:
        print(f"[WARN] No exp2_results.json found under: {search_root}")
        return

    rows_out: List[Row] = []

    for jp in json_paths:
        payload = json.loads(jp.read_text(encoding="utf-8"))
        cfg = payload.get("config", {})
        run_name = str(cfg.get("run_name") or jp.parent.name)

        k = cfg.get("k")
        try:
            k = int(k) if k is not None else None
        except Exception:
            k = None

        w_mode = str(cfg.get("w_mode") or "unknown")
        lda_reg = cfg.get("lda_reg")
        lda_reg_f = _safe_float(lda_reg) if lda_reg is not None else float("nan")

        max_abs_w, top3_abs_w = _w_concentration(payload.get("w"))

        sweep_rot = _normalize_sweep(payload.get("sweep_rotated"))
        sweep_star = _normalize_sweep(payload.get("sweep_star"))

        rot_kl95_he = _kl_at_flip_target(sweep_rot, "flip_he_pct", args.flip_target)
        star_kl95_he = _kl_at_flip_target(sweep_star, "flip_he_pct", args.flip_target)
        d_kl95_he = (rot_kl95_he - star_kl95_he) if (not math.isnan(rot_kl95_he) and not math.isnan(star_kl95_he)) else float("nan")

        rot_flip_he, rot_other, _ = _flip_at_kl_budget(sweep_rot, "flip_he_pct", args.kl_budget)
        star_flip_he, star_other, _ = _flip_at_kl_budget(sweep_star, "flip_he_pct", args.kl_budget)

        rows_out.append(Row(
            run_name=run_name,
            json_path=str(jp),
            k=k,
            w_mode=w_mode,
            lda_reg=None if math.isnan(lda_reg_f) else lda_reg_f,
            max_abs_w=max_abs_w,
            top3_abs_w=top3_abs_w,
            rot_kl95_he=rot_kl95_he,
            star_kl95_he=star_kl95_he,
            d_kl95_he=d_kl95_he,
            rot_flip_he_at_budget=rot_flip_he,
            star_flip_he_at_budget=star_flip_he,
            rot_other_at_budget=rot_other,
            star_other_at_budget=star_other,
        ))

    # Sort by best (lowest) ROT_KL@95% he flips
    rows_out.sort(key=lambda r: float("inf") if math.isnan(r.rot_kl95_he) else r.rot_kl95_he)

    print(f"\n[SUMMARY] runs={len(rows_out)} flip_target={args.flip_target}% kl_budget={args.kl_budget}")
    print("run_name\tk\tw_mode\tlda_reg\tmax|w|\tROT_KL@95(he)\tSTAR_KL@95(he)\tΔKL\tROT_flip_he@KL\tSTAR_flip_he@KL\tROT_other@KL\tSTAR_other@KL\ttop3|w|")

    def fmt(x: Any) -> str:
        if x is None:
            return ""
        if isinstance(x, float):
            if math.isnan(x):
                return "nan"
            return f"{x:.4g}"
        return str(x)

    for r in rows_out[: args.topn]:
        print(
            f"{r.run_name}\t{r.k}\t{r.w_mode}\t{fmt(r.lda_reg)}\t{fmt(r.max_abs_w)}\t"
            f"{fmt(r.rot_kl95_he)}\t{fmt(r.star_kl95_he)}\t{fmt(r.d_kl95_he)}\t"
            f"{fmt(r.rot_flip_he_at_budget)}\t{fmt(r.star_flip_he_at_budget)}\t"
            f"{fmt(r.rot_other_at_budget)}\t{fmt(r.star_other_at_budget)}\t"
            f"{r.top3_abs_w}"
        )

    if args.save_csv:
        outp = Path(args.save_csv)
        outp.parent.mkdir(parents=True, exist_ok=True)
        with outp.open("w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow([
                "run_name","json_path","k","w_mode","lda_reg",
                "max_abs_w","top3_abs_w",
                "rot_kl95_he","star_kl95_he","d_kl95_he",
                "rot_flip_he_at_budget","star_flip_he_at_budget",
                "rot_other_at_budget","star_other_at_budget",
            ])
            for r in rows_out:
                w.writerow([
                    r.run_name,r.json_path,r.k,r.w_mode,r.lda_reg,
                    r.max_abs_w,r.top3_abs_w,
                    r.rot_kl95_he,r.star_kl95_he,r.d_kl95_he,
                    r.rot_flip_he_at_budget,r.star_flip_he_at_budget,
                    r.rot_other_at_budget,r.star_other_at_budget,
                ])
        print(f"\n[SAVE_CSV] {outp}")


if __name__ == "__main__":
    main()
