#!/usr/bin/env python3
"""gp_exp3_summarize_sweep.py

Experiment 3 (post-processing, no retraining / no forward passes):
Summarize Exp2 sweep outputs and compute the headline metrics.

Input: exp2_results.json files created by gp_exp2_rotate_subspace_v5_sweep.py
(default location: <out_dir>/exp2_rotate/**/exp2_results.json)

Outputs:
- prints a ranked table of runs (rotated vs STAR) with:
  * KL@flip_target (he->she and she->he)
  * flip@KL_target (he->she and she->he)
  * other%@KL_target
  * w concentration: max|w| and top-3 weights (idx, w)

This script is intentionally lightweight so you can run it after a long sweep and
paste one consolidated summary into another window.
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from typing import Any, Dict, List, Optional, Tuple


def _safe_float(x: Any) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _topk_weights(idx_list: List[int], w: List[float], k: int = 3) -> List[Tuple[int, float]]:
    pairs = list(zip(idx_list, w))
    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    return pairs[:k]


def _kl_at_flip_target(
    sweep: List[Dict[str, Any]],
    flip_key: str,
    flip_target: float,
) -> Optional[float]:
    # Return the minimum KL among points that reach the flip target.
    best: Optional[float] = None
    for row in sweep:
        flip = _safe_float(row.get(flip_key))
        kl = _safe_float(row.get("kl_mean"))
        if flip is None or kl is None:
            continue
        if flip >= flip_target:
            if best is None or kl < best:
                best = kl
    return best


def _interp_y_at_x(points: List[Tuple[float, float]], x: float) -> Optional[float]:
    """Linear interpolation for y(x) given (x_i, y_i) points.

    points: list of (x, y) pairs. x doesn't have to be sorted.
    If x is outside range, we clamp to nearest endpoint.
    """
    if not points:
        return None
    pts = sorted(points, key=lambda t: t[0])
    # If duplicates in x, keep the one with larger y? (doesn't matter much). We'll just keep all.
    if x <= pts[0][0]:
        return pts[0][1]
    if x >= pts[-1][0]:
        return pts[-1][1]
    for (x1, y1), (x2, y2) in zip(pts[:-1], pts[1:]):
        if x1 <= x <= x2:
            if math.isclose(x1, x2):
                return (y1 + y2) / 2.0
            t = (x - x1) / (x2 - x1)
            return y1 + t * (y2 - y1)
    return None


def _flip_at_kl_target(
    sweep: List[Dict[str, Any]],
    flip_key: str,
    kl_target: float,
) -> Optional[float]:
    pts: List[Tuple[float, float]] = []
    for row in sweep:
        flip = _safe_float(row.get(flip_key))
        kl = _safe_float(row.get("kl_mean"))
        if flip is None or kl is None:
            continue
        pts.append((kl, flip))
    return _interp_y_at_x(pts, kl_target)


def _other_at_kl_target(sweep: List[Dict[str, Any]], kl_target: float) -> Optional[float]:
    pts: List[Tuple[float, float]] = []
    for row in sweep:
        other = _safe_float(row.get("other_pct"))
        kl = _safe_float(row.get("kl_mean"))
        if other is None or kl is None:
            continue
        pts.append((kl, other))
    return _interp_y_at_x(pts, kl_target)


def _fmt(x: Optional[float], nd: int = 3) -> str:
    if x is None:
        return "NA"
    return f"{x:.{nd}f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, required=True, help="Same out_dir you used for Exp2 (e.g., outputs/gp)")
    ap.add_argument("--flip_target", type=float, default=95.0, help="Flip%% target for KL@flip (default 95)")
    # Allow both names so older/newer notebooks keep working.
    ap.add_argument(
        "--kl_target",
        "--kl_budget",
        dest="kl_target",
        type=float,
        default=2.0,
        help="KL budget for flip@KL (default 2.0). Alias: --kl_budget",
    )
    ap.add_argument(
        "--search_root",
        type=str,
        default=None,
        help="Root to search for exp2_results.json (default: <out_dir>/exp2_rotate)",
    )
    ap.add_argument(
        "--max_rows",
        "--topn",
        dest="max_rows",
        type=int,
        default=50,
        help="How many runs to print (default 50). Alias: --topn",
    )
    ap.add_argument(
        "--save_csv",
        type=str,
        default=None,
        help="If set, save the printed summary table to this CSV path.",
    )
    args = ap.parse_args()

    search_root = args.search_root or os.path.join(args.out_dir, "exp2_rotate")
    pattern = os.path.join(search_root, "**", "exp2_results.json")
    paths = sorted(glob.glob(pattern, recursive=True))
    if not paths:
        raise SystemExit(f"No exp2_results.json found under: {pattern}")

    rows: List[Dict[str, Any]] = []

    for p in paths:
        try:
            d = _load_json(p)
        except Exception:
            continue

        cfg = {
            "path": p,
            "layer": d.get("layer"),
            "head": d.get("head"),
            "k": d.get("k"),
            "w_mode": d.get("w_mode"),
            "lda_reg": d.get("lda_reg"),
            "lda_shrink": d.get("lda_shrink"),
            "run_name": d.get("run_name"),
            "selection_mode": d.get("selection_mode"),
        }

        idx_list = d.get("idx_list") or []
        w = d.get("w") or []
        # Be robust: sometimes idx_list may be saved as strings.
        try:
            idx_list = [int(x) for x in idx_list]
        except Exception:
            idx_list = list(idx_list)
        try:
            w = [float(x) for x in w]
        except Exception:
            w = list(w)

        max_abs_w = max((abs(x) for x in w), default=float("nan"))
        top3 = _topk_weights(idx_list, w, k=3) if idx_list and w else []

        sweep_rot = d.get("sweep_rotated") or []
        sweep_star = d.get("sweep_star") or []

        rot_kl95_he = _kl_at_flip_target(sweep_rot, "flip_he_pct", args.flip_target)
        rot_kl95_she = _kl_at_flip_target(sweep_rot, "flip_she_pct", args.flip_target)
        star_kl95_he = _kl_at_flip_target(sweep_star, "flip_he_pct", args.flip_target)
        star_kl95_she = _kl_at_flip_target(sweep_star, "flip_she_pct", args.flip_target)

        rot_flip_at_kl_he = _flip_at_kl_target(sweep_rot, "flip_he_pct", args.kl_target)
        rot_flip_at_kl_she = _flip_at_kl_target(sweep_rot, "flip_she_pct", args.kl_target)
        star_flip_at_kl_he = _flip_at_kl_target(sweep_star, "flip_he_pct", args.kl_target)
        star_flip_at_kl_she = _flip_at_kl_target(sweep_star, "flip_she_pct", args.kl_target)

        rot_other_at_kl = _other_at_kl_target(sweep_rot, args.kl_target)
        star_other_at_kl = _other_at_kl_target(sweep_star, args.kl_target)

        # Aggregate for ranking: worst-case KL to reach flip_target (lower is better).
        rot_kl95_max = None
        if rot_kl95_he is not None and rot_kl95_she is not None:
            rot_kl95_max = max(rot_kl95_he, rot_kl95_she)
        star_kl95_max = None
        if star_kl95_he is not None and star_kl95_she is not None:
            star_kl95_max = max(star_kl95_he, star_kl95_she)

        rows.append(
            {
                **cfg,
                "max_abs_w": max_abs_w,
                "top3": top3,
                "rot_kl95_he": rot_kl95_he,
                "rot_kl95_she": rot_kl95_she,
                "rot_kl95_max": rot_kl95_max,
                "star_kl95_he": star_kl95_he,
                "star_kl95_she": star_kl95_she,
                "star_kl95_max": star_kl95_max,
                "rot_flip_at_kl_he": rot_flip_at_kl_he,
                "rot_flip_at_kl_she": rot_flip_at_kl_she,
                "star_flip_at_kl_he": star_flip_at_kl_he,
                "star_flip_at_kl_she": star_flip_at_kl_she,
                "rot_other_at_kl": rot_other_at_kl,
                "star_other_at_kl": star_other_at_kl,
            }
        )

    # Rank: prefer runs with a defined rot_kl95_max; then smaller rot_kl95_max.
    def _rank_key(r: Dict[str, Any]) -> Tuple[int, float]:
        kl = r.get("rot_kl95_max")
        if kl is None:
            return (1, float("inf"))
        return (0, float(kl))

    rows.sort(key=_rank_key)

    print("\n===== EXP3 SUMMARY (from Exp2 JSONs) =====")
    print(f"Found {len(rows)} runs under: {search_root}")
    print(f"Flip target for KL@flip: {args.flip_target}%")
    print(f"KL target for flip@KL:   {args.kl_target}")

    kl_lab = str(args.kl_target)
    header = (
        "k  w_mode   lda_reg   run_name\t"
        "ROT: KL@95(he,she,max)\tROT: flip@KL" + kl_lab + "(he,she)\tROT other@KL" + kl_lab + "\t"
        "STAR: KL@95(he,she,max)\tSTAR: flip@KL" + kl_lab + "(he,she)\tSTAR other@KL" + kl_lab + "\t"
        "max|w|  top3(idx:w)"
    )
    print("\n" + header)
    print("-" * len(header))

    for r in rows[: args.max_rows]:
        lda_reg = r.get("lda_reg")
        lda_reg_s = "NA" if lda_reg is None else str(lda_reg)
        top3_s = ",".join([f"{idx}:{w:+.3f}" for idx, w in (r.get("top3") or [])])

        line = (
            f"{str(r.get('k')).rjust(2)} "
            f"{str(r.get('w_mode')).ljust(8)} "
            f"{lda_reg_s.ljust(8)} "
            f"{str(r.get('run_name'))}\t"
            f"({_fmt(r.get('rot_kl95_he'))},{_fmt(r.get('rot_kl95_she'))},{_fmt(r.get('rot_kl95_max'))})\t"
            f"({_fmt(r.get('rot_flip_at_kl_he'),2)},{_fmt(r.get('rot_flip_at_kl_she'),2)})\t"
            f"{_fmt(r.get('rot_other_at_kl'),2)}\t"
            f"({_fmt(r.get('star_kl95_he'))},{_fmt(r.get('star_kl95_she'))},{_fmt(r.get('star_kl95_max'))})\t"
            f"({_fmt(r.get('star_flip_at_kl_he'),2)},{_fmt(r.get('star_flip_at_kl_she'),2)})\t"
            f"{_fmt(r.get('star_other_at_kl'),2)}\t"
            f"{_fmt(r.get('max_abs_w'),3)}  {top3_s}"
        )
        print(line)

    if args.save_csv:
        os.makedirs(os.path.dirname(args.save_csv) or ".", exist_ok=True)
        import csv

        # Save full sorted list (not truncated) so you can filter in Excel later.
        with open(args.save_csv, "w", newline="", encoding="utf-8") as f:
            wtr = csv.writer(f)
            wtr.writerow(
                [
                    "k",
                    "w_mode",
                    "lda_reg",
                    "lda_shrink",
                    "run_name",
                    "rot_kl95_he",
                    "rot_kl95_she",
                    "rot_kl95_max",
                    f"rot_flip_at_kl_{args.kl_target}_he",
                    f"rot_flip_at_kl_{args.kl_target}_she",
                    f"rot_other_at_kl_{args.kl_target}",
                    "star_kl95_he",
                    "star_kl95_she",
                    "star_kl95_max",
                    f"star_flip_at_kl_{args.kl_target}_he",
                    f"star_flip_at_kl_{args.kl_target}_she",
                    f"star_other_at_kl_{args.kl_target}",
                    "max_abs_w",
                    "top3",
                    "path",
                ]
            )
            for r in rows:
                top3_s = ",".join([f"{idx}:{w:+.6f}" for idx, w in (r.get("top3") or [])])
                wtr.writerow(
                    [
                        r.get("k"),
                        r.get("w_mode"),
                        r.get("lda_reg"),
                        r.get("lda_shrink"),
                        r.get("run_name"),
                        r.get("rot_kl95_he"),
                        r.get("rot_kl95_she"),
                        r.get("rot_kl95_max"),
                        r.get("rot_flip_at_kl_he"),
                        r.get("rot_flip_at_kl_she"),
                        r.get("rot_other_at_kl"),
                        r.get("star_kl95_he"),
                        r.get("star_kl95_she"),
                        r.get("star_kl95_max"),
                        r.get("star_flip_at_kl_he"),
                        r.get("star_flip_at_kl_she"),
                        r.get("star_other_at_kl"),
                        r.get("max_abs_w"),
                        top3_s,
                        r.get("path"),
                    ]
                )
        print(f"\n[SAVE] wrote CSV summary to: {args.save_csv}")

    print("\nNotes:")
    print("- KL@95 uses the *minimum* KL among your sampled sigma_scales that reaches >=95% flips.")
    print("  If your scales are coarse, this is a step-function estimate (good enough for ranking).")
    print(
        f"- flip@KL{args.kl_target} linearly interpolates flip% as a function of KL across your sampled points."
    )


if __name__ == "__main__":
    main()
