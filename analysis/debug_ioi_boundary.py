#!/usr/bin/env python3
"""
Tiny debug utility: IOI boundary correctness (tab leak) check.

It does NOT require a run_dir. Just point it at an IOI CSV.

Example:
  python analysis/debug_ioi_boundary.py --ioi_csv data_main/train_ioi.csv --out sanity_ioi_boundary.txt --n 20
"""
from __future__ import annotations

import argparse
import os
import random
from typing import Any, Dict, List

import pandas as pd


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def sanity_ioi_boundary(ioi_csv_path: str, out_path: str, n: int = 20, seed: int = 0) -> Dict[str, Any]:
    rng = random.Random(seed)
    df = pd.read_csv(ioi_csv_path)

    for c in ["ioi_sentences_input", "ioi_sentences_labels"]:
        if c not in df.columns:
            raise ValueError(f"Missing column {c} in {ioi_csv_path}")

    idxs = list(range(len(df)))
    rng.shuffle(idxs)
    idxs = idxs[: min(n, len(idxs))]

    lines: List[str] = []
    tab_rows = 0
    split_mismatch = 0
    leak_at_end = 0

    for i in idxs:
        inp = str(df.loc[i, "ioi_sentences_input"])
        lab = str(df.loc[i, "ioi_sentences_labels"])

        lines.append(f"ROW {i}")
        lines.append(f"  input_repr: {inp!r}")
        lines.append(f"  label_repr: {lab!r}")

        if "\t" in inp:
            tab_rows += 1
            prompt_part, label_part = inp.split("\t", 1)

            if label_part.strip() != lab.strip():
                split_mismatch += 1
                lines.append(f"  TAB_SPLIT_MISMATCH: label_part.strip()={label_part.strip()!r} vs label_col={lab.strip()!r}")

            if prompt_part.rstrip().endswith(lab.strip()):
                leak_at_end += 1
                lines.append("  LEAK_AT_END: label appears at end of prompt_part")

            lines.append(f"  prompt_part_repr: {prompt_part!r}")
            lines.append(f"  label_part_repr: {label_part!r}")

        lines.append("")

    passed = (split_mismatch == 0) and (leak_at_end == 0)
    summary = {
        "ioi_csv": ioi_csv_path,
        "checked_rows": len(idxs),
        "rows_with_tab": tab_rows,
        "tab_split_mismatch_rows": split_mismatch,
        "label_leak_at_end_rows": leak_at_end,
        "passed": passed,
    }

    lines.append("SUMMARY")
    for k, v in summary.items():
        lines.append(f"{k}: {v}")

    _write_text(out_path, "\n".join(lines))
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ioi_csv", type=str, required=True)
    p.add_argument("--out", type=str, default="sanity_ioi_boundary.txt")
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    res = sanity_ioi_boundary(args.ioi_csv, out_path=args.out, n=args.n, seed=args.seed)
    print(f"[OK] wrote: {args.out}")
    print(f"PASS={res['passed']}  rows_with_tab={res['rows_with_tab']}  split_mismatch={res['tab_split_mismatch_rows']}  leak_at_end={res['label_leak_at_end_rows']}")


if __name__ == "__main__":
    main()
