from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import matplotlib.pyplot as plt

from transformers import AutoTokenizer  # lightweight, avoid loading full model

from .data import PromptDataset, load_task_examples
from .utils import json_dump


def _load_history(metrics_path: Path):
    obj = json.loads(metrics_path.read_text(encoding="utf-8"))

    # Our code saves train_result dict that usually has "history"
    if isinstance(obj, dict) and "history" in obj:
        return obj["history"], obj
    # Fallback: sometimes history itself might be dumped
    if isinstance(obj, list):
        return obj, {"history": obj}
    raise ValueError(f"Unrecognized metrics.json structure: {metrics_path}")


def _extract_series(history, split: str, key: str):
    # history entries in separate mode:
    # {"epoch": e, "train": {...}, "val": {...}, "l1_mean_mask": ...}
    out = []
    for rec in history:
        if split in rec and isinstance(rec[split], dict) and key in rec[split]:
            out.append(float(rec[split][key]))
        else:
            out.append(float("nan"))
    return out


def _extract_l1(history):
    out = []
    for rec in history:
        if "l1_mean_mask" in rec:
            out.append(float(rec["l1_mean_mask"]))
        elif "l1" in rec:
            out.append(float(rec["l1"]))
        else:
            out.append(float("nan"))
    return out


def _mask_stats(mask_values: torch.Tensor) -> Dict[str, float]:
    x = mask_values.flatten().float()
    qs = [0.0, 0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 1.0]
    quant = torch.quantile(x, torch.tensor(qs, device=x.device)).cpu().tolist()

    def frac_lt(t): return float((x < t).float().mean().item())
    def frac_gt(t): return float((x > t).float().mean().item())

    return {
        "min": float(quant[0]),
        "p1": float(quant[1]),
        "p5": float(quant[2]),
        "p10": float(quant[3]),
        "p25": float(quant[4]),
        "p50": float(quant[5]),
        "p75": float(quant[6]),
        "p90": float(quant[7]),
        "p95": float(quant[8]),
        "p99": float(quant[9]),
        "max": float(quant[10]),
        "frac_lt_0.1": frac_lt(0.1),
        "frac_lt_0.01": frac_lt(0.01),
        "frac_gt_0.9": frac_gt(0.9),
        "frac_gt_0.99": frac_gt(0.99),
    }


def _plot_curves(epochs, train_vals, val_vals, outpath: Path, title: str, ylabel: str):
    plt.figure()
    plt.plot(epochs, train_vals, label="train")
    plt.plot(epochs, val_vals, label="val")
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def _plot_l1(epochs, l1_vals, outpath: Path):
    plt.figure()
    plt.plot(epochs, l1_vals)
    plt.xlabel("epoch")
    plt.ylabel("L1_mean_mask")
    plt.title("Mask sparsity (mean(sigmoid(theta)))")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def _plot_hist(mask_values: torch.Tensor, outpath: Path):
    x = mask_values.flatten().float().cpu().numpy()
    plt.figure()
    plt.hist(x, bins=50)
    plt.xlabel("mask value (sigmoid(theta))")
    plt.ylabel("count")
    plt.title("Mask value distribution")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def _load_selected(selected_path: Path) -> Dict[str, list]:
    obj = json.loads(selected_path.read_text(encoding="utf-8"))
    # expected: selected["l{l}_h{h}"]["ranked_indices"] = [...]
    out = {}
    for k, v in obj.items():
        out[k] = v["ranked_indices"]
    return out


def _topk_overlap(sel_a: Dict[str, list], sel_b: Dict[str, list], k: int) -> Dict[str, float]:
    overlaps = {}
    for head_key in sel_a.keys():
        if head_key not in sel_b:
            continue
        A = set(sel_a[head_key][:k])
        B = set(sel_b[head_key][:k])
        overlaps[head_key] = float(len(A & B) / max(1, k))
    return overlaps


def _multi_token_rate_from_config(config: dict, run_dir: Path) -> Dict[str, dict]:
    """
    Recompute multi-token label counts/rates using PromptDataset logic.
    This answers: "Is multi-token label rate non-trivial?"
    """
    train_cfg = config.get("train_config", {})
    model_name = train_cfg.get("model_name", "gpt2")

    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    data_dir = Path(config.get("data_dir", "data_main"))

    mode = config.get("mode", "separate")
    results = {}

    if mode == "joint":
        csvs = config["csvs"]
        for task, spec in csvs.items():
            train_csv = spec["train"]
            examples = load_task_examples(task, data_dir / train_csv)
            ds = PromptDataset(examples, tokenizer=tok, task=task)
            N = len(ds)
            counts = ds.multi_token_warnings
            results[task] = {
                "N": N,
                "counts": counts,
                "rates": {k: float(v / max(1, N)) for k, v in counts.items()},
            }
        return results

    # separate
    task = config["task"]
    train_csv = config["train_csv"]
    examples = load_task_examples(task, data_dir / train_csv)
    ds = PromptDataset(examples, tokenizer=tok, task=task)
    N = len(ds)
    counts = ds.multi_token_warnings
    results[task] = {
        "N": N,
        "counts": counts,
        "rates": {k: float(v / max(1, N)) for k, v in counts.items()},
    }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", type=str, required=True)
    ap.add_argument("--compare_run_dir", type=str, default=None)
    ap.add_argument("--topk", type=int, default=50)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    out_dir = run_dir / "posthoc"
    out_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))

    history, metrics_obj = _load_history(run_dir / "metrics.json")
    epochs = [int(rec.get("epoch", i)) for i, rec in enumerate(history)]

    train_kl = _extract_series(history, "train", "kl_total")
    val_kl = _extract_series(history, "val", "kl_total")
    l1_vals = _extract_l1(history)

    _plot_curves(epochs, train_kl, val_kl, out_dir / "kl_curve.png", "KL_total over epochs", "KL_total")
    _plot_l1(epochs, l1_vals, out_dir / "l1_curve.png")

    mask_path = run_dir / "masks" / "mask_values.pt"
    mask_values = torch.load(mask_path, map_location="cpu")
    _plot_hist(mask_values, out_dir / "mask_hist.png")
    ms = _mask_stats(mask_values)
    json_dump(ms, out_dir / "mask_stats.json")

    # Multi-token label rate recompute
    mtr = _multi_token_rate_from_config(config, run_dir)
    json_dump(mtr, out_dir / "multi_token_report.json")

    report = {
        "run_dir": str(run_dir),
        "epochs_ran": len(history),
        "final_train_kl": float(train_kl[-1]) if len(train_kl) else None,
        "final_val_kl": float(val_kl[-1]) if len(val_kl) else None,
        "final_l1": float(l1_vals[-1]) if len(l1_vals) else None,
        "mask_stats": ms,
        "multi_token_report": mtr,
    }

    # Selected overlap if compare is given
    if args.compare_run_dir is not None:
        other = Path(args.compare_run_dir)
        sel_a = _load_selected(run_dir / "selected.json")
        sel_b = _load_selected(other / "selected.json")
        overlaps = _topk_overlap(sel_a, sel_b, args.topk)
        json_dump(overlaps, out_dir / "selected_overlap.json")
        report["compare_run_dir"] = str(other)
        report["topk"] = int(args.topk)
        report["mean_overlap"] = float(sum(overlaps.values()) / max(1, len(overlaps)))

    json_dump(report, out_dir / "report.json")
    print(f"[posthoc] wrote outputs to: {out_dir}")


if __name__ == "__main__":
    main()
