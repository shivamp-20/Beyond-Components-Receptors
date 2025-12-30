#!/usr/bin/env python3
"""
Analyze an OV-mask training run directory.

This script is intentionally simple and read-only: it reads saved artifacts from
a run_dir and writes analysis outputs under <save_dir> (default: <run_dir>/analysis/).

Typical usage (separate task):
  python analysis/analyze_ov_mask_run.py \
    --run_dir runs/gp/ov_mask_YYYYMMDD_HHMMSS \
    --task gp \
    --data_dir data_main \
    --data_csv train_1k_gp.csv val_gp.csv test_gp.csv

Typical usage (joint):
  python analysis/analyze_ov_mask_run.py --run_dir runs/joint/ov_mask_... --task joint
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

# headless-safe plotting
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# I/O helpers
# -----------------------------
def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _quantile(x: np.ndarray, q: float) -> float:
    return float(np.quantile(x, q))


def _resolve_csv_path(data_dir: str, maybe_rel: str) -> str:
    return maybe_rel if os.path.isabs(maybe_rel) else os.path.join(data_dir, maybe_rel)


# -----------------------------
# Artifact loading
# -----------------------------
def _load_mask_values(run_dir: str) -> torch.Tensor:
    """
    Returns mask values m in [0,1] with shape [L,H,r] on CPU float32.

    Preference order:
      1) masks/mask_values.pt
      2) masks/best_mask_logits.pt -> sigmoid
      3) masks/mask_logits.pt      -> sigmoid
    """
    mv = os.path.join(run_dir, "masks", "mask_values.pt")
    if os.path.exists(mv):
        return torch.load(mv, map_location="cpu").float()

    bl = os.path.join(run_dir, "masks", "best_mask_logits.pt")
    if os.path.exists(bl):
        theta = torch.load(bl, map_location="cpu").float()
        return torch.sigmoid(theta)

    ml = os.path.join(run_dir, "masks", "mask_logits.pt")
    if os.path.exists(ml):
        theta = torch.load(ml, map_location="cpu").float()
        return torch.sigmoid(theta)

    raise FileNotFoundError("No mask values found under run_dir/masks/ (mask_values.pt or (best_)mask_logits.pt).")


def _load_singular_values_from_cache(svd_cache_dir: str, L: int, H: int, r: int) -> torch.Tensor:
    """
    Loads singular values S[l,h,:] from:
      <svd_cache_dir>/svd/l{l}_h{h}_S.pt
    into CPU float32 tensor [L,H,r].
    """
    S_all = torch.empty((L, H, r), dtype=torch.float32, device="cpu")
    for l in range(L):
        for h in range(H):
            p = os.path.join(svd_cache_dir, "svd", f"l{l}_h{h}_S.pt")
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing SVD file: {p}")
            S = torch.load(p, map_location="cpu").float().reshape(-1)
            if S.numel() != r:
                raise ValueError(f"Bad S shape at l={l},h={h}: got {tuple(S.shape)}, expected ({r},)")
            S_all[l, h] = S
    return S_all


# -----------------------------
# Math helpers
# -----------------------------
def _effective_rank(weights: torch.Tensor, tau: float = 0.9) -> int:
    """
    Effective rank definition used here:
      sort w_i desc,
      smallest k such that sum_{i<=k} w_i / sum_i w_i >= tau
    """
    w = weights.detach().float().cpu()
    total = float(w.sum().item())
    if total <= 0.0:
        return 0
    w_sorted, _ = torch.sort(w, descending=True)
    c = torch.cumsum(w_sorted, dim=0) / total
    idx = int(torch.searchsorted(c, torch.tensor(tau)).item())
    return min(idx + 1, w.numel())


# -----------------------------
# Plotting
# -----------------------------
def _plot_lines(x: List[int], series: Dict[str, List[float]], title: str, ylabel: str, out_path: str) -> None:
    plt.figure()
    for name, y in series.items():
        if y is None or len(y) == 0:
            continue
        plt.plot(x, y, label=name)
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path)
    plt.close()


def _plot_hist(values: np.ndarray, title: str, xlabel: str, out_path: str, bins: int = 60) -> None:
    plt.figure()
    plt.hist(values, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("count")
    plt.tight_layout()
    _ensure_dir(os.path.dirname(out_path))
    plt.savefig(out_path)
    plt.close()


# -----------------------------
# Sanity checks
# -----------------------------
def sanity_ioi_boundary(ioi_csv_path: str, out_path: str, n: int = 20, seed: int = 0) -> Dict[str, Any]:
    """
    Checks for IOI tab leakage (answer token embedded in prompt).
    Writes details to out_path.
    """
    import pandas as pd

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


def sanity_tokenization(task: str, csv_triplet: Tuple[str, str, str], data_dir: str, out_path: str,
                       max_examples: int = 200) -> Dict[str, Any]:
    """
    Checks % of labels that are single-token under GPT-2 tokenizer for:
      encode(" "+label_str)
    Writes a compact report to out_path.
    """
    from transformers import GPT2TokenizerFast
    from ovmask.data import load_task_csv

    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    def check_split(split_name: str, csv_name: str) -> Dict[str, Any]:
        path = _resolve_csv_path(data_dir, csv_name)
        exs = load_task_csv(task, path)
        n = min(max_examples, len(exs))

        lens = []
        multi = []
        for i in range(n):
            lab = str(exs[i].label_clean_str)
            ids = tok.encode(" " + lab, add_special_tokens=False)
            lens.append(len(ids))
            if len(ids) != 1 and len(multi) < 10:
                multi.append({
                    "label": lab,
                    "token_ids": ids,
                    "token_strs": [tok.decode([t]) for t in ids],
                })

        lens = np.asarray(lens, dtype=np.int64)
        pct1 = float((lens == 1).mean() * 100.0) if lens.size else 0.0
        return {"split": split_name, "n_checked": int(n), "pct_single_token": pct1, "multi_token_examples": multi}

    train_csv, val_csv, test_csv = csv_triplet
    out = {
        "task": task,
        "train": check_split("train", train_csv),
        "val": check_split("val", val_csv),
        "test": check_split("test", test_csv),
    }

    lines = []
    for split in ["train", "val", "test"]:
        s = out[split]
        lines.append(f"{task}/{split}: n={s['n_checked']}  pct_len1={s['pct_single_token']:.2f}%")
    lines.append("")
    for split in ["train", "val", "test"]:
        s = out[split]
        if not s["multi_token_examples"]:
            continue
        lines.append(f"Multi-token examples ({task}/{split}):")
        for ex in s["multi_token_examples"]:
            lines.append(f"  label={ex['label']!r}  token_ids={ex['token_ids']}  token_strs={ex['token_strs']}")
        lines.append("")
    _write_text(out_path, "\n".join(lines))
    return out


@torch.no_grad()
def sanity_all_ones_kl(cfg: Dict[str, Any], task: str, csv_triplet: Tuple[str, str, str],
                       data_dir: str, device: str, out_path: str, n_batch: int = 32) -> Dict[str, Any]:
    """
    All-ones mask invariance check:
      set theta=+20 everywhere => m≈1,
      run masked model on a small VAL batch,
      compare to teacher logp cache and compute KL exactly like training.

    PASS heuristic: KL_total < 1e-4
    """
    from transformer_lens import HookedTransformer
    from ovmask.data import load_task_csv
    from ovmask.svd_cache import SVDSpec, load_all_svd
    from ovmask.masked_ov import freeze_model_params, build_mask_state, MaskedOVHook
    from ovmask.teacher_cache import TeacherShardReader
    from ovmask.train_eval import forward_logp_next, kl_from_teacher_logp
    from ovmask.utils import dtype_from_str

    dev = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
    dtype = dtype_from_str(str(cfg.get("dtype", "float32")))

    svd_cache_dir = str(cfg.get("svd_cache_dir", "svd_cache/gpt2"))
    meta = _read_json(os.path.join(svd_cache_dir, "svd_meta.json"))
    spec = SVDSpec(
        model_name=meta["model_name"],
        n_layers=int(meta["n_layers"]),
        n_heads=int(meta["n_heads"]),
        d_head=int(meta["d_head"]),
        d_model=int(meta["d_model"]),
        r=int(meta["r"]),
    )
    U, S, Vh = load_all_svd(svd_cache_dir, spec, device=dev, dtype=dtype)

    student = HookedTransformer.from_pretrained("gpt2", device=dev, dtype=dtype)
    from ovmask.masked_ov import freeze_model_params
    freeze_model_params(student)
    student.eval()

    state = build_mask_state(U, S, Vh, init_theta=4.0, device=dev)
    state.theta.data.fill_(20.0)  # force m≈1
    hook = MaskedOVHook(state).to(dev)

    student.reset_hooks()
    for l in range(spec.n_layers):
        student.add_hook(f"blocks.{l}.attn.hook_z", hook.hook_store_z, dir="fwd")
        student.add_hook(f"blocks.{l}.hook_attn_out", hook.hook_replace_attn_out, dir="fwd")

    train_csv, val_csv, test_csv = csv_triplet
    exs = load_task_csv(task, _resolve_csv_path(data_dir, val_csv))
    n = min(n_batch, len(exs))
    batch = exs[:n]

    idx = torch.tensor([e.idx for e in batch], device=dev, dtype=torch.long)
    prompts_clean = [e.prompt_clean for e in batch]
    prompts_corr = [e.prompt_corr for e in batch]

    teacher_cache_dir = str(cfg.get("teacher_cache_dir", "teacher_cache"))
    t_clean = TeacherShardReader(teacher_cache_dir, task, "val", "clean")
    t_corr = TeacherShardReader(teacher_cache_dir, task, "val", "corr")

    logp_clean = forward_logp_next(student, prompts_clean, device=dev)
    logp_corr = forward_logp_next(student, prompts_corr, device=dev)
    tp_clean = t_clean.get(idx, device=dev)
    tp_corr = t_corr.get(idx, device=dev)

    kl_clean = float(kl_from_teacher_logp(tp_clean, logp_clean).item())
    kl_corr = float(kl_from_teacher_logp(tp_corr, logp_corr).item())
    kl_total = 0.5 * (kl_clean + kl_corr)
    passed = kl_total < 1e-4

    txt = (
        f"All-ones mask invariance check (theta=+20 => m≈1)\n"
        f"task={task} split=val n={n}\n"
        f"KL_clean={kl_clean:.8e}\n"
        f"KL_corr={kl_corr:.8e}\n"
        f"KL_total={kl_total:.8e}\n"
        f"PASS={passed} (threshold KL_total < 1e-4)\n"
    )
    _write_text(out_path, txt)
    return {"KL_clean": kl_clean, "KL_corr": kl_corr, "KL_total": kl_total, "passed": passed}


# -----------------------------
# Main analysis
# -----------------------------
def analyze_run(
    run_dir: str,
    task: str,
    save_dir: Optional[str],
    n_examples_debug: int,
    device: str,
    data_dir: Optional[str],
    data_csv: Optional[Tuple[str, str, str]],
) -> None:
    run_dir = os.path.abspath(run_dir)
    save_dir = os.path.abspath(save_dir or os.path.join(run_dir, "analysis"))
    plots_dir = os.path.join(save_dir, "plots")
    _ensure_dir(plots_dir)

    cfg = _read_json(os.path.join(run_dir, "config.json"))
    metrics = _read_json(os.path.join(run_dir, "metrics.json"))
    final_eval_path = os.path.join(run_dir, "final_eval.json")
    final_eval = _read_json(final_eval_path) if os.path.exists(final_eval_path) else None

    if data_dir is None:
        data_dir = str(cfg.get("data_dir", "data_main"))

    # infer csv from config when possible (separate)
    if data_csv is None and task != "joint":
        if all(k in cfg for k in ["train_csv", "val_csv", "test_csv"]):
            data_csv = (cfg["train_csv"], cfg["val_csv"], cfg["test_csv"])

    # ----- curves from metrics.json -----
    if not isinstance(metrics, list):
        raise ValueError("metrics.json must be list[dict].")

    joint_mode = len(metrics) > 0 and ("KL_total_mean" in metrics[0])

    epochs: List[int] = []
    val_kl_total: List[float] = []
    train_kl: List[float] = []
    val_l1: List[float] = []
    train_l1: List[float] = []
    val_kl_clean: List[float] = []
    val_kl_corr: List[float] = []

    if joint_mode:
        for row in metrics:
            epochs.append(int(row.get("epoch", len(epochs) + 1)))
            val_kl_total.append(float(row.get("KL_total_mean", float("nan"))))
            # shared mask -> L1 can be read from any per-task dict if present
            if isinstance(row.get("gp"), dict):
                val_l1.append(float(row["gp"].get("L1_mean_mask", float("nan"))))
            else:
                val_l1.append(float("nan"))
    else:
        for row in metrics:
            epochs.append(int(row.get("epoch", len(epochs) + 1)))
            train_kl.append(float(row.get("train_KL", float("nan"))))
            val_kl_total.append(float(row.get("val_KL_total", float("nan"))))
            train_l1.append(float(row.get("train_L1_mean", float("nan"))))
            val_l1.append(float(row.get("val_L1_mean_mask", float("nan"))))
            if "val_KL_clean" in row:
                val_kl_clean.append(float(row.get("val_KL_clean", float("nan"))))
            if "val_KL_corr" in row:
                val_kl_corr.append(float(row.get("val_KL_corr", float("nan"))))

    best_i = int(np.nanargmin(np.asarray(val_kl_total))) if val_kl_total else 0
    best_epoch = int(epochs[best_i]) if epochs else None
    best_val_kl = float(np.nanmin(np.asarray(val_kl_total))) if val_kl_total else float("nan")

    # KL plot
    kl_series: Dict[str, List[float]] = {}
    if joint_mode:
        kl_series["val_KL_total_mean"] = val_kl_total
    else:
        kl_series["train_KL_total"] = train_kl
        kl_series["val_KL_total"] = val_kl_total
        if len(val_kl_clean) == len(epochs):
            kl_series["val_KL_clean"] = val_kl_clean
        if len(val_kl_corr) == len(epochs):
            kl_series["val_KL_corr"] = val_kl_corr

    _plot_lines(
        x=epochs,
        series=kl_series,
        title="KL curves",
        ylabel="KL(teacher || student)",
        out_path=os.path.join(plots_dir, "kl_curves.png"),
    )

    # Sparsity curve (base)
    sp_series: Dict[str, List[float]] = {}
    if joint_mode:
        sp_series["val_L1_mean_mask"] = val_l1
    else:
        sp_series["train_L1_mean_mask"] = train_l1
        sp_series["val_L1_mean_mask"] = val_l1

    _plot_lines(
        x=epochs,
        series=sp_series,
        title="Sparsity curves",
        ylabel="mean(mask) = mean(sigmoid(theta))",
        out_path=os.path.join(plots_dir, "sparsity_curves.png"),
    )

    # ----- mask stats -----
    m = _load_mask_values(run_dir)  # [L,H,r]
    m_np = m.numpy().reshape(-1)

    mask_stats = {
        "mask_mean": float(m_np.mean()),
        "mask_median": float(np.median(m_np)),
        "mask_p10": _quantile(m_np, 0.10),
        "mask_p90": _quantile(m_np, 0.90),
        "dead_frac_0.05": float((m_np < 0.05).mean()),
        "mid_frac_0.05_0.95": float(((m_np >= 0.05) & (m_np <= 0.95)).mean()),
        "active_frac_0.95": float((m_np > 0.95).mean()),
        "active_frac_0.9": float((m_np > 0.9).mean()),
    }

    _plot_hist(
        values=m_np,
        title="Mask histogram",
        xlabel="m = sigmoid(theta)",
        out_path=os.path.join(plots_dir, "mask_hist.png"),
        bins=70,
    )

    # Overwrite sparsity plot including post-hoc fractions from BEST mask (constant lines).
    # This is useful when mean(mask) is ~1.0 and you want to see "any pruning" at all.
    if len(epochs) > 0:
        sp_series_with_fracs = dict(sp_series)
        sp_series_with_fracs["dead_frac_0.05(best)"] = [mask_stats["dead_frac_0.05"]] * len(epochs)
        sp_series_with_fracs["active_frac_0.9(best)"] = [mask_stats["active_frac_0.9"]] * len(epochs)
        sp_series_with_fracs["active_frac_0.95(best)"] = [mask_stats["active_frac_0.95"]] * len(epochs)
        _plot_lines(
            x=epochs,
            series=sp_series_with_fracs,
            title="Sparsity curves (+ post-hoc best-mask fractions)",
            ylabel="mask statistics",
            out_path=os.path.join(plots_dir, "sparsity_curves.png"),
        )

    # ----- effective rank -----
    L, H, r = m.shape
    svd_cache_dir = str(cfg.get("svd_cache_dir", "svd_cache/gpt2"))
    S_all = _load_singular_values_from_cache(svd_cache_dir, L, H, r)

    w = (m.float() * S_all).clamp_min(0.0)  # [L,H,r]
    eff = np.zeros((L, H), dtype=np.int32)
    strength = np.zeros((L, H), dtype=np.float32)

    for l in range(L):
        for h in range(H):
            ww = w[l, h]
            eff[l, h] = _effective_rank(ww, tau=0.9)
            strength[l, h] = float(ww.sum().item())

    eff_flat = eff.reshape(-1).astype(np.float32)
    eff_stats = {
        "eff_rank_median": float(np.median(eff_flat)),
        "eff_rank_p90": _quantile(eff_flat, 0.90),
        "eff_rank_min": int(eff_flat.min()) if eff_flat.size else None,
        "eff_rank_max": int(eff_flat.max()) if eff_flat.size else None,
    }

    _plot_hist(
        values=eff_flat,
        title="Effective rank (tau=0.9)",
        xlabel="effective_rank",
        out_path=os.path.join(plots_dir, "effective_rank.png"),
        bins=int(min(65, max(10, eff_flat.max() + 1))) if eff_flat.size else 20,
    )

    # Top heads table
    flat: List[Tuple[float, int, int, int]] = []
    for l in range(L):
        for h in range(H):
            flat.append((float(strength[l, h]), l, h, int(eff[l, h])))
    flat.sort(reverse=True, key=lambda x: x[0])
    top10 = flat[:10]

    lines = []
    lines.append("Top heads by sum_i (m_i * S_i) (larger = more 'kept energy')")
    lines.append("columns: rank, layer, head, strength_sum, eff_rank_tau0.9, top5_dirs(i:w_i)")
    for rank, (s, l, h, er) in enumerate(top10, start=1):
        ww = w[l, h]
        order = torch.argsort(ww, descending=True)[:5].tolist()
        pairs = [f"{i}:{float(ww[i].item()):.4f}" for i in order]
        lines.append(f"{rank:2d}  L{l:02d}H{h:02d}  strength={s:.6f}  eff_rank={er:2d}  top5={pairs}")
    _write_text(os.path.join(plots_dir, "top_heads_table.txt"), "\n".join(lines))

    # ----- sanity checks -----
    sanity: Dict[str, Any] = {}

    # IOI boundary check
    if task in {"ioi", "joint"}:
        ioi_train_csv = None
        if task == "ioi" and data_csv is not None:
            ioi_train_csv = _resolve_csv_path(data_dir, data_csv[0])
        elif task == "joint" and "ioi_train_csv" in cfg:
            ioi_train_csv = _resolve_csv_path(data_dir, cfg["ioi_train_csv"])

        out = os.path.join(save_dir, "sanity_ioi_boundary.txt")
        if ioi_train_csv is None or (not os.path.exists(ioi_train_csv)):
            _write_text(out, "SKIPPED: IOI CSV not provided / not found.\n")
            sanity["ioi_boundary"] = {"skipped": True}
        else:
            sanity["ioi_boundary"] = sanity_ioi_boundary(ioi_train_csv, out_path=out, n=20, seed=int(cfg.get("seed", 0)))

    # Tokenization sanity
    if task != "joint":
        out = os.path.join(save_dir, "sanity_tokenization.txt")
        if data_csv is None:
            _write_text(out, "SKIPPED: provide --data_csv TRAIN VAL TEST to run tokenization sanity.\n")
            sanity["tokenization"] = {"skipped": True}
        else:
            sanity["tokenization"] = sanity_tokenization(task, data_csv, data_dir, out_path=out, max_examples=200)
    else:
        # joint: write per-task reports
        wrote = []
        tok_sum = {}
        for t in ["gp", "ioi", "gt"]:
            k = f"{t}_train_csv"
            if k not in cfg:
                continue
            trip = (cfg[f"{t}_train_csv"], cfg[f"{t}_val_csv"], cfg[f"{t}_test_csv"])
            out = os.path.join(save_dir, f"sanity_tokenization_{t}.txt")
            tok_sum[t] = sanity_tokenization(t, trip, data_dir, out_path=out, max_examples=200)
            wrote.append(f"wrote sanity_tokenization_{t}.txt")
        _write_text(os.path.join(save_dir, "sanity_tokenization.txt"), "\n".join(wrote) + ("\n" if wrote else "SKIPPED\n"))
        sanity["tokenization"] = tok_sum if tok_sum else {"skipped": True}

    # All-ones invariance (separate only, because needs teacher cache + CSVs)
    out = os.path.join(save_dir, "sanity_all_ones_kl.txt")
    if task == "joint":
        _write_text(out, "SKIPPED for joint by default. Run per-task if you want.\n")
        sanity["all_ones_kl"] = {"skipped": True}
    else:
        if data_csv is None:
            _write_text(out, "SKIPPED: provide --data_csv TRAIN VAL TEST to run all-ones sanity.\n")
            sanity["all_ones_kl"] = {"skipped": True}
        else:
            try:
                sanity["all_ones_kl"] = sanity_all_ones_kl(cfg, task, data_csv, data_dir, device=device, out_path=out, n_batch=32)
            except Exception as e:
                _write_text(out, f"FAILED: {e}\n")
                sanity["all_ones_kl"] = {"failed": True, "error": str(e)}

    # ----- summary.json -----
    summary = {
        "run_dir": run_dir,
        "task": task,
        "joint_mode": joint_mode,
        "best_epoch_by_val_KL": best_epoch,
        "val_KL_total_best": best_val_kl,
        "final_eval": final_eval,
        **mask_stats,
        **eff_stats,
        "top_heads": [{"rank": i + 1, "layer": l, "head": h, "strength_sum": s, "eff_rank_tau0.9": er}
                      for i, (s, l, h, er) in enumerate(top10)],
        "sanity": sanity,
    }
    _write_json(os.path.join(save_dir, "summary.json"), summary)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--task", type=str, required=True, choices=["gp", "ioi", "gt", "joint"])
    p.add_argument("--data_csv", type=str, nargs=3, default=None, metavar=("TRAIN", "VAL", "TEST"),
                   help="Optional: train/val/test CSV names (needed for sanity checks).")
    p.add_argument("--data_dir", type=str, default=None, help="Directory holding CSVs (default: from config.json or data_main).")
    p.add_argument("--n_examples_debug", type=int, default=8)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default=None)
    return p


def main() -> None:
    args = build_parser().parse_args()
    analyze_run(
        run_dir=args.run_dir,
        task=args.task,
        save_dir=args.save_dir,
        n_examples_debug=args.n_examples_debug,
        device=args.device,
        data_dir=args.data_dir,
        data_csv=tuple(args.data_csv) if args.data_csv is not None else None,
    )
    print(f"[OK] Analysis written to: {os.path.abspath(args.save_dir or os.path.join(args.run_dir, 'analysis'))}")


if __name__ == "__main__":
    main()
