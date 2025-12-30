#!/usr/bin/env python3
"""
Tiny debug utility: All-ones mask invariance check for a saved run.

Example:
  python analysis/debug_all_ones_kl.py \
    --run_dir runs/gp/ov_mask_... \
    --task gp \
    --data_dir data_main \
    --data_csv train_1k_gp.csv val_gp.csv test_gp.csv \
    --device cuda \
    --out sanity_all_ones_kl.txt
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, Tuple

import torch


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_text(path: str, text: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def _resolve_csv_path(data_dir: str, maybe_rel: str) -> str:
    return maybe_rel if os.path.isabs(maybe_rel) else os.path.join(data_dir, maybe_rel)


@torch.no_grad()
def run_all_ones_kl(run_dir: str, task: str, csv_triplet: Tuple[str, str, str], data_dir: str, device: str, out: str) -> Dict[str, Any]:
    from transformer_lens import HookedTransformer
    from ovmask.data import load_task_csv
    from ovmask.svd_cache import SVDSpec, load_all_svd
    from ovmask.masked_ov import freeze_model_params, build_mask_state, MaskedOVHook
    from ovmask.teacher_cache import TeacherShardReader
    from ovmask.train_eval import forward_logp_next, kl_from_teacher_logp
    from ovmask.utils import dtype_from_str

    cfg = _read_json(os.path.join(run_dir, "config.json"))

    dev = torch.device(device if (device != "cpu" and torch.cuda.is_available()) else "cpu")
    dtype = dtype_from_str(str(cfg.get("dtype", "float32")))

    # eval-mode to avoid dropout mismatch
    student = HookedTransformer.from_pretrained("gpt2", device=dev, dtype=dtype)
    freeze_model_params(student)
    student.eval()

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

    state = build_mask_state(U, S, Vh, init_theta=4.0, device=dev)
    state.theta.data.fill_(20.0)
    hook = MaskedOVHook(state).to(dev)

    student.reset_hooks()
    for l in range(spec.n_layers):
        student.add_hook(f"blocks.{l}.attn.hook_z", hook.hook_store_z, dir="fwd")
        student.add_hook(f"blocks.{l}.hook_attn_out", hook.hook_replace_attn_out, dir="fwd")

    # use val split for the check
    train_csv, val_csv, test_csv = csv_triplet
    exs = load_task_csv(task, _resolve_csv_path(data_dir, val_csv))
    n = min(32, len(exs))
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
    _write_text(out, txt)
    return {"KL_clean": kl_clean, "KL_corr": kl_corr, "KL_total": kl_total, "passed": passed}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run_dir", type=str, required=True)
    p.add_argument("--task", type=str, required=True, choices=["gp","ioi","gt"])
    p.add_argument("--data_dir", type=str, default="data_main")
    p.add_argument("--data_csv", type=str, nargs=3, required=True, metavar=("TRAIN","VAL","TEST"))
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--out", type=str, default="sanity_all_ones_kl.txt")
    args = p.parse_args()

    res = run_all_ones_kl(args.run_dir, args.task, tuple(args.data_csv), args.data_dir, args.device, args.out)
    print(f"[OK] wrote: {args.out}")
    print(f"PASS={res['passed']}  KL_total={res['KL_total']:.4e}")


if __name__ == "__main__":
    main()
