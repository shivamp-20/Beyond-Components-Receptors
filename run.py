#!/usr/bin/env python3
"""
Entry point for OV-mask training.

Examples:
  # GP
  python run.py train --task gp --data_dir data_main --train_csv train_gp.csv --val_csv val_gp.csv --test_csv test_gp.csv

  # IOI
  python run.py train --task ioi --data_dir data_main --train_csv train_ioi.csv --val_csv val_ioi.csv --test_csv test_ioi.csv

  # GT
  python run.py train --task gt --data_dir data_main --train_csv train_gt.csv --val_csv val_gt.csv --test_csv test_gt.csv

  # Joint (one shared mask)
  python run.py train_joint --data_dir data_main \
    --gp_train_csv train_gp.csv --gp_val_csv val_gp.csv --gp_test_csv test_gp.csv \
    --ioi_train_csv train_ioi.csv --ioi_val_csv val_ioi.csv --ioi_test_csv test_ioi.csv \
    --gt_train_csv train_gt.csv --gt_val_csv val_gt.csv --gt_test_csv test_gt.csv
"""

import argparse
from ovmask.train_eval import (
    train_separate_task,
    train_joint_tasks,
)
from ovmask.utils import str2bool


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()

    sub = p.add_subparsers(dest="cmd", required=True)

    def add_common_train_args(sp: argparse.ArgumentParser):
        sp.add_argument("--data_dir", type=str, default="data_main")

        sp.add_argument("--batch_size", type=int, default=64)
        sp.add_argument("--epochs", type=int, default=15)
        sp.add_argument("--lr", type=float, default=1e-2)
        sp.add_argument("--lambda_l1", type=float, default=0.1)
        sp.add_argument("--optimizer", type=str, default="adamw", choices=["adamw"])

        sp.add_argument("--early_stopping_patience", type=int, default=3)
        sp.add_argument("--grad_clip", type=float, default=1.0)

        sp.add_argument("--seed", type=int, default=0)
        sp.add_argument("--device", type=str, default="cuda")
        sp.add_argument("--dtype", type=str, default="float32", choices=["float32", "float16", "bfloat16"])

        sp.add_argument("--num_workers", type=int, default=0)
        sp.add_argument("--shuffle_train", type=str2bool, default=True)

        # caching / storage
        sp.add_argument("--svd_cache_dir", type=str, default="svd_cache/gpt2")
        sp.add_argument("--teacher_cache_dir", type=str, default="teacher_cache")
        sp.add_argument("--teacher_shard_size", type=int, default=2048)  # shards to avoid huge RAM
        sp.add_argument("--force_recompute_svd", type=str2bool, default=False)
        sp.add_argument("--force_recompute_teacher", type=str2bool, default=False)

        # logging
        sp.add_argument("--log_every_steps", type=int, default=200)  # not too spammy
        sp.add_argument("--notes", type=str, default="")

    # ---- train (single task) ----
    sp_train = sub.add_parser("train")
    add_common_train_args(sp_train)
    sp_train.add_argument("--task", type=str, required=True, choices=["gp", "ioi", "gt"])
    sp_train.add_argument("--train_csv", type=str, required=True)
    sp_train.add_argument("--val_csv", type=str, required=True)
    sp_train.add_argument("--test_csv", type=str, required=True)

    # ---- train_joint ----
    sp_joint = sub.add_parser("train_joint")
    add_common_train_args(sp_joint)
    sp_joint.add_argument("--gp_train_csv", type=str, required=True)
    sp_joint.add_argument("--gp_val_csv", type=str, required=True)
    sp_joint.add_argument("--gp_test_csv", type=str, required=True)

    sp_joint.add_argument("--ioi_train_csv", type=str, required=True)
    sp_joint.add_argument("--ioi_val_csv", type=str, required=True)
    sp_joint.add_argument("--ioi_test_csv", type=str, required=True)

    sp_joint.add_argument("--gt_train_csv", type=str, required=True)
    sp_joint.add_argument("--gt_val_csv", type=str, required=True)
    sp_joint.add_argument("--gt_test_csv", type=str, required=True)

    return p


def main():
    args = build_parser().parse_args()

    if args.cmd == "train":
        train_separate_task(args)
    elif args.cmd == "train_joint":
        train_joint_tasks(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()
