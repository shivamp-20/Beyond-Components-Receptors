from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from transformer_lens import HookedTransformer  # type: ignore

from .data import PromptDataset, load_task_examples
from .masked_model import OVMaskParams, OVMaskedRunner
from .svd_cache import load_svd_tensors, maybe_compute_and_save_svd_cache
from .teacher_cache import load_teacher_cache, maybe_build_teacher_cache, teacher_cache_paths
from .train import TrainConfig, _collate_fn, eval_split, train_loop
from .utils import ensure_dir, get_device, get_git_commit_hash, json_dump, parse_dtype, set_seed, setup_logger, timestamp


def _set_use_attn_result(model) -> None:
    """
    TransformerLens requires use_attn_result=True for hook_result to exist. :contentReference[oaicite:3]{index=3}
    Different versions expose different helpers, so we set defensively.
    """
    if hasattr(model, "cfg") and hasattr(model.cfg, "use_attn_result"):
        model.cfg.use_attn_result = True
    if hasattr(model, "set_use_attn_result"):
        try:
            model.set_use_attn_result(True)
        except Exception:
            pass


def _zero_out_b_O(model) -> None:
    """
    IMPORTANT:
    We fold b_O/n_heads into A_aug and apply it per head via the appended ones dimension.
    Therefore we must prevent model from also adding b_O.
    Easiest: set attn.b_O = 0 in the STUDENT model.
    """
    with torch.no_grad():
        for l in range(model.cfg.n_layers):
            attn = model.blocks[l].attn
            if hasattr(attn, "b_O"):
                attn.b_O.zero_()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train diagonal OV singular direction masks (GP/IOI/GT).")

    p.add_argument("--mode", type=str, default="separate", choices=["separate", "joint"])

    # Separate mode args
    p.add_argument("--task", type=str, default=None, choices=["gp", "ioi", "gt"])
    p.add_argument("--train_csv", type=str, default=None)
    p.add_argument("--val_csv", type=str, default=None)
    p.add_argument("--test_csv", type=str, default=None)

    # Joint mode args
    p.add_argument("--gp_train_csv", type=str, default=None)
    p.add_argument("--gp_val_csv", type=str, default=None)
    p.add_argument("--gp_test_csv", type=str, default=None)

    p.add_argument("--ioi_train_csv", type=str, default=None)
    p.add_argument("--ioi_val_csv", type=str, default=None)
    p.add_argument("--ioi_test_csv", type=str, default=None)

    p.add_argument("--gt_train_csv", type=str, default=None)
    p.add_argument("--gt_val_csv", type=str, default=None)
    p.add_argument("--gt_test_csv", type=str, default=None)

    p.add_argument("--data_dir", type=str, default="data_main")
    p.add_argument("--cache_root", type=str, default="cache")

    # Hyperparams (configurable defaults per spec)
    p.add_argument("--model_name", type=str, default="gpt2")
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--lambda_l1", type=float, default=0.1)
    p.add_argument("--early_stopping_patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="auto")
    p.add_argument("--dtype", type=str, default="float16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--init_theta", type=float, default=4.0)

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--force_recompute_svd", action="store_true")
    p.add_argument("--force_recompute_teacher", action="store_true")

    return p


def _load_models(model_name: str, device: torch.device, dtype: torch.dtype):
    # Teacher (original)
    teacher = HookedTransformer.from_pretrained(model_name)
    _set_use_attn_result(teacher)
    # teacher.to(device=device, dtype=dtype)
    teacher.to(device)
    teacher.to(dtype)  
    teacher.eval()

    # Student (same weights, but b_O zeroed; OV replaced by hooks)
    student = HookedTransformer.from_pretrained(model_name)
    _set_use_attn_result(student)
    student.to(device=device, dtype=dtype)
    _zero_out_b_O(student)
    student.eval()

    # Freeze all params for both (mask params live outside model)
    for m in [teacher, student]:
        for p in m.parameters():
            p.requires_grad_(False)

    # Ensure tokenizer has padding token
    if student.tokenizer.pad_token_id is None:
        student.tokenizer.pad_token = student.tokenizer.eos_token

    return teacher, student


# def _load_models(model_name: str, device, dtype: torch.dtype):
#     # TransformerLens expects device as a string like "cuda" / "cpu"
#     device_str = str(device)

#     # Load directly on the right device/dtype (supported by TL) :contentReference[oaicite:1]{index=1}
#     teacher = HookedTransformer.from_pretrained(model_name, device=device_str, dtype=dtype)
#     teacher.eval()
#     for p in teacher.parameters():
#         p.requires_grad_(False)

#     student_base = HookedTransformer.from_pretrained(model_name, device=device_str, dtype=dtype)
#     student_base.eval()
#     for p in student_base.parameters():
#         p.requires_grad_(False)

#     return teacher, student_base



def _prepare_split(
    *,
    task: str,
    split: str,
    csv_path: Path,
    tokenizer,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
):
    examples = load_task_examples(task, csv_path)
    ds = PromptDataset(examples, tokenizer=tokenizer, task=task)

    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate_fn,
        drop_last=False,
    )
    return ds, loader


def run_separate(args) -> None:
    repo_root = Path(".").resolve()
    data_dir = Path(args.data_dir)
    cache_root = Path(args.cache_root)

    device = get_device(args.device)
    dtype = parse_dtype(args.dtype)

    set_seed(args.seed)

    # Run dir
    run_dir = ensure_dir(Path("runs") / args.task / f"ov_mask_{timestamp()}")
    ensure_dir(run_dir / "masks")
    ensure_dir(run_dir / "teacher")
    ensure_dir(run_dir / "logs")

    logger = setup_logger(run_dir / "train.log")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Device: {device}, dtype: {dtype}")

    git_hash = get_git_commit_hash(repo_root)

    cfg = TrainConfig(
        model_name=args.model_name,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        lambda_l1=args.lambda_l1,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
        grad_clip=args.grad_clip,
        device=str(device),
        dtype=args.dtype,
        init_theta=args.init_theta,
        num_workers=args.num_workers,
        force_recompute_svd=args.force_recompute_svd,
        force_recompute_teacher=args.force_recompute_teacher,
    )
    json_dump(
        {
            "train_config": cfg.__dict__,
            "task": args.task,
            "train_csv": args.train_csv,
            "val_csv": args.val_csv,
            "test_csv": args.test_csv,
            "data_dir": str(data_dir),
            "cache_root": str(cache_root),
            "git_commit_hash": git_hash,
        },
        run_dir / "config.json",
    )

    # Load models
    teacher, student = _load_models(args.model_name, device=device, dtype=dtype)
    tokenizer = student.tokenizer

    # SVD cache (computed from teacher weights)
    svd_dir, svd_meta = maybe_compute_and_save_svd_cache(
        model=teacher,
        model_name=args.model_name,
        cache_root=cache_root,
        force_recompute=args.force_recompute_svd,
    )
    logger.info(f"SVD cache: {svd_dir}")

    svd_tensors = load_svd_tensors(
        svd_dir,
        n_layers=teacher.cfg.n_layers,
        n_heads=teacher.cfg.n_heads,
        device=device,
        dtype=dtype,
    )

    # Mask params
    r = teacher.cfg.d_head + 1
    mask_params = OVMaskParams(teacher.cfg.n_layers, teacher.cfg.n_heads, r, init_theta=args.init_theta).to(device)
    runner = OVMaskedRunner(model=student, svd_tensors=svd_tensors, mask_params=mask_params)

    # Data
    train_path = data_dir / args.train_csv
    val_path = data_dir / args.val_csv
    test_path = data_dir / args.test_csv

    train_ds, train_loader = _prepare_split(
        task=args.task, split="train", csv_path=train_path, tokenizer=tokenizer,
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    val_ds, val_loader = _prepare_split(
        task=args.task, split="val", csv_path=val_path, tokenizer=tokenizer,
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )
    test_ds, test_loader = _prepare_split(
        task=args.task, split="test", csv_path=test_path, tokenizer=tokenizer,
        batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    # Warn if labels often multi-token
    logger.info(f"Multi-token label stats (counts): {train_ds.multi_token_warnings}")

    # Teacher cache
    vocab_size = teacher.cfg.d_vocab
    tpaths_train = teacher_cache_paths(run_dir / "teacher", args.task, "train")
    tpaths_val = teacher_cache_paths(run_dir / "teacher", args.task, "val")
    tpaths_test = teacher_cache_paths(run_dir / "teacher", args.task, "test")

    # Build teacher caches (no grads) BEFORE training
    maybe_build_teacher_cache(
        model=teacher,
        tokenizer=tokenizer,
        dataloader=DataLoader(train_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_fn),
        n_items=len(train_ds),
        vocab_size=vocab_size,
        cache_paths=tpaths_train,
        device=device,
        force_recompute=args.force_recompute_teacher,
        desc_prefix=f"{args.task}/train",
    )
    maybe_build_teacher_cache(
        model=teacher,
        tokenizer=tokenizer,
        dataloader=DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_fn),
        n_items=len(val_ds),
        vocab_size=vocab_size,
        cache_paths=tpaths_val,
        device=device,
        force_recompute=args.force_recompute_teacher,
        desc_prefix=f"{args.task}/val",
    )
    maybe_build_teacher_cache(
        model=teacher,
        tokenizer=tokenizer,
        dataloader=DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, collate_fn=_collate_fn),
        n_items=len(test_ds),
        vocab_size=vocab_size,
        cache_paths=tpaths_test,
        device=device,
        force_recompute=args.force_recompute_teacher,
        desc_prefix=f"{args.task}/test",
    )

    teacher_train = load_teacher_cache(tpaths_train)
    teacher_val = load_teacher_cache(tpaths_val)
    teacher_test = load_teacher_cache(tpaths_test)

    # Train
    train_result = train_loop(
        runner=runner,
        mask_params=mask_params,
        tokenizer=tokenizer,
        teacher_cache_train=teacher_train,
        teacher_cache_val=teacher_val,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        cfg=cfg,
        logger=logger,
        run_dir=run_dir,
    )

    # Load best and save final artifacts
    best_theta = torch.load(train_result["best_theta_path"], map_location="cpu")
    mask_params.theta.data.copy_(best_theta.to(device=device, dtype=mask_params.theta.dtype))

    # Final eval on test
    test_stats = eval_split(
        runner=runner,
        tokenizer=tokenizer,
        teacher_cache=teacher_test,
        dataloader=test_loader,
        device=device,
    )

    # Save required outputs
    torch.save(mask_params.theta.detach().cpu(), run_dir / "masks" / "mask_logits.pt")
    torch.save(mask_params.mask_values().detach().cpu(), run_dir / "masks" / "mask_values.pt")

    # selected.json: rank by m_i * S_i
    with torch.no_grad():
        m = mask_params.mask_values().detach().cpu()  # [L,H,R]
        S = svd_tensors["S"].detach().cpu()           # [L,H,R] (already includes bias fold in SVD)
        scores = m * S
        selected = {}
        for l in range(scores.shape[0]):
            for h in range(scores.shape[1]):
                order = torch.argsort(scores[l, h], descending=True).tolist()
                selected[f"l{l}_h{h}"] = {
                    "ranked_indices": order,
                    "scores": scores[l, h][order].tolist(),
                }

    json_dump(selected, run_dir / "selected.json")
    json_dump(train_result, run_dir / "metrics.json")
    json_dump(test_stats, run_dir / "final_eval.json")

    logger.info(f"Saved final_eval.json: {test_stats}")
    logger.info("DONE.")


def run_joint(args) -> None:
    raise NotImplementedError(
        "Joint mode wiring is intentionally omitted in this minimal first pass.\n"
        "You said optional; separate mode is the default and fully implemented.\n"
        "If you want, I can extend joint mode next (it is straightforward)."
    )


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "separate":
        if not args.task or not args.train_csv or not args.val_csv or not args.test_csv:
            raise ValueError("Separate mode requires --task and --train_csv/--val_csv/--test_csv.")
        run_separate(args)
    else:
        run_joint(args)
