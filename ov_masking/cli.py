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
from ov_masking.train import train_loop_joint, JointSplitCaches, eval_joint


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
    # student.to(device=device, dtype=dtype)
    student.to(device)
    student.to(dtype)
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


# def run_joint(args) -> None:
#     raise NotImplementedError(
#         "Joint mode wiring is intentionally omitted in this minimal first pass.\n"
#         "You said optional; separate mode is the default and fully implemented.\n"
#         "If you want, I can extend joint mode next (it is straightforward)."
#     )

# def run_joint(args):
#     """
#     Joint mode:
#       - Build 3 task datasets (gp/ioi/gt)
#       - Build per-task teacher caches (clean/corr for train/val/test)
#       - Train ONE shared mask with equal minibatches per task
#       - Save one run_dir under runs/joint/...
#     """
#     import json
#     from pathlib import Path
#     from datetime import datetime

#     import torch

#     # --- setup ---
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     dtype = torch.float16 if getattr(args, "fp16", True) and device.type == "cuda" else torch.float32

#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     run_dir = Path("runs") / "joint" / f"ov_mask_{timestamp}"
#     run_dir.mkdir(parents=True, exist_ok=True)
#     (run_dir / "masks").mkdir(exist_ok=True)
#     (run_dir / "teacher").mkdir(exist_ok=True)

#     # logger: reuse your existing logger if you have one; otherwise minimal
#     log_path = run_dir / "train.log"
#     def log_fn(msg: str):
#         print(msg, flush=True)
#         with open(log_path, "a", encoding="utf-8") as f:
#             f.write(msg + "\n")

#     # --- load models ---
#     teacher, student = _load_models(args.model_name, device=device, dtype=dtype)

#     # Ensure hook_result exists (TransformerLens feature flag)
#     student.cfg.use_attn_result = True

#     tokenizer = student.tokenizer
#     if tokenizer is None:
#         raise RuntimeError("Tokenizer missing on TransformerLens model.")

#     # --- SVD cache + mask params + runner ---
#     svd_cache = build_or_load_svd_cache(student, cache_root=Path("cache") / "svd", log_fn=log_fn)
#     mask_params = init_mask_params_like_svd(svd_cache, init_logit=4.0, device=device)
#     runner = MaskedOVRunner(student, svd_cache, mask_params)

#     # Freeze base weights (masks only)
#     for p in student.parameters():
#         p.requires_grad_(False)
#     mask_params.theta.requires_grad_(True)

#     # --- load datasets (reuse your existing loader used in run_separate) ---
#     data_dir = Path(args.data_dir)

#     task_specs = {
#         "gp":  {"train": args.gp_train_csv,  "val": args.gp_val_csv,  "test": args.gp_test_csv},
#         "ioi": {"train": args.ioi_train_csv, "val": args.ioi_val_csv, "test": args.ioi_test_csv},
#         "gt":  {"train": args.gt_train_csv,  "val": args.gt_val_csv,  "test": args.gt_test_csv},
#     }

#     splits = ["train", "val", "test"]

#     # examples[task][split] = list of examples (same type you already use in separate mode)
#     examples = {t: {} for t in task_specs}
#     for task, spec in task_specs.items():
#         for split in splits:
#             csv_path = data_dir / spec[split]
#             examples[task][split] = load_examples_for_task(task, csv_path)  # <-- same helper as separate mode

#     # --- teacher cache build/load ---
#     # teacher/{task}_{split}_{clean|corr}_logp.pt
#     def teacher_path(task: str, split: str, variant: str) -> Path:
#         return run_dir / "teacher" / f"{task}_{split}_{variant}_logp.pt"

#     @torch.no_grad()
#     def build_teacher_cache(task: str, split: str, variant: str):
#         out_path = teacher_path(task, split, variant)
#         if out_path.exists():
#             return
#         exs = examples[task][split]
#         prompts = [ex.prompt_clean if variant == "clean" else ex.prompt_corr for ex in exs]
#         # tokenize in batches
#         all_logp = []
#         bs = args.batch_size  # reuse
#         max_len = getattr(args, "max_length", 1024)
#         teacher.eval()
#         for i in range(0, len(prompts), bs):
#             chunk = prompts[i:i+bs]
#             enc = tokenizer(
#                 chunk,
#                 return_tensors="pt",
#                 padding=True,
#                 truncation=True,
#                 max_length=max_len,
#                 add_special_tokens=False,
#             )
#             toks = enc["input_ids"].to(device)
#             am = enc["attention_mask"].to(device)
#             logits = teacher(toks, attention_mask=am)  # TL forward supports attention_mask
#             logits_last = logits[:, -1, :]
#             logp = torch.log_softmax(logits_last.float(), dim=-1).to(torch.float16).cpu()
#             all_logp.append(logp)
#         full = torch.cat(all_logp, dim=0)
#         torch.save(full, out_path)
#         log_fn(f"[teacher] saved {out_path.name} shape={tuple(full.shape)} dtype={full.dtype}")

#     # Build all caches
#     log_fn("[joint] Building/loading teacher caches...")
#     for task in task_specs:
#         for split in splits:
#             build_teacher_cache(task, split, "clean")
#             build_teacher_cache(task, split, "corr")

#     # Load caches into CPU tensors
#     train_caches = JointSplitCaches(teacher_logp={})
#     val_caches = JointSplitCaches(teacher_logp={})
#     test_caches = JointSplitCaches(teacher_logp={})

#     for task in task_specs:
#         train_caches.teacher_logp[task] = {
#             "clean": torch.load(teacher_path(task, "train", "clean"), map_location="cpu"),
#             "corr":  torch.load(teacher_path(task, "train", "corr"),  map_location="cpu"),
#         }
#         val_caches.teacher_logp[task] = {
#             "clean": torch.load(teacher_path(task, "val", "clean"), map_location="cpu"),
#             "corr":  torch.load(teacher_path(task, "val", "corr"),  map_location="cpu"),
#         }
#         test_caches.teacher_logp[task] = {
#             "clean": torch.load(teacher_path(task, "test", "clean"), map_location="cpu"),
#             "corr":  torch.load(teacher_path(task, "test", "corr"),  map_location="cpu"),
#         }

#     # --- dataloaders: equal minibatches per task ---
#     per_task_bs = args.batch_size // 3
#     train_loaders = {}
#     val_loaders = {}
#     test_loaders = {}

#     for task in task_specs:
#         train_loaders[task] = make_dataloader(examples[task]["train"], batch_size=per_task_bs, shuffle=True, seed=args.seed)
#         val_loaders[task]   = make_dataloader(examples[task]["val"],   batch_size=per_task_bs, shuffle=False, seed=args.seed)
#         test_loaders[task]  = make_dataloader(examples[task]["test"],  batch_size=per_task_bs, shuffle=False, seed=args.seed)

#     # --- save config ---
#     config = vars(args).copy()
#     config["mode"] = "joint"
#     config["run_dir"] = str(run_dir)
#     config["per_task_batch_size"] = per_task_bs
#     with open(run_dir / "config.json", "w", encoding="utf-8") as f:
#         json.dump(config, f, indent=2)

#     # --- training ---
#     best_mask_logits_path = run_dir / "masks" / "best_mask_logits.pt"
#     best_mask_values_path = run_dir / "masks" / "best_mask_values.pt"

#     def save_best_fn(epoch: int):
#         torch.save(mask_params.theta.detach().cpu(), best_mask_logits_path)
#         torch.save(torch.sigmoid(mask_params.theta.detach()).cpu(), best_mask_values_path)
#         log_fn(f"[joint] Saved best masks at epoch {epoch}")

#     train_result = train_loop_joint(
#         runner=runner,
#         mask_theta=mask_params.theta,
#         tokenizer=tokenizer,
#         train_loaders=train_loaders,
#         val_loaders=val_loaders,
#         train_caches=train_caches,
#         val_caches=val_caches,
#         epochs=args.epochs,
#         lr=args.lr,
#         lambda_l1=args.lambda_l1,
#         grad_clip=args.grad_clip,
#         early_stopping_patience=args.early_stopping_patience,
#         device=device,
#         max_length=getattr(args, "max_length", 1024),
#         log_fn=log_fn,
#         save_best_fn=save_best_fn,
#     )

#     # Save full masks (final)
#     torch.save(mask_params.theta.detach().cpu(), run_dir / "masks" / "mask_logits.pt")
#     torch.save(torch.sigmoid(mask_params.theta.detach()).cpu(), run_dir / "masks" / "mask_values.pt")

#     # Save metrics
#     with open(run_dir / "metrics.json", "w", encoding="utf-8") as f:
#         json.dump(train_result, f, indent=2)

#     # --- test eval using best masks (if present) ---
#     if best_mask_logits_path.exists():
#         mask_params.theta.data.copy_(torch.load(best_mask_logits_path, map_location=device))

#     test_report = eval_joint(
#         runner=runner,
#         tokenizer=tokenizer,
#         loaders=test_loaders,
#         caches=test_caches,
#         device=device,
#         max_length=getattr(args, "max_length", 1024),
#     )
#     with open(run_dir / "final_eval.json", "w", encoding="utf-8") as f:
#         json.dump(test_report, f, indent=2)

#     # selected.json (rank by m_i * S_i)
#     selected = build_selected_directions_json(svd_cache, torch.sigmoid(mask_params.theta.detach()).cpu())
#     with open(run_dir / "selected.json", "w", encoding="utf-8") as f:
#         json.dump(selected, f, indent=2)

#     log_fn(f"[joint] Done. Outputs in: {run_dir}")

def run_joint(args) -> None:
    """
    Joint mode: train ONE shared mask across GP/IOI/GT.
    - Uses per-task equal batch size (args.batch_size // 3).
    - Loss is equal-weight average of task KLs + lambda_l1 * mean(sigmoid(theta)).
    - Early stopping on equal-weight average val KL_total.
    """
    repo_root = Path(".").resolve()
    data_dir = Path(args.data_dir)
    cache_root = Path(args.cache_root)

    device = get_device(args.device)
    dtype = parse_dtype(args.dtype)
    set_seed(args.seed)

    # Run dir
    run_dir = ensure_dir(Path("runs") / "joint" / f"ov_mask_{timestamp()}")
    ensure_dir(run_dir / "masks")
    ensure_dir(run_dir / "teacher")
    ensure_dir(run_dir / "logs")

    logger = setup_logger(run_dir / "train.log")
    logger.info(f"Run dir: {run_dir}")
    logger.info(f"Device: {device}, dtype: {dtype}")

    git_hash = get_git_commit_hash(repo_root)

    per_task_bs = args.batch_size // 3

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
            "mode": "joint",
            "data_dir": str(data_dir),
            "cache_root": str(cache_root),
            "git_commit_hash": git_hash,
            "per_task_batch_size": per_task_bs,
            "csvs": {
                "gp": {"train": args.gp_train_csv, "val": args.gp_val_csv, "test": args.gp_test_csv},
                "ioi": {"train": args.ioi_train_csv, "val": args.ioi_val_csv, "test": args.ioi_test_csv},
                "gt": {"train": args.gt_train_csv, "val": args.gt_val_csv, "test": args.gt_test_csv},
            },
        },
        run_dir / "config.json",
    )

    # Models
    teacher, student = _load_models(args.model_name, device=device, dtype=dtype)
    tokenizer = student.tokenizer

    # SVD cache (computed from teacher weights)
    svd_dir, _svd_meta = maybe_compute_and_save_svd_cache(
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

    # Mask + runner
    r = teacher.cfg.d_head + 1
    mask_params = OVMaskParams(teacher.cfg.n_layers, teacher.cfg.n_heads, r, init_theta=args.init_theta).to(device)
    runner = OVMaskedRunner(model=student, svd_tensors=svd_tensors, mask_params=mask_params)

    # Data loaders per task
    task_to_csv = {
        "gp": (args.gp_train_csv, args.gp_val_csv, args.gp_test_csv),
        "ioi": (args.ioi_train_csv, args.ioi_val_csv, args.ioi_test_csv),
        "gt": (args.gt_train_csv, args.gt_val_csv, args.gt_test_csv),
    }

    train_loaders = {}
    val_loaders = {}
    test_loaders = {}
    teacher_train = {}
    teacher_val = {}
    teacher_test = {}

    vocab_size = teacher.cfg.d_vocab

    for task, (tr, va, te) in task_to_csv.items():
        train_path = data_dir / tr
        val_path = data_dir / va
        test_path = data_dir / te

        train_ds, train_loader = _prepare_split(
            task=task, split="train", csv_path=train_path, tokenizer=tokenizer,
            batch_size=per_task_bs, num_workers=args.num_workers, shuffle=True
        )
        val_ds, val_loader = _prepare_split(
            task=task, split="val", csv_path=val_path, tokenizer=tokenizer,
            batch_size=per_task_bs, num_workers=args.num_workers, shuffle=False
        )
        test_ds, test_loader = _prepare_split(
            task=task, split="test", csv_path=test_path, tokenizer=tokenizer,
            batch_size=per_task_bs, num_workers=args.num_workers, shuffle=False
        )

        train_loaders[task] = train_loader
        val_loaders[task] = val_loader
        test_loaders[task] = test_loader

        # Teacher caches (saved under run_dir/teacher)
        tpaths_train = teacher_cache_paths(run_dir / "teacher", task, "train")
        tpaths_val = teacher_cache_paths(run_dir / "teacher", task, "val")
        tpaths_test = teacher_cache_paths(run_dir / "teacher", task, "test")

        maybe_build_teacher_cache(
            model=teacher,
            tokenizer=tokenizer,
            dataloader=DataLoader(train_ds, batch_size=per_task_bs, shuffle=False, collate_fn=_collate_fn),
            n_items=len(train_ds),
            vocab_size=vocab_size,
            cache_paths=tpaths_train,
            device=device,
            force_recompute=args.force_recompute_teacher,
            desc_prefix=f"{task}/train",
        )
        maybe_build_teacher_cache(
            model=teacher,
            tokenizer=tokenizer,
            dataloader=DataLoader(val_ds, batch_size=per_task_bs, shuffle=False, collate_fn=_collate_fn),
            n_items=len(val_ds),
            vocab_size=vocab_size,
            cache_paths=tpaths_val,
            device=device,
            force_recompute=args.force_recompute_teacher,
            desc_prefix=f"{task}/val",
        )
        maybe_build_teacher_cache(
            model=teacher,
            tokenizer=tokenizer,
            dataloader=DataLoader(test_ds, batch_size=per_task_bs, shuffle=False, collate_fn=_collate_fn),
            n_items=len(test_ds),
            vocab_size=vocab_size,
            cache_paths=tpaths_test,
            device=device,
            force_recompute=args.force_recompute_teacher,
            desc_prefix=f"{task}/test",
        )

        teacher_train[task] = load_teacher_cache(tpaths_train)
        teacher_val[task] = load_teacher_cache(tpaths_val)
        teacher_test[task] = load_teacher_cache(tpaths_test)

    # ---- JOINT TRAIN ----
    train_result = train_loop_joint(
        runner=runner,
        mask_params=mask_params,
        tokenizer=tokenizer,
        teacher_cache_train_by_task=teacher_train,
        teacher_cache_val_by_task=teacher_val,
        train_loaders_by_task=train_loaders,
        val_loaders_by_task=val_loaders,
        device=device,
        cfg=cfg,
        logger=logger,
        run_dir=run_dir,
    )

    # Load best theta for final eval
    best_theta = torch.load(train_result["best_theta_path"], map_location="cpu")
    mask_params.theta.data.copy_(best_theta.to(device=device, dtype=mask_params.theta.dtype))

    # Save final masks
    torch.save(mask_params.theta.detach().cpu(), run_dir / "masks" / "mask_logits.pt")
    torch.save(mask_params.mask_values().detach().cpu(), run_dir / "masks" / "mask_values.pt")

    # Test eval per task + overall
    per_task_test = {}
    for task in ["gp", "ioi", "gt"]:
        per_task_test[task] = eval_split(
            runner=runner,
            tokenizer=tokenizer,
            teacher_cache=teacher_test[task],
            dataloader=test_loaders[task],
            device=device,
        )

    overall = {}
    keys = list(per_task_test["gp"].keys())
    for k in keys:
        overall[k] = float(sum(per_task_test[t][k] for t in ["gp", "ioi", "gt"]) / 3.0)

    json_dump({"per_task": per_task_test, "overall": overall}, run_dir / "final_eval.json")
    json_dump(train_result, run_dir / "metrics.json")

    # selected.json: rank by m_i * S_i
    with torch.no_grad():
        m = mask_params.mask_values().detach().cpu()
        S = svd_tensors["S"].detach().cpu()
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

    logger.info("JOINT DONE.")



# def main() -> None:
#     parser = build_arg_parser()
#     args = parser.parse_args()

#     if args.mode == "separate":
#         if not args.task or not args.train_csv or not args.val_csv or not args.test_csv:
#             raise ValueError("Separate mode requires --task and --train_csv/--val_csv/--test_csv.")
#         run_separate(args)
#     elif args.mode == "joint":
#         needed = [
#             args.gp_train_csv, args.gp_val_csv, args.gp_test_csv,
#             args.ioi_train_csv, args.ioi_val_csv, args.ioi_test_csv,
#             args.gt_train_csv, args.gt_val_csv, args.gt_test_csv,
#         ]
#         if any(x is None for x in needed):
#             raise ValueError("Joint mode requires all 9 CSV args: gp_*, ioi_*, gt_*.")
#         if args.batch_size % 3 != 0:
#             raise ValueError("Joint mode requires --batch_size divisible by 3 (equal minibatches per task).")
#     else:
#         run_joint(args)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.mode == "separate":
        if not args.task or not args.train_csv or not args.val_csv or not args.test_csv:
            raise ValueError("Separate mode requires --task and --train_csv/--val_csv/--test_csv.")
        run_separate(args)
        return

    if args.mode == "joint":
        needed = [
            args.gp_train_csv, args.gp_val_csv, args.gp_test_csv,
            args.ioi_train_csv, args.ioi_val_csv, args.ioi_test_csv,
            args.gt_train_csv, args.gt_val_csv, args.gt_test_csv,
        ]
        if any(x is None for x in needed):
            raise ValueError("Joint mode requires all 9 CSV args: gp_*, ioi_*, gt_*.")
        if args.batch_size % 3 != 0:
            raise ValueError("Joint mode requires --batch_size divisible by 3 (equal minibatches per task).")

        run_joint(args)
        return

    raise ValueError(f"Unknown mode: {args.mode}")
