from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Iterable, Iterator, Any

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import accuracy_from_logits, kl_teacher_student, logit_diff
from .teacher_cache import tokenize_prompts


@dataclass
class TrainConfig:
    model_name: str = "gpt2"
    batch_size: int = 64
    epochs: int = 15
    lr: float = 1e-2
    lambda_l1: float = 0.1
    early_stopping_patience: int = 3
    seed: int = 0
    grad_clip: float = 1.0
    device: str = "auto"
    dtype: str = "float16"
    init_theta: float = 4.0

    num_workers: int = 0
    force_recompute_svd: bool = False
    force_recompute_teacher: bool = False


def _collate_fn(batch):
    idxs, pclean, pcorr, lcid, lcoid, wcid, wcoid = zip(*batch)
    return (
        torch.tensor(idxs, dtype=torch.long),
        list(pclean),
        list(pcorr),
        torch.tensor(lcid, dtype=torch.long),
        torch.tensor(lcoid, dtype=torch.long),
        torch.tensor(wcid, dtype=torch.long),
        torch.tensor(wcoid, dtype=torch.long),
    )


@torch.no_grad()
def eval_split(
    *,
    runner,
    tokenizer,
    teacher_cache: Dict[str, torch.Tensor],  # CPU float16
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    runner.model.eval()

    sums = {
        "kl_clean": 0.0,
        "kl_corr": 0.0,
        "acc_clean": 0.0,
        "acc_corr": 0.0,
        "logitdiff_clean": 0.0,
        "logitdiff_corr": 0.0,
    }
    n_batches = 0

    for batch in tqdm(dataloader, desc="eval", leave=False):
        idxs, prompts_clean, prompts_corr, label_clean_ids, label_corr_ids, wrong_clean_ids, wrong_corr_ids = batch
        idxs_list = idxs.tolist()

        # Clean forward
        tokens, attn_mask, last_pos = tokenize_prompts(tokenizer, prompts_clean, device=device)
        logits = runner(tokens, attn_mask)
        bsz = logits.size(0)
        logits_last = logits[torch.arange(bsz, device=device), last_pos]  # [B,V]
        logp_student = torch.log_softmax(logits_last.float(), dim=-1)
        logp_teacher = teacher_cache["clean"][idxs_list].to(device=device, dtype=torch.float32)
        kl_c = kl_teacher_student(logp_student, logp_teacher).item()

        acc_c = accuracy_from_logits(logits_last, label_clean_ids.to(device)).item()
        ld_c = logit_diff(logits_last, label_clean_ids.to(device), wrong_clean_ids.to(device)).item()

        # Corr forward
        tokens, attn_mask, last_pos = tokenize_prompts(tokenizer, prompts_corr, device=device)
        logits = runner(tokens, attn_mask)
        bsz = logits.size(0)
        logits_last = logits[torch.arange(bsz, device=device), last_pos]
        logp_student = torch.log_softmax(logits_last.float(), dim=-1)
        logp_teacher = teacher_cache["corr"][idxs_list].to(device=device, dtype=torch.float32)
        kl_k = kl_teacher_student(logp_student, logp_teacher).item()

        acc_k = accuracy_from_logits(logits_last, label_corr_ids.to(device)).item()
        ld_k = logit_diff(logits_last, label_corr_ids.to(device), wrong_corr_ids.to(device)).item()

        sums["kl_clean"] += kl_c
        sums["kl_corr"] += kl_k
        sums["acc_clean"] += acc_c
        sums["acc_corr"] += acc_k
        sums["logitdiff_clean"] += ld_c
        sums["logitdiff_corr"] += ld_k
        n_batches += 1

    if n_batches == 0:
        raise ValueError("Empty dataloader in eval_split.")

    out = {k: v / n_batches for k, v in sums.items()}
    out["kl_total"] = 0.5 * (out["kl_clean"] + out["kl_corr"])
    out["acc_total"] = 0.5 * (out["acc_clean"] + out["acc_corr"])
    out["logitdiff_total"] = 0.5 * (out["logitdiff_clean"] + out["logitdiff_corr"])
    return out


def train_loop(
    *,
    runner,
    mask_params,
    tokenizer,
    teacher_cache_train: Dict[str, torch.Tensor],  # CPU float16
    teacher_cache_val: Dict[str, torch.Tensor],    # CPU float16
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    cfg: TrainConfig,
    logger,
    run_dir: Path,
) -> Dict[str, any]:
    """
    Optimizes theta with loss = KL + lambda_l1 * mean(sigmoid(theta)).
    Early-stops on val KL_total.
    """
    runner.model.train()
    mask_params.train()

    opt = torch.optim.AdamW([mask_params.theta], lr=cfg.lr)

    best_val = float("inf")
    best_epoch = -1
    patience = 0

    history: List[Dict[str, float]] = []

    best_theta_path = run_dir / "masks" / "best_mask_logits.pt"
    best_mask_path = run_dir / "masks" / "best_mask_values.pt"

    for epoch in range(cfg.epochs):
        runner.model.train()
        mask_params.train()

        sums = {
            "kl_clean": 0.0,
            "kl_corr": 0.0,
            "acc_clean": 0.0,
            "acc_corr": 0.0,
            "logitdiff_clean": 0.0,
            "logitdiff_corr": 0.0,
            "loss": 0.0,
            "l1": 0.0,
        }
        n_batches = 0

        for batch in tqdm(train_loader, desc=f"train epoch {epoch}", leave=False):
            idxs, prompts_clean, prompts_corr, label_clean_ids, label_corr_ids, wrong_clean_ids, wrong_corr_ids = batch
            idxs_list = idxs.tolist()

            opt.zero_grad(set_to_none=True)

            # ---- Clean ----
            tokens, attn_mask, last_pos = tokenize_prompts(tokenizer, prompts_clean, device=device)
            logits = runner(tokens, attn_mask)
            bsz = logits.size(0)
            logits_last = logits[torch.arange(bsz, device=device), last_pos]
            logp_student_clean = torch.log_softmax(logits_last.float(), dim=-1)
            logp_teacher_clean = teacher_cache_train["clean"][idxs_list].to(device=device, dtype=torch.float32)
            kl_clean = kl_teacher_student(logp_student_clean, logp_teacher_clean)

            acc_clean = accuracy_from_logits(logits_last, label_clean_ids.to(device))
            ld_clean = logit_diff(logits_last, label_clean_ids.to(device), wrong_clean_ids.to(device))

            # ---- Corr ----
            tokens, attn_mask, last_pos = tokenize_prompts(tokenizer, prompts_corr, device=device)
            logits = runner(tokens, attn_mask)
            bsz = logits.size(0)
            logits_last = logits[torch.arange(bsz, device=device), last_pos]
            logp_student_corr = torch.log_softmax(logits_last.float(), dim=-1)
            logp_teacher_corr = teacher_cache_train["corr"][idxs_list].to(device=device, dtype=torch.float32)
            kl_corr = kl_teacher_student(logp_student_corr, logp_teacher_corr)

            acc_corr = accuracy_from_logits(logits_last, label_corr_ids.to(device))
            ld_corr = logit_diff(logits_last, label_corr_ids.to(device), wrong_corr_ids.to(device))

            kl_total = 0.5 * (kl_clean + kl_corr)

            l1 = mask_params.l1_mean()
            loss = kl_total + cfg.lambda_l1 * l1

            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([mask_params.theta], cfg.grad_clip)
            opt.step()

            sums["kl_clean"] += kl_clean.item()
            sums["kl_corr"] += kl_corr.item()
            sums["acc_clean"] += acc_clean.item()
            sums["acc_corr"] += acc_corr.item()
            sums["logitdiff_clean"] += ld_clean.item()
            sums["logitdiff_corr"] += ld_corr.item()
            sums["loss"] += loss.item()
            sums["l1"] += l1.item()
            n_batches += 1

        # Train epoch averages
        train_stats = {k: v / n_batches for k, v in sums.items()}
        train_stats["kl_total"] = 0.5 * (train_stats["kl_clean"] + train_stats["kl_corr"])
        train_stats["acc_total"] = 0.5 * (train_stats["acc_clean"] + train_stats["acc_corr"])
        train_stats["logitdiff_total"] = 0.5 * (train_stats["logitdiff_clean"] + train_stats["logitdiff_corr"])

        # Validation
        val_stats = eval_split(
            runner=runner,
            tokenizer=tokenizer,
            teacher_cache=teacher_cache_val,
            dataloader=val_loader,
            device=device,
        )

        epoch_rec = {
            "epoch": epoch,
            "train": train_stats,
            "val": val_stats,
            "l1_mean_mask": float(mask_params.l1_mean().item()),
        }
        history.append(epoch_rec)

        logger.info(
            f"[epoch {epoch}] "
            f"train KL={train_stats['kl_total']:.4f} (c={train_stats['kl_clean']:.4f}, k={train_stats['kl_corr']:.4f}) "
            f"val KL={val_stats['kl_total']:.4f} (c={val_stats['kl_clean']:.4f}, k={val_stats['kl_corr']:.4f}) "
            f"L1={epoch_rec['l1_mean_mask']:.4f} "
            f"val acc={val_stats['acc_total']:.4f}"
        )

        # Early stopping based on val KL_total
        if val_stats["kl_total"] < best_val - 1e-6:
            best_val = val_stats["kl_total"]
            best_epoch = epoch
            patience = 0

            # Save best masks
            best_theta_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(mask_params.theta.detach().cpu(), best_theta_path)
            torch.save(mask_params.mask_values().detach().cpu(), best_mask_path)
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                logger.info(f"Early stopping: no improvement for {cfg.early_stopping_patience} epochs.")
                break

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_kl": best_val,
        "best_theta_path": str(best_theta_path),
        "best_mask_path": str(best_mask_path),
    }

# ============================
# Joint-mode training utilities
# ============================

def _cycle(loader: DataLoader):
    while True:
        for batch in loader:
            yield batch

def _avg_metrics_equal_weight(per_task: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    tasks = list(per_task.keys())
    out: Dict[str, float] = {}
    for k in per_task[tasks[0]].keys():
        out[k] = float(sum(per_task[t][k] for t in tasks) / len(tasks))
    return out

def train_loop_joint(
    *,
    runner,
    mask_params,
    tokenizer,
    teacher_cache_train_by_task: Dict[str, Dict[str, torch.Tensor]],  # task -> {"clean": [N,V], "corr": [N,V]} CPU
    teacher_cache_val_by_task: Dict[str, Dict[str, torch.Tensor]],    # same
    train_loaders_by_task: Dict[str, DataLoader],
    val_loaders_by_task: Dict[str, DataLoader],
    device: torch.device,
    cfg: TrainConfig,
    logger,
    run_dir: Path,
) -> Dict[str, Any]:
    """
    Joint training with one shared theta:
      - Each step: take one minibatch from each task (equal task batch sizes by construction)
      - Do ONE clean forward + ONE corr forward on the concatenated batch
      - KL_total = mean_task( 0.5*(KL_clean_task + KL_corr_task) )
      - loss = KL_total + lambda_l1 * mean(sigmoid(theta))
      - early stop on equal-weight avg val KL_total
    """
    runner.model.train()
    mask_params.train()
    opt = torch.optim.AdamW([mask_params.theta], lr=cfg.lr)

    tasks = list(train_loaders_by_task.keys())
    iters = {t: _cycle(train_loaders_by_task[t]) for t in tasks}
    steps_per_epoch = min(len(train_loaders_by_task[t]) for t in tasks)

    best_val = float("inf")
    best_epoch = -1
    patience = 0
    history: List[Dict[str, Any]] = []

    best_theta_path = run_dir / "masks" / "best_mask_logits.pt"
    best_mask_path = run_dir / "masks" / "best_mask_values.pt"
    best_theta_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(cfg.epochs):
        runner.model.train()
        mask_params.train()

        train_sums = {t: {"kl_clean": 0.0, "kl_corr": 0.0, "kl_total": 0.0,
                          "acc_clean": 0.0, "acc_corr": 0.0, "acc_total": 0.0,
                          "logitdiff_clean": 0.0, "logitdiff_corr": 0.0, "logitdiff_total": 0.0}
                      for t in tasks}
        loss_sum = 0.0
        l1_sum = 0.0
        n_steps = 0

        for _ in tqdm(range(steps_per_epoch), desc=f"joint train epoch {epoch}", leave=False):
            # fetch one batch per task
            batches = {t: next(iters[t]) for t in tasks}

            # concatenate
            idxs_all = []
            prompts_clean_all = []
            prompts_corr_all = []
            label_clean_all = []
            label_corr_all = []
            wrong_clean_all = []
            wrong_corr_all = []

            sizes = {}
            for t in tasks:
                idxs, pclean, pcorr, lcid, lcoid, wcid, wcoid = batches[t]
                sizes[t] = int(idxs.shape[0])
                idxs_all.append(idxs)
                prompts_clean_all += list(pclean)
                prompts_corr_all += list(pcorr)
                label_clean_all.append(lcid)
                label_corr_all.append(lcoid)
                wrong_clean_all.append(wcid)
                wrong_corr_all.append(wcoid)

            idxs_all = torch.cat(idxs_all, dim=0)
            label_clean_all = torch.cat(label_clean_all, dim=0).to(device)
            label_corr_all = torch.cat(label_corr_all, dim=0).to(device)
            wrong_clean_all = torch.cat(wrong_clean_all, dim=0).to(device)
            wrong_corr_all = torch.cat(wrong_corr_all, dim=0).to(device)

            # tokenize + forward clean
            tokens, attn_mask, last_pos = tokenize_prompts(tokenizer, prompts_clean_all, device=device)
            logits = runner(tokens, attn_mask)
            bsz = logits.size(0)
            logits_last_clean = logits[torch.arange(bsz, device=device), last_pos]
            logp_student_clean = torch.log_softmax(logits_last_clean.float(), dim=-1)

            # tokenize + forward corr
            tokens, attn_mask, last_pos = tokenize_prompts(tokenizer, prompts_corr_all, device=device)
            logits = runner(tokens, attn_mask)
            bsz = logits.size(0)
            logits_last_corr = logits[torch.arange(bsz, device=device), last_pos]
            logp_student_corr = torch.log_softmax(logits_last_corr.float(), dim=-1)

            # build teacher batches (concat in same order)
            teacher_clean_parts = []
            teacher_corr_parts = []
            offset = 0
            slices = {}
            for t in tasks:
                n = sizes[t]
                sl = slice(offset, offset + n)
                slices[t] = sl
                idxs_t = idxs_all[sl].tolist()

                teacher_clean_parts.append(teacher_cache_train_by_task[t]["clean"][idxs_t])
                teacher_corr_parts.append(teacher_cache_train_by_task[t]["corr"][idxs_t])

                offset += n

            teacher_clean = torch.cat(teacher_clean_parts, dim=0).to(device=device, dtype=torch.float32)
            teacher_corr = torch.cat(teacher_corr_parts, dim=0).to(device=device, dtype=torch.float32)

            # per-task KLs (equal-weight)
            task_kl_totals = []
            for t in tasks:
                sl = slices[t]
                kcl = kl_teacher_student(logp_student_clean[sl], teacher_clean[sl])
                kco = kl_teacher_student(logp_student_corr[sl], teacher_corr[sl])
                kt = 0.5 * (kcl + kco)
                task_kl_totals.append(kt)

                # metrics for logging
                acc_c = accuracy_from_logits(logits_last_clean[sl], label_clean_all[sl])
                acc_k = accuracy_from_logits(logits_last_corr[sl], label_corr_all[sl])
                ld_c = logit_diff(logits_last_clean[sl], label_clean_all[sl], wrong_clean_all[sl])
                ld_k = logit_diff(logits_last_corr[sl], label_corr_all[sl], wrong_corr_all[sl])

                train_sums[t]["kl_clean"] += float(kcl.item())
                train_sums[t]["kl_corr"] += float(kco.item())
                train_sums[t]["kl_total"] += float(kt.item())
                train_sums[t]["acc_clean"] += float(acc_c.item())
                train_sums[t]["acc_corr"] += float(acc_k.item())
                train_sums[t]["acc_total"] += 0.5 * float(acc_c.item() + acc_k.item())
                train_sums[t]["logitdiff_clean"] += float(ld_c.item())
                train_sums[t]["logitdiff_corr"] += float(ld_k.item())
                train_sums[t]["logitdiff_total"] += 0.5 * float(ld_c.item() + ld_k.item())

            kl_total = sum(task_kl_totals) / len(task_kl_totals)
            l1 = mask_params.l1_mean()
            loss = kl_total + (cfg.lambda_l1 * l1)

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip is not None and cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_([mask_params.theta], cfg.grad_clip)
            opt.step()

            loss_sum += float(loss.item())
            l1_sum += float(l1.item())
            n_steps += 1

        # train averages
        train_per_task = {t: {k: v / max(n_steps, 1) for k, v in train_sums[t].items()} for t in tasks}
        train_overall = _avg_metrics_equal_weight(train_per_task)

        # val eval per task (full pass) using existing eval_split
        val_per_task = {}
        for t in tasks:
            val_per_task[t] = eval_split(
                runner=runner,
                tokenizer=tokenizer,
                teacher_cache=teacher_cache_val_by_task[t],
                dataloader=val_loaders_by_task[t],
                device=device,
            )
        val_overall = _avg_metrics_equal_weight(val_per_task)

        rec = {
            "epoch": epoch,
            "train": {"per_task": train_per_task, "overall": train_overall, "loss": loss_sum / max(n_steps,1), "l1": l1_sum / max(n_steps,1)},
            "val": {"per_task": val_per_task, "overall": val_overall},
            "l1_mean_mask": float(mask_params.l1_mean().item()),
        }
        history.append(rec)

        logger.info(
            f"[joint epoch {epoch}] "
            f"train KL={train_overall['kl_total']:.4f} "
            f"val KL={val_overall['kl_total']:.4f} "
            f"L1={rec['l1_mean_mask']:.4f}"
        )

        # early stop on overall val KL_total
        if val_overall["kl_total"] < best_val - 1e-6:
            best_val = val_overall["kl_total"]
            best_epoch = epoch
            patience = 0
            torch.save(mask_params.theta.detach().cpu(), best_theta_path)
            torch.save(mask_params.mask_values().detach().cpu(), best_mask_path)
        else:
            patience += 1
            if patience >= cfg.early_stopping_patience:
                logger.info(f"[joint] Early stopping: no improvement for {cfg.early_stopping_patience} epochs.")
                break

    return {
        "mode": "joint",
        "history": history,
        "best_epoch": best_epoch,
        "best_val_kl": best_val,
        "best_theta_path": str(best_theta_path),
        "best_mask_path": str(best_mask_path),
    }
