from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

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

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import torch


def _cycle(loader: Iterable):
    """Infinite iterator over a dataloader."""
    while True:
        for batch in loader:
            yield batch


def _encode_first_token_id(tokenizer, label_strs: List[str]) -> torch.Tensor:
    """
    Encode ' ' + label_str and take the first token id (per spec).
    Returns int64 tensor on CPU, shape [B].
    """
    ids: List[int] = []
    for s in label_strs:
        tok = tokenizer.encode(" " + str(s), add_special_tokens=False)
        ids.append(int(tok[0]))
    return torch.tensor(ids, dtype=torch.long)


def _tokenize_prompts(tokenizer, prompts: List[str], device: torch.device, max_length: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Tokenize prompts into (tokens, attention_mask), moved to device.
    Truncation protects against > context window.
    """
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
        add_special_tokens=False,
    )
    tokens = enc["input_ids"].to(device)
    attn_mask = enc["attention_mask"].to(device)
    return tokens, attn_mask


def _kl_teacher_student(teacher_logp: torch.Tensor, student_logp: torch.Tensor) -> torch.Tensor:
    """
    KL(teacher || student) where both are log-probs: E_t[ log t - log s ].
    teacher_logp, student_logp: [B, V] float tensors on same device.
    """
    teacher_p = teacher_logp.exp()
    return (teacher_p * (teacher_logp - student_logp)).sum(dim=-1).mean()


@dataclass
class JointSplitCaches:
    # teacher_logp[task]["clean"|"corr"] = tensor [N, V] on CPU (float16 or float32)
    teacher_logp: Dict[str, Dict[str, torch.Tensor]]


@torch.no_grad()
def _eval_one_task(
    runner,
    tokenizer,
    loader,
    caches: JointSplitCaches,
    task: str,
    device: torch.device,
    max_length: int,
) -> Dict[str, Optional[float]]:
    """
    Full-pass evaluation for one task over loader.
    Returns mean metrics for clean/corr/total.
    """
    sums = {
        "kl_clean": 0.0, "kl_corr": 0.0,
        "acc_clean": 0.0, "acc_corr": 0.0,
        "logitdiff_clean": 0.0, "logitdiff_corr": 0.0,
    }
    counts = {"n": 0, "n_logitdiff": 0}

    teacher_clean_all = caches.teacher_logp[task]["clean"]
    teacher_corr_all = caches.teacher_logp[task]["corr"]

    for batch in loader:
        # batch is a list of examples
        B = len(batch)
        if B == 0:
            continue

        idxs = torch.tensor([ex.idx for ex in batch], dtype=torch.long)
        prompts_clean = [ex.prompt_clean for ex in batch]
        prompts_corr = [ex.prompt_corr for ex in batch]

        label_clean_ids = _encode_first_token_id(tokenizer, [ex.label_clean_str for ex in batch]).to(device)
        label_corr_ids  = _encode_first_token_id(tokenizer, [ex.label_corr_str  for ex in batch]).to(device)

        # wrong labels (may be missing)
        have_wrong = all(getattr(ex, "wrong_clean_str", None) is not None and getattr(ex, "wrong_corr_str", None) is not None for ex in batch)
        if have_wrong:
            wrong_clean_ids = _encode_first_token_id(tokenizer, [ex.wrong_clean_str for ex in batch]).to(device)
            wrong_corr_ids  = _encode_first_token_id(tokenizer, [ex.wrong_corr_str  for ex in batch]).to(device)
        else:
            wrong_clean_ids = None
            wrong_corr_ids = None

        tokens_clean, attn_clean = _tokenize_prompts(tokenizer, prompts_clean, device, max_length)
        logits_clean = runner(tokens_clean, attn_clean)  # [B,P,V]
        logits_last_clean = logits_clean[:, -1, :]
        logp_student_clean = torch.log_softmax(logits_last_clean.float(), dim=-1)

        teacher_clean = teacher_clean_all.index_select(0, idxs).to(device).float()
        kl_clean = _kl_teacher_student(teacher_clean, logp_student_clean)

        pred_clean = logits_last_clean.argmax(dim=-1)
        acc_clean = (pred_clean == label_clean_ids).float().mean()

        if have_wrong:
            lc = logits_last_clean.gather(-1, label_clean_ids.unsqueeze(-1)).squeeze(-1)
            lw = logits_last_clean.gather(-1, wrong_clean_ids.unsqueeze(-1)).squeeze(-1)
            logitdiff_clean = (lc - lw).float().mean()
        else:
            logitdiff_clean = None

        tokens_corr, attn_corr = _tokenize_prompts(tokenizer, prompts_corr, device, max_length)
        logits_corr = runner(tokens_corr, attn_corr)
        logits_last_corr = logits_corr[:, -1, :]
        logp_student_corr = torch.log_softmax(logits_last_corr.float(), dim=-1)

        teacher_corr = teacher_corr_all.index_select(0, idxs).to(device).float()
        kl_corr = _kl_teacher_student(teacher_corr, logp_student_corr)

        pred_corr = logits_last_corr.argmax(dim=-1)
        acc_corr = (pred_corr == label_corr_ids).float().mean()

        if have_wrong:
            lc2 = logits_last_corr.gather(-1, label_corr_ids.unsqueeze(-1)).squeeze(-1)
            lw2 = logits_last_corr.gather(-1, wrong_corr_ids.unsqueeze(-1)).squeeze(-1)
            logitdiff_corr = (lc2 - lw2).float().mean()
        else:
            logitdiff_corr = None

        # accumulate
        sums["kl_clean"] += float(kl_clean.item()) * B
        sums["kl_corr"] += float(kl_corr.item()) * B
        sums["acc_clean"] += float(acc_clean.item()) * B
        sums["acc_corr"] += float(acc_corr.item()) * B

        if have_wrong:
            sums["logitdiff_clean"] += float(logitdiff_clean.item()) * B
            sums["logitdiff_corr"] += float(logitdiff_corr.item()) * B
            counts["n_logitdiff"] += B

        counts["n"] += B

    n = max(counts["n"], 1)
    out: Dict[str, Optional[float]] = {}
    out["kl_clean"] = sums["kl_clean"] / n
    out["kl_corr"] = sums["kl_corr"] / n
    out["kl_total"] = 0.5 * (out["kl_clean"] + out["kl_corr"])
    out["acc_clean"] = sums["acc_clean"] / n
    out["acc_corr"] = sums["acc_corr"] / n
    out["acc_total"] = 0.5 * (out["acc_clean"] + out["acc_corr"])

    if counts["n_logitdiff"] > 0:
        out["logitdiff_clean"] = sums["logitdiff_clean"] / n
        out["logitdiff_corr"] = sums["logitdiff_corr"] / n
        out["logitdiff_total"] = 0.5 * (out["logitdiff_clean"] + out["logitdiff_corr"])
    else:
        out["logitdiff_clean"] = None
        out["logitdiff_corr"] = None
        out["logitdiff_total"] = None

    return out


@torch.no_grad()
def eval_joint(
    runner,
    tokenizer,
    loaders: Dict[str, Iterable],
    caches: JointSplitCaches,
    device: torch.device,
    max_length: int,
) -> Dict[str, object]:
    """
    Evaluate each task fully and average metrics equally across tasks.
    Returns:
      {
        "per_task": {task: metrics},
        "overall": averaged_metrics
      }
    """
    per_task: Dict[str, Dict[str, Optional[float]]] = {}
    for task, loader in loaders.items():
        per_task[task] = _eval_one_task(runner, tokenizer, loader, caches, task, device, max_length)

    # equal-weight average across tasks
    tasks = list(per_task.keys())
    def avg(key: str) -> Optional[float]:
        vals = [per_task[t][key] for t in tasks]
        if any(v is None for v in vals):
            return None
        return float(sum(vals) / len(vals))  # type: ignore

    overall = {
        "kl_clean": avg("kl_clean"),
        "kl_corr": avg("kl_corr"),
        "kl_total": avg("kl_total"),
        "acc_clean": avg("acc_clean"),
        "acc_corr": avg("acc_corr"),
        "acc_total": avg("acc_total"),
        "logitdiff_clean": avg("logitdiff_clean"),
        "logitdiff_corr": avg("logitdiff_corr"),
        "logitdiff_total": avg("logitdiff_total"),
    }
    return {"per_task": per_task, "overall": overall}


def train_loop_joint(
    runner,
    mask_theta: torch.nn.Parameter,
    tokenizer,
    train_loaders: Dict[str, Iterable],
    val_loaders: Dict[str, Iterable],
    train_caches: JointSplitCaches,
    val_caches: JointSplitCaches,
    *,
    epochs: int,
    lr: float,
    lambda_l1: float,
    grad_clip: float,
    early_stopping_patience: int,
    device: torch.device,
    max_length: int,
    log_fn,
    save_best_fn,
) -> Dict[str, object]:
    """
    Joint training:
      - Each step samples one minibatch per task (equal batch size per task)
      - Concatenates them, runs one clean forward + one corr forward
      - Loss = avg_task(KL_total) + lambda_l1 * mean(sigmoid(theta))
      - Early stop on overall val KL_total (equal-weighted across tasks)
    `save_best_fn(epoch)` is called when val improves (you save masks in CLI).
    """
    optimizer = torch.optim.AdamW([mask_theta], lr=lr)

    # cycle iterators so all tasks contribute equally each epoch
    iters = {t: _cycle(dl) for t, dl in train_loaders.items()}
    tasks = list(train_loaders.keys())

    # define steps/epoch as the minimum number of batches among loaders
    # (so we don't hang if one loader is short)
    steps_per_epoch = min(len(train_loaders[t]) for t in tasks)

    history: List[Dict[str, object]] = []
    best_val = float("inf")
    best_epoch = -1
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        runner.model.eval()  # model frozen; but keeps deterministic behavior
        total_loss_sum = 0.0

        # accumulators
        train_task_sums = {t: {"kl_clean": 0.0, "kl_corr": 0.0, "n": 0} for t in tasks}

        for _ in range(steps_per_epoch):
            # fetch one batch per task
            batches = {t: next(iters[t]) for t in tasks}
            lens = {t: len(batches[t]) for t in tasks}
            if any(lens[t] == 0 for t in tasks):
                continue

            # build combined batch
            combined = []
            for t in tasks:
                combined.extend(batches[t])

            B = len(combined)
            # prompts / idxs
            idxs_by_task = {t: torch.tensor([ex.idx for ex in batches[t]], dtype=torch.long) for t in tasks}
            prompts_clean = [ex.prompt_clean for ex in combined]
            prompts_corr  = [ex.prompt_corr  for ex in combined]

            # labels
            label_clean_ids = _encode_first_token_id(tokenizer, [ex.label_clean_str for ex in combined]).to(device)
            label_corr_ids  = _encode_first_token_id(tokenizer, [ex.label_corr_str  for ex in combined]).to(device)

            have_wrong = all(getattr(ex, "wrong_clean_str", None) is not None and getattr(ex, "wrong_corr_str", None) is not None for ex in combined)
            if have_wrong:
                wrong_clean_ids = _encode_first_token_id(tokenizer, [ex.wrong_clean_str for ex in combined]).to(device)
                wrong_corr_ids  = _encode_first_token_id(tokenizer, [ex.wrong_corr_str  for ex in combined]).to(device)
            else:
                wrong_clean_ids = None
                wrong_corr_ids = None

            # tokenize once
            tokens_clean, attn_clean = _tokenize_prompts(tokenizer, prompts_clean, device, max_length)
            logits_clean = runner(tokens_clean, attn_clean)
            logits_last_clean = logits_clean[:, -1, :]
            logp_student_clean = torch.log_softmax(logits_last_clean.float(), dim=-1)

            tokens_corr, attn_corr = _tokenize_prompts(tokenizer, prompts_corr, device, max_length)
            logits_corr = runner(tokens_corr, attn_corr)
            logits_last_corr = logits_corr[:, -1, :]
            logp_student_corr = torch.log_softmax(logits_last_corr.float(), dim=-1)

            # build teacher batches by concatenating per-task gathered tensors
            teacher_clean_parts = []
            teacher_corr_parts = []
            for t in tasks:
                tc = train_caches.teacher_logp[t]["clean"].index_select(0, idxs_by_task[t])
                tr = train_caches.teacher_logp[t]["corr"].index_select(0, idxs_by_task[t])
                teacher_clean_parts.append(tc)
                teacher_corr_parts.append(tr)
            teacher_clean = torch.cat(teacher_clean_parts, dim=0).to(device).float()
            teacher_corr  = torch.cat(teacher_corr_parts, dim=0).to(device).float()

            # KL per task (slice)
            offset = 0
            task_kls_clean = {}
            task_kls_corr = {}
            for t in tasks:
                n = lens[t]
                sl = slice(offset, offset + n)
                task_kls_clean[t] = _kl_teacher_student(teacher_clean[sl], logp_student_clean[sl])
                task_kls_corr[t]  = _kl_teacher_student(teacher_corr[sl],  logp_student_corr[sl])
                offset += n

            # equal-weight KL across tasks
            kl_clean = sum(task_kls_clean.values()) / len(tasks)
            kl_corr  = sum(task_kls_corr.values()) / len(tasks)
            kl_total = 0.5 * (kl_clean + kl_corr)

            # sparsity
            l1_mean = torch.sigmoid(mask_theta).mean()
            loss = kl_total + (lambda_l1 * l1_mean)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([mask_theta], grad_clip)
            optimizer.step()

            total_loss_sum += float(loss.item())

            # update train sums (store per-task KLs as weighted by n)
            for t in tasks:
                n = lens[t]
                train_task_sums[t]["kl_clean"] += float(task_kls_clean[t].item()) * n
                train_task_sums[t]["kl_corr"]  += float(task_kls_corr[t].item()) * n
                train_task_sums[t]["n"]        += n

        # compute epoch train averages
        train_per_task = {}
        for t in tasks:
            n = max(train_task_sums[t]["n"], 1)
            kcl = train_task_sums[t]["kl_clean"] / n
            kco = train_task_sums[t]["kl_corr"] / n
            train_per_task[t] = {
                "kl_clean": kcl,
                "kl_corr": kco,
                "kl_total": 0.5 * (kcl + kco),
            }

        # val full eval per task, equal-weighted overall
        val_report = eval_joint(runner, tokenizer, val_loaders, val_caches, device, max_length)
        val_kl_total = float(val_report["overall"]["kl_total"])

        # log
        log_fn(f"[joint][epoch {epoch}] train_loss={total_loss_sum/max(steps_per_epoch,1):.6f} "
               f"val_kl_total={val_kl_total:.6f} l1_mean={float(torch.sigmoid(mask_theta).mean().item()):.6f}")

        history.append({
            "epoch": epoch,
            "train_per_task": train_per_task,
            "val": val_report,
            "l1_mean_mask": float(torch.sigmoid(mask_theta).mean().item()),
        })

        # early stopping on overall val KL_total
        if val_kl_total < best_val - 1e-8:
            best_val = val_kl_total
            best_epoch = epoch
            bad_epochs = 0
            save_best_fn(epoch)
        else:
            bad_epochs += 1
            if bad_epochs >= early_stopping_patience:
                log_fn(f"[joint] Early stopping at epoch {epoch} (best_epoch={best_epoch}, best_val_kl={best_val:.6f})")
                break

    return {
        "history": history,
        "best_epoch": best_epoch,
        "best_val_kl_total": best_val,
    }
