import os
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Optional, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from transformer_lens import HookedTransformer

from ovmask.data import (
    load_task_csv,
    attach_label_token_ids,
    SimpleDataset,
    collate_fn,
    resolve_csv,
)
from ovmask.svd_cache import SVDSpec, precompute_svd, load_all_svd
from ovmask.teacher_cache import build_teacher_cache, TeacherShardReader
from ovmask.masked_ov import freeze_model_params, build_mask_state, MaskedOVHook
from ovmask.utils import (
    set_seed,
    make_run_dir,
    setup_logger,
    save_json,
    append_json_list,
    get_git_commit_hash,
    dtype_from_str,
    tokenize_prompts,
    last_token_indices,
    select_last_logits,
)


@torch.no_grad()
def forward_logp_next(model, prompts, device: torch.device) -> torch.Tensor:
    """
    Returns logp over vocab for next token after each prompt.
    Shape: [batch, vocab]
    """
    tokenizer = model.tokenizer
    input_ids, attn_mask = tokenize_prompts(tokenizer, prompts, device=device)
    logits = model(input_ids)  # [b, seq, vocab]
    last_idx = last_token_indices(attn_mask)
    logits_last = select_last_logits(logits, last_idx)  # [b, vocab]
    logp = torch.log_softmax(logits_last.float(), dim=-1)
    return logp


def kl_from_teacher_logp(logp_teacher: torch.Tensor, logp_student: torch.Tensor) -> torch.Tensor:
    """
    KL(teacher || student) = sum p_t * (logp_t - logp_s)
    Both: [batch, vocab]
    """
    p_teacher = torch.exp(logp_teacher)
    return (p_teacher * (logp_teacher - logp_student)).sum(dim=-1).mean()


def batch_accuracy_from_logits(logp_student: torch.Tensor, label_ids: torch.Tensor) -> float:
    pred = torch.argmax(logp_student, dim=-1)
    return float((pred == label_ids).float().mean().item())


def batch_logit_diff(logp_student: torch.Tensor, label_ids: torch.Tensor, wrong_ids: Optional[torch.Tensor]) -> Optional[float]:
    if wrong_ids is None:
        return None
    # logp is log-softmax; logit diff requested is on logits, but ranking is same.
    # We'll compute diff on logp (monotone with logits) - acceptable for reporting;
    # If you want exact logits, change to logits_last (easy).
    b = logp_student.shape[0]
    label_lp = logp_student[torch.arange(b, device=logp_student.device), label_ids]
    wrong_lp = logp_student[torch.arange(b, device=logp_student.device), wrong_ids]
    return float((label_lp - wrong_lp).mean().item())


def completion_logprob_sum(
    model,
    prompts: List[str],
    completions: List[Optional[str]],
    device: torch.device,
) -> torch.Tensor:
    """Teacher-forced multi-token scoring.

    Returns a float tensor [batch] with the sum of log-probabilities of the
    completion tokens under the model, conditioned on the prompt.

    Why: IOI (names) are often *multi-token* in GPT-2 BPE, so single-token
    accuracy/logit-diff is misleading.
    """
    tok = model.tokenizer
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    input_id_tensors: List[torch.Tensor] = []
    start_idxs: List[int] = []
    seq_lens: List[int] = []

    for p, c in zip(prompts, completions):
        if c is None:
            ids = tok.encode(p, add_special_tokens=False)
            if len(ids) == 0:
                ids = [tok.eos_token_id]
            input_id_tensors.append(torch.tensor(ids, dtype=torch.long))
            start_idxs.append(len(ids))  # no completion
            seq_lens.append(len(ids))
            continue

        full = p + c

        # Default boundary: prompt tokenization is a prefix of full tokenization.
        prompt_ids = tok.encode(p, add_special_tokens=False)
        full_ids = tok.encode(full, add_special_tokens=False)
        start = len(prompt_ids)

        # Rare fallback: if prefix property fails, try to locate boundary using offsets.
        if len(prompt_ids) > len(full_ids) or full_ids[: len(prompt_ids)] != prompt_ids:
            try:
                enc = tok(full, add_special_tokens=False, return_offsets_mapping=True)
                full_ids = enc["input_ids"]
                offsets = enc["offset_mapping"]
                plen = len(p)
                start = len(full_ids)
                for i, (s, e) in enumerate(offsets):
                    if (s, e) == (0, 0):
                        continue
                    if s >= plen:
                        start = i
                        break
            except TypeError:
                # Tokenizer doesn't support offset mapping; keep the prompt-length guess.
                pass

        input_id_tensors.append(torch.tensor(full_ids, dtype=torch.long))
        start_idxs.append(start)
        seq_lens.append(len(full_ids))

    # Pad to a batch tensor.
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_id_tensors, batch_first=True, padding_value=pad_id
    ).to(device)

    # Attention mask based on true lengths (do NOT trust pad_id==eos semantics).
    bsz, max_len = input_ids.shape
    attn_mask = torch.zeros((bsz, max_len), device=device, dtype=torch.long)
    for b, L in enumerate(seq_lens):
        attn_mask[b, :L] = 1

    logits = model(input_ids)  # [b, seq, vocab]
    logp_step = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)  # predicts token t_{i+1} from position i

    scores = torch.zeros((bsz,), device=device, dtype=torch.float32)
    for b in range(bsz):
        start = int(start_idxs[b])
        L = int(seq_lens[b])
        if start >= L:
            continue
        # Completion tokens are at positions [start, ..., L-1] in input_ids.
        start_eff = max(start, 1)  # need a previous position to predict from
        if start_eff >= L:
            continue
        pos = torch.arange(start_eff, L, device=device, dtype=torch.long)
        prev = pos - 1
        tgt = input_ids[b, pos]
        scores[b] = logp_step[b, prev, tgt].sum()

    return scores


def pairwise_acc_and_diff(label_scores: torch.Tensor, wrong_scores: torch.Tensor) -> Tuple[float, float]:
    """Return (accuracy, mean_diff) for label vs wrong using summed log-probs."""
    diff = label_scores - wrong_scores
    acc = float((diff > 0).float().mean().item())
    mean_diff = float(diff.mean().item())
    return acc, mean_diff

@torch.no_grad()
def eval_split(
    student_model,
    masked_hook: MaskedOVHook,
    teacher_clean: TeacherShardReader,
    teacher_corr: TeacherShardReader,
    dl: DataLoader,
    device: torch.device,
) -> Dict[str, Any]:
    student_model.eval()

    total_kl_clean = 0.0
    total_kl_corr = 0.0
    total_n = 0

    total_acc_clean = 0.0
    total_acc_corr = 0.0

    diffs_clean = []
    diffs_corr = []

    for batch in tqdm(dl, desc="eval", leave=False):
        idx = torch.tensor(batch["idx"], device=device, dtype=torch.long)

        prompts_all = batch["prompt_clean"] + batch["prompt_corr"]
        logp_all = forward_logp_next(student_model, prompts_all, device=device)
        bsz = len(batch["prompt_clean"])
        logp_clean = logp_all[:bsz]
        logp_corr = logp_all[bsz:]

        # teacher logp
        t_clean = teacher_clean.get(idx, device=device)
        t_corr = teacher_corr.get(idx, device=device)

        kl_clean = kl_from_teacher_logp(t_clean, logp_clean)
        kl_corr = kl_from_teacher_logp(t_corr, logp_corr)

        label_clean = torch.tensor(batch["label_clean_id"], device=device, dtype=torch.long)
        label_corr = torch.tensor(batch["label_corr_id"], device=device, dtype=torch.long)

        # Accuracy/logit-diff metrics (reporting only; loss is KL).
        # Prefer multi-token scoring when label strings are available (important for IOI names).
        use_multitoken = (
            ("label_clean_str" in batch)
            and ("wrong_clean_str" in batch)
            and (not all(x is None for x in batch["wrong_clean_str"]))
        )
        if use_multitoken:
            # Sum log p(label | prompt) over all label tokens, and compare vs wrong label.
            label_lp_clean = completion_logprob_sum(student_model, batch["prompt_clean"], batch["label_clean_str"], device)
            wrong_lp_clean = completion_logprob_sum(student_model, batch["prompt_clean"], batch["wrong_clean_str"], device)
            label_lp_corr = completion_logprob_sum(student_model, batch["prompt_corr"], batch["label_corr_str"], device)
            wrong_lp_corr = completion_logprob_sum(student_model, batch["prompt_corr"], batch["wrong_corr_str"], device)
            acc_clean, diff_clean = pairwise_acc_and_diff(label_lp_clean, wrong_lp_clean)
            acc_corr, diff_corr = pairwise_acc_and_diff(label_lp_corr, wrong_lp_corr)
        else:
            # Fallback: single-token accuracy/logit-diff (only valid when labels are truly 1 token).
            acc_clean = batch_accuracy_from_logits(logp_clean, label_clean)
            acc_corr = batch_accuracy_from_logits(logp_corr, label_corr)

            wrong_clean_list = batch["wrong_clean_id"]
            wrong_corr_list = batch["wrong_corr_id"]
            wrong_clean = None if all(x is None for x in wrong_clean_list) else torch.tensor(
                [(-1 if x is None else x) for x in wrong_clean_list], device=device, dtype=torch.long
            )
            wrong_corr = None if all(x is None for x in wrong_corr_list) else torch.tensor(
                [(-1 if x is None else x) for x in wrong_corr_list], device=device, dtype=torch.long
            )

            diff_clean = None if wrong_clean is None else batch_logit_diff(logp_clean, label_clean, wrong_clean)
            diff_corr = None if wrong_corr is None else batch_logit_diff(logp_corr, label_corr, wrong_corr)

        total_kl_clean += float(kl_clean.item()) * bsz
        total_kl_corr += float(kl_corr.item()) * bsz
        total_acc_clean += acc_clean * bsz
        total_acc_corr += acc_corr * bsz
        total_n += bsz

        if diff_clean is not None:
            diffs_clean.append(diff_clean)
        if diff_corr is not None:
            diffs_corr.append(diff_corr)

    out = {
        "KL_clean": total_kl_clean / max(1, total_n),
        "KL_corr": total_kl_corr / max(1, total_n),
        "KL_total": 0.5 * ((total_kl_clean / max(1, total_n)) + (total_kl_corr / max(1, total_n))),
        "acc_clean": total_acc_clean / max(1, total_n),
        "acc_corr": total_acc_corr / max(1, total_n),
        "L1_mean_mask": float(masked_hook.l1_mean().item()),
    }
    if len(diffs_clean) > 0:
        out["logit_diff_clean_mean"] = sum(diffs_clean) / len(diffs_clean)
    if len(diffs_corr) > 0:
        out["logit_diff_corr_mean"] = sum(diffs_corr) / len(diffs_corr)
    return out


def _build_models_and_data_single(args, task: str, train_csv: str, val_csv: str, test_csv: str, run_dir: str, logger):
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_str(args.dtype)

    # Teacher model (original)
    teacher_model = HookedTransformer.from_pretrained("gpt2", device=device, dtype=dtype)
    freeze_model_params(teacher_model)
    teacher_model.eval()

    # Student model (we add hooks that override attention output)
    student_model = HookedTransformer.from_pretrained("gpt2", device=device, dtype=dtype)
    freeze_model_params(student_model)
    student_model.eval()

    tokenizer = teacher_model.tokenizer

    # Load data
    train_path = resolve_csv(args.data_dir, train_csv)
    val_path = resolve_csv(args.data_dir, val_csv)
    test_path = resolve_csv(args.data_dir, test_csv)

    train_exs = load_task_csv(task, train_path)
    val_exs = load_task_csv(task, val_path)
    test_exs = load_task_csv(task, test_path)

    attach_label_token_ids(train_exs, tokenizer)
    attach_label_token_ids(val_exs, tokenizer)
    attach_label_token_ids(test_exs, tokenizer)

    # SVD precompute/load
    spec = precompute_svd(args.svd_cache_dir, teacher_model, force=args.force_recompute_svd)

    U, S, Vh = load_all_svd(args.svd_cache_dir, spec, device=device, dtype=dtype)
    state = build_mask_state(U, S, Vh, init_theta=4.0, device=device)
    masked_hook = MaskedOVHook(state).to(device)

    # Register hooks (IMPORTANT: reset first so we don't stack hooks if user runs multiple times)
    student_model.reset_hooks()
    for l in range(spec.n_layers):
        student_model.add_hook(f"blocks.{l}.attn.hook_z", masked_hook.hook_store_z, dir="fwd")
        student_model.add_hook(f"blocks.{l}.hook_attn_out", masked_hook.hook_replace_attn_out, dir="fwd")

    # Teacher cache for clean/corr prompts on each split
    def prompts_from(exs, which: str):
        return [getattr(e, f"prompt_{which}") for e in exs]

    for split_name, exs in [("train", train_exs), ("val", val_exs), ("test", test_exs)]:
        build_teacher_cache(
            base_dir=args.teacher_cache_dir,
            task=task,
            split=split_name,
            variant="clean",
            prompts=prompts_from(exs, "clean"),
            teacher_model=teacher_model,
            shard_size=args.teacher_shard_size,
            force=args.force_recompute_teacher,
            logger=logger,
        )
        build_teacher_cache(
            base_dir=args.teacher_cache_dir,
            task=task,
            split=split_name,
            variant="corr",
            prompts=prompts_from(exs, "corr"),
            teacher_model=teacher_model,
            shard_size=args.teacher_shard_size,
            force=args.force_recompute_teacher,
            logger=logger,
        )

    teacher_train_clean = TeacherShardReader(args.teacher_cache_dir, task, "train", "clean")
    teacher_train_corr = TeacherShardReader(args.teacher_cache_dir, task, "train", "corr")
    teacher_val_clean = TeacherShardReader(args.teacher_cache_dir, task, "val", "clean")
    teacher_val_corr = TeacherShardReader(args.teacher_cache_dir, task, "val", "corr")
    teacher_test_clean = TeacherShardReader(args.teacher_cache_dir, task, "test", "clean")
    teacher_test_corr = TeacherShardReader(args.teacher_cache_dir, task, "test", "corr")

    train_dl = DataLoader(
        SimpleDataset(train_exs),
        batch_size=args.batch_size,
        shuffle=args.shuffle_train,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_dl = DataLoader(
        SimpleDataset(val_exs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    test_dl = DataLoader(
        SimpleDataset(test_exs),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )

    return (teacher_model, student_model, masked_hook, spec,
            train_dl, val_dl, test_dl,
            teacher_train_clean, teacher_train_corr,
            teacher_val_clean, teacher_val_corr,
            teacher_test_clean, teacher_test_corr,
            device)


def _save_masks_and_selected(run_dir: str, masked_hook: MaskedOVHook):
    os.makedirs(os.path.join(run_dir, "masks"), exist_ok=True)
    theta = masked_hook.state.theta.detach().cpu()
    m = torch.sigmoid(masked_hook.state.theta.detach()).cpu()
    torch.save(theta, os.path.join(run_dir, "masks", "mask_logits.pt"))
    torch.save(m, os.path.join(run_dir, "masks", "mask_values.pt"))

    # selected.json: per (l,h) directions sorted by m_i * S_i
    scores = masked_hook.scores_m_times_s().detach().cpu()  # [L,H,r]
    selected = {}
    L, H, r = scores.shape
    for l in range(L):
        for h in range(H):
            vals = scores[l, h]
            order = torch.argsort(vals, descending=True).tolist()
            selected[f"l{l}_h{h}"] = [{"i": int(i), "score": float(vals[i].item())} for i in order]
    save_json(os.path.join(run_dir, "selected.json"), selected)


def train_separate_task(args):
    set_seed(args.seed)
    run_dir = make_run_dir(args.task)
    logger = setup_logger(run_dir)

    logger.info(f"Run dir: {run_dir}")

    commit = get_git_commit_hash()
    config = vars(args).copy()
    config["git_commit"] = commit
    save_json(os.path.join(run_dir, "config.json"), config)

    (teacher_model, student_model, masked_hook, spec,
     train_dl, val_dl, test_dl,
     teacher_train_clean, teacher_train_corr,
     teacher_val_clean, teacher_val_corr,
     teacher_test_clean, teacher_test_corr,
     device) = _build_models_and_data_single(args, args.task, args.train_csv, args.val_csv, args.test_csv, run_dir, logger)

    # Only theta is learnable
    params = [masked_hook.state.theta]
    opt = torch.optim.AdamW(params, lr=args.lr)

    best_val = float("inf")
    best_path = os.path.join(run_dir, "masks", "best_mask_logits.pt")
    patience = 0

    metrics_path = os.path.join(run_dir, "metrics.json")

    logger.info(f"Training: task={args.task}, epochs={args.epochs}, batch_size={args.batch_size}, lr={args.lr}, lambda_l1={args.lambda_l1}")

    global_step = 0
    for epoch in range(1, args.epochs + 1):
        student_model.train()
        total_loss = 0.0
        total_kl = 0.0
        total_l1 = 0.0
        total_n = 0

        for batch in tqdm(train_dl, desc=f"epoch {epoch}/{args.epochs}", leave=False):
            idx = torch.tensor(batch["idx"], device=device, dtype=torch.long)

            # forward student for clean+corr
            prompts_all = batch["prompt_clean"] + batch["prompt_corr"]
            logp_all = forward_logp_next(student_model, prompts_all, device=device)
            bsz = len(batch["prompt_clean"])
            logp_clean = logp_all[:bsz]
            logp_corr = logp_all[bsz:]

            # teacher logp
            t_clean = teacher_train_clean.get(idx, device=device)
            t_corr = teacher_train_corr.get(idx, device=device)

            kl_clean = kl_from_teacher_logp(t_clean, logp_clean)
            kl_corr = kl_from_teacher_logp(t_corr, logp_corr)
            kl = 0.5 * (kl_clean + kl_corr)

            l1 = masked_hook.l1_mean()
            loss = kl + args.lambda_l1 * l1

            opt.zero_grad(set_to_none=True)
            loss.backward()

            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

            opt.step()

            total_loss += float(loss.item()) * bsz
            total_kl += float(kl.item()) * bsz
            total_l1 += float(l1.item()) * bsz
            total_n += bsz

            global_step += 1
            if global_step % args.log_every_steps == 0:
                logger.info(f"[step {global_step}] loss={loss.item():.6f} kl={kl.item():.6f} l1={l1.item():.6f}")

        # ---- end epoch: eval val ----
        val_metrics = eval_split(student_model, masked_hook, teacher_val_clean, teacher_val_corr, val_dl, device=device)

        train_epoch = {
            "epoch": epoch,
            "train_loss": total_loss / max(1, total_n),
            "train_KL": total_kl / max(1, total_n),
            "train_L1_mean": total_l1 / max(1, total_n),
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        append_json_list(metrics_path, train_epoch)

        logger.info(
            f"[epoch {epoch}] train_loss={train_epoch['train_loss']:.6f} "
            f"train_KL={train_epoch['train_KL']:.6f} train_L1={train_epoch['train_L1_mean']:.6f} "
            f"val_KL_total={val_metrics['KL_total']:.6f} val_L1={val_metrics['L1_mean_mask']:.6f}"
        )

        # Early stopping
        if val_metrics["KL_total"] < best_val - 1e-8:
            best_val = val_metrics["KL_total"]
            patience = 0
            os.makedirs(os.path.join(run_dir, "masks"), exist_ok=True)
            torch.save(masked_hook.state.theta.detach().cpu(), best_path)
            logger.info(f"New best val KL_total={best_val:.6f} -> saved best mask logits")
        else:
            patience += 1
            logger.info(f"No improvement. patience={patience}/{args.early_stopping_patience}")
            if patience >= args.early_stopping_patience:
                logger.info("Early stopping triggered.")
                break

    # Load best mask for final test eval
    if os.path.exists(best_path):
        best_theta = torch.load(best_path, map_location="cpu").to(device=device)
        masked_hook.state.theta.data.copy_(best_theta)

    # Save final masks + selected
    _save_masks_and_selected(run_dir, masked_hook)

    # Test eval
    test_metrics = eval_split(student_model, masked_hook, teacher_test_clean, teacher_test_corr, test_dl, device=device)
    save_json(os.path.join(run_dir, "final_eval.json"), test_metrics)
    logger.info(f"Test: KL_total={test_metrics['KL_total']:.6f}, acc_clean={test_metrics['acc_clean']:.4f}, acc_corr={test_metrics['acc_corr']:.4f}")

    logger.info("DONE.")


def train_joint_tasks(args):
    """
    Joint training: one shared mask, batches sampled from each task and averaged.
    """
    set_seed(args.seed)
    run_dir = make_run_dir("joint")
    logger = setup_logger(run_dir)
    logger.info(f"Run dir: {run_dir}")

    commit = get_git_commit_hash()
    config = vars(args).copy()
    config["git_commit"] = commit
    save_json(os.path.join(run_dir, "config.json"), config)

    # Build each task data + teacher caches, but share one student model + one masked_hook state.
    # We do that by:
    #   - build teacher_model once
    #   - precompute/load SVD once
    #   - build one MaskedOVHook (theta shared)
    #   - build dataloaders + teacher readers per task

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    dtype = dtype_from_str(args.dtype)

    teacher_model = HookedTransformer.from_pretrained("gpt2", device=device, dtype=dtype)
    freeze_model_params(teacher_model)
    teacher_model.eval()

    student_model = HookedTransformer.from_pretrained("gpt2", device=device, dtype=dtype)
    freeze_model_params(student_model)
    student_model.eval()

    # SVD once
    spec = precompute_svd(args.svd_cache_dir, teacher_model, force=args.force_recompute_svd)
    U, S, Vh = load_all_svd(args.svd_cache_dir, spec, device=device, dtype=dtype)
    state = build_mask_state(U, S, Vh, init_theta=4.0, device=device)
    masked_hook = MaskedOVHook(state).to(device)

    student_model.reset_hooks()
    for l in range(spec.n_layers):
        student_model.add_hook(f"blocks.{l}.attn.hook_z", masked_hook.hook_store_z, dir="fwd")
        student_model.add_hook(f"blocks.{l}.hook_attn_out", masked_hook.hook_replace_attn_out, dir="fwd")

    # helper to prep task
    def prep_task(task: str, train_csv: str, val_csv: str, test_csv: str):
        tokenizer = teacher_model.tokenizer
        train_exs = load_task_csv(task, resolve_csv(args.data_dir, train_csv))
        val_exs = load_task_csv(task, resolve_csv(args.data_dir, val_csv))
        test_exs = load_task_csv(task, resolve_csv(args.data_dir, test_csv))
        attach_label_token_ids(train_exs, tokenizer)
        attach_label_token_ids(val_exs, tokenizer)
        attach_label_token_ids(test_exs, tokenizer)

        def prompts_from(exs, which: str):
            return [getattr(e, f"prompt_{which}") for e in exs]

        for split_name, exs in [("train", train_exs), ("val", val_exs), ("test", test_exs)]:
            build_teacher_cache(args.teacher_cache_dir, task, split_name, "clean", prompts_from(exs, "clean"), teacher_model,
                               args.teacher_shard_size, args.force_recompute_teacher, logger)
            build_teacher_cache(args.teacher_cache_dir, task, split_name, "corr", prompts_from(exs, "corr"), teacher_model,
                               args.teacher_shard_size, args.force_recompute_teacher, logger)

        dls = {
            "train": DataLoader(SimpleDataset(train_exs), batch_size=args.batch_size, shuffle=args.shuffle_train,
                                num_workers=args.num_workers, collate_fn=collate_fn),
            "val": DataLoader(SimpleDataset(val_exs), batch_size=args.batch_size, shuffle=False,
                              num_workers=args.num_workers, collate_fn=collate_fn),
            "test": DataLoader(SimpleDataset(test_exs), batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=collate_fn),
        }
        teachers = {
            "train_clean": TeacherShardReader(args.teacher_cache_dir, task, "train", "clean"),
            "train_corr": TeacherShardReader(args.teacher_cache_dir, task, "train", "corr"),
            "val_clean": TeacherShardReader(args.teacher_cache_dir, task, "val", "clean"),
            "val_corr": TeacherShardReader(args.teacher_cache_dir, task, "val", "corr"),
            "test_clean": TeacherShardReader(args.teacher_cache_dir, task, "test", "clean"),
            "test_corr": TeacherShardReader(args.teacher_cache_dir, task, "test", "corr"),
        }
        return dls, teachers

    gp_dls, gp_t = prep_task("gp", args.gp_train_csv, args.gp_val_csv, args.gp_test_csv)
    ioi_dls, ioi_t = prep_task("ioi", args.ioi_train_csv, args.ioi_val_csv, args.ioi_test_csv)
    gt_dls, gt_t = prep_task("gt", args.gt_train_csv, args.gt_val_csv, args.gt_test_csv)

    params = [masked_hook.state.theta]
    opt = torch.optim.AdamW(params, lr=args.lr)

    best_val = float("inf")
    best_path = os.path.join(run_dir, "masks", "best_mask_logits.pt")
    patience = 0
    metrics_path = os.path.join(run_dir, "metrics.json")

    def joint_val():
        v_gp = eval_split(student_model, masked_hook, gp_t["val_clean"], gp_t["val_corr"], gp_dls["val"], device=device)
        v_ioi = eval_split(student_model, masked_hook, ioi_t["val_clean"], ioi_t["val_corr"], ioi_dls["val"], device=device)
        v_gt = eval_split(student_model, masked_hook, gt_t["val_clean"], gt_t["val_corr"], gt_dls["val"], device=device)
        return {
            "gp": v_gp, "ioi": v_ioi, "gt": v_gt,
            "KL_total_mean": (v_gp["KL_total"] + v_ioi["KL_total"] + v_gt["KL_total"]) / 3.0
        }

    logger.info("Joint training started (shared mask).")

    for epoch in range(1, args.epochs + 1):
        student_model.train()

        it_gp = iter(gp_dls["train"])
        it_ioi = iter(ioi_dls["train"])
        it_gt = iter(gt_dls["train"])

        steps = min(len(gp_dls["train"]), len(ioi_dls["train"]), len(gt_dls["train"]))
        total_loss = 0.0

        for step in tqdm(range(steps), desc=f"joint epoch {epoch}", leave=False):
            batches = []
            for it_ in [it_gp, it_ioi, it_gt]:
                try:
                    batches.append(next(it_))
                except StopIteration:
                    return

            losses = []
            for batch, t_clean, t_corr in [
                (batches[0], gp_t["train_clean"], gp_t["train_corr"]),
                (batches[1], ioi_t["train_clean"], ioi_t["train_corr"]),
                (batches[2], gt_t["train_clean"], gt_t["train_corr"]),
            ]:
                idx = torch.tensor(batch["idx"], device=device, dtype=torch.long)
                prompts_all = batch["prompt_clean"] + batch["prompt_corr"]
                logp_all = forward_logp_next(student_model, prompts_all, device=device)
                bsz = len(batch["prompt_clean"])
                logp_clean = logp_all[:bsz]
                logp_corr = logp_all[bsz:]
                tc = t_clean.get(idx, device=device)
                tr = t_corr.get(idx, device=device)
                kl = 0.5 * (kl_from_teacher_logp(tc, logp_clean) + kl_from_teacher_logp(tr, logp_corr))
                losses.append(kl)

            kl_mean = sum(losses) / 3.0
            l1 = masked_hook.l1_mean()
            loss = kl_mean + args.lambda_l1 * l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            if args.grad_clip is not None and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(params, args.grad_clip)
            opt.step()

            total_loss += float(loss.item())

        val = joint_val()
        append_json_list(metrics_path, {"epoch": epoch, "train_loss_mean": total_loss / max(1, steps), **val})

        logger.info(f"[epoch {epoch}] train_loss_mean={total_loss/max(1,steps):.6f} val_KL_total_mean={val['KL_total_mean']:.6f}")

        if val["KL_total_mean"] < best_val - 1e-8:
            best_val = val["KL_total_mean"]
            patience = 0
            os.makedirs(os.path.join(run_dir, "masks"), exist_ok=True)
            torch.save(masked_hook.state.theta.detach().cpu(), best_path)
            logger.info("New best -> saved best mask.")
        else:
            patience += 1
            if patience >= args.early_stopping_patience:
                logger.info("Early stopping (joint).")
                break

    if os.path.exists(best_path):
        best_theta = torch.load(best_path, map_location="cpu").to(device=device)
        masked_hook.state.theta.data.copy_(best_theta)

    _save_masks_and_selected(run_dir, masked_hook)

    # joint test
    t_gp = eval_split(student_model, masked_hook, gp_t["test_clean"], gp_t["test_corr"], gp_dls["test"], device=device)
    t_ioi = eval_split(student_model, masked_hook, ioi_t["test_clean"], ioi_t["test_corr"], ioi_dls["test"], device=device)
    t_gt = eval_split(student_model, masked_hook, gt_t["test_clean"], gt_t["test_corr"], gt_dls["test"], device=device)

    save_json(os.path.join(run_dir, "final_eval.json"), {"gp": t_gp, "ioi": t_ioi, "gt": t_gt})
    logger.info("DONE (joint).")
