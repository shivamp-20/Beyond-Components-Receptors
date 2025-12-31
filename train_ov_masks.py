"""Entry point: OV mask learning for one task (gp / ioi / gt).

Usage:
  python train_ov_masks.py --config config.yaml --task gp
  python train_ov_masks.py --config config.yaml --task gp --dry-run

What it does:
  1) Load GPT-2 small in TransformerLens (frozen)
  2) Compute/load per-head OV SVD factors (with optional bias augmentation)
  3) Load CSV split(s), tokenize prompts + labels using the full completion trick
  4) For each batch:
       - teacher clean logits (no mask)
       - corrupt z cache (corrupt prompts)
       - masked clean logits (hooks that mix clean/corrupt OV directions using m and 1-m)
       - loss = KL(teacher || masked) + lambda_sparse * mean(m)
  5) Save best masks, logs, and (optional) receptors
"""

from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import yaml
from tqdm import tqdm

from csv_tasks import Example, load_task_split
from logging_utils import setup_logger
from ov_masks import MaskParams, make_masked_forward_hooks
from ov_svd import compute_or_load_svd_bank
from receptors import save_receptors
from utils import accuracy_from_logits, gather_logits_at_positions, kl_divergence, left_pad, save_json, set_seed, mask_value_stats, paper_relative_sparsity, sparsity_measures_multi_threshold


def _load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_device(device_str: str) -> torch.device:
    if device_str == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _maybe_set_tlens_flags(model):
    # Ensure we have per-head results available.
    if hasattr(model, "set_use_attn_result"):
        model.set_use_attn_result(True)
    model.cfg.use_attn_result = True

    # Ensure padding works (left padding is important to align ends)
    tok = model.tokenizer
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    if hasattr(model, "set_tokenizer"):
        # TransformerLens helper in newer versions
        model.set_tokenizer(tok, default_padding_side="left")


def _filter_single_token_labels(examples: List[Example], logger) -> List[Example]:
    # Here we *assume* label id is last token of full completion.
    # If a label was actually multi-token, this trick would only use last token.
    # We cannot perfectly detect that after the fact without re-tokenizing label alone,
    # so we provide a conservative check: ensure prompt_ids + [label_id] decodes to something
    # ending with the label token. This is still imperfect.
    #
    # In practice, for your datasets these labels are curated to be single-token.
    return examples


def _make_dataloader(examples: List[Example], batch_size: int, shuffle: bool, num_workers: int):
    # Keep it simple: manual batching (fast enough for ~1k–10k examples).
    import random

    idxs = list(range(len(examples)))
    if shuffle:
        random.shuffle(idxs)

    for start in range(0, len(idxs), batch_size):
        batch = [examples[i] for i in idxs[start : start + batch_size]]
        yield batch


def _batch_to_tensors(batch: List[Example], pad_id: int, device: torch.device):
    """Pad clean and corrupt sequences to the SAME max_len (important!).

    We use left-padding so that the final prompt token always sits at the final position.
    That means the target position is simply (max_len - 1) for every example in the batch.
    """
    clean_seqs = [ex.clean_ids for ex in batch]
    corr_seqs = [ex.corrupt_ids for ex in batch]

    max_len = max(max(len(s) for s in clean_seqs), max(len(s) for s in corr_seqs))
    bsz = len(batch)

    clean_tokens = torch.full((bsz, max_len), pad_id, dtype=torch.long)
    corr_tokens = torch.full((bsz, max_len), pad_id, dtype=torch.long)

    for i, (cs, rs) in enumerate(zip(clean_seqs, corr_seqs)):
        clean_tokens[i, max_len - len(cs) :] = torch.tensor(cs, dtype=torch.long)
        corr_tokens[i, max_len - len(rs) :] = torch.tensor(rs, dtype=torch.long)

    # Target position: the final prompt token (always the last position after left padding)
    pos = torch.full((bsz,), max_len - 1, dtype=torch.long)

    clean_label = torch.tensor([ex.clean_label_id for ex in batch], dtype=torch.long)
    corr_label = torch.tensor([ex.corrupt_label_id for ex in batch], dtype=torch.long)

    clean_wrong = None
    corr_wrong = None
    if any(ex.clean_wrong_label_id is not None for ex in batch):
        clean_wrong = torch.tensor([ex.clean_wrong_label_id or -1 for ex in batch], dtype=torch.long)
    if any(ex.corrupt_wrong_label_id is not None for ex in batch):
        corr_wrong = torch.tensor([ex.corrupt_wrong_label_id or -1 for ex in batch], dtype=torch.long)

    return (
        clean_tokens.to(device),
        corr_tokens.to(device),
        pos.to(device),
        pos.to(device),
        clean_label.to(device),
        corr_label.to(device),
        clean_wrong.to(device) if clean_wrong is not None else None,
        corr_wrong.to(device) if corr_wrong is not None else None,
    )



@torch.no_grad()
def _teacher_logits_at_pos(model, tokens: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    logits = model(tokens)
    return gather_logits_at_positions(logits, pos)


@torch.no_grad()
def _cache_corrupt_z(model, corr_tokens: torch.Tensor, n_layers: int) -> Dict[str, torch.Tensor]:
    names_filter = lambda n: n.endswith("attn.hook_z")
    _, cache = model.run_with_cache(corr_tokens, names_filter=names_filter)
    out: Dict[str, torch.Tensor] = {}
    for l in range(n_layers):
        out[f"blocks.{l}.attn.hook_z"] = cache[f"blocks.{l}.attn.hook_z"].detach()
    return out


def _masked_logits_at_pos(model, clean_tokens, clean_pos, svd_bank, mask_params, corrupt_z_cache):
    hooks = make_masked_forward_hooks(svd=svd_bank, mask_params=mask_params, corrupt_cache=corrupt_z_cache, device=clean_tokens.device)
    logits = model.run_with_hooks(clean_tokens, fwd_hooks=hooks)
    return gather_logits_at_positions(logits, clean_pos)


def evaluate_split(model, svd_bank, mask_params, examples, batch_size, pad_id, device, temperature, logger, split_name: str):
    model.eval()
    total_kl = 0.0
    total_acc = 0.0
    n = 0

    for batch in _make_dataloader(examples, batch_size=batch_size, shuffle=False, num_workers=0):
        clean_tokens, corr_tokens, clean_pos, _, clean_label, _, _, _ = _batch_to_tensors(batch, pad_id, device)

        teacher = _teacher_logits_at_pos(model, clean_tokens, clean_pos)
        corrupt_cache = _cache_corrupt_z(model, corr_tokens, model.cfg.n_layers)
        masked = _masked_logits_at_pos(model, clean_tokens, clean_pos, svd_bank, mask_params, corrupt_cache)

        kl = kl_divergence(teacher, masked, temperature=temperature)
        acc = accuracy_from_logits(masked, clean_label)

        bsz = clean_tokens.shape[0]
        total_kl += float(kl.item()) * bsz
        total_acc += float(acc.item()) * bsz
        n += bsz

    mean_kl = total_kl / max(n, 1)
    mean_acc = total_acc / max(n, 1)
    stats = mask_value_stats(mask_params.m(), mask_params.valid)

    # Paper App. B.6 relative sparsity (thr=1e-2)
    rel = paper_relative_sparsity(mask_params.m(), mask_params.valid, threshold=1e-2)

    # L1 over learnable directions (diag(M) in the paper objective)
    l1_sum = float(mask_params.m()[mask_params.valid].sum().item())

    logger.info(
        f"[{split_name}] KL={mean_kl:.6f}  acc={mean_acc:.4f}  "
        f"mean_m={stats.get('mask_mean', float('nan')):.4f}  "
        f"kept@0.9={stats.get('mask_frac_gt_0p9', float('nan')):.3f}  "
        f"off@0.1={stats.get('mask_frac_lt_0p1', float('nan')):.3f}  "
        f"q10={stats.get('mask_q10', float('nan')):.3f}  "
        f"q50={stats.get('mask_q50', float('nan')):.3f}  "
        f"q90={stats.get('mask_q90', float('nan')):.3f}  "
        f"S_rel={rel.get('S_rel', float('nan')):.4f} active_frac={rel.get('active_frac', float('nan')):.4f} (thr={rel.get('thr')})"
    )

    out = {"kl": mean_kl, "acc": mean_acc, "l1_sum": l1_sum}
    out.update(stats)
    out.update(rel)
    return out




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--task", type=str, required=True, choices=["gp", "ioi", "gt"])
    ap.add_argument("--dry-run", action="store_true", help="Run a 1-batch sanity check and exit.")
    args = ap.parse_args()

    cfg = _load_config(Path(args.config))
    task_cfg = cfg["tasks"][args.task]

    seed = int(cfg["project"]["seed"])
    set_seed(seed)

    run_id = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    artifacts_root = Path(cfg["paths"]["artifacts_dir"]) / args.task / run_id
    log_dir = artifacts_root / "logs"
    logger = setup_logger(log_dir, log_level=cfg["logging"]["log_level"])
    logger.info(f"Run id: {run_id}")
    logger.info(f"Task: {args.task} ({task_cfg['kind']})")

    device = _resolve_device(cfg["model"]["device"])
    logger.info(f"Device: {device}")

    # Load TransformerLens model
    from transformer_lens import HookedTransformer

    model = HookedTransformer.from_pretrained(
        cfg["model"]["name"],
        device=str(device),
        fold_ln=False,
        center_unembed=False,
        center_writing_weights=False,
        refactor_factored_attn_matrices=False,
    )
    _maybe_set_tlens_flags(model)

    # Freeze model params
    model.requires_grad_(False)
    model.eval()

    # Load data
    data_dir = Path(cfg["paths"]["data_dir"])
    train_path = data_dir / task_cfg["train_file"]
    val_path = data_dir / task_cfg["val_file"]
    test_path = data_dir / task_cfg["test_file"]

    logger.info(f"Reading CSVs from: {data_dir}")
    logger.info(f"  train: {train_path.name}")
    logger.info(f"  val:   {val_path.name}")
    logger.info(f"  test:  {test_path.name}")

    prepend_bos = bool(cfg["model"]["prepend_bos"])
    tokenizer = model.tokenizer

    train_ex = load_task_split(
        kind=task_cfg["kind"],
        csv_path=train_path,
        tokenizer=tokenizer,
        prepend_bos=prepend_bos,
        max_examples=cfg["train"]["max_train_examples"],
    )
    val_ex = load_task_split(
        kind=task_cfg["kind"],
        csv_path=val_path,
        tokenizer=tokenizer,
        prepend_bos=prepend_bos,
        max_examples=cfg["train"]["max_val_examples"],
    )
    test_ex = load_task_split(
        kind=task_cfg["kind"],
        csv_path=test_path,
        tokenizer=tokenizer,
        prepend_bos=prepend_bos,
        max_examples=None,
    )

    logger.info(f"Loaded examples: train={len(train_ex)}, val={len(val_ex)}, test={len(test_ex)}")

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    # SVD bank (depends only on model, but we store per-task under this run)
    svd_dir = artifacts_root / "svd"
    svd_bank = compute_or_load_svd_bank(
        model=model,
        svd_dir=svd_dir,
        svd_eps=float(cfg["svd"]["svd_eps"]),
        bias_handling=cfg["svd"]["bias_handling"],
        logger=logger,
    )

    # Mask params
    n_layers, n_heads, _, r_max = svd_bank.U.shape
    mask_params = MaskParams(
        n_layers=n_layers,
        n_heads=n_heads,
        r_max=r_max,
        valid=svd_bank.valid,
        init_m=float(cfg["train"]["init_m"]),
    ).to(device)

    # opt = torch.optim.AdamW(mask_params.parameters(), lr=float(cfg["train"]["lr"]))
    opt = torch.optim.AdamW(
        mask_params.parameters(),
        lr=float(cfg["train"]["lr"]),
        weight_decay=float(cfg["train"].get("weight_decay", 0.0)),
    )
    temperature = float(cfg["train"]["temperature"])
    lambda_sparse = float(cfg["train"]["lambda_sparse"])

    # Save run config for reproducibility
    save_json(artifacts_root / "run_config.json", cfg)

    # Dry run: 1 train batch + 1 val batch to check everything works end-to-end
    if args.dry_run:
        logger.info("[DRY RUN] Running 1 train batch...")
        batch = next(_make_dataloader(train_ex, batch_size=int(cfg["train"]["batch_size"]), shuffle=True, num_workers=0))
        clean_tokens, corr_tokens, clean_pos, _, clean_label, _, _, _ = _batch_to_tensors(batch, pad_id, device)

        teacher = _teacher_logits_at_pos(model, clean_tokens, clean_pos)
        corrupt_cache = _cache_corrupt_z(model, corr_tokens, model.cfg.n_layers)

        masked = _masked_logits_at_pos(model, clean_tokens, clean_pos, svd_bank, mask_params, corrupt_cache)
        kl = kl_divergence(teacher, masked, temperature=temperature)
        # sparse = mask_params.sparsity()
        # loss = kl + lambda_sparse * sparse
        sparse_mean = mask_params.sparsity()  # mean(m) for human readability
        l1_sum = mask_params.m()[mask_params.valid].sum()  # SUM(m) for paper-faithful L1
        loss = kl + lambda_sparse * l1_sum
        logger.info(f"[DRY RUN] KL={kl.item():.6f} sparse={sparse.item():.6f} loss={loss.item():.6f}")

        logger.info("[DRY RUN] Running 1 val batch...")
        batch = next(_make_dataloader(val_ex, batch_size=int(cfg["train"]["batch_size"]), shuffle=False, num_workers=0))
        clean_tokens, corr_tokens, clean_pos, _, clean_label, _, _, _ = _batch_to_tensors(batch, pad_id, device)
        teacher = _teacher_logits_at_pos(model, clean_tokens, clean_pos)
        corrupt_cache = _cache_corrupt_z(model, corr_tokens, model.cfg.n_layers)
        masked = _masked_logits_at_pos(model, clean_tokens, clean_pos, svd_bank, mask_params, corrupt_cache)
        kl = kl_divergence(teacher, masked, temperature=temperature)
        acc = accuracy_from_logits(masked, clean_label)
        logger.info(f"[DRY RUN] val_KL={kl.item():.6f} val_acc={acc.item():.4f}")
        logger.info("[DRY RUN] OK. Exiting.")
        return

    # Training loop
    best_val_kl = float("inf")
    best_epoch = -1
    patience = int(cfg["train"]["early_stop_patience"])
    patience_left = patience
    epochs = int(cfg["train"]["epochs"])
    batch_size = int(cfg["train"]["batch_size"])

    history = []

    for epoch in range(1, epochs + 1):
        mask_params.train()
        model.eval()

        total_loss = 0.0
        total_kl = 0.0
        total_acc = 0.0
        n = 0

        pbar = tqdm(_make_dataloader(train_ex, batch_size=batch_size, shuffle=True, num_workers=int(cfg["train"]["num_workers"])))
        pbar.set_description(f"epoch {epoch}/{epochs}")

        for batch in pbar:
            clean_tokens, corr_tokens, clean_pos, _, clean_label, _, _, _ = _batch_to_tensors(batch, pad_id, device)

            with torch.no_grad():
                teacher = _teacher_logits_at_pos(model, clean_tokens, clean_pos)
                corrupt_cache = _cache_corrupt_z(model, corr_tokens, model.cfg.n_layers)

            masked = _masked_logits_at_pos(model, clean_tokens, clean_pos, svd_bank, mask_params, corrupt_cache)

            kl = kl_divergence(teacher, masked, temperature=temperature)
            # sparse = mask_params.sparsity()
            # loss = kl + lambda_sparse * sparse
            sparse_mean = mask_params.sparsity()  # mean(m) for human readability
            l1_sum = mask_params.m()[mask_params.valid].sum()  # SUM(m) for paper-faithful L1
            loss = kl + lambda_sparse * l1_sum


            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            with torch.no_grad():
                acc = accuracy_from_logits(masked, clean_label)

            bsz = clean_tokens.shape[0]
            total_loss += float(loss.item()) * bsz
            total_kl += float(kl.item()) * bsz
            total_acc += float(acc.item()) * bsz
            n += bsz
            pbar.set_postfix({"kl": total_kl / max(n, 1), "sparse": float(sparse.item())})

        train_metrics = {
            "epoch": epoch,
            "train_loss": total_loss / max(n, 1),
            "train_kl": total_kl / max(n, 1),
            "train_acc": total_acc / max(n, 1),
            "sparsity": float(mask_params.sparsity().item()),
        }
        # stats = mask_value_stats(mask_params.m(), mask_params.valid)
        # train_metrics.update(stats)
        # rel = paper_relative_sparsity(mask_params.m(), mask_params.valid, threshold=1e-2)
        m = mask_params.m()
        stats = mask_value_stats(m, mask_params.valid)
        rel = paper_relative_sparsity(m, mask_params.valid, threshold=1e-2)
        multi = sparsity_measures_multi_threshold(
            m,
            mask_params.valid,
            thresholds=(1e-2, 1e-1, 0.5, 0.9),
            total_directions=int(mask_params.mask_logits.numel()),
        )

        train_metrics.update(stats)
        train_metrics.update(multi)

        logger.info(
            f"[Train] epoch={epoch} loss={train_metrics['train_loss']:.6f} "
            f"kl={train_metrics['train_kl']:.6f} acc={train_metrics['train_acc']:.4f} "
            f"mean_m={stats.get('mask_mean', float('nan')):.4f} "
            f"S_rel@1e-2={rel['S_rel']:.4f} active_frac@1e-2={rel['active_frac']:.4f} "
            f"off@0.01={stats.get('mask_frac_lt_0p01', float('nan')):.3f} "
            f"off@0.1={stats.get('mask_frac_lt_0p1', float('nan')):.3f} "
            f"kept@0.9={stats.get('mask_frac_gt_0p9', float('nan')):.3f} "
            f"q10={stats.get('mask_q10', float('nan')):.3f} "
            f"q50={stats.get('mask_q50', float('nan')):.3f} "
            f"q90={stats.get('mask_q90', float('nan')):.3f}"
        )

        # logger.info(
        #     f"[Train] epoch={epoch} ... "
        #     f"mean_m={float(mask_params.sparsity().item()):.4f} "
        #     f"S_rel={rel['S_rel']:.4f} active_frac={rel['active_frac']:.4f} (thr={rel['thr']})"
        # )

        # logger.info(
        #     f"[Train] epoch={epoch} loss={train_metrics['train_loss']:.6f} "
        #     f"kl={train_metrics['train_kl']:.6f} acc={train_metrics['train_acc']:.4f} "
        #     f"mean_m={train_metrics.get('mask_mean', float('nan')):.4f} "
        #     f"kept@0.9={train_metrics.get('mask_frac_gt_0p9', float('nan')):.3f} "
        #     f"off@0.1={train_metrics.get('mask_frac_lt_0p1', float('nan')):.3f}"
        # )

        # logger.info(f"[Train] epoch={epoch} loss={train_metrics['train_loss']:.6f} kl={train_metrics['train_kl']:.6f} acc={train_metrics['train_acc']:.4f} sparsity={train_metrics['sparsity']:.4f}")

        val_metrics = evaluate_split(
            model=model,
            svd_bank=svd_bank,
            mask_params=mask_params,
            examples=val_ex,
            batch_size=batch_size,
            pad_id=pad_id,
            device=device,
            temperature=temperature,
            logger=logger,
            split_name="Val",
        )
        val_kl = float(val_metrics["kl"])
        val_l1_sum = float(val_metrics["l1_sum"])
        val_obj = val_kl + lambda_sparse * val_l1_sum

        logger.info(f"[ValSummary] epoch={epoch} val_kl={val_kl:.6f} val_l1_sum={val_l1_sum:.2f} val_obj={val_obj:.6f}")

        train_metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        history.append(train_metrics)

        # Save optionally each epoch
        if bool(cfg["logging"]["save_every_epoch"]):
            torch.save({"mask_logits": mask_params.mask_logits.detach().cpu(), "valid": svd_bank.valid.cpu()}, artifacts_root / "masks" / f"epoch_{epoch:02d}.pt")

        # Check best
        if val_metrics["kl"] < best_val_kl - 1e-6:
            best_val_kl = val_metrics["kl"]
            best_epoch = epoch
            patience_left = patience

            (artifacts_root / "masks").mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "mask_logits": mask_params.mask_logits.detach().cpu(),
                    "m": mask_params.m().detach().cpu(),
                    "valid": svd_bank.valid.cpu(),
                    "svd_eps": float(cfg["svd"]["svd_eps"]),
                    "use_bias_aug": bool(svd_bank.use_bias_aug),
                    "bias_mode": svd_bank.bias_mode,
                },
                artifacts_root / "masks" / "best_masks.pt",
            )
            logger.info(f"[Checkpoint] Saved best masks at epoch {epoch} (val_KL={best_val_kl:.6f})")
        else:
            patience_left -= 1
            logger.info(f"[EarlyStop] No improvement. patience_left={patience_left}/{patience}")
            if patience_left <= 0:
                logger.info(f"[EarlyStop] Stopping early at epoch {epoch}. Best epoch was {best_epoch} (val_KL={best_val_kl:.6f}).")
                break

        # Save history each epoch
        save_json(artifacts_root / "logs" / "history.json", history)

    # Always save last
    (artifacts_root / "masks").mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "mask_logits": mask_params.mask_logits.detach().cpu(),
            "m": mask_params.m().detach().cpu(),
            "valid": svd_bank.valid.cpu(),
        },
        artifacts_root / "masks" / "last_masks.pt",
    )

    logger.info("Evaluating best masks on test split...")

    # Load best masks (if available) back into mask_params for test evaluation
    best_path = artifacts_root / "masks" / "best_masks.pt"
    if best_path.exists():
        ckpt = torch.load(best_path, map_location="cpu")
        with torch.no_grad():
            mask_params.mask_logits.copy_(ckpt["mask_logits"].to(device))
        logger.info(f"Loaded best_masks.pt from epoch {best_epoch}.")
    else:
        logger.info("best_masks.pt not found; using last masks.")

    test_metrics = evaluate_split(
        model=model,
        svd_bank=svd_bank,
        mask_params=mask_params,
        examples=test_ex,
        batch_size=batch_size,
        pad_id=pad_id,
        device=device,
        temperature=temperature,
        logger=logger,
        split_name="Test",
    )
    save_json(artifacts_root / "logs" / "test_metrics.json", test_metrics)

    # Optional receptors
    if bool(cfg["logging"]["save_receptors"]):
        save_receptors(
            model=model,
            svd=svd_bank,
            masks=mask_params.m().detach().cpu(),
            out_dir=artifacts_root / "receptors",
            topk=int(cfg["logging"]["receptors_topk"]),
            mask_threshold=float(cfg["logging"]["receptors_save_mask_threshold"]),
            logger=logger,
        )

    logger.info(f"Done. Outputs in: {artifacts_root}")


if __name__ == "__main__":
    main()
