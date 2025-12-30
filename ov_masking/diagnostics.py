from __future__ import annotations

from typing import Dict, List

import torch

from .teacher_cache import tokenize_prompts
from .metrics import kl_teacher_student


@torch.no_grad()
def run_identity_check(
    *,
    teacher,
    runner,
    mask_params,
    tokenizer,
    prompts: List[str],
    device: torch.device,
    mask_logit: float,
    max_abs_tol: float,
    kl_tol: float,
) -> Dict[str, object]:
    """
    Identity check: force mask ≈ 1 (theta = mask_logit, sigmoid(theta) ~ 1)
    and verify student logits match teacher logits on the SAME prompts, without using cached teacher logprobs.

    We compare last-token logits at the true last (unpadded) position using last_pos from tokenize_prompts.
    """

    # Backup current theta
    theta_backup = mask_params.theta.detach().clone()

    try:
        # Force mask ~ 1
        mask_params.theta.fill_(float(mask_logit))

        # Tokenize
        tokens, attn_mask, last_pos = tokenize_prompts(tokenizer, prompts, device=device)

        # Forward teacher
        teacher.eval()
        logits_t = teacher(tokens, attention_mask=attn_mask)  # [B, T, V]

        # Forward student (masked runner)
        runner.model.eval()
        logits_s = runner(tokens, attn_mask)  # [B, T, V]

        bsz = tokens.size(0)
        idx = torch.arange(bsz, device=device)

        logits_t_last = logits_t[idx, last_pos].float()  # [B, V]
        logits_s_last = logits_s[idx, last_pos].float()  # [B, V]

        # Compare logits
        diff = logits_t_last - logits_s_last
        max_abs_diff = float(diff.abs().max().item())
        mean_abs_diff = float(diff.abs().mean().item())

        # Compare distributions via KL(teacher || student)
        logp_t = torch.log_softmax(logits_t_last, dim=-1)
        logp_s = torch.log_softmax(logits_s_last, dim=-1)
        kl = float(kl_teacher_student(logp_s, logp_t).item())

        passed = (max_abs_diff <= float(max_abs_tol)) and (kl <= float(kl_tol))

        return {
            "passed": passed,
            "n_prompts": len(prompts),
            "mask_logit": float(mask_logit),
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": mean_abs_diff,
            "kl_teacher_student": kl,
            "thresholds": {"max_abs_tol": float(max_abs_tol), "kl_tol": float(kl_tol)},
        }

    finally:
        # Restore theta
        mask_params.theta.copy_(theta_backup)
