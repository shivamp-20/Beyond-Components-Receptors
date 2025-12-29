from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn.functional as F


@dataclass
class BatchMetrics:
    kl_clean: float
    kl_corr: float
    kl_total: float
    acc_clean: float
    acc_corr: float
    acc_total: float
    logitdiff_clean: float
    logitdiff_corr: float
    logitdiff_total: float


def kl_teacher_student(logp_student: torch.Tensor, logp_teacher: torch.Tensor) -> torch.Tensor:
    """
    Spec KL:
      mean_b sum_v exp(logp_teacher) * (logp_teacher - logp_student)
    PyTorch: F.kl_div(input=logp_student, target=logp_teacher, log_target=True)
    with reduction='batchmean' gives sum over vocab, mean over batch.
    """
    return F.kl_div(logp_student, logp_teacher, reduction="batchmean", log_target=True)


def accuracy_from_logits(logits_last: torch.Tensor, label_ids: torch.Tensor) -> torch.Tensor:
    """
    logits_last: [B,V], label_ids: [B]
    """
    pred = torch.argmax(logits_last, dim=-1)
    return (pred == label_ids).float().mean()


def logit_diff(logits_last: torch.Tensor, label_ids: torch.Tensor, wrong_ids: torch.Tensor) -> torch.Tensor:
    """
    mean_b (logit[label] - logit[wrong])
    """
    bsz = logits_last.size(0)
    label_logits = logits_last[torch.arange(bsz, device=logits_last.device), label_ids]
    wrong_logits = logits_last[torch.arange(bsz, device=logits_last.device), wrong_ids]
    return (label_logits - wrong_logits).mean()
