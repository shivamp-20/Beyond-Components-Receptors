import json
import logging
import os
import random
import subprocess
from dataclasses import asdict
from datetime import datetime
from typing import Any, Dict, Optional

import numpy as np
import torch


def str2bool(x: str) -> bool:
    if isinstance(x, bool):
        return x
    x = x.lower().strip()
    if x in {"1", "true", "t", "yes", "y"}:
        return True
    if x in {"0", "false", "f", "no", "n"}:
        return False
    raise ValueError(f"Cannot parse bool from: {x}")


def normalize(text: str) -> str:
    """
    Must match spec exactly:
      text = text.strip()
      text = text + " "
      return text
    """
    text = text.strip()
    text = text + " "
    return text


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_git_commit_hash() -> Optional[str]:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode().strip()
        return out
    except Exception:
        return None


def make_run_dir(task_name: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join("runs", task_name, f"ov_mask_{ts}")
    os.makedirs(run_dir, exist_ok=False)  # never overwrite
    return run_dir


def setup_logger(run_dir: str) -> logging.Logger:
    """
    Logs to:
      - stdout
      - runs/.../train.log
    """
    logger = logging.getLogger("ovmask")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("[%(asctime)s] %(levelname)s: %(message)s")

    # file
    fh = logging.FileHandler(os.path.join(run_dir, "train.log"), mode="w", encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # console
    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    return logger


def save_json(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_json_list(path: str, item: Dict[str, Any]) -> None:
    """
    Keeps metrics.json as a list[dict]. Overwrites file each time (simple & robust).
    """
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = []
    data.append(item)
    save_json(path, data)


def dtype_from_str(dtype_str: str) -> torch.dtype:
    if dtype_str == "float32":
        return torch.float32
    if dtype_str == "float16":
        return torch.float16
    if dtype_str == "bfloat16":
        return torch.bfloat16
    raise ValueError(dtype_str)


def ensure_pad_token(tokenizer) -> None:
    """
    GPT-2 tokenizer often has no pad token. We set pad_token = eos_token for batching.
    This is a standard approach in GPT-2 workflows.
    """
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token


def tokenize_prompts(tokenizer, prompts, device: torch.device):
    """
    Returns:
      input_ids: [batch, seq]
      attention_mask: [batch, seq] (1 for real tokens, 0 for padding)
    """
    ensure_pad_token(tokenizer)
    enc = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    return enc["input_ids"].to(device), enc["attention_mask"].to(device)


def last_token_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    """
    For each row, returns index of final real token (length-1).
    attention_mask: [batch, seq] with 1s then 0s (right padded).
    """
    lengths = attention_mask.sum(dim=1)  # [batch]
    return (lengths - 1).long()


def select_last_logits(logits: torch.Tensor, last_idx: torch.Tensor) -> torch.Tensor:
    """
    logits: [batch, seq, vocab]
    last_idx: [batch]
    returns logits_last: [batch, vocab]
    """
    b = logits.shape[0]
    return logits[torch.arange(b, device=logits.device), last_idx, :]


def label_to_token_id(tokenizer, label_str: str) -> int:
    """
    Spec: encode " "+label_str and take FIRST token id.
    """
    toks = tokenizer.encode(" " + label_str, add_special_tokens=False)
    if len(toks) == 0:
        raise ValueError(f"Label produced no tokens: {label_str!r}")
    return int(toks[0])
