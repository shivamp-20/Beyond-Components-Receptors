from __future__ import annotations

import hashlib
import json
import logging
import os
import random
import subprocess
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch


def timestamp() -> str:
    """UTC-ish timestamp string safe for filenames."""
    return time.strftime("%Y%m%d_%H%M%S", time.gmtime())


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Determinism tradeoffs; keep default performance unless user wants strict determinism.


def get_git_commit_hash(repo_root: Path) -> Optional[str]:
    """Return current git commit hash, or None if not available."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_root),
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        return out
    except Exception:
        return None


def setup_logger(log_file: Path) -> logging.Logger:
    """
    Logger that prints to stdout and also writes to a file.
    Avoids super-verbose logs; use logger.info for epoch-level summaries.
    """
    logger = logging.getLogger("ov_masking")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    logger.propagate = False

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    # Console
    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)

    # File
    fh = logging.FileHandler(str(log_file), mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(fmt)

    logger.addHandler(sh)
    logger.addHandler(fh)
    return logger


def json_dump(obj: Any, path: Path) -> None:
    """Dump JSON with stable formatting."""
    if is_dataclass(obj):
        obj = asdict(obj)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def json_load(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def read_table_auto_sep(path: Path) -> pd.DataFrame:
    """
    Read a file that is usually CSV, but your examples look TSV-like.
    We try comma first; if it yields 1 column, retry with tab.
    """
    df = pd.read_csv(path)
    if df.shape[1] <= 1:
        df = pd.read_csv(path, sep="\t")
    return df


def file_md5(path: Path) -> str:
    """MD5 for dataset identity checks."""
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def get_device(device_str: str) -> torch.device:
    if device_str == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_str)


def parse_dtype(dtype_str: str) -> torch.dtype:
    s = dtype_str.lower()
    if s in {"fp16", "float16"}:
        return torch.float16
    if s in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if s in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unknown dtype: {dtype_str}. Use float16|bfloat16|float32.")


def normalize(text: str) -> str:
    """
    MUST match spec exactly:
      text = text.strip()
      text = text + " "
    """
    text = str(text).strip()
    text = text + " "
    return text
