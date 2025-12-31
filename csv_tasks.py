"""CSV task loaders for your 3 datasets (GP / IOI / GT).

This file implements the **full completion tokenization trick** you specified:

Given prompt and label string:
    full = prompt.strip() + " " + label.strip()
    tokens_full = tokenize(full)
    label_token_id = tokens_full[-1]
    prompt_ids = tokens_full[:-1]

This avoids the common GPT-2 BPE "leading space" bug.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class Example:
    """A single training example with already-tokenized prompt ids."""
    clean_ids: List[int]
    corrupt_ids: List[int]
    clean_label_id: int
    corrupt_label_id: int
    # Optional: only used for extra reporting on IOI
    clean_wrong_label_id: Optional[int] = None
    corrupt_wrong_label_id: Optional[int] = None


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read a CSV that might be comma-separated or tab-separated.

    Your sample rows look tab-delimited, but we'll detect robustly.
    """
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        # likely a tab-separated file that got read as a single column
        df = pd.read_csv(path, sep="\t")
    return df


def _tokenize_full_completion(tokenizer, text: str, prepend_bos: bool) -> List[int]:
    """Tokenize a full completion string into token ids.

    We do NOT use `add_special_tokens` because GPT-2 doesn't use them,
    but we optionally prepend a BOS token (TransformerLens often does).
    """
    text = text.strip()
    ids = tokenizer.encode(text, add_special_tokens=False)
    if prepend_bos:
        bos = tokenizer.bos_token_id
        if bos is None:
            # GPT-2 usually has no BOS; TransformerLens often uses EOS as BOS.
            bos = tokenizer.eos_token_id
        ids = [bos] + ids
    return ids


def _make_prompt_and_label_ids(tokenizer, prompt: str, label: str, prepend_bos: bool) -> Tuple[List[int], int]:
    """Return (prompt_ids, label_token_id) using the full completion trick."""
    full = prompt.strip() + " " + str(label).strip()
    full_ids = _tokenize_full_completion(tokenizer, full, prepend_bos=prepend_bos)
    if len(full_ids) < 2:
        raise ValueError(f"Too-short tokenization for full completion: {full!r}")
    prompt_ids = full_ids[:-1]
    label_id = full_ids[-1]
    return prompt_ids, label_id


def load_task_split(
    *,
    kind: str,
    csv_path: Path,
    tokenizer,
    prepend_bos: bool,
    max_examples: Optional[int] = None,
) -> List[Example]:
    """Load one split (train/val/test) for a given task kind.

    kind ∈ {"gp", "ioi", "gt"}
    """
    df = _read_csv_robust(csv_path)

    examples: List[Example] = []
    for _, row in df.iterrows():
        if kind == "gp":
            clean_prompt = str(row["prefix"])
            corrupt_prompt = str(row["corr_prefix"])
            clean_label = str(row["pronoun"])
            corrupt_label = str(row["corr_pronoun"])

            clean_ids, clean_label_id = _make_prompt_and_label_ids(tokenizer, clean_prompt, clean_label, prepend_bos)
            corrupt_ids, corrupt_label_id = _make_prompt_and_label_ids(tokenizer, corrupt_prompt, corrupt_label, prepend_bos)

            ex = Example(
                clean_ids=clean_ids,
                corrupt_ids=corrupt_ids,
                clean_label_id=clean_label_id,
                corrupt_label_id=corrupt_label_id,
            )

        elif kind == "ioi":
            clean_prompt = str(row["ioi_sentences_input"])
            corrupt_prompt = str(row["corr_ioi_sentences_input"])
            clean_label = str(row["ioi_sentences_labels"])
            corrupt_label = str(row["corr_ioi_sentences_labels"])

            clean_ids, clean_label_id = _make_prompt_and_label_ids(tokenizer, clean_prompt, clean_label, prepend_bos)
            corrupt_ids, corrupt_label_id = _make_prompt_and_label_ids(tokenizer, corrupt_prompt, corrupt_label, prepend_bos)

            # Optional wrong labels for reporting
            clean_wrong_id = None
            corrupt_wrong_id = None
            if "ioi_sentences_labels_wrong" in row and pd.notna(row["ioi_sentences_labels_wrong"]):
                _, clean_wrong_id = _make_prompt_and_label_ids(tokenizer, clean_prompt, str(row["ioi_sentences_labels_wrong"]), prepend_bos)
            if "corr_ioi_sentences_labels_wrong" in row and pd.notna(row["corr_ioi_sentences_labels_wrong"]):
                _, corrupt_wrong_id = _make_prompt_and_label_ids(tokenizer, corrupt_prompt, str(row["corr_ioi_sentences_labels_wrong"]), prepend_bos)

            ex = Example(
                clean_ids=clean_ids,
                corrupt_ids=corrupt_ids,
                clean_label_id=clean_label_id,
                corrupt_label_id=corrupt_label_id,
                clean_wrong_label_id=clean_wrong_id,
                corrupt_wrong_label_id=corrupt_wrong_id,
            )

        elif kind == "gt":
            # digits may be numeric; convert to string
            clean_prompt = str(row["prefix"])
            corrupt_prompt = str(row["corr_prefix"])
            clean_label = str(row["digits"])
            corrupt_label = str(row["corr_digits"])

            clean_ids, clean_label_id = _make_prompt_and_label_ids(tokenizer, clean_prompt, clean_label, prepend_bos)
            corrupt_ids, corrupt_label_id = _make_prompt_and_label_ids(tokenizer, corrupt_prompt, corrupt_label, prepend_bos)

            ex = Example(
                clean_ids=clean_ids,
                corrupt_ids=corrupt_ids,
                clean_label_id=clean_label_id,
                corrupt_label_id=corrupt_label_id,
            )

        else:
            raise ValueError(f"Unknown task kind: {kind}")

        examples.append(ex)
        if max_examples is not None and len(examples) >= max_examples:
            break

    return examples
