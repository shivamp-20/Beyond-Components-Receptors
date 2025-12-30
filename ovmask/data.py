import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from ovmask.utils import normalize, label_to_token_id


@dataclass
class Example:
    idx: int
    prompt_clean: str
    prompt_corr: str
    label_clean_str: str
    label_corr_str: str
    wrong_clean_str: Optional[str] = None
    wrong_corr_str: Optional[str] = None

    # token ids (computed once after tokenizer known)
    label_clean_id: Optional[int] = None
    label_corr_id: Optional[int] = None
    wrong_clean_id: Optional[int] = None
    wrong_corr_id: Optional[int] = None


def load_task_csv(task: str, csv_path: str) -> List[Example]:
    """
    Builds examples exactly per the spec (columns differ per task).
    """
    df = pd.read_csv(csv_path)
    exs: List[Example] = []

    if task == "gp":
        required = ["prefix", "pronoun", "template", "name", "corr_prefix", "corr_pronoun", "corr_template", "corr_name"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in {csv_path}")

        for i, row in df.iterrows():
            prompt_clean = normalize(str(row["prefix"]))
            prompt_corr = normalize(str(row["corr_prefix"]))
            label_clean = str(row["pronoun"])
            label_corr = str(row["corr_pronoun"])
            exs.append(
                Example(
                    idx=i,
                    prompt_clean=prompt_clean,
                    prompt_corr=prompt_corr,
                    label_clean_str=label_clean,
                    label_corr_str=label_corr,
                    # for reporting (not KL): GP can use corr_pronoun as wrong label for clean
                    wrong_clean_str=label_corr,
                    wrong_corr_str=label_clean,
                )
            )

    elif task == "ioi":
        required = [
            "ioi_sentences_input",
            "ioi_sentences_labels",
            "corr_ioi_sentences_input",
            "corr_ioi_sentences_labels",
            "ioi_sentences_labels_wrong",
            "corr_ioi_sentences_labels_wrong",
        ]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in {csv_path}")

        for i, row in df.iterrows():
            prompt_clean = normalize(str(row["ioi_sentences_input"]))
            prompt_corr = normalize(str(row["corr_ioi_sentences_input"]))
            label_clean = str(row["ioi_sentences_labels"])
            label_corr = str(row["corr_ioi_sentences_labels"])
            wrong_clean = str(row["ioi_sentences_labels_wrong"])
            wrong_corr = str(row["corr_ioi_sentences_labels_wrong"])
            exs.append(
                Example(
                    idx=i,
                    prompt_clean=prompt_clean,
                    prompt_corr=prompt_corr,
                    label_clean_str=label_clean,
                    label_corr_str=label_corr,
                    wrong_clean_str=wrong_clean,
                    wrong_corr_str=wrong_corr,
                )
            )

    elif task == "gt":
        required = ["template", "century", "noun", "digits", "prefix", "corr_template", "corr_century", "corr_noun", "corr_digits", "corr_prefix"]
        for c in required:
            if c not in df.columns:
                raise ValueError(f"Missing column {c} in {csv_path}")

        for i, row in df.iterrows():
            prompt_clean = normalize(str(row["prefix"]))
            prompt_corr = normalize(str(row["corr_prefix"]))
            label_clean = str(row["digits"])
            label_corr = str(row["corr_digits"])
            exs.append(
                Example(
                    idx=i,
                    prompt_clean=prompt_clean,
                    prompt_corr=prompt_corr,
                    label_clean_str=label_clean,
                    label_corr_str=label_corr,
                    # for reporting (not KL): GT can use corr_digits as wrong for clean
                    wrong_clean_str=label_corr,
                    wrong_corr_str=label_clean,
                )
            )
    else:
        raise ValueError(f"Unknown task: {task}")

    return exs


def attach_label_token_ids(exs: List[Example], tokenizer) -> None:
    """
    Mutates Example objects to include token ids for clean/corr label + wrong labels (if available).
    """
    for ex in exs:
        ex.label_clean_id = label_to_token_id(tokenizer, ex.label_clean_str)
        ex.label_corr_id = label_to_token_id(tokenizer, ex.label_corr_str)
        if ex.wrong_clean_str is not None:
            ex.wrong_clean_id = label_to_token_id(tokenizer, ex.wrong_clean_str)
        if ex.wrong_corr_str is not None:
            ex.wrong_corr_id = label_to_token_id(tokenizer, ex.wrong_corr_str)


class SimpleDataset:
    """
    Minimal dataset wrapper: returns dict with strings + ids + index.
    """
    def __init__(self, examples: List[Example]):
        self.examples = examples

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int) -> Dict:
        ex = self.examples[i]
        return {
            "idx": ex.idx,
            "prompt_clean": ex.prompt_clean,
            "prompt_corr": ex.prompt_corr,
            "label_clean_id": ex.label_clean_id,
            "label_corr_id": ex.label_corr_id,
            "wrong_clean_id": ex.wrong_clean_id,
            "wrong_corr_id": ex.wrong_corr_id,
        }


def collate_fn(batch: List[Dict]) -> Dict:
    """
    Keeps it simple: just return lists; tokenization happens later.
    """
    return {
        "idx": [b["idx"] for b in batch],
        "prompt_clean": [b["prompt_clean"] for b in batch],
        "prompt_corr": [b["prompt_corr"] for b in batch],
        "label_clean_id": [b["label_clean_id"] for b in batch],
        "label_corr_id": [b["label_corr_id"] for b in batch],
        "wrong_clean_id": [b["wrong_clean_id"] for b in batch],
        "wrong_corr_id": [b["wrong_corr_id"] for b in batch],
    }


def resolve_csv(data_dir: str, filename: str) -> str:
    path = os.path.join(data_dir, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(f"CSV not found: {path}")
    return path
