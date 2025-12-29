from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from torch.utils.data import Dataset

from .utils import normalize, read_table_auto_sep


@dataclass(frozen=True)
class Example:
    idx: int
    prompt_clean: str
    prompt_corr: str
    label_clean_str: str
    label_corr_str: str
    wrong_clean_str: Optional[str]
    wrong_corr_str: Optional[str]


EXPECTED_COLUMNS = {
    "gp": [
        "prefix", "pronoun", "template", "name",
        "corr_prefix", "corr_pronoun", "corr_template", "corr_name"
    ],
    "ioi": [
        "ioi_sentences_input", "ioi_sentences_labels",
        "corr_ioi_sentences_input", "corr_ioi_sentences_labels",
        "ioi_sentences_labels_wrong", "corr_ioi_sentences_labels_wrong"
    ],
    "gt": [
        "template", "century", "noun", "digits", "prefix",
        "corr_template", "corr_century", "corr_noun", "corr_digits", "corr_prefix"
    ],
}


def _check_columns(task: str, df: pd.DataFrame) -> None:
    expected = EXPECTED_COLUMNS[task]
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{task}] Missing columns: {missing}\n"
            f"Found columns: {list(df.columns)}"
        )


def load_task_examples(task: str, csv_path: Path) -> List[Example]:
    """
    Convert a task CSV into a list of normalized prompt/label examples.

    IMPORTANT:
    - Normalization is applied ONLY to prompts, not labels.
    - wrong_* strings are included for metrics only.
    """
    df = read_table_auto_sep(csv_path)
    _check_columns(task, df)

    examples: List[Example] = []
    for i, row in enumerate(df.itertuples(index=False)):
        r = row._asdict()

        if task == "gp":
            prompt_clean = normalize(r["prefix"])
            prompt_corr = normalize(r["corr_prefix"])
            label_clean = str(r["pronoun"])
            label_corr = str(r["corr_pronoun"])
            # Spec: GP can use corr_pronoun as "wrong" (for clean), and vice versa.
            wrong_clean = label_corr
            wrong_corr = label_clean

        elif task == "ioi":
            prompt_clean = normalize(r["ioi_sentences_input"])
            prompt_corr = normalize(r["corr_ioi_sentences_input"])
            label_clean = str(r["ioi_sentences_labels"])
            label_corr = str(r["corr_ioi_sentences_labels"])
            wrong_clean = str(r["ioi_sentences_labels_wrong"])
            wrong_corr = str(r["corr_ioi_sentences_labels_wrong"])

        elif task == "gt":
            prompt_clean = normalize(r["prefix"])
            prompt_corr = normalize(r["corr_prefix"])
            label_clean = str(r["digits"])
            label_corr = str(r["corr_digits"])
            # Spec: GT can use corr_digits as "wrong" (for clean), and vice versa.
            wrong_clean = label_corr
            wrong_corr = label_clean

        else:
            raise ValueError(f"Unknown task: {task}")

        examples.append(
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

    return examples


def label_str_to_first_token_id(tokenizer, label_str: str) -> Tuple[int, int]:
    """
    Spec: "encode ' '+label_str and take the first token id"
    Returns: (token_id, num_tokens_from_encoding)
    """
    text = " " + str(label_str)
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) < 1:
        raise ValueError(f"Tokenizer produced 0 tokens for label: {label_str!r}")
    return ids[0], len(ids)


class PromptDataset(Dataset):
    """
    Dataset returns raw strings + precomputed label/wrong token IDs.
    We keep idx to align with teacher logp tensors.
    """

    def __init__(self, examples: List[Example], tokenizer, task: str):
        self.examples = examples
        self.task = task

        # Precompute label token IDs (first token of " "+label_str)
        self.label_clean_id: List[int] = []
        self.label_corr_id: List[int] = []
        self.wrong_clean_id: List[int] = []
        self.wrong_corr_id: List[int] = []

        multi_token_warnings: Dict[str, int] = {"clean_label": 0, "corr_label": 0, "clean_wrong": 0, "corr_wrong": 0}

        for ex in examples:
            lid, ln = label_str_to_first_token_id(tokenizer, ex.label_clean_str)
            cid, cn = label_str_to_first_token_id(tokenizer, ex.label_corr_str)

            wid_clean, wn_clean = label_str_to_first_token_id(tokenizer, ex.wrong_clean_str) if ex.wrong_clean_str is not None else (cid, 1)
            wid_corr, wn_corr = label_str_to_first_token_id(tokenizer, ex.wrong_corr_str) if ex.wrong_corr_str is not None else (lid, 1)

            if ln != 1:
                multi_token_warnings["clean_label"] += 1
            if cn != 1:
                multi_token_warnings["corr_label"] += 1
            if wn_clean != 1:
                multi_token_warnings["clean_wrong"] += 1
            if wn_corr != 1:
                multi_token_warnings["corr_wrong"] += 1

            self.label_clean_id.append(lid)
            self.label_corr_id.append(cid)
            self.wrong_clean_id.append(wid_clean)
            self.wrong_corr_id.append(wid_corr)

        self.multi_token_warnings = multi_token_warnings

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, i: int):
        ex = self.examples[i]
        return (
            ex.idx,
            ex.prompt_clean,
            ex.prompt_corr,
            self.label_clean_id[i],
            self.label_corr_id[i],
            self.wrong_clean_id[i],
            self.wrong_corr_id[i],
        )
