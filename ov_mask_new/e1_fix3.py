"""Fix3: Logit-footprint clustering + task-weighted mass.

This script is intentionally self-contained and path-robust for the existing
`run_ov_mask.py` output layout:

  <out_dir>/cache/{train,val,test}.pt
  <out_dir>/svd/layer{l:02d}_head{h:02d}.pt
  <out_dir>/masks/layer{l:02d}_head{h:02d}.pt

It will:
  0) Ensure nu_clean_at_target exists in cache files (computed from tokens_clean).
  1) Build active direction set D = {(l,h,k): m[k] >= tau_active}.
  2) Compute per-direction logit footprints via (sigma * v)^T @ W_U.
  3) Compute per-direction task relevance w_d = m * sigma * |corr(a_clean, margin)|.
  4) Cluster directions by weighted-Jaccard overlap of token footprints (pos/neg).
     Threshold is selected by a tiny grid search to maximize top-K mass.
  5) Save a new receptor bank + state tensors X_* for train/val/test.

Run (after you ran run_ov_mask.py for the same task):

  python e1_fix3.py --task gp --out_dir outputs/gp --data_dir data_main

Artifacts:
  <out_dir>/artifacts/e1_fix3/
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn
from tqdm.auto import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import GPT2LMHeadModel, GPT2TokenizerFast


# -------------------------
# Small utilities
# -------------------------


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _exists_any(p: Path, names: Sequence[str]) -> Optional[Path]:
    for n in names:
        cand = p / n
        if cand.exists() and cand.is_dir():
            return cand
    return None


def resolve_out_layout(out_dir: Path) -> Tuple[Path, Path, Path]:
    """Return (cache_dir, svd_dir, masks_dir) for an out_dir.

    We keep this robust because earlier scripts used inconsistent names.
    """
    if not out_dir.exists():
        raise FileNotFoundError(f"out_dir does not exist: {out_dir}")

    cache_dir = _exists_any(out_dir, ["cache", "caches"]) or (out_dir / "cache")
    svd_dir = _exists_any(out_dir, ["svd", "svd_dir"]) or (out_dir / "svd")
    masks_dir = _exists_any(out_dir, ["masks", "mask", "mask_dir"]) or (out_dir / "masks")

    if not cache_dir.exists():
        raise FileNotFoundError(f"Missing cache directory: {cache_dir}")
    if not svd_dir.exists():
        raise FileNotFoundError(f"Missing SVD directory: {svd_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Missing masks directory: {masks_dir}")

    return cache_dir, svd_dir, masks_dir


def find_csv(data_dir: Path, task: str, split: str) -> Path:
    """Try to find the right CSV for (task, split) in data_dir."""
    assert split in {"train", "val", "test"}
    pats = [
        f"{split}_*{task}*.csv",
        f"*{split}*{task}*.csv",
        f"*{task}*{split}*.csv",
    ]
    cands: List[Path] = []
    for pat in pats:
        cands.extend(sorted(data_dir.glob(pat)))
    # de-duplicate
    uniq = []
    seen = set()
    for p in cands:
        if p.name not in seen:
            uniq.append(p)
            seen.add(p.name)
    if not uniq:
        raise FileNotFoundError(
            f"Could not find a CSV for task={task} split={split} under {data_dir}. "
            f"Expected something like '{split}_*{task}*.csv'."
        )

    # Heuristic preferences:
    # - train often has '1k'
    # - otherwise shortest filename
    if split == "train":
        one_k = [p for p in uniq if "1k" in p.name.lower()]
        if one_k:
            uniq = one_k
    uniq = sorted(uniq, key=lambda p: (len(p.name), p.name))
    return uniq[0]


def safe_token_id(tok: GPT2TokenizerFast, s: str) -> Tuple[int, bool]:
    """Return a token id for label string s.

    If multi-token, returns the *last* token id and (is_multitoken=True).
    """
    # GPT-2 uses leading-space tokenization; we try both.
    ids = tok.encode(s, add_special_tokens=False)
    if len(ids) == 0:
        ids = tok.encode(" " + s, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"Could not tokenize label: {s!r}")
    return ids[-1], (len(ids) != 1)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y))
    if denom < 1e-12:
        return 0.0
    return float(np.dot(x, y) / denom)


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy()


def softmax_np(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x - x.max()
    e = np.exp(x)
    s = e.sum()
    if s <= 0:
        return np.ones_like(x) / max(1, x.size)
    return (e / s).astype(np.float32)


def weighted_jaccard(ids_a: np.ndarray, w_a: np.ndarray, ids_b: np.ndarray, w_b: np.ndarray) -> float:
    """Weighted Jaccard over token ids with nonnegative weights."""
    # small (L<=64), so dict is fine.
    da = {int(i): float(w) for i, w in zip(ids_a, w_a)}
    db = {int(i): float(w) for i, w in zip(ids_b, w_b)}
    keys = set(da.keys()) | set(db.keys())
    if not keys:
        return 0.0
    num = 0.0
    den = 0.0
    for k in keys:
        wa = da.get(k, 0.0)
        wb = db.get(k, 0.0)
        num += min(wa, wb)
        den += max(wa, wb)
    return float(num / (den + 1e-12))


# -------------------------
# Cache helpers
# -------------------------


def cache_path(cache_dir: Path, split: str) -> Path:
    """Resolve cache file name. Primary expected: cache/{split}.pt."""
    cand = cache_dir / f"{split}.pt"
    if cand.exists():
        return cand
    # fallbacks
    for alt in [cache_dir / f"cache_{split}.pt", cache_dir / f"cache_{split}.pth"]:
        if alt.exists():
            return alt
    raise FileNotFoundError(f"Missing cache file for split={split}. Looked for {cand} and fallbacks in {cache_dir}.")


class CProjInputCacher:
    """Capture attention c_proj input at target positions for all layers.

    We store per-layer/head value vectors z at target position.
    Output shape: [B, L, H, d_head]. We then augment to [B,L,H,65] by appending 1.
    """

    def __init__(self, model: GPT2LMHeadModel):
        self.model = model
        self.handles: List[torch.utils.hooks.RemovableHandle] = []
        self.target_pos: Optional[torch.Tensor] = None
        self._buf: List[torch.Tensor] = []
        self.n_layers = model.config.n_layer
        self.n_heads = model.config.n_head
        self.d_head = model.config.n_embd // model.config.n_head

    def _hook(self, layer_idx: int):
        def fn(_mod, inputs, _output):
            # inputs[0] is attn output before c_proj: [B, T, n_embd]
            x = inputs[0]
            assert self.target_pos is not None
            # gather at target positions
            B, T, D = x.shape
            tp = self.target_pos.to(x.device)
            assert tp.shape == (B,)
            x_t = x[torch.arange(B, device=x.device), tp]  # [B, D]
            x_t = x_t.view(B, self.n_heads, self.d_head)  # [B, H, d_head]
            self._buf[layer_idx] = x_t.detach()
        return fn

    def __enter__(self):
        self._buf = [torch.empty(0) for _ in range(self.n_layers)]
        for l in range(self.n_layers):
            h = self.model.transformer.h[l].attn.c_proj.register_forward_hook(self._hook(l))
            self.handles.append(h)
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles = []
        self.target_pos = None

    def run(self, tokens: torch.Tensor, target_pos: torch.Tensor, batch_size: int = 64) -> torch.Tensor:
        """Return nu_clean_at_target: [N, L, H, 65]"""
        dev = next(self.model.parameters()).device
        tokens = tokens.to(dev)
        target_pos = target_pos.to(dev)
        N, T = tokens.shape
        out = torch.empty((N, self.n_layers, self.n_heads, self.d_head + 1), dtype=torch.float32)

        self.model.eval()
        with torch.no_grad():
            for i0 in tqdm(range(0, N, batch_size), desc="[Fix3] nu_clean batches"):
                i1 = min(N, i0 + batch_size)
                tok_b = tokens[i0:i1]
                tp_b = target_pos[i0:i1]
                self.target_pos = tp_b
                # ensure buffer sized
                self._buf = [torch.empty((i1 - i0, self.n_heads, self.d_head), device=dev) for _ in range(self.n_layers)]
                _ = self.model(tok_b)
                z = torch.stack(self._buf, dim=1)  # [B, L, H, d_head]
                ones = torch.ones((z.shape[0], z.shape[1], z.shape[2], 1), device=z.device, dtype=z.dtype)
                nu = torch.cat([z, ones], dim=-1).float().cpu()  # [B, L, H, 65]
                out[i0:i1] = nu
        return out


def ensure_nu_clean(out_dir: Path, cache_dir: Path, model: GPT2LMHeadModel, batch_size: int = 64) -> None:
    """If any cache split lacks nu_clean_at_target, compute and write it back."""
    for split in ["train", "val", "test"]:
        p = cache_path(cache_dir, split)
        cache = torch.load(p, map_location="cpu")
        if "nu_clean_at_target" in cache:
            continue
        tokens_clean = cache["tokens_clean"]
        target_pos = cache["target_pos"]
        with CProjInputCacher(model) as cacher:
            nu_clean = cacher.run(tokens_clean, target_pos, batch_size=batch_size)
        cache["nu_clean_at_target"] = nu_clean
        torch.save(cache, p)
        print(f"[Fix3] added nu_clean_at_target -> {p}")


# -------------------------
# Direction extraction
# -------------------------


@dataclass
class Direction:
    l: int
    h: int
    k: int
    sigma: float
    mask: float
    flip: bool  # whether we flipped sign to make corr>=0
    corr: float
    mass: float
    v: np.ndarray  # [768], possibly flipped
    # footprints
    pos_ids: np.ndarray
    pos_w: np.ndarray
    pos_scores: np.ndarray
    neg_ids: np.ndarray
    neg_w: np.ndarray
    neg_scores: np.ndarray

    @property
    def dir_id(self) -> str:
        return f"l{self.l}_h{self.h}_k{self.k}"


def load_mask_vec(masks_dir: Path, l: int, h: int) -> torch.Tensor:
    p = masks_dir / f"layer{l:02d}_head{h:02d}.pt"
    obj = torch.load(p, map_location="cpu")
    if isinstance(obj, dict) and "m" in obj:
        m = obj["m"]
    else:
        m = obj
    return m.float().cpu()


def load_svd(svd_dir: Path, l: int, h: int) -> Dict[str, torch.Tensor]:
    p = svd_dir / f"layer{l:02d}_head{h:02d}.pt"
    obj = torch.load(p, map_location="cpu")
    # Expect keys: U_keep, S_keep, V_keep
    return obj


def build_task_margin(
    task: str,
    split_csv: Path,
    logits_base_at_target: torch.Tensor,
    tokenizer: GPT2TokenizerFast,
) -> Tuple[np.ndarray, Dict[str, int]]:
    """Build margin[n] = logit(correct) - logit(wrong) for each row in CSV, aligned with cache order."""
    df = pd.read_csv(split_csv)
    if task == "gp":
        corr_col = "pronoun"
        wrong_col = "corr_pronoun"
    elif task == "ioi":
        corr_col = "ioi_sentences_labels"
        wrong_col = "ioi_sentences_labels_wrong"
    elif task == "gt":
        # per Fix3 spec
        corr_col = "digits"
        wrong_col = "corr_digits"
    else:
        raise ValueError(f"Unknown task: {task}")

    if corr_col not in df.columns or wrong_col not in df.columns:
        raise KeyError(
            f"CSV {split_csv} missing required columns for task={task}: {corr_col}, {wrong_col}. "
            f"Columns present: {list(df.columns)[:30]}"
        )

    correct_str = df[corr_col].astype(str).tolist()
    wrong_str = df[wrong_col].astype(str).tolist()
    if logits_base_at_target.shape[0] != len(df):
        raise ValueError(
            f"Cache/logits N mismatch for {split_csv}: logits has N={logits_base_at_target.shape[0]} but CSV has {len(df)} rows."
        )

    multi = 0
    token_stats: Dict[str, int] = {"multi_correct": 0, "multi_wrong": 0}
    correct_ids = []
    wrong_ids = []
    for cs, ws in zip(correct_str, wrong_str):
        cid, cm = safe_token_id(tokenizer, cs)
        wid, wm = safe_token_id(tokenizer, ws)
        correct_ids.append(cid)
        wrong_ids.append(wid)
        token_stats["multi_correct"] += int(cm)
        token_stats["multi_wrong"] += int(wm)

    logits = logits_base_at_target.detach().cpu().float().numpy()
    correct_ids = np.array(correct_ids, dtype=np.int64)
    wrong_ids = np.array(wrong_ids, dtype=np.int64)
    margin = logits[np.arange(len(df)), correct_ids] - logits[np.arange(len(df)), wrong_ids]
    return margin.astype(np.float32), token_stats


def compute_direction_corr_and_mass(
    nu_clean_at_target: torch.Tensor,  # [N, L, H, 65]
    margin: np.ndarray,  # [N]
    U_keep: torch.Tensor,  # [65, r]
    S_keep: torch.Tensor,  # [r]
    V_keep: torch.Tensor,  # [768, r]
    m_vec: torch.Tensor,  # [r]
    tau_active: float,
    l: int,
    h: int,
    corr_min_abs_for_flip: float,
) -> List[Tuple[int, float, float, bool, np.ndarray]]:
    """Return list of (k, corr, mass, flip, v_dir[768]) for active k."""
    # active indices
    m_np = m_vec.detach().cpu().numpy()
    active_ks = np.where(m_np >= tau_active)[0]
    if active_ks.size == 0:
        return []

    # Pull the [N, 65] slice once.
    nu_lh = nu_clean_at_target[:, l, h, :].float()  # [N,65]
    # pre-center margin once
    margin_np = margin
    out = []

    for k in active_ks.tolist():
        u = U_keep[:, k].float()  # [65]
        a = (nu_lh @ u).detach().cpu().numpy().astype(np.float32)  # [N]
        corr = pearson_corr(a, margin_np)
        flip = False
        v = V_keep[:, k].detach().cpu().numpy().astype(np.float32)  # [768]
        sigma = float(S_keep[k].item())
        mask = float(m_np[k])
        # orient so corr >= 0 when it's confident enough
        if corr < 0 and abs(corr) >= corr_min_abs_for_flip:
            flip = True
            corr = -corr
            v = -v
        mass = mask * sigma * abs(corr)
        out.append((k, corr, mass, flip, v))

    return out


def compute_logit_footprints(
    model: GPT2LMHeadModel,
    dirs_meta: List[Tuple[int, int, int, float, float, bool, float, np.ndarray]],
    topL: int,
    fp16: bool = True,
    chunk: int = 256,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Given directions metadata, compute topL pos/neg ids and scores.

    dirs_meta rows: (l,h,k,sigma,mask,flip,corr,v[768]) where v already oriented.
    """
    # W_U: [V, 768]
    W = model.lm_head.weight.detach()
    dev = W.device

    # stack (sigma * v) into [N,768]
    vecs = np.stack([sigma * v for (_l, _h, _k, sigma, _m, _flip, _corr, v) in dirs_meta], axis=0).astype(np.float32)
    N = vecs.shape[0]
    pos_ids_list: List[np.ndarray] = []
    pos_scores_list: List[np.ndarray] = []
    neg_ids_list: List[np.ndarray] = []
    neg_scores_list: List[np.ndarray] = []

    dtype = torch.float16 if (fp16 and dev.type == "cuda") else torch.float32

    with torch.no_grad():
        for i0 in tqdm(range(0, N, chunk), desc="[Fix3] logit footprints"):
            i1 = min(N, i0 + chunk)
            Vb = torch.tensor(vecs[i0:i1], device=dev, dtype=dtype)  # [B,768]
            # R = Vb @ W.T -> [B, V]
            R = Vb @ W.T
            # top positive
            vals_pos, ids_pos = torch.topk(R, k=topL, dim=-1)
            # top negative
            vals_neg, ids_neg = torch.topk(-R, k=topL, dim=-1)
            # store
            pos_ids_list.extend([_to_numpy(x) for x in ids_pos])
            pos_scores_list.extend([_to_numpy(x) for x in vals_pos.float()])
            neg_ids_list.extend([_to_numpy(x) for x in ids_neg])
            neg_scores_list.extend([_to_numpy((-x).float()) for x in vals_neg])  # convert back to negative scores

    return pos_ids_list, pos_scores_list, neg_ids_list, neg_scores_list


# -------------------------
# Clustering
# -------------------------


def similarity_token_footprint(
    pos_ids_i: np.ndarray,
    pos_w_i: np.ndarray,
    neg_ids_i: np.ndarray,
    neg_w_i: np.ndarray,
    pos_ids_j: np.ndarray,
    pos_w_j: np.ndarray,
    neg_ids_j: np.ndarray,
    neg_w_j: np.ndarray,
) -> float:
    j_pos = weighted_jaccard(pos_ids_i, pos_w_i, pos_ids_j, pos_w_j)
    j_neg = weighted_jaccard(neg_ids_i, neg_w_i, neg_ids_j, neg_w_j)
    return 0.5 * (j_pos + j_neg)


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def connected_components_from_sim(sim_mat: np.ndarray, threshold: float) -> List[List[int]]:
    n = sim_mat.shape[0]
    uf = UnionFind(n)
    for i in range(n):
        # only upper triangle
        for j in range(i + 1, n):
            if sim_mat[i, j] >= threshold:
                uf.union(i, j)
    comps: Dict[int, List[int]] = {}
    for i in range(n):
        r = uf.find(i)
        comps.setdefault(r, []).append(i)
    return list(comps.values())


def compute_cluster_metrics(
    comps: List[List[int]],
    masses: np.ndarray,
    sim_mat: np.ndarray,
    topk: int = 10,
) -> Dict[str, float]:
    masses = masses.astype(np.float64)
    total = float(masses.sum() + 1e-12)
    cluster_mass = np.array([masses[c].sum() for c in comps], dtype=np.float64)
    cluster_mass_sorted = np.sort(cluster_mass)[::-1]
    topk_mass = float(cluster_mass_sorted[:topk].sum())
    top20_mass = float(cluster_mass_sorted[:20].sum())
    singleton_frac = float(np.mean([len(c) == 1 for c in comps]))
    # mean intra sim across all within-cluster pairs
    pair_sims = []
    for c in comps:
        if len(c) <= 1:
            continue
        for ii in range(len(c)):
            for jj in range(ii + 1, len(c)):
                pair_sims.append(sim_mat[c[ii], c[jj]])
    mean_intra = float(np.mean(pair_sims)) if pair_sims else 0.0
    return {
        "n_clusters": float(len(comps)),
        "singleton_frac": singleton_frac,
        "top10_mass_frac": topk_mass / total,
        "top20_mass_frac": top20_mass / total,
        "mean_intra_sim": mean_intra,
        "total_mass": total,
    }


def select_threshold_grid(
    sim_mat: np.ndarray,
    masses: np.ndarray,
    grid: Sequence[float],
    min_mean_intra: float,
) -> Tuple[float, Dict[str, float], List[List[int]]]:
    best_t = None
    best_metrics = None
    best_comps = None
    candidates: List[Tuple[float, Dict[str, float], List[List[int]]]] = []
    for t in grid:
        comps = connected_components_from_sim(sim_mat, t)
        metrics = compute_cluster_metrics(comps, masses, sim_mat)
        candidates.append((t, metrics, comps))

    # choose best by top10_mass_frac with constraint
    feasible = [c for c in candidates if c[1]["mean_intra_sim"] >= min_mean_intra]
    pool = feasible if feasible else candidates
    pool = sorted(
        pool,
        key=lambda x: (x[1]["top10_mass_frac"], -x[1]["n_clusters"]),
        reverse=True,
    )
    best_t, best_metrics, best_comps = pool[0]
    return float(best_t), best_metrics, best_comps


# -------------------------
# Bank + X rebuild
# -------------------------


def build_bank_medoid(
    comps: List[List[int]],
    sim_mat: np.ndarray,
    v_dirs: np.ndarray,  # [N,768]
    dirs: List[Direction],
    masses: np.ndarray,
    tokenizer: GPT2TokenizerFast,
) -> Tuple[np.ndarray, List[dict]]:
    """Return (v_bank[C,768], rep_meta[C])."""
    v_bank = []
    rep_meta = []

    for cid, members in enumerate(comps):
        if len(members) == 1:
            med = members[0]
        else:
            # medoid: max sum sim within cluster
            sub = sim_mat[np.ix_(members, members)]
            sums = sub.sum(axis=1)
            med = members[int(np.argmax(sums))]

        drep = dirs[med]
        v_bank.append(v_dirs[med])

        # human-readable top tokens
        pos_tok = [tokenizer.decode([int(t)]).replace("\n", "\\n") for t in drep.pos_ids[:10]]
        neg_tok = [tokenizer.decode([int(t)]).replace("\n", "\\n") for t in drep.neg_ids[:10]]
        m_masses = [float(masses[i]) for i in members]
        rep_meta.append(
            {
                "cluster_id": cid,
                "n_members": len(members),
                "mass_total": float(sum(m_masses)),
                "rep_dir": {"layer": drep.l, "head": drep.h, "k": drep.k},
                "rep_dir_id": drep.dir_id,
                "rep_corr": float(drep.corr),
                "rep_mask": float(drep.mask),
                "rep_sigma": float(drep.sigma),
                "top_pos_tokens": pos_tok,
                "top_neg_tokens": neg_tok,
                "members": [dirs[i].dir_id for i in members],
                "member_mass": m_masses,
            }
        )

    v_bank = np.stack(v_bank, axis=0).astype(np.float32)
    return v_bank, rep_meta


def rebuild_X(
    out_dir: Path,
    cache_dir: Path,
    svd_dir: Path,
    masks_dir: Path,
    dirs: List[Direction],
    dir_to_cluster: Dict[str, int],
    C_new: int,
    x_scale: str = "sigma",
    store_masked: bool = True,
) -> Dict[str, Dict[str, str]]:
    """Rebuild X tensors for train/val/test.

    x_scale: "none" or "sigma".
    If store_masked, scales by mask as in spec.
    """
    assert x_scale in {"none", "sigma"}
    out_art = out_dir / "artifacts" / "e1_fix3"
    out_art.mkdir(parents=True, exist_ok=True)

    # Preload per (l,h) SVD U_keep and S_keep and masks.
    # Access patterns are sparse; cache dicts.
    svd_cache: Dict[Tuple[int, int], Dict[str, torch.Tensor]] = {}
    m_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    def get_lh(l: int, h: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        key = (l, h)
        if key not in svd_cache:
            svd_cache[key] = load_svd(svd_dir, l, h)
        if key not in m_cache:
            m_cache[key] = load_mask_vec(masks_dir, l, h)
        U = svd_cache[key]["U_keep"].float()  # [65,r]
        S = svd_cache[key]["S_keep"].float()  # [r]
        m = m_cache[key].float()  # [r]
        return U, S, m

    results: Dict[str, Dict[str, str]] = {}
    for split in ["train", "val", "test"]:
        p = cache_path(cache_dir, split)
        cache = torch.load(p, map_location="cpu")
        nu_clean = cache["nu_clean_at_target"].float()  # [N,L,H,65]
        nu_corr = cache["nu_corrupt_at_target"].float()
        N, L, H, D = nu_clean.shape
        X_clean = torch.zeros((N, L, C_new), dtype=torch.float32)
        X_corr = torch.zeros((N, L, C_new), dtype=torch.float32)

        for d in tqdm(dirs, desc=f"[Fix3] rebuild X {split}"):
            c = dir_to_cluster[d.dir_id]
            U, S, m_vec = get_lh(d.l, d.h)
            u = U[:, d.k]  # [65]
            a_clean = (nu_clean[:, d.l, d.h, :] @ u).float()  # [N]
            a_corr = (nu_corr[:, d.l, d.h, :] @ u).float()
            if x_scale == "sigma":
                a_clean = a_clean * float(S[d.k].item())
                a_corr = a_corr * float(S[d.k].item())
            if d.flip:
                a_clean = -a_clean
                a_corr = -a_corr
            if store_masked:
                a_clean = a_clean * float(m_vec[d.k].item())
                a_corr = a_corr * float(m_vec[d.k].item())
            X_clean[:, d.l, c] += a_clean
            X_corr[:, d.l, c] += a_corr

        out_paths = {}
        p_clean = out_art / f"X_clean_{split}_fix3.pt"
        p_corr = out_art / f"X_corr_{split}_fix3.pt"
        torch.save(X_clean, p_clean)
        torch.save(X_corr, p_corr)
        out_paths["X_clean"] = str(p_clean)
        out_paths["X_corr"] = str(p_corr)
        results[split] = out_paths
        print(f"[Fix3] wrote X for {split}: {out_paths}")

    return results


# -------------------------
# Main
# -------------------------


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=["gp", "ioi", "gt"])
    ap.add_argument("--out_dir", required=True, type=str, help="Output dir, e.g. outputs/gp")
    ap.add_argument("--data_dir", required=True, type=str, help="Dataset dir, e.g. data_main")
    ap.add_argument("--tau_active", type=float, default=1e-2)
    ap.add_argument("--topL", type=int, default=32)
    ap.add_argument("--threshold_grid", type=str, default="0.20,0.25,0.30,0.35,0.40")
    ap.add_argument("--min_mean_intra", type=float, default=0.20)
    ap.add_argument("--corr_min_abs_for_flip", type=float, default=0.05)
    ap.add_argument("--x_scale", type=str, default="sigma", choices=["none", "sigma"])
    ap.add_argument("--store_masked_X", action="store_true")
    ap.add_argument("--no_fp16", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch_size_nu", type=int, default=64)
    args = ap.parse_args()

    _seed_all(args.seed)

    task = args.task
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    cache_dir, svd_dir, masks_dir = resolve_out_layout(out_dir)
    art_dir = out_dir / "artifacts" / "e1_fix3"
    art_dir.mkdir(parents=True, exist_ok=True)

    print("[Fix3] Using:")
    print(f"  out_dir   = {out_dir}")
    print(f"  cache_dir = {cache_dir}")
    print(f"  svd_dir   = {svd_dir}")
    print(f"  masks_dir = {masks_dir}")
    print(f"  art_dir   = {art_dir}")

    # Load model
    dev = _device()
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    model.to(dev)
    model.eval()

    # Ensure nu_clean exists in caches
    ensure_nu_clean(out_dir, cache_dir, model, batch_size=args.batch_size_nu)

    # Load val cache + CSV to compute margin
    val_cache = torch.load(cache_path(cache_dir, "val"), map_location="cpu")
    logits_val = val_cache["logits_base_at_target"].float()
    val_csv = find_csv(data_dir, task, "val")
    margin_val, token_stats = build_task_margin(task, val_csv, logits_val, tokenizer)
    print(f"[Fix3] val_csv = {val_csv.name}  token_stats={token_stats}")

    nu_clean_val = val_cache["nu_clean_at_target"].float()  # [N,L,H,65]
    L = nu_clean_val.shape[1]
    H = nu_clean_val.shape[2]

    # Enumerate active directions + compute corr/mass + oriented v
    dirs_meta: List[Tuple[int, int, int, float, float, bool, float, np.ndarray]] = []
    # (l,h,k,sigma,mask,flip,corr,v)

    for l in tqdm(range(L), desc="[Fix3] directions layers"):
        for h in range(H):
            svd = load_svd(svd_dir, l, h)
            U_keep = svd["U_keep"]
            S_keep = svd["S_keep"]
            V_keep = svd["V_keep"]
            m_vec = load_mask_vec(masks_dir, l, h)
            triples = compute_direction_corr_and_mass(
                nu_clean_val,
                margin_val,
                U_keep,
                S_keep,
                V_keep,
                m_vec,
                tau_active=args.tau_active,
                l=l,
                h=h,
                corr_min_abs_for_flip=args.corr_min_abs_for_flip,
            )
            if not triples:
                continue
            m_np = m_vec.detach().cpu().numpy()
            for (k, corr, mass, flip, v) in triples:
                sigma = float(S_keep[k].item())
                mask = float(m_np[k])
                dirs_meta.append((l, h, k, sigma, mask, flip, float(corr), v))

    if len(dirs_meta) == 0:
        raise RuntimeError(
            f"No active directions found at tau_active={args.tau_active}. "
            f"Check your masks or lower --tau_active."
        )

    print(f"[Fix3] N_dir_active = {len(dirs_meta)} (tau_active={args.tau_active})")

    # Footprints
    pos_ids_list, pos_scores_list, neg_ids_list, neg_scores_list = compute_logit_footprints(
        model,
        dirs_meta,
        topL=args.topL,
        fp16=(not args.no_fp16),
        chunk=256,
    )

    # Build Direction objects with softmax-normalized weights within pos/neg
    dirs: List[Direction] = []
    v_dirs = []
    masses = []

    for i, (l, h, k, sigma, mask, flip, corr, v) in enumerate(dirs_meta):
        pos_ids = pos_ids_list[i]
        pos_scores = pos_scores_list[i]
        neg_ids = neg_ids_list[i]
        neg_scores = neg_scores_list[i]
        pos_w = softmax_np(pos_scores)
        neg_w = softmax_np(-neg_scores)  # weights over negative magnitudes
        d = Direction(
            l=int(l),
            h=int(h),
            k=int(k),
            sigma=float(sigma),
            mask=float(mask),
            flip=bool(flip),
            corr=float(corr),
            mass=float(mask * sigma * abs(corr)),
            v=v,
            pos_ids=pos_ids.astype(np.int64),
            pos_w=pos_w.astype(np.float32),
            pos_scores=pos_scores.astype(np.float32),
            neg_ids=neg_ids.astype(np.int64),
            neg_w=neg_w.astype(np.float32),
            neg_scores=neg_scores.astype(np.float32),
        )
        dirs.append(d)
        v_dirs.append(v)
        masses.append(d.mass)

    v_dirs = np.stack(v_dirs, axis=0).astype(np.float32)  # [N,768]
    masses = np.array(masses, dtype=np.float32)

    # Print quantiles
    mask_q = np.quantile([d.mask for d in dirs], [0, 0.5, 0.9, 0.99, 1.0]).tolist()
    sigma_q = np.quantile([d.sigma for d in dirs], [0, 0.5, 0.9, 0.99, 1.0]).tolist()
    mass_q = np.quantile(masses, [0, 0.5, 0.9, 0.99, 1.0]).tolist()
    print(f"[Fix3] mask quantiles (0,50,90,99,100%): {mask_q}")
    print(f"[Fix3] sigma quantiles (0,50,90,99,100%): {sigma_q}")
    print(f"[Fix3] mass quantiles (0,50,90,99,100%): {mass_q}")

    # Build full similarity matrix (N<=~1000 typically)
    N = len(dirs)
    sim_mat = np.zeros((N, N), dtype=np.float32)
    for i in tqdm(range(N), desc="[Fix3] sim matrix"):
        sim_mat[i, i] = 1.0
        for j in range(i + 1, N):
            sim = similarity_token_footprint(
                dirs[i].pos_ids, dirs[i].pos_w, dirs[i].neg_ids, dirs[i].neg_w,
                dirs[j].pos_ids, dirs[j].pos_w, dirs[j].neg_ids, dirs[j].neg_w,
            )
            sim_mat[i, j] = sim
            sim_mat[j, i] = sim

    # Threshold selection
    grid = [float(x) for x in args.threshold_grid.split(",") if x.strip()]
    t_best, metrics_best, comps_best = select_threshold_grid(
        sim_mat,
        masses,
        grid=grid,
        min_mean_intra=args.min_mean_intra,
    )
    print(f"[Fix3] Selected threshold t={t_best} with metrics={metrics_best}")

    # Build mapping dir->cluster
    dir_to_cluster: Dict[str, int] = {}
    for cid, members in enumerate(comps_best):
        for idx in members:
            dir_to_cluster[dirs[idx].dir_id] = cid

    C_new = len(comps_best)
    v_bank, rep_meta = build_bank_medoid(comps_best, sim_mat, v_dirs, dirs, masses, tokenizer)

    # Save bank
    bank = {
        "v_bank": torch.tensor(v_bank),
        "v_c": torch.tensor(v_bank),  # alias for older code
        "rep_meta": rep_meta,
        "threshold": t_best,
        "tau_active": float(args.tau_active),
        "topL": int(args.topL),
        "x_scale": args.x_scale,
        "store_masked_X": bool(args.store_masked_X),
    }
    torch.save(bank, art_dir / "receptor_bank.pt")
    with open(art_dir / "dir_to_cluster.json", "w") as f:
        json.dump(dir_to_cluster, f, indent=2)

    # Cluster mass table
    cluster_mass = []
    for cid, members in enumerate(comps_best):
        cluster_mass.append(
            {
                "cluster_id": cid,
                "mass_total": float(masses[members].sum()),
                "n_members": len(members),
            }
        )
    df_mass = pd.DataFrame(cluster_mass).sort_values("mass_total", ascending=False)
    df_mass.to_csv(art_dir / "cluster_mass.csv", index=False)

    # Plots
    masses_sorted = df_mass["mass_total"].to_numpy()
    cdf = np.cumsum(masses_sorted) / (masses_sorted.sum() + 1e-12)
    plt.figure()
    plt.plot(np.arange(1, len(cdf) + 1), cdf)
    plt.xlabel("#clusters")
    plt.ylabel("cumulative mass fraction")
    plt.title("Fix3 mass CDF")
    plt.savefig(art_dir / "mass_cdf.png", dpi=160)
    plt.close()

    sizes = df_mass["n_members"].to_numpy()
    plt.figure()
    plt.hist(sizes, bins=min(50, max(5, int(sizes.max()))))
    plt.xlabel("cluster size")
    plt.ylabel("count")
    plt.title("Fix3 cluster size histogram")
    plt.savefig(art_dir / "cluster_size_hist.png", dpi=160)
    plt.close()

    sim_vals = sim_mat[np.triu_indices(N, k=1)]
    plt.figure()
    plt.hist(sim_vals, bins=50)
    plt.xlabel("pairwise similarity")
    plt.ylabel("count")
    plt.title("Fix3 pairwise similarity (all pairs)")
    plt.savefig(art_dir / "intra_sim_hist.png", dpi=160)
    plt.close()

    # Rebuild X tensors
    x_paths = rebuild_X(
        out_dir,
        cache_dir,
        svd_dir,
        masks_dir,
        dirs,
        dir_to_cluster,
        C_new=C_new,
        x_scale=args.x_scale,
        store_masked=args.store_masked_X,
    )

    # Final report
    report = {
        "task": task,
        "tau_active": float(args.tau_active),
        "topL": int(args.topL),
        "threshold_grid": grid,
        "threshold_selected": float(t_best),
        "metrics": metrics_best,
        "N_dir_active": int(N),
        "C_new": int(C_new),
        "x_scale": args.x_scale,
        "store_masked_X": bool(args.store_masked_X),
        "token_stats": token_stats,
        "x_paths": x_paths,
    }
    with open(art_dir / "report.json", "w") as f:
        json.dump(report, f, indent=2)

    # Print compact summary for copying
    print("\n=== FIX3 SUMMARY ===")
    print(f"N_dir_active: {N}")
    print(f"C_new: {C_new}")
    print(f"threshold_selected: {t_best}")
    print(f"top10_mass_frac: {metrics_best['top10_mass_frac']}")
    print(f"top20_mass_frac: {metrics_best['top20_mass_frac']}")
    print(f"singleton_frac: {metrics_best['singleton_frac']}")
    print(f"mean_intra_sim: {metrics_best['mean_intra_sim']}")
    print(f"Saved under: {art_dir}")


if __name__ == "__main__":
    main()
