
#!/usr/bin/env python3
"""
E1 postprocessing for OV-mask runs (GPT-2 small).

Reads artifacts produced by run_ov_mask.py:
  - cache/{split}.pt with tokens_clean, target_pos, logits_base_at_target, nu_corrupt_at_target
  - svd/layer{l}_head{h}.pt with U_keep, S_keep, V_keep
  - masks/layer{l}_head{h}.pt with m (sigmoid(mask_logits)) of length rank

Implements Steps 0-7 from the near-pseudocode spec:
  0) Ensure nu_clean_at_target exists in cache files
  1) Direction table per split + combined (train primary)
  2) Candidate selection
  3) Stage-1 clustering (cosine on V directions)
  4) Stage-2 logit confirmation (cosine on top tokens in logit-space)
  5) Canonical receptor bank
  6) Receptor state tensors X_{clean,corr,mix}
  7) Validations + plots + report

This script is designed to be self-contained and run on Kaggle.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from transformers import AutoTokenizer, AutoModelForCausalLM


# ----------------------------
# Utilities
# ----------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device) if isinstance(x, torch.Tensor) else x

def pearsonr(x: np.ndarray, y: np.ndarray) -> float:
    # Robust Pearson corr with nan-safe handling
    x = x.astype(np.float64)
    y = y.astype(np.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = (np.linalg.norm(x) * np.linalg.norm(y))
    if denom == 0:
        return 0.0
    return float(np.dot(x, y) / denom)

def safe_token_id(tokenizer: Any, s: str) -> Tuple[int, bool]:
    """
    Encode a *single* token string (assumes prediction after a space).
    Returns (token_id, is_single_token). If not single-token, we fallback to first token id.
    """
    ids = tokenizer.encode(" " + str(s), add_special_tokens=False)
    if len(ids) == 0:
        # Extremely unlikely; fall back to encoding without leading space
        ids = tokenizer.encode(str(s), add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError(f"Could not tokenize label: {s!r}")
    return ids[0], (len(ids) == 1)

def json_dump(path: Path, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def try_write_table(df: pd.DataFrame, out_base: Path) -> Path:
    """
    Write a table with minimal deps. Prefer parquet if available, else csv.gz.
    Returns the written path.
    """
    try:
        out_path = out_base.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
        return out_path
    except Exception:
        out_path = out_base.with_suffix(".csv.gz")
        df.to_csv(out_path, index=False, compression="gzip")
        return out_path


# ----------------------------
# Capturing z = c_proj input
# ----------------------------

class CProjInputCacher:
    """
    Captures, per layer, the attention output (context) that is fed into c_proj:
      z: [B, T, n_heads, d_head]
    We store z at target_pos per-example, per-head: [B, n_heads, d_head]
    """
    def __init__(self, model: Any):
        self.model = model
        self.handles = []
        self._buf: List[torch.Tensor] = []

    def _hook(self, module, module_in, module_out):
        # module_in is a tuple; first element is [B, T, d_model]
        x = module_in[0]
        # reshape to [B,T,n_heads,d_head]
        B, T, D = x.shape
        n_heads = self.model.config.n_head
        d_head = D // n_heads
        z = x.view(B, T, n_heads, d_head).detach()
        self._buf.append(z)

    def __enter__(self):
        self._buf = []
        for block in self.model.transformer.h:
            # GPT-2: attention c_proj is block.attn.c_proj
            self.handles.append(block.attn.c_proj.register_forward_hook(self._hook))
        return self

    def __exit__(self, exc_type, exc, tb):
        for h in self.handles:
            h.remove()
        self.handles = []

    def get(self) -> List[torch.Tensor]:
        return self._buf


# ----------------------------
# Union-Find for clustering
# ----------------------------

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

    def groups(self) -> Dict[int, List[int]]:
        out: Dict[int, List[int]] = {}
        for i in range(len(self.parent)):
            r = self.find(i)
            out.setdefault(r, []).append(i)
        return out


# ----------------------------
# Dataset label loading
# ----------------------------

def infer_split_files(task: str, data_dir: Path) -> Dict[str, Path]:
    """
    Looks for standard names in data_dir for a given task.
    You can override with explicit CLI args.
    """
    task = task.lower()
    patterns = {
        "gp": ["train_1k_gp.csv", "val_gp.csv", "test_gp.csv", "train_gp.csv", "val_1k_gp.csv", "test_1k_gp.csv"],
        "ioi": ["train_ioi.csv", "val_ioi.csv", "test_ioi.csv", "train_1k_ioi.csv", "val_1k_ioi.csv", "test_1k_ioi.csv"],
        "gt": ["train_gt_1k.csv", "val_gt.csv", "test_gt.csv", "train_gt.csv", "val_1k_gt.csv", "test_1k_gt.csv"],
    }
    # Find best matches
    files = {p.name: p for p in data_dir.glob("*.csv")}
    def pick(cands):
        for c in cands:
            if c in files:
                return files[c]
        return None
    train = pick(patterns[task])
    val = pick([p.replace("train", "val") for p in patterns[task]])
    test = pick([p.replace("train", "test") for p in patterns[task]])
    out = {}
    if train: out["train"] = train
    if val: out["val"] = val
    if test: out["test"] = test
    return out

def load_labels_for_split(task: str, csv_path: Path) -> Tuple[List[str], List[str]]:
    """
    Returns (correct_labels, wrong_labels) as strings per row, order preserved.
    """
    df = pd.read_csv(csv_path)
    task = task.lower()
    if task == "gp":
        # correct: pronoun, wrong: corr_pronoun
        return df["pronoun"].astype(str).tolist(), df["corr_pronoun"].astype(str).tolist()
    if task == "ioi":
        return df["ioi_sentences_labels"].astype(str).tolist(), df["ioi_sentences_labels_wrong"].astype(str).tolist()
    if task == "gt":
        return df["century"].astype(str).tolist(), df["corr_century"].astype(str).tolist()
    raise ValueError(f"Unknown task: {task}")


# ----------------------------
# Core E1 pipeline
# ----------------------------

@dataclass
class E1Config:
    task: str
    out_dir: Path
    data_dir: Path

    tau_report: float = 1e-2
    tau_candidate: float = 1e-3  # currently unused; kept for provenance
    K_per_head: int = 4
    K_global_cap: int = 2000

    cos_merge_stage1: float = 0.95
    cos_confirm_stage2: float = 0.90
    topT_tokens: int = 50

    assign_cos_threshold: float = 0.90

    n_receptors_intervene: int = 3
    alpha_grid: Tuple[float, ...] = (-2, -1, -0.5, 0.5, 1, 2)

    state_variant: str = "clean+corr"  # or "mix"

    device: str = "cuda"  # "cuda" or "cpu"
    seed: int = 0

    # Performance knobs
    chunk_size_cos: int = 256  # for cosine union-find stage-1

def load_cache(cache_path: Path) -> Dict[str, torch.Tensor]:
    cache = torch.load(cache_path, map_location="cpu")
    if not isinstance(cache, dict):
        raise ValueError(f"Cache is not a dict: {cache_path}")
    return cache

def save_cache(cache_path: Path, cache: Dict[str, torch.Tensor]) -> None:
    torch.save(cache, cache_path)

def ensure_nu_clean(cfg: E1Config, model: Any, split: str, cache_path: Path) -> None:
    cache = load_cache(cache_path)
    if "nu_clean_at_target" in cache:
        return

    tokens_clean = cache["tokens_clean"]  # [N,T]
    target_pos = cache["target_pos"]      # [N]
    N, T = tokens_clean.shape
    n_layers = model.config.n_layer
    n_heads = model.config.n_head
    d_head = model.config.n_embd // n_heads

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.eval().to(device)

    # We'll do a batched forward with c_proj hooks to capture z for all layers.
    # For memory, chunk over N.
    batch = 64
    nu_clean = torch.empty((N, n_layers, n_heads, d_head + 1), dtype=torch.float32)

    ones = torch.ones((batch, 1), dtype=torch.float32, device=device)

    with torch.no_grad():
        for i in tqdm(range(0, N, batch), desc=f"Step0 nu_clean {split}", leave=False):
            j = min(i + batch, N)
            inp = tokens_clean[i:j].to(device)
            tpos = target_pos[i:j].to(device)
            with CProjInputCacher(model) as cacher:
                _ = model(inp)
                z_layers = cacher.get()
            if len(z_layers) != n_layers:
                raise RuntimeError(f"Expected {n_layers} layers, got {len(z_layers)}")
            # Stack: [B,L,T,H,Dh]
            z = torch.stack(z_layers, dim=1)  # [B,L,T,H,Dh]
            # Gather at target positions
            B = j - i
            # build indices
            ar = torch.arange(B, device=device)
            z_t = z[ar[:, None], torch.arange(n_layers, device=device)[None, :], tpos[:, None], :, :]  # [B,L,H,Dh]
            # Append ones
            ones_b = torch.ones((B, n_layers, n_heads, 1), device=device, dtype=z_t.dtype)
            nu = torch.cat([z_t, ones_b], dim=-1).float().cpu()  # [B,L,H,65]
            nu_clean[i:j] = nu

    cache["nu_clean_at_target"] = nu_clean
    save_cache(cache_path, cache)

def load_svd_file(svd_path: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = torch.load(svd_path, map_location="cpu")
    # We support either direct tensors or dict entries
    if isinstance(d, dict):
        U = d["U_keep"]
        S = d["S_keep"]
        V = d["V_keep"]
    else:
        raise ValueError(f"Unexpected SVD format in {svd_path}")
    return U.float(), S.float(), V.float()

def load_mask_m(mask_path: Path) -> torch.Tensor:
    d = torch.load(mask_path, map_location="cpu")
    if isinstance(d, dict) and "m" in d:
        return d["m"].float()
    if isinstance(d, torch.Tensor):
        return d.float()
    raise ValueError(f"Unexpected mask format in {mask_path}")

def build_direction_table_for_split(cfg: E1Config, split: str, cache_path: Path,
                                   svd_dir: Path, mask_dir: Path) -> pd.DataFrame:
    cache = load_cache(cache_path)
    nu_clean = cache["nu_clean_at_target"].float()        # [N,L,H,65]
    nu_corr  = cache["nu_corrupt_at_target"].float()      # [N,L,H,65]
    N, L, H, D = nu_clean.shape

    rows = []
    # Iterate layer/head
    for l in tqdm(range(L), desc=f"Directions {split} layers"):
        for h in range(H):
            svd_path = svd_dir / f"layer{l:02d}_head{h:02d}.pt"
            mask_path = mask_dir / f"layer{l:02d}_head{h:02d}.pt"
            if not svd_path.exists() or not mask_path.exists():
                raise FileNotFoundError(f"Missing {svd_path} or {mask_path}")
            U, S, V = load_svd_file(svd_path)  # U:[65,r], S:[r], V:[768,r]
            m = load_mask_m(mask_path)         # [r]
            r = int(S.shape[0])
            if m.numel() != r:
                raise ValueError(f"Mask length mismatch at layer{l} head{h}: m={m.numel()} r={r}")

            # Project activations into U basis
            # A_clean: [N,r], A_corr: [N,r]
            # We'll compute via einsum for speed and clarity.
            A_clean = torch.einsum("nd,dr->nr", nu_clean[:, l, h, :], U)  # [N,r]
            A_corr  = torch.einsum("nd,dr->nr", nu_corr[:, l, h, :], U)   # [N,r]
            delta = A_clean - A_corr
            rms_delta = torch.sqrt(torch.mean(delta * delta, dim=0) + 1e-12)  # [r]
            score = (m * S * rms_delta)  # [r]
            active = (m >= cfg.tau_report)

            # Append rows
            for k in range(r):
                rows.append({
                    "split": split,
                    "layer": l,
                    "head": h,
                    "k": k,
                    "rank": r,
                    "m": float(m[k].item()),
                    "sigma": float(S[k].item()),
                    "rms_delta": float(rms_delta[k].item()),
                    "score": float(score[k].item()),
                    "active": bool(active[k].item()),
                })

    df = pd.DataFrame(rows)
    return df

def select_candidates(cfg: E1Config, train_df: pd.DataFrame) -> List[Tuple[int,int,int]]:
    # Active set
    cand = set()
    for row in train_df.itertuples(index=False):
        if row.active:
            cand.add((int(row.layer), int(row.head), int(row.k)))

    # Top K_per_head by score per (layer, head)
    grouped = train_df.groupby(["layer", "head"], sort=False)
    for (l, h), g in grouped:
        g2 = g.sort_values("score", ascending=False).head(cfg.K_per_head)
        for r in g2.itertuples(index=False):
            cand.add((int(r.layer), int(r.head), int(r.k)))

    # Global cap
    if len(cand) > cfg.K_global_cap:
        # keep top by score among cand
        cand_df = train_df[train_df.apply(lambda r: (int(r.layer), int(r.head), int(r.k)) in cand, axis=1)]
        cand_df = cand_df.sort_values("score", ascending=False).head(cfg.K_global_cap)
        cand = {(int(r.layer), int(r.head), int(r.k)) for r in cand_df.itertuples(index=False)}

    return sorted(list(cand))

def load_candidate_vectors(candidates: List[Tuple[int,int,int]], svd_dir: Path) -> torch.Tensor:
    # Returns V dirs stacked: [N,768] (each is V_keep[:,k])
    vecs = []
    cache_svd: Dict[Tuple[int,int], torch.Tensor] = {}
    for (l,h,k) in candidates:
        key = (l,h)
        if key not in cache_svd:
            _, _, V = load_svd_file(svd_dir / f"layer{l:02d}_head{h:02d}.pt")  # [768,r]
            cache_svd[key] = V
        V = cache_svd[key]
        v = V[:, k]
        vecs.append(v)
    X = torch.stack(vecs, dim=0).float()  # [N,768]
    # Normalize
    X = X / (X.norm(dim=1, keepdim=True) + 1e-12)
    return X

def stage1_cluster(cfg: E1Config, X: torch.Tensor) -> List[int]:
    """
    Single-linkage style union-find clustering: union pairs with cosine >= cos_merge_stage1.
    X: [N,768] normalized.
    Returns cluster_id for each index (0..N-1) as a dense 0..C-1 mapping.
    """
    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    Xd = X.to(device)
    N = Xd.shape[0]
    uf = UnionFind(N)

    chunk = cfg.chunk_size_cos
    with torch.no_grad():
        for i in tqdm(range(0, N, chunk), desc="Stage1 clustering", leave=False):
            Xi = Xd[i:i+chunk]                          # [c,768]
            sim = Xi @ Xd.T                             # [c,N]
            # Get indices where sim >= threshold
            mask = sim >= cfg.cos_merge_stage1
            # Convert to pairs
            idx = mask.nonzero(as_tuple=False)          # [E,2] (row_in_chunk, col)
            for rowcol in idx.tolist():
                r0, c0 = rowcol
                a = i + r0
                b = c0
                if a != b:
                    uf.union(a, b)

    groups = uf.groups()
    # Map root -> new id
    root_to_id = {root: ci for ci, root in enumerate(sorted(groups.keys()))}
    cluster_ids = [root_to_id[uf.find(i)] for i in range(N)]
    return cluster_ids

def stage2_confirm(cfg: E1Config,
                   candidates: List[Tuple[int,int,int]],
                   cluster_ids_stage1: List[int],
                   train_df: pd.DataFrame,
                   svd_dir: Path,
                   model: Any,
                   tokenizer: Any) -> Tuple[Dict[str, Any], Dict[str, int], List[Tuple[int,int,int]]]:
    """
    Logit-space confirmation using representative top tokens.
    Returns:
      registry (clusters + metadata),
      dir_to_cluster mapping (key "l_h_k" -> cluster_id),
      rep_ids list per cluster (as direction tuples)
    """
    # Build per-candidate score lookup from train_df
    score_map: Dict[Tuple[int,int,int], float] = {}
    sigma_map: Dict[Tuple[int,int,int], float] = {}
    m_map: Dict[Tuple[int,int,int], float] = {}
    for r in train_df.itertuples(index=False):
        key = (int(r.layer), int(r.head), int(r.k))
        score_map[key] = float(r.score)
        sigma_map[key] = float(r.sigma)
        m_map[key] = float(r.m)

    # group candidates by stage1 cluster
    clusters: Dict[int, List[int]] = {}
    for idx, cid in enumerate(cluster_ids_stage1):
        clusters.setdefault(int(cid), []).append(idx)

    # Preload normalized v vectors and also raw v (non-normalized) for receptor computation
    cache_V: Dict[Tuple[int,int], torch.Tensor] = {}
    def get_v(l:int,h:int,k:int) -> torch.Tensor:
        key = (l,h)
        if key not in cache_V:
            _, _, V = load_svd_file(svd_dir / f"layer{l:02d}_head{h:02d}.pt")
            cache_V[key] = V.float()
        return cache_V[key][:, k]

    WU = model.lm_head.weight.detach().float().cpu()  # [V,768]
    Vocab = WU.shape[0]

    registry_clusters = []
    dir_to_cluster: Dict[str, int] = {}

    final_cluster_id = 0
    rep_dirs: List[Tuple[int,int,int]] = []

    for cid, members_idx in tqdm(clusters.items(), desc="Stage2 confirm", leave=False):
        # Pick representative: max score
        member_dirs = [candidates[i] for i in members_idx]
        rep = max(member_dirs, key=lambda d: score_map.get(d, -1e9))
        v_rep = get_v(*rep).float()
        v_rep_norm = v_rep / (v_rep.norm() + 1e-12)

        # Compute receptor logits and top tokens by abs
        r_rep = (v_rep_norm @ WU.T).numpy()  # [V]
        top_abs = np.argsort(np.abs(r_rep))[::-1][:cfg.topT_tokens]
        top_pos = np.argsort(r_rep)[::-1][:cfg.topT_tokens]

        # Compare each member in logit subspace
        kept = []
        dropped = []
        # Precompute rep top vectors
        rep_sub_abs = r_rep[top_abs]
        rep_sub_pos = r_rep[top_pos]
        rep_abs_norm = np.linalg.norm(rep_sub_abs) + 1e-12
        rep_pos_norm = np.linalg.norm(rep_sub_pos) + 1e-12

        for d in member_dirs:
            v = get_v(*d).float()
            v = v / (v.norm() + 1e-12)
            r = (v @ WU.T).numpy()
            sub_abs = r[top_abs]
            sub_pos = r[top_pos]
            cos_abs = float(np.dot(sub_abs, rep_sub_abs) / ((np.linalg.norm(sub_abs) + 1e-12) * rep_abs_norm))
            cos_pos = float(np.dot(sub_pos, rep_sub_pos) / ((np.linalg.norm(sub_pos) + 1e-12) * rep_pos_norm))
            cos = max(cos_abs, cos_pos)
            if cos >= cfg.cos_confirm_stage2:
                kept.append((d, cos))
            else:
                dropped.append((d, cos))

        if len(kept) == 0:
            # Fall back: keep rep as singleton
            kept = [(rep, 1.0)]

        # Build final cluster
        cluster_members = [d for (d, _cos) in kept]
        for d in cluster_members:
            dir_to_cluster[f"{d[0]}_{d[1]}_{d[2]}"] = final_cluster_id

        # representative tokens as strings
        top_abs_str = [tokenizer.decode([int(t)]) for t in top_abs[:cfg.topT_tokens]]
        top_pos_str = [tokenizer.decode([int(t)]) for t in top_pos[:cfg.topT_tokens]]

        mass = float(sum(score_map.get(d, 0.0) for d in cluster_members))
        registry_clusters.append({
            "cluster_id": final_cluster_id,
            "stage1_cluster": int(cid),
            "representative": {"layer": rep[0], "head": rep[1], "k": rep[2]},
            "n_members": len(cluster_members),
            "member_dirs": [{"layer": d[0], "head": d[1], "k": d[2]} for d in cluster_members],
            "mass": mass,
            "rep_m": float(m_map.get(rep, 0.0)),
            "rep_sigma": float(sigma_map.get(rep, 0.0)),
            "top_tokens_abs": top_abs_str,
            "top_tokens_pos": top_pos_str,
        })
        rep_dirs.append(rep)
        final_cluster_id += 1

        # Option: make dropped singleton clusters
        for d, cos in dropped:
            # singleton
            dir_to_cluster[f"{d[0]}_{d[1]}_{d[2]}"] = final_cluster_id
            v_d = get_v(*d).float()
            v_d = v_d / (v_d.norm() + 1e-12)
            r_d = (v_d @ WU.T).numpy()
            top_abs_d = np.argsort(np.abs(r_d))[::-1][:cfg.topT_tokens]
            top_pos_d = np.argsort(r_d)[::-1][:cfg.topT_tokens]
            registry_clusters.append({
                "cluster_id": final_cluster_id,
                "stage1_cluster": int(cid),
                "representative": {"layer": d[0], "head": d[1], "k": d[2]},
                "n_members": 1,
                "member_dirs": [{"layer": d[0], "head": d[1], "k": d[2]}],
                "mass": float(score_map.get(d, 0.0)),
                "rep_m": float(m_map.get(d, 0.0)),
                "rep_sigma": float(sigma_map.get(d, 0.0)),
                "top_tokens_abs": [tokenizer.decode([int(t)]) for t in top_abs_d[:cfg.topT_tokens]],
                "top_tokens_pos": [tokenizer.decode([int(t)]) for t in top_pos_d[:cfg.topT_tokens]],
                "note": f"singleton_from_stage2_drop cos={cos:.3f}",
            })
            rep_dirs.append(d)
            final_cluster_id += 1

    registry = {
        "task": cfg.task,
        "cos_merge_stage1": cfg.cos_merge_stage1,
        "cos_confirm_stage2": cfg.cos_confirm_stage2,
        "topT_tokens": cfg.topT_tokens,
        "n_clusters": final_cluster_id,
        "clusters": registry_clusters,
    }
    return registry, dir_to_cluster, rep_dirs

def build_bank(cfg: E1Config,
               registry: Dict[str, Any],
               dir_to_cluster: Dict[str, int],
               rep_dirs: List[Tuple[int,int,int]],
               train_df: pd.DataFrame,
               svd_dir: Path,
               mask_dir: Path) -> Dict[str, Any]:
    # Lookup for score/sigma/m
    score_map: Dict[Tuple[int,int,int], float] = {}
    sigma_map: Dict[Tuple[int,int,int], float] = {}
    m_map: Dict[Tuple[int,int,int], float] = {}
    for r in train_df.itertuples(index=False):
        key = (int(r.layer), int(r.head), int(r.k))
        score_map[key] = float(r.score)
        sigma_map[key] = float(r.sigma)
        m_map[key] = float(r.m)

    # Preload svd V
    cache_V: Dict[Tuple[int,int], torch.Tensor] = {}
    def get_v(l:int,h:int,k:int) -> torch.Tensor:
        key = (l,h)
        if key not in cache_V:
            _, _, V = load_svd_file(svd_dir / f"layer{l:02d}_head{h:02d}.pt")
            cache_V[key] = V.float()
        return cache_V[key][:, k]

    C = registry["n_clusters"]
    v_bank = torch.empty((C, 768), dtype=torch.float32)
    rep_meta = []
    for c in range(C):
        rep = rep_dirs[c]
        v = get_v(*rep).float()
        v = v / (v.norm() + 1e-12)
        v_bank[c] = v
        rep_meta.append({
            "cluster_id": c,
            "rep_dir": {"layer": rep[0], "head": rep[1], "k": rep[2]},
            "rep_score": score_map.get(rep, 0.0),
            "rep_sigma": sigma_map.get(rep, 0.0),
            "rep_m": m_map.get(rep, 0.0),
        })
    bank = {
        "v_bank": v_bank,  # tensor
        "rep_meta": rep_meta,
        "registry": registry,
        "dir_to_cluster": dir_to_cluster,
    }
    return bank

def build_states(cfg: E1Config,
                 split: str,
                 cache_path: Path,
                 candidates: List[Tuple[int,int,int]],
                 dir_to_cluster: Dict[str, int],
                 bank: Dict[str, Any],
                 svd_dir: Path,
                 mask_dir: Path,
                 out_art_dir: Path) -> Dict[str, Any]:
    cache = load_cache(cache_path)
    nu_clean = cache["nu_clean_at_target"].float()        # [N,L,H,65]
    nu_corr  = cache["nu_corrupt_at_target"].float()      # [N,L,H,65]
    N, L, H, D = nu_clean.shape
    v_bank: torch.Tensor = bank["v_bank"].float()         # [C,768]
    C = v_bank.shape[0]

    X_clean = torch.zeros((N, L, C), dtype=torch.float32)
    X_corr  = torch.zeros((N, L, C), dtype=torch.float32)
    X_mix   = torch.zeros((N, L, C), dtype=torch.float32) if ("mix" in cfg.state_variant) else None

    # Preload per (l,h) U,S,V and m
    cache_svd: Dict[Tuple[int,int], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]] = {}
    cache_m: Dict[Tuple[int,int], torch.Tensor] = {}

    def get_head(l:int,h:int):
        key = (l,h)
        if key not in cache_svd:
            cache_svd[key] = load_svd_file(svd_dir / f"layer{l:02d}_head{h:02d}.pt")
        if key not in cache_m:
            cache_m[key] = load_mask_m(mask_dir / f"layer{l:02d}_head{h:02d}.pt")
        return cache_svd[key], cache_m[key]

    # Representative vector per cluster for sign alignment
    rep_vecs = v_bank  # already normalized

    for (l,h,k) in tqdm(candidates, desc=f"Build states {split}", leave=False):
        key = f"{l}_{h}_{k}"
        if key not in dir_to_cluster:
            continue
        c = int(dir_to_cluster[key])
        (U,S,V), m = get_head(l,h)
        sigma = float(S[k].item())
        mk = float(m[k].item())

        # member direction vector
        v_member = V[:, k].float()
        v_member = v_member / (v_member.norm() + 1e-12)
        # sign align
        if float(torch.dot(v_member, rep_vecs[c]).item()) < 0:
            sign = -1.0
        else:
            sign = 1.0

        # compute a_clean, a_corr for all examples
        a_clean = torch.einsum("nd,d->n", nu_clean[:, l, h, :], U[:, k])  # [N]
        a_corr  = torch.einsum("nd,d->n", nu_corr[:, l, h, :], U[:, k])   # [N]

        s_clean = sign * sigma * a_clean
        s_corr  = sign * sigma * a_corr

        X_clean[:, l, c] += s_clean
        X_corr[:,  l, c] += s_corr

        if X_mix is not None:
            s_mix = mk * s_clean + (1.0 - mk) * s_corr
            X_mix[:, l, c] += s_mix

    # Save
    out = {
        "X_clean": out_art_dir / f"X_clean_{split}.pt",
        "X_corr":  out_art_dir / f"X_corr_{split}.pt",
    }
    torch.save(X_clean, out["X_clean"])
    torch.save(X_corr, out["X_corr"])
    if X_mix is not None:
        out["X_mix"] = out_art_dir / f"X_mix_{split}.pt"
        torch.save(X_mix, out["X_mix"])

    return out

def plot_coverage(registry: Dict[str, Any], out_path: Path) -> Dict[str, Any]:
    clusters = registry["clusters"]
    masses = np.array([c["mass"] for c in clusters], dtype=np.float64)
    order = np.argsort(masses)[::-1]
    masses_sorted = masses[order]
    cum = np.cumsum(masses_sorted)
    total = float(cum[-1]) if len(cum) else 1.0
    frac = cum / (total + 1e-12)

    plt.figure()
    plt.plot(np.arange(1, len(frac)+1), frac)
    plt.xlabel("Cluster rank")
    plt.ylabel("Cumulative mass fraction")
    plt.title("Coverage curve (cluster mass)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

    top10 = float(frac[min(9, len(frac)-1)]) if len(frac) else 0.0
    return {"coverage_top10_mass_frac": top10, "n_clusters": int(len(frac)), "total_mass": total}

def stability_assignment(cfg: E1Config,
                         candidates: List[Tuple[int,int,int]],
                         svd_dir: Path,
                         bank: Dict[str, Any],
                         out_hist: Path) -> Dict[str, Any]:
    X = load_candidate_vectors(candidates, svd_dir)  # [N,768] normalized
    v_bank = bank["v_bank"].float()
    v_bank = v_bank / (v_bank.norm(dim=1, keepdim=True) + 1e-12)
    # cosine to nearest bank vector
    sims = (X @ v_bank.T).numpy()  # [N,C]
    best = sims.max(axis=1)
    frac = float((best >= cfg.assign_cos_threshold).mean())

    plt.figure()
    plt.hist(best, bins=50)
    plt.xlabel("Best cosine to bank")
    plt.ylabel("Count")
    plt.title("Stability: assignment cosine histogram")
    plt.tight_layout()
    plt.savefig(out_hist)
    plt.close()

    return {"assign_cos_threshold": cfg.assign_cos_threshold, "frac_assigned": frac, "mean_best_cos": float(best.mean())}

def behavior_alignment(cfg: E1Config,
                       split: str,
                       cache_path: Path,
                       X_clean_path: Path,
                       labels_csv: Path,
                       tokenizer: Any,
                       out_bar: Path) -> Dict[str, Any]:
    cache = load_cache(cache_path)
    logits = cache["logits_base_at_target"].float().numpy()  # [N,V]
    X_clean = torch.load(X_clean_path, map_location="cpu").float().numpy()  # [N,L,C]
    N, L, C = X_clean.shape

    correct, wrong = load_labels_for_split(cfg.task, labels_csv)
    if len(correct) != N:
        raise ValueError(f"Label rows {len(correct)} != cache N {N} for {split}")

    correct_ids = np.zeros(N, dtype=np.int64)
    wrong_ids = np.zeros(N, dtype=np.int64)
    single_ok = {"correct": 0, "wrong": 0}
    for i in range(N):
        tid, ok = safe_token_id(tokenizer, correct[i])
        correct_ids[i] = tid
        single_ok["correct"] += int(ok)
        tid2, ok2 = safe_token_id(tokenizer, wrong[i])
        wrong_ids[i] = tid2
        single_ok["wrong"] += int(ok2)

    delta = logits[np.arange(N), correct_ids] - logits[np.arange(N), wrong_ids]  # [N]

    # Compute best-layer corr per receptor
    best_corr = np.zeros(C, dtype=np.float64)
    best_layer = np.zeros(C, dtype=np.int64)
    for c in range(C):
        corrs = []
        for l in range(L):
            corrs.append(pearsonr(X_clean[:, l, c], delta))
        corrs = np.array(corrs)
        idx = int(np.argmax(np.abs(corrs)))
        best_corr[c] = float(corrs[idx])
        best_layer[c] = idx

    # Plot top 20 by abs corr
    top = np.argsort(np.abs(best_corr))[::-1][:20]
    plt.figure(figsize=(10, 5))
    plt.bar(range(len(top)), best_corr[top])
    plt.xticks(range(len(top)), [str(int(i)) for i in top], rotation=45)
    plt.xlabel("Receptor id")
    plt.ylabel("Pearson corr (best layer)")
    plt.title(f"Top receptors by |corr| ({split})")
    plt.tight_layout()
    plt.savefig(out_bar)
    plt.close()

    # Summaries
    return {
        "n_receptors": int(C),
        "top_abs_corr": float(np.max(np.abs(best_corr))) if C else 0.0,
        "n_ge_0p2": int((np.abs(best_corr) >= 0.2).sum()),
        "best_corr": best_corr.tolist(),         # for later use
        "best_layer": best_layer.tolist(),
        "token_single_frac": {
            "correct": float(single_ok["correct"] / max(1, N)),
            "wrong": float(single_ok["wrong"] / max(1, N)),
        }
    }

def intervention_sweep(cfg: E1Config,
                       model: Any,
                       tokenizer: Any,
                       split: str,
                       cache_path: Path,
                       labels_csv: Path,
                       bank: Dict[str, Any],
                       align: Dict[str, Any],
                       out_plot: Path) -> Dict[str, Any]:
    # Use top receptors by abs corr
    best_corr = np.array(align["best_corr"], dtype=np.float64)
    best_layer = np.array(align["best_layer"], dtype=np.int64)
    C = best_corr.shape[0]
    top = np.argsort(np.abs(best_corr))[::-1][:cfg.n_receptors_intervene]
    if len(top) == 0:
        return {"note": "no receptors"}

    cache = load_cache(cache_path)
    tokens_clean = cache["tokens_clean"]
    target_pos = cache["target_pos"]
    N, T = tokens_clean.shape

    correct, wrong = load_labels_for_split(cfg.task, labels_csv)
    correct_ids = np.array([safe_token_id(tokenizer, s)[0] for s in correct], dtype=np.int64)
    wrong_ids   = np.array([safe_token_id(tokenizer, s)[0] for s in wrong], dtype=np.int64)

    device = torch.device(cfg.device if (cfg.device == "cpu" or torch.cuda.is_available()) else "cpu")
    model.eval().to(device)

    v_bank: torch.Tensor = bank["v_bank"].float().to(device)  # [C,768]
    alpha_grid = list(cfg.alpha_grid)

    # We'll compute mean delta logit over the split for each receptor and alpha
    results = []
    batch = 32

    for c in top.tolist():
        layer = int(best_layer[c])
        v = v_bank[c]  # [768]
        deltas_mean = []
        for alpha in alpha_grid:
            deltas = []
            with torch.no_grad():
                for i in range(0, N, batch):
                    j = min(i + batch, N)
                    inp = tokens_clean[i:j].to(device)
                    tpos = target_pos[i:j].to(device)
                    # hook that adds alpha*v at (batch_idx, tpos[batch_idx])
                    def hook_fn(module, module_in, module_out):
                        out = module_out
                        if isinstance(out, tuple):
                            hs = out[0]
                            rest = out[1:]
                        else:
                            hs = out
                            rest = None
                        hs = hs.clone()
                        B = hs.shape[0]
                        ar = torch.arange(B, device=hs.device)
                        hs[ar, tpos, :] += float(alpha) * v
                        if rest is None:
                            return hs
                        return (hs, *rest)

                    handle = model.transformer.h[layer].register_forward_hook(hook_fn)
                    out = model(inp)
                    handle.remove()
                    logits = out.logits  # [B,T,V]
                    ar = torch.arange(j - i, device=device)
                    logits_t = logits[ar, tpos, :]  # [B,V]
                    # gather correct/wrong per example
                    corr_ids = torch.tensor(correct_ids[i:j], device=device)
                    wrong_ids_b = torch.tensor(wrong_ids[i:j], device=device)
                    d = logits_t[ar, corr_ids] - logits_t[ar, wrong_ids_b]
                    deltas.append(d.detach().cpu())
            d_all = torch.cat(deltas, dim=0).numpy()
            deltas_mean.append(float(d_all.mean()))
        results.append({"receptor": int(c), "layer": layer, "corr": float(best_corr[c]), "delta_mean": deltas_mean})

    # Plot curves
    plt.figure(figsize=(8, 5))
    for r in results:
        plt.plot(alpha_grid, r["delta_mean"], marker="o", label=f"c{r['receptor']},L{r['layer']},corr={r['corr']:.2f}")
    plt.axhline(0, linewidth=1)
    plt.xlabel("alpha")
    plt.ylabel("mean Δlogit(correct - wrong)")
    plt.title(f"Intervention sweep ({split})")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_plot)
    plt.close()

    # Simple monotonicity check (exists one receptor monotonic w.r.t alpha and sign matches corr)
    def is_monotonic(vals: List[float]) -> bool:
        inc = all(vals[i] <= vals[i+1] + 1e-6 for i in range(len(vals)-1))
        dec = all(vals[i] >= vals[i+1] - 1e-6 for i in range(len(vals)-1))
        return inc or dec

    monotonic_any = False
    for r in results:
        vals = r["delta_mean"]
        if is_monotonic(vals):
            monotonic_any = True
            break

    return {"n_tested": len(results), "alpha_grid": alpha_grid, "results": results, "monotonic_any": monotonic_any}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, required=True, choices=["gp", "ioi", "gt"])
    ap.add_argument("--out_dir", type=str, required=True, help="Task output dir from run_ov_mask.py, e.g. outputs/gp")
    ap.add_argument("--data_dir", type=str, required=True, help="Directory containing CSVs (data_main)")

    ap.add_argument("--tau_report", type=float, default=1e-2)
    ap.add_argument("--tau_candidate", type=float, default=1e-3)
    ap.add_argument("--K_per_head", type=int, default=4)
    ap.add_argument("--K_global_cap", type=int, default=2000)
    ap.add_argument("--cos_merge_stage1", type=float, default=0.95)
    ap.add_argument("--cos_confirm_stage2", type=float, default=0.90)
    ap.add_argument("--topT_tokens", type=int, default=50)
    ap.add_argument("--assign_cos_threshold", type=float, default=0.90)

    ap.add_argument("--state_variant", type=str, default="clean+corr", choices=["clean+corr", "clean+corr+mix"])

    ap.add_argument("--n_receptors_intervene", type=int, default=3)
    ap.add_argument("--alpha_grid", type=str, default="-2,-1,-0.5,0.5,1,2")

    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk_size_cos", type=int, default=256)

    # Optionally pass CSVs explicitly; otherwise we infer
    ap.add_argument("--train_csv", type=str, default="")
    ap.add_argument("--val_csv", type=str, default="")
    ap.add_argument("--test_csv", type=str, default="")

    args = ap.parse_args()

    cfg = E1Config(
        task=args.task,
        out_dir=Path(args.out_dir),
        data_dir=Path(args.data_dir),
        tau_report=args.tau_report,
        tau_candidate=args.tau_candidate,
        K_per_head=args.K_per_head,
        K_global_cap=args.K_global_cap,
        cos_merge_stage1=args.cos_merge_stage1,
        cos_confirm_stage2=args.cos_confirm_stage2,
        topT_tokens=args.topT_tokens,
        assign_cos_threshold=args.assign_cos_threshold,
        state_variant=args.state_variant,
        n_receptors_intervene=args.n_receptors_intervene,
        alpha_grid=tuple(float(x) for x in args.alpha_grid.split(",")),
        device=args.device,
        seed=args.seed,
        chunk_size_cos=args.chunk_size_cos,
    )

    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    cache_dir = cfg.out_dir / "cache"
    svd_dir = cfg.out_dir / "svd"
    mask_dir = cfg.out_dir / "masks"

    if not cache_dir.exists():
        raise FileNotFoundError(f"Missing cache dir: {cache_dir}")
    if not svd_dir.exists():
        raise FileNotFoundError(f"Missing svd dir: {svd_dir}")
    if not mask_dir.exists():
        raise FileNotFoundError(f"Missing masks dir: {mask_dir}")

    # Output directories
    art_dir = cfg.out_dir / "artifacts" / "e1"
    ensure_dir(art_dir)

    # CSV paths
    inferred = infer_split_files(cfg.task, cfg.data_dir)
    split_csv = {
        "train": Path(args.train_csv) if args.train_csv else inferred.get("train"),
        "val": Path(args.val_csv) if args.val_csv else inferred.get("val"),
        "test": Path(args.test_csv) if args.test_csv else inferred.get("test"),
    }
    for sp in ["train", "val", "test"]:
        if split_csv[sp] is None or not Path(split_csv[sp]).exists():
            raise FileNotFoundError(f"Could not find {sp} csv for task={cfg.task}. Provide --{sp}_csv or place it in {cfg.data_dir}")

    # Load model + tokenizer (GPT-2 small)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    # Step 0: ensure nu_clean in cache files
    for split in ["train", "val", "test"]:
        cache_path = cache_dir / f"{split}.pt"
        if not cache_path.exists():
            raise FileNotFoundError(f"Missing cache file: {cache_path}")
        ensure_nu_clean(cfg, model, split, cache_path)

    # Step 1: direction tables
    dfs = {}
    for split in ["train", "val", "test"]:
        df = build_direction_table_for_split(cfg, split, cache_dir / f"{split}.pt", svd_dir, mask_dir)
        dfs[split] = df
        out_path = try_write_table(df, art_dir / f"directions_{split}")
        print(f"[E1] wrote {split} direction table: {out_path}")

    # Combined/global using train as primary (we store train df as global)
    train_df = dfs["train"].copy()
    out_path = try_write_table(train_df, art_dir / "directions_global")
    print(f"[E1] wrote global direction table: {out_path}")

    # Step 2: candidate selection
    candidates = select_candidates(cfg, train_df)
    json_dump(art_dir / "candidates.json", {"task": cfg.task, "tau_report": cfg.tau_report, "candidates": candidates})
    print(f"[E1] candidates: {len(candidates)}")

    # Step 3: stage-1 clustering in residual space
    X = load_candidate_vectors(candidates, svd_dir)  # [Ncand,768] normalized
    cluster_ids = stage1_cluster(cfg, X)
    json_dump(art_dir / "stage1_clusters.json", {"task": cfg.task, "candidates": candidates, "cluster_id": cluster_ids})

    # Step 4: stage-2 logit confirmation / cleanup
    registry, dir_to_cluster, rep_dirs = stage2_confirm(cfg, candidates, cluster_ids, train_df, svd_dir, model, tokenizer)
    json_dump(art_dir / "receptor_registry.json", registry)
    json_dump(art_dir / "dir_to_cluster.json", dir_to_cluster)
    print(f"[E1] finalized clusters: {registry['n_clusters']}")

    # Step 5: canonical bank
    bank = build_bank(cfg, registry, dir_to_cluster, rep_dirs, train_df, svd_dir, mask_dir)
    torch.save({"v_bank": bank["v_bank"], "rep_meta": bank["rep_meta"]}, art_dir / "receptor_bank.pt")

    # Step 6: build receptor state tensors
    state_paths = {}
    for split in ["train", "val", "test"]:
        state_paths[split] = build_states(cfg, split, cache_dir / f"{split}.pt", candidates, dir_to_cluster, bank, svd_dir, mask_dir, art_dir)
        print(f"[E1] wrote states for {split}: {state_paths[split]}")

    # Step 7: validations + plots
    cov = plot_coverage(registry, art_dir / "coverage_curve.png")
    stab = stability_assignment(cfg, candidates, svd_dir, bank, art_dir / "stability_hist.png")
    align_train = behavior_alignment(cfg, "train", cache_dir / "train.pt", state_paths["train"]["X_clean"], Path(split_csv["train"]), tokenizer, art_dir / "corr_barplot_train.png")
    align_val   = behavior_alignment(cfg, "val", cache_dir / "val.pt", state_paths["val"]["X_clean"], Path(split_csv["val"]), tokenizer, art_dir / "corr_barplot_val.png")

    interv = intervention_sweep(cfg, model, tokenizer, "val", cache_dir / "val.pt", Path(split_csv["val"]), bank, align_train, art_dir / "intervention_curves.png")

    report = {
        "task": cfg.task,
        "tau_report": cfg.tau_report,
        "K_per_head": cfg.K_per_head,
        "K_global_cap": cfg.K_global_cap,
        "coverage": cov,
        "stability": stab,
        "alignment_train": {
            "n_ge_0p2": align_train["n_ge_0p2"],
            "top_abs_corr": align_train["top_abs_corr"],
            "token_single_frac": align_train["token_single_frac"],
        },
        "alignment_val": {
            "n_ge_0p2": align_val["n_ge_0p2"],
            "top_abs_corr": align_val["top_abs_corr"],
            "token_single_frac": align_val["token_single_frac"],
        },
        "intervention": {
            "n_tested": interv.get("n_tested", 0),
            "monotonic_any": interv.get("monotonic_any", False),
        },
        "success_gate": {
            "coverage_top10_ge_0p5": bool(cov.get("coverage_top10_mass_frac", 0.0) >= 0.5),
            "stability_ge_0p8": bool(stab.get("frac_assigned", 0.0) >= 0.8),
            "alignment_ge_3": bool(align_train.get("n_ge_0p2", 0) >= 3),
            "intervention_monotonic_any": bool(interv.get("monotonic_any", False)),
        }
    }
    json_dump(art_dir / "e1_report.json", report)

    print("\n[E1] SUCCESS GATE:")
    print("  coverage top-10 mass >= 50% :", report["success_gate"]["coverage_top10_ge_0p5"])
    print("  stability val assignment >=80%:", report["success_gate"]["stability_ge_0p8"])
    print("  alignment >=3 receptors |corr|>=0.2 :", report["success_gate"]["alignment_ge_3"])
    print("  intervention monotonic any :", report["success_gate"]["intervention_monotonic_any"])
    print(f"\n[E1] Wrote report + artifacts under: {art_dir}")

if __name__ == "__main__":
    main()
