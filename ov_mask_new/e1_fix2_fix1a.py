
"""
E1 Fix2 + Fix1a postprocess module

Purpose:
- Recluster OV-mask directions using token-level logit signatures (Fix 2)
- Use mass_fix1a = mask * sigma^2 as the clustering mass / coverage metric (Fix 1a)
- Rebuild receptor bank + per-example receptor state tensors X_clean/X_corr (optionally X_*_masked)
- Write a compact report so you can paste metrics back into your planning window.

Expected pre-req:
- You have already run run_ov_mask.py for a given task, producing:
  outputs/<task>/{cache,svd,masks}/...
- You have either run e1_postprocess.py once (recommended), OR you at least have cache files that contain:
  nu_corrupt_at_target and (optionally) nu_clean_at_target (this script can add nu_clean if missing).

Run (Kaggle):
  python e1_fix2_fix1a.py --task gp --out_dir outputs/gp --data_dir data_main

Outputs:
  outputs/<task>/artifacts/e1_fix2_fix1a/...
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
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

# Transformers is already a dependency in your repo (run_ov_mask.py uses it)
from transformers import AutoTokenizer, AutoModelForCausalLM


# -----------------------------
# Utilities
# -----------------------------

def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_device(x: torch.Tensor, device: torch.device) -> torch.Tensor:
    return x.to(device=device, non_blocking=True)


def _safe_norm(x: np.ndarray, eps: float = 1e-12) -> float:
    return float(np.sqrt(np.sum(x * x)) + eps)


def _normalize_vec(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = _safe_norm(v, eps)
    return (v / n).astype(np.float32)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _read_json(p: Path) -> Any:
    return json.loads(p.read_text())


def _write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, indent=2))


# -----------------------------
# Cache / nu_clean extraction
# -----------------------------

@torch.no_grad()
def compute_nu_clean_at_target(
    model: AutoModelForCausalLM,
    tokens_clean: torch.Tensor,          # [N,T]
    target_pos: torch.Tensor,            # [N]
    device: torch.device,
) -> torch.Tensor:
    """
    Matches run_ov_mask.py's notion of nu: attention value vector at target position, augmented with 1.
    Returns: nu_clean_at_target [N, L, H, 65]
    """
    model.eval()
    tokens_clean = _to_device(tokens_clean, device)
    target_pos = _to_device(target_pos, device)

    # GPT-2: n_layer=12, n_head=12, head_dim=64
    cfg = model.config
    L = int(cfg.n_layer)
    H = int(cfg.n_head)
    head_dim = int(cfg.n_embd // cfg.n_head)
    d_aug = head_dim + 1

    # We'll capture attention value vectors via hooks at each layer's attn
    # In HF GPT-2, block.attn.c_attn produces qkv, but grabbing "value" vectors robustly is messy.
    # Your existing code already computes nu_corrupt_at_target in build_cache; we mirror its logic
    # by calling the model with output_attentions=True and reconstructing V via internal modules.
    #
    # To keep this stable: we use forward hooks on each block.attn to capture the value vectors
    # AFTER projection and split into heads, before attention weighting.
    #
    # This is consistent with your run_ov_mask implementation if it uses the same hook point.
    # If your run_ov_mask uses a different hook point, nu_clean and nu_corrupt would mismatch.
    # In that case, prefer running e1_postprocess.py (Step0) which already matched your codebase.

    # We'll attempt to import and reuse the exact hook logic from run_ov_mask if present.
    # Otherwise, fall back to a conservative approach that should still work given your repo.
    try:
        import run_ov_mask  # type: ignore
        if hasattr(run_ov_mask, "capture_nu_at_target"):
            return run_ov_mask.capture_nu_at_target(model, tokens_clean, target_pos, device=device, which="clean")  # type: ignore
    except Exception:
        pass

    # Fallback: use model outputs with output_hidden_states and approximate nu using head outputs.
    # NOTE: This fallback is less "paper-faithful" than using your repo's capture function.
    # It is only used if we cannot import/call run_ov_mask.capture_nu_at_target.
    outputs = model(tokens_clean, output_hidden_states=True, use_cache=False)
    hidden_states = outputs.hidden_states  # tuple len L+1, each [N,T,d_model]
    # We'll approximate "value" vectors by taking the per-head input to O projection in each layer.
    # HF GPT2 stores attn output before resid add as block.attn(hidden_states)[0] but not values.
    # So we just return zeros to avoid silent mismatch.
    # You should NOT rely on this fallback; run e1_postprocess.py first.
    raise RuntimeError(
        "Could not compute nu_clean via repo hook. Please run e1_postprocess.py first "
        "so nu_clean_at_target is added to caches using the same capture point as training."
    )


def ensure_nu_clean_in_cache(
    cache_path: Path,
    model: AutoModelForCausalLM,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    cache = torch.load(cache_path, map_location="cpu")
    if "nu_clean_at_target" in cache:
        return cache
    if "tokens_clean" not in cache or "target_pos" not in cache:
        raise KeyError(f"{cache_path} missing tokens_clean/target_pos; cannot compute nu_clean.")
    nu_clean = compute_nu_clean_at_target(model, cache["tokens_clean"], cache["target_pos"], device)
    cache["nu_clean_at_target"] = nu_clean.cpu()
    torch.save(cache, cache_path)
    return cache


# -----------------------------
# Direction table + signatures
# -----------------------------

@dataclass
class DirectionRow:
    dir_id: str
    l: int
    h: int
    k: int
    sigma: float
    mask: float
    mass_fix1a: float
    active: bool


def load_direction_table(
    out_dir: Path,
    tau_active: float,
    use_candidates_json: bool = True,
) -> Tuple[pd.DataFrame, torch.Tensor, List[str]]:
    """
    Builds a direction table from svd + masks.

    Returns:
      df: one row per direction (all in-scope; active flagged)
      v_mat: [N_dir, 768] float32 unit vectors (same row order)
      dir_ids: list of dir_id strings aligned with v_mat / df
    """
    svd_dir = out_dir / "svd"
    if not svd_dir.exists():
        svd_dir = out_dir / "svd_dir"
    if not svd_dir.exists():
        raise FileNotFoundError(
            f"Missing SVD directory. Looked for {out_dir/'svd'} and {out_dir/'svd_dir'}."
        )

    # svd_dir = out_dir / "svd_dir"
    # if not svd_dir.exists():
    #     alt = out_dir / "svd"
    #     if alt.exists():
    #         svd_dir = alt

    # if not svd_dir.exists():
    #     raise FileNotFoundError(
    #         f"Missing SVD directory. Looked for {out_dir/'svd_dir'} and {out_dir/'svd'}. "
    #         "Did you run run_ov_mask.py with the same --out_dir and --task?"
    #     )
    # masks_dir = out_dir / "masks"
    # if not svd_dir.exists():
    #     raise FileNotFoundError(f"Missing {svd_dir}. Did you run run_ov_mask.py?")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Missing {masks_dir}. Did you run run_ov_mask.py?")

    # Optional candidate list from the earlier E1 run
    cand_ids: Optional[set] = None
    cand_path = out_dir / "artifacts" / "e1" / "candidates.json"
    if use_candidates_json and cand_path.exists():
        cand_obj = _read_json(cand_path)

        # e1_postprocess.py writes {"task":..., "tau_report":..., "candidates":[(l,h,k), ...]}
        if isinstance(cand_obj, dict) and "candidates" in cand_obj:
            cand_obj = cand_obj["candidates"]

        cand_ids = set()
        if isinstance(cand_obj, list):
            for x in cand_obj:
                # allow either strings OR tuple/list OR dicts
                if isinstance(x, str):
                    cand_ids.add(x)
                elif isinstance(x, (list, tuple)) and len(x) == 3:
                    l, h, k = int(x[0]), int(x[1]), int(x[2])
                    cand_ids.add(f"l{l}_h{h}_k{k}")
                elif isinstance(x, dict) and all(k in x for k in ("layer", "head", "k")):
                    cand_ids.add(f"l{int(x['layer'])}_h{int(x['head'])}_k{int(x['k'])}")

        # If parsing failed, fall back to None (meaning: don't filter)
        if len(cand_ids) == 0:
            cand_ids = None


    rows: List[DirectionRow] = []
    v_list: List[np.ndarray] = []
    dir_ids: List[str] = []

    # Determine layers/heads from existing files
    svd_files = sorted(svd_dir.glob("layer*_head*.pt"))
    if len(svd_files) == 0:
        raise FileNotFoundError(f"No SVD files found in {svd_dir}")

    for p in svd_files:
        m = re.match(r"layer(\d+)_head(\d+)\.pt", p.name)
        if m is None:
            continue
        l = int(m.group(1))
        h = int(m.group(2))

        svd = torch.load(p, map_location="cpu")
        V_keep = svd["V_keep"].float()          # [d_model, r]
        S_keep = svd["S_keep"].float()          # [r]
        r = int(S_keep.numel())

        mask_p = masks_dir / f"layer{l:02d}_head{h:02d}.pt"
        if not mask_p.exists():
            raise FileNotFoundError(f"Missing mask file: {mask_p}")
        mobj = torch.load(mask_p, map_location="cpu")
        if "m" in mobj:
            m_vec = mobj["m"].float().view(-1)
        elif "mask_logits" in mobj:
            m_vec = torch.sigmoid(mobj["mask_logits"].float().view(-1))
        else:
            raise KeyError(f"Mask file {mask_p} missing keys; expected 'm' or 'mask_logits'.")
        if m_vec.numel() != r:
            raise ValueError(f"Mask length mismatch in {mask_p}: got {m_vec.numel()} expected {r}")

        for k in range(r):
            dir_id = f"l{l}_h{h}_k{k}"
            if cand_ids is not None and dir_id not in cand_ids:
                continue

            sigma = float(S_keep[k].item())
            mask = float(m_vec[k].item())
            mass = float(mask * (sigma ** 2))
            active = bool(mask >= tau_active)

            v = V_keep[:, k].numpy().astype(np.float32)  # [768]
            v = _normalize_vec(v)
            rows.append(DirectionRow(dir_id, l, h, k, sigma, mask, mass, active))
            v_list.append(v)
            dir_ids.append(dir_id)

    df = pd.DataFrame([r.__dict__ for r in rows])
    v_mat = torch.tensor(np.stack(v_list, axis=0), dtype=torch.float32)  # [N,768]
    return df, v_mat, dir_ids


def build_direction_signatures(
    v_mat: torch.Tensor,                # [N,768] on CPU
    dir_ids: List[str],
    tokenizer: AutoTokenizer,
    model: AutoModelForCausalLM,
    device: torch.device,
    T_pos: int,
    T_neg: int,
    use_abs_weight_norm: bool,
) -> pd.DataFrame:
    """
    For each direction v, compute s = W_U @ v (logit signature), then keep top pos/neg tokens.
    Returns a DataFrame aligned with dir_ids/v_mat.
    """
    W_U = model.lm_head.weight.detach()  # [V, d_model]
    W_U = _to_device(W_U, device)

    v_mat_dev = _to_device(v_mat, device)
    rows = []
    V = W_U.shape[0]

    for i in tqdm(range(v_mat.shape[0]), desc="Signatures", leave=False):
        v = v_mat_dev[i]  # [768]
        # s: [V]
        s = torch.mv(W_U, v)  # W_U @ v

        # top positive
        pos_vals, pos_idx = torch.topk(s, k=T_pos, largest=True)
        # top negative
        neg_vals, neg_idx = torch.topk(-s, k=T_neg, largest=True)  # values are -s, positive
        neg_vals = neg_vals  # magnitude

        pos_idx_np = pos_idx.detach().cpu().numpy().astype(int).tolist()
        neg_idx_np = neg_idx.detach().cpu().numpy().astype(int).tolist()
        pos_w = pos_vals.detach().cpu().numpy().astype(np.float32)
        neg_w = neg_vals.detach().cpu().numpy().astype(np.float32)

        # Convert to positive weights
        pos_w = np.maximum(pos_w, 0.0)
        neg_w = np.maximum(neg_w, 0.0)

        if use_abs_weight_norm:
            ps = float(pos_w.sum())
            ns = float(neg_w.sum())
            if ps > 0:
                pos_w = pos_w / ps
            if ns > 0:
                neg_w = neg_w / ns

        # Token strings (best-effort; GPT-2 BPE sometimes yields weird prefix spaces)
        pos_tok = [tokenizer.decode([tid]) for tid in pos_idx_np]
        neg_tok = [tokenizer.decode([tid]) for tid in neg_idx_np]

        rows.append({
            "dir_id": dir_ids[i],
            "sig_pos_ids": pos_idx_np,
            "sig_pos_w": pos_w.tolist(),
            "sig_neg_ids": neg_idx_np,
            "sig_neg_w": neg_w.tolist(),
            "sig_pos_tok": pos_tok,
            "sig_neg_tok": neg_tok,
        })
    return pd.DataFrame(rows)


# -----------------------------
# Signed weighted Jaccard similarity
# -----------------------------

SigMap = Dict[Tuple[str, int], float]  # keys: ("P", token_id) or ("N", token_id)


def sig_to_map(pos_ids: List[int], pos_w: List[float], neg_ids: List[int], neg_w: List[float]) -> SigMap:
    d: SigMap = {}
    for tid, w in zip(pos_ids, pos_w):
        if w > 0:
            d[("P", int(tid))] = float(w)
    for tid, w in zip(neg_ids, neg_w):
        if w > 0:
            d[("N", int(tid))] = float(w)
    return d


def flip_sig_map(d: SigMap) -> SigMap:
    out: SigMap = {}
    for (s, tid), w in d.items():
        out[("N", tid) if s == "P" else ("P", tid)] = float(w)
    return out


def weighted_jaccard(a: SigMap, b: SigMap) -> Tuple[float, int]:
    """
    Returns (J_w, n_common_keys)
    J_w = sum_{t in A∩B} min(w_a, w_b) / sum_{t in A∪B} max(w_a, w_b)
    """
    if len(a) == 0 or len(b) == 0:
        return 0.0, 0
    # Iterate over smaller
    if len(a) > len(b):
        a, b = b, a
    common = 0
    num = 0.0
    den = 0.0
    # union keys
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        wa = a.get(k, 0.0)
        wb = b.get(k, 0.0)
        if wa > 0 and wb > 0:
            common += 1
            num += min(wa, wb)
        den += max(wa, wb)
    if den <= 0:
        return 0.0, common
    return float(num / den), common


def best_sig_similarity(a: SigMap, b: SigMap) -> Tuple[float, int, bool]:
    """
    Returns (best_sim, n_common, needs_flip)
    where needs_flip indicates b should be flipped for best match.
    """
    sim_same, common_same = weighted_jaccard(a, b)
    b_flip = flip_sig_map(b)
    sim_flip, common_flip = weighted_jaccard(a, b_flip)
    if sim_flip > sim_same:
        return sim_flip, common_flip, True
    return sim_same, common_same, False


def renorm_sig_map(d: SigMap) -> SigMap:
    """
    Renormalize weights separately within P and N so each sign sums to 1 (if non-empty).
    """
    sum_p = 0.0
    sum_n = 0.0
    for (s, _), w in d.items():
        if s == "P":
            sum_p += w
        else:
            sum_n += w
    out: SigMap = {}
    for (s, tid), w in d.items():
        if s == "P":
            out[(s, tid)] = float(w / sum_p) if sum_p > 0 else float(w)
        else:
            out[(s, tid)] = float(w / sum_n) if sum_n > 0 else float(w)
    return out


# -----------------------------
# Greedy clustering (Fix 2)
# -----------------------------

@dataclass
class Cluster:
    members: List[int]          # indices into active arrays
    mass_total: float
    sig_sum: SigMap             # mass-weighted sum of signature weights (not renormalized)
    sig_rep: SigMap             # representative signature map (renormalized)
    v_sum: np.ndarray           # mass-weighted sum of oriented v_dir vectors
    v_rep: np.ndarray           # unit-normalized v_sum
    rep_dir_idx: int            # index of best-mass member (for metadata)


def cluster_active_directions(
    active_idx: np.ndarray,            # indices in full direction arrays
    masses: np.ndarray,                # [N_active] mass_fix1a
    sig_maps: List[SigMap],            # [N_active]
    v_active: np.ndarray,              # [N_active,768] (unit vectors)
    sim_threshold: float,
    min_common_signed_tokens: int,
) -> Tuple[List[Cluster], np.ndarray, np.ndarray]:
    """
    Mass-sorted greedy clustering using signature similarity.

    Returns:
      clusters: list of clusters
      dir_to_cluster: [N_active] cluster_id
      dir_flip: [N_active] bool whether direction was flipped for alignment
    """
    N = int(active_idx.shape[0])
    order = np.argsort(-masses)  # high mass first

    clusters: List[Cluster] = []
    dir_to_cluster = -np.ones((N,), dtype=np.int64)
    dir_flip = np.zeros((N,), dtype=np.bool_)

    # Work copies that we mutate when flipping directions
    sig_maps_work = list(sig_maps)
    v_work = v_active.copy()

    for oi in tqdm(order, desc="Clustering", leave=False):
        m_i = float(masses[oi])
        sig_i = sig_maps_work[oi]

        # If signature is empty, keep it as a singleton
        if len(sig_i) == 0:
            v_sum = v_work[oi] * m_i
            c = Cluster(
                members=[oi],
                mass_total=m_i,
                sig_sum={},  # empty
                sig_rep={},
                v_sum=v_sum.astype(np.float32),
                v_rep=_normalize_vec(v_sum),
                rep_dir_idx=oi,
            )
            clusters.append(c)
            dir_to_cluster[oi] = len(clusters) - 1
            continue

        best_c = -1
        best_sim = -1.0
        best_needs_flip = False

        for cid, c in enumerate(clusters):
            sim, common, needs_flip = best_sig_similarity(sig_i, c.sig_rep)
            if common < min_common_signed_tokens:
                continue
            if sim > best_sim:
                best_sim = sim
                best_c = cid
                best_needs_flip = needs_flip

        if best_c >= 0 and best_sim >= sim_threshold:
            # Assign to existing cluster
            if best_needs_flip:
                dir_flip[oi] = True
                v_work[oi] = -v_work[oi]
                sig_maps_work[oi] = flip_sig_map(sig_maps_work[oi])
                sig_i = sig_maps_work[oi]

            c = clusters[best_c]
            old_mass = float(c.mass_total)
            new_mass = old_mass + m_i
            c.mass_total = new_mass

            # Update sig_sum (mass-weighted)
            for k, w in sig_i.items():
                c.sig_sum[k] = float(c.sig_sum.get(k, 0.0) + w * m_i)

            # Recompute sig_rep as (sig_sum / mass_total) then renorm within sign
            sig_avg = {k: float(v / new_mass) for k, v in c.sig_sum.items()}
            c.sig_rep = renorm_sig_map(sig_avg)

            # Update v_sum / v_rep
            c.v_sum = (c.v_sum + v_work[oi] * m_i).astype(np.float32)
            c.v_rep = _normalize_vec(c.v_sum)

            c.members.append(oi)

            # Update representative member (highest mass)
            if m_i > float(masses[c.rep_dir_idx]):
                c.rep_dir_idx = oi

            dir_to_cluster[oi] = best_c
        else:
            # New cluster
            sig_sum = {k: float(w * m_i) for k, w in sig_i.items()}
            sig_avg = {k: float(v / m_i) for k, v in sig_sum.items()} if m_i > 0 else dict(sig_i)
            v_sum = v_work[oi] * m_i
            c = Cluster(
                members=[oi],
                mass_total=m_i,
                sig_sum=sig_sum,
                sig_rep=renorm_sig_map(sig_avg),
                v_sum=v_sum.astype(np.float32),
                v_rep=_normalize_vec(v_sum),
                rep_dir_idx=oi,
            )
            clusters.append(c)
            dir_to_cluster[oi] = len(clusters) - 1

    return clusters, dir_to_cluster, dir_flip


# -----------------------------
# Rebuild bank + X states
# -----------------------------

# def _load_cache(out_dir: Path, split: str) -> Dict[str, torch.Tensor]:
#     p = out_dir / "cache" / f"cache_{split}.pt"
#     if not p.exists():
#         raise FileNotFoundError(f"Missing cache file: {p}")
#     return torch.load(p, map_location="cpu")

def _load_cache(out_dir: Path, split: str) -> Dict[str, Any]:
    cache_dir = out_dir / "cache"
    candidates = [
        cache_dir / f"{split}.pt",           # correct (E1 / run_ov_mask convention)
        cache_dir / f"cache_{split}.pt",     # legacy / your earlier assumption
        cache_dir / f"cache_{split}.pt",     # harmless duplicate, keep simple
        cache_dir / f"cache_{split}.pt",     # (you can delete extras later)
    ]
    for p in candidates:
        if p.exists():
            return torch.load(p, map_location="cpu")

    raise FileNotFoundError(
        "Missing cache file. Looked for:\n  " + "\n  ".join(str(p) for p in candidates)
    )



def rebuild_X_tensors(
    out_dir: Path,
    splits: List[str],
    df_all: pd.DataFrame,
    v_mat: torch.Tensor,
    dir_ids: List[str],
    active_mask: np.ndarray,          # [N_dir] bool
    dir_to_cluster_active: np.ndarray,# [N_active] cluster ids
    dir_flip_active: np.ndarray,      # [N_active] bool
    clusters: List[Cluster],
    store_masked: bool,
) -> Dict[str, Dict[str, Path]]:
    """
    Builds X_clean/X_corr with shape [N_examples, L, C_new] for each split.
    Accumulates sigma * (nu_* @ U_keep[:,k]) into the cluster channel.
    """
    # Map active local index -> global direction index
    active_global_idx = np.where(active_mask)[0]
    assert active_global_idx.shape[0] == dir_to_cluster_active.shape[0]

    # Build a lookup from global direction index -> (cluster_id, flip)
    global_to_cluster = {}
    global_to_flip = {}
    for j, gi in enumerate(active_global_idx.tolist()):
        global_to_cluster[int(gi)] = int(dir_to_cluster_active[j])
        global_to_flip[int(gi)] = bool(dir_flip_active[j])

    # svd_dir = out_dir / "svd_dir"

    svd_dir = out_dir / "svd"
    if not svd_dir.exists():
        svd_dir = out_dir / "svd_dir"
    if not svd_dir.exists():
        raise FileNotFoundError(
            f"Missing SVD directory. Looked for {out_dir/'svd'} and {out_dir/'svd_dir'}."
        )

    # svd_dir = out_dir / "svd_dir"
    # if not svd_dir.exists():
    #     alt = out_dir / "svd"
    #     if alt.exists():
    #         svd_dir = alt

    # if not svd_dir.exists():
    #     raise FileNotFoundError(
    #         f"Missing SVD directory. Looked for {out_dir/'svd_dir'} and {out_dir/'svd'}. "
    #         "Did you run run_ov_mask.py with the same --out_dir and --task?"
    #     )

    outputs: Dict[str, Dict[str, Path]] = {}

    for split in splits:
        cache = _load_cache(out_dir, split)
        if "nu_clean_at_target" not in cache:
            raise KeyError(
                f"Cache for {split} missing nu_clean_at_target. "
                f"Run e1_postprocess.py first (Step0) so it is computed with the correct hook."
            )
        nu_clean = cache["nu_clean_at_target"].float()     # [N,L,H,65]
        nu_corr = cache["nu_corrupt_at_target"].float()    # [N,L,H,65]
        N, L, H, d_aug = nu_clean.shape
        C = len(clusters)

        X_clean = torch.zeros((N, L, C), dtype=torch.float32)
        X_corr = torch.zeros((N, L, C), dtype=torch.float32)
        X_clean_masked = torch.zeros((N, L, C), dtype=torch.float32) if store_masked else None
        X_corr_masked = torch.zeros((N, L, C), dtype=torch.float32) if store_masked else None

        # Iterate over all directions (global), but only active are assigned to clusters
        for gi in tqdm(range(df_all.shape[0]), desc=f"Build X {split}", leave=False):
            if not bool(active_mask[gi]):
                continue
            row = df_all.iloc[gi]
            l = int(row["l"]); h = int(row["h"]); k = int(row["k"])
            sigma = float(row["sigma"])
            mask_val = float(row["mask"])
            c = global_to_cluster[gi]
            flip = global_to_flip[gi]

            svd_p = svd_dir / f"layer{l:02d}_head{h:02d}.pt"
            svd = torch.load(svd_p, map_location="cpu")
            U_keep = svd["U_keep"].float()  # [65,r]
            u = U_keep[:, k]               # [65]

            # a_* = nu_* @ u
            a_clean = torch.matmul(nu_clean[:, l, h, :], u) * sigma
            a_corr = torch.matmul(nu_corr[:, l, h, :], u) * sigma
            if flip:
                a_clean = -a_clean
                a_corr = -a_corr

            X_clean[:, l, c] += a_clean
            X_corr[:, l, c] += a_corr

            if store_masked:
                assert X_clean_masked is not None and X_corr_masked is not None
                X_clean_masked[:, l, c] += a_clean * mask_val
                X_corr_masked[:, l, c] += a_corr * mask_val

        out_sub = out_dir / "artifacts" / "e1_fix2_fix1a"
        _ensure_dir(out_sub)

        paths = {
            "X_clean": out_sub / f"X_clean_{split}_fix2.pt",
            "X_corr": out_sub / f"X_corr_{split}_fix2.pt",
        }
        torch.save(X_clean, paths["X_clean"])
        torch.save(X_corr, paths["X_corr"])

        if store_masked:
            paths["X_clean_masked"] = out_sub / f"X_clean_{split}_masked_fix2.pt"
            paths["X_corr_masked"] = out_sub / f"X_corr_{split}_masked_fix2.pt"
            torch.save(X_clean_masked, paths["X_clean_masked"])  # type: ignore[arg-type]
            torch.save(X_corr_masked, paths["X_corr_masked"])    # type: ignore[arg-type]

        outputs[split] = paths

    return outputs


# -----------------------------
# Reporting helpers
# -----------------------------

def top_tokens_from_sig(sig: SigMap, tokenizer: AutoTokenizer, topk: int = 8) -> Tuple[List[str], List[str]]:
    pos = [(tid, w) for (s, tid), w in sig.items() if s == "P"]
    neg = [(tid, w) for (s, tid), w in sig.items() if s == "N"]
    pos = sorted(pos, key=lambda x: -x[1])[:topk]
    neg = sorted(neg, key=lambda x: -x[1])[:topk]
    pos_t = [tokenizer.decode([tid]) for tid, _ in pos]
    neg_t = [tokenizer.decode([tid]) for tid, _ in neg]
    return pos_t, neg_t


def compute_cluster_stats(clusters: List[Cluster], total_mass: float) -> Dict[str, Any]:
    masses = np.array([c.mass_total for c in clusters], dtype=np.float64)
    sizes = np.array([len(c.members) for c in clusters], dtype=np.int64)
    order = np.argsort(-masses)
    top10 = masses[order[:10]].sum() if masses.size else 0.0
    singleton_frac = float((sizes == 1).mean()) if sizes.size else 0.0
    frac_top10 = float(top10 / total_mass) if total_mass > 0 else 0.0
    # HHI on clusters
    p = masses / (total_mass + 1e-12)
    hhi = float(np.sum(p * p))
    return {
        "n_clusters": int(len(clusters)),
        "singleton_frac": singleton_frac,
        "top10_mass_frac": frac_top10,
        "top1_mass_frac": float(masses[order[0]] / total_mass) if masses.size and total_mass > 0 else 0.0,
        "hhi": hhi,
        "cluster_size_stats": {
            "min": int(sizes.min()) if sizes.size else 0,
            "median": float(np.median(sizes)) if sizes.size else 0.0,
            "max": int(sizes.max()) if sizes.size else 0,
        },
        "mass_stats": {
            "min": float(masses.min()) if masses.size else 0.0,
            "median": float(np.median(masses)) if masses.size else 0.0,
            "max": float(masses.max()) if masses.size else 0.0,
        }
    }


def stability_by_signature(
    sig_maps_active: List[SigMap],
    dir_to_cluster: np.ndarray,
    clusters: List[Cluster],
) -> Dict[str, Any]:
    """
    Assign each active direction to the best cluster rep by signature similarity,
    and report fraction above a few thresholds.
    """
    sims = []
    for i, sig in enumerate(sig_maps_active):
        best = -1.0
        for c in clusters:
            sim, common, _ = best_sig_similarity(sig, c.sig_rep)
            if sim > best:
                best = sim
        sims.append(best)
    sims = np.array(sims, dtype=np.float64)
    return {
        "mean_best_sim": float(sims.mean()) if sims.size else 0.0,
        "frac_best_sim_ge_0.25": float((sims >= 0.25).mean()) if sims.size else 0.0,
        "frac_best_sim_ge_0.30": float((sims >= 0.30).mean()) if sims.size else 0.0,
        "frac_best_sim_ge_0.40": float((sims >= 0.40).mean()) if sims.size else 0.0,
    }


# -----------------------------
# Main
# -----------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, required=True, choices=["gp", "ioi", "gt"])
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)

    # Active definition and mass
    parser.add_argument("--tau_active", type=float, default=1e-2, help="Active mask threshold for clustering (default 1e-2)")
    parser.add_argument("--use_candidates_json", action="store_true", help="If set, restrict directions to candidates.json when present.")
    parser.add_argument("--no_use_candidates_json", action="store_true", help="If set, ignore candidates.json even if present.")

    # Signature settings
    parser.add_argument("--T_pos", type=int, default=20)
    parser.add_argument("--T_neg", type=int, default=20)
    parser.add_argument("--use_abs_weight_norm", action="store_true", help="Normalize weights within pos/neg (recommended).")
    parser.add_argument("--no_use_abs_weight_norm", action="store_true")

    # Clustering similarity
    parser.add_argument("--sim_threshold", type=float, default=0.25)
    parser.add_argument("--min_common_signed_tokens", type=int, default=3)

    # Rebuild X settings
    parser.add_argument("--store_masked_X", action="store_true", help="Also store X_*_masked where each direction is weighted by its mask.")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()

    _seed_all(args.seed)

    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    if not out_dir.exists():
        raise FileNotFoundError(out_dir)

    # Load tokenizer + model (for signatures)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)
    model.eval()

    # Build direction table
    use_cands = args.use_candidates_json or (not args.no_use_candidates_json)
    df_all, v_mat, dir_ids = load_direction_table(out_dir, tau_active=args.tau_active, use_candidates_json=use_cands)

    # Active filtering
    active_mask = df_all["active"].values.astype(bool)
    df_active = df_all[active_mask].copy().reset_index(drop=True)
    v_active = v_mat[active_mask].numpy()

    # Print quantiles for sanity
    def q(arr, qs=(0.0,0.1,0.5,0.9,0.99,1.0)):
        arr = np.asarray(arr, dtype=np.float64)
        return {str(qq): float(np.quantile(arr, qq)) for qq in qs}

    print("[Fix2/Fix1a] Direction table built.")
    print("  N_dir_total:", int(df_all.shape[0]))
    print("  N_dir_active:", int(df_active.shape[0]), f"(tau_active={args.tau_active:g})")
    if df_active.shape[0] == 0:
        raise RuntimeError("No active directions. Lower tau_active or check that masks were saved correctly.")

    print("  mask quantiles (active):", q(df_active["mask"].values))
    print("  sigma quantiles (active):", q(df_active["sigma"].values))
    print("  mass_fix1a quantiles (active):", q(df_active["mass_fix1a"].values))

    # Save direction table + v matrix
    out_sub = out_dir / "artifacts" / "e1_fix2_fix1a"
    _ensure_dir(out_sub)

    df_all.to_parquet(out_sub / "direction_table_all.parquet", index=False)
    df_active.to_parquet(out_sub / "direction_table_active.parquet", index=False)
    torch.save(v_mat, out_sub / "v_dir_matrix_all.pt")
    torch.save(v_mat[active_mask], out_sub / "v_dir_matrix_active.pt")
    _write_json(out_sub / "dir_ids_all.json", dir_ids)
    _write_json(out_sub / "dir_ids_active.json", df_active["dir_id"].tolist())

    # Build signatures for active directions only (cheaper)
    use_abs = True if args.use_abs_weight_norm else (False if args.no_use_abs_weight_norm else True)
    sig_df = build_direction_signatures(
        v_mat=v_mat[active_mask],
        dir_ids=df_active["dir_id"].tolist(),
        tokenizer=tokenizer,
        model=model,
        device=device,
        T_pos=args.T_pos,
        T_neg=args.T_neg,
        use_abs_weight_norm=use_abs,
    )
    sig_df.to_parquet(out_sub / "direction_signatures_active.parquet", index=False)

    # Turn signatures into maps aligned with active directions
    sig_maps_active: List[SigMap] = []
    for _, r in sig_df.iterrows():
        sig_maps_active.append(sig_to_map(r["sig_pos_ids"], r["sig_pos_w"], r["sig_neg_ids"], r["sig_neg_w"]))

    # Cluster active directions
    masses = df_active["mass_fix1a"].values.astype(np.float64)
    active_idx_global = np.where(active_mask)[0]
    clusters, dir_to_cluster, dir_flip = cluster_active_directions(
        active_idx=active_idx_global,
        masses=masses,
        sig_maps=sig_maps_active,
        v_active=v_active,
        sim_threshold=float(args.sim_threshold),
        min_common_signed_tokens=int(args.min_common_signed_tokens),
    )

    # Cluster stats
    total_mass = float(masses.sum())
    stats = compute_cluster_stats(clusters, total_mass=total_mass)

    print("\n[Fix2/Fix1a] Clustering done.")
    print("  C_new:", stats["n_clusters"])
    print("  singleton_frac:", f"{stats['singleton_frac']:.3f}")
    print("  top10_mass_frac:", f"{stats['top10_mass_frac']:.3f}")
    print("  HHI:", f"{stats['hhi']:.3f}")

    # Write clustering artifacts
    # Map from dir_id -> cluster_id and flip
    dir_id_active = df_active["dir_id"].tolist()
    dir_to_cluster_map = {dir_id_active[i]: int(dir_to_cluster[i]) for i in range(len(dir_id_active))}
    dir_flip_map = {dir_id_active[i]: bool(dir_flip[i]) for i in range(len(dir_id_active))}
    _write_json(out_sub / "dir_to_cluster_fix2.json", dir_to_cluster_map)
    _write_json(out_sub / "dir_flip_fix2.json", dir_flip_map)

    cluster_members = {str(ci): [dir_id_active[i] for i in c.members] for ci, c in enumerate(clusters)}
    _write_json(out_sub / "cluster_members_fix2.json", cluster_members)

    # Save cluster mass table + top tokens
    cm_rows = []
    for ci, c in enumerate(clusters):
        pos_t, neg_t = top_tokens_from_sig(c.sig_rep, tokenizer, topk=10)
        cm_rows.append({
            "cluster_id": ci,
            "mass_total": float(c.mass_total),
            "n_members": int(len(c.members)),
            "top_pos_tokens": pos_t,
            "top_neg_tokens": neg_t,
            "rep_dir_id": dir_id_active[c.rep_dir_idx],
        })
    cm = pd.DataFrame(cm_rows).sort_values("mass_total", ascending=False)
    cm.to_parquet(out_sub / "cluster_mass_fix2.parquet", index=False)

    # Rebuild receptor bank (include both v_bank and v_c for backwards-compat with your old cell)
    v_bank = torch.tensor(np.stack([c.v_rep for c in clusters], axis=0), dtype=torch.float32)
    rep_meta = []
    for ci, c in enumerate(clusters):
        pos_t, neg_t = top_tokens_from_sig(c.sig_rep, tokenizer, topk=20)
        rep_meta.append({
            "cluster_id": ci,
            "mass_total": float(c.mass_total),
            "n_members": int(len(c.members)),
            "rep_dir_id": dir_id_active[c.rep_dir_idx],
            "top_pos_tokens": pos_t,
            "top_neg_tokens": neg_t,
            "member_dir_ids": [dir_id_active[i] for i in c.members],
        })
    bank = {"v_bank": v_bank, "v_c": v_bank, "rep_meta": rep_meta}
    torch.save(bank, out_sub / "receptor_bank_fix2_fix1a.pt")

    # Rebuild X tensors
    # IMPORTANT: relies on nu_clean_at_target existing in caches (paper-faithful hook point).
    x_paths = rebuild_X_tensors(
        out_dir=out_dir,
        splits=["train", "val", "test"],
        df_all=df_all,
        v_mat=v_mat,
        dir_ids=dir_ids,
        active_mask=active_mask,
        dir_to_cluster_active=dir_to_cluster,
        dir_flip_active=dir_flip,
        clusters=clusters,
        store_masked=bool(args.store_masked_X),
    )

    # Stability sanity (signature-based; nontrivial mean_best_sim expected)
    stab = stability_by_signature(sig_maps_active, dir_to_cluster, clusters)

    # Load old report if present for comparison
    old_report_path = out_dir / "artifacts" / "e1" / "e1_report.json"
    old = None
    if old_report_path.exists():
        try:
            old = _read_json(old_report_path)
        except Exception:
            old = None

    report = {
        "task": args.task,
        "tau_active": float(args.tau_active),
        "T_pos": int(args.T_pos),
        "T_neg": int(args.T_neg),
        "sim_threshold": float(args.sim_threshold),
        "min_common_signed_tokens": int(args.min_common_signed_tokens),
        "direction_counts": {
            "N_dir_total": int(df_all.shape[0]),
            "N_dir_active": int(df_active.shape[0]),
        },
        "coverage_fix1a": {
            "total_mass": float(total_mass),
            "top10_mass_frac": float(stats["top10_mass_frac"]),
            "top1_mass_frac": float(stats["top1_mass_frac"]),
        },
        "clusters_fix2": stats,
        "stability_fix2": stab,
        "paths": {k: {kk: str(vv) for kk, vv in d.items()} for k, d in x_paths.items()},
        "old_e1_report": old,
    }
    _write_json(out_sub / "e1_report_fix2_fix1a.json", report)

    # Final compact print (pasteable)
    print("\n=== FIX2/FIX1A COMPACT REPORT ===")
    if old is not None:
        try:
            old_cov = old.get("coverage", {})
            print("C_old:", old_cov.get("n_clusters", "NA"), " | C_new:", stats["n_clusters"])
            print("top10_mass_old:", old_cov.get("coverage_top10_mass_frac", "NA"),
                  " | top10_mass_new_fix1a:", f"{stats['top10_mass_frac']:.3f}")
        except Exception:
            print("C_new:", stats["n_clusters"])
    else:
        print("C_new:", stats["n_clusters"])
        print("top10_mass_new_fix1a:", f"{stats['top10_mass_frac']:.3f}")

    print("singleton_frac_new:", f"{stats['singleton_frac']:.3f}")
    print("HHI_new:", f"{stats['hhi']:.3f}")
    print("stability mean_best_sim:", f"{stab['mean_best_sim']:.3f}",
          "| frac>=0.25:", f"{stab['frac_best_sim_ge_0.25']:.3f}")
    print("Bank v_bank shape:", tuple(v_bank.shape))
    print("Example X_clean_val shape:", tuple(torch.load(x_paths["val"]["X_clean"], map_location="cpu").shape))
    print("Wrote:", out_sub)

    # Show top-5 clusters
    print("\nTop-5 clusters (mass, size, rep_dir, top tokens):")
    for row in cm.head(5).itertuples(index=False):
        print(f"  c{row.cluster_id}: mass={row.mass_total:.4f} size={row.n_members} rep={row.rep_dir_id}")
        print("     +", row.top_pos_tokens[:8])
        print("     -", row.top_neg_tokens[:8])


if __name__ == "__main__":
    main()
