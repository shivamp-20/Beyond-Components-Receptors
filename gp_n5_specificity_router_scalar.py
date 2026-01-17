#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""gp_n5_specificity_router_scalar.py

N5 Specificity (Router → Scalar) for GPTask.

Goal (per direction group):
  - Target restoration high: applying router head patches S_r (clean → corrupt on hook_z)
    moves target scalar a_r(x) from corrupt toward clean.
  - Off-target restoration low: the SAME patches do not similarly restore a_d(x) for many
    other active OV directions d.

Definitions (match your Experiment B2 script):
  - a(x) at (L, H, sv) is computed from TransformerLens cached activations at layer L:
      x_ln1 = cache[f"blocks.{L}.ln1.hook_normalized"]            # (B,S,D)
      pat   = cache[f"blocks.{L}.attn.hook_pattern"]              # (B,H,S,S)
      weights = pat[b, H, t*, :]  where t* = last_idx[b]
      ctx_t = Σ_s weights[s] * x_ln1[s]
      nu = [ctx_t, 1]
      a = <nu, u_vec> where u_vec = ov[L][H].U[:, sv]              # (D+1,)

  - Router patch set S_r is a set of heads {(L_i, H_i)}.
    Patching primitive (head-level): overwrite blocks.{L_i}.attn.hook_z for head H_i
    at all positions p <= t* (per-example last_idx) using CLEAN cached z.

Notes on TL conventions used:
  - Attention pattern tensors are [batch, head, query_pos, key_pos]. citeturn0search1turn0search10
  - hook_z tensors are [batch, pos, head, d_head]. citeturn0search3
  - Hook fns have signature fn(tensor, hook). citeturn0search19turn0search17

Outputs:
  - outputs/gp/n5_specificity_{direction}.csv
  - outputs/gp/n5_specificity_{direction}_summary.json

Typical usage (both directions; router_json contains "{direction}"):

python gp_n5_specificity_router_scalar.py \
  --data_dir data_main \
  --pair_csvs train_1k_gp.csv,val_gp.csv,test_gp.csv \
  --out_dir outputs/gp \
  --layer_star 10 --head_star 9 --sv_idx 0 \
  --router_json outputs/gp/b2_router_greedy_{direction}.json \
  --router_take 5 \
  --receptors_jsonl outputs/gp/ov_logit_receptors.jsonl \
  --candidate_mask_min 0.01 \
  --K_controls 60 \
  --N_pairs 256 \
  --batch_size 64 \
  --device cuda \
  --save_json 1

If you don't want to run your router script first, pass an explicit patch set:
  --router_spec "4:3,6:0,9:7"

Prereqs:
  - outputs/gp/svd_cache.pt must exist (this script can build it if missing).
  - outputs/gp/ov_logit_receptors.jsonl should exist for control selection.
    (Produced by train_gp_masks_and_dump_ov_logit_receptors.py when dumping receptors.)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer

import train_gp_masks_and_dump_ov_logit_receptors as gp

DIR_HE_TO_SHE = "he_to_she"
DIR_SHE_TO_HE = "she_to_he"
DIRS = (DIR_HE_TO_SHE, DIR_SHE_TO_HE)


# -----------------------------
# Utils
# -----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def batches_idx(n: int, batch_size: int):
    for i in range(0, n, batch_size):
        yield i, min(n, i + batch_size)


def norm_pronoun(x: str) -> str:
    return (x or "").strip().lower()


def safe_float(x: Any, default: float = float("nan")) -> float:
    try:
        return float(x)
    except Exception:
        return default


# -----------------------------
# Pair building (same as router)
# -----------------------------

@dataclass
class PairRec:
    clean_ids: List[int]
    corr_ids: List[int]
    L: int
    pos_first: int
    pos_last: int
    pos_pred: int
    direction: str


def build_pairs_from_rows(tokenizer: GPT2TokenizerFast, rows: List[Dict[str, str]]) -> List[PairRec]:
    pairs: List[PairRec] = []
    for r in rows:
        p = norm_pronoun(r.get("pronoun", ""))
        cp = norm_pronoun(r.get("corr_pronoun", ""))
        if p not in ("he", "she") or cp not in ("he", "she") or p == cp:
            continue
        direction = DIR_HE_TO_SHE if (p == "he" and cp == "she") else DIR_SHE_TO_HE if (p == "she" and cp == "he") else None
        if direction is None:
            continue

        ids_c = tokenizer.encode(r["prefix"], add_special_tokens=False)
        ids_k = tokenizer.encode(r["corr_prefix"], add_special_tokens=False)
        if len(ids_c) != len(ids_k) or len(ids_c) == 0:
            continue
        diff = [t for t in range(len(ids_c)) if ids_c[t] != ids_k[t]]
        if len(diff) == 0:
            continue

        pairs.append(
            PairRec(
                clean_ids=ids_c,
                corr_ids=ids_k,
                L=len(ids_c),
                pos_first=diff[0],
                pos_last=diff[-1],
                pos_pred=len(ids_c) - 1,
                direction=direction,
            )
        )
    return pairs


def pad_batch_ids(tokenizer: GPT2TokenizerFast, ids_list: List[List[int]], pad_len: int, device: str):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id is None; set pad_token to eos.")
    tokens, _attn, last_idx = gp._pad_to_length(ids_list, pad_len, pad_id, device)
    return tokens, last_idx


def make_pair_batch(
    tokenizer: GPT2TokenizerFast,
    pairs_batch: List[PairRec],
    device: str,
    pad_len_override: Optional[int] = None,
):
    pad_len = int(pad_len_override) if pad_len_override is not None else max(p.L for p in pairs_batch)
    clean_ids_list = [p.clean_ids for p in pairs_batch]
    corr_ids_list = [p.corr_ids for p in pairs_batch]
    tokens_clean, last_idx = pad_batch_ids(tokenizer, clean_ids_list, pad_len, device)
    tokens_corr, _ = pad_batch_ids(tokenizer, corr_ids_list, pad_len, device)
    pos_slots = torch.tensor(
        [[p.pos_first, p.pos_last, p.pos_pred] for p in pairs_batch], dtype=torch.long, device=device
    )
    return tokens_clean, tokens_corr, last_idx, pos_slots


# -----------------------------
# a(x) and nu from TL cache
# -----------------------------

@torch.no_grad()
def ai_from_cache(cache: Any, last_idx: torch.Tensor, layer_L: int, head_h: int, u_vec: torch.Tensor) -> torch.Tensor:
    x_ln1 = cache[f"blocks.{layer_L}.ln1.hook_normalized"]  # (B,S,D)
    pat = cache[f"blocks.{layer_L}.attn.hook_pattern"]      # (B,H,S,S)
    B, S, D = x_ln1.shape
    idx = torch.arange(B, device=x_ln1.device)
    weights = pat[idx, head_h, last_idx, :]                 # (B,S)
    ctx_t = torch.einsum("bs,bsd->bd", weights, x_ln1)      # (B,D)
    ones = torch.ones((B, 1), device=x_ln1.device, dtype=ctx_t.dtype)
    nu = torch.cat([ctx_t, ones], dim=-1)                   # (B,D+1)
    return (nu * u_vec[None, :]).sum(dim=-1)                # (B,)


@torch.no_grad()
def compute_nu_by_head_from_cache(cache: Any, last_idx: torch.Tensor, layer_L: int) -> torch.Tensor:
    """Return nu[b, h, :] for all heads at layer_L. Shape (B, n_heads, D+1)."""
    x_ln1 = cache[f"blocks.{layer_L}.ln1.hook_normalized"]  # (B,S,D)
    pat = cache[f"blocks.{layer_L}.attn.hook_pattern"]      # (B,H,S,S)
    B, S, D = x_ln1.shape
    idx = torch.arange(B, device=x_ln1.device)
    weights = pat[idx, :, last_idx, :]                      # (B,H,S)
    ctx = torch.einsum("bhs,bsd->bhd", weights, x_ln1)      # (B,H,D)
    ones = torch.ones((B, ctx.shape[1], 1), device=x_ln1.device, dtype=ctx.dtype)
    nu = torch.cat([ctx, ones], dim=-1)                     # (B,H,D+1)
    return nu


@torch.no_grad()
def run_and_cache_layer(model: HookedTransformer, tokens: torch.Tensor, layer_L: int):
    names = [
        f"blocks.{layer_L}.ln1.hook_normalized",
        f"blocks.{layer_L}.attn.hook_pattern",
    ]
    _out, cache = model.run_with_cache(tokens, names_filter=names, stop_at_layer=layer_L + 1)
    return cache


# -----------------------------
# hook_z caching and head patching
# -----------------------------

@torch.no_grad()
def cache_z_full_for_layer(model: HookedTransformer, tokens: torch.Tensor, layer_to_cache: int) -> torch.Tensor:
    hook_name = f"blocks.{layer_to_cache}.attn.hook_z"
    _out, cache = model.run_with_cache(tokens, names_filter=[hook_name], stop_at_layer=layer_to_cache + 1)
    z = cache[hook_name]  # (B,S_pad,H,Dh)
    return z.to(torch.float16).cpu()


def make_hook_z_patch_fn_head(heads_to_patch: List[int], z_clean_full_batch: torch.Tensor, last_idx: torch.Tensor):
    heads = [int(h) for h in heads_to_patch]

    def hook_fn(z: torch.Tensor, hook=None):
        z2 = z.clone()
        B, S_pad, H, Dh = z2.shape
        idx_pos = torch.arange(S_pad, device=z2.device)[None, :]
        mask = (idx_pos <= last_idx[:, None])                # (B,S_pad)
        mask3 = mask[:, :, None]                             # (B,S_pad,1)
        for h in heads:
            z2[:, :, h, :] = torch.where(
                mask3,
                z_clean_full_batch[:, :, h, :].to(z2.dtype),
                z2[:, :, h, :],
            )
        return z2

    return hook_fn


# -----------------------------
# Router patch set parsing
# -----------------------------

def parse_router_spec(router_spec: str) -> Dict[int, List[int]]:
    """Parse "L:H,L:H,..." into patches_by_layer dict."""
    patches_by_layer: Dict[int, List[int]] = {}
    if not router_spec:
        return patches_by_layer
    parts = [p.strip() for p in router_spec.split(",") if p.strip()]
    for part in parts:
        if ":" not in part:
            raise ValueError(f"Bad router_spec entry '{part}', expected 'L:H'.")
        Ls, Hs = part.split(":", 1)
        L, H = int(Ls), int(Hs)
        patches_by_layer.setdefault(L, []).append(H)
    # dedupe
    for L in list(patches_by_layer.keys()):
        patches_by_layer[L] = sorted(list(set(int(h) for h in patches_by_layer[L])))
    return patches_by_layer


def load_router_json(router_json_path: Path, router_take: int) -> Dict[int, List[int]]:
    j = json.loads(router_json_path.read_text(encoding="utf-8"))
    chosen = j.get("chosen_set", [])
    chosen = chosen[: int(router_take)]
    patches_by_layer: Dict[int, List[int]] = {}
    for item in chosen:
        L = int(item["layer"])
        H = int(item["head"])
        patches_by_layer.setdefault(L, []).append(H)
    for L in list(patches_by_layer.keys()):
        patches_by_layer[L] = sorted(list(set(int(h) for h in patches_by_layer[L])))
    return patches_by_layer


def resolve_router_json_path(router_json_template: str, direction: str) -> Path:
    s = router_json_template.replace("{direction}", direction)
    return Path(s)


# -----------------------------
# Receptor candidates loading
# -----------------------------

def load_receptor_candidates_jsonl(path: Path, candidate_mask_min: float) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(
            f"Missing receptors_jsonl at {path}. This script expects outputs/gp/ov_logit_receptors.jsonl "
            f"(dumped by train_gp_masks_and_dump_ov_logit_receptors.py)."
        )

    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            mv = safe_float(rec.get("mask", float("nan")))
            if math.isnan(mv) or mv < float(candidate_mask_min):
                continue
            out.append(
                {
                    "layer": int(rec["layer"]),
                    "head": int(rec["head"]),
                    "sv_idx": int(rec["sv_idx"]),
                    "mask": float(mv),
                    "sigma": float(rec.get("singular_value", rec.get("sigma", 0.0))),
                }
            )
    return out


def load_pronoun_bucket(pronoun_json: Optional[Path]) -> List[Tuple[int, int, int]]:
    if pronoun_json is None:
        return []
    if not pronoun_json.exists():
        raise FileNotFoundError(f"pronoun_json not found: {pronoun_json}")
    j = json.loads(pronoun_json.read_text(encoding="utf-8"))
    out = []
    for d in j:
        out.append((int(d["layer"]), int(d["head"]), int(d["sv_idx"])))
    return out


# -----------------------------
# Restoration metrics
# -----------------------------

def compute_restore_stats(
    a_clean: torch.Tensor,
    a_corr: torch.Tensor,
    a_patch: torch.Tensor,
    delta_eps: float,
) -> Dict[str, Any]:
    """Compute both mean-based and per-example restoration metrics."""
    a_clean = a_clean.float()
    a_corr = a_corr.float()
    a_patch = a_patch.float()

    # mean-based
    denom_mean = (a_clean.mean() - a_corr.mean()).item()
    if abs(denom_mean) < 1e-12:
        restore_mean_based = float("nan")
    else:
        restore_mean_based = ((a_patch.mean() - a_corr.mean()).item()) / denom_mean

    # per-example normalized
    delta = a_clean - a_corr
    valid = delta.abs() >= float(delta_eps)
    n_valid = int(valid.sum().item())
    if n_valid == 0:
        r_mean, r_std = float("nan"), float("nan")
    else:
        r = ((a_patch - a_corr) / delta)[valid]
        r_mean = float(r.mean().item())
        r_std = float(r.std(unbiased=False).item())

    return {
        "restore_mean_based": float(restore_mean_based),
        "restore_per_example_mean": float(r_mean),
        "restore_per_example_std": float(r_std),
        "n_valid": n_valid,
        "valid_frac": float(n_valid / max(1, a_clean.numel())),
        "a_clean_mean": float(a_clean.mean().item()),
        "a_corr_mean": float(a_corr.mean().item()),
        "a_patch_mean": float(a_patch.mean().item()),
    }


# -----------------------------
# nu caching (clean/corr/patch)
# -----------------------------

class NuCache:
    def __init__(self):
        self.clean: Dict[int, torch.Tensor] = {}
        self.corr: Dict[int, torch.Tensor] = {}
        self.patch: Dict[int, torch.Tensor] = {}


@torch.no_grad()
def get_nu_clean_for_layer(
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    pairs_sel: List[PairRec],
    layer_L: int,
    batch_size: int,
    device: str,
    pad_len_global: int,
    nu_cache: NuCache,
) -> torch.Tensor:
    if layer_L in nu_cache.clean:
        return nu_cache.clean[layer_L]

    chunks = []
    for i, j in batches_idx(len(pairs_sel), batch_size):
        batch_pairs = pairs_sel[i:j]
        tokens_clean, _tokens_corr, last_idx, _pos_slots = make_pair_batch(
            tokenizer, batch_pairs, device=device, pad_len_override=pad_len_global
        )
        cache = run_and_cache_layer(model, tokens_clean, layer_L)
        nu = compute_nu_by_head_from_cache(cache, last_idx, layer_L)
        chunks.append(nu.detach().cpu().float())

    nu_full = torch.cat(chunks, dim=0).contiguous()  # (N,H,D+1)
    nu_cache.clean[layer_L] = nu_full
    return nu_full


@torch.no_grad()
def get_nu_corr_for_layer(
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    pairs_sel: List[PairRec],
    layer_L: int,
    batch_size: int,
    device: str,
    pad_len_global: int,
    nu_cache: NuCache,
) -> torch.Tensor:
    if layer_L in nu_cache.corr:
        return nu_cache.corr[layer_L]

    chunks = []
    for i, j in batches_idx(len(pairs_sel), batch_size):
        batch_pairs = pairs_sel[i:j]
        _tokens_clean, tokens_corr, last_idx, _pos_slots = make_pair_batch(
            tokenizer, batch_pairs, device=device, pad_len_override=pad_len_global
        )
        cache = run_and_cache_layer(model, tokens_corr, layer_L)
        nu = compute_nu_by_head_from_cache(cache, last_idx, layer_L)
        chunks.append(nu.detach().cpu().float())

    nu_full = torch.cat(chunks, dim=0).contiguous()
    nu_cache.corr[layer_L] = nu_full
    return nu_full


@torch.no_grad()
def get_nu_patch_for_layer(
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    pairs_sel: List[PairRec],
    layer_L: int,
    patches_by_layer: Dict[int, List[int]],
    clean_z_cache_layers_cpu: Dict[int, torch.Tensor],
    batch_size: int,
    device: str,
    pad_len_global: int,
    nu_cache: NuCache,
) -> torch.Tensor:
    if layer_L in nu_cache.patch:
        return nu_cache.patch[layer_L]

    # Only need hooks for router layers that are executed before or at layer_L
    eligible_patch_layers = [Lp for Lp in patches_by_layer.keys() if int(Lp) <= int(layer_L)]

    chunks = []
    for i, j in batches_idx(len(pairs_sel), batch_size):
        batch_pairs = pairs_sel[i:j]
        _tokens_clean, tokens_corr, last_idx, _pos_slots = make_pair_batch(
            tokenizer, batch_pairs, device=device, pad_len_override=pad_len_global
        )

        fwd_hooks = []
        for Lp in eligible_patch_layers:
            hook_name = f"blocks.{int(Lp)}.attn.hook_z"
            z_clean_full = clean_z_cache_layers_cpu[int(Lp)][slice(i, j)].to(tokens_corr.device)
            fwd_hooks.append((hook_name, make_hook_z_patch_fn_head(patches_by_layer[int(Lp)], z_clean_full, last_idx)))

        with model.hooks(fwd_hooks=fwd_hooks):
            cache = run_and_cache_layer(model, tokens_corr, layer_L)
        nu = compute_nu_by_head_from_cache(cache, last_idx, layer_L)
        chunks.append(nu.detach().cpu().float())

    nu_full = torch.cat(chunks, dim=0).contiguous()
    nu_cache.patch[layer_L] = nu_full
    return nu_full


# -----------------------------
# Compute a_clean/a_corr for target for selection
# -----------------------------

@torch.no_grad()
def compute_ai_for_pairs(
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    pairs: List[PairRec],
    layer_L: int,
    head_h: int,
    u_vec: torch.Tensor,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    a_clean_chunks = []
    a_corr_chunks = []
    for i, j in batches_idx(len(pairs), batch_size):
        batch_pairs = pairs[i:j]
        tokens_clean, tokens_corr, last_idx, _pos_slots = make_pair_batch(tokenizer, batch_pairs, device=device, pad_len_override=None)

        cache_clean = run_and_cache_layer(model, tokens_clean, layer_L)
        cache_corr = run_and_cache_layer(model, tokens_corr, layer_L)

        a_clean = ai_from_cache(cache_clean, last_idx, layer_L, head_h, u_vec)
        a_corr = ai_from_cache(cache_corr, last_idx, layer_L, head_h, u_vec)

        a_clean_chunks.append(a_clean.detach().cpu().float())
        a_corr_chunks.append(a_corr.detach().cpu().float())

    return torch.cat(a_clean_chunks, dim=0), torch.cat(a_corr_chunks, dim=0)


# -----------------------------
# Parent sanity (cache-based)
# -----------------------------

@torch.no_grad()
def parent_sanity_check(
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    pairs_sel: List[PairRec],
    layer_star: int,
    head_star: int,
    u_vec: torch.Tensor,
    batch_size: int,
    device: str,
    pad_len_global: int,
    delta_eps_restore: float,
):
    """Upper bound: patch ln1.hook_normalized for all positions <= t* from clean into corrupt; should restore a(x) ~ 1."""

    print("\n[SANITY-PARENT] direct-parent patch upper bound check (must pass)")

    ln1_key = f"blocks.{layer_star}.ln1.hook_normalized"
    pat_key = f"blocks.{layer_star}.attn.hook_pattern"

    def make_ln1_patch_hook(ln1_clean_batch: torch.Tensor, last_idx_batch: torch.Tensor, mode: str, pos_slots_batch: torch.Tensor):
        def hook_fn(x: torch.Tensor, hook=None):
            x2 = x.clone()
            B, S, D = x2.shape
            if mode == "none":
                return x2
            if mode == "slots":
                for slot_id in [0, 1, 2]:
                    pos = pos_slots_batch[:, slot_id]
                    x2[torch.arange(B, device=x2.device), pos, :] = ln1_clean_batch[torch.arange(B, device=x2.device), pos, :].to(x2.dtype)
                return x2
            if mode == "all_upto_last":
                idx_pos = torch.arange(S, device=x2.device)[None, :]
                mask = idx_pos <= last_idx_batch[:, None]
                x2 = torch.where(mask[:, :, None], ln1_clean_batch.to(x2.dtype), x2)
                return x2
            raise ValueError("bad ln1 patch mode")
        return hook_fn

    def make_patrow_patch_hook(pat_clean_batch: torch.Tensor, last_idx_batch: torch.Tensor, mode: str):
        def hook_fn(pat: torch.Tensor, hook=None):
            if mode == "none":
                return pat
            if mode != "tstar_row":
                raise ValueError("bad pat patch mode")
            pat2 = pat.clone()
            B = pat2.shape[0]
            idx = torch.arange(B, device=pat2.device)
            pat2[idx, head_star, last_idx_batch, :] = pat_clean_batch[idx, head_star, last_idx_batch, :].to(pat2.dtype)
            return pat2
        return hook_fn

    modes = [
        ("ln1_slots", "slots", "none"),
        ("ln1_all", "all_upto_last", "none"),
        ("patrow_only", "none", "tstar_row"),
        ("ln1_all+patrow", "all_upto_last", "tstar_row"),
    ]

    stats = {}

    # Gather clean/corr/patched a in batches
    for name, ln1_mode, pat_mode in modes:
        all_clean, all_corr, all_patch = [], [], []

        for i, j in batches_idx(len(pairs_sel), batch_size):
            batch_pairs = pairs_sel[i:j]
            tokens_clean, tokens_corr, last_idx, pos_slots = make_pair_batch(
                tokenizer, batch_pairs, device=device, pad_len_override=pad_len_global
            )

            # baseline clean & corrupt caches
            _o1, cache_clean = model.run_with_cache(tokens_clean, names_filter=[ln1_key, pat_key], stop_at_layer=layer_star + 1)
            _o2, cache_corr = model.run_with_cache(tokens_corr, names_filter=[ln1_key, pat_key], stop_at_layer=layer_star + 1)

            a_clean = ai_from_cache(cache_clean, last_idx, layer_star, head_star, u_vec)
            a_corr = ai_from_cache(cache_corr, last_idx, layer_star, head_star, u_vec)

            # patched corrupt run
            ln1_clean_batch = cache_clean[ln1_key]
            pat_clean_batch = cache_clean[pat_key]

            fwd_hooks = []
            fwd_hooks.append((ln1_key, make_ln1_patch_hook(ln1_clean_batch, last_idx, ln1_mode, pos_slots)))
            fwd_hooks.append((pat_key, make_patrow_patch_hook(pat_clean_batch, last_idx, pat_mode)))

            with model.hooks(fwd_hooks=fwd_hooks):
                _o3, cache_patch = model.run_with_cache(tokens_corr, names_filter=[ln1_key, pat_key], stop_at_layer=layer_star + 1)

            a_patch = ai_from_cache(cache_patch, last_idx, layer_star, head_star, u_vec)

            all_clean.append(a_clean.detach().cpu())
            all_corr.append(a_corr.detach().cpu())
            all_patch.append(a_patch.detach().cpu())

        a_clean_full = torch.cat(all_clean)
        a_corr_full = torch.cat(all_corr)
        a_patch_full = torch.cat(all_patch)
        st = compute_restore_stats(a_clean_full, a_corr_full, a_patch_full, delta_eps=float(delta_eps_restore))
        stats[name] = st

        print(f"  {name:>12}: mean_based={st['restore_mean_based']:+.4f} | per_ex={st['restore_per_example_mean']:+.4f}±{st['restore_per_example_std']:.4f} (n={st['n_valid']})")

    # null-check
    # run with none/none and ensure exact equality
    all_corr, all_id = [], []
    for i, j in batches_idx(len(pairs_sel), batch_size):
        batch_pairs = pairs_sel[i:j]
        _tc, tokens_corr, last_idx, _pos_slots = make_pair_batch(tokenizer, batch_pairs, device=device, pad_len_override=pad_len_global)
        _o2, cache_corr = model.run_with_cache(tokens_corr, names_filter=[ln1_key, pat_key], stop_at_layer=layer_star + 1)
        a_corr = ai_from_cache(cache_corr, last_idx, layer_star, head_star, u_vec)
        with model.hooks(fwd_hooks=[(ln1_key, (lambda x, hook=None: x)), (pat_key, (lambda x, hook=None: x))]):
            _o3, cache_id = model.run_with_cache(tokens_corr, names_filter=[ln1_key, pat_key], stop_at_layer=layer_star + 1)
        a_id = ai_from_cache(cache_id, last_idx, layer_star, head_star, u_vec)
        all_corr.append(a_corr.detach().cpu())
        all_id.append(a_id.detach().cpu())
    a_corr_full = torch.cat(all_corr)
    a_id_full = torch.cat(all_id)
    maxdiff = float((a_id_full - a_corr_full).abs().max().item())
    print(f"  null-check (none/none): max|a_patch-a_corr|={maxdiff:.3e} (should be tiny)")

    if stats["ln1_all"]["restore_mean_based"] < 0.90:
        raise RuntimeError("[SANITY-PARENT] FAILED: ln1_all did not restore a(x).")
    if stats["ln1_all+patrow"]["restore_mean_based"] < 0.90:
        raise RuntimeError("[SANITY-PARENT] FAILED: ln1_all+patrow did not restore a(x).")
    print("[SANITY-PARENT] PASS")


# -----------------------------
# Identity sanity (head-level)
# -----------------------------

@torch.no_grad()
def identity_z_head_sanity(
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    pairs_sel: List[PairRec],
    layer_star: int,
    head_star: int,
    u_vec: torch.Tensor,
    batch_size: int,
    device: str,
    pad_len_global: int,
    layer_check: int,
    head_check: int,
    tol: float = 1e-5,
):
    print(f"[SANITY-ID-Z-HEAD] identity patch on hook_z at (L={layer_check}, H={head_check}) across all p<=t*")

    # Use first batch (sufficient to catch indexing bugs)
    i0, j0 = 0, min(len(pairs_sel), batch_size)
    batch_pairs = pairs_sel[i0:j0]
    _tc, tokens_corr, last_idx, _pos_slots = make_pair_batch(tokenizer, batch_pairs, device=device, pad_len_override=pad_len_global)

    # Baseline corrupt a
    a_corr, _ = run_ai_for_layer_head(model, tokens_corr, last_idx, layer_star, head_star, u_vec)

    # Cache corrupt z full for layer_check
    z_corr_full = cache_z_full_for_layer(model, tokens_corr, layer_check).to(tokens_corr.device)

    # Patch hook that writes z_corr back into itself (<=t*)
    def hook_fn(z: torch.Tensor, hook=None):
        z2 = z.clone()
        B, S_pad, H, Dh = z2.shape
        idx_pos = torch.arange(S_pad, device=z2.device)[None, :]
        mask = (idx_pos <= last_idx[:, None])
        mask3 = mask[:, :, None]
        z2[:, :, head_check, :] = torch.where(mask3, z_corr_full[:, :, head_check, :].to(z2.dtype), z2[:, :, head_check, :])
        return z2

    ln1_key = f"blocks.{layer_star}.ln1.hook_normalized"
    pat_key = f"blocks.{layer_star}.attn.hook_pattern"
    hook_name = f"blocks.{layer_check}.attn.hook_z"
    with model.hooks(fwd_hooks=[(hook_name, hook_fn)]):
        _o, cache = model.run_with_cache(tokens_corr, names_filter=[ln1_key, pat_key], stop_at_layer=layer_star + 1)
    a_id = ai_from_cache(cache, last_idx, layer_star, head_star, u_vec)

    maxdiff = float((a_id - a_corr).abs().max().item())
    print(f"  max|a_id - a_corr| = {maxdiff:.3e} (tol={tol:.1e})")
    if maxdiff > tol:
        raise RuntimeError("SANITY-ID-Z-HEAD FAILED: identity head patch changed a(x).")
    print("[SANITY-ID-Z-HEAD] PASS")


@torch.no_grad()
def run_ai_for_layer_head(
    model: HookedTransformer,
    tokens: torch.Tensor,
    last_idx: torch.Tensor,
    layer_L: int,
    head_h: int,
    u_vec: torch.Tensor,
):
    names = [
        f"blocks.{layer_L}.ln1.hook_normalized",
        f"blocks.{layer_L}.attn.hook_pattern",
    ]
    _out, cache = model.run_with_cache(tokens, names_filter=names, stop_at_layer=layer_L + 1)
    a = ai_from_cache(cache, last_idx, layer_L, head_h, u_vec)
    return a, cache


# -----------------------------
# Control selection (activity filter)
# -----------------------------

def make_relax_schedule(valid_frac_min: float, mean_delta_min: float, relax_steps: int) -> List[Tuple[float, float]]:
    """Return list of (valid_frac_min_i, mean_delta_min_i) from strict -> relaxed."""
    # Multipliers chosen to mimic 0.15 -> 0.10 -> 0.07 -> 0.05 when mean_delta_min=0.15
    multipliers = [1.0, 2.0 / 3.0, 0.47, 1.0 / 3.0]
    out = []
    for step in range(max(0, int(relax_steps)) + 1):
        vf = float(valid_frac_min) - 0.05 * step
        vf = max(0.0, vf)
        mult = multipliers[min(step, len(multipliers) - 1)]
        md = float(mean_delta_min) * mult
        out.append((vf, md))
    return out


def rank_key(rec: Dict[str, Any], score_mode: str) -> float:
    if score_mode == "mask_sigma":
        return float(rec["mask"]) * float(rec["sigma"])
    return float(rec["mask"])


def select_controls(
    candidates: List[Dict[str, Any]],
    exclude: set,
    ov: List[List[Any]],
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    pairs_sel: List[PairRec],
    pad_len_global: int,
    batch_size: int,
    device: str,
    K_controls: int,
    score_mode: str,
    valid_frac_min: float,
    delta_valid_min: float,
    mean_delta_min: float,
    relax_steps: int,
) -> Tuple[List[Tuple[int, int, int]], Dict[Tuple[int, int, int], Dict[str, Any]], Dict[str, Any]]:

    # Filter excluded
    cand2 = [c for c in candidates if (c["layer"], c["head"], c["sv_idx"]) not in exclude]

    cand2.sort(key=lambda r: rank_key(r, score_mode), reverse=True)

    nu_cache = NuCache()

    controls: List[Tuple[int, int, int]] = []
    diag: Dict[Tuple[int, int, int], Dict[str, Any]] = {}

    schedule = make_relax_schedule(valid_frac_min, mean_delta_min, relax_steps)
    thresholds_used = None

    for (vf_min_i, md_min_i) in schedule:
        if len(controls) >= int(K_controls):
            thresholds_used = {"valid_frac_min": vf_min_i, "mean_delta_min": md_min_i}
            break

        for c in cand2:
            if len(controls) >= int(K_controls):
                thresholds_used = {"valid_frac_min": vf_min_i, "mean_delta_min": md_min_i}
                break

            key = (int(c["layer"]), int(c["head"]), int(c["sv_idx"]))
            if key in diag:
                continue

            layer_L, head_h, sv = key
            # guard
            svd = ov[layer_L][head_h]
            if sv < 0 or sv >= int(svd.r):
                continue

            # compute a_clean/a_corr using nu caches
            nu_clean = get_nu_clean_for_layer(model, tokenizer, pairs_sel, layer_L, batch_size, device, pad_len_global, nu_cache)
            nu_corr = get_nu_corr_for_layer(model, tokenizer, pairs_sel, layer_L, batch_size, device, pad_len_global, nu_cache)
            u = svd.U[:, sv].detach().cpu().float()  # (D+1,)

            a_clean = (nu_clean[:, head_h, :] * u[None, :]).sum(dim=-1)
            a_corr = (nu_corr[:, head_h, :] * u[None, :]).sum(dim=-1)
            delta = (a_clean - a_corr)

            valid = delta.abs() >= float(delta_valid_min)
            valid_frac = float(valid.float().mean().item())

            if valid.any():
                mean_abs_delta = float(delta[valid].abs().mean().item())
                mean_delta = float(delta[valid].mean().item())
            else:
                mean_abs_delta = 0.0
                mean_delta = 0.0

            pass_valid_frac = valid_frac >= float(vf_min_i)
            pass_mean = (abs(mean_delta) >= float(md_min_i)) or (mean_abs_delta >= float(md_min_i))

            diag[key] = {
                "mask": float(c["mask"]),
                "sigma": float(c["sigma"]),
                "rank": float(rank_key(c, score_mode)),
                "valid_frac": valid_frac,
                "mean_abs_delta": mean_abs_delta,
                "mean_delta": mean_delta,
            }

            if pass_valid_frac and pass_mean:
                controls.append(key)

        if len(controls) >= int(K_controls):
            break

    if thresholds_used is None:
        # last schedule entry
        thresholds_used = {"valid_frac_min": schedule[-1][0], "mean_delta_min": schedule[-1][1]}

    meta = {
        "thresholds_schedule": [{"valid_frac_min": vf, "mean_delta_min": md} for (vf, md) in schedule],
        "thresholds_used": thresholds_used,
        "K_controls": int(K_controls),
        "num_candidates_after_exclude": len(cand2),
    }

    return controls, diag, meta


# -----------------------------
# Main per-direction run
# -----------------------------

@torch.no_grad()
def run_direction(
    direction: str,
    pairs_all: List[PairRec],
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    ov: List[List[Any]],
    out_dir: Path,
    layer_star: int,
    head_star: int,
    sv_star: int,
    router_json_template: Optional[str],
    router_take: int,
    router_spec: Optional[str],
    receptors_jsonl: Path,
    candidate_mask_min: float,
    pronoun_json: Optional[Path],
    N_pairs: int,
    batch_size: int,
    device: str,
    K_controls: int,
    score_mode: str,
    valid_frac_min: float,
    delta_valid_min: float,
    mean_delta_min: float,
    relax_steps: int,
    delta_eps_restore: float,
    save_json: int,
) -> None:

    pairs_dir = [p for p in pairs_all if p.direction == direction]
    print("\n" + "=" * 30)
    print(f"[GROUP] direction={direction} | aligned_pairs={len(pairs_dir)}")
    print("=" * 30)

    if len(pairs_dir) == 0:
        print(f"[WARN] No aligned pairs for direction {direction}. Skipping.")
        return

    # target u_vec
    u_star = ov[layer_star][head_star].U[:, sv_star].detach().to(device)

    # Baseline a_clean/a_corr for all pairs in direction (for selection)
    print("[BASELINE] computing a_clean/a_corr for this direction group ...")
    a_clean_all, a_corr_all = compute_ai_for_pairs(
        model=model,
        tokenizer=tokenizer,
        pairs=pairs_dir,
        layer_L=layer_star,
        head_h=head_star,
        u_vec=u_star,
        batch_size=batch_size,
        device=device,
    )

    abs_delta = (a_clean_all - a_corr_all).abs()
    print(f"[BASELINE] |Δa| mean={abs_delta.mean().item():.6g} median={abs_delta.median().item():.6g} max={abs_delta.max().item():.6g}")
    print(f"[BASELINE] Δa mean={(a_clean_all - a_corr_all).mean().item():+.6g}")

    k = min(int(N_pairs), len(pairs_dir))
    top_idx = torch.topk(abs_delta, k=k, largest=True).indices
    top_idx = top_idx.detach().cpu().tolist()
    pairs_sel = [pairs_dir[i] for i in top_idx]

    # For consistency with head-patching caches, compute a_clean/a_corr again on selected pool with global pad
    pad_len_global = max(p.L for p in pairs_sel)
    print(f"[SELECT] N_pairs={len(pairs_sel)}/{len(pairs_dir)} by |Δa| (min selected |Δa|={float(abs_delta[top_idx].min().item()):.6g})")

    # Build nu caches on the selected pool at layer_star to get exact means
    nu_cache_tmp = NuCache()
    nu_clean_star = get_nu_clean_for_layer(model, tokenizer, pairs_sel, layer_star, batch_size, device, pad_len_global, nu_cache_tmp)
    nu_corr_star = get_nu_corr_for_layer(model, tokenizer, pairs_sel, layer_star, batch_size, device, pad_len_global, nu_cache_tmp)
    u_star_cpu = ov[layer_star][head_star].U[:, sv_star].detach().cpu().float()
    a_star_clean_sel = (nu_clean_star[:, head_star, :] * u_star_cpu[None, :]).sum(dim=-1)
    a_star_corr_sel = (nu_corr_star[:, head_star, :] * u_star_cpu[None, :]).sum(dim=-1)
    print(f"[SELECT] a_clean_mean={a_star_clean_sel.mean().item():+.6g} a_corr_mean={a_star_corr_sel.mean().item():+.6g} Δmean={(a_star_clean_sel.mean()-a_star_corr_sel.mean()).item():+.6g}")
    print(f"[PAD] pad_len_global={pad_len_global} (max token length in selected pool)")

    # Router patch set
    if router_spec:
        patches_by_layer = parse_router_spec(router_spec)
    else:
        if not router_json_template:
            raise ValueError("Must provide either --router_spec or --router_json.")
        router_path = resolve_router_json_path(router_json_template, direction)
        if not router_path.exists():
            raise FileNotFoundError(f"router_json not found for direction={direction}: {router_path}")
        patches_by_layer = load_router_json(router_path, router_take=router_take)

    print(f"[ROUTER] patches_by_layer={ {int(k): [int(h) for h in v] for k,v in patches_by_layer.items()} }")

    # Parent sanity
    parent_sanity_check(
        model=model,
        tokenizer=tokenizer,
        pairs_sel=pairs_sel,
        layer_star=layer_star,
        head_star=head_star,
        u_vec=u_star,
        batch_size=batch_size,
        device=device,
        pad_len_global=pad_len_global,
        delta_eps_restore=delta_eps_restore,
    )

    # Identity sanity (head-level)
    identity_z_head_sanity(
        model=model,
        tokenizer=tokenizer,
        pairs_sel=pairs_sel,
        layer_star=layer_star,
        head_star=head_star,
        u_vec=u_star,
        batch_size=batch_size,
        device=device,
        pad_len_global=pad_len_global,
        layer_check=0,
        head_check=0,
        tol=1e-5,
    )

    # Load candidates and exclusion sets
    candidates = load_receptor_candidates_jsonl(receptors_jsonl, candidate_mask_min=candidate_mask_min)
    pronoun_bucket = set(load_pronoun_bucket(pronoun_json))
    target_key = (int(layer_star), int(head_star), int(sv_star))
    if target_key in pronoun_bucket:
        pronoun_bucket.remove(target_key)

    exclude = set([target_key]) | pronoun_bucket
    print(f"[CAND] candidates after mask_min={candidate_mask_min}: {len(candidates)} | exclude={len(exclude)}")

    # Select controls by activity filter
    controls, diag, ctrl_meta = select_controls(
        candidates=candidates,
        exclude=exclude,
        ov=ov,
        model=model,
        tokenizer=tokenizer,
        pairs_sel=pairs_sel,
        pad_len_global=pad_len_global,
        batch_size=batch_size,
        device=device,
        K_controls=K_controls,
        score_mode=score_mode,
        valid_frac_min=valid_frac_min,
        delta_valid_min=delta_valid_min,
        mean_delta_min=mean_delta_min,
        relax_steps=relax_steps,
    )

    print(f"[CONTROLS] selected {len(controls)}/{K_controls} controls (score_mode={score_mode})")
    print(f"[CONTROLS] thresholds_used: {ctrl_meta['thresholds_used']}")

    # Cache clean hook_z full for router layers
    router_layers = sorted(int(L) for L in patches_by_layer.keys())
    clean_z_cache_layers_cpu: Dict[int, torch.Tensor] = {}
    if len(router_layers) == 0:
        print("[WARN] Router patch set is empty; patched run == corrupt baseline.")
    else:
        print("[CACHE] caching clean Z-full for router layers ...")
        for L in router_layers:
            chunks = []
            for i, j in batches_idx(len(pairs_sel), batch_size):
                batch_pairs = pairs_sel[i:j]
                tokens_clean, _tokens_corr, _last_idx, _pos_slots = make_pair_batch(
                    tokenizer, batch_pairs, device=device, pad_len_override=pad_len_global
                )
                z_full = cache_z_full_for_layer(model, tokens_clean, layer_to_cache=L)
                chunks.append(z_full)
            z_cat = torch.cat(chunks, dim=0).contiguous()  # (N,S_pad,H,Dh)
            clean_z_cache_layers_cpu[L] = z_cat
            print(f"  cached layer {L:>2}: {tuple(z_cat.shape)} dtype={z_cat.dtype}")

    # Build nu caches needed for: target + controls (+ optional pronoun bucket for reporting)
    nu_cache = NuCache()

    needed_layers = {layer_star}
    needed_layers |= {int(L) for (L, _H, _sv) in controls}
    needed_layers |= {int(L) for (L, _H, _sv) in pronoun_bucket}

    print(f"[NU] computing nu caches for layers: {sorted(list(needed_layers))}")

    # Clean/corr for needed layers
    for L in sorted(list(needed_layers)):
        _ = get_nu_clean_for_layer(model, tokenizer, pairs_sel, L, batch_size, device, pad_len_global, nu_cache)
        _ = get_nu_corr_for_layer(model, tokenizer, pairs_sel, L, batch_size, device, pad_len_global, nu_cache)

    # Patched nu for needed layers
    if len(patches_by_layer) > 0:
        for L in sorted(list(needed_layers)):
            _ = get_nu_patch_for_layer(
                model, tokenizer, pairs_sel, L,
                patches_by_layer=patches_by_layer,
                clean_z_cache_layers_cpu=clean_z_cache_layers_cpu,
                batch_size=batch_size, device=device,
                pad_len_global=pad_len_global,
                nu_cache=nu_cache,
            )
    else:
        # If no patches, patched == corrupt
        for L in sorted(list(needed_layers)):
            nu_cache.patch[L] = nu_cache.corr[L]

    # Helper to compute (a_clean, a_corr, a_patch) for a direction key
    def compute_a_triplet(key: Tuple[int, int, int]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        L, H, sv = key
        svd = ov[L][H]
        u = svd.U[:, sv].detach().cpu().float()
        nu_clean = nu_cache.clean[L]
        nu_corr = nu_cache.corr[L]
        nu_patch = nu_cache.patch[L]
        a_clean = (nu_clean[:, H, :] * u[None, :]).sum(dim=-1)
        a_corr = (nu_corr[:, H, :] * u[None, :]).sum(dim=-1)
        a_patch = (nu_patch[:, H, :] * u[None, :]).sum(dim=-1)
        return a_clean, a_corr, a_patch

    # Target stats
    a_clean_t, a_corr_t, a_patch_t = compute_a_triplet(target_key)
    st_target = compute_restore_stats(a_clean_t, a_corr_t, a_patch_t, delta_eps=float(delta_eps_restore))

    # Controls stats
    rows_csv = []
    abs_R = []
    for key in controls:
        a_clean_d, a_corr_d, a_patch_d = compute_a_triplet(key)
        st = compute_restore_stats(a_clean_d, a_corr_d, a_patch_d, delta_eps=float(delta_eps_restore))
        R_ex = st["restore_per_example_mean"]
        if not math.isnan(R_ex):
            abs_R.append(abs(float(R_ex)))

        d = diag.get(key, {})
        rows_csv.append(
            {
                "layer": key[0],
                "head": key[1],
                "sv_idx": key[2],
                "mask": d.get("mask", float("nan")),
                "sigma": d.get("sigma", float("nan")),
                "rank": d.get("rank", float("nan")),
                "denom_valid_frac": d.get("valid_frac", float("nan")),
                "mean_abs_delta": d.get("mean_abs_delta", float("nan")),
                "mean_delta": d.get("mean_delta", float("nan")),
                "R_ex": st["restore_per_example_mean"],
                "Std_ex": st["restore_per_example_std"],
                "n_valid": st["n_valid"],
                "valid_frac_restore": st["valid_frac"],
                "R_mean": st["restore_mean_based"],
                "a_clean_mean": st["a_clean_mean"],
                "a_corr_mean": st["a_corr_mean"],
                "a_patch_mean": st["a_patch_mean"],
            }
        )

    off_mean_abs_R = float(sum(abs_R) / max(1, len(abs_R)))
    spec_ratio = abs(float(st_target["restore_per_example_mean"])) / (off_mean_abs_R + 0.01) if not math.isnan(float(st_target["restore_per_example_mean"])) else float("nan")

    print("\n[N5] Target restoration:")
    print(f"  target (L={layer_star},H={head_star},sv={sv_star}): R_ex={st_target['restore_per_example_mean']:+.4f}±{st_target['restore_per_example_std']:.4f} (n={st_target['n_valid']}) | R_mean={st_target['restore_mean_based']:+.4f}")
    print("[N5] Off-target controls:")
    if len(abs_R) > 0:
        abs_R_t = torch.tensor(abs_R, dtype=torch.float32)
        print(f"  mean(|R_ex|)={abs_R_t.mean().item():.4f}  median={abs_R_t.median().item():.4f}  max={abs_R_t.max().item():.4f}  (K={len(abs_R)})")
    else:
        print("  (no valid off-target R_ex values)")
    print(f"[N5] SpecRatio = |R_ex(target)| / (mean(|R_ex(controls)|)+0.01) = {spec_ratio:.3f}")

    # Save CSV
    out_csv = out_dir / f"n5_specificity_{direction}.csv"
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = list(rows_csv[0].keys()) if rows_csv else [
            "layer", "head", "sv_idx", "mask", "sigma", "rank",
            "denom_valid_frac", "mean_abs_delta", "mean_delta",
            "R_ex", "Std_ex", "n_valid", "valid_frac_restore",
            "R_mean", "a_clean_mean", "a_corr_mean", "a_patch_mean"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows_csv:
            w.writerow(r)
    print(f"[SAVE] wrote: {out_csv}")

    # Save summary JSON
    if int(save_json) == 1:
        out_summary = out_dir / f"n5_specificity_{direction}_summary.json"
        summary = {
            "direction": direction,
            "target": {
                "layer": layer_star,
                "head": head_star,
                "sv_idx": sv_star,
                **st_target,
            },
            "offtarget_mean_abs_R_ex": off_mean_abs_R,
            "spec_ratio": spec_ratio,
            "controls_abs_R_ex_stats": {
                "K": len(abs_R),
                "mean": float(torch.tensor(abs_R).mean().item()) if len(abs_R) else float("nan"),
                "median": float(torch.tensor(abs_R).median().item()) if len(abs_R) else float("nan"),
                "max": float(torch.tensor(abs_R).max().item()) if len(abs_R) else float("nan"),
            },
            "router_patchset": {int(L): [int(h) for h in hs] for L, hs in patches_by_layer.items()},
            "N_pairs": len(pairs_sel),
            "pad_len_global": pad_len_global,
            "controls_selection_meta": ctrl_meta,
            "delta_eps_restore": float(delta_eps_restore),
            "candidate_mask_min": float(candidate_mask_min),
            "excluded_pronoun_bucket_size": int(len(pronoun_bucket)),
        }
        out_summary.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[SAVE] wrote: {out_summary}")


# -----------------------------
# Main
# -----------------------------

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="data_main")
    ap.add_argument("--pair_csvs", type=str, default="train_1k_gp.csv,val_gp.csv,test_gp.csv")
    ap.add_argument("--out_dir", type=str, default="outputs/gp")

    ap.add_argument("--layer_star", type=int, required=True)
    ap.add_argument("--head_star", type=int, required=True)
    ap.add_argument("--sv_idx", type=int, required=True)

    ap.add_argument("--router_json", type=str, default=None,
                    help="Path template to b2_router_greedy_{direction}.json (use {direction} placeholder).")
    ap.add_argument("--router_take", type=int, default=5)
    ap.add_argument("--router_spec", type=str, default=None,
                    help='Explicit router heads "L:H,L:H,..." (overrides router_json).')

    ap.add_argument("--receptors_jsonl", type=str, default="outputs/gp/ov_logit_receptors.jsonl")
    ap.add_argument("--candidate_mask_min", type=float, default=0.01)
    ap.add_argument("--pronoun_json", type=str, default=None)

    ap.add_argument("--directions", type=str, default=f"{DIR_HE_TO_SHE},{DIR_SHE_TO_HE}")

    ap.add_argument("--N_pairs", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")

    ap.add_argument("--K_controls", type=int, default=60)
    ap.add_argument("--score_mode", type=str, default="mask_sigma", choices=["mask_sigma", "mask"])
    ap.add_argument("--valid_frac_min", type=float, default=0.80)
    ap.add_argument("--delta_valid_min", type=float, default=0.05)
    ap.add_argument("--mean_delta_min", type=float, default=0.15)
    ap.add_argument("--relax_steps", type=int, default=3)

    ap.add_argument("--delta_eps_restore", type=float, default=0.05)

    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save_json", type=int, default=1)

    args = ap.parse_args()

    set_seed(int(args.seed))

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    he_ids = tokenizer.encode(" he", add_special_tokens=False)
    she_ids = tokenizer.encode(" she", add_special_tokens=False)
    print(f"[TOKENS] he_ids= {he_ids} decoded= {tokenizer.decode(he_ids)}")
    print(f"[TOKENS] she_ids= {she_ids} decoded= {tokenizer.decode(she_ids)}")

    # Load rows
    rows_all: List[Dict[str, str]] = []
    csvs = [c.strip() for c in str(args.pair_csvs).split(",") if c.strip()]
    for csv_name in csvs:
        path = data_dir / csv_name
        rows = gp.load_gp_csv(str(path))
        rows_all.extend(rows)
        print(f"[DATA] loaded {csv_name}: {len(rows)} rows")
    print(f"[DATA] total rows pooled: {len(rows_all)} from csvs={csvs}")

    # Build aligned pairs
    pairs_all = build_pairs_from_rows(tokenizer, rows_all)
    n_he = sum(1 for p in pairs_all if p.direction == DIR_HE_TO_SHE)
    n_she = sum(1 for p in pairs_all if p.direction == DIR_SHE_TO_HE)
    print(f"[PAIRS] aligned pairs pooled: {len(pairs_all)}")
    print(f"        he_to_she: {n_he} | she_to_he: {n_she}")

    # Model
    model = HookedTransformer.from_pretrained("gpt2-small", device=args.device)
    model.eval()

    # Load SVD cache (build if missing)
    svd_path = out_dir / "svd_cache.pt"
    if not svd_path.exists():
        print(f"[SVD] Missing {svd_path}. Building SVD cache now (CPU)...")
        # build on CPU to reduce GPU memory spikes
        _cpu = model.to("cpu")
        gp.build_svd_cache(_cpu, str(svd_path), svd_eps=1e-5)
        model = _cpu.to(args.device)
    qk, ov, mlp_in, mlp_out, rank_total_ov = gp.load_svd_cache(str(svd_path), device=args.device)

    # Mask file optional (only for printing; not needed)
    masks_path = out_dir / "masks.pt"
    if not masks_path.exists():
        print(f"[MASK] masks.pt not found at {masks_path} (ok). Skipping mask print.")

    # Direction list
    directions = [d.strip() for d in str(args.directions).split(",") if d.strip()]
    for d in directions:
        if d not in DIRS:
            raise ValueError(f"Bad direction '{d}', expected one of {DIRS}.")

    pronoun_json = Path(args.pronoun_json) if args.pronoun_json else None

    for direction in directions:
        run_direction(
            direction=direction,
            pairs_all=pairs_all,
            model=model,
            tokenizer=tokenizer,
            ov=ov,
            out_dir=out_dir,
            layer_star=int(args.layer_star),
            head_star=int(args.head_star),
            sv_star=int(args.sv_idx),
            router_json_template=str(args.router_json) if args.router_json else None,
            router_take=int(args.router_take),
            router_spec=str(args.router_spec) if args.router_spec else None,
            receptors_jsonl=Path(args.receptors_jsonl),
            candidate_mask_min=float(args.candidate_mask_min),
            pronoun_json=pronoun_json,
            N_pairs=int(args.N_pairs),
            batch_size=int(args.batch_size),
            device=str(args.device),
            K_controls=int(args.K_controls),
            score_mode=str(args.score_mode),
            valid_frac_min=float(args.valid_frac_min),
            delta_valid_min=float(args.delta_valid_min),
            mean_delta_min=float(args.mean_delta_min),
            relax_steps=int(args.relax_steps),
            delta_eps_restore=float(args.delta_eps_restore),
            save_json=int(args.save_json),
        )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
