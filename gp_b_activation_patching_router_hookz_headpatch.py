#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gp_b_activation_patching_router_hookz_headpatch.py

Experiment B (router discovery) — Step-2: head-level, all-positions hook_z patching.

Key changes vs the slot-based hook_z router (gp_b_activation_patching_router_hookz_v2.py):
  - Candidate nodes are (L, H) only (no slots)
  - Patch primitive overwrites an entire head's hook_z across ALL positions p <= t* (per-example last_idx)
  - Cache clean hook_z FULL tensors per layer:
        (N_sel, S_pad_global, n_heads, d_head) float16 on CPU
  - Uses a global pad length for the selected pool so cached tensors concatenate cleanly.

Metric a(x) is unchanged:
  - Compute a(x) at (layer_star, head_star, sv_idx) from TL cache:
      blocks.{L*}.ln1.hook_normalized  (B,S,D)
      blocks.{L*}.attn.hook_pattern    (B,H,S,S)
    ctx_t* = sum_s pat[head_star, t*, s] * x_ln1[s]
    a = < [ctx_t*, 1], u_vec >

Sanity checks (must pass, per direction group):
  - SANITY-PARENT: ln1_all restores a(x) ~ 1.0 (upper bound)
  - SANITY-PATCH-NOTHING: no head patches => restore ~ 0
  - SANITY-ID-Z-HEAD: patch hook_z head with corrupt-cached z => no change

References for TL hook conventions:
  - hook_z shape & meaning: (batch, position, head, d_head) (weighted sum of V) citeturn0search5
  - hook function signature: fn(tensor, hook) citeturn0search6turn0search8
"""

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer

import train_gp_masks_and_dump_ov_logit_receptors as gp


DIR_HE_TO_SHE = "he_to_she"
DIR_SHE_TO_HE = "she_to_he"
DIRS = (DIR_HE_TO_SHE, DIR_SHE_TO_HE)


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


@dataclass
class PairRec:
    clean_ids: List[int]
    corr_ids: List[int]
    L: int
    pos_first: int
    pos_last: int
    pos_pred: int
    direction: str   # "he_to_she" or "she_to_he"


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

        pairs.append(PairRec(
            clean_ids=ids_c,
            corr_ids=ids_k,
            L=len(ids_c),
            pos_first=diff[0],
            pos_last=diff[-1],
            pos_pred=len(ids_c) - 1,
            direction=direction,
        ))
    return pairs


def pad_batch_ids(tokenizer: GPT2TokenizerFast, ids_list: List[List[int]], pad_len: int, device: str):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id is None; set pad_token to eos.")
    tokens, _attn, last_idx = gp._pad_to_length(ids_list, pad_len, pad_id, device)
    return tokens, last_idx


def make_pair_batch(tokenizer: GPT2TokenizerFast, pairs_batch: List[PairRec], device: str, pad_len_override: Optional[int] = None):
    pad_len = int(pad_len_override) if pad_len_override is not None else max(p.L for p in pairs_batch)
    clean_ids_list = [p.clean_ids for p in pairs_batch]
    corr_ids_list = [p.corr_ids for p in pairs_batch]
    tokens_clean, last_idx = pad_batch_ids(tokenizer, clean_ids_list, pad_len, device)
    tokens_corr, _ = pad_batch_ids(tokenizer, corr_ids_list, pad_len, device)
    pos_slots = torch.tensor([[p.pos_first, p.pos_last, p.pos_pred] for p in pairs_batch],
                             dtype=torch.long, device=device)
    return tokens_clean, tokens_corr, last_idx, pos_slots


# -----------------------------
# Metric: a(x) from TL cache
# -----------------------------

@torch.no_grad()
def ai_from_cache(cache: Any, last_idx: torch.Tensor, layer_star: int, head_star: int, u_vec: torch.Tensor) -> torch.Tensor:
    x_ln1 = cache[f"blocks.{layer_star}.ln1.hook_normalized"]    # (B,S,D)
    pat = cache[f"blocks.{layer_star}.attn.hook_pattern"]        # (B,H,S,S)
    B, S, D = x_ln1.shape
    idx = torch.arange(B, device=x_ln1.device)

    weights = pat[idx, head_star, last_idx, :]                  # (B,S)
    ctx_t = torch.einsum("bs,bsd->bd", weights, x_ln1)           # (B,D)

    ones = torch.ones((B, 1), device=x_ln1.device, dtype=ctx_t.dtype)
    nu = torch.cat([ctx_t, ones], dim=-1)                        # (B,D+1)
    return (nu * u_vec[None, :]).sum(dim=-1)                     # (B,)


@torch.no_grad()
def run_and_get_ai(model: HookedTransformer, tokens: torch.Tensor, last_idx: torch.Tensor,
                   layer_star: int, head_star: int, u_vec: torch.Tensor):
    names = [
        f"blocks.{layer_star}.ln1.hook_normalized",
        f"blocks.{layer_star}.attn.hook_pattern",
    ]
    _out, cache = model.run_with_cache(tokens, names_filter=names, stop_at_layer=layer_star + 1)
    ai = ai_from_cache(cache, last_idx, layer_star, head_star, u_vec)
    return ai, cache


# -----------------------------
# Cache full hook_z (clean/corr)
# -----------------------------

@torch.no_grad()
def cache_z_full_for_layer(model: HookedTransformer, tokens: torch.Tensor, layer_to_cache: int) -> torch.Tensor:
    hook_name = f"blocks.{layer_to_cache}.attn.hook_z"
    _out, cache = model.run_with_cache(tokens, names_filter=[hook_name], stop_at_layer=layer_to_cache + 1)
    z = cache[hook_name]  # (B,S_pad,H,Dh)
    return z.to(torch.float16).cpu()


# -----------------------------
# Head-level patching: overwrite all positions <= t*
# -----------------------------

def make_hook_z_patch_fn_head(heads_to_patch: List[int], z_clean_full_batch: torch.Tensor, last_idx: torch.Tensor):
    heads = [int(h) for h in heads_to_patch]

    def hook_fn(z: torch.Tensor, hook=None):
        z2 = z.clone()
        B, S_pad, H, Dh = z2.shape
        idx_pos = torch.arange(S_pad, device=z2.device)[None, :]               # (1,S_pad)
        mask = (idx_pos <= last_idx[:, None])                                  # (B,S_pad)
        mask3 = mask[:, :, None]                                               # (B,S_pad,1)

        for h in heads:
            z2[:, :, h, :] = torch.where(
                mask3,
                z_clean_full_batch[:, :, h, :].to(z2.dtype),
                z2[:, :, h, :],
            )
        return z2

    return hook_fn


@torch.no_grad()
def run_corr_with_head_patches_get_ai(
    model: HookedTransformer,
    tokens_corr: torch.Tensor,
    last_idx: torch.Tensor,
    layer_star: int,
    head_star: int,
    u_vec: torch.Tensor,
    patches_by_layer: Dict[int, List[int]],
    clean_z_full_cache_layers_cpu: Dict[int, torch.Tensor],
    batch_slice: slice,
):
    if len(patches_by_layer) == 0:
        a_corr, _ = run_and_get_ai(model, tokens_corr, last_idx, layer_star, head_star, u_vec)
        return a_corr

    fwd_hooks = []
    for L, heads in patches_by_layer.items():
        hook_name = f"blocks.{L}.attn.hook_z"
        z_clean_full = clean_z_full_cache_layers_cpu[L][batch_slice].to(tokens_corr.device)
        fwd_hooks.append((hook_name, make_hook_z_patch_fn_head(heads, z_clean_full, last_idx)))

    names = [
        f"blocks.{layer_star}.ln1.hook_normalized",
        f"blocks.{layer_star}.attn.hook_pattern",
    ]
    with model.hooks(fwd_hooks=fwd_hooks):
        _out, cache = model.run_with_cache(tokens_corr, names_filter=names, stop_at_layer=layer_star + 1)

    return ai_from_cache(cache, last_idx, layer_star, head_star, u_vec)


# -----------------------------
# Restore stats
# -----------------------------

def compute_restore_stats(a_clean: torch.Tensor, a_corr: torch.Tensor, a_patch: torch.Tensor, delta_eps: float) -> Dict[str, float]:
    a_clean = a_clean.float()
    a_corr = a_corr.float()
    a_patch = a_patch.float()

    a_clean_mean = float(a_clean.mean().item())
    a_corr_mean = float(a_corr.mean().item())
    a_patch_mean = float(a_patch.mean().item())

    denom = (a_clean_mean - a_corr_mean)
    mean_based = float("nan") if abs(denom) < 1e-12 else float((a_patch_mean - a_corr_mean) / denom)

    delta = (a_clean - a_corr)
    valid = delta.abs() > float(delta_eps)
    if valid.any():
        r = (a_patch[valid] - a_corr[valid]) / delta[valid]
        per_ex_mean = float(r.mean().item())
        per_ex_std = float(r.std(unbiased=False).item())
        n_valid = int(r.numel())
    else:
        per_ex_mean, per_ex_std, n_valid = float("nan"), float("nan"), 0

    return {
        "mean_based": mean_based,
        "per_ex_mean": per_ex_mean,
        "per_ex_std": per_ex_std,
        "n_valid": n_valid,
        "a_patch_mean": a_patch_mean,
        "a_clean_mean": a_clean_mean,
        "a_corr_mean": a_corr_mean,
    }


# -----------------------------
# Baseline ai for selection
# -----------------------------

@torch.no_grad()
def compute_ai_for_pairs(model: HookedTransformer, tokenizer: GPT2TokenizerFast, pairs: List[PairRec],
                         layer_star: int, head_star: int, u_vec: torch.Tensor,
                         batch_size: int, device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    a_clean_all, a_corr_all = [], []
    for i, j in batches_idx(len(pairs), batch_size):
        batch = pairs[i:j]
        tokens_clean, tokens_corr, last_idx, _pos_slots = make_pair_batch(tokenizer, batch, device)
        a_clean, _ = run_and_get_ai(model, tokens_clean, last_idx, layer_star, head_star, u_vec)
        a_corr, _ = run_and_get_ai(model, tokens_corr, last_idx, layer_star, head_star, u_vec)
        a_clean_all.append(a_clean.detach().float().cpu())
        a_corr_all.append(a_corr.detach().float().cpu())
    return torch.cat(a_clean_all, dim=0), torch.cat(a_corr_all, dim=0)


# -----------------------------
# Parent sanity (canonical hooks)
# -----------------------------

def make_ln1_patch_hook(x_ln1_clean: torch.Tensor, last_idx: torch.Tensor, pos_slots: torch.Tensor, mode: str):
    def hook_fn(x_ln1: torch.Tensor, hook=None):
        x = x_ln1.clone()
        B, S, D = x.shape
        idx = torch.arange(B, device=x.device)
        if mode == "none":
            return x
        if mode == "slots":
            for slot_id in (0, 1, 2):
                pos = pos_slots[:, slot_id]
                x[idx, pos, :] = x_ln1_clean[idx, pos, :].to(x.dtype)
            return x
        if mode == "all_upto_last":
            mask = (torch.arange(S, device=x.device)[None, :] <= last_idx[:, None])  # (B,S)
            return torch.where(mask[:, :, None], x_ln1_clean.to(x.dtype), x)
        raise ValueError(f"bad ln1 patch mode: {mode}")
    return hook_fn


def make_patrow_patch_hook(pat_clean: torch.Tensor, last_idx: torch.Tensor, head_star: int, mode: str):
    def hook_fn(pat: torch.Tensor, hook=None):
        p = pat.clone()
        if mode == "none":
            return p
        if mode != "tstar_row":
            raise ValueError(f"bad patrow patch mode: {mode}")
        B = p.shape[0]
        idx = torch.arange(B, device=p.device)
        p[idx, head_star, last_idx, :] = pat_clean[idx, head_star, last_idx, :].to(p.dtype)
        return p
    return hook_fn


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
    delta_eps: float,
    pad_len_override: int,
) -> None:
    print("\n[SANITY-PARENT] direct-parent patch upper bound check (must pass)")

    max_abs = 0.0
    for i, j in batches_idx(len(pairs_sel), batch_size):
        batch = pairs_sel[i:j]
        _tc, tokens_corr, last_idx, _pos = make_pair_batch(tokenizer, batch, device, pad_len_override=pad_len_override)
        a_corr, _ = run_and_get_ai(model, tokens_corr, last_idx, layer_star, head_star, u_vec)
        a_null, _ = run_and_get_ai(model, tokens_corr, last_idx, layer_star, head_star, u_vec)
        max_abs = max(max_abs, float((a_null - a_corr).abs().max().item()))
    print(f"  null-check (none/none): max|a_patch-a_corr|={max_abs:.3e} (should be tiny)")
    if max_abs > 1e-5:
        raise RuntimeError("SANITY-PARENT FAILED: null-check differs from corrupt baseline.")

    modes = [
        ("ln1_slots", "slots", "none"),
        ("ln1_all", "all_upto_last", "none"),
        ("patrow_only", "none", "tstar_row"),
        ("ln1_all+patrow", "all_upto_last", "tstar_row"),
    ]
    stats: Dict[str, Dict[str, float]] = {}

    for name, ln1_mode, pat_mode in modes:
        all_clean, all_corr, all_patch = [], [], []
        for i, j in batches_idx(len(pairs_sel), batch_size):
            batch = pairs_sel[i:j]
            tokens_clean, tokens_corr, last_idx, pos_slots = make_pair_batch(tokenizer, batch, device, pad_len_override=pad_len_override)

            _o, cache_clean = model.run_with_cache(
                tokens_clean,
                names_filter=[
                    f"blocks.{layer_star}.ln1.hook_normalized",
                    f"blocks.{layer_star}.attn.hook_pattern",
                ],
                stop_at_layer=layer_star + 1,
            )
            x_ln1_clean = cache_clean[f"blocks.{layer_star}.ln1.hook_normalized"]
            pat_clean = cache_clean[f"blocks.{layer_star}.attn.hook_pattern"]
            a_clean = ai_from_cache(cache_clean, last_idx, layer_star, head_star, u_vec)

            a_corr, _ = run_and_get_ai(model, tokens_corr, last_idx, layer_star, head_star, u_vec)

            fwd_hooks = []
            if ln1_mode != "none":
                fwd_hooks.append((f"blocks.{layer_star}.ln1.hook_normalized",
                                  make_ln1_patch_hook(x_ln1_clean, last_idx, pos_slots, ln1_mode)))
            if pat_mode != "none":
                fwd_hooks.append((f"blocks.{layer_star}.attn.hook_pattern",
                                  make_patrow_patch_hook(pat_clean, last_idx, head_star, pat_mode)))

            with model.hooks(fwd_hooks=fwd_hooks):
                _o2, cache_patch = model.run_with_cache(
                    tokens_corr,
                    names_filter=[
                        f"blocks.{layer_star}.ln1.hook_normalized",
                        f"blocks.{layer_star}.attn.hook_pattern",
                    ],
                    stop_at_layer=layer_star + 1,
                )
            a_patch = ai_from_cache(cache_patch, last_idx, layer_star, head_star, u_vec)

            all_clean.append(a_clean.detach().float().cpu())
            all_corr.append(a_corr.detach().float().cpu())
            all_patch.append(a_patch.detach().float().cpu())

        st = compute_restore_stats(torch.cat(all_clean), torch.cat(all_corr), torch.cat(all_patch), delta_eps)
        stats[name] = st
        print(f"  {name:>12}: mean_based={st['mean_based']:+.4f} | per_ex={st['per_ex_mean']:+.4f}±{st['per_ex_std']:.4f} (n={st['n_valid']})")

    if not (stats["ln1_all"]["mean_based"] >= 0.90 and stats["ln1_all+patrow"]["mean_based"] >= 0.90):
        raise RuntimeError("SANITY-PARENT FAILED: ln1_all did not restore a(x). Stop and debug before router search.")
    print("[SANITY-PARENT] PASS")


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
    pad_len_override: int,
    layer_check: int,
    head_check: int,
) -> None:
    print(f"\n[SANITY-ID-Z-HEAD] identity patch on hook_z at (L={layer_check}, H={head_check}) across all p<=t*")

    max_abs = 0.0
    for i, j in batches_idx(len(pairs_sel), batch_size):
        batch = pairs_sel[i:j]
        _tc, tokens_corr, last_idx, _pos = make_pair_batch(tokenizer, batch, device, pad_len_override=pad_len_override)

        a_corr, _ = run_and_get_ai(model, tokens_corr, last_idx, layer_star, head_star, u_vec)

        z_corr_full = cache_z_full_for_layer(model, tokens_corr, layer_to_cache=layer_check).to(tokens_corr.device)

        def hook_fn(z: torch.Tensor, hook=None):
            z2 = z.clone()
            B, S_pad, H, Dh = z2.shape
            mask = (torch.arange(S_pad, device=z2.device)[None, :] <= last_idx[:, None])  # (B,S_pad)
            mask3 = mask[:, :, None]
            z2[:, :, head_check, :] = torch.where(mask3, z_corr_full[:, :, head_check, :].to(z2.dtype), z2[:, :, head_check, :])
            return z2

        with model.hooks(fwd_hooks=[(f"blocks.{layer_check}.attn.hook_z", hook_fn)]):
            a_id, _ = run_and_get_ai(model, tokens_corr, last_idx, layer_star, head_star, u_vec)

        max_abs = max(max_abs, float((a_id - a_corr).abs().max().item()))

    print(f"  max|a_id - a_corr| = {max_abs:.3e} (should be tiny)")
    if max_abs > 1e-5:
        raise RuntimeError("SANITY-ID-Z-HEAD FAILED: identity head patch changed a(x).")
    print("[SANITY-ID-Z-HEAD] PASS")


# -----------------------------
# Router evaluation
# -----------------------------

def candidate_list_heads(candidate_layers_max: int, n_heads: int) -> List[Tuple[int, int]]:
    return [(l, h) for l in range(candidate_layers_max + 1) for h in range(n_heads)]


@torch.no_grad()
def eval_patchset_stats(
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    selected_pairs: List[PairRec],
    a_clean_sel: torch.Tensor,   # CPU
    a_corr_sel: torch.Tensor,    # CPU
    layer_star: int,
    head_star: int,
    u_vec: torch.Tensor,
    patches_by_layer: Dict[int, List[int]],
    clean_z_full_cache_layers_cpu: Dict[int, torch.Tensor],
    batch_size: int,
    device: str,
    delta_eps: float,
    pad_len_override: int,
) -> Dict[str, float]:
    N = len(selected_pairs)
    a_patch_all = []

    for i, j in batches_idx(N, batch_size):
        batch_pairs = selected_pairs[i:j]
        _tc, tokens_corr, last_idx, _pos_slots = make_pair_batch(tokenizer, batch_pairs, device, pad_len_override=pad_len_override)

        a_patch = run_corr_with_head_patches_get_ai(
            model=model,
            tokens_corr=tokens_corr,
            last_idx=last_idx,
            layer_star=layer_star,
            head_star=head_star,
            u_vec=u_vec,
            patches_by_layer=patches_by_layer,
            clean_z_full_cache_layers_cpu=clean_z_full_cache_layers_cpu,
            batch_slice=slice(i, j),
        ).detach().float().cpu()

        a_patch_all.append(a_patch)

    a_patch_cat = torch.cat(a_patch_all, dim=0)
    st = compute_restore_stats(a_clean_sel, a_corr_sel, a_patch_cat, delta_eps)
    return {
        "a_patch_mean": float(a_patch_cat.mean().item()),
        "restore_mean_based": float(st["mean_based"]),
        "restore_per_example_mean": float(st["per_ex_mean"]),
        "restore_per_example_std": float(st["per_ex_std"]),
        "n_valid": float(st["n_valid"]),
        "a_clean_mean": float(st["a_clean_mean"]),
        "a_corr_mean": float(st["a_corr_mean"]),
    }


def safe_key(x: float) -> float:
    return -1e18 if (x is None or math.isnan(x)) else x


def run_group(
    direction: str,
    pairs_all: List[PairRec],
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    ov,
    out_dir: Path,
    layer_star: int,
    head_star: int,
    sv_idx: int,
    batch_size: int,
    device: str,
    N_pairs: int,
    candidate_layers_max: int,
    topK: int,
    Jmax: int,
    stop_frac: float,
    random_trials: int,
    delta_eps: float,
    greedy_score: str,
    save_json: int,
) -> None:
    pairs = [p for p in pairs_all if p.direction == direction]
    if len(pairs) == 0:
        print(f"\n[SKIP] direction={direction}: no eligible pairs.")
        return

    print("\n==============================")
    print(f"[GROUP] direction={direction} | aligned_pairs={len(pairs)}")
    print("==============================")

    svd_star = ov[layer_star][head_star]
    if not (0 <= sv_idx < svd_star.r):
        raise ValueError(f"--sv_idx {sv_idx} out of range (0..{svd_star.r-1})")
    u_vec = svd_star.U[:, sv_idx].contiguous()

    print("[BASELINE] computing a_clean/a_corr for this direction group ...")
    a_clean_all, a_corr_all = compute_ai_for_pairs(
        model=model, tokenizer=tokenizer, pairs=pairs,
        layer_star=layer_star, head_star=head_star, u_vec=u_vec,
        batch_size=batch_size, device=device
    )
    delta_all = (a_clean_all - a_corr_all)
    abs_delta = delta_all.abs()
    print(f"[BASELINE] |Δa| mean={float(abs_delta.mean()):.6g} median={float(abs_delta.median()):.6g} max={float(abs_delta.max()):.6g}")
    print(f"[BASELINE] Δa mean={float(delta_all.mean()):+.6g}")

    N_sel = min(int(N_pairs), len(pairs))
    top_vals, top_idx = torch.topk(abs_delta, k=N_sel, largest=True)
    pairs_sel = [pairs[int(i)] for i in top_idx.tolist()]
    a_clean_sel = a_clean_all[top_idx].contiguous()
    a_corr_sel = a_corr_all[top_idx].contiguous()
    delta_sel = (a_clean_sel - a_corr_sel)

    print(f"[SELECT] N_pairs={N_sel}/{len(pairs)} by |Δa| (min selected |Δa|={float(top_vals.min()):.6g})")
    print(f"[SELECT] a_clean_mean={float(a_clean_sel.mean()):+.6g} a_corr_mean={float(a_corr_sel.mean()):+.6g} Δmean={float(delta_sel.mean()):+.6g}")

    pad_len_global = max(p.L for p in pairs_sel)
    print(f"[PAD] pad_len_global={pad_len_global} (max token length in selected pool)")

    parent_sanity_check(
        model=model, tokenizer=tokenizer, pairs_sel=pairs_sel,
        layer_star=layer_star, head_star=head_star, u_vec=u_vec,
        batch_size=batch_size, device=device, delta_eps=delta_eps,
        pad_len_override=pad_len_global
    )
    identity_z_head_sanity(
        model=model, tokenizer=tokenizer, pairs_sel=pairs_sel,
        layer_star=layer_star, head_star=head_star, u_vec=u_vec,
        batch_size=batch_size, device=device,
        pad_len_override=pad_len_global,
        layer_check=0, head_check=0
    )

    n_heads = model.cfg.n_heads
    candidates = candidate_list_heads(candidate_layers_max, n_heads)
    print(f"[CAND] layers=0..{candidate_layers_max} ({candidate_layers_max+1}) * heads={n_heads} => {len(candidates)} candidates")

    print("[CACHE] caching clean Z-full for candidate layers ...")
    clean_z_full_cache_layers_cpu: Dict[int, torch.Tensor] = {}
    for L in range(candidate_layers_max + 1):
        chunks = []
        for i, j in batches_idx(N_sel, batch_size):
            batch_pairs = pairs_sel[i:j]
            tokens_clean, _tc2, _last_idx, _pos_slots = make_pair_batch(tokenizer, batch_pairs, device, pad_len_override=pad_len_global)
            z_full = cache_z_full_for_layer(model, tokens_clean, layer_to_cache=L)
            chunks.append(z_full)
        clean_z_full_cache_layers_cpu[L] = torch.cat(chunks, dim=0).contiguous()
        print(f"  cached layer {L:>2}: {tuple(clean_z_full_cache_layers_cpu[L].shape)} dtype={clean_z_full_cache_layers_cpu[L].dtype}")

    st0 = eval_patchset_stats(
        model=model, tokenizer=tokenizer, selected_pairs=pairs_sel,
        a_clean_sel=a_clean_sel, a_corr_sel=a_corr_sel,
        layer_star=layer_star, head_star=head_star, u_vec=u_vec,
        patches_by_layer={},
        clean_z_full_cache_layers_cpu=clean_z_full_cache_layers_cpu,
        batch_size=batch_size, device=device,
        delta_eps=delta_eps,
        pad_len_override=pad_len_global,
    )
    print(f"\n[SANITY] patch-nothing (head patch path): mean_based={st0['restore_mean_based']:+.6g} | "
          f"per_ex={st0['restore_per_example_mean']:+.6g}±{st0['restore_per_example_std']:.6g} (n={int(st0['n_valid'])})")

    print("\n[SINGLE] evaluating single-head candidates ...")
    single_results: List[Dict[str, Any]] = []
    for idx_c, (L, H) in enumerate(candidates):
        st = eval_patchset_stats(
            model=model, tokenizer=tokenizer, selected_pairs=pairs_sel,
            a_clean_sel=a_clean_sel, a_corr_sel=a_corr_sel,
            layer_star=layer_star, head_star=head_star, u_vec=u_vec,
            patches_by_layer={L: [H]},
            clean_z_full_cache_layers_cpu=clean_z_full_cache_layers_cpu,
            batch_size=batch_size, device=device,
            delta_eps=delta_eps,
            pad_len_override=pad_len_global,
        )
        single_results.append({"layer": int(L), "head": int(H), **st})
        if (idx_c + 1) % 20 == 0 or (idx_c + 1) == len(candidates):
            print(f"  done {idx_c+1}/{len(candidates)}")

    def score_of(r: Dict[str, Any]) -> float:
        return float(r["restore_per_example_mean"]) if greedy_score == "per_example" else float(r["restore_mean_based"])

    single_sorted = sorted(single_results, key=lambda r: safe_key(score_of(r)), reverse=True)

    print(f"\n[TOP-20] single-head candidates by score={greedy_score}")
    print("rank\tL\tH\tmean_based\tper_ex(mean±std)\ta_patch_mean")
    for i, r in enumerate(single_sorted[:20], start=1):
        print(f"{i}\t{r['layer']}\t{r['head']}\t"
              f"{r['restore_mean_based']:+.4f}\t"
              f"{r['restore_per_example_mean']:+.4f}±{r['restore_per_example_std']:.4f}\t"
              f"{r['a_patch_mean']:+.6g}")

    if int(save_json) == 1:
        out_path = out_dir / f"b2_router_single_head_{direction}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"meta": {"direction": direction, "N_sel": N_sel, "score_mode": greedy_score, "pad_len_global": pad_len_global},
                       "results_sorted": single_sorted}, f, indent=2)
        print(f"[SAVE] wrote: {out_path}")

    print("\n[GREEDY] building a small head-router set ...")
    pool = single_sorted[:min(int(topK), len(single_sorted))]
    chosen: List[Dict[str, Any]] = []
    chosen_set = set()
    greedy_curve: List[Dict[str, Any]] = []

    for t in range(1, int(Jmax) + 1):
        best = None
        best_stats = None
        best_score = -1e18

        for cand in pool:
            key = (cand["layer"], cand["head"])
            if key in chosen_set:
                continue

            patches_by_layer: Dict[int, List[int]] = {}
            for c in chosen:
                patches_by_layer.setdefault(int(c["layer"]), []).append(int(c["head"]))
            patches_by_layer.setdefault(int(cand["layer"]), []).append(int(cand["head"]))

            st = eval_patchset_stats(
                model=model, tokenizer=tokenizer, selected_pairs=pairs_sel,
                a_clean_sel=a_clean_sel, a_corr_sel=a_corr_sel,
                layer_star=layer_star, head_star=head_star, u_vec=u_vec,
                patches_by_layer=patches_by_layer,
                clean_z_full_cache_layers_cpu=clean_z_full_cache_layers_cpu,
                batch_size=batch_size, device=device,
                delta_eps=delta_eps,
                pad_len_override=pad_len_global,
            )
            sc = float(st["restore_per_example_mean"]) if greedy_score == "per_example" else float(st["restore_mean_based"])
            if (not math.isnan(sc)) and (sc > best_score):
                best_score = sc
                best = cand
                best_stats = st

        if best is None:
            print("[GREEDY] No more candidates improved (or all were NaN). Stopping.")
            break

        chosen.append(best)
        chosen_set.add((best["layer"], best["head"]))
        greedy_curve.append({
            "step": int(t),
            "added": {"layer": int(best["layer"]), "head": int(best["head"])},
            "score_mode": greedy_score,
            "score": float(best_score),
            **best_stats,
        })

        print(f"  step {t:>2}: add (L={best['layer']},H={best['head']}) "
              f"=> score={best_score:+.4f} | mean_based={best_stats['restore_mean_based']:+.4f} "
              f"| per_ex={best_stats['restore_per_example_mean']:+.4f}±{best_stats['restore_per_example_std']:.4f}")

        if best_score >= float(stop_frac):
            print(f"[GREEDY] Reached stop_frac={stop_frac} under score_mode={greedy_score}.")
            break

    if int(save_json) == 1:
        out_path = out_dir / f"b2_router_greedy_{direction}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"meta": {"direction": direction, "score_mode": greedy_score, "pad_len_global": pad_len_global},
                       "greedy_curve": greedy_curve, "chosen_set": chosen}, f, indent=2)
        print(f"[SAVE] wrote: {out_path}")

    print("\n[RANDOM] random baseline band ...")
    m_max = len(chosen)
    random_summary = []
    for m in range(1, m_max + 1):
        scores = []
        for _ in range(int(random_trials)):
            rs = random.sample(candidates, k=m)
            patches_by_layer: Dict[int, List[int]] = {}
            for (L, H) in rs:
                patches_by_layer.setdefault(L, []).append(H)
            st = eval_patchset_stats(
                model=model, tokenizer=tokenizer, selected_pairs=pairs_sel,
                a_clean_sel=a_clean_sel, a_corr_sel=a_corr_sel,
                layer_star=layer_star, head_star=head_star, u_vec=u_vec,
                patches_by_layer=patches_by_layer,
                clean_z_full_cache_layers_cpu=clean_z_full_cache_layers_cpu,
                batch_size=batch_size, device=device,
                delta_eps=delta_eps,
                pad_len_override=pad_len_global,
            )
            sc = float(st["restore_per_example_mean"]) if greedy_score == "per_example" else float(st["restore_mean_based"])
            if not math.isnan(sc):
                scores.append(float(sc))
        if len(scores) == 0:
            mean_sc, std_sc = float("nan"), float("nan")
        else:
            tt = torch.tensor(scores, dtype=torch.float32)
            mean_sc = float(tt.mean().item())
            std_sc = float(tt.std(unbiased=False).item())
        greedy_sc = float(greedy_curve[m - 1]["score"]) if (m - 1) < len(greedy_curve) else float("nan")
        random_summary.append({"m": int(m), "greedy_score": greedy_sc, "random_mean": mean_sc, "random_std": std_sc, "trials_used": len(scores)})
        print(f"  m={m:>2}: greedy={greedy_sc:+.4f} | random={mean_sc:+.4f}±{std_sc:.4f} (n={len(scores)})")

    if int(save_json) == 1:
        out_path = out_dir / f"b2_router_summary_{direction}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"meta": {"direction": direction, "score_mode": greedy_score, "pad_len_global": pad_len_global},
                       "random_baseline": random_summary, "greedy_curve": greedy_curve}, f, indent=2)
        print(f"[SAVE] wrote: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, default="data_main")
    ap.add_argument("--train_csv", type=str, default="train_1k_gp.csv")
    ap.add_argument("--test_csv", type=str, default="test_gp.csv")
    ap.add_argument("--pair_csvs", type=str, default=None,
                    help="Comma-separated CSV filenames to pool pairs from. Default: train_csv,test_csv. Include val_gp.csv if desired.")
    ap.add_argument("--out_dir", type=str, default="outputs/gp")

    ap.add_argument("--layer_star", type=int, required=True)
    ap.add_argument("--head_star", type=int, required=True)
    ap.add_argument("--sv_idx", type=int, required=True)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    ap.add_argument("--N_pairs", type=int, default=256)
    ap.add_argument("--candidate_layers_max", type=int, default=None, help="Default: layer_star-1")
    ap.add_argument("--topK", type=int, default=50)
    ap.add_argument("--Jmax", type=int, default=12)
    ap.add_argument("--stop_frac", type=float, default=0.8)
    ap.add_argument("--random_trials", type=int, default=30)
    ap.add_argument("--delta_eps", type=float, default=1e-6)
    ap.add_argument("--greedy_score", type=str, default="per_example", choices=["per_example", "mean_based"])
    ap.add_argument("--save_json", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--masks_path", type=str, default=None)

    args = ap.parse_args()
    set_seed(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda requested but not available; falling back to cpu.")
        device = "cpu"

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    he_ids = tokenizer.encode(" he", add_special_tokens=False)
    she_ids = tokenizer.encode(" she", add_special_tokens=False)
    print("[TOKENS] he_ids=", he_ids, "decoded=", tokenizer.decode(he_ids, clean_up_tokenization_spaces=False))
    print("[TOKENS] she_ids=", she_ids, "decoded=", tokenizer.decode(she_ids, clean_up_tokenization_spaces=False))

    csv_names = [args.train_csv, args.test_csv] if args.pair_csvs is None else [x.strip() for x in args.pair_csvs.split(",") if x.strip()]
    rows_all: List[Dict[str, str]] = []
    for name in csv_names:
        rs = gp.load_gp_csv(str(data_dir / name))
        rows_all.extend(rs)
        print(f"[DATA] loaded {name}: {len(rs)} rows")
    print(f"[DATA] total rows pooled: {len(rows_all)} from csvs={csv_names}")

    pairs_all = build_pairs_from_rows(tokenizer, rows_all)
    counts = {d: 0 for d in DIRS}
    for p in pairs_all:
        counts[p.direction] += 1
    print(f"[PAIRS] aligned pairs pooled: {len(pairs_all)}")
    print(f"        {DIR_HE_TO_SHE}: {counts[DIR_HE_TO_SHE]} | {DIR_SHE_TO_HE}: {counts[DIR_SHE_TO_HE]}")

    if len(pairs_all) == 0:
        raise RuntimeError("No aligned pairs found. Check token length alignment and pronoun labels.")

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    svd_path = out_dir / "svd_cache.pt"
    if not svd_path.exists():
        print(f"[SVD] Missing {svd_path}. Building SVD cache now (CPU)...")
        gp.build_svd_cache(model, str(svd_path), svd_eps=1e-6, device_for_svd="cpu")
    _qk, ov, _mi, _mo, _rank = gp.load_svd_cache(str(svd_path), device=device)

    masks_path = Path(args.masks_path) if args.masks_path else (out_dir / "masks.pt")
    if masks_path.exists():
        masks_dict = torch.load(str(masks_path), map_location="cpu")
        mask_star = float(masks_dict["ov"][args.layer_star][args.head_star][args.sv_idx].item())
        print(f"[MASK] mask_star={mask_star:.6g} from {masks_path}")
    else:
        print(f"[MASK] masks.pt not found at {masks_path} (ok). Skipping mask print.")

    cand_max = args.candidate_layers_max if args.candidate_layers_max is not None else (args.layer_star - 1)
    if cand_max < 0:
        raise ValueError("candidate_layers_max < 0 (layer_star must be >= 1).")
    cand_max = min(cand_max, model.cfg.n_layers - 1)

    for direction in DIRS:
        run_group(
            direction=direction,
            pairs_all=pairs_all,
            model=model,
            tokenizer=tokenizer,
            ov=ov,
            out_dir=out_dir,
            layer_star=args.layer_star,
            head_star=args.head_star,
            sv_idx=args.sv_idx,
            batch_size=args.batch_size,
            device=device,
            N_pairs=args.N_pairs,
            candidate_layers_max=cand_max,
            topK=args.topK,
            Jmax=args.Jmax,
            stop_frac=args.stop_frac,
            random_trials=args.random_trials,
            delta_eps=args.delta_eps,
            greedy_score=args.greedy_score,
            save_json=args.save_json,
        )

    print("\n[DONE]")


if __name__ == "__main__":
    main()
