#!/usr/bin/env python
# -*- coding: utf-8 -*-

# gp_b_activation_patching_router.py
#
# EXPERIMENT B (extended):
# Activation patching / denoising to find upstream router nodes that set the receptor scalar a(x).
#
# Changes vs the previous version:
#   1) Pair pool can be built from MULTIPLE CSVs (default: train + test) for more data.
#   2) Results are computed SEPARATELY for the two directions using labels:
#        - he -> she  (clean pronoun="he", corr_pronoun="she")
#        - she -> he  (clean pronoun="she", corr_pronoun="he")
#      This avoids mean cancellation when mixing opposite directions.
#   3) For every patch, we report BOTH:
#        - mean-based RestoreFrac (direction-split)
#        - per-example normalized RestoreFrac_i = (a_patch_i - a_corr_i) / (a_clean_i - a_corr_i)
#
# Denoising / causal tracing pattern:
#   (1) clean run: cache activations
#   (2) corrupt run: overwrite selected activations with clean cached values
#
# Patched activation (per head):
#   v = (context_resid @ W_V[h]) + b_V[h], where context_resid = pat @ x_ln1
#
# Slots (token positions) we allow patching at:
#   - first: first token position where clean/corr differ
#   - last:  last token position where clean/corr differ
#   - pred:  last token of prefix (L-1)
#
# Receptor scalar:
#   a(x) at (layer_star, head_star) = < [ctx_t*, 1], u_vec >
#   where ctx = pat @ x_ln1 and u_vec is OV.SVD.U[:, sv_idx] from svd_cache.pt
#
# Outputs (in out_dir):
#   b_router_single_node_{direction}.json
#   b_router_greedy_{direction}.json
#   b_router_summary_{direction}.json
#
# Example:
#   python gp_b_activation_patching_router.py \
#     --data_dir data_main \
#     --pair_csvs train_1k_gp.csv,test_gp.csv \
#     --out_dir outputs/gp \
#     --layer_star 10 --head_star 9 --sv_idx 0 \
#     --N_pairs 256 --batch_size 64 --candidate_layers_max 9 \
#     --slots first,last,pred --topK 50 --Jmax 12 --random_trials 30 \
#     --delta_eps 1e-6 --greedy_score per_example \
#     --device cuda --save_json 1

from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer

import train_gp_masks_and_dump_ov_logit_receptors as gp


SLOT_NAME_TO_ID = {"first": 0, "last": 1, "pred": 2}
SLOT_ID_TO_NAME = {0: "first", 1: "last", 2: "pred"}

DIR_HE_TO_SHE = "he_to_she"
DIR_SHE_TO_HE = "she_to_he"
DIRS = (DIR_HE_TO_SHE, DIR_SHE_TO_HE)


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def parse_slots(s: str) -> List[int]:
    s = s.strip().lower()
    if not s:
        return [0, 1, 2]
    parts = [p.strip() for p in (s.split(",") if "," in s else s.split()) if p.strip()]
    out = []
    for p in parts:
        if p not in SLOT_NAME_TO_ID:
            raise ValueError(f"Unknown slot '{p}'. Valid: first,last,pred")
        out.append(SLOT_NAME_TO_ID[p])
    return sorted(set(out))


def batches_idx(n: int, batch_size: int):
    for i in range(0, n, batch_size):
        j = min(n, i + batch_size)
        yield i, j


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

        clean = r["prefix"]
        corr = r["corr_prefix"]
        ids_c = tokenizer.encode(clean, add_special_tokens=False)
        ids_k = tokenizer.encode(corr, add_special_tokens=False)
        if len(ids_c) != len(ids_k):
            continue

        L = len(ids_c)
        if L == 0:
            continue

        diff = [t for t in range(L) if ids_c[t] != ids_k[t]]
        if len(diff) == 0:
            continue

        pairs.append(PairRec(
            clean_ids=ids_c,
            corr_ids=ids_k,
            L=L,
            pos_first=diff[0],
            pos_last=diff[-1],
            pos_pred=L - 1,
            direction=direction,
        ))
    return pairs


def pad_batch_ids(tokenizer: GPT2TokenizerFast, ids_list: List[List[int]], pad_len: int, device: str):
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id is None; set pad_token to eos.")
    tokens, _attn, last_idx = gp._pad_to_length(ids_list, pad_len, pad_id, device)
    return tokens, last_idx


def make_pair_batch(tokenizer: GPT2TokenizerFast, pairs_batch: List[PairRec], device: str):
    pad_len = max(p.L for p in pairs_batch)
    clean_ids_list = [p.clean_ids for p in pairs_batch]
    corr_ids_list = [p.corr_ids for p in pairs_batch]

    tokens_clean, last_clean = pad_batch_ids(tokenizer, clean_ids_list, pad_len, device)
    tokens_corr, last_corr = pad_batch_ids(tokenizer, corr_ids_list, pad_len, device)

    pos_slots = torch.tensor(
        [[p.pos_first, p.pos_last, p.pos_pred] for p in pairs_batch],
        dtype=torch.long,
        device=device
    )
    return tokens_clean, tokens_corr, last_clean, pos_slots


@torch.no_grad()
def forward_ai_with_v_patches(
    model: HookedTransformer,
    tokens: torch.Tensor,
    last_idx: torch.Tensor,
    pos_slots: torch.Tensor,
    layer_star: int,
    head_star: int,
    u_vec: torch.Tensor,
    patches_by_layer: Dict[int, List[Tuple[int, int]]],
    clean_v_cache_by_layer: Dict[int, torch.Tensor],
) -> torch.Tensor:
    device = tokens.device
    act_fn = gp.get_act_fn(model)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    if not (0 <= layer_star < n_layers):
        raise ValueError(f"layer_star out of range: {layer_star} (n_layers={n_layers})")
    if not (0 <= head_star < n_heads):
        raise ValueError(f"head_star out of range: {head_star} (n_heads={n_heads})")

    B, S = tokens.shape
    causal = gp.make_causal_mask(S, device=str(device))
    x = model.embed(tokens) + model.pos_embed(tokens)

    # Run layers 0..layer_star-1 with optional V patches
    for l in range(layer_star):
        block = model.blocks[l]
        attn = block.attn
        mlp = block.mlp

        x_ln1 = block.ln1(x)

        head_to_slots: Dict[int, List[int]] = {}
        if l in patches_by_layer:
            for (h_patch, slot_id) in patches_by_layer[l]:
                head_to_slots.setdefault(int(h_patch), []).append(int(slot_id))

        head_outs = []
        for h in range(n_heads):
            pat = gp.attention_pattern_original(
                x_ln1,
                attn.W_Q[h], attn.b_Q[h],
                attn.W_K[h], attn.b_K[h],
                causal,
                d_head
            )
            ctx = pat @ x_ln1
            v = (ctx @ attn.W_V[h]) + attn.b_V[h]

            if h in head_to_slots:
                cache = clean_v_cache_by_layer[l]  # (B,3,H,Dh)
                for slot_id in head_to_slots[h]:
                    pos = pos_slots[:, slot_id]
                    v[torch.arange(B, device=device), pos, :] = cache[:, slot_id, h, :].to(v.dtype)

            head_outs.append(v @ attn.W_O[h])

        x = x + (torch.stack(head_outs, dim=0).sum(dim=0) + attn.b_O)

        x_ln2 = block.ln2(x)
        pre = (x_ln2 @ mlp.W_in) + mlp.b_in
        h_act = act_fn(pre)
        x = x + ((h_act @ mlp.W_out) + mlp.b_out)

    # At layer_star, compute a(x) from head_star context
    block = model.blocks[layer_star]
    attn = block.attn
    x_ln1 = block.ln1(x)

    pat = gp.attention_pattern_original(
        x_ln1,
        attn.W_Q[head_star], attn.b_Q[head_star],
        attn.W_K[head_star], attn.b_K[head_star],
        causal,
        d_head
    )
    ctx = pat @ x_ln1
    ctx_t = ctx[torch.arange(B, device=device), last_idx, :]
    ones = torch.ones((B, 1), device=device, dtype=ctx_t.dtype)
    nu = torch.cat([ctx_t, ones], dim=-1)
    return (nu * u_vec[None, :]).sum(dim=-1)


@torch.no_grad()
def cache_clean_v_slots_for_layer(
    model: HookedTransformer,
    tokens_clean: torch.Tensor,
    pos_slots: torch.Tensor,
    layer_to_cache: int,
) -> torch.Tensor:
    device = tokens_clean.device
    act_fn = gp.get_act_fn(model)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    if not (0 <= layer_to_cache < n_layers):
        raise ValueError(f"layer_to_cache out of range: {layer_to_cache}")

    B, S = tokens_clean.shape
    causal = gp.make_causal_mask(S, device=str(device))
    x = model.embed(tokens_clean) + model.pos_embed(tokens_clean)

    for l in range(layer_to_cache):
        block = model.blocks[l]
        attn = block.attn
        mlp = block.mlp

        x_ln1 = block.ln1(x)
        head_outs = []
        for h in range(n_heads):
            pat = gp.attention_pattern_original(
                x_ln1,
                attn.W_Q[h], attn.b_Q[h],
                attn.W_K[h], attn.b_K[h],
                causal,
                d_head
            )
            ctx = pat @ x_ln1
            v = (ctx @ attn.W_V[h]) + attn.b_V[h]
            head_outs.append(v @ attn.W_O[h])
        x = x + (torch.stack(head_outs, dim=0).sum(dim=0) + attn.b_O)

        x_ln2 = block.ln2(x)
        pre = (x_ln2 @ mlp.W_in) + mlp.b_in
        h_act = act_fn(pre)
        x = x + ((h_act @ mlp.W_out) + mlp.b_out)

    block = model.blocks[layer_to_cache]
    attn = block.attn
    x_ln1 = block.ln1(x)

    v_slots = torch.empty((B, 3, n_heads, d_head), device=device, dtype=torch.float16)
    for h in range(n_heads):
        pat = gp.attention_pattern_original(
            x_ln1,
            attn.W_Q[h], attn.b_Q[h],
            attn.W_K[h], attn.b_K[h],
            causal,
            d_head
        )
        ctx = pat @ x_ln1
        v = (ctx @ attn.W_V[h]) + attn.b_V[h]
        for slot_id in range(3):
            pos = pos_slots[:, slot_id]
            vv = v[torch.arange(B, device=device), pos, :]
            v_slots[:, slot_id, h, :] = vv.to(torch.float16)

    
@torch.no_grad()
def run_to_pre_layer_x(model: HookedTransformer, tokens: torch.Tensor, stop_layer: int) -> torch.Tensor:
    """
    Returns x after finishing layers [0 .. stop_layer-1], i.e. the residual stream input to ln1 at stop_layer.
    Must match the exact math in forward_ai_with_v_patches for layers < layer_star.
    """
    device = tokens.device
    act_fn = gp.get_act_fn(model)
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    B, S = tokens.shape
    causal = gp.make_causal_mask(S, device=str(device))

    x = model.embed(tokens) + model.pos_embed(tokens)

    for l in range(stop_layer):
        block = model.blocks[l]
        attn = block.attn
        mlp = block.mlp

        x_ln1 = block.ln1(x)
        head_outs = []
        for h in range(n_heads):
            pat = gp.attention_pattern_original(
                x_ln1,
                attn.W_Q[h], attn.b_Q[h],
                attn.W_K[h], attn.b_K[h],
                causal,
                d_head
            )
            ctx = pat @ x_ln1
            v = (ctx @ attn.W_V[h]) + attn.b_V[h]
            head_outs.append(v @ attn.W_O[h])

        x = x + (torch.stack(head_outs, dim=0).sum(dim=0) + attn.b_O)

        x_ln2 = block.ln2(x)
        pre = (x_ln2 @ mlp.W_in) + mlp.b_in
        h_act = act_fn(pre)
        x = x + ((h_act @ mlp.W_out) + mlp.b_out)

    return x


@torch.no_grad()
def compute_ai_with_parent_patch(
    model: HookedTransformer,
    tokens_clean: torch.Tensor,
    tokens_corr: torch.Tensor,
    last_idx: torch.Tensor,
    pos_slots: torch.Tensor,
    layer_star: int,
    head_star: int,
    u_vec: torch.Tensor,
    patch_ln1_mode: str,      # "none" | "slots" | "all_upto_last"
    patch_patrow_mode: str,   # "none" | "tstar_row"
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs clean and corrupt up to layer_star, then patches *parents* of a(x) at layer_star/head_star
    using clean values computed at the SAME layer_star.

    Returns:
      a_clean (B,), a_corr (B,), a_patched (B,)
    """
    device = tokens_corr.device
    B, S = tokens_corr.shape
    d_head = model.cfg.d_head
    causal = gp.make_causal_mask(S, device=str(device))

    # 1) x_pre at layer_star (input to ln1) for clean and corrupt
    x_pre_clean = run_to_pre_layer_x(model, tokens_clean, stop_layer=layer_star)
    x_pre_corr = run_to_pre_layer_x(model, tokens_corr, stop_layer=layer_star)

    # 2) ln1 at layer_star
    block = model.blocks[layer_star]
    attn = block.attn

    x_ln1_clean = block.ln1(x_pre_clean)  # (B,S,D)
    x_ln1_corr = block.ln1(x_pre_corr)    # (B,S,D)

    # 3) pat for head_star on clean and corrupt
    pat_clean = gp.attention_pattern_original(
        x_ln1_clean,
        attn.W_Q[head_star], attn.b_Q[head_star],
        attn.W_K[head_star], attn.b_K[head_star],
        causal,
        d_head
    )  # (B,S,S)

    pat_corr = gp.attention_pattern_original(
        x_ln1_corr,
        attn.W_Q[head_star], attn.b_Q[head_star],
        attn.W_K[head_star], attn.b_K[head_star],
        causal,
        d_head
    )  # (B,S,S)

    # 4) baseline a_clean and a_corr
    def ai_from(x_ln1: torch.Tensor, pat: torch.Tensor) -> torch.Tensor:
        ctx = pat @ x_ln1
        ctx_t = ctx[torch.arange(B, device=device), last_idx, :]  # (B,D)
        ones = torch.ones((B, 1), device=device, dtype=ctx_t.dtype)
        nu = torch.cat([ctx_t, ones], dim=-1)  # (B,D+1)
        return (nu * u_vec[None, :]).sum(dim=-1)

    a_clean = ai_from(x_ln1_clean, pat_clean)
    a_corr = ai_from(x_ln1_corr, pat_corr)

    # 5) Patch x_ln1 (parent #1)
    x_ln1_patched = x_ln1_corr.clone()

    if patch_ln1_mode == "slots":
        for slot_id in [0, 1, 2]:
            pos = pos_slots[:, slot_id]
            x_ln1_patched[torch.arange(B, device=device), pos, :] = x_ln1_clean[torch.arange(B, device=device), pos, :]
    elif patch_ln1_mode == "all_upto_last":
        mask = (torch.arange(S, device=device)[None, :] <= last_idx[:, None])  # (B,S)
        x_ln1_patched = torch.where(mask[:, :, None], x_ln1_clean, x_ln1_patched)
    elif patch_ln1_mode == "none":
        pass
    else:
        raise ValueError(f"bad patch_ln1_mode: {patch_ln1_mode}")

    # 5b) Recompute pat on patched ln1, then optionally patch row at t* (parent #2)
    pat_patched = gp.attention_pattern_original(
        x_ln1_patched,
        attn.W_Q[head_star], attn.b_Q[head_star],
        attn.W_K[head_star], attn.b_K[head_star],
        causal,
        d_head
    )

    if patch_patrow_mode == "tstar_row":
        pat_patched[torch.arange(B, device=device), last_idx, :] = pat_clean[torch.arange(B, device=device), last_idx, :]
    elif patch_patrow_mode == "none":
        pass
    else:
        raise ValueError(f"bad patch_patrow_mode: {patch_patrow_mode}")

    a_patched = ai_from(x_ln1_patched, pat_patched)
    return a_clean, a_corr, a_patched


def compute_restore_stats(a_clean: torch.Tensor, a_corr: torch.Tensor, a_patch: torch.Tensor, delta_eps: float) -> Dict[str, float]:
    """
    Compute mean-based and per-example restoration stats for already-computed a_* tensors (CPU).
    """
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


return v_slots


def restore_frac_mean_based(a_patch_mean: float, a_corr_mean: float, a_clean_mean: float, eps: float = 1e-12) -> float:
    denom = (a_clean_mean - a_corr_mean)
    if abs(denom) < eps:
        return float("nan")
    return float((a_patch_mean - a_corr_mean) / denom)


@torch.no_grad()
def compute_ai_for_pairs(
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    pairs: List[PairRec],
    layer_star: int,
    head_star: int,
    u_vec: torch.Tensor,
    batch_size: int,
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    N = len(pairs)
    a_clean_all, a_corr_all = [], []

    empty_patches: Dict[int, List[Tuple[int, int]]] = {}
    empty_cache: Dict[int, torch.Tensor] = {}

    for i, j in batches_idx(N, batch_size):
        batch = pairs[i:j]
        tokens_clean, tokens_corr, last_idx, pos_slots = make_pair_batch(tokenizer, batch, device)

        a_clean = forward_ai_with_v_patches(
            model, tokens_clean, last_idx, pos_slots,
            layer_star=layer_star, head_star=head_star, u_vec=u_vec,
            patches_by_layer=empty_patches, clean_v_cache_by_layer=empty_cache
        )
        a_corr = forward_ai_with_v_patches(
            model, tokens_corr, last_idx, pos_slots,
            layer_star=layer_star, head_star=head_star, u_vec=u_vec,
            patches_by_layer=empty_patches, clean_v_cache_by_layer=empty_cache
        )
        a_clean_all.append(a_clean.detach().float().cpu())
        a_corr_all.append(a_corr.detach().float().cpu())

    return torch.cat(a_clean_all, dim=0), torch.cat(a_corr_all, dim=0)


@torch.no_grad()
def eval_patchset_stats(
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    selected_pairs: List[PairRec],
    a_clean_sel: torch.Tensor,
    a_corr_sel: torch.Tensor,
    layer_star: int,
    head_star: int,
    u_vec: torch.Tensor,
    patches_by_layer: Dict[int, List[Tuple[int, int]]],
    clean_cache_layers_cpu: Dict[int, torch.Tensor],
    batch_size: int,
    device: str,
    delta_eps: float,
) -> Dict[str, float]:
    N = len(selected_pairs)
    a_patch_all = []
    restore_vals: List[float] = []

    a_clean_mean = float(a_clean_sel.mean().item())
    a_corr_mean = float(a_corr_sel.mean().item())

    for i, j in batches_idx(N, batch_size):
        batch_pairs = selected_pairs[i:j]
        _tokens_clean, tokens_corr, last_idx, pos_slots = make_pair_batch(tokenizer, batch_pairs, device)

        cache_by_layer: Dict[int, torch.Tensor] = {}
        for l in patches_by_layer.keys():
            cache_by_layer[l] = clean_cache_layers_cpu[l][i:j].to(device)

        a_patch = forward_ai_with_v_patches(
            model, tokens_corr, last_idx, pos_slots,
            layer_star=layer_star, head_star=head_star, u_vec=u_vec,
            patches_by_layer=patches_by_layer, clean_v_cache_by_layer=cache_by_layer
        ).detach().float().cpu()
        a_patch_all.append(a_patch)

        delta = (a_clean_sel[i:j] - a_corr_sel[i:j])
        num = (a_patch - a_corr_sel[i:j])
        valid = delta.abs() > float(delta_eps)
        if valid.any():
            restore_vals.extend((num[valid] / delta[valid]).tolist())

    a_patch_cat = torch.cat(a_patch_all, dim=0)
    a_patch_mean = float(a_patch_cat.mean().item())
    restore_mb = restore_frac_mean_based(a_patch_mean, a_corr_mean, a_clean_mean)

    if len(restore_vals) == 0:
        r_mean, r_std, n_valid = float("nan"), float("nan"), 0
    else:
        rr = torch.tensor(restore_vals, dtype=torch.float32)
        r_mean = float(rr.mean().item())
        r_std = float(rr.std(unbiased=False).item())
        n_valid = int(rr.numel())

    return {
        "a_patch_mean": a_patch_mean,
        "restore_mean_based": float(restore_mb),
        "restore_per_example_mean": float(r_mean),
        "restore_per_example_std": float(r_std),
        "n_valid": int(n_valid),
        "a_clean_mean": float(a_clean_mean),
        "a_corr_mean": float(a_corr_mean),
    }


def candidate_list(candidate_layers_max: int, n_heads: int, slot_ids: List[int]) -> List[Tuple[int, int, int]]:
    out = []
    for l in range(candidate_layers_max + 1):
        for h in range(n_heads):
            for s in slot_ids:
                out.append((l, h, s))
    return out


def pick_score(r: Dict[str, float], mode: str) -> float:
    if mode == "per_example":
        return float(r["restore_per_example_mean"])
    if mode == "mean_based":
        return float(r["restore_mean_based"])
    raise ValueError(f"Unknown score mode: {mode}")


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
    slot_ids_requested: List[int],
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
        raise ValueError(f"--sv_idx {sv_idx} out of range (0..{svd_star.r-1}) for ov[L*][H*].r={svd_star.r}")
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

    same_fl = sum(1 for p in pairs_sel if p.pos_first == p.pos_last)
    print(f"[SLOTS] first==last in selected pool: {same_fl}/{len(pairs_sel)} = {100.0*same_fl/max(1,len(pairs_sel)):.2f}%")

    slot_ids = list(slot_ids_requested)
    if (0 in slot_ids) and (1 in slot_ids) and (same_fl == len(pairs_sel)):
        slot_ids = [s for s in slot_ids if s != 1]
        print("[SLOTS] All selected pairs have first==last -> dropping slot 'last' for this group.")

    n_heads = model.cfg.n_heads
    candidates = candidate_list(candidate_layers_max, n_heads, slot_ids)
    print(f"[CAND] layers=0..{candidate_layers_max} ({candidate_layers_max+1}) * heads={n_heads} * slots={len(slot_ids)} => {len(candidates)} candidates")


    # ------------------------------------------------------------
    # SANITY CHECK (upper bound): direct-parent patch at layer_star/head_star
    # If this fails, router search is not interpretable.
    # ------------------------------------------------------------
    print("\n[SANITY-PARENT] direct-parent patch upper bound check (must pass)")

    parent_modes = [
        ("ln1_slots", "slots", "none"),
        ("ln1_all", "all_upto_last", "none"),
        ("patrow_only", "none", "tstar_row"),
        ("ln1_all+patrow", "all_upto_last", "tstar_row"),
    ]

    stats_parent: Dict[str, Dict[str, float]] = {}

    # Null/self check: no patches => a_patched must equal a_corr (numerical tolerance)
    all_corr_0, all_patch_0 = [], []
    for i, j in batches_idx(N_sel, batch_size):
        batch_pairs = pairs_sel[i:j]
        tokens_clean, tokens_corr, last_idx, pos_slots = make_pair_batch(tokenizer, batch_pairs, device)
        _a_clean, a_corr, a_patch = compute_ai_with_parent_patch(
            model=model,
            tokens_clean=tokens_clean,
            tokens_corr=tokens_corr,
            last_idx=last_idx,
            pos_slots=pos_slots,
            layer_star=layer_star,
            head_star=head_star,
            u_vec=u_vec,
            patch_ln1_mode="none",
            patch_patrow_mode="none",
        )
        all_corr_0.append(a_corr.detach().float().cpu())
        all_patch_0.append(a_patch.detach().float().cpu())
    a_corr0 = torch.cat(all_corr_0)
    a_patch0 = torch.cat(all_patch_0)
    max_abs = float((a_corr0 - a_patch0).abs().max().item())
    print(f"  null-check (none/none): max|a_patch-a_corr|={max_abs:.3e} (should be tiny)")
    if max_abs > 1e-4:
        raise RuntimeError("SANITY-PARENT FAILED: null parent-patch does not reproduce a_corr. Likely bug in sanity code.")

    # Evaluate parent-patch modes
    for name, ln1_mode, pat_mode in parent_modes:
        all_clean, all_corr, all_patch = [], [], []
        for i, j in batches_idx(N_sel, batch_size):
            batch_pairs = pairs_sel[i:j]
            tokens_clean, tokens_corr, last_idx, pos_slots = make_pair_batch(tokenizer, batch_pairs, device)

            a_clean_b, a_corr_b, a_patch_b = compute_ai_with_parent_patch(
                model=model,
                tokens_clean=tokens_clean,
                tokens_corr=tokens_corr,
                last_idx=last_idx,
                pos_slots=pos_slots,
                layer_star=layer_star,
                head_star=head_star,
                u_vec=u_vec,
                patch_ln1_mode=ln1_mode,
                patch_patrow_mode=pat_mode,
            )
            all_clean.append(a_clean_b.detach().float().cpu())
            all_corr.append(a_corr_b.detach().float().cpu())
            all_patch.append(a_patch_b.detach().float().cpu())

        a_clean = torch.cat(all_clean)
        a_corr = torch.cat(all_corr)
        a_patch = torch.cat(all_patch)

        st = compute_restore_stats(a_clean, a_corr, a_patch, delta_eps=delta_eps)
        stats_parent[name] = st
        print(f"  {name:>12}: mean_based={st['mean_based']:+.4f} | per_ex={st['per_ex_mean']:+.4f}±{st['per_ex_std']:.4f} (n={st['n_valid']})")

    # Hard pass/fail: ln1_all+patrow should strongly restore a(x)
    ub = stats_parent["ln1_all+patrow"]["mean_based"]
    if (not math.isnan(ub)) and (ub < 0.90):
        raise RuntimeError(
            f"SANITY-PARENT FAILED: direct-parent patch did not restore a(x) (mean_based={ub:.3f} < 0.90). "
            "This suggests a mismatch between how a(x) is computed and what is being patched; stop and debug."
        )

    print("[SANITY-PARENT] PASS")

    print("[CACHE] caching clean V-slots for candidate layers ...")
    clean_cache_layers_cpu: Dict[int, torch.Tensor] = {}
    for l in range(candidate_layers_max + 1):
        chunks = []
        for i, j in batches_idx(N_sel, batch_size):
            batch_pairs = pairs_sel[i:j]
            tokens_clean, _tokens_corr, _last_idx, pos_slots = make_pair_batch(tokenizer, batch_pairs, device)
            v_slots = cache_clean_v_slots_for_layer(model, tokens_clean, pos_slots, layer_to_cache=l)
            chunks.append(v_slots.detach().cpu())
        clean_cache_layers_cpu[l] = torch.cat(chunks, dim=0).contiguous()
        print(f"  cached layer {l:>2}: {tuple(clean_cache_layers_cpu[l].shape)} dtype={clean_cache_layers_cpu[l].dtype}")

    stats0 = eval_patchset_stats(
        model=model, tokenizer=tokenizer, selected_pairs=pairs_sel,
        a_clean_sel=a_clean_sel, a_corr_sel=a_corr_sel,
        layer_star=layer_star, head_star=head_star, u_vec=u_vec,
        patches_by_layer={},
        clean_cache_layers_cpu=clean_cache_layers_cpu,
        batch_size=batch_size, device=device,
        delta_eps=delta_eps
    )
    print(f"[SANITY] patch-nothing: mean_based={stats0['restore_mean_based']:+.6g} | "
          f"per_ex={stats0['restore_per_example_mean']:+.6g}±{stats0['restore_per_example_std']:.6g} (n={stats0['n_valid']})")

    print("\n[SINGLE] evaluating single-node candidates ...")
    single_results: List[Dict[str, Any]] = []
    for idx_c, (l, h, s) in enumerate(candidates):
        patches_by_layer = {l: [(h, s)]}
        st = eval_patchset_stats(
            model=model, tokenizer=tokenizer, selected_pairs=pairs_sel,
            a_clean_sel=a_clean_sel, a_corr_sel=a_corr_sel,
            layer_star=layer_star, head_star=head_star, u_vec=u_vec,
            patches_by_layer=patches_by_layer,
            clean_cache_layers_cpu=clean_cache_layers_cpu,
            batch_size=batch_size, device=device,
            delta_eps=delta_eps
        )
        single_results.append({
            "layer": int(l),
            "head": int(h),
            "slot_id": int(s),
            "slot": SLOT_ID_TO_NAME[int(s)],
            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in st.items()},
        })
        if (idx_c + 1) % 25 == 0 or (idx_c + 1) == len(candidates):
            print(f"  done {idx_c+1}/{len(candidates)}")

    single_results_sorted = sorted(
        single_results,
        key=lambda r: safe_key(pick_score(r, greedy_score)),
        reverse=True
    )

    print(f"\n[TOP-20] single-node candidates by score={greedy_score}")
    print("rank\tL\tH\tslot\tmean_based\tper_ex(mean±std)\ta_patch_mean")
    for i, r in enumerate(single_results_sorted[:20], start=1):
        print(
            f"{i}\t{r['layer']}\t{r['head']}\t{r['slot']}\t"
            f"{r['restore_mean_based']:+.4f}\t"
            f"{r['restore_per_example_mean']:+.4f}±{r['restore_per_example_std']:.4f}\t"
            f"{r['a_patch_mean']:+.6g}"
        )

    if int(save_json) == 1:
        out_path = out_dir / f"b_router_single_node_{direction}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "experiment": "B_router_activation_patching",
                    "direction": direction,
                    "selected_N_pairs": int(N_sel),
                    "candidate_layers_max": int(candidate_layers_max),
                    "slots_requested": [SLOT_ID_TO_NAME[s] for s in slot_ids_requested],
                    "slots_used": [SLOT_ID_TO_NAME[s] for s in slot_ids],
                    "layer_star": int(layer_star),
                    "head_star": int(head_star),
                    "sv_idx": int(sv_idx),
                    "delta_eps": float(delta_eps),
                    "score_mode": greedy_score,
                    "baseline_selected": {
                        "a_clean_mean": float(a_clean_sel.mean().item()),
                        "a_corr_mean": float(a_corr_sel.mean().item()),
                        "delta_mean": float(delta_sel.mean().item()),
                        "abs_delta_mean": float(delta_sel.abs().mean().item()),
                    }
                },
                "results_sorted": single_results_sorted,
            }, f, indent=2)
        print(f"[SAVE] wrote: {out_path}")

    print("\n[GREEDY] building a small router set ...")
    pool = single_results_sorted[:min(int(topK), len(single_results_sorted))]
    chosen: List[Dict[str, Any]] = []
    chosen_set = set()
    greedy_curve: List[Dict[str, Any]] = []

    for t in range(1, int(Jmax) + 1):
        best_c = None
        best_stats = None
        best_score = -1e18

        for cand in pool:
            key = (cand["layer"], cand["head"], cand["slot_id"])
            if key in chosen_set:
                continue

            patches_by_layer: Dict[int, List[Tuple[int, int]]] = {}
            for c in chosen:
                patches_by_layer.setdefault(int(c["layer"]), []).append((int(c["head"]), int(c["slot_id"])))
            patches_by_layer.setdefault(int(cand["layer"]), []).append((int(cand["head"]), int(cand["slot_id"])))

            st = eval_patchset_stats(
                model=model, tokenizer=tokenizer, selected_pairs=pairs_sel,
                a_clean_sel=a_clean_sel, a_corr_sel=a_corr_sel,
                layer_star=layer_star, head_star=head_star, u_vec=u_vec,
                patches_by_layer=patches_by_layer,
                clean_cache_layers_cpu=clean_cache_layers_cpu,
                batch_size=batch_size, device=device,
                delta_eps=delta_eps
            )
            sc = pick_score(st, greedy_score)
            if (not math.isnan(sc)) and (sc > best_score):
                best_score = sc
                best_c = cand
                best_stats = st

        if best_c is None:
            print("[GREEDY] No more candidates improved (or all were NaN). Stopping.")
            break

        chosen.append(best_c)
        chosen_set.add((best_c["layer"], best_c["head"], best_c["slot_id"]))
        greedy_curve.append({
            "step": int(t),
            "added": {"layer": int(best_c["layer"]), "head": int(best_c["head"]), "slot": best_c["slot"], "slot_id": int(best_c["slot_id"])},
            "score_mode": greedy_score,
            "score": float(best_score),
            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in best_stats.items()},
        })

        print(f"  step {t:>2}: add (L={best_c['layer']},H={best_c['head']},slot={best_c['slot']}) "
              f"=> score={best_score:+.4f} | mean_based={best_stats['restore_mean_based']:+.4f} "
              f"| per_ex={best_stats['restore_per_example_mean']:+.4f}±{best_stats['restore_per_example_std']:.4f}")

        if pick_score(best_stats, greedy_score) >= float(stop_frac):
            print(f"[GREEDY] Reached stop_frac={stop_frac} under score_mode={greedy_score}.")
            break

    if int(save_json) == 1:
        out_path = out_dir / f"b_router_greedy_{direction}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "experiment": "B_router_activation_patching_greedy",
                    "direction": direction,
                    "topK_pool": int(min(int(topK), len(single_results_sorted))),
                    "Jmax": int(Jmax),
                    "stop_frac": float(stop_frac),
                    "score_mode": greedy_score,
                },
                "chosen_set": [{"layer": int(c["layer"]), "head": int(c["head"]), "slot": c["slot"], "slot_id": int(c["slot_id"])} for c in chosen],
                "greedy_curve": greedy_curve,
            }, f, indent=2)
        print(f"[SAVE] wrote: {out_path}")

    print("\n[RANDOM] random baseline band ...")
    full_candidates = candidates[:]
    m_max = len(chosen)
    random_summary = []

    for m in range(1, m_max + 1):
        scores = []
        for _ in range(int(random_trials)):
            rs = random.sample(full_candidates, k=m)
            patches_by_layer: Dict[int, List[Tuple[int, int]]] = {}
            for (l, h, s) in rs:
                patches_by_layer.setdefault(l, []).append((h, s))

            st = eval_patchset_stats(
                model=model, tokenizer=tokenizer, selected_pairs=pairs_sel,
                a_clean_sel=a_clean_sel, a_corr_sel=a_corr_sel,
                layer_star=layer_star, head_star=head_star, u_vec=u_vec,
                patches_by_layer=patches_by_layer,
                clean_cache_layers_cpu=clean_cache_layers_cpu,
                batch_size=batch_size, device=device,
                delta_eps=delta_eps
            )
            sc = pick_score(st, greedy_score)
            if not math.isnan(sc):
                scores.append(float(sc))

        if len(scores) == 0:
            mean_sc, std_sc = float("nan"), float("nan")
        else:
            tt = torch.tensor(scores, dtype=torch.float32)
            mean_sc = float(tt.mean().item())
            std_sc = float(tt.std(unbiased=False).item())

        greedy_sc = float(greedy_curve[m - 1]["score"]) if (m - 1) < len(greedy_curve) else float("nan")
        random_summary.append({"m": int(m), "greedy_score": float(greedy_sc), "random_mean": float(mean_sc), "random_std": float(std_sc), "trials_used": int(len(scores))})
        print(f"  m={m:>2}: greedy={greedy_sc:+.4f} | random={mean_sc:+.4f}±{std_sc:.4f} (n={len(scores)})")

    if int(save_json) == 1:
        out_path = out_dir / f"b_router_summary_{direction}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "experiment": "B_router_activation_patching_summary",
                    "direction": direction,
                    "selected_N_pairs": int(N_sel),
                    "layer_star": int(layer_star),
                    "head_star": int(head_star),
                    "sv_idx": int(sv_idx),
                    "candidate_layers_max": int(candidate_layers_max),
                    "slots_used": [SLOT_ID_TO_NAME[s] for s in slot_ids],
                    "delta_eps": float(delta_eps),
                    "score_mode": greedy_score,
                },
                "greedy_curve": greedy_curve,
                "random_baseline": random_summary,
                "top20_single_nodes": single_results_sorted[:20],
            }, f, indent=2)
        print(f"[SAVE] wrote: {out_path}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="data_main")
    ap.add_argument("--train_csv", type=str, default="train_1k_gp.csv")
    ap.add_argument("--test_csv", type=str, default="test_gp.csv")
    ap.add_argument("--pair_csvs", type=str, default=None,
                    help="Comma-separated CSV filenames to build the pair pool from. Default: train_csv,test_csv. You can include val_gp.csv too.")
    ap.add_argument("--out_dir", type=str, default="outputs/gp")

    ap.add_argument("--layer_star", type=int, required=True)
    ap.add_argument("--head_star", type=int, required=True)
    ap.add_argument("--sv_idx", type=int, required=True)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    ap.add_argument("--N_pairs", type=int, default=256, help="Top |Δa| pairs PER DIRECTION group.")
    ap.add_argument("--slots", type=str, default="first,last,pred")
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

    if args.pair_csvs is None:
        csv_names = [args.train_csv, args.test_csv]
    else:
        csv_names = [x.strip() for x in args.pair_csvs.split(",") if x.strip()]

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
        raise RuntimeError("No aligned pairs found. Ensure clean/corr token lengths match and pronouns are he/she with opposite labels.")

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    svd_path = out_dir / "svd_cache.pt"
    if not svd_path.exists():
        print(f"[SVD] Missing {svd_path}. Building SVD cache now (CPU)...")
        gp.build_svd_cache(model, str(svd_path), svd_eps=1e-6, device_for_svd="cpu")
    _qk, ov, _mlp_in, _mlp_out, _rank_total_ov = gp.load_svd_cache(str(svd_path), device=device)

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

    slot_ids_requested = parse_slots(args.slots)

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
            slot_ids_requested=slot_ids_requested,
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
