# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# \"\"
# FILE: gp_b_activation_patching_router.py

# EXPERIMENT B: Activation patching / denoising to find upstream router nodes that set the receptor scalar a(x).

# Goal:
#   Find a small upstream set S of nodes c=(L, H, SLOT) such that patching those node activations
#   from CLEAN into CORRUPT restores the receptor scalar a(x) at (L*,H*,SV*) toward its clean value.

# This implements the 3-pass patching structure:
#   (1) clean run (cache V activations to transplant)
#   (2) corrupt run baseline (measure receptor scalar)
#   (3) corrupt-with-patch run (overwrite some V activations with clean cached values)

# IMPORTANT CONVENTIONS (match train_gp_masks_and_dump_ov_logit_receptors.py):
#   - prefix normalization is rstrip() (NO trailing space)
#   - tokenization uses gp._encode_texts + gp._pad_to_length
#   - attention pattern uses gp.attention_pattern_original
#   - patched activation is the per-head V-activation:
#         v = (context_resid @ W_V[h]) + b_V[h]
#     where context_resid = pat @ x_ln1
#   - receptor scalar a(x) uses ctx at chosen (L*,H*) and augments [ctx_t*, 1] then dot with u_vec
#     where u_vec is OV SVD U[:,K*] from outputs/gp/svd_cache.pt

# Outputs (in --out_dir):
#   b_router_single_node.json    (all candidates scored; sorted)
#   b_router_greedy.json         (greedy set + curve)
#   b_router_summary.json        (overall summary, including random baseline)

# Run example (Kaggle):
#   python gp_b_activation_patching_router.py \
#     --data_dir data_main --train_csv train_1k_gp.csv --test_csv test_gp.csv \
#     --out_dir outputs/gp --layer_star 10 --head_star 9 --sv_idx 0 \
#     --N_pairs 256 --batch_size 64 --candidate_layers_max 9 \
#     --slots first,last,pred --topK 50 --Jmax 12 --random_trials 30 \
#     --device cuda --save_json 1
# \"\"

from __future__ import annotations

import argparse
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


SLOT_NAME_TO_ID = {"first": 0, "last": 1, "pred": 2}
SLOT_ID_TO_NAME = {0: "first", 1: "last", 2: "pred"}


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


@dataclass
class PairRec:
    clean_ids: List[int]
    corr_ids: List[int]
    L: int
    pos_first: int
    pos_last: int
    pos_pred: int


def build_pairs(tokenizer: GPT2TokenizerFast, rows: List[Dict[str, str]]) -> List[PairRec]:
    pairs: List[PairRec] = []
    for r in rows:
        clean = r["prefix"]
        corr = r["corr_prefix"]
        ids_c = tokenizer.encode(clean, add_special_tokens=False)
        ids_k = tokenizer.encode(corr, add_special_tokens=False)
        if len(ids_c) != len(ids_k):
            continue
        L = len(ids_c)
        if L == 0:
            continue
        diff = [p for p in range(L) if ids_c[p] != ids_k[p]]
        if len(diff) == 0:
            continue
        pairs.append(PairRec(
            clean_ids=ids_c,
            corr_ids=ids_k,
            L=L,
            pos_first=diff[0],
            pos_last=diff[-1],
            pos_pred=L - 1,
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

    return v_slots


def restore_frac(a_patch_mean: float, a_corr_mean: float, a_clean_mean: float, eps: float = 1e-12) -> float:
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
def eval_patchset_mean_ai(
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    selected_pairs: List[PairRec],                    # pool order
    layer_star: int,
    head_star: int,
    u_vec: torch.Tensor,
    patches_by_layer: Dict[int, List[Tuple[int, int]]],
    clean_cache_layers_cpu: Dict[int, torch.Tensor],  # layer -> (N_pairs,3,H,Dh) fp16 CPU aligned to pool order
    batch_size: int,
    device: str,
) -> float:
    N = len(selected_pairs)
    a_all = []

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
        )
        a_all.append(a_patch.detach().float().cpu())

    a = torch.cat(a_all, dim=0)
    return float(a.mean().item())


def candidate_list(candidate_layers_max: int, n_heads: int, slot_ids: List[int]) -> List[Tuple[int, int, int]]:
    out = []
    for l in range(candidate_layers_max + 1):
        for h in range(n_heads):
            for s in slot_ids:
                out.append((l, h, s))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="data_main")
    ap.add_argument("--train_csv", type=str, default="train_1k_gp.csv")
    ap.add_argument("--test_csv", type=str, default="test_gp.csv")
    ap.add_argument("--out_dir", type=str, default="outputs/gp")

    ap.add_argument("--layer_star", type=int, required=True)
    ap.add_argument("--head_star", type=int, required=True)
    ap.add_argument("--sv_idx", type=int, required=True)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    ap.add_argument("--N_pairs", type=int, default=256)
    ap.add_argument("--slots", type=str, default="first,last,pred")
    ap.add_argument("--candidate_layers_max", type=int, default=None)  # default: layer_star-1
    ap.add_argument("--topK", type=int, default=50)
    ap.add_argument("--Jmax", type=int, default=12)
    ap.add_argument("--stop_frac", type=float, default=0.8)
    ap.add_argument("--random_trials", type=int, default=30)
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

    # Regime A sanity prints
    he_ids = tokenizer.encode(" he", add_special_tokens=False)
    she_ids = tokenizer.encode(" she", add_special_tokens=False)
    print("[TOKENS] he_ids=", he_ids, "decoded=", tokenizer.decode(he_ids, clean_up_tokenization_spaces=False))
    print("[TOKENS] she_ids=", she_ids, "decoded=", tokenizer.decode(she_ids, clean_up_tokenization_spaces=False))

    train_rows = gp.load_gp_csv(str(data_dir / args.train_csv))
    test_rows = gp.load_gp_csv(str(data_dir / args.test_csv))
    print(f"[DATA] train_rows={len(train_rows)} test_rows={len(test_rows)}")

    test_pairs_all = build_pairs(tokenizer, test_rows)
    kept_pct = 100.0 * len(test_pairs_all) / max(1, len(test_rows))
    print(f"[PAIRS] aligned test pairs kept: {len(test_pairs_all)}/{len(test_rows)} = {kept_pct:.2f}%")
    if len(test_pairs_all) == 0:
        raise RuntimeError("No aligned pairs found (clean/corr token lengths must match).")

    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    svd_path = out_dir / "svd_cache.pt"
    if not svd_path.exists():
        print(f"[SVD] Missing {svd_path}. Building SVD cache now (CPU)...")
        gp.build_svd_cache(model, str(svd_path), svd_eps=1e-6, device_for_svd="cpu")

    _qk, ov, _mlp_in, _mlp_out, _rank_total_ov = gp.load_svd_cache(str(svd_path), device=device)

    Ls, Hs, Ks = args.layer_star, args.head_star, args.sv_idx
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    if not (0 <= Ls < n_layers):
        raise ValueError(f"--layer_star {Ls} out of range (0..{n_layers-1})")
    if not (0 <= Hs < n_heads):
        raise ValueError(f"--head_star {Hs} out of range (0..{n_heads-1})")

    svd_star = ov[Ls][Hs]
    if not (0 <= Ks < svd_star.r):
        raise ValueError(f"--sv_idx {Ks} out of range (0..{svd_star.r-1}) for ov[L*][H*].r={svd_star.r}")

    u_vec = svd_star.U[:, Ks].contiguous()
    print(f"[STAR] (L*,H*,K*)=({Ls},{Hs},{Ks}) | u_vec.shape={tuple(u_vec.shape)}")

    masks_path = Path(args.masks_path) if args.masks_path else (out_dir / "masks.pt")
    if masks_path.exists():
        masks_dict = torch.load(str(masks_path), map_location="cpu")
        mask_star = float(masks_dict["ov"][Ls][Hs][Ks].item())
        print(f"[MASK] mask_star={mask_star:.6g} from {masks_path}")
    else:
        print(f"[MASK] masks.pt not found at {masks_path} (ok). Skipping mask print.")

    print("\n[BASELINE] computing a_clean/a_corr for all aligned test pairs ...")
    a_clean_all, a_corr_all = compute_ai_for_pairs(
        model=model, tokenizer=tokenizer, pairs=test_pairs_all,
        layer_star=Ls, head_star=Hs, u_vec=u_vec,
        batch_size=args.batch_size, device=device
    )
    d = (a_clean_all - a_corr_all).abs()
    print(f"[BASELINE] |a_clean-a_corr|: mean={float(d.mean()):.6g}  median={float(d.median()):.6g}  max={float(d.max()):.6g}")

    N_pairs = min(int(args.N_pairs), len(test_pairs_all))
    top_vals, top_idx = torch.topk(d, k=N_pairs, largest=True)
    print(f"[SELECT] using top N_pairs={N_pairs} by |a_clean-a_corr| (min selected diff={float(top_vals.min()):.6g})")

    a_clean_sel = a_clean_all[top_idx]
    a_corr_sel = a_corr_all[top_idx]
    a_clean_mean = float(a_clean_sel.mean().item())
    a_corr_mean = float(a_corr_sel.mean().item())
    print(f"[SELECT] a_clean_mean={a_clean_mean:.6g}  a_corr_mean={a_corr_mean:.6g}  delta={a_clean_mean-a_corr_mean:+.6g}")

    selected_pairs = [test_pairs_all[int(i)] for i in top_idx.tolist()]

    slot_ids = parse_slots(args.slots)
    if len(slot_ids) == 0:
        raise ValueError("No slots selected.")

    cand_max = args.candidate_layers_max if args.candidate_layers_max is not None else (Ls - 1)
    if cand_max < 0:
        raise ValueError("candidate_layers_max < 0 (layer_star must be >= 1 for upstream routing).")
    if cand_max >= Ls:
        print(f"[WARN] candidate_layers_max={cand_max} >= layer_star={Ls}. Upstream routing usually uses < layer_star.")
    cand_max = min(cand_max, n_layers - 1)
    print(f"[CAND] candidate_layers_max={cand_max} | slots={[(sid, SLOT_ID_TO_NAME[sid]) for sid in slot_ids]}")

    print("\n[CACHE] caching clean V-slots for candidate layers ...")
    clean_cache_layers_cpu: Dict[int, torch.Tensor] = {}
    for l in range(cand_max + 1):
        chunks = []
        for i, j in batches_idx(N_pairs, args.batch_size):
            batch_pairs = selected_pairs[i:j]
            tokens_clean, _tokens_corr, _last_idx, pos_slots = make_pair_batch(tokenizer, batch_pairs, device)
            v_slots = cache_clean_v_slots_for_layer(model, tokens_clean, pos_slots, layer_to_cache=l)
            chunks.append(v_slots.detach().cpu())
        clean_cache_layers_cpu[l] = torch.cat(chunks, dim=0).contiguous()
        print(f"  cached layer {l:>2}: {tuple(clean_cache_layers_cpu[l].shape)} dtype={clean_cache_layers_cpu[l].dtype}")

    a_patch0 = eval_patchset_mean_ai(
        model=model, tokenizer=tokenizer, selected_pairs=selected_pairs,
        layer_star=Ls, head_star=Hs, u_vec=u_vec,
        patches_by_layer={}, clean_cache_layers_cpu=clean_cache_layers_cpu,
        batch_size=args.batch_size, device=device
    )
    rf0 = restore_frac(a_patch0, a_corr_mean, a_clean_mean)
    print(f"\n[SANITY] patch nothing: a_patch_mean={a_patch0:.6g}  RestoreFrac={rf0:+.6g} (should be ~0)")

    print("\n[SINGLE] evaluating single-node candidates ...")
    candidates = candidate_list(candidate_layers_max=cand_max, n_heads=n_heads, slot_ids=slot_ids)
    single_results: List[Dict[str, Any]] = []

    for idx_c, (l, h, s) in enumerate(candidates):
        patches_by_layer = {l: [(h, s)]}
        a_patch_mean = eval_patchset_mean_ai(
            model=model, tokenizer=tokenizer, selected_pairs=selected_pairs,
            layer_star=Ls, head_star=Hs, u_vec=u_vec,
            patches_by_layer=patches_by_layer,
            clean_cache_layers_cpu=clean_cache_layers_cpu,
            batch_size=args.batch_size, device=device
        )
        rf = restore_frac(a_patch_mean, a_corr_mean, a_clean_mean)
        single_results.append({
            "layer": int(l),
            "head": int(h),
            "slot_id": int(s),
            "slot": SLOT_ID_TO_NAME[int(s)],
            "a_patch_mean": float(a_patch_mean),
            "restore_frac": float(rf),
        })
        if (idx_c + 1) % 25 == 0 or (idx_c + 1) == len(candidates):
            print(f"  done {idx_c+1}/{len(candidates)}")

    single_results_sorted = sorted(single_results, key=lambda r: (-(r["restore_frac"] if not math.isnan(r["restore_frac"]) else -1e9)))
    print("\n[TOP-20] single-node candidates by RestoreFrac")
    print("rank\tL\tH\tslot\tRestoreFrac\ta_patch_mean")
    for i, r in enumerate(single_results_sorted[:20], start=1):
        print(f"{i}\t{r['layer']}\t{r['head']}\t{r['slot']}\t{r['restore_frac']:+.4f}\t{r['a_patch_mean']:+.6g}")

    if int(args.save_json) == 1:
        out_path = out_dir / "b_router_single_node.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "experiment": "B_router_activation_patching",
                    "selected_N_pairs": N_pairs,
                    "candidate_layers_max": cand_max,
                    "slots": [SLOT_ID_TO_NAME[s] for s in slot_ids],
                    "layer_star": Ls,
                    "head_star": Hs,
                    "sv_idx": Ks,
                    "a_clean_mean": a_clean_mean,
                    "a_corr_mean": a_corr_mean,
                },
                "results_sorted": single_results_sorted,
            }, f, indent=2)
        print(f"[SAVE] wrote: {out_path}")

    print("\n[GREEDY] building a small router set ...")
    topK = min(int(args.topK), len(single_results_sorted))
    pool = single_results_sorted[:topK]
    chosen: List[Dict[str, Any]] = []
    chosen_set = set()
    greedy_curve = []

    for t in range(1, int(args.Jmax) + 1):
        best_c = None
        best_c_rf = -1e18
        best_c_ap = None

        for cand in pool:
            key = (cand["layer"], cand["head"], cand["slot_id"])
            if key in chosen_set:
                continue

            patches_by_layer: Dict[int, List[Tuple[int, int]]] = {}
            for c in chosen:
                patches_by_layer.setdefault(c["layer"], []).append((c["head"], c["slot_id"]))
            patches_by_layer.setdefault(cand["layer"], []).append((cand["head"], cand["slot_id"]))

            a_patch_mean = eval_patchset_mean_ai(
                model=model, tokenizer=tokenizer, selected_pairs=selected_pairs,
                layer_star=Ls, head_star=Hs, u_vec=u_vec,
                patches_by_layer=patches_by_layer,
                clean_cache_layers_cpu=clean_cache_layers_cpu,
                batch_size=args.batch_size, device=device
            )
            rf = restore_frac(a_patch_mean, a_corr_mean, a_clean_mean)

            if (not math.isnan(rf)) and (rf > best_c_rf):
                best_c_rf = rf
                best_c = cand
                best_c_ap = a_patch_mean

        if best_c is None:
            print("[GREEDY] No more candidates improved (or all were NaN). Stopping.")
            break

        chosen.append(best_c)
        chosen_set.add((best_c["layer"], best_c["head"], best_c["slot_id"]))
        greedy_curve.append({
            "step": t,
            "added": {"layer": best_c["layer"], "head": best_c["head"], "slot": best_c["slot"], "slot_id": best_c["slot_id"]},
            "restore_frac": float(best_c_rf),
            "a_patch_mean": float(best_c_ap),
        })

        print(f"  step {t:>2}: add (L={best_c['layer']},H={best_c['head']},slot={best_c['slot']}) => RestoreFrac={best_c_rf:+.4f}  a_patch_mean={best_c_ap:+.6g}")
        if best_c_rf >= float(args.stop_frac):
            print(f"[GREEDY] Reached stop_frac={args.stop_frac}.")
            break

    if int(args.save_json) == 1:
        out_path = out_dir / "b_router_greedy.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {"experiment": "B_router_activation_patching_greedy", "topK_pool": topK, "Jmax": int(args.Jmax), "stop_frac": float(args.stop_frac)},
                "chosen_set": [{"layer": c["layer"], "head": c["head"], "slot": c["slot"], "slot_id": c["slot_id"]} for c in chosen],
                "greedy_curve": greedy_curve,
            }, f, indent=2)
        print(f"[SAVE] wrote: {out_path}")

    print("\n[RANDOM] random baseline band ...")
    full_candidates = candidate_list(candidate_layers_max=cand_max, n_heads=n_heads, slot_ids=slot_ids)
    m_max = len(chosen)
    random_summary = []

    for m in range(1, m_max + 1):
        rfs = []
        for _ in range(int(args.random_trials)):
            rs = random.sample(full_candidates, k=m)
            patches_by_layer: Dict[int, List[Tuple[int, int]]] = {}
            for (l, h, s) in rs:
                patches_by_layer.setdefault(l, []).append((h, s))

            a_patch_mean = eval_patchset_mean_ai(
                model=model, tokenizer=tokenizer, selected_pairs=selected_pairs,
                layer_star=Ls, head_star=Hs, u_vec=u_vec,
                patches_by_layer=patches_by_layer,
                clean_cache_layers_cpu=clean_cache_layers_cpu,
                batch_size=args.batch_size, device=device
            )
            rf = restore_frac(a_patch_mean, a_corr_mean, a_clean_mean)
            if not math.isnan(rf):
                rfs.append(rf)

        if len(rfs) == 0:
            mean_rf, std_rf = float("nan"), float("nan")
        else:
            t = torch.tensor(rfs, dtype=torch.float32)
            mean_rf = float(t.mean().item())
            std_rf = float(t.std(unbiased=False).item())

        greedy_rf = float(greedy_curve[m - 1]["restore_frac"]) if (m - 1) < len(greedy_curve) else float("nan")
        random_summary.append({"m": m, "greedy_restore_frac": greedy_rf, "random_mean": mean_rf, "random_std": std_rf, "trials_used": len(rfs)})
        print(f"  m={m:>2}: greedy={greedy_rf:+.4f} | random={mean_rf:+.4f}±{std_rf:.4f} (n={len(rfs)})")

    if int(args.save_json) == 1:
        out_path = out_dir / "b_router_summary.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "experiment": "B_router_activation_patching_summary",
                    "selected_N_pairs": N_pairs,
                    "layer_star": Ls,
                    "head_star": Hs,
                    "sv_idx": Ks,
                    "a_clean_mean": a_clean_mean,
                    "a_corr_mean": a_corr_mean,
                    "candidate_layers_max": cand_max,
                    "slots": [SLOT_ID_TO_NAME[s] for s in slot_ids],
                },
                "greedy_curve": greedy_curve,
                "random_baseline": random_summary,
                "top20_single_nodes": single_results_sorted[:20],
            }, f, indent=2)
        print(f"[SAVE] wrote: {out_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
