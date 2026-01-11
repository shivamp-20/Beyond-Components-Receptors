#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gp_a4_single_direction_intervention.py

Paper-faithful (Regime A) GP Task "A4" style single-direction intervention.

Inputs on disk (defaults):
  data_main/train_1k_gp.csv
  data_main/test_gp.csv
  outputs/gp/svd_cache.pt
  outputs/gp/masks.pt

What it prints:
  - pronoun token IDs + decoded strings (must be " he" and " she")
  - mask value for chosen (layer, head, sv_idx) + rank info
  - mu_he, mu_she (mean±std of a_i) on train
  - for each sigma_scale: baseline ΔLogit mean±std, intervention ΔLogit mean±std, flip% and baseline counts
What it saves (optional):
  outputs/gp/a4_L{L}_H{H}_SV{K}.json

Run:
  python gp_a4_single_direction_intervention.py \
    --data_dir data_main --train_csv train_1k_gp.csv --test_csv test_gp.csv \
    --out_dir outputs/gp --layer 8 --head 6 --sv_idx 12 --device cuda
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer

# Reuse exact helpers/conventions from your training file
import train_gp_masks_and_dump_ov_logit_receptors as gp


def parse_sigma_scales(s: str) -> List[float]:
    s = s.strip()
    if not s:
        return [0, 1, 2, 5, 10, 15, 20]
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in s.split() if p.strip()]
    return [float(p) for p in parts]


def batches(items: List[Dict[str, str]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def expand_rows_to_examples(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Expand each CSV row into two labeled examples (clean + corrupt)."""
    out: List[Dict[str, str]] = []
    for r in rows:
        p = (r.get("pronoun") or "").strip().lower()
        cp = (r.get("corr_pronoun") or "").strip().lower()
        if p in ("he", "she"):
            out.append({"text": r["prefix"], "label": p})
        if cp in ("he", "she"):
            out.append({"text": r["corr_prefix"], "label": cp})
    return out


def tokenize_texts(tokenizer: GPT2TokenizerFast, texts: List[str], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """Tokenize using the exact low-level helpers from your training script."""
    ids_list = gp._encode_texts(tokenizer, texts)
    pad_len = max(len(x) for x in ids_list)
    pad_id = tokenizer.pad_token_id
    tokens, _attn, last_idx = gp._pad_to_length(ids_list, pad_len, pad_id, device)
    return tokens, last_idx


@torch.no_grad()
def forward_ai_only(
    model: HookedTransformer,
    tokens: torch.Tensor,
    last_idx: torch.Tensor,
    layer: int,
    head: int,
    u_vec: torch.Tensor,  # (d_model+1,)
) -> torch.Tensor:
    """Compute a_i(x) at chosen (layer,head): <[context_resid_t*,1], u_vec>."""
    device = tokens.device
    act_fn = gp.get_act_fn(model)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    if not (0 <= layer < n_layers):
        raise ValueError(f"layer out of range: {layer} (n_layers={n_layers})")
    if not (0 <= head < n_heads):
        raise ValueError(f"head out of range: {head} (n_heads={n_heads})")

    B, S = tokens.shape
    causal = gp.make_causal_mask(S, device=str(device))

    x = model.embed(tokens) + model.pos_embed(tokens)

    # Run blocks up to layer-1 to get correct x at chosen layer
    for l in range(layer):
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

    # Chosen layer: compute ai from chosen head's context_resid
    block = model.blocks[layer]
    attn = block.attn
    x_ln1 = block.ln1(x)
    pat = gp.attention_pattern_original(
        x_ln1,
        attn.W_Q[head], attn.b_Q[head],
        attn.W_K[head], attn.b_K[head],
        causal,
        d_head
    )
    ctx = pat @ x_ln1  # (B,S,D)
    ctx_t = ctx[torch.arange(B, device=device), last_idx, :]  # (B,D)
    ones = torch.ones((B, 1), device=device, dtype=ctx_t.dtype)
    nu = torch.cat([ctx_t, ones], dim=-1)  # (B,D+1)
    return (nu * u_vec[None, :]).sum(dim=-1)  # (B,)


@torch.no_grad()
def forward_full_get_ai_resid_logits_last(
    model: HookedTransformer,
    tokens: torch.Tensor,
    last_idx: torch.Tensor,
    layer: int,
    head: int,
    u_vec: torch.Tensor,  # (d_model+1,)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full original forward: returns ai, resid_final(t*), logits_last(t*)."""
    device = tokens.device
    act_fn = gp.get_act_fn(model)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    B, S = tokens.shape
    causal = gp.make_causal_mask(S, device=str(device))

    x = model.embed(tokens) + model.pos_embed(tokens)
    ai_out = torch.zeros((B,), device=device, dtype=torch.float32)

    for l in range(n_layers):
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
            ctx = pat @ x_ln1  # (B,S,D)

            if (l == layer) and (h == head):
                ctx_t = ctx[torch.arange(B, device=device), last_idx, :]
                ones = torch.ones((B, 1), device=device, dtype=ctx_t.dtype)
                nu = torch.cat([ctx_t, ones], dim=-1)
                ai_out = (nu * u_vec[None, :]).sum(dim=-1).to(torch.float32)

            v = (ctx @ attn.W_V[h]) + attn.b_V[h]
            head_outs.append(v @ attn.W_O[h])

        x = x + (torch.stack(head_outs, dim=0).sum(dim=0) + attn.b_O)

        x_ln2 = block.ln2(x)
        pre = (x_ln2 @ mlp.W_in) + mlp.b_in
        h_act = act_fn(pre)
        x = x + ((h_act @ mlp.W_out) + mlp.b_out)

    resid_fin = x[torch.arange(B, device=device), last_idx, :]  # (B,D)
    logits = model.ln_final(resid_fin) @ model.W_U + model.b_U  # (B,V)
    return ai_out, resid_fin, logits


@torch.no_grad()
def estimate_mu_ai(
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    examples: List[Dict[str, str]],
    layer: int,
    head: int,
    u_vec: torch.Tensor,
    batch_size: int,
    device: str,
) -> Tuple[float, float, int]:
    if len(examples) == 0:
        return float("nan"), float("nan"), 0

    all_ai = []
    for batch in batches(examples, batch_size):
        tokens, last_idx = tokenize_texts(tokenizer, [e["text"] for e in batch], device=device)
        ai = forward_ai_only(model, tokens, last_idx, layer=layer, head=head, u_vec=u_vec)
        all_ai.append(ai.detach().float().cpu())

    a = torch.cat(all_ai, dim=0)
    return float(a.mean().item()), float(a.std(unbiased=False).item()), int(a.numel())


@torch.no_grad()
def collect_baseline_cache(
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    examples: List[Dict[str, str]],
    layer: int,
    head: int,
    u_vec: torch.Tensor,
    he_id: int,
    she_id: int,
    batch_size: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    ais, resids, base_pred, logit_he, logit_she = [], [], [], [], []
    for batch in batches(examples, batch_size):
        tokens, last_idx = tokenize_texts(tokenizer, [e["text"] for e in batch], device=device)
        ai, resid_fin, logits = forward_full_get_ai_resid_logits_last(
            model, tokens, last_idx, layer=layer, head=head, u_vec=u_vec
        )
        pred = torch.argmax(logits, dim=-1)

        ais.append(ai.detach())
        resids.append(resid_fin.detach())
        base_pred.append(pred.detach())
        logit_he.append(logits[:, he_id].detach())
        logit_she.append(logits[:, she_id].detach())

    return {
        "ai": torch.cat(ais, dim=0),
        "resid": torch.cat(resids, dim=0),
        "base_pred": torch.cat(base_pred, dim=0),
        "logit_he": torch.cat(logit_he, dim=0),
        "logit_she": torch.cat(logit_she, dim=0),
    }


@torch.no_grad()
def sigma_sweep_eval(
    model: HookedTransformer,
    cache: Dict[str, torch.Tensor],
    label: str,
    mu_he: float,
    mu_she: float,
    he_id: int,
    she_id: int,
    v_row: torch.Tensor,   # (D,)
    sigma: torch.Tensor,   # scalar
    sigma_scales: List[float],
) -> Dict[str, Any]:
    ai = cache["ai"]
    resid = cache["resid"]
    base_pred = cache["base_pred"]
    logit_he = cache["logit_he"]
    logit_she = cache["logit_she"]

    if label == "he":
        a_target = mu_she
        correct_id, opp_id = he_id, she_id
        diff_base = (logit_he - logit_she)
        denom_mask = (base_pred == he_id)
    elif label == "she":
        a_target = mu_he
        correct_id, opp_id = she_id, he_id
        diff_base = (logit_she - logit_he)
        denom_mask = (base_pred == she_id)
    else:
        raise ValueError("label must be 'he' or 'she'")

    base_counts = {
        "he": int((base_pred == he_id).sum().item()),
        "she": int((base_pred == she_id).sum().item()),
        "other": int(((base_pred != he_id) & (base_pred != she_id)).sum().item()),
    }

    out: Dict[str, Any] = {}
    sigma_f = float(sigma.item())

    for scale in sigma_scales:
        scale_f = float(scale)
        delta = ((a_target - ai) * (scale_f * sigma_f)).to(resid.dtype)[:, None] * v_row[None, :]
        resid_cf = resid + delta

        logits_cf = model.ln_final(resid_cf) @ model.W_U + model.b_U  # (N,V)
        cf_pred = torch.argmax(logits_cf, dim=-1)
        diff_cf = (logits_cf[:, correct_id] - logits_cf[:, opp_id])

        denom = int(denom_mask.sum().item())
        if label == "he":
            flips = int(((denom_mask) & (cf_pred == she_id)).sum().item())
        else:
            flips = int(((denom_mask) & (cf_pred == he_id)).sum().item())
        flip_pct = 100.0 * flips / max(1, denom)

        out[str(scale_f)] = {
            "baseline_logitdiff_mean": float(diff_base.mean().item()),
            "baseline_logitdiff_std": float(diff_base.std(unbiased=False).item()),
            "interv_logitdiff_mean": float(diff_cf.mean().item()),
            "interv_logitdiff_std": float(diff_cf.std(unbiased=False).item()),
            "base_pred_counts": base_counts,
            "flip_count": flips,
            "flip_denom": denom,
            "flip_pct": flip_pct,
        }

    return out


def print_sigma_table(label: str, sigma_scales: List[float], results: Dict[str, Any]) -> None:
    print(f"\n=== TEST GROUP: {label.upper()} contexts ===")
    print("sigma_scale\tΔLogit_base(mean±std)\tΔLogit_int(mean±std)\tflip% (count/denom)\tbase_pred_counts")
    for s in sigma_scales:
        key = str(float(s))
        r = results[key]
        base = f"{r['baseline_logitdiff_mean']:+.4f}±{r['baseline_logitdiff_std']:.4f}"
        cf = f"{r['interv_logitdiff_mean']:+.4f}±{r['interv_logitdiff_std']:.4f}"
        flip = f"{r['flip_pct']:.2f}% ({r['flip_count']}/{r['flip_denom']})"
        print(f"{float(s):g}\t{base}\t{cf}\t{flip}\t{r['base_pred_counts']}")


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="data_main")
    ap.add_argument("--train_csv", type=str, default="train_1k_gp.csv")
    ap.add_argument("--test_csv", type=str, default="test_gp.csv")
    ap.add_argument("--out_dir", type=str, default="outputs/gp")

    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--sv_idx", type=int, required=True)

    ap.add_argument("--tau", type=float, default=1e-2, help="Mask threshold; used only for printing/check.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--sigma_scales", type=str, default="0,1,2,5,10,15,20")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--save_json", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda requested but not available; falling back to cpu.")
        device = "cpu"

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer (same as training)
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # Regime A pronoun tokens: leading-space single tokens
    he_ids = tokenizer.encode(" he", add_special_tokens=False)
    she_ids = tokenizer.encode(" she", add_special_tokens=False)
    if len(he_ids) != 1 or len(she_ids) != 1:
        raise ValueError(f'" he"/" she" not single tokens: he_ids={he_ids}, she_ids={she_ids}')
    he_id, she_id = he_ids[0], she_ids[0]
    print("[TOKENS] he_id =", he_id, "decoded =", tokenizer.decode([he_id], clean_up_tokenization_spaces=False))
    print("[TOKENS] she_id =", she_id, "decoded =", tokenizer.decode([she_id], clean_up_tokenization_spaces=False))

    # Load data via exact loader from training script (keeps prefix normalization)
    train_rows = gp.load_gp_csv(str(data_dir / args.train_csv))
    test_rows = gp.load_gp_csv(str(data_dir / args.test_csv))
    print(f"[DATA] train_rows={len(train_rows)} test_rows={len(test_rows)}")
    gp.sanity_check_pronoun_tokenization(tokenizer, train_rows, max_checks=50)

    train_ex = expand_rows_to_examples(train_rows)
    test_ex = expand_rows_to_examples(test_rows)
    train_he = [e for e in train_ex if e["label"] == "he"]
    train_she = [e for e in train_ex if e["label"] == "she"]
    test_he = [e for e in test_ex if e["label"] == "he"]
    test_she = [e for e in test_ex if e["label"] == "she"]
    print(f"[COUNTS] train_he={len(train_he)} train_she={len(train_she)} | test_he={len(test_he)} test_she={len(test_she)}")

    # Model (original)
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load artifacts from training
    svd_path = out_dir / "svd_cache.pt"
    masks_path = out_dir / "masks.pt"
    if not svd_path.exists():
        raise FileNotFoundError(f"Missing svd_cache.pt: {svd_path}")
    if not masks_path.exists():
        raise FileNotFoundError(f"Missing masks.pt: {masks_path}")

    qk, ov, mlp_in, mlp_out, rank_total_ov = gp.load_svd_cache(str(svd_path), device=device)
    masks_dict = torch.load(str(masks_path), map_location="cpu")  # sigmoid masks

    L, H, K = args.layer, args.head, args.sv_idx
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if not (0 <= L < n_layers):
        raise ValueError(f"--layer {L} out of range (0..{n_layers-1})")
    if not (0 <= H < n_heads):
        raise ValueError(f"--head {H} out of range (0..{n_heads-1})")

    svd = ov[L][H]
    if not (0 <= K < svd.r):
        raise ValueError(f"--sv_idx {K} out of range (0..{svd.r-1}) for ov[L][H].r={svd.r}")

    mask_val = float(masks_dict["ov"][L][H][K].item())
    rank_trainable = int(svd.r)
    rank_total = int(rank_total_ov[L][H])
    print(f"[CHOSEN] (L,H,SV)=({L},{H},{K}) | mask={mask_val:.6g} | tau={args.tau:g} | rank_trainable={rank_trainable} | rank_total_ov={rank_total}")

    u_vec = svd.U[:, K].contiguous()    # (D+1,)
    sigma = svd.S[K].contiguous()       # scalar
    v_row = svd.Vh[K, :].contiguous()   # (D,)
    print(f"[DIR] sigma={float(sigma.detach().cpu().item()):.6g} | u.shape={tuple(u_vec.shape)} | v.shape={tuple(v_row.shape)}")

    sigma_scales = parse_sigma_scales(args.sigma_scales)
    if 0.0 not in [float(x) for x in sigma_scales]:
        sigma_scales = [0.0] + sigma_scales
    print(f"[SWEEP] sigma_scales={sigma_scales}")

    # TRAIN means
    mu_he, std_he, n_he = estimate_mu_ai(model, tokenizer, train_he, L, H, u_vec, args.batch_size, device)
    mu_she, std_she, n_she = estimate_mu_ai(model, tokenizer, train_she, L, H, u_vec, args.batch_size, device)
    print(f"[TRAIN a_i] mu_he={mu_he:.6g} ± {std_he:.6g} (n={n_he})")
    print(f"[TRAIN a_i] mu_she={mu_she:.6g} ± {std_she:.6g} (n={n_she})")

    # TEST caches
    print("\n[CACHE] computing baseline cache for TEST he-group ...")
    cache_he = collect_baseline_cache(model, tokenizer, test_he, L, H, u_vec, he_id, she_id, args.batch_size, device)
    print("[CACHE] computing baseline cache for TEST she-group ...")
    cache_she = collect_baseline_cache(model, tokenizer, test_she, L, H, u_vec, he_id, she_id, args.batch_size, device)

    res_test_he = sigma_sweep_eval(model, cache_he, "he", mu_he, mu_she, he_id, she_id, v_row, sigma, sigma_scales)
    res_test_she = sigma_sweep_eval(model, cache_she, "she", mu_he, mu_she, he_id, she_id, v_row, sigma, sigma_scales)

    print_sigma_table("he", sigma_scales, res_test_he)
    print_sigma_table("she", sigma_scales, res_test_she)

    # scale=0 sanity
    r0_he = res_test_he[str(0.0)]
    r0_she = res_test_she[str(0.0)]
    print("\n[SANITY] scale=0 should match baseline: ΔLogit_base == ΔLogit_int up to numerical noise.")
    print(f"  he-group:  base_mean={r0_he['baseline_logitdiff_mean']:+.6g}  int_mean={r0_he['interv_logitdiff_mean']:+.6g}  absdiff={abs(r0_he['baseline_logitdiff_mean']-r0_he['interv_logitdiff_mean']):.3g}")
    print(f"  she-group: base_mean={r0_she['baseline_logitdiff_mean']:+.6g}  int_mean={r0_she['interv_logitdiff_mean']:+.6g}  absdiff={abs(r0_she['baseline_logitdiff_mean']-r0_she['interv_logitdiff_mean']):.3g}")

    if int(args.save_json) == 1:
        out_path = out_dir / f"a4_L{L}_H{H}_SV{K}.json"
        payload = {
            "chosen": {
                "layer": L, "head": H, "sv_idx": K,
                "mask": mask_val, "tau": float(args.tau),
                "sigma": float(sigma.detach().cpu().item()),
                "rank_trainable": rank_trainable,
                "rank_total_ov": rank_total,
            },
            "token_ids": {
                "he_id": int(he_id),
                "she_id": int(she_id),
                "he_str": tokenizer.decode([he_id], clean_up_tokenization_spaces=False),
                "she_str": tokenizer.decode([she_id], clean_up_tokenization_spaces=False),
            },
            "mu": {
                "mu_he": mu_he, "std_he": std_he, "n_he": n_he,
                "mu_she": mu_she, "std_she": std_she, "n_she": n_she,
            },
            "sigma_scales": [float(x) for x in sigma_scales],
            "results_test_he": res_test_he,
            "results_test_she": res_test_she,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        print(f"\n[SAVE] wrote: {out_path}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
