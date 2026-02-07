#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gp_exp0_receptor_activation_vectors.py

Experiment 0 (Phase 1): Receptor activation scalar + activation-vector computation.

Given one or more OV logit receptors (layer, head, sv_idx), compute:
  - a_i(x) = <[context_resid_t*, 1], u_i> for each example
  - per-label mean/std (mu_he, mu_she)
  - effect size (Cohen's d), ROC-AUC for separating labels using a_i
  - activation vectors in residual space:
        vec_he   = mu_he  * (sigma * v)
        vec_she  = mu_she * (sigma * v)
        vec_delta = (mu_he - mu_she) * (sigma * v)
    (optionally multiplied by the learned mask value)

Saves:
  out_dir/exp0_activation_vectors.pt      (tensors: u, v, vectors)
  out_dir/exp0_activation_vectors.json    (summary scalars)

Run example:
  python gp_exp0_receptor_activation_vectors.py \
    --data_dir data_main --train_csv train_1k_gp.csv --out_dir outputs/gp \
    --receptors "10,9,0;11,7,0;9,7,1" --batch_size 64 --device cuda --use_mask 1
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import torch

from transformer_lens import HookedTransformer
from transformers import GPT2TokenizerFast

import train_gp_masks_and_dump_ov_logit_receptors_ddp as gp


def parse_receptors(s: str) -> List[Tuple[int, int, int]]:
    # "L,H,SV;L,H,SV;..."
    out = []
    s = (s or "").strip()
    if not s:
        raise ValueError("--receptors is required, e.g. '10,9,0;11,7,0;9,7,1'")
    for chunk in s.split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Bad receptor spec '{chunk}'. Expected 'L,H,SV'.")
        out.append((int(parts[0]), int(parts[1]), int(parts[2])))
    if not out:
        raise ValueError("No receptors parsed.")
    return out


def auc_roc(scores: torch.Tensor, labels01: torch.Tensor) -> float:
    """
    ROC-AUC via rank statistic (Mann–Whitney U).
    scores: (N,) float
    labels01: (N,) in {0,1}
    """
    scores = scores.detach().cpu().float()
    labels01 = labels01.detach().cpu().long()
    n_pos = int(labels01.sum().item())
    n = int(labels01.numel())
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # ranks of scores (average ranks for ties)
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels01[order]

    ranks = torch.zeros_like(sorted_scores)
    i = 0
    r = 1.0
    while i < n:
        j = i + 1
        while j < n and sorted_scores[j].item() == sorted_scores[i].item():
            j += 1
        # average rank for [i, j)
        avg = (r + (r + (j - i) - 1.0)) / 2.0
        ranks[i:j] = avg
        r += (j - i)
        i = j

    sum_ranks_pos = float((ranks * sorted_labels.float()).sum().item())
    # U for positives
    u = sum_ranks_pos - (n_pos * (n_pos + 1)) / 2.0
    auc = u / (n_pos * n_neg)
    return float(auc)


def cohen_d(x0: torch.Tensor, x1: torch.Tensor) -> float:
    x0 = x0.detach().cpu().float()
    x1 = x1.detach().cpu().float()
    m0 = x0.mean().item()
    m1 = x1.mean().item()
    s0 = x0.std(unbiased=False).item()
    s1 = x1.std(unbiased=False).item()
    n0 = x0.numel()
    n1 = x1.numel()
    if n0 < 2 or n1 < 2:
        return float("nan")
    sp = math.sqrt(((n0 - 1) * s0 * s0 + (n1 - 1) * s1 * s1) / ((n0 - 1) + (n1 - 1) + 1e-12))
    if sp == 0:
        return float("inf") if (m0 != m1) else 0.0
    return float((m1 - m0) / sp)


def rows_to_texts_with_labels(rows: List[Dict[str, str]], include_corrupted: bool) -> List[Dict[str, str]]:
    out = []
    for r in rows:
        p = (r.get("pronoun", "") or "").strip().lower()
        if p in ("he", "she"):
            out.append({"text": r["prefix"], "label": p, "kind": "clean"})
        if include_corrupted:
            cp = (r.get("corr_pronoun", "") or "").strip().lower()
            if cp in ("he", "she"):
                out.append({"text": r["corr_prefix"], "label": cp, "kind": "corr"})
    return out


def tokenize_texts(tokenizer: GPT2TokenizerFast, texts: List[str], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    # exactly like your intervention script: use gp._encode_texts + gp._pad_to_length
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
    """
    Compute a_i(x) at chosen (layer,head): <[context_resid_t*, 1], u_vec>.
    Matches gp_a4_single_direction_intervention.py.
    """
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

    # run blocks up to layer-1
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

    # chosen layer: context_resid for chosen head
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--train_csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--receptors", type=str, required=True,
                    help='e.g. "10,9,0;11,7,0;9,7,1"')
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--use_mask", type=int, default=1, help="multiply activation vectors by learned mask value")
    ap.add_argument("--include_corrupted", type=int, default=1, help="include corr_prefix rows in mu estimates")
    ap.add_argument("--max_items", type=int, default=0, help="0 = all, else limit number of labeled texts")
    ap.add_argument("--save_per_example", type=int, default=0, help="save all a_i(x) values (bigger pt file)")
    args = ap.parse_args()

    torch.set_grad_enabled(False)

    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- load model ----
    tl_name = "gpt2-small"
    model = HookedTransformer.from_pretrained(tl_name, device=args.device)
    model.eval()

    tokenizer: GPT2TokenizerFast = model.tokenizer
    # GPT2 has no pad token by default; mirror your other scripts:
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---- load SVD + masks ----
    svd_path = out_dir / "svd_cache.pt"
    masks_path = out_dir / "masks.pt"
    if not svd_path.exists():
        raise FileNotFoundError(f"Missing {svd_path}")
    if not masks_path.exists():
        raise FileNotFoundError(f"Missing {masks_path}")

    svd_cache = gp.load_svd_cache(str(svd_path), device=args.device)
    masks_dict = torch.load(masks_path, map_location="cpu")

    receptors = parse_receptors(args.receptors)

    # gather per-receptor vectors
    rec_info = []
    for (layer, head, sv_idx) in receptors:
        svd = svd_cache["ov"][layer][head]
        U = svd.U  # (d_model+1, r)
        S = svd.S  # (r,)
        Vh = svd.Vh  # (r, d_model)

        if sv_idx < 0 or sv_idx >= S.numel():
            raise ValueError(f"sv_idx {sv_idx} out of range for L{layer}.H{head} with rank={S.numel()}")

        u_vec = U[:, sv_idx].to(args.device)
        sigma = float(S[sv_idx].item())
        v_row = Vh[sv_idx, :].to(args.device)

        mask_val = float(masks_dict["ov"][layer][head][sv_idx])
        rec_info.append({
            "layer": layer, "head": head, "sv_idx": sv_idx,
            "mask": mask_val, "sigma": sigma,
            "u_vec": u_vec, "v_row": v_row,
        })

    # ---- load labeled texts ----
    train_rows = gp.load_gp_csv(str(data_dir / args.train_csv))
    items = rows_to_texts_with_labels(train_rows, include_corrupted=bool(args.include_corrupted))
    if args.max_items and args.max_items > 0:
        items = items[:args.max_items]

    labels = [it["label"] for it in items]
    texts = [it["text"] for it in items]

    # encode labels to 0/1: she=0, he=1
    y = torch.tensor([1 if lab == "he" else 0 for lab in labels], device="cpu", dtype=torch.long)

    # ---- compute a_i(x) for each receptor ----
    all_ai = {f"L{r['layer']}_H{r['head']}_SV{r['sv_idx']}": [] for r in rec_info}

    bs = args.batch_size
    for start in range(0, len(texts), bs):
        batch_texts = texts[start:start + bs]
        tokens, last_idx = tokenize_texts(tokenizer, batch_texts, device=args.device)

        for r in rec_info:
            key = f"L{r['layer']}_H{r['head']}_SV{r['sv_idx']}"
            ai = forward_ai_only(model, tokens, last_idx, r["layer"], r["head"], r["u_vec"])
            all_ai[key].append(ai.detach().cpu())

    # concat
    for k in list(all_ai.keys()):
        all_ai[k] = torch.cat(all_ai[k], dim=0)  # (N,)

    # ---- compute summary + activation vectors ----
    summary = {"meta": {
        "tl_name": tl_name,
        "train_csv": args.train_csv,
        "include_corrupted": bool(args.include_corrupted),
        "use_mask": bool(args.use_mask),
        "n_items": int(len(texts)),
    }, "receptors": []}

    pt_payload = {"meta": summary["meta"], "receptors": []}

    for r in rec_info:
        key = f"L{r['layer']}_H{r['head']}_SV{r['sv_idx']}"
        ai = all_ai[key]  # cpu tensor

        he_mask = (y == 1)
        she_mask = (y == 0)

        ai_he = ai[he_mask]
        ai_she = ai[she_mask]

        mu_he = float(ai_he.mean().item())
        mu_she = float(ai_she.mean().item())
        sd_he = float(ai_he.std(unbiased=False).item())
        sd_she = float(ai_she.std(unbiased=False).item())
        d = cohen_d(ai_she, ai_he)
        auc = auc_roc(ai, y)

        sigma = r["sigma"]
        mask_val = r["mask"]
        scale = mask_val if args.use_mask else 1.0

        v = r["v_row"].detach().cpu().float()
        u = r["u_vec"].detach().cpu().float()

        base_dir = (sigma * v) * scale  # (d_model,)
        vec_he = (mu_he * base_dir)
        vec_she = (mu_she * base_dir)
        vec_delta = ((mu_he - mu_she) * base_dir)

        rec_sum = {
            "key": key,
            "layer": r["layer"], "head": r["head"], "sv_idx": r["sv_idx"],
            "mask": mask_val, "sigma": sigma, "used_scale": scale,
            "mu_he": mu_he, "sd_he": sd_he,
            "mu_she": mu_she, "sd_she": sd_she,
            "delta_mu": mu_he - mu_she,
            "cohen_d(he_vs_she)": d,
            "auc(ai->he)": auc,
            "norm_base_dir": float(base_dir.norm().item()),
            "norm_vec_delta": float(vec_delta.norm().item()),
        }
        summary["receptors"].append(rec_sum)

        pt_payload["receptors"].append({
            **rec_sum,
            "u_vec": u,
            "v_row": v,
            "base_dir": base_dir,
            "vec_he": vec_he,
            "vec_she": vec_she,
            "vec_delta": vec_delta,
            **({"ai_all": ai} if args.save_per_example else {}),
            **({"labels01": y} if args.save_per_example else {}),
        })

        # print to console
        print("\n" + "=" * 80)
        print(f"{key}  mask={mask_val:.6g}  sigma={sigma:.6g}  use_mask={bool(args.use_mask)}")
        print(f"  mu_he  ={mu_he:.6g} ± {sd_he:.6g}   (n={int(ai_he.numel())})")
        print(f"  mu_she ={mu_she:.6g} ± {sd_she:.6g}  (n={int(ai_she.numel())})")
        print(f"  delta_mu = {mu_he - mu_she:.6g}")
        print(f"  Cohen's d (he vs she) = {d:.4g}")
        print(f"  ROC-AUC(ai→he)        = {auc:.4g}")
        print(f"  ||sigma*v||           = {rec_sum['norm_base_dir']:.6g}")
        print(f"  ||delta_vec||         = {rec_sum['norm_vec_delta']:.6g}")

    # ---- save ----
    pt_path = out_dir / "exp0_activation_vectors.pt"
    js_path = out_dir / "exp0_activation_vectors.json"
    torch.save(pt_payload, pt_path)
    js_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nSaved:")
    print(f"  {pt_path}")
    print(f"  {js_path}")


if __name__ == "__main__":
    main()
