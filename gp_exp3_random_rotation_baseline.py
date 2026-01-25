#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gp_exp3_random_rotation_baseline.py

Experiment 3: Random-rotation baseline for Exp2's rotated-axis result.

Goal:
  Inside the same mask-selected k-dim OV subspace (same head),
  compare LDA axis vs random Haar rotations.

We:
  - Fix (layer, head), i_star, and idx_list (k SV indices).
  - Cache k-dim features A = nu @ U_sel for train/test (same as Exp2).
  - STAR baseline = i_star
  - LDA baseline = (Sw + reg I)^(-1) (mu_he - mu_she), normalized (same as Exp2)
  - Random trials:
      * sample Haar random orthogonal R via QR + sign-fix
      * rotate train features A_rot = A_train @ R
      * pick axis j with max |Cohen's d|
      * w = R[:, j] (back in original coords)
      * evaluate flips + KL under the SAME intervention rule as Exp2

Outputs:
  {out_dir}/exp3_random_rot/L{L}_H{H}_k{k}/trials.jsonl
  {out_dir}/exp3_random_rot/L{L}_H{H}_k{k}/summary.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer

import train_gp_masks_and_dump_ov_logit_receptors as gp


# ----------------------------
# Small utils
# ----------------------------

def _parse_float_list(s: str, default: List[float]) -> List[float]:
    s = (s or "").strip()
    if not s:
        return list(default)
    if "," in s:
        parts = [p.strip() for p in s.split(",") if p.strip()]
    else:
        parts = [p.strip() for p in s.split() if p.strip()]
    return [float(p) for p in parts]


def parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    parts = [p.strip() for p in s.replace(" ", ",").split(",") if p.strip()]
    return [int(p) for p in parts]


def batches(items: List[Dict[str, str]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i:i + batch_size]


def expand_rows_to_examples(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """
    Same convention as Exp2:
    Each CSV row yields up to two labeled examples:
      (prefix, pronoun) and (corr_prefix, corr_pronoun)
    """
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
    """
    Exact low-level tokenizer path from your training script.
    Returns:
      tokens: (B,S)
      last_idx: (B,)   last non-pad position index
    """
    ids_list = gp._encode_texts(tokenizer, texts)
    pad_len = max(len(x) for x in ids_list)
    pad_id = tokenizer.pad_token_id
    tokens, _attn, last_idx = gp._pad_to_length(ids_list, pad_len, pad_id, device)
    return tokens, last_idx


def kl_pq_from_logits(base_logits: torch.Tensor, cf_logits: torch.Tensor) -> torch.Tensor:
    """
    KL(P||Q) per example, P=softmax(base_logits), Q=softmax(cf_logits).
    """
    logp = F.log_softmax(base_logits, dim=-1)
    p = logp.exp()
    logq = F.log_softmax(cf_logits, dim=-1)
    return (p * (logp - logq)).sum(dim=-1)


# ----------------------------
# Forward (explicit, same style as your A4/Exp2)
# ----------------------------

@torch.no_grad()
def forward_collect_nu_resid_logits_last(
    model: HookedTransformer,
    tokens: torch.Tensor,         # (B,S)
    last_idx: torch.Tensor,       # (B,)
    layer: int,
    head: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      nu_out: (B, d_model+1)  = concat(ctx_t, 1)
      resid_last: (B, d_model)
      logits_last: (B, vocab)
    """
    device = tokens.device
    act_fn = gp.get_act_fn(model)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    B, S = tokens.shape
    causal = gp.make_causal_mask(S, device=str(device))

    x = model.embed(tokens) + model.pos_embed(tokens)
    nu_out = None

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
                ctx_t = ctx[torch.arange(B, device=device), last_idx, :]  # (B,D)
                ones = torch.ones((B, 1), device=device, dtype=ctx_t.dtype)
                nu_out = torch.cat([ctx_t, ones], dim=-1)  # (B,D+1)

            v = (ctx @ attn.W_V[h]) + attn.b_V[h]
            head_outs.append(v @ attn.W_O[h])

        x = x + (torch.stack(head_outs, dim=0).sum(dim=0) + attn.b_O)

        x_ln2 = block.ln2(x)
        pre = (x_ln2 @ mlp.W_in) + mlp.b_in
        h_act = act_fn(pre)
        x = x + ((h_act @ mlp.W_out) + mlp.b_out)

    if nu_out is None:
        raise RuntimeError("nu_out was not computed (check layer/head indices).")

    resid_last = x[torch.arange(B, device=device), last_idx, :]           # (B,D)
    logits_last = model.ln_final(resid_last) @ model.W_U + model.b_U      # (B,V)
    return nu_out, resid_last, logits_last


# ----------------------------
# idx_list selection (same as Exp2)
# ----------------------------

def choose_indices(
    mask_vec: torch.Tensor,
    k: int,
    selection_mode: str,
    tau: float,
    explicit_list: List[int],
    must_include: int,
) -> List[int]:
    mask_vec = mask_vec.detach().float().cpu()
    r_total = int(mask_vec.numel())
    if not (0 <= must_include < r_total):
        raise ValueError(f"i_star out of range: {must_include} (r_total={r_total})")

    if selection_mode == "mask_topk":
        idx = torch.argsort(mask_vec, descending=True).tolist()[:k]
    elif selection_mode == "mask_threshold_then_topk":
        cand = [i for i in range(r_total) if float(mask_vec[i].item()) > float(tau)]
        cand = cand if len(cand) > 0 else list(range(r_total))
        cand_sorted = sorted(cand, key=lambda i: float(mask_vec[i].item()), reverse=True)
        idx = cand_sorted[:k]
    elif selection_mode == "explicit_list":
        if len(explicit_list) != k:
            raise ValueError(f"--explicit_list must have exactly k={k} integers, got {len(explicit_list)}")
        idx = list(explicit_list)
    else:
        raise ValueError(f"Unknown selection_mode: {selection_mode}")

    if must_include not in idx:
        idx[-1] = must_include

    seen = set()
    uniq: List[int] = []
    for i in idx:
        if int(i) not in seen:
            uniq.append(int(i))
            seen.add(int(i))
    if len(uniq) < k:
        all_sorted = torch.argsort(mask_vec, descending=True).tolist()
        for i in all_sorted:
            if int(i) not in seen:
                uniq.append(int(i))
                seen.add(int(i))
            if len(uniq) == k:
                break

    if selection_mode != "explicit_list":
        uniq = sorted(uniq, key=lambda i: float(mask_vec[i].item()), reverse=True)

    return uniq[:k]


# ----------------------------
# Random Haar orthogonal via QR + sign fix
# ----------------------------

def sample_haar_orthogonal(k: int, seed: int) -> torch.Tensor:
    """
    Haar-uniform Q in R^{k×k}:
      G ~ N(0,1)^{k×k}
      Q,R = qr(G)
      Q = Q * sign(diag(R))  (column-wise)
    """
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    G = torch.randn((k, k), generator=gen, dtype=torch.float32)
    Q, R = torch.linalg.qr(G)
    diag = torch.diagonal(R)
    s = torch.sign(diag)
    s[s == 0] = 1.0
    Q = Q * s.unsqueeze(0)
    return Q.contiguous()


def cohen_d_per_axis(A: torch.Tensor, he_mask: torch.Tensor) -> torch.Tensor:
    """
    Vectorized Cohen's d for A: (N,k) on CPU.
    """
    he_mask = he_mask.bool()
    she_mask = ~he_mask
    Ah = A[he_mask]
    As = A[she_mask]
    mu_h = Ah.mean(dim=0)
    mu_s = As.mean(dim=0)
    std_h = Ah.std(dim=0, unbiased=False)
    std_s = As.std(dim=0, unbiased=False)
    denom = torch.sqrt(0.5 * (std_h ** 2 + std_s ** 2) + 1e-12)
    return (mu_h - mu_s) / denom


def orient_w_by_receptor(
    w_cpu: torch.Tensor,           # (k,) CPU
    V_sel_d: torch.Tensor,         # (k,D) device
    model: HookedTransformer,
    he_id: int,
    she_id: int,
    device: str,
) -> torch.Tensor:
    """
    Same sign convention as Exp2: make receptor[he] > receptor[she].
    receptor is computed from v_rot = sum_t w_t * V_sel[t].
    """
    w_d = w_cpu.to(device=device, dtype=torch.float32)
    v_rot = (w_d[:, None] * V_sel_d).sum(dim=0)
    receptor = (v_rot @ model.W_U).detach()
    if float(receptor[he_id].item()) < float(receptor[she_id].item()):
        w_cpu = -w_cpu
    return w_cpu


def build_intervention_direction(
    w_cpu: torch.Tensor,           # (k,) CPU
    S_sel_d: torch.Tensor,         # (k,) device
    V_sel_d: torch.Tensor,         # (k,D) device
    device: str,
) -> Tuple[float, torch.Tensor]:
    """
    Exp2 scaling:
      alpha = w ⊙ S_sel
      sigma_eff = ||alpha||
      v_eff_unit = sum((alpha/||alpha||) * V_sel_row)
    """
    w_d = w_cpu.to(device=device, dtype=torch.float32)
    alpha = w_d * S_sel_d
    sigma_eff = float(alpha.norm().item())
    if sigma_eff < 1e-8 or not math.isfinite(sigma_eff):
        raise RuntimeError("sigma_eff is degenerate.")
    v_eff_unit = ((alpha / alpha.norm())[:, None] * V_sel_d).sum(dim=0).to(dtype=torch.float32)
    return sigma_eff, v_eff_unit


def headline_kl_at_flip(
    sweep: Dict[str, Any],
    scales: List[float],
    flip_threshold: float,
) -> float:
    for s in scales:
        key = str(float(s))
        if key not in sweep:
            continue
        r = sweep[key]
        if float(r.get("flip_he_pct", 0.0)) >= flip_threshold and float(r.get("flip_she_pct", 0.0)) >= flip_threshold:
            return float(r.get("kl_mean", float("inf")))
    return float("inf")


@torch.no_grad()
def eval_sweep(
    model: HookedTransformer,
    resid_test_cpu: torch.Tensor,     # (N,D) CPU
    base_logits_cpu: torch.Tensor,    # (N,V) CPU
    denom_he_mask_cpu: torch.Tensor,  # (N,) CPU bool
    denom_she_mask_cpu: torch.Tensor, # (N,) CPU bool
    he_id: int,
    she_id: int,
    a_scalar_cpu: torch.Tensor,       # (N,) CPU
    target_cpu: torch.Tensor,         # (N,) CPU
    v_dir_d: torch.Tensor,            # (D,) device
    sigma_value: float,
    scales: List[float],
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    """
    Exact Exp2 convention:
      delta = (target - a) * (scale * sigma_value)
      resid_cf = resid + delta * v_dir
      logits_cf = ln_final(resid_cf) @ W_U + b_U
      metrics: flips, other%, KL(P||Q)
    """
    N = int(a_scalar_cpu.numel())
    denom_he = int(denom_he_mask_cpu.sum().item())
    denom_she = int(denom_she_mask_cpu.sum().item())

    out: Dict[str, Any] = {}
    for scale in scales:
        scale_f = float(scale)
        flips_he = 0
        flips_she = 0
        other_all = 0
        other_he = 0
        other_she = 0
        kl_sum = 0.0

        for i0 in range(0, N, batch_size):
            i1 = min(N, i0 + batch_size)

            resid_b = resid_test_cpu[i0:i1].to(device).float()
            base_logits_b = base_logits_cpu[i0:i1].to(device).float()
            a_b = a_scalar_cpu[i0:i1].to(device).float()
            t_b = target_cpu[i0:i1].to(device).float()

            delta_coeff = (t_b - a_b) * (scale_f * float(sigma_value))
            resid_cf = resid_b + delta_coeff[:, None] * v_dir_d[None, :]

            logits_cf = model.ln_final(resid_cf) @ model.W_U + model.b_U
            pred_cf = torch.argmax(logits_cf, dim=-1)

            denom_he_b = denom_he_mask_cpu[i0:i1].to(device)
            denom_she_b = denom_she_mask_cpu[i0:i1].to(device)
            flips_he += int((denom_he_b & (pred_cf == she_id)).sum().item())
            flips_she += int((denom_she_b & (pred_cf == he_id)).sum().item())

            other_mask = (pred_cf != he_id) & (pred_cf != she_id)
            other_all += int(other_mask.sum().item())
            other_he += int((denom_he_b & other_mask).sum().item())
            other_she += int((denom_she_b & other_mask).sum().item())

            kl_sum += float(kl_pq_from_logits(base_logits_b, logits_cf).sum().item())

        out[str(scale_f)] = {
            "flip_he_pct": 100.0 * flips_he / max(1, denom_he),
            "flip_she_pct": 100.0 * flips_she / max(1, denom_she),
            "other_pct_all": 100.0 * other_all / max(1, N),
            "kl_mean": float(kl_sum / max(1, N)),
        }

    return out


def print_table(title: str, scales: List[float], res: Dict[str, Any]) -> None:
    print(f"\n=== {title} ===")
    print("scale\tflip_he%\tflip_she%\tother%\tKL_mean")
    for s in scales:
        r = res[str(float(s))]
        print(f"{float(s):g}\t{r['flip_he_pct']:.2f}\t{r['flip_she_pct']:.2f}\t{r['other_pct_all']:.2f}\t{r['kl_mean']:.4g}")


# ----------------------------
# Main
# ----------------------------

def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="data_main")
    ap.add_argument("--train_csv", type=str, default="train_1k_gp.csv")
    ap.add_argument("--test_csv", type=str, default="test_gp.csv")
    ap.add_argument("--out_dir", type=str, default="outputs/gp")

    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--i_star", type=int, required=True)

    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--selection_mode", type=str, default="mask_topk",
                    choices=["mask_topk", "mask_threshold_then_topk", "explicit_list"])
    ap.add_argument("--explicit_list", type=str, default="")
    ap.add_argument("--tau", type=float, default=1e-2)

    ap.add_argument("--n_trials", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--screen_scales", type=str, default="0,1,2,5")
    ap.add_argument("--full_scales", type=str, default="0,1,2,5,10,15,20,40,60,80")
    ap.add_argument("--topm_full", type=int, default=10)
    ap.add_argument("--flip_threshold", type=float, default=95.0)

    ap.add_argument("--lda_results_json", type=str, default="")
    ap.add_argument("--lda_reg", type=float, default=0.0)
    ap.add_argument("--lda_shrink", type=float, default=0.0)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])

    args = ap.parse_args()

    # Seeds
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARN] cuda requested but not available; using cpu.")
        device = "cpu"

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    exp_dir = out_dir / "exp3_random_rot" / f"L{args.layer}_H{args.head}_k{args.k}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    trials_jsonl = exp_dir / "trials.jsonl"
    summary_json = exp_dir / "summary.json"

    screen_scales = _parse_float_list(args.screen_scales, default=[0, 1, 2, 5])
    full_scales = _parse_float_list(args.full_scales, default=[0, 1, 2, 5, 10, 15, 20])
    if 0.0 not in [float(x) for x in screen_scales]:
        screen_scales = [0.0] + screen_scales
    if 0.0 not in [float(x) for x in full_scales]:
        full_scales = [0.0] + full_scales

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    he_ids = tokenizer.encode(" he", add_special_tokens=False)
    she_ids = tokenizer.encode(" she", add_special_tokens=False)
    if len(he_ids) != 1 or len(she_ids) != 1:
        raise ValueError(f'" he"/" she" not single tokens: he_ids={he_ids}, she_ids={she_ids}')
    he_id, she_id = int(he_ids[0]), int(she_ids[0])
    print("[TOKENS] he_id =", he_id, "she_id =", she_id)

    # Data
    train_rows = gp.load_gp_csv(str(data_dir / args.train_csv))
    test_rows = gp.load_gp_csv(str(data_dir / args.test_csv))
    print(f"[DATA] train_rows={len(train_rows)} test_rows={len(test_rows)}")
    gp.sanity_check_pronoun_tokenization(tokenizer, train_rows, max_checks=50)

    train_ex = expand_rows_to_examples(train_rows)
    test_ex = expand_rows_to_examples(test_rows)

    # Sanity: no trailing spaces (your setup)
    for i in range(min(20, len(train_ex))):
        if train_ex[i]["text"].endswith(" "):
            raise ValueError("Found a prefix ending with space. Your GP setup assumes NO trailing spaces.")

    # Model
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Artifacts
    svd_path = out_dir / "svd_cache.pt"
    masks_path = out_dir / "masks.pt"
    if not svd_path.exists():
        raise FileNotFoundError(f"Missing svd_cache.pt: {svd_path} (run train_gp_masks_and_dump_ov_logit_receptors.py first)")
    if not masks_path.exists():
        raise FileNotFoundError(f"Missing masks.pt: {masks_path} (run train_gp_masks_and_dump_ov_logit_receptors.py first)")

    _qk, ov, _mlp_in, _mlp_out, _rank_total_ov = gp.load_svd_cache(str(svd_path), device=device)
    masks_dict = torch.load(str(masks_path), map_location="cpu")

    L, H, i_star = int(args.layer), int(args.head), int(args.i_star)
    svd = ov[L][H]
    if not (0 <= i_star < svd.r):
        raise ValueError(f"--i_star {i_star} out of range (0..{svd.r-1})")

    mask_vec = masks_dict["ov"][L][H].detach().float().cpu()
    idx_explicit = parse_int_list(args.explicit_list)
    idx_list = choose_indices(mask_vec, int(args.k), str(args.selection_mode), float(args.tau), idx_explicit, must_include=i_star)

    print("\n[SELECT] idx_list:")
    for j in idx_list:
        print(f"  idx={j:4d}  mask={float(mask_vec[j]):.6g}  sigma={float(svd.S[j].detach().cpu().item()):.6g}")

    # Selected SVD pieces
    U_sel = torch.stack([svd.U[:, j] for j in idx_list], dim=-1).contiguous().to(device)
    V_sel = torch.stack([svd.Vh[j, :] for j in idx_list], dim=0).contiguous().to(device)
    S_sel = torch.tensor([float(svd.S[j].detach().cpu().item()) for j in idx_list], device=device, dtype=torch.float32)

    # STAR pieces
    sigma_star = float(svd.S[i_star].detach().cpu().item())
    v_star = svd.Vh[i_star, :].to(device).to(torch.float32)

    # Cache TRAIN A_train
    print("\n[TRAIN] caching A_train ...")
    A_chunks: List[torch.Tensor] = []
    y_train: List[str] = []
    for batch in batches(train_ex, int(args.batch_size)):
        texts = [e["text"] for e in batch]
        labels = [e["label"] for e in batch]
        tokens, last_idx = tokenize_texts(tokenizer, texts, device=device)
        nu, _resid, _logits = forward_collect_nu_resid_logits_last(model, tokens, last_idx, layer=L, head=H)
        A_chunks.append((nu @ U_sel).detach().float().cpu())
        y_train.extend(labels)

    A_train = torch.cat(A_chunks, dim=0)
    he_mask_train = torch.tensor([1 if y == "he" else 0 for y in y_train], dtype=torch.bool)
    if int(he_mask_train.sum().item()) == 0 or int((~he_mask_train).sum().item()) == 0:
        raise ValueError("Need both he and she examples in train.")

    # Cache TEST A_test, resid_test, base_logits, base_pred
    print("\n[TEST] caching A_test / resid_test / base_logits ...")
    Atest_chunks: List[torch.Tensor] = []
    y_test: List[str] = []
    resid_chunks: List[torch.Tensor] = []
    logits_chunks: List[torch.Tensor] = []
    pred_chunks: List[torch.Tensor] = []

    for batch in batches(test_ex, int(args.batch_size)):
        texts = [e["text"] for e in batch]
        labels = [e["label"] for e in batch]
        tokens, last_idx = tokenize_texts(tokenizer, texts, device=device)
        nu, resid_last, logits_last = forward_collect_nu_resid_logits_last(model, tokens, last_idx, layer=L, head=H)

        Atest_chunks.append((nu @ U_sel).detach().float().cpu())
        y_test.extend(labels)
        resid_chunks.append(resid_last.detach().float().cpu())
        logits_chunks.append(logits_last.detach().float().cpu())
        pred_chunks.append(torch.argmax(logits_last, dim=-1).detach().cpu())

    A_test = torch.cat(Atest_chunks, dim=0)
    resid_test = torch.cat(resid_chunks, dim=0)
    base_logits = torch.cat(logits_chunks, dim=0).half()
    base_pred = torch.cat(pred_chunks, dim=0)

    y_is_he = torch.tensor([1 if y == "he" else 0 for y in y_test], dtype=torch.bool)
    y_is_she = ~y_is_he
    denom_he_mask = y_is_he & (base_pred == he_id)
    denom_she_mask = y_is_she & (base_pred == she_id)
    print(f"[DENOM] he={int(denom_he_mask.sum())}/{int(y_is_he.sum())}  she={int(denom_she_mask.sum())}/{int(y_is_she.sum())}")

    # STAR targets (swap means)
    t_star = idx_list.index(i_star)
    a_star_train = A_train[:, t_star]
    a_star_test = A_test[:, t_star]
    mu_he_star = float(a_star_train[he_mask_train].mean().item())
    mu_she_star = float(a_star_train[~he_mask_train].mean().item())
    target_star_test = torch.where(y_is_he, torch.tensor(mu_she_star), torch.tensor(mu_he_star)).float()

    star_screen = eval_sweep(model, resid_test, base_logits, denom_he_mask, denom_she_mask,
                             he_id, she_id, a_star_test.float(), target_star_test,
                             v_star, sigma_star, screen_scales, int(args.batch_size), device)
    print_table("STAR (screen)", screen_scales, star_screen)
    star_headline = headline_kl_at_flip(star_screen, screen_scales, float(args.flip_threshold))
    print(f"[STAR] headline_screen KL@{args.flip_threshold}% = {star_headline:.6g}")

    # LDA baseline (compute, and also optionally load exp2 json just for reference)
    lda_loaded_headline = None
    if (args.lda_results_json or "").strip():
        p = Path(args.lda_results_json)
        if p.exists():
            with open(p, "r", encoding="utf-8") as f:
                j = json.load(f)
            sweep_rot = j.get("sweep_rotated", {})
            scales_json = [float(x) for x in (j.get("config", {}).get("sigma_scales", []) or [])]
            lda_loaded_headline = headline_kl_at_flip(sweep_rot, scales_json, float(args.flip_threshold))
            print(f"\n[LDA] loaded exp2 json headline (on its scales) = {lda_loaded_headline:.6g}")

    print("\n[LDA] computing internally ...")
    mu_he_vec = A_train[he_mask_train].mean(dim=0)
    mu_she_vec = A_train[~he_mask_train].mean(dim=0)
    d = (mu_he_vec - mu_she_vec)

    Xh = (A_train[he_mask_train] - mu_he_vec).to(dtype=torch.float32)
    Xs = (A_train[~he_mask_train] - mu_she_vec).to(dtype=torch.float32)
    cov_h = (Xh.T @ Xh) / float(max(1, Xh.shape[0]))
    cov_s = (Xs.T @ Xs) / float(max(1, Xs.shape[0]))
    Sw = cov_h + cov_s

    k_dim = int(Sw.shape[0])
    shrink = float(args.lda_shrink)
    if shrink > 0.0:
        tr = float(torch.trace(Sw).item())
        avg_var = tr / max(1, k_dim)
        Sw = (1.0 - shrink) * Sw + shrink * (avg_var * torch.eye(k_dim))

    reg = float(args.lda_reg)
    if reg <= 0.0:
        tr = float(torch.trace(Sw).item())
        avg_var = tr / max(1, k_dim)
        reg = 1e-3 * avg_var + 1e-6

    w_raw = torch.linalg.solve(Sw + reg * torch.eye(k_dim), d.to(dtype=torch.float32))
    w_lda = (w_raw / w_raw.norm()).cpu()

    w_lda = orient_w_by_receptor(w_lda, V_sel, model, he_id, she_id, device=device)

    a_lda_train = A_train @ w_lda
    mu_he_lda = float(a_lda_train[he_mask_train].mean().item())
    mu_she_lda = float(a_lda_train[~he_mask_train].mean().item())
    target_lda_test = torch.where(y_is_he, torch.tensor(mu_she_lda), torch.tensor(mu_he_lda)).float()

    sigma_eff_lda, v_eff_unit_lda = build_intervention_direction(w_lda, S_sel, V_sel, device=device)
    a_lda_test = (A_test @ w_lda).float()

    lda_screen = eval_sweep(model, resid_test, base_logits, denom_he_mask, denom_she_mask,
                            he_id, she_id, a_lda_test, target_lda_test,
                            v_eff_unit_lda, sigma_eff_lda, screen_scales, int(args.batch_size), device)
    print_table("LDA (screen)", screen_scales, lda_screen)
    lda_headline = headline_kl_at_flip(lda_screen, screen_scales, float(args.flip_threshold))
    print(f"[LDA] headline_screen KL@{args.flip_threshold}% = {lda_headline:.6g}")

    # ----------------------------
    # Random trials: screening
    # ----------------------------
    print(f"\n[RANDOM] n_trials={args.n_trials}  topm_full={args.topm_full}")
    with open(trials_jsonl, "w", encoding="utf-8") as f:
        f.write("")

    trial_records: List[Dict[str, Any]] = []
    for t in range(int(args.n_trials)):
        trial_seed = int(args.seed) + t

        R = sample_haar_orthogonal(int(args.k), trial_seed)
        A_rot = A_train @ R
        d_axis = cohen_d_per_axis(A_rot, he_mask_train)
        j_best = int(torch.argmax(torch.abs(d_axis)).item())

        w = R[:, j_best].contiguous()
        w = w / max(1e-12, float(w.norm().item()))
        w = orient_w_by_receptor(w, V_sel, model, he_id, she_id, device=device)

        a_train = A_train @ w
        mu_he = float(a_train[he_mask_train].mean().item())
        mu_she = float(a_train[~he_mask_train].mean().item())
        target_test = torch.where(y_is_he, torch.tensor(mu_she), torch.tensor(mu_he)).float()

        sigma_eff, v_eff_unit = build_intervention_direction(w, S_sel, V_sel, device=device)
        a_test = (A_test @ w).float()

        sweep_screen = eval_sweep(model, resid_test, base_logits, denom_he_mask, denom_she_mask,
                                  he_id, she_id, a_test, target_test,
                                  v_eff_unit, sigma_eff, screen_scales, int(args.batch_size), device)
        headline_screen = headline_kl_at_flip(sweep_screen, screen_scales, float(args.flip_threshold))

        rec = {
            "type": "screen",
            "trial_id": int(t),
            "seed": int(trial_seed),
            "j_best": int(j_best),
            "headline_screen": float(headline_screen),
            "w": [float(x) for x in w.tolist()],
            "sigma_eff": float(sigma_eff),
            "screen_sweep": sweep_screen,
        }
        trial_records.append(rec)
        with open(trials_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

        top3 = torch.topk(torch.abs(w), k=min(3, w.numel()))
        top3_pairs = [(int(i), float(v)) for v, i in zip(top3.values.tolist(), top3.indices.tolist())]
        print(f"[trial {t:03d}] seed={trial_seed} j_best={j_best} max|w|={float(torch.abs(w).max()):.3f} top3={top3_pairs} headline={headline_screen:.4g}")

    # Pick top-M by screening headline
    valid = [r for r in trial_records if math.isfinite(float(r["headline_screen"]))]
    valid_sorted = sorted(valid, key=lambda r: float(r["headline_screen"]))
    topm = valid_sorted[: max(0, min(int(args.topm_full), len(valid_sorted)))]

    # Full sweeps for top-M
    full_records: List[Dict[str, Any]] = []
    for r in topm:
        w = torch.tensor(r["w"], dtype=torch.float32)
        w = orient_w_by_receptor(w, V_sel, model, he_id, she_id, device=device)

        a_train = A_train @ w
        mu_he = float(a_train[he_mask_train].mean().item())
        mu_she = float(a_train[~he_mask_train].mean().item())
        target_test = torch.where(y_is_he, torch.tensor(mu_she), torch.tensor(mu_he)).float()

        sigma_eff, v_eff_unit = build_intervention_direction(w, S_sel, V_sel, device=device)
        a_test = (A_test @ w).float()

        sweep_full = eval_sweep(model, resid_test, base_logits, denom_he_mask, denom_she_mask,
                                he_id, she_id, a_test, target_test,
                                v_eff_unit, sigma_eff, full_scales, int(args.batch_size), device)
        headline_full = headline_kl_at_flip(sweep_full, full_scales, float(args.flip_threshold))

        rec_full = {
            "type": "full",
            "trial_id": int(r["trial_id"]),
            "seed": int(r["seed"]),
            "headline_full": float(headline_full),
            "full_sweep": sweep_full,
        }
        full_records.append(rec_full)
        with open(trials_jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec_full) + "\n")

        print(f"[FULL trial {r['trial_id']:03d}] headline_full={headline_full:.4g}")

    # Summary + percentile
    rand_scores = [float(r["headline_screen"]) for r in trial_records if math.isfinite(float(r["headline_screen"]))]
    if len(rand_scores) > 0 and math.isfinite(float(lda_headline)):
        lda_percentile = 100.0 * sum(1 for x in rand_scores if x >= float(lda_headline)) / len(rand_scores)
    else:
        lda_percentile = float("nan")

    summary = {
        "config": {
            "layer": L, "head": H, "i_star": i_star,
            "k": int(args.k), "idx_list": idx_list,
            "selection_mode": str(args.selection_mode), "tau": float(args.tau),
            "n_trials": int(args.n_trials), "seed": int(args.seed),
            "screen_scales": screen_scales, "full_scales": full_scales,
            "topm_full": int(args.topm_full),
            "flip_threshold": float(args.flip_threshold),
            "lda_reg": float(args.lda_reg),
            "lda_shrink": float(args.lda_shrink),
            "lda_results_json": str(args.lda_results_json),
        },
        "token_ids": {"he_id": he_id, "she_id": she_id},
        "STAR": {"headline_screen": float(star_headline), "screen_sweep": star_screen},
        "LDA": {
            "headline_screen": float(lda_headline),
            "screen_sweep": lda_screen,
            "loaded_exp2_headline": float(lda_loaded_headline) if lda_loaded_headline is not None else None,
        },
        "random": {
            "n_finite": int(len(rand_scores)),
            "median": float(torch.quantile(torch.tensor(rand_scores), 0.5).item()) if rand_scores else None,
            "p10": float(torch.quantile(torch.tensor(rand_scores), 0.1).item()) if rand_scores else None,
            "p90": float(torch.quantile(torch.tensor(rand_scores), 0.9).item()) if rand_scores else None,
            "lda_percentile_screen": float(lda_percentile),
        },
    }

    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\n[SUMMARY]")
    print(f"STAR headline_screen KL@{args.flip_threshold}% = {star_headline:.6g}")
    print(f"LDA  headline_screen KL@{args.flip_threshold}% = {lda_headline:.6g}")
    print(f"LDA better than {lda_percentile:.2f}% of random rotations (screen headline)")
    print(f"[SAVE] {trials_jsonl}")
    print(f"[SAVE] {summary_json}")


if __name__ == "__main__":
    main()
