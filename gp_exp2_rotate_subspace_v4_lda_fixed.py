
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
gp_exp2_rotate_subspace.py

Experiment 2: Rotate a mask-selected OV subspace inside one (layer, head) and test whether a learned
1D axis inside that subspace yields a "cleaner" logit receptor + stronger/lower-collateral interventions
than the best raw singular direction (sv_idx = i_star).

This script is intentionally "Regime A" compatible with your existing codebase:
- Uses the SAME tokenizer conventions (" he"/" she" are leading-space single tokens)
- Uses the SAME explicit forward pass logic (copied/adapted from gp_a4_single_direction_intervention.py)
- Reuses helpers from train_gp_masks_and_dump_ov_logit_receptors.py (imported as gp)

Requires (under --out_dir):
  svd_cache.pt   (OV SVD cache)
  masks.pt       (sigmoid masks)

Suggested run order on a fresh output directory:
  1) train masks (builds SVD cache if missing)
  2) run this script (optionally also run A4 to pick a good i_star)

Example:
  python gp_exp2_rotate_subspace.py \
    --data_dir data_main --train_csv train_1k_gp.csv --test_csv test_gp.csv \
    --out_dir outputs/gp --layer 11 --head 8 --best_sv_idx 6 \
    --k 16 --selection_mode mask_topk --tau 1e-2 \
    --sigma_scales 0,1,2,5,10,15,20 \
    --batch_size 64 --device cuda --save_json 1
"""

from __future__ import annotations

import argparse
import json
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer

# Reuse exact helpers/conventions from your training file
import train_gp_masks_and_dump_ov_logit_receptors as gp


# ----------------------------
# Small utilities
# ----------------------------

def parse_sigma_scales(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return [0, 1, 2, 5, 10, 15, 20]
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


def make_gender_token_ids(tokenizer: GPT2TokenizerFast, strs: List[str]) -> List[int]:
    ids: List[int] = []
    for s in strs:
        tok_ids = tokenizer.encode(s, add_special_tokens=False)
        if len(tok_ids) == 1:
            ids.append(tok_ids[0])
    return ids


def receptor_metrics(
    tokenizer: GPT2TokenizerFast,
    receptor: torch.Tensor,  # (vocab,)
    male_ids: List[int],
    female_ids: List[int],
    top_k: int,
) -> Dict[str, Any]:
    top = gp.topk_tokens(tokenizer, receptor, top_k)
    bot = gp.bottomk_tokens(tokenizer, receptor, top_k)

    gender = set(male_ids) | set(female_ids)
    top_ids = [t["token_id"] for t in top]
    bot_ids = [t["token_id"] for t in bot]

    purity_top = sum(int(i in gender) for i in top_ids) / max(1, top_k)
    purity_bot = sum(int(i in gender) for i in bot_ids) / max(1, top_k)

    polarity = (sum(int(i in set(male_ids)) for i in top_ids) +
                sum(int(i in set(female_ids)) for i in bot_ids)) / max(1, 2 * top_k)

    return {
        "top_tokens": top,
        "bottom_tokens": bot,
        "purity_top": float(purity_top),
        "purity_bottom": float(purity_bot),
        "polarity": float(polarity),
    }


def kl_pq_from_logits(base_logits: torch.Tensor, cf_logits: torch.Tensor) -> torch.Tensor:
    """
    Returns KL(P||Q) per example, where P=softmax(base_logits), Q=softmax(cf_logits).
    base_logits, cf_logits: (N, V)
    """
    logp = F.log_softmax(base_logits, dim=-1)
    p = logp.exp()
    logq = F.log_softmax(cf_logits, dim=-1)
    return (p * (logp - logq)).sum(dim=-1)


# ----------------------------
# Forward pass (explicit, A4-style)
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
    Full original forward:
      - nu = concat(context_resid_t*, ones) at (layer, head), (B, d_model+1)
      - resid_last = final residual stream at position t*, (B, d_model)
      - logits_last = logits at t*, (B, vocab)

    Logic is adapted from gp_a4_single_direction_intervention.py for exact conventions.
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
# Main experiment
# ----------------------------

def choose_indices(
    mask_vec: torch.Tensor,
    k: int,
    selection_mode: str,
    tau: float,
    explicit_list: List[int],
    must_include: int,
) -> List[int]:
    """
    Returns list of length k, sorted by descending mask, guaranteed to include must_include.
    """
    mask_vec = mask_vec.detach().float().cpu()
    r_total = int(mask_vec.numel())
    if not (0 <= must_include < r_total):
        raise ValueError(f"best_sv_idx out of range: {must_include} (r_total={r_total})")

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
        # Replace the last element with must_include.
        idx[-1] = must_include

    # Remove duplicates while preserving order, then re-fill if needed.
    seen = set()
    uniq = []
    for i in idx:
        if i not in seen:
            uniq.append(int(i))
            seen.add(int(i))
    if len(uniq) < k:
        # Fill with next-best by mask
        all_sorted = torch.argsort(mask_vec, descending=True).tolist()
        for i in all_sorted:
            if i not in seen:
                uniq.append(int(i))
                seen.add(int(i))
            if len(uniq) == k:
                break

    # Final order: descending mask for logging stability (except explicit_list case where order is user intent).
    if selection_mode != "explicit_list":
        uniq = sorted(uniq, key=lambda i: float(mask_vec[i].item()), reverse=True)

    return uniq[:k]


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--data_dir", type=str, default="data_main")
    ap.add_argument("--train_csv", type=str, default="train_1k_gp.csv")
    ap.add_argument("--test_csv", type=str, default="test_gp.csv")
    ap.add_argument("--out_dir", type=str, default="outputs/gp")

    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--best_sv_idx", type=int, required=True, help="i_star (baseline singular direction to compare)")

    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--selection_mode", type=str, default="mask_topk",
                    choices=["mask_topk", "mask_threshold_then_topk", "explicit_list"])
    ap.add_argument("--explicit_list", type=str, default="", help="Comma-separated list of k indices when selection_mode=explicit_list")
    ap.add_argument("--tau", type=float, default=1e-2, help="Used for mask_threshold_then_topk")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--sigma_scales", type=str, default="0,1,2,5,10,15,20")
    ap.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--seed", type=int, default=0)

    ap.add_argument("--w_mode", type=str, default="mean_diff", choices=["mean_diff", "lda"],
                    help="How to choose the 1D axis inside the k-dim subspace: mean_diff or Fisher/LDA")
    ap.add_argument("--lda_reg", type=float, default=0.0,
                    help="Ridge regularization added to within-class scatter. If 0, choose an automatic small value.")
    ap.add_argument("--lda_shrink", type=float, default=0.0,
                    help="Optional shrinkage toward identity in [0,1). 0 disables shrinkage.")


    ap.add_argument("--topk_print_tokens", type=int, default=35)
    ap.add_argument("--save_json", type=int, default=1)

    ap.add_argument("--male_strings", type=str, default=" he, him, his, himself")
    ap.add_argument("--female_strings", type=str, default=" she, her, hers, herself")

    args = ap.parse_args()

    # Seed
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

    exp_dir = out_dir / "exp2_rotate" / f"L{args.layer}_H{args.head}_k{args.k}"
    exp_dir.mkdir(parents=True, exist_ok=True)

    # Tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad_token by default

    # Regime A pronoun tokens: leading-space single tokens
    he_ids = tokenizer.encode(" he", add_special_tokens=False)
    she_ids = tokenizer.encode(" she", add_special_tokens=False)
    if len(he_ids) != 1 or len(she_ids) != 1:
        raise ValueError(f'" he"/" she" not single tokens: he_ids={he_ids}, she_ids={she_ids}')
    he_id, she_id = he_ids[0], she_ids[0]
    print("[TOKENS] he_id =", he_id, "decoded =", tokenizer.decode([he_id], clean_up_tokenization_spaces=False))
    print("[TOKENS] she_id =", she_id, "decoded =", tokenizer.decode([she_id], clean_up_tokenization_spaces=False))

    # Gender token sets (must be leading-space variants)
    # Gender token sets (keep leading spaces; GP uses leading-space pronoun tokens)
    def _parse_gender_strings(s: str) -> List[str]:
        out: List[str] = []
        for part in (s or "").split(","):
            # Preserve leading spaces; only remove trailing whitespace/newlines/tabs
            p = part.replace("\n", "").replace("\r", "").replace("\t", "")
            p = p.rstrip()
            if p == "":
                continue
            # If user provided bare lowercase words like 'he', force leading-space form ' he'.
            # Preserve capitalization-only tokens like 'He' (no leading space) if the user wants them.
            if (not p.startswith(" ")) and p.isalpha() and p.islower():
                p = " " + p
            out.append(p)
        return out

    male_strs = _parse_gender_strings(args.male_strings)
    female_strs = _parse_gender_strings(args.female_strings)
    male_ids = make_gender_token_ids(tokenizer, male_strs)
    female_ids = make_gender_token_ids(tokenizer, female_strs)
    print(f"[GENDER TOKS] male={list(zip(male_strs, male_ids))} | female={list(zip(female_strs, female_ids))}")

    # Load data (keeps prefix normalization via gp.normalize_prefix)
    train_rows = gp.load_gp_csv(str(data_dir / args.train_csv))
    test_rows = gp.load_gp_csv(str(data_dir / args.test_csv))
    print(f"[DATA] train_rows={len(train_rows)} test_rows={len(test_rows)}")
    gp.sanity_check_pronoun_tokenization(tokenizer, train_rows, max_checks=50)

    train_ex = expand_rows_to_examples(train_rows)
    test_ex = expand_rows_to_examples(test_rows)

    # Tokenization sanity: prefixes should NOT end with a space (per your setup)
    for i in range(min(10, len(train_ex))):
        if train_ex[i]["text"].endswith(" "):
            raise ValueError("Found a prefix ending with space. Your GP setup assumes NO trailing spaces.")

    # Model
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    # Load artifacts
    svd_path = out_dir / "svd_cache.pt"
    masks_path = out_dir / "masks.pt"
    if not svd_path.exists():
        raise FileNotFoundError(f"Missing svd_cache.pt: {svd_path} (run train_gp_masks_and_dump_ov_logit_receptors.py first)")
    if not masks_path.exists():
        raise FileNotFoundError(f"Missing masks.pt: {masks_path} (run train_gp_masks_and_dump_ov_logit_receptors.py first)")

    qk, ov, mlp_in, mlp_out, rank_total_ov = gp.load_svd_cache(str(svd_path), device=device)
    masks_dict = torch.load(str(masks_path), map_location="cpu")

    L, H, i_star = int(args.layer), int(args.head), int(args.best_sv_idx)
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    if not (0 <= L < n_layers):
        raise ValueError(f"--layer {L} out of range (0..{n_layers-1})")
    if not (0 <= H < n_heads):
        raise ValueError(f"--head {H} out of range (0..{n_heads-1})")

    svd = ov[L][H]
    if not (0 <= i_star < svd.r):
        raise ValueError(f"--best_sv_idx {i_star} out of range (0..{svd.r-1}) for ov[L][H].r={svd.r}")

    mask_vec = masks_dict["ov"][L][H].detach().float().cpu()
    print(f"[MASK] ov[L][H] has r={svd.r} trainable dirs; i_star mask={float(mask_vec[i_star]):.6g}")

    idx_explicit = parse_int_list(args.explicit_list)
    idx_list = choose_indices(
        mask_vec=mask_vec,
        k=int(args.k),
        selection_mode=str(args.selection_mode),
        tau=float(args.tau),
        explicit_list=idx_explicit,
        must_include=i_star,
    )

    print("[SELECT] idx_list (sorted):")
    for j in idx_list:
        print(f"  idx={j:4d}  mask={float(mask_vec[j]):.6g}  sigma={float(svd.S[j].detach().cpu().item()):.6g}")

    # Cache selected SVD pieces
    # U_sel: (D+1,k), V_sel: (k,D), S_sel: (k,)
    U_sel = torch.stack([svd.U[:, j] for j in idx_list], dim=-1).contiguous()
    V_sel = torch.stack([svd.Vh[j, :] for j in idx_list], dim=0).contiguous()
    S_sel = torch.tensor([float(svd.S[j].detach().cpu().item()) for j in idx_list], device=device, dtype=torch.float32)

    U_sel = U_sel.to(device=device)
    V_sel = V_sel.to(device=device)
    # S_sel already on device

    # Baseline direction pieces (star)
    u_star = svd.U[:, i_star].to(device)
    sigma_star = svd.S[i_star].to(device)
    v_star = svd.Vh[i_star, :].to(device)

    # ----------------------------
    # TRAIN: collect A_train and fit w
    # ----------------------------
    print("\n[TRAIN] collecting a_mat = nu @ U_sel ...")
    A_chunks: List[torch.Tensor] = []
    y_chunks: List[str] = []

    for batch in batches(train_ex, args.batch_size):
        texts = [e["text"] for e in batch]
        labels = [e["label"] for e in batch]
        tokens, last_idx = tokenize_texts(tokenizer, texts, device=device)

        nu, _resid, _logits = forward_collect_nu_resid_logits_last(model, tokens, last_idx, layer=L, head=H)
        a_mat = nu @ U_sel  # (B,k)

        A_chunks.append(a_mat.detach().float().cpu())
        y_chunks.extend(labels)

    A_train = torch.cat(A_chunks, dim=0)  # (N,k)
    y_train = y_chunks

    he_mask = torch.tensor([1 if y == "he" else 0 for y in y_train], dtype=torch.bool)
    she_mask = ~he_mask

    if int(he_mask.sum().item()) == 0 or int(she_mask.sum().item()) == 0:
        raise ValueError("Need both he and she examples in train set after expansion.")

    mu_he_vec = A_train[he_mask].mean(dim=0)
    mu_she_vec = A_train[she_mask].mean(dim=0)
    d = (mu_he_vec - mu_she_vec)
    d_norm = float(d.norm().item())
    if d_norm < 1e-8:
        raise RuntimeError("Mean difference is ~0 in this subspace (no separable direction).")
    # Choose 1D axis w in the selected k-dim subspace
    if args.w_mode == "mean_diff":
        w = (d / d.norm()).to(dtype=torch.float32)  # (k,)
    else:
        # Fisher/LDA: w ∝ (S_w + λI)^{-1} (μ_he - μ_she)
        # where S_w is within-class scatter (sum of class covariances).
        Xh = (A_train[he_mask] - mu_he_vec).to(dtype=torch.float32)
        Xs = (A_train[she_mask] - mu_she_vec).to(dtype=torch.float32)
        nh = max(1, Xh.shape[0])
        ns = max(1, Xs.shape[0])
        cov_h = (Xh.T @ Xh) / float(nh)
        cov_s = (Xs.T @ Xs) / float(ns)
        Sw = cov_h + cov_s
        k_dim = int(Sw.shape[0])
        if not (0.0 <= float(args.lda_shrink) < 1.0):
            raise ValueError("--lda_shrink must be in [0,1).")
        if float(args.lda_shrink) > 0.0:
            tr = float(torch.trace(Sw).item())
            avg_var = tr / max(1, k_dim)
            Sw = (1.0 - float(args.lda_shrink)) * Sw + float(args.lda_shrink) * (avg_var * torch.eye(k_dim))
        reg = float(args.lda_reg)
        if reg <= 0.0:
            # Automatic small ridge based on average variance
            tr = float(torch.trace(Sw).item())
            avg_var = tr / max(1, k_dim)
            reg = 1e-3 * avg_var + 1e-6
        A_mat = Sw + reg * torch.eye(k_dim)
        d_f = d.to(dtype=torch.float32)
        w_raw = torch.linalg.solve(A_mat, d_f)
        w_norm = float(w_raw.norm().item())
        if w_norm < 1e-8 or not math.isfinite(w_norm):
            raise RuntimeError("LDA produced degenerate w (norm too small or non-finite).")
        w = (w_raw / w_raw.norm()).to(dtype=torch.float32)

    # Orientation: make 'he' score above 'she' on rotated receptor
    v_rot = (w.to(device)[:, None] * V_sel).sum(dim=0)  # (D,)
    receptor_rot = (v_rot @ model.W_U).detach().float().cpu()  # (vocab,)
    if float(receptor_rot[he_id].item()) < float(receptor_rot[she_id].item()):
        w = -w
        v_rot = -v_rot
        receptor_rot = -receptor_rot
    print(f"[W] mode={args.w_mode} | learned w over k={args.k} dims, ||w||={float(w.norm().item()):.6g}, oriented so ' he' > ' she'.")
    print("[W components] (aligned with idx_list order)")
    for t, idx_j in enumerate(idx_list):
        print(f"  t={t:2d} idx={idx_j:4d} w={float(w[t].item()):+.6g}  mask={float(mask_vec[idx_j].item()):.6g}  sigma={float(svd.S[idx_j].detach().cpu().item()):.6g}")

    # Compute rotated receptor metrics
    rec_metrics_rot = receptor_metrics(tokenizer, receptor_rot, male_ids, female_ids, args.topk_print_tokens)

    # Baseline receptor metrics (star)
    receptor_star = (v_star @ model.W_U).detach().float().cpu()
    rec_metrics_star = receptor_metrics(tokenizer, receptor_star, male_ids, female_ids, args.topk_print_tokens)

    print("\n[RECEPTOR] rotated axis top/bottom preview:")
    print("  top:", [t["token_str"] for t in rec_metrics_rot["top_tokens"][:10]])
    print("  bot:", [t["token_str"] for t in rec_metrics_rot["bottom_tokens"][:10]])
    print(f"  purity_top={rec_metrics_rot['purity_top']:.3f} purity_bottom={rec_metrics_rot['purity_bottom']:.3f} polarity={rec_metrics_rot['polarity']:.3f}")

    print("\n[RECEPTOR] star sv_idx receptor top/bottom preview:")
    print("  top:", [t["token_str"] for t in rec_metrics_star["top_tokens"][:10]])
    print("  bot:", [t["token_str"] for t in rec_metrics_star["bottom_tokens"][:10]])
    print(f"  purity_top={rec_metrics_star['purity_top']:.3f} purity_bottom={rec_metrics_star['purity_bottom']:.3f} polarity={rec_metrics_star['polarity']:.3f}")

    # Scalar separability on TRAIN
    a_new_train = A_train @ w.cpu()
    mu_he = float(a_new_train[he_mask].mean().item())
    mu_she = float(a_new_train[she_mask].mean().item())
    std_he = float(a_new_train[he_mask].std(unbiased=False).item())
    std_she = float(a_new_train[she_mask].std(unbiased=False).item())
    denom = math.sqrt(0.5 * (std_he ** 2 + std_she ** 2) + 1e-12)
    cohen_d = (mu_he - mu_she) / denom
    print(f"\n[SEPARABILITY TRAIN] a_new: mu_he={mu_he:.6g}±{std_he:.6g} | mu_she={mu_she:.6g}±{std_she:.6g} | cohen_d={cohen_d:.4g}")

    # Targets (swap means)
    target_for_he = mu_she
    target_for_she = mu_he

    # Build rotated intervention direction
    # alpha = w ⊙ S_sel, sigma_eff = ||alpha||, v_eff_unit = sum(alpha/sigma_eff * V_sel_j)
    alpha = (w.to(device) * S_sel)  # (k,)
    sigma_eff = float(alpha.norm().item())
    if sigma_eff < 1e-8:
        raise RuntimeError("sigma_eff is ~0 (unexpected).")
    v_eff_unit = ((alpha / alpha.norm())[:, None] * V_sel).sum(dim=0)  # (D,)
    v_eff_norm = float(v_eff_unit.norm().item())
    print(f"\n[ROT INTERV] sigma_eff={sigma_eff:.6g} | ||v_eff_unit||={v_eff_norm:.6g} (should be ~1).")

    # ----------------------------
    # TEST: cache baseline once
    # ----------------------------
    sigma_scales = parse_sigma_scales(args.sigma_scales)
    if 0.0 not in [float(x) for x in sigma_scales]:
        sigma_scales = [0.0] + sigma_scales
    print(f"[SWEEP] sigma_scales={sigma_scales}")

    print("\n[TEST] caching baseline nu/resid/logits for test examples ...")
    Atest_chunks: List[torch.Tensor] = []
    y_test: List[str] = []
    resid_chunks: List[torch.Tensor] = []
    logits_chunks: List[torch.Tensor] = []
    base_pred_chunks: List[torch.Tensor] = []
    a_star_chunks: List[torch.Tensor] = []

    for batch in batches(test_ex, args.batch_size):
        texts = [e["text"] for e in batch]
        labels = [e["label"] for e in batch]
        tokens, last_idx = tokenize_texts(tokenizer, texts, device=device)

        nu, resid_last, logits_last = forward_collect_nu_resid_logits_last(model, tokens, last_idx, layer=L, head=H)
        a_mat = nu @ U_sel  # (B,k)
        ai_star = (nu * u_star[None, :]).sum(dim=-1)  # (B,)

        Atest_chunks.append(a_mat.detach().float().cpu())
        a_star_chunks.append(ai_star.detach().float().cpu())
        y_test.extend(labels)

        resid_chunks.append(resid_last.detach().float().cpu())
        logits_chunks.append(logits_last.detach().float().cpu())
        base_pred_chunks.append(torch.argmax(logits_last, dim=-1).detach().cpu())

    A_test = torch.cat(Atest_chunks, dim=0)              # (N,k)
    a_new_test = (A_test @ w.cpu())                      # (N,)
    ai_star_test = torch.cat(a_star_chunks, dim=0)       # (N,)
    resid_test = torch.cat(resid_chunks, dim=0)          # (N,D)
    base_logits = torch.cat(logits_chunks, dim=0).half()  # (N,V) store fp16 to save RAM
    base_pred = torch.cat(base_pred_chunks, dim=0)       # (N,)

    N = int(base_pred.numel())
    print(f"[TEST] N={N} examples cached.")

    # Denominators: only examples where baseline predicts the correct pronoun token
    y_is_he = torch.tensor([1 if y == "he" else 0 for y in y_test], dtype=torch.bool)
    y_is_she = ~y_is_he
    denom_he_mask = y_is_he & (base_pred == he_id)
    denom_she_mask = y_is_she & (base_pred == she_id)
    print(f"[DENOM] he_denom={int(denom_he_mask.sum().item())}/{int(y_is_he.sum().item())} | "
          f"she_denom={int(denom_she_mask.sum().item())}/{int(y_is_she.sum().item())}")

    # Precompute baseline logitdiffs
    base_logit_he = base_logits[:, he_id]
    base_logit_she = base_logits[:, she_id]
    base_diff_he = (base_logit_he - base_logit_she)      # for he-labeled analysis
    base_diff_she = (base_logit_she - base_logit_he)     # for she-labeled analysis

    # ----------------------------
    # Sweep: rotated intervention (chunked to avoid GPU OOM)
    # ----------------------------
    results_rot: Dict[str, Any] = {}
    results_star: Dict[str, Any] = {}

    # Compute baseline logitdiffs on CPU (cast to fp32)
    base_logits_f = base_logits.float()
    base_logit_he = base_logits_f[:, he_id]
    base_logit_she = base_logits_f[:, she_id]
    base_diff_he = (base_logit_he - base_logit_she)      # for he-labeled analysis
    base_diff_she = (base_logit_she - base_logit_he)     # for she-labeled analysis

    denom_he = int(denom_he_mask.sum().item())
    denom_she = int(denom_she_mask.sum().item())

    # Vectorized targets per example (CPU)

    # ----------------------------
    # STAR TRAIN: compute means for A4 baseline direction (i_star) so the swap-targets are well-defined
    # ----------------------------
    print("\n[STAR TRAIN] computing mu_he_star/mu_she_star for A4 baseline ...")
    star_ai_chunks: List[torch.Tensor] = []
    star_y: List[str] = []
    for batch in batches(train_ex, args.batch_size):
        texts = [e["text"] for e in batch]
        labels = [e["label"] for e in batch]
        tokens, last_idx = tokenize_texts(tokenizer, texts, device=device)
        nu, _resid, _logits = forward_collect_nu_resid_logits_last(model, tokens, last_idx, layer=L, head=H)
        ai_star = (nu * u_star[None, :]).sum(dim=-1)  # (B,)
        star_ai_chunks.append(ai_star.detach().float().cpu())
        star_y.extend(labels)

    ai_star_train = torch.cat(star_ai_chunks, dim=0)
    y_star_is_he = torch.tensor([1 if y == "he" else 0 for y in star_y], dtype=torch.bool)
    if int(y_star_is_he.sum().item()) == 0 or int((~y_star_is_he).sum().item()) == 0:
        raise ValueError("Need both he and she examples in train set for STAR baseline mean computation.")
    mu_he_star = float(ai_star_train[y_star_is_he].mean().item())
    mu_she_star = float(ai_star_train[~y_star_is_he].mean().item())
    print(f"[STAR TRAIN] mu_he_star={mu_he_star:.6g}  mu_she_star={mu_she_star:.6g}")
    target_new_cpu = torch.where(y_is_he, torch.tensor(target_for_he), torch.tensor(target_for_she)).float()
    target_star_cpu = torch.where(y_is_he, torch.tensor(mu_she_star), torch.tensor(mu_he_star)).float()

    # Direction vectors + sigmas (device)
    v_eff_unit_d = v_eff_unit.to(device).to(torch.float32)
    v_star_d = v_star.to(device).to(torch.float32)
    sigma_star_f = float(sigma_star.detach().cpu().item())

    def run_sweep(
        a_scalar_cpu: torch.Tensor,          # (N,)
        target_cpu: torch.Tensor,            # (N,)
        v_dir_d: torch.Tensor,               # (D,)
        sigma_value: float,
    ) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        N_local = int(a_scalar_cpu.numel())

        for scale in sigma_scales:
            scale_f = float(scale)
            flips_he = 0
            flips_she = 0
            other_all = 0
            other_he = 0
            other_she = 0
            kl_sum = 0.0

            diff_he_all: List[torch.Tensor] = []
            diff_she_all: List[torch.Tensor] = []

            for i0 in range(0, N_local, args.batch_size):
                i1 = min(N_local, i0 + args.batch_size)

                resid_b = resid_test[i0:i1].to(device).float()                     # (B,D)
                base_logits_b = base_logits[i0:i1].to(device).float()              # (B,V)
                a_b = a_scalar_cpu[i0:i1].to(device).float()                       # (B,)
                t_b = target_cpu[i0:i1].to(device).float()                         # (B,)

                delta_coeff = (t_b - a_b) * (scale_f * float(sigma_value))         # (B,)
                resid_cf = resid_b + delta_coeff[:, None] * v_dir_d[None, :]       # (B,D)

                logits_cf = model.ln_final(resid_cf) @ model.W_U + model.b_U       # (B,V)
                pred_cf = torch.argmax(logits_cf, dim=-1)

                # flips (denom masks are on CPU; slice + move)
                denom_he_b = denom_he_mask[i0:i1].to(device)
                denom_she_b = denom_she_mask[i0:i1].to(device)
                flips_he += int((denom_he_b & (pred_cf == she_id)).sum().item())
                flips_she += int((denom_she_b & (pred_cf == he_id)).sum().item())

                # "Third-token wins": predictions that are neither ' he' nor ' she'.
                other_mask = (pred_cf != he_id) & (pred_cf != she_id)
                other_all += int(other_mask.sum().item())
                other_he += int((denom_he_b & other_mask).sum().item())
                other_she += int((denom_she_b & other_mask).sum().item())

                # KL collateral
                kl_sum += float(kl_pq_from_logits(base_logits_b, logits_cf).sum().item())

                # diffs for later stats on CPU
                diff_he_all.append((logits_cf[:, he_id] - logits_cf[:, she_id]).detach().cpu())
                diff_she_all.append((logits_cf[:, she_id] - logits_cf[:, he_id]).detach().cpu())

            diff_he_cf = torch.cat(diff_he_all, dim=0)
            diff_she_cf = torch.cat(diff_she_all, dim=0)

            out[str(scale_f)] = {
                "flip_he_pct": 100.0 * flips_he / max(1, denom_he),
                "flip_he_count": int(flips_he),
                "flip_he_denom": int(denom_he),
                "flip_she_pct": 100.0 * flips_she / max(1, denom_she),
                "flip_she_count": int(flips_she),
                "flip_she_denom": int(denom_she),
                "other_pct_all": 100.0 * other_all / max(1, N_local),
                "other_pct_he_denom": 100.0 * other_he / max(1, denom_he),
                "other_pct_she_denom": 100.0 * other_she / max(1, denom_she),
                "kl_mean": float(kl_sum / max(1, N_local)),
                "base_diff_he_mean": float(base_diff_he[y_is_he].mean().item()),
                "base_diff_he_std": float(base_diff_he[y_is_he].std(unbiased=False).item()),
                "cf_diff_he_mean": float(diff_he_cf[y_is_he].mean().item()),
                "cf_diff_he_std": float(diff_he_cf[y_is_he].std(unbiased=False).item()),
                "base_diff_she_mean": float(base_diff_she[y_is_she].mean().item()),
                "base_diff_she_std": float(base_diff_she[y_is_she].std(unbiased=False).item()),
                "cf_diff_she_mean": float(diff_she_cf[y_is_she].mean().item()),
                "cf_diff_she_std": float(diff_she_cf[y_is_she].std(unbiased=False).item()),
            }

        return out

    results_rot = run_sweep(
        a_scalar_cpu=a_new_test.float(),
        target_cpu=target_new_cpu,
        v_dir_d=v_eff_unit_d,
        sigma_value=float(sigma_eff),
    )

    results_star = run_sweep(
        a_scalar_cpu=ai_star_test.float(),
        target_cpu=target_star_cpu,
        v_dir_d=v_star_d,
        sigma_value=float(sigma_star_f),
    )

    # Pretty print table
    def print_table(title: str, res: Dict[str, Any]) -> None:
        print(f"\n=== {title} ===")
        print("scale\tflip_he%\tflip_she%\tother%\tKL_mean")
        for s in sigma_scales:
            k = str(float(s))
            r = res[k]
            print(f"{float(s):g}\t{r['flip_he_pct']:.2f}\t{r['flip_she_pct']:.2f}\t{r['other_pct_all']:.2f}\t{r['kl_mean']:.4g}")

    print_table("ROTATED AXIS", results_rot)
    print_table("STAR (A4 baseline)", results_star)

    # Sanity: scale=0 should match baseline (KL ~= 0)
    r0 = results_rot[str(0.0)]
    r0s = results_star[str(0.0)]
    print("\n[SANITY] scale=0 should not change outputs (KL ~ 0, flips 0 unless numerical argmax ties).")
    print(f"  rotated: KL={r0['kl_mean']:.6g}, flip_he={r0['flip_he_count']}/{r0['flip_he_denom']}, flip_she={r0['flip_she_count']}/{r0['flip_she_denom']}")
    print(f"  star:    KL={r0s['kl_mean']:.6g}, flip_he={r0s['flip_he_count']}/{r0s['flip_he_denom']}, flip_she={r0s['flip_she_count']}/{r0s['flip_she_denom']}")

    # Save JSON
    payload = {
        "config": {
            "data_dir": str(args.data_dir),
            "train_csv": str(args.train_csv),
            "test_csv": str(args.test_csv),
            "out_dir": str(args.out_dir),
            "layer": L,
            "head": H,
            "best_sv_idx": i_star,
            "k": int(args.k),
            "selection_mode": str(args.selection_mode),
            "tau": float(args.tau),
            "idx_list": idx_list,
            "sigma_scales": [float(x) for x in sigma_scales],
            "batch_size": int(args.batch_size),
            "seed": int(args.seed),
            "w_mode": str(args.w_mode),
            "lda_reg": float(args.lda_reg),
            "lda_shrink": float(args.lda_shrink),
            "male_strings": male_strs,
            "female_strings": female_strs,
        },
        "selected_mask_values": {str(i): float(mask_vec[i].item()) for i in idx_list},
        "w": [float(x) for x in w.tolist()],
        "sigma_eff": float(sigma_eff),
        "v_eff_norm": float(v_eff_norm),
        "train_separability": {
            "mu_he": float(mu_he),
            "mu_she": float(mu_she),
            "std_he": float(std_he),
            "std_she": float(std_she),
            "cohen_d": float(cohen_d),
        },
        "receptor_rotated": {
            "purity_top": rec_metrics_rot["purity_top"],
            "purity_bottom": rec_metrics_rot["purity_bottom"],
            "polarity": rec_metrics_rot["polarity"],
            "top_tokens": rec_metrics_rot["top_tokens"],
            "bottom_tokens": rec_metrics_rot["bottom_tokens"],
        },
        "receptor_star": {
            "purity_top": rec_metrics_star["purity_top"],
            "purity_bottom": rec_metrics_star["purity_bottom"],
            "polarity": rec_metrics_star["polarity"],
            "top_tokens": rec_metrics_star["top_tokens"],
            "bottom_tokens": rec_metrics_star["bottom_tokens"],
        },
        "sweep_rotated": results_rot,
        "sweep_star": results_star,
        "train_ai_star": {"mu_he_star": float(mu_he_star), "mu_she_star": float(mu_she_star)},
    }

    if int(args.save_json) == 1:
        out_json = exp_dir / "exp2_results.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\n[SAVE] {out_json}")

    print("\n[DONE]")


if __name__ == "__main__":
    main()
