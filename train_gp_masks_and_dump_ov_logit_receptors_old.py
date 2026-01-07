#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
FILE: train_gp_masks_and_dump_ov_logit_receptors.py

Implements (a faithful, explicit version of) the Beyond-Components-style masking:
- Build augmented matrices (bias folded) for QK, OV, MLP_in, MLP_out
- Compute SVD once, discard singular values <= svd_eps
- Freeze GPT-2 weights, learn diagonal masks m=sigmoid(m_raw) with KL + L1
- QK: masked only (no complement)
- OV + MLP: masked + complement from corrupted activations
- After training: dump OV logit receptors for active OV directions (mask > tau)

Outputs (under out_dir):
  svd_cache.pt
  masks.pt
  ov_logit_receptors.jsonl
  summary.json
"""

import argparse
import csv
import json
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GPT2TokenizerFast

from transformer_lens import HookedTransformer


# ----------------------------
# Utils
# ----------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def normalize_prefix(s: str) -> str:
    s = s.rstrip()
    return s + " "


def detect_delimiter(path: str) -> str:
    # Try sniffing; fallback to '|', which is common when text contains commas.
    with open(path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "|", "\t", ";"])
        return dialect.delimiter
    except Exception:
        return "|"


def load_gp_csv(path: str) -> List[Dict[str, str]]:
    delim = detect_delimiter(path)
    rows: List[Dict[str, str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        required = ["prefix", "pronoun", "template", "name",
                    "corr_prefix", "corr_pronoun", "corr_template", "corr_name"]
        for col in required:
            if col not in reader.fieldnames:
                raise ValueError(f"{path}: missing required column '{col}'. "
                                 f"Found columns={reader.fieldnames} with delimiter='{delim}'")
        for r in reader:
            r["prefix"] = normalize_prefix(r["prefix"])
            r["corr_prefix"] = normalize_prefix(r["corr_prefix"])
            rows.append(r)
    return rows


def gelu_new(x: torch.Tensor) -> torch.Tensor:
    # GPT-2 "gelu_new" (tanh approximation)
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * (x ** 3))))


def get_act_fn(model: HookedTransformer):
    # Prefer model-configured activation if available; GPT-2 is gelu_new.
    act = getattr(model.cfg, "act_fn", "gelu_new")
    act = str(act).lower()
    if act in ("gelu_new", "gelu"):
        return gelu_new if act == "gelu_new" else F.gelu
    if act == "relu":
        return F.relu
    if act == "silu":
        return F.silu
    # Fallback
    return gelu_new


@dataclass
class SVDMat:
    # Compact SVD storage for W = U diag(S) Vh  (full_matrices=False)
    U: torch.Tensor    # (in_dim, r)
    S: torch.Tensor    # (r,)
    Vh: torch.Tensor   # (r, out_dim)
    # For QK we also want V = Vh.T sometimes; can derive on the fly

    @property
    def r(self) -> int:
        return int(self.S.shape[0])


def svd_compact(W: torch.Tensor, svd_eps: float) -> Tuple[SVDMat, int]:
    """
    Returns compact SVD with singular values > svd_eps.
    Also returns "rank_total" = count(S > 0) (numerically thresholded).
    """
    # W: (m, n)
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)  # U: (m,k), S: (k), Vh: (k,n)
    # rank_total before truncation: count of numerically-nonzero singular values
    rank_total = int((S > 1e-12).sum().item())
    keep = S > svd_eps
    if keep.sum().item() == 0:
        raise ValueError("All singular values were <= svd_eps. Increase svd_eps? Or check matrix construction.")
    U = U[:, keep]
    S = S[keep]
    Vh = Vh[keep, :]
    return SVDMat(U=U.contiguous(), S=S.contiguous(), Vh=Vh.contiguous()), rank_total


def topk_tokens(tokenizer: GPT2TokenizerFast, vec: torch.Tensor, k: int) -> List[Dict[str, Any]]:
    # vec: (vocab,)
    vals, idx = torch.topk(vec, k=k, largest=True)
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        tok = tokenizer.decode([i], clean_up_tokenization_spaces=False)
        out.append({"token_id": int(i), "token_str": tok, "score": float(v)})
    return out


def bottomk_tokens(tokenizer: GPT2TokenizerFast, vec: torch.Tensor, k: int) -> List[Dict[str, Any]]:
    vals, idx = torch.topk(vec, k=k, largest=False)
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        tok = tokenizer.decode([i], clean_up_tokenization_spaces=False)
        out.append({"token_id": int(i), "token_str": tok, "score": float(v)})
    return out


# ----------------------------
# Mask bank (trainable params)
# ----------------------------

class MaskBank(nn.Module):
    def __init__(self, qk: List[List[SVDMat]], ov: List[List[SVDMat]], mlp_in: List[SVDMat], mlp_out: List[SVDMat]):
        super().__init__()
        n_layers = len(qk)
        n_heads = len(qk[0])

        self.qk = nn.ModuleList([
            nn.ParameterList([nn.Parameter(torch.zeros(qk[l][h].r)) for h in range(n_heads)])
            for l in range(n_layers)
        ])
        self.ov = nn.ModuleList([
            nn.ParameterList([nn.Parameter(torch.zeros(ov[l][h].r)) for h in range(n_heads)])
            for l in range(n_layers)
        ])
        self.mlp_in = nn.ParameterList([nn.Parameter(torch.zeros(mlp_in[l].r)) for l in range(n_layers)])
        self.mlp_out = nn.ParameterList([nn.Parameter(torch.zeros(mlp_out[l].r)) for l in range(n_layers)])

    @staticmethod
    def sig(p: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(p)

    def l1_sum(self) -> torch.Tensor:
        acc = 0.0
        for l in range(len(self.qk)):
            for h in range(len(self.qk[l])):
                acc = acc + self.sig(self.qk[l][h]).mean()
                acc = acc + self.sig(self.ov[l][h]).mean()
            acc = acc + self.sig(self.mlp_in[l]).mean()
            acc = acc + self.sig(self.mlp_out[l]).mean()
        return acc


# ----------------------------
# Build SVD cache
# ----------------------------

@torch.no_grad()
def build_svd_cache(
    model: HookedTransformer,
    out_path: str,
    svd_eps: float,
    device_for_svd: str = "cpu"
) -> Dict[str, Any]:
    """
    Builds SVD cache for QK, OV (per layer/head) and MLP (per layer).
    Stores on CPU by default and saves to out_path.
    """
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_model = model.cfg.d_model
    d_head = model.cfg.d_head
    d_mlp = model.cfg.d_mlp

    # Pull shared b_O once per layer
    svd_cache: Dict[str, Any] = {
        "meta": {
            "n_layers": n_layers, "n_heads": n_heads,
            "d_model": d_model, "d_head": d_head, "d_mlp": d_mlp,
            "svd_eps": svd_eps,
        },
        "qk": [[None for _ in range(n_heads)] for _ in range(n_layers)],
        "ov": [[None for _ in range(n_heads)] for _ in range(n_layers)],
        "mlp_in": [None for _ in range(n_layers)],
        "mlp_out": [None for _ in range(n_layers)],
        "rank_total_ov": [[None for _ in range(n_heads)] for _ in range(n_layers)],  # for "full sparsity"
    }

    for l in range(n_layers):
        attn = model.blocks[l].attn
        mlp = model.blocks[l].mlp

        # MLP in/out
        W_in = mlp.W_in.to(device_for_svd)          # (d_model, d_mlp)
        b_in = mlp.b_in.to(device_for_svd)          # (d_mlp,)
        W_in_aug = torch.cat([W_in, b_in[None, :]], dim=0)  # (d_model+1, d_mlp)
        mlp_in_svd, _ = svd_compact(W_in_aug, svd_eps)
        svd_cache["mlp_in"][l] = {
            "U": mlp_in_svd.U.cpu(), "S": mlp_in_svd.S.cpu(), "Vh": mlp_in_svd.Vh.cpu()
        }

        W_out = mlp.W_out.to(device_for_svd)        # (d_mlp, d_model)
        b_out = mlp.b_out.to(device_for_svd)        # (d_model,)
        W_out_aug = torch.cat([W_out, b_out[None, :]], dim=0)  # (d_mlp+1, d_model)
        mlp_out_svd, _ = svd_compact(W_out_aug, svd_eps)
        svd_cache["mlp_out"][l] = {
            "U": mlp_out_svd.U.cpu(), "S": mlp_out_svd.S.cpu(), "Vh": mlp_out_svd.Vh.cpu()
        }

        # QK + OV per head
        b_O = attn.b_O.to(device_for_svd)  # (d_model,) shared across heads in TL

        for h in range(n_heads):
            W_Q = attn.W_Q[h].to(device_for_svd)  # (d_model, d_head)
            b_Q = attn.b_Q[h].to(device_for_svd)  # (d_head,)
            W_K = attn.W_K[h].to(device_for_svd)
            b_K = attn.b_K[h].to(device_for_svd)

            A = torch.cat([W_Q, b_Q[None, :]], dim=0)  # (d_model+1, d_head)
            B = torch.cat([W_K, b_K[None, :]], dim=0)  # (d_model+1, d_head)
            W_QK_aug = A @ B.T                          # (d_model+1, d_model+1)
            qk_svd, _ = svd_compact(W_QK_aug, svd_eps)
            svd_cache["qk"][l][h] = {
                "U": qk_svd.U.cpu(), "S": qk_svd.S.cpu(), "Vh": qk_svd.Vh.cpu()
            }

            W_V = attn.W_V[h].to(device_for_svd)  # (d_model, d_head)
            b_V = attn.b_V[h].to(device_for_svd)  # (d_head,)
            W_O = attn.W_O[h].to(device_for_svd)  # (d_head, d_model)

            W_OV = W_V @ W_O                      # (d_model, d_model)
            b_OV = (b_V @ W_O) + (b_O / n_heads)   # (d_model,) split shared bias across heads
            W_OV_aug = torch.cat([W_OV, b_OV[None, :]], dim=0)  # (d_model+1, d_model)
            ov_svd, rank_total = svd_compact(W_OV_aug, svd_eps)

            svd_cache["ov"][l][h] = {
                "U": ov_svd.U.cpu(), "S": ov_svd.S.cpu(), "Vh": ov_svd.Vh.cpu()
            }
            svd_cache["rank_total_ov"][l][h] = rank_total

        print(f"[SVD] layer {l+1}/{n_layers} done")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    torch.save(svd_cache, out_path)
    print(f"[SVD] saved: {out_path}")
    return svd_cache


def load_svd_cache(path: str, device: str) -> Tuple[List[List[SVDMat]], List[List[SVDMat]], List[SVDMat], List[SVDMat], List[List[int]]]:
    cache = torch.load(path, map_location="cpu")
    n_layers = cache["meta"]["n_layers"]
    n_heads = cache["meta"]["n_heads"]

    qk: List[List[SVDMat]] = [[None for _ in range(n_heads)] for _ in range(n_layers)]
    ov: List[List[SVDMat]] = [[None for _ in range(n_heads)] for _ in range(n_layers)]
    mlp_in: List[SVDMat] = [None for _ in range(n_layers)]
    mlp_out: List[SVDMat] = [None for _ in range(n_layers)]
    rank_total_ov: List[List[int]] = [[0 for _ in range(n_heads)] for _ in range(n_layers)]

    for l in range(n_layers):
        mlp_in[l] = SVDMat(
            U=cache["mlp_in"][l]["U"].to(device),
            S=cache["mlp_in"][l]["S"].to(device),
            Vh=cache["mlp_in"][l]["Vh"].to(device),
        )
        mlp_out[l] = SVDMat(
            U=cache["mlp_out"][l]["U"].to(device),
            S=cache["mlp_out"][l]["S"].to(device),
            Vh=cache["mlp_out"][l]["Vh"].to(device),
        )
        for h in range(n_heads):
            qk[l][h] = SVDMat(
                U=cache["qk"][l][h]["U"].to(device),
                S=cache["qk"][l][h]["S"].to(device),
                Vh=cache["qk"][l][h]["Vh"].to(device),
            )
            ov[l][h] = SVDMat(
                U=cache["ov"][l][h]["U"].to(device),
                S=cache["ov"][l][h]["S"].to(device),
                Vh=cache["ov"][l][h]["Vh"].to(device),
            )
            rank_total_ov[l][h] = int(cache["rank_total_ov"][l][h])

    return qk, ov, mlp_in, mlp_out, rank_total_ov


# ----------------------------
# Forward passes
# ----------------------------

def make_causal_mask(seq_len: int, device: str) -> torch.Tensor:
    return torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))


def attention_pattern_original(
    x_ln1: torch.Tensor,  # (B,S,D)
    W_Q: torch.Tensor, b_Q: torch.Tensor,  # (D,Dh), (Dh,)
    W_K: torch.Tensor, b_K: torch.Tensor,  # (D,Dh), (Dh,)
    causal: torch.Tensor,  # (S,S) bool
    d_head: int
) -> torch.Tensor:
    # q,k: (B,S,Dh)
    q = x_ln1 @ W_Q + b_Q
    k = x_ln1 @ W_K + b_K
    scores = (q @ k.transpose(-1, -2)) / math.sqrt(d_head)  # (B,S,S)
    scores = scores.masked_fill(~causal, -1e9)
    return F.softmax(scores, dim=-1)


# def attention_pattern_masked_qk(
#     x_ln1: torch.Tensor,       # (B,S,D)
#     qk_svd: SVDMat,             # U: (D+1,r), Vh: (r,D+1)
#     m: torch.Tensor,            # (r,)
#     causal: torch.Tensor        # (S,S)
# ) -> torch.Tensor:
#     B, S, D = x_ln1.shape
#     ones = torch.ones((B, S, 1), device=x_ln1.device, dtype=x_ln1.dtype)
#     x_aug = torch.cat([x_ln1, ones], dim=-1)  # (B,S,D+1)

#     U = qk_svd.U               # (D+1,r)
#     V = qk_svd.Vh.transpose(0, 1)  # (D+1,r)
#     Svals = qk_svd.S           # (r,)

#     # A = x_aug @ U  ; Bv = x_aug @ V
#     A = x_aug @ U              # (B,S,r)
#     Bv = x_aug @ V             # (B,S,r)
#     A = A * (m * Svals)        # (B,S,r)

#     scores = (A @ Bv.transpose(-1, -2)) / math.sqrt(D // (D // 64))  # scale by sqrt(d_head); robust fallback
#     scores = scores.masked_fill(~causal, -1e9)
#     return F.softmax(scores, dim=-1)


def attention_pattern_masked_qk(
    x_ln1: torch.Tensor,       # (B,S,D)
    qk_svd: SVDMat,             # U: (D+1,r), Vh: (r,D+1)
    m: torch.Tensor,            # (r,)
    causal: torch.Tensor,       # (S,S)
    d_head: int
) -> torch.Tensor:
    B, S, D = x_ln1.shape
    ones = torch.ones((B, S, 1), device=x_ln1.device, dtype=x_ln1.dtype)
    x_aug = torch.cat([x_ln1, ones], dim=-1)  # (B,S,D+1)

    U = qk_svd.U                           # (D+1,r)
    V = qk_svd.Vh.transpose(0, 1)          # (D+1,r)
    Svals = qk_svd.S                       # (r,)

    A = x_aug @ U                          # (B,S,r)
    Bv = x_aug @ V                         # (B,S,r)

    A = A * (m * Svals)                    # (B,S,r)
    scores = (A @ Bv.transpose(-1, -2)) / math.sqrt(d_head)  # (B,S,S)
    scores = scores.masked_fill(~causal, -1e9)
    return F.softmax(scores, dim=-1)


def apply_masked_linear_from_projections(
    z_clean: torch.Tensor,      # (B,S,r) == x_clean_aug @ U
    z_corr: torch.Tensor,       # (B,S,r) == x_corr_aug @ U   (cached)
    svd: SVDMat,                # S: (r,), Vh: (r,out_dim)
    m: torch.Tensor             # (r,)
) -> torch.Tensor:
    # out = (z_clean*(m*S) + z_corr*((1-m)*S)) @ Vh
    Svals = svd.S
    mix = (z_clean * (m * Svals)) + (z_corr * ((1.0 - m) * Svals))
    return mix @ svd.Vh  # (B,S,out_dim)


@torch.no_grad()
def corrupt_projections(
    model: HookedTransformer,
    tokens: torch.Tensor,            # (B,S)
    qk: List[List[SVDMat]],
    ov: List[List[SVDMat]],
    mlp_in: List[SVDMat],
    mlp_out: List[SVDMat],
    dtype_cache: torch.dtype = torch.float16,
) -> Tuple[List[List[torch.Tensor]], List[torch.Tensor], List[torch.Tensor]]:
    """
    Runs an ORIGINAL forward pass on corrupted prompts, but stores only low-dim SVD-space
    projections needed for complement terms:
      z_corr_ov[l][h]   = context_resid_aug @ U_ov[l][h]
      z_corr_mlp_in[l]  = x_ln2_aug @ U_mlp_in[l]
      z_corr_mlp_out[l] = gelu(pre)_aug @ U_mlp_out[l]
    """
    device = tokens.device
    act_fn = get_act_fn(model)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    d_head = model.cfg.d_head

    # embeddings
    x = model.embed(tokens) + model.pos_embed(tokens)

    B, S = tokens.shape
    causal = make_causal_mask(S, device=device)

    z_corr_ov: List[List[torch.Tensor]] = [[None for _ in range(n_heads)] for _ in range(n_layers)]
    z_corr_mlp_in: List[torch.Tensor] = [None for _ in range(n_layers)]
    z_corr_mlp_out: List[torch.Tensor] = [None for _ in range(n_layers)]

    for l in range(n_layers):
        block = model.blocks[l]
        attn = block.attn
        mlp = block.mlp

        # ln1
        x_ln1 = block.ln1(x)

        # attention per head
        head_outs = []
        for h in range(n_heads):
            pat = attention_pattern_original(
                x_ln1,
                attn.W_Q[h], attn.b_Q[h],
                attn.W_K[h], attn.b_K[h],
                causal,
                d_head
            )  # (B,S,S)

            # context_resid = pat @ x_ln1  (move W_V after averaging)
            context_resid = pat @ x_ln1  # (B,S,D)

            # cache OV projection: context_resid_aug @ U_ov
            ones = torch.ones((B, S, 1), device=device, dtype=context_resid.dtype)
            ctx_aug = torch.cat([context_resid, ones], dim=-1)  # (B,S,D+1)
            z = (ctx_aug @ ov[l][h].U).to(dtype_cache)  # (B,S,r)
            z_corr_ov[l][h] = z

            # original head output for continuation:
            v = (context_resid @ attn.W_V[h]) + attn.b_V[h]     # (B,S,Dh)
            out = (v @ attn.W_O[h])                             # (B,S,D)
            head_outs.append(out)

        attn_out = torch.stack(head_outs, dim=0).sum(dim=0) + attn.b_O  # (B,S,D)
        x = x + attn_out

        # ln2
        x_ln2 = block.ln2(x)

        # cache MLP_in projection: x_ln2_aug @ U_in
        ones2 = torch.ones((B, S, 1), device=device, dtype=x_ln2.dtype)
        x_ln2_aug = torch.cat([x_ln2, ones2], dim=-1)  # (B,S,D+1)
        z_corr_mlp_in[l] = (x_ln2_aug @ mlp_in[l].U).to(dtype_cache)  # (B,S,r_in)

        # original MLP forward (for continuation)
        pre = (x_ln2 @ mlp.W_in) + mlp.b_in
        h_act = act_fn(pre)

        # cache MLP_out projection: h_act_aug @ U_out
        ones3 = torch.ones((B, S, 1), device=device, dtype=h_act.dtype)
        h_aug = torch.cat([h_act, ones3], dim=-1)  # (B,S,d_mlp+1)
        z_corr_mlp_out[l] = (h_aug @ mlp_out[l].U).to(dtype_cache)  # (B,S,r_out)

        mlp_out_orig = (h_act @ mlp.W_out) + mlp.b_out
        x = x + mlp_out_orig

    return z_corr_ov, z_corr_mlp_in, z_corr_mlp_out


def masked_forward_logits_last(
    model: HookedTransformer,
    tokens: torch.Tensor,                # (B,S)
    qk: List[List[SVDMat]],
    ov: List[List[SVDMat]],
    mlp_in: List[SVDMat],
    mlp_out: List[SVDMat],
    masks: MaskBank,
    z_corr_ov: List[List[torch.Tensor]],
    z_corr_mlp_in: List[torch.Tensor],
    z_corr_mlp_out: List[torch.Tensor],
    last_indices: torch.Tensor,          # (B,)
) -> torch.Tensor:
    """
    Masked forward on CLEAN tokens using:
      - QK masked (no complement)
      - OV masked + complement from cached z_corr_ov
      - MLP masked + complement from cached z_corr_mlp_{in,out}
    Returns logits at each sample's last index: (B, vocab)
    """
    device = tokens.device
    act_fn = get_act_fn(model)

    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads

    x = model.embed(tokens) + model.pos_embed(tokens)
    B, S = tokens.shape
    causal = make_causal_mask(S, device=device)

    for l in range(n_layers):
        block = model.blocks[l]
        attn = block.attn

        # ln1
        x_ln1 = block.ln1(x)

        # attention heads
        head_outs = []
        for h in range(n_heads):
            m_qk = MaskBank.sig(masks.qk[l][h])
            # pat = attention_pattern_masked_qk(x_ln1, qk[l][h], m_qk, causal)  # (B,S,S)
            pat = attention_pattern_masked_qk(x_ln1, qk[l][h], m_qk, causal, model.cfg.d_head)

            # context_resid clean
            context_resid = pat @ x_ln1  # (B,S,D)

            # project clean context into OV-U space
            ones = torch.ones((B, S, 1), device=device, dtype=context_resid.dtype)
            ctx_aug = torch.cat([context_resid, ones], dim=-1)  # (B,S,D+1)
            z_clean = ctx_aug @ ov[l][h].U                       # (B,S,r)

            m_ov = MaskBank.sig(masks.ov[l][h])
            zc = z_corr_ov[l][h].to(z_clean.dtype)               # (B,S,r)
            out = apply_masked_linear_from_projections(z_clean, zc, ov[l][h], m_ov)  # (B,S,D)
            head_outs.append(out)

        attn_out = torch.stack(head_outs, dim=0).sum(dim=0)  # bias already folded into OV_aug (via /n_heads), do NOT add b_O here
        x = x + attn_out

        # MLP with complement
        mlp = block.mlp
        x_ln2 = block.ln2(x)

        ones2 = torch.ones((B, S, 1), device=device, dtype=x_ln2.dtype)
        x_ln2_aug = torch.cat([x_ln2, ones2], dim=-1)
        z_clean_in = x_ln2_aug @ mlp_in[l].U  # (B,S,r_in)

        m_in = MaskBank.sig(masks.mlp_in[l])
        zc_in = z_corr_mlp_in[l].to(z_clean_in.dtype)
        pre = apply_masked_linear_from_projections(z_clean_in, zc_in, mlp_in[l], m_in)  # (B,S,d_mlp)

        h_act = act_fn(pre)

        ones3 = torch.ones((B, S, 1), device=device, dtype=h_act.dtype)
        h_aug = torch.cat([h_act, ones3], dim=-1)
        z_clean_out = h_aug @ mlp_out[l].U  # (B,S,r_out)

        m_out = MaskBank.sig(masks.mlp_out[l])
        zc_out = z_corr_mlp_out[l].to(z_clean_out.dtype)
        mlp_out_masked = apply_masked_linear_from_projections(z_clean_out, zc_out, mlp_out[l], m_out)  # (B,S,D)

        x = x + mlp_out_masked

    x_final = model.ln_final(x)
    logits = x_final @ model.W_U + model.b_U  # (B,S,vocab)

    # gather last-position logits
    return logits[torch.arange(B, device=device), last_indices, :]


# ----------------------------
# Metrics + dumping
# ----------------------------

@torch.no_grad()
def compute_sparsities(
    masks: MaskBank,
    tau: float,
    rank_total_ov: List[List[int]]
) -> Dict[str, float]:
    n_layers = len(masks.qk)
    n_heads = len(masks.qk[0])

    # relative sparsity over ALL trainable directions (after svd_eps)
    n_trainable = 0
    n_kept = 0

    # full sparsity over OV directions "before truncation" (rank_total_ov)
    n_total_ov = 0
    n_kept_ov = 0

    for l in range(n_layers):
        for h in range(n_heads):
            m_qk = torch.sigmoid(masks.qk[l][h]).detach()
            m_ov = torch.sigmoid(masks.ov[l][h]).detach()
            n_trainable += m_qk.numel()
            n_trainable += m_ov.numel()
            n_kept += int((m_qk > tau).sum().item())
            n_kept += int((m_ov > tau).sum().item())

            n_total_ov += int(rank_total_ov[l][h])
            n_kept_ov += int((m_ov > tau).sum().item())

        m_in = torch.sigmoid(masks.mlp_in[l]).detach()
        m_out = torch.sigmoid(masks.mlp_out[l]).detach()
        n_trainable += m_in.numel() + m_out.numel()
        n_kept += int((m_in > tau).sum().item()) + int((m_out > tau).sum().item())

    rel_sparsity = 1.0 - (n_kept / max(1, n_trainable))
    full_sparsity_ov = 1.0 - (n_kept_ov / max(1, n_total_ov))

    return {
        "n_trainable": float(n_trainable),
        "n_kept": float(n_kept),
        "relative_sparsity_all": float(rel_sparsity),
        "n_total_ov_rank": float(n_total_ov),
        "n_kept_ov": float(n_kept_ov),
        "full_sparsity_ov": float(full_sparsity_ov),
    }


@torch.no_grad()
def save_masks(path: str, masks: MaskBank) -> None:
    out = {"qk": [], "ov": [], "mlp_in": [], "mlp_out": []}
    n_layers = len(masks.qk)
    n_heads = len(masks.qk[0])

    for l in range(n_layers):
        out["qk"].append([])
        out["ov"].append([])
        for h in range(n_heads):
            out["qk"][l].append(torch.sigmoid(masks.qk[l][h]).cpu())
            out["ov"][l].append(torch.sigmoid(masks.ov[l][h]).cpu())
        out["mlp_in"].append(torch.sigmoid(masks.mlp_in[l]).cpu())
        out["mlp_out"].append(torch.sigmoid(masks.mlp_out[l]).cpu())

    torch.save(out, path)
    print(f"[SAVE] masks -> {path}")


@torch.no_grad()
def dump_ov_logit_receptors(
    out_jsonl: str,
    model: HookedTransformer,
    tokenizer: GPT2TokenizerFast,
    ov: List[List[SVDMat]],
    masks: MaskBank,
    tau: float,
    top_k: int
) -> Dict[str, Any]:
    """
    For each (layer, head, sv_idx) with mask > tau:
      v = right singular vector (length d_model) from OV SVD
      receptor = v^T @ W_U
      dump top/bottom tokens
    """
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)
    n_layers = len(ov)
    n_heads = len(ov[0])

    count = 0
    with open(out_jsonl, "w", encoding="utf-8") as f:
        for l in range(n_layers):
            for h in range(n_heads):
                m = torch.sigmoid(masks.ov[l][h]).detach()
                svd = ov[l][h]
                # right singular vectors are rows of Vh: (r, d_model)
                for k in range(svd.r):
                    mv = float(m[k].item())
                    if mv <= tau:
                        continue
                    v_row = svd.Vh[k, :]  # (d_model,)
                    receptor = v_row @ model.W_U  # (vocab,)
                    rec = receptor.detach().cpu()
                    record = {
                        "layer": l,
                        "head": h,
                        "sv_idx": k,
                        "mask": mv,
                        "singular_value": float(svd.S[k].detach().cpu().item()),
                        "top_tokens": topk_tokens(tokenizer, rec, top_k),
                        "bottom_tokens": bottomk_tokens(tokenizer, rec, top_k),
                    }
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    count += 1

    print(f"[DUMP] ov logit receptors -> {out_jsonl} (active={count})")
    return {"num_active_ov_directions": count}


# ----------------------------
# Training
# ----------------------------

def batch_iter(rows: List[Dict[str, str]], batch_size: int, shuffle: bool = True):
    idx = list(range(len(rows)))
    if shuffle:
        random.shuffle(idx)
    for i in range(0, len(idx), batch_size):
        batch = [rows[j] for j in idx[i:i+batch_size]]
        yield batch


# def tokenize_batch(tokenizer: GPT2TokenizerFast, texts: List[str], device: str) -> Tuple[torch.Tensor, torch.Tensor]:
#     enc = tokenizer(
#         texts,
#         return_tensors="pt",
#         padding=True,
#         truncation=False,
#         add_special_tokens=False
#     )
#     tokens = enc["input_ids"].to(device)
#     attn = enc["attention_mask"].to(device)
#     last_idx = attn.sum(dim=1) - 1
#     return tokens, last_idx


def _encode_texts(tokenizer: GPT2TokenizerFast, texts: List[str]) -> List[List[int]]:
    # No padding here; we pad manually so clean+corr use the SAME max_len.
    return [tokenizer.encode(t, add_special_tokens=False) for t in texts]

def _pad_to_length(
    ids_list: List[List[int]],
    pad_len: int,
    pad_id: int,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B = len(ids_list)
    tokens = torch.full((B, pad_len), pad_id, dtype=torch.long)
    attn = torch.zeros((B, pad_len), dtype=torch.long)
    last_idx = torch.empty((B,), dtype=torch.long)

    for i, ids in enumerate(ids_list):
        L = len(ids)
        if L == 0:
            raise ValueError("Got an empty tokenized sequence. Check your prompts.")
        if L > pad_len:
            raise ValueError(f"Sequence length {L} > pad_len {pad_len} (should never happen).")
        tokens[i, :L] = torch.tensor(ids, dtype=torch.long)
        attn[i, :L] = 1
        last_idx[i] = L - 1

    return tokens.to(device), attn.to(device), last_idx.to(device)

def tokenize_pair_batch(
    tokenizer: GPT2TokenizerFast,
    clean_texts: List[str],
    corr_texts: List[str],
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Returns:
      tokens_clean: (B, Smax)
      last_idx_clean: (B,)
      tokens_corr: (B, Smax)
      last_idx_corr: (B,)
    where Smax is the SAME for clean and corr in this batch.
    """
    ids_clean = _encode_texts(tokenizer, clean_texts)
    ids_corr = _encode_texts(tokenizer, corr_texts)

    max_len = max(max(len(x) for x in ids_clean), max(len(x) for x in ids_corr))
    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        raise ValueError("tokenizer.pad_token_id is None. Ensure tokenizer.pad_token is set.")

    tokens_clean, _, last_clean = _pad_to_length(ids_clean, max_len, pad_id, device)
    tokens_corr, _, last_corr = _pad_to_length(ids_corr, max_len, pad_id, device)

    return tokens_clean, last_clean, tokens_corr, last_corr


@torch.no_grad()
def sanity_check_pronoun_tokenization(tokenizer: GPT2TokenizerFast, rows: List[Dict[str, str]], max_checks: int = 50) -> None:
    # Ensure " " + pronoun is a single token (recommended)
    bad = 0
    for r in rows[:max_checks]:
        for key in ("pronoun", "corr_pronoun"):
            p = r[key].strip()
            ids = tokenizer.encode(" " + p, add_special_tokens=False)
            if len(ids) != 1:
                bad += 1
    if bad > 0:
        print(f"[WARN] {bad} pronoun tokenizations (in first {max_checks} rows) were not single-token under GPT-2.")
        print("       This doesn't break KL training, but it breaks any 'single target token index' assumption.")


def should_stop(val_kl: float, metrics: Dict[str, float],
                target_val_kl: Optional[float],
                target_rel_sparsity: Optional[float],
                target_full_sparsity: Optional[float]) -> bool:
    # Stop when ALL provided targets are met.
    ok = True
    if target_val_kl is not None:
        ok = ok and (val_kl <= target_val_kl)
    if target_rel_sparsity is not None:
        ok = ok and (metrics["relative_sparsity_all"] >= target_rel_sparsity)
    if target_full_sparsity is not None:
        ok = ok and (metrics["full_sparsity_ov"] >= target_full_sparsity)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="gp")
    ap.add_argument("--data_dir", default="data_main")
    ap.add_argument("--train_csv", default="train_1k_gp.csv")
    ap.add_argument("--val_csv", default="val_gp.csv")
    ap.add_argument("--test_csv", default="test_gp.csv")

    ap.add_argument("--model_name", default="gpt2")  # will map to TL name
    ap.add_argument("--device", default="cuda")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--max_epochs", type=int, default=150)
    ap.add_argument("--lr", type=float, default=1e-2)
    ap.add_argument("--weight_decay", type=float, default=0.0)
    ap.add_argument("--l1_lambda", type=float, default=1e-3)

    ap.add_argument("--svd_eps", type=float, default=1e-6)
    ap.add_argument("--tau", type=float, default=1e-2)

    ap.add_argument("--target_val_kl", type=float, default=None)
    ap.add_argument("--target_rel_sparsity", type=float, default=None)
    ap.add_argument("--target_full_sparsity", type=float, default=None)

    ap.add_argument("--top_k_tokens", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=20)

    ap.add_argument("--out_dir", default="outputs/gp")
    ap.add_argument("--rebuild_svd", action="store_true", help="Force rebuild SVD cache even if svd_cache.pt exists.")
    args = ap.parse_args()

    set_seed(args.seed)

    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    svd_path = str(out_dir / "svd_cache.pt")
    masks_path = str(out_dir / "masks.pt")
    receptors_path = str(out_dir / "ov_logit_receptors.jsonl")
    summary_path = str(out_dir / "summary.json")

    # Load tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    # GPT-2 has no pad token by default; set to EOS for batch padding
    tokenizer.pad_token = tokenizer.eos_token

    # Load data
    train_rows = load_gp_csv(str(Path(args.data_dir) / args.train_csv))
    val_rows = load_gp_csv(str(Path(args.data_dir) / args.val_csv))
    test_rows = load_gp_csv(str(Path(args.data_dir) / args.test_csv))

    print(f"[DATA] train={len(train_rows)} val={len(val_rows)} test={len(test_rows)}")
    sanity_check_pronoun_tokenization(tokenizer, train_rows, max_checks=50)

    # Load model (TransformerLens name for GPT-2 small is "gpt2-small")
    tl_name = args.model_name
    if tl_name == "gpt2":
        tl_name = "gpt2-small"
    model = HookedTransformer.from_pretrained(tl_name, device=device)
    model.eval()

    # Freeze all model weights
    for p in model.parameters():
        p.requires_grad_(False)

    # SVD cache
    if args.rebuild_svd or (not os.path.exists(svd_path)):
        print("[SVD] building cache...")
        build_svd_cache(model, svd_path, svd_eps=args.svd_eps, device_for_svd="cpu")

    qk, ov, mlp_in, mlp_out, rank_total_ov = load_svd_cache(svd_path, device=device)

    # Trainable masks
    masks = MaskBank(qk=qk, ov=ov, mlp_in=mlp_in, mlp_out=mlp_out).to(device)
    opt = torch.optim.AdamW(masks.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_val = float("inf")
    bad_epochs = 0
    last_summary: Dict[str, Any] = {}

    for epoch in range(1, args.max_epochs + 1):
        masks.train()
        total_loss = 0.0
        total_kl = 0.0
        total_batches = 0

        for batch in batch_iter(train_rows, args.batch_size, shuffle=True):
            clean_prompts = [r["prefix"] for r in batch]
            corr_prompts = [r["corr_prefix"] for r in batch]

            # tokens_clean, last_idx_clean = tokenize_batch(tokenizer, clean_prompts, device)
            # tokens_corr, _ = tokenize_batch(tokenizer, corr_prompts, device)

            tokens_clean, last_idx_clean, tokens_corr, _ = tokenize_pair_batch(
                tokenizer, clean_prompts, corr_prompts, device
            )


            # original clean logits -> p (detach)
            with torch.no_grad():
                logits_clean_full = model(tokens_clean)  # (B,S,vocab)
                logits_clean = logits_clean_full[torch.arange(tokens_clean.size(0), device=device), last_idx_clean, :]
                p = F.softmax(logits_clean, dim=-1).detach()
                logp = F.log_softmax(logits_clean, dim=-1).detach()

            # corrupted projections for complement
            zc_ov, zc_mlp_in, zc_mlp_out = corrupt_projections(
                model=model,
                tokens=tokens_corr,
                qk=qk, ov=ov, mlp_in=mlp_in, mlp_out=mlp_out,
                dtype_cache=torch.float16
            )

            # masked clean logits -> q
            logits_masked = masked_forward_logits_last(
                model=model,
                tokens=tokens_clean,
                qk=qk, ov=ov, mlp_in=mlp_in, mlp_out=mlp_out,
                masks=masks,
                z_corr_ov=zc_ov,
                z_corr_mlp_in=zc_mlp_in,
                z_corr_mlp_out=zc_mlp_out,
                last_indices=last_idx_clean
            )

            logq = F.log_softmax(logits_masked, dim=-1)
            # KL(p || q) = sum p * (logp - logq)
            kl = (p * (logp - logq)).sum(dim=-1).mean()

            l1 = masks.l1_sum()
            loss = kl + args.l1_lambda * l1

            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

            total_loss += float(loss.detach().cpu().item())
            total_kl += float(kl.detach().cpu().item())
            total_batches += 1

        # Validation
        masks.eval()
        with torch.no_grad():
            val_kls = []
            for batch in batch_iter(val_rows, args.batch_size, shuffle=False):
                clean_prompts = [r["prefix"] for r in batch]
                corr_prompts = [r["corr_prefix"] for r in batch]
                # tokens_clean, last_idx_clean = tokenize_batch(tokenizer, clean_prompts, device)
                # tokens_corr, _ = tokenize_batch(tokenizer, corr_prompts, device)
                tokens_clean, last_idx_clean, tokens_corr, _ = tokenize_pair_batch(
                    tokenizer, clean_prompts, corr_prompts, device
                )


                logits_clean_full = model(tokens_clean)
                logits_clean = logits_clean_full[torch.arange(tokens_clean.size(0), device=device), last_idx_clean, :]
                p = F.softmax(logits_clean, dim=-1)
                logp = F.log_softmax(logits_clean, dim=-1)

                zc_ov, zc_mlp_in, zc_mlp_out = corrupt_projections(
                    model=model,
                    tokens=tokens_corr,
                    qk=qk, ov=ov, mlp_in=mlp_in, mlp_out=mlp_out,
                    dtype_cache=torch.float16
                )

                logits_masked = masked_forward_logits_last(
                    model=model,
                    tokens=tokens_clean,
                    qk=qk, ov=ov, mlp_in=mlp_in, mlp_out=mlp_out,
                    masks=masks,
                    z_corr_ov=zc_ov,
                    z_corr_mlp_in=zc_mlp_in,
                    z_corr_mlp_out=zc_mlp_out,
                    last_indices=last_idx_clean
                )
                logq = F.log_softmax(logits_masked, dim=-1)
                kl = (p * (logp - logq)).sum(dim=-1).mean()
                val_kls.append(float(kl.cpu().item()))

            val_kl = float(sum(val_kls) / max(1, len(val_kls)))

        spars = compute_sparsities(masks, tau=args.tau, rank_total_ov=rank_total_ov)
        train_loss = total_loss / max(1, total_batches)
        train_kl = total_kl / max(1, total_batches)

        last_summary = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_kl": train_kl,
            "val_kl": val_kl,
            "tau": args.tau,
            "svd_eps": args.svd_eps,
            "l1_lambda": args.l1_lambda,
            **spars
        }

        print(f"[EPOCH {epoch}] train_kl={train_kl:.6g} val_kl={val_kl:.6g} "
              f"rel_sparsity={spars['relative_sparsity_all']:.4f} full_sparsity_ov={spars['full_sparsity_ov']:.4f}")

        # Early stop check
        if should_stop(val_kl, spars, args.target_val_kl, args.target_rel_sparsity, args.target_full_sparsity):
            print("[STOP] targets reached -> stopping early.")
            break

        # Track best val_kl for patience
        if val_kl < best_val:
            best_val = val_kl
            bad_epochs = 0
            save_masks(masks_path, masks)
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(last_summary, f, indent=2)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print("[STOP] patience exhausted.")
                break

    # Always save final masks + summary
    save_masks(masks_path, masks)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(last_summary, f, indent=2)
    print(f"[SAVE] summary -> {summary_path}")

    # Dump receptors
    receptor_stats = dump_ov_logit_receptors(
        out_jsonl=receptors_path,
        model=model,
        tokenizer=tokenizer,
        ov=ov,
        masks=masks,
        tau=args.tau,
        top_k=args.top_k_tokens
    )

    # Add receptor stats to summary
    last_summary.update(receptor_stats)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(last_summary, f, indent=2)
    print("[DONE]")


if __name__ == "__main__":
    main()
