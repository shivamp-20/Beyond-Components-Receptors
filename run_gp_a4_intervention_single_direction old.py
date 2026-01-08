#!/usr/bin/env python3
"""
A4 / Algorithm 2 causal test for a *single* OV SVD direction on the GP task.

This script is designed to run *after* you have already run:
  train_gp_masks_and_dump_ov_logit_receptors.py

It expects:
  - {out_dir}/svd_cache.pt      (produced by build_svd_cache)
  - (optional) {out_dir}/masks.pt  (saved masks; used only for sanity-checking)
  - GP CSVs with columns like: prefix, pronoun, corr_prefix, corr_pronoun, ...

It implements the intervention you described:

  a_i(x) = < [context_resid(x), 1] , u_i >        where context_resid = pattern @ x_ln1
  ΔR     = (a_target - a_i(x)) * (sigma_scale * σ_i) * v_i
  resid_cf = resid_final_at_t* + ΔR
  logits_cf = unembed( ln_final(resid_cf) )

Then it reports baseline vs counterfactual logit-diffs and flip rates on the test split,
and saves a JSON + plots.

Notes / assumptions:
- "he" and "she" must be single-token when encoded as " he" and " she" for GPT-2.
- sv_idx is 0-based (k=0 is the first kept singular direction in svd_cache.pt).
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformer_lens import HookedTransformer
from transformers import GPT2TokenizerFast


# -------------------------
# Small utilities (mirrors the training script conventions)
# -------------------------

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def detect_delimiter(csv_path: Path) -> str:
    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
    try:
        return csv.Sniffer().sniff(sample).delimiter
    except Exception:
        return ","

def normalize_prefix(s: str) -> str:
    # training script uses: rstrip + single trailing space
    return str(s).rstrip() + " "

def load_gp_csv(csv_path: Path) -> List[Dict[str, str]]:
    delim = detect_delimiter(csv_path)
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter=delim)
        rows = []
        for r in reader:
            # normalize clean + corrupt prefixes so we consistently predict the pronoun as the *next token*
            if "prefix" in r: r["prefix"] = normalize_prefix(r["prefix"])
            if "corr_prefix" in r: r["corr_prefix"] = normalize_prefix(r["corr_prefix"])
            rows.append(r)
    return rows

def batch_iter(xs: List[str], batch_size: int) -> Iterable[List[str]]:
    for i in range(0, len(xs), batch_size):
        yield xs[i:i+batch_size]

def make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # True where allowed (k <= q)
    return torch.tril(torch.ones((seq_len, seq_len), device=device, dtype=torch.bool))

def gelu_new(x: torch.Tensor) -> torch.Tensor:
    # GPT-2 GELU approximation (matches TransformerLens & HF)
    return 0.5 * x * (1.0 + torch.tanh(np.sqrt(2.0 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))

def attention_pattern_original(
    x_ln1: torch.Tensor,            # [B, S, D]
    W_Q: torch.Tensor, b_Q: torch.Tensor,  # [D, d_head], [d_head]
    W_K: torch.Tensor, b_K: torch.Tensor,  # [D, d_head], [d_head]
    causal: torch.Tensor,           # [S, S] bool
    d_head: int,
) -> torch.Tensor:
    # Computes attention pattern for ONE head: softmax( (QK^T)/sqrt(d_head) + causal_mask )
    q = x_ln1 @ W_Q + b_Q           # [B, S, d_head]
    k = x_ln1 @ W_K + b_K           # [B, S, d_head]
    scores = torch.einsum("bqd,bkd->bqk", q, k) / (d_head ** 0.5)  # [B, S, S]
    # mask out future positions
    scores = scores.masked_fill(~causal[None, :, :], float("-inf"))
    return F.softmax(scores, dim=-1)  # [B, S, S]


def tokenize_prefixes(
    tok: GPT2TokenizerFast,
    prefixes: List[str],
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      tokens: [B, S_max] padded on the right with pad_token_id (=eos)
      last_idx: [B] index of last non-pad token (t*)
    """
    enc = tok(prefixes, return_tensors="pt", padding=True, truncation=False, add_special_tokens=False)
    tokens = enc["input_ids"].to(device)
    pad_id = tok.pad_token_id
    lengths = (tokens != pad_id).sum(dim=1)
    last_idx = lengths - 1
    return tokens, last_idx


# -------------------------
# Core: forward + extract a_i, resid_final_t*, logits_t*
# -------------------------

@torch.no_grad()
def forward_original_extract(
    model: HookedTransformer,
    tokens: torch.Tensor,       # [B, S]
    last_idx: torch.Tensor,     # [B]
    layer: int,
    head: int,
    u_vec: torch.Tensor,        # [D+1] (column of U for sv_idx)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Runs a manual forward matching the training script's OV input convention:
      context_resid = pattern @ x_ln1
      ctx_aug = [context_resid, 1]
      a_i(x) = <ctx_aug, u_i>

    Returns:
      ai:        [B]
      resid_t:   [B, D]   final residual stream at position t* (pre ln_final)
      logits_t:  [B, V]   next-token logits at position t*
    """
    device = tokens.device
    cfg = model.cfg
    B, S = tokens.shape
    causal = make_causal_mask(S, device=device)

    # embeddings (TransformerLens)
    x = model.embed(tokens) + model.pos_embed(tokens)  # [B,S,D]

    ai: Optional[torch.Tensor] = None

    for l in range(cfg.n_layers):
        block = model.blocks[l]
        attn = block.attn
        mlp  = block.mlp

        x_ln1 = block.ln1(x)  # [B,S,D]

        # Attention: sum over heads
        head_outs = []
        for h in range(cfg.n_heads):
            pat = attention_pattern_original(
                x_ln1,
                attn.W_Q[h], attn.b_Q[h],
                attn.W_K[h], attn.b_K[h],
                causal=causal,
                d_head=cfg.d_head,
            )  # [B,S,S]

            context_resid = pat @ x_ln1  # [B,S,D]  (matches training script definition)

            # Extract a_i(x) at the requested (layer, head) and t* per example
            if l == layer and h == head:
                idx = torch.arange(B, device=device)
                c_t = context_resid[idx, last_idx, :]  # [B,D]
                ones = torch.ones((B, 1), device=device, dtype=c_t.dtype)
                ctx_aug = torch.cat([c_t, ones], dim=-1)  # [B,D+1]
                ai = (ctx_aug * u_vec[None, :]).sum(dim=-1)  # [B]

            # Standard V/O path (per-head)
            v = context_resid @ attn.W_V[h] + attn.b_V[h]      # [B,S,d_head]
            out = v @ attn.W_O[h]                              # [B,S,D]
            head_outs.append(out)

        attn_out = torch.stack(head_outs, dim=0).sum(dim=0) + attn.b_O  # [B,S,D]
        x = x + attn_out

        # MLP
        x_ln2 = block.ln2(x)
        pre = x_ln2 @ mlp.W_in + mlp.b_in                       # [B,S,d_mlp]
        act = gelu_new(pre)
        mlp_out = act @ mlp.W_out + mlp.b_out                   # [B,S,D]
        x = x + mlp_out

    assert ai is not None, "Internal error: did not compute ai (check layer/head indices)."

    idx = torch.arange(B, device=device)
    resid_t = x[idx, last_idx, :]  # [B,D] pre ln_final

    logits_t = (model.ln_final(resid_t) @ model.W_U) + model.b_U  # [B,V]
    return ai, resid_t, logits_t


# -------------------------
# Statistics / evaluation
# -------------------------

@dataclass
class EvalStats:
    n: int = 0
    base_diff_sum: float = 0.0
    base_diff_sq_sum: float = 0.0
    cf_diff_sum: float = 0.0
    cf_diff_sq_sum: float = 0.0
    flips: int = 0
    denom: int = 0

    def update_diffs(self, base: torch.Tensor, cf: torch.Tensor):
        # base, cf are [B] float tensors on CPU/GPU
        b = base.detach().float().cpu().numpy()
        c = cf.detach().float().cpu().numpy()
        self.n += int(b.shape[0])
        self.base_diff_sum += float(b.sum())
        self.base_diff_sq_sum += float((b ** 2).sum())
        self.cf_diff_sum += float(c.sum())
        self.cf_diff_sq_sum += float((c ** 2).sum())

    def update_flips(self, flips: int, denom: int):
        self.flips += int(flips)
        self.denom += int(denom)

    def summary(self) -> Dict[str, float]:
        def mean_std(sum_, sq_sum_, n_):
            if n_ <= 0: return (float("nan"), float("nan"))
            mu = sum_ / n_
            var = max(0.0, (sq_sum_ / n_) - mu * mu)
            return (mu, var ** 0.5)

        base_mean, base_std = mean_std(self.base_diff_sum, self.base_diff_sq_sum, self.n)
        cf_mean, cf_std = mean_std(self.cf_diff_sum, self.cf_diff_sq_sum, self.n)
        flip_rate = self.flips / max(1, self.denom)
        return {
            "n": self.n,
            "base_logit_diff_mean": base_mean,
            "base_logit_diff_std": base_std,
            "cf_logit_diff_mean": cf_mean,
            "cf_logit_diff_std": cf_std,
            "flips": self.flips,
            "flip_denom": self.denom,
            "flip_rate": flip_rate,
        }


@torch.no_grad()
def estimate_mu(
    model: HookedTransformer,
    tok: GPT2TokenizerFast,
    prefixes: List[str],
    layer: int,
    head: int,
    u_vec: torch.Tensor,
    device: torch.device,
    batch_size: int,
    max_n: Optional[int] = None,
) -> float:
    if max_n is not None:
        prefixes = prefixes[:max_n]

    total = 0.0
    n = 0
    for batch in batch_iter(prefixes, batch_size):
        tokens, last_idx = tokenize_prefixes(tok, batch, device)
        ai, _, _ = forward_original_extract(model, tokens, last_idx, layer, head, u_vec)
        total += float(ai.detach().sum().cpu())
        n += int(ai.numel())
    return total / max(1, n)


@torch.no_grad()
def eval_split(
    model: HookedTransformer,
    tok: GPT2TokenizerFast,
    prefixes: List[str],
    true_class: str,         # "he" or "she" (label of this subset)
    mu_he: float,
    mu_she: float,
    he_id: int,
    she_id: int,
    layer: int,
    head: int,
    u_vec: torch.Tensor,
    sigma: float,
    v_vec: torch.Tensor,     # [D] (row of Vh for sv_idx)
    sigma_scales: List[float],
    device: torch.device,
    batch_size: int,
    max_n: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    if max_n is not None:
        prefixes = prefixes[:max_n]

    results: Dict[str, Dict[str, float]] = {}
    true_class = true_class.lower().strip()
    assert true_class in ("he", "she")

    for scale in sigma_scales:
        stats = EvalStats()

        for batch in batch_iter(prefixes, batch_size):
            tokens, last_idx = tokenize_prefixes(tok, batch, device)
            ai, resid_t, logits_base = forward_original_extract(model, tokens, last_idx, layer, head, u_vec)

            base_pred = torch.argmax(logits_base, dim=-1)

            if true_class == "he":
                base_diff = logits_base[:, he_id] - logits_base[:, she_id]
                a_target = mu_she
                denom = int((base_pred == he_id).sum().item())
            else:
                base_diff = logits_base[:, she_id] - logits_base[:, he_id]
                a_target = mu_he
                denom = int((base_pred == she_id).sum().item())

            # ΔR = (a_target - ai) * (scale * sigma) * v
            delta = ((a_target - ai) * (scale * sigma)).unsqueeze(-1) * v_vec.unsqueeze(0)  # [B,D]
            resid_cf = resid_t + delta
            logits_cf = (model.ln_final(resid_cf) @ model.W_U) + model.b_U

            cf_pred = torch.argmax(logits_cf, dim=-1)

            if true_class == "he":
                cf_diff = logits_cf[:, he_id] - logits_cf[:, she_id]
                flips = int(((base_pred == he_id) & (cf_pred == she_id)).sum().item())
            else:
                cf_diff = logits_cf[:, she_id] - logits_cf[:, he_id]
                flips = int(((base_pred == she_id) & (cf_pred == he_id)).sum().item())

            stats.update_diffs(base_diff, cf_diff)
            stats.update_flips(flips=flips, denom=denom)

        results[str(scale)] = stats.summary()

    return results


def plot_curves(out_dir: Path, tag: str, sigma_scales: List[float], metrics: Dict[str, Dict[str, float]]) -> None:
    xs = sigma_scales
    base_means = [metrics[str(s)]["base_logit_diff_mean"] for s in xs]
    cf_means   = [metrics[str(s)]["cf_logit_diff_mean"] for s in xs]
    flip_rates = [metrics[str(s)]["flip_rate"] for s in xs]

    ensure_dir(out_dir)

    plt.figure()
    plt.plot(xs, base_means, marker="o", label="baseline logit-diff mean")
    plt.plot(xs, cf_means, marker="o", label="counterfactual logit-diff mean")
    plt.xlabel("sigma_scale")
    plt.ylabel("logit diff")
    plt.title(tag)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_logit_diff.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(xs, flip_rates, marker="o")
    plt.xlabel("sigma_scale")
    plt.ylabel("flip rate")
    plt.title(tag)
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_flip_rate.png", dpi=200)
    plt.close()


def parse_sigma_scales(s: str) -> List[float]:
    # accepts "0,0.5,1,2" or "0 0.5 1 2"
    s = s.strip()
    if not s:
        return [0.0, 0.5, 1.0, 2.0, 4.0, 8.0]
    parts = [p for p in s.replace(",", " ").split(" ") if p]
    return [float(p) for p in parts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_main")
    ap.add_argument("--train_csv", default="train_1k_gp.csv")
    ap.add_argument("--test_csv", default="test_gp.csv")
    ap.add_argument("--out_dir", default="outputs/gp")

    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--head", type=int, required=True)
    ap.add_argument("--sv_idx", type=int, required=True)

    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch_size", type=int, default=64)

    ap.add_argument("--sigma_scales", type=str, default="0,0.5,1,2,4,8")
    ap.add_argument("--tau", type=float, default=1e-2, help="Only for mask sanity-check print.")
    ap.add_argument("--max_mu", type=int, default=None, help="Optional cap on #train examples used to estimate μ.")
    ap.add_argument("--max_test", type=int, default=None, help="Optional cap on #test examples used in eval.")
    ap.add_argument("--save_json", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)

    svd_path = out_dir / "svd_cache.pt"
    if not svd_path.exists():
        raise FileNotFoundError(f"Missing {svd_path}. Run your training script first (stage B).")

    # Load data
    train_rows = load_gp_csv(data_dir / args.train_csv)
    test_rows  = load_gp_csv(data_dir / args.test_csv)

    # Group by pronoun label
    def by_label(rows, label: str) -> List[Dict[str,str]]:
        label = label.lower().strip()
        return [r for r in rows if str(r.get("pronoun","")).strip().lower() == label]

    train_he  = by_label(train_rows, "he")
    train_she = by_label(train_rows, "she")
    test_he   = by_label(test_rows,  "he")
    test_she  = by_label(test_rows,  "she")

    if len(train_he) == 0 or len(train_she) == 0:
        raise ValueError("Could not find both 'he' and 'she' in train pronoun column.")

    # Load tokenizer + model
    device = torch.device(args.device)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = HookedTransformer.from_pretrained("gpt2-small", device=device).eval()

    # Pronoun token sanity
    he_ids  = tok.encode("he", add_special_tokens=False)
    she_ids = tok.encode("she", add_special_tokens=False)
    if len(he_ids) != 1 or len(she_ids) != 1:
        raise ValueError(f"' he' or ' she' is not a single token: he={he_ids}, she={she_ids}")
    he_id, she_id = int(he_ids[0]), int(she_ids[0])

    # Load SVD direction
    svd_cache = torch.load(svd_path, map_location="cpu")
    try:
        svd_obj = svd_cache["ov"][args.layer][args.head]
        U = svd_obj["U"]   # [D+1, r]
        S = svd_obj["S"]   # [r]
        Vh = svd_obj["Vh"] # [r, D]
    except Exception as e:
        raise KeyError(f"Could not access ov[{args.layer}][{args.head}] in {svd_path}: {e}")

    r = int(S.numel())
    if not (0 <= args.sv_idx < r):
        raise ValueError(f"sv_idx out of range: got {args.sv_idx}, but r={r} for this head.")

    u_vec = U[:, args.sv_idx].to(device)
    sigma = float(S[args.sv_idx].item())
    v_vec = Vh[args.sv_idx, :].to(device)

    print(f"[LOAD] out_dir={out_dir.resolve()}")
    print(f"[LOAD] svd_cache.pt OK. (layer={args.layer}, head={args.head}) r={r}, sv_idx={args.sv_idx}, sigma={sigma:.6g}")

    # Optional: mask sanity-check
    masks_path = out_dir / "masks.pt"
    if masks_path.exists():
        masks = torch.load(masks_path, map_location="cpu")
        try:
            m_val = float(masks["ov"][args.layer][args.head][args.sv_idx].item())
            print(f"[SANITY] mask m={m_val:.6g}  (tau={args.tau:g})  -> active={m_val > args.tau}")
        except Exception as e:
            print(f"[SANITY] masks.pt exists but could not read masks['ov'][layer][head][sv_idx]: {e}")
    else:
        print("[SANITY] masks.pt not found (OK). Skipping mask value print.")

    # Estimate class means μ
    train_prefixes_he  = [r["prefix"] for r in train_he]
    train_prefixes_she = [r["prefix"] for r in train_she]

    print(f"[MU] estimating on train: he={len(train_prefixes_he)} she={len(train_prefixes_she)} (max_mu={args.max_mu})")
    mu_he = estimate_mu(model, tok, train_prefixes_he,  args.layer, args.head, u_vec, device, args.batch_size, max_n=args.max_mu)
    mu_she = estimate_mu(model, tok, train_prefixes_she, args.layer, args.head, u_vec, device, args.batch_size, max_n=args.max_mu)
    print(f"[MU] mu_he={mu_he:.6g}  mu_she={mu_she:.6g}")

    sigma_scales = parse_sigma_scales(args.sigma_scales)
    print(f"[EVAL] sigma_scales={sigma_scales}")

    # Eval
    test_prefixes_he  = [r["prefix"] for r in test_he]
    test_prefixes_she = [r["prefix"] for r in test_she]

    print(f"[EVAL] test sizes: he={len(test_prefixes_he)} she={len(test_prefixes_she)} (max_test={args.max_test})")

    metrics_he = eval_split(
        model, tok, test_prefixes_he, "he",
        mu_he=mu_he, mu_she=mu_she,
        he_id=he_id, she_id=she_id,
        layer=args.layer, head=args.head,
        u_vec=u_vec, sigma=sigma, v_vec=v_vec,
        sigma_scales=sigma_scales, device=device,
        batch_size=args.batch_size, max_n=args.max_test,
    )

    metrics_she = eval_split(
        model, tok, test_prefixes_she, "she",
        mu_he=mu_he, mu_she=mu_she,
        he_id=he_id, she_id=she_id,
        layer=args.layer, head=args.head,
        u_vec=u_vec, sigma=sigma, v_vec=v_vec,
        sigma_scales=sigma_scales, device=device,
        batch_size=args.batch_size, max_n=args.max_test,
    )

    # Pretty print
    def print_table(tag: str, metrics: Dict[str, Dict[str, float]]):
        print(f"\n== {tag} ==")
        for s in sigma_scales:
            m = metrics[str(s)]
            print(
                f"scale={s:<6}  "
                f"base_diff={m['base_logit_diff_mean']:+.4f}±{m['base_logit_diff_std']:.4f}  "
                f"cf_diff={m['cf_logit_diff_mean']:+.4f}±{m['cf_logit_diff_std']:.4f}  "
                f"flip_rate={100*m['flip_rate']:.2f}%  (n={m['n']}, denom={m['flip_denom']})"
            )

    print_table("TEST he→(target she mean)", metrics_he)
    print_table("TEST she→(target he mean)", metrics_she)

    # Save JSON + plots
    results = {
        "layer": args.layer,
        "head": args.head,
        "sv_idx": args.sv_idx,
        "sigma": sigma,
        "mu_he": float(mu_he),
        "mu_she": float(mu_she),
        "sigma_scales": sigma_scales,
        "metrics_he": metrics_he,
        "metrics_she": metrics_she,
        "note": "logit_diff is (he-she) on he-split and (she-he) on she-split; flips counted among baseline argmax==true pronoun token.",
    }

    if args.save_json:
        out_json = out_dir / f"a4_intervention_l{args.layer}_h{args.head}_k{args.sv_idx}.json"
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n[SAVE] {out_json}")

    plot_dir = out_dir / "a4_plots"
    ensure_dir(plot_dir)
    plot_curves(plot_dir, f"he_l{args.layer}_h{args.head}_k{args.sv_idx}", sigma_scales, metrics_he)
    plot_curves(plot_dir, f"she_l{args.layer}_h{args.head}_k{args.sv_idx}", sigma_scales, metrics_she)
    print(f"[SAVE] plots -> {plot_dir}")

    print("[DONE]")


if __name__ == "__main__":
    main()
