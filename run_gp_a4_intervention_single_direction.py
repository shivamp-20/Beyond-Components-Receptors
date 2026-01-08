#!/usr/bin/env python3
"""
A4 / Algorithm 2 causal test for a *single* OV SVD direction on the GP task.

This is a DEBUG+SANITY version of run_gp_a4_intervention_single_direction.py.

Key fixes / additions:
- Robust CSV delimiter sniff (supports '|' like the GP dataset).
- Robust pronoun-token ID inference for *this dataset formatting*:
    We infer the next-token ID for "he"/"she" by comparing tokenization of:
      prefix
    vs
      prefix + pronoun
  This avoids the common GPT-2 gotcha where the next token might be "he" (no leading space)
  rather than " he" (leading-space token), depending on whether the trailing space is an
  explicit token in the prefix encoding.
- Prints lots of tokenization + prediction sanity diagnostics so you can paste logs back.

It expects (same as the original A4 script):
  - {out_dir}/svd_cache.pt
  - (optional) {out_dir}/masks.pt
  - GP CSVs with columns: prefix, pronoun, corr_prefix, corr_pronoun, ...

Intervention implemented:
  a_i(x) = < [context_resid(x), 1] , u_i >        where context_resid = pattern @ x_ln1
  ΔR     = (a_target - a_i(x)) * (sigma_scale * σ_i) * v_i
  resid_cf = resid_final_at_t* + ΔR
  logits_cf = unembed( ln_final(resid_cf) )

We report:
- baseline vs counterfactual he-vs-she logit-diff means
- flip rates under two definitions:
    (A) top-1 flip (original behavior): baseline argmax is true pronoun AND cf argmax is opposite pronoun
    (B) pairwise flip (recommended): baseline prefers true pronoun within {he,she} AND cf prefers opposite
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Iterable, Optional, Counter as CounterT
from collections import Counter

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
    # Mirror the training script: GP CSVs often use '|'
    with open(csv_path, "r", encoding="utf-8") as f:
        sample = f.read(4096)
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=[",", "|", "\t", ";"])
        return dialect.delimiter
    except Exception:
        return "|"

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
# Tokenization diagnostics
# -------------------------

def next_token_id_from_prefix(
    tok: GPT2TokenizerFast,
    prefix: str,
    word: str,
) -> int:
    """Return the token-id the model would emit for `word` as the *next token* after `prefix`.

    We compute it by encoding `prefix + word` and taking the last token id. This is robust to GPT-2
    byte-level BPE quirks where appending characters can change the tokenization of the prefix tail
    (e.g., a trailing space token can merge into the following word token like " he").
    """
    ids = tok.encode(prefix + word, add_special_tokens=False)
    if len(ids) == 0:
        raise ValueError("Got empty tokenization for prefix+word; unexpected.")
    return int(ids[-1])


def collect_next_token_id_stats(
    tok: GPT2TokenizerFast,
    prefixes: List[str],
    word: str,
    sample_n: int = 400,
) -> Dict[str, Any]:
    """Collect distribution of next-token IDs for `word` across a sample of prefixes."""
    word = word.strip()
    prefixes = prefixes[:sample_n]
    counts: Counter[int] = Counter()
    errors = 0
    decoded: Dict[int, str] = {}
    for p in prefixes:
        try:
            tid = next_token_id_from_prefix(tok, p, word)
            counts[tid] += 1
            if tid not in decoded:
                decoded[tid] = tok.decode([tid], clean_up_tokenization_spaces=False)
        except Exception:
            errors += 1

    most_common = counts.most_common(5)
    return {
        "sampled": len(prefixes),
        "errors": int(errors),
        "num_unique_ids": int(len(counts)),
        "top_ids": [{"id": int(tid), "count": int(c), "decoded": decoded.get(tid, "")} for tid, c in most_common],
        "all_counts": {int(k): int(v) for k, v in counts.items()},
    }

def print_prefix_debug(
    model: HookedTransformer,
    tok: GPT2TokenizerFast,
    prefixes: List[str],
    device: torch.device,
    n: int = 5,
) -> None:
    """
    For a few prefixes, print:
      - raw prefix tail
      - token IDs tail
      - decoded tail tokens
      - the per-prefix inferred next-token IDs for he/she (via encode(prefix+word)[-1])
      - baseline top tokens at t*
      - ranks / logits for those per-prefix he/she token IDs
    """
    n = min(n, len(prefixes))
    if n <= 0:
        return

    # Per-prefix token-ids for he/she as NEXT tokens
    he_ids = [next_token_id_from_prefix(tok, p, "he") for p in prefixes[:n]]
    she_ids = [next_token_id_from_prefix(tok, p, "she") for p in prefixes[:n]]

    tokens, last_idx = tokenize_prefixes(tok, prefixes[:n], device)
    B, S = tokens.shape
    logits = model(tokens)  # [B,S,V]
    idx = torch.arange(B, device=device)
    t = last_idx
    lt = logits[idx, t, :]  # [B,V]

    topv, topi = torch.topk(lt, k=10, dim=-1)

    for i in range(n):
        p = prefixes[i]
        tail = p[-120:]
        print("\n---")
        print("prefix tail:", repr(tail))
        ids = tokens[i].tolist()
        # show last 20 non-pad tokens
        nonpad = [x for x in ids if x != tok.pad_token_id]
        tail_ids = nonpad[-20:]
        tail_dec = [tok.decode([x], clean_up_tokenization_spaces=False) for x in tail_ids]
        print("token tail ids:", tail_ids)
        print("token tail dec:", tail_dec)

        hid = he_ids[i]
        sid = she_ids[i]
        print(f"inferred next-token ids: he={hid} ({tok.decode([hid], clean_up_tokenization_spaces=False)!r}), "
              f"she={sid} ({tok.decode([sid], clean_up_tokenization_spaces=False)!r})")

        hv = float(lt[i, hid].item())
        sv = float(lt[i, sid].item())
        rank_he = int((lt[i] > lt[i, hid]).sum().item() + 1)
        rank_she = int((lt[i] > lt[i, sid]).sum().item() + 1)
        print(f"logit(he)={hv:.4f} rank={rank_he} | logit(she)={sv:.4f} rank={rank_she}")

        tops = [(int(topi[i, j]), float(topv[i, j]), tok.decode([int(topi[i, j])], clean_up_tokenization_spaces=False))
                for j in range(10)]
        print("top10 @t*:", tops)

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

            if l == layer and h == head:
                idx = torch.arange(B, device=device)
                c_t = context_resid[idx, last_idx, :]  # [B,D]
                ones = torch.ones((B, 1), device=device, dtype=c_t.dtype)
                ctx_aug = torch.cat([c_t, ones], dim=-1)  # [B,D+1]
                ai = (ctx_aug * u_vec[None, :]).sum(dim=-1)  # [B]

            v = context_resid @ attn.W_V[h] + attn.b_V[h      ]  # [B,S,d_head]
            out = v @ attn.W_O[h]                                # [B,S,D]
            head_outs.append(out)

        attn_out = torch.stack(head_outs, dim=0).sum(dim=0) + attn.b_O  # [B,S,D]
        x = x + attn_out

        x_ln2 = block.ln2(x)
        pre = x_ln2 @ mlp.W_in + mlp.b_in
        act = gelu_new(pre)
        mlp_out = act @ mlp.W_out + mlp.b_out
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

    flips_top1: int = 0
    denom_top1: int = 0

    flips_pair: int = 0
    denom_pair: int = 0

    ai_sum: float = 0.0
    ai_sq_sum: float = 0.0
    delta_l2_sum: float = 0.0
    delta_l2_sq_sum: float = 0.0

    def update_diffs(self, base: torch.Tensor, cf: torch.Tensor):
        b = base.detach().float().cpu().numpy()
        c = cf.detach().float().cpu().numpy()
        self.n += int(b.shape[0])
        self.base_diff_sum += float(b.sum())
        self.base_diff_sq_sum += float((b ** 2).sum())
        self.cf_diff_sum += float(c.sum())
        self.cf_diff_sq_sum += float((c ** 2).sum())

    def update_flips(self, flips_top1: int, denom_top1: int, flips_pair: int, denom_pair: int):
        self.flips_top1 += int(flips_top1)
        self.denom_top1 += int(denom_top1)
        self.flips_pair += int(flips_pair)
        self.denom_pair += int(denom_pair)

    def update_ai_delta(self, ai: torch.Tensor, delta: torch.Tensor):
        # ai: [B], delta: [B,D]
        a = ai.detach().float().cpu().numpy()
        dl2 = delta.detach().float().norm(dim=-1).cpu().numpy()
        self.ai_sum += float(a.sum())
        self.ai_sq_sum += float((a ** 2).sum())
        self.delta_l2_sum += float(dl2.sum())
        self.delta_l2_sq_sum += float((dl2 ** 2).sum())

    def summary(self) -> Dict[str, float]:
        def mean_std(sum_, sq_sum_, n_):
            if n_ <= 0: return (float("nan"), float("nan"))
            mu = sum_ / n_
            var = max(0.0, (sq_sum_ / n_) - mu * mu)
            return (mu, var ** 0.5)

        base_mean, base_std = mean_std(self.base_diff_sum, self.base_diff_sq_sum, self.n)
        cf_mean, cf_std = mean_std(self.cf_diff_sum, self.cf_diff_sq_sum, self.n)
        ai_mean, ai_std = mean_std(self.ai_sum, self.ai_sq_sum, self.n)
        dl2_mean, dl2_std = mean_std(self.delta_l2_sum, self.delta_l2_sq_sum, self.n)

        flip_rate_top1 = self.flips_top1 / max(1, self.denom_top1)
        flip_rate_pair = self.flips_pair / max(1, self.denom_pair)

        return {
            "n": self.n,
            "base_logit_diff_mean": base_mean,
            "base_logit_diff_std": base_std,
            "cf_logit_diff_mean": cf_mean,
            "cf_logit_diff_std": cf_std,

            "flips_top1": self.flips_top1,
            "flip_denom_top1": self.denom_top1,
            "flip_rate_top1": flip_rate_top1,

            "flips_pair": self.flips_pair,
            "flip_denom_pair": self.denom_pair,
            "flip_rate_pair": flip_rate_pair,

            "ai_mean": ai_mean,
            "ai_std": ai_std,
            "delta_l2_mean": dl2_mean,
            "delta_l2_std": dl2_std,
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
) -> Tuple[float, float]:
    if max_n is not None:
        prefixes = prefixes[:max_n]

    total = 0.0
    total_sq = 0.0
    n = 0
    for batch in batch_iter(prefixes, batch_size):
        tokens, last_idx = tokenize_prefixes(tok, batch, device)
        ai, _, _ = forward_original_extract(model, tokens, last_idx, layer, head, u_vec)
        a = ai.detach().float().cpu().numpy()
        total += float(a.sum())
        total_sq += float((a**2).sum())
        n += int(ai.numel())

    if n == 0:
        return float("nan"), float("nan")
    mu = total / n
    var = max(0.0, (total_sq / n) - mu*mu)
    return float(mu), float(var ** 0.5)


@torch.no_grad()
def eval_split(
    model: HookedTransformer,
    tok: GPT2TokenizerFast,
    prefixes: List[str],
    true_class: str,         # "he" or "she"
    mu_he: float,
    mu_she: float,
    layer: int,
    head: int,
    u_vec: torch.Tensor,
    sigma: float,
    v_vec: torch.Tensor,     # [D]
    sigma_scales: List[float],
    device: torch.device,
    batch_size: int = 64,
    max_test: Optional[int] = None,
) -> Dict[str, Dict[str, float]]:
    # NOTE: We compute pronoun token IDs PER-EXAMPLE as:
    #   he_id_ex  = encode(prefix + "he")[-1]
    #   she_id_ex = encode(prefix + "she")[-1]
    # This avoids whitespace/BPE quirks (" he" vs "he") that can make a single global ID wrong.

    if max_test is not None:
        prefixes = prefixes[:max_test]

    results: Dict[str, Dict[str, float]] = {}
    mu_tgt = mu_she if true_class == "he" else mu_he

    for scale in sigma_scales:
        stats = RunningStats()

        for batch in batch_iter(prefixes, batch_size):
            tokens, last_idx = tokenize_prefixes(tok, batch, device)

            # Per-example next-token IDs for the two pronouns
            he_ids = torch.tensor([next_token_id_from_prefix(tok, p, "he") for p in batch], device=device, dtype=torch.long)
            she_ids = torch.tensor([next_token_id_from_prefix(tok, p, "she") for p in batch], device=device, dtype=torch.long)

            ai, resid_t, logits_t = forward_original_extract(model, tokens, last_idx, layer, head, u_vec)

            he_logits = logits_t.gather(1, he_ids.unsqueeze(1)).squeeze(1)
            she_logits = logits_t.gather(1, she_ids.unsqueeze(1)).squeeze(1)

            if true_class == "he":
                base_diff = he_logits - she_logits
            else:
                base_diff = she_logits - he_logits

            # Mean-swap: set a_cf to the opposite-class mean, then scale delta by (scale*sigma)
            a_cf = torch.full_like(ai, float(mu_tgt))
            delta = (a_cf - ai)[:, None] * (float(scale) * float(sigma)) * v_vec[None, :].to(device)

            resid_cf = resid_t + delta.to(resid_t.dtype)
            x_final = model.ln_final(resid_cf)
            logits_cf = x_final @ model.W_U + model.b_U  # [B,V]

            he_logits_cf = logits_cf.gather(1, he_ids.unsqueeze(1)).squeeze(1)
            she_logits_cf = logits_cf.gather(1, she_ids.unsqueeze(1)).squeeze(1)

            if true_class == "he":
                cf_diff = he_logits_cf - she_logits_cf
                true_ids = he_ids
                other_ids = she_ids
            else:
                cf_diff = she_logits_cf - he_logits_cf
                true_ids = she_ids
                other_ids = he_ids

            base_pred = logits_t.argmax(dim=-1)
            cf_pred = logits_cf.argmax(dim=-1)

            denom_top1 = int((base_pred == true_ids).sum().item())
            flips_top1 = int(((base_pred == true_ids) & (cf_pred == other_ids)).sum().item())

            denom_pair = int((base_diff > 0).sum().item())
            flips_pair = int(((base_diff > 0) & (cf_diff < 0)).sum().item())

            stats.update_diffs(base_diff, cf_diff)
            stats.update_flips(flips_top1=flips_top1, denom_top1=denom_top1,
                               flips_pair=flips_pair, denom_pair=denom_pair)
            stats.update_ai_delta(ai=ai, delta=delta)

        results[str(scale)] = stats.summary()

    return results

def plot_curves(out_dir: Path, tag: str, sigma_scales: List[float], metrics: Dict[str, Dict[str, float]]) -> None:
    xs = sigma_scales
    base_means = [metrics[str(s)]["base_logit_diff_mean"] for s in xs]
    cf_means   = [metrics[str(s)]["cf_logit_diff_mean"] for s in xs]
    flip_top1  = [metrics[str(s)]["flip_rate_top1"] for s in xs]
    flip_pair  = [metrics[str(s)]["flip_rate_pair"] for s in xs]

    ensure_dir(out_dir)

    plt.figure()
    plt.plot(xs, base_means, marker="o", label="baseline diff mean (true-other)")
    plt.plot(xs, cf_means, marker="o", label="counterfactual diff mean (true-other)")
    plt.xlabel("sigma_scale")
    plt.ylabel("logit diff")
    plt.title(tag)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_logit_diff.png", dpi=200)
    plt.close()

    plt.figure()
    plt.plot(xs, flip_top1, marker="o", label="flip_rate_top1 (argmax)")
    plt.plot(xs, flip_pair, marker="o", label="flip_rate_pair (sign flip)")
    plt.xlabel("sigma_scale")
    plt.ylabel("flip rate")
    plt.title(tag)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{tag}_flip_rate.png", dpi=200)
    plt.close()


def parse_sigma_scales(s: str) -> List[float]:
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

    ap.add_argument("--sigma_scales", type=str, default="0,0.5,1,2,4,8,10,15")
    ap.add_argument("--tau", type=float, default=1e-2, help="Only for mask sanity-check print.")
    ap.add_argument("--max_mu", type=int, default=None, help="Optional cap on #train examples used to estimate μ.")
    ap.add_argument("--max_test", type=int, default=None, help="Optional cap on #test examples used in eval.")
    ap.add_argument("--debug_n", type=int, default=5, help="How many example prefixes to print tokenization diagnostics for.")
    ap.add_argument("--debug_token_infer_n", type=int, default=200, help="How many prefixes to sample when inferring pronoun next-token IDs.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)

    svd_path = out_dir / "svd_cache.pt"
    if not svd_path.exists():
        raise FileNotFoundError(f"Missing {svd_path}. Run your training script first.")

    # Load data
    train_rows = load_gp_csv(data_dir / args.train_csv)
    test_rows  = load_gp_csv(data_dir / args.test_csv)

    # Group by pronoun label (CSV has 'he'/'she' without leading space — that's fine)
    def by_label(rows, label: str) -> List[Dict[str,str]]:
        label = label.lower().strip()
        return [r for r in rows if str(r.get("pronoun","")).strip().lower() == label]

    train_he  = by_label(train_rows, "he")
    train_she = by_label(train_rows, "she")
    test_he   = by_label(test_rows,  "he")
    test_she  = by_label(test_rows,  "she")

    if len(train_he) == 0 or len(train_she) == 0:
        raise ValueError("Could not find both 'he' and 'she' in train pronoun column.")

    # Sanity: prefixes should end with exactly one space after normalize_prefix
    bad_space = sum(1 for r in train_rows[:1000] if not str(r["prefix"]).endswith(" "))
    print(f"[DATA] train={len(train_rows)} (he={len(train_he)} she={len(train_she)}) test={len(test_rows)} (he={len(test_he)} she={len(test_she)})")
    print(f"[SANITY] prefix endswith space? bad_in_first_1000={bad_space}")

    # Load tokenizer + model
    device = torch.device(args.device)
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    model = HookedTransformer.from_pretrained("gpt2-small", device=device).eval()

    # Tokenization sanity for *next token* IDs of he/she under THIS prefix formatting.
    # We compute per-prefix next-token IDs as encode(prefix + "he")[-1] (same for "she").
    prefixes_for_debug = [r["prefix"] for r in (train_he[:args.debug_token_infer_n] + train_she[:args.debug_token_infer_n])]
    he_stats = collect_next_token_id_stats(tok, prefixes_for_debug, "he", sample_n=len(prefixes_for_debug))
    she_stats = collect_next_token_id_stats(tok, prefixes_for_debug, "she", sample_n=len(prefixes_for_debug))

    print(f"[TOK] encode(' he')={tok.encode(' he', add_special_tokens=False)} ; encode('he')={tok.encode('he', add_special_tokens=False)}")
    print(f"[TOK] encode(' she')={tok.encode(' she', add_special_tokens=False)} ; encode('she')={tok.encode('she', add_special_tokens=False)}")
    print(f"[TOK-INFER] per-prefix he next-token id distribution: {he_stats}")
    print(f"[TOK-INFER] per-prefix she next-token id distribution: {she_stats}")

    # Print a few sample tokenization/prediction diagnostics
    print("[DEBUG] baseline top tokens @t* for a few HE prefixes:")
    print_prefix_debug(model, tok, [r["prefix"] for r in train_he[:args.debug_n]], device, n=args.debug_n)
    print("[DEBUG] baseline top tokens @t* for a few SHE prefixes:")
    print_prefix_debug(model, tok, [r["prefix"] for r in train_she[:args.debug_n]], device, n=args.debug_n)

    # Load SVD direction
    svd_cache = torch.load(svd_path, map_location="cpu")
    svd_obj = svd_cache["ov"][args.layer][args.head]
    U = svd_obj["U"]   # [D+1, r]
    S = svd_obj["S"]   # [r]
    Vh = svd_obj["Vh"] # [r, D]

    r = int(S.numel())
    if not (0 <= args.sv_idx < r):
        raise ValueError(f"sv_idx out of range: got {args.sv_idx}, but r={r} for this head.")

    u_vec = U[:, args.sv_idx].to(device)
    sigma = float(S[args.sv_idx].item())
    v_vec = Vh[args.sv_idx, :].to(device)

    print(f"\n[LOAD] out_dir={out_dir.resolve()}")
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

    # Estimate class means μ (+ std for debug)
    train_prefixes_he  = [r["prefix"] for r in train_he]
    train_prefixes_she = [r["prefix"] for r in train_she]

    print(f"[MU] estimating on train: he={len(train_prefixes_he)} she={len(train_prefixes_she)} (max_mu={args.max_mu})")
    mu_he, std_he = estimate_mu(model, tok, train_prefixes_he,  args.layer, args.head, u_vec, device, args.batch_size, max_n=args.max_mu)
    mu_she, std_she = estimate_mu(model, tok, train_prefixes_she, args.layer, args.head, u_vec, device, args.batch_size, max_n=args.max_mu)
    print(f"[MU] mu_he={mu_he:.6g}±{std_he:.6g}  mu_she={mu_she:.6g}±{std_she:.6g}")

    sigma_scales = parse_sigma_scales(args.sigma_scales)
    print(f"[EVAL] sigma_scales={sigma_scales}")
    print(f"[EVAL] test sizes: he={len(test_he)} she={len(test_she)} (max_test={args.max_test})")

    # Eval
    metrics_he = eval_split(
        model, tok, [r["prefix"] for r in test_he], "he",
        mu_he=mu_he, mu_she=mu_she,
        
        layer=args.layer, head=args.head,
        u_vec=u_vec, sigma=sigma, v_vec=v_vec,
        sigma_scales=sigma_scales, device=device,
        batch_size=args.batch_size, max_test=args.max_test,
    )

    metrics_she = eval_split(
        model, tok, [r["prefix"] for r in test_she], "she",
        mu_he=mu_he, mu_she=mu_she,
        
        layer=args.layer, head=args.head,
        u_vec=u_vec, sigma=sigma, v_vec=v_vec,
        sigma_scales=sigma_scales, device=device,
        batch_size=args.batch_size, max_test=args.max_test,
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
                f"flip_top1={100*m['flip_rate_top1']:.2f}% (den={m['flip_denom_top1']}, flips={m['flips_top1']})  "
                f"flip_pair={100*m['flip_rate_pair']:.2f}% (den={m['flip_denom_pair']}, flips={m['flips_pair']})  "
                f"| ai={m['ai_mean']:+.3f}±{m['ai_std']:.3f}  Δ||.||={m['delta_l2_mean']:.3f}±{m['delta_l2_std']:.3f}"
            )

    print_table("TEST he subset (diff = he - she)", metrics_he)
    print_table("TEST she subset (diff = she - he)", metrics_she)

    # Save JSON + plots
    out_json = out_dir / f"a4_intervention_l{args.layer}_h{args.head}_k{args.sv_idx}_debug.json"
    plots_dir = out_dir / "a4_plots_debug"
    ensure_dir(plots_dir)

    payload = {
        "layer": args.layer,
        "head": args.head,
        "sv_idx": args.sv_idx,
        "sigma": sigma,
        "mu_he": mu_he,
        "mu_she": mu_she,
        "std_he": std_he,
        "std_she": std_she,
        "he_token_ids_candidates": {" he": tok.encode(" he", add_special_tokens=False)[0], "he": tok.encode("he", add_special_tokens=False)[0]},
        "she_token_ids_candidates": {" she": tok.encode(" she", add_special_tokens=False)[0], "she": tok.encode("she", add_special_tokens=False)[0]},
        "he_infer_stats": he_stats,
        "she_infer_stats": she_stats,
        "sigma_scales": sigma_scales,
        "metrics_he": metrics_he,
        "metrics_she": metrics_she,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    plot_curves(plots_dir, "he_subset", sigma_scales, metrics_he)
    plot_curves(plots_dir, "she_subset", sigma_scales, metrics_she)

    print(f"\n[SAVE] {out_json}")
    print(f"[SAVE] plots -> {plots_dir}")
    print("[DONE]")


if __name__ == "__main__":
    main()