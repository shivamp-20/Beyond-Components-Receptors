#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Experiment 0: Receptor Activation Vector Computation

Given 3 OV receptors specified by (layer, head, sv_idx) and their polarities (+1 or -1),
compute per-layer receptor activations a(l) = [r1^T x_l, r2^T x_l, r3^T x_l]
and gender score s(l) = v^T a(l), for each GP prompt at the decision position.

Notes:
- Your training script defines OV "logit receptors" by taking v_row = Vh[k,:] (d_model)
  and interpreting tokens via receptor_vocab = v_row @ W_U. We follow that: the receptor
  direction used for activations is v_row in residual-stream space. (Token printouts
  are just interpretation.)
  See dump_ov_logit_receptors in your code. :contentReference[oaicite:2]{index=2}
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict
import math

import torch
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast
from transformer_lens import HookedTransformer

# We use TransformerLens run_with_cache to collect residual streams layer-by-layer. :contentReference[oaicite:3]{index=3}
try:
    from transformer_lens import utils as tl_utils
except Exception:
    tl_utils = None


# -------------------------
# Import your training utils (robustly)
# -------------------------
def import_gp_module():
    # Prefer ddp filename (what you ran), but fall back if you also have the non-ddp copy.
    try:
        import train_gp_masks_and_dump_ov_logit_receptors_ddp as gp
        return gp
    except Exception:
        import train_gp_masks_and_dump_ov_logit_receptors as gp
        return gp


gp = import_gp_module()


# -------------------------
# Parsing / data structures
# -------------------------
@dataclass
class ReceptorSpec:
    layer: int
    head: int
    sv_idx: int
    polarity: int  # +1 (male-on-top) or -1 (female-on-top)


def parse_receptors(s: str) -> List[ReceptorSpec]:
    """
    Accepts:
      "L:H:K:P,L:H:K:P,L:H:K:P"
    e.g. "10:9:0:+1,3:5:7:-1,11:2:4:+1"
    """
    parts = [p.strip() for p in s.replace(";", ",").split(",") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"--receptors must specify exactly 3 receptors; got {len(parts)}: {parts}")

    out: List[ReceptorSpec] = []
    for p in parts:
        fields = p.split(":")
        if len(fields) != 4:
            raise ValueError(f"Bad receptor spec '{p}'. Expected 'L:H:K:P'")
        L, H, K, P = fields
        pol = int(P)
        if pol not in (-1, +1):
            raise ValueError(f"Polarity must be +1 or -1, got {pol} in '{p}'")
        out.append(ReceptorSpec(layer=int(L), head=int(H), sv_idx=int(K), polarity=pol))
    return out


def expand_rows_to_examples(rows: List[Dict[str, str]], use_both: bool) -> List[Dict[str, str]]:
    """
    Each GP CSV row contains clean + corrupted versions (prefix/corr_prefix) and their pronouns. :contentReference[oaicite:4]{index=4}
    We treat each as a separate example, labeled by its own pronoun.
    """
    ex = []
    for r in rows:
        ex.append({"text": r["prefix"], "pronoun": r["pronoun"]})
        if use_both:
            ex.append({"text": r["corr_prefix"], "pronoun": r["corr_pronoun"]})
    return ex


# def pronoun_to_y(p: str) -> int:
#     # y=+1 for male-correct, y=-1 for female-correct
#     t = p.strip().lower()
#     if "he" in t:   # robust to "he", "He", etc.
#         return +1
#     if "she" in t:
#         return -1
#     raise ValueError(f"Unknown pronoun label '{p}' (expected contains 'he' or 'she').")

def pronoun_to_y(p: str) -> int:
    t = p.strip().lower()
    if t == "he":
        return +1
    if t == "she":
        return -1
    raise ValueError(f"Unknown pronoun label: {p!r}")



# -------------------------
# Metrics
# -------------------------
def auc_rank(scores: torch.Tensor, labels01: torch.Tensor) -> float:
    """
    AUC via average-rank method with tie handling.
    scores: (N,) float
    labels01: (N,) in {0,1}
    """
    scores = scores.detach().cpu()
    labels01 = labels01.detach().cpu().long()
    N = scores.numel()
    n_pos = int(labels01.sum().item())
    n_neg = N - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    # sort by score
    order = torch.argsort(scores)
    sorted_scores = scores[order]
    sorted_labels = labels01[order]

    # compute average ranks for ties
    ranks = torch.empty(N, dtype=torch.float64)
    i = 0
    rank = 1
    while i < N:
        j = i + 1
        while j < N and sorted_scores[j].item() == sorted_scores[i].item():
            j += 1
        # tie group [i, j)
        avg_rank = (rank + (rank + (j - i) - 1)) / 2.0
        ranks[i:j] = avg_rank
        rank += (j - i)
        i = j

    sum_ranks_pos = float((ranks * sorted_labels.double()).sum().item())
    # Mann–Whitney U
    U = sum_ranks_pos - n_pos * (n_pos + 1) / 2.0
    return float(U / (n_pos * n_neg))


def best_threshold_accuracy(scores: torch.Tensor, labels01: torch.Tensor) -> Tuple[float, float]:
    """Return (best_threshold, best_accuracy) for predicting male if score > threshold."""
    scores = scores.detach().cpu().float()
    labels01 = labels01.detach().cpu().long()
    N = scores.numel()
    if N == 0:
        return float("nan"), float("nan")

    # Candidate thresholds = midpoints between sorted unique scores, plus +/- inf.
    uniq = torch.unique(scores)
    uniq, _ = torch.sort(uniq)
    if uniq.numel() == 1:
        thr = float(uniq[0].item())
        pred = (scores > thr).long()
        acc = float((pred == labels01).float().mean().item())
        return thr, acc

    mids = (uniq[:-1] + uniq[1:]) / 2.0
    candidates = torch.cat([torch.tensor([uniq[0] - 1.0]), mids, torch.tensor([uniq[-1] + 1.0])])

    best_thr = float(candidates[0].item())
    best_acc = -1.0
    for thr_t in candidates:
        pred = (scores > thr_t).long()
        acc = float((pred == labels01).float().mean().item())
        if acc > best_acc:
            best_acc = acc
            best_thr = float(thr_t.item())

    return best_thr, best_acc


def rcs_from_abs_contrib(abs_contrib: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Receptor Concentration Score (RCS) per example.
    abs_contrib: (..., 3) >= 0, where each is |v_k * g_k| (but since v_k is ±1, it's just |g_k|).
    Returns: (...) in [0, 1] where 1=one receptor dominates, 0=equal contributions.
    """
    abs_contrib = abs_contrib.clamp_min(0.0)
    denom = abs_contrib.sum(dim=-1, keepdim=True).clamp_min(eps)
    p = abs_contrib / denom
    H = -(p * (p + eps).log()).sum(dim=-1)
    return 1.0 - H / math.log(3.0)

# -------------------------
# Core experiment
# -------------------------
@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--csv", type=str, required=True, help="e.g. test_gp.csv (must have prefix/pronoun/corr_prefix/corr_pronoun).")
    ap.add_argument("--out_dir", type=str, required=True, help="e.g. outputs/gp (must contain svd_cache.pt).")
    ap.add_argument("--receptors", type=str, required=True, help="Exactly 3 receptors: 'L:H:K:P,L:H:K:P,L:H:K:P'")
    ap.add_argument("--use_both", type=int, default=1, help="1=use prefix + corr_prefix as separate examples. 0=only prefix.")
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_examples", type=int, default=0, help="0=all")
    ap.add_argument("--use_ln_final", type=int, default=0, help="1=apply final LN before dotting with v_row (optional sanity toggle).")
    ap.add_argument("--plot_mode", type=str, default="mean", choices=["mean", "all"])
    ap.add_argument("--save_prefix", type=str, default="exp0")

    ap.add_argument(
        "--center_mode",
        type=str,
        default="none",
        choices=["none", "score", "acts"],
        help=(
            "Optional zero-centering to remove constant offsets in receptor scores. "
            "'none': no centering. "
            "'score': subtract mean s(l) across ALL prompts at each layer (s <- s - mean(s)). "
            "'acts': subtract mean activation per receptor at each layer (a <- a - mean(a)), then recompute s=v^T a. "
            "Note: centering does not affect AUC much (ranking), but makes acc@0 more meaningful."
        ),
    )
    args = ap.parse_args()

    device = args.device
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load model + tokenizer
    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = HookedTransformer.from_pretrained("gpt2-small", device=device)
    model.eval()

    # Load GP rows (your loader enforces required columns). :contentReference[oaicite:5]{index=5}
    csv_path = Path(args.data_dir) / args.csv
    rows = gp.load_gp_csv(str(csv_path))
    examples = expand_rows_to_examples(rows, use_both=(int(args.use_both) == 1))
    if args.max_examples and args.max_examples > 0:
        examples = examples[: args.max_examples]

    # Labels
    y = torch.tensor([pronoun_to_y(e["pronoun"]) for e in examples], device="cpu")  # (N,)
    print("label counts:", {+1: int((y==1).sum()), -1: int((y==-1).sum())})
    print("unique pronoun strings:", sorted(set([e["pronoun"] for e in examples]))[:20])
    labels01 = (y == 1).long()

    # Parse receptor specs
    specs = parse_receptors(args.receptors)
    polarity = torch.tensor([s.polarity for s in specs], dtype=torch.float32, device=device)  # (3,)

    # Load SVD cache and extract the 3 v_rows (right singular vectors in Vh). :contentReference[oaicite:6]{index=6}
    svd_path = out_dir / "svd_cache.pt"
    if not svd_path.exists():
        raise FileNotFoundError(f"Missing {svd_path} (run training first).")
    _, ov, _, _, _ = gp.load_svd_cache(str(svd_path), device=device)

    v_rows = []
    print("\n[RECEPTORS] Using OV right singular vectors v_row = Vh[sv_idx] (d_model).")
    print("[RECEPTORS] Token interpretation uses receptor_vocab = v_row @ W_U. :contentReference[oaicite:7]{index=7}")

    # Pronoun token IDs for quick sanity prints
    he_id = tokenizer.encode(" he", add_special_tokens=False)[0]
    she_id = tokenizer.encode(" she", add_special_tokens=False)[0]

    for i, s in enumerate(specs, start=1):
        svd = ov[s.layer][s.head]
        if s.sv_idx < 0 or s.sv_idx >= svd.r:
            raise ValueError(f"Receptor {i}: sv_idx={s.sv_idx} out of range (rank_trainable={svd.r}) at (L={s.layer},H={s.head}).")
        v_row = svd.Vh[s.sv_idx, :].contiguous()  # (d_model,) :contentReference[oaicite:8]{index=8}
        v_rows.append(v_row)

        # Print top/bottom tokens just for sanity, same as your dump_ov_logit_receptors. :contentReference[oaicite:9]{index=9}
        rec_vocab = (v_row @ model.W_U).detach().cpu()  # (vocab,)
        top = gp.topk_tokens(tokenizer, rec_vocab, k=10)
        bot = gp.bottomk_tokens(tokenizer, rec_vocab, k=10)
        he_score = float(rec_vocab[he_id].item())
        she_score = float(rec_vocab[she_id].item())

        print(f"\n  R{i}: (layer={s.layer}, head={s.head}, sv_idx={s.sv_idx}) polarity={s.polarity:+d}")
        print(f"      pronoun scores: ' he'={he_score:+.4g}, ' she'={she_score:+.4g}")
        # print("      TOP   :", ", ".join([f\"{d['token_str']!r}:{float(d['score']):+.3g}\" for d in top[:10]]))
        # print("      BOTTOM:", ", ".join([f\"{d['token_str']!r}:{float(d['score']):+.3g}\" for d in bot[:10]]))
        print("      TOP   :", ", ".join([f"{d['token_str']!r}:{float(d['score']):+.3g}" for d in top[:10]]))
        print("      BOTTOM:", ", ".join([f"{d['token_str']!r}:{float(d['score']):+.3g}" for d in bot[:10]]))


    # Stack receptor directions: R is (3, d_model)
    R = torch.stack(v_rows, dim=0)  # (3, D)

    # Tokenize all texts (using your helper logic: encode then pad manually). :contentReference[oaicite:10]{index=10}
    texts = [e["text"] for e in examples]
    ids_list = gp._encode_texts(tokenizer, texts)
    max_len = max(len(x) for x in ids_list)
    tokens, attn, last_idx = gp._pad_to_length(ids_list, max_len, tokenizer.pad_token_id, device=device)

    N = tokens.shape[0]
    n_layers = model.cfg.n_layers
    all_s = torch.empty((N, n_layers), dtype=torch.float32)
    all_a = torch.empty((N, n_layers, 3), dtype=torch.float32)
    # Final-output diagnostics (at decision position):
    # - top1 token (full vocab)
    # - he_vs_she_margin = logit(' he') - logit(' she')
    all_top1 = torch.empty((N,), dtype=torch.long)
    all_margin = torch.empty((N,), dtype=torch.float32)


    # batching
    bs = int(args.batch_size)
    print(f"\n[RUN] N={N} examples | seq_len={max_len} | layers={n_layers} | batch_size={bs}")

    def run_with_cache_safe(toks):
        # TransformerLens run_with_cache returns (logits, cache). :contentReference[oaicite:11]{index=11}
        try:
            return model.run_with_cache(toks, remove_batch_dim=False)
        except TypeError:
            return model.run_with_cache(toks)

    for start in range(0, N, bs):
        end = min(N, start + bs)
        toks_b = tokens[start:end]
        last_b = last_idx[start:end]
        B = toks_b.shape[0]

        logits, cache = run_with_cache_safe(toks_b)
        # final logits at decision position (for sanity checks only)
        ar_logits = torch.arange(B, device=device)
        logits_pos = logits[ar_logits, last_b, :]  # (B, vocab)
        top1 = torch.argmax(logits_pos, dim=-1)  # (B,)
        margin = logits_pos[:, he_id] - logits_pos[:, she_id]  # (B,)
        all_top1[start:end] = top1.detach().cpu()
        all_margin[start:end] = margin.detach().cpu()

        ar = torch.arange(B, device=device)
        for l in range(n_layers):
            key = None
            if tl_utils is not None:
                key = tl_utils.get_act_name("resid_post", l)
            else:
                key = f"blocks.{l}.hook_resid_post"

            if key not in cache:
                # give a helpful error with nearby keys
                sample_keys = [k for k in cache.keys() if "resid" in str(k)]
                raise KeyError(f"Cache missing key '{key}'. Resid-related keys include: {sample_keys[:20]}")

            resid = cache[key]  # (B,S,D)
            x_l = resid[ar, last_b, :]  # (B,D)

            if int(args.use_ln_final) == 1:
                x_l = model.ln_final(x_l)

            # a_l = x_l @ R^T  => (B,3)
            a_l = x_l @ R.T
            s_l = a_l @ polarity

            all_a[start:end, l, :] = a_l.detach().cpu()
            all_s[start:end, l] = s_l.detach().cpu()

        if (start // bs) % 10 == 0 or end == N:
            print(f"  processed {end}/{N}")


    # -------------------------
    # Optional zero-centering (quick cleanup)
    # -------------------------
    # Motivation: s(l) can have a constant offset because receptor activations are not zero-centered.
    # This does NOT affect AUC (ranking), but it can make acc@0 meaningless if scores are shifted.
    center_mode = str(args.center_mode).lower()

    # Keep raw copies so you can compare centered vs raw without re-running
    all_s_raw = all_s.clone()
    all_a_raw = all_a.clone()

    # Raw per-layer means (computed BEFORE any centering)
    mean_s_layer_raw = all_s.mean(dim=0)  # (L,)
    mean_a_layer_raw = all_a.mean(dim=0)  # (L,3)

    if center_mode == "none":
        pass
    elif center_mode == "score":
        # s(l) <- s(l) - mean_{examples}(s(l))
        all_s = all_s - mean_s_layer_raw.view(1, -1)
    elif center_mode == "acts":
        # a(l) <- a(l) - mean_{examples}(a(l)); then recompute s(l) = v^T a(l)
        all_a = all_a - mean_a_layer_raw.view(1, mean_a_layer_raw.shape[0], 3)
        all_s = torch.einsum("nlk,k->nl", all_a, polarity.detach().cpu().float())
    else:
        raise ValueError(f"Unknown --center_mode {args.center_mode!r}. Expected one of: none, score, acts")

    if center_mode != "none":
        mean_after = all_s.mean(dim=0)
        print(
            f"\n[CENTER] center_mode={center_mode} | "
            f"mean_s before (L_last)={float(mean_s_layer_raw[-1]):+.4g} "
            f"after={float(mean_after[-1]):+.4g} (should be ~0)"
        )
    # -------------------------
    # Summaries per layer
    # -------------------------
    y_cpu = y.cpu()
    labels01_cpu = labels01.cpu()
    is_male = (y_cpu == 1)
    is_female = (y_cpu == -1)
    polarity_cpu = polarity.detach().cpu()  # (3,)
    # Overall model-output diagnostics at the decision position (final layer only):
    expected_tok = torch.where(labels01_cpu == 1, torch.tensor(he_id), torch.tensor(she_id))
    acc_top1 = float((all_top1 == expected_tok).float().mean().item())
    other_rate = float(((all_top1 != he_id) & (all_top1 != she_id)).float().mean().item())
    acc_he_vs_she = float(((all_margin > 0).long() == labels01_cpu).float().mean().item())


    layer_stats = []
    aucs = []
    accs = []
    accs_best = []

    for l in range(n_layers):
        s_l = all_s[:, l]
        s_m = s_l[is_male]
        s_f = s_l[is_female]

        mean_m = float(s_m.mean().item()) if s_m.numel() else float("nan")
        mean_f = float(s_f.mean().item()) if s_f.numel() else float("nan")
        sep = mean_m - mean_f

        # threshold at 0: predict male if s>0
        pred_male = (s_l > 0).long()
        acc = float((pred_male == labels01_cpu).float().mean().item())
        auc = auc_rank(s_l, labels01_cpu)

        # best threshold (for diagnostics): predict male if s > thr_best
        thr_best, acc_best = best_threshold_accuracy(s_l, labels01_cpu)

        # receptor contributions + RCS per example
        a_l = all_a[:, l, :]  # (N,3)
        abs_contrib = (a_l * polarity_cpu.view(1, 3)).abs()  # (N,3)
        rcs = rcs_from_abs_contrib(abs_contrib)  # (N,)
        rcs_all = float(rcs.mean().item())
        rcs_male = float(rcs[is_male].mean().item()) if int(is_male.sum()) else float('nan')
        rcs_female = float(rcs[is_female].mean().item()) if int(is_female.sum()) else float('nan')
        # dominance diagnostics: how often one receptor takes most of the mass
        p = abs_contrib / abs_contrib.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        pmax = p.max(dim=-1).values  # (N,)
        pmax_mean = float(pmax.mean().item())
        dom80 = float((pmax > 0.80).float().mean().item())
        dom90 = float((pmax > 0.90).float().mean().item())

        # receptor dominance: average |v_k * a_k| contribution
        contrib = (a_l * polarity_cpu.view(1, 3)).abs().mean(dim=0)  # (3,)
        contrib = contrib / (contrib.sum() + 1e-12)

        layer_stats.append({
            "layer": l,
            "mean_s_male": mean_m,
            "mean_s_female": mean_f,
            "sep_mean": float(sep),
            "acc_sign0": acc,
            "thr_best": float(thr_best),
            "acc_best": float(acc_best),
            "rcs_mean": float(rcs_all),
            "rcs_mean_male": float(rcs_male),
            "rcs_mean_female": float(rcs_female),
            "pmax_mean": float(pmax_mean),
            "dom_frac_gt_0.80": float(dom80),
            "dom_frac_gt_0.90": float(dom90),
            "auc": float(auc),
            "contrib_frac": [float(x.item()) for x in contrib],
        })
        aucs.append(auc)
        accs.append(acc)
        accs_best.append(acc_best)

    # L*: earliest layer where separation is strong (you can change thresholds later)
    L_star_auc = None
    for l, auc in enumerate(aucs):
        if auc == auc and auc >= 0.75:  # not nan and >= 0.75
            L_star_auc = l
            break

    L_star_acc = None
    for l, acc in enumerate(accs):
        if acc >= 0.75:
            L_star_acc = l
            break

    L_star_acc_best = None
    for l, acc in enumerate(accs_best):
        if acc >= 0.75:
            L_star_acc_best = l
            break

    summary = {
        "receptors": [{"layer": s.layer, "head": s.head, "sv_idx": s.sv_idx, "polarity": s.polarity} for s in specs],
        "N": int(N),
        "n_layers": int(n_layers),
        "center_mode": center_mode,
        "mean_s_layer_raw": [float(x) for x in mean_s_layer_raw.tolist()],
        "mean_a_layer_raw": [[float(y) for y in row] for row in mean_a_layer_raw.tolist()],
        "final_acc_top1_expected": float(acc_top1),
        "final_acc_he_vs_she_only": float(acc_he_vs_she),
        "final_other_rate_top1": float(other_rate),
        "L_star_auc>=0.75": L_star_auc,
        "L_star_acc>=0.75": L_star_acc,
        "L_star_acc_best>=0.75": L_star_acc_best,
        "per_layer": layer_stats,
    }

    # Save outputs
    prefix = args.save_prefix
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    payload = {
        "all_s": all_s,
        "all_a": all_a,
        "y": y_cpu,
        "all_margin": all_margin,
        "all_top1": all_top1,
        "he_id": int(he_id),
        "she_id": int(she_id),
        "center_mode": center_mode,
        # raw copies + raw means
        "all_s_raw": all_s_raw,
        "all_a_raw": all_a_raw,
        "mean_s_layer_raw": mean_s_layer_raw,
        "mean_a_layer_raw": mean_a_layer_raw,
    }
    torch.save(payload, out_dir / f"{prefix}_tensors.pt")

    # Plot
    xs = list(range(n_layers))
    plt.figure()
    if args.plot_mode == "all":
        # plot all trajectories (can be many lines)
        for i in range(N):
            plt.plot(xs, all_s[i].tolist(), alpha=0.12)
    else:
        # plot group means
        mean_m = [float(all_s[is_male, l].mean().item()) for l in range(n_layers)]
        mean_f = [float(all_s[is_female, l].mean().item()) for l in range(n_layers)]
        plt.plot(xs, mean_m, label="male-correct mean")
        plt.plot(xs, mean_f, label="female-correct mean")
        plt.legend()

    plt.axhline(0.0)
    plt.xlabel("Layer")
    plt.ylabel("Gender score s(l) = v^T a(l)")
    plt.title("Experiment 0: receptor-score trajectories")
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_traj.png", dpi=200)

    if center_mode != "none":
        plt.figure()
        if args.plot_mode == "all":
            for i in range(N):
                plt.plot(xs, all_s_raw[i].tolist(), alpha=0.12)
        else:
            mean_m_raw = [float(all_s_raw[is_male, l].mean().item()) for l in range(n_layers)]
            mean_f_raw = [float(all_s_raw[is_female, l].mean().item()) for l in range(n_layers)]
            plt.plot(xs, mean_m_raw, label="male-correct mean (raw)")
            plt.plot(xs, mean_f_raw, label="female-correct mean (raw)")
            plt.legend()
        plt.axhline(0.0)
        plt.xlabel("Layer")
        plt.ylabel("Gender score s(l) (raw)")
        plt.title("Experiment 0: receptor-score trajectories (raw)")
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_traj_raw.png", dpi=200)

    # Plot RCS trajectories (group means)
    plt.figure()
    rcs_m = [st['rcs_mean_male'] for st in layer_stats]
    rcs_f = [st['rcs_mean_female'] for st in layer_stats]
    plt.plot(xs, rcs_m, label='male-correct mean RCS')
    plt.plot(xs, rcs_f, label='female-correct mean RCS')
    plt.ylim(0.0, 1.0)
    plt.xlabel('Layer')
    plt.ylabel('RCS')
    plt.title('Experiment 0: receptor concentration (RCS)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / f"{prefix}_rcs.png", dpi=200)

    if center_mode != "none":
        rcs_m_raw = []
        rcs_f_raw = []
        for l in range(n_layers):
            a_l_raw = all_a_raw[:, l, :]  # (N,3)
            abs_contrib_raw = (a_l_raw * polarity_cpu.view(1, 3)).abs()
            rcs_raw = rcs_from_abs_contrib(abs_contrib_raw)
            rcs_m_raw.append(float(rcs_raw[is_male].mean().item()))
            rcs_f_raw.append(float(rcs_raw[is_female].mean().item()))
        plt.figure()
        plt.plot(xs, rcs_m_raw, label="male-correct mean RCS (raw)")
        plt.plot(xs, rcs_f_raw, label="female-correct mean RCS (raw)")
        plt.ylim(0.0, 1.0)
        plt.xlabel("Layer")
        plt.ylabel("RCS (raw)")
        plt.title("Experiment 0: receptor concentration (RCS, raw)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"{prefix}_rcs_raw.png", dpi=200)

    # Print a compact table
    print(f"\n[SAVED] {out_dir / f'{prefix}_summary.json'}")
    print(f"[SAVED] {out_dir / f'{prefix}_traj.png'}")
    print(f"[SAVED] {out_dir / f'{prefix}_rcs.png'}")
    print(f"[SAVED] {out_dir / f'{prefix}_tensors.pt'}")

    if center_mode != "none":
        print(f"[SAVED] {out_dir / f'{prefix}_traj_raw.png'}")
        print(f"[SAVED] {out_dir / f'{prefix}_rcs_raw.png'}")

    print("\n[FINAL OUTPUT] Decision-position diagnostics (final layer logits):")
    print(f"  acc_top1_expected={acc_top1:.3f}  acc_he_vs_she_only={acc_he_vs_she:.3f}  other_rate_top1={other_rate:.3f}")

    print("\n[RESULT] Per-layer (acc@0, acc*, AUC, RCS, pmax/dom80, mean_male, mean_female, sep, contrib):")
    for st in layer_stats:
        contrib_str = [f"{c:.2f}" for c in st["contrib_frac"]]
        print(
            f"  L{st['layer']:02d}  acc@0={st['acc_sign0']:.3f}  acc*={st['acc_best']:.3f} (thr={st['thr_best']:+.3g})  "
            f"auc={st['auc']:.3f}  rcs={st['rcs_mean']:.3f}  pmax={st['pmax_mean']:.2f}  dom80={st['dom_frac_gt_0.80']:.2f}  "
            f"μ_m={st['mean_s_male']:+.3g}  μ_f={st['mean_s_female']:+.3g}  Δ={st['sep_mean']:+.3g}  contrib={contrib_str}"
        )

    print("\n[CHECK] Success heuristic:")
    print("  - If AUC ramps above ~0.75 at some layer and stays high -> trajectories separate (framework works).")
    print("  - acc@0 uses threshold 0 (often suboptimal if scores are shifted). acc* reports best-threshold accuracy per layer.")
    print("  - L* (AUC) and L* (acc*) are in the summary JSON.")
    print("  - RCS in [0,1]: ~0 means contributions are spread across receptors; ~1 means one receptor dominates.")


if __name__ == "__main__":
    main()
