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


def pronoun_to_y(p: str) -> int:
    # y=+1 for male-correct, y=-1 for female-correct
    t = p.strip().lower()
    if "he" in t:   # robust to "he", "He", etc.
        return +1
    if "she" in t:
        return -1
    raise ValueError(f"Unknown pronoun label '{p}' (expected contains 'he' or 'she').")


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
    # Summaries per layer
    # -------------------------
    y_cpu = y.cpu()
    labels01_cpu = labels01.cpu()
    is_male = (y_cpu == 1)
    is_female = (y_cpu == -1)

    layer_stats = []
    aucs = []
    accs = []

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

        # receptor dominance: average |v_k * a_k| contribution
        a_l = all_a[:, l, :]  # (N,3)
        contrib = (a_l * torch.tensor([s.polarity for s in specs]).view(1, 3)).abs().mean(dim=0)  # (3,)
        contrib = contrib / (contrib.sum() + 1e-12)

        layer_stats.append({
            "layer": l,
            "mean_s_male": mean_m,
            "mean_s_female": mean_f,
            "sep_mean": float(sep),
            "acc_sign0": acc,
            "auc": float(auc),
            "contrib_frac": [float(x.item()) for x in contrib],
        })
        aucs.append(auc)
        accs.append(acc)

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

    summary = {
        "receptors": [{"layer": s.layer, "head": s.head, "sv_idx": s.sv_idx, "polarity": s.polarity} for s in specs],
        "N": int(N),
        "n_layers": int(n_layers),
        "L_star_auc>=0.75": L_star_auc,
        "L_star_acc>=0.75": L_star_acc,
        "per_layer": layer_stats,
    }

    # Save outputs
    prefix = args.save_prefix
    (out_dir / f"{prefix}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save({"all_s": all_s, "all_a": all_a, "y": y_cpu}, out_dir / f"{prefix}_tensors.pt")

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

    # Print a compact table
    print(f"\n[SAVED] {out_dir / f'{prefix}_summary.json'}")
    print(f"[SAVED] {out_dir / f'{prefix}_traj.png'}")
    print(f"[SAVED] {out_dir / f'{prefix}_tensors.pt'}")

    print("\n[RESULT] Per-layer (acc@0, AUC, mean_male, mean_female, sep):")
    for st in layer_stats:
        print(
            f"  L{st['layer']:02d}  acc={st['acc_sign0']:.3f}  auc={st['auc']:.3f}  "
            f"μ_m={st['mean_s_male']:+.3g}  μ_f={st['mean_s_female']:+.3g}  Δ={st['sep_mean']:+.3g}  "
            f"contrib={['%.2f'%c for c in st['contrib_frac']]}"
        )

    print("\n[CHECK] Success heuristic:")
    print("  - If AUC and acc ramp above ~0.75 at some layer and stay high -> trajectories separate (framework works).")
    print("  - L* is the earliest layer where AUC>=0.75 / acc>=0.75 (see summary JSON).")
    print("  - contrib_frac tells if one receptor dominates (RCS-ish diagnostic).")


if __name__ == "__main__":
    main()
