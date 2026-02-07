#!/usr/bin/env python3
"""
GP Experiment 0b: Component contribution ranking in receptor space.

For each component n in GPT-2 Small:
  - attention heads: 12 layers * 12 heads = 144 components
  - MLPs: 12 layers = 12 components
Total = 156 components.

We compute (at the decision position t* for each prompt):
  C[k, n] = r_k · output_n(t*)       for k in {1,2,3}
  mu_n    = y * sum_k v_k * C[k,n]   where y=+1 (male-correct), y=-1 (female-correct)
We rank components by mean(|mu_n|) over the dataset.

This script assumes your receptors are in residual-stream space (d_model), i.e. OV right singular vectors Vh[sv_idx].
That matches your existing exp0 printing: receptor_vocab = v_row @ W_U (so v_row is d_model).

Implementation detail for heads:
  We hook `blocks.{l}.attn.hook_z` (shape: [B, S, H, d_head]) and map to residual via W_O:
    head_out = z @ W_O  -> [B, H, d_model]
  Then dot with receptors.

Outputs:
  - exp0b_topk.txt (printed table)
  - exp0b_summary.json (machine-readable)
  - exp0b_head_LxHy_ex*_pattern.png + exp0b_head_LxHy_topkeys.json for top heads (optional)
"""

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import Dict, List

import torch
import matplotlib.pyplot as plt
from transformers import GPT2TokenizerFast

try:
    from transformer_lens import HookedTransformer
except Exception as e:
    raise ImportError(
        "transformer_lens is required. In Kaggle: pip install transformer_lens"
    ) from e

import train_gp_masks_and_dump_ov_logit_receptors_ddp as gp


@dataclass
class ReceptorSpec:
    layer: int
    head: int
    sv_idx: int
    polarity: int  # +1 or -1


def parse_receptors_arg(s: str) -> List[ReceptorSpec]:
    """Parse --receptors like: "10,9,0,+1;11,8,6,+1;9,7,1,-1"""

    out: List[ReceptorSpec] = []
    parts = [p.strip() for p in s.split(";") if p.strip()]
    if len(parts) != 3:
        raise ValueError(f"--receptors must have exactly 3 entries separated by ';'. Got {len(parts)}: {parts}")
    for p in parts:
        fields = [x.strip() for x in p.split(",")]
        if len(fields) != 4:
            raise ValueError(f"Bad receptor spec '{p}'. Expected layer,head,sv_idx,polarity")
        layer, head, sv_idx, pol = fields
        pol_i = int(pol)
        if pol_i not in (-1, +1):
            raise ValueError(f"polarity must be +1 or -1, got {pol_i} in '{p}'")
        out.append(ReceptorSpec(layer=int(layer), head=int(head), sv_idx=int(sv_idx), polarity=pol_i))
    return out


def expand_rows_to_examples(rows: List[Dict[str, str]], use_both: int = 1) -> List[Dict[str, str]]:
    out: List[Dict[str, str]] = []
    for r in rows:
        p = (r.get("pronoun") or "").strip().lower()
        if p in ("he", "she"):
            out.append({"text": r["prefix"], "label": p})
        if use_both == 1:
            cp = (r.get("corr_pronoun") or "").strip().lower()
            if cp in ("he", "she"):
                out.append({"text": r["corr_prefix"], "label": cp})
    return out


def token_strs(tokenizer: GPT2TokenizerFast, tok_ids: List[int]) -> List[str]:
    return [tokenizer.decode([t]) for t in tok_ids]


class Collector:
    def __init__(self, n_layers: int, n_heads: int, K: int, device: torch.device):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.K = K
        self.device = device

        self.n_head_components = n_layers * n_heads
        self.n_mlp_components = n_layers
        self.n_components = self.n_head_components + self.n_mlp_components

        self.sum_abs_mu = torch.zeros((self.n_components,), device=device)
        self.sum_mu = torch.zeros((self.n_components,), device=device)
        self.sum_mu_k = torch.zeros((self.n_components, K), device=device)
        self.sum_abs_mu_k = torch.zeros((self.n_components, K), device=device)
        self.N = 0

    def add_heads(self, layer: int, mu_bh: torch.Tensor, mu_k_bhk: torch.Tensor):
        B, H = mu_bh.shape
        base = layer * self.n_heads
        self.sum_abs_mu[base:base + H] += mu_bh.abs().sum(dim=0)
        self.sum_mu[base:base + H] += mu_bh.sum(dim=0)
        self.sum_mu_k[base:base + H, :] += mu_k_bhk.sum(dim=0)
        self.sum_abs_mu_k[base:base + H, :] += mu_k_bhk.abs().sum(dim=0)

    def add_mlp(self, layer: int, mu_b: torch.Tensor, mu_k_bk: torch.Tensor):
        idx = self.n_head_components + layer
        self.sum_abs_mu[idx] += mu_b.abs().sum()
        self.sum_mu[idx] += mu_b.sum()
        self.sum_mu_k[idx, :] += mu_k_bk.sum(dim=0)
        self.sum_abs_mu_k[idx, :] += mu_k_bk.abs().sum(dim=0)

    def add_count(self, B: int):
        self.N += int(B)

    def finalize(self):
        if self.N == 0:
            raise RuntimeError("No examples processed.")
        return dict(
            mean_abs_mu=(self.sum_abs_mu / self.N),
            mean_mu=(self.sum_mu / self.N),
            mean_mu_k=(self.sum_mu_k / self.N),
            mean_abs_mu_k=(self.sum_abs_mu_k / self.N),
        )


def component_name(idx: int, n_layers: int, n_heads: int) -> str:
    head_total = n_layers * n_heads
    if idx < head_total:
        l = idx // n_heads
        h = idx % n_heads
        return f"Attn L{l:02d}H{h:02d}"
    l = idx - head_total
    return f"MLP  L{l:02d}"


def component_layer(idx: int, n_layers: int, n_heads: int) -> int:
    head_total = n_layers * n_heads
    if idx < head_total:
        return idx // n_heads
    return idx - head_total


def component_is_attn(idx: int, n_layers: int, n_heads: int) -> bool:
    return idx < (n_layers * n_heads)


def compute_rcs(contrib_abs: torch.Tensor, eps: float = 1e-12) -> float:
    K = contrib_abs.numel()
    s = float(contrib_abs.sum().item())
    if s <= eps:
        return 0.0
    p = (contrib_abs / (s + eps)).clamp(min=eps)
    H = float(-(p * p.log()).sum().item())
    return float(1.0 - H / math.log(K))


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--csv", type=str, required=True)
    ap.add_argument("--out_dir", type=str, required=True)
    ap.add_argument("--receptors", type=str, required=True)
    ap.add_argument("--use_both", type=int, default=1)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--max_examples", type=int, default=-1)
    ap.add_argument("--topk", type=int, default=15)
    ap.add_argument("--viz_heads", type=int, default=3)
    ap.add_argument("--viz_examples", type=int, default=3)
    ap.add_argument("--save_prefix", type=str, default="exp0b")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device(args.device)

    model = HookedTransformer.from_pretrained("gpt2-small")
    model.to(device)
    model.eval()
    print("Loaded pretrained model gpt2-small into HookedTransformer")

    tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    csv_path = os.path.join(args.data_dir, args.csv)
    rows = gp.load_gp_csv(csv_path)
    examples = expand_rows_to_examples(rows, use_both=int(args.use_both))
    if args.max_examples and args.max_examples > 0:
        examples = examples[: int(args.max_examples)]

    texts = [e["text"] for e in examples]
    labels = [e["label"] for e in examples]
    y = torch.tensor([+1 if l == "he" else -1 for l in labels], dtype=torch.float32, device=device)

    counts = {+1: int((y == 1).sum().item()), -1: int((y == -1).sum().item())}
    print(f"label counts: {counts}")
    print(f"unique pronoun strings: {sorted(list(set(labels)))}\n")

    receptor_specs = parse_receptors_arg(args.receptors)
    svd_path = os.path.join(args.out_dir, "svd_cache.pt")
    qk, ov, mlp_in, mlp_out, rank_total_ov = gp.load_svd_cache(svd_path, device=args.device)
he_id = tokenizer.encode(" he", add_special_tokens=False)[0]
    she_id = tokenizer.encode(" she", add_special_tokens=False)[0]

    v_rows = []
    v_pols = []
    print("[RECEPTORS] Using OV right singular vectors v_row = Vh[sv_idx] (d_model).")
    print("[RECEPTORS] Token interpretation uses receptor_vocab = v_row @ W_U.\n")

    for i, s in enumerate(receptor_specs, start=1):
        Vh = ov[s.layer][s.head].Vh  # (rank, d_model)
v_row = Vh[s.sv_idx].to(device)
        v_rows.append(v_row)
        v_pols.append(s.polarity)

        rec_vocab = (v_row @ model.W_U).detach().cpu()
        he_score = float(rec_vocab[he_id].item())
        she_score = float(rec_vocab[she_id].item())
        top = gp.topk_tokens(tokenizer, rec_vocab, k=10)
        bot = gp.bottomk_tokens(tokenizer, rec_vocab, k=10)

        print(f"  R{i}: (layer={s.layer}, head={s.head}, sv_idx={s.sv_idx}) polarity={s.polarity:+d}")
        print(f"      pronoun scores: ' he'={he_score:+.4g}, ' she'={she_score:+.4g}")
        print("      TOP   :", ", ".join([f"{d['token_str']!r}:{float(d['score']):+.3g}" for d in top[:10]]))
        print("      BOTTOM:", ", ".join([f"{d['token_str']!r}:{float(d['score']):+.3g}" for d in bot[:10]]))
        print()

    R = torch.stack(v_rows, dim=0)  # (K,d_model)
    K = R.shape[0]
    polarity = torch.tensor(v_pols, dtype=torch.float32, device=device)

    ids_list = gp._encode_texts(tokenizer, texts)
    max_len = max(len(x) for x in ids_list)
    tokens, _attn, last_idx = gp._pad_to_length(ids_list, max_len, tokenizer.pad_token_id, device=str(device))

    N = tokens.shape[0]
    n_layers = model.cfg.n_layers
    n_heads = model.cfg.n_heads
    print(f"[RUN] N={N} examples | seq_len={max_len} | layers={n_layers} | heads={n_heads} | batch_size={args.batch_size}")

    collector = Collector(n_layers=n_layers, n_heads=n_heads, K=K, device=device)
    W_O = model.W_O  # (layers, heads, d_head, d_model) in TransformerLens

    # state updated each batch
    state = {"last_b": None, "y_b": None}

    def make_attn_hook(layer: int):
        def hook_fn(z: torch.Tensor, hook):
            last_b = state["last_b"]
            y_b = state["y_b"]
            B = z.shape[0]
            ar = torch.arange(B, device=z.device)
            z_dec = z[ar, last_b, :, :]  # (B,H,d_head)
            head_out = torch.einsum("bhd,hdm->bhm", z_dec, W_O[layer])  # (B,H,d_model)
            C = torch.einsum("bhm,km->bhk", head_out, R)  # (B,H,K)
            mu_k = (y_b[:, None, None]) * (polarity[None, None, :]) * C  # (B,H,K)
            mu = mu_k.sum(dim=-1)  # (B,H)
            collector.add_heads(layer, mu, mu_k)
        return hook_fn

    def make_mlp_hook(layer: int):
        def hook_fn(mlp_out: torch.Tensor, hook):
            last_b = state["last_b"]
            y_b = state["y_b"]
            B = mlp_out.shape[0]
            ar = torch.arange(B, device=mlp_out.device)
            out_dec = mlp_out[ar, last_b, :]  # (B,d_model)
            C = torch.einsum("bm,km->bk", out_dec, R)  # (B,K)
            mu_k = (y_b[:, None]) * (polarity[None, :]) * C  # (B,K)
            mu = mu_k.sum(dim=-1)  # (B,)
            collector.add_mlp(layer, mu, mu_k)
        return hook_fn

    fwd_hooks = []
    for l in range(n_layers):
        fwd_hooks.append((f"blocks.{l}.attn.hook_z", make_attn_hook(l)))
        fwd_hooks.append((f"blocks.{l}.hook_mlp_out", make_mlp_hook(l)))

    bs = int(args.batch_size)
    for start in range(0, N, bs):
        end = min(N, start + bs)
        state["last_b"] = last_idx[start:end]
        state["y_b"] = y[start:end]
        toks_b = tokens[start:end]
        _ = model.run_with_hooks(toks_b, fwd_hooks=fwd_hooks, return_type="logits")
        collector.add_count(end - start)
        if (start // bs) % 10 == 0 or end == N:
            print(f"  processed {end}/{N}")

    st = collector.finalize()
    mean_abs_mu = st["mean_abs_mu"].detach().cpu()
    mean_mu = st["mean_mu"].detach().cpu()
    mean_mu_k = st["mean_mu_k"].detach().cpu()
    mean_abs_mu_k = st["mean_abs_mu_k"].detach().cpu()

    order = torch.argsort(mean_abs_mu, descending=True).tolist()
    total = float(mean_abs_mu.sum().item()) + 1e-12

    cum = 0.0
    cum_share = []
    n80 = None
    for i, idx in enumerate(order):
        cum += float(mean_abs_mu[idx].item())
        cs = cum / total
        cum_share.append(cs)
        if n80 is None and cs >= 0.80:
            n80 = i + 1

    head_total = n_layers * n_heads
    attn_share = float(mean_abs_mu[:head_total].sum().item() / total)
    mlp_share = float(mean_abs_mu[head_total:].sum().item() / total)

    topk = min(int(args.topk), len(order))
    lines = []
    header = f"{'rank':>4}  {'component':<12}  {'|mu|':>10}  {'mu':>10}  {'mu_k (signed contrib)':<34}  {'RCS':>6}  {'cum%':>6}"
    lines.append(header)
    lines.append("-" * len(header))

    for r in range(topk):
        idx = order[r]
        muabs = float(mean_abs_mu[idx].item())
        mus = float(mean_mu[idx].item())
        muvec = [float(x) for x in mean_mu_k[idx].tolist()]
        rcs = compute_rcs(mean_abs_mu_k[idx])
        lines.append(
            f"{r+1:>4}  {component_name(idx,n_layers,n_heads):<12}  {muabs:>10.4g}  {mus:>10.4g}  "
            f"[{muvec[0]:+.3g}, {muvec[1]:+.3g}, {muvec[2]:+.3g}]".ljust(34)
            + f"  {rcs:>6.3f}  {100*cum_share[r]:>5.1f}"
        )

    out_txt = os.path.join(args.out_dir, f"{args.save_prefix}_topk.txt")
    with open(out_txt, "w") as f:
        f.write("\n".join(lines) + "\n")

    print("\n[TOP COMPONENTS] ranked by mean(|mu_n|) over dataset\n")
    print("\n".join(lines))
    print(f"\n[SAVED] {out_txt}")

    # layer distribution in top-k
    layer_counts = {}
    for r in range(topk):
        l = component_layer(order[r], n_layers, n_heads)
        layer_counts[l] = layer_counts.get(l, 0) + 1
    print("\n[LAYER DISTRIBUTION] among top-k:")
    print(dict(sorted(layer_counts.items(), key=lambda x: x[0])))

    print("\n[SPARSITY & TYPE SPLIT]")
    print(f"  components needed for 80% of total |mu|: {n80} / {head_total + n_layers}")
    print(f"  attention share of total |mu|: {attn_share:.3f}")
    print(f"  mlp share of total |mu|:      {mlp_share:.3f}")

    summary = {
        "N": int(N),
        "seq_len": int(max_len),
        "n_layers": int(n_layers),
        "n_heads": int(n_heads),
        "attn_share": attn_share,
        "mlp_share": mlp_share,
        "n80": int(n80) if n80 is not None else None,
        "receptors": [s.__dict__ for s in receptor_specs],
        "topk": [],
    }
    for r in range(topk):
        idx = order[r]
        is_attn = component_is_attn(idx, n_layers, n_heads)
        l = component_layer(idx, n_layers, n_heads)
        h = (idx % n_heads) if is_attn else None
        summary["topk"].append(
            {
                "rank": r+1,
                "index": int(idx),
                "name": component_name(idx,n_layers,n_heads),
                "kind": "attn" if is_attn else "mlp",
                "layer": int(l),
                "head": int(h) if h is not None else None,
                "mean_abs_mu": float(mean_abs_mu[idx].item()),
                "mean_mu": float(mean_mu[idx].item()),
                "mean_mu_k": [float(x) for x in mean_mu_k[idx].tolist()],
                "rcs": float(compute_rcs(mean_abs_mu_k[idx])),
                "cum_share": float(cum_share[r]),
            }
        )

    out_json = os.path.join(args.out_dir, f"{args.save_prefix}_summary.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[SAVED] {out_json}")

    # Optional visualization of attention patterns for top heads
    viz_heads = int(args.viz_heads)
    viz_examples = int(args.viz_examples)
    if viz_heads > 0:
        top_head_indices = [idx for idx in order if idx < head_total][:viz_heads]
        if top_head_indices:
            # choose some examples (balanced)
            male = [i for i, lab in enumerate(labels) if lab == "he"]
            fem = [i for i, lab in enumerate(labels) if lab == "she"]
            chosen = []
            while len(chosen) < viz_examples and (male or fem):
                if male:
                    chosen.append(male.pop(0))
                    if len(chosen) >= viz_examples: break
                if fem:
                    chosen.append(fem.pop(0))
                    if len(chosen) >= viz_examples: break
            if not chosen:
                chosen = list(range(min(viz_examples, N)))

            toks_ex = tokens[chosen].to(device)
            last_ex = last_idx[chosen].to(device)
            texts_ex = [texts[i] for i in chosen]

            for idx in top_head_indices:
                l = idx // n_heads
                h = idx % n_heads
                key = f"blocks.{l}.attn.hook_pattern"

                # cache just this pattern
                try:
                    logits, cache = model.run_with_cache(
                        toks_ex,
                        names_filter=lambda name: name == key,
                        remove_batch_dim=False,
                    )
                except TypeError:
                    # older TransformerLens versions don't accept remove_batch_dim
                    logits, cache = model.run_with_cache(
                        toks_ex,
                        names_filter=lambda name: name == key,
                    )
                if key not in cache:
                    pattern_keys = [k for k in cache.keys() if "pattern" in str(k)]
                    raise KeyError(f"Cache missing '{key}'. Available pattern keys: {pattern_keys[:20]}")
                pat = cache[key]
                # standardize to (B,H,Q,K)
                if pat.ndim == 4 and pat.shape[1] == n_heads:
                    pat_bhqk = pat
                elif pat.ndim == 4 and pat.shape[-1] == n_heads:
                    pat_bhqk = pat.permute(0, 3, 1, 2)
                else:
                    raise ValueError(f"Unexpected pattern shape {tuple(pat.shape)} for key {key}")

                rows_txt = []
                for bi in range(pat_bhqk.shape[0]):
                    q = int(last_ex[bi].item())
                    attn_row = pat_bhqk[bi, h, q, : q+1].detach().cpu()
                    tok_ids = toks_ex[bi, : q+1].detach().cpu().tolist()
                    tok_str = token_strs(tokenizer, tok_ids)
                    topv, topi = torch.topk(attn_row, k=min(8, attn_row.numel()))
                    rows_txt.append(
                        {
                            "example_index": int(chosen[bi]),
                            "label": labels[chosen[bi]],
                            "query_pos": q,
                            "top_keys": [
                                {"pos": int(ii.item()), "token": tok_str[int(ii.item())], "weight": float(vv.item())}
                                for vv, ii in zip(topv, topi)
                            ],
                            "text": texts_ex[bi],
                        }
                    )

                    fig = plt.figure(figsize=(min(14, 0.5*len(tok_str)+2), 3.6))
                    xs = list(range(len(tok_str)))
                    plt.bar(xs, attn_row.numpy())
                    plt.xticks(xs, [t.replace("\n","\\n") for t in tok_str], rotation=70, ha="right")
                    plt.title(f"Attention from decision pos (q={q}) — Head L{l}H{h} — label={labels[chosen[bi]]}")
                    plt.tight_layout()
                    out_png = os.path.join(args.out_dir, f"{args.save_prefix}_head_L{l}H{h}_ex{bi}_pattern.png")
                    plt.savefig(out_png, dpi=160)
                    plt.close(fig)

                out_json2 = os.path.join(args.out_dir, f"{args.save_prefix}_head_L{l}H{h}_topkeys.json")
                with open(out_json2, "w") as f:
                    json.dump(rows_txt, f, indent=2)

                print(f"\n[VIZ] Head L{l}H{h}: saved bar plots + top-keys JSON")
                print(f"      {out_json2}")
                print(f"      PNGs: {args.out_dir}/{args.save_prefix}_head_L{l}H{h}_ex*_pattern.png")

    print("\n[DONE] Exp0b complete.")


if __name__ == "__main__":
    main()