import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch
import matplotlib.pyplot as plt

# Make repo root importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from experiments.train import load_config  # same loader as training
from src.data.data_loader import load_ioi_dataset, load_gp_dataset, load_gt_dataset
from src.models.masked_transformer_circuit import MaskedTransformerCircuit
from src.utils.utils import get_data_column_names, get_label_column_names
from src.utils.receptor_fisher_geometry import (
    make_logit_receptors,
    apply_D,
)


# ---------------------------
# Small utilities
# ---------------------------

def _now_ts() -> str:
    return time.strftime("%Y%m%d_%H%M%S", time.localtime())

def _ensure_leading_space(s: str) -> str:
    if s is None:
        return ""
    return s if s.startswith(" ") else (" " + s)

def _set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def _torch_dtype_from_str(s: str) -> torch.dtype:
    s = s.lower()
    if s in ["float64", "fp64", "double"]:
        return torch.float64
    if s in ["float32", "fp32", "float"]:
        return torch.float32
    raise ValueError(f"Unsupported dtype: {s}")

@dataclass
class RunningStats:
    n: int = 0
    s: float = 0.0
    s2: float = 0.0

    def update(self, x: torch.Tensor) -> None:
        # x can be shape [B] or scalar
        x = x.detach()
        self.n += x.numel()
        self.s += x.sum().item()
        self.s2 += (x * x).sum().item()

    def mean(self) -> float:
        return self.s / max(self.n, 1)

    def std(self) -> float:
        if self.n <= 1:
            return 0.0
        m = self.mean()
        var = max(self.s2 / self.n - m * m, 0.0)
        return math.sqrt(var)

def _json_save(path: str, obj: Any) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def _pt_save(path: str, obj: Any) -> None:
    torch.save(obj, path)

def _get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

def _build_raw_dataloader(config: Dict[str, Any]):
    data_type = config["data"]["type"]
    batch_size = config["training"]["batch_size"]
    seed = config["data"].get("seed", 0)

    if data_type == "ioi":
        dataset = load_ioi_dataset(seed=seed)
    elif data_type == "gp":
        dataset = load_gp_dataset(seed=seed)
    elif data_type == "gt":
        dataset = load_gt_dataset(seed=seed)
    else:
        raise ValueError(f"Unknown data_type: {data_type}")

    # dataset already returns dict-like items; torch DataLoader default collate -> dict of lists
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

def _infer_d_model_from_unembed(W_U_raw: torch.Tensor) -> int:
    if W_U_raw.ndim != 2:
        raise ValueError(f"Expected W_U_raw to be rank-2, got shape={tuple(W_U_raw.shape)}")
    # Either [vocab, d_model] or [d_model, vocab]
    if W_U_raw.shape[0] < W_U_raw.shape[1]:
        # could be [d_model, vocab] but not guaranteed; use explicit checks below
        pass
    # just use robust checks:
    # if shape is [vocab, d_model], d_model = dim1
    # if shape is [d_model, vocab], d_model = dim0
    # We'll decide by comparing against typical vocab sizes? No. Decide by matching with V rows later.
    # For now used only for augmented drop decision in the script, so we handle both:
    return W_U_raw.shape[1]  # default assumption (vocab, d_model) or (d_model, vocab) will be handled later

def _tokenize_texts(model, texts: List[str], device: torch.device):
    tok = model.tokenizer(
        texts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    input_ids = tok["input_ids"].to(device)
    attention_mask = tok["attention_mask"].to(device)
    lengths = attention_mask.sum(dim=1)
    last_idx = (lengths - 1).clamp_min(0).to(torch.long)
    return input_ids, attention_mask, last_idx

def _tokenize_labels_one_token(model, labels: List[str], device: torch.device) -> torch.Tensor:
    labels_sp = [_ensure_leading_space(x) for x in labels]
    tok = model.tokenizer(
        labels_sp,
        return_tensors="pt",
        padding=True,
        truncation=False,
        add_special_tokens=False,
    )
    ids = tok["input_ids"].to(device)
    # We assume first token is the label token (repo datasets are usually curated this way)
    return ids[:, 0].to(torch.long)

def _select_logits_at_positions(logits: torch.Tensor, pos_idx: torch.Tensor) -> torch.Tensor:
    # logits: [B, seq, vocab], pos_idx: [B]
    b = torch.arange(logits.shape[0], device=logits.device)
    return logits[b, pos_idx, :]

def _score_logit_diff(logits_last: torch.Tensor, tok_correct: torch.Tensor, tok_wrong: torch.Tensor) -> torch.Tensor:
    # logits_last: [B, vocab], tok_*: [B]
    b = torch.arange(logits_last.shape[0], device=logits_last.device)
    lc = logits_last[b, tok_correct]
    lw = logits_last[b, tok_wrong]
    return lc - lw

def _abs_offdiag_top_pairs(pairs: List[Dict[str, Any]], num_pairs: int) -> List[Dict[str, Any]]:
    return pairs[: min(num_pairs, len(pairs))]

def _group_pairs_by_i(pairs: List[Dict[str, Any]]) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {}
    for p in pairs:
        i = int(p["i"])
        j = int(p["j"])
        out.setdefault(i, []).append(j)
    return out

def _fisher_inner_abs_batch(delta: torch.Tensor, R: torch.Tensor, p: torch.Tensor) -> torch.Tensor:
    """
    delta: [B, vocab]
    R:     [vocab, m]
    p:     [B, vocab] softmax
    Returns abs(<delta, r_j>_F) for all j: [B, m]
    <a,b>_F = sum_k p_k a_k b_k - (sum_k p_k a_k)(sum_k p_k b_k)
    """
    # term1_j = sum_k p_k * delta_k * R_kj
    term1 = (p * delta) @ R  # [B, m]
    # s_delta = sum_k p_k * delta_k
    s_delta = (p * delta).sum(dim=1, keepdim=True)  # [B, 1]
    # s_r_j = sum_k p_k * R_kj
    s_r = p @ R  # [B, m]
    inner = term1 - s_delta * s_r
    return inner.abs()

def _leakage_from_c(c_abs: torch.Tensor, i_idx: int, partners: List[int], eps: float = 1e-12):
    """
    c_abs: [B, m] where c_abs[:, j] = abs(<delta, r_j>) or abs(<delta, r_j>_F)
    returns:
      mean_other: [B]
      max_other:  [B]
      partner_map: {j: [B]}
    """
    B, m = c_abs.shape
    ci = c_abs[:, i_idx].clamp_min(eps)  # [B]
    sum_all = c_abs.sum(dim=1)  # [B]
    mean_other = (sum_all - c_abs[:, i_idx]) / max(m - 1, 1)
    mean_other = mean_other / ci

    # max over j != i
    c_tmp = c_abs.clone()
    c_tmp[:, i_idx] = -1.0
    max_other = c_tmp.max(dim=1).values.clamp_min(0.0) / ci

    partner_map = {}
    for j in partners:
        partner_map[int(j)] = (c_abs[:, int(j)] / ci)
    return mean_other, max_other, partner_map

def _save_heatmap(mat: torch.Tensor, title: str, path: str):
    mat = mat.detach().cpu()
    plt.figure(figsize=(9, 8))
    plt.imshow(mat, aspect="auto")
    plt.colorbar()
    plt.title(title)
    plt.xlabel("j")
    plt.ylabel("i")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def _save_bar_compare(labels: List[str], raw_vals: List[float], dec_vals: List[float], title: str, path: str):
    x = list(range(len(labels)))
    plt.figure(figsize=(10, 4))
    plt.bar([i - 0.2 for i in x], raw_vals, width=0.4, label="raw")
    plt.bar([i + 0.2 for i in x], dec_vals, width=0.4, label="dec")
    plt.xticks(x, labels, rotation=20, ha="right")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


# ---------------------------
# Patch builders
# ---------------------------

def _make_patch_hook_variant_A(
    head: int,
    last_idx_corr: torch.Tensor,   # [B]
    diff_scalar: torch.Tensor,     # [B] where diff = (s_clean - s_corr)
    v_dir: torch.Tensor,           # [d_model]
):
    """
    Patch:
      y_patched = y_corr + (s_clean - s_corr) * v
    implemented as: act[...] += diff * v
    Hook point: blocks.{layer}.attn.hook_result  shape [B, seq, n_heads, d_model]
    """
    v = v_dir

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:
        # act: [B, seq, H, d_model]
        B = act.shape[0]
        act = act.clone()
        b = torch.arange(B, device=act.device)
        act[b, last_idx_corr, head, :] += diff_scalar[:, None] * v[None, :]
        return act

    return hook_fn

def _make_patch_hook_variant_B(
    head: int,
    last_idx_corr: torch.Tensor,     # [B]
    delta_alpha: torch.Tensor,       # [B] (alpha_clean - alpha_corr)
    v_dir: torch.Tensor,             # [d_model] (raw v_i)
):
    """
    Patch:
      y_patched = y_corr + (alpha_clean - alpha_corr) * v
    """
    v = v_dir

    def hook_fn(act: torch.Tensor, hook) -> torch.Tensor:
        B = act.shape[0]
        act = act.clone()
        b = torch.arange(B, device=act.device)
        act[b, last_idx_corr, head, :] += delta_alpha[:, None] * v[None, :]
        return act

    return hook_fn


# ---------------------------
# Main
# ---------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    ap.add_argument("--checkpoint", type=str, default=None)

    ap.add_argument("--exp1_run_dir", type=str, required=True)
    ap.add_argument("--pairs_source", type=str, choices=["before", "after"], default="before")
    ap.add_argument("--num_pairs", type=int, default=30)

    ap.add_argument("--position", type=str, default="last")
    ap.add_argument("--max_batches", type=int, default=30)
    ap.add_argument("--ridge", type=float, default=1e-3)  # not used directly here, but kept for parity

    ap.add_argument("--out_dir", type=str, default="logs/receptor_fisher_causal_selectivity")
    ap.add_argument("--dtype", type=str, default="float64")
    ap.add_argument("--patch_variant", type=str, choices=["A", "B", "both"], default="both")
    ap.add_argument("--leakage_metric", type=str, choices=["1", "2", "both"], default="both")
    ap.add_argument("--seed", type=int, default=0)

    args = ap.parse_args()

    if args.position != "last":
        raise ValueError("This script currently supports --position last only (matches your spec defaults).")

    _set_seed(args.seed)
    device = _get_device()
    geom_dtype = _torch_dtype_from_str(args.dtype)

    # ----- Load config + data -----
    config = load_config(args.config)
    data_type = config["data"]["type"]
    raw_loader = _build_raw_dataloader(config)

    # ----- Build model (same as training style) -----
    circuit = MaskedTransformerCircuit(config)
    circuit.to(device)
    circuit.eval()

    # If checkpoint is provided, load like other scripts do
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device)
        # common patterns: state_dict directly or nested
        if isinstance(ckpt, dict) and "model_state_dict" in ckpt:
            circuit.load_state_dict(ckpt["model_state_dict"], strict=False)
        elif isinstance(ckpt, dict):
            circuit.load_state_dict(ckpt, strict=False)
        else:
            raise ValueError("Unsupported checkpoint format")

    model = circuit.model  # underlying TransformerLens HookedTransformer
    model.to(device)
    model.eval()

    # Ensure pad token exists
    if model.tokenizer.pad_token is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    # ----- Load Exp1 artifacts -----
    exp1_dir = args.exp1_run_dir
    D = torch.load(os.path.join(exp1_dir, "D.pt"), map_location=device)
    V_write_global = torch.load(os.path.join(exp1_dir, "V_write_global.pt"), map_location=device)

    with open(os.path.join(exp1_dir, "metadata.json"), "r", encoding="utf-8") as f:
        metadata: List[Dict[str, Any]] = json.load(f)

    with open(os.path.join(exp1_dir, f"top_pairs_{args.pairs_source}.json"), "r", encoding="utf-8") as f:
        pairs_all = json.load(f)

    pairs = _abs_offdiag_top_pairs(pairs_all, args.num_pairs)
    pairs_by_i = _group_pairs_by_i(pairs)
    unique_i = sorted(pairs_by_i.keys())

    # ----- Build receptor bases -----
    W_U_raw = circuit.get_unembed_weight().detach().to(device)

    # Shape guard: if augmented d_model+1, drop last row before projecting into logits
    # Determine d_model expected by unembedding:
    # - W_U_raw: [d_model, vocab] (TransformerLens) OR [vocab, d_model]
    if W_U_raw.ndim != 2:
        raise ValueError(f"W_U_raw must be rank-2, got {tuple(W_U_raw.shape)}")
    if W_U_raw.shape[0] < W_U_raw.shape[1]:
        # could be [d_model, vocab] or [vocab, d_model]; decide by matching V rows
        pass

    # Decide d_model by checking which dimension matches V_write_global rows or rows-1
    V_rows = V_write_global.shape[0]
    candidates = []
    # candidate A: W_U_raw is [d_model, vocab]
    candidates.append(W_U_raw.shape[0])
    # candidate B: W_U_raw is [vocab, d_model]
    candidates.append(W_U_raw.shape[1])
    # choose one that matches V_rows or V_rows-1
    d_model = None
    for c in candidates:
        if V_rows == c or V_rows == c + 1:
            d_model = c
            break
    if d_model is None:
        raise ValueError(
            f"Could not reconcile V_write_global rows={V_rows} with W_U_raw shape={tuple(W_U_raw.shape)}"
        )

    V_raw = V_write_global
    if V_raw.shape[0] == d_model + 1:
        V_raw = V_raw[:d_model, :]

    # Make sure indices exist
    m = V_raw.shape[1]
    for i in unique_i:
        if not (0 <= i < m):
            raise ValueError(f"Receptor index {i} out of range for m={m}")

    # Deconfounded basis
    V_dec = apply_D(V_raw, D, renorm_cols=True)

    # For patching we keep directions float32 (logits are float32 anyway)
    V_raw_f32 = V_raw.detach().to(device=device, dtype=torch.float32)
    V_dec_f32 = V_dec.detach().to(device=device, dtype=torch.float32)

    # Logit receptors
    # (These can be big; keep them float32 unless you explicitly want float64)
    R_raw = make_logit_receptors(V_raw_f32, W_U_raw).to(device=device, dtype=torch.float32)  # [vocab, m]
    R_dec = make_logit_receptors(V_dec_f32, W_U_raw).to(device=device, dtype=torch.float32)  # [vocab, m]

    print(f"[Sanity] W_U_raw shape: {tuple(W_U_raw.shape)}")
    print(f"[Sanity] V_raw shape:   {tuple(V_raw_f32.shape)}  (m={m})")
    print(f"[Sanity] R_raw shape:   {tuple(R_raw.shape)}")

    # Needed layers for caching hook_result/pattern/ln1
    layers_needed = sorted({int(metadata[i]["layer"]) for i in unique_i})
    hook_result_names = [f"blocks.{l}.attn.hook_result" for l in layers_needed]
    hook_pattern_names = [f"blocks.{l}.attn.hook_pattern" for l in layers_needed]
    hook_ln1_names = [f"blocks.{l}.ln1.hook_normalized" for l in layers_needed]

    do_A = args.patch_variant in ["A", "both"]
    do_B = args.patch_variant in ["B", "both"]
    do_leak1 = args.leakage_metric in ["1", "both"]
    do_leak2 = args.leakage_metric in ["2", "both"]

    # Variant B needs extra caches + OV SVD cache
    if do_B:
        # Ensure SVD cache is populated
        if not hasattr(circuit, "svd_cache") or len(getattr(circuit, "svd_cache", {})) == 0:
            if hasattr(circuit, "_load_or_compute_svd"):
                circuit._load_or_compute_svd()
            else:
                raise RuntimeError("Variant B requested but circuit has no SVD cache loader.")

    # Output dir
    run_name = f"{data_type}_pairs{len(pairs)}_uniq{len(unique_i)}_{_now_ts()}"
    out_run_dir = os.path.join(args.out_dir, run_name)
    os.makedirs(out_run_dir, exist_ok=True)

    # Save evaluated indices
    _pt_save(os.path.join(out_run_dir, "indices.pt"), torch.tensor(unique_i, dtype=torch.long))

    # Stats containers
    # We mainly report raw(A) vs dec(A). Raw(B) optionally.
    results: Dict[str, Dict[str, Dict[str, RunningStats]]] = {
        "raw": {},
        "dec": {},
    }

    def _init_bucket(basis: str, variant: str):
        if variant not in results[basis]:
            results[basis][variant] = {
                "recovery": RunningStats(),
                "leak1_mean_other": RunningStats(),
                "leak1_max_other": RunningStats(),
                "leak1_partner": RunningStats(),
                "leak2_mean_other": RunningStats(),
                "leak2_max_other": RunningStats(),
                "leak2_partner": RunningStats(),
            }

    if do_A:
        _init_bucket("raw", "A")
        _init_bucket("dec", "A")
    if do_B:
        _init_bucket("raw", "B")

    # Per-pair stats
    per_pair_stats: Dict[str, Dict[str, Dict[str, RunningStats]]] = {}
    def _pair_key(i: int, j: int) -> str:
        return f"{i}_{j}"

    for p in pairs:
        i = int(p["i"]); j = int(p["j"])
        k = _pair_key(i, j)
        per_pair_stats[k] = {"raw": {}, "dec": {}}
        if do_A:
            per_pair_stats[k]["raw"]["A"] = {
                "recovery": RunningStats(),
                "leak1_partner": RunningStats(),
                "leak2_partner": RunningStats(),
            }
            per_pair_stats[k]["dec"]["A"] = {
                "recovery": RunningStats(),
                "leak1_partner": RunningStats(),
                "leak2_partner": RunningStats(),
            }
        if do_B:
            per_pair_stats[k]["raw"]["B"] = {
                "recovery": RunningStats(),
                "leak1_partner": RunningStats(),
                "leak2_partner": RunningStats(),
            }

    # Heatmap subsets (option 2 only, variant A only)
    subset = unique_i[: min(64, len(unique_i))]
    subset_set = set(subset)
    idx_to_row = {idx: r for r, idx in enumerate(subset)}
    heat_raw_num = torch.zeros((len(subset), len(subset)), dtype=torch.float64)
    heat_raw_den = torch.zeros((len(subset),), dtype=torch.float64)
    heat_dec_num = torch.zeros((len(subset), len(subset)), dtype=torch.float64)
    heat_dec_den = torch.zeros((len(subset),), dtype=torch.float64)

    # ----- Main evaluation loop -----
    eps = 1e-12
    total_examples = 0
    batches_done = 0

    clean_col, corr_col = get_data_column_names(data_type)
    label_col, wrong_col = get_label_column_names(data_type)

    if data_type == "gt":
        print("[Warning] GT in this repo is often evaluated by exact-match; this script uses token logit-diff for recovery/leakage.")

    for batch_idx, batch in enumerate(raw_loader):
        if args.max_batches and batch_idx >= args.max_batches:
            break
        batches_done += 1

        # Prepare texts
        clean_texts = list(batch[clean_col])
        corr_texts = list(batch[corr_col])

        # Labels: use CLEAN labels for clean/corr/patched (recovery-style measurement)
        labels = list(batch[label_col])
        wrongs = list(batch[wrong_col])

        # Tokenize
        input_ids_clean, attn_mask_clean, last_idx_clean = _tokenize_texts(model, clean_texts, device)
        input_ids_corr, attn_mask_corr, last_idx_corr = _tokenize_texts(model, corr_texts, device)
        tok_correct = _tokenize_labels_one_token(model, labels, device)
        tok_wrong = _tokenize_labels_one_token(model, wrongs, device)

        B = input_ids_corr.shape[0]
        total_examples += B

        # Cache clean + corrupt at required hooks
        hooks_to_cache = set(hook_result_names)
        if do_B:
            hooks_to_cache.update(hook_pattern_names)
            hooks_to_cache.update(hook_ln1_names)

        def names_filter(name: str) -> bool:
            return name in hooks_to_cache

        with torch.no_grad():
            logits_clean, cache_clean = model.run_with_cache(
                input_ids_clean, attention_mask=attn_mask_clean, names_filter=names_filter
            )
            logits_corr, cache_corr = model.run_with_cache(
                input_ids_corr, attention_mask=attn_mask_corr, names_filter=names_filter
            )

        logits_clean_last = _select_logits_at_positions(logits_clean, last_idx_clean)
        logits_corr_last = _select_logits_at_positions(logits_corr, last_idx_corr)

        score_clean = _score_logit_diff(logits_clean_last, tok_correct, tok_wrong)
        score_corr = _score_logit_diff(logits_corr_last, tok_correct, tok_wrong)

        denom = (score_clean - score_corr).clamp_min(eps)

        # For each patched receptor i
        for i in unique_i:
            meta_i = metadata[i]
            layer = int(meta_i["layer"])
            head = int(meta_i["head"])
            sv_idx = int(meta_i["sv_idx"])
            partners = pairs_by_i[i]

            hook_name = f"blocks.{layer}.attn.hook_result"

            # Extract y_clean/y_corr at the site (variant A uses these)
            y_clean_all = cache_clean[hook_name]  # [B, seq, H, d_model]
            y_corr_all = cache_corr[hook_name]

            b_ar = torch.arange(B, device=device)
            y_clean = y_clean_all[b_ar, last_idx_clean, head, :]  # [B, d_model]
            y_corr = y_corr_all[b_ar, last_idx_corr, head, :]     # [B, d_model]

            # -------- Variant A: RAW basis --------
            if do_A:
                v_raw_i = V_raw_f32[:, i]  # [d_model]
                s_clean_raw = (y_clean * v_raw_i[None, :]).sum(dim=1)  # [B]
                s_corr_raw = (y_corr * v_raw_i[None, :]).sum(dim=1)    # [B]
                diff_raw = (s_clean_raw - s_corr_raw)  # [B]

                patch_fn_rawA = _make_patch_hook_variant_A(head, last_idx_corr, diff_raw, v_raw_i)

                with torch.no_grad():
                    logits_patched = model.run_with_hooks(
                        input_ids_corr,
                        attention_mask=attn_mask_corr,
                        fwd_hooks=[(hook_name, patch_fn_rawA)],
                    )
                logits_patched_last = _select_logits_at_positions(logits_patched, last_idx_corr)
                score_patched = _score_logit_diff(logits_patched_last, tok_correct, tok_wrong)
                recovery = (score_patched - score_corr) / denom

                results["raw"]["A"]["recovery"].update(recovery)

                delta_logits = (logits_patched_last - logits_corr_last)  # [B, vocab]

                # Leakage option 1 (Euclidean projection on receptor logits)
                if do_leak1:
                    c_abs = (delta_logits @ R_raw).abs()  # [B, m]
                    mean_other, max_other, partner_map = _leakage_from_c(c_abs, i, partners, eps=eps)
                    results["raw"]["A"]["leak1_mean_other"].update(mean_other)
                    results["raw"]["A"]["leak1_max_other"].update(max_other)
                    # partner: average over partners for the global summary
                    partner_vals = torch.stack([partner_map[int(j)] for j in partners], dim=1).mean(dim=1)
                    results["raw"]["A"]["leak1_partner"].update(partner_vals)

                # Leakage option 2 (Fisher inner product at corrupted p)
                if do_leak2:
                    p_corr = torch.softmax(logits_corr_last, dim=1)  # [B, vocab]
                    fisher_abs = _fisher_inner_abs_batch(delta_logits, R_raw, p_corr)  # [B, m]
                    mean_other2, max_other2, partner_map2 = _leakage_from_c(fisher_abs, i, partners, eps=eps)
                    results["raw"]["A"]["leak2_mean_other"].update(mean_other2)
                    results["raw"]["A"]["leak2_max_other"].update(max_other2)
                    partner_vals2 = torch.stack([partner_map2[int(j)] for j in partners], dim=1).mean(dim=1)
                    results["raw"]["A"]["leak2_partner"].update(partner_vals2)

                    # Heatmap accumulation (subset only)
                    if i in subset_set:
                        row = idx_to_row[i]
                        # accumulate numerator for subset columns
                        fisher_sub = fisher_abs[:, subset].double()  # [B, |subset|]
                        heat_raw_num[row, :] += fisher_sub.sum(dim=0).cpu()
                        heat_raw_den[row] += fisher_abs[:, i].double().sum().cpu()

                # Per-pair stats for raw(A)
                for j in partners:
                    k = _pair_key(i, j)
                    per_pair_stats[k]["raw"]["A"]["recovery"].update(recovery)
                    if do_leak1:
                        per_pair_stats[k]["raw"]["A"]["leak1_partner"].update(partner_map[int(j)])
                    if do_leak2:
                        per_pair_stats[k]["raw"]["A"]["leak2_partner"].update(partner_map2[int(j)])

            # -------- Variant A: DEC basis --------
            if do_A:
                v_dec_i = V_dec_f32[:, i]  # [d_model]
                s_clean_dec = (y_clean * v_dec_i[None, :]).sum(dim=1)
                s_corr_dec = (y_corr * v_dec_i[None, :]).sum(dim=1)
                diff_dec = (s_clean_dec - s_corr_dec)

                patch_fn_decA = _make_patch_hook_variant_A(head, last_idx_corr, diff_dec, v_dec_i)

                with torch.no_grad():
                    logits_patched_dec = model.run_with_hooks(
                        input_ids_corr,
                        attention_mask=attn_mask_corr,
                        fwd_hooks=[(hook_name, patch_fn_decA)],
                    )

                logits_patched_dec_last = _select_logits_at_positions(logits_patched_dec, last_idx_corr)
                score_patched_dec = _score_logit_diff(logits_patched_dec_last, tok_correct, tok_wrong)
                recovery_dec = (score_patched_dec - score_corr) / denom

                results["dec"]["A"]["recovery"].update(recovery_dec)

                delta_logits_dec = (logits_patched_dec_last - logits_corr_last)  # [B, vocab]

                if do_leak1:
                    c_abs_dec = (delta_logits_dec @ R_dec).abs()  # [B, m]
                    mean_other_d, max_other_d, partner_map_d = _leakage_from_c(c_abs_dec, i, partners, eps=eps)
                    results["dec"]["A"]["leak1_mean_other"].update(mean_other_d)
                    results["dec"]["A"]["leak1_max_other"].update(max_other_d)
                    partner_vals_d = torch.stack([partner_map_d[int(j)] for j in partners], dim=1).mean(dim=1)
                    results["dec"]["A"]["leak1_partner"].update(partner_vals_d)

                if do_leak2:
                    p_corr = torch.softmax(logits_corr_last, dim=1)
                    fisher_abs_dec = _fisher_inner_abs_batch(delta_logits_dec, R_dec, p_corr)  # [B, m]
                    mean_other2_d, max_other2_d, partner_map2_d = _leakage_from_c(fisher_abs_dec, i, partners, eps=eps)
                    results["dec"]["A"]["leak2_mean_other"].update(mean_other2_d)
                    results["dec"]["A"]["leak2_max_other"].update(max_other2_d)
                    partner_vals2_d = torch.stack([partner_map2_d[int(j)] for j in partners], dim=1).mean(dim=1)
                    results["dec"]["A"]["leak2_partner"].update(partner_vals2_d)

                    if i in subset_set:
                        row = idx_to_row[i]
                        fisher_sub = fisher_abs_dec[:, subset].double()
                        heat_dec_num[row, :] += fisher_sub.sum(dim=0).cpu()
                        heat_dec_den[row] += fisher_abs_dec[:, i].double().sum().cpu()

                # Per-pair stats for dec(A)
                for j in partners:
                    k = _pair_key(i, j)
                    per_pair_stats[k]["dec"]["A"]["recovery"].update(recovery_dec)
                    if do_leak1:
                        per_pair_stats[k]["dec"]["A"]["leak1_partner"].update(partner_map_d[int(j)])
                    if do_leak2:
                        per_pair_stats[k]["dec"]["A"]["leak2_partner"].update(partner_map2_d[int(j)])

            # -------- Variant B: RAW basis only --------
            if do_B:
                # This uses OV SVD coefficient alpha_i = sigma_i * <x, u_i>
                # where x is the augmented OV input [1, context_standard] at the site.
                meta = metadata[i]
                layer = int(meta["layer"])
                head = int(meta["head"])
                sv_idx = int(meta["sv_idx"])

                head_key = f"differential_head_{layer}_{head}"
                ov_cache_key = f"{head_key}_ov"
                if ov_cache_key not in circuit.svd_cache:
                    circuit._load_or_compute_svd()
                U_ov, S_ov, Vh_ov, W_OV_orig = circuit.svd_cache[ov_cache_key]

                # u_i (augmented dim), sigma_i
                u_i = U_ov[:, sv_idx].detach().to(device=device, dtype=torch.float32)  # [d_model+1]
                sigma_i = float(S_ov[sv_idx].detach().cpu().item())

                # Compute x_clean and x_corr from caches:
                # context_standard = sum_k attn_w[q=last, k] * attn_in[k]
                # then augment with ones.
                hook_pat = f"blocks.{layer}.attn.hook_pattern"
                hook_ln1 = f"blocks.{layer}.ln1.hook_normalized"
                attn_w_clean = cache_clean[hook_pat]   # [B, H, Q, K]
                attn_w_corr = cache_corr[hook_pat]
                attn_in_clean = cache_clean[hook_ln1]  # [B, seq, d_model]
                attn_in_corr = cache_corr[hook_ln1]

                w_clean = attn_w_clean[b_ar, head, last_idx_clean, :]  # [B, seq]
                w_corr = attn_w_corr[b_ar, head, last_idx_corr, :]     # [B, seq]

                ctx_clean = (w_clean.unsqueeze(-1) * attn_in_clean).sum(dim=1)  # [B, d_model]
                ctx_corr = (w_corr.unsqueeze(-1) * attn_in_corr).sum(dim=1)     # [B, d_model]

                ones = torch.ones((B, 1), device=device, dtype=torch.float32)
                x_clean = torch.cat([ones, ctx_clean], dim=1)  # [B, d_model+1]
                x_corr = torch.cat([ones, ctx_corr], dim=1)    # [B, d_model+1]

                # alpha = sigma * <x, u>
                alpha_clean = sigma_i * (x_clean * u_i[None, :]).sum(dim=1)
                alpha_corr = sigma_i * (x_corr * u_i[None, :]).sum(dim=1)
                delta_alpha = (alpha_clean - alpha_corr)  # [B]

                v_raw_i = V_raw_f32[:, i]
                patch_fn_rawB = _make_patch_hook_variant_B(head, last_idx_corr, delta_alpha, v_raw_i)

                hook_name = f"blocks.{layer}.attn.hook_result"
                with torch.no_grad():
                    logits_patched_B = model.run_with_hooks(
                        input_ids_corr,
                        attention_mask=attn_mask_corr,
                        fwd_hooks=[(hook_name, patch_fn_rawB)],
                    )

                logits_patched_B_last = _select_logits_at_positions(logits_patched_B, last_idx_corr)
                score_patched_B = _score_logit_diff(logits_patched_B_last, tok_correct, tok_wrong)
                recovery_B = (score_patched_B - score_corr) / denom
                results["raw"]["B"]["recovery"].update(recovery_B)

                delta_logits_B = (logits_patched_B_last - logits_corr_last)

                if do_leak1:
                    c_abs_B = (delta_logits_B @ R_raw).abs()
                    mean_otherB, max_otherB, partner_mapB = _leakage_from_c(c_abs_B, i, partners, eps=eps)
                    results["raw"]["B"]["leak1_mean_other"].update(mean_otherB)
                    results["raw"]["B"]["leak1_max_other"].update(max_otherB)
                    partner_valsB = torch.stack([partner_mapB[int(j)] for j in partners], dim=1).mean(dim=1)
                    results["raw"]["B"]["leak1_partner"].update(partner_valsB)

                if do_leak2:
                    p_corr = torch.softmax(logits_corr_last, dim=1)
                    fisher_abs_B = _fisher_inner_abs_batch(delta_logits_B, R_raw, p_corr)
                    mean_other2B, max_other2B, partner_map2B = _leakage_from_c(fisher_abs_B, i, partners, eps=eps)
                    results["raw"]["B"]["leak2_mean_other"].update(mean_other2B)
                    results["raw"]["B"]["leak2_max_other"].update(max_other2B)
                    partner_vals2B = torch.stack([partner_map2B[int(j)] for j in partners], dim=1).mean(dim=1)
                    results["raw"]["B"]["leak2_partner"].update(partner_vals2B)

                # Per-pair stats for raw(B)
                for j in partners:
                    k = _pair_key(i, j)
                    per_pair_stats[k]["raw"]["B"]["recovery"].update(recovery_B)
                    if do_leak1:
                        per_pair_stats[k]["raw"]["B"]["leak1_partner"].update(partner_mapB[int(j)])
                    if do_leak2:
                        per_pair_stats[k]["raw"]["B"]["leak2_partner"].update(partner_map2B[int(j)])

        print(f"[Batch {batch_idx}] done (B={B}).")

    # ----- Finalize + save -----
    def _bucket_to_dict(bucket: Dict[str, RunningStats]) -> Dict[str, Dict[str, float]]:
        return {k: {"mean": v.mean(), "std": v.std(), "n": v.n} for k, v in bucket.items()}

    summary = {
        "args": vars(args),
        "data_type": data_type,
        "exp1_run_dir": exp1_dir,
        "pairs_source": args.pairs_source,
        "num_pairs": len(pairs),
        "unique_i": len(unique_i),
        "m": int(m),
        "batches_done": int(batches_done),
        "num_examples": int(total_examples),
        "results": {
            basis: {variant: _bucket_to_dict(stats) for variant, stats in variants.items()}
            for basis, variants in results.items()
        },
    }

    _json_save(os.path.join(out_run_dir, "metrics.json"), summary)

    # Per-pair json
    per_pair_out = []
    for p in pairs:
        i = int(p["i"]); j = int(p["j"])
        k = _pair_key(i, j)
        rec = {
            "i": i, "j": j,
            "meta_i": metadata[i],
            "meta_j": metadata[j],
            "pair_abs": float(p.get("abs", 0.0)),
            "pair_signed": float(p.get("signed", 0.0)),
            "raw": {},
            "dec": {},
        }
        if do_A:
            rec["raw"]["A"] = {
                "recovery_mean": per_pair_stats[k]["raw"]["A"]["recovery"].mean(),
                "recovery_std": per_pair_stats[k]["raw"]["A"]["recovery"].std(),
                "leak1_partner_mean": per_pair_stats[k]["raw"]["A"]["leak1_partner"].mean() if do_leak1 else None,
                "leak2_partner_mean": per_pair_stats[k]["raw"]["A"]["leak2_partner"].mean() if do_leak2 else None,
            }
            rec["dec"]["A"] = {
                "recovery_mean": per_pair_stats[k]["dec"]["A"]["recovery"].mean(),
                "recovery_std": per_pair_stats[k]["dec"]["A"]["recovery"].std(),
                "leak1_partner_mean": per_pair_stats[k]["dec"]["A"]["leak1_partner"].mean() if do_leak1 else None,
                "leak2_partner_mean": per_pair_stats[k]["dec"]["A"]["leak2_partner"].mean() if do_leak2 else None,
            }
        if do_B:
            rec["raw"]["B"] = {
                "recovery_mean": per_pair_stats[k]["raw"]["B"]["recovery"].mean(),
                "recovery_std": per_pair_stats[k]["raw"]["B"]["recovery"].std(),
                "leak1_partner_mean": per_pair_stats[k]["raw"]["B"]["leak1_partner"].mean() if do_leak1 else None,
                "leak2_partner_mean": per_pair_stats[k]["raw"]["B"]["leak2_partner"].mean() if do_leak2 else None,
            }
        per_pair_out.append(rec)

    _json_save(os.path.join(out_run_dir, "per_pair.json"), per_pair_out)

    # Heatmaps (Option2, Variant A)
    if do_A and do_leak2 and len(subset) > 0:
        raw_L = heat_raw_num / (heat_raw_den[:, None].clamp_min(1e-12))
        dec_L = heat_dec_num / (heat_dec_den[:, None].clamp_min(1e-12))
        _save_heatmap(raw_L, f"Leakage heatmap (raw, opt2), subset={len(subset)}", os.path.join(out_run_dir, "leak_heatmap_raw_opt2.png"))
        _save_heatmap(dec_L, f"Leakage heatmap (dec, opt2), subset={len(subset)}", os.path.join(out_run_dir, "leak_heatmap_dec_opt2.png"))

    # Bar chart summary (Variant A only, if present)
    if do_A:
        labels = ["recovery"]
        raw_vals = [results["raw"]["A"]["recovery"].mean()]
        dec_vals = [results["dec"]["A"]["recovery"].mean()]

        if do_leak1:
            labels += ["leak1_mean_other", "leak1_partner"]
            raw_vals += [results["raw"]["A"]["leak1_mean_other"].mean(), results["raw"]["A"]["leak1_partner"].mean()]
            dec_vals += [results["dec"]["A"]["leak1_mean_other"].mean(), results["dec"]["A"]["leak1_partner"].mean()]
        if do_leak2:
            labels += ["leak2_mean_other", "leak2_partner"]
            raw_vals += [results["raw"]["A"]["leak2_mean_other"].mean(), results["raw"]["A"]["leak2_partner"].mean()]
            dec_vals += [results["dec"]["A"]["leak2_mean_other"].mean(), results["dec"]["A"]["leak2_partner"].mean()]

        _save_bar_compare(
            labels, raw_vals, dec_vals,
            title=f"Causal selectivity summary (Variant A) | {data_type} | N={total_examples}",
            path=os.path.join(out_run_dir, "bar_compare_raw_vs_dec_variantA.png"),
        )

    # Stdout summary
    print("\n=== Experiment 2 Summary ===")
    print(f"run_dir: {out_run_dir}")
    print(f"data_type={data_type} | pairs={len(pairs)} | unique_i={len(unique_i)} | m={m} | N={total_examples}")
    if do_A:
        print(f"[A/raw] recovery mean={results['raw']['A']['recovery'].mean():.4f} std={results['raw']['A']['recovery'].std():.4f}")
        print(f"[A/dec] recovery mean={results['dec']['A']['recovery'].mean():.4f} std={results['dec']['A']['recovery'].std():.4f}")
        if do_leak1:
            print(f"[A/raw] leak1 mean_other={results['raw']['A']['leak1_mean_other'].mean():.4f} partner={results['raw']['A']['leak1_partner'].mean():.4f}")
            print(f"[A/dec] leak1 mean_other={results['dec']['A']['leak1_mean_other'].mean():.4f} partner={results['dec']['A']['leak1_partner'].mean():.4f}")
        if do_leak2:
            print(f"[A/raw] leak2 mean_other={results['raw']['A']['leak2_mean_other'].mean():.4f} partner={results['raw']['A']['leak2_partner'].mean():.4f}")
            print(f"[A/dec] leak2 mean_other={results['dec']['A']['leak2_mean_other'].mean():.4f} partner={results['dec']['A']['leak2_partner'].mean():.4f}")

    if do_B:
        print(f"[B/raw] recovery mean={results['raw']['B']['recovery'].mean():.4f} std={results['raw']['B']['recovery'].std():.4f}")

    print("============================\n")


if __name__ == "__main__":
    main()
