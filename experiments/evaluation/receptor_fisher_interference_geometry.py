"""
Evaluate Fisher/KL interference geometry of OV-SVD logit receptors and deconfound via Fisher whitening.

Goal:
- Build logit receptors from RIGHT singular vectors (V_write) as in Beyond Components Appendix B.2
- Measure interference via Fisher Gram G
- Compute Fisher-whitening D (ZCA-style whitening)
- Apply D in residual write-space (V_write) and show interference drops after deconfounding
- Save heatmaps + top interfering receptor pairs for before/after

Outputs (under out_dir/<run_name>/):
- metrics.json
- G.pt, G_dec.pt, D.pt, V_write_global.pt, V_dec.pt
- metadata.json
- top_pairs_before.json, top_pairs_after.json
- heatmaps (matplotlib): G_heatmap.png / G_heatmap_first128.png, etc.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import matplotlib.pyplot as plt
from transformer_lens import HookedTransformer

# Make repo root importable (this file lives in experiments/evaluation/)
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
import sys
sys.path.append(str(ROOT_DIR))

from experiments.train import load_config
from src.data.data_loader import load_ioi_dataset, load_gp_dataset, load_gt_dataset
from src.utils.utils import get_data_column_names
from src.models.masked_transformer_circuit import MaskedTransformerCircuit
from src.utils.receptor_fisher_geometry import (
    make_logit_receptors,
    fisher_gram_streaming,
    build_fisher_D,
    apply_D,
    interference_metrics,
)


def _parse_int_list(arg: str, all_n: int) -> Optional[List[int]]:
    s = arg.strip().lower()
    if s == "all" or s == "":
        return None
    parts = [p.strip() for p in s.split(",") if p.strip() != ""]
    out: List[int] = []
    for p in parts:
        v = int(p)
        if v < 0 or v >= all_n:
            raise ValueError(f"Index {v} out of range [0, {all_n-1}]")
        out.append(v)
    return out


def _top_offdiag_pairs(G: torch.Tensor, metadata: List[Dict[str, Any]], k: int = 50) -> List[Dict[str, Any]]:
    """
    Return top-k |G_ij| off-diagonal pairs (unique with i<j), with metadata.
    """
    m = int(G.shape[0])
    if G.ndim != 2 or G.shape[1] != m:
        raise ValueError("G must be square")

    absG = G.abs().clone()
    absG.fill_diagonal_(0.0)

    flat = absG.flatten()
    take = min(flat.numel(), max(k * 10, 100))  # oversample due to symmetry duplicates
    _, idxs = torch.topk(flat, k=take, largest=True)

    pairs: List[Dict[str, Any]] = []
    seen = set()
    for idx in idxs.tolist():
        i = idx // m
        j = idx % m
        if i == j:
            continue
        if i > j:
            i, j = j, i
        key = (i, j)
        if key in seen:
            continue
        seen.add(key)

        pairs.append({
            "i": int(i),
            "j": int(j),
            "abs": float(abs(G[i, j]).item()),
            "signed": float(G[i, j].item()),
            "meta_i": metadata[i],
            "meta_j": metadata[j],
        })
        if len(pairs) >= k:
            break
    return pairs


def _save_heatmap(mat: torch.Tensor, path: Path, title: str, max_show: int = 128) -> None:
    mat = mat.detach().cpu()
    m = int(mat.shape[0])
    show = min(m, max_show)
    mat_show = mat[:show, :show]

    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(mat_show, aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("receptor index")
    ax.set_ylabel("receptor index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _build_raw_dataloader(config: Dict[str, Any]):
    data_type = config["data_type"]
    # data_dir = config["data_dir"]
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"].get("num_workers", 4))

    if data_type == "ioi":
        fn = load_ioi_dataset
    elif data_type == "gp":
        fn = load_gp_dataset
    elif data_type == "gt":
        fn = load_gt_dataset
    else:
        raise ValueError(f"Unknown data_type='{data_type}'")

    # Use validation split for evaluation
    return fn(
        data_dir=None,
        batch_size=batch_size,
        full_batch=False,
        shuffle=False,
        num_workers=num_workers,
        validation=True,
        train=False,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML (configs/*.yaml)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Optional checkpoint to load")
    parser.add_argument("--layers", type=str, default="all", help='e.g. "all" or "9,10,11"')
    parser.add_argument("--heads", type=str, default="all", help='e.g. "all" or "0,1,2"')
    parser.add_argument("--top_k", type=int, default=8)
    parser.add_argument("--position", type=str, default="last")
    parser.add_argument("--max_batches", type=int, default=50)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--out_dir", type=str, default="logs/receptor_fisher_geometry")
    parser.add_argument("--top_pairs", type=int, default=50)
    args = parser.parse_args()

    config = load_config(args.config)
    data_type = config["data_type"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load base model (same pattern as repo training/intervention)
    model = HookedTransformer.from_pretrained(
        config["model"]["name"],
        cache_dir=config["model"]["pretrained_cache_dir"],
    ).to(device)

    # Ensure padding is defined (repo pattern)
    if getattr(model.tokenizer, "pad_token", None) is None:
        model.tokenizer.pad_token = model.tokenizer.eos_token

    # Instantiate circuit like training does
    circuit = MaskedTransformerCircuit(
        model=model,
        device=device,
        cache_svd=bool(config["masking"].get("cache_svd", True)),
        mask_init_value=config["masking"]["mask_init_value"],
    )

    # Optional: load checkpoint masks (same keys as experiments/ablation/intervention.py)
    if args.checkpoint is not None:
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        for key in ["qk_masks", "ov_masks", "mlp_in_masks", "mlp_out_masks"]:
            if key in ckpt:
                setattr(circuit, key, ckpt[key])

    raw_loader = _build_raw_dataloader(config)

    layers_list = _parse_int_list(args.layers, model.cfg.n_layers)
    heads_list = _parse_int_list(args.heads, model.cfg.n_heads)

    # Extract unembedding + OV write singular vectors (RIGHT singular vectors)
    W_U_raw = circuit.get_unembed_weight()
    sv_dict = circuit.get_ov_write_singular_vectors(
        layers=layers_list,
        heads=heads_list,
        top_k=args.top_k,
        use_augmented=True,
    )

    some_key = next(iter(sv_dict.keys()))
    print("W_U_raw.shape:", tuple(W_U_raw.shape))
    print("Example V_write shape:", tuple(sv_dict[some_key]["V_write"].shape), "for", some_key)

    # Concatenate all selected heads’ V_write columns
    V_cols = []
    meta: List[Dict[str, Any]] = []
    col = 0
    for (layer, head), d in sorted(sv_dict.items(), key=lambda x: (x[0][0], x[0][1])):
        Vw = d["V_write"]
        S = d["S"]
        for k in range(Vw.shape[1]):
            V_cols.append(Vw[:, k:k + 1])
            meta.append({
                "col": int(col),
                "layer": int(layer),
                "head": int(head),
                "sv_idx": int(k),
                "sigma": float(S[k].item()) if k < S.numel() else None,
            })
            col += 1

    if len(V_cols) == 0:
        raise RuntimeError("No V_write columns found. Check --layers/--heads and SVD cache.")
    V_write_global = torch.cat(V_cols, dim=1)
    m = int(V_write_global.shape[1])

    # IMPORTANT shape guard (Prompt 3 step 6)
    if W_U_raw.shape[0] == model.cfg.d_model:
        d_model = int(W_U_raw.shape[0])
    elif W_U_raw.shape[1] == model.cfg.d_model:
        d_model = int(W_U_raw.shape[1])
    else:
        d_model = int(min(W_U_raw.shape))

    if V_write_global.shape[0] == d_model + 1:
        V_write_global = V_write_global[:d_model, :]
        print(f"[shape guard] Dropped augmented row: V_write_global now {tuple(V_write_global.shape)}")

    # Tokenize clean inputs (raw_loader yields strings)
    clean_column_name, _ = get_data_column_names(data_type)

    def tokenized_iter():
        for batch in raw_loader:
            input_ids = model.tokenizer(
                batch[clean_column_name],
                return_tensors="pt",
                padding=True,
            )["input_ids"].to(device)
            lengths = (input_ids != model.tokenizer.pad_token_id).sum(dim=1)
            attention_mask = torch.arange(input_ids.size(1), device=device)[None, :] < lengths[:, None]
            yield {"input_ids": input_ids, "attention_mask": attention_mask}

    # Convert to logit receptors
    R_logit = make_logit_receptors(V_write_global, W_U_raw)

    # Fisher Gram BEFORE
    G, N = fisher_gram_streaming(
        R_logit=R_logit,
        model=model,
        dataloader=tokenized_iter(),
        device=device,
        position=args.position,
        max_batches=args.max_batches,
        return_N=True,
        dtype=torch.float64,
    )
    metrics_before = interference_metrics(G)

    # Fisher whitening D
    D = build_fisher_D(G, ridge=args.ridge)

    # Apply D in write space and reproject
    V_dec = apply_D(V_write_global, D, renorm_cols=True)
    R_logit_dec = make_logit_receptors(V_dec, W_U_raw)

    G_dec, N2 = fisher_gram_streaming(
        R_logit=R_logit_dec,
        model=model,
        dataloader=tokenized_iter(),
        device=device,
        position=args.position,
        max_batches=args.max_batches,
        return_N=True,
        dtype=torch.float64,
    )
    metrics_after = interference_metrics(G_dec)

    # Create run dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{data_type}_m{m}_ridge{args.ridge}_{ts}"
    run_dir = Path(args.out_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save tensors
    torch.save(G.cpu(), run_dir / "G.pt")
    torch.save(G_dec.cpu(), run_dir / "G_dec.pt")
    torch.save(D.cpu(), run_dir / "D.pt")
    torch.save(V_write_global.cpu(), run_dir / "V_write_global.pt")
    torch.save(V_dec.cpu(), run_dir / "V_dec.pt")

    # Save metadata
    with open(run_dir / "metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    # Save top pairs
    with open(run_dir / "top_pairs_before.json", "w") as f:
        json.dump(_top_offdiag_pairs(G, meta, k=args.top_pairs), f, indent=2)
    with open(run_dir / "top_pairs_after.json", "w") as f:
        json.dump(_top_offdiag_pairs(G_dec, meta, k=args.top_pairs), f, indent=2)

    # Save metrics.json
    off_b = float(metrics_before["offdiag_ratio"])
    off_a = float(metrics_after["offdiag_ratio"])
    improvement_factor = off_b / max(off_a, 1e-12)

    metrics_payload = {
        "args": vars(args),
        "N": int(N),
        "N2": int(N2),
        "m": int(m),
        "metrics_before": metrics_before,
        "metrics_after": metrics_after,
        "delta_offdiag_ratio": float(off_a - off_b),
        "improvement_factor": float(improvement_factor),
    }
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics_payload, f, indent=2)

    # Heatmaps (always save; downsample only for visualization)
    def _hm_name(base: str) -> str:
        if m <= 128:
            return base
        stem = base[:-4] if base.endswith(".png") else base
        return f"{stem}_first128.png"

    suffix = " first128" if m > 128 else ""
    _save_heatmap(G, run_dir / _hm_name("G_heatmap.png"),
                  title=f"G (before){suffix} | m={m} ridge={args.ridge} N={N}")
    _save_heatmap(G_dec, run_dir / _hm_name("G_dec_heatmap.png"),
                  title=f"G (after){suffix} | m={m} ridge={args.ridge} N={N2}")
    abs_delta = (G_dec.abs() - G.abs())
    _save_heatmap(abs_delta, run_dir / _hm_name("G_abs_delta_heatmap.png"),
                  title=f"|G_after|-|G_before|{suffix} | m={m} ridge={args.ridge} N={N}")

    # Stdout summary
    print("\n=== receptor_fisher_interference_geometry summary ===")
    print("run_dir:", str(run_dir))
    print("m:", m, "N:", N)
    print("before:", {k: metrics_before[k] for k in ["offdiag_ratio", "max_abs_offdiag", "mean_abs_offdiag", "cond_number"]})
    print("after :", {k: metrics_after[k] for k in ["offdiag_ratio", "max_abs_offdiag", "mean_abs_offdiag", "cond_number"]})
    print("improvement_factor:", improvement_factor)


if __name__ == "__main__":
    main()
