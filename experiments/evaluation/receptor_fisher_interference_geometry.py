import argparse
import os
import json
import torch
from pathlib import Path
from tqdm import tqdm

from src.utils.receptor_fisher_geometry import (
    make_logit_receptors, fisher_gram_streaming, build_fisher_D, apply_D, interference_metrics
)

# 1. CLI args
def parse_args():
    parser = argparse.ArgumentParser(description="Fisher interference geometry for logit receptors")
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--layers', type=str, default="all")
    parser.add_argument('--heads', type=str, default="all")
    parser.add_argument('--top_k', type=int, default=8)
    parser.add_argument('--position', type=str, default="last")
    parser.add_argument('--max_batches', type=int, default=50)
    parser.add_argument('--ridge', type=float, default=1e-3)
    parser.add_argument('--out_dir', type=str, default="logs/receptor_fisher_geometry")
    return parser.parse_args()

# 2. Config loader (reuse from train.py)
def load_config(config_path):
    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 3. Data loader (reuse from src/data/data_loader.py)
def build_dataloader(config):
    from src.data.data_loader import build_data_loader
    return build_data_loader(config, is_train=False)

# 4. Model loader
def load_model(config, checkpoint=None, device=None):
    from src.models.masked_transformer_circuit import MaskedTransformerCircuit
    from src.models.model_loader import load_hooked_model
    model = load_hooked_model(config)
    mtc = MaskedTransformerCircuit(model)
    if checkpoint:
        state = torch.load(checkpoint, map_location=device or 'cpu')
        mtc.load_state_dict(state['model'], strict=False)
    mtc.eval()
    mtc.to(device or torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    return mtc

# 5-10. Main logic
def main():
    args = parse_args()
    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataloader = build_dataloader(config)
    model = load_model(config, args.checkpoint, device)

    # Parse layers/heads
    n_layers = model.n_layers
    n_heads = model.n_heads
    if args.layers == "all":
        layers = list(range(n_layers))
    else:
        layers = [int(x) for x in args.layers.split(",") if x.strip()]
    if args.heads == "all":
        heads = list(range(n_heads))
    else:
        heads = [int(x) for x in args.heads.split(",") if x.strip()]

    # 5. Extract W_U and OV SVDs
    W_U_raw = model.get_unembed_weight()
    sv_dict = model.get_ov_write_singular_vectors(layers=layers, heads=heads, top_k=args.top_k, use_augmented=True)

    # 6. Build V_write_global and metadata
    V_write_list = []
    metadata = []
    for (layer, head), d in sv_dict.items():
        V = d['V_write']  # [d_model, top_k]
        S = d['S']        # [top_k]
        for i in range(V.shape[1]):
            V_write_list.append(V[:, i:i+1])
            metadata.append({
                'layer': layer,
                'head': head,
                'sv_idx': i,
                'singular_value': float(S[i].item())
            })
    V_write_global = torch.cat(V_write_list, dim=1)  # [d_model, m]
    m = V_write_global.shape[1]

    # 7. Logit receptors
    R_logit = make_logit_receptors(V_write_global, W_U_raw)

    # 8. Fisher Gram BEFORE
    G, N = fisher_gram_streaming(R_logit, model, dataloader, device, args.position, args.max_batches, return_N=True)
    metrics_before = interference_metrics(G)

    # 9. Fisher D
    D = build_fisher_D(G, ridge=args.ridge)

    # 10. Apply D, reproject, recompute
    V_dec = apply_D(V_write_global, D, renorm_cols=True)
    R_logit_dec = make_logit_receptors(V_dec, W_U_raw)
    G_dec, N2 = fisher_gram_streaming(R_logit_dec, model, dataloader, device, args.position, args.max_batches, return_N=True)
    metrics_after = interference_metrics(G_dec)

    # 11. Save outputs
    run_name = Path(args.config).stem + ("_ckpt" if args.checkpoint else "_base")
    out_dir = Path(args.out_dir) / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(G, out_dir / "G.pt")
    torch.save(G_dec, out_dir / "G_dec.pt")
    torch.save(D, out_dir / "D.pt")
    torch.save(V_write_global, out_dir / "V_write_global.pt")
    torch.save(V_dec, out_dir / "V_dec.pt")
    with open(out_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    metrics = {
        'args': vars(args),
        'N': N,
        'm': m,
        'metrics_before': metrics_before,
        'metrics_after': metrics_after,
        'delta': {k: metrics_after[k] - metrics_before[k] if isinstance(metrics_before[k], float) else None for k in metrics_before}
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # 12. Optional visualization
    try:
        from src.utils.visualization import save_heatmap
        save_heatmap(G.cpu().numpy(), out_dir / "G_heatmap.png", title="Fisher Gram BEFORE")
        save_heatmap(G_dec.cpu().numpy(), out_dir / "G_dec_heatmap.png", title="Fisher Gram AFTER")
    except Exception:
        pass

    print("Done. Results saved to", out_dir)

if __name__ == "__main__":
    main()
