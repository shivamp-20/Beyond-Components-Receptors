# OV mask learning (paper-faithful) — GPT-2 small + TransformerLens

This repo implements **Part A: OV-only mask learning** exactly in the style you specified:
- **paired clean + corrupt activations** (cache corrupt `hook_z` then run masked clean pass),
- **complementary term** (`m` and `1-m` mixing),
- **teacher KL loss** + sparsity,
- saves **SVD factors**, **trained masks**, and optional **logit receptors**.

TransformerLens exposes per-head weights and activations (e.g., `W_O` is `[head, d_head, d_model]`, `hook_z` is `[batch, pos, head, d_head]`). See docs. 

## Folder layout

Put your 9 CSVs in:
- `data_main/` (gitignored)

Outputs go to:
- `artifacts/<task>/<run_id>/...`

A run never overwrites a previous run; it always creates a new timestamped `run_id`.

## Setup (local)

```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt
cp config.example.yaml config.yaml
# edit config.yaml to put your real CSV filenames
```

## Run (local / Colab / Kaggle)

Train one task:

```bash
python train_ov_masks.py --config config.yaml --task gp
python train_ov_masks.py --config config.yaml --task ioi
python train_ov_masks.py --config config.yaml --task gt
```

Quick sanity check (1 batch train + 1 batch val):

```bash
python train_ov_masks.py --config config.yaml --task gp --dry-run
```

## Notes

- If some examples produce **multi-token labels**, we **drop them** (and log how many were dropped). The training objective here is *single next-token* KL/accuracy.
- We automatically set `use_attn_result=True` so `blocks.<l>.attn.hook_result` exists (standard TransformerLens usage).

