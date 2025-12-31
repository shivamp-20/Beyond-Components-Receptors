# OV-mask (E1 Part A–C, OV-only) — GPT-2 small

This repo trains an OV-only mask per attention head for GPT-2 small, following the plan you pasted.

## Quickstart

1) Put your 9 CSV files in `data_main/` (3 tasks × {train,val,test}).

2) Install deps:

```bash
pip install -r requirements.txt
```

3) Run a task end-to-end (A+B+C+receptors):

```bash
python run_ov_mask.py --task gp --data_dir data_main \
  --train_csv train_1k_gp.csv --val_csv val_1k_gp.csv --test_csv test_1k_gp.csv
```

Outputs go to `outputs/<task>/`:
- `cache/*.pt`
- `svd/layerXX_headYY.pt`
- `masks/layerXX_headYY.pt`
- `receptors/layerXX_headYY.json`
- `metrics.jsonl`
- `plots/*.png`
