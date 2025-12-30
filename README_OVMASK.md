# OV SVD Diagonal Mask Training (GP / IOI / GT + Joint)

Goal: Learn diagonal masks over OV SVD directions so a masked GPT-2 matches the original next-token distribution on clean + corrupted prompts, with L1 sparsity.

## Put CSVs here

`data_main/` contains 9 csv files:

- GP: train*.csv val*.csv test*.csv
- IOI: train*.csv val*.csv test*.csv
- GT: train*.csv val*.csv test*.csv

## Install

pip install -r requirements.txt

## Run (single task)

python run.py train --task gp --data_dir data_main --train_csv train_gp.csv --val_csv val_gp.csv --test_csv test_gp.csv

## Run (joint)

python run.py train_joint --data_dir data_main 
  --gp_train_csv train_gp.csv --gp_val_csv val_gp.csv --gp_test_csv test_gp.csv 
  --ioi_train_csv train_ioi.csv --ioi_val_csv val_ioi.csv --ioi_test_csv test_ioi.csv 
  --gt_train_csv train_gt.csv --gt_val_csv val_gt.csv --gt_test_csv test_gt.csv

## Outputs

Each run creates:
runs/`<task>`/ov_mask_`<timestamp>`/
  config.json
  train.log
  metrics.json
  final_eval.json
  masks/mask_logits.pt
  masks/mask_values.pt
  selected.json

SVD cache (shared):
svd_cache/gpt2/svd_meta.json
svd_cache/gpt2/svd/l{l}_h{h}_{U,S,Vh}.pt

Teacher cache (shared, sharded):
teacher_cache/`<task>`/`<split>`/`<variant>`/meta.json
teacher_cache/`<task>`/`<split>`/`<variant>`/shard_XXXXX.pt
