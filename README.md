# OV SVD Diagonal Mask Training (GP / IOI / GT)

Goal: learn sigmoid-diagonal masks over OV singular directions so a masked GPT-2 matches
the teacher next-token distribution on clean + corrupted prompts while staying sparse (L1).

## Setup

Create `data_main/` and place the 9 CSV files there.

Install deps:

```bash
pip install -r requirements.txt
```
