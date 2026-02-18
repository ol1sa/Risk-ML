# Risk-ML

## How to download / build data

This project expects a transaction dataset in `data/raw/` and builds a DuckDB feature table used for training.

### 1) Create a virtual environment + install deps
'''bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Goal
Given a fixed review budget (fraction of transactions that can be reviewed), select cases to minimize expected cost:
- False Positive cost = 1
- False Negative cost = 20

## Final approach
- Model: HistGradientBoosting (HGB)
- Score used for ranking: `s_raw` (model raw score)
- Decision policy: strict top-k (`topk_strict`)
- Review budget: `k_frac = 0.015` (review top 1.5%)

Strict top-k means we always flag exactly `k` cases (no “tie inflation”).
Ties are deterministically resolved by sorting on:
1) score desc
2) event_time asc
3) event_id asc

## How to run
1) Train models + write prediction files:
```bash
python scripts/train.py --db data/db/riskml.duckdb
