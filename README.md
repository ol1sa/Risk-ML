# Risk-ML

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