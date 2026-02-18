#!/bin/bash
set -euo pipefail

PRED_DIR="data/derived"
COST_FP=1
COST_FN=20

echo "=== HGB (s_raw): FINAL @ k_frac=0.015 (topk_strict) ==="
python scripts/evaluate.py --pred_dir "$PRED_DIR" \
  --valid_file preds_valid_hgb.parquet --test_file preds_test_hgb.parquet \
  --use s_raw --policy topk_strict --k_frac 0.015 --cost_fp "$COST_FP" --cost_fn "$COST_FN"

echo
echo "=== Logistic (p_uncal): baseline @ k_frac=0.01 (topk threshold) ==="
python scripts/evaluate.py --pred_dir "$PRED_DIR" \
  --valid_file preds_valid.parquet --test_file preds_test.parquet \
  --use p_uncal --policy topk --k_frac 0.01 --cost_fp "$COST_FP" --cost_fn "$COST_FN"

echo
echo "=== PACKAGE FINAL ARTIFACTS ==="
python scripts/final_package.py
