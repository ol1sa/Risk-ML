from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


def main():
    # ---- FINAL SETTINGS (keep these aligned with your run_all.sh / final choice) ----
    pred_dir = Path("data/derived")
    valid_file = "preds_valid_hgb.parquet"
    test_file = "preds_test_hgb.parquet"

    score_col = "s_raw"
    policy = "topk_strict"
    k_frac = 0.015
    cost_fp = 1
    cost_fn = 20

    out_flagged = Path("data/derived/flagged_test_hgb_topk_strict_k015.parquet")
    out_report = Path("artifacts_eval/final_report.json")
    out_report.parent.mkdir(parents=True, exist_ok=True)

    # ---- Load test preds ----
    dft = pd.read_parquet(pred_dir / test_file).copy()
    if score_col not in dft.columns:
        raise RuntimeError(f"Missing score_col={score_col} in {test_file}. Columns: {list(dft.columns)}")
    for c in ["event_id", "event_time", "y_true"]:
        if c not in dft.columns:
            raise RuntimeError(f"Missing required col {c} in {test_file}.")

    y = dft["y_true"].astype(int).to_numpy()
    n = len(dft)
    k = max(1, int(round(k_frac * n)))

    # ---- Strict top-k decision ----
    if policy != "topk_strict":
        raise RuntimeError("This packaging script is meant for the FINAL strict top-k policy only.")

    dft = dft.sort_values([score_col, "event_time", "event_id"], ascending=[False, True, True]).reset_index(drop=True)
    flagged = dft.head(k).copy()
    thr = float(dft.loc[k - 1, score_col])

    yhat = np.zeros(n, dtype=int)
    yhat[:k] = 1

    # ---- Metrics ----
    tn, fp, fn, tp = confusion_matrix(dft["y_true"].astype(int).to_numpy(), yhat, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    test_cost = float(fp * cost_fp + fn * cost_fn)
    flag_none_cost = float(int(y.sum()) * cost_fn)
    savings = float(flag_none_cost - test_cost)
    cost_per_review = float(test_cost / k)

    report = {
        "final_policy": {
            "model": "hgb",
            "score_col": score_col,
            "policy": policy,
            "k_frac": k_frac,
            "k": int(k),
            "threshold_at_k": thr,
            "cost_fp": cost_fp,
            "cost_fn": cost_fn,
        },
        "test_metrics": {
            "n": int(n),
            "flagged": int(k),
            "flagged_frac": float(k / n),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "precision": float(precision),
            "recall": float(recall),
            "test_cost": float(test_cost),
            "flag_none_cost": float(flag_none_cost),
            "savings": float(savings),
            "cost_per_review": float(cost_per_review),
            "fp_per_tp": float(fp / tp) if tp else None,
        },
    }

    # ---- Save artifacts ----
    out_flagged.parent.mkdir(parents=True, exist_ok=True)
    flagged.to_parquet(out_flagged, index=False)

    out_report.write_text(json.dumps(report, indent=2))

    print("Saved flagged queue:", out_flagged)
    print("Saved report:", out_report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
