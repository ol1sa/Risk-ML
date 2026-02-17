from __future__ import annotations

import argparse
from pathlib import Path
import sys
import json

import duckdb
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.append(str(Path(__file__).resolve().parents[1]))
from riskml.duckdb_utils import connect, ensure_schemas 


NON_FEATURE_COLS = {"event_id", "event_time", "label", "step", "event_type", "entity_orig", "entity_dest"} 


def get_feature_columns(con: duckdb.DuckDBPyConnection, table: str) -> list[str]:
    cols = con.execute(f"PRAGMA table_info('{table}')").fetchall()
    names = [r[1] for r in cols]
    feats = [c for c in names if c not in NON_FEATURE_COLS]
    if not feats:
        raise RuntimeError("No feature columns found after exclusions.")
    return feats


def compute_time_cutoffs(con: duckdb.DuckDBPyConnection, table: str, q1: float, q2: float):
    row = con.execute(
        f"""
        SELECT
          quantile_cont(event_time, {q1}) AS t1,
          quantile_cont(event_time, {q2}) AS t2
        FROM {table}
        """
    ).fetchone()
    t1, t2 = row[0], row[1]
    if t1 is None or t2 is None or not (t1 < t2):
        raise RuntimeError(f"Bad cutoffs: t1={t1}, t2={t2}")
    return t1, t2


def load_split(con: duckdb.DuckDBPyConnection, table: str, where_sql: str, cols: list[str]) -> pd.DataFrame:
    col_sql = ", ".join(cols)
    return con.execute(f"SELECT {col_sql} FROM {table} WHERE {where_sql}").df()


def pick_threshold_cost(y_true: np.ndarray, y_prob: np.ndarray, cost_fp: float, cost_fn: float):
    # Evaluate thresholds from quantiles of probabilities
    qs = np.linspace(0.01, 0.99, 99)
    thresholds = np.unique(np.quantile(y_prob, qs))
    best = None
    for thr in thresholds:
        y_hat = (y_prob >= thr).astype(int)
        fp = int(((y_hat == 1) & (y_true == 0)).sum())
        fn = int(((y_hat == 0) & (y_true == 1)).sum())
        cost = fp * cost_fp + fn * cost_fn
        if best is None or cost < best["cost"]:
            best = {"threshold": float(thr), "fp": fp, "fn": fn, "cost": float(cost)}
    return best


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=Path("data/db/riskml.duckdb"))
    p.add_argument("--table", type=str, default="feat.model_table")
    p.add_argument("--outdir", type=Path, default=Path("data/derived"))
    p.add_argument("--train_q", type=float, default=0.70)
    p.add_argument("--valid_q", type=float, default=0.85)
    p.add_argument("--C", type=float, default=1.0)
    p.add_argument("--max_iter", type=int, default=300)
    p.add_argument("--calibration", type=str, choices=["sigmoid", "isotonic"], default="sigmoid")
    p.add_argument("--cost_fp", type=float, default=1.0)
    p.add_argument("--cost_fn", type=float, default=20.0)
    p.add_argument("--valid_file", default="preds_valid.parquet")
    p.add_argument("--test_file", default="preds_test.parquet")
    args = p.parse_args()

    # To avoid file-store deprecation
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("riskml_train")

    args.outdir.mkdir(parents=True, exist_ok=True)

    valid_path = args.outdir / args.valid_file
    test_path  = args.outdir / args.test_file

    con = connect(args.db)
    ensure_schemas(con)

    con.execute(f"SELECT 1 FROM {args.table} LIMIT 1;")

    feats = get_feature_columns(con, args.table)
    needed_cols = ["event_id", "event_time", "label"] + feats

    t1, t2 = compute_time_cutoffs(con, args.table, args.train_q, args.valid_q)

    # Time separation
    train_where = f"event_time < TIMESTAMP '{t1}'"
    valid_where = f"event_time >= TIMESTAMP '{t1}' AND event_time < TIMESTAMP '{t2}'"
    test_where  = f"event_time >= TIMESTAMP '{t2}'"

    with mlflow.start_run(run_name="logreg_baseline_time_split"):
        mlflow.log_param("db", str(args.db))
        mlflow.log_param("table", args.table)
        mlflow.log_param("train_q", args.train_q)
        mlflow.log_param("valid_q", args.valid_q)
        mlflow.log_param("cutoff_t1", str(t1))
        mlflow.log_param("cutoff_t2", str(t2))
        mlflow.log_param("model", "logistic_regression")
        mlflow.log_param("C", args.C)
        mlflow.log_param("max_iter", args.max_iter)
        mlflow.log_param("class_weight", "balanced")
        mlflow.log_param("calibration", args.calibration)
        mlflow.log_param("cost_fp", args.cost_fp)
        mlflow.log_param("cost_fn", args.cost_fn)

        # Loading data splits
        df_train = load_split(con, args.table, train_where, needed_cols)
        df_valid = load_split(con, args.table, valid_where, needed_cols)
        df_test  = load_split(con, args.table, test_where,  needed_cols)

        # Checks
        def split_stats(df: pd.DataFrame, name: str):
            n = len(df)
            rate = float(df["label"].mean()) if n else 0.0
            tmin = str(df["event_time"].min()) if n else "NA"
            tmax = str(df["event_time"].max()) if n else "NA"
            return {"name": name, "n": n, "fraud_rate": rate, "tmin": tmin, "tmax": tmax}

        s_train, s_valid, s_test = split_stats(df_train, "train"), split_stats(df_valid, "valid"), split_stats(df_test, "test")
        mlflow.log_dict({"train": s_train, "valid": s_valid, "test": s_test}, "split_summary.json")

        # Time separation
        if pd.to_datetime(df_train["event_time"].max()) >= pd.to_datetime(df_valid["event_time"].min()):
            raise RuntimeError("Time split leakage: train max time overlaps valid min time.")
        if pd.to_datetime(df_valid["event_time"].max()) >= pd.to_datetime(df_test["event_time"].min()):
            raise RuntimeError("Time split leakage: valid max time overlaps test min time.")

        X_train, y_train = df_train[feats], df_train["label"].astype(int).to_numpy()
        X_valid, y_valid = df_valid[feats], df_valid["label"].astype(int).to_numpy()
        X_test,  y_test  = df_test[feats],  df_test["label"].astype(int).to_numpy()

        # Model preprocessing + logistic regression
        base = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("clf", LogisticRegression(
                C=args.C,
                max_iter=args.max_iter,
                solver="lbfgs",
                class_weight="balanced",
                n_jobs=None
            )),
        ])

        base.fit(X_train, y_train)

        # Uncalibrated probabilities
        p_valid_uncal = base.predict_proba(X_valid)[:, 1]
        p_test_uncal  = base.predict_proba(X_test)[:, 1]

        pr_valid_uncal = float(average_precision_score(y_valid, p_valid_uncal))
        pr_test_uncal  = float(average_precision_score(y_test,  p_test_uncal))
        brier_valid_uncal = float(brier_score_loss(y_valid, p_valid_uncal))
        brier_test_uncal  = float(brier_score_loss(y_test,  p_test_uncal))

        mlflow.log_metric("pr_auc_valid_uncal", pr_valid_uncal)
        mlflow.log_metric("pr_auc_test_uncal", pr_test_uncal)
        mlflow.log_metric("brier_valid_uncal", brier_valid_uncal)
        mlflow.log_metric("brier_test_uncal", brier_test_uncal)

        # Calibration on validation
        calib = CalibratedClassifierCV(estimator=FrozenEstimator(base), method=args.calibration, cv=5)
        calib.fit(X_valid, y_valid)

        p_valid = calib.predict_proba(X_valid)[:, 1]
        p_test  = calib.predict_proba(X_test)[:, 1]

        pr_valid = float(average_precision_score(y_valid, p_valid))
        pr_test  = float(average_precision_score(y_test,  p_test))
        brier_valid = float(brier_score_loss(y_valid, p_valid))
        brier_test  = float(brier_score_loss(y_test,  p_test))

        mlflow.log_metric("pr_auc_valid", pr_valid)
        mlflow.log_metric("pr_auc_test", pr_test)
        mlflow.log_metric("brier_valid", brier_valid)
        mlflow.log_metric("brier_test", brier_test)

        # Threshold selection on validation
        best = pick_threshold_cost(y_valid, p_valid, args.cost_fp, args.cost_fn)
        mlflow.log_dict(best, "threshold_selection.json")

        valid_out = pd.DataFrame({
            "event_id": df_valid["event_id"],
            "event_time": df_valid["event_time"],
            "y_true": y_valid,
            "p_uncal": p_valid_uncal,
            "p_cal": p_valid,
            })
        test_out = pd.DataFrame({
            "event_id": df_test["event_id"],
            "event_time": df_test["event_time"],
            "y_true": y_test,
            "p_uncal": p_test_uncal,
            "p_cal": p_test,
            })
        valid_path = args.outdir / "preds_valid.parquet"
        test_path  = args.outdir / "preds_test.parquet"
        valid_out.to_parquet(valid_path, index=False)
        test_out.to_parquet(test_path, index=False)

        mlflow.log_artifact(str(valid_path))
        mlflow.log_artifact(str(test_path))

        # Log calibrated model
        mlflow.sklearn.log_model(calib, artifact_path="model")

        print("\n=== TRAIN DONE ===")
        print(json.dumps({"train": s_train, "valid": s_valid, "test": s_test}, indent=2))
        print("Saved:", valid_path, test_path)
        print("Chosen threshold:", best["threshold"], "valid_cost:", best["cost"])

    con.close()
    print("\n=== TRAIN HGB START ===")
    with mlflow.start_run(run_name="hgb_baseline_time_split"):
        pos = (y_train == 1).sum()
        neg_ = int((y_train == 0).sum())
        w_pos = float(neg_ / max(pos, 1))
        sample_weight = np.where(y_train == 1, w_pos, 1.0)

        mlflow.log_param("pos_weight", float(w_pos))
        mlflow.log_param("model", "hist_gradient_boosting")
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_iter", 200)
        mlflow.log_param("model", "hist_gradient_boosting")
        mlflow.log_param("pos_weight", w_pos)
        mlflow.log_param("max_depth", 6)
        mlflow.log_param("learning_rate", 0.1)
        mlflow.log_param("max_iter", 200)

        hgb = HistGradientBoostingClassifier(
            max_depth=6,
            learning_rate=0.1,
            max_iter=200,
            random_state=42
        )
        hgb.fit(X_train, y_train)

        p_valid_hgb = hgb.predict_proba(X_valid)[:, 1]
        p_test_hgb  = hgb.predict_proba(X_test)[:, 1]

        s_valid_hgb = hgb.decision_function(X_valid)
        s_test_hgb  = hgb.decision_function(X_test)

        pr_valid_hgb = float(average_precision_score(y_valid, p_valid_hgb))
        pr_test_hgb  = float(average_precision_score(y_test,  p_test_hgb))
        
        mlflow.log_metric("pr_auc_valid", pr_valid_hgb)
        mlflow.log_metric("pr_auc_test",  pr_test_hgb)

        valid_out_hgb = pd.DataFrame({
            "event_id": df_valid["event_id"],
            "event_time": df_valid["event_time"],
            "y_true": y_valid,
            "p_uncal": p_valid_hgb,
            "s_raw": s_valid_hgb,
            "amount": df_valid["amount"].astype(float),
            "orig_cnt_24h": df_valid["orig_cnt_24h"].astype(float),
            "pair_cnt_24h": df_valid["pair_cnt_24h"].astype(float),
            })
            
        test_out_hgb = pd.DataFrame({
            "event_id": df_test["event_id"],
            "event_time": df_test["event_time"],
            "y_true": y_test,
            "p_uncal": p_test_hgb,
            "s_raw": s_test_hgb,  # <-- ADD THIS
            "amount": df_test["amount"].astype(float),
            "orig_cnt_24h": df_test["orig_cnt_24h"].astype(float),
            "pair_cnt_24h": df_test["pair_cnt_24h"].astype(float),
            })

        valid_path_hgb = args.outdir / "preds_valid_hgb.parquet"
        test_path_hgb  = args.outdir / "preds_test_hgb.parquet"
        valid_out_hgb.to_parquet(valid_path_hgb, index=False)
        test_out_hgb.to_parquet(test_path_hgb, index=False)

        mlflow.log_artifact(str(valid_path_hgb))
        mlflow.log_artifact(str(test_path_hgb))

        mlflow.sklearn.log_model(hgb, artifact_path="model")

    print(f"HGB PR-AUC valid={pr_valid_hgb:.4f} test={pr_test_hgb:.4f}")
    print("Saved:", valid_path_hgb, test_path_hgb)

if __name__ == "__main__":
    main()

