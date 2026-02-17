from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import matplotlib.pyplot as plt

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    confusion_matrix,
)
from sklearn.calibration import calibration_curve

def decision_metrics_from_counts(tp: int, fp: int, fn: int, tn: int, cost_fp: float, cost_fn: float, k: int | None = None):
    cost = fp * cost_fp + fn * cost_fn
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    fp_per_tp = fp / max(tp, 1)

    out = {
        "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
        "precision": float(precision),
        "recall": float(recall),
        "cost": float(cost),
        "fp_per_tp": float(fp_per_tp),
    }
    if k is not None and k > 0:
        out["cost_per_review"] = float(cost / k)
    return out

def strict_topk_on_df(
    df: pd.DataFrame,
    score_col: str,
    y_col: str,
    k_frac: float,
    cost_fp: float,
    cost_fn: float,
    tie_cols: list[str] | None = None,
):
    if not (0 < k_frac <= 1):
        raise ValueError("k_frac must be in (0, 1].")

    tie_cols = tie_cols or ["event_time", "event_id"]

    n = len(df)
    k = max(1, int(round(k_frac * n)))

    sort_cols = [score_col] + tie_cols
    asc = [False] + [True] * len(tie_cols)

    ranked = df.sort_values(sort_cols, ascending=asc).reset_index(drop=True)

    y_true = ranked[y_col].astype(int).to_numpy()
    y_hat = np.zeros(n, dtype=int)
    y_hat[:k] = 1

    thr = float(ranked.loc[k - 1, score_col])
    flagged = int(y_hat.sum())
    flagged_frac = flagged / n

    tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
    metrics = decision_metrics_from_counts(tp, fp, fn, tn, cost_fp, cost_fn, k=k)
    metrics.update({"k": int(k), "threshold": float(thr), "flagged": flagged, "flagged_frac": float(flagged_frac)})

    return metrics

def expected_cost_curve(y_true: np.ndarray, p: np.ndarray, cost_fp: float, cost_fn: float):
    qs = np.linspace(0.0, 1.0, 2001)
    thr = np.quantile(p, qs)
    thresholds = np.unique(thr)
    thresholds = np.concatenate([
        np.array([-np.inf]),
        thresholds,
        np.array([np.inf]),
    ])
    rows = []
    for thr in thresholds:
        y_hat = (p >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_hat, labels=[0, 1]).ravel()
        cost = fp * cost_fp + fn * cost_fn
        rows.append((float(thr), int(tp), int(fp), int(fn), int(tn), float(cost)))
    out = pd.DataFrame(rows, columns=["threshold", "tp", "fp", "fn", "tn", "cost"])
    min_cost = out["cost"].min()
    tied = out[out["cost"] == min_cost].copy()
    best = tied.sort_values("threshold", ascending=True).iloc[0].to_dict()
    return out, best

def precision_recall_at_k(y_true: np.ndarray, p: np.ndarray, k_frac: float):
    if not (0 < k_frac <= 1):
        raise ValueError("k_frac must be in (0, 1].")
    y = y_true.astype(int)
    p = p.astype(float)

    n = len(y)
    k = max(1, int(round(k_frac * n)))

    order = np.argsort(-p)          # sort descending by score
    thr = float(p[order][k - 1])    # threshold is score of k-th ranked instance (1-based k)

    y_hat = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, y_hat, labels=[0, 1]).ravel()

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0

    return float(precision), float(recall), int(tp), int(fp), int(k), float(thr)

def strict_topk_mask(p: np.ndarray, event_time: np.ndarray, event_id: np.ndarray, k_frac: float) -> tuple[np.ndarray, int, float]:
    if not (0 < k_frac <= 1):
        raise ValueError("k_frac must be in (0, 1].")

    p = p.astype(float)
    n = len(p)
    k = max(1, int(round(k_frac * n)))

    order = np.lexsort((event_id, event_time, -p)) # sort by p desc, then event_time asc, then event_id asc
    top = order[:k]

    y_hat = np.zeros(n, dtype=int)
    y_hat[top] = 1

    thr = float(p[top[-1]])
    return y_hat, k, thr


def prior_adjust(p, pi_train, pi_deploy):
    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    odds = p / (1 - p)
    k = (pi_deploy / (1 - pi_deploy)) / (pi_train / (1 - pi_train))
    odds2 = odds * k
    return odds2 / (1 + odds2)

def plot_pr(y_true, p, title: str, outpath: Path):
    prec, rec, _ = precision_recall_curve(y_true, p)
    ap = average_precision_score(y_true, p)

    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"{title} | PR-AUC={ap:.4f}")
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def sigmoid(x: np.ndarray) -> np.ndarray:
    # stable sigmoid for converting raw scores to (0,1) for calibration plots only
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x, dtype=float)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    expx = np.exp(x[~pos])
    out[~pos] = expx / (1.0 + expx)
    return out

def plot_calibration(y_true, p, title: str, outpath: Path, n_bins: int = 20):
    frac_pos, mean_pred = calibration_curve(y_true, p, n_bins=n_bins, strategy="quantile")

    plt.figure()
    plt.plot(mean_pred, frac_pos)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Mean predicted probability")
    plt.ylabel("Fraction of positives")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_cost_curve(df_cost: pd.DataFrame, title: str, outpath: Path):
    plt.figure()
    plt.plot(df_cost["threshold"], df_cost["cost"])
    plt.xlabel("Threshold")
    plt.ylabel("Expected cost")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred_dir", type=Path, default=Path("data/derived"))
    p.add_argument("--use", choices=["p_cal", "p_uncal", "s_raw"], default="p_cal")
    p.add_argument("--cost_fp", type=float, default=1.0)
    p.add_argument("--cost_fn", type=float, default=20.0)
    p.add_argument("--policy", choices=["cost_threshold", "topk", "topk_strict"], default="topk_strict")
    p.add_argument("--k_frac", type=float, default=0.01)
    p.add_argument("--adjust-prior", action="store_true",help="Apply label-shift (prior) correction to calibrated probabilities before thresholding.")
    p.add_argument("--pi-train", type=float, default=None,help="Training/calibration base rate (if omitted, uses valid y rate).")
    p.add_argument("--pi-deploy", type=float, default=None,help="Deployment base rate (if omitted, uses test y rate — OK for offline demo only).")
    p.add_argument("--valid_file", type=str, default="preds_valid.parquet")
    p.add_argument("--test_file", type=str, default="preds_test.parquet")
    p.add_argument("--sweep", action="store_true", help="Run k_frac sweep (0.001,0.005,0.01,0.02,0.05) and exit.")
    p.add_argument("--k_fracs", type=str, default="0.001,0.005,0.01,0.02,0.05", help="Comma-separated k_fracs for sweep.")

    args = p.parse_args()

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("riskml_eval")

    valid_path = args.pred_dir / args.valid_file
    test_path  = args.pred_dir / args.test_file
    if not valid_path.exists() or not test_path.exists():
        raise FileNotFoundError("Missing preds files. Run train.py first to create preds_valid.parquet and preds_test.parquet")

    dfv = pd.read_parquet(valid_path)
    dft = pd.read_parquet(test_path)

    if args.use not in dfv.columns or args.use not in dft.columns:
        raise KeyError(
            f"--use {args.use} not found in preds files. "
            f"valid cols={sorted(dfv.columns)} test cols={sorted(dft.columns)}"
        )

    yv = dfv["y_true"].astype(int).to_numpy()
    yt = dft["y_true"].astype(int).to_numpy()

    pv = dfv[args.use].to_numpy()
    pt = dft[args.use].to_numpy()

    pv_use, pt_use = pv, pt

    # Prior-adjust only makes sense for probabilities, not raw scores.
    if args.adjust_prior:
        if args.use == "s_raw":
            raise ValueError("--adjust-prior cannot be used with --use s_raw (raw scores are not probabilities).")

        pi_train = float(args.pi_train) if args.pi_train is not None else float(yv.mean())
        pi_deploy = float(args.pi_deploy) if args.pi_deploy is not None else float(yt.mean())

        pv_use = prior_adjust(pv, pi_train=pi_train, pi_deploy=pi_deploy)
        pt_use = prior_adjust(pt, pi_train=pi_train, pi_deploy=pi_deploy)
    
    def parse_kfracs(s: str):
        return [float(x.strip()) for x in s.split(",") if x.strip()]

    if args.sweep:
        k_fracs = parse_kfracs(args.k_fracs)

        # Build a df for strict mode sweeps (uses event_time/event_id only)
        dft_use2 = dft.copy()
        dft_use2["score_use"] = pt_use

        print("\n=== SWEEP (policy=topk_strict) ===")
        for kf in [0.01, 0.0125, 0.015, 0.0175, 0.02, 0.0225, 0.025, 0.0275, 0.03]:
            m = strict_topk_on_df(
                dft_use2,
                score_col="score_use",
                y_col="y_true",
                k_frac=kf,
                cost_fp=args.cost_fp,
                cost_fn=args.cost_fn,
                tie_cols=["event_time", "event_id"],
            )
            print(
                f"k={m['k']:6d} ({kf*100:4.1f}%) | thr≈{m['threshold']:.6f} | "
                f"prec={m['precision']:.3f} rec={m['recall']:.3f} | "
                f"cost={m['cost']:.0f} | cost/review={m['cost']/m['k']:.3f} | FP/TP={m['fp_per_tp']:.2f}"
            )

        print("\n=== SWEEP DONE ===")
        return

    with mlflow.start_run(run_name=f"eval_{args.use}"):
        mlflow.log_param("use_prob", args.use)
        mlflow.log_param("cost_fp", args.cost_fp)
        mlflow.log_param("cost_fn", args.cost_fn)
        mlflow.log_param("policy", args.policy)
        mlflow.log_param("k_frac", float(args.k_frac))
        mlflow.log_param("adjust_prior", bool(args.adjust_prior))
        if args.adjust_prior:
            mlflow.log_param("pi_train", pi_train)
            mlflow.log_param("pi_deploy", pi_deploy)

        pr_v = float(average_precision_score(yv, pv_use))
        pr_t = float(average_precision_score(yt, pt_use))
        mlflow.log_metric("pr_auc_valid", pr_v)
        mlflow.log_metric("pr_auc_test", pr_t)

        best = None
        thr = None
        yhat_eval = None

        if args.policy == "cost_threshold":
            cost_df, best = expected_cost_curve(yv, pv_use, args.cost_fp, args.cost_fn)
            thr = float(best["threshold"])
            yhat_eval = (pt_use >= thr).astype(int)
            policy_info = {"policy": "cost_threshold_tuned_on_valid", "threshold": thr, "best_valid": best}

        elif args.policy == "topk":
            prec_k, rec_k, tp_k, fp_k, k, thr_k = precision_recall_at_k(yt, pt_use, args.k_frac)
            thr = float(thr_k)
            yhat_eval = (pt_use >= thr).astype(int)
            flagged = int(yhat_eval.sum())
            print(f"Flagged={flagged} ({flagged/len(yhat_eval):.4%}) at threshold={thr:.6f}")
            policy_info = {"policy": "topk_threshold", "k_frac": args.k_frac, "k_target": k, "threshold": thr}
            cost_df, _ = expected_cost_curve(yv, pv_use, args.cost_fp, args.cost_fn)

        else:  # topk_strict
            if "event_id" not in dft.columns or "event_time" not in dft.columns:
                raise RuntimeError("topk_strict requires event_id and event_time columns in preds parquet.")
            event_id = dft["event_id"].to_numpy()
            event_time = dft["event_time"].to_numpy()

            yhat_eval, k, thr = strict_topk_mask(
                p=pt_use,
                event_time=event_time,
                event_id=event_id,
                k_frac=args.k_frac,
            )
            flagged = int(yhat_eval.sum())
            print(f"Flagged={flagged} ({flagged/len(yhat_eval):.4%}) strict topk k={k} thr≈{thr:.6f}")
            policy_info = {"policy": "topk_strict", "k_frac": args.k_frac, "k": k, "threshold": float(thr)}
            cost_df, _ = expected_cost_curve(yv, pv_use, args.cost_fp, args.cost_fn)

        mlflow.log_dict(policy_info, "policy.json")

        if yhat_eval is None:
            raise RuntimeError("Policy did not set yhat_eval.")

        test_flag_none_cost = int(yt.sum()) * args.cost_fn
        print("test flag-none cost:", float(test_flag_none_cost))
        if best is not None:
            print("best row:", best)
        else:
            print(f"best row: (n/a for policy={args.policy})")

        # Apply on test and log results
        tn, fp, fn, tp = confusion_matrix(yt, yhat_eval, labels=[0, 1]).ravel()
        test_cost = float(fp * args.cost_fp + fn * args.cost_fn)

        positives = int(yt.sum())
        flag_none_cost = positives * args.cost_fn
        savings = float(flag_none_cost - test_cost)

        flagged = int(yhat_eval.sum())
        cost_per_review = float(test_cost / max(flagged, 1))
        fp_per_tp = float(fp / max(tp, 1))

        mlflow.log_metric("test_flag_none_cost", float(flag_none_cost))
        mlflow.log_metric("test_savings_vs_flag_none", savings)
        mlflow.log_metric("test_cost_per_review", cost_per_review)
        mlflow.log_metric("test_fp_per_tp", fp_per_tp)

        print(f"Flag-none cost={flag_none_cost:.0f} | Savings={savings:.0f} | Cost/review={cost_per_review:.3f} | FP/TP={fp_per_tp:.2f}")

        flagged = int(yhat_eval.sum())
        print(f"Final flagged={flagged} ({flagged/len(yhat_eval):.4%}) policy={args.policy}")
        mlflow.log_metric("flagged_frac_test", flagged / len(yhat_eval))

        mlflow.log_metric("test_tp", tp)
        mlflow.log_metric("test_fp", fp)
        mlflow.log_metric("test_fn", fn)
        mlflow.log_metric("test_tn", tn)
        mlflow.log_metric("test_cost", test_cost)

        mlflow.log_metric("p_mean_test", float(pt_use.mean()))
        mlflow.log_metric("p_p95_test", float(np.quantile(pt_use, 0.95)))
        mlflow.log_metric("p_p99_test", float(np.quantile(pt_use, 0.99)))
        mlflow.log_metric("p_max_test", float(pt_use.max()))
        print("p_test: mean", pt_use.mean(), "p95", np.quantile(pt_use,0.95), "p99", np.quantile(pt_use,0.99), "max", pt_use.max())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        mlflow.log_metric("test_precision", precision)
        mlflow.log_metric("test_recall", recall)
        print(f"Precision={precision:.3f} Recall={recall:.3f}")

        k_fracs = [0.001, 0.005, 0.01]

        if args.policy == "topk_strict": # Strict top-k (flags exactly k, breaking ties by event_time asc, then event_id asc)
            dft_use2 = dft.copy()
            dft_use2["score_use"] = pt_use

            for kf in k_fracs:
                m = strict_topk_on_df(
                    dft_use2,
                    score_col="score_use",
                    y_col="y_true",
                    k_frac=kf,
                    cost_fp=args.cost_fp,
                    cost_fn=args.cost_fn,
                    tie_cols=["event_time", "event_id"],
                )
                mlflow.log_metric(f"strict_cost_at_{kf}", m["cost"])
                mlflow.log_metric(f"strict_precision_at_{kf}", m["precision"])
                mlflow.log_metric(f"strict_recall_at_{kf}", m["recall"])
                print(f"StrictCost@{kf*100:.1f}% = {m['cost']:.0f} (k={m['k']}, thr≈{m['threshold']:.6f})")

            else:
                # Threshold top-k (may flag >k when ties exist)
                for kf in k_fracs:
                    prec_k, rec_k, tp_k, fp_k, k, thr_k = precision_recall_at_k(yt, pt_use, kf)
                    fn_k = int(yt.sum()) - tp_k
                    cost_k = fp_k * args.cost_fp + fn_k * args.cost_fn
                    mlflow.log_metric(f"cost_at_{kf}", float(cost_k))
                    mlflow.log_metric(f"precision_at_{kf}", float(prec_k))
                    mlflow.log_metric(f"recall_at_{kf}", float(rec_k))
                    print(f"Cost@{kf*100:.1f}% = {cost_k:.0f} (k_target={k}, thr={thr_k:.6f})")
            

        # Plots
        art_dir = Path("artifacts_eval")
        art_dir.mkdir(exist_ok=True)

        pr_valid_png = art_dir / "pr_valid.png"
        pr_test_png  = art_dir / "pr_test.png"
        cal_valid_png = art_dir / "calibration_valid.png"
        cal_test_png  = art_dir / "calibration_test.png"
        cost_png = art_dir / "cost_curve_valid.png"

        plot_pr(yv, pv_use, "VALID", pr_valid_png)
        plot_pr(yt, pt_use, "TEST", pr_test_png)

        if args.use == "s_raw":
            plot_calibration(yv, sigmoid(pv_use), "VALID calibration (sigmoid(s_raw))", cal_valid_png)
            plot_calibration(yt, sigmoid(pt_use), "TEST calibration (sigmoid(s_raw))", cal_test_png)
        else:
            plot_calibration(yv, pv_use, "VALID calibration", cal_valid_png)
            plot_calibration(yt, pt_use, "TEST calibration", cal_test_png)

        plot_cost_curve(cost_df, "VALID cost vs threshold", cost_png)

        for f in [pr_valid_png, pr_test_png, cal_valid_png, cal_test_png, cost_png]:
            mlflow.log_artifact(str(f))

        print("\n=== EVAL DONE ===")
        print(f"Using: {args.use}")
        print(f"PR-AUC valid={pr_v:.4f} test={pr_t:.4f}")
        print(f"Chosen threshold={thr:.4f} | test_cost={test_cost:.2f} | TP={tp} FP={fp} FN={fn} TN={tn}")


if __name__ == "__main__":
    main()

