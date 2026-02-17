from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mlflow
import duckdb

sys.path.append(str(Path(__file__).resolve().parents[1]))
from riskml.duckdb_utils import connect, ensure_schemas  # noqa: E402


FEATURE_SQL = r"""
CREATE TABLE feat.model_table AS
WITH base AS (
    SELECT
        event_id,
        event_time,
        step,
        event_type,
        amount,
        entity_orig,
        entity_dest,
        oldbalance_orig,
        oldbalance_dest,
        label
    FROM core.events
),
typed AS (
    SELECT
        *,
        CAST(step % 24 AS INTEGER) AS hour_of_day,
        CAST(step / 24 AS INTEGER) AS day_index,
        LN(amount + 1.0) AS amount_log,

        CASE WHEN event_type = 'CASH_IN'  THEN 1 ELSE 0 END AS is_cash_in,
        CASE WHEN event_type = 'CASH_OUT' THEN 1 ELSE 0 END AS is_cash_out,
        CASE WHEN event_type = 'PAYMENT' THEN 1 ELSE 0 END AS is_payment,
        CASE WHEN event_type = 'TRANSFER' THEN 1 ELSE 0 END AS is_transfer,
        CASE WHEN event_type = 'DEBIT'   THEN 1 ELSE 0 END AS is_debit,

        amount / (oldbalance_orig + 1.0) AS orig_amt_over_oldbal,
        amount / (oldbalance_dest + 1.0) AS dest_amt_over_oldbal,

        CASE WHEN oldbalance_orig <= 0 THEN 1 ELSE 0 END AS orig_oldbal_is_zero,
        CASE WHEN oldbalance_dest <= 0 THEN 1 ELSE 0 END AS dest_oldbal_is_zero
    FROM base
),
feat AS (
    SELECT
        t.*,

        -- Origin rolling counts
        COUNT(*) OVER (
            PARTITION BY entity_orig
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ) AS orig_cnt_1h,

        COUNT(*) OVER (
            PARTITION BY entity_orig
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '6 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ) AS orig_cnt_6h,

        COUNT(*) OVER (
            PARTITION BY entity_orig
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ) AS orig_cnt_24h,

        COALESCE(SUM(amount) OVER (
            PARTITION BY entity_orig
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ), 0.0) AS orig_amt_sum_24h,

        -- Destination rolling counts
        COUNT(*) OVER (
            PARTITION BY entity_dest
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '1 hour' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ) AS dest_cnt_1h,

        COUNT(*) OVER (
            PARTITION BY entity_dest
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '6 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ) AS dest_cnt_6h,

        COUNT(*) OVER (
            PARTITION BY entity_dest
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ) AS dest_cnt_24h,

        COALESCE(SUM(amount) OVER (
            PARTITION BY entity_dest
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ), 0.0) AS dest_amt_sum_24h,

        -- Pair interaction history
        COUNT(*) OVER (
            PARTITION BY entity_orig, entity_dest
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ) AS pair_cnt_24h,

        -- Time since last (origin/dest)
        COALESCE(
            EXTRACT(EPOCH FROM (event_time - LAG(event_time) OVER (PARTITION BY entity_orig ORDER BY event_time))),
            -1
        ) AS orig_time_since_last_s,

        COALESCE(
            EXTRACT(EPOCH FROM (event_time - LAG(event_time) OVER (PARTITION BY entity_dest ORDER BY event_time))),
            -1
        ) AS dest_time_since_last_s

    FROM typed t
)
SELECT
  event_id,
  event_time,
  step,
  label,

  -- base numeric + simple transforms
  amount,
  hour_of_day,
  day_index,
  amount_log,

  -- type indicators (keep these, drop event_type itself)
  is_cash_in,
  is_cash_out,
  is_payment,
  is_transfer,
  is_debit,

  -- ratio/flags based on PRE balances only
  orig_amt_over_oldbal,
  dest_amt_over_oldbal,
  orig_oldbal_is_zero,
  dest_oldbal_is_zero,

  -- rolling history
  orig_cnt_1h,
  orig_cnt_6h,
  orig_cnt_24h,
  orig_amt_sum_24h,
  orig_time_since_last_s,

  dest_cnt_1h,
  dest_cnt_6h,
  dest_cnt_24h,
  dest_amt_sum_24h,
  dest_time_since_last_s,

  pair_cnt_24h
FROM feat
;
"""


AUDIT_SQL = r"""
WITH audit AS (
    SELECT
        event_id,
        event_time,
        entity_orig,                          -- FIX: expose column so outer query can use it
        -- Max timestamp seen in the strict look-back window (must be < event_time)
        MAX(event_time) OVER (
            PARTITION BY entity_orig
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ) AS orig_last_time_used_24h,
        -- Count excluding current row (what the feature pipeline computes)
        COUNT(*) OVER (
            PARTITION BY entity_orig
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND INTERVAL '1 microsecond' PRECEDING
        ) AS cnt_excl_current,
        -- Count including current row (would indicate leakage if different)
        COUNT(*) OVER (
            PARTITION BY entity_orig
            ORDER BY event_time
            RANGE BETWEEN INTERVAL '24 hours' PRECEDING AND CURRENT ROW
        ) AS cnt_incl_current
    FROM core.events
)
SELECT
    COUNT(*)                                                                      AS n,
    -- FIX: a non-NULL orig_last_time_used_24h that equals event_time means the
    --      current row leaked into its own window (timestamp collision edge case).
    SUM(
        CASE WHEN orig_last_time_used_24h IS NOT NULL
              AND orig_last_time_used_24h >= event_time
             THEN 1 ELSE 0 END
    )                                                                             AS bad_rows_timestamp,
    -- Secondary check: if cnt_incl_current > cnt_excl_current the window is
    -- working correctly (current row excluded). Flag rows where they are EQUAL
    -- AND cnt_incl_current > 0, which would indicate CURRENT ROW was never
    -- excluded (i.e. the PRECEDING boundary has no effect).
    SUM(
        CASE WHEN cnt_incl_current > 0
              AND cnt_incl_current = cnt_excl_current
             THEN 1 ELSE 0 END
    )                                                                             AS bad_rows_window_boundary
FROM audit;
"""


def build_features(db_path: Path) -> None:
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("riskml_features")

    with mlflow.start_run(run_name="build_features_paysim"):
        mlflow.log_param("db_path", str(db_path))
        mlflow.log_param("windows_hours", "1,6,24")
        mlflow.log_param("pair_window_hours", "24")
        mlflow.log_param("frame_excludes_current", True)
        mlflow.log_param("uses_post_transaction_balances", False)

        con = connect(db_path)
        ensure_schemas(con)

        con.execute("DROP TABLE IF EXISTS feat.model_table;")
        con.execute(FEATURE_SQL)

        n = con.execute("SELECT COUNT(*) FROM feat.model_table").fetchone()[0]
        fraud_rate = con.execute("SELECT AVG(label)::DOUBLE FROM feat.model_table").fetchone()[0]

        # Basic null scan
        nulls = con.execute("""
            SELECT
              SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS n_amount_null,
              SUM(CASE WHEN orig_time_since_last_s IS NULL THEN 1 ELSE 0 END) AS n_orig_tsl_null,
              SUM(CASE WHEN dest_time_since_last_s IS NULL THEN 1 ELSE 0 END) AS n_dest_tsl_null
            FROM feat.model_table;
        """).fetchone()

        mlflow.log_metric("rows", n)
        mlflow.log_metric("fraud_rate", fraud_rate)
        mlflow.log_metric("n_amount_null", nulls[0])
        mlflow.log_metric("n_orig_tsl_null", nulls[1])
        mlflow.log_metric("n_dest_tsl_null", nulls[2])

        print("\n=== FEATURES BUILT ===")
        print(f"Rows: {n:,}")
        print(f"Fraud rate: {fraud_rate:.6f}")
        print(f"Nulls: amount={nulls[0]}, orig_tsl={nulls[1]}, dest_tsl={nulls[2]}")

        con.close()


def inspect(db_path: Path) -> None:
    con = connect(db_path)
    ensure_schemas(con)

    print("\n=== feat.model_table columns ===")
    cols = con.execute("PRAGMA table_info('feat.model_table')").fetchall()
    for r in cols:
        print(f"{r[1]:<28} {r[2]}")

    print("\n=== sample rows ===")
    print(con.execute("SELECT * FROM feat.model_table ORDER BY event_time, event_id LIMIT 5").df())

    con.close()


def audit(db_path: Path) -> None:
    con = connect(db_path)
    ensure_schemas(con)
    n, bad_ts, bad_window = con.execute(AUDIT_SQL).fetchone()
    print("\n=== LEAKAGE AUDIT (origin_last_time_used_24h) ===")
    print(f"Rows checked:                    {n:,}")
    print(f"Bad rows – timestamp violation:  {bad_ts:,}   (should be 0)")
    print(f"Bad rows – window boundary:      {bad_window:,}   (should be 0)")
    con.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--db", type=Path, default=Path("data/db/riskml.duckdb"))
    p.add_argument("--inspect", action="store_true")
    p.add_argument("--audit", action="store_true")
    args = p.parse_args()

    if args.inspect:
        inspect(args.db)
        return
    if args.audit:
        audit(args.db)
        return

    build_features(args.db)


if __name__ == "__main__":
    main()