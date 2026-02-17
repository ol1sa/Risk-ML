from __future__ import annotations

import argparse
from pathlib import Path
import sys

import mlflow
import duckdb


sys.path.append(str(Path(__file__).resolve().parents[1]))
from riskml.duckdb_utils import connect, ensure_schemas, table_columns  # noqa: E402


CANONICAL = {
    "step": ["step"],
    "type": ["type"],
    "amount": ["amount"],
    "name_orig": ["nameOrig", "nameorig", "name_orig"],
    "oldbalance_orig": ["oldbalanceOrg", "oldBalanceOrig", "oldbalanceOrig", "oldbalance_org"],
    "newbalance_orig": ["newbalanceOrig", "newBalanceOrig", "newbalance_org", "newbalanceOrig"],
    "name_dest": ["nameDest", "namedest", "name_dest"],
    "oldbalance_dest": ["oldbalanceDest", "oldBalanceDest", "oldbalance_dest"],
    "newbalance_dest": ["newbalanceDest", "newBalanceDest", "newbalance_dest"],
    "is_fraud": ["isFraud", "is_fraud"],
    "is_flagged_fraud": ["isFlaggedFraud", "is_flagged_fraud"],
}


def resolve_column(existing_cols: list[str], candidates: list[str]) -> str:
    existing_lower = {c.lower(): c for c in existing_cols}
    for cand in candidates:
        key = cand.lower()
        if key in existing_lower:
            return existing_lower[key]
    raise ValueError(f"Missing expected column. Tried: {candidates}. Found columns: {existing_cols}")


def build_canonical_select(con: duckdb.DuckDBPyConnection, raw_table: str) -> str:
    cols = table_columns(con, raw_table)

    mapped = {k: resolve_column(cols, v) for k, v in CANONICAL.items()}

    # Fabricate a timestamp from step (hours since start). This gives you time ordering for splits/windows.
    # Using a fixed anchor date keeps it deterministic and portable.
    return f"""
    SELECT
        md5(
            concat_ws('|',
                cast({mapped['step']} AS VARCHAR),
                cast({mapped['type']} AS VARCHAR),
                cast({mapped['amount']} AS VARCHAR),
                cast({mapped['name_orig']} AS VARCHAR),
                cast({mapped['name_dest']} AS VARCHAR),
                cast({mapped['oldbalance_orig']} AS VARCHAR),
                cast({mapped['newbalance_orig']} AS VARCHAR),
                cast({mapped['oldbalance_dest']} AS VARCHAR),
                cast({mapped['newbalance_dest']} AS VARCHAR)
            )
        ) AS event_id,

        CAST({mapped['step']} AS INTEGER) AS step,
        CAST({mapped['type']} AS VARCHAR) AS event_type,
        CAST({mapped['amount']} AS DOUBLE) AS amount,

        CAST({mapped['name_orig']} AS VARCHAR) AS entity_orig,
        CAST({mapped['name_dest']} AS VARCHAR) AS entity_dest,

        CAST({mapped['oldbalance_orig']} AS DOUBLE) AS oldbalance_orig,
        CAST({mapped['newbalance_orig']} AS DOUBLE) AS newbalance_orig,
        CAST({mapped['oldbalance_dest']} AS DOUBLE) AS oldbalance_dest,
        CAST({mapped['newbalance_dest']} AS DOUBLE) AS newbalance_dest,

        CAST({mapped['is_fraud']} AS INTEGER) AS label,
        CAST({mapped['is_flagged_fraud']} AS INTEGER) AS is_flagged_fraud,

        (TIMESTAMP '2020-01-01 00:00:00' + (CAST({mapped['step']} AS BIGINT) * INTERVAL '1 hour')) AS event_time
    FROM {raw_table}
    """


def ingest(csv_path: Path, db_path: Path, sample_frac: float, seed: float) -> None:
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("riskml_ingest")

    with mlflow.start_run(run_name="ingest_paysim") as run:
        mlflow.log_param("csv_path", str(csv_path))
        mlflow.log_param("db_path", str(db_path))
        mlflow.log_param("sample_frac", sample_frac)
        mlflow.log_param("seed", seed)

        con = connect(db_path)
        ensure_schemas(con)

        if not csv_path.exists():
            raise FileNotFoundError(f"CSV not found: {csv_path}")

        con.execute("DROP TABLE IF EXISTS raw.paysim_raw;")

        # Deterministic sampling for dev speed
        # DuckDB setseed expects [0,1]
        seed01 = max(0.0, min(1.0, seed))
        con.execute("SELECT setseed(?);", [seed01])

        if sample_frac < 1.0:
            con.execute(
                f"""
                CREATE TABLE raw.paysim_raw AS
                SELECT *
                FROM read_csv_auto(?, header=true)
                WHERE random() < ?;
                """,
                [str(csv_path), sample_frac],
            )
        else:
            con.execute(
                """
                CREATE TABLE raw.paysim_raw AS
                SELECT *
                FROM read_csv_auto(?, header=true);
                """,
                [str(csv_path)],
            )

        # Canonical table
        con.execute("DROP TABLE IF EXISTS core.events;")
        canonical_sql = build_canonical_select(con, "raw.paysim_raw")
        con.execute(f"CREATE TABLE core.events AS {canonical_sql}")

        # Basic constraints / helpful indexes (DuckDB "indexes" are limited, but these help for filtering/sorting)
        con.execute("CREATE OR REPLACE VIEW core.events_ordered AS SELECT * FROM core.events ORDER BY event_time, event_id;")

        # Metrics + sanity checks
        row_count = con.execute("SELECT COUNT(*) FROM core.events").fetchone()[0]
        fraud_rate = con.execute("SELECT AVG(label)::DOUBLE FROM core.events").fetchone()[0]
        min_time, max_time = con.execute("SELECT MIN(event_time), MAX(event_time) FROM core.events").fetchone()

        # Missingness (simple)
        miss = con.execute("""
            SELECT
              SUM(CASE WHEN amount IS NULL THEN 1 ELSE 0 END) AS miss_amount,
              SUM(CASE WHEN entity_orig IS NULL THEN 1 ELSE 0 END) AS miss_entity_orig,
              SUM(CASE WHEN entity_dest IS NULL THEN 1 ELSE 0 END) AS miss_entity_dest
            FROM core.events;
        """).fetchone()

        mlflow.log_metric("rows", row_count)
        mlflow.log_metric("fraud_rate", fraud_rate)
        mlflow.log_metric("miss_amount", miss[0])
        mlflow.log_metric("miss_entity_orig", miss[1])
        mlflow.log_metric("miss_entity_dest", miss[2])

        print("\n=== INGEST SUMMARY ===")
        print(f"DB: {db_path}")
        print(f"Rows: {row_count:,}")
        print(f"Fraud rate (label=1): {fraud_rate:.6f}")
        print(f"Time range: {min_time} -> {max_time}")
        print(f"Missing: amount={miss[0]}, entity_orig={miss[1]}, entity_dest={miss[2]}")

        # Quick distribution for event types (top 10)
        print("\nTop event types:")
        for t, n in con.execute("""
            SELECT event_type, COUNT(*) AS n
            FROM core.events
            GROUP BY 1
            ORDER BY n DESC
            LIMIT 10;
        """).fetchall():
            print(f"  {t:<12} {n:,}")

        con.close()
        print(f"\nMLflow run logged: {run.info.run_id}")


def inspect(db_path: Path) -> None:
    con = connect(db_path)
    ensure_schemas(con)

    tables = con.execute("""
        SELECT table_schema, table_name
        FROM information_schema.tables
        WHERE table_schema IN ('raw','core','feat','model')
        ORDER BY table_schema, table_name;
    """).fetchall()

    print("\n=== TABLES ===")
    for s, t in tables:
        print(f"{s}.{t}")

    if ("core", "events") in tables:
        print("\n=== core.events schema ===")
        for row in con.execute("PRAGMA table_info('core.events')").fetchall():
            # (cid, name, type, notnull, dflt_value, pk)
            print(f"{row[1]:<18} {row[2]}")

        print("\n=== core.events sample ===")
        print(con.execute("SELECT * FROM core.events_ordered LIMIT 5").df())

    con.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=Path, default=Path("data/raw/paysim.csv"))
    p.add_argument("--db", type=Path, default=Path("data/db/riskml.duckdb"))
    p.add_argument("--sample-frac", type=float, default=1.0)
    p.add_argument("--seed", type=float, default=0.42)
    p.add_argument("--inspect", action="store_true")
    args = p.parse_args()

    if args.inspect:
        inspect(args.db)
        return

    if not (0.0 < args.sample_frac <= 1.0):
        raise ValueError("--sample-frac must be in (0, 1].")

    ingest(args.csv, args.db, args.sample_frac, args.seed)


if __name__ == "__main__":
    main()

