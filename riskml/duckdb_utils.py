from __future__ import annotations

from pathlib import Path
import duckdb

def connect(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute("PRAGMA threads=4;")
    con.execute("PRAGMA enable_progress_bar=true;")
    return con

def ensure_schemas(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("CREATE SCHEMA IF NOT EXISTS raw;")
    con.execute("CREATE SCHEMA IF NOT EXISTS core;")
    con.execute("CREATE SCHEMA IF NOT EXISTS feat;")
    con.execute("CREATE SCHEMA IF NOT EXISTS model;")

def table_columns(con: duckdb.DuckDBPyConnection, table_name: str) -> list[str]:
    rows = con.execute(f"PRAGMA table_info('{table_name}')").fetchall()
    return [r[1] for r in rows]

