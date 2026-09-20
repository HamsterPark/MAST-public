"""DuckDB ATTACH helpers for OLAP and training-data export.

Per compass §4.4, complex aggregations and training-data dumps are run via
DuckDB attached against the same SQLite v2 db file in READ_ONLY mode. This
avoids ETL and keeps SQLite as the canonical store.

DuckDB is an optional dep; everything here gracefully degrades if it is
missing (the caller gets ``DuckDBNotInstalled`` raised eagerly).
"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)


class DuckDBNotInstalled(RuntimeError):
    pass


def have_duckdb() -> bool:
    try:
        import duckdb  # noqa: F401
        return True
    except ImportError:
        return False


@contextmanager
def attach(db_path: str | Path, *, alias: str = "m", read_only: bool = True) -> Iterator[Any]:
    """Context manager yielding a DuckDB connection with the SQLite file attached."""
    if not have_duckdb():
        raise DuckDBNotInstalled(
            "DuckDB is required for OLAP queries. "
            "Install with: .venv-v2-py313/Scripts/python.exe -m pip install duckdb"
        )
    import duckdb
    conn = duckdb.connect(":memory:")
    try:
        conn.execute("INSTALL sqlite; LOAD sqlite;")
    except Exception as exc:
        logger.debug("sqlite extension already loaded or auto-loaded: %s", exc)
    ro_clause = ", READ_ONLY" if read_only else ""
    conn.execute(f"ATTACH '{db_path}' AS {alias} (TYPE sqlite{ro_clause});")
    try:
        yield conn
    finally:
        try:
            conn.execute(f"DETACH {alias};")
        except Exception:
            pass
        conn.close()


# ── Canned aggregations ───────────────────────────────────────────────

def actions_per_day(db_path: str | Path) -> list[dict]:
    """Daily action volume across all experiments."""
    with attach(db_path) as conn:
        return conn.execute(
            "SELECT substr(hlc, 1, 8) AS day_ms_prefix, "
            "       count(*) AS n_actions, "
            "       sum(CASE WHEN status='succeeded' THEN 1 ELSE 0 END) AS n_succeeded, "
            "       sum(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS n_failed "
            "FROM m.actions GROUP BY 1 ORDER BY 1"
        ).fetchdf().to_dict(orient="records")


def observables_summary(db_path: str | Path) -> list[dict]:
    """Per-observable count + scalar stats."""
    with attach(db_path) as conn:
        return conn.execute(
            "SELECT observable, "
            "       count(*) AS n, "
            "       min(scalar_value) AS min_v, "
            "       max(scalar_value) AS max_v, "
            "       avg(scalar_value) AS avg_v, "
            "       count(scan_file_id) AS n_with_scan "
            "FROM m.observations GROUP BY observable ORDER BY n DESC"
        ).fetchdf().to_dict(orient="records")


def skill_usage(db_path: str | Path, *, limit: int = 50) -> list[dict]:
    with attach(db_path) as conn:
        return conn.execute(
            "SELECT action_type, agent_id, count(*) AS n, "
            "       sum(CASE WHEN status='succeeded' THEN 1 ELSE 0 END) AS n_ok, "
            "       sum(CASE WHEN status='failed' THEN 1 ELSE 0 END) AS n_err, "
            "       avg(duration_ms) AS avg_duration_ms "
            f"FROM m.actions GROUP BY action_type, agent_id ORDER BY n DESC LIMIT {int(limit)}"
        ).fetchdf().to_dict(orient="records")


# ── Parquet export ────────────────────────────────────────────────────

def export_observations_parquet(
    db_path: str | Path,
    output_path: str | Path,
    *,
    where_clause: str = "1=1",
    compression: str = "ZSTD",
) -> int:
    """Stream observations (joined with their scan_files + actions) to a Parquet file.

    Returns the row count written. ``where_clause`` is interpolated raw — the
    caller is trusted not to inject SQL since this is a developer/training tool.
    """
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with attach(db_path) as conn:
        conn.execute(
            f"COPY ("
            f"  SELECT o.id AS observation_id, o.hlc, o.observable, o.channel, "
            f"         o.scalar_value, o.units, o.result_summary_json, "
            f"         o.action_id, a.action_type, a.agent_id, a.params_json, "
            f"         a.experiment_id, "
            f"         sf.sha256, sf.current_path, sf.format_kind, sf.meta_json "
            f"  FROM m.observations o "
            f"  JOIN m.actions a ON o.action_id = a.id "
            f"  LEFT JOIN m.scan_files sf ON o.scan_file_id = sf.id "
            f"  WHERE {where_clause}"
            f") TO '{output_path}' (FORMAT PARQUET, COMPRESSION '{compression}');"
        )
        n = conn.execute(
            f"SELECT count(*) FROM m.observations o "
            f"JOIN m.actions a ON o.action_id = a.id WHERE {where_clause}"
        ).fetchone()[0]
        return int(n)


def export_actions_parquet(
    db_path: str | Path,
    output_path: str | Path,
    *,
    where_clause: str = "1=1",
) -> int:
    output_path = str(output_path)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with attach(db_path) as conn:
        conn.execute(
            f"COPY (SELECT * FROM m.actions WHERE {where_clause}) "
            f"TO '{output_path}' (FORMAT PARQUET, COMPRESSION 'ZSTD');"
        )
        return int(conn.execute(
            f"SELECT count(*) FROM m.actions WHERE {where_clause}"
        ).fetchone()[0])


# ── Custom SQL ────────────────────────────────────────────────────────

def query(db_path: str | Path, sql: str, params: tuple = ()) -> list[dict]:
    """Run an arbitrary SQL query against the attached SQLite via DuckDB."""
    with attach(db_path) as conn:
        rows = conn.execute(sql, params).fetchdf().to_dict(orient="records")
        return rows
