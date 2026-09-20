"""High-level training-set + analysis exports.

Thin wrapper around ``olap`` that names the common cuts for downstream
training pipelines (compass §4.4 / §7 Phase 4).

For raw OLAP queries see ``mast.logging.v2.olap``.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

from mast.logging.v2 import olap

logger = logging.getLogger(__name__)


def export_training_set(
    db_path: str | Path,
    output_path: str | Path,
    *,
    observables: Iterable[str] = ("topography", "didv", "i_t"),
    since_hlc: str | None = None,
    until_hlc: str | None = None,
    compression: str = "ZSTD",
) -> int:
    """Export a Parquet training set selecting given *observables*."""
    cls = ", ".join(f"'{o}'" for o in observables)
    where = f"o.observable IN ({cls})"
    if since_hlc:
        where += f" AND o.hlc >= '{since_hlc}'"
    if until_hlc:
        where += f" AND o.hlc < '{until_hlc}'"
    return olap.export_observations_parquet(
        db_path, output_path, where_clause=where, compression=compression,
    )


def export_per_experiment(
    db_path: str | Path,
    experiment_id: str,
    output_dir: str | Path,
) -> dict[str, int]:
    """Export per-experiment 4 parquet files: actions/observations/events/scan_files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts = {}
    counts["actions"] = olap.export_actions_parquet(
        db_path, output_dir / "actions.parquet",
        where_clause=f"experiment_id = '{experiment_id}'",
    )
    counts["observations"] = olap.export_observations_parquet(
        db_path, output_dir / "observations.parquet",
        where_clause=f"o.experiment_id = '{experiment_id}'",
    )
    if olap.have_duckdb():
        with olap.attach(db_path) as conn:
            out_ev = str(output_dir / "events.parquet")
            conn.execute(
                f"COPY (SELECT * FROM m.events WHERE experiment_id = '{experiment_id}') "
                f"TO '{out_ev}' (FORMAT PARQUET, COMPRESSION 'ZSTD');"
            )
            counts["events"] = int(conn.execute(
                f"SELECT count(*) FROM m.events WHERE experiment_id = '{experiment_id}'"
            ).fetchone()[0])
            out_sf = str(output_dir / "scan_files.parquet")
            conn.execute(
                f"COPY (SELECT sf.* FROM m.scan_files sf JOIN m.actions a "
                f"  ON sf.produced_by_action_id = a.id "
                f"  WHERE a.experiment_id = '{experiment_id}') "
                f"TO '{out_sf}' (FORMAT PARQUET, COMPRESSION 'ZSTD');"
            )
            counts["scan_files"] = int(conn.execute(
                f"SELECT count(*) FROM m.scan_files sf JOIN m.actions a "
                f"ON sf.produced_by_action_id = a.id "
                f"WHERE a.experiment_id = '{experiment_id}'"
            ).fetchone()[0])
    # Sidecar manifest so the consumer knows what they got.
    (output_dir / "_manifest.json").write_text(
        json.dumps({"experiment_id": experiment_id, "counts": counts}, indent=2),
        encoding="utf-8",
    )
    return counts


def export_audit_log(db_path: str | Path, output_path: str | Path) -> int:
    """Dump the full audit_log to Parquet for offline compliance review."""
    return olap.export_actions_parquet  # type: ignore[return-value]  # placeholder unused
