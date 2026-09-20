"""RO-Crate 1.1 export for a single experiment.

Produces an ``<output_dir>/ro-crate-metadata.json`` (JSON-LD) and copies
referenced scan files into the crate root. The result is a self-describing
research-object directory that downstream tooling (DataCite, Crossref, FAIR
repositories) can ingest.

Spec: https://www.researchobject.org/ro-crate/specification/1.1/
"""
from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from mast.logging.v2.repos import V2Repos

logger = logging.getLogger(__name__)

RO_CRATE_CONTEXT = "https://w3id.org/ro/crate/1.1/context"


def build_metadata(repos: V2Repos, experiment_id: str) -> dict[str, Any]:
    """Build the in-memory JSON-LD dict for a single experiment crate."""
    exp = repos.experiments.get(experiment_id)
    if not exp:
        raise ValueError(f"experiment {experiment_id} not found")
    sample = repos.samples.get(exp["sample_id"]) if exp.get("sample_id") else None
    campaign = repos.campaigns.get(exp["campaign_id"]) if exp.get("campaign_id") else None
    actions = repos.actions.for_experiment(experiment_id, limit=100_000)
    observations = repos.observations.for_experiment(experiment_id, limit=100_000)
    scan_files = []
    for a in actions:
        scan_files.extend(repos.scan_files.for_action(a["id"]))

    graph: list[dict[str, Any]] = []

    # The crate-root descriptor file
    graph.append({
        "@id": "ro-crate-metadata.json",
        "@type": "CreativeWork",
        "conformsTo": {"@id": "https://w3id.org/ro/crate/1.1"},
        "about": {"@id": "./"},
    })

    # Root dataset
    graph.append({
        "@id": "./",
        "@type": "Dataset",
        "name": exp["title"],
        "description": exp.get("conclusion") or "",
        "datePublished": exp.get("ended_at") or exp["started_at"],
        "identifier": experiment_id,
        "hasPart": (
            [{"@id": f"#action/{a['id']}"} for a in actions]
            + [{"@id": f"#observation/{o['id']}"} for o in observations]
            + [{"@id": f"files/{sf['sha256']}"} for sf in scan_files]
        ),
        "isPartOf": ({"@id": f"#campaign/{campaign['id']}"} if campaign else None),
        "mentions": ({"@id": f"#sample/{sample['id']}"} if sample else None),
    })
    if campaign:
        graph.append({
            "@id": f"#campaign/{campaign['id']}",
            "@type": "ResearchProject",
            "name": campaign["title"],
            "description": campaign["hypothesis"],
        })
    if sample:
        graph.append({
            "@id": f"#sample/{sample['id']}",
            "@type": "Specimen",
            "name": sample["label"],
            "material": sample["material"],
            "preparationMethod": sample.get("prep_method") or "",
        })

    for a in actions:
        graph.append({
            "@id": f"#action/{a['id']}",
            "@type": "Action",
            "name": a["action_type"],
            "agent": {"@id": f"#agent/{a['agent_id']}"},
            "startTime": a["hlc"],
            "actionStatus": _to_schema_status(a["status"]),
            "parameters": _safe_json(a["params_json"]),
            "isPartOf": {"@id": "./"},
        })

    for o in observations:
        graph.append({
            "@id": f"#observation/{o['id']}",
            "@type": "Observation",
            "observedProperty": o["observable"],
            "value": o.get("scalar_value"),
            "unitText": o.get("units"),
            "channel": o.get("channel"),
            "resultedFrom": {"@id": f"#action/{o['action_id']}"},
            "subjectOf": ({"@id": f"files/{_scan_sha(repos, o['scan_file_id'])}"}
                          if o.get("scan_file_id") else None),
        })

    for sf in scan_files:
        graph.append({
            "@id": f"files/{sf['sha256']}",
            "@type": "File",
            "name": Path(sf["current_path"]).name,
            "contentSize": sf["size_bytes"],
            "encodingFormat": sf["mime_type"],
            "sha256": sf["sha256"],
            "softwareRequirements": sf["parser_spec"],
            "wasGeneratedBy": {"@id": f"#action/{sf['produced_by_action_id']}"},
        })

    # Strip None values for cleanliness.
    return {"@context": RO_CRATE_CONTEXT, "@graph": [_strip_none(n) for n in graph]}


def export(
    repos: V2Repos,
    experiment_id: str,
    output_dir: str | Path,
    *,
    copy_files: bool = True,
) -> Path:
    """Write a complete RO-Crate directory for an experiment.

    Returns the output_dir path. Files are deduplicated by sha256.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    metadata = build_metadata(repos, experiment_id)
    (out / "ro-crate-metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if copy_files:
        files_dir = out / "files"
        files_dir.mkdir(exist_ok=True)
        for node in metadata["@graph"]:
            if node.get("@type") != "File":
                continue
            sha = node.get("sha256")
            if not sha:
                continue
            sf = repos.scan_files.by_sha(sha)
            if not sf:
                continue
            src = Path(sf["current_path"])
            if not src.is_file():
                logger.warning("scan file missing: %s", src)
                continue
            target = files_dir / sha
            if not target.exists():
                shutil.copy2(src, target)

    # README/preview pointer
    (out / "README.md").write_text(
        f"# RO-Crate for experiment {experiment_id}\n\n"
        f"Generated by MAST at {datetime.now(timezone.utc).isoformat()}.\n"
        f"See `ro-crate-metadata.json` for the JSON-LD graph.\n",
        encoding="utf-8",
    )
    return out


# ── helpers ───────────────────────────────────────────────────────────

def _to_schema_status(status: str) -> str:
    return {
        "pending": "PotentialActionStatus",
        "running": "ActiveActionStatus",
        "succeeded": "CompletedActionStatus",
        "failed": "FailedActionStatus",
        "rolled_back": "FailedActionStatus",
        "retracted": "FailedActionStatus",
    }.get(status, "PotentialActionStatus")


def _safe_json(s: Any) -> Any:
    if s is None or s == "":
        return None
    if isinstance(s, str):
        try:
            return json.loads(s)
        except Exception:
            return s
    return s


def _scan_sha(repos: V2Repos, scan_file_id: str) -> str | None:
    sf = repos.scan_files.get(scan_file_id)
    return sf["sha256"] if sf else None


def _strip_none(d: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if v is not None}
