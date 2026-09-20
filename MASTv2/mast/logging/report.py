"""Generate experiment reports from logged data."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from mast.core.types import ActionRecord
from mast.logging.storage import ExperimentStorage


class ReportGenerator:
    """Generate experiment reports from logged data."""

    def __init__(self, storage: ExperimentStorage):
        self._storage = storage

    # ── Markdown report ──────────────────────────────────────────────

    def generate_markdown(
        self, experiment_id: str, output_path: str | None = None
    ) -> str:
        """Generate a Markdown experiment report."""
        exp = self._storage.get_experiment(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment {experiment_id} not found.")

        actions = self._storage.get_actions(experiment_id)
        samples = self._storage.get_samples(experiment_id)
        sample_map = {s["id"]: s for s in samples}
        success_count = sum(
            1 for a in actions if a.result and a.result.success
        )
        fail_count = sum(
            1 for a in actions if a.result and not a.result.success
        )

        lines: list[str] = []
        lines.append(f"# Experiment: {exp['name']}")
        lines.append("")
        lines.append(f"**Goal**: {exp.get('goal_text', '')}")
        lines.append(
            f"**Duration**: {exp['start_time']} — {exp.get('end_time', 'ongoing')}"
        )
        lines.append(f"**Status**: {exp['status']}")
        if samples:
            lines.append(f"**Samples**: {len(samples)}")
        lines.append("")
        lines.append("## Timeline")
        lines.append("")

        # Group actions by sample
        from collections import OrderedDict
        groups: OrderedDict[str, list[ActionRecord]] = OrderedDict()
        for action in actions:
            key = action.sample_id or ""
            groups.setdefault(key, []).append(action)

        # Render in sample order
        sample_order = [s["id"] for s in samples]
        ordered_keys: list[str] = []
        for sid in sample_order:
            if sid in groups:
                ordered_keys.append(sid)
        if "" in groups:
            ordered_keys.append("")
        for key in groups:
            if key not in ordered_keys:
                ordered_keys.append(key)

        for key in ordered_keys:
            group_actions = groups[key]
            if key and key in sample_map:
                s = sample_map[key]
                lines.append(f"### Sample: {s['name']} ({s['status']})")
                if s.get("description"):
                    lines.append(f"_{s['description']}_")
                lines.append("")
            elif key == "" and samples:
                lines.append("### Early actions")
                lines.append("")

            for action in group_actions:
                version_str = f" v{action.skill_version}" if action.skill_version else ""
                lines.append(
                    f"#### {action.timestamp} — {action.skill_name}{version_str}"
                )
                lines.append("")

                if action.parameters:
                    lines.append("**Parameters**:")
                    lines.append("")
                    lines.append("| Parameter | Value |")
                    lines.append("|-----------|-------|")
                    for k, v in action.parameters.items():
                        lines.append(f"| {k} | {v} |")
                    lines.append("")

                if action.result:
                    status_str = "Success" if action.result.success else "FAILURE"
                    lines.append(f"**Result**: {status_str}")
                    if action.result.error:
                        lines.append(f"  - Error: {action.result.error}")
                    if action.result.data:
                        summary = json.dumps(action.result.data, default=str)
                        if len(summary) > 200:
                            summary = summary[:200] + "..."
                        lines.append(f"  - Data: `{summary}`")
                    lines.append("")

                lines.append(f"**Duration**: {action.duration_s:.3f}s")
                lines.append(f"**Approval**: {action.approval_source}")
                lines.append("")

        # Summary
        lines.append("## Summary")
        lines.append("")
        lines.append(
            f"Total actions: {len(actions)}, "
            f"Successful: {success_count}, "
            f"Failed: {fail_count}"
        )
        lines.append("")

        md = "\n".join(lines)
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(md, encoding="utf-8")
        return md

    # ── Reproducible script ──────────────────────────────────────────

    def generate_reproducible_script(
        self, experiment_id: str, output_path: str | None = None
    ) -> str:
        """Generate a standalone Python script that replays the exact Nanonis call sequence.

        Each NanonisCallRecord becomes a direct nanonis method call with
        comments showing timestamps and parameters.
        """
        exp = self._storage.get_experiment(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment {experiment_id} not found.")

        actions = self._storage.get_actions(experiment_id)
        samples = self._storage.get_samples(experiment_id)
        sample_map = {s["id"]: s for s in samples}

        # The nanonis_spm Nanonis client takes an already-connected socket
        # (``Nanonis(connection)``), NOT a host/port pair. Emitting
        # ``Nanonis("127.0.0.1", 6501)`` produced a script that crashed on the
        # very first line (TypeError: too many positional args). Open a TCP
        # socket first and hand it to the constructor — mirrors how
        # mast.core.connection.ConnectionPool builds its clients.
        lines: list[str] = [
            '"""',
            f"Reproducible script for experiment: {exp['name']}",
            f"Goal: {exp.get('goal_text', '')}",
            f"Original run: {exp['start_time']}",
            '"""',
            "",
            "import socket",
            "",
            "from nanonis_spm import Nanonis",
            "",
            "# Connect to Nanonis (adjust host/port for your instrument)",
            "sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)",
            'sock.connect(("127.0.0.1", 6501))',
            "nanonis = Nanonis(sock)",
            "",
        ]

        # Group by sample
        from collections import OrderedDict
        groups: OrderedDict[str, list] = OrderedDict()
        for action in actions:
            key = action.sample_id or ""
            groups.setdefault(key, []).append(action)

        sample_order = [s["id"] for s in samples]
        ordered_keys: list[str] = []
        for sid in sample_order:
            if sid in groups:
                ordered_keys.append(sid)
        if "" in groups:
            ordered_keys.append("")
        for key in groups:
            if key not in ordered_keys:
                ordered_keys.append(key)

        for key in ordered_keys:
            group_actions = groups[key]
            if key and key in sample_map:
                s = sample_map[key]
                lines.append(f"# === Sample: {s['name']} ===")
                lines.append("")
            elif key == "" and samples:
                lines.append("# === Early actions (no sample) ===")
                lines.append("")

            for action in group_actions:
                lines.append(f"# --- {action.timestamp} | {action.skill_name} ---")
                if action.parameters:
                    lines.append(f"# Parameters: {json.dumps(action.parameters, default=str)}")

                for call in action.nanonis_calls:
                    lines.append(f"# [{call.timestamp}] elapsed={call.elapsed_s:.4f}s")
                    # Security: a logged method name is interpolated verbatim into
                    # executable source. Reject anything that is not a bare Python
                    # identifier so a corrupted/poisoned log can't smuggle code
                    # (e.g. "Scan_Action(); __import__('os').system('rm -rf /')#").
                    method = str(call.method)
                    if not method.isidentifier():
                        lines.append(
                            f"# SKIPPED non-identifier method name: {method!r}"
                        )
                        lines.append("")
                        continue
                    args_str = ", ".join(repr(a) for a in call.args)
                    # kwargs keys also land verbatim as `key=...`, so they must
                    # be identifiers too; drop the whole call if any key isn't.
                    bad_kwargs = [k for k in call.kwargs if not str(k).isidentifier()]
                    if bad_kwargs:
                        lines.append(
                            f"# SKIPPED call with non-identifier kwarg name(s): {bad_kwargs!r}"
                        )
                        lines.append("")
                        continue
                    kwargs_str = ", ".join(
                        f"{k}={v!r}" for k, v in call.kwargs.items()
                    )
                    all_args = ", ".join(filter(None, [args_str, kwargs_str]))
                    lines.append(f"nanonis.{method}({all_args})")
                    lines.append("")

        lines.append("# End of experiment replay")
        lines.append("")

        script = "\n".join(lines)
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(script, encoding="utf-8")
        return script

    # ── JSON export ──────────────────────────────────────────────────

    def export_json(
        self, experiment_id: str, output_path: str | None = None
    ) -> str:
        """Export full experiment data as JSON for downstream agents."""
        exp = self._storage.get_experiment(experiment_id)
        if exp is None:
            raise ValueError(f"Experiment {experiment_id} not found.")

        actions = self._storage.get_actions(experiment_id)
        samples = self._storage.get_samples(experiment_id)

        payload = {
            "experiment": exp,
            "samples": samples,
            "actions": [_action_to_export_dict(a) for a in actions],
        }

        output = json.dumps(payload, indent=2, default=str)
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(output, encoding="utf-8")
        return output


    def generate_markdown_with_citations(
        self, experiment_id: str, output_path: str | None = None
    ) -> str:
        """Generate Markdown report with recommended citations appended."""
        from mast.citations.manager import CitationManager

        md = self.generate_markdown(experiment_id)
        cm = CitationManager(self._storage)
        md = cm.append_to_report(experiment_id, md)
        if output_path:
            p = Path(output_path)
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(md, encoding="utf-8")
        return md


def _action_to_export_dict(record: ActionRecord) -> dict:
    """Convert ActionRecord to a fully-expanded dict for JSON export."""
    d: dict = {
        "id": record.id,
        "experiment_id": record.experiment_id,
        "sample_id": record.sample_id,
        "timestamp": record.timestamp,
        "skill_name": record.skill_name,
        "skill_version": record.skill_version,
        "parameters": record.parameters,
        "context": record.context,
        "duration_s": record.duration_s,
        "approval_source": record.approval_source,
    }
    if record.result:
        d["result"] = asdict(record.result)
    else:
        d["result"] = None
    d["state_before"] = asdict(record.state_before) if record.state_before else None
    d["state_after"] = asdict(record.state_after) if record.state_after else None
    d["nanonis_calls"] = [asdict(c) for c in record.nanonis_calls]
    return d
