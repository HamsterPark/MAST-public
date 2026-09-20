"""the reproducible-script generator must not interpolate an
unvalidated Nanonis method name (or kwargs key) into executable Python source.

A corrupted / poisoned log row could otherwise smuggle arbitrary code into the
generated replay script. We assert the generator validates with ``isidentifier()``
and neutralises any non-identifier method / kwarg name instead of emitting it.

Runs fully offline: a tiny in-memory storage stub stands in for ExperimentStorage.
"""
from __future__ import annotations

import sys
from pathlib import Path
_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import ast

import pytest

from mast.core.types import ActionRecord, NanonisCallRecord
from mast.logging.report import ReportGenerator


class _StubStorage:
    """Minimal stand-in for ExperimentStorage used by ReportGenerator."""

    def __init__(self, exp, actions, samples):
        self._exp = exp
        self._actions = actions
        self._samples = samples

    def get_experiment(self, experiment_id):
        return self._exp

    def get_actions(self, experiment_id):
        return self._actions

    def get_samples(self, experiment_id):
        return self._samples


def _make_gen(actions):
    exp = {"name": "inj-test", "goal_text": "g", "start_time": "2026-01-01T00:00:00"}
    return ReportGenerator(_StubStorage(exp, actions, []))


def _assert_no_dangerous_calls(script: str) -> None:
    """Parse the generated script and assert no executable Call references
    ``__import__`` or ``system``. A neutralised payload may survive verbatim
    inside a ``# ...`` comment (which the AST ignores) — only executable code
    matters for the injection property.
    """
    tree = ast.parse(script)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            dumped = ast.dump(node)
            assert "system" not in dumped, f"injected system() call: {dumped}"
            assert "__import__" not in dumped, f"injected __import__ call: {dumped}"


def test_malicious_method_name_is_not_emitted_as_code():
    evil = "Scan_Action(); __import__('os').system('echo pwned')#"
    action = ActionRecord(
        experiment_id="e1",
        skill_name="scan",
        nanonis_calls=[NanonisCallRecord(method=evil, args=(1,))],
    )
    script = _make_gen([action]).generate_reproducible_script("e1")

    # The poisoned payload must never become an executable statement.
    _assert_no_dangerous_calls(script)
    # And the only nanonis.* call line must not carry the payload.
    assert "nanonis.Scan_Action(); __import__" not in script
    # It must be neutralised into a comment.
    assert "# SKIPPED non-identifier method name" in script


def test_malicious_kwarg_name_is_not_emitted_as_code():
    action = ActionRecord(
        experiment_id="e1",
        skill_name="scan",
        nanonis_calls=[
            NanonisCallRecord(
                method="Bias_Set",
                kwargs={"x=1)\n__import__('os').system('pwned')#": 1},
            )
        ],
    )
    script = _make_gen([action]).generate_reproducible_script("e1")
    _assert_no_dangerous_calls(script)
    assert "# SKIPPED call with non-identifier kwarg name" in script


def test_valid_method_still_emitted():
    action = ActionRecord(
        experiment_id="e1",
        skill_name="scan",
        nanonis_calls=[
            NanonisCallRecord(method="Bias_Set", args=(-2.0,), kwargs={"wait": True})
        ],
    )
    script = _make_gen([action]).generate_reproducible_script("e1")
    assert "nanonis.Bias_Set(-2.0, wait=True)" in script


def test_generated_script_always_parses_even_with_poisoned_log():
    """Whatever the log contains, the emitted file must be valid Python."""
    actions = [
        ActionRecord(
            experiment_id="e1",
            skill_name="scan",
            nanonis_calls=[
                NanonisCallRecord(method="Bias_Set", args=(-2.0,)),
                NanonisCallRecord(method="evil; rm -rf /"),
                NanonisCallRecord(
                    method="Scan_Action", kwargs={"bad key": "x"}
                ),
            ],
        )
    ]
    script = _make_gen(actions).generate_reproducible_script("e1")
    # Must parse as a complete module — injection would raise SyntaxError or
    # (worse) parse into extra executable nodes — and contain no dangerous call.
    _assert_no_dangerous_calls(script)
    # Sanity: the one valid call survived.
    assert "nanonis.Bias_Set(-2.0)" in script
