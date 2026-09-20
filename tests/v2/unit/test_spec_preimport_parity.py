"""mast2.spec's pre-import list must name modules that still exist.

KNOWN_ISSUES §2.4: ``scripts/_check_spec_preimport.py`` imported ``mast.gui.app``
for weeks after ``mast.gui`` was deleted in the TS rewrite. Nothing caught it
because that script is not on ``installer/mast2_build.ps1``'s critical path —
"a checker nobody runs" and "a checker that passes" look identical from outside.

The list now lives in exactly one place (``mast2.spec``) and the checker parses
it. This test is the thing that fails when a module in that list is renamed or
deleted: it resolves every name WITHOUT executing it (``find_spec``), so it is
cheap enough to run in the normal unit suite.

Why find_spec and not import: importing ``mast.agents.orchestrator`` &co. pulls
in LangGraph and six agent packages. Resolution is what the spec actually needs
to be true — an unimportable-but-present module is a different bug, and the
checker script (run manually / at build time) does the real import.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO_ROOT / "mast2.spec"
CHECKER_PATH = REPO_ROOT / "scripts" / "_check_spec_preimport.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location(
        "_check_spec_preimport", CHECKER_PATH
    )
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def preimport_modules() -> list[str]:
    if not SPEC_PATH.exists() or not CHECKER_PATH.exists():
        pytest.skip("packaging tree not present")
    return _load_checker().spec_preimport_modules(str(SPEC_PATH))


def test_spec_preimport_block_is_parsed_not_empty(preimport_modules):
    """A parser that silently returns [] would make the checker a no-op."""
    assert len(preimport_modules) >= 20
    assert all(n.startswith("mast") for n in preimport_modules)


def test_every_preimported_module_still_exists(preimport_modules):
    """The §2.4 regression: `mast.gui.app` outlived `mast.gui` by weeks."""
    missing = []
    for name in preimport_modules:
        try:
            found = importlib.util.find_spec(name)
        except (ImportError, AttributeError, ValueError) as exc:
            missing.append(f"{name} ({type(exc).__name__}: {exc})")
            continue
        if found is None:
            missing.append(name)
    assert not missing, (
        "mast2.spec pre-imports modules that no longer resolve: "
        + ", ".join(missing)
        + " — fix the name in mast2.spec (the spec swallows import errors, so "
        "this only ever shows up as a mysteriously missing module in the frozen "
        "build)."
    )


def test_checker_script_does_not_hardcode_a_second_module_list():
    """The whole point of §2.4's fix: one list, in the spec.

    A future edit that pastes module names back into the checker re-creates the
    drift this test exists to prevent.
    """
    source = CHECKER_PATH.read_text(encoding="utf-8")
    body = source.split('if __name__ == "__main__"')[0]
    # Strip the docstring (it legitimately names mast.gui / mast.webui when
    # explaining the incident) before looking for hardcoded imports.
    if body.count('"""') >= 2:
        body = body.split('"""', 2)[2]
    hardcoded = [
        line for line in body.splitlines()
        if line.strip().startswith(("import mast.", "from mast."))
    ]
    assert not hardcoded, (
        "scripts/_check_spec_preimport.py hardcodes module names again: "
        f"{hardcoded}. Read them from mast2.spec instead."
    )
