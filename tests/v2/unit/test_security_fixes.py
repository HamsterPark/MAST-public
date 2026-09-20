"""Security fixes for review findings [28] [38] [37] [20] (group: security).

Real-logic tests (no mocking of the unit under test):

  [28] llm/skill_author.py — skill-name path-traversal + dunder AST escape +
       exec gated behind explicit allow_exec confirmation.
  [38] update/client.py — manifest.filename path-traversal rejection.
  [37] admin/override_store.py — save_and_reload / save_or_delete_and_reload.
  [20] core/safety.py — malformed admin safety_checks.json no longer crashes
       SafetyGuard construction.

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/test_security_fixes.py -x -v
"""
from __future__ import annotations

# ── path bootstrap (canonical block) ───────────────────────────────────────
import sys
from pathlib import Path
_MASTV2_ROOT = str(Path(__file__).resolve().parents[3] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import json

import pytest

# Module-level imports so the cached `mast` resolves to v2 (MASTv2) BEFORE
# pytest's rootdir-driven sys.path setup re-adds the v1 repo-root copy. (Same
# pattern as tests/v2/unit/test_update.py — in-function imports would resolve
# to the v1 copy once another test file has imported it.)
import mast.llm.skill_author as skill_author_mod
from mast.llm.skill_author import SkillAuthor, _validate_skill_name
import mast.update.client as update_client_mod
from mast.update.client import _is_safe_filename, load_pending, pending_dir, check_and_download
from mast.update.manifest import Manifest, write_manifest
from mast.admin.override_store import ConfigOverrideRegistry
from mast.config import SafetyLimits
from mast.core.safety import SafetyGuard


# ════════════════════════════════════════════════════════════════════════════
# [28] SkillAuthor — name validation, dunder AST block, exec gating
# ════════════════════════════════════════════════════════════════════════════

def test_skill_name_validation_rejects_traversal():
    _validate_skill_name  # use module-level import
    # Each of these previously interpolated straight into the output path /
    # dotted module name → path traversal or real-code overwrite.
    for bad in (
        "../../core/registry",
        "..\\..\\evil",
        "a/b",
        "a\\b",
        ".hidden",
        "has space",
        "has-dash",
        "9starts_with_digit",
        "__dunder__",
        "name.with.dots",
        "",
    ):
        with pytest.raises(ValueError):
            _validate_skill_name(bad)


def test_skill_name_validation_accepts_plain_identifier():
    from mast.llm.skill_author import _validate_skill_name

    for good in ("MyScan", "scan_and_sts", "Skill1", "a", "Foo_Bar_2"):
        _validate_skill_name(good)  # must not raise


def test_ast_check_blocks_subclasses_escape():
    """The classic deny-list bypass via attribute chain must be caught."""
    from mast.llm.skill_author import SkillAuthor

    author = SkillAuthor.__new__(SkillAuthor)  # no client/registry needed
    escape = (
        "x = ().__class__.__bases__[0].__subclasses__()\n"
        "y = (1).__class__\n"
    )
    violations = author._ast_safety_check(escape)
    assert violations, "dunder attribute escape was NOT flagged"
    assert any("dunder" in v.lower() for v in violations)


def test_ast_check_blocks_builtins_reacharound():
    from mast.llm.skill_author import SkillAuthor

    author = SkillAuthor.__new__(SkillAuthor)
    code = "g = (lambda: 0).__globals__\nb = __builtins__\n"
    violations = author._ast_safety_check(code)
    assert violations
    assert any("__globals__" in v or "__builtins__" in v for v in violations)


def test_ast_check_still_allows_simple_skill():
    """A normal skill body (no dunder reach-around) passes the static check."""
    from mast.llm.skill_author import SkillAuthor

    author = SkillAuthor.__new__(SkillAuthor)
    code = (
        "import numpy as np\n"
        "class Foo:\n"
        "    def execute(self, ctx, params):\n"
        "        return np.mean([1, 2, 3])\n"
    )
    assert author._ast_safety_check(code) == []


def test_create_skill_does_not_exec_by_default(tmp_path, monkeypatch):
    """allow_exec defaults False: code is written but never exec'd/registered."""
    import mast.llm.skill_author as sa

    # Redirect the custom-skills dir into a temp location so we don't touch the
    # real package, and confirm exec_module is never called.
    monkeypatch.setattr(sa, "_CUSTOM_SKILLS_DIR", tmp_path / "custom")

    generated = "class Foo:\n    pass\n"

    class _FakeClient:
        def single_turn(self, prompt, system):
            return generated

    class _FakeRegistry:
        def __init__(self):
            self.registered = []

        def list_skills(self):
            return []

        def register(self, obj):
            self.registered.append(obj)

    reg = _FakeRegistry()
    author = sa.SkillAuthor(_FakeClient(), reg)

    # Sabotage exec_module so any attempt to run it fails loudly.
    def _boom(*a, **k):
        raise AssertionError("exec_module must NOT run when allow_exec=False")

    monkeypatch.setattr(sa.importlib.util, "spec_from_file_location",
                        lambda *a, **k: (_ for _ in ()).throw(
                            AssertionError("spec creation must not run")))

    code = author.create_skill("desc", "MySkill")  # allow_exec defaults False
    # _extract_code strips surrounding whitespace; compare on stripped form.
    assert code == generated.strip()
    assert reg.registered == []  # nothing registered
    # File written for human review (the stripped code).
    written = (tmp_path / "custom" / "MySkill.py").read_text(encoding="utf-8")
    assert written == generated.strip()


def test_create_skill_rejects_bad_name_before_llm(tmp_path):
    import mast.llm.skill_author as sa

    class _FakeClient:
        def single_turn(self, prompt, system):
            raise AssertionError("LLM must not be called for a bad name")

    class _FakeRegistry:
        def list_skills(self):
            return []

    author = sa.SkillAuthor(_FakeClient(), _FakeRegistry())
    with pytest.raises(ValueError):
        author.create_skill("desc", "../../evil")


# ════════════════════════════════════════════════════════════════════════════
# [38] update client — manifest.filename path-traversal rejection
# ════════════════════════════════════════════════════════════════════════════

def test_is_safe_filename_rejects_traversal():
    from mast.update.client import _is_safe_filename

    for bad in (
        "../../evil.exe",
        "..\\..\\evil.exe",
        "sub/dir.exe",
        "sub\\dir.exe",
        "/abs/evil.exe",
        "\\abs\\evil.exe",
        ".hidden",
        "C:evil.exe",
        "\\\\host\\share\\evil.exe",
        "with\x00nul.exe",
        "",
    ):
        assert not _is_safe_filename(bad), f"{bad!r} should be rejected"


def test_is_safe_filename_accepts_plain_name():
    from mast.update.client import _is_safe_filename

    for good in (
        "MAST2-Setup-v2.0.0-windows-x64.exe",
        "delta_2.0.0_to_2.0.1.zip",
        "setup.exe",
    ):
        assert _is_safe_filename(good), f"{good!r} should be accepted"


def test_load_pending_rejects_traversal_manifest(tmp_path):
    """A pending manifest naming a traversal filename is discarded, not honored."""
    from mast.update.client import load_pending, pending_dir
    from mast.update.manifest import Manifest, write_manifest

    pdir = pending_dir(tmp_path)
    # Plant a file OUTSIDE the pending dir that the bad filename would resolve to.
    outside = tmp_path / "evil.exe"
    outside.write_bytes(b"x" * 10)

    bad = Manifest(
        version="9.9.9",
        filename="..\\evil.exe",  # escapes pending dir
        sha256="00" * 32,
        size_bytes=10,
        published_at="2026-05-30T00:00:00+00:00",
    )
    write_manifest(pdir / "manifest.json", bad)

    m, setup = load_pending(tmp_path)
    assert m is None and setup is None


def test_check_and_download_rejects_traversal_manifest(tmp_path, monkeypatch):
    """End-to-end: a server manifest with a traversal filename is refused
    before any file is written outside the pending dir."""
    import mast.update.client as client

    bad_manifest = {
        "version": "9.9.9",
        "filename": "..\\..\\evil.exe",
        "sha256": "00" * 32,
        "size_bytes": 10,
        "published_at": "2026-05-30T00:00:00+00:00",
    }

    class _Resp:
        status_code = 200
        text = ""

        def json(self):
            return bad_manifest

    class _FakeClientCtx:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None):
            return _Resp()

    # Patch only the network client; the function under test runs for real.
    import httpx
    monkeypatch.setattr(httpx, "Client", _FakeClientCtx)

    # Isolate from the developer machine: with a real release pubkey configured
    # (push_defaults_mast2.json after keygen), the missing-signature rejection
    # fires BEFORE the filename check. Clear it so this test exercises the
    # traversal guard specifically.
    import mast.update.signing as signing
    monkeypatch.setattr(signing, "get_release_public_key", lambda: None)

    # Use an https:// URL so the HTTPS-enforcement guard passes and we exercise
    # the manifest-filename traversal check specifically (the network client is
    # mocked, so no real connection is made).
    status, detail = client.check_and_download(
        "https://server", "tok", tmp_path, current_version="1.0.0",
    )
    assert status == "error"
    assert "bad filename" in detail
    # Nothing escaped the pending dir.
    assert not (tmp_path.parent / "evil.exe").exists()


# ════════════════════════════════════════════════════════════════════════════
# [37] ConfigOverrideRegistry.save_and_reload / save_or_delete_and_reload
# ════════════════════════════════════════════════════════════════════════════

def test_save_and_reload_writes_file_and_reload_sentinel(tmp_path):
    from mast.admin.override_store import ConfigOverrideRegistry

    reg = ConfigOverrideRegistry(overrides_dir=tmp_path)
    reg.save_and_reload("safety_limits.json", {"bias_max_v": 5.0})

    saved = tmp_path / "safety_limits.json"
    assert saved.exists()
    assert json.loads(saved.read_text(encoding="utf-8")) == {"bias_max_v": 5.0}
    # reload sentinel was dropped
    assert (tmp_path / ".reload").exists()


def test_save_or_delete_and_reload_deletes_on_empty(tmp_path):
    from mast.admin.override_store import ConfigOverrideRegistry

    reg = ConfigOverrideRegistry(overrides_dir=tmp_path)
    reg.save("skill_overrides.json", {"SomeSkill": {"safety_level": "dangerous"}})
    assert (tmp_path / "skill_overrides.json").exists()

    # Empty payload → file removed (revert to code defaults).
    reg.save_or_delete_and_reload("skill_overrides.json", {})
    assert not (tmp_path / "skill_overrides.json").exists()
    assert (tmp_path / ".reload").exists()


def test_save_or_delete_and_reload_saves_on_nonempty(tmp_path):
    from mast.admin.override_store import ConfigOverrideRegistry

    reg = ConfigOverrideRegistry(overrides_dir=tmp_path)
    reg.save_or_delete_and_reload("safety_constraints.json", {"Au111": {"bias_max_v": 2.0}})
    assert (tmp_path / "safety_constraints.json").exists()


# ════════════════════════════════════════════════════════════════════════════
# [20] Malformed safety_checks.json must not crash SafetyGuard construction
# ════════════════════════════════════════════════════════════════════════════

def _registry_with_checks(tmp_path, checks_data):
    from mast.admin.override_store import ConfigOverrideRegistry
    ConfigOverrideRegistry.reset()
    reg = ConfigOverrideRegistry(overrides_dir=tmp_path)
    reg.save("safety_checks.json", checks_data)
    # The singleton is what core.safety reads via ConfigOverrideRegistry.get().
    ConfigOverrideRegistry._instance = reg
    return reg


def test_safetyguard_survives_bad_min_attr(tmp_path):
    """An addition with a typo'd min_attr must be dropped, not crash __init__."""
    from mast.config import SafetyLimits
    from mast.admin.override_store import ConfigOverrideRegistry

    try:
        _registry_with_checks(tmp_path, {
            "additions": [
                {"pattern": "foo_v", "unit": "v",
                 "min_attr": "typo_does_not_exist", "max_attr": "bias_max_v"},
            ],
        })
        from mast.core.safety import SafetyGuard
        # Previously raised AttributeError here → whole executor disabled.
        guard = SafetyGuard(SafetyLimits())
        # The bad entry was dropped; built-in checks survive.
        patterns = {c[0] for c in guard._resolved_checks}
        assert "foo_v" not in patterns
        assert "bias_v" in patterns  # genuine check still present
    finally:
        ConfigOverrideRegistry.reset()


def test_safetyguard_survives_missing_keys(tmp_path):
    from mast.config import SafetyLimits
    from mast.admin.override_store import ConfigOverrideRegistry

    try:
        _registry_with_checks(tmp_path, {
            "additions": [
                {"pattern": "foo_v"},          # missing unit/min_attr/max_attr
                "not even a dict",             # garbage
                {"pattern": "bar_v", "unit": "v",
                 "min_attr": "bias_min_v", "max_attr": "bias_max_v"},  # valid
            ],
        })
        from mast.core.safety import SafetyGuard
        guard = SafetyGuard(SafetyLimits())  # must not raise
        patterns = {c[0] for c in guard._resolved_checks}
        assert "foo_v" not in patterns      # dropped (missing keys)
        assert "bar_v" in patterns          # kept (valid)
    finally:
        ConfigOverrideRegistry.reset()


def test_safetyguard_bad_override_keeps_builtin(tmp_path):
    """A bad 'overrides' entry must not corrupt the matching built-in check."""
    from mast.config import SafetyLimits
    from mast.admin.override_store import ConfigOverrideRegistry

    try:
        _registry_with_checks(tmp_path, {
            "overrides": [
                {"pattern": "bias_v", "unit": "v",
                 "min_attr": "nope", "max_attr": "nope2"},
            ],
        })
        from mast.core.safety import SafetyGuard
        guard = SafetyGuard(SafetyLimits())  # must not raise
        # bias_v built-in check survives with its real attrs (resolved values).
        bias_checks = [c for c in guard._resolved_checks if c[0] == "bias_v"]
        assert bias_checks, "bias_v check was lost"
        _, _, gmin, gmax = bias_checks[0]
        assert gmin == SafetyLimits().bias_min_v
        assert gmax == SafetyLimits().bias_max_v
    finally:
        ConfigOverrideRegistry.reset()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
