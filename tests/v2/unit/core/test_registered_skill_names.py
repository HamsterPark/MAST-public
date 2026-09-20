"""Registry enumeration must agree with callable skill names.

The registry may contain version-to-class dictionaries or SkillMetadata objects.
Tests verify a known present skill, agreement with dispatch in both directions,
plain names instead of object representations, and a genuinely missing skill.
A nonempty set alone cannot prove that enumeration works."""
from __future__ import annotations

# ── path bootstrap ──
import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _name in list(sys.modules):
    if _name == "mast" or _name.startswith("mast."):
        _f = getattr(sys.modules[_name], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_name]

import pytest  # noqa: E402

from mast.core.registry import SkillRegistry, registered_skill_names  # noqa: E402


@pytest.fixture(scope="module")
def live_registry() -> SkillRegistry:
    reg = SkillRegistry()
    reg.discover("mast.skills.builtins", "mast.skills.composite")
    return reg


# ── 1. the scan is not blind ────────────────────────────────────────────────

def test_scan_finds_a_skill_that_is_definitely_registered(live_registry):
    """THE FIRST TEST OF ANY SCAN: prove it can see something that IS there.

    ``GetLatestScanFile`` is named on purpose — it is the one the caught being called successfully on the instrument in the same session where
    this scan declared it missing.
    """
    names = registered_skill_names(live_registry)
    assert live_registry.has("GetLatestScanFile"), "fixture is broken, not the scan"
    assert "GetLatestScanFile" in names
    assert len(names) > 100, f"only {len(names)} names — the scan is under-reading"


# ── 2. it agrees with the thing that actually dispatches ────────────────────

def test_scan_matches_the_dispatcher_exactly(live_registry):
    """Cross-validate against ``registry.has()`` — the dict ``ctx.run()`` looks in.

    Both directions. "Everything I report is dispatchable" alone would pass for
    a scan that reports only one skill; "everything dispatchable is reported"
    alone would pass for a scan that reports the whole dictionary plus junk.
    """
    names = registered_skill_names(live_registry)
    assert all(live_registry.has(n) for n in names)
    assert names == set(live_registry._skills)


# ── 3. names, not reprs — the 2026-08-04 regression, nailed down ────────────

def test_scan_returns_names_not_object_reprs(live_registry):
    """The exact bug: ``str(SkillMetadata(...))`` instead of ``.name``.

    A repr contains characters no skill name can contain. Checking the SHAPE of
    every entry catches this whole family, including the next variant of it,
    which is why this is not just ``assert "ScanAt" in names``.
    """
    names = registered_skill_names(live_registry)
    for n in names:
        assert isinstance(n, str) and n
        assert n.isidentifier(), f"not a skill name: {n[:80]!r}"


def test_metadata_objects_are_read_by_name_attribute():
    """A registry handing back ``SkillMetadata`` must be read via ``.name``."""
    from mast.core.types import SkillMetadata

    class MetadataRegistry:
        def list_skills(self):
            return [SkillMetadata(name="ScanAt"), SkillMetadata(name="SaveScan")]

    assert registered_skill_names(MetadataRegistry()) == {"ScanAt", "SaveScan"}


def test_a_registry_that_lists_bare_names_also_works():
    """The other legal shape, accepted without being coerced through ``str()``."""
    class NameRegistry:
        def list_skills(self):
            return ["ScanAt", "SaveScan"]

    assert registered_skill_names(NameRegistry()) == {"ScanAt", "SaveScan"}


def test_junk_entries_are_dropped_not_stringified():
    """Anything that is neither ``.name`` nor a string is DROPPED.

    Coercing it is what produced the field failure. Dropping it makes the
    affected skill report as missing — a false alarm, which is loud and gets
    fixed, rather than a false name, which is silent and never matches.
    """
    class JunkRegistry:
        def list_skills(self):
            return [object(), 42, None, "ScanAt"]

    assert registered_skill_names(JunkRegistry()) == {"ScanAt"}


# ── 4. it can still say NO ──────────────────────────────────────────────────

def test_scan_reports_a_genuinely_absent_skill_as_absent(live_registry):
    """Sensitivity. Without this, a scan stuck on "yes" passes everything above."""
    names = registered_skill_names(live_registry)
    assert "NoSuchSkillNameEverRegistered" not in names


def test_unregistering_a_skill_removes_it_from_the_scan():
    """The scan tracks the registry's real contents, not a static list."""
    reg = SkillRegistry()
    reg.discover("mast.skills.builtins")
    assert "GetBias" in registered_skill_names(reg)
    reg.unregister("GetBias")
    assert "GetBias" not in registered_skill_names(reg)


# ── fallbacks: never raise, and answer the BUILD question when asked ────────

def test_no_registry_falls_back_to_a_fresh_discovery():
    """A bare context (unit test, CLI probe) still gets the build's real answer."""
    names = registered_skill_names(None)
    assert "GetLatestScanFile" in names
    assert "ScanAt" in names


def test_a_broken_registry_falls_back_to_the_build_answer():
    """An exploding registry must not take the self-check down with it — and
    must not be reported as "nothing is installed" either.

    The fallthrough direction is a deliberate safety choice. A false "present"
    fails LOUDLY at the point of use (``ctx.run`` raises "skill not found" and
    names it); a false "missing" stops the agent SILENTLY before it starts,
    which is the exact failure this whole fix exists to remove. When the live
    answer is unavailable, the build answer is the better of the two errors.
    """
    class ExplodingRegistry:
        def list_skills(self):
            raise RuntimeError("boom")

    assert "ScanAt" in registered_skill_names(ExplodingRegistry())


def test_an_object_that_is_not_a_registry_falls_back_the_same_way():
    class NotARegistry:
        pass

    assert "ScanAt" in registered_skill_names(NotARegistry())


def test_returns_empty_rather_than_raising_when_even_discovery_fails(monkeypatch):
    """Last resort. A self-check that dies while checking is worse than one
    that reports zero — the caller can read an empty set as "unknown"."""
    def boom(self, *packages):
        raise RuntimeError("frozen build, no packages")

    monkeypatch.setattr(SkillRegistry, "discover", boom)
    assert registered_skill_names(None) == set()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-x", "-v"]))
