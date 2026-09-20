"""The advanced-capability gate + the PIN that guards it.

These four are not "hardware you may not own" — they are powers that step around a
protection. So the gate is the same mechanism as the hardware modules (off ⇒ the
skills are not in the agent's tool list at all) with a harder lock: the write path
is PIN-checked and FAILS CLOSED when no PIN is set.

The single most important test in this file is
``test_load_refuses_to_overwrite_a_vetted_slot``. The Nanonis script allow-list is
the ONLY barrier for scripts — they run on the real-time controller, where MAST's
safety gate, mode gate, abort gate and HITL are all blind. The allow-list approves
*a script, in a slot*. If a Load can replace the contents of a vetted slot, the
approval means nothing and the barrier is gone, even though the file still lists the
slot and everything still looks fine.
"""

from __future__ import annotations

import hashlib

import pytest

from mast.agents.instrument_control.tools import (
    build_instrument_skill_tools,
    discover_instrument_skills,
)
from mast.core.types import SafetyLevel
from mast.skills import advanced_capabilities as ac
from mast.skills import hardware_modules as hm
from mast.skills.builtins.nanonis_script_files import LoadNanonisScript


@pytest.fixture(autouse=True)
def _restore():
    before_ac, before_hm = ac.enabled_ids(), hm.enabled_ids()
    yield
    ac.set_enabled(before_ac)
    hm.set_enabled(before_hm)


@pytest.fixture(scope="module")
def registry():
    return discover_instrument_skills()


class _Rec:
    def __init__(self, body=(0.0,), error=None):
        self.return_value = [0, 0, list(body)]
        self.error = error


class FakeCtx:
    def __init__(self):
        self.calls: list[str] = []

    def safe_call(self, verb, *args, **kw):
        self.calls.append(verb)
        return _Rec()

    def check_abort(self):
        return False


# ─────────────────────────────────────────────────────────────────────────────
# Registry ↔ reality — the same invariant that keeps the hardware gate honest
# ─────────────────────────────────────────────────────────────────────────────

def test_every_registered_skill_name_actually_exists(registry):
    real = {m.name for m in registry.list_skills()}
    missing = sorted(set(ac.SKILL_OWNER) - real)
    assert not missing, (
        f"advanced_capabilities.CAPABILITIES 里这些 skill 不存在：{missing}\n"
        "门控按名字匹配——名字错了就等于没门控。"
    )


def test_every_capability_ships_off():
    on = [c.id for c in ac.CAPABILITIES if c.default_on]
    assert not on, f"这些高级能力默认开着：{on}。它们每一个都能绕过某道保护，必须默认全关。"


def test_every_capability_states_which_protection_it_steps_around():
    for c in ac.CAPABILITIES:
        assert c.risk.strip(), f"{c.id} 没写它绕过了哪道保护"
        assert c.description.strip(), f"{c.id} 没有说明"


def test_no_overlap_with_the_hardware_module_gate():
    """Two gates, two settings keys. A skill owned by both would be switchable from
    two places with different rules — and one of them would silently win."""
    both = sorted(set(ac.SKILL_OWNER) & set(hm.SKILL_OWNER))
    assert not both, f"这些 skill 同时被两个门认领：{both}"


def test_disabled_capabilities_are_withheld_from_the_agent(registry):
    ac.set_enabled({})
    hm.set_enabled({})
    names_off = {t.name for t in build_instrument_skill_tools(registry, lambda: None)}
    for skill in ac.SKILL_OWNER:
        assert skill not in names_off, f"{skill} 属于关闭的高级能力，却在 agent 工具表里"

    ac.set_enabled({"quit_nanonis": True})
    names = {t.name for t in build_instrument_skill_tools(registry, lambda: None)}
    assert "QuitNanonis" in names
    assert "LoadNanonisScript" not in names, "只开了 quit_nanonis，脚本能力却也进来了"


def test_multipass_MODE_is_not_gated_only_the_config_FILES_are(registry):
    """SetMultiPass is an ordinary scan feature — scanning a line twice at different
    biases. It is the CONFIG FILE I/O that needed the gate, not the mode."""
    ac.set_enabled({})
    hm.set_enabled({})
    names = {t.name for t in build_instrument_skill_tools(registry, lambda: None)}
    assert "SetMultiPass" in names, "多程扫描本身是普通扫描功能，不该被门控挡掉"
    assert "LoadMultiPassConfig" not in names


@pytest.mark.parametrize("garbage", ["quit_nanonis", 42, 3.14, True])
def test_unparseable_state_falls_back_to_all_off(garbage):
    ac.set_enabled(garbage)
    assert ac.enabled_ids() == frozenset()


# ─────────────────────────────────────────────────────────────────────────────
# THE one: a Load must not launder a vetted slot
# ─────────────────────────────────────────────────────────────────────────────

def test_load_refuses_to_overwrite_a_vetted_slot(monkeypatch):
    """The allow-list approves A SCRIPT, IN A SLOT — not a slot number.

    A Nanonis script runs on the real-time controller and issues no safe_call, so
    SafetyGate / the mode gate / the abort gate / HITL are ALL blind to what it does.
    The allow-list is the only barrier there is. If LoadNanonisScript could put a
    different file into slot 3 after a human vetted slot 3, the barrier would still
    be there on paper and gone in fact — and nothing would look wrong.
    """
    monkeypatch.setattr(
        "mast.skills.builtins.nanonis_script_files.load_allowlist",
        lambda: {3: {"slot": 3, "name": "vetted_recipe"}},
    )
    ctx = FakeCtx()
    r = LoadNanonisScript().execute(ctx, {"slot": 3, "file_path": "C:/other.ns"})
    assert r.success is False
    assert "Script_Load" not in ctx.calls, "拒绝了，却还是发出了 Script_Load"
    assert "白名单" in (r.error or "")


def test_load_into_an_unvetted_slot_is_allowed_but_not_runnable(monkeypatch):
    """And that is not a dead end — it is the right division of labour: the agent does
    the mechanical file-loading, the human does the approving."""
    monkeypatch.setattr(
        "mast.skills.builtins.nanonis_script_files.load_allowlist",
        lambda: {3: {"slot": 3}},
    )
    ctx = FakeCtx()
    r = LoadNanonisScript().execute(ctx, {"slot": 7, "file_path": "C:/new.ns"})
    assert r.success is True
    assert "Script_Load" in ctx.calls
    assert r.data["runnable"] is False
    assert "还不能运行" in (r.summary or "")


def test_load_is_dangerous_and_quit_is_dangerous(registry):
    for name in ("LoadNanonisScript", "QuitNanonis", "LoadMultiPassConfig"):
        meta = next(m for m in registry.list_skills() if m.name == name)
        assert meta.safety_level == SafetyLevel.DANGEROUS, (
            f"{name} 是 {meta.safety_level.name}——它能绕过一道保护，必须 DANGEROUS（进 HITL）"
        )


# ─────────────────────────────────────────────────────────────────────────────
# The PIN — it used to be dead code
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def pin_root(tmp_path, monkeypatch):
    monkeypatch.setattr("mast._runtime_paths.project_root", lambda: tmp_path)
    return tmp_path


def test_no_pin_set_means_REFUSED_not_wide_open(pin_root):
    """Fail-closed. Handing an agent the ability to overwrite a vetted script slot
    should not happen on a machine where nobody has yet claimed to be the admin."""
    from mast.api.admin_pin import pin_is_set, verify_pin
    assert pin_is_set() is False
    ok, reason = verify_pin("anything")
    assert ok is False and reason == "no_pin_set"


def test_set_then_verify(pin_root):
    from mast.api.admin_pin import pin_is_set, set_pin, verify_pin
    assert set_pin("1234") == (True, "")
    assert pin_is_set() is True
    assert verify_pin("1234")[0] is True
    assert verify_pin("9999") == (False, "wrong")
    assert verify_pin("") == (False, "empty")


def test_only_the_hash_reaches_the_disk(pin_root):
    from mast.api.admin_pin import set_pin
    set_pin("hunter2")
    written = (pin_root / "config" / "admin_pin.txt").read_text(encoding="utf-8").strip()
    assert written == hashlib.sha256(b"hunter2").hexdigest()
    assert "hunter2" not in written


def test_changing_a_pin_needs_the_current_one(pin_root):
    from mast.api.admin_pin import set_pin, verify_pin
    set_pin("1234")
    assert set_pin("5678") == (False, "wrong_current")
    assert set_pin("5678", current_pin="1234") == (True, "")
    assert verify_pin("5678")[0] is True


def test_too_short_is_refused(pin_root):
    from mast.api.admin_pin import set_pin
    assert set_pin("12") == (False, "too_short")


def test_guarded_keys_are_exactly_the_known_set():
    """Guard the guard: if a key is dropped from this set, its toggle silently stops
    needing the PIN and nothing fails.

    Two kinds of key live here now. ``advanced_capabilities`` and
    ``hardware_modules`` grant the AGENT a power it does not have by default.
    ``coarse_drive`` (2026-07-31) is different in kind: it is a CLAIM ABOUT THE
    HARDWARE that nothing can verify — the controller will happily output a
    voltage the piezo stack does not survive, and no reading says which rig you
    are on. It is guarded for the same practical reason (a stray human hand, not
    the model — the model has no path to this API) and inherits the same
    fail-closed rule: with no PIN configured it cannot be written at all, so an
    unattended rig stays "undeclared", which refuses every drive write.

    ``tip_conditioning_overrides`` (2026-08-10) is a third kind, and the closest
    to ``hardware_modules``: it can WIDEN a hardware envelope. ``max_abs_pulse_v``
    and ``max_poke_depth_m`` live in that table, and those two numbers decide
    which pulses and which plunges get refused. One mis-click raises the ceiling
    on a physical action that cannot be undone (a quartz tuning fork does not
    come back). It got its write path the same day — before that the key was
    read on every resolve and writable by nothing, so the operator could not
    change it at all.

    ⚠️ Guarding it is a TRADE-OFF, not obviously right: it also means a rig with
    no PIN configured can never change the tip policy. Listed for the operator to
    overrule — see docs/v2/design/forge_scan_working_point.md §八."""
    from mast.api.admin_pin import GUARDED_KEYS
    assert GUARDED_KEYS == {"advanced_capabilities", "hardware_modules",
                            "coarse_drive", "tip_conditioning_overrides"}, (
        f"受 PIN 保护的键变了：{sorted(GUARDED_KEYS)}。"
        "少一个就等于那个开关/声明不再需要 PIN。"
    )


def test_settings_key_is_in_known_keys():
    from mast.webui.settings_store import KNOWN_KEYS
    assert ac.SETTINGS_KEY in KNOWN_KEYS, (
        f"{ac.SETTINGS_KEY!r} 不在 KNOWN_KEYS 里——开关会被静默丢弃，重启即失效。"
    )
