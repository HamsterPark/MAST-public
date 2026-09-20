"""The optional-hardware module gate.

The gate is a NAME MATCH between a hand-written registry and a discovered skill
class. That is exactly the kind of coupling that rots silently: mistype one name in
``hardware_modules.MODULES`` and the gate withholds nothing, the settings page
still renders a satisfying green toggle, and the agent quietly keeps a KPFM tool it
should not have. Nothing errors. So the first two tests here are the load-bearing
ones — they pin the registry to reality in both directions.
"""

from __future__ import annotations

import pytest

from mast.agents.instrument_control.tools import (
    build_instrument_skill_tools,
    discover_instrument_skills,
)
from mast.core.execution_context import _ABORT_SAFE_WRITES
from mast.skills import hardware_modules as hm


@pytest.fixture(autouse=True)
def _restore_holder():
    """Every test here mutates the process-level holder. Put it back."""
    before = hm.enabled_ids()
    yield
    hm.set_enabled(before)


@pytest.fixture(scope="module")
def registry():
    return discover_instrument_skills()


# ─────────────────────────────────────────────────────────────────────────────
# The registry ↔ reality invariant — both directions
# ─────────────────────────────────────────────────────────────────────────────

def test_every_registered_skill_name_actually_exists(registry):
    """A typo'd name in MODULES gates NOTHING and reports success. Catch it here."""
    real = {m.name for m in registry.list_skills()}
    missing = sorted(set(hm.SKILL_OWNER) - real)
    assert not missing, (
        f"hardware_modules.MODULES 里这些 skill 名字不存在：{missing}\n"
        "门控按名字匹配——名字错了就等于没门控，而设置页依然显示开关正常。"
    )


def test_every_optional_hardware_skill_is_owned(registry):
    """The reverse: a skill tagged optional-hardware that no module claims is a skill
    that can NEVER be gated off. It would sit in the agent's tool list forever."""
    orphans = sorted(
        m.name for m in registry.list_skills()
        if "optional-hardware" in (m.tags or []) and m.name not in hm.SKILL_OWNER
    )
    assert not orphans, (
        f"这些 skill 标了 optional-hardware 但没有模块认领：{orphans}\n"
        "无人认领 = 永远关不掉 = 永远占着 agent 的工具表。"
    )


def test_no_skill_is_owned_by_two_modules():
    seen: dict[str, str] = {}
    dupes = []
    for mod in hm.MODULES:
        for skill in mod.skills:
            if skill in seen:
                dupes.append(f"{skill}（{seen[skill]} 与 {mod.id}）")
            seen[skill] = mod.id
    assert not dupes, f"skill 被两个模块同时认领：{dupes}"


def test_module_ids_are_unique():
    ids = [m.id for m in hm.MODULES]
    assert len(ids) == len(set(ids)), f"模块 id 重复：{ids}"


# ─────────────────────────────────────────────────────────────────────────────
# Off by default — the whole premise
# ─────────────────────────────────────────────────────────────────────────────

def test_every_optional_module_ships_off():
    on = [m.id for m in hm.MODULES if m.default_on]
    assert not on, (
        f"这些模块默认开着：{on}。用户没有这些硬件——默认必须全关。"
    )


def test_default_holder_state_is_empty():
    hm.set_enabled(None)
    assert hm.enabled_ids() == frozenset()


def test_disabled_skills_are_withheld_from_the_agent(registry):
    """The actual point of the feature: off ⇒ not in the tool list at all."""
    hm.set_enabled({})   # everything off
    def ctx():
        return None
    names_off = {t.name for t in build_instrument_skill_tools(registry, ctx)}
    for skill in hm.SKILL_OWNER:
        assert skill not in names_off, f"{skill} 属于关闭的模块，却出现在 agent 工具表里"

    hm.set_enabled({"laser": True})
    names_laser = {t.name for t in build_instrument_skill_tools(registry, ctx)}
    for skill in hm.MODULE_BY_ID["laser"].skills:
        assert skill in names_laser, f"laser 模块已开，{skill} 却不在工具表里"
    # And only that module came back.
    for skill in hm.MODULE_BY_ID["multiprobe"].skills:
        assert skill not in names_laser, f"只开了 laser，{skill}（多探针）却也进来了"


def test_base_skills_are_never_gated(registry):
    """Gating a base skill would mean the agent cannot drive the microscope."""
    hm.set_enabled({})
    def ctx():
        return None
    names = {t.name for t in build_instrument_skill_tools(registry, ctx)}
    for essential in ("SetBias", "Withdraw", "StartScan"):
        if any(m.name == essential for m in registry.list_skills()):
            assert essential in names, f"基础 skill {essential} 被门控挡掉了"


# ─────────────────────────────────────────────────────────────────────────────
# The holder: fail-closed on garbage
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("garbage", ["laser", 42, 3.14, True, object()])
def test_unparseable_state_falls_back_to_defaults_not_everything_on(garbage):
    """An unknown stored state is not 'all hardware present'. A bare string is the
    nastiest case: it is iterable, so a naive set() would enable modules named 'l',
    'a', 's'… — or, worse, silently succeed."""
    hm.set_enabled(garbage)
    assert hm.enabled_ids() == hm.DEFAULT_ENABLED == frozenset()


def test_unknown_ids_are_dropped_not_kept():
    hm.set_enabled({"laser": True, "no_such_module": True})
    assert hm.enabled_ids() == frozenset({"laser"})


def test_false_values_disable():
    hm.set_enabled({"laser": True, "kelvin": False})
    assert hm.enabled_ids() == frozenset({"laser"})


def test_accepts_a_plain_id_list():
    hm.set_enabled(["laser", "kelvin"])
    assert hm.enabled_ids() == frozenset({"laser", "kelvin"})


def test_module_states_reports_live_truth():
    hm.set_enabled({"osci_hr": True})
    states = {s["id"]: s for s in hm.module_states()}
    assert states["osci_hr"]["enabled"] is True
    assert states["laser"]["enabled"] is False
    assert states["osci_hr"]["skill_count"] == len(hm.MODULE_BY_ID["osci_hr"].skills)
    # Every module tells the operator what hardware it actually needs.
    for s in states.values():
        assert s["hardware"].strip(), f"{s['id']} 没说明需要什么硬件"


# ─────────────────────────────────────────────────────────────────────────────
# Stops must survive an abort, gate or no gate
# ─────────────────────────────────────────────────────────────────────────────

def test_optional_hardware_stops_are_abort_safe():
    """The module gate decides what the agent can SEE. It must never decide what an
    abort can STOP. If the operator has a multi-probe rig and presses 中止, the
    probes must retract — an abort gate that refused the retract would be the worst
    possible failure."""
    for verb in ("HSSwp_Stop", "APRFGen_SwpStop", "PLLPhasSwp_Stop",
                 "MProbeScanner_Stop", "MProbeZCtrl_Withdraw"):
        assert verb in _ABORT_SAFE_WRITES, f"{verb} 不在中止白名单里——按下中止后它会被拒"
    # And the overloaded ones only in their OFF form.
    assert _ABORT_SAFE_WRITES["APRFGen_RFOutOnOffSet"] == (0, frozenset({0}))
    assert _ABORT_SAFE_WRITES["Laser_OnOffSet"] == (0, frozenset({0}))


def test_feedback_loop_offs_are_NOT_abort_safe():
    """Switching a feedback loop off is not a stop — it parks whatever the loop was
    holding with nothing holding it. The safe post-abort action for a Z loop is
    Withdraw. This test pins that distinction so nobody 'helpfully' adds them."""
    for verb in ("MProbeZCtrl_OnOffSet", "KelvinCtrl_CtrlOnOffSet",
                 "PICtrl_OnOffSet", "Interf_CtrlOnOffSet", "ZCtrl_OnOffSet"):
        assert verb not in _ABORT_SAFE_WRITES, (
            f"{verb} 进了中止白名单。关反馈环不是「停止」——它把环原本托着的东西"
            "原地松手。中止后 Z 环的正确动作是 Withdraw。"
        )


def _real_nanonis_verbs() -> set[str]:
    """The REAL nanonis_spm API surface.

    conftest replaces ``nanonis_spm`` with a MagicMock so the suite runs without
    hardware — and a MagicMock's ``dir()`` answers yes to everything. Asking IT
    whether a verb exists is a test that can never fail, which is worse than no
    test. So load the real module from disk, past the mock in sys.modules.
    """
    import importlib.util
    import sysconfig
    from pathlib import Path

    # find_spec() consults sys.modules first and the mock has no __spec__, so go
    # straight to the file on disk.
    site = Path(sysconfig.get_paths()["purelib"])
    src = site / "nanonis_spm" / "NanonisClass.py"
    if not src.is_file():  # pragma: no cover
        # NOT a skip. A skip here reads as a pass, and this is the one probe that
        # keeps the phantom-verb test honest — if it cannot run, say so loudly.
        raise AssertionError(f"找不到真实的 nanonis_spm：{src}（幽灵动词检测形同虚设）")
    spec = importlib.util.spec_from_file_location("_real_nanonis_probe", src)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # NOT put into sys.modules — the mock stays intact
    return set(dir(mod.Nanonis))


def test_the_real_api_probe_is_not_a_mock():
    """Guard the guard. If conftest's mock ever leaks into _real_nanonis_verbs, the
    phantom-verb test below silently becomes a no-op."""
    verbs = _real_nanonis_verbs()
    assert "ZCtrl_Withdraw" in verbs           # a verb that really exists
    assert "Definitely_Not_A_Verb" not in verbs  # a MagicMock would say yes here
    assert len(verbs) > 500, f"只看到 {len(verbs)} 个动词——真实 API 有 600+，这多半还是 mock"


def test_abort_safe_verbs_all_exist_in_the_real_api():
    """Phantom entries in the allow-list are dead weight that reads as protection.
    (Two of them — BiasSpectrMLS_Stop, GenSweep_Stop — were exactly that until
    property testing found they do not exist.)"""
    real = _real_nanonis_verbs()
    phantom = sorted(v for v in _ABORT_SAFE_WRITES if v not in real)
    assert not phantom, f"中止白名单里这些动词在真实 API 里不存在：{phantom}"


# ─────────────────────────────────────────────────────────────────────────────
# The settings key must be persistable — the silent-no-op trap
# ─────────────────────────────────────────────────────────────────────────────

def test_settings_key_is_in_known_keys():
    """A settings key outside SettingsStore.KNOWN_KEYS is dropped on save. It looks
    like it works and does nothing. That trap once made the admin PIN gate accept
    any text; here it would mean the toggles never persist across a restart."""
    from mast.webui.settings_store import KNOWN_KEYS
    assert hm.SETTINGS_KEY in KNOWN_KEYS, (
        f"{hm.SETTINGS_KEY!r} 不在 KNOWN_KEYS 里——开关会被静默丢弃，重启即失效。"
    )


def test_settings_key_round_trips_through_the_store(tmp_path):
    from mast.webui.settings_store import SettingsStore
    store = SettingsStore(tmp_path)
    store.update(**{hm.SETTINGS_KEY: {"laser": True, "kelvin": False}})
    assert SettingsStore(tmp_path).get(hm.SETTINGS_KEY) == {"laser": True, "kelvin": False}


# ─────────────────────────────────────────────────────────────────────────────
# Every optional skill carries a safety level and says what it drives
# ─────────────────────────────────────────────────────────────────────────────

def test_optional_skills_all_declare_safety(registry):
    for name in hm.SKILL_OWNER:
        meta = next(m for m in registry.list_skills() if m.name == name)
        assert meta.safety_level is not None, f"{name} 没有 safety_level"
        assert meta.description.strip(), f"{name} 没有描述"


def test_the_unbounded_hazards_are_dangerous(registry):
    """DANGEROUS is reserved for the hazards Nanonis' own limits do NOT bound.

    The wider safety model (2026-06-11 re-scoping) is that MOST hardware writes are
    safe to leave un-gated because Nanonis bounds them — which is why TipShape, a
    skill that deliberately crashes the tip into the surface, is merely AUTO. These
    four are the ones that premise does not cover. They must never quietly drop to
    AUTO: DANGEROUS is what makes an autonomous run STOP and ask a human.

    The authoritative list lives in tests/v2/agents/instrument_control/
    test_hitl_derivation.py (EXPECTED_DANGEROUS) — this is the reason for each.
    """
    from mast.core.types import SafetyLevel
    unbounded = {
        "SetLaserOnOff":        "人眼伤害；Nanonis 不 bound 光",
        "StartRfGenerator":     "dBm 远超结的耐受，「在量程内」与「安全」无关",
        "RunRfFrequencySweep":  "同上，且整个扫描期间都在出功率",
        "MoveProbeXY":          "探针互撞——Nanonis 不知道别的探针在哪",
        "SetPiControllerOnOff": "把环闭到任意输出上，可驱动压电且无接触停止",
    }
    for name, why in sorted(unbounded.items()):
        meta = next((m for m in registry.list_skills() if m.name == name), None)
        assert meta is not None, f"{name} 不存在"
        assert meta.safety_level == SafetyLevel.DANGEROUS, (
            f"{name} 的 safety_level 是 {meta.safety_level.name}，应为 DANGEROUS：{why}"
        )


def test_the_bounded_ones_are_not_over_gated(registry):
    """The inverse, and it matters just as much. Gating a bounded action as DANGEROUS
    means an autonomous run halts to ask a human about something Nanonis already
    prevents — the operator learns to click 'approve' without reading, and the gate
    stops being a gate. PulseProbeBias is the sharpest case: its core twin BiasPulse
    is AUTO, so gating the multi-probe copy harder would be incoherent."""
    from mast.core.types import SafetyLevel
    bounded = ("RunHighSpeedSweep", "SetKelvinControllerOnOff", "RunCpdCompensation",
               "SetProbeZController", "PulseProbeBias", "SetGenericPiOutput")
    for name in bounded:
        meta = next((m for m in registry.list_skills() if m.name == name), None)
        assert meta is not None, f"{name} 不存在"
        assert meta.safety_level == SafetyLevel.CONFIRM, (
            f"{name} 是 {meta.safety_level.name}；它的危害被 Nanonis 限值 bound 住，"
            "应为 CONFIRM。过度门控会训练用户盲目点「批准」。"
        )


def test_no_skill_claims_a_gate_it_does_not_have(registry):
    """A description that says 'DANGEROUS' or a 'dangerous' tag on a CONFIRM skill is
    a lie the model reads on every turn. Scoped to the optional set — three core
    skills (MotorMove/MotorMoveClosedLoop/SetBiasCalibration) carry the tag
    descriptively, as the parameter-gated hazard class, which predates this."""
    from mast.core.types import SafetyLevel
    for name in hm.SKILL_OWNER:
        meta = next(m for m in registry.list_skills() if m.name == name)
        if meta.safety_level == SafetyLevel.DANGEROUS:
            continue
        assert "dangerous" not in (meta.tags or []), f"{name} 不是 DANGEROUS 却带 dangerous 标签"
        assert "DANGEROUS" not in meta.description, (
            f"{name} 的描述里写着 DANGEROUS，但它的 safety_level 是 "
            f"{meta.safety_level.name} —— 描述在向模型承诺一个并不存在的人工闸门。"
        )
        # 上面那道只认英文大写 token。SKILL_OWNER 这 58 条描述现在全是中文写的
        # （2026-08-24 技能描述中文化），所以最可能犯这个错的写法是中文的「危险」。
        # 双向变异实测（2026-08-25）：注入 "DANGEROUS" 会报红，注入「危险」不会 ——
        # 守卫的中文那一侧是空的。补上，两边才都响。
        # 想描述性地讲危害，用「危害」「有风险」这类不与 DANGEROUS 档位同名的词。
        zh_marker = next((w for w in ("危险", "高危") if w in meta.description), None)
        assert zh_marker is None, (
            f"{name} 的描述里写着「{zh_marker}」，但它的 safety_level 是 "
            f"{meta.safety_level.name} —— 描述在向模型承诺一个并不存在的人工闸门。"
        )
        # 参数说明和技能自述进的是**同一个** LLM schema，同一句谎话在那里同样成立 ——
        # 而上面几道断言只看 meta.description，参数面此前任何语言都没人看。
        # 实测（2026-08-25）扩过来代价为零：这 58 个技能的参数说明里，英文 DANGEROUS
        # 与中文「危险/高危」命中都是 0 条。迭代范围仍是 SKILL_OWNER，没有改动。
        for p in (meta.parameters or []):
            pd = p.description or ""
            assert "DANGEROUS" not in pd, (
                f"{name}.{p.name} 的参数说明里写着 DANGEROUS，但技能的 safety_level 是 "
                f"{meta.safety_level.name} —— 说明在向模型承诺一个并不存在的人工闸门。"
            )
            zh_p = next((w for w in ("危险", "高危") if w in pd), None)
            assert zh_p is None, (
                f"{name}.{p.name} 的参数说明里写着「{zh_p}」，但技能的 safety_level 是 "
                f"{meta.safety_level.name} —— 说明在向模型承诺一个并不存在的人工闸门。"
            )
