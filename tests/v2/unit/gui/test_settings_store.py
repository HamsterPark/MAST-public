"""Persistent UI settings store (closes the settings-reset-on-restart gap).

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/gui/test_settings_store.py -x -v
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

from mast.webui.settings_store import KNOWN_KEYS, SettingsStore  # noqa: E402


def test_update_persists_across_instances(tmp_path):
    s = SettingsStore(tmp_path)
    s.update(model_alias="deepseek-v3", thinking="off", font_scale="大")
    # a fresh store on the same dir reads it back (survives "restart")
    s2 = SettingsStore(tmp_path)
    d = s2.load()
    assert d["model_alias"] == "deepseek-v3"
    assert d["thinking"] == "off" and d["font_scale"] == "大"


def test_only_whitelisted_keys_persist(tmp_path):
    s = SettingsStore(tmp_path)
    s.update(model_alias="kimi-k2.6", bogus_key="x", evil="y")
    d = s.load()
    assert "bogus_key" not in d and "evil" not in d
    assert set(d) <= KNOWN_KEYS


def test_none_values_ignored(tmp_path):
    s = SettingsStore(tmp_path)
    s.update(model_alias="kimi-k2.6")
    s.update(model_alias=None, thinking="max")   # None must not wipe model_alias
    d = s.load()
    assert d["model_alias"] == "kimi-k2.6" and d["thinking"] == "max"


def test_corrupt_file_degrades(tmp_path):
    (tmp_path / "ui_settings.json").write_text("{not json", encoding="utf-8")
    s = SettingsStore(tmp_path)            # must not raise
    assert s.load() == {}
    s.update(theme="Light")                 # still usable
    assert s.get("theme") == "Light"


def test_nanonis_ports_roundtrip(tmp_path):
    s = SettingsStore(tmp_path)
    s.update(nanonis_host="192.168.1.5", nanonis_port_main=7001,
             nanonis_port_emergency=7004)
    d = SettingsStore(tmp_path).load()
    assert d["nanonis_host"] == "192.168.1.5"
    assert d["nanonis_port_main"] == 7001 and d["nanonis_port_emergency"] == 7004


def test_apply_to_config_hydrates(tmp_path):
    from mast.config import MASTConfig
    s = SettingsStore(tmp_path)
    s.update(model_alias="deepseek-v3",
             nanonis_host="10.0.0.9", nanonis_port_main=6601,
             voice="Cherry", voice_autoplay=False)
    cfg = MASTConfig()
    applied = s.apply_to_config(cfg)
    assert cfg.nanonis.host == "10.0.0.9" and cfg.nanonis.port_main == 6601
    assert cfg.voice.autoplay is False
    assert "model_alias" in applied and "nanonis_host" in applied


def test_a_removed_key_is_not_silently_accepted(tmp_path):
    """2026-08-24 删掉的 knowledge_mode 不能悄悄回来。

    白名单外的键在 ``_load`` 里被过滤掉，所以「存得进去、下次启动没了」是它
    唯一可能的表现 —— 而那正是幽灵开关的形状：界面说已保存，行为一个字节
    不变。这条钉住它连白名单都进不去。
    """
    from mast.webui.settings_store import KNOWN_KEYS

    assert "knowledge_mode" not in KNOWN_KEYS, (
        "knowledge_mode 又回到白名单里了 —— 加回来之前先回答「谁读它」："
        "注入侧要有一处真的按这个值改变注入内容，否则它还是那个幽灵开关。")


def test_apply_to_config_ignores_missing(tmp_path):
    # empty store → apply is a no-op, never raises
    from mast.config import MASTConfig
    cfg = MASTConfig()
    before = cfg.nanonis.host
    SettingsStore(tmp_path).apply_to_config(cfg)
    assert cfg.nanonis.host == before


def test_qa_model_is_whitelisted_and_round_trips(tmp_path):
    """The 查询助手 model picker now persists via the qa_model key (its own
    independent client, separate from the global chat model_alias)."""
    assert "qa_model" in KNOWN_KEYS
    s = SettingsStore(tmp_path)
    s.update(qa_model="deepseek-v4-pro", model_alias="kimi-k2.6")
    d = SettingsStore(tmp_path).load()
    assert d["qa_model"] == "deepseek-v4-pro"
    # qa_model is INDEPENDENT of the global chat model.
    assert d["model_alias"] == "kimi-k2.6"


def test_apply_to_config_does_not_touch_qa_model(tmp_path):
    """qa_model is consumed by the GUI (independent QuickAsk client), NOT by
    apply_to_config — applying must not crash and must not leak into config.llm."""
    from mast.config import MASTConfig
    s = SettingsStore(tmp_path)
    s.update(qa_model="qwen3.7-max", model_alias="kimi-k2.6")
    cfg = MASTConfig()
    applied = s.apply_to_config(cfg)
    # model_alias is applied to the main LLM; qa_model is not an apply target.
    assert "qa_model" not in applied
    assert cfg.llm.model_alias == "kimi-k2.6"


def test_vision_thresholds_is_whitelisted_and_dict_round_trips(tmp_path):
    """The VIGIL tip-quality valves persist as ONE structured dict key (not 6
    scalar keys). Survives a 'restart' via the JSON round-trip."""
    assert "vision_thresholds" in KNOWN_KEYS
    th = {"coarse_good_threshold": 0.4, "multi_apex_p_max": 0.65, "m0_quality_min": 65.0}
    s = SettingsStore(tmp_path)
    s.update(vision_thresholds=th)
    d = SettingsStore(tmp_path).load()
    assert d["vision_thresholds"] == th
    assert isinstance(d["vision_thresholds"]["coarse_good_threshold"], float)


def test_vision_thresholds_unchanged_dict_does_not_rewrite(tmp_path):
    """update() compares dicts by value, so re-POSTing the identical valves is a
    no-op (no spurious disk write); a changed value does save."""
    s = SettingsStore(tmp_path)
    s.update(vision_thresholds={"coarse_good_threshold": 0.4})
    calls: list[int] = []
    s._save = lambda: calls.append(1)  # type: ignore[method-assign]
    s.update(vision_thresholds={"coarse_good_threshold": 0.4})  # same value
    assert calls == []
    s.update(vision_thresholds={"coarse_good_threshold": 0.3})  # changed
    assert calls == [1]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
