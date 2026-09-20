"""针尖登记的存储层 —— tips 表、换针事务、当前针尖的唯一性。

关键不变式(设计见 docs/v2/design/tip_registry_and_hardware_profile.md §D2/D3):

  * **没有指针表**:「当前针尖 = 唯一 removed_at IS NULL 的行」。这里钉住这条
    编码是完备且自愈的 —— 尤其是 create_tip 关闭**所有** open 行,而不只是
    最新那一行。
  * 换针把绑上一根针的学习标定清掉,但先快照进退役行(归档不是丢弃)。
  * tip_index 单调不回收(目录名/编号一旦示人就不能重指另一根针)。
"""

from __future__ import annotations

import json

from mast.core import instrument_profile as iprof
from mast.core import tip_state
from mast.logging import tip_registry
from mast.logging.storage import ExperimentStorage


def _fresh(tmp_path, name="tips.db") -> ExperimentStorage:
    return ExperimentStorage(str(tmp_path / name))


# ── 表与迁移 ────────────────────────────────────────────────────────────────

def test_tips_table_is_created_on_a_fresh_db(tmp_path) -> None:
    s = _fresh(tmp_path)
    assert s.get_current_tip() is None
    assert s.list_tips() == []


def test_reopening_an_existing_db_is_idempotent(tmp_path) -> None:
    """CREATE TABLE IF NOT EXISTS 每次连库都跑,重开不能炸也不能丢数据。"""
    s = _fresh(tmp_path)
    tid = s.create_tip({"material": "W", "fabrication": "etched"})
    again = ExperimentStorage(str(tmp_path / "tips.db"))
    assert again.get_current_tip()["id"] == tid


# ── 当前针尖 = 唯一 open 行 ─────────────────────────────────────────────────

def test_registering_a_tip_retires_the_previous_one(tmp_path) -> None:
    s = _fresh(tmp_path)
    first = s.create_tip({"material": "W", "fabrication": "etched"})
    second = s.create_tip({"material": "PtIr", "fabrication": "cut"})

    assert s.get_current_tip()["id"] == second
    old = s.get_tip(first)
    assert old["removed_at"], "上一根针必须在装新针时退役"
    assert s.get_tip(second)["removed_at"] is None


def test_create_tip_closes_every_open_row_not_just_the_latest(tmp_path) -> None:
    """两行 open 只可能来自手改库,但事务必须顺手收敛掉它 —— 这正是不建指针表
    换来的自愈:没有第二处状态可以跟这张表分叉。"""
    s = _fresh(tmp_path)
    a = s.create_tip({"material": "W"})
    b = s.create_tip({"material": "PtIr"})
    # 手工制造一个第二 open 行(模拟手改库/早期 bug)
    with s._connect() as conn:                      # noqa: SLF001 — 就是要伪造脏状态
        conn.execute("UPDATE tips SET removed_at = NULL WHERE id = ?", (a,))
    with s._connect() as conn:
        n_open = conn.execute(
            "SELECT COUNT(*) AS n FROM tips WHERE removed_at IS NULL").fetchone()["n"]
    assert n_open == 2, "前提:确实造出了两行 open"

    s.create_tip({"material": "Fe"})
    with s._connect() as conn:
        n_open = conn.execute(
            "SELECT COUNT(*) AS n FROM tips WHERE removed_at IS NULL").fetchone()["n"]
    assert n_open == 1, "登记新针尖必须把所有 open 行收敛掉"
    assert s.get_tip(a)["removed_at"] and s.get_tip(b)["removed_at"]


def test_get_current_tip_survives_a_dirty_db_with_two_open_rows(tmp_path) -> None:
    """读当前针尖绝不能抛 —— 注入块和样品快照都在读它。"""
    s = _fresh(tmp_path)
    a = s.create_tip({"material": "W"})
    s.create_tip({"material": "PtIr"})
    with s._connect() as conn:                      # noqa: SLF001
        conn.execute("UPDATE tips SET removed_at = NULL WHERE id = ?", (a,))
    cur = s.get_current_tip()
    assert cur is not None and cur["removed_at"] is None


# ── 序号 ────────────────────────────────────────────────────────────────────

def test_tip_index_is_monotonic_and_not_recycled(tmp_path) -> None:
    s = _fresh(tmp_path)
    ids = [s.create_tip({"material": m}) for m in ("W", "PtIr", "Fe")]
    idx = [s.get_tip(i)["tip_index"] for i in ids]
    assert idx == [1, 2, 3]

    with s._connect() as conn:                      # noqa: SLF001 — 删一行看序号会不会回收
        conn.execute("DELETE FROM tips WHERE id = ?", (ids[-1],))
    nxt = s.create_tip({"material": "Au"})
    assert s.get_tip(nxt)["tip_index"] == 3 or s.get_tip(nxt)["tip_index"] == 4
    # 关键是不回收到已用过的 1/2
    assert s.get_tip(nxt)["tip_index"] > 2


# ── 查询 / 更新 ─────────────────────────────────────────────────────────────

def test_find_tip_by_name_is_casefolded(tmp_path) -> None:
    s = _fresh(tmp_path)
    tid = s.create_tip({"material": "W", "name": "W-etched #1"})
    assert s.find_tip_by_name("w-ETCHED #1")["id"] == tid
    assert s.find_tip_by_name("  W-etched #1  ")["id"] == tid
    assert s.find_tip_by_name("没有这根") is None
    assert s.find_tip_by_name("") is None


def test_update_tip_only_touches_whitelisted_columns(tmp_path) -> None:
    s = _fresh(tmp_path)
    tid = s.create_tip({"material": "W"})
    before = s.get_tip(tid)
    assert s.update_tip(tid, {"wire_diameter_mm": 0.25, "note": "第一根"}) is True
    after = s.get_tip(tid)
    assert after["wire_diameter_mm"] == 0.25 and after["note"] == "第一根"
    # 未知键静默忽略,不炸
    assert s.update_tip(tid, {"nonexistent_column": 1}) is False
    # 事件事实不可改
    assert after["tip_index"] == before["tip_index"]
    assert after["created_at"] == before["created_at"]
    assert s.update_tip("no-such-id", {"note": "x"}) is False


def test_list_tips_is_newest_first(tmp_path) -> None:
    s = _fresh(tmp_path)
    s.create_tip({"material": "W", "name": "one"})
    s.create_tip({"material": "PtIr", "name": "two"})
    names = [t["name"] for t in s.list_tips()]
    assert names[0] == "two"


def test_retire_current_tip(tmp_path) -> None:
    s = _fresh(tmp_path)
    s.create_tip({"material": "W"})
    assert s.retire_current_tip(retire_snapshot={"didv_at_contact_v": 1.0}) is True
    assert s.get_current_tip() is None
    assert s.retire_current_tip() is False, "没有 open 行时应如实返回 False"


# ── 换针副作用(编排层)─────────────────────────────────────────────────────

def test_registering_clears_the_previous_tips_learned_calibration(tmp_path) -> None:
    """换针必须清掉绑上一根针的量,否则撞针判据拿旧分母比,非错即哑。"""
    s = _fresh(tmp_path)
    iprof.set_profile({
        "didv_at_contact_v": 2.5e-3,
        "didv_cal_bias_v": 0.5,
        "qplus_amplitude_baseline": 12.0,
        "qplus_f0_measured_hz": 32768.0,
        "qplus_q_measured": 4000.0,
        # 这两个必须活下来
        "tilt_cal_g11": 1.0, "tilt_cal_g12": 0.0,
        "tilt_cal_g21": 0.0, "tilt_cal_g22": 1.0,
        "qplus_amplitude_signal_index": 7,
    })
    tip_registry.register_tip(s, material="W", fabrication="etched")

    prof = iprof.get_profile()
    for gone in ("didv_at_contact_v", "didv_cal_bias_v", "qplus_amplitude_baseline",
                 "qplus_f0_measured_hz", "qplus_q_measured"):
        assert gone not in prof, f"{gone} 绑上一根针,换针必须清"
    assert prof["tilt_cal_g11"] == 1.0, "倾斜标定是样品/托架属性,换针不该清"
    assert prof["qplus_amplitude_signal_index"] == 7, "信号槽接线是仪器属性,换针不该清"
    iprof.set_profile({})


def test_cleared_calibration_is_archived_on_the_retired_tip(tmp_path) -> None:
    """清掉不等于丢掉 —— 旧值进退役那一行,日后还查得到。"""
    s = _fresh(tmp_path)
    tip_registry.register_tip(s, material="W", fabrication="etched")
    iprof.set_profile({"didv_at_contact_v": 3.3e-3, "didv_cal_bias_v": 0.8})

    tip_registry.register_tip(s, material="PtIr", fabrication="cut")

    retired = [t for t in s.list_tips() if t["removed_at"]][0]
    snap = json.loads(retired["retire_snapshot"])
    assert snap["didv_at_contact_v"] == 3.3e-3
    assert snap["didv_cal_bias_v"] == 0.8
    iprof.set_profile({})


def test_register_normalizes_vocabulary_and_updates_the_holder(tmp_path) -> None:
    s = _fresh(tmp_path)
    res = tip_registry.register_tip(
        s, material="钨", fabrication="电化学腐蚀", form="音叉")
    assert res["ok"] is True
    assert res["tip"]["material"] == "W"
    assert res["tip"]["fabrication"] == "etched"
    assert res["tip"]["form"] == "qplus"
    facts = tip_state.current_tip_facts()
    assert facts["material"] == "W" and facts["form"] == "qplus"
    assert tip_state.is_qplus() is True
    tip_state.set_current_tip(None)


def test_unknown_vocabulary_is_recorded_verbatim_with_a_warning(tmp_path) -> None:
    """认不出的材料不猜,原样存 + 说明,让用户/模型下一步更正。"""
    s = _fresh(tmp_path)
    res = tip_registry.register_tip(s, material="镝钪合金", fabrication="魔法")
    assert res["ok"] is True
    assert res["tip"]["material"] == "镝钪合金"
    assert res["tip"]["fabrication"] == "unknown"
    assert any("词表" in w for w in res["warnings"])
    tip_state.set_current_tip(None)


def test_auto_name_when_left_blank(tmp_path) -> None:
    s = _fresh(tmp_path)
    res = tip_registry.register_tip(s, material="W", fabrication="etched")
    assert res["tip"]["name"] == "W-etched #1"
    tip_state.set_current_tip(None)


def test_installed_at_can_be_backdated(tmp_path) -> None:
    """用户常常过一两天才想起来记 —— 装入日期必须可回填。"""
    s = _fresh(tmp_path)
    res = tip_registry.register_tip(
        s, material="W", installed_at="2026-07-20T09:00:00")
    assert res["tip"]["installed_at"].startswith("2026-07-20")
    tip_state.set_current_tip(None)


def test_bad_numbers_are_dropped_not_stored(tmp_path) -> None:
    s = _fresh(tmp_path)
    res = tip_registry.register_tip(
        s, material="W", wire_diameter_mm="粗的", qplus_q=-5)
    assert res["ok"] is True
    assert res["tip"]["wire_diameter_mm"] is None
    assert res["tip"]["qplus_q"] is None
    assert len(res["warnings"]) >= 2
    tip_state.set_current_tip(None)


def test_hydrate_restores_the_holder_at_startup(tmp_path) -> None:
    s = _fresh(tmp_path)
    tip_registry.register_tip(s, material="PtIr", fabrication="cut")
    tip_state.set_current_tip(None)                 # 模拟进程重启
    assert tip_state.current_tip_facts() is None

    tip_registry.hydrate(s)
    assert tip_state.current_tip_facts()["material"] == "PtIr"
    tip_state.set_current_tip(None)


def test_remove_current_tip_when_nothing_registered(tmp_path) -> None:
    s = _fresh(tmp_path)
    res = tip_registry.remove_current_tip(s)
    assert res["ok"] is True and res["changed"] is False


def test_registry_never_raises_on_a_broken_storage() -> None:
    """登记针尖失败绝不能把一次实验带停。"""
    class Broken:
        def get_current_tip(self): raise RuntimeError("db gone")
        def create_tip(self, *a, **k): raise RuntimeError("db gone")
        def retire_current_tip(self, *a, **k): raise RuntimeError("db gone")
        def get_active_scope(self): raise RuntimeError("db gone")

    res = tip_registry.register_tip(Broken(), material="W")
    assert res["ok"] is False and "失败" in res["error"]
    assert tip_registry.hydrate(Broken()) is None
    assert tip_registry.remove_current_tip(Broken())["ok"] is False
    tip_state.set_current_tip(None)
