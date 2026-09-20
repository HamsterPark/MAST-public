"""逐偏压账 —— 键、append-only、epoch 对账、五栏分列。

S2 逐偏压原子序列设计 D4 / §3.2 / §3.3。

这一层承载的是整个 S2 的可信度:序列跑完之后，「哪个偏压上有原子分辨」这个问题
只能由这本账回答。所以四件事要钉死：

1. **键是规范化的偏压** —— `expand_series` 真实吐出 `-0.30000000000000004`，
   不规范化就是同一个偏压两把钥匙（设计陷阱 12），而重排之后序号根本不是偏压
   （陷阱 13）；
2. **append-only** —— 同偏压第二次尝试是第二条记录，不是把第一条改掉；
3. **`stale` 是读出来的不是存出来的**，且对不上账时是 `None` 不是 `False`；
4. **五栏分列且加得起来** —— `undecided` 与 `absent` 合并会把「证据不足」说成
   「这里没有原子」，那是这本账最不能犯的错。

Run from repo root:
    .venv-v2-py313/Scripts/python.exe -m pytest tests/v2/unit/core/test_s2_bias_ledger.py -x -v
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.core import s2_bias_ledger as L  # noqa: E402
from mast.core.scan_planner import expand_series, order_series_monotonic  # noqa: E402


def _rec(bias_v: float, attempt: int = 1, **kw) -> dict:
    kw.setdefault("evidence_epoch", 3)
    kw.setdefault("coord_epoch", 7)
    return L.make_record(bias_v=bias_v, attempt=attempt, **kw)


def _write(tmp_path: Path, records: list[dict]) -> Path:
    p = L.ledger_path(tmp_path)
    for r in records:
        L.append(p, r)
    return p


# ══════════════════════════════════════════════════════════════════════
# 1. 键
# ══════════════════════════════════════════════════════════════════════

def test_the_key_survives_expand_series_float_dust():
    """**用 `expand_series` 的真实输出喂**，不是手写一个漂亮的浮点。

    linear 展开做的是 `start + step * i`，-0.3 真的会以
    `-0.30000000000000004` 出现。这条测试的价值全在「数据是真的从那个函数来的」。
    """
    values = expand_series({"start": -0.5, "stop": 0.5, "n": 11})
    assert any(abs(v) > 0 and f"{v!r}".count("0000000") for v in values), (
        f"这批值里没有浮点尘，测试就没在测它要测的东西:{values!r}")
    keys = [L.bias_key(v) for v in values]
    assert len(set(keys)) == len(values), f"同一个偏压长出了两把钥匙:{keys}"
    assert L.bias_key(-0.30000000000000004) == L.bias_key(-0.3)


def test_negative_zero_is_not_a_second_key_for_zero():
    """`round(-1e-9, 6)` 得到 `-0.0`，格式化出来是 `-0.000000` —— 与
    `0.000000` 指着同一个偏压却是两把钥匙。"""
    assert L.bias_key(-1e-9) == L.bias_key(0.0) == "0.000000"
    assert not L.bias_key(-1e-9).startswith("-")


def test_key_resolution_is_microvolts():
    assert L.bias_key(0.0000004) == L.bias_key(0.0)      # 0.4 µV → 同键
    assert L.bias_key(0.0000015) != L.bias_key(0.0)      # 1.5 µV → 不同键


def test_bias_human_reads_like_an_stm_person_writes_it():
    assert L.bias_human(-0.3) == "-300 mV"
    assert L.bias_human(1.5) == "1.5 V"
    assert L.bias_human(0.0) == "0 V"


def test_bias_human_does_not_use_format_si():
    """`format_si` 的契约是「印出来要能抄回参数」，于是 2 V 被印成 `2000m`。
    `live_state_mw.py` 已经为偏压做过这个判断，这里跟随它而不是另立一套。"""
    from mast.core.si_quantity import format_si

    assert format_si(2.0) == "2000m"          # 先证明那个函数确实这么干
    assert L.bias_human(2.0) == "2 V"         # 再证明这里没照抄它


# ══════════════════════════════════════════════════════════════════════
# 2. append-only
# ══════════════════════════════════════════════════════════════════════

def test_a_second_attempt_is_a_second_row(tmp_path):
    p = _write(tmp_path, [
        _rec(-0.3, 1, verdict=L.VERDICT_UNDECIDABLE, remedy="rescan_frame"),
        _rec(-0.3, 2, verdict=L.VERDICT_RESOLVED),
    ])
    rows = L.read_ledger(p, current_evidence_epoch=3).rows
    assert len(rows) == 2, "第二次尝试把第一条改写了 —— 尝试史没了"
    assert [r["attempt"] for r in rows] == [1, 2]
    assert p.read_text(encoding="utf-8").count("\n") == 2


def test_rows_are_one_self_contained_json_object_per_line(tmp_path):
    p = _write(tmp_path, [_rec(-0.3, 1, verdict=L.VERDICT_RESOLVED)])
    line = p.read_text(encoding="utf-8").strip()
    obj = json.loads(line)
    assert obj["schema"] == L.SCHEMA
    assert obj["bias_key"] == "-0.300000" and obj["bias_v"] == -0.3


def test_the_ledger_stores_an_absolute_frame_path(tmp_path):
    """`scan_registry._MAX_KEEP = 50` 会把多天序列的早期帧淘汰掉（陷阱 11），
    所以账里必须自带绝对路径，不能只留一个 scan_id 去查它。"""
    frame = tmp_path / "f001.sxm"
    p = _write(tmp_path, [_rec(-0.3, 1, frame_path=str(frame),
                               scan_id="s-001", verdict=L.VERDICT_RESOLVED)])
    row = L.read_ledger(p, current_evidence_epoch=3).rows[0]
    assert Path(row["frame_path"]).is_absolute()
    assert row["scan_id"] == "s-001", "交叉查用的 scan_id 也要留着"


# ══════════════════════════════════════════════════════════════════════
# 3. epoch 对账:stale 是读出来的
# ══════════════════════════════════════════════════════════════════════

def test_bumping_the_evidence_epoch_makes_old_rows_stale_without_deleting_them(tmp_path):
    p = _write(tmp_path, [_rec(-0.3, 1, verdict=L.VERDICT_RESOLVED,
                               evidence_epoch=3)])
    after = L.read_ledger(p, current_evidence_epoch=4)
    assert len(after.rows) == 1, "陈旧的记录被删了 —— 设计要求是标记不是删除"
    assert after.rows[0]["stale"] is True
    assert p.read_text(encoding="utf-8").count("\n") == 1, "文件被重写了"


def test_an_unknown_current_epoch_yields_none_not_false(tmp_path):
    """对不了账要说对不了账。`False` 是在说「我核对过，它是新鲜的」。"""
    p = _write(tmp_path, [_rec(-0.3, 1, verdict=L.VERDICT_RESOLVED)])
    row = L.read_ledger(p, current_evidence_epoch=None).rows[0]
    assert row["stale"] is None
    assert row["stale"] is not False


def test_a_row_without_an_epoch_is_also_none(tmp_path):
    p = _write(tmp_path, [_rec(-0.3, 1, verdict=L.VERDICT_RESOLVED,
                               evidence_epoch=None)])
    assert L.read_ledger(p, current_evidence_epoch=4).rows[0]["stale"] is None


def test_the_stored_stale_flag_is_never_trusted(tmp_path):
    """文件里写着 `stale: false`（写入那一刻的实情），但当前代次说它陈旧。
    读出来必须是 True —— 存着的那个值不是答案。"""
    p = L.ledger_path(tmp_path)
    r = _rec(-0.3, 1, verdict=L.VERDICT_RESOLVED, evidence_epoch=1)
    assert r["stale"] is False
    L.append(p, r)
    assert L.read_ledger(p, current_evidence_epoch=9).rows[0]["stale"] is True


def test_a_half_written_line_is_reported_not_dropped(tmp_path):
    """丢掉一条 resolved，汇总就会把那个偏压报成 undecided —— 一句假话。"""
    p = _write(tmp_path, [_rec(-0.3, 1, verdict=L.VERDICT_RESOLVED)])
    with open(p, "a", encoding="utf-8") as fh:
        fh.write('{"schema": "s2_bias_ledg')      # 断电断在半截
    read = L.read_ledger(p, current_evidence_epoch=3)
    assert len(read.rows) == 1
    assert read.unreadable and "不是合法 JSON" in read.unreadable[0]["why"]


def test_a_missing_ledger_says_so(tmp_path):
    read = L.read_ledger(L.ledger_path(tmp_path), current_evidence_epoch=1)
    assert read.exists is False and read.rows == [] and read.unreadable == []


# ══════════════════════════════════════════════════════════════════════
# 4. 每偏压终态(闭集,每态至少一条)
# ══════════════════════════════════════════════════════════════════════

def test_any_resolved_attempt_wins():
    assert L.derive_final_state([
        {"verdict": L.VERDICT_UNDECIDABLE},
        {"verdict": L.VERDICT_RESOLVED},
    ]) == L.STATE_RESOLVED


def test_absent_confirmed_needs_a_frame_that_was_admitted():
    """「这个偏压上没有原子对比度」是个**科学结论**，它要求至少有一帧是合格的。
    帧准入没过就下这个结论，等于在自己刚制造的坑上判样品。"""
    admitted = [{"verdict": L.VERDICT_ABSENT, "frame_admission_passed": True},
                {"verdict": L.VERDICT_ABSENT, "frame_admission_passed": True}]
    assert L.derive_final_state(admitted) == L.STATE_ABSENT_CONFIRMED

    never_admitted = [{"verdict": L.VERDICT_ABSENT,
                       "frame_admission_passed": False}]
    assert L.derive_final_state(never_admitted) == L.STATE_UNDECIDED


def test_a_mix_of_absent_and_undecidable_sinks_to_undecided():
    """闭集在这里本来有个洞（`absent_confirmed` 要求全 absent，`undecided`
    要求从未 absent，混合两条都不满足）。**代价不对称决定补洞方向**：
    补成 absent 是拿不足的证据下结论，补成 undecided 只是说证据不够。"""
    assert L.derive_final_state([
        {"verdict": L.VERDICT_ABSENT, "frame_admission_passed": True},
        {"verdict": L.VERDICT_UNDECIDABLE},
    ]) == L.STATE_UNDECIDED


def test_a_tip_abort_outranks_the_absence_evidence():
    """针尖出过事之后，这个偏压上的其余证据都得等 S1 绕道回来再看。"""
    assert L.derive_final_state([
        {"verdict": L.VERDICT_ABSENT, "frame_admission_passed": True},
        {"outcome": L.OUTCOME_TIP_ABORT},
    ]) == L.STATE_TIP_ABORTED


def test_blocked_only_when_nothing_was_ever_judged():
    assert L.derive_final_state(
        [{"outcome": L.OUTCOME_BLOCKED}]) == L.STATE_BLOCKED
    # 拒过一次但后来量到了 —— 那就不是「被拒」的故事
    assert L.derive_final_state([
        {"outcome": L.OUTCOME_BLOCKED},
        {"verdict": L.VERDICT_ABSENT, "frame_admission_passed": True},
    ]) == L.STATE_ABSENT_CONFIRMED


def test_no_attempts_at_all_is_undecided_not_absent():
    assert L.derive_final_state([]) == L.STATE_UNDECIDED


def test_every_final_state_is_reachable():
    """闭集穷举:五个终态每一个都要有一条能到达它的输入。少一个就是死枚举。"""
    reached = {
        L.derive_final_state([{"verdict": L.VERDICT_RESOLVED}]),
        L.derive_final_state([{"verdict": L.VERDICT_ABSENT,
                               "frame_admission_passed": True}]),
        L.derive_final_state([{"verdict": L.VERDICT_UNDECIDABLE}]),
        L.derive_final_state([{"outcome": L.OUTCOME_BLOCKED}]),
        L.derive_final_state([{"outcome": L.OUTCOME_TIP_ABORT}]),
    }
    assert reached == set(L.FINAL_STATES)


# ══════════════════════════════════════════════════════════════════════
# 5. 汇总:五栏分列 + 加得起来
# ══════════════════════════════════════════════════════════════════════

def _mixed_ledger(tmp_path) -> tuple[Path, list[float]]:
    planned = [-0.4, -0.2, 0.0, 0.2, 0.4]
    _write(tmp_path, [
        _rec(-0.4, 1, verdict=L.VERDICT_RESOLVED, frame_admission_passed=True),
        _rec(-0.2, 1, verdict=L.VERDICT_ABSENT, frame_admission_passed=True),
        _rec(-0.2, 2, verdict=L.VERDICT_ABSENT, frame_admission_passed=True),
        _rec(0.0, 1, verdict=L.VERDICT_UNDECIDABLE, remedy="rescan_frame"),
        _rec(0.0, 2, verdict=L.VERDICT_UNDECIDABLE, remedy="move_site"),
        _rec(0.2, 1, outcome=L.OUTCOME_BLOCKED),
        # 0.4 一条记录都没有 —— 没跑到
    ])
    return L.ledger_path(tmp_path), planned


def test_the_five_columns_are_listed_separately(tmp_path):
    p, planned = _mixed_ledger(tmp_path)
    s = L.summarize(L.read_ledger(p, current_evidence_epoch=3),
                    planned_biases=planned)
    assert s["n_resolved"] == 1
    assert s["n_absent_confirmed"] == 1
    assert s["n_undecided"] == 1
    assert s["n_blocked"] == 1
    assert s["n_tip_aborted"] == 0
    assert len(s["unreadable"]) == 1 and "没跑到" in s["unreadable"][0]["why"]


def test_the_columns_plus_unreadable_account_for_every_planned_bias(tmp_path):
    """恒等式。少掉的那个偏压必须出现在**某一栏**里，不能凭空消失。"""
    p, planned = _mixed_ledger(tmp_path)
    s = L.summarize(L.read_ledger(p, current_evidence_epoch=3),
                    planned_biases=planned)
    total = (s["n_resolved"] + s["n_absent_confirmed"] + s["n_undecided"]
             + s["n_blocked"] + s["n_tip_aborted"] + len(s["unreadable"]))
    assert total == s["n_biases"] == 5, f"账加不起来:{s}"


def test_undecided_is_never_folded_into_absent(tmp_path):
    """这本账最不能犯的错:把「证据不足」说成「这里没有原子」。"""
    p, planned = _mixed_ledger(tmp_path)
    s = L.summarize(L.read_ledger(p, current_evidence_epoch=3),
                    planned_biases=planned)
    states = {r["bias_key"]: r["final_state"] for r in s["per_bias"]}
    assert states[L.bias_key(0.0)] == L.STATE_UNDECIDED
    assert states[L.bias_key(-0.2)] == L.STATE_ABSENT_CONFIRMED


def test_stale_rows_drop_out_of_the_summary_but_stay_in_the_file(tmp_path):
    p, planned = _mixed_ledger(tmp_path)
    s = L.summarize(L.read_ledger(p, current_evidence_epoch=4),  # 全部陈旧
                    planned_biases=planned)
    assert s["n_resolved"] == 0
    assert len(s["unreadable"]) == 5, "五个偏压全变成「没跑到」"
    assert len(L.read_ledger(p, current_evidence_epoch=4).rows) == 6


def test_rows_that_cannot_be_reconciled_go_to_unreadable_not_to_a_count(tmp_path):
    p, planned = _mixed_ledger(tmp_path)
    s = L.summarize(L.read_ledger(p, current_evidence_epoch=None),
                    planned_biases=planned)
    assert s["n_resolved"] == 0 and s["n_undecided"] == 0
    assert len(s["unreadable"]) >= 6
    assert any("evidence_epoch" in u["why"] for u in s["unreadable"])


def test_summary_uses_the_bias_as_key_not_the_frame_order(tmp_path):
    """`order_series_monotonic` 会重排 ⇒ 帧序号 ≠ 偏压序号（陷阱 13）。
    按重排后的顺序落账，汇总仍要按偏压对齐。"""
    values = expand_series({"start": -0.4, "stop": 0.4, "n": 5})
    order = order_series_monotonic(values, current=0.4)
    assert order != list(range(len(values))), "这次没重排，测试没测到东西"
    _write(tmp_path, [_rec(values[i], 1, verdict=L.VERDICT_RESOLVED,
                           frame_admission_passed=True)
                      for i in order])
    s = L.summarize(L.read_ledger(L.ledger_path(tmp_path),
                                  current_evidence_epoch=3),
                    planned_biases=values)
    assert s["n_resolved"] == 5
    assert [r["bias_key"] for r in s["per_bias"]] == [L.bias_key(v)
                                                      for v in values]


# ══════════════════════════════════════════════════════════════════════
# 6. 锚点 + 序列出口
# ══════════════════════════════════════════════════════════════════════

def test_anchor_frames_are_counted_and_zero_anchors_is_visible(tmp_path):
    _write(tmp_path, [
        _rec(-0.4, 1, verdict=L.VERDICT_RESOLVED, frame_admission_passed=True),
        _rec(-0.4, 2, role=L.ROLE_ANCHOR, verdict=L.VERDICT_RESOLVED,
             frame_admission_passed=True),
    ])
    s = L.summarize(L.read_ledger(L.ledger_path(tmp_path),
                                  current_evidence_epoch=3),
                    planned_biases=[-0.4])
    a = s["anchor_consistency"]
    assert a["n_anchors"] == 1 and a["n_anchor_resolved"] == 1
    assert L.anchor_consistency([])["n_anchors"] == 0, (
        "一个锚点都没有要看得见 —— 没有锚点时「扫的是同一片原子」无从证明")


def test_drift_flagged_biases_surface_as_segments(tmp_path):
    _write(tmp_path, [_rec(-0.4, 1, verdict=L.VERDICT_RESOLVED,
                           frame_admission_passed=True,
                           drift_shift_px=41.0, drift_flag="drift_excessive")])
    s = L.summarize(L.read_ledger(L.ledger_path(tmp_path),
                                  current_evidence_epoch=3),
                    planned_biases=[-0.4])
    assert s["anchor_consistency"]["drifted_segments"] == ["-400 mV"]


# ── anchor_verdict:同一份证据的顶层名字(2026-08-15)────────────────────

@pytest.mark.parametrize("rows,expect", [
    ([], "unproven_no_anchor"),
    ([{"role": L.ROLE_SERIES, "verdict": L.VERDICT_RESOLVED}],
     "unproven_no_anchor"),
    ([{"role": L.ROLE_ANCHOR, "verdict": L.VERDICT_RESOLVED}],
     "corroborated"),
    ([{"role": L.ROLE_ANCHOR, "verdict": L.VERDICT_UNDECIDABLE}],
     "unproven_anchor_unresolved"),
    # 漂移优先:锚点自己判出了原子分辨也不算数 —— 锚点证明的是「回到起始偏压
    # 还能看见原子」,不是「中间那些帧没跑偏」。
    ([{"role": L.ROLE_ANCHOR, "verdict": L.VERDICT_RESOLVED},
      {"role": L.ROLE_SERIES, "verdict": L.VERDICT_RESOLVED,
       "drift_flag": "drift_excessive", "bias_human": "-400 mV"}],
     "contradicted"),
])
def test_the_anchor_verdict_is_a_closed_set_of_four(rows, expect):
    """**刻意不是布尔。**

    「一张锚点都没拍」补不回来(那一段已经扫完了)、「拍了没判出」可以补拍一张、
    「检测到漂移」是实打实的否定证据 —— 三件事下一步做的完全不同,压成 ``False``
    就都没了,而且会和第四种(佐证成立)的反面混在一起。
    """
    got = L.anchor_verdict(L.anchor_consistency(rows))
    assert got == expect
    assert got in L.ANCHOR_VERDICTS


def test_the_flat_verdict_and_the_nested_detail_come_from_one_computation(tmp_path):
    """两个字段是同一份证据的两种读法 —— 各算一遍就会有两个答案。

    这条钉的是 ``summarize`` 里那一次 ``ac = anchor_consistency(usable)``:
    嵌套详情与顶层裁决都由它派生。哪天有人让顶层那个再算一遍(比如从别的行集合
    里算),这条会红。
    """
    _write(tmp_path, [
        _rec(-0.4, 1, verdict=L.VERDICT_RESOLVED, frame_admission_passed=True),
        _rec(-0.4, 2, role=L.ROLE_ANCHOR, verdict=L.VERDICT_RESOLVED,
             frame_admission_passed=True),
    ])
    s = L.summarize(L.read_ledger(L.ledger_path(tmp_path),
                                  current_evidence_epoch=3),
                    planned_biases=[-0.4])
    assert s["anchor_verdict"] == L.anchor_verdict(s["anchor_consistency"])
    assert s["anchor_verdict"] == "corroborated"


@pytest.mark.parametrize("summary,kw,expect", [
    ({"n_biases": 3, "n_resolved": 3, "n_absent_confirmed": 0,
      "n_blocked": 0, "unreadable": []}, {}, L.EXIT_COMPLETE),
    ({"n_biases": 3, "n_resolved": 1, "n_absent_confirmed": 2,
      "n_blocked": 0, "unreadable": []}, {}, L.EXIT_COMPLETE),
    ({"n_biases": 3, "n_resolved": 1, "n_absent_confirmed": 0,
      "n_blocked": 0, "unreadable": []}, {}, L.EXIT_PARTIAL),
    ({"n_biases": 3, "n_resolved": 0, "n_absent_confirmed": 0,
      "n_blocked": 3, "unreadable": []}, {}, L.EXIT_BLOCKED),
    ({"n_biases": 3, "n_resolved": 1, "n_absent_confirmed": 0,
      "n_blocked": 0, "unreadable": []}, {"aborted_tip": True},
     L.EXIT_ABORTED_TIP),
    ({"n_biases": 3, "n_resolved": 1, "n_absent_confirmed": 0,
      "n_blocked": 0, "unreadable": []}, {"budget_exhausted": True},
     L.EXIT_ABORTED_BUDGET),
])
def test_series_exit_closed_set(summary, kw, expect):
    got = L.series_exit(summary, **kw)
    assert got == expect
    assert got in L.SERIES_EXITS


def test_complete_requires_no_unreadable(tmp_path):
    """全部有结论、但有一栏读不到 ⇒ **不许**报 complete。
    「读不到」非空时闸门不得判 pass 是设计里的铁律。"""
    s = {"n_biases": 2, "n_resolved": 2, "n_absent_confirmed": 0,
         "n_blocked": 0, "unreadable": [{"what": "x", "why": "y"}]}
    assert L.series_exit(s) == L.EXIT_PARTIAL


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
