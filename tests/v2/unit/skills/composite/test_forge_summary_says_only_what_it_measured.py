"""没有采集到判据的运行不得声称针尖未达标；未知、通过、不合格保持三态。"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.skills.composite.forge_au_tip import _summary  # noqa: E402


class _Progress:
    """`_summary` 只从 progress 里取中止事实,给一个最小替身。"""

    def __init__(self) -> None:
        self.partial_data: dict = {}
        self.aborted = False
        self.aborted_reason = ""

    def to_dict(self) -> dict:
        return {}


_MEASURED = {"bias_v": 0.05, "current_a": 9.7e-11, "any_read": True,
             "unreadable": {}, "read_at_iso": "16:04:31"}

# 模拟未发生测量的空阶段结果。
_NOTHING_RAN = [{"site": 1, "rounds": 0, "outcome": "incomplete", "phases": []}]

#: 真的测过:一轮大修 + 一轮验证,验证给出了数。
_SOMETHING_RAN = [{
    "site": 1, "rounds": 1, "outcome": "verify_exhausted",
    "phases": [{"phase": "pulse", "satisfied": True, "fired": 1},
               {"phase": "verify", "passed": False, "similarity": 0.43}],
}]


def test_a_run_that_measured_nothing_does_not_judge_the_tip():
    """零阶段 ⇒ 结案句里**不许**出现「针尖未达标」。"""
    s = _summary("aborted", _NOTHING_RAN, _Progress(), measured=_MEASURED)
    assert "针尖未达标" not in s, (
        "一次未测量的运行生成了不应出现的针尖结论："
        "第一步 TCP 抖动中止,0 张图、0 个判据,而它告诉用户「针尖未达标」——"
        "照着这句话,下一步是去修一根根本没被碰过的针。")


def test_it_says_plainly_that_nothing_was_measured():
    """而且要**说出来**:这不是沉默,是一句明确的「没测」。

    只删掉那句错话不够 —— 留一个空洞会让读者自己去填,
    而他会填回「大概是针不好」。
    """
    s = _summary("aborted", _NOTHING_RAN, _Progress(), measured=_MEASURED)
    assert "没有测到任何东西" in s
    assert "没测" in s
    # 并且要把人指向真原因,而不是指向针尖
    assert "last_failure" in s


def test_a_run_that_did_measure_still_says_the_tip_missed():
    """真的测过而没达标 ⇒ 那句话必须还在。

    没有这一条,上面两条会诱使人把它整个删掉 ——
    而「测过了、确实不达标」是一个**真的**结论,用户需要它。
    """
    s = _summary("verify_exhausted", _SOMETHING_RAN, _Progress(),
                 measured=_MEASURED)
    assert "针尖未达标" in s, "测过而不达标时,那句结论不该消失"
    assert "没有测到任何东西" not in s


def test_ready_never_prints_either_sentence():
    """成功报告应准确描述完成的步骤。"""
    ok = [{"site": 1, "rounds": 1, "outcome": "ready",
           "phases": [{"phase": "accept", "kind": "undecidable"}]}]
    s = _summary("ready", ok, _Progress(), measured=_MEASURED)
    assert "针尖未达标" not in s
    assert "没有测到任何东西" not in s


@pytest.mark.parametrize("sites", [None, [], [{}], [{"phases": None}],
                                   [{"phases": []}, {"phases": []}]])
def test_every_shape_of_nothing_counts_as_nothing(sites):
    """没有测量时保留未知状态，不能生成针尖质量结论。"""
    s = _summary("aborted", sites, _Progress(), measured=_MEASURED)
    assert "针尖未达标" not in s


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
