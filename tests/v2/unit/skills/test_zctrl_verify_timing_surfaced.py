"""「这是不是一个误判？」—— 回答这个问题的两个数一直没被说出来。

现场返回的错误信息本身就是一个问句::

    工具 [ZControllerOnOff] failed: Z 反馈开关未生效：要求 ON，
    实时控制器回报 OFF。这是不是一个误判？

他答不上来，看日志的人也答不上来 —— 因为能分辨的那两个数
（``waited_s`` / ``switch_off_delay_s``）由 ``verify_z_controller`` 算了出来，
然后被丢掉了，从没进过给用户的消息。又一次「量出来了但没人用」。

补上之后两种成因一眼分开：

* 等待时间 ≈ 机器声明的延迟 → 写入确实没生效（接线 / 模块 / RT 配置的问题）
* 等待时间 ≪ 机器声明的延迟 → 是判早了，该调的是 settle 预算

这条**不改判据本身**。写后回读不一致仍然判失败（Nanonis 手册要求以实时控制器
为准，而反馈环还闭着时跑开环粗进针就是撞针）—— 改的只是让失败可诊断。

从仓库根运行::

    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/skills/test_zctrl_verify_timing_surfaced.py -q
"""
from __future__ import annotations

# ── path bootstrap ───────────────────────────────────────────────────────────
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.modules.setdefault("nanonis_spm", MagicMock())


def _find_mastv2_root() -> str:
    p = Path(__file__).resolve()
    while p.parent != p:
        candidate = p / "MASTv2"
        if candidate.is_dir():
            return str(candidate)
        p = p.parent
    raise RuntimeError("MASTv2 dir not found")


_MASTV2_ROOT = _find_mastv2_root()
if sys.path and sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)
for _n in list(sys.modules):
    if _n == "mast" or _n.startswith("mast."):
        _f = getattr(sys.modules[_n], "__file__", "") or ""
        if "MASTv2" not in _f.replace("\\", "/"):
            del sys.modules[_n]

import pytest  # noqa: E402

from mast.core.types import NanonisCallRecord  # noqa: E402
from mast.skills.builtins.zcontrol import ZControllerOnOff  # noqa: E402


class _Ctx:
    def safe_call(self, verb, *a, **k):
        return NanonisCallRecord(method=verb, args=a, kwargs={},
                                 return_value=("", b"", [0]), error="")


def _run(waited, budget, *, on=False, enable=True):
    """Drive ZControllerOnOff with a mismatching read-back and given timing."""
    verdict = {"on": on, "verified": True, "matches": False, "error": None,
               "record": _Ctx().safe_call("ZCtrl_OnOffGet"),
               "waited_s": waited, "switch_off_delay_s": budget,
               "module_status": 1}
    with patch("mast.skills.builtins.zcontrol.verify_z_controller",
               return_value=verdict):
        return ZControllerOnOff().execute(_Ctx(), {"enable": enable})


# ════════════════════════════════════════════════════════════════════════════
# 两种成因必须能分开 —— 这正是那句错误信息里带出的问题
# ════════════════════════════════════════════════════════════════════════════

def test_waiting_far_less_than_the_declared_delay_says_so():
    r = _run(0.20, 5.0)
    assert not r.success
    assert "0.20s" in r.error and "5.00s" in r.error
    assert "判早" in r.error, "没告诉用户这更可能是判早了"


def test_waiting_out_the_declared_delay_says_the_write_did_not_land():
    r = _run(4.90, 5.0)
    assert "已等满" in r.error
    assert "没生效" in r.error


def test_a_rig_that_declares_no_delay_is_stated_not_guessed():
    r = _run(0.20, None)
    assert "未声明" in r.error
    for word in ("判早", "已等满"):
        assert word not in r.error, "机器没给预算却下了结论"


def test_the_numbers_are_also_in_data_not_only_in_the_prose():
    """让后续分析（诊断台账 / 记录）拿得到，而不是只能去解析中文。"""
    r = _run(1.25, 3.0)
    assert r.data["waited_s"] == pytest.approx(1.25)
    assert r.data["switch_off_delay_s"] == pytest.approx(3.0)


# ════════════════════════════════════════════════════════════════════════════
# 判据本身不许被这次改动动到
# ════════════════════════════════════════════════════════════════════════════

def test_a_mismatch_is_still_a_failure():
    """写后回读不一致仍然判失败 —— 这条是撞针保护，不是措辞问题。"""
    assert _run(9.0, 5.0).success is False


def test_the_read_back_state_is_still_reported_honestly():
    r = _run(0.5, 2.0, on=False, enable=True)
    assert r.data["z_controller_on"] is False
    assert r.data["requested"] is True
    assert r.data["verified"] is True


def test_the_original_sentence_is_still_there():
    """增加诊断字段时保留可识别的用户提示。"""
    r = _run(0.2, 5.0)
    assert "Z 反馈开关未生效" in r.error
    assert "实时控制器回报" in r.error


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
