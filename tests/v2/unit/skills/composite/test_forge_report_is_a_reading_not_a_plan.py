"""收尾报告必须来自回读且带采样时刻；读不到时明确缺失，不回退到计划值。"""
from __future__ import annotations

import sys
from pathlib import Path

_MASTV2_ROOT = str(Path(__file__).resolve().parents[5] / "MASTv2")
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

from mast.core.noble_tip_workflow import NOBLE_METAL_BASELINE  # noqa: E402
from mast.core.types import SkillResult  # noqa: E402
from mast.skills.composite import forge_au_tip as F  # noqa: E402

#: 流程表里的**计划**值 —— 报文里出现这两个数(而硬件不是这个数)就是在说谎。
PLAN_BIAS_V = NOBLE_METAL_BASELINE.junction_bias_v          # 0.05 V
PLAN_SETPOINT_PA = NOBLE_METAL_BASELINE.junction_setpoint_a * 1e12   # 1000 pA


class _Rec:
    def __init__(self, value=None, error=""):
        self.error = error
        # Nanonis 回包形状:(header, error, body) —— 载荷在 index 2。
        self.return_value = None if error else ("", b"", [value])
        self.method = ""
        self.args = ()


class RigCtx:
    """假 ExecutionContext。**硬件读数是这个类唯一认真的部分。**

    ``hw`` 里给什么,回读就读到什么;``dead=True`` 让每个读都报错(E_STOP 之后
    TCP 死掉的样子)。子技能一律成功但什么都不改 —— 这个文件不关心流程怎么走,
    只关心收尾那句话说的是不是它刚量到的东西。
    """

    def __init__(self, hw: dict | None = None, *, dead: bool = False):
        self.hw = dict(hw or {})
        self.dead = dead
        self.run_id = "forge-report-test"
        self.reads: list[str] = []

    def safe_call(self, method, *args, role="main", allow_on_abort=False):
        self.reads.append(method)
        if self.dead:
            return _Rec(error="TCP 连接已断开")
        if method not in self.hw:
            return _Rec(error=f"{method} 本机不支持")
        return _Rec(self.hw[method])

    def run(self, skill_name, params, version=None):
        return SkillResult(skill_name=skill_name, success=True, data={})

    def check_abort(self):
        return False

    def check_halt(self):
        return ""


# 合成回读和计划设置不同，用于检测报告是否错误照抄计划。
SYNTHETIC_HW = {
    "Bias_Get": 0.8,
    "ZCtrl_SetpntGet": 350e-12,
    "Current_Get": 340e-12,
    "ZCtrl_ZPosGet": -2.0e-8,
    "ZCtrl_OnOffGet": 1,      # 实时控制器:反馈闭合
    "ZCtrl_StatusGet": 2,     # 模块:On
}


@pytest.fixture(autouse=True)
def _isolate_project_root(tmp_path, monkeypatch):
    """sidecar 落在 tmp,别写进用户真实的 experiments/。"""
    monkeypatch.setenv("MAST2_PROJECT_ROOT", str(tmp_path))


@pytest.fixture
def _gates_open(monkeypatch):
    """两道入口闸默认放行 —— 除非某条测试自己关上。"""
    monkeypatch.setattr(F, "_tip_shaper_preflight", lambda executor: False)
    monkeypatch.setattr(F, "_qplus_blocked", lambda executor, name, params: False)


def _run(ctx, **params) -> dict:
    """跑一次 ForgeAuTip,返回它的 result.data。"""
    skill = F.ForgeAuTip()
    res = skill.execute(ctx, dict(params))
    return dict(res.data or {})


# ── 决定性对照:入口闸门拒绝,什么都没改 ────────────────────────────────────

def test_gate_rejected_run_does_not_claim_the_instrument_is_at_the_forge_junction(
        monkeypatch, _gates_open):
    """入口拒绝后的报告仍须读取状态，不能引用计划设置。"""
    monkeypatch.setattr(F, "_qplus_blocked", lambda executor, name, params: True)
    ctx = RigCtx(SYNTHETIC_HW)

    data = _run(ctx, max_sites=1, max_rounds_per_site=2)
    summary = data["summary_cn"]

    # 说谎的那两个数(计划值)不许出现。
    assert f"{PLAN_BIAS_V:g} V" not in summary, summary
    assert f"{PLAN_SETPOINT_PA:g} pA" not in summary, summary
    assert "仪器留在" not in summary, summary
    # 该出现的是刚量到的那两个。
    assert "0.8 V" in summary, summary
    assert "350 pA" in summary, summary
    # 而且它真的去读了硬件,不是从哪个缓存里抄的。
    assert "Bias_Get" in ctx.reads
    assert "ZCtrl_SetpntGet" in ctx.reads


def test_the_plan_is_kept_but_never_presented_as_state(_gates_open):
    """计划值仍然留在机器可读的数据里(它是有用的),但**不进那句话**。"""
    ctx = RigCtx(SYNTHETIC_HW)
    data = _run(ctx, max_sites=1, max_rounds_per_site=1)

    intended = data.get("intended_junction") or {}
    assert intended.get("bias_v") == PLAN_BIAS_V
    assert intended.get("is_intent_not_reading") is True
    assert f"{PLAN_BIAS_V:g} V" not in data["summary_cn"]


# ── 反馈断开:报文不许说「停在隧穿态」 ──────────────────────────────────────

def test_feedback_off_is_reported_as_not_tunnelling(_gates_open):
    """反馈关闭时必须报告为非隧穿态。"""
    hw = dict(SYNTHETIC_HW, **{"ZCtrl_OnOffGet": 0, "ZCtrl_StatusGet": 1,
                           "Current_Get": 60e-15})
    data = _run(RigCtx(hw), max_sites=1, max_rounds_per_site=1)
    summary = data["summary_cn"]

    assert "停在隧穿态" not in summary, summary
    assert "不是隧穿态" in summary, summary
    assert "断开" in summary, summary


def test_feedback_on_is_not_called_not_tunnelling(_gates_open):
    """反过来也要对:反馈闭合时不许平白说「不是隧穿态」。

    没有这一条,把结论写死成「不是隧穿态」也能让上面那条变绿。"""
    summary = _run(RigCtx(SYNTHETIC_HW), max_sites=1,
                   max_rounds_per_site=1)["summary_cn"]
    assert "不是隧穿态" not in summary, summary
    assert "闭合" in summary, summary


# ── 读不到:说读不到,不许回落到计划值 ──────────────────────────────────────

def test_unreadable_says_so_with_a_reason_and_never_falls_back_to_the_plan(
        _gates_open):
    """TCP 死了。这时候「偏压 0.05 V」比没有这句话更危险。"""
    ctx = RigCtx(dead=True)
    data = _run(ctx, max_sites=1, max_rounds_per_site=1)
    summary = data["summary_cn"]

    assert f"{PLAN_BIAS_V:g} V" not in summary, summary
    assert f"{PLAN_SETPOINT_PA:g} pA" not in summary, summary
    # 逐个量点名 —— 只断言「摘要里某处有『读不到』」的话,某一个量悄悄变成 0
    # 也能让这条测试变绿(读失败静默填零就是这么混过去的)。
    assert "偏压:读不到" in summary, summary
    assert "电流设定:读不到" in summary, summary
    assert "Z 反馈:读不到" in summary, summary
    # 为什么读不到也要说 —— 链路报错和回包读不懂指向不同的下一步。
    assert "TCP 连接已断开" in summary, summary


def test_one_dead_read_does_not_blank_the_others(_gates_open):
    """只有偏压读不到时,其余三个量照报 —— 逐量独立,不是一损俱损。

    也钉住反方向:读不到的那个量**不许**被别处的值顶上。"""
    hw = {k: v for k, v in SYNTHETIC_HW.items() if k != "Bias_Get"}
    summary = _run(RigCtx(hw), max_sites=1,
                   max_rounds_per_site=1)["summary_cn"]

    assert "偏压:读不到" in summary, summary
    assert "350 pA" in summary, summary          # 设定点仍然读到了
    assert f"{PLAN_BIAS_V:g} V" not in summary, summary


# ── 读数必须带时刻:真的数 + 过期 = 最难发现的说谎 ────────────────────────

class _FakeClock:
    """``verify._time`` 的替身。每次 ``time()`` 前进 ``step`` 秒。"""

    def __init__(self, start=1_800_000_000.0, step=0.0):
        self.now = float(start)
        self.step = float(step)

    def time(self):
        v = self.now
        self.now += self.step
        return v

    def localtime(self, t=None):
        return __import__("time").localtime(t)

    def strftime(self, fmt, t=None):
        return "12:00:00"


def test_the_report_says_WHEN_the_reading_was_taken(_gates_open, monkeypatch):
    """回读必须带采样时刻，避免把历史状态说成当前状态。"""
    from mast.skills import verify as V

    monkeypatch.setattr(V, "_time", _FakeClock())
    summary = _run(RigCtx(SYNTHETIC_HW), max_sites=1,
                   max_rounds_per_site=1)["summary_cn"]

    assert "12:00:00" in summary, summary
    assert "截至" in summary, summary
    # 而且要说清楚它是一张快照,不是一个持续成立的断言。
    assert "快照" in summary, summary


def test_a_reading_that_took_seconds_admits_it_is_not_one_moment(
        _gates_open, monkeypatch):
    """六个寄存器是**依次**读的。卡住时它们不属于同一时刻 —— 不许并排摆成一张
    一致的快照。"""
    from mast.skills import verify as V

    monkeypatch.setattr(V, "_time", _FakeClock(step=1.5))   # 跨度 > 1 s
    summary = _run(RigCtx(SYNTHETIC_HW), max_sites=1,
                   max_rounds_per_site=1)["summary_cn"]

    assert "不是同时读的" in summary, summary


def test_a_fast_reading_does_not_cry_wolf(_gates_open, monkeypatch):
    """反过来:读得快的时候不许平白说「不是同时读的」。

    没有这一条,把警告写成无条件的也能让上面那条变绿。"""
    from mast.skills import verify as V

    monkeypatch.setattr(V, "_time", _FakeClock(step=0.0))
    summary = _run(RigCtx(SYNTHETIC_HW), max_sites=1,
                   max_rounds_per_site=1)["summary_cn"]

    assert "不是同时读的" not in summary, summary


def test_the_stamp_is_when_the_read_started_not_when_it_finished(monkeypatch):
    """时刻取自**第一个读之前**,不是最后一个读之后。

    差别只有在回读本身很慢的时候才显出来 —— 而那正是它要紧的时候:一次卡了
    十秒的回读,若自称发生在结束那一刻,就把十秒前的偏压说成了刚刚量到的。
    所以这里用假时钟把「开始」和「结束」拉开,再问它记了哪一个。"""
    from mast.skills import verify as V
    from mast.skills.verify import read_junction_state

    monkeypatch.setattr(V, "_time", _FakeClock(start=1000.0, step=1.5))
    out = read_junction_state(RigCtx(SYNTHETIC_HW))

    assert out["read_at"] == 1000.0, "记的是结束时刻,不是开始时刻"
    assert out["span_s"] >= 1.5, out["span_s"]


def test_each_read_calls_the_verb_it_blames_in_its_error_message():
    """回读表里动词写了两遍(一次给 grep,一次给人话),两遍必须是同一个。

    为什么允许写两遍:``safe_call`` 的动词必须是字面量,否则全仓靠 grep 找 Nanonis
    调用的安全工具都看不见它(``test_safe_call_verbs_are_literal``)。代价是可能
    漂移 —— 于是这条测试从**观测到的调用**反查:让每个读各自失败一次,看它报出来
    的动词是不是它真的调过的那个。错标的话,用户会拿着一个没人调过的方法名去查。
    """
    from mast.skills.verify import _JUNCTION_READS, read_junction_state

    for field, verb, _thunk in _JUNCTION_READS:
        ctx = RigCtx(dead=True)
        out = read_junction_state(ctx)
        assert verb in ctx.reads, f"{field} 标着 {verb},却没有调过它"
        assert verb in out["unreadable"][field], (
            f"{field} 读不到,但原因里写的不是 {verb}:{out['unreadable'][field]}")


def test_measured_values_are_machine_readable_too(_gates_open):
    """给下游的那份不能只有一句人话 —— 而且读不到就是 None,不是 0。"""
    hw = {k: v for k, v in SYNTHETIC_HW.items() if k != "Bias_Get"}
    data = _run(RigCtx(hw), max_sites=1, max_rounds_per_site=1)

    m = data["left_at_measured"]
    assert m["bias_v"] is None
    assert m["unreadable"]["bias_v"]
    assert m["setpoint_a"] == pytest.approx(350e-12)
    assert m["feedback_on"] is True
