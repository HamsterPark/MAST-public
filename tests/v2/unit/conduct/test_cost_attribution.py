"""花销口径 —— **`usd_max` 是一个装得像在拦的闸**,以及它为什么拦不住。

M3-d。这一组钉三件事:

1. **归集维度**:账本只有 ``source`` 一个逐调用归集口(``meta`` 列写得进去,
   但 ``summary()`` / ``recent()`` 都不 SELECT 它)。conduct 的判决调用必须把
   ``source`` 写成 ``conduct:<id>``,否则它记在 ``orchestrator`` 名下,与
   supervisor 自己的路由调用**分不开**;
2. **读不到 ≠ 花了 0**:账本文件不在 ⇒ ``None``;账本在、这份 conduct 名下
   没有记录 ⇒ 一个**答得上来的空**;
3. **不合成一个数**:账本按 provider 原生币种记账,合并需要汇率,而
   ``usd_to_cny_rate()`` 在没有覆盖文件时是**写死的 7.2**。所以这里逐币种给,
   并且**说出来为什么没有合计**。

全程 tmp_path:绝不碰真实 ``experiments/usage_ledger.sqlite``。
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_MASTV2_ROOT = str(Path(__file__).resolve().parents[4] / "MASTv2")
if sys.path[0] != _MASTV2_ROOT:
    while _MASTV2_ROOT in sys.path:
        sys.path.remove(_MASTV2_ROOT)
    sys.path.insert(0, _MASTV2_ROOT)

from mast.conduct.adapters import ConductCost, RuntimeConductCost  # noqa: E402
from mast.conduct.llm_seat import COST_SOURCE_PREFIX, cost_source  # noqa: E402


class _FakeLedger:
    """只实现 ``summary()`` —— 那是读侧唯一用到的方法。"""

    def __init__(self, rows=None, boom=False):
        self.rows = list(rows or [])
        self.boom = boom

    def summary(self, since=None, until=None):
        if self.boom:
            raise RuntimeError("账本锁住了")
        return {"by_source": self.rows}


def _reader(ledger):
    return RuntimeConductCost(ledger_getter=lambda: ledger)


def _row(source, currency, cost, count=1, all_priced=True):
    return {"key": source, "currency": currency, "cost": cost,
            "count": count, "all_priced": all_priced}


# ── 1. 归集维度 ────────────────────────────────────────────────────────

def _sql_consts(fn) -> list:
    """一个函数里所有的字符串常量 —— 取自**已加载的 code object**,不碰磁盘。

    SQL 的列名是**字符串字面量**,不是属性访问,所以 ``co_names`` 看不见它们
    (实测:``summary.__code__.co_names`` 里没有 ``meta``)。而 ``co_consts``
    看得见,并且与 ``inspect.getsource`` 不同 —— 它不会因为别人在同一个文件上方
    插了几行就切错位置。
    """
    return [c for c in fn.__code__.co_consts if isinstance(c, str)]


def test_the_ledger_has_exactly_one_per_call_attribution_dimension():
    """``source`` 是唯一的。``meta`` 写得进去,但**没有任何查询读它** ——
    往 meta 里塞 conduct_id 会是一个没有读端的维度(生产方接好了、消费方不在)。

    这条测试变红 = 有人给 ``meta`` 加了读端,那时归集可以更细。
    """
    from mast.billing import ledger as L

    # 列在不在:直接读那个模块级常量,不读源码。
    assert "meta" in L._SCHEMA, "meta 列没了?"

    for fn in (L.UsageLedger.summary, L.UsageLedger.recent):
        consts = _sql_consts(fn)
        blob = " ".join(consts)
        # **自检**:先证明这个判据看得见 SELECT 的列名。少了这一句,
        # 「meta not in blob」会在「常量里根本没有 SQL」时同样通过 ——
        # 那时它什么都没验(比如有人把 SQL 改成动态拼装)。
        assert "cost" in blob and "currency" in blob, (
            f"{fn.__name__} 的字符串常量里找不到 SELECT 的列名 —— "
            f"这个判据失去了区分力(SQL 改成动态拼的了?),换判法,别让它空转。"
            f"常量:{consts[:4]}")
        assert "meta" not in blob, (
            f"{fn.__name__} 开始读 meta 了 —— 归集维度多了一个,"
            f"去看能不能按 conduct_id 直接查,而不是靠 source 前缀")


def test_the_seat_books_its_spend_under_this_conduct(monkeypatch):
    """判决调用的花销要记在**这份 conduct** 名下。

    不写的话 ``source`` 由 ``agent`` 推出来 = ``"orchestrator"``,与 supervisor
    自己的路由调用混在一起 —— 事后分不开,于是预算没有任何东西可读。
    """
    from mast.conduct.llm_seat import make_decide_route
    from mast.skills.composite import llm_node
    import mast.agents._shared.models as models_mod

    seen: dict = {}

    def _fake_make(agent=None, **kw):
        seen.update(kw)
        seen["agent"] = agent
        return object()

    monkeypatch.setattr(models_mod, "make_chat_model", _fake_make)
    monkeypatch.setattr(llm_node, "log_decision", lambda rec: None)
    monkeypatch.setattr(llm_node, "decide_route",
                        lambda n, i, model=None: {"route": "go"})
    seat = make_decide_route(context=lambda: {"conduct_id": "c-42"})
    seat({"id": "n", "routes": {"go": "继续", "hold": "停"}, "escape": "hold"}, {})
    assert seen["usage_source"] == "conduct:c-42"
    assert seen["agent"] == "orchestrator", "模型档位不该被改,改的只是记账归属"


def test_without_a_conduct_id_the_seat_does_not_invent_an_attribution(monkeypatch):
    """拿不到 conduct_id ⇒ 不传 ``usage_source``,退回默认。

    编一个 ``conduct:`` 前缀的空 id 会造出一个谁都不是的归集桶,而那个桶里的钱
    看起来「已归集」。少记比乱记好 —— 两者都要说得出来。
    """
    from mast.conduct.llm_seat import make_decide_route
    from mast.skills.composite import llm_node
    import mast.agents._shared.models as models_mod

    seen: dict = {}
    monkeypatch.setattr(models_mod, "make_chat_model",
                        lambda agent=None, **kw: seen.update(kw) or object())
    monkeypatch.setattr(llm_node, "log_decision", lambda rec: None)
    monkeypatch.setattr(llm_node, "decide_route",
                        lambda n, i, model=None: {"route": "go"})
    seat = make_decide_route(context=lambda: {})
    seat({"id": "n", "routes": {"go": "继续", "hold": "停"}, "escape": "hold"}, {})
    assert seen.get("usage_source") is None


def test_a_test_double_factory_without_the_kwarg_still_works():
    """按签名决定给不给 ``usage_source``,不靠 try/except TypeError。

    靠捕 TypeError 的话,工厂**内部**抛的 TypeError 会被当成签名不收 ——
    于是静默退回一次不带归集的调用,而没有任何地方会说这件事发生过。
    """
    from mast.conduct.llm_seat import _build_model

    assert _build_model(lambda node: "no-kwarg", {}, "conduct:c-1") == "no-kwarg"
    assert _build_model(lambda node, usage_source="": usage_source,
                        {}, "conduct:c-1") == "conduct:c-1"

    # 工厂**收**这个参数,而它的内部抛了 TypeError。正确行为:原样抛出、只调一次。
    # 靠 try/except 的实现会把它当成「签名不匹配」,退回去**再调一次**不带归集的
    # —— 那一次成功了,于是花销记到 orchestrator 名下,而没有任何地方会说话。
    calls: list = []

    def _boom_with_kwarg(node, usage_source=""):
        calls.append(usage_source)
        if usage_source:
            raise TypeError("这是工厂内部的错,不是签名不匹配")
        return "偷偷退回来的、没有归集的那个模型"

    with pytest.raises(TypeError):
        _build_model(_boom_with_kwarg, {}, "conduct:c-1")
    assert calls == ["conduct:c-1"], (
        f"工厂被调了 {len(calls)} 次 —— 第二次是静默退回的无归集调用")


# ── 2. 读不到 ≠ 花了 0 ─────────────────────────────────────────────────

def test_no_ledger_file_reads_unreadable_and_never_creates_one(tmp_path):
    """账本文件不在 ⇒ ``None``,而且**不建一个空的**。

    ``get_ledger()`` 会建库(与监控 store 同一个坑)。一个被动读者建出来的空库
    按构造就没有东西可读,而且在测试里那是往用户真实实验目录里写。
    """
    path = tmp_path / "nope" / "usage_ledger.sqlite"
    r = RuntimeConductCost(path_getter=lambda: path)
    assert r("c-1") is None
    assert not path.exists(), "被动读者建了一个空账本"


def test_a_conduct_with_no_calls_is_an_answerable_empty_not_unreadable():
    got = _reader(_FakeLedger([_row("orchestrator", "CNY", 1.5)]))("c-1")
    assert isinstance(got, ConductCost)
    assert got.by_currency == {} and got.count == 0
    assert "还没有任何一条调用" in got.reason
    # 「这个币种下什么都没有」⇒ None,不是 0.0
    assert got.amount("CNY") is None


def test_a_ledger_that_explodes_is_unreadable():
    assert _reader(_FakeLedger(boom=True))("c-1") is None


def test_an_empty_conduct_id_has_no_bucket_at_all():
    assert _reader(_FakeLedger())("") is None


def test_only_this_conducts_rows_are_counted():
    rows = [_row(cost_source("c-1"), "CNY", 2.0, count=3),
            _row(cost_source("c-2"), "CNY", 99.0, count=50),
            _row("orchestrator", "CNY", 7.0, count=9)]
    got = _reader(_FakeLedger(rows))("c-1")
    assert got.amount("CNY") == pytest.approx(2.0)
    assert got.count == 3


# ── 3. 不合成一个数 ────────────────────────────────────────────────────

def test_multi_currency_spend_is_reported_per_currency_with_no_total():
    """合并需要汇率,而 ``usd_to_cny_rate()`` 没有覆盖文件时是写死的 7.2 ——
    账本自己都把折算总额标成「never written to the ledger」。

    在这里乘一个汇率会得到一个看起来精确、实际是编的数字,而它会被拿去和
    ``usd_max`` 比大小,然后决定要不要停掉一个跑了六小时的实验。
    """
    rows = [_row(cost_source("c-1"), "CNY", 3.0, count=2),
            _row(cost_source("c-1"), "USD", 0.4, count=1)]
    got = _reader(_FakeLedger(rows))("c-1")
    assert got.currencies == ("CNY", "USD")
    assert got.amount("CNY") == pytest.approx(3.0)
    assert got.amount("USD") == pytest.approx(0.4)
    assert not hasattr(got, "total"), "出现了一个合计 —— 它只能是编的"
    assert "汇率" in got.reason and "不给合计" in got.reason


def test_the_hardcoded_fx_rate_is_still_hardcoded():
    """⚠️ 这条钉的是**「为什么不能合计」的那个前提**,不是一个待办。

    ``usd_to_cny_rate()`` 在没有覆盖文件时返回写死的 7.2。哪天它变成实测对账的
    汇率,这条会红 —— **那时才轮到问**「可以合成一个总额了吗」。
    """
    from mast.billing import pricing

    # 数值字面量住在 ``co_consts`` 里 —— 取自已加载的 code object,不碰磁盘,
    # 于是免疫「别人在同一个文件上方插了几行」那个错位机理。
    # 这是一条**正向**断言,没有空转风险:常量表空了它就红。
    consts = pricing.usd_to_cny_rate.__code__.co_consts
    assert 7.2 in consts, (
        f"汇率不再是写死的 7.2 了 —— 去看它现在是不是实测对账来的。是的话,"
        f"「不能合成总额」这条理由就不成立了,预算口径可以重新设计。"
        f"当前数值常量:{[c for c in consts if isinstance(c, (int, float))]}")


def test_unpriced_calls_are_called_out_not_silently_dropped():
    """单价查不到的调用(``cost_known=0``)会让金额**偏低**。说出来。"""
    rows = [_row(cost_source("c-1"), "CNY", 1.0, all_priced=False)]
    got = _reader(_FakeLedger(rows))("c-1")
    assert got.all_priced is False
    assert "偏低" in got.reason


# ── 4. 那条上限必须承认自己拦不住 ───────────────────────────────────────

def test_the_snapshot_admits_the_usd_cap_does_not_stop_anything():
    """人读快照是用户 approve 之前唯一会读的东西。

    一条印成 `上限 $20.00` 的数字会被读成一道守卫,而它今天什么都不守。
    **装得像在拦的闸比没有闸更危险** —— 看的人会据此放心。
    """
    from mast.conduct.journal import USD_MAX_NOT_ENFORCEABLE, render_spec_markdown

    from tests.v2.unit.conduct._harness import spec as make_spec, stage, step

    s = make_spec([stage("A", [step("A.01", "ScanAt")])])
    text = render_spec_markdown(s, {}, conduct_id="c", experiment_id="e")
    assert "拦不住任何东西" in text
    assert USD_MAX_NOT_ENFORCEABLE in text
    # 每阶段唤醒预算是**真的在拦**的那一条,别把两者说成一样
    assert "这一条真的在拦" in text


def test_the_snapshot_still_names_no_model():
    """这句解释里也不许出现具体模型名 —— 与 llm 闸门那一行同一条理由:
    真正花钱的是回退链当时选中的那家。(这条逮到过那句话的第一版。)"""
    from mast.conduct.journal import USD_MAX_NOT_ENFORCEABLE

    for name in ("kimi", "deepseek", "claude", "gpt", "qwen", "glm", "minimax"):
        assert name not in USD_MAX_NOT_ENFORCEABLE.lower()


def test_the_source_prefix_has_one_home():
    """前缀写在 ``llm_seat``、读在 ``adapters``。两处各写一份字面量,
    改一处就会静默漏掉另一处的所有花销。"""
    assert COST_SOURCE_PREFIX == "conduct:"
    assert cost_source("c-9") == "conduct:c-9"

    from mast.conduct import adapters

    code = adapters.RuntimeConductCost.__call__.__code__
    # 读端**走那个函数**,不自己拼前缀(正向断言,不会空转)。
    assert "cost_source" in code.co_names, (
        f"读端不再走 cost_source() 了 —— 它多半自己拼了一遍前缀。"
        f"它现在访问的名字:{sorted(code.co_names)}")
    literals = [c for c in code.co_consts if isinstance(c, str)]
    # **自检**:先证明这个判据看得见这个函数里的字符串字面量。少了它,
    # 下面那条 not-any 会在「co_consts 抓不到这类字面量」时同样通过。
    assert any("by_source" in c for c in literals), (
        f"co_consts 里找不到已知的字面量 —— 判据失去区分力,换判法。"
        f"常量:{literals[:5]}")
    assert not any("conduct:" in c for c in literals), (
        f"读端里出现了写死的 conduct: 前缀 —— 前缀有了两处真源,"
        f"改一处就会静默漏掉另一处的全部花销。常量:{literals}")
