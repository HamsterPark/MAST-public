"""「本实验期间在役的是哪根针」——区间交集，两端都可能开口。

2026-08-06 要求：「针尖记录等是不是也应该在实验记录中?」

针尖是仪器域的真源，不搬家：一根针跨很多次实验，一次实验也可能换好几根，
所以它与实验 / 样品**并列**而不是它们的下级。这里做的是**展示层聚合** ——
``tips`` 表一个字都不写。

这份测试几乎全在钉 **NULL 的处理**，因为这一块的每一种写错法都是**静默变空**：
``WHERE a.x <= b.y`` 形式的条件碰上 NULL 求值成 NULL、被 WHERE 丢掉，而
「实验还没结束」（``end_time IS NULL``，大多数行就是这样）和「针尖还在役」
（``removed_at IS NULL``）恰恰是**最常见**的两种情况。两条都漏掉的结果是一个
空区块，读起来像「这次实验没换过针」—— 一句关于仪器历史的假陈述，而且看不出
它是假的。

从仓库根运行::

    .venv-v2-py313/Scripts/python.exe -m pytest \
        tests/v2/unit/logging/test_tips_in_service_during.py -q
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

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

from mast.logging.storage import ExperimentStorage  # noqa: E402


@pytest.fixture()
def store(tmp_path) -> ExperimentStorage:
    return ExperimentStorage(str(tmp_path / "exp.db"))


def _experiment(store, exp_id: str, start: str, end: str | None) -> None:
    """直接写库。走 create_experiment 的话 start_time 只能是「现在」，
    而这一组要摆布的正是时间轴。"""
    with store._connect() as conn:
        conn.execute(
            "INSERT INTO experiments (id, name, goal_text, start_time, end_time, "
            "status, notes) VALUES (?,?,?,?,?,?,?)",
            (exp_id, exp_id, "", start, end, "running", ""),
        )


def _tip(store, name: str, installed: str | None, removed: str | None,
         created: str | None = None) -> str:
    import uuid
    tip_id = str(uuid.uuid4())
    with store._connect() as conn:
        n = conn.execute("SELECT COUNT(*) AS n FROM tips").fetchone()["n"]
        conn.execute(
            "INSERT INTO tips (id, tip_index, name, material, fabrication, form, "
            "installed_at, removed_at, installed_by, note, created_at) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (tip_id, n + 1, name, "W", "etched", "stm_wire",
             installed, removed, "", "", created or installed or "2026-01-01T00:00:00"),
        )
    return tip_id


def _names(rows) -> list[str]:
    return [r["name"] for r in rows]


# ══════════════════════════════════════════════════════════════════════════
# 基本区间交集
# ══════════════════════════════════════════════════════════════════════════

def test_a_tip_installed_and_removed_inside_the_experiment(store):
    _experiment(store, "e1", "2026-08-01T00:00:00", "2026-08-10T00:00:00")
    _tip(store, "整段都在里面", "2026-08-03T00:00:00", "2026-08-05T00:00:00")
    assert _names(store.tips_in_service_during("e1")) == ["整段都在里面"]


def test_a_tip_that_spans_the_whole_experiment(store):
    """装得比实验早、拆得比实验晚 —— 整场都是它。

    只比「装入时刻落在实验区间里」的写法会漏掉这一根，而它恰恰是**最典型**的
    情况：一根针用上好几周，中间做了七八次实验。
    """
    _experiment(store, "e1", "2026-08-01T00:00:00", "2026-08-10T00:00:00")
    _tip(store, "跨越整场", "2026-07-01T00:00:00", "2026-09-01T00:00:00")
    assert _names(store.tips_in_service_during("e1")) == ["跨越整场"]


def test_tips_entirely_before_or_after_are_excluded(store):
    _experiment(store, "e1", "2026-08-01T00:00:00", "2026-08-10T00:00:00")
    _tip(store, "太早", "2026-07-01T00:00:00", "2026-07-20T00:00:00")
    _tip(store, "太晚", "2026-08-20T00:00:00", "2026-08-25T00:00:00")
    assert store.tips_in_service_during("e1") == []


def test_several_tips_come_back_in_install_order(store):
    """一次实验里换了三根针，按装入先后排 —— 那就是它们服役的顺序。"""
    _experiment(store, "e1", "2026-08-01T00:00:00", "2026-08-10T00:00:00")
    _tip(store, "第三根", "2026-08-07T00:00:00", None)
    _tip(store, "第一根", "2026-08-02T00:00:00", "2026-08-04T00:00:00")
    _tip(store, "第二根", "2026-08-04T00:00:00", "2026-08-07T00:00:00")
    assert _names(store.tips_in_service_during("e1")) == ["第一根", "第二根", "第三根"]


# ══════════════════════════════════════════════════════════════════════════
# NULL —— 每一条都是「静默变空」的入口
# ══════════════════════════════════════════════════════════════════════════

def test_an_unfinished_experiment_still_finds_its_tips(store):
    """``end_time IS NULL`` = 实验还没结束。

    这是**大多数行**的状态（这个仓的纪律是实验没有结束、更没有归档）。
    naive 的 `installed_at < end_time` 在 NULL 上求值成 NULL、被 WHERE 丢掉，
    于是「正在做的这次实验」永远看不到任何针尖 —— 而那正是最想看的那一次。
    """
    _experiment(store, "e1", "2026-08-01T00:00:00", None)
    _tip(store, "当前针", "2026-08-03T00:00:00", None)
    _tip(store, "以后才装的", "2026-09-01T00:00:00", None)
    got = _names(store.tips_in_service_during("e1"))
    assert "当前针" in got
    # 开口右端 ⇒ 之后装的针也算在这次「还在进行」的实验期间内。
    assert "以后才装的" in got


def test_empty_string_end_time_counts_as_unfinished(store):
    """空串和 NULL 都要当成「没结束」。

    库里两种都有（老行、导入的行）。只判 NULL 的话，空串那一行会去和每个
    ``installed_at`` 比大小 —— 而任何 ISO 串都 > ''，于是**一根都匹配不上**。
    """
    _experiment(store, "e1", "2026-08-01T00:00:00", "")
    _tip(store, "当前针", "2026-08-03T00:00:00", None)
    assert _names(store.tips_in_service_during("e1")) == ["当前针"]


def test_a_tip_still_in_service_is_found(store):
    """``removed_at IS NULL`` = 这根针还装在仪器上。同样是最常见的那一行。"""
    _experiment(store, "e1", "2026-08-01T00:00:00", "2026-08-10T00:00:00")
    _tip(store, "还在役", "2026-08-03T00:00:00", None)
    rows = store.tips_in_service_during("e1")
    assert _names(rows) == ["还在役"]
    assert rows[0]["removed_at"] is None


def test_both_ends_open(store):
    """实验没结束 + 针尖还在役 —— 两个 NULL 撞一起，最常见的组合。"""
    _experiment(store, "e1", "2026-08-01T00:00:00", None)
    _tip(store, "当前针", "2026-08-03T00:00:00", None)
    assert _names(store.tips_in_service_during("e1")) == ["当前针"]


# ══════════════════════════════════════════════════════════════════════════
# 口径：宁可多列一根，而且要说出来是哪一根不确定
# ══════════════════════════════════════════════════════════════════════════

def test_a_backfilled_date_only_install_still_matches(store):
    """``installed_at`` 允许人工回填成纯日期（UI 明确鼓励）。

    字符串比较下 "2026-08-03" 等价于当天 00:00，也就是**偏早**。
    偏早只会把边界上的针多算进来，不会漏掉。
    """
    _experiment(store, "e1", "2026-08-03T09:00:00", "2026-08-10T00:00:00")
    _tip(store, "回填日期", "2026-08-03", None)
    assert _names(store.tips_in_service_during("e1")) == ["回填日期"]


def test_a_date_only_install_is_flagged_inexact(store):
    """「确实在役」和「大概在役」要看得出区别。

    没有这个标记的话，一个下界会被印得和一个确切时刻一模一样 —— 而读的人
    无从知道哪一根是推出来的。
    """
    _experiment(store, "e1", "2026-08-01T00:00:00", None)
    _tip(store, "回填日期", "2026-08-03", None)
    _tip(store, "确切时刻", "2026-08-04T11:22:33", None)
    by = {r["name"]: r["overlap_exact"] for r in store.tips_in_service_during("e1")}
    assert by == {"回填日期": False, "确切时刻": True}


def test_a_tip_with_no_install_time_falls_back_to_created_at(store):
    """``installed_at`` 可以为空。退回 ``created_at`` —— 那一列全长、必非空。

    不退回的话这一行的左端是 NULL，比较求值成 NULL，这根针从此在任何实验里
    都查不到 —— 一行不完整的历史数据把它自己从记录里抹掉了。
    """
    _experiment(store, "e1", "2026-08-01T00:00:00", "2026-08-10T00:00:00")
    _tip(store, "没填装入时刻", None, None, created="2026-08-05T00:00:00")
    rows = store.tips_in_service_during("e1")
    assert _names(rows) == ["没填装入时刻"]
    assert rows[0]["overlap_exact"] is False, "用的是下界，必须说出来"


def test_the_created_at_fallback_still_respects_the_window(store):
    """退回 ``created_at`` 不等于「无条件算进来」。"""
    _experiment(store, "e1", "2026-08-01T00:00:00", "2026-08-10T00:00:00")
    _tip(store, "库里写得太晚", None, None, created="2026-12-01T00:00:00")
    assert store.tips_in_service_during("e1") == []


# ══════════════════════════════════════════════════════════════════════════
# 边界与退化
# ══════════════════════════════════════════════════════════════════════════

def test_a_tip_removed_exactly_at_the_experiment_start_is_excluded(store):
    """半开区间：拆针那一刻正是实验开始那一刻 = 这次实验没用过它。"""
    _experiment(store, "e1", "2026-08-01T00:00:00", "2026-08-10T00:00:00")
    _tip(store, "刚好拆掉", "2026-07-01T00:00:00", "2026-08-01T00:00:00")
    assert store.tips_in_service_during("e1") == []


def test_an_unknown_experiment_returns_empty_not_everything(store):
    """查不到的实验返回空。

    「找不到实验」若被当成「没有时间窗约束」，就会把**全部**针尖列出来 ——
    一个看起来信息很丰富的错答案。
    """
    _tip(store, "某根针", "2026-08-03T00:00:00", None)
    assert store.tips_in_service_during("no-such-experiment") == []


def test_no_tips_at_all_is_empty_not_an_error(store):
    _experiment(store, "e1", "2026-08-01T00:00:00", None)
    assert store.tips_in_service_during("e1") == []


def test_it_never_writes_to_the_tips_table(store):
    """**展示层聚合**：tips 表是针尖的唯一真源，这个方法一个字都不写。"""
    _experiment(store, "e1", "2026-08-01T00:00:00", None)
    _tip(store, "针", "2026-08-03T00:00:00", None)
    with store._connect() as conn:
        before = [dict(r) for r in conn.execute("SELECT * FROM tips").fetchall()]
    store.tips_in_service_during("e1")
    with store._connect() as conn:
        after = [dict(r) for r in conn.execute("SELECT * FROM tips").fetchall()]
    assert before == after
