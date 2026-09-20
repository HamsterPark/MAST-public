"""旁白文案 —— 确定性中文模板，**没有一条走 LLM**。

形状照抄 ``mast/agents/buffer_summarizer/node.py`` 的 ``describe(kind, payload)``
（那句 docstring 写着 "Deterministic, instant, non-LLM Chinese narration"），但多了
一条它没有的纪律，而这条纪律是本模块存在的全部理由：

**这里没有任何一个参数能让调用方把一句写好的话（或一个编出来的数字）传进来。**

``narrate()`` 的签名里没有 ``text=``。技能只交出它**真的下发给硬件的那份
``params``**，句子由这里拼。于是「10 V / 500 ms 是不是真的」不再取决于谁写得认真：
数字要么从 ``step.params`` 里按 key 取到，要么这句话里**根本不出现数字**
（``fallback``）。这是本仓「移除诱因，别在提示词里说服模型」的第四次落法。

三条硬纪律
----------

1. **``requires`` 里的 key 少一个，就走 ``fallback``。** ``fallback`` 是一句
   *不带任何数字* 的话。缺 ``duration_s`` 时说「我们要打一发脉冲（宽度没记下来）」，
   **不是**「我们要打一发 10 V 脉冲」—— 后者把一个没读到的值说成了读到的。
2. **数字一律过 :func:`si`**，绝不在句子里手写 ``f"{v*1e12:.0f} pm"``。本仓已经为
   单位换算提供统一入口；模板不应另写转换公式。
3. **``render`` 函数体里不许出现字面数字，字符串里不许出现数字。**
   由 ``tests/v2/unit/chat/test_narration_templates_gate.py`` 用 AST 钉住。要算百分比
   就用模块级 helper（:func:`pct`），因为 helper 有名字、有测试，而一个埋在句子里的
   ``* 100`` 只有出事那天才会被人读到。

数字从哪来 —— 单一真源
----------------------

composite 类旁白的 ``data`` 是 ``{"skill", "params", "step_id", ...}``，其中
``params`` 由 ``GraphExecutor`` **原样交出**（``graph_executor.py`` 里
``self._context.run(step.skill_name, step.params)`` 用的就是这一份）。所以「句子里
的电压等于真正下发的电压」不是因为模板写得好，而是因为**中间没有任何人有机会改写
它**。``meta.facts`` 再把用到的原始值存回行里，让这件事**可核对**而不只是可相信。

⚠️ key 名以**代码**为准，不以设计文档为准（2026-08-11 逐个核对过）：

* ``TipPulse``（composite/tip_pulse.py）── ``pulse_v`` / ``duration_s`` / ``count``；
* ``BiasPulse``（builtins/bias_pulse.py）── ``bias_v`` / ``width_s``（**不是**
  ``pulse_v``，两个技能是两套 key，设计文档把它们混成了一套）；
* ``TipShapeWithReadback``（builtins/tip_shaper_readback.py）── 下压深度是
  ``tip_lift_m``（**不是**文档写的 ``depth_m``），负值 = 往表面里压；
* ``ScanAt``（composite/scan_at.py）── ``center_x_m`` / ``center_y_m`` / ``size_m``。
"""

from __future__ import annotations

import math

import logging
from dataclasses import dataclass, field
from typing import Any, Callable

from mast.core.si_quantity import format_si_readable

logger = logging.getLogger(__name__)

#: 句子里嵌入的自由文本（失败原因等）的上限。存储层还会再截到 8000，这里截是为了
#: 让**句子**保持一句话，而不是把一段堆栈糊到用户脸上。
_REASON_MAX = 220

# ── 取值 helper（render 函数体里只许调这些，不许自己算） ────────────────────


#: 米 → 纳米。写成常量而不是散在各处的 ``1e9`` —— 见 ``nm()`` 的自述。
_M_TO_NM = 1e9
#: 「距离 0」的判定阈值(米)。0.0 是一个**有意义**的读数:针尖脚下就是干净的,
#: 那一步**不发移动指令**。浮点比较不写 ``== 0``,而这个数也不许散进句子里。
_ZERO_DIST_M = 1e-12
#: Z 跳变报几位小数 —— 见 ``dz_nm()``。
_DZ_DIGITS = 2


def _finite(v: Any) -> "float | None":
    """能拿来渲染的有限数,否则 None。``num()`` 的无路径版本。"""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def dig(data: dict, path: str) -> Any:
    """按 ``"params.pulse_v"`` 这样的点号路径取值；取不到返回 None。"""
    cur: Any = data
    for part in path.split("."):
        if not isinstance(cur, dict):
            return None
        cur = cur.get(part)
    return cur


def num(data: dict, path: str) -> "float | None":
    """路径上的**有限**数值，否则 None。

    ``float(v or 0)`` 是不行的 —— ``0.0`` 是一个真实的高度/偏压。
    ``bool`` 被排除掉：``True`` 是 ``1``，而「打一发 1 V」和「change_bias=True」
    是两句完全不同的话。NaN / inf 当作没读到：一个渲染成 "nan" 的旁白比没有旁白更糟。
    """
    v = dig(data, path)
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    return f


def text_of(data: dict, path: str) -> str:
    """路径上的字符串（截断到一句话的长度）。取不到 → ``""``。"""
    v = dig(data, path)
    if v is None or isinstance(v, (dict, list, tuple)):
        return ""
    s = str(v).strip().replace("\n", " ")
    return s if len(s) <= _REASON_MAX else s[:_REASON_MAX] + "…"


#: 完整处置建议使用比失败短码更宽的长度上限，避免截掉建议的下一步。
#: 存储层另有 8000 字符上限。
_ADVICE_MAX = 700


def advice_of(data: dict, path: str) -> str:
    """路径上那**一整段**处置建议(比 :func:`text_of` 放得宽)。取不到 → ``""``。

    单独一个函数而不是给 ``text_of`` 加一个 ``limit=`` 形参:上限是一个**决定**
    (「这句话属于哪一类」),不是调用点的排版参数。有名字才会有测试,而散在调用点
    的 ``limit=700`` 迟早会出现第二个、第三个不一样的数。
    """
    v = dig(data, path)
    if v is None or isinstance(v, (dict, list, tuple)):
        return ""
    s = str(v).strip().replace("\n", " ")
    return s if len(s) <= _ADVICE_MAX else s[:_ADVICE_MAX] + "…"


#: 句末标点。转述别人写的那句话时，它可能自带句号，也可能不带。
_SENTENCE_END = "。！？!?.；;"


def unstop(s: str) -> str:
    """把一句转述过来的话的**句末标点去掉** —— 因为后面还要接东西。

    :func:`stop` 的反面。两个都需要,是因为转述别人写的句子时我们不知道它带没带
    句号,而**接在中间**和**放在末尾**要的正好相反。
    """
    return (s or "").rstrip().rstrip(_SENTENCE_END)


def stop(s: str) -> str:
    """给一句转述过来的话补句号 —— **它自己带了就不补**。

    存在的理由是 2026-08-18 念出来的那句「……先跑一次 TiltCalibrate。。」。
    技能的 ``detail`` 是给人读的完整句子（自带句号），而模板又在末尾拼了一个 ——
    两个句号本身无伤大雅，但它是同一个形状的第一次出现：**模板不知道自己转述的
    是一个词还是一句话**，而这条流程正在把越来越多的原文直接念出去。
    """
    s = (s or "").rstrip()
    return s if (not s or s[-1] in _SENTENCE_END) else s + "。"


#: 使用 core.si_quantity 的共用格式化器，将 SI 值转为可读单位。
#: 保留 si 这个短名供模板调用，AST 闸门要求转换集中在模块级 helper。
si = format_si_readable


def pct(frac: float) -> str:
    """``0.125`` → ``"12.5%"``。存在的唯一理由：让 ``* 100`` 有个名字。"""
    try:
        return f"{float(frac) * 100:.4g}%"
    except (TypeError, ValueError):
        return ""


def nm(value: "float | None", *, digits: int = 0) -> str:
    """米 → 纳米。``1.0407e-6 → "-1041"``(不带单位,由调用处写)。

    为什么不用 :func:`si`:坐标要让用户**一眼比得出远近**,而 si 会按量级换
    前缀 —— 同一串落点会一会儿 nm 一会儿 µm,``(959, 1030) nm`` 和 ``(1, 1) µm``
    读起来像两个地方。整条修针流程里横向尺度都在 nm 量级,固定成 nm 才对得起来。

    ⚠️ 它存在的第一理由是 ``test_no_numeric_literal_anywhere_in_a_render_body``:
    **换算不许写在句子里**。``v * 1e9`` 那个 1e9 不在字符串里,却正是
    「3e-12 → 3 米」那一类事故的形状 —— 要算就给它起个名字,
    有名字就会有测试(就是下面这几行的存在理由)。
    """
    v = _finite(value)
    return "" if v is None else f"{v * _M_TO_NM:.{digits}f}"


def ratio(value: "float | None", *, digits: int = 3) -> str:
    """无量纲比值(轴比之类)。``0.7765… → "0.777"``。

    存在的理由和 :func:`nm` 一样:``f"{axis:.3f}"`` 那个 ``.3f`` 也是**在句子里算**
    —— ``test_no_numeric_literal_anywhere_in_a_render_body`` 拦的正是这个形状。
    位数是这里的一个决定(轴比第三位才分得开 0.777 与 0.775),
    写在 helper 里就只有一处;散在句子里就会有第二处、第三处,而且迟早不一致。
    """
    v = _finite(value)
    return "" if v is None else f"{v:.{digits}f}"


def dz_nm(value: "float | None") -> str:
    """Z 跳变量,纳米,**固定两位小数**。

    精度是一个决定,不是排版:``0.03 nm`` 与 ``0 nm`` 是两句不同的话 ——
    前者说「量到了,很小」,后者读起来像「没量」。判定阈值在 ±20 nm 量级,
    而「没打动」那一档的读数常在百分之几 nm,两位小数正好把这两端都说清楚。

    写成 helper 而不是在句子里传 ``digits=2``:那个 2 也是字面数字,
    ``test_no_numeric_literal_anywhere_in_a_render_body`` 一样拦 ——
    **要算就给它起个名字**,而这个名字同时保证了三个分支的精度一致。
    """
    v = _finite(value)
    return "" if v is None else f"{v * _M_TO_NM:+.{_DZ_DIGITS}f}"


def pair(v) -> "tuple | None":
    """一个两元素序列 → ``(a, b)``,否则 None。

    存在的理由是那道字面量闸门:``sep[0]`` / ``len(sep) == 2`` 里的 0/1/2 是
    **下标**不是句子里的数,但闸门分不出来 —— 而它分不出来是对的
    (``v * 1e9`` 里那个 1e9 也「不在字符串里」)。**要下标就给它起个名字。**
    """
    try:
        a, b = v
    except (TypeError, ValueError):
        return None
    return (a, b)


def count_zh(n: float) -> str:
    """脉冲发数之类的整数计数。``1.0 → "1"``（不写成 "1.0"）。"""
    try:
        return f"{int(round(float(n)))}"
    except (TypeError, ValueError):
        return ""


def into_surface(tip_lift_m: float) -> bool:
    """``tip_lift_m`` 是往表面里压（True）还是往上抬（False）。

    Nanonis 的 ``tip_lift_m`` 符号约定：**负 = 压进去**。存在这个 helper 不是为了
    绕过「render 体里不许有字面数字」那道闸门，而是因为闸门问的问题是对的 ——
    一个埋在句子里的 ``< 0`` 只有出事那天才会被人读到，而「针在往哪个方向走」
    正是这条旁白唯一要说清的事。
    """
    try:
        return float(tip_lift_m) < 0
    except (TypeError, ValueError):
        return False


_ZH_CACHE: "dict[str, dict] | None" = None


def skill_zh(name: str) -> str:
    """技能的中文名；查不到就**原样用英文名**（那也是一句真话）。

    真源是 ``mast/webui/skill_zh.json``，读法复用 ``builder_api._zh_sidecar``
    （它同时处理 ``config/skill_zh.json`` 用户覆盖）—— 不在这里抄第二份读法。
    整个查表失败（比如 webui 不可导入）也只是退回英文名。
    """
    global _ZH_CACHE
    if _ZH_CACHE is None:
        try:
            from mast.webui.builder_api import _zh_sidecar

            _ZH_CACHE = _zh_sidecar() or {}
        except Exception:  # noqa: BLE001 — 查不到中文名不该让旁白消失
            logger.debug("skill_zh sidecar unavailable", exc_info=True)
            _ZH_CACHE = {}
    entry = _ZH_CACHE.get(str(name or ""))
    if isinstance(entry, dict):
        zh = str(entry.get("zh") or "").strip()
        if zh:
            return zh
    return str(name or "这一步")


# ── 模板 ────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class Template:
    """一种旁白。

    ``requires`` 里的每一条都是 ``data`` 上的一条**点号路径**；少任何一条都走
    ``fallback``。``records`` 是「也想记进 ``meta.facts`` 但不是必需」的路径 ——
    facts 存在的理由是让事后对账成立（「这句话里的 10 V 是不是真的下发值」），
    所以句子用到的值都该在里面。
    """

    kind: str
    requires: tuple[str, ...]
    render: Callable[[dict], str]
    fallback: str
    tone: str = "info"          # info | good | warn —— 只做配色，不做判断
    records: tuple[str, ...] = field(default=())
    #: 三态旁白的配色**跟着判决走**（2026-08-18）。
    #:
    #: 加这一项的理由是一条模板一个颜色装不下三个结局：``step_split`` 从
    #: 「只在劈裂时开口」改成三态全说之后，``tone="warn"`` 会把「针尖是单尖」
    #: 也画成一条警告；``fwd_bwd_result`` 同理（通过 / 不通过 / 判不了）。
    #:
    #: 它仍然**只做配色，不做判断** —— 判断已经在 ``render`` 写出的那句话里，
    #: 这里只是别让颜色和那句话说两件事。返回空串或抛异常都退回 ``tone``，
    #: 而 ``fallback``（必需字段没读到）永远用 ``tone``：一句不带数字的话
    #: 不该被染成「通过」。
    tone_of: "Callable[[dict], str] | None" = None


@dataclass(frozen=True)
class Rendered:
    text: str
    tone: str
    facts: dict
    #: True = 有必需字段没读到，句子里**没有数字**。前端不用它，但事后查
    #: 「为什么这条旁白没说电压」时它是答案。
    degraded: bool = False


# ── 注册表 ──────────────────────────────────────────────────────────────
#
# 每条模板下面那行注释写的是它的 key 来自哪个文件 —— 因为 key 写错的后果不是报错，
# 是**永远走 fallback**：一句语法完全正确、永远不带数字的话（本仓 `lookup(错名字)
# || 默认` 那一类，已经记过一次）。``test_narration_templates_gate`` 会拿真实的技能
# 元数据核对这些 key 确实存在。



#: 团簇判读三态的**主句**。提成常量不是为了复用 —— 只有一个调用方 ——
#: 是为了让纪律测试断言「第三态没被写成第二态」时,**不必写死任何形容词**。
#:
#: 第一版措辞「圆了 / 判不了」改写之后,
#: 四条钉着那几个字的测试立刻变红。它们钉的其实是三态纪律,不是形容词;
#: 红得没错,却红错了理由。文案还会再改,纪律不会。
CLUSTER_ROUND_HEAD = "这一针的团簇**形状规整**"
CLUSTER_NOT_ROUND_HEAD = "这一针的团簇**形状还不够规整**"
#: ⚠️ 第三态**不是**「不合格」,是「这一张图给不出读数」。两者天差地别:
#: 前者是一个判决,后者是承认没判。这个仓库最常复发的缺陷族就是把后者写成前者。
CLUSTER_UNDECIDABLE_HEAD = "这一针的团簇**给不出可靠读数**"
#: 多针尖同样三态。量不出就说量不出 —— 默不作声等于让人以为「查过了,没有」。
CLUSTER_MULTI_TIP_YES = "**疑似多针尖**"
CLUSTER_MULTI_TIP_UNDECIDABLE = "多针尖这一项这张图看不出来"


def _cluster_sentence(d: dict) -> str:
    """团簇判读的一句话。**三态如实**,「量不出」绝不写成「不合格」。

    ``is_round`` / ``multi_tip`` 都是三态。把 None 折成 False 会让用户读到
    一个系统其实没有下过的判决 —— 这个仓库最常复发的缺陷族,而旁白正是他
    唯一会读的那一层。

    ## 措辞

    第一版写「这一针的团簇**圆**了」/「**判不了**」,这类措辞不够合适。
    ——「圆了」像在夸自己,「判不了」像在推卸。换成读起来像**一份读数**:

        形状规整 / 形状还不够规整 / 团簇太小,这一张给不出可靠读数

    含义一个字没变(尤其第三种仍然**不是**「不合格」),变的是它听起来像
    一台仪器在报数,而不是像一个人在下结论。
    """
    res = d.get("result") if isinstance(d.get("result"), dict) else {}

    def g(k):
        return res.get(k)

    # 字段缺席与字段值为 None 不同，dig 对两者都返回 None。
    # 通用 requires 检查排除 bool 以免把真假当作物理量，但 is_round 是合法布尔判定。
    # 因此该模板直接检查键是否存在，不能把 False 或 True 误判为分析结果缺失。
    if "is_round" not in res:
        return "扎完这一针的团簇图分析完了,但读数没记下来。"

    axis = num(d, "result.equivalent_axis_ratio")
    n = g("n_components")
    area = g("area_px")
    is_round = g("is_round")
    multi = g("multi_tip")

    # 主句:三态
    if is_round is None:
        why = str(g("roundness_undecidable") or "").strip()
        head = CLUSTER_UNDECIDABLE_HEAD
        if why:
            head += f"——{why[:_REASON_MAX]}"
    elif is_round:
        head = CLUSTER_ROUND_HEAD
        if axis is not None:
            head += f",等效轴比 {ratio(axis)}"
    else:
        head = CLUSTER_NOT_ROUND_HEAD
        if axis is not None:
            head += f",等效轴比 {ratio(axis)}"

    bits = []
    if area is not None:
        bits.append(f"面积 {area} px")
    if n is not None:
        bits.append(f"{n} 个连通域")
    # 多针尖:三态。量不出就说量不出,不说「没有」。
    if multi is True:
        bits.append(CLUSTER_MULTI_TIP_YES)
    elif multi is None:
        bits.append(CLUSTER_MULTI_TIP_UNDECIDABLE)

    # 「算了就说」:判决之外那几个量也报出来 —— 它们不参与判决,
    # 但下一次要问「当时到底量到了什么」时,只有这里有。
    wa = num(d, "result.weighted_axis_ratio")
    ba = num(d, "result.boundary_axis_ratio")
    if wa is not None and ba is not None:
        bits.append(f"边界轴比 {ratio(ba)} / 加权轴比 {ratio(wa)}")
    thr_mode = text_of(d, "result.threshold_mode")
    shape_mode = text_of(d, "result.shape_mode")
    if thr_mode or shape_mode:
        bits.append(f"判法 {thr_mode or '—'}+{shape_mode or '—'}")
    bg = num(d, "result.background_level_m")
    if bg is not None:
        bits.append(f"背景 {si(bg, 'm')}")

    tail = ("(" + "、".join(bits) + ")") if bits else ""
    return head + tail + "。"



#: 「去问地图」这一句里,可用区那个数从哪来。写成常量是为了让测试断言它,
#: 而不是断言一句中文 —— 那句话还会被改,这个来源不该跟着改。
FIND_SPOT_ZONE_PATH = "result.effective_half_range_m"


def _find_spot_sentence(d: dict) -> str:
    """地图给了什么 —— **一句拿来 debug 的话**。

    ## 出处

    「问地图要干净区域这种事情也应该做进旁白,方便 debug。
    你再想想,旁白要大幅地细致一点。」

    起因是他在屏幕前看着 forge 连打了 76 发脉冲,以为针尖一直没挪窝。查下来
    **挪了**(每 4 发一次,每轮 5 个落点),但**旁白里一个字都没有** ——
    移动这件事只存在于 ``completed_steps`` 里,而那个要事后翻 sidecar 才看得到。

    ## 三态在这里尤其要紧

    ``map_known=False`` 是**读不到实验记录**,不是「地图上很干净」。两者在几何上
    完全一样,只有这个布尔分得开。读不到的时候往表面打 10 V 脉冲,和在确认干净
    的地方打,是两件不同的事 —— 所以这句话必须说出来。
    """
    x, y = num(d, "result.x_m"), num(d, "result.y_m")
    dist = num(d, "result.distance_m")
    if x is None or y is None:
        return "地图给了一个落点(坐标没记下来)。"

    head = f"地图给的落点:({nm(x)}, {nm(y)}) nm"
    if dist is not None:
        # 0 nm 是一个**有意义**的读数:针尖脚下就是干净的,这一步不发移动指令。
        head += (",就在针尖脚下(不用移动)" if dist < _ZERO_DIST_M
                 else f",距当前位置 {nm(dist)} nm")

    bits = []
    cands = dig(d, "result.candidates")
    if isinstance(cands, list):
        bits.append(f"候选 {len(cands)} 个")
    zone = num(d, FIND_SPOT_ZONE_PATH)
    if zone is not None:
        bits.append(f"可用区 ±{nm(zone)} nm")
    r = num(d, "result.spot_radius_m")
    if r is not None:
        bits.append(f"避让半径 {nm(r)} nm")

    # 避让依据 —— 「谁答上了」。空名单和「两个来源都说干净」几何上一样。
    known = dig(d, "result.map_known")
    seen = num(d, "result.markers_seen")
    if known is False:
        bits.append("⚠️ **读不到实验记录** —— 无法确认此处干净")
    elif known is True:
        bits.append(f"地图上 {int(seen)} 个标记" if seen is not None else "地图读到了")
    else:
        bits.append("地图读没读到**没记下来**")

    crash = num(d, "result.crash_memory_points")
    if crash:
        bits.append(f"另避开本进程记着的 {int(crash)} 个撞针点")
    unloc = num(d, "result.crash_memory_unlocated")
    if unloc:
        bits.append(f"⚠️ {int(unloc)} 次撞针**读不到坐标**,圈画不出来")
    if dig(d, "result.recentred") is True:
        # 换过原点必须说 —— 否则调用方会以为落点就在手边,而它可能在一微米外。
        bits.append("针尖原在可用区外,**已从区中心重搜**")

    return head + ("(" + "、".join(bits) + ")" if bits else "") + "。"


def _pulse_result_sentence(d: dict) -> str:
    """这一发**打动针尖了吗** —— 判定三态,读不到绝不说成没动。

    ``insufficient_data`` 是「Z 读回来有问题」,``none`` 是「Z 确实没跳」。
    两句话指向完全不同的下一步(查信号链 vs 再打一发),混在一起就都问不出来了
    —— ``_tip_phases`` 里 ``unreadable`` 与 ``no_effect`` 分开记账,正是这个道理。
    """
    direction = text_of(d, "result.step.direction")
    dz = num(d, "result.step.delta_m")
    if not direction:
        return "这一发打完了(Z 判定没记下来)。"
    if direction == "insufficient_data":
        return "这一发打完了,但 **Z 读不出来** —— 判不了针尖有没有改变(该查信号链,不是再打一发)。"
    if direction == "none":
        return ("这一发**未改变针尖**(Z 无跳变"
                + (f",{dz_nm(dz)} nm" if dz is not None else "") + ")。")
    zh = "向上" if direction == "up" else "向下"
    # 读数带符号,所以方向词后面不再重复一次正负("向上跳变 +22.00 nm" 里那个 +
    # 是废话)。方向词留着是因为 delta_m 读不到时它仍然说得出发生了什么。
    return ("这一发**改变了针尖** —— Z " + zh + "跳变"
            + (f" {dz_nm(dz)} nm" if dz is not None else "") + "。")



#: ``AutoTilt`` 的 outcome → 这句话的主语。**真源是 ``composite/auto_tilt.py``
#: 的 ``report()``**,而它有**五**个结局,不是三个。
#:
#: ``rolled_back`` 与 ``failed`` 都是「做了没做成」,但下一步不同:前者已经把
#: 压电倾斜写回动手之前的值(现场干净),后者保留了改善过的中间态(现场已经不是
#: 原来那个了)。把两者揉成一句,用户就无法判断「现在这块面是什么状态」。
_TILT_OUTCOME_ZH = {
    "applied": "调平**已执行**",
    "no_action_needed": "调平**无需执行** —— 这一帧的倾斜已在 Z 量程预算内",
    "skipped": "调平**未执行**",
    "failed": "调平**执行失败**(改善过的中间态保留着)",
    "rolled_back": "调平**执行失败**,压电倾斜已写回动手之前的值",
}

#: 哪几个 outcome 算「这块面现在是平的」。``no_action_needed`` 也算 ——
#: 下游问的是「这块地方现在平不平」,不是「我们有没有动过压电」。
#: 与 ``_tip_phases._TILT_DID_ACT`` / ``_TILT_NO_ACT`` 是同一批取值。
_TILT_GOOD_OUTCOMES = ("applied", "no_action_needed")


def _tilt_sentence(d: dict) -> str:
    """根据 AutoTilt 的实际返回说明是否执行、调整量与效果。
    
    读取 outcome、before.z_span_m 和 after.z_span_m，与 _tip_phases._tilt_readout
    使用同一组字段。skipped、z_span_before_m、z_span_after_m 或 action 不是该回包
    契约；字段缺失时应报告未知，不猜测成功或失败。"""
    outcome = text_of(d, "result.outcome")
    if not outcome:
        # 「读不到」不是一个结局。**绝不猜**,也绝不说成「没做成」。
        return "调平这一步跑完了,但是否执行没记下来。"
    head = _TILT_OUTCOME_ZH.get(outcome) or f"调平结局「{outcome}」"

    before = num(d, "result.before.z_span_m")
    after = num(d, "result.after.z_span_m")
    slope = num(d, "result.before.measured_slope_deg")
    trigger = num(d, "result.trigger_z_span_m")
    rounds = num(d, "result.iterations")

    # 「效果」= 帧内 Z 落差前后各是多少。用 si() 而不是 nm():落差跨好几个
    # 量级(nm → pm),而 nm() 取整会把 0.3 nm 报成 "0" —— 读起来像「完全平了」。
    if before is not None and after is not None:
        span = f":帧内 Z 落差 {si(before, 'm')} → {si(after, 'm')}"
    elif before is not None:
        # **失败/未执行时也要报这个数。** 「做了没做成」和「本来就多斜」是
        # 两种处境,而分开它们的就是这个读数。
        span = f"(测到帧内 Z 落差 {si(before, 'm')}"
        if trigger is not None:
            span += f",触发阈 {si(trigger, 'm')}"
        span += ")"
    else:
        span = ""

    bits = []
    if slope is not None:
        # ``:.3g`` 而不是 ``:.3f``:定点会把 0.0004° 印成 "0.000°",读起来像
        # 「正好是零」—— 与 ``_tilt_sentence`` 用 si() 而不用 nm() 是同一条理由
        # (「把 0.3 nm 报成 0」)。有效数字不会把一个非零值抹成零。
        bits.append(f"测到倾斜 {slope:.3g}°")
    if rounds is not None:
        bits.append(f"迭代 {int(rounds)} 轮")
    detail = ("〔" + "、".join(bits) + "〕") if bits else ""

    # ``detail`` 是技能写给**人**读的那句完整话(自带句号)—— 念它,不念
    # ``reason`` 那个短码:「calibration_missing」对用户等于没说。
    why = text_of(d, "result.detail")
    return stop(head + span + detail + (f"——{unstop(why)}" if why else ""))



def _find_terrace_begin(d: dict) -> str:
    x, y = num(d, "x_m"), num(d, "y_m")
    frame, win = num(d, "frame_nm"), num(d, "window_nm")
    seen = num(d, "searched_frames")
    head = "我们去找一块能扎针的台面"
    if x is not None and y is not None:
        head += f":先挪到 ({nm(x)}, {nm(y)}) nm"
    bits = []
    if frame is not None:
        bits.append(f"扫 {frame:.0f} nm 的图")
    if win is not None:
        bits.append(f"要一块 {win:.0f} nm 见方的单台面")
    if seen:
        # 说出来:这是「换一块」而不是「又在原地找」。
        bits.append(f"已经搜过 {int(seen)} 片,这次避开它们")
    return head + ("(" + "、".join(bits) + ")" if bits else "") + "。"


def _terrace_window_shrink(d: dict) -> str:
    """判据说「换小一档就有」—— 说出来我们听了它的话。"""
    a, b = num(d, "from_nm"), num(d, "to_nm")
    if a is None or b is None:
        return "这一片的窗口不够平,按判据的建议换小一档再找。"
    return (f"{a:.0f} nm 的窗在这张图上不够平,判据说 {b:.0f} nm 行 —— "
            f"**听它的**,换小一档再找(比换位置便宜)。")


def _find_terrace_result(d: dict) -> str:
    """找台面的**过程与全部读数** —— 算了就说。

    「有些分析的结果值虽然程序不会参考,但是既然分析了就都说出来。」
    一个算出来却不报的数,下一次出问题时没人知道它当时是多少。
    """
    n = num(d, "found")
    proc = []
    checked, skipped = num(d, "windows_checked"), num(d, "windows_skipped")
    cross = num(d, "windows_cross_terrace")
    if checked is not None:
        proc.append(f"量了 {count_zh(checked)} 个窗")
    if cross:
        proc.append(f"{count_zh(cross)} 个跨台阶被排除")
    if skipped:
        proc.append(f"{count_zh(skipped)} 个被避让点挡住")
    usable = num(d, "usable_rms_pm")
    if usable is not None:
        proc.append(f"可用阈值 {usable:.0f} pm")
    # 多针尖那一问的读数**不在这里报**(2026-08-18)。它有自己的一张卡片
    # (``step_split``,同一张图、紧挨着这一条),而那张卡片报得更全:除了
    # 重影强度和台面能级数,还有判决本身、重影间隔。同一个数在相邻两条旁白里
    # 各说一遍是噪声,而且两处的取法迟早会分家。
    # 调用方仍然传 ``split_score`` / ``split_threshold``(进本模板的 ``records``,
    # 事后对账要用),只是不进句子。
    tail = ("〔" + "、".join(proc) + "〕" if proc else "")

    if not n:
        why = text_of(d, "why")
        return "没找到台面" + (f"——{why}" if why else "") + tail + "。"
    x, y = num(d, "x_m"), num(d, "y_m")
    win, rms = num(d, "window_nm"), num(d, "rms_pm")
    head = f"找到 {count_zh(n)} 个落点"
    bits = []
    if win is not None:
        bits.append(f"窗 {win:.0f} nm")
    if rms is not None:
        bits.append(f"残差 {rms:.0f} pm")
    if x is not None and y is not None:
        bits.append(f"第一个在 ({nm(x)}, {nm(y)}) nm")
    return head + ("(" + "、".join(bits) + ")" if bits else "") + tail + "。"


def _level_begin(d: dict) -> str:
    win, rms = num(d, "window_nm"), num(d, "rms_pm")
    tail = ""
    if win is not None and rms is not None:
        tail = f"({win:.0f} nm 窗,当前残差 {rms:.0f} pm)"
    return f"**就在这块台面上调平**{tail}。"


def _level_result(d: dict) -> str:
    """调平这一步的结局。**四态**,而且四句话指向四个不同的下一步。

    ``AutoTilt`` 的四个 outcome 一一对应(真源 ``composite/auto_tilt.py``):

    * ``applied``          —— 真动了压电,复测达标
    * ``no_action_needed`` —— **本来就够平,一个字没改**
    * ``skipped``          —— 没标定 / 测不到:**没做**,不是没做成
    * ``failed``           —— 做了没做成(发散 / 硬件拒绝 / 迭代用尽)

    2026-08-18 之前这里只有三支,而且判「跳过」读的是一个**不存在的字段**
    (``skipped``,AutoTilt 从来没有回过这个键)—— 于是 ``skipped`` 恒为假,
    每一次「没做」都被念成「没做成」。一个前置条件缺失被报成了一次失败:
    前者要去跑 ``TiltCalibrate``,后者要去查标定为什么失效。

    「没做成」而又说不出原因时**明说没记下来**。原来那一支在没有原因时输出
    「调平没做成。」—— 一句读起来像结论、实际不含任何信息的话。
    """
    before, after = num(d, "before_m"), num(d, "after_m")
    if before is not None and after is not None:
        span = f":台面内落差 {si(before, 'm')} → {si(after, 'm')}"
    elif before is not None:
        span = f"(测到台面内落差 {si(before, 'm')})"
    else:
        span = ""
    why = text_of(d, "reason")
    if dig(d, "done") is True:
        return "调平**已执行**" + (span or "(落差没记下来)") + "。"
    if dig(d, "no_action") is True:
        return "调平**无需执行** —— 该台面的倾斜已在 Z 量程预算内" + span + "。"
    # ``why`` 是技能的 ``detail``，一句自带句号的完整话 —— 走 ``stop`` 补标点。
    if dig(d, "skipped") is True:
        return stop("调平**未执行**"
                    + (f"——{why}" if why else "(多半是本机尚未做过倾斜响应标定)"))
    return stop("调平**执行失败**" + (f"——{why}" if why else "(原因没记下来)"))


def _poke_begin(d: dict) -> str:
    x, y, depth = num(d, "x_m"), num(d, "y_m"), num(d, "depth_pm")
    n, stage = num(d, "n"), text_of(d, "stage")
    head = "我们扎一针"
    if n is not None:
        head += f"(第 {int(n)} 针"
        head += ("、深扎期)" if stage == "deep" else "、临界期)" if stage else ")")
    if depth is not None:
        head += f":下压深度 {depth:.0f} pm"
    if x is not None and y is not None:
        head += f",落点 ({nm(x)}, {nm(y)}) nm"
    return head + "。"


#: ``step_verdict`` 的四态 → 一句中文。**四态不是三态** ——
#: ``insufficient_data`` 与「没扎上」指向完全相反的下一步。
_INDENT_ZH = {
    "cluster": "**已接触**，表面已形成一个团簇",
    "no_change": "**未接触**（Z 没有净变化）",
    "tip_changed_or_pit": "**针尖已改变，或扎出一个坑**",
    "insufficient_data": "**判不了**",
}

#: ``feedback_segment_source`` → 为什么判不了 / 这次判定读的是哪一段。
_FB_SEG_ZH = {
    "current": "第四段由电流定位",
    "no_press": "电流始终没升上去 —— 这一针可能根本没压到表面",
    "no_return": "电流压上去之后**再没回到 setpoint**，采集在反馈接管之前就结束了",
    "too_short": "反馈恢复之后录到的时间太短",
    "no_current": "没有电流通道，段界只能靠尾窗猜",
    "too_few": "电流样本太少",
}


def _poke_indent(d: dict) -> str:
    """扎完这一针，**Z 曲线说了什么** —— 这条此前一个字都没有。

    「修针尖过程中，扎针的检测应该写进旁白，包括读取到的
    z 曲线也应该画出来。」

    ═══════════════════════════════════════════════════════════════════════
    为什么这一句必须带上「读的是哪一段」
    ═══════════════════════════════════════════════════════════════════════

    同一天查出来的事：这个判定此前读的是**第三段** —— 「Z 被斜坡抬回原位、
    反馈还没开」。那一段按构造等于基线，Δz 因此恒为 0，输出「没扎上」。
    168 条历史曲线里 159 条的采集在反馈接管**之前**就结束了，而它们每一条
    都给出了确定的判定。

    所以这句话里 ``feedback_segment_source`` 不是脚注：它区分「读到了该读的那一段，
    结论是未接触」和「压根没读到那一段」。前者该加深，后者该加长采集 ——
    照后者去加深，就是拿针尖去补一个软件问题。
    """
    v = text_of(d, "verdict")
    dz = num(d, "dz_pm")
    tol = num(d, "tol_pm")
    src = text_of(d, "feedback_segment_source")
    seg4 = num(d, "feedback_segment_s")

    head = _INDENT_ZH.get(v, f"判定「{v}」")
    bits = []
    if dz is not None:
        bits.append(f"Δz = {dz:+.0f} pm")
    if tol is not None:
        bits.append(f"容差 ±{tol:.0f} pm")
    body = ("（" + "、".join(bits) + "）") if bits else ""

    why = _FB_SEG_ZH.get(src, "")
    if v == "insufficient_data":
        tail = f" —— {why}。" if why else "。"
        # **不要**在这里劝人加深。判不了的下一步是把采集加长。
        #
        # 句子里不报具体秒数：真源是 ``_tip_phases._POKE_POST_ROLL_S``，
        # 在这里抄一个数就是第二个真源，而两处一旦漂开，念出来的那个数会
        # 稳稳地指向一个没人配过的值。
        return (head + body + tail
                + "这**不是**「未接触」：先把采集窗口 post_roll_s 调长再扎一次，"
                  "别据此增大下压深度。")
    if src == "current" and seg4 is not None:
        body += f"，读的是反馈恢复之后那 {seg4:.2f} s"
    elif why:
        body += f"（{why}）"
    return head + body + "。"


#: ``VerifyAtomicResolution`` 的三态 → 一句中文。
#:
#: **真源是 ``skills/composite/verify_atomic_resolution.py`` 的 ``VERDICT_*``**
#: (``atomic_resolved`` / ``atomic_absent`` / ``undecidable``)。这里抄一份字符串
#: 而不是 import,是因为 ``mast.chat`` 不该反向依赖 ``mast.skills``;两处的平价由
#: ``test_atomic_result_narration.py::test_the_three_verdicts_are_the_skills_own``
#: 钉住 —— 那边改了名字,这边会当场红,而不是安静地退回「判定「xxx」」。
#:
#: ⚠️ **三态不是两态**:``undecidable`` 与 ``atomic_absent`` 指向完全不同的下一步
#: (前者要换视野/像素或重扫,后者要动针尖),把「判不了」念成「没有」等于把一个
#: 测量条件问题报成了针尖问题。
_ATOMIC_RESOLVED = "atomic_resolved"
_ATOMIC_UNDECIDABLE = "undecidable"
_ATOMIC_VERDICT_ZH = {
    _ATOMIC_RESOLVED: "**拿到原子分辨**",
    "atomic_absent": "**这一帧上没有原子分辨**",
    _ATOMIC_UNDECIDABLE: "**判不了** —— 这**不等于**没有原子分辨",
}

#: 角向集中度报几位小数。真实跨度是 1.8（针尖抖动）到 7645（真晶格），
#: 一位小数在两端都读得出来。写成常量而不是句子里的 ``:.1f`` —— 见 ``ratio()``。
_CONC_DIGITS = 1


def _atomic_result(d: dict) -> str:
    """扫到原子分辨这一轮的**结论与全部读数**（2026-08-23）。

    要求：原子分辨这个 skill 应当有自己的旁白体系，展示最终的原子分辨
    图与分析结果，并尽量附上分析图像。图由发出方
    （``composite/achieve_atomic.py::_narrate_atomic_result``）画好挂在
    ``image`` 上，这里负责那句话。

    ═══════════════════════════════════════════════════════════════════════
    为什么这句话要把**四个数**都摆出来，而不是只报一个「过了」
    ═══════════════════════════════════════════════════════════════════════

    区分晶格候选与针尖抖动条纹时，需要报告角向集中度及其判定依据，
    不能只给出“有原子分辨”的结论。集中度应结合当前数据与其他判据解释。

    另外两个数各自排除一种假阳性，缺哪个都不行：

    * **反扫**：只在一个扫描方向上出现的周期是**针尖产物**，不是表面的。
      两个方向都有，这条才排得掉。
    * **帧内前后两半**：证书描述的是**算它的那一段**，不是它结束的那一刻。
      若前后半帧差异明显，整帧结果可能只描述已消失的状态，必须并排报告。

    ``rungs_used`` 是「爬到第几档才拿到」：它回答的不是「成没成」，是**代价花在
    哪**，而那是用户下一次决定从哪一档起步的依据。
    """
    v = text_of(d, "verdict")
    if not v:
        # 「读不到」不是一个判定。不猜，也不默认成「没有」。
        return "原子分辨这一轮跑完了,但结论没记下来。"
    head = _ATOMIC_VERDICT_ZH.get(v) or f"原子分辨判定「{v}」"

    conc, per = num(d, "concentration"), num(d, "period_nm")
    bits = []
    if conc is not None:
        bits.append(f"角向集中度 {ratio(conc, digits=_CONC_DIGITS)}")
    if per is not None:
        bits.append(f"快扫方向周期 {ratio(per)} nm")
    head += ("(" + "、".join(bits) + ")") if bits else ""

    tail = list(_atomic_frame_readings(d))
    rungs = [s.strip() for s in text_of(d, "rungs_used").split(",") if s.strip()]
    if rungs:
        tail.append(_atomic_ladder_tally(rungs))
    return head + "。" + ("；".join(tail) + "。" if tail else "")


def _atomic_frame_readings(d: dict) -> list:
    """三联分析图**本来就量过**的那几个数,说成话。

    两个调用点共用:拿到了那一条(:func:`_atomic_resolution_achieved`)和
    **没拿到**那一条(:func:`_atomic_run_result`)。

    要求详细介绍其分析结果,包括分析图的旁白,
    **无论成或不成**——而在此之前只有拿到了那一支摆得出这几个数,
    没拿到那一支只说一句「没有拿到」。**失败那一次恰恰最需要解释**:
    反扫有没有、周期是多少、帧内两半差多少,正是「为什么不算」的证据,
    而它们本来就在渲染三联图时算过了,只是从没往这条路上交。

    ⚠️ **只摆数,不判决**:两半差多少才算「变过」没有标定过的阈值,
    而未标定的阈值没资格否决人。摆出来用户一眼就看得出
    5731.9 / 17.0 不是同一根针尖。
    """
    out: list = []
    bwd = num(d, "concentration_bwd")
    if bwd is not None:
        out.append(f"反扫 {ratio(bwd, digits=_CONC_DIGITS)}"
                   "(只在一个方向上出现的周期是针尖产物,两个方向都要有)")
    a, b = num(d, "half_a"), num(d, "half_b")
    if a is not None and b is not None:
        out.append(f"帧内前后两半 {ratio(a, digits=_CONC_DIGITS)} / "
                   f"{ratio(b, digits=_CONC_DIGITS)}"
                   "(按扫描顺序切,两半本应相当;差得远说明认证期间针尖变过,"
                   "整帧那个数是两段的混合)")
    return out


#: ``AchieveAtomicResolution`` 的**档位名** → 一句中文。
#:
#: 真源是 ``skills/composite/achieve_atomic.py::plan_dynamic`` 里 ``_note()`` 的
#: 第一个实参(``r1`` / ``r3a:recheck`` / ``surface#2`` …)。这里只认**去掉
#: ``#2`` 站点后缀和 ``:recheck`` 子步骤后缀**的那个词根 —— 后缀由句子另说,
#: 抄进表里会变成一张永远补不齐的表。
#:
#: ⚠️ 查不到就**原样用档位名**(见 :func:`_atomic_rung`)。绝不编一句
#: 「某一档跑完了」—— 一个没被翻译的档位名看起来刺眼,而一句通用话看起来
#: 完全正常,于是漏掉的那一档永远不会被发现(与 :func:`forge_outcome_zh` 同理)。
_ATOMIC_RUNG_ZH: dict[str, str] = {
    "p0": "现状",
    "r1": "重扫档",
    "r2": "偏压抖动 + 阈值浅扎档",
    # 2026-08-25 顺序换过来了(浅扎更温和,见 achieve_atomic 的 R3 自述)。
    # a/b 保持「先/后」的含义,所以标签跟着换 —— 只换一边就会让旁白说反话。
    "r3a": "浅扎档",
    "r3b": "脉冲档",
    "r4": "锻造档",
    "triage": "分诊",
    "surface": "表面分诊",
    "relocate": "粗动换区",
}

#: 每一档的**结局码** → 那句话里不带数字的部分。
#:
#: 真源是 ``achieve_atomic.py`` 里 ``_note(rung, outcome, …)`` 的**第二个实参**。
#: 那是整条流程唯一的记账口,旁白也接在那里 —— 于是「加了一档忘了写旁白」在结构上
#: 不可能发生,而「加了一档忘了写**措辞**」由
#: ``tests/v2/unit/chat/test_achieve_atomic_narration.py::
#: test_every_outcome_the_emitter_can_note_has_a_phrase`` 当场判红。
#:
#: ⚠️ 每一条都要说清**下一步**,不只是「发生了什么」。用户看这条流程时问的是
#: 「它在想什么、为什么升级、为什么停」——一句只报告状态的话答不了那三个问题。
_ATOMIC_OUTCOME_ZH: dict[str, str] = {
    "begin": "开这一档",
    "not_imaging": (
        "**反馈没开、针不在隧道结上** —— 这不是针尖的问题,是我们根本不在成像状态,"
        "后面每一帧都是废的。本流程**不替你进针**(进针要知道样品换没换、粗动在哪、"
        "是不是 4 K,猜错的代价是撞针),流程停止"),
    "found": "判据说**有原子分辨** —— 交给正反扫验证复核",
    "no_lattice": "仍无原子分辨",
    "undecidable_only": (
        "**全部判不了** —— 这**不等于**没有原子分辨。所以**一次针尖都不动**:"
        "在判不了的证据上修针,是在自己造出来的空白上判读,会把一根本来就好的针磨掉。"
        "先按判据给的 remedy 改成像条件(视野太小就扩、像素太粗就缩、残帧就重扫、"
        "死平区就换落点),再跑一次"),
    "no_frame_to_ask": (
        "R1 没交出存盘路径,**表面这一道分诊问不了** —— 这不是「认定它是针尖的问题」,"
        "是这道分诊根本没问成"),
    "surface_ok": (
        "这片表面还有干净台面 ⇒ 这是**针尖**的问题,不是表面的 ⇒ 继续往上爬阶梯"),
    "no_usable_region": (
        "判据说**本片表面已无可用台面** ⇒ 这是**表面**的问题,不是针尖的 ⇒ "
        "去粗动换一片新表面,而不是接着修针(没有台面时修针是治错的病)"),
    "no_site_left": (
        "粗动大地图说**没有下一站**了 ⇒ 流程停止,不接着修针 —— 通常意味着要换样品"),
    "moving": "粗动换区",
    "moved": (
        "粗动换区完成 ⇒ 扫描地图进入**新的坐标代次**,旧坐标全部作废 ⇒ "
        "回到最便宜那一档重来"),
    "tip_conditioning_disabled": (
        "调用方关掉了动针尖(allow_tip_conditioning)⇒ 只做前两档,流程停止"),
    "skipped_budget": (
        "总预算已用完,这一档**没有开**。预算只决定「还要不要开下一档」,"
        "**绝不会在一档中途因为超时把它判失败**"),
    "done": "这一档跑完了 ⇒ 复评读数再决定要不要继续往上爬",
    "not_allowed": (
        "ForgeAuTip **没开**(allow_forge 默认关)。**「前面几档都失败了」不是开它的"
        "理由** —— 它治的是「针尖顶端已经毁掉、要重新锻造」;这片表面用完了则该粗动"
        "换区。都不是的话,再给便宜那档一次机会往往更划算"),
}

#: 「这一档拿到了」的结局码。提成常量是为了让配色那一行不必写死字符串
#: (措辞会改,而三态纪律不会 —— 见 ``_CLUSTER_*`` 那一段的同一条理由)。
_ATOMIC_FOUND = "found"

#: 阶梯上真正的**档**(代价从低到高:重扫 → 偏压抖动 → 脉冲 → 浅扎 → 锻造)。
#:
#: ``rungs_used`` 是一份**记账清单**,里面除了档,还有分诊(``surface`` /
#: ``triage``)和粗动换区(``relocate``,而且它一次动作记两条:moving + moved)。
#: 把它们统统念成「档」,屏幕上就会出现「走过 12 档」—— 而这部阶梯**只有四档**。
#: 用户看到那句话的第一个问题必然是「哪来的 12 档」,而这条旁白本来是要回答
#: 「代价花在哪」的。所以两个数分开报:开过几档,一共走过几步。
_ATOMIC_LADDER_RUNGS: frozenset = frozenset({"r1", "r2", "r3a", "r3b", "r4"})


def _atomic_ladder_tally(rungs: list) -> str:
    """``rungs_used`` → 「阶梯上开过 N 档,连同分诊与换区一共走过 M 步」。"""
    roots = {_atomic_rung_root(s) for s in rungs}
    n_ladder = len(roots & _ATOMIC_LADDER_RUNGS)
    return (f"阶梯上开过 {count_zh(n_ladder)} 档,连同分诊与换区一共走过 "
            f"{count_zh(len(rungs))} 步:" + "、".join(rungs))

#: 走到这些结局就**不会再往上爬了**。配色要显眼:一次跑完而用户没拿到他要的
#: 东西,不该和拿到了长一个样。**只做配色,不做判断** —— 判断已经在句子里。
_ATOMIC_STOP_OUTCOMES: frozenset = frozenset({
    "not_imaging", "undecidable_only", "no_site_left", "no_usable_region",
    "tip_conditioning_disabled", "skipped_budget", "not_allowed",
})

#: 一条 ``rung`` 里表示「这是某一档之后的复评」的后缀。真源同上。
_ATOMIC_RECHECK_SUFFIX = ":recheck"
#: 第二个及以后的站点在档位名后缀上的分隔符(``r1#2``)。真源同上。
_ATOMIC_SITE_SEP = "#"


def _atomic_rung_root(rung: str) -> str:
    """``r3a:recheck#2`` → ``r3a``:剥掉复评后缀与站点后缀,只留档位名词根。

    两个调用点(说成中文、数阶梯上开过几档)共用它 —— 各写一遍这两次
    ``split`` 就是各漂各的,而漂开之后的症状是「有的句子认得出档位、有的认不出」。
    """
    s = str(rung or "")
    return s.split(_ATOMIC_RECHECK_SUFFIX)[0].split(_ATOMIC_SITE_SEP)[0]


def _atomic_rung_zh(rung: str) -> str:
    """档位名 → 「〔重扫档〕」这样的一个前缀。**查不到就原样用它**。

    后缀(``#2`` 站点、``:recheck`` 复评)在这里拆开,由 :func:`_atomic_rung`
    把它们说成话 —— 抄进 :data:`_ATOMIC_RUNG_ZH` 会变成一张笛卡尔积大小的表,
    而且每加一个站点就得补一行。
    """
    base = _atomic_rung_root(rung)
    return _ATOMIC_RUNG_ZH.get(base) or (base or str(rung or ""))


def _atomic_site_no(rung: str) -> "float | None":
    """``r1#2`` → 第 2 个站点。没有后缀 = 第一个站点 → None(句子里不提)。"""
    s = str(rung or "").split(_ATOMIC_RECHECK_SUFFIX)[0]
    if _ATOMIC_SITE_SEP not in s:
        return None
    try:
        return float(s.split(_ATOMIC_SITE_SEP)[1])
    except (TypeError, ValueError, IndexError):
        return None


def _atomic_begin(d: dict) -> str:
    """开跑之前**先把现状念出来** —— 反馈、setpoint、Z,以及这一跑的权限。

    「每一步判断都不出旁白……应该有一个详细介绍其分析结果」。
    在此之前 ``AchieveAtomicResolution`` 只在**最后成功**时说一句话,于是屏幕上
    看到的是一连串通用的「我们开始扫一张图」,而它**先量了什么、据此决定从哪一档
    起步**一个字都没有。

    ⚠️ ``controller_on`` 是**三态**:True / False / **读不到**。
    ``if not controller_on`` 会把「读不到」念成「没开」—— 而本仓这一类
    (「读不到」被折叠成一个具体的值)一天里出现过五次。所以这里逐个比 ``is``。
    """
    on = dig(d, "controller_on")
    if on is True:
        head = "我们在成像状态(Z 反馈**开着**)"
    elif on is False:
        head = "Z 反馈**没开**"
    else:
        head = "Z 反馈状态**读不到**"
    bits = []
    sp = num(d, "setpoint_a")
    if sp is not None:
        bits.append(f"setpoint {si(sp, 'A')}")
    z = num(d, "z_m")
    if z is not None:
        bits.append(f"Z {si(z, 'm')}")
    budget = num(d, "budget_min")
    if budget is not None:
        bits.append(f"预算 {ratio(budget, digits=_CONC_DIGITS)} 分钟")
    head += ("〔" + "、".join(bits) + "〕") if bits else ""
    opts = []
    if dig(d, "allow_tip") is False:
        opts.append("不动针尖")
    if dig(d, "allow_forge") is True:
        opts.append("允许锻造")
    if dig(d, "allow_relocate") is False:
        opts.append("不粗动换区")
    tail = ("(" + "、".join(opts) + ")") if opts else ""
    return ("开始追原子分辨:" + head + tail
            + "。先自己把事实量出来,再按读数决定从阶梯的哪一档起步。")


def _atomic_rung(d: dict) -> str:
    """阶梯上**每一档的判断**,带读数(2026-08-24)。

    ═══════════════════════════════════════════════════════════════════════
    为什么接在 ``_note()`` 上,而不是在每一档各写一句
    ═══════════════════════════════════════════════════════════════════════

    ``_note(rung, outcome, **kw)`` 是 ``plan_dynamic`` 里**唯一**的记账口 ——
    每一档、每一次分诊、每一个停下的理由都从那里过。旁白接在那里,于是
    「加了一档忘了发旁白」在结构上不可能发生;而「加了一档忘了写措辞」由
    :data:`_ATOMIC_OUTCOME_ZH` 的那道闸门当场判红。

    在每一档各写一句是另一种做法,而本仓已经知道它怎么坏:
    ``forge_au_tip.py`` 整个文件曾经**一句旁白都没有**,因为每一处都要有人记得。

    ⚠️ 认不出的结局码**原样念出来**,不编一句通用话 —— 与
    :func:`forge_outcome_zh` 同一条纪律。
    """
    code = text_of(d, "outcome")
    phrase = _ATOMIC_OUTCOME_ZH.get(code) or f"结局「{code}」"
    where = _atomic_rung_zh(text_of(d, "rung"))
    site = _atomic_site_no(text_of(d, "rung"))
    head = f"〔{where}"
    if _ATOMIC_RECHECK_SUFFIX in text_of(d, "rung"):
        head += "复评"
    if site is not None:
        head += f",第 {count_zh(site)} 站"
    head += "〕"

    # ── 读数:有什么摆什么。**一个都不硬凑** ──────────────────────────────
    bits: list[str] = []
    # 采集次数与实际判过的不同帧数必须分别报告。
    # 重复保存或重复评估同一帧不构成新的采集证据；优先使用 frames_judged，
    # 与 attempts 不一致时同时显示，便于发现重复输入。
    n_try = num(d, "attempts")
    n_judged = num(d, "frames_judged")
    if n_judged is not None:
        bits.append(f"真的采到并判过 {count_zh(n_judged)} 帧")
        if n_try is not None and n_try != n_judged:
            bits.append(f"发起过 {count_zh(n_try)} 次 attempt")
    elif n_try is not None:
        bits.append(f"扫了 {count_zh(n_try)} 帧")
    # 下面两条**只在非零时说**：0 次没采到不是新闻,而每多一行都会把有信息量的
    # 那几个数挤淡。读不到(None)时同样不说 —— 「读不到」不是「零次」。
    n_failed = num(d, "scans_failed")
    if n_failed:
        bits.append(f"其中 {count_zh(n_failed)} 次**根本没采到**"
                    "(扫描那一步没成 ⇒ 那几次不存盘、不判定,"
                    "否则存的是上一帧的缓冲)")
    n_repeat = num(d, "repeat_frames")
    if n_repeat:
        bits.append(f"另有 {count_zh(n_repeat)} 次**没给出新样本**"
                    "(交回来的是已经判过的那张,或读数与上一帧逐位相同)")
    n_nojudge = num(d, "assess_failed")
    if n_nojudge:
        # 「采到了但判据没跑成」和「没有原子分辨」是两回事。不说的话,
        # 屏幕上只剩一个「真的采到并判过 0 帧」,而**为什么是 0** 无从查起。
        bits.append(f"还有 {count_zh(n_nojudge)} 帧**采到了、判据却没跑成**"
                    "(这不是「没有原子分辨」——那几帧还在盘上,可以直接重判)")
    concs = dig(d, "concentrations")
    if isinstance(concs, (list, tuple)) and concs:
        shown = [ratio(c, digits=_CONC_DIGITS) for c in concs
                 if _finite(c) is not None]
        if shown:
            bits.append("角向集中度 " + "/".join(shown))
    n_undet = num(d, "n_undetermined")
    if n_undet is not None:
        bits.append(f"其中判不了 {count_zh(n_undet)} 帧")
    n_frames = num(d, "n_frames")
    if n_frames is not None:
        bits.append(f"{count_zh(n_frames)} 帧")
    verdict = text_of(d, "verdict")
    if verdict:
        bits.append(f"判据 verdict={verdict}")
    axis, direction = text_of(d, "axis"), text_of(d, "direction")
    steps = num(d, "steps")
    if axis and direction and steps is not None:
        bits.append(f"沿 {axis}{direction} 走 {count_zh(steps)} 步"
                    "(往哪走、走多少步由粗动大地图给,本流程一个换算都不做 —— "
                    "粗动步长随驱动幅度/负载/温度漂移)")
    outcome_of_skill = text_of(d, "skill_outcome")
    if outcome_of_skill:
        bits.append(f"技能结局 {outcome_of_skill}")
    why = unstop(text_of(d, "note") or text_of(d, "hint") or text_of(d, "reason"))
    if why:
        bits.append(why)
    body = ("〔" + "、".join(bits) + "〕") if bits else ""

    at = num(d, "at_min")
    when = f"(起跑后 {ratio(at, digits=_CONC_DIGITS)} 分钟)" if at is not None else ""
    return head + phrase + body + when + "。"


def _atomic_run_result(d: dict) -> str:
    """整跑的结局 —— **没拿到的时候也要说**。

    「无论成或不成」都要有一段带分析图的旁白。在此之前只有**验证通过**那一支
    发得出声音,于是最需要解释的那一次跑(没拿到)反而是全程最沉默的 ——
    用户屏幕上只剩下一串通用的扫描进度。

    ``advice`` 是 ``aggregate()`` 写给人读的**整段处置建议**,原样转述 ——
    它和这里读的是同一批局部变量,数字不可能对不上;而让调用方把一句成品话传进来
    正是本模块要挡的事,所以它只走 ``data``,不走 ``text=``(那个参数不存在)。
    """
    got = dig(d, "achieved")
    verdict = text_of(d, "verified")
    if got is True and verdict:
        head = ("原子分辨这一跑结束:**拿到了**,验证结论 "
                + (_ATOMIC_VERDICT_ZH.get(verdict) or f"「{verdict}」"))
    elif got is True:
        head = "原子分辨这一跑结束:**拿到了一帧**,但验证结论没记下来"
    elif got is False:
        head = "原子分辨这一跑结束:**没有拿到**"
    else:
        head = "原子分辨这一跑结束,结论没记下来"

    bits = []
    rungs = [s.strip() for s in text_of(d, "rungs_used").split(",") if s.strip()]
    if rungs:
        bits.append(_atomic_ladder_tally(rungs))
    reloc = num(d, "relocations")
    if reloc is not None:
        bits.append(f"粗动换区 {count_zh(reloc)} 次")
    at = num(d, "elapsed_min")
    if at is not None:
        bits.append(f"用时 {ratio(at, digits=_CONC_DIGITS)} 分钟")
    conc = num(d, "concentration")
    if conc is not None:
        bits.append(f"最后一帧角向集中度 {ratio(conc, digits=_CONC_DIGITS)}")
    per = num(d, "period_nm")
    if per is not None:
        bits.append(f"快扫方向周期 {ratio(per)} nm")
    # 三联图本来就量过的那几个数 —— **失败那一次最需要它们**:反扫有没有、
    # 帧内两半差多少,正是「为什么不算」的证据。以前只有拿到了那一支摆得出来。
    bits += _atomic_frame_readings(d)
    head += ("〔" + "、".join(bits) + "〕") if bits else ""
    advice = unstop(advice_of(d, "advice"))
    return stop(head + (f"。{advice}" if advice else ""))


def _step_failed(d: dict) -> str:
    """一步没成:短码 + **技能写给人读的那句带数字的话**(2026-08-23 补 detail)。

    ``reason`` 是 ``result.error``,形如 ``AutoTilt failed: rolled_back: diverged``
    —— 一个短码。而技能真正说清楚发生了什么的那句在 ``data["detail"]`` 里::

        第 1 轮后残余 Z 占用 6.4 nm,未降到上一轮 9.6 nm 的 70% 以下

    失败的步骤不进 ``sub_results``(``graph_executor`` 只在成功分支收),所以
    ``_narrate_step_result`` 不发、``_data()`` 也读不到 —— **技能说得最清楚的
    那句话,正好在出事的时候被丢掉了**。「调平时的操作和
    效果宜记入旁白。」出事那一次的读数比顺利那一次更该留下来。

    ``detail`` 已经被短码包含时不重复念(有些技能把同一句同时放两处)。
    """
    head = f"{skill_zh(text_of(d, 'skill'))}这一步没成：{text_of(d, 'reason')}"
    why = unstop(text_of(d, "detail"))
    if why and why not in head:
        head += f" —— {why}"
    return head + ("（这一步不是必须的，我们继续往下走。）" if dig(d, "continued")
                   else "（这一步是必须的，任务到此停下。）")


def _back_to_pulse(d: dict) -> str:
    n = num(d, "tries")
    return (f"连续 {int(n) if n else '数'} 针团簇圆度均不达标 —— "
            f"**退回去打脉冲**重修一次,再来扎。"
            f"(继续扎针是把落点问题当成针尖形状问题处理。)")



#: ``AssessTipSharpness`` 的三态(真源 ``skills/builtins/tip_sharpness.py`` 与
#: ``forge_au_tip._accept`` 的 ``kind``)。写成常量是因为有三处要认它们,
#: 而三处各写一遍字符串就是三处各错一次的机会。
_ACCEPT_PASS = "pass"
_ACCEPT_FAIL = "fail"

#: 唯一那个「好消息」的结局代码。见 ``FORGE_OUTCOME_ZH``。
_OUT_READY = "ready"

#: 扎完一针之后的决策。**受控词表**,不是自由文本 —— 见 :func:`_poke_decision`。
#: 真源是 ``_tip_phases.poke_phase`` 里 ``_decide()`` 的第一个实参。
_POKE_DEEPER = "deeper"          # 图上没东西 ⇒ 没扎上 ⇒ 加深
_POKE_SAME_DEPTH = "same_depth"  # 扎上了不圆 ⇒ 换地方,深度不动
_POKE_SHALLOWER = "shallower"    # 同深度试满 ⇒ 搬多了 ⇒ 变浅
_POKE_FLOOR = "floor"            # 已在最浅档,不再变浅
_POKE_ROUND = "round"            # 单峰且圆,连续计数 +1
_POKE_STAGE = "stage_switch"     # 扫图确认表面变了 ⇒ 转临界深度搜索

#: 修针外环的结局代码 → **一行**中文。「旁白应该尽可能多
#: 涵盖内容。」在此之前 ``forge_au_tip.py`` 整个文件**一句旁白都没有** ——
#: 站点为什么结束、为什么换区、验收判了什么,全在沉默里,只有跑完那份报告里才有。
#:
#: ⚠️ **这不是 ``forge_au_tip._OUTCOME_CN`` 的副本,是它的另一种排版。**
#: 那一张是**报告**用的,每条两三句、带处置建议(「不建议直接加大预算重跑」);
#: 这一张是**卡片**用的,一行。两者是同一个代码的两种呈现,不是两份事实。
#:
#: 真正的风险是**一边加了代码另一边忘了** —— 那会让一个新结局在旁白里显示成
#: 一个英文 slug。所以有一道闸门钉住两张表的键完全相同:
#: ``tests/v2/unit/chat/test_forge_outcomes_are_all_narratable.py``。
FORGE_OUTCOME_ZH: dict[str, str] = {
    "ready": "**针尖已达标** —— 正反扫描线重合、团簇单峰且圆度合格",
    # ``verified`` **不在这里** —— 那是 ``PrepareNobleTip`` 的结局代码,
    # 修针外环从不赋它。放进来会被闸门当成「只有旁白认得的死代码」判红,
    # 而那正是它该做的:一条永远不会被用到的翻译,等于一句没有证据的
    # 「已经支持了」(本仓 `producer_wired_consumer_absent` 的同一形状)。
    "surface_spent": "本片表面已无干净落点 —— 需粗动换区",
    "verify_exhausted": "反复大修后正反扫描线仍不重合",
    "verify_inconclusive": "正反扫一致性判不了 —— 既非合格也非不合格,换位置重测",
    "refine_incomplete": "精修未能使团簇达到判据",
    "sharpness_not_met": "台阶边缘锐度判为钝(blunt)—— 未通过验收",
    "pulse_rescue_exhausted": "脉冲救援次数已用尽,本站不再重试",
    "time_budget_exhausted": "时间预算耗尽 —— **需人工介入排查**",
    "sites_exhausted": "站点预算耗尽,针尖仍未达标",
    "spinning": "连续多站**未产出任何测量** —— 这不是「未达标」,是「未测量」",
    "critical_alert": "电流监控 CRITICAL —— **立即停止**,先查针尖与信号链",
    "hard_cap": "触发失控保险(站点数硬上限)—— 这本身是缺陷信号",
    "round_hard_cap": "触发失控保险(单站轮数硬上限)—— **这不是关于针尖的结论**",
    "coarse_budget_exhausted": "没有可去的新站点(受粗动行程 / 站点间距限制)",
    "relocate_failed": "粗动换位失败,流程停止",
    "stopped_early": "扫描被中止 —— 这是事实不是判断",
    "aborted": "被中止",
    "incomplete": "未走完",
    "relocated": "已换位",
}


def forge_outcome_zh(code: str) -> str:
    """结局代码 → 一行中文。**查不到就原样用代码** —— 那也是一句真话。

    绝不编一句「流程已结束」之类的通用话:一个没被翻译的代码看起来刺眼,
    而一句通用话看起来完全正常 —— 后者会让一个漏掉的结局永远不被发现。
    """
    return FORGE_OUTCOME_ZH.get(str(code or ""), str(code or ""))


def _site_begin(d: dict) -> str:
    """开一个新站点。一站几十分钟,这一条不算多 —— 它是后面所有旁白的坐标系。"""
    n = num(d, "site_no")
    head = f"── 第 {count_zh(n)} 站开工" if n is not None else "── 新站点开工"
    x, y = num(d, "x_m"), num(d, "y_m")
    if x is not None and y is not None:
        head += f",落在 ({nm(x)}, {nm(y)}) nm"
    if dig(d, "entry_check") is True:
        head += " —— **先验针尖状态,不预先施加脉冲**"
    elif dig(d, "first_site") is True:
        head += " —— 第一站,修针的第一发脉冲是强制的"
    return head + "。"


def _site_result(d: dict) -> str:
    """一个站点结束了,结论是什么、跑了几轮。"""
    n, rounds = num(d, "site_no"), num(d, "rounds")
    who = f"第 {count_zh(n)} 站" if n is not None else "这一站"
    tail = f"(大修⇄验证 {count_zh(rounds)} 轮)" if rounds is not None else ""
    return f"{who}结束:{forge_outcome_zh(text_of(d, 'outcome'))}{tail}。"


def _run_result(d: dict) -> str:
    """整条修针跑完了。走过几站、结论是什么。"""
    sites = num(d, "sites")
    tail = f"(一共走了 {count_zh(sites)} 站)" if sites is not None else ""
    return f"修针结束:{forge_outcome_zh(text_of(d, 'outcome'))}{tail}。"


def _accept_result(d: dict) -> str:
    """报告台阶边缘锐度的三态结果。
    
    缺少阈值或无法测量时只报读数和原因，不把未知当成不合格。
    只有明确测得 blunt 才构成拒绝；旁白需展示该判定及其依据。"""
    kind = text_of(d, "kind")
    res = num(d, "edge_resolution_nm")
    limit = num(d, "sharp_edge_nm")
    floor = num(d, "sampling_floor_nm")
    bits = []
    if res is not None:
        bits.append(f"边缘 {res:.2f} nm")
    if limit is not None:
        bits.append(f"阈值 {limit:.2f} nm")
    if floor is not None:
        bits.append(f"这张图的采样极限 {floor:.2f} nm")
    inst = num(d, "fwd_bwd_instability")
    if inst is not None:
        bits.append(f"正反扫不稳定度 {ratio(inst)}")
    tail = ("〔" + "、".join(bits) + "〕" if bits else "")
    why = text_of(d, "reason")
    if kind == _ACCEPT_PASS:
        return "台阶边缘锐度**通过验收**" + tail + "。"
    if kind == _ACCEPT_FAIL:
        return ("台阶边缘**锐度不足 —— 未通过验收**" + tail
                + "。这是一次确切的否定测量,不是「未测出」。")
    # ``why`` 是技能写的、自带句号的完整句子 —— 拼之前先把它的句号去掉,
    # 不然会得到「…(不是不合格)。〔采样极限 0.78 nm〕。」这种两个句号夹一个括号。
    return stop("台阶边缘锐度**判不了**" + (f"——{unstop(why)}" if why else "")
                + tail + "。**这不是「针尖不合格」**,是本站无法完成验收")


def _pulse_result(d: dict) -> str:
    """一批大修脉冲打完了 —— 打了几发、Z 到底动没动。

    ⚠️ 收的是 ``dz_m``(**米**)。:func:`dz_nm` 那个名字说的是**输出**单位,
    它自己做 m→nm 的换算，调用方不能按函数名误传纳米值。旁白使用 SI 基本单位
    (``before_m`` / ``x_m`` / ``delta_m``),这里跟着来。
    """
    fired = num(d, "fired")
    head = f"这一轮打了 {count_zh(fired)} 发脉冲" if fired is not None else "脉冲这一轮打完了"
    if dig(d, "satisfied") is True:
        dz = num(d, "dz_m")
        return head + (f",Z 跳变 {dz_nm(dz)} nm —— **针尖已改变**。"
                       if dz is not None else ",**针尖已改变**。")
    why = text_of(d, "reason")
    return stop(head + ",**Z 无跳变**" + (f"——{why}" if why else ""))


def _poke_decision(d: dict) -> str:
    """扎完一针之后**它决定干什么** —— 一条流程里最能说明「它在想什么」的一句。

    这些决策此前只存在于 ``entry["note"]`` 里(进报文、进 sidecar),
    而屏幕上看到的是一针接一针,中间没有任何理由。「在原地反复打脉冲却不说明为什么」
    正是这个形状:动作换了,旁白一个字都没有。

    ⚠️ 句子由这里拼,**不接收调用方写好的那句话**。`entry["note"]` 是报告用的,
    它和这里读的是同一批局部变量,所以两者的数字不可能对不上;而让调用方把
    一句成品话传进来,就等于把这个模块的全部约束绕过去(见模块自述第一条)。
    """
    kind = text_of(d, "kind")
    now, nxt = num(d, "depth_pm"), num(d, "next_depth_pm")
    tries, streak, need = num(d, "tries"), num(d, "streak"), num(d, "need")
    if kind == _POKE_DEEPER:
        return (f"图上没有可判读的团簇 —— **未接触**,下压深度**加深**到 {nxt:.0f} pm"
                f"(原 {now:.0f} pm)。" if nxt is not None and now is not None
                else "图上没有可判读的团簇 —— **未接触**,加深下压再来。")
    if kind == _POKE_SAME_DEPTH:
        return (f"已接触,但团簇圆度不达标 —— **换落点**,深度不变"
                f"({now:.0f} pm)。圆度不达标是落点位置的问题,不是下压深度不足。"
                if now is not None
                else "已接触,但团簇圆度不达标 —— **换落点**,深度不变。")
    if kind == _POKE_SHALLOWER:
        return (f"同一深度换了 {count_zh(tries)} 个落点圆度均不达标 —— 已接触但**材料转移过量**,"
                f"下压深度**减小**到 {nxt:.0f} pm(原 {now:.0f} pm)。"
                if None not in (tries, now, nxt)
                else "同一深度换过几个落点圆度均不达标 —— 减小下压深度再来。")
    if kind == _POKE_FLOOR:
        return (f"同一深度换了 {count_zh(tries)} 个落点圆度均不达标,但**已在最小下压深度**"
                f"({now:.0f} pm)—— 不再减小,继续换落点。"
                if tries is not None and now is not None
                else "已在最小下压深度,不再减小,继续换落点。")
    if kind == _POKE_ROUND:
        return (f"本针团簇**单峰、圆度达标** —— 连续第 {count_zh(streak)} 次"
                f"(判据要求连续 {count_zh(need)} 次)。"
                if streak is not None and need is not None
                else "本针团簇**单峰、圆度达标** —— 连续计数递增。")
    if kind == _POKE_STAGE:
        return (f"扫图确认表面形貌已改变 —— **转入临界深度搜索**,从 {nxt:.0f} pm 起。"
                if nxt is not None else "扫图确认表面形貌已改变 —— 转入临界深度搜索。")
    # 不认识的决策码。**不编一个决策**,也不返回空串 —— 空串会被 ``render``
    # 当成「必需字段没读到」而报 ``degraded``,而这里缺的不是字段,是词表里的一项。
    # (``degraded`` 有确切含义,借用它会让事后查「为什么这条旁白没说数」时读错。)
    return f"本针结束,下一步是「{kind}」—— 旁白未收录该决策码。"


def _poke_result(d: dict) -> str:
    """扎针相结束 —— 达标没有、扎了多少针。"""
    n = num(d, "pokes")
    tail = f"(一共扎了 {count_zh(n)} 针)" if n is not None else ""
    if dig(d, "refined") is True:
        return f"**精修达标**:团簇已达单峰、圆度合格{tail}。"
    why = text_of(d, "reason")
    return stop("精修**未达标**" + (f"——{why}" if why else "") + tail)


def _relocate_result(d: dict) -> str:
    """粗动换区做成了没有。**换完必定重新进针**,所以这一句要说出来。"""
    if dig(d, "moved") is True:
        axis, steps = text_of(d, "axis"), num(d, "steps")
        bits = []
        if axis:
            bits.append(f"{axis} 轴")
        if steps is not None:
            bits.append(f"{count_zh(steps)} 步")
        return ("**已粗动换区**" + ("(" + "、".join(bits) + ")" if bits else "")
                + ",并已重新进针 —— 下一站是一片未经处理的表面。")
    return stop("**无法换区**" + (f"——{forge_outcome_zh(text_of(d, 'outcome'))}"
                                  if text_of(d, "outcome") else ""))


def _fwd_bwd_result(d: dict) -> str:
    """正反扫描线重不重合 —— 修针环里的**第一道**判据。**三态**。

    判据:正反扫描重合是基本判据,阈值取得较宽;团簇
    圆度排在其后。它是 B 相唯一的判决,而 2026-08-18 之前 **B 相一句旁白都没有** ——
    屏幕上只看得见「扫了一张图」,紧接着针尖要么被放行、要么挨一批脉冲,
    中间那个理由从来不出现——旁白没有展示扫描线重合的分析。

    为什么必须是三态:``inconclusive``(读不到线数据、或者这块地方太平)
    **不是**「针尖不好」。当成不好会拿脉冲去打一根可能好好的针 ——
    这条纪律 ``verify_phase`` 早就写下来了,这句话只是把它说出来。

    「重合度」而不是「相似度」:用户看的是同一条线来回走出的两条曲线叠不叠得上,
    这也是 ``forge_au_tip`` 全篇的用词(「正反扫描线重合」)。
    """
    sim, thr = num(d, "similarity"), num(d, "threshold")
    reading = ""
    if sim is not None:
        reading = f"重合度 {ratio(sim)}"
        if thr is not None:
            reading += f"、阈值 {ratio(thr)}"
    tail = f"({reading})" if reading else ""
    # ⚠️ **判决没读到 ≠ 不通过。**
    #
    # 三态的最后一支是 `return 不重合`,所以任何一次「两个键都没传到」都会念出
    # 一句「针尖还没到基本态,回去打脉冲」—— 一个**根本没做过的判决**,而且它
    # 指向的是打脉冲。本仓 `read_failure_folded_into_a_value` 的又一次:
    # 「读不到」被折叠成了一个具体的、合理得没人会去核的值。
    #
    # 判的是**键在不在**,不是它的真假值:`passed=False` 是一次真的不通过,
    # 两个键都缺才是「没读到」。(bool 进不了 `requires`,所以这道门只能在这儿。)
    if "passed" not in d and "inconclusive" not in d:
        return "正反扫描线这一步跑完了,**判决没记下来**" + tail + "。"
    if dig(d, "inconclusive") is True:
        why = text_of(d, "reason")
        # 判不了的时候**不报重合度** —— 那个数要么根本没有,要么正是它不可信
        # 才走到这一支。把它印出来会让人拿一个已经被判为不可用的读数去比阈值。
        return (stop("正反扫描线这一问**判不了**" + (f"——{why}" if why else ""))
                + "**这不等于针尖没问题**,也不等于针尖有问题;换位置重新测量。")
    if dig(d, "passed") is True:
        return "正反扫描线**重合**" + tail + ",针尖过了这道基本判据。"
    # 重合度过了、却被多针尖否决 —— **这一句非说不可**。
    #
    # 不说的话看到的是「重合度 0.87 通过」紧接着一批脉冲,而那正是
    # 「在原地反复打脉冲却不说明为什么」的形状:动作换了、旁白一个字都没有。
    # 两条判据看的是两件事:一根劈成两个顶点的针尖**照样来回重复**,
    # 两趟画出的是同一对重影,所以重合度量不到它。
    if dig(d, "fwd_bwd_ok") is True and dig(d, "split_tip") is True:
        return ("正反扫描线**重合**" + tail
                + ",但同一条台阶出现重影 —— **多针尖,一票否决**,"
                  "针尖未达标,回去打脉冲重修。"
                  "(正反扫重复性好只说明扫描稳定,不排除针尖存在多个成像顶点。)")
    return ("正反扫描线**不重合**" + tail
            + " —— 针尖未达基本判据,回去打脉冲重修。")


#: ``vision.double_tip.step_splitting`` 的三个 verdict。真源在那个模块;
#: 写成常量是因为 2026-08-18 起有**两处**要认它们(句子 + 配色),
#: 而两处各写一遍字符串就是两处各错一次的机会。
_V_SPLIT = "split"
_V_SINGLE = "single"


def _step_split_sentence(d: dict) -> str:
    """大图上台阶有没有被劈开 —— **三态**。

    判据:先调平再看所有点的高度统计分组,看台阶高度,
    劈裂就是多针尖,多针尖就直接 pulse。

    尺子是 Au(111) 单原子台阶 —— 一个**有物理零点**的量。单针尖成像时相邻台面
    的高度差是它的整数倍;多针尖把每条台阶重复成两道,直方图里冒出半整数能级。
    (10 nm 簇图上没有这把尺子,所以那里判不了 —— 换一张图才有信号。)
    """
    v = text_of(d, "result.verdict")
    n = len(dig(d, "result.levels_pm") or [])
    score = num(d, "result.score")
    thr = num(d, "result.score_threshold")
    sep = dig(d, "result.separation_px")
    if v == _V_SPLIT:
        head = "**多针尖** —— 同一条台阶在大图上被成像成两道重影"
    elif v == _V_SINGLE:
        head = "大图上未见重影,判为**单针尖**"
    else:
        why = text_of(d, "result.reason")
        return "多针尖这一项**判不了**" + (f"——{why}" if why else "") + "。"
    bits = []
    if score is not None and thr is not None:
        bits.append(f"重影强度 {ratio(score)}(阈值 {ratio(thr)})")
    sp = pair(sep)
    if sp is not None:
        # 解包拿名字,不用下标 —— 下标里的 0/1 也是字面数字,闸门分不出它们
        # 是下标还是句子里的数,而它分不出来是对的(见 ``pair`` 的自述)。
        sx, sy = sp
        bits.append(f"重影间隔 {count_zh(sx)}×{count_zh(sy)} 像素")
    if n:
        bits.append(f"图上 {count_zh(n)} 个台面")
    return head + ("(" + "、".join(bits) + ")" if bits else "") + "。"


TEMPLATES: dict[str, Template] = {
    # ── 扫描 ──
    "scan_at": Template(
        kind="scan_at",
        # composite/scan_at.py ParameterSpec: center_x_m / center_y_m / size_m
        requires=("params.size_m",),
        records=("params.center_x_m", "params.center_y_m", "params.purpose"),
        render=lambda d: (
            f"我们要在 ({si(num(d, 'params.center_x_m'), 'm')}, "
            f"{si(num(d, 'params.center_y_m'), 'm')}) 扫一张 "
            f"{si(num(d, 'params.size_m'), 'm')} 见方的图。"
            if num(d, "params.center_x_m") is not None
            and num(d, "params.center_y_m") is not None
            else f"我们要扫一张 {si(num(d, 'params.size_m'), 'm')} 见方的图。"
        ),
        fallback="我们要扫一张图（尺寸没记下来）。",
    ),
    "scan_start": Template(
        kind="scan_start",
        # builtins/imaging.py StartScan —— 它**没有参数**，扫描参数是仪器当前的那份。
        # 所以这句话不带数字不是降级，是它本来就没有可说的数字。
        requires=(),
        render=lambda d: "我们开始扫一张图（用仪器当前的扫描参数）。",
        fallback="我们开始扫一张图。",
    ),
    "scan_milestone": Template(
        kind="scan_milestone",
        # vision/scan_monitor.py: frac_acquired + summary_zh（那句判读**复用**，
        # 旁白不重新判读画面）
        requires=("frac",),
        records=("scan_id", "ordinal"),
        render=lambda d: (
            f"扫到 {pct(num(d, 'frac'))}。{text_of(d, 'summary_zh')}"
            if text_of(d, "summary_zh")
            else f"扫到 {pct(num(d, 'frac'))}。"
        ),
        fallback="扫描进行中（进度没记下来）。",
    ),
    "scan_done": Template(
        kind="scan_done",
        # scan_monitor 的 100% 里程碑 —— 只在 `saw_idle and completed` 时发。
        requires=(),
        records=("scan_id",),
        render=lambda d: (
            f"这张图扫完了。{text_of(d, 'summary_zh')}"
            if text_of(d, "summary_zh") else "这张图扫完了。"
        ),
        tone="good",
        fallback="这张图扫完了。",
    ),
    "scan_stopped_early": Template(
        kind="scan_stopped_early",
        # **必须和 scan_done 是两句话**。「停止」不等于「达标」—— # 与「假成功」四根因之一。判据不是时间到了，是扫描缓冲里的 NaN 前沿
        # （``_confirm_complete``）。
        requires=("frac",),
        records=("scan_id",),
        render=lambda d: (
            f"扫描提前停下了，只扫到 {pct(num(d, 'frac'))} —— **没有**记为完成。"),
        tone="warn",
        fallback="扫描提前停下了 —— **没有**记为完成（扫到多少没记下来）。",
    ),
    # ── 脉冲 ──
    "tip_pulse": Template(
        kind="tip_pulse",
        # composite/tip_pulse.py ParameterSpec: pulse_v / duration_s / count
        requires=("params.pulse_v", "params.duration_s"),
        records=("params.count",),
        render=lambda d: (
            f"我们要打 {count_zh(num(d, 'params.count'))} 发 "
            f"{si(num(d, 'params.pulse_v'), 'V')} / "
            f"{si(num(d, 'params.duration_s'), 's')} 的脉冲。"
            if num(d, "params.count") is not None
            else f"我们要打一发 {si(num(d, 'params.pulse_v'), 'V')} / "
                 f"{si(num(d, 'params.duration_s'), 's')} 的脉冲。"
        ),
        fallback="我们要打脉冲修针（电压/宽度没记下来 —— 它们由针尖策略表在下发时决定）。",
        tone="warn",
    ),
    "bias_pulse": Template(
        kind="bias_pulse",
        # builtins/bias_pulse.py ParameterSpec: bias_v / width_s
        # ——**不是** pulse_v/duration_s，那是 TipPulse 的 key。
        requires=("params.bias_v", "params.width_s"),
        render=lambda d: (
            f"我们要打一发 {si(num(d, 'params.bias_v'), 'V')} / "
            f"{si(num(d, 'params.width_s'), 's')} 的偏压脉冲。"),
        fallback="我们要打一发偏压脉冲（电压/宽度没记下来）。",
        tone="warn",
    ),
    # ── 扎针 ──
    "poke": Template(
        kind="poke",
        # builtins/tip_shaper_readback.py ParameterSpec: tip_lift_m（负 = 往里压）
        requires=("params.tip_lift_m",),
        records=("params.bias_lift_v", "params.lift_height_m"),
        render=lambda d: (
            f"我们要把针尖压入表面 {si(abs(num(d, 'params.tip_lift_m')), 'm')}。"
            if into_surface(num(d, "params.tip_lift_m"))
            # 正的 tip_lift_m 是**抬起来**，不是扎下去。两件事两句话 —— 把它们
            # 说成同一句，用户就没法从旁白里看出针在往哪个方向走。
            else f"我们要把针抬起 {si(num(d, 'params.tip_lift_m'), 'm')}。"
        ),
        fallback="我们要动针尖整形（深度没记下来 —— 它由针尖策略表在下发时决定）。",
        tone="warn",
    ),
    # ── 换地方 ──
    "relocate": Template(
        kind="relocate",
        # composite/relocate_coarse_xy.py ParameterSpec: axis / direction / steps
        requires=("params.axis", "params.direction", "params.steps"),
        records=("params.reapproach",),
        render=lambda d: (
            f"我们换个地方：沿 {text_of(d, 'params.axis')} 轴 "
            f"{text_of(d, 'params.direction')} 方向粗动 "
            f"{count_zh(num(d, 'params.steps'))} 步。"),
        fallback="我们要换个地方（走法没记下来）。",
        tone="warn",
    ),
    # ── 地图落点选择 ──
    "find_spot": Template(
        kind="find_spot",
        # builtins/clean_spot.py ParameterSpec: purpose
        requires=("params.purpose",),
        records=("params.exclude_spots", "params.count"),
        render=lambda d: (
            f"我们去问地图:哪儿还有干净的地方做"
            f"{'脉冲' if text_of(d, 'params.purpose') == 'pulse' else '扎针'}?"),
        fallback="我们去问地图要一个干净落点。",
        tone="info",
    ),
    "find_spot_result": Template(
        kind="find_spot_result",
        requires=("result.x_m", "result.y_m"),
        records=("result.distance_m", "result.map_known", "result.markers_seen",
                 "result.effective_half_range_m", "result.spot_radius_m",
                 "result.crash_memory_points", "result.crash_memory_unlocated",
                 "result.recentred", "result.piezo_range_source"),
        render=_find_spot_sentence,
        fallback="地图回答了,但落点没记下来。",
        tone="info",
    ),
    # ── 挪窝 ──
    "move_xy": Template(
        kind="move_xy",
        # builtins/navigation.py MoveToXY ParameterSpec: x_m / y_m
        requires=("params.x_m", "params.y_m"),
        render=lambda d: (
            f"我们把针尖挪到 ({nm(num(d, 'params.x_m'))}, "
            f"{nm(num(d, 'params.y_m'))}) nm。"),
        fallback="我们把针尖挪个地方(坐标没记下来)。",
        tone="info",
    ),
    # ── 脉冲打完之后 ──
    "pulse_readback": Template(
        kind="pulse_readback",
        # builtins/bias_pulse_readback.py: data["step"] = verdict
        requires=("result.step.direction",),
        records=("result.step.delta_m", "result.bias_v", "result.width_s"),
        render=_pulse_result_sentence,
        fallback="这一发打完了(Z 判定没记下来)。",
        tone="info",
    ),
    # ── 调平(已知问题:做没做过无法判断,旁白没有说明) ──
    "auto_tilt": Template(
        kind="auto_tilt",
        # builtins 的 AutoTilt:调用方 _tip_phases.level_phase 读的就是这几个键。
        requires=(),
        records=("params.surface_rms_m",),
        render=lambda d: "我们把扫描面调平(把整体倾斜归零)。",
        fallback="我们把扫描面调平。",
        tone="info",
    ),
    "auto_tilt_result": Template(
        kind="auto_tilt_result",
        # 五态都要说得出来,所以 requires 空 —— 由句子自己分辨。
        requires=(),

        # facts 路径以 composite/auto_tilt.py 的 report() 为准。
        # 必须读取实际存在的键，否则旁白会回落且诊断记录为空。
        records=("result.outcome", "result.reason", "result.detail",
                 "result.before.z_span_m", "result.after.z_span_m",
                 "result.before.measured_slope_deg",
                 "result.trigger_z_span_m", "result.accept_z_span_m",
                 "result.iterations"),
        render=_tilt_sentence,
        fallback="调平这一步跑完了,但是否执行没记下来。",
        tone="info",
        # 「没做」(缺标定)与「没做成」(发散/回滚)都要显眼 —— 两者都意味着
        # 后面每一步都建立在一块**没调平**的面上。「不用做」是好消息。
        # 与 ``level_result`` 同一把尺子。
        tone_of=lambda d: ("good" if text_of(d, "result.outcome")
                           in _TILT_GOOD_OUTCOMES else "warn"
                           if text_of(d, "result.outcome") else "info"),
    ),
    "flat_region": Template(
        kind="flat_region",
        requires=(),
        records=("params.same_terrace", "params.scan_path"),
        render=lambda d: "我们在这张图上找一块够平的地方落脚。",
        fallback="我们在找一块够平的地方。",
        tone="info",
    ),
    # ── 找台面 / 调平 / 扎针的状态旁白 ──
    "find_terrace_begin": Template(
        kind="find_terrace_begin", requires=(),
        render=_find_terrace_begin,
        fallback="我们去找一块能扎针的台面。", tone="info"),
    "terrace_window_shrink": Template(
        kind="terrace_window_shrink", requires=(),
        render=_terrace_window_shrink,
        fallback="按判据的建议把平区窗换小一档再找。", tone="info"),
    "find_terrace_result": Template(
        kind="find_terrace_result", requires=(),
        # 2026-08-18 补:这条模板此前**一个 records 都没有** —— 它是本流程里
        # 读数最多的一句话,而事后一个数都对不回去。``records`` 只收标量
        # (``render`` 里那个过滤器),所以 ``levels_pm``(列表)不在这儿;
        # 它连同判决一起进 ``step_split`` 那张卡片的 facts。
        records=("found", "window_nm", "rms_pm", "usable_rms_pm",
                 "windows_checked", "windows_skipped", "windows_cross_terrace",
                 "split_score", "split_threshold", "x_m", "y_m"),
        render=_find_terrace_result,
        fallback="找台面这一步跑完了,结果没记下来。", tone="info"),
    "level_begin": Template(
        kind="level_begin", requires=(),
        render=_level_begin,
        fallback="就在这块台面上调平。", tone="info"),
    "level_result": Template(
        kind="level_result", requires=(),
        # ``code`` 是 AutoTilt 的机器可读短码(calibration_missing / diverged /
        # not_converged …)。句子里念的是 ``reason``(那句人话),短码只进 facts:
        # 念 `calibration_missing` 给用户听等于没说,但事后按短码检索这一跑
        # 出了什么事,靠的正是它。``hint`` 同理(run_tilt_calibrate)。
        records=("done", "no_action", "skipped", "code", "hint",
                 "before_m", "after_m"),
        render=_level_result,
        fallback="调平跑完了,做没做没记下来。", tone="info",
        # 「没做」(缺标定)与「没做成」(发散/硬件拒绝)都要显眼 —— 两者都意味着
        # 后面那一批针扎在一块**没调平的**台面上。「不用做」是好消息。
        tone_of=lambda d: ("good" if dig(d, "done") is True
                           or dig(d, "no_action") is True else "warn")),
    "poke_begin": Template(
        kind="poke_begin", requires=(),
        render=_poke_begin,
        fallback="我们扎一针。", tone="warn"),
    "poke_indent": Template(
        kind="poke_indent",
        # ``verdict`` 真的必需：没有它这句话就没有内容。缺了走 fallback 并标
        # degraded —— 那正是 degraded 的本义。
        requires=("verdict",),
        records=("verdict", "dz_pm", "tol_pm", "feedback_segment_source", "feedback_segment_s",
                 "depth_pm"),
        render=_poke_indent,
        fallback="扎完了，但 Z 曲线的判读没记下来。", tone="info",
        tone_of=lambda d: ("good" if text_of(d, "verdict") == "cluster"
                           else "warn" if text_of(d, "verdict") == "insufficient_data"
                           else "info")),
    # ── 原子分辨结果旁白 ────────
    "atomic_resolution_achieved": Template(
        kind="atomic_resolution_achieved",
        # ⚠️ 路径**全是根级**。发出方
        # (``composite/achieve_atomic.py::_narrate_atomic_result``)是平铺发的,
        # 而同一天 ``poke_indent`` 刚因为「发出方包了一层 result= 而模板写根级」
        # 30/30 全走兜底。两边同层这件事由
        # ``test_atomic_result_narration.py::test_the_emitted_payload_renders_numbers``
        # 从**发出方**那一头钉住 —— 模板自测证明不了发出方发对了形状。
        requires=("verdict",),
        records=("verdict", "concentration", "concentration_bwd", "period_nm",
                 "half_a", "half_b", "rungs_used", "scan_path"),
        render=_atomic_result,
        fallback="原子分辨这一轮跑完了,但结论没记下来。", tone="info",
        # 「没拿到」和「判不了」都要显眼:一次 AchieveAtomicResolution 跑完而
        # 用户没拿到他要的东西,不该和拿到了长一个样。**只做配色,不做判断** ——
        # 判断已经在句子里,而两者的下一步不同(换视野/重扫 vs 动针尖)。
        tone_of=lambda d: ("good" if text_of(d, "verdict") == _ATOMIC_RESOLVED
                           else "warn" if text_of(d, "verdict") else "info")),
    # ── 原子分辨阶梯中每一档的判断 ──
    #
    # 三条一组:开跑先念现状 → 每一档念判断和读数 → 跑完念结局(成或不成)。
    # 中间那条接在 ``_note()`` 上,那是 ``plan_dynamic`` 唯一的记账口 ——
    # 见 :func:`_atomic_rung` 的自述。路径**全是根级**(发出方平铺发),
    # 由 test_achieve_atomic_narration.py 从**发出方**那一头钉住。
    "atomic_begin": Template(
        kind="atomic_begin",
        # ``controller_on`` 是**布尔**,进 requires 就永远走 fallback
        # (``_satisfied`` 排除 bool —— 那是为了防 True 被念成「1 V」)。
        # 「读没读到」由句子自己按 ``is True`` / ``is False`` 分三态。
        requires=(),
        records=("controller_on", "setpoint_a", "z_m", "budget_min",
                 "allow_tip", "allow_forge", "allow_relocate"),
        render=_atomic_begin,
        fallback="我们开始追原子分辨。", tone="info"),
    "atomic_rung": Template(
        kind="atomic_rung",
        # ``outcome`` 真的必需:没有它这句话就没有内容,而它是字符串,进得了
        # requires。缺了走 fallback 并标 degraded —— 那正是 degraded 的本义。
        requires=("outcome",),
        # ⚠️ ``records`` 是「句子里的数**对得回**原始读数」的那张表。加一个
        # 读数时这里漏掉,句子照样念得出来,只是 ``meta.facts`` 里查不到它 ——
        # 一个查不到出处的数,和一个编出来的数在报告里长得一模一样。
        records=("rung", "outcome", "at_min", "attempts", "frames_judged",
                 "scans_failed", "repeat_frames", "assess_failed",
                 "n_undetermined",
                 "n_frames", "verdict", "axis", "direction", "steps",
                 "skill_outcome", "note", "hint", "reason", "path"),
        render=_atomic_rung,
        fallback="原子分辨这一档跑完了,判断没记下来。", tone="info",
        # 「拿到了」是好消息;停下来的那几种都要显眼(用户没拿到他要的东西,
        # 不该和拿到了长一个样);其余是过程,不染色。
        tone_of=lambda d: ("good" if text_of(d, "outcome") == _ATOMIC_FOUND
                           else "warn"
                           if text_of(d, "outcome") in _ATOMIC_STOP_OUTCOMES
                           else "info")),
    "atomic_run_result": Template(
        kind="atomic_run_result",
        # ``achieved`` 是**布尔** —— 同 atomic_begin,不能进 requires。
        requires=(),
        # 三联图量到的四个数也要能对回去 —— 失败那一次它们就是「为什么不算」
        # 的全部证据,而一个查不到出处的数和一个编出来的数在报告里长得一样。
        records=("achieved", "verified", "rungs_used", "relocations",
                 "elapsed_min", "concentration", "concentration_bwd",
                 "period_nm", "half_a", "half_b", "scan_path", "advice"),
        render=_atomic_run_result,
        fallback="原子分辨这一跑结束了,结论没记下来。", tone="info",
        tone_of=lambda d: ("good" if dig(d, "achieved") is True
                           else "warn" if dig(d, "achieved") is False
                           else "info")),
    "back_to_pulse": Template(
        kind="back_to_pulse", requires=(),
        render=_back_to_pulse,
        fallback="扎不出圆团簇 —— 退回去打脉冲重修。", tone="warn"),
    # ── 修针外环:站点 / 验收 / 换区 / 整跑(2026-08-18 之前 forge_au_tip.py
    #    **整个文件一句旁白都没有**) ────────────────────────────────────
    "site_begin": Template(
        kind="site_begin", requires=(),
        records=("site_no", "x_m", "y_m", "entry_check", "first_site"),
        render=_site_begin,
        fallback="我们换到一个新站点开工。", tone="info"),
    "site_result": Template(
        kind="site_result", requires=(),
        records=("site_no", "outcome", "rounds"),
        render=_site_result,
        fallback="这一站结束了,结论没记下来。", tone="info",
        # 只有「修好了」是好消息;别的都值得看一眼。
        tone_of=lambda d: ("good" if text_of(d, "outcome") == _OUT_READY
                           else "warn")),
    "run_result": Template(
        kind="run_result", requires=(),
        records=("outcome", "sites"),
        render=_run_result,
        fallback="修针跑完了,结论没记下来。", tone="info",
        tone_of=lambda d: ("good" if text_of(d, "outcome") == _OUT_READY
                           else "warn")),
    "accept_result": Template(
        kind="accept_result", requires=(),
        records=("kind", "verdict", "edge_resolution_nm", "sharp_edge_nm",
                 "sampling_floor_nm", "fwd_bwd_instability", "reason"),
        render=_accept_result,
        fallback="台阶锐度验收跑完了,读数没记下来。", tone="info",
        # 「量不出来」是 info:它既不是通过也不是不合格。
        tone_of=lambda d: ("good" if text_of(d, "kind") == _ACCEPT_PASS
                           else "warn" if text_of(d, "kind") == _ACCEPT_FAIL
                           else "info")),
    "relocate_result": Template(
        kind="relocate_result", requires=(),
        records=("moved", "axis", "steps", "outcome"),
        render=_relocate_result,
        fallback="粗动换区这一步跑完了,结果没记下来。", tone="info",
        tone_of=lambda d: ("info" if dig(d, "moved") is True else "warn")),
    # ── 脉冲相 / 扎针相的**结论与逐针决策** ──────────────────────────────
    "pulse_result": Template(
        kind="pulse_result", requires=(),
        records=("satisfied", "fired", "dz_m", "reason"),
        render=_pulse_result,
        fallback="这一轮脉冲打完了,结果没记下来。", tone="info",
        tone_of=lambda d: ("info" if dig(d, "satisfied") is True else "warn")),
    "poke_decision": Template(
        kind="poke_decision",
        # ``kind`` 是**真的必需**:没有它就没有决策可说,而它是字符串,
        # 进得了 requires(bool 才进不了)。缺了就走 fallback 并标 ``degraded`` ——
        # 那正是 degraded 的本义。
        requires=("kind",),
        records=("kind", "depth_pm", "next_depth_pm", "tries", "streak", "need"),
        render=_poke_decision,
        # 认不出的 kind ⇒ render 返回空串 ⇒ 走 fallback。**不编一个决策**。
        fallback="扎完这一针,下一步怎么走没记下来。", tone="info",
        tone_of=lambda d: ("good" if text_of(d, "kind") == _POKE_ROUND
                           else "info")),
    "poke_result": Template(
        kind="poke_result", requires=(),
        records=("refined", "pokes", "reason"),
        render=_poke_result,
        fallback="精修这一相跑完了,达标没有没记下来。", tone="info",
        tone_of=lambda d: ("good" if dig(d, "refined") is True else "warn")),
    # ── 正反扫描线重合度:B 相唯一的判决(2026-08-18 才有旁白) ──
    "fwd_bwd_result": Template(
        kind="fwd_bwd_result",
        requires=(),   # passed / inconclusive 是真布尔,进 requires 就永远 fallback
        records=("passed", "inconclusive", "fwd_bwd_ok", "split_tip",
                 "similarity", "threshold", "reason"),
        render=_fwd_bwd_result,
        fallback="正反扫描线看过了,结果没记下来。",
        tone="info",
        # 判不了 = info(既不是好消息也不是坏消息,是没消息)。
        tone_of=lambda d: ("info" if dig(d, "inconclusive") is True
                           else "good" if dig(d, "passed") is True else "warn")),
    # ── 大图上的多针尖:台阶劈裂 ──
    "step_split": Template(
        kind="step_split",
        requires=(),
        records=("result.verdict", "result.score", "result.score_threshold",
                 "result.separation_px", "result.levels_pm", "result.gaps_au",
                 "result.deviations", "result.step_height_pm", "result.frame_m"),
        render=_step_split_sentence,
        fallback="看了一眼大图上的台阶,结果没记下来。",
        tone="warn",
        # 2026-08-18 三态全说之后配色必须跟着判决走:一句「针尖是单尖」
        # 被画成警告,和把它说成警告是同一件事。
        tone_of=lambda d: ("warn" if text_of(d, "result.verdict") == _V_SPLIT
                           else "good" if text_of(d, "result.verdict") == _V_SINGLE
                           else "info")),
    # ── 没成 ──
    "step_failed": Template(
        kind="step_failed",
        # graph_executor._handle_failure 的 msg —— 原样转述，不改写、不安慰。
        requires=("reason",),
        records=("skill", "step_id", "continued", "detail"),
        # ``continued`` 而不是 ``optional``：用户看到「没成」之后要知道的第一件事
        # 是**这次跑还在不在跑**。``optional`` 只是决定它的三个输入之一
        # （``on_step_failed`` 回调也能决定继续），拿它当答案会在回调放行时说错。
        render=_step_failed,
        fallback="有一步没成（原因没记下来）。",
        tone="warn",
    ),
    # ── 结论类(阶段 4,2026-08-16)────────────────────────────────────
    "cluster_roundness": Template(
        kind="cluster_roundness",
        # 字段真源:skills/builtins/cluster_roundness.py 的返回 data。
        # 2026-08-15/16 把这份词汇表钉死之前,这条模板不该存在 ——
        # 见 RESULT_KIND_FOR_SKILL 下面那段。
        requires=(),   # ← 见 _cluster_sentence:bool 进不了 requires
        # ``_satisfied`` 排除所有 bool(防 True 被念成 1 V),
        # 而 is_round 是真正的布尔判读 —— 写进 requires 就永远 fallback。
        # 「读没读到」由句子自己按**键在不在**判。
        records=("result.equivalent_axis_ratio", "result.aspect_ratio",
                 "result.multi_tip", "result.n_components", "result.area_px",
                 "result.shape_mode", "result.threshold_mode",
                 "result.roundness_undecidable"),
        render=lambda d: _cluster_sentence(d),
        # requires 少了 is_round 就走这里。**不编一个判决**。
        fallback="扎完这一针的团簇图分析完了,但读数没记下来。",
        tone="info",
    ),
}


#: 子步骤**开始之前**发哪一条。放在开始之前不是排版偏好：要求的措辞是
#: 未来时的「**即将**打 10V 500ms 脉冲」——它的价值在于用户在
#: 脉冲落下**之前**就知道要发生什么。跑完再说一遍「刚才打了」是另一件事。
BEGIN_KIND_FOR_SKILL: dict[str, str] = {
    "ScanAt": "scan_at",
    "StartScan": "scan_start",
    "FullScan": "scan_start",
    "TipPulse": "tip_pulse",
    "BiasPulse": "bias_pulse",
    "BiasPulseWithReadback": "bias_pulse",
    "TipShapeWithReadback": "poke",
    "RelocateCoarseXY": "relocate",
    # 2026-08-17 加的两条。要求:向地图申请干净区域这件事也要进旁白,
    # 便于 debug。在此之前「换地方」只存在于 completed_steps 里,
    # 而屏幕前看到的是连着 4 发脉冲 —— 于是以为针尖压根没挪。
    "FindCleanSpot": "find_spot",
    "MoveToXY": "move_xy",
    # 已知问题:调平这一步是否执行过无法判断,旁白没有说明。
    "AutoTilt": "auto_tilt",
    "FindFlatRegion": "flat_region",
}

#: 子步骤**成功之后**发哪一条。**故意是空的**（2026-08-11）。
#:
#: 结论类旁白（圆不圆 / 一致不一致 / 扎出团簇没有）要读子技能的 ``result.data``，
#: 而那些字段的真源文件（``_tip_phases.py`` / ``assess_quality.py`` /
#: ``tip_sharpness.py`` / ``forge_au_tip.py``）当时正在被另一条线改。照着一份会变的
#: 词汇表写模板，写出来的是**一句永远走 fallback、看起来却完全正常**的话。
#: 设计文档把这些排在阶段 4，这里就留在阶段 4 —— 一个空 dict 是诚实的，
#: 一条读错字段的模板不是。
RESULT_KIND_FOR_SKILL: dict[str, str] = {
    # 2026-08-16 起有第一条。上面那段说明的前提("词汇表在变")已经不成立:
    # AssessClusterRoundness 的返回字段在 08-15/16 被钉死,有单测和产物声明看着。
    # 其余几个(assess_quality / tip_sharpness)仍留空 —— **它们的词汇表还没钉**,
    # 而一条读错字段的模板会永远走 fallback 且看起来完全正常。
    "AssessClusterRoundness": "cluster_roundness",
    # 2026-08-17。这两个的返回字段各自都钉住了:
    #   · FindCleanSpot  —— provenance 那一组(map_known / markers_seen /
    #     effective_half_range_m / recentred / crash_memory_*),
    #     由 test_find_spot_narration_reads_real_fields.py 对着技能源码核;
    #   · BiasPulseWithReadback —— data["step"] = verdict(direction / delta_m)。
    # 没钉住的仍然不接(assess_quality / tip_sharpness)—— 同一条纪律。
    "FindCleanSpot": "find_spot_result",
    "BiasPulseWithReadback": "pulse_readback",
    # AutoTilt 的返回字段由 _tip_phases.level_phase 逐个读,
    # 那就是它们被钉住的地方(改了那边这里立刻走 fallback)。
    "AutoTilt": "auto_tilt_result",
}


def _satisfied(d: dict, path: str) -> bool:
    """这条必需路径**真的读到了一个能说出口的值**吗。

    三种「看着有其实没有」都要算缺：``None``；空字符串（会渲染成「没成：」后面
    什么都没有）；以及非字符串却不是有限数值的东西 —— ``NaN`` / ``True`` /
    一个 dict。最后一条是关键：``bool`` 是 ``int`` 的子类，``True`` 会安安静静
    地被念成「打一发 1 V」。
    """
    raw = dig(d, path)
    if raw is None:
        return False
    if isinstance(raw, str):
        return bool(raw.strip())
    return num(d, path) is not None


#: 合法的配色。``tone_of`` 返回表外的值一律退回模板默认 —— 一个前端不认识的
#: tone 在界面上会静默变成「没有颜色」,那是「颜色说错了」之外的第二种坏法。
_TONES = ("info", "good", "warn")


def _tone_for(tpl: "Template", d: dict) -> str:
    """这一条该用什么颜色。``tone_of`` 说了算，说不出来就用模板默认。"""
    fn = getattr(tpl, "tone_of", None)
    if fn is None:
        return tpl.tone
    try:
        t = str(fn(d) or "")
    except Exception:  # noqa: BLE001 — 配色算错绝不许把一句真话弄没
        logger.debug("narration tone_of %s raised", tpl.kind, exc_info=True)
        return tpl.tone
    return t if t in _TONES else tpl.tone


def render(kind: str, data: dict) -> "Rendered | None":
    """渲染一条旁白。**不认识的 kind 返回 None** —— 不发，而不是发一句通用的。

    「发一句通用的」听起来更友好，实际是往转录里灌一堆「执行了某个步骤」——
    那既没有信息量，又让真正有信息量的那几条被淹掉。
    """
    tpl = TEMPLATES.get(str(kind or ""))
    if tpl is None:
        return None
    d = data if isinstance(data, dict) else {}
    facts: dict = {}
    for path in tpl.requires + tpl.records:
        v = dig(d, path)
        if v is not None and isinstance(v, (int, float, str, bool)):
            facts[path] = v
    if any(not _satisfied(d, p) for p in tpl.requires):
        return Rendered(text=tpl.fallback, tone=tpl.tone, facts=facts, degraded=True)
    try:
        text = tpl.render(d)
    except Exception:  # noqa: BLE001 — 模板抛了就退回不带数字的那句
        logger.debug("narration template %s raised", kind, exc_info=True)
        return Rendered(text=tpl.fallback, tone=tpl.tone, facts=facts, degraded=True)
    text = (text or "").strip()
    if not text:
        return Rendered(text=tpl.fallback, tone=tpl.tone, facts=facts, degraded=True)
    return Rendered(text=text, tone=_tone_for(tpl, d), facts=facts, degraded=False)


__all__ = [
    "BEGIN_KIND_FOR_SKILL",
    "FIND_SPOT_ZONE_PATH",
    "nm",
    "dz_nm",
    "ratio",
    "RESULT_KIND_FOR_SKILL",
    "Rendered",
    "TEMPLATES",
    "Template",
    "advice_of",
    "count_zh",
    "pair",
    "dig",
    "into_surface",
    "num",
    "pct",
    "render",
    "si",
    "skill_zh",
    "text_of",
]
