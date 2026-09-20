"""SpectroscopyAtPositions —— 按一份**显式位置列表**逐点取谱。

坐标来自别处(用户 / 上游搜索 / conduct spec)，本技能只负责「去、采、记账」。
与 :class:`~mast.skills.composite.grid_sts.GridSTS` 的分工判据是坐标的来源：规则
网格用五个数就完整描述四百个点，没有人(更没有模型)需要枚举它们；而一份不规则的
位置列表除了逐个列出没有别的表达方式。

## 与 GridSTS 的三处实质差别(不是风格差别)

**一、生成器 plan，移动失败就不采**(S4 STS 设计 D11)。`GridSTS` 的 plan 是静态的，
所以移动失败之后那条谱**照样采**，只能事后标 `suspect`。生成器可以直接跳过：既不
浪费一次采集，也不产生一条位置是错的数据。`suspect` 因此被收窄成「采了，但位置
不可信」——这才是它该说的话。

**二、.dat 归属自证**(D12)。`AcquireSTS` 找 .dat 的办法是「最近 120 秒内最新的那
一个」——单次采集没问题，**批量循环里是一颗定时炸弹**：只要有一个点没落盘
(autosave 关了 / 磁盘满 / 采集失败)，下一次查询就会拿到**上一个点**的文件，并把它
当成这一个点的数据。三层防线在本文件里做(不改那个共用原语)：采前记水位、每点唯一
basename、采后两条都得对得上。任一不满足 ⇒ 这一点 `dat_path=None` /
`verdict="unrated"` / `reason="dat_attribution_failed"`。**绝不让一个点继承上一个点
的 .dat。**

**三、坐标代次闸门**(D13)。一次横向粗动之后，同样的 (x, y) 指的是另一片表面。
一份跨天存活的坐标列表正是最容易踩这一条的东西，所以 ``expected_coord_epoch``
是**必填**，且这里比 :mod:`mast.core.coord_epoch` 的默认策略更严：那个模块对
「查不到当前代次」的默认处置是放行加告警(它要服务的调用方里有省略代次是常态的)，
而这一批坐标是**别人在别的时刻算出来的**，「读不到」不能当成「对得上」——
**只有确凿的 MATCH 才放行**。加严在调用方，模块本身一个字不改。

## 判决与预算是两件事

逐点的 `AssessSpectrum` 只**记账**，不打断(D10：质量判定在 conduct 的步边界消费)。
本 composite 唯一的早停是 D14 的**预算**：连续 N 条 `discard`、或某一点耗时超出
`per_point_timeout_s`。`unrated` **不计入** consecutive_discard ——「读不到」不是
「不合格」。早停时**不做任何补救**(不修针、不换点)，只如实停下并写进 summary；
补救是上层的事。

## 收尾

含 "sts"/"spectr" 的技能一律被判定为「用 lock-in」，所以**没有任何人会替本流程关
调制**(`_preflight.uses_lockin`)。责任在终端消费者，也就是这里。判据是
``executor.progress.aborted`` —— **中止时不关**：中止不是流程结束。
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

from mast.agents._shared.skill_adapter import wrap_skill
from mast.core.sts_workflow import DEFAULT_CONDITION
from mast.core.types import (
    ParameterSpec,
    SafetyLevel,
    SkillCategory,
    SkillMetadata,
    SkillResult,
)
from mast.skills.composite._base import CompositeSkillGraph
from mast.skills.composite.graph_executor import (
    CompositeProgress,
    CompositeStep,
    GraphExecutor,
    abort_facts,
)

logger = logging.getLogger(__name__)

#: 一批的硬上限(S4 STS 设计 D14)。与 ``BatchRegionsScan._MAX_REGIONS`` 对齐 ——
#: 理由是 checkpoint 载荷与无人值守的爆炸半径，不是物理。逐点记录比 GridSTS 的
#: ``str→str`` 重得多，所以不跟 GridSTS 的 400。
_MAX_POINTS = 64

#: 合理性守卫(米)。±1 mm 是压电+粗动的量程上限，用来抓单位滑落(把 100 nm 写成
#: 100)。与 ``BatchRegionsScan`` 同一个数，同一个理由。
_MAX_ABS_XY = 1e-3

#: 逐点状态闭集。写在这里是为了让「又多了一种状态」这件事必须动这一行。
POINT_STATUSES = ("ok", "suspect", "move_failed", "acquire_failed", "skipped")

#: 位置核对的三态。``unverified`` 不是 ``match`` 也不是 ``mismatch`` ——
#: .dat 头读不出 xy 时我们**不知道**它采在哪，而「不知道」既不该被当成对得上
#: (那会让一条错位的谱以 ok 落在地图上)，也不该被当成对不上(那会把一批好数据
#: 全打成 suspect)。它自己一个计数，并且必须出现在 summary 里。
POSITION_CHECKS = ("match", "mismatch", "unverified")


def _sanitize_tag(raw: str) -> str:
    """把 run_tag 收成一个能进文件名的短串。"""
    return re.sub(r"[^A-Za-z0-9_-]", "_", str(raw or "").strip())[:24]


def _dat_mtime_watermark(context) -> "float | None":
    """采集**之前**候选保存目录里 ``*.dat`` 的最高 mtime;**枚举不了返回 None**。

    三态而不是两态：

    * ``float`` —— 水位。目录枚举得动但一个 .dat 都没有时是 ``0.0``(那是一个真实
      答案：任何新文件都比它新)，不是 ``None``；
    * ``None`` —— **枚举不了**(拿不到候选目录 / 全部读失败)。这一点的归属校验
      因此无法成立，它会让该点变成 ``unrated`` 而不是被放行 —— 「查不了」不是
      「通过」。

    代价是每点多一次 rglob。``AcquireSTS`` 内部本来就要为同一批目录做一次
    (``find_latest_saved``)，所以量级没变。
    """
    try:
        from mast.skills.builtins.scan_extra import _candidate_save_dirs

        roots = _candidate_save_dirs(context)
    except Exception as exc:  # noqa: BLE001 —— 枚举不了就是枚举不了
        logger.debug("SpectroscopyAtPositions: 候选保存目录取不到: %s", exc)
        return None
    if not roots:
        return None
    best = 0.0
    scanned_any = False
    for root in roots:
        try:
            for p in root.rglob("*.dat"):
                try:
                    best = max(best, p.stat().st_mtime)
                except OSError:
                    continue
            scanned_any = True
        except OSError:
            continue
    return best if scanned_any else None


def _check_attribution(path, basename: str,
                       watermark: "float | None") -> "tuple[bool, str]":
    """这条 .dat 到底是不是**这一个点**刚采的(S4 STS 设计 D12 的第三层)。

    两条独立的证据都必须成立：

    1. **文件名含本点的 basename** —— 挡住「Nanonis 没落盘，于是拿到了上一个点的
       文件」。每点 basename 唯一，所以上一个点的文件名对不上。
    2. **mtime 严格高于采前水位** —— 挡住「同一个 run_tag 重跑，上一轮的同名文件
       还躺在那里」。这一条与第 1 条是独立的：第 1 条管不了重名的旧文件，第 2 条
       管不了 Nanonis 忽略 basename 的情形。

    ⚠️ **第 2 条在哪里是薄的，以及为什么可以接受。** 保存目录里一个 .dat 都没有时
    水位是 ``0.0``，任何文件都比它新 —— 这一条那时没有牙齿。但它恰好只发生在
    **本批第一个点**(从第二个点起，上一个点的文件就在目录里，水位就有了参照)，
    而第一个点根本没有「上一个点的文件」可继承 —— 它要防的那件事在那一刻不存在。
    要推翻这个判断，需要举出一个「目录里没有任何 .dat、却能拿到一个不属于本点的
    .dat」的路径。不用墙上时钟去补这个缺口是有意的：文件 mtime 由写文件的那台机器
    盖章，跨时钟比较会在网络盘上把一整批好数据判成 unrated。

    返回 ``(通过, 不通过的原因)``。
    """
    if not path:
        return False, "AcquireSTS 没有返回 .dat 路径(autosave 关着 / 没落盘)"
    name = Path(str(path)).name
    if basename and basename not in name:
        return False, (f"落盘文件名 {name!r} 不含本点的 basename {basename!r} —— "
                       f"这多半是上一个点的文件")
    if watermark is None:
        return False, "候选保存目录枚举不了,采前 mtime 水位未知,无法确认这是新文件"
    try:
        mtime = Path(str(path)).stat().st_mtime
    except OSError as exc:
        return False, f"读不到 {name!r} 的 mtime: {exc}"
    if not mtime > watermark:
        return False, (f"{name!r} 的 mtime 不高于采集前的水位 —— 它在这一点开采"
                       f"之前就已经存在")
    return True, ""


class SpectroscopyAtPositions(CompositeSkillGraph):
    """在一份显式的 (x, y) 列表上逐点取谱，逐点记账。"""

    def metadata(self) -> SkillMetadata:
        return SkillMetadata(
            name="SpectroscopyAtPositions",
            version="1.0.0",
            category=SkillCategory.COMPOSITE,
            safety_level=SafetyLevel.CONFIRM,
            description=(
                "按一份**显式的位置列表**逐点采点谱(JSON 数组,元素是 "
                "{x_m, y_m, [label]},单位**米**)。适用于坐标来自别处的场合 —— "
                "用户、上游的一次搜索、一份 conduct spec —— 而不是来自规则网格"
                "(那是 GridSTS)。MoveToXY 失败的点**不会**采谱:谱采在错的地方,"
                "比缺一条谱更糟。每一个 .dat 都必须自证它属于它自己那个点"
                "(基名唯一 + mtime 晚于采集前打下的水位线);证不了的点被记成 "
                "unrated 且不给路径,**绝不**去继承上一个点的文件。"
                "expected_coord_epoch 是**必填**的,代次对不上要拒绝,"
                "**当前代次读不到时同样拒绝** —— 横向粗动之后,同样的 (x, y) "
                "指的是另一片表面。"
                "部分成功是正常的,缺口会写在 summary 里;"
                "一个点都没成功则算失败。"
            ),
            parameters=[
                ParameterSpec(
                    name="positions",
                    type="str",
                    description=(
                        "由位置对象组成的 JSON 数组,每一项都要有 x_m 与 y_m,"
                        "单位**米**(label 可选)。例: "
                        '[{"x_m": 1e-8, "y_m": -2e-8, "label": "A"}, '
                        '{"x_m": 3e-8, "y_m": 0}]。最多 64 个点。'
                    ),
                    required=True,
                ),
                ParameterSpec(
                    name="expected_coord_epoch",
                    type="int",
                    description=(
                        "这批位置是在哪一代坐标系下算出来的(取自它们来源的地图"
                        "标记 / 计划)。对不上、**或者**当前代次读不到,都拒绝"
                        "执行;那样的坐标会指向另一片表面。**不提供任何换算** —— "
                        "粗动步进是开环的,所以一个换算出来的坐标看上去和真坐标"
                        "一模一样,而它是编的。"
                    ),
                    required=True,
                    min_value=0,
                ),
                ParameterSpec(
                    name="condition",
                    type="str",
                    description=(
                        "采谱所依据的条件**组的名字** —— **不是数字**。稳定 bias "
                        "与 setpoint、扫描窗口、点数、调制幅度/频率,全都放在"
                        "那一个组里(core.sts_workflow)。挑一个组;**不要**自己编"
                        "这些值,也不要用别的方式把它们传进来 —— 没有别的方式。"
                        "名字不认识会被拒绝,**并附上已知组的清单**。组里样品相关"
                        "的字段还没标定的,同样拒绝,并**指名**到底缺哪几个:"
                        "稳定条件绝不编一个出厂值出来,因为编出来的那个跑起来完美"
                        "无缺,量的却是一个没有人声明过的假设。"
                    ),
                    required=False, default=DEFAULT_CONDITION,
                ),
                ParameterSpec(
                    name="settle_s",
                    type="float",
                    description=(
                        "**只**覆写所选条件组的整定时间,单位秒。它是一个时间"
                        "旋钮,不是一个物理设定值 —— 被测量本身不会因它而变。"
                        "省略就用组自己的值。其余一切都来自那个组。"
                    ),
                    unit="s", required=False, default=None,
                    min_value=0.0, max_value=60.0,
                ),
                ParameterSpec(
                    name="spectral_family",
                    type="str",
                    description=(
                        "原样转交给 AssessSpectrum:哪些子判据可以起闸。"
                        "留空 = unknown,而 unknown **不是** metallic。"
                    ),
                    required=False, default="",
                    allowed_values=["", "metallic", "gapped", "unknown"],
                ),
                ParameterSpec(
                    name="assess",
                    type="bool",
                    description=(
                        "对每一个采到的 .dat 跑一次 AssessSpectrum。"
                        "它只记账 —— 一个差的判定绝不会打断整批。"
                    ),
                    required=False, default=True,
                ),
                ParameterSpec(
                    name="position_tol_nm",
                    type="float",
                    description=(
                        "存下来的 .dat 头里的 xy 与下发的 xy 最多可以差多少;"
                        "差得超过这个值,这个点就被标成可疑。"
                    ),
                    unit="nm", required=False, default=2.0,
                    min_value=0.0, max_value=1000.0,
                ),
                ParameterSpec(
                    name="min_point_separation_nm",
                    type="float",
                    description=(
                        "某个点与**同一批**里更早的某个点靠得比这个值还近,就把它"
                        "丢掉。0 = 每个点都保留。"
                        "重复测只是浪费时间,不会损坏任何东西。"
                    ),
                    unit="nm", required=False, default=0.0,
                    min_value=0.0, max_value=10000.0,
                ),
                ParameterSpec(
                    name="stop_after_consecutive_discard",
                    type="int",
                    description=(
                        "连续这么多次判成 'discard' 就停掉整批。0 = 永不停。"
                        "'unrated' **不**计入 —— 判不了不等于谱不好。"
                        "它不会尝试去补救什么;整批就是停下来,"
                        "并把这件事说出来。"
                    ),
                    required=False, default=3,
                    min_value=0, max_value=64,
                ),
                ParameterSpec(
                    name="per_point_timeout_s",
                    type="float",
                    description=(
                        "某一个点花的时间比这个值还长,就在开始下一个点**之前**"
                        "停下来。它打断不了一个已经在跑的子技能,所以它是在每个点"
                        "采完**之后**才检查的。"
                    ),
                    unit="s", required=False, default=300.0,
                    min_value=1.0, max_value=7200.0,
                ),
                ParameterSpec(
                    name="run_tag",
                    type="str",
                    description=(
                        "逐点 .dat 文件基名的前缀(为了可追溯)。"
                        "留空 = 用一个时间戳,这样重跑一次产生的文件"
                        "也能和这一次的区分开。"
                    ),
                    required=False, default="",
                ),
            ],
            preconditions=["z_controller_on"],
            estimated_duration_s=600.0,
            composition_level=3,
            tags=["spectroscopy", "sts", "batch", "positions", "composite"],
        )

    # ------------------------------------------------------------------
    # 输入解析
    # ------------------------------------------------------------------

    def _parse_positions(self, params: dict) -> "tuple[list[dict], str]":
        """``(positions, 错误原文)``。任何一项不合格 ⇒ **整批拒绝**并说清是第几个。

        照 ``BatchRegionsScan._parse_regions``：一个 64 点的批次里混着一个坏坐标，
        跑到那里再失败已经浪费了半小时，而且那时候没人在看。
        """
        raw = params.get("positions")
        if raw is None or (isinstance(raw, str) and not raw.strip()):
            return [], "positions 是必填项(一个 JSON 数组，元素含 x_m / y_m，单位米)。"
        if isinstance(raw, (list, tuple)):
            items: Any = list(raw)
        else:
            try:
                items = json.loads(raw)
            except (ValueError, TypeError) as exc:
                return [], f"positions 不是合法 JSON: {exc}"
        if not isinstance(items, list):
            return [], "positions 必须是一个 JSON 数组(list)。"
        if not items:
            return [], "positions 是空数组。"
        if len(items) > _MAX_POINTS:
            return [], (f"点数过多({len(items)} > {_MAX_POINTS})。上限是 checkpoint "
                        f"载荷与无人值守爆炸半径的限制，不是物理限制 —— "
                        f"请拆成几批。")
        out: list[dict] = []
        for i, it in enumerate(items):
            if not isinstance(it, dict):
                return [], f"第 {i} 个点不是一个对象。"
            try:
                x = float(it["x_m"])
                y = float(it["y_m"])
            except (KeyError, TypeError, ValueError):
                return [], f"第 {i} 个点缺少数值型的 x_m / y_m(单位米)。"
            if not (math.isfinite(x) and math.isfinite(y)):
                return [], f"第 {i} 个点的坐标不是有限数。"
            if not (abs(x) <= _MAX_ABS_XY and abs(y) <= _MAX_ABS_XY):
                return [], (f"第 {i} 个点超出量程(|x|,|y| 必须 ≤ "
                            f"{_MAX_ABS_XY:.0e} m) —— 检查一下是米还是纳米。")
            out.append({"x_m": x, "y_m": y,
                        "label": str(it.get("label") or f"P{i + 1}")})
        return out, ""

    @staticmethod
    def _resolve_condition(params: dict) -> "tuple[Any, str]":
        """``(条件 spec, 错误原文)`` —— 组名 → 一整套数字(S4 STS 设计 D19)。

        这条流程**不再收显式数值**。理由不是洁癖:七八个数值参数摆在工具表里，
        等于在请调用方(经常是语言模型)填数，而它们每一个都会变成真实的硬件动作。
        「移除诱因，别在提示词里说服模型」在本仓已经记过四次。

        三种情况都是**拒绝整次运行**，而且各说各的下一步:

        * 组名认不出来 ⇒ 报出已知的组名单子(「不知道有哪些」是最没用的回答)；
        * 覆写越界 ⇒ 拒绝，**不夹紧也不丢弃**(夹紧会让调用方以为自己设的是 X)；
        * 组里的样品事实还没标定 ⇒ 拒绝并**点名缺哪几个**。这不是「还没做完」，
          是「这个问题现在没有答案」——编一组稳定条件出来，整批谱会正常落盘、
          正常被判据吃掉，而它们测的是一个没人给过的假设。
        """
        from mast.core.sts_workflow import resolve_condition

        overrides = {}
        if params.get("settle_s") is not None:
            overrides["settle_s"] = params["settle_s"]
        res = resolve_condition(params.get("condition"), overrides or None)
        if not res.usable:
            return None, (f"谱学条件组 {res.name!r} 用不了：{res.refusal_detail()}")
        return res.spec, ""

    #: 回读比对的相对容差。仪器把 5.000 mV 报成 5.0001 mV 是量化，不是没写进去；
    #: 报成 0.0 或 2 mV 才是。1% 远宽于任何 DAC 量化，远窄于任何真实的写失败。
    _LOCKIN_READBACK_RTOL = 0.01

    def _check_lockin_readback(self, executor: GraphExecutor, spec) -> str:
        """回读到的调制参数与条件表对不对得上。**矛盾返回一句话(停);其余 ``""``。**

        三种结果**互不折叠**——这正是这个方法存在的理由：

        * **对得上** ⇒ ``""``，照跑。
        * **读不到**(``GetLockInConfig`` 没跑起来，或某个字段是 ``None``) ⇒ ``""``
          + 一条 warning。「读不到」不是「不一致」，把它当矛盾会让一次读失败废掉
          整批已经排好的谱；但它也不是「对得上」，所以必须出声说这一轮没有这层保护。
        * **对不上** ⇒ 返回理由，整批**一帧都不采**。写进去的调制幅度不是要的那个，
          意味着整批 dI/dV 落在一个未知幅度上，而每一条谱看上去都完全正常。
          这是本仓「一个调制错信号的 lock-in 会产出完全干净、完全错误的 dI/dV」
          那条注释的同一个形状。
        """
        res = executor.sub_results.get("setup:lockin_readback")
        data = dict(getattr(res, "data", None) or {}) if res is not None else {}
        executor.set_partial("lockin_readback", data)
        if res is None or not getattr(res, "success", False) or not data:
            self._note_warning(executor,
                               "lock-in 配置回读没跑成 —— 这一轮没有「写进去的调制"
                               "真的是要的那个」这层保护。**读不到不等于不一致**。")
            return ""
        want = spec.lockin_config()
        # 键名以 GetLockInConfig 实际返回的为准(amplitude，不是 amplitude_v)——
        # 拼错的键会读成 None，而 None 走的是「读不到」那条分支，于是一次真正的
        # 不一致会被静静地降级成一条 warning。
        pairs = (("mod_on", "mod_on", None),
                 ("amplitude", "amplitude_v", "V"),
                 ("frequency_hz", "frequency_hz", "Hz"))
        unreadable: list[str] = []
        bad: list[str] = []
        for got_key, want_key, unit in pairs:
            got = data.get(got_key)
            if got is None:
                unreadable.append(got_key)
                continue
            target = want[want_key]
            if unit is None:
                if bool(got) != bool(target):
                    bad.append(f"{got_key}: 要 {bool(target)}，读回 {bool(got)}")
                continue
            if abs(float(got) - float(target)) > abs(float(target)) * self._LOCKIN_READBACK_RTOL:
                bad.append(f"{got_key}: 要 {float(target):g} {unit}，"
                           f"读回 {float(got):g} {unit}")
        if bad:
            return ("lock-in 回读与条件表对不上（" + "；".join(bad) + "）—— "
                    "一帧都没有采。整批 dI/dV 会落在一个未知的调制幅度上，"
                    "而每一条谱看上去都完全正常。先查 ConfigureLockIn 的写入"
                    "（它自己不做写后回读，修复项 是这个结构缺口的症状）。")
        if unreadable:
            self._note_warning(
                executor,
                f"lock-in 回读里这几个字段读不到：{unreadable} —— "
                f"这一轮它们没有对账。**读不到不等于对得上。**")
        return ""

    @staticmethod
    def _note_warning(executor: GraphExecutor, text: str) -> None:
        warns = list(executor.progress.partial_data.get("warnings") or ())
        warns.append(text)
        executor.set_partial("warnings", warns)

    # ------------------------------------------------------------------
    # 逐点记录
    # ------------------------------------------------------------------

    def _records(self) -> "dict[str, dict]":
        return self._executor.progress.partial_data.setdefault("point_records", {})

    def _record(self, i: int) -> dict:
        recs = self._records()
        rec = recs.get(str(i))
        if rec is None:
            positions = self._executor.progress.partial_data.get("positions") or []
            p = positions[i] if i < len(positions) else {}
            rec = {
                "index": i + 1,
                "label": p.get("label") or f"P{i + 1}",
                "x_m": p.get("x_m"),
                "y_m": p.get("y_m"),
                # success 必须**显式**写。``runtime._marker_subrecords`` 对缺省的
                # success 视为 True ⇒ 一个没跑成的点会以 done 落在地图上。
                "success": False,
                "status": "skipped",
                "error": None,
                "path": None,
                "verdict": None,
                "order": None,
                "position_check": None,
                "arrived": None,
            }
            recs[str(i)] = rec
        return rec

    def _save(self) -> None:
        """把逐点记录推回 partial_data(顺带更新 last_update_at)。"""
        self._executor.set_partial("point_records", self._records())

    @staticmethod
    def _finish(rec: dict, status: str, *, error: "str | None" = None) -> None:
        """收口一个点的状态。``success`` 由 status 派生，只有 ``ok`` 是成功。

        ``suspect`` 不是成功：那条谱确实采到了，但它坐在一个我们无法确认的位置上
        —— 与 ``GridSTS`` 2026-07-03 那次 review 同一个判断(此前错位置的谱被当成
        有效数据计入产量)。数据仍然在 ``path`` 里，想要的人拿得到。
        """
        rec["status"] = status
        rec["success"] = (status == "ok")
        if error is not None:
            rec["error"] = error
        elif status == "ok":
            rec["error"] = None

    # ------------------------------------------------------------------
    # 生成器 plan
    # ------------------------------------------------------------------

    def plan_dynamic(self, params: dict,
                     executor: GraphExecutor) -> Iterator[CompositeStep]:
        pd = executor.progress.partial_data
        positions: list[dict] = list(pd.get("positions") or [])
        if not positions:
            return

        assess = bool(params.get("assess", True))
        family = str(params.get("spectral_family") or "")
        tol_nm = float(params.get("position_tol_nm", 2.0) or 0.0)
        sep_nm = float(params.get("min_point_separation_nm", 0.0) or 0.0)
        stop_after = int(params.get("stop_after_consecutive_discard", 3) or 0)
        point_budget_s = float(params.get("per_point_timeout_s", 300.0) or 0.0)
        run_tag = str(pd.get("run_tag") or "sts")

        # ── 条件建立走正门(S4 STS 设计 D16)────────────────────────────
        #
        # 全部 optional=False：调用方要的是「在这个条件下取的谱」。条件没建立起来
        # 还照采，出来的是一批**看上去正常、其实条件未知**的数据 —— 而且没有任何
        # 下游判据会报警。
        spec = self._spec
        # BiasSettleChange 而**不是** SetBias：恒流反馈下偏压穿零会把针尖推进
        # 样品，而一份位置列表的稳定偏压完全可能与当前偏压异号(D20)。
        yield CompositeStep(
            step_id="setup:bias", skill_name="BiasSettleChange",
            params=spec.bias_settle_params(), optional=False,
            checkpoint_after=False, tags=("setup",))
        yield CompositeStep(
            step_id="setup:setpoint", skill_name="SetSetpoint",
            params=spec.setpoint_params(), optional=False,
            checkpoint_after=False, tags=("setup",))
        # 调制:数字来自条件表，走 ConfigureLockIn 的**显式**传参(D18) ——
        # ApplyLockInPreset 更安全(它自带写后回读)，但它给不了「逐条谱不同幅度」，
        # 而条件表整件事就是在不同调制幅度下各取一条。显式传参同时绕开 修复项
        # (省略的幅度/频率会被报成 0.0)，回读那一半由下一步自己补上。
        yield CompositeStep(
            step_id="setup:lockin", skill_name="ConfigureLockIn",
            params=spec.lockin_config(), optional=False,
            checkpoint_after=False, tags=("setup",))
        # 回读比对。``ConfigureLockIn`` **不做写后回读**(而 SetSetpoint/SetZCtrlGain
        # 都做)——没有这一步，「调制已配好」只是这条流程在把自己的请求复述给自己听。
        yield CompositeStep(
            step_id="setup:lockin_readback", skill_name="GetLockInConfig",
            params={}, optional=True, checkpoint_after=False, tags=("setup", "read"))
        lockin_problem = self._check_lockin_readback(executor, spec)
        yield CompositeStep(
            step_id="setup:sts", skill_name="ConfigureSTS",
            params=spec.sweep_config(), optional=False, checkpoint_after=True,
            tags=("setup",))
        mls = spec.mls_arrays()
        if mls is not None:
            # 段内结构继续由 Nanonis 的 MLS 负责，S4 不建第二套分段抽象(D18)。
            yield CompositeStep(
                step_id="setup:mls_mode", skill_name="SetSTSMLSMode",
                params={"mode": "MLS"}, optional=False, checkpoint_after=False,
                tags=("setup",))
            yield CompositeStep(
                step_id="setup:mls_vals", skill_name="SetSTSMLSVals",
                params=dict(mls), optional=False, checkpoint_after=False,
                tags=("setup",))

        # ── 逐点 ──────────────────────────────────────────────────────
        order = 0
        streak = 0            # 连续 discard 计数(D14)
        taken: list[tuple[float, float]] = []   # 已采点，用于批内去重
        # 回读**否定**了调制设置 ⇒ 一帧都不采。整批 dI/dV 会落在一个未知的调制
        # 幅度上，而每一条谱看上去都完全正常 —— 那是几小时之后才发现的一种错误。
        # ⚠️ 「读不到」不走这条路(它在 warnings 里)：读不到不是矛盾。
        stop_reason = lockin_problem
        if stop_reason:
            # 就地记账。循环里那处只在**跑到一半才停**时写，而这一次是**一点都没
            # 开跑** —— 不在这里写，结果会说「一切正常，只是零个点」。
            executor.set_partial("stopped_early", True)
            executor.set_partial("stopped_reason", stop_reason)

        for i, p in enumerate(positions):
            rec = self._record(i)
            if stop_reason:
                self._finish(rec, "skipped", error=f"早停后未执行：{stop_reason}")
                self._save()
                continue

            # 批内去重。「已测过」不等于「已损坏」—— 复测浪费时间，不伤表面，
            # 所以默认(0)是不去重；这只是一个省时间的开关。
            if sep_nm > 0.0:
                near = next(
                    (k for k, (tx, ty) in enumerate(taken)
                     if math.hypot(p["x_m"] - tx, p["y_m"] - ty) * 1e9 < sep_nm),
                    None)
                if near is not None:
                    self._finish(rec, "skipped",
                                 error=(f"与本批第 {near + 1} 个已采点相距不足 "
                                        f"{sep_nm:g} nm，按去重跳过"))
                    self._save()
                    continue

            order += 1
            rec["order"] = order
            self._save()
            t_point = time.monotonic()

            move_id = f"p{i:03d}:move"
            yield CompositeStep(
                step_id=move_id, skill_name="MoveToXY",
                params={"x_m": p["x_m"], "y_m": p["y_m"], "wait": True},
                optional=True, checkpoint_after=False,
                tags=("move", f"point={i}", f"label={p['label']}"))

            move_res = executor.sub_results.get(move_id)
            resumed_move = move_res is None and executor.is_completed(move_id)
            if move_res is None and not resumed_move:
                # **移动失败就不采**(D11)。静态 plan 做不到这一点：它只能照采一条
                # 位置是错的谱，然后事后标记。跳过既省一次采集，也不产生错数据。
                self._finish(rec, "move_failed",
                             error="移动到该点失败，未采谱(位置不对的谱不如没有)")
                self._save()
                continue

            arrived = None
            if move_res is not None:
                arrived = (getattr(move_res, "data", None) or {}).get("arrived")
                rec["arrived"] = arrived
            # ``arrived`` 是三态。``wait=False`` 路径返回 None，而把 None 当成 True
            # 就是又一次「读不到被当成答了」。这里只有**显式 True** 才算到位。
            arrived_ok = (arrived is True) or resumed_move

            # 采前记水位(D12 第一层)。必须在 AcquireSTS **之前**取，否则本点自己
            # 落的盘会把水位顶上去，这条判据就永远为假。
            watermark = _dat_mtime_watermark(executor.context)
            basename = f"{run_tag}_p{i:03d}"
            acq_id = f"p{i:03d}:acquire"
            yield CompositeStep(
                step_id=acq_id, skill_name="AcquireSTS",
                params={"save_basename": basename},
                optional=True, checkpoint_after=True,
                tags=("sts", f"point={i}", f"label={p['label']}"))

            acq_res = executor.sub_results.get(acq_id)
            if acq_res is None and not executor.is_completed(acq_id):
                self._finish(rec, "acquire_failed", error="采谱失败")
                self._save()
                taken.append((p["x_m"], p["y_m"]))
                continue
            taken.append((p["x_m"], p["y_m"]))

            acq_data = (getattr(acq_res, "data", None) or {}) if acq_res else {}
            path = acq_data.get("path")
            attributed, why = _check_attribution(path, basename, watermark)
            if not attributed:
                # 归属证不了 ⇒ 这一点**没有** .dat。绝不退而求其次去用那个路径：
                # 「最近 120 秒最新」在批量循环里指向的很可能是上一个点。
                rec["path"] = None
                rec["verdict"] = "unrated"
                rec["reason"] = "dat_attribution_failed"
                self._finish(rec, "acquire_failed",
                             error=f"谱文件归属校验未通过：{why}")
                self._save()
                continue

            rec["path"] = str(path)
            rec.pop("reason", None)

            # 位置核对(D11 第二条)：验证的是**动作**，不是「发过命令」。
            check = self._verify_position(str(path), p, tol_nm)
            rec["position_check"] = check
            status = "ok"
            err = None
            if not arrived_ok:
                status = "suspect"
                err = ("MoveToXY 报成功但没有确认到位(arrived 不是 True)，"
                       "这条谱的位置不可信")
            elif check == "mismatch":
                status = "suspect"
                err = (f".dat 头里的 xy 与命令 xy 相差超过 {tol_nm:g} nm，"
                       f"这条谱的位置不可信")
            self._finish(rec, status, error=err)
            self._save()

            if assess:
                assess_id = f"p{i:03d}:assess"
                yield CompositeStep(
                    step_id=assess_id, skill_name="AssessSpectrum",
                    params={"dat_path": str(path), "spectral_family": family},
                    optional=True, checkpoint_after=False,
                    tags=("assess", f"point={i}"))
                ares = executor.sub_results.get(assess_id)
                adata = (getattr(ares, "data", None) or {}) if ares else {}
                verdict = adata.get("verdict")
                if not verdict:
                    # 判据自己没跑成 ⇒ **判不了**，不是不合格。
                    verdict = "unrated"
                    rec["reason"] = "assess_failed"
                rec["verdict"] = verdict
                rec["gated_criteria"] = list(adata.get("gated_criteria") or [])
                rec["ungated_criteria"] = list(adata.get("ungated_criteria") or [])
                rec["assess_reasons"] = list(adata.get("reasons") or [])
                self._save()

                if verdict == "discard":
                    streak += 1
                elif verdict in ("keep", "keep_flagged"):
                    streak = 0
                # unrated：既不加也不清零 ——「读不到」不是「不合格」，也不是
                # 「好转了」。
                if stop_after > 0 and streak >= stop_after:
                    stop_reason = (f"连续 {streak} 条谱被判为 discard "
                                   f"(阈值 {stop_after})")

            if point_budget_s > 0.0 and not stop_reason:
                elapsed = time.monotonic() - t_point
                if elapsed > point_budget_s:
                    # 这是**预算**不是判决。它拦不住一个已经在跑的子技能(那需要
                    # 抢占，本层没有)，能做的只有「不开始下一个点」—— 所以判据
                    # 放在点**结束之后**，并如实说出它是事后检查。
                    stop_reason = (f"第 {order} 个点耗时 {elapsed:.0f} s，"
                                   f"超出每点预算 {point_budget_s:.0f} s")

            if stop_reason:
                executor.set_partial("stopped_early", True)
                executor.set_partial("stopped_reason", stop_reason)

    def _verify_position(self, path: str, p: dict, tol_nm: float) -> str:
        """.dat 头里的真实 xy 与命令 xy 对不对得上 —— 三态，见 POSITION_CHECKS。"""
        try:
            from mast.io.exp_map import extract_dat_position

            xy = extract_dat_position(path)
        except Exception as exc:  # noqa: BLE001 —— 读不到就是读不到
            logger.debug("SpectroscopyAtPositions: 读 .dat 位置失败: %s", exc)
            xy = None
        if not xy:
            return "unverified"
        d_nm = math.hypot(xy[0] - p["x_m"], xy[1] - p["y_m"]) * 1e9
        return "mismatch" if d_nm > tol_nm else "match"

    # ------------------------------------------------------------------
    # 汇总
    # ------------------------------------------------------------------

    def aggregate(self, sub_results: dict, progress: CompositeProgress) -> dict:
        pd = progress.partial_data
        positions = list(pd.get("positions") or [])
        recs: dict = pd.get("point_records") or {}
        points: list[dict] = []
        for i in range(len(positions)):
            rec = recs.get(str(i))
            if rec is None:
                p = positions[i]
                rec = {"index": i + 1, "label": p.get("label") or f"P{i + 1}",
                       "x_m": p.get("x_m"), "y_m": p.get("y_m"),
                       # 显式 False：缺省会被记录层当成 True。
                       "success": False, "status": "skipped",
                       "error": "未执行到这一点", "path": None,
                       "verdict": None, "order": None,
                       "position_check": None, "arrived": None}
            points.append(rec)

        def _count(pred) -> int:
            return sum(1 for r in points if pred(r))

        gated: list[str] = []
        ungated: list[str] = []
        for r in points:
            for name in r.get("gated_criteria") or ():
                if name not in gated:
                    gated.append(name)
            for name in r.get("ungated_criteria") or ():
                if name not in ungated:
                    ungated.append(name)

        return {
            # 键名**必须**是 points：``runtime._marker_subrecords`` 只认
            # ``regions`` / ``points`` 两个键，别的名字会静默退回单标记路径。
            "points": points,
            "point_count": len(points),
            "n_planned": len(points),
            "n_attempted": _count(lambda r: r.get("order") is not None),
            "success_count": _count(lambda r: r.get("success") is True),
            "n_keep": _count(lambda r: r.get("verdict") == "keep"),
            "n_flagged": _count(lambda r: r.get("verdict") == "keep_flagged"),
            "n_discard": _count(lambda r: r.get("verdict") == "discard"),
            "n_unrated": _count(lambda r: r.get("verdict") == "unrated"),
            # 「没跑判据」与「判不了」是两件事，不合并。只数**真的采到了谱**的点：
            # 一个移动失败的点也没有 verdict，但它缺的是数据，不是判据。
            "n_unassessed": _count(
                lambda r: r.get("verdict") is None and r.get("path")),
            "n_move_failed": _count(lambda r: r.get("status") == "move_failed"),
            "n_acquire_failed": _count(
                lambda r: r.get("status") == "acquire_failed"),
            "n_suspect": _count(lambda r: r.get("status") == "suspect"),
            "n_skipped": _count(lambda r: r.get("status") == "skipped"),
            "n_dat_attribution_failed": _count(
                lambda r: r.get("reason") == "dat_attribution_failed"),
            "n_position_unverified": _count(
                lambda r: r.get("position_check") == "unverified"),
            # 仅向下游传递成功采集且通过位置归属验证的路径。
            "dat_paths": [r["path"] for r in points
                          if r.get("success") is True and r.get("path")],
            "gated_criteria": gated,
            "ungated_criteria": ungated,
            "stopped_early": bool(pd.get("stopped_early", False)),
            "stopped_reason": str(pd.get("stopped_reason", "") or ""),
            "coord_epoch": pd.get("coord_epoch"),
            "run_tag": pd.get("run_tag"),
            # 条件:**组名与展开后的数字两个都记**。只记组名的话，流程表改版之后
            # 同一个名字指向另一组数，这批结果就没法回答「它们是在什么条件下取的」；
            # 只记数字的话，对不上是哪一组，也就没法复现(S4 STS 设计 D19)。
            "condition": pd.get("condition"),
            "condition_values": pd.get("condition_values") or {},
            "condition_summary": pd.get("condition_summary") or "",
            "lockin_readback": pd.get("lockin_readback") or {},
            "warnings": list(pd.get("warnings") or ()),
        }

    # ------------------------------------------------------------------
    # run_composite
    # ------------------------------------------------------------------

    def run_composite(self, context, params: dict) -> SkillResult:
        positions, err = self._parse_positions(params)
        if err:
            return self.fail(err)
        spec, err = self._resolve_condition(params)
        if err:
            return self.fail(err)
        self._spec = spec

        gate = self._epoch_gate(params)
        if gate is not None:
            return gate

        executor = GraphExecutor(
            composite_name=self._skill_name(),
            context=context,
            on_step_result=self.on_step_result,
            on_step_failed=self.on_step_failed,
        )
        executor.set_partial("positions", positions)
        executor.set_partial("condition", str(params.get("condition")
                                              or DEFAULT_CONDITION).strip())
        # 条件**展开成数字**存一份进结果:事后要能回答「这批谱是在什么条件下取的」，
        # 而组名会随流程表改版指向另一组数。组名与数字两个都记，缺一不可。
        executor.set_partial("condition_values", asdict(spec))
        executor.set_partial("condition_summary", spec.describe())
        executor.set_partial("coord_epoch", int(params["expected_coord_epoch"]))
        # run_tag 用 set_partial_default：续跑必须沿用**同一个** tag，否则恢复出来
        # 的点会用另一套 basename 去核对已经落盘的文件。
        executor.set_partial_default(
            "run_tag",
            _sanitize_tag(params.get("run_tag") or "")
            or time.strftime("sts%m%d_%H%M%S"))
        executor.set_partial_default("point_records", {})
        executor.set_partial_default("stopped_early", False)
        executor.set_partial_default("stopped_reason", "")
        self._executor = executor

        executor.run_plan(self.plan_dynamic(params, executor))
        data = self.aggregate(executor.sub_results, executor.progress)
        data["_progress"] = executor.progress.to_dict()
        data.update(abort_facts(executor.progress))

        if not executor.progress.aborted:
            # 收尾关调制(D16)。含 "sts"/"spectr" 的技能被判定为「用 lock-in」，
            # 所以没有任何人会替这条流程关 —— 责任在终端消费者。
            # **中止时不关**：中止不是流程结束，而且那一刻我们一个字都还没改完。
            #
            # ⚠️ 已知的张力，留着不动是有意的：本流程**自己**用
            # ``ApplyLockInPreset`` 打开过调制时，按「谁改谁还」它中止时也该关，
            # 而这里按「流程结束 = OFF」的策略判据(``aborted``)不关。三条理由让
            # 本轮选后者：①中止那一刻 ``safe_call`` 一律拒写，试也写不进去；
            # ②「还」的正确动作是**放回原值**，而 ``close_modulation`` 无条件写
            # OFF，那是策略不是归还；③判据只有一个才有单一真源。要改口径请连同
            # ``_preflight`` 的那段分界一起改，别在这里单独开一个例外。
            from mast.skills.composite._preflight import close_modulation

            data.update(close_modulation(context, skill_name=self._skill_name(),
                                         calls=self._all_calls))
            # 跑到头的运行绝不能给下一次同规模运行留 sidecar(2026-07-10 #95
            # 「第 5 个点扫描不到」)。
            executor.clear_sidecar()

        return self._finalize(executor, data)

    def _epoch_gate(self, params: dict) -> "SkillResult | None":
        """坐标代次闸门(S4 STS 设计 D13)。放行返回 None，否则返回拒绝结果。

        **只有确凿的 MATCH 放行。** :mod:`mast.core.coord_epoch` 的默认策略对
        「查不到当前代次」是放行加告警，那是为省略代次是常态的调用方定的；这一批
        坐标是**别人在别的时刻算出来的**，「查不到」不能当成「对得上」。加严在
        这里，模块本身一个字不改 —— 另一个调用方并不欠这份严格。
        """
        from mast.core import coord_epoch as ce

        verdict = ce.verify(params.get("expected_coord_epoch"),
                            what="这批取谱坐标")
        if verdict.verified:
            return None
        extra = ""
        if verdict.state in (ce.UNVERIFIABLE, ce.UNSTAMPED):
            extra = ("——本技能在这里比通用策略更严：一批**别处算出来的**坐标，"
                     "「查不到当前代次」不能当成「对得上」。请确认记录存储可用、"
                     "或按当前代次重新规划后再跑。")
        return self.fail(
            f"{verdict.message}{extra}",
            refusal_code=verdict.state,
            requested_coord_epoch=verdict.stamped,
            current_coord_epoch=verdict.current,
            points=[], point_count=0, dat_paths=[],
        )

    def _finalize(self, executor: GraphExecutor, data: dict) -> SkillResult:
        """成败判定 + 把缺口写进 summary(照 ``BatchRegionsScan`` 的既有语义)。"""
        if executor.progress.aborted:
            from mast.skills.composite.graph_executor import abort_error_text

            return self.fail(abort_error_text(executor.progress), **data)

        planned = int(data.get("n_planned", 0) or 0)
        ok_n = int(data.get("success_count", 0) or 0)
        if planned > 0 and ok_n == 0:
            return self.fail(
                f"0/{planned} 个点取到可用的谱 —— 每一个点都没成", **data)

        res = self.ok(**data)
        gaps: list[str] = []
        for key, word in (("n_move_failed", "移动失败"),
                          ("n_acquire_failed", "采集/归属失败"),
                          ("n_suspect", "位置存疑"),
                          ("n_skipped", "未执行")):
            n = int(data.get(key, 0) or 0)
            if n:
                gaps.append(f"{word} {n}")
        head = f"{ok_n}/{planned} 个点取谱成功"
        if gaps:
            # 失败数量及原因进入 summary，不能仅藏在 data 字段。
            res.summary = f"部分完成：{head}（{'、'.join(gaps)}；详见 points[].error）。"
        else:
            res.summary = f"{head}。"
        if data.get("stopped_early"):
            res.summary += (f" 早停：{data.get('stopped_reason')}"
                            f"——本技能不做任何补救(不修针、不换点)，"
                            f"剩下的点没有执行。")
        n_unver = int(data.get("n_position_unverified", 0) or 0)
        if n_unver:
            res.summary += (f" 另有 {n_unver} 个点的 .dat 头里读不到 xy，"
                            f"它们的位置**没有被核对过**(既不是对得上，"
                            f"也不是对不上)。")
        if ok_n and not data.get("gated_criteria"):
            res.summary += (" ⚠️ 这一批没有任何谱质量判据当闸(阈值全未标定)，"
                            "所以「成功」只等于「采到了」，不等于「合格」。")
        return res


# v2 tool export
def make_tool(context_provider):
    return wrap_skill(SpectroscopyAtPositions, context_provider)


__all__ = ["SpectroscopyAtPositions", "POINT_STATUSES", "POSITION_CHECKS"]
