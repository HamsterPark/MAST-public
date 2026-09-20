"""新仪器初始化 —— 「装到一台新机器上，还差哪些数」的那份清单。

设计文档：``docs/v2/design/new_instrument_initialization.md``

这个模块存在的理由
==================
MAST 已经**能存下**每台仪器的事实 —— ``instrument_profile`` 有 40 个字段、
``scan_policy`` 有一张档位表、``coarse_drive`` 有驱动电压声明、安全包络有覆写
通道。缺的从来不是存储，而是：

    **没有任何一个地方回答「这台机器还差哪些数」。**

这些字段散在 8 个存储里、界面上散在 3 个页面的 12 个折叠块里，默认值有三种完全
不同的性质（出厂猜测 / 保守起点 / 根本没有默认），而没有任何东西告诉用户哪些
是必须填的。出厂安全包络不能替代目标仪器的独立核对。

本模块是**目录**（catalog）：每一项是什么、属于哪个真源、什么级别、为什么只能人
来填、填错了物理上会怎样。

它**不存任何数值**
==================
一个数值都不存。值住在它们原本的真源里（``instrument_profile`` /
``safety_limits`` 覆写 / ``scan_policy`` / ``coarse_drive`` / SettingsStore），
本模块只做三件事：

1. 描述这些项（元数据）；
2. 拿调用方给的当前值，算出每一项的完成状态与总体是否「需要初始化」；
3. 提供硬件指纹 + 前放一致性核对这两个纯函数。

把 87 项的值再存一份会得到一张漂亮的、自包含的、**和真源不一致的**第二张表 ——
同一个教训在 ``rig_operating_parameters.md``「为什么扫图那套不进
instrument_profile」和 ``scan_policy`` 的 docstring 里各写过一次。

依赖：仅 stdlib（api 层与 skills 层都要读它）。
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from dataclasses import dataclass
from typing import Any, Iterable

logger = logging.getLogger(__name__)

#: 持久化键（``SettingsStore.KNOWN_KEYS``）。只装元数据，不装数值。
SETTINGS_KEY = "instrument_init"

#: 导出包的 schema 标识。导入时不匹配即拒绝 —— 一个来路不明的 json 被当成
#: 仪器档案灌进去，比没有导入功能糟得多。
BUNDLE_SCHEMA = "mast.instrument_init/1"

#: 严重级别。
REQUIRED = "required"
RECOMMENDED = "recommended"
OPTIONAL = "optional"

#: 前放输入端的电压量程（Nanonis 模拟输入 ±10 V）。用于「增益 ↔ 满量程」交叉
#: 核对：两个数是同一件事的两种说法，对不上就是有一个填错了。
ADC_INPUT_RANGE_V = 10.0

#: 交叉核对的容差。前放标称增益与实测满量程差 20% 以内视为一致（标称值本来就
#: 是圆整过的；差一个量级才是我们要抓的）。
PREAMP_CONSISTENCY_REL_TOL = 0.2


# ── 分组 ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class InitGroup:
    """一组初始化项。``intro`` 是**一句**导语，不是这一组的教科书。"""

    id: str
    title: str
    intro: str


# / #20 / #30（三轮）：「初始化页面的废话太多，删掉一些。代码注释中
# 保留即可。」每条 intro 原来是两三段物理背景，装机时要先读几屏才见到第一个输入框。
#
# 现在每组只留**一句**。被删掉的段落原样搬到本文件的注释里 —— 它们没有蒸发，
# 只是不再挡在填表的人和输入框之间。判断保留什么的规则只有一条：
# **单位 / 哪里找这个值 / 安全关键的方向性警示**，其余一律进注释。
GROUPS: tuple[InitGroup, ...] = (
    # 删掉的原文：「这一组决定 MAST 怎么把电压读数翻译成电流、以及正偏压到底在
    # 探测什么。它们错了不会有任何报错——图照样出、谱照样出，只是每一个数都错，
    # 或者整套能态归属是反的。」
    InitGroup(
        id="signal_chain",
        title="信号链（前置放大器 / 偏压极性）",
        intro="填错不会报错：图照样出，只是每个电流值都错，或者能态归属整个反了。",
    ),
    # 删掉的原文：「MAST 拦截越界命令用的那条线。出厂默认是占位符——与任何一台
    # 真实机器都不对应：它可能比你的压电小（好用的表面白白够不着），也可能比你的
    # 大（越界命令拦不住）。设定点上限尤其要紧：高于前放量程的设定点物理上达不到，
    # 反馈环会一路进 Z 直到撞针。这一组可以从仪器直接读一次来对账（下面的按钮）。
    # 对账只报告不一致，改不改由你定；自动化只允许收紧，绝不允许自动放宽。」
    # ⚠️ **2026-08-25 更正：上面那句「自动化只允许收紧」目前没有任何东西在执行。**
    #
    # 这一行原本写的是「『只允许收紧』那条纪律在 `apply_probe` 的实现与测试里，
    # 不靠这段话执行」—— 三项全是假的：
    #
    #   * **没有 `apply_probe` 这个函数**（全仓只有这句注释提过它）；
    #   * 真正写值的是 `api/routes/instrument_init.py::apply_instrument_init`，
    #     它**写什么就是什么**，校验转交 write_settings / write_override
    #     （schema + PIN），**没有任何方向检查**；
    #   * `tests/v2/unit/api/test_instrument_init_*.py` 里**没有**钉这条的用例。
    #
    # 它一直没出事，是因为**今天根本没有自动化** —— 这条路只有人点的初始化页
    # （POST /instrument-init/probe 读仪器 → /instrument-init/apply 写值），
    # 人在环里。而**人是需要放宽的**：出厂默认偏小时，够不着好表面、还会过早
    # 建议粗动换区。所以「只允许收紧」是给**自动化**定的规矩，不该套在这一页上。
    #
    # ⇒ 谁要做「自动识别包络并更新」的技能，**那道闸门要跟着新建，不在这里**。
    #   参考 `core/envelope_reconcile.py` 的立场（只产 findings、刻意不写）与
    #   `skills/builtins/instrument_limits.py` 开头那句：
    #   "An agent that can widen its own limits has, in the strict sense, no limits."
    InitGroup(
        id="envelope",
        title="安全包络（压电量程 / 设定点上限）",
        intro="出厂默认是占位符，用下面的按钮从仪器读一次对账。",
    ),
    # 参数组由代码直接读取、写入并回读核对，避免模型再次转述数值。
    InitGroup(
        id="approach",
        title="进针",
        intro="进针那套 Z 反馈参数，以及这台机器进针会不会碰到样品。",
    ),
    InitGroup(
        id="retract",
        title="退针（粗动马达）",
        intro="换样品 / 关机时怎么把针退开 —— **方向搞反就是把针往样品里送**。",
    ),
    # 删掉的原文：「『有些 Nanonis 控制器支持 400 V，但是有时候 300 V 就烧坏了』，
    # 而没有任何读数、状态位或报错告诉你在哪一台机器上。未声明 = 拒绝一切粗动
    # （这是刻意的：不知道叠堆能承受多少，是不动的理由，不是回退到别人机器上那个
    # 数的理由）。它在管理员 PIN 门后面——没设 PIN 就写不了。」
    # —— 完整版留在 max_amplitude_v 那一项的 consequence 里（安全关键，不删）。
    InitGroup(
        id="coarse",
        title="粗动与换区",
        intro="驱动电压是整页里唯一一个填错了硬件当场报废的数（写入需管理员 PIN）。",
    ),
    # 删掉的原文：「在中间真空区（约 0.1–1000 Pa，Paschen 极小值附近）给粗动压电
    # 加几百伏会打火，电弧爬过绝缘层就把叠堆废了——抽气和放气途中正好穿过这个区间。
    # 判据本身与真空计型号无关；型号相关的只有量程那两个数。特别注意：粗糙真空规
    # （Pirani / 电容薄膜规）下限只到 ~1 Pa，把它的下限填成冷阴极规的 5e-8 就是
    # fail-open——触底读数看起来『真空非常好』，实际可能是 0.5 Pa，正在放电区里。」
    # —— fail-open 那半段留在 vacuum_gauge_min_pa 的 consequence 里（不删）。
    InitGroup(
        id="vacuum",
        title="粗动真空互锁",
        intro="中间真空区（约 0.1–1000 Pa）给粗动加几百伏会打火，抽 / 放气途中正好穿过。",
    ),
    # 删掉的原文：「Z 压电量程是自动调平判据的分母；扫描档位表是全系统扫图默认
    # 参数的单一真源（`ScanAt` 每帧都从那里下发）。出厂档位是从文献协议数值化来的
    # 起点，不是你这台机器的真值。」
    InitGroup(
        id="scan",
        title="扫图",
        intro="出厂档位是从文献协议数值化来的起点，不是你这台机器的真值。",
    ),
    # 删掉的原文：「修针尖的碎屑实际扩散多远，是这根针、这个样品、这个温度的属性
    # ——只有看过的人知道。默认值是起点不是测量值。填小了，地图会把针尖领回一片
    # 已经毁掉的表面；填大了只是少扫几张图。」
    InitGroup(
        id="damage",
        title="破坏范围（避让半径）",
        intro="碎屑扩散多远随针 / 样品 / 温度变，默认值是起点不是测量值。",
    ),
    # 删掉的原文：「全部默认关闭是刻意的：为不存在的硬件保留技能比没有技能更糟——
    # agent 会看到它、调用它、拿到 Nanonis 报错、然后（如果它固执）再调一次；
    # 而工具表是模型每一轮都要读的菜单。」
    InitGroup(
        id="rig",
        title="连接与硬件清单",
        intro="这台机器实际装了哪些可选模块，**全部默认关闭**。",
    ),
)

_GROUP_IDS = frozenset(g.id for g in GROUPS)


# ── 单项 ─────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class InitItem:
    """一个「换机器就要重定」的量。

    ``store`` 指向**真源**，不是本模块自己的存储。``id`` = ``store.key``。
    """

    key: str
    store: str
    group: str
    severity: str
    label: str
    #: 这一项在页面上的**全部**说明 —— 一句话：填什么 + 哪里读。
    #:
    #: 这个字段是 ``what`` + ``where`` 合并来的。多轮删字之后字数总会长回来
    #: —— 因为结构还在：只要页面上有「这是什么 / 哪里找 / 填错会怎样 /
    #: 可对账」四个格子，就总有理由把每个格子填满，而「加起来太长了」不属于任何
    #: 一次改动。这一轮拆的是格子本身：**只剩一个格子，而且它只装得下一句话。**
    #:
    #: 两条内容纪律，都由 ``tests/v2/unit/core/test_instrument_init.py`` 钉住：
    #: 一句（一个句号）、``_ONE_LINE`` 字以内；不许重复 ``label`` 已经说过的话
    #: （从前 label「Z 压电总量程」配 what「Z 压电从头到尾能走多远」，同一句说两遍）。
    #: 单位不写进来 —— 它有自己的字段 ``unit``，页面挨着标签印。
    hint: str
    #: 填错 / 不填会怎样 —— **只有 ``safety_critical`` 的项有，而且必须有。**
    #:
    #: 这里写**方向性**警示（往哪边错、会撞什么），而且**不压成一行**：它们是
    #: 这一页存在的理由。其余项一个字都不在页面上，原文躺在各项上方的注释里。
    #:
    #: 「非安全项不许有」这一半和「安全项必须有」那一半同样重要。只写前一半的话，
    #: 下一轮删字会把持械字段的方向性警告一起压掉 —— 而那是事故路径。
    consequence: str = ""
    unit: str = ""
    #: 有出厂默认时为 True。没有默认的项，「未填」就是硬缺口；有默认的项，
    #: 「未填」需要用户**明确核对过**才算数（见 :func:`evaluate`）。
    has_factory: bool = True
    #: 存下来但**不算答案**的值（``bias_applied_to`` 的 "unknown" 就是刻意的
    #: 「未声明」占位，不是一个回答）。
    unanswered_values: tuple[str, ...] = ()
    #: 只有当另一项取到某些值时才需要问（``()`` = 永远需要）。
    depends_on: tuple[str, ...] = ()
    depends_values: tuple[str, ...] = ()
    #: 能从仪器读出来对账的项，写上读法（只用于界面提示）。
    probe: str = ""
    #: 填错会**伤到硬件**（撞针 / 毁针尖 / 烧压电 / 让安全判据反向）。
    #:
    #: 与 ``severity`` 是两件事，而它们的差别正是(内容冗长)的落点：
    #: 23 项 required 每一项都把「填错会怎样」那一整段直接摊在页面上，而其中
    #: 只有 9 项的后果是硬件损伤。其余的后果是**测量不准**——一样重要，但读它
    #: 不必抢在填表之前，behind「说明」就够了。
    #:
    #: 起这个名字而不是复用 severity，是因为这两个区分迟早会被当成冗余合并掉。
    #: 有测试钉住「这一族非空、且包含那几个已知会撞针的键」。
    safety_critical: bool = False

    @property
    def id(self) -> str:
        return f"{self.store}.{self.key}"


# ── 目录 ─────────────────────────────────────────────────────────────────────
# 每一项在页面上只剩**一句**（``hint``：填什么 + 哪里读），加上 ``label`` 与
# ``unit``。只有 ``safety_critical`` 的项多一条方向性警示，常显。
#
# 教学性、背景性、以及「我们为什么这么设计」的段落，全部在各项**上方的注释**里。
# 五轮反馈（「废话略多」/ 「说明基本上可以删去」/ 「废话太多，代码注释中
# 保留即可」/ #39 / 「这些废话还是没有清理？」）说的是同一件事：这一页是**填表**，
# 不是读物。
#
# 前三轮都是删字，字每次都长回来。第四轮改的是**结构**：从前页面上有四个格子
# （这是什么 / 哪里找 / 填错会怎样 / 可对账），于是总有理由把每个格子填满 ——
# 而「加起来太长了」不属于任何一次改动。现在只有一个格子，且它只装得下一句话。
#
# ⚠️ 这里**不写具体数字**（出厂值 / 实测值）。数字由调用方从真源现取，摆在项目
# 旁边显示。在这里再抄一份就会漂——这份清单存在的理由本身就是「同一个数不该有
# 两个来源」。
CATALOG: tuple[InitItem, ...] = (
    # ── ① 信号链 ────────────────────────────────────────────────────────────
    # 为什么只能人填：Nanonis 的 `Current_GainsGet` 只给得到一个**增益索引**和
    # 一串标签，索引→实际 V/A 倍数的对照表既不在 Nanonis 里，也不在 MAST 里 ——
    # 它印在前放机箱上、写在它的手册里。
    # 后果展开版：错一个量级，MAST 报出去的每一个电流值就整体错一个量级，而且
    # 不会有任何报错：你会看到一张标着「100 pA」的图，实际是 100 nA。设定点、
    # 饱和判据、I(z) 势垒、dI/dV 标定全部跟着错。
    # 后果（页面上已删，#39/#40）：错一个量级，MAST 报出去的每一个电流值都
    # 跟着错一个量级，且不报错。
    InitItem(
        key="preamp_gain_v_per_a", store="instrument_profile", group="signal_chain",
        severity=REQUIRED, unit="V/A", has_factory=False,
        label="前置放大器跨阻增益",
        hint="电流 = 读数电压 ÷ 此值；看前放铭牌 / 手册。",
        probe="GetMiscInstrumentConfig.current_gains（只给索引与标签，用于交叉核对）",
    ),
    # 为什么单独再问一遍满量程：Nanonis 侧同样读不到，而你**记得住的是这个数**
    # （「±10 nA 档」），不是 1e9 V/A；两个都填还能互相验一次
    # （见 ADC_INPUT_RANGE_V / PREAMP_CONSISTENCY_REL_TOL）。
    InitItem(
        key="preamp_full_scale_a", store="instrument_profile", group="signal_chain",
        severity=REQUIRED, safety_critical=True, unit="A", has_factory=False,
        label="前置放大器满量程电流",
        hint="前放铭牌 / 手册上「±10 nA」那个数。",
        consequence=(
            "**这一个数决定两条安全判据，而两条出厂默认都是错的方向。**\n\n"
            "① **设定点上限** `setpoint_max_a`（出厂 100 nA）—— 设定点高于前放量程时"
            "电流**恒小于**设定点，反馈环拿不到目标值，就会一路把 Z 推向样品直到撞针。"
            "在一台 ±10 nA 的机器上，出厂值留了一条完整的撞针路径。\n\n"
            "② **电流贴轨判据** `cm_sat_current_a`（出厂 90 nA）—— 前放在 10 nA 就"
            "贴轨了，读数停在 10 nA，**永远够不到 90**。这条 CRITICAL 告警一直是"
            "关着的，而它本来是「针已经扎进去了」最直接的证据之一。\n\n"
            "两条都不会报错，只会安静地什么都不做。"),
    ),
    # 为什么默认留在「未声明」而不是「样品」：虽然样品偏压是绝大多数机器的约定，
    # 但「按最常见约定默认下去」正是这类错误的来源。这是**接线**不是设置 ——
    # Nanonis 不知道线接在哪儿，任何读数里都没有这个信息。
    # 后果展开版：弄反了整套能态归属就是反的，而这种错在数据里完全看不出来：
    # 谱一样漂亮，结论一样自洽，只是把导带说成了价带。
    # 后果（页面上已删，#39/#40）：决定 dI/dV 谱里正偏压对应占据态还是空
    # 态；弄反了，数据上看不出来。
    InitItem(
        key="bias_applied_to", store="instrument_profile", group="signal_chain",
        severity=REQUIRED, unanswered_values=("unknown",),
        label="偏压加在样品还是针尖",
        hint="看接线；Nanonis 读数里没有这个信息。",
    ),
    # 后果（页面上已删，#39/#40）：缺了只影响「这份数据当时用的什么前放」这类事
    # 后追溯。
    InitItem(
        key="preamp_model", store="instrument_profile", group="signal_chain",
        severity=OPTIONAL, label="前置放大器型号",
        hint="铭牌上的字，只用于事后追溯。",
    ),

    # ── ② 安全包络 ──────────────────────────────────────────────────────────
    # 出厂默认是占位符，真值 = `Piezo_RangeGet` 全程 ÷ 2，而且**随温度变** ——
    # 压电系数在低温下显著变小，所以还要知道那份标定是在什么温度下做的。
    # 后果展开版：填小了，一部分好用的表面永远够不着，而且扫描地图会在还剩几倍
    # 面积可用时就建议粗动换区（而粗动会让全部既有坐标失效）。
    # 后果（页面上已删，#39/#40）：填小了够不着好表面、过早建议换区；填大了越界
    # 命令拦不住。
    InitItem(
        key="xy_max_m", store="safety_limits", group="envelope",
        severity=REQUIRED, unit="m",
        label="XY 压电半程",
        hint="`Piezo_RangeGet` 全程 ÷ 2，从中心算起。",
        probe="GetPiezoConfig.range（全程；除以 2 得半程）",
    ),
    InitItem(
        key="z_min_m", store="safety_limits", group="envelope",
        severity=REQUIRED, safety_critical=True, unit="m",
        label="Z 绝对位置下限",
        hint="朝样品那一侧的最负位置；`ZCtrl_LimitsGet`。",
        consequence=(
            "出厂默认 0.0 会把**隧穿侧那半个行程整个挡在门外**。它看起来像一条保守"
            "的安全设计，其实不是：从退针位命令到 0.0 本身就已经是几百 nm 的下扎。"
            "真正的撞针保护在 SafeTip、Z 控制器和进针逻辑里，从来不在这条包络上。"),
        probe="GetZControllerState.z_limits / GetPiezoConfig.range",
    ),
    # 后果（页面上已删，#39/#40）：填小了会让正常的退针位落在门外，退针动作被自
    # 己的安全门拒绝。
    InitItem(
        key="z_max_m", store="safety_limits", group="envelope",
        severity=REQUIRED, unit="m",
        label="Z 绝对位置上限",
        hint="退针方向的最正位置；`ZCtrl_LimitsGet`。",
        probe="GetZControllerState.z_limits",
    ),
    # 后果（页面上已删，#39/#40）：填大了，一次「抬针 1 µm」的谱就能把针推
    # 出量程或压进样品。
    InitItem(
        key="z_offset_max_m", store="safety_limits", group="envelope",
        severity=REQUIRED, unit="m",
        label="Z 相对位移上限",
        hint="STS 抬针这类**有符号位移**的上限，等于 Z 全程。",
    ),
    # 后果（页面上已删，#39/#40）：没有它，一次 width_m=2（两米）的扫
    # 描请求会被放行——v0.3.5 真的发生过。
    InitItem(
        key="scan_size_max_m", store="safety_limits", group="envelope",
        severity=REQUIRED, unit="m",
        label="扫描尺寸上限",
        hint="单帧边长上限，等于 XY 全程。",
    ),
    # 它**不是**一个偏好，是前放的物理量程 —— 应当等于上面填的「前放满量程」，
    # 页面会用那个数给出建议值。
    InitItem(
        key="setpoint_max_a", store="safety_limits", group="envelope",
        severity=REQUIRED, safety_critical=True, unit="A",
        label="电流设定点上限",
        hint="等于上面的「前放满量程」。",
        consequence=(
            "高于前放量程的设定点**物理上达不到**：反馈环拿不到目标电流，就会一路"
            "把 Z 推向样品，直到撞针。这是这一页里后果最直接的一条。"),
    ),
    # 后果（页面上已删，#39/#40）：填大了，一次写错小数点的偏压就直接送到结上。
    InitItem(
        key="bias_max_v", store="safety_limits", group="envelope",
        severity=RECOMMENDED, unit="V",
        label="偏压上限",
        hint="`Bias_RangeGet` 对账。",
        probe="GetMiscInstrumentConfig.bias_range",
    ),

    # ── ③ 进针 ──────────────────────────────────────────────────────────────
    # 这是这套进针机构的机械性质，没有任何读数能告诉 MAST。**只有看过进针后
    # 那块地方的人知道。**
    InitItem(
        key="approach_damages_surface", store="instrument_profile", group="approach",
        severity=REQUIRED, safety_critical=True, unanswered_values=("unknown",),
        label="这台机器进针会不会碰伤表面",
        hint="只有看过进针后那块地方的人知道。",
        consequence=(
            "填「不会」而实际会：地图不设避让区，之后的扫描会挑到扎痕上——"
            "浪费一张图，还可能把已经调好的针撞钝。填「会」而实际不会：只赔掉"
            "进针点周围一两百纳米的表面。所以「未知」按保守（＝会）处理。"),
    ),
    # 与上一项是两件事：上一项问的是会不会点伤**表面**（代价是一片表面），
    # 这一项问的是会不会毁掉**针尖**（代价是针，加之后几小时的修针）。
    # 取决于这套进针机构的机械设计与整定，只有守过几次进针的人知道。
    InitItem(
        key="approach_supervision", store="instrument_profile", group="approach",
        severity=REQUIRED, safety_critical=True, unanswered_values=("unknown",),
        label="进针要不要有人在场",
        hint="问的是会不会**扎针**（毁针尖，不是表面）。",
        consequence=(
            "它决定 **agent 敢不敢自己发起进针**。填「可无人值守」而这台机器其实"
            "会扎针，自主流程就会在没人看着的时候把针送进去。\n\n"
            "⚠ 老实说明这一项**目前只注入给模型，不是硬门**：HITL 审批门是在建图时"
            "从技能的 safety_level 派生的（`AutoApproach` 现在是 AUTO），运行时"
            "填的值改不了它。"),
    ),
    # 增益取决于仪器和针尖整定，没有普适值。面板的 SI 前缀必须正确解析；
    # 参数由代码直接写入，避免模型转述时丢失指数。
    InitItem(
        key="approach_p_gain_m", store="instrument_profile", group="approach",
        severity=REQUIRED, unit="m", has_factory=False,
        label="进针 Z 比例增益 (P)",
        hint="照抄面板的 Proportional：`3.000p` 就是 `3e-12`。",
    ),
    # 后果（页面上已删，#39/#40）：**扫图前要改回扫图那一档**，否则会得到一
    # 张带振铃的图（档位表填了就自动下发）。
    InitItem(
        key="approach_i_gain_m_per_s", store="instrument_profile", group="approach",
        severity=REQUIRED, unit="m/s", has_factory=False,
        label="进针 Z 积分增益 (I)",
        hint="照抄面板上的 Integral。",
    ),
    # 后果（页面上已删，#39/#40）：太大 = 建立隧道时电流冲得很猛；太小 = 
    # 判据在噪声里，进针停不下来。
    InitItem(
        key="approach_setpoint_a", store="instrument_profile", group="approach",
        severity=REQUIRED, unit="A", has_factory=False,
        label="进针电流设定点",
        hint="进针时的目标电流，越大进针越快。",
    ),
    # 后果（页面上已删，#39/#40）：缺了只是无法在自检里核对面板配置与预期是否一
    # 致。
    InitItem(
        key="approach_steps_per_cycle", store="instrument_profile", group="approach",
        severity=RECOMMENDED, unit="步", has_factory=False,
        label="Auto Approach 每轮步数",
        hint="面板上的 Number of Pulses，MAST 读不到。",
    ),
    # 后果（页面上已删，#39/#40）：缺了就判断不出「这次进针是不是异常」（走了三
    # 倍步数还没到通常是针掉了 / 空滑 / 方向反）。
    InitItem(
        key="approach_expected_steps", store="instrument_profile", group="approach",
        severity=RECOMMENDED, unit="步", has_factory=False,
        label="进针预计总步数",
        hint="从退针位到建立隧道的经验步数。",
    ),
    # 信号槽分配是这台机器的接线，每台都可能不同；哪一路解调是 dI/dV、哪一路是
    # 二阶（IETS），也只有配这台机器的人知道。
    # 后果（页面上已删，#39/#40）：缺了，**进针 dI/dV 测距整个不可用*
    # *（会明说，不会静默失败）。
    InitItem(
        key="lockin_signal_index", store="instrument_profile", group="approach",
        severity=RECOMMENDED, has_factory=False,
        label="dI/dV lock-in 信号索引",
        hint="在下拉列表中按通道名选择所需信号。",
        probe="ListSignalChannels（Signals_NamesGet）",
    ),
    # 解调器可提供 X/Y 或 R/Amplitude，具体取决于通道配置。读数本身
    # 看不出来 —— X 和 R 都只是一个电压数。
    # 后果展开版：R 是幅度，恒为正，随接近单调（指数）增大。X 是对参考相位的带符号
    # 投影：只有相位调到「信号全落在 X 上」时 |X| 才≈幅度；相位没调好、或在接近
    # 途中转动（结电容随距离变，相位真的会转），|X| 会在相位扫过 90° 时穿零 ——
    # 于是「越近越大」中途掉下去，而那是相位问题，不是针尖退开了。声明之后，
    # 进针播报会改用正确的符号并带上相位提醒。
    InitItem(
        key="lockin_readout_form", store="instrument_profile", group="approach",
        severity=RECOMMENDED, safety_critical=True, unanswered_values=("unknown",),
        label="lock-in 解调形式（上面那个信号是什么）",
        hint="看 Nanonis 里这一路解调怎么配的；读数看不出来。",
        consequence=(
            "**它决定「越近越大」这条判据成不成立。** X 是带符号投影，接近途中"
            "相位会转，|X| 扫过 90° 时穿零 —— 系统会把这次穿零播报成"
            "「离样品变远了」，而用户据此继续进针。"
        ),
    ),
    # 后果（页面上已删，#39/#40）：缺了，独立于电流的那条撞针判据没有输入。
    InitItem(
        key="qplus_amplitude_signal_index", store="instrument_profile", group="approach",
        severity=RECOMMENDED,
        label="qPlus 振幅信号索引",
        hint="下拉里按通道名选（−1 = 自动查找）。",
        probe="ListSignalChannels（Signals_NamesGet）",
    ),
    # 它是这一支音叉、这一次装配的值。**换针即失效** —— MAST 会在登记新针尖时
    # 自动清掉它（先归档进退役针尖那一行）。
    # 后果（页面上已删，#39/#40）：缺了判据没有分母，永远报不出撞针；拿上一根针
    # 的基线比，则要么恒报要么恒不报。
    InitItem(
        key="qplus_amplitude_baseline", store="instrument_profile", group="approach",
        severity=RECOMMENDED, has_factory=False,
        label="qPlus 自由振荡振幅基线",
        hint="针尖未接触时跑 `ReadTipOscillationAmplitude`。",
    ),

    # ── ④ 退针 ──────────────────────────────────────────────────────────────
    InitItem(
        key="retract_motor_dir", store="instrument_profile", group="retract",
        severity=REQUIRED, safety_critical=True,
        label="退针方向（粗动马达远离样品）",
        hint="装置接线约定，多数 Nanonis 是 Z+ = 远离。",
        consequence=(
            "搞反了，一次「退针 3000 步」就是往样品里送 3000 步。"
            "**运行时有兜底**：退针分级进行，第一级只走 1 步就回读 Z 压电走向，"
            "发现在逼近立刻停并撤针——所以填错最多赔 1 步。但请如实填。"),
    ),
    InitItem(
        key="z_extend_sign", store="instrument_profile", group="retract",
        severity=REQUIRED, safety_critical=True,
        label="压电伸长（趋向样品）对应 Z 读数符号",
        hint="Nanonis 内部约定 + 接线，两种都见过。",
        consequence=(
            "上面那条退针自检就是靠这个符号把「Z 在动」翻译成「在远离还是在靠近」。"
            "符号错了，兜底判据本身会读反。"),
    ),
    # 后果（页面上已删，#39/#40）：退不够就拔样品，样品架会刮到针。
    InitItem(
        key="retract_total_steps", store="instrument_profile", group="retract",
        severity=RECOMMENDED, unit="步",
        label="换样品退针总步数",
        hint="取决于样品架几何与粗动步长。",
    ),
    # 后果（页面上已删，#39/#40）：设太小会把噪声当成「在远离」；设太大会把正常
    # 退针误判成失败。
    InitItem(
        key="z_recede_min_nm", store="instrument_profile", group="retract",
        severity=RECOMMENDED, unit="nm",
        label="退针自检阈值（Z 伸长超过此值判定为远离）",
        hint="取决于本机的 Z 噪声与粗动步长。",
    ),
    # 不是固定等待 —— Z 一停就立刻读，这个数只决定「等到什么时候放弃」。
    # 反馈从退针位找到表面要多久是这台机器的性质（增益、温度、针尖都影响），
    # 没有哪个读数能替你回答。
    # 后果展开版：太短的话反馈还没走完就去读 Z，读到的是斜坡中段，量到的是
    # 「等了多久」而不是「表面在哪」。超时应报未得出结论并停下，不会
    # 假装有答案，所以填短了是**慢+拒绝**，不是误判。太长则马达真卡住、或者粗逼近
    # 高压没开的时候，每一级都要把这个预算等满才肯拒绝，换区变慢。
    # 后果（页面上已删，#39/#40）：太短会读到 Z 斜坡中段（会如实报「没得出结
    # 论」，不会误判）；太长则马达卡住时换区变慢。
    InitItem(
        key="z_settle_timeout_s", store="instrument_profile", group="retract",
        severity=RECOMMENDED, unit="s",
        label="退针自检等 Z 稳定的时间预算",
        hint="须在目标仪器上测量并设置，确保 Z 有足够时间稳定。",
    ),

    # ── ⑤ 粗动 ──────────────────────────────────────────────────────────────
    # 没有 XY 粗动的机器上，**当前压电范围内的表面就是「插拔样品之前你能看到的
    # 全部表面」** —— 修针尖的污染会不可逆地吃掉可用面积，所以选点策略必须省着来
    # （外圈→内圈）。
    # 后果（页面上已删，#39/#40）：填成「有」而实际没有，MAST 会用坏了就打
    # 算换区，而它根本换不了。
    InitItem(
        key="xy_coarse_motion", store="instrument_profile", group="coarse",
        severity=REQUIRED,
        label="这台机器有没有 XY 粗动",
        hint="机械配置。",
    ),
    InitItem(
        key="max_amplitude_v", store="coarse_drive", group="coarse",
        severity=REQUIRED, safety_critical=True, unit="V", has_factory=False,
        label="粗动驱动幅度上限（本机声明）",
        hint="叠堆能承受的电压，只有搭这台机器的人知道。",
        consequence=(
            "**这是整页里唯一一个填错了硬件当场报废的数**，而且软件救不回来。"
            "控制器的上限和**叠堆**的上限是两个数，只有后者要紧：「有些 Nanonis "
            "控制器支持 400 V，但是有时候 300 V 就烧坏了」。\n\n"
            "所以未声明时 MAST 拒绝一切粗动写入——不知道叠堆能承受多少，是不动的"
            "理由，不是回退到别人机器上那个数的理由。写入需要管理员 PIN。"),
        probe="GetMiscInstrumentConfig.motor_freq_amp（读回当前值，用于移动前核对）",
    ),
    # 后果（页面上已删，#39/#40）：声明的上限只约束 MAST **写**什么，
    # 管不了驱动**实际**是多少——所以每次粗动前都读回来比一次，读不到就拒绝。
    InitItem(
        key="expected_frequency_hz", store="coarse_drive", group="coarse",
        severity=RECOMMENDED, unit="Hz", has_factory=False,
        label="粗动驱动频率（移动前核对用）",
        hint="本机整定值；留空 = 不核对频率。",
    ),
    # 真值只能实机标：移一次、扫一张图、看有没有刮痕。取决于样品倾斜、台面跳动
    # 和针的长度。压电退针只有 1–2 µm 余量，而样品台侧滑时的垂直跳动、样品倾斜和
    # 针尖长度都远不止这个数。
    # 后果（页面上已删，#39/#40）：退得不够，横移的时候针就在样品上犁过去（压电
    # 退针只有 1–2 µm 余量）。
    InitItem(
        key="xy_prewithdraw_steps", store="instrument_profile", group="coarse",
        severity=RECOMMENDED, unit="步",
        depends_on=("instrument_profile.xy_coarse_motion",), depends_values=("yes",),
        label="横向粗动前的退针步数（清障）",
        hint="实机标：移一次、扫一张图、看有没有刮痕。",
    ),
    # 后果（页面上已删，#39/#40）：不够远的话「换了区」只是把旧区域挪进视野，扫
    # 出来还是同一片被污染过的表面。
    InitItem(
        key="xy_site_spacing_steps", store="instrument_profile", group="coarse",
        severity=RECOMMENDED, unit="步",
        depends_on=("instrument_profile.xy_coarse_motion",), depends_values=("yes",),
        label="粗动站点最小间距",
        hint="取决于本机粗动步长，只能实机标。",
    ),
    # 后果（页面上已删，#39/#40）：填太小，可能重叠的站点会被画成不重叠——好样
    # 品被判成用完，或在毁掉的区域上重新开工。
    InitItem(
        key="xy_step_uncertainty_frac", store="instrument_profile", group="coarse",
        severity=RECOMMENDED,
        depends_on=("instrument_profile.xy_coarse_motion",), depends_values=("yes",),
        label="粗动单步位移的相对不确定度",
        hint="随驱动幅度 / 温度 / 负载漂移，低温下更大。",
    ),
    # 后果（页面上已删，#39/#40）：**缺了只是少一句注释**——粗动标记上不写
    # 「大约走了这么远」，而不是编一个数出来。
    InitItem(
        key="xy_motor_step_m", store="instrument_profile", group="coarse",
        severity=OPTIONAL, unit="m", has_factory=False,
        depends_on=("instrument_profile.xy_coarse_motion",), depends_values=("yes",),
        label="XY 粗动单步位移标定",
        hint="只能实测，且会漂；**仅供估算**。",
    ),

    # ── ⑥ 真空互锁 ──────────────────────────────────────────────────────────
    # 判据本身与规的型号无关，**型号相关的只有量程这两个数**。
    InitItem(
        key="vacuum_gauge_min_pa", store="instrument_profile", group="vacuum",
        severity=REQUIRED, safety_critical=True, unit="Pa",
        label="真空计量程下限",
        hint="规的手册：冷阴极规约 5e-8 Pa，Pirani 只到 ~1 Pa。",
        consequence=(
            "**fail-open 的入口。** 规触底时发出的数看起来正好像「真空非常好」。"
            "填成 DL-7 的值而实际是 Pirani，触底只说明「低于 1 Pa」—— 那"
            "**包含 0.5 Pa，正在放电区里**，互锁形同虚设。"
        ),
    ),
    InitItem(
        key="vacuum_gauge_full_scale_pa", store="instrument_profile", group="vacuum",
        severity=REQUIRED, safety_critical=True, unit="Pa",
        label="真空计满量程",
        hint="规的手册。",
        consequence=(
            "超量程的读数不是数据是饱和。DL-7 的帧解析不校验指数位，超量程可能"
            "解出一个**看着合理的小数** —— 把它当成有效读数就会在高压下放行粗动。"
        ),
    ),
    # 后果（页面上已删，#39/#40）：设高了，抽 / 放气途中的粗动会在 Pasc
    # hen 极小值附近打火。
    InitItem(
        key="coarse_motion_max_pressure_pa", store="instrument_profile", group="vacuum",
        severity=RECOMMENDED, unit="Pa",
        label="允许粗动的压强上限",
        hint="用出厂默认即可（比放电区下沿低一个量级）。",
    ),
    # 选「关闭阻断」时仍然会计算并记录裁决 ——「我们选择不检查」要留得下痕迹，
    # 不能悄悄变成「没什么可检查的」。
    # 后果（页面上已删，#39/#40）：没接真空计的机器如果选「只认真空计」，粗动会
    # 一直被拒绝。
    InitItem(
        key="vacuum_interlock_mode", store="instrument_profile", group="vacuum",
        severity=RECOMMENDED,
        label="真空互锁模式",
        hint="拿不到可信压强读数时怎么办。",
    ),

    # ── ⑦ 扫图 ──────────────────────────────────────────────────────────────
    # 它是自动调平触发判据的**分母**：判据是「这一帧的斜坡吃掉多少 Z 量程」。
    # 填错了，同一个倾角在大图和小图上给出的紧迫程度就全错。
    # 后果（页面上已删，#39/#40）：它是自动调平判据的分母：填错了，要么该调平时
    # 不调，要么每一帧都嚷着要调平。
    InitItem(
        key="z_range_m", store="instrument_profile", group="scan",
        severity=REQUIRED, unit="m",
        label="Z 压电总量程",
        hint="`GetPiezoConfig.range` 的 Z 分量。",
        probe="GetPiezoConfig.range（Z 分量）",
    ),
    # 后果（页面上已删，#39/#40）：像素数 / 每线时间 / 帧宽单独看都合法，
    # **乘起来**才知道扫得多快——只有这条拦得住那个组合。
    InitItem(
        key="v_tip_max_m_s", store="instrument_profile", group="scan",
        severity=RECOMMENDED, unit="m/s",
        label="针尖横向扫描速度上限",
        hint="经验值：这根针、这个样品能扛多快。",
    ),
    # 后果（页面上已删，#39/#40）：转过头会吃掉 XY 行程，并让 Z 在帧角上
    # 打满。
    InitItem(
        key="tilt_limit_deg", store="instrument_profile", group="scan",
        severity=RECOMMENDED, unit="°",
        label="压电倾斜补偿上限（单轴）",
        hint="本机压电与样品安装的失配角。",
    ),
    # 出厂 4 档是从文献协议数值化来的**起点**；增益一档出厂根本不敢给数
    # （强依赖仪器与针尖态）。把扫图那套 P / T 填进来之后，就不需要在进针和扫图
    # 之间手动切增益了 ——`ScanAt` 每帧都从这里下发。
    # 后果（页面上已删，#39/#40）：不改也能跑（用出厂档），但每一帧都在用别人机
    # 器的速度。
    InitItem(
        key="tiers", store="scan_policy", group="scan",
        severity=RECOMMENDED,
        label="扫描参数档位表（扫图数值组）",
        hint="在「设置 → 扫描参数档位」里编辑。",
    ),

    # ── ⑧ 破坏范围 ──────────────────────────────────────────────────────────
    # 碎屑实际扎出去多远，是**这根针、这个样品、这个温度**的属性 —— 只有看过修针
    # 后那一块的人知道。默认值是起点，不是测量值。
    # 后果（页面上已删，#39/#40）：填小了，扫描地图会把针尖领回一片已经被自己毁
    # 掉的表面。
    InitItem(
        key="avoid_radius_tip_shape_nm", store="instrument_profile", group="damage",
        severity=RECOMMENDED, unit="nm",
        label="修针尖避让半径",
        hint="只有看过修针后那一块的人知道。",
    ),
    # 后果（页面上已删，#39/#40）：同上。
    InitItem(
        key="avoid_radius_pulse_nm", store="instrument_profile", group="damage",
        severity=RECOMMENDED, unit="nm",
        label="电脉冲避让半径",
        hint="同上，取决于脉冲幅度与样品。",
    ),
    # 后果（页面上已删，#39/#40）：同上。
    InitItem(
        key="avoid_radius_crash_nm", store="instrument_profile", group="damage",
        severity=RECOMMENDED, unit="nm",
        label="撞针避让半径",
        hint="同上。",
    ),
    # 后果（页面上已删，#39/#40）：同上。上面那一项填「不会」时这一项不起作用。
    InitItem(
        key="avoid_radius_approach_nm", store="instrument_profile", group="damage",
        severity=RECOMMENDED, unit="nm",
        depends_on=("instrument_profile.approach_damages_surface",),
        depends_values=("yes", "unknown"),
        label="进针扎痕避让半径",
        hint="同上；只在「进针会扎表面」时生效。",
    ),

    # ── ⑨ 连接与硬件清单 ────────────────────────────────────────────────────
    # 打开一个不存在的模块，agent 的工具表里就多出几十个**必然失败**的工具：
    # 它会看到、调用、拿到 Nanonis 报错、然后（如果固执）再调一次。而工具表是模型
    # 每一轮都要读的菜单 —— 为不存在的硬件付路由代价，会降低那些真的能用的工具的
    # 命中率。
    # 后果（页面上已删，#39/#40）：打开一个不存在的模块，agent 的工具表里
    # 就多出几十个必然失败的工具，挤掉真能用的那些。
    InitItem(
        key="modules", store="hardware_modules", group="rig",
        severity=REQUIRED,
        label="装了哪些可选硬件模块",
        hint="只有站在机器旁边的人知道机箱里插了什么。",
    ),
    InitItem(
        key="cm_sat_current_a", store="current_monitor", group="rig",
        severity=REQUIRED, safety_critical=True, unit="A",
        label="电流贴轨判据的饱和值",
        hint="等于前放满量程。",
        consequence=(
            "出厂 90 nA 是按 100 nA 量程的前放定的。**在一台 ±10 nA 的机器上，"
            "读数贴轨后就停在 10 nA，永远够不到 90** —— 这条 CRITICAL 告警等于"
            "一直关着，而贴轨恰恰是「针已经扎进去了」最直接的证据之一。"
            "它不会报错，只会安静地什么都不做。"),
    ),
)

_BY_ID: dict[str, InitItem] = {it.id: it for it in CATALOG}

#: 只用于 UI 分组渲染的顺序。
ITEM_IDS: tuple[str, ...] = tuple(it.id for it in CATALOG)


def item(item_id: str) -> "InitItem | None":
    return _BY_ID.get(item_id)


def items_for_group(group_id: str) -> tuple[InitItem, ...]:
    return tuple(it for it in CATALOG if it.group == group_id)


def catalog_payload() -> list[dict[str, Any]]:
    """目录的可序列化形式（给 API / 前端渲染）。**不含任何数值。**"""
    return [
        {
            "id": it.id,
            "key": it.key,
            "store": it.store,
            "group": it.group,
            "severity": it.severity,
            "label": it.label,
            "unit": it.unit,
            "hint": it.hint,
            "consequence": it.consequence,
            "has_factory": it.has_factory,
            "safety_critical": it.safety_critical,
            "unanswered_values": list(it.unanswered_values),
            "depends_on": list(it.depends_on),
            "depends_values": list(it.depends_values),
            "probe": it.probe,
        }
        for it in CATALOG
    ]


def groups_payload() -> list[dict[str, str]]:
    return [{"id": g.id, "title": g.title, "intro": g.intro} for g in GROUPS]


# ── 状态判定 ─────────────────────────────────────────────────────────────────
#
# 「未初始化」的判据是**内容判据**，不是文件判据 —— 这是刻意的。覆写目录与设置
# 文件都住在安装目录里（KNOWN_ISSUES §3.2），靠「安装包里恰好没有同名文件」活着，
# 所以「文件在不在」根本不能用来回答「这台机器配过没有」。
#
# 两层，方向不同：
#
#   * **必须弹**（权威）：有 required 项没有答案 —— 完全从**值本身**算，不看任何
#     标记。假阴性（该弹没弹）是危险的，所以这一层不接受任何「我确认过了」之外的
#     豁免。
#   * **不再骚扰**（仅抑制）：有出厂默认、且用户明确核对过的项算完成。假阳性
#     （多弹一次）只是烦，所以这一层可以依赖标记。
#
# 关键性质：核对记录与它描述的数据**存在同一个存储单元里**（``instrument_init``
# 与 ``instrument_profile`` 同在 ui_settings.json）。值活着记录就活着；值被抹了
# 记录也一起没了 —— 记录不可能比数据活得久。那才是「文件存不存在」真正的毛病。

STATUS_SET = "set"                  # 用户填了一个明确的值
STATUS_ACKNOWLEDGED = "acknowledged"  # 没填，但明确核对过「出厂值就是对的」
STATUS_DEFAULT = "default"          # 没填，有出厂默认，没核对过
STATUS_MISSING = "missing"          # 没填，且**没有**出厂默认 —— 硬缺口
STATUS_NOT_APPLICABLE = "n/a"       # 依赖条件不成立（如没有 XY 粗动的机器）

_COMPLETE_STATUSES = frozenset({STATUS_SET, STATUS_ACKNOWLEDGED, STATUS_NOT_APPLICABLE})


def _is_blank(value: Any, it: InitItem) -> bool:
    """True 表示「这个存储值不算一个答案」。

    ``None`` / 空串显然不算。``unanswered_values`` 里的枚举值也不算 ——
    ``bias_applied_to = "unknown"`` 是刻意的**未声明占位**，不是回答；把它当成
    「已回答」就正好回到了「按最常见约定默认下去」那条错误路径上。
    """
    if value is None:
        return True
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return True
        return s in it.unanswered_values
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    if isinstance(value, float) and not math.isfinite(value):
        return True
    return False


def _applicable(it: InitItem, values: dict[str, Any]) -> bool:
    """依赖条件是否成立。

    两条规则，方向都偏「显示出来」：

    * 依赖项**自己还没答**时按适用处理 —— 一个空白不该把另一项悄悄藏起来
      （否则一台什么都没填的机器上，清单会短得让人以为没什么要填的）；
    * 只有依赖项**明确答了**、且答案不在 ``depends_values`` 里，才判不适用。

    多个依赖时**全部**要求不适用才算不适用（不短路）—— 早期版本在第一个空白
    依赖上就 ``return True``，后面的依赖再也不会被看，而那种「只对了第一项」的
    行为在只有一个依赖时完全看不出来。
    """
    if not it.depends_on:
        return True
    for dep_id in it.depends_on:
        dep = _BY_ID.get(dep_id)
        raw = values.get(dep_id)
        if dep is not None and _is_blank(raw, dep):
            continue                      # 没答 → 这一条不构成「不适用」
        if it.depends_values and str(raw).strip() not in it.depends_values:
            return False
    return True


#: 选项值 → 中文。只覆盖三态问句那一族；不认识的值原样回显。
#:
#: 回显**原始值**而不是猜一个说法：这段话的用处就是让人能回到上面那一项去改，
#: 而一个和选项框里写的不一样的词会让人找不到该改哪个。
_CHOICE_LABEL: dict[str, str] = {"yes": "是", "no": "否", "unknown": "未声明"}


def na_reason(it: InitItem, values: dict[str, Any]) -> str:
    """这一项为什么不适用 —— 指名是**哪一项的哪个回答**让它失去意义。

    界面原来只说「这台机器不适用（上面某一项的回答让它失去
    意义）」。那句话把读者留在一道谜题前 —— 48 项里是哪一项？改了它这一项会
    不会回来？正确的措辞是：「直接写明上面已经选择了这台机器
    进针不扎」。

    依赖信息（``depends_on`` / ``depends_values``）本来就在目录里、也本来就随
    ``catalog_payload`` 发给了前端；缺的只是把它和**用户实际填的值**接起来
    的这一句话。
    """
    for dep_id in it.depends_on:
        dep = _BY_ID.get(dep_id)
        raw = values.get(dep_id)
        if dep is not None and _is_blank(raw, dep):
            continue
        if it.depends_values and str(raw).strip() not in it.depends_values:
            answer = str(raw).strip()
            shown = _CHOICE_LABEL.get(answer, answer or "（空）")
            label = dep.label if dep is not None else dep_id
            return f"上面「{label}」已答「{shown}」，所以这一项不适用。"
    return ""


@dataclass(frozen=True)
class ItemStatus:
    id: str
    status: str
    complete: bool
    severity: str
    value: Any = None
    #: 仅 ``n/a`` 时非空 —— 见 :func:`na_reason`。
    na_reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "status": self.status, "complete": self.complete,
            "severity": self.severity, "value": self.value,
            "na_reason": self.na_reason,
        }


def evaluate(
    values: dict[str, Any],
    acknowledged: "Iterable[str] | None" = None,
) -> list[ItemStatus]:
    """算出每一项的状态。

    ``values``：``item.id`` → **用户实际存下来的值**（不是「生效值」）。
    有出厂默认的项，未存过就传 ``None`` —— 传出厂默认进来会让「填过」和
    「用着默认」永远分不开，而这两者恰恰是本模块要区分的东西。

    ``acknowledged``：用户明确核对过的 ``item.id`` 集合。
    """
    ack = set(acknowledged or ())
    out: list[ItemStatus] = []
    for it in CATALOG:
        raw = values.get(it.id)
        if not _applicable(it, values):
            out.append(ItemStatus(it.id, STATUS_NOT_APPLICABLE, True, it.severity, raw,
                                  na_reason(it, values)))
            continue
        if not _is_blank(raw, it):
            out.append(ItemStatus(it.id, STATUS_SET, True, it.severity, raw))
            continue
        # 「核对过」只能对**有东西可核对**的项生效。
        #
        # 一个 required 且**没有出厂默认**的项(前放增益、进针 P/I/设定点、粗动
        # 驱动上限……)未填时,系统手里既没有值也没有兜底 —— 那不是「默认值恰好
        # 对」,那是空的。允许在这里签一笔字,等于让人**把一个不存在的数标成已确认**,
        # 而下游每一条依赖它的判据仍然是瞎的。
        #
        # 非 required 的无默认项可以签(比如没有 qPlus 音叉的机器上那条振幅基线):
        # 缺它只让某个功能明说自己不可用,不会让任何安全网静默失效。
        if it.id in ack and (it.has_factory or it.severity != REQUIRED):
            out.append(ItemStatus(it.id, STATUS_ACKNOWLEDGED, True, it.severity, None))
            continue
        status = STATUS_DEFAULT if it.has_factory else STATUS_MISSING
        # optional 项永远不阻塞：它们缺了只是少一句注释。
        complete = it.severity == OPTIONAL
        out.append(ItemStatus(it.id, status, complete, it.severity, None))
    return out


def summarise(statuses: "Iterable[ItemStatus]") -> dict[str, Any]:
    """按级别汇总 + 给出「要不要弹」。"""
    rows = list(statuses)
    counts: dict[str, dict[str, int]] = {
        sev: {"total": 0, "complete": 0} for sev in (REQUIRED, RECOMMENDED, OPTIONAL)
    }
    outstanding_required: list[str] = []
    for st in rows:
        bucket = counts.setdefault(st.severity, {"total": 0, "complete": 0})
        if st.status == STATUS_NOT_APPLICABLE:
            continue                      # 不适用的项不进分母
        bucket["total"] += 1
        if st.complete:
            bucket["complete"] += 1
        elif st.severity == REQUIRED:
            outstanding_required.append(st.id)
    return {
        "counts": counts,
        "outstanding_required": outstanding_required,
        "needs_setup": bool(outstanding_required),
    }


# ── 硬件指纹 ─────────────────────────────────────────────────────────────────
#
# 第三个触发器：**这还是同一台机器吗**。压电量程 / Z 限位 / 前放增益索引 /
# 偏压量程凑成一个摘要；哈希变了就意味着换了机器（或者有人重做了标定，
# 那同样值得重新看一遍这份清单）。
#
# 它是**补充**触发器：读不到就不算，永远不因为读不到而阻塞。

#: 参与指纹的字段。少而稳 —— 会随温度轻微漂的量（实测共振、噪声底）不进来，
#: 否则指纹每天都在变，很快就没人看了。
FINGERPRINT_FIELDS: tuple[str, ...] = (
    "piezo_range", "z_limits", "current_gain_index", "bias_range",
)

#: 指纹里数值的保留有效数字。压电量程的回包在最低位上会抖，取 4 位有效数字
#: 既能吸收抖动，又能分辨两台真正不同的机器。
_FINGERPRINT_SIG_DIGITS = 4


def _round_sig(value: Any, digits: int = _FINGERPRINT_SIG_DIGITS) -> Any:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v) or v == 0.0:
        return 0.0 if v == 0.0 else None
    return round(v, -int(math.floor(math.log10(abs(v)))) + (digits - 1))


def _canonical(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _canonical(value[k]) for k in sorted(value, key=str)}
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return _round_sig(value)
    if value is None:
        return None
    return str(value)


def rig_fingerprint(readings: Any) -> "str | None":
    """仪器指纹（16 位 hex），字段不足 2 个时返回 ``None``。

    ``readings``：``FINGERPRINT_FIELDS`` 的子集。缺字段是常态（没连上、模块没装），
    所以门槛设在 2 —— 一个字段的哈希太容易碰撞成「同一台机器」。
    """
    if not isinstance(readings, dict):
        return None
    payload = {
        k: _canonical(readings[k])
        for k in FINGERPRINT_FIELDS
        if k in readings and readings[k] is not None
    }
    if len(payload) < 2:
        return None
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


# ── 完成记录（唯一持久化的东西，且一个数值都不存） ───────────────────────────
_RECORD_TEXT_KEYS: tuple[str, ...] = ("rig_label", "completed_by", "app_version",
                                      "rig_fingerprint")
_RECORD_MAX_TEXT = 200


def sanitize_record(raw: Any) -> dict[str, Any]:
    """清洗完成记录。永不抛；非 dict → ``{}``。

    只保留：核对过的项、时间戳、指纹、几个自由文本。**任何数值字段都不收** ——
    这个键不是第二份仪器档案。
    """
    if not isinstance(raw, dict):
        return {}
    out: dict[str, Any] = {}
    ack = raw.get("acknowledged")
    if isinstance(ack, (list, tuple, set)):
        # 只留目录里认识的 id：一个已经被删掉的字段留在这里会让计数永远对不上。
        out["acknowledged"] = sorted({str(x) for x in ack} & set(ITEM_IDS))
    for key in ("completed_at", "dismissed_at"):
        try:
            v = float(raw[key])          # type: ignore[index]
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(v) and v > 0:
            out[key] = v
    for key in _RECORD_TEXT_KEYS:
        v = raw.get(key)
        if isinstance(v, str) and v.strip():
            out[key] = v.strip()[:_RECORD_MAX_TEXT]
    return out


def acknowledge(record: Any, item_ids: "Iterable[str]") -> dict[str, Any]:
    """把一批项标成「已核对」，返回新的记录（纯函数）。"""
    rec = sanitize_record(record)
    ack = set(rec.get("acknowledged") or ())
    ack |= {str(i) for i in item_ids} & set(ITEM_IDS)
    rec["acknowledged"] = sorted(ack)
    return rec


def stamp_completion(record: Any, *, fingerprint: "str | None" = None,
                     by: "str | None" = None,
                     app_version: "str | None" = None) -> dict[str, Any]:
    """盖一个「这台机器配过了」的时间戳（纯函数）。"""
    rec = sanitize_record(record)
    rec["completed_at"] = time.time()
    if fingerprint:
        rec["rig_fingerprint"] = str(fingerprint)[:_RECORD_MAX_TEXT]
    if by:
        rec["completed_by"] = str(by)[:_RECORD_MAX_TEXT]
    if app_version:
        rec["app_version"] = str(app_version)[:_RECORD_MAX_TEXT]
    return rec


def fingerprint_changed(record: Any, current: "str | None") -> bool:
    """指纹变了 = 大概率换了机器（或重做了标定）→ 重新弹。

    读不到当前指纹（没连上 Nanonis）**不算变** —— 一个因为没连硬件就每次都弹的
    横幅，一周之内就会被无视。
    """
    if not current:
        return False
    stored = sanitize_record(record).get("rig_fingerprint")
    return bool(stored) and str(stored) != str(current)


# ── 前放一致性核对 ───────────────────────────────────────────────────────────
def preamp_consistency(gain_v_per_a: Any, full_scale_a: Any,
                       adc_range_v: float = ADC_INPUT_RANGE_V) -> dict[str, Any]:
    """「增益」与「满量程」是同一件事的两种说法，对不上就是有一个填错了。

    返回 ``{"expected_full_scale_a", "ratio", "consistent", "note"}``；
    任一输入缺失 → ``consistent=None``（不知道，不是「一致」）。
    """
    try:
        gain = float(gain_v_per_a)
        full = float(full_scale_a)
    except (TypeError, ValueError):
        return {"expected_full_scale_a": None, "ratio": None,
                "consistent": None, "note": "两个数都填了才能互相核对。"}
    if not (math.isfinite(gain) and math.isfinite(full)) or gain <= 0 or full <= 0:
        return {"expected_full_scale_a": None, "ratio": None,
                "consistent": None, "note": "两个数都填了才能互相核对。"}
    expected = adc_range_v / gain
    ratio = full / expected
    consistent = abs(ratio - 1.0) <= PREAMP_CONSISTENCY_REL_TOL
    if consistent:
        note = (f"一致：{gain:.3g} V/A 对应满量程 ±{expected:.3g} A "
                f"(= ±{expected * 1e9:.4g} nA)。")
    else:
        decades = abs(math.log10(ratio)) if ratio > 0 else 0.0
        note = (
            f"⚠ 对不上：按增益 {gain:.3g} V/A 算，满量程应当是 ±{expected:.3g} A "
            f"(= ±{expected * 1e9:.4g} nA)，你填的是 ±{full:.3g} A "
            f"(= ±{full * 1e9:.4g} nA)，差 {ratio:.3g} 倍"
            + (f"（约 {decades:.1f} 个数量级）" if decades >= 0.5 else "")
            + "。请核对是哪一个填错了 —— 这两个数错任何一个，"
              "MAST 报出去的电流都会整体错同样的倍数。")
    return {"expected_full_scale_a": expected, "ratio": ratio,
            "consistent": consistent, "note": note}


def derived_from_preamp(full_scale_a: Any) -> dict[str, Any]:
    """由前放满量程导出的两条下游线（**建议值**，不自动写入）。

    为什么不自动写：`skills/builtins/instrument_limits.py` 开头那句
    "An agent that can widen its own limits has, in the strict sense, no limits."
    —— 对账只负责报告，改不改由人定。这里连 agent 都不涉及，但同一条纪律仍然
    适用：一个安全上限被程序悄悄改动过，就不再是用户声明过的那条线。
    """
    try:
        full = float(full_scale_a)
    except (TypeError, ValueError):
        return {}
    if not math.isfinite(full) or full <= 0:
        return {}
    return {
        "safety_limits.setpoint_max_a": {
            "value": full,
            "why": ("设定点上限应当等于前放量程：高于它的设定点物理上达不到，"
                    "反馈环会一路把 Z 推向样品直到撞针。"),
        },
        "current_monitor.cm_sat_current_a": {
            "value": full,
            "why": "超过前放满量程的读数不是测量值，是贴轨。",
        },
    }


# ── 导出 / 导入 ──────────────────────────────────────────────────────────────
#: 导出包里带的存储。**刻意不含学习量**（``instrument_profile`` 的
#: ``_CALIB_KEYS``）—— ``didv_at_contact_v`` 是那根针在那台机器上学出来的，
#: 搬到另一台机器上就是一个自信的错值。剥离由调用方做（它才拿得到 _CALIB_KEYS）。
BUNDLE_STORES: tuple[str, ...] = (
    "instrument_profile", "safety_limits", "scan_policy",
    "current_monitor", "coarse_drive",
)

#: 导入后**不自动生效**、必须重新签一次的存储。
#: ``coarse_drive`` 是「填错了叠堆就废」的那一个，且在 PIN 门后面：同型号 ≠ 同一台。
BUNDLE_NEVER_AUTO_APPLY: frozenset[str] = frozenset({"coarse_drive"})


def build_bundle(stores: dict[str, Any], *, app_version: str = "",
                 rig_label: str = "") -> dict[str, Any]:
    """打一个可带走的导出包（纯函数；调用方负责先剥掉学习量）。"""
    return {
        "schema": BUNDLE_SCHEMA,
        "exported_at": time.time(),
        "app_version": str(app_version or "")[:_RECORD_MAX_TEXT],
        "rig_label": str(rig_label or "")[:_RECORD_MAX_TEXT],
        "stores": {k: stores.get(k) for k in BUNDLE_STORES if stores.get(k)},
    }


class BundleRejected(ValueError):
    """导入包不可用。

    做成异常而不是「静默丢掉坏的部分」：半个导入包比没有导入更危险 —— 用户以为
    整台机器的配置都带过来了，实际只到了一半，而这个差异在界面上看不出来。
    与 ``scan_policy.PolicyRejected`` 同一类。
    """


def parse_bundle(raw: Any) -> dict[str, Any]:
    """校验并拆开一个导入包。结构不对就抛 :class:`BundleRejected`。

    **返回的是「待复核」的值，不是「已生效」的值** —— 导入是把数字填进表单，
    不是写进硬件。同型号不等于同一台：前放可能不一样，压电标定一定不一样。
    """
    if isinstance(raw, (str, bytes)):
        try:
            raw = json.loads(raw)
        except (ValueError, TypeError) as exc:
            raise BundleRejected(f"不是合法的 JSON：{exc}") from exc
    if not isinstance(raw, dict):
        raise BundleRejected("导入内容必须是一个 JSON 对象。")
    schema = str(raw.get("schema") or "")
    if schema != BUNDLE_SCHEMA:
        raise BundleRejected(
            f"不认识的格式标识 {schema!r}（期望 {BUNDLE_SCHEMA!r}）。"
            "这个文件不是 MAST 的仪器初始化导出包 —— 拒绝导入，"
            "而不是猜着把它灌进仪器档案。")
    stores = raw.get("stores")
    if not isinstance(stores, dict) or not stores:
        raise BundleRejected("导入包里没有任何配置内容。")
    kept = {k: v for k, v in stores.items() if k in BUNDLE_STORES and v}
    if not kept:
        raise BundleRejected(
            f"导入包里没有可识别的配置段（认识的有：{', '.join(BUNDLE_STORES)}）。")
    return {
        "schema": schema,
        "exported_at": raw.get("exported_at"),
        "app_version": str(raw.get("app_version") or "")[:_RECORD_MAX_TEXT],
        "rig_label": str(raw.get("rig_label") or "")[:_RECORD_MAX_TEXT],
        "stores": kept,
        "needs_resign": sorted(set(kept) & BUNDLE_NEVER_AUTO_APPLY),
    }


__all__ = [
    "SETTINGS_KEY", "BUNDLE_SCHEMA", "ADC_INPUT_RANGE_V",
    "REQUIRED", "RECOMMENDED", "OPTIONAL",
    "STATUS_SET", "STATUS_ACKNOWLEDGED", "STATUS_DEFAULT", "STATUS_MISSING",
    "STATUS_NOT_APPLICABLE",
    "InitItem", "InitGroup", "ItemStatus",
    "CATALOG", "GROUPS", "ITEM_IDS", "FINGERPRINT_FIELDS",
    "BUNDLE_STORES", "BUNDLE_NEVER_AUTO_APPLY", "BundleRejected",
    "item", "items_for_group", "catalog_payload", "groups_payload",
    "evaluate", "summarise", "rig_fingerprint", "fingerprint_changed",
    "sanitize_record", "acknowledge", "stamp_completion",
    "preamp_consistency", "derived_from_preamp",
    "build_bundle", "parse_bundle",
]
