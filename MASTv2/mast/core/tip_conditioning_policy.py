"""按针尖材料、制法与形态逐级回退的修针参数方案表。

参数默认值与安全包络上限是不同字段；超出包络的请求应拒绝，不静默夹紧。
未登记针尖使用通用方案，显式覆写优先于内建方案，来源痕迹随解析结果返回。
内建值是可配置的算法基线，部署前须在目标仪器与针尖条件下核验，
不代表本站或任何目标仪器的已完成标定。
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

#: 通配符:表里用它表示「这一维不区分」。
ANY = "*"


# ── 参数字段 ────────────────────────────────────────────────────────────────
#
# 字段名与各技能的 ParameterSpec 一一对应,resolver 按名字发给对应技能。

#: 一档方案里可能出现的参数,以及它属于哪个技能。
FIELD_OWNERS: dict[str, tuple[str, ...]] = {
    "pulse_v": ("TipPulse", "ConditionTip"),
    "pulse_duration_s": ("TipPulse",),
    "pulse_count": ("TipPulse",),
    "shaper_bias_v": ("TipShape",),
    "shaper_lift_v": ("TipShape",),
    "shaper_depth_m": ("TipShape",),
    "poke_shallow_depth_m": ("ShapeTipOnSurface",),
    "poke_deep_depth_m": ("ShapeTipOnSurface",),
    "poke_steps": ("ShapeTipOnSurface",),
    "target_quality": ("ConditionTip",),
    "max_attempts": ("ConditionTip",),
}

#: 安全包络字段(上限,超了拒绝)。
LIMIT_FIELDS: tuple[str, ...] = (
    "max_abs_pulse_v",     # 脉冲电压绝对值上限
    "max_poke_depth_m",    # 向表面下压的最大深度(正数,米)

    "max_pulse_count",     # 单次最多打几发
)



_DEFAULT: dict[str, Any] = {
    # 未登记针尖 / 认不出的组合落到这里。刻意保守:不知道针是什么的时候,
    # 宁可处理不够也不要一发把针打没。
    "pulse_v": 3.0,
    "pulse_duration_s": 0.1,
    "pulse_count": 1,

    "shaper_depth_m": -1.0e-9,
    "poke_shallow_depth_m": -5.0e-10,
    "poke_deep_depth_m": -2.0e-9,
    "poke_steps": 5,
    "target_quality": 0.3,
    "max_attempts": 5,
    "max_abs_pulse_v": 10.0,
    "max_poke_depth_m": 1.0e-8,
    "max_pulse_count": 5,
    "_note": "针尖未登记或组合未收录 —— 用保守通用参数。登记针尖可得到更合适的方案。",
}

#: (material, fabrication, form) → 参数覆盖(只写与上一级不同的字段)。
#:
#: 查表顺序见 :func:`resolve_policy`。写表时**只写差异**,便于看出「这一档特殊
#: 在哪」。
_POLICY: dict[tuple[str, str, str], dict[str, Any]] = {

    # ── 普通金属丝针尖 ────────────────────────────────────────────────
    (ANY, ANY, "stm_wire"): {
        "pulse_v": 4.0,
        "max_abs_pulse_v": 10.0,      # Nanonis bias 常见量程,也是 TipPulse 的 spec 上限
        "max_poke_depth_m": 1.0e-8,
        "_note": "金属丝针尖通用档。",
    },

    # 钨:硬、耐打,但暴露大气后有氧化层,初次进针常需要更强的处理才出好态。
    # 文献常见的钨针场成形/脉冲量级在几伏到十伏;这里取中段起步。
    ("W", ANY, "stm_wire"): {
        "pulse_v": 5.0,
        "shaper_bias_v": 4.0,
        "shaper_depth_m": -1.5e-9,
        "poke_deep_depth_m": -3.0e-9,
        "_note": "钨针较硬、耐处理;氧化层可能需要偏强的首轮处理。待真机标定。",
    },
    ("W", "etched", "stm_wire"): {
        # 腐蚀针顶端细,单次不宜过猛;宁可多来一轮。
        "pulse_v": 5.0,
        "max_attempts": 6,
        "_note": "钨电化学腐蚀针:顶端细,宁可多轮轻处理。待真机标定。",
    },
    ("W", "cut", "stm_wire"): {
        # 剪切钨针顶端不规则、常有多个微尖 —— 双针尖先验高,需要更多轮成形。
        "pulse_v": 5.5,
        "max_attempts": 8,
        "poke_deep_depth_m": -4.0e-9,
        "_note": "钨剪切针:多微尖先验高,通常需要更多轮成形。待真机标定。",
    },

    # 铂铱:不氧化、惰性好、适合谱学,但材质软 —— 重手法容易钝掉甚至粘针。
    ("PtIr", ANY, "stm_wire"): {
        "pulse_v": 3.0,
        "shaper_bias_v": 2.5,
        "shaper_depth_m": -8.0e-10,
        "poke_shallow_depth_m": -3.0e-10,
        "poke_deep_depth_m": -1.5e-9,
        "max_abs_pulse_v": 10.0,
        "max_poke_depth_m": 1.0e-8,
        "_note": "铂铱软:处理要比钨针轻,深压容易钝尖/粘针。待真机标定。",
    },
    ("PtIr", "cut", "stm_wire"): {
        "max_attempts": 7,
        "_note": "铂铱剪切针:软 + 多微尖,轻处理多轮。待真机标定。",
    },

    ("Pt", ANY, "stm_wire"): {
        "pulse_v": 3.0, "shaper_bias_v": 2.5,
        "max_abs_pulse_v": 10.0,
        "_note": "铂针软且惰性,处理宜轻。待真机标定。",
    },
    ("Ir", ANY, "stm_wire"): {
        "pulse_v": 4.0,
        "_note": "铱针硬度高于铂铱、惰性好。待真机标定。",
    },

    # 磁性针尖(自旋极化):磁构型是实验对象的一部分 —— 大力处理不只改几何,
    # 还会改磁性,把「修好了」变成「换了一根不同的针」。
    ("Fe", ANY, "stm_wire"): {
        "pulse_v": 2.5,
        "shaper_bias_v": 2.0,
        "shaper_depth_m": -5.0e-10,
        "poke_shallow_depth_m": -2.0e-10,
        "poke_deep_depth_m": -1.0e-9,
        "max_abs_pulse_v": 10.0,
        "max_poke_depth_m": 1.0e-8,
        "max_attempts": 4,
        "_note": ("磁性针尖:处理会改变磁构型,不只是几何形状。"
                  "自旋极化实验中「修针」可能使之前的磁对比不可比。待真机标定。"),
    },
    ("Ni", ANY, "stm_wire"): {
        "pulse_v": 2.5, "max_abs_pulse_v": 10.0,
        "_note": "镍针为磁性针尖,处理会改变磁构型。待真机标定。",
    },
    ("Co", ANY, "stm_wire"): {
        "pulse_v": 2.5, "max_abs_pulse_v": 10.0,
        "_note": "钴针为磁性针尖,处理会改变磁构型。待真机标定。",
    },
    ("Cr", ANY, "stm_wire"): {
        "pulse_v": 3.0, "max_abs_pulse_v": 10.0,
        "_note": "铬针为反铁磁针尖(杂散场小),处理会改变磁构型。待真机标定。",
    },

    # 超导针尖:回温、污染、强脉冲都可能破坏针尖超导能隙 —— 而那正是测量对象。
    ("Nb", ANY, "stm_wire"): {
        "pulse_v": 2.0,
        "max_abs_pulse_v": 10.0,
        "max_poke_depth_m": 1.0e-8,
        "max_attempts": 3,
        "_note": ("超导针尖:强处理会破坏针尖能隙,而能隙正是测量对象。"
                  "优先换针而不是反复修。待真机标定。"),
    },

    # ── qPlus 传感器 ──────────────────────────────────────────────────
    #
    # 石英音叉被戳坏**不可逆**,要拆机重装(往往还要重新粘针、重新标定 f₀/Q)。
    # 所以这一档的包络最硬,而且戳表面类技能另有一道软门(见 tip_state 注入块)。
    (ANY, ANY, "qplus"): {
        "pulse_v": 2.0,
        "pulse_duration_s": 0.05,
        "pulse_count": 1,

        "shaper_depth_m": -2.0e-10,
        "poke_shallow_depth_m": -1.0e-10,
        "poke_deep_depth_m": -3.0e-10,
        "poke_steps": 3,
        "max_attempts": 3,

        "max_abs_pulse_v": 10.0,

        "max_poke_depth_m": 1.0e-8,
        "max_pulse_count": 2,
        "_note": ("qPlus 传感器:石英音叉损坏不可逆。操作受当前针尖安全包络约束；"
                  "超过已配置上限时拒绝执行，不夹紧。参数需在目标仪器上核验。"),
    },
    ("PtIr", ANY, "qplus"): {
        "pulse_v": 1.5,

        "_note": "qPlus + 铂铱针:软材质 + 易损传感器,最轻的一档。待真机标定。",
    },
    ("W", ANY, "qplus"): {
        "pulse_v": 2.0,
        "_note": "qPlus + 钨针:材质耐打但传感器不耐打,按传感器的包络来。待真机标定。",
    },
}


def _lookup_chain(material: str, fabrication: str, form: str) -> list[dict[str, Any]]:
    """从粗到细的命中序列(后者覆盖前者)。"""
    mat = (material or "").strip() or ANY
    fab = (fabrication or "").strip() or ANY
    frm = (form or "").strip() or "stm_wire"
    keys = [
        (ANY, ANY, frm),        # 形态通用
        (mat, ANY, frm),        # 材料 × 形态
        (mat, fab, frm),        # 精确组合
    ]
    return [_POLICY[k] for k in keys if k in _POLICY]


def resolve_policy(
    facts: "dict[str, Any] | None",
    *,
    overrides: "dict[str, Any] | None" = None,
) -> dict[str, Any]:
    """当前针尖对应的一整档方案(参数 + 安全包络 + 说明 + 来源痕迹)。

    *facts* 是 :func:`mast.core.tip_state.current_tip_facts` 的返回(None = 针尖
    未登记 → 通用保守档)。*overrides* 是用户在设置里填的覆盖,优先级最高。

    返回值里的 ``_sources`` 记录每个字段是哪一级给的 —— 与 scan_resolver 的
    trace 同款:一个数字从哪来,事后必须查得到。
    """
    out: dict[str, Any] = dict(_DEFAULT)
    sources: dict[str, str] = {k: "factory_default" for k in out if not k.startswith("_")}
    notes: list[str] = []
    if _DEFAULT.get("_note") and not facts:
        notes.append(str(_DEFAULT["_note"]))

    if facts:
        for layer in _lookup_chain(
                str(facts.get("material") or ""),
                str(facts.get("fabrication") or ""),
                str(facts.get("form") or "stm_wire")):
            for key, val in layer.items():
                if key == "_note":
                    notes.append(str(val))
                    continue
                out[key] = val
                sources[key] = "policy_table"

    if overrides:
        for key, val in overrides.items():
            if key.startswith("_") or key not in out:
                continue
            if val is None:
                continue
            try:
                out[key] = type(out[key])(val) if isinstance(out[key], (int, float)) else val
            except (TypeError, ValueError):
                logger.debug("修针方案覆写 %s=%r 类型不符,忽略", key, val)
                continue
            sources[key] = "operator_override"

    out["_sources"] = sources
    out["_notes"] = notes
    return out


def policy_summary(facts: "dict[str, Any] | None") -> str:
    """一行人读的方案说明(注入块/技能结果里用)。"""
    pol = resolve_policy(facts)
    notes = pol.get("_notes") or []
    return " ".join(str(n) for n in notes) if notes else ""


__all__ = [
    "ANY", "FIELD_OWNERS", "LIMIT_FIELDS",
    "resolve_policy", "policy_summary",
]
