"""Tip state — 当前装在仪器里的针尖是哪一根，以及它意味着什么。

Data + render layer(stdlib only,无 langchain / 无 agent 依赖),与
:mod:`mast.core.instrument_profile` 同样的分层理由:
  * skill 层直接 import 本模块读针尖属性(skills→core 是既有方向);
  * 注入中间件住在 ``mast.agents._shared.tip_context_mw`` 并 import 本模块;
  * 持久化与换针事务住在 ``mast.logging.tip_registry``(它需要 storage)。

**为什么针尖要单独成一层**(2026-07-31 用户需求):

  针尖是仪器级的耗材 —— 换实验、换样品都未必换针尖,但换了针尖,一批"学出来
  的"量立刻失效(dI/dV 接触标定、qPlus 自由振幅基线),而且修针方案本身就依赖
  针尖是什么:钨腐蚀针、铂铱剪切针、qPlus 传感器,能承受的处理完全不同。

  在此之前系统里**没有针尖实体这个概念**:vision 的 tip_quality / tip_change
  是"当前这根针的瞬时状态",tip_crash_tracker 是位置状态,
  ``instrument_profile`` 里只有两个 qPlus 振幅键。"针尖是钨还是铂铱、腐蚀还是
  剪切"这一维在系统里完全不存在,于是修针技能的参数只能硬编码成一个常数
  (``tip_shaper`` 的 bias 3.0 V 对钨针和铂铱针是同一个数)。

**术语纪律**:本仓「换针 / tip change」已被 vision 占用(mid-scan 针尖态突变
检测,见 ``mast/vision/tip_change.py``)。物理更换针尖一律用 register / install
/ 装入 / 登记,面向模型的文本同此。
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)


# ── 受控词表 ─────────────────────────────────────────────────────────────────
#
# 存库时是软约束(TEXT 列,不加 SQL CHECK) —— 与 samples.sample_type 同款:归一
# 在工具层做,收紧词表时不必迁移老库。

#: 针尖材料。``other`` + ``material_detail`` 兜住合金/涂层等长尾。
TIP_MATERIALS: tuple[str, ...] = (
    "W", "PtIr", "Pt", "Ir", "Fe", "Ni", "Co", "Cr", "Au", "Ag", "Nb", "other")

#: 制备方式(要求的四种 + unknown)。
TIP_FABRICATIONS: tuple[str, ...] = ("etched", "cut", "ground", "fib", "unknown")

#: 针尖形态。qPlus 与普通金属丝针尖在"能承受什么"上差别最大。
TIP_FORMS: tuple[str, ...] = ("stm_wire", "qplus")

MATERIAL_LABELS: dict[str, str] = {
    "W": "钨 (W)", "PtIr": "铂铱 (Pt/Ir)", "Pt": "铂 (Pt)", "Ir": "铱 (Ir)",
    "Fe": "铁 (Fe)", "Ni": "镍 (Ni)", "Co": "钴 (Co)", "Cr": "铬 (Cr)",
    "Au": "金 (Au)", "Ag": "银 (Ag)", "Nb": "铌 (Nb)", "other": "其它",
}
FABRICATION_LABELS: dict[str, str] = {
    "etched": "电化学腐蚀", "cut": "钳子剪切", "ground": "机械打磨",
    "fib": "FIB 切割", "unknown": "制备方式未记录",
}
FORM_LABELS: dict[str, str] = {
    "stm_wire": "普通 STM 金属丝针尖", "qplus": "qPlus 型针尖(石英音叉)",
}

#: 归一映射:用户/模型可能写的各种写法 → 词表值。键一律小写无空格无连字符。
_MATERIAL_ALIASES: dict[str, str] = {
    "w": "W", "tungsten": "W", "钨": "W", "钨丝": "W",
    "ptir": "PtIr", "pt/ir": "PtIr", "ptir alloy": "PtIr",
    "platinumiridium": "PtIr", "platinum-iridium": "PtIr",
    "pt80ir20": "PtIr", "pt90ir10": "PtIr", "ptir90/10": "PtIr",
    "铂铱": "PtIr", "铂铱合金": "PtIr", "铂/铱": "PtIr",
    "pt": "Pt", "platinum": "Pt", "铂": "Pt", "白金": "Pt",
    "ir": "Ir", "iridium": "Ir", "铱": "Ir",
    "fe": "Fe", "iron": "Fe", "铁": "Fe",
    "ni": "Ni", "nickel": "Ni", "镍": "Ni",
    "co": "Co", "cobalt": "Co", "钴": "Co",
    "cr": "Cr", "chromium": "Cr", "铬": "Cr",
    "au": "Au", "gold": "Au", "金": "Au",
    "ag": "Ag", "silver": "Ag", "银": "Ag",
    "nb": "Nb", "niobium": "Nb", "铌": "Nb",
}
_FABRICATION_ALIASES: dict[str, str] = {
    "etched": "etched", "etch": "etched", "electrochemical": "etched",
    "electrochemicaletching": "etched", "electrochemicallyetched": "etched",
    "电化学腐蚀": "etched", "电化学": "etched", "腐蚀": "etched", "电解腐蚀": "etched",
    "cut": "cut", "clipped": "cut", "snipped": "cut", "mechanicalcut": "cut",
    "钳子剪": "cut", "钳剪": "cut", "剪切": "cut", "剪的": "cut", "剪": "cut",
    "ground": "ground", "grinding": "ground", "polished": "ground",
    "mechanicalgrinding": "ground",
    "打磨": "ground", "机械打磨": "ground", "研磨": "ground", "抛光": "ground",
    "fib": "fib", "focusedionbeam": "fib", "fibmilled": "fib", "fibcut": "fib",
    "聚焦离子束": "fib", "离子束切割": "fib",
}
_FORM_ALIASES: dict[str, str] = {
    "qplus": "qplus", "q-plus": "qplus", "qplussensor": "qplus",
    "tuningfork": "qplus", "音叉": "qplus", "石英音叉": "qplus", "qplus针尖": "qplus",
    "stmwire": "stm_wire", "stm": "stm_wire", "wire": "stm_wire",
    "metalwire": "stm_wire", "normal": "stm_wire", "plain": "stm_wire",
    "普通": "stm_wire", "普通stm针尖": "stm_wire", "金属丝": "stm_wire",
    "常规": "stm_wire",
}


def _key(text: Any) -> str:
    """归一比较用的键:小写、去空白/连字符/下划线。"""
    return "".join(str(text or "").split()).replace("-", "").replace("_", "").lower()


def _lookup(text: Any, aliases: dict[str, str], vocab: tuple[str, ...]) -> "str | None":
    """词表归一。命中返回词表值,认不出返回 None(**绝不猜**)。"""
    raw = str(text or "").strip()
    if not raw:
        return None
    for v in vocab:                       # 已经是词表值(大小写不敏感)
        if raw.casefold() == v.casefold():
            return v
    return aliases.get(_key(raw))


def normalize_material(text: Any) -> "str | None":
    """「钨」「tungsten」「Pt80Ir20」→ 词表值;认不出返回 None。"""
    return _lookup(text, _MATERIAL_ALIASES, TIP_MATERIALS)


def normalize_fabrication(text: Any) -> "str | None":
    """「电化学腐蚀」「clipped」「FIB」→ 词表值;认不出返回 None。"""
    return _lookup(text, _FABRICATION_ALIASES, TIP_FABRICATIONS)


def normalize_form(text: Any) -> "str | None":
    """「qPlus」「音叉」「普通」→ 词表值;认不出返回 None。"""
    return _lookup(text, _FORM_ALIASES, TIP_FORMS)


def material_candidates() -> list[str]:
    """给模型的可选材料清单(词表 miss 时回给它挑)。"""
    return [f"{m}（{MATERIAL_LABELS.get(m, m)}）" for m in TIP_MATERIALS]


def fabrication_candidates() -> list[str]:
    return [f"{f}（{FABRICATION_LABELS.get(f, f)}）" for f in TIP_FABRICATIONS]


def form_candidates() -> list[str]:
    return [f"{f}（{FORM_LABELS.get(f, f)}）" for f in TIP_FORMS]


def auto_name(material: str, fabrication: str, index: Any) -> str:
    """没给名字时生成一个人能认的默认名,如 ``W-etched #3``。"""
    mat = (material or "tip").strip() or "tip"
    fab = (fabrication or "").strip()
    stem = f"{mat}-{fab}" if fab and fab != "unknown" else mat
    try:
        return f"{stem} #{int(index)}"
    except (TypeError, ValueError):
        return stem


# ── 进程级 holder(live-read) ────────────────────────────────────────────────
#
# 注入中间件每次 model call 都要读当前针尖,不该为此打一次 SQLite。持久化真源
# 仍是 tips 表,本 holder 由 tip_registry 在启动 hydrate 与每次登记后刷新。

_lock = threading.RLock()
_current: "dict[str, Any] | None" = None


def set_current_tip(row: "dict[str, Any] | None") -> None:
    """刷新 holder(tip_registry 调用)。``None`` = 仪器里没有已登记的针尖。"""
    global _current
    with _lock:
        _current = dict(row) if isinstance(row, dict) else None


def get_current_tip() -> "dict[str, Any] | None":
    """当前针尖的完整行副本;未登记返回 None。"""
    with _lock:
        return dict(_current) if _current else None


def current_tip_facts() -> "dict[str, Any] | None":
    """给 skill 层的精简事实(材料/制备/形态/qPlus 参数);未登记返回 None。

    **读不到就返回 None,绝不猜**(与 ``qplus_amplitude`` 读基线同款):未登记
    针尖时修针技能退回通用保守参数,而不是假装它是钨腐蚀针。
    """
    tip = get_current_tip()
    if not tip:
        return None
    return {
        "tip_id": tip.get("id"),
        "name": tip.get("name") or "",
        "material": tip.get("material") or "",
        "fabrication": tip.get("fabrication") or "unknown",
        "form": tip.get("form") or "stm_wire",
        "wire_diameter_mm": tip.get("wire_diameter_mm"),
        "installed_at": tip.get("installed_at"),
        "qplus_sensor_model": tip.get("qplus_sensor_model") or "",
        "qplus_f0_hz": tip.get("qplus_f0_hz"),
        "qplus_q": tip.get("qplus_q"),
        "qplus_k_n_per_m": tip.get("qplus_k_n_per_m"),
    }


def is_qplus() -> bool:
    """当前针尖是否 qPlus 传感器。未登记 → False(fail-open,不拦操作)。"""
    facts = current_tip_facts()
    return bool(facts and facts.get("form") == "qplus")


# ── 纯渲染 ───────────────────────────────────────────────────────────────────

def _service_days(installed_at: Any, now: "float | None" = None) -> "int | None":
    """已服役天数;解析不出返回 None。"""
    raw = str(installed_at or "").strip()
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except (TypeError, ValueError):
        return None
    ref = datetime.fromtimestamp(now) if now is not None else datetime.now()
    try:
        return max(0, int((ref - dt).total_seconds() // 86400))
    except (OverflowError, ValueError):      # pragma: no cover
        return None


def _fmt_hz(v: Any) -> str:
    """共振频率:人读单位 + SI(注入文本的铁律,见 test_injection_si_units)。"""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return ""
    if not (f == f and abs(f) != float("inf")):        # NaN / inf
        return ""
    if abs(f) >= 1e3:
        return f"{f / 1e3:.4g} kHz (= {f:.4e} Hz)"
    return f"{f:.4g} Hz"


def _bias_polarity_line(applied_to: Any) -> str:
    """偏压加在哪一侧 —— 决定 dI/dV 谱的能态归属,写错了整套解释就是反的。

    加在样品上(常规约定):正样品偏压把样品能级相对针尖压低,电子由针尖占据态
    隧穿进样品**空态**;负偏压反过来,探测样品**占据态**。
    加在针尖上时符号整体反号。
    未声明时**明说未声明** —— 让模型知道自己不知道,好过让它按默认约定断言。
    """
    side = str(applied_to or "unknown").strip().lower()
    if side == "sample":
        return ("- **偏压极性**：本机偏压加在**样品**上（常规约定）。"
                "正偏压 → 电子由针尖隧穿进样品**空态**（探测未占据态/导带/LUMO）；"
                "负偏压 → 探测样品**占据态**（价带/HOMO）。"
                "解释 STS / dI/dV 谱的能态归属按这条。\n")
    if side == "tip":
        return ("- **偏压极性**：本机偏压加在**针尖**上 —— 符号与常规样品偏压"
                "**整体反号**。正偏压 → 探测样品**占据态**；"
                "负偏压 → 探测样品**空态**。"
                "文献里绝大多数图按样品偏压画，与本机数据比对时先把符号翻过来。\n")
    return ("- **偏压极性：未声明**（不知道偏压加在样品还是针尖）。"
            "在用户补充之前，**不要断言**某个偏压符号对应占据态还是空态，"
            "也不要照搬文献的符号约定；需要时先问用户或让他在设置里填。\n")


def _preamp_line(model: Any, gain: Any) -> str:
    """前置放大器 —— 增益错一个量级，所有报出去的电流就整体错一个量级。"""
    name = str(model or "").strip()
    try:
        g = float(gain)
        if not (g == g) or g <= 0:
            g = None
    except (TypeError, ValueError):
        g = None
    if not name and g is None:
        return ("- **前置放大器：未登记**（型号与跨阻增益都没填）。"
                "电流读数的绝对标度因此无法核对；报电流数值时说明这一点。\n")
    parts = []
    if name:
        parts.append(f"型号 {name}")
    if g is not None:
        parts.append(f"跨阻增益 {g:.4g} V/A（电流 = 读数电压 ÷ 增益）")
    else:
        parts.append("跨阻增益未登记")
    return "- **前置放大器**：" + "，".join(parts) + "。\n"


#: 按 材料 / 形态 给的一句话方案要点。**只讲定性注意事项,不给数字** ——
#: 具体参数一律走 tip_conditioning_resolver 的方案表,不靠模型记提示词里的数。
_MATERIAL_NOTES: dict[str, str] = {
    "W": "钨针硬、易成形，但暴露大气后表面有氧化层，初次进针常需要更强的处理才出好态；氧化层也让它在低偏压 STS 上不如铂铱干净。",
    "PtIr": "铂铱针不氧化、化学惰性好，适合谱学；但材质软，重手法容易钝掉甚至粘针，处理要比钨针轻。",
    "Pt": "铂针软且惰性，处理宜轻。",
    "Ir": "铱针硬度高于铂铱，惰性好。",
    "Fe": "铁针用于自旋极化测量，磁性态是实验对象的一部分——大力处理会改变磁构型，不只是改变几何形状。",
    "Ni": "镍针为磁性针尖，处理会改变磁构型。",
    "Co": "钴针为磁性针尖，处理会改变磁构型。",
    "Cr": "铬针为反铁磁针尖，杂散场小，但处理会改变磁构型。",
    "Nb": "铌针用于超导谱学，回温或污染会破坏针尖超导能隙。",
}
_FAB_NOTES: dict[str, str] = {
    "etched": "电化学腐蚀针尖形貌取决于腐蚀参数，个体差异大。",
    "cut": "剪切针尖顶端不规则、常有多个微尖，双针尖成像的先验概率高于腐蚀针。",
    "ground": "打磨针尖顶端曲率半径通常较大。",
    "fib": "FIB 切割针尖几何可控，但离子注入层可能影响谱学。",
}


#: 注入给模型的 qPlus 说明。**提成模块级常量**是为了让产物声明查得到它 ——
#: 一段每轮都进模型上下文、而且刚刚被推翻过一次的文案,必须能被验证「真的进了
#: 冻结产物」,否则「我改了措辞」这句话在装机之后无从证实。
_QPLUS_POKE_NOTE = (
    "- **qPlus 针尖**：几纳米量级的下压（TipShape / ShapeTipOnSurface）"
    "**可以做，而且是常规手段**，不需要额外征求同意 —— 直接做。"
    "两件事由代码保证，你不必操心也不要重复询问："
    "① 扎针前偏压会自动缓降到 20 mV (2e-2 V)（不降的话音叉起振，那才是真正"
    "毁针的原因，与「扎不扎」无关）；"
    "② 下压深度有包络（``max_poke_depth_m``），超了会被**拒绝**而不是"
    "夹到上限。真正会毁掉音叉的是**远超包络的深扎**，不是扎针本身。\n")


def format_tip_block(
    tip: "dict[str, Any] | None",
    profile: "dict[str, Any] | None" = None,
    *,
    now: "float | None" = None,
) -> str:
    """把「当前针尖 + 信号链约定」渲染成注入块。纯函数。

    **永远返回非空块**(与 instrument_profile 同款,与 experiment_prefs 相反):
    「针尖未登记」「偏压极性未声明」本身就是模型需要知道的事实 —— 不说,它就
    会按最常见的约定默认下去,而那正是解释谱图时最贵的一类错。
    """
    p = dict(profile or {})
    t = dict(tip or {})
    lines = ["## 当前针尖与信号链（针尖登记 / instrument_profile）\n"]

    if not t:
        lines.append(
            "- **当前针尖：未登记**。仪器里物理上当然有针，但没人登记它是什么，"
            "所以修针方案只能用通用保守参数，dI/dV 接触标定与 qPlus 振幅基线"
            "也无法归属到具体某根针。要登记请调 `register_tip`（或在右栏针尖卡片里填）。\n")
    else:
        mat = str(t.get("material") or "").strip()
        fab = str(t.get("fabrication") or "unknown").strip()
        form = str(t.get("form") or "stm_wire").strip()
        bits = [
            MATERIAL_LABELS.get(mat, mat or "材料未记录"),
            FABRICATION_LABELS.get(fab, fab),
            FORM_LABELS.get(form, form),
        ]
        detail = str(t.get("material_detail") or "").strip()
        if detail:
            bits.append(detail)
        dia = t.get("wire_diameter_mm")
        try:
            d = float(dia)
            if d > 0:
                bits.append(f"线材直径 {d:.4g} mm (= {d * 1e-3:.3e} m)")
        except (TypeError, ValueError):
            pass

        name = str(t.get("name") or "").strip() or "未命名"
        installed = str(t.get("installed_at") or "").strip()
        days = _service_days(installed, now)
        when = ""
        if installed:
            when = f"，{installed[:10]} 装入"
            if days is not None:
                when += f"（已服役 {days} 天）"
        lines.append(f"- **当前针尖**：「{name}」—— " + "、".join(bits) + when + "。\n")

        if form == "qplus":
            qb = []
            if str(t.get("qplus_sensor_model") or "").strip():
                qb.append(f"型号 {str(t['qplus_sensor_model']).strip()}")
            f0 = _fmt_hz(t.get("qplus_f0_hz"))
            if f0:
                qb.append(f"标称 f₀ {f0}")
            for key, label in (("qplus_q", "标称 Q"), ("qplus_k_n_per_m", "标称 k")):
                try:
                    v = float(t.get(key))
                except (TypeError, ValueError):
                    continue
                # Q 常在 1e4–1e5,``%.4g`` 会渲染成 "3e+04" —— 对一个用户刚亲手
                # 填进去的整数来说这读着像另一个数。整数就照整数写。
                shown = f"{v:,.0f}" if v == int(v) and abs(v) >= 1000 else f"{v:.4g}"
                qb.append(f"{label} {shown}"
                          + (" N/m" if key.endswith("n_per_m") else ""))
            mf0 = _fmt_hz(p.get("qplus_f0_measured_hz"))
            if mf0:
                mq = p.get("qplus_q_measured")
                qb.append(f"**实测** f₀ {mf0}"
                          + (f"、Q {float(mq):.4g}" if isinstance(mq, (int, float)) else ""))
            if qb:
                lines.append("- **qPlus 传感器**：" + "，".join(qb) + "。\n")
            # ⚠️ 这段话**每一轮都注入进模型的上下文**，所以它必须是真的。
            #
            # 2026-08-17 之前这里写的是：「戳表面类处理有毁掉音叉传感器的风险，
            # 且传感器损坏不可逆……这类技能对 qPlus 针尖默认拒绝，确需执行要显式
            # 置 allow_on_qplus=true。」
            #
            # 两处都不成立了，而且它造成了真实伤害：
            #
            #   · **事实上就是错的**：qPlus 针尖在 nm 尺度下可以安全地扎针，
            #     这个误解从源头上污染了我们的系统。
            #   · **默认值也早就不是「拒绝」了**（08-16 改的，见
            #     ``_tip_policy._guard_on``），所以这句话连描述现状都不对。
            #
            # 后果不是「多一句废话」：模型每一轮都读到「有毁掉传感器的风险 +
            # 默认拒绝」，于是它在该动手的时候回来问「要不要扎针」——
            # 现场反馈:原文案过于犹疑，读起来只是在吓唬人。
            #
            # ⇒ 这里写**真正要注意的那两件事**，而不是一句吓唬人的话。
            #   把危险说过头，和把危险瞒下来，一样是在污染判断。
            lines.append(_QPLUS_POKE_NOTE)

        note = _MATERIAL_NOTES.get(mat, "")
        fnote = _FAB_NOTES.get(fab, "")
        if note or fnote:
            lines.append("- **针尖特性**：" + " ".join(x for x in (note, fnote) if x) + "\n")

    lines.append(_bias_polarity_line(p.get("bias_applied_to")))
    lines.append(_preamp_line(p.get("preamp_model"), p.get("preamp_gain_v_per_a")))
    lines.append(
        "- **修针参数不要自己发明**：调用修针类技能（ConditionTip / TipShape / "
        "TipPulse / ShapeTipOnSurface）时，把你没有把握的参数**留空**——"
        "系统会按当前针尖的「材料 × 制备 × 形态」查方案表填进去，并在结果里"
        "告诉你每个值的来源。你显式给的数会被采纳，但超出该针尖安全包络的会被拒绝。\n")
    return "".join(lines)


__all__ = [
    "TIP_MATERIALS", "TIP_FABRICATIONS", "TIP_FORMS",
    "MATERIAL_LABELS", "FABRICATION_LABELS", "FORM_LABELS",
    "normalize_material", "normalize_fabrication", "normalize_form",
    "material_candidates", "fabrication_candidates", "form_candidates",
    "auto_name",
    "set_current_tip", "get_current_tip", "current_tip_facts", "is_qplus",
    "format_tip_block",
]
