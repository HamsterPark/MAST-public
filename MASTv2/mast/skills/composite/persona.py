"""Persona context profiles — agent 上下文切片注册表（P2-C，RFC §8）。

llm 节点经 ``persona`` + ``context_sections`` 字段借用某个 agent 的领域上下文
做一次受限决策（不带工具、不跑 agent 循环）。切片带版本号：决策日志记录
``persona@version``，agent prompt 演化后旧工作流的行为漂移可被审计。

设计取舍：各 agent 的 SYSTEM_PROMPT 是单一大常量（无结构化 section），按行
切片脆弱——所以这里维护**策展切片**（短、聚焦、中文，专为单次决策注入而写），
另保留 ``full_system_prompt`` 切片懒加载真实 agent prompt（token 大，慎用）。
新增/修改切片必须 bump 对应 persona 的 version。
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# {persona_id: {"version": int, "description": str,
#               "sections": {section_id: {"title": str, "text": str}}}}
PERSONAS: dict[str, dict] = {
    "instrument_control": {
        # v2 (2026-08-01): 加 ``safe_text`` —— 安全模式下的替代文本。非安全模式
        # 注入内容逐字未变。见 render_sections 里的说明。
        "version": 2,
        "description": "仪器控制（IC）agent 的 STM 操作领域上下文",
        "sections": {
            "tip_conditioning": {
                "title": "针尖整备",
                "text": (
                    "针尖质量评判与修复常识：质量分数低/双针尖/不稳定时，常用"
                    "手段从温和到激进依次为：小幅偏压脉冲（TipPulse，约 ±2~4 V）、"
                    "重复脉冲、受控轻触表面（poke）。每次干预后必须重新扫描评估"
                    "再决定下一步；连续多次无改善应升级人工处理而不是无限重试。"
                    "修复动作只能从预设技能选择，参数有硬上限。"),
                "safe_text": (
                    "当前为**安全模式**：针尖视为状态良好，本模式下不做任何针尖"
                    "处理——不修针、不打偏压脉冲、不做 tip shaping。请直接用当前"
                    "针尖继续采集数据，不要评估是否需要修针。针尖确有问题时由"
                    "用户人工处理，或由用户切换到 semi/auto 模式。"),
            },
            "scanning": {
                "title": "扫描成像",
                "text": (
                    "扫描质量受针尖状态、漂移、反馈参数影响。图像质量不佳时先"
                    "区分：针尖问题（条纹/重影/突变）→ 修针；漂移（倾斜拉伸）→ "
                    "等待稳定或重扫；参数问题（过快/增益不当）→ 调整扫速或"
                    "setpoint。换区域是针尖反复修不好时的替代选项。"),
                "safe_text": (
                    "扫描质量受针尖状态、漂移、反馈参数影响。图像质量不佳时先"
                    "区分：漂移（倾斜拉伸）→ 等待稳定或重扫；参数问题（过快/"
                    "增益不当）→ 调整扫速或 setpoint；换区域也是可选项。"
                    "**安全模式下不修针**——即使怀疑是针尖问题，也请继续采集或"
                    "换区域，不要尝试任何修针动作。"),
            },
            "safety": {
                "title": "安全边界",
                "text": (
                    "Withdraw 是静态安全态（Z 控制器关 + Z 最高位）。粗逼近"
                    "（z-approach）是唯一物理高危动作，绝不在自动工作流里做，"
                    "永远交给人。偏压/电流/扫描范围由系统钳位，你的职责只是"
                    "在给定选项中做判断，不输出任何硬件参数。"),
            },
        },
    },
    "data_processing": {
        "version": 1,
        "description": "数据处理（DP）agent 的分析领域上下文",
        "sections": {
            "image_quality": {
                "title": "图像质量评估",
                "text": (
                    "评估扫描图像时综合：噪声水平、台阶/原子分辨可见度、扫描"
                    "伪影（条纹、重影、漂移畸变）。质量分数是近似指标，临界值"
                    "附近的判断应倾向保守（宁可重扫/升级人工，不要放行坏数据）。"),
            },
        },
    },
}


def list_personas() -> list[dict]:
    """前端/API 用的档案目录：[{id, version, description, sections:[{id,title}]}]"""
    out = []
    for pid, p in PERSONAS.items():
        out.append({
            "id": pid, "version": p["version"],
            "description": p["description"],
            "sections": [{"id": sid, "title": s["title"]}
                         for sid, s in p["sections"].items()],
        })
    return out


def _section_text(section: dict) -> str:
    """切片正文；安全模式下优先用 ``safe_text``。

    这条注入路径是 composite 内 LLM 决策节点专用的（``llm_node``），**不经过
    agent 图**，所以 ``ModeBeliefMiddleware`` 那层安全模式信念根本够不到它。
    2026-08-01 审计发现：安全模式下全系统只剩这里还在原样告诉模型「质量分数低/
    双针尖/不稳定 → 打脉冲、poke」「图像不好 → 修针」，而且**完全没有任何对冲**。

    非安全模式逐字不变：没有 ``safe_text`` 的切片，或非 SAFE 时，都走 ``text``。
    """
    try:
        from mast.core.operating_mode import safe_mode_active
        if safe_mode_active():
            alt = section.get("safe_text")
            if alt:
                return str(alt)
    except Exception:  # noqa: BLE001 — 注入不能因为读模式失败而中断
        pass
    return str(section.get("text", ""))


def render_sections(persona: str, sections=None) -> str:
    """拼出注入文本。persona 可带版本后缀（"instrument_control@1"）——版本不符
    记警告但仍注入当前版（行为漂移可审计，不可静默失败）。

    sections=None → 注入该 persona 全部策展切片；含 "full_system_prompt" 时
    追加真实 agent SYSTEM_PROMPT（懒加载，失败降级跳过）。未知 persona 抛
    KeyError（调用方 llm_node 捕获并降级为无 persona）。"""
    pid, _, ver = str(persona).partition("@")
    p = PERSONAS[pid]   # KeyError → llm_node 降级
    if ver and str(p["version"]) != ver:
        logger.warning("persona %s requested @%s but current is @%s — "
                       "injecting current (drift is auditable via the "
                       "decision log)", pid, ver, p["version"])
    wanted = list(sections) if sections else list(p["sections"].keys())
    parts: list[str] = []
    for sid in wanted:
        if sid == "full_system_prompt":
            try:
                import importlib
                mod = importlib.import_module(f"mast.agents.{pid}.prompts")
                parts.append(str(getattr(mod, "SYSTEM_PROMPT", "")))
            except Exception as exc:  # noqa: BLE001 — 注入降级，不阻断决策
                logger.warning("full_system_prompt for %s unavailable: %s",
                               pid, exc)
            continue
        s = p["sections"].get(sid)
        if s is None:
            logger.warning("persona %s has no section %r — skipped", pid, sid)
            continue
        parts.append(f"【{s['title']}】{_section_text(s)}")
    return "\n".join(parts)


__all__ = ["PERSONAS", "list_personas", "render_sections"]
