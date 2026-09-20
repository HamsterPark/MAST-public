"""六家 provider 的**踩过的坑**，写成一张可执行的表 —— 二期的前置。

这就是那张表。它现在还不驱动任何代码路径（今天那些约束由 ``llm_route`` /
``client.py`` / ``config.py`` 各自实现），它的作用是**让二期换 provider 层的时候，
每一条教训都有一个会红的判据**。

为什么非要有它
--------------
二期要把 ``mast/llm/client.py`` 扩成 6 provider 的自有客户端。那时今天这些约束会被
重新实现一遍 —— 而它们**每一条都对应可复现的 API 兼容性约束**，散落在三个文件的
docstring 散文里。散文不会在重写时变红。

⚠️ 它**刻意不覆盖**的东西（避免造第二真源）
--------------------------------------------
* **thinking 模式**（none / fixed / tunable）—— ``config.model_thinking_mode()`` 是
  单一真源，聊天客户端与设置页都读它。这里只**引用**它，一个字都不复制。
  仓库已经因为「同一个名单抄第二遍」出过事故，不再添一次。
* **模型价格 / 上下文窗口** —— 分别在 ``billing/pricing.py`` 与 ``config`` 里。
* **key 从哪来** —— ``agents/_shared/models._PROVIDER_KEY_FILE``。

每条 trait 带三样东西
---------------------
``why``（约束是什么）· ``symptom``（违反了会看见什么）· ``source``（谁说的）。

**``symptom`` 那一栏不是装饰**：这些坑的共同点是**症状离原因很远**。「多轮第二次
400」看起来像网络问题，「思维链被截断」看起来像模型笨了，「思考选了 max 却没变化」
看起来像参数没生效 —— 而它们各自的原因是 reasoning_content 没回传、max_tokens 太小、
档位表缺一个键。**没有 symptom 那一栏，这张表在真正需要它的那天派不上用场。**
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Trait:
    """一条踩过的坑。"""

    why: str
    symptom: str
    source: str
    #: 今天已经有代码在守它吗？有的话写在这里，测试会去核。
    #: 空 = 只是记录，**还没有任何东西在防它** —— 那本身是要看见的信息。
    enforced_by: str = ""


#: ── 全六家共有的约束（来自 ``agents/_shared/llm_route.py`` 的 docstring）────
#:
#: 那段散文是 2026-06-08「F5 provider-portability fix」的产物，一次修了五家。
#: 原文标注了归属的就照抄归属；**没标注的不替它编** —— 「response_format / prefill /
#: replayed tool ids 各破坏一家」原文只给了候选集合 (deepseek/qwen/glm/sonnet/minimax)，
#: 没说哪条对哪家。编一个精确归属出来，比留着「未逐一归属」更糟：它会变成一句谁也
#: 核不动、又看起来很确定的话。
SHARED: dict[str, Trait] = {
    "structured_output_method": Trait(
        why='``with_structured_output`` 必须用 ``method="function_calling"``，'
            "不能用 response_format。",
        symptom="换一家 provider 就抛/返回空，而同样的 schema 在另一家上好好的。",
        source="llm_route.py docstring（2026-06-08 F5 provider-portability fix）；"
               "原文：response_format / prefill / replayed tool ids 各破坏一家"
               "（deepseek/qwen/glm/sonnet/minimax，**未逐一归属**）",
        enforced_by="agents/_shared/llm_route.py::route_decision",
    ),
    "flatten_before_routing": Trait(
        why="路由前必须把消息拍平成纯 {role, content} 文本，**剥掉 tool-call id**。",
        symptom="minimax / sonnet 在重放带 tool_call_id 的历史时 400。",
        source="llm_route.py::flatten_messages_for_router docstring",
        enforced_by="agents/_shared/llm_route.py::flatten_messages_for_router",
    ),
    "json_hint_contains_literal_json": Trait(
        why='文本回退的提示里必须**字面**出现 "json" 这个词。',
        symptom="qwen 不进 JSON 模式，返回散文，解析失败后整轮路由落空。",
        source="llm_route.py docstring（点名 qwen）",
        enforced_by="agentruntime/routing.py::make_llm_router 的 json_hint",
    ),
    "conversation_ends_on_user_turn": Trait(
        why="送出去的消息列表必须以 user 轮结尾。",
        symptom="sonnet 拒绝以 assistant 轮结尾的请求。",
        source="llm_route.py docstring（点名 sonnet）",
        enforced_by="agentruntime/routing.py::make_llm_router",
    ),
    "no_parallel_tool_calls": Trait(
        why="OpenAI 兼容后端显式发 ``parallel_tool_calls: False``。",
        symptom="并行工具调用回来之后，配对与顺序假设全部落空。",
        source="mast/llm/client.py:193",
        enforced_by="mast/llm/client.py::_chat_openai_compat",
    ),
    "bare_name_needs_exactly_one_mention": Trait(
        why="裸名字兜底只在**恰好一个**目标被词边界命中时才认。",
        symptom="0 个或 2 个命中却猜一个出来 —— 而猜出来的目标会去驱动真实仪器。",
        source="llm_route.py::parse_route_text（2026-06-08 对抗性复核）",
        enforced_by="agents/_shared/llm_route.py::parse_route_text",
    ),
    "brace_balanced_json": Trait(
        why="从每一个 '{' 起做**括号配平**解码，不能按第一个 '}' 截断。",
        symptom='reason 字符串里出现 "}" 时，JSON 被腰斩、整条决策丢失。',
        source="llm_route.py::parse_route_text docstring",
        enforced_by="agents/_shared/llm_route.py::parse_route_text",
    ),
}


#: ── 各家自己的坑 ──────────────────────────────────────────────────────
#:
#: 全部来自 ``docs/api_providers/AUDIT.md``（2026-06-01 逐家对照官方文档的审计）。
#: 项目规约 的硬规矩：**改 LLM provider 参数前先读 docs/api_providers/**。
PER_PROVIDER: dict[str, dict[str, Trait]] = {
    "anthropic": {
        "adaptive_thinking_required": Trait(
            why="Opus 4.6/4.7/4.8 与 Sonnet 4.6 **拒绝**手动 "
                "``thinking={type:enabled, budget_tokens}``，要发 "
                "``thinking={type:adaptive}`` + ``output_config={effort}``。",
            symptom="400。而 claude-opus-4-7 在改之前就是一直报错。",
            source="AUDIT.md §5（2026-06-01，实测主聊天 + agents 路径均通过）",
            enforced_by="mast/llm/client.py::_chat_anthropic 的 ADAPTIVE_THINKING_MODELS",
        ),
        "thinking_blocks_carry_signature": Trait(
            why="thinking block 带 ``signature``，多轮里必须**原样保留**。",
            symptom="签名丢了之后再带 tool_use 回去会被拒。",
            source="AUDIT.md §多轮 reasoning/thinking 回传",
            enforced_by="mast/llm/client.py::_anthropic_block_to_dict",
        ),
    },
    "minimax": {
        "base_url_is_minimaxi": Trait(
            why="接口在 **api.minimaxi.com**（``/anthropic`` 兼容面），不是 .io。",
            symptom="连不上 / 401，而域名看起来完全合理。",
            source="项目规约「模型 provider」段 + docs/api_providers/minimax.md",
            enforced_by="",
        ),
        "multi_turn_thinking_verbatim": Trait(
            why="官方要求多轮 thinking **原样回带**。",
            symptom="第二轮起模型行为漂移或直接拒绝。",
            source="AUDIT.md §多轮 reasoning/thinking 回传",
            enforced_by="mast/llm/client.py（走 anthropic-compat 路径）",
        ),
    },
    "moonshot": {
        "max_tokens_floor_16000": Trait(
            why="Kimi K2.6 要求 max_tokens ≥ 16000（预设已从 8192 提到 16384）。",
            symptom="**reasoning_content + 答案一起被截断** —— 现场看起来像"
                    "「模型突然变笨了」，而不是像一个配置问题。",
            source="AUDIT.md 已应用的修复 1 + 3",
            enforced_by="mast/config.py 预设 + models.make_chat_model 的下限",
        ),
        "v1_128k_has_no_thinking": Trait(
            why="``moonshot-v1-128k`` **没有** thinking；thinking 参数一个都不能发。",
            symptom="400，或者更糟：静默地不对。",
            source="config.model_thinking_mode docstring（点名这个型号）",
            enforced_by="mast/config.py::model_thinking_mode → \"none\"",
        ),
    },
    "deepseek": {
        "echo_reasoning_when_tools": Trait(
            why="V4 thinking_mode：**有工具调用时必须回传 reasoning_content**；"
                "无工具调用可省。我们始终保留（安全的那一侧）。",
            symptom="带工具的第二轮 400。",
            source="AUDIT.md §多轮（并注明 ``deepseek-reasoner`` 规则**相反** ——"
                   "回传会 400，所以我们不用它）",
            enforced_by="mast/llm/client.py 的 reasoning_content round-trip",
        ),
    },
    "qwen": {
        "no_preserve_thinking_param": Trait(
            why="DashScope **没有** ``preserve_thinking`` 这个参数（非标准）。"
                "CoT 回传靠消息历史，不靠请求标志。",
            symptom="发一个不存在的参数 —— 而它不报错，只是什么也不做。",
            source="AUDIT.md 已应用的修复 2",
            enforced_by="",
        ),
        "round_trip_is_unconfirmed": Trait(
            why="Qwen 的 reasoning_content 多轮回传**官方未明确要求**；我们保留。",
            symptom="如遇 400 再调 —— 这一条是**已知的不确定**，不是已知的事实。",
            source="AUDIT.md 仍可选的增强",
            enforced_by="",
        ),
    },
    "zhipu": {
        "glm_returns_reasoning_content": Trait(
            why="GLM-5.1（open.bigmodel.cn/api/paas/v4，OpenAI 兼容）返回 "
                "reasoning_content，纳入 REASONING_MODELS ⇒ round-trip + "
                "max_tokens 下限。",
            symptom="不纳入的话，第二轮丢 CoT / 被截断。",
            source="AUDIT.md §7（2026-06-01，实测主聊天 + agents 路径均通过）",
            enforced_by="mast/config.py REASONING_MODELS",
        ),
    },
}


def canonical_providers() -> tuple[str, ...]:
    """六家的名单 —— **派生**自 ``models._FALLBACK_MODEL_BY_PROVIDER``。

    不在这里另抄一份：「名单抄第二遍就会漂」是本仓记过的事故形状。加第七家 provider
    而没来补 trait 时，测试会点名说是哪一家。
    """
    from mast.agents._shared.models import _FALLBACK_MODEL_BY_PROVIDER

    return tuple(sorted(_FALLBACK_MODEL_BY_PROVIDER))


def thinking_mode(model_id: str) -> str:
    """转发给**单一真源** ``config.model_thinking_mode``。

    这里刻意只做转发、不缓存也不复制：thinking 能力表已经有主人（聊天客户端与设置页
    都读它）。这个函数存在只是为了让二期的读者在这张表里找 thinking 时**找得到路**，
    而不是顺手在这里再列一张。
    """
    from mast.config import model_thinking_mode

    return model_thinking_mode(model_id)


def unenforced() -> dict[str, list[str]]:
    """今天**还没有任何代码在守**的 trait —— 按 provider 列出来。

    ★ 这个函数是这张表最有用的一半。一张全是「已经在守了」的表读起来很安心，而
    安心正是它最没用的时候：**要紧的是哪几条只写在纸上。** 二期重写 provider 层时，
    这些是最容易悄悄丢掉的 —— 因为没有东西会变红。
    """
    out: dict[str, list[str]] = {}
    for name, t in SHARED.items():
        if not t.enforced_by:
            out.setdefault("_shared", []).append(name)
    for prov, traits in PER_PROVIDER.items():
        for name, t in traits.items():
            if not t.enforced_by:
                out.setdefault(prov, []).append(name)
    return out


__all__ = ["Trait", "SHARED", "PER_PROVIDER", "canonical_providers",
           "thinking_mode", "unenforced"]

