# LLM provider 适配参考

本目录记录 MAST 对多家 LLM 的接口适配，包括跨协议消息转换、多轮推理字段保留、
工具调用完整性和图像输入。专题资料提供协议背景，源码路线说明这些差异怎样进入
直接客户端与 agent 客户端，历史检查记录说明适配过程中解决的问题。

资料主要整理于 2026 年 6 月至 8 月，各文件保留其日期与来源；模型名、参数和调用观察
按相应记录日期理解。当前云端规格、SDK 行为及账户可用性需要结合目标厂商的最新官方文档
另行核对。本次公开文档整理保留历史范围，没有重新进行云端验证。

## 按接口查阅

| Provider / 能力 | 随仓参考 |
|---|---|
| Anthropic：思考内容与 Messages API | [anthropic_extended_thinking.md](anthropic_extended_thinking.md) |
| Moonshot / Kimi：思考内容与工具调用 | [moonshot_kimi_k2_thinking.md](moonshot_kimi_k2_thinking.md) |
| Moonshot / Kimi：图像输入 | [moonshot_kimi_vision.md](moonshot_kimi_vision.md) |
| DeepSeek：reasoning 接口与 V4 适配记录 | [deepseek_reasoning_model.md](deepseek_reasoning_model.md)、[deepseek_v4_thinking_mode.md](deepseek_v4_thinking_mode.md) |
| Qwen / DashScope：思考内容 | [qwen_dashscope_thinking.md](qwen_dashscope_thinking.md) |
| MiniMax：Anthropic 兼容接口 | [minimax_anthropic_compat.md](minimax_anthropic_compat.md) |
| Zhipu GLM：适配记录 | [zhipu_glm.md](zhipu_glm.md) |
| 多 provider 图像输入对照 | [vision_support.md](vision_support.md) |
| DashScope 实时语音 | [dashscope_realtime_voice.md](dashscope_realtime_voice.md) |

[AUDIT.md](AUDIT.md) 汇总消息回传、思考参数和工具绑定方面的检查与修复。
专题文档中的“当前”“默认”等措辞描述其记录日期的状态，实际部署以选定配置为准。

## 从请求到响应的代码路线

以下路径相对于仓库根目录。

| 工程问题 | 当前源码入口 |
|---|---|
| 直接客户端的请求构造与跨协议消息转换 | `MASTv2/mast/llm/client.py`：`_anthropic_messages_to_openai`、`_openai_response_to_anthropic` |
| agent 客户端的 SDK、参数与工具调用配置 | `MASTv2/mast/agents/_shared/models.py`：`make_chat_model` |
| 多轮工具调用的推理字段保留 | `MASTv2/mast/agents/_shared/reasoning_chat_model.py` 与直接客户端的消息转换函数 |
| 模型能力、思考模式与上下文限制 | `MASTv2/mast/config.py`：`model_thinking_mode`、`model_supports_vision`、`model_input_context`、`model_output_limit` |
| 工具参数在完整接收后进入执行校验 | `MASTv2/mast/agents/_shared/models.py` 的 `_DISABLE_STREAMING_FOR_TOOLS`；`MASTv2/mast/agents/_shared/skill_adapter.py` 的参数转换与校验 |

能力表按具体 model ID 登记图像输入和思考模式。请求构造再结合运行配置选择参数；
这种分层让 provider 差异集中在适配层，同时保留技能执行层的统一约束。

## 验证入口

`tests/v2/unit/llm/test_qwen_client_and_models.py` 和
`tests/v2/agents/_shared/test_vision_channel.py` 提供参数适配与消息转换的回归入口。
阅读测试使用的替身、SDK 和断言即可判断其覆盖范围。当前云端兼容性通过另行记录的真实请求验证；
普通代码审阅可直接从源码和离线测试开始，环境见
[仓库指南](../../AGENTS.md) 与[公开版本说明](../OPEN_SOURCE_NOTES.md)。
