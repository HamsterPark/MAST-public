# LLM 接口适配：历史检查记录与源码入口

MAST 的 provider 适配检查覆盖了两条客户端路径、跨协议消息转换、多轮推理字段、
思考档位和工具绑定。检查推动了输出预算、档位映射和 provider 专属参数的修复，
并将工具调用完整性与仪器执行约束连接起来。

本页整理 2026-06-01 及 2026 年 8 月的检查记录。历史调用结果按当时模型与环境理解；
当前云端规格和 SDK 行为需另行核对。公开源码与回归测试提供实现证据，原始云端响应及
探针脚本未随仓。本次整理没有重新调用云端模型。专题来源见[参考索引](README.md)。
以下路径相对于仓库根目录。

## 可在公开源码中追踪的适配机制

| 工程问题 | 实现入口 | 机制 |
|---|---|---|
| 直接聊天与 agent 使用不同客户端 | `MASTv2/mast/llm/client.py`、`MASTv2/mast/agents/_shared/models.py` | 两条路径分别构造请求，检查需覆盖各自的参数与 SDK 边界。 |
| 多轮工具调用保留 provider 返回的推理字段 | `MASTv2/mast/llm/client.py` 的 `_anthropic_block_to_dict`、`_anthropic_messages_to_openai`、`_openai_response_to_anthropic`；`MASTv2/mast/agents/_shared/reasoning_chat_model.py` | 在消息格式转换与再次发送之间保留所需字段，使工具往返具有连续上下文。 |
| 不同模型使用不同思考参数 | `MASTv2/mast/config.py`、`MASTv2/mast/agents/_shared/models.py::make_chat_model` | 按 model ID 和档位选择 adaptive、固定预算或 provider 专属参数。 |
| 工具参数完整性影响物理数量级 | `MASTv2/mast/agents/_shared/models.py` 的 `_DISABLE_STREAMING_FOR_TOOLS`；`MASTv2/mast/agents/_shared/skill_adapter.py` | 工具调用采用完整参数后再解析，普通无工具文本仍可流式输出；执行层继续检查参数范围与 SI 前缀。 |
| 图像输入格式与模型能力存在差异 | `MASTv2/mast/config.py::model_supports_vision`；`tests/v2/agents/_shared/test_vision_channel.py` | 具体模型的能力登记与消息块转换分别承担选择和序列化职责。 |

保存的设置与 per-agent 覆盖共同决定实际运行配置。复核一次请求时，将源码版本、
生效配置与请求内容对应起来，就能追踪界面选项如何变成发送参数。

## 2026-06-01：消息回传、输出预算与思考档位

历史检查及后续补充记录了以下修复：

| 检查发现 | 当时完成的处理 |
|---|---|
| 多轮工具调用需要同时保留可见回答与推理内容 | 检查直接客户端的消息转换与 agent 客户端的推理字段回传；保留对应字段。 |
| 较小输出上限可能截断推理内容与答案 | 调整 Kimi 预设和 agent 推理模型的输出预算下限。 |
| 设置页提供的 `max` 档位缺少对应映射 | 补全 agent 思考档位映射，修复选择 `max` 后静默关闭思考的问题。 |
| 不同 provider 的思考控制参数有差异 | 移除 Qwen 的 `preserve_thinking` 请求标志，并按模型选择 adaptive 或固定预算。 |
| 新 provider 与模型需要同时接入两条调用路径 | 加入 GLM 适配；记录了部分 Anthropic 和 GLM 模型在直接客户端及 agent 路径的调用成功。 |

这些工作体现了从设置项、请求构造到多轮响应回传的完整检查范围。
当前默认档位、token 上限与模型名单由源码和生效配置确定；历史数值用于解释修复背景。

## 2026-08：工具可见性与强制调用的接口差异

历史探针研究了一个执行层问题：当请求要求调用未绑定的工具，同时要求模型必须发起工具调用时，
部分 provider 返回了已绑定的另一个工具名；其他 provider 返回文本拒绝、忽略强制设置，
或返回未绑定的名字。探针据此区分了 provider、模型与调用条件，而没有使用统一的兼容性假设。

这个观察将适配工作推进到工具调用的语义层：合法的工具名和参数形状还需要与用户意图对应。
模型的解释文字与实际工具调用也可能出现差异。沿以下路径可以检查处理边界：

1. 本轮实际绑定的工具集合，以及按需加载后可见性的变化。
2. `tool_choice` 等控制参数的设置位置与目标模型要求。
3. 未知工具名、参数拒绝和中止状态如何进入可观察的结果。
4. 运行模式、仪器占用权和具体操作检查如何约束实际执行。

工具数量探针也记录过不同 provider 对大工具集合的响应，适用范围是当时的模型与请求条件。
原记录提出了后续候选改进；当前实现进度应沿实际工具绑定路径及其测试核对。

## 复核入口

从 `tests/v2/unit/llm/test_qwen_client_and_models.py`、
`tests/v2/agents/_shared/test_vision_channel.py` 和
`tests/v2/unit/core/test_execution_context_mode_gate.py` 分别检查参数构造、消息形状与执行边界。

任务需要云端验证时，记录模型 ID、SDK 版本、请求形状、工具集合、响应和日期，
分别确认请求被接受、工具调用语义和任务结果。源码与离线测试则支持无需密钥的日常代码审阅。
