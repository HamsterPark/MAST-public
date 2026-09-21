# MAST — Modular Autonomous SPM Toolkit

**从 AI 推理到真实仪器操作的全栈实验系统。**

MAST 面向扫描隧道显微镜（STM），将多智能体协作、仪器控制、视觉观测、数据处理与操作员界面连接起来，
把 AI 的实验意图落实为带有执行约束、状态反馈和人工介入入口的物理操作。

MAST is a full-stack multi-agent system for real STM experiments, connecting experiment planning,
instrument control, perception, data analysis and an operator interface.

![Python 3.13](https://img.shields.io/badge/python-3.13-blue) ![TypeScript](https://img.shields.io/badge/frontend-React%2018%20%2B%20TypeScript-blue) ![License: MIT](https://img.shields.io/badge/license-MIT-green) ![snapshot](https://img.shields.io/badge/snapshot-2026-09-21-lightgrey)

## 已有成果

**上一版本中，外部 AI agent 已通过 MAST 控制真实 STM 运行五天五夜。**
这段跨昼夜的运行经历，将 agent 决策、仪器执行与状态反馈带入了实际实验流程。
MAST 6.5.0 在此基础上新增 `/api/ext/v1` 与配套 MCP 集成：新接口已完成软件测试，真机验证尚待完成。

**In the previous version, an external AI agent operated a real STM through MAST over five days and five nights.**
Version 6.5.0 builds on that experience with a new external API and MCP integration,
which have undergone software testing and await hardware validation.

![Tip conditioning on Au through MAST: eight STM frames on a relative timeline](docs/assets/au-tip-repair-overview.png)

*外部 agent 通过上一版本 MAST 控制路径在 Au 上修针的真实 STM 图像序列。时间相对首张展示帧；#0652 采用逐行一阶调平。*

本次公开源码基线 `9884ff5` 已于 **2026-09-21** 在 Windows / Python 3.13 环境完成无需仪器的软件验证：

| 验证项目 | 已记录结果 |
|---|---|
| 后端测试 | **14,356 通过，0 失败**；46 跳过、2 个预期失败 |
| 前端单元测试 | **985 / 985 通过** |
| 前端构建与类型检查 | 通过 |
| API 契约 | 302 条 OpenAPI 路径、577 个 schema；前端类型已同步 |
| 发布检查 | Python 语法、import 闭包、公开内容清理检查通过 |

测试使用替身与合成数据；真机经历与软件验证分别记录。环境、命令、跳过项及复现前提见[验证记录](docs/OPEN_SOURCE_NOTES.md#四import-闭包与测试)。

## 核心工程能力

MAST 的工程工作横跨 AI 编排、物理仪器、科学数据与 Web 应用。公开源码保留了以下主线：

| 工程问题 | 已实现的机制 | 代表入口 |
|---|---|---|
| 将模型动作接入物理仪器 | 技能参数范围与 SI 数量级校验、执行模式与安全检查、仪器占用权、中止处理；原子技能与声明式复合流程共享执行约束 | [执行上下文](MASTv2/mast/core/execution_context.py)、[安全检查](MASTv2/mast/core/safety.py) |
| 在多个 agent 之间延续实验任务 | 编排器与七个域 agent 分工；以类型化产物引用、有界摘要和状态归并传递工作成果 | [状态契约](MASTv2/mast/agents/state.py)、[产物通道](MASTv2/mast/agents/_shared/artifact_channel.py) |
| 将持续观测供给较慢的推理 | 视觉、监控与缓冲层组织仪器观测，向 agent 暴露观测年龄与扫描推进状态 | [缓冲服务](MASTv2/mast/buffer/service.py)、[观测工具](MASTv2/mast/agents/_shared/buffer_tools.py) |
| 处理真实系统中的故障 | 控制器协议的分段读取、连接中断与超时处理；WebSocket 重连与轮询回退；SSE 区分完成、断流与静默超时 | [协议补丁](MASTv2/mast/core/nanonis_patch.py)、[前端流式通信](frontend/src/lib/) |
| 将能力开放给外部 agent | 外部 API、MCP 集成和技能投稿接口；作业请求去重、内容冲突检查与重启后的状态处理 | [外部作业](MASTv2/mast/api/ext/jobs.py)、[Claude Code 集成](integrations/claude-code/) |

[中文审阅指南](AGENTS.zh.md) / [English review guide](AGENTS.md) 将这些机制逐项连接到源码与回归测试，
适合人工阅读，也适合 Codex 等代码助手从实现出发检查项目。

## 系统架构

```text
实验意图 / 操作员
        │
        ▼
多 agent 协作 ── 类型化产物与状态交接
        │
        ▼
instrument_control ── 技能与复合流程 ── 执行校验与安全检查 ── Nanonis / STM
        ▲                                                          │
        │                                                          ▼
观测上下文 ◀──────── 缓冲与状态层 ◀─────────────── 视觉 / 仪器监控
        │
        └──────────────── API / React 操作员界面
```

分层设计将不同时间尺度的工作接起来：控制器承担底层闭环，视觉与监控产生观测，
缓冲层整理状态，agent 完成较慢的推理、任务拆分与交接，操作员通过界面观察与介入执行。
域 agent 中由 `instrument_control` 调用仪器技能；手动操作与外部接口的执行入口另见审阅指南。

`core`、`skills`、七个域 agent、内部 API、视觉与监控、数据 I/O、缓冲、
签名增量更新及网络模块均已有在维护者仪器上的功能运行经历，具体经验对应当时使用的功能与版本。
长时间规划 `conduct/`、自研运行时 `agentruntime/`、`goals/`、技能工坊与市场及 qPlus 路径仍处于实验阶段，
尚待真机验证。自研运行时与 LangGraph 路径共存，`engine_v2_*` 开关默认关闭。

进一步阅读：[agent 拓扑](docs/v2/agent-topology.md)、[技能目录](docs/v2/skill-catalog.md)、[模型服务适配](docs/api_providers/)。

## 公开源码规模

| 量 | 值 |
|---|---|
| Python 源文件 / 物理行（`MASTv2/mast/`） | 843 / 315,846 |
| 手写 TypeScript 行（`frontend/src/`，不含生成的 `schema.d.ts`） | 65,628 |
| 定义了 `execute()` 的技能类（builtins + composite） | 444 |
| HTTP / WebSocket 路由声明（静态计数） | 365 |
| 测试函数定义（`tests/`，不展开参数化用例） | 11,659 |
| agent | orchestrator + 七个域 agent（research_director、literature、experiment_design、instrument_control、data_processing、paper_writing、paper_review）+ brainstorm、buffer_summarizer 两个辅助节点 |


以上按发布树静态统计：技能按类定义、路由按声明、测试按函数计数；运行时注册项、
OpenAPI 路径及参数化测试用例使用不同口径。项目自 2026-03-17 起迭代，导出时私有仓累计 1016 次提交。

## 公开范围

本仓是 **MAST 6.5.0 的公开精简源码版**，保留多 agent 编排、通用仪器技能、执行与安全机制、
感知和数据处理框架、React 操作界面、外部 agent 接口、技能投稿示例及相应测试。
通用图库与文献管理代码随仓提供，初始数据为空。

公开范围综合考虑**第三方版权与许可、知识产权与商业化安排，以及未发表研究和现场数据保护**。
部分专用模块、论文移植技能、知识资产、视觉权重、仪器标定、站点配置与私有开发历史不随仓提供。
本版本面向源码审阅、技术交流与无需仪器的软件测试；完整部署需补齐相应配置与资产。
预处理内置 profile 是未标定示例，须使用自备数据评估并配置阈值。

*This public source edition preserves the project's general engineering framework, interfaces and tests.
Selected specialized modules and assets are excluded for licensing, intellectual-property and commercialization
considerations, and to protect unpublished research and site data. Instrument deployment requires additional
configuration and assets.*

具体保留项、删减处理及影响见[公开版本说明](docs/OPEN_SOURCE_NOTES.md)；
公开代码的许可见 [LICENSE](LICENSE) 与 [THIRD_PARTY.md](THIRD_PARTY.md)。

## 阅读、验证与参与

- **审阅实现**：[AGENTS.zh.md](AGENTS.zh.md) / [AGENTS.md](AGENTS.md) 提供六条实现与测试阅读路线。
- **本地验证**：同一指南提供 Python 3.13、Node 24 的准备方法与检查命令，可运行测试、构建前端，无需连接仪器。
  上述无需硬件的测试使用仪器替身，本仓不附带完整仪器模拟器；PDF 取全文与 OCR 的可选依赖另见 `MASTv2/requirements-pdf.txt`。
- **扩展技能**：欢迎向 `contrib/skills/` 投稿，社区技能经维护者真机验证后可进入官方树。
  规则见 [CONTRIBUTING.zh.md](CONTRIBUTING.zh.md) / [CONTRIBUTING.md](CONTRIBUTING.md)。
- **接入外部 agent**：[使用指南](docs/external/)与 [Claude Code 插件](integrations/claude-code/)提供集成入口。

## 许可 · 第三方 · 投稿 · 安全

[LICENSE](LICENSE)（MIT）· [THIRD_PARTY.md](THIRD_PARTY.md) · [CONTRIBUTING.md](CONTRIBUTING.md) / [CONTRIBUTING.zh.md](CONTRIBUTING.zh.md) · [SECURITY.md](SECURITY.md) / [SECURITY.zh.md](SECURITY.zh.md)
