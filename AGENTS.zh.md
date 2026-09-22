# MAST — 代码审阅与开发指南

[English](AGENTS.md)

MAST（Modular Autonomous SPM Toolkit）是面向真实扫描隧道显微镜（STM）实验的全栈多智能体系统。
系统连接实验规划、仪器技能、确定性的执行校验、视觉观测、数据处理与 React 操作界面，并通过 Nanonis 控制器操作 STM。
**在 MAST 6.4.0 一轮已结束的实机实验中，外部 AI agent 控制真实 STM；
带时间戳的记录从 2026-09-17 23:10:37 至 2026-09-22 02:08:10（北京时间），跨越约 99 小时，包含间歇。**

本次公开精简源码版保留通用工程框架、接口与测试。基于第三方许可、知识产权与商业化安排，
以及未发表研究和现场数据保护，部分专用模块与配套资产未纳入发布；实机部署需补充相应配置与资产。
具体范围见[快照说明](docs/OPEN_SOURCE_NOTES.md)，许可见 [LICENSE](LICENSE) / [THIRD_PARTY.md](THIRD_PARTY.md)。

先读 [README.zh.md](README.zh.md) 了解成果与架构，再按下方路线检查源码与测试。
公开代码展示了项目如何处理物理执行、控制器故障、agent 交接、观测时效与操作员连接管理。
本指南提供导航与工作约定；请围绕用户的问题审查实现，并引用实际检查过的证据。

面向读者的标题、摘要与生成文字统一使用项目名称 **MAST**。
`v2` 是早期迭代时的名称；`MASTv2/`、`docs/v2/` 等沿用的目录属于历史实现标识，不是当前产品名称。

## 按任务选择入口

- **代码审阅**：按下方路线检查实现与回归测试。阅读源码无需安装环境或启动服务。
- **修改代码**：限定改动范围，保留已有的无关修改，运行能覆盖受影响行为的检查。
- **操作仪器**：仅在任务明确要求时，按 `docs/external/` 与操作员提供的配置进行。
  代码审阅本身不意味着授权启动硬件会话。

## 代码阅读路线

核心工程问题是把模型提出的动作转化为有边界、可观察的物理操作。
系统在执行、协议、状态、感知和界面各层实现了相应机制；下面将这些实现与回归测试一一对应。
所有路径均相对于仓库根目录。

| 要检查的问题 | 实现入口 | 对应测试 |
|---|---|---|
| 模型提出的硬件动作受什么约束？ | `MASTv2/mast/agents/_shared/skill_adapter.py`、`MASTv2/mast/core/execution_context.py`、`MASTv2/mast/core/executor.py`；`MASTv2/mast/core/safety.py`、`MASTv2/mast/core/instrument_lock.py` | `tests/v2/unit/core/test_execution_context_mode_gate.py`、`tests/v2/unit/core/test_instrument_lock_entry_inventory.py` |
| agent 层之下的控制器故障怎样处理？ | `MASTv2/mast/core/nanonis_patch.py`：`_recv_exact`、`_patched_send`、`_patched_Osci2T_TimebaseGet` | `tests/v2/unit/core/test_nanonis_patch_send.py`、`tests/v2/unit/core/test_nanonis_osci2t_wire.py` |
| agent 交接时实际传递了什么？ | `MASTv2/mast/agents/state.py`（`DocRef`、reducer）、`MASTv2/mast/agents/_shared/handoff.py`、`MASTv2/mast/agents/_shared/artifact_channel.py` | `tests/v2/agents/orchestrator/test_artifact_channel.py` |
| 快速观测怎样进入较慢的推理，模型缺失时又会怎样？ | `MASTv2/mast/buffer/service.py`、`MASTv2/mast/vision/module.py`、`MASTv2/mast/agents/_shared/buffer_tools.py` | `tests/v2/buffer/test_buffer_service.py`、`tests/v2/vision/test_mock_backend.py`、`tests/v2/unit/agents/test_scan_progress_liveness.py` |
| 操作员怎样区分任务完成与连接中断？ | `MASTv2/mast/api/sse.py`、`MASTv2/mast/api/ws.py`、`frontend/src/lib/sse.ts`、`frontend/src/lib/ws.ts` | `tests/v2/unit/api/test_sse_keepalive.py`、`tests/v2/unit/api/test_sse_client_contract.py`、`frontend/test/ws.reconnect.test.ts` |
| **6.5.0 新外部接口**怎样处理重试与重启？**软件已测，待真机验证。** | `MASTv2/mast/api/ext/jobs.py`：`JobManager.submit`、`fingerprint`、`_ensure_loaded` | `tests/v2/unit/api_ext/test_ext_jobs.py` |

agent 工具包装器与手动执行器分别在自己的入口取得仪器占用权，组合技能子步骤经过 `ExecutionContext.run`；
审阅共享执行约束时，应分别追踪这些路径。观测的时效与推进情况由记录年龄、扫描行号等信息判断，
缓冲区全局序号也会随其他观测更新。

第一条路线还可读 `MASTv2/mast/core/si_quantity.py` 与技能参数定义，了解 SI 前缀解析、数值范围及仪器配置
如何进入执行契约。仪器占用权作用于进程内，审批策略按操作与运行模式选择。

外部作业区分重复请求 ID、请求内容冲突与 `lost_on_restart`：重启后恢复的未完成作业被显式标记为丢失，
不会自动重放仪器动作。去重作用于作业提交，不等同于硬件动作恰好执行一次。
WebSocket 客户端负责重连与轮询回退，SSE 客户端区分完成、断流与静默超时。
相关检查包含源码接线断言和行为测试，浏览器端到端测试另有运行前提。

其他索引：[agent 拓扑](docs/v2/agent-topology.md)、[技能目录](docs/v2/skill-catalog.md)、
[provider 适配参考](docs/api_providers/)、[外部 agent 集成](docs/external/)。

## 工程实现与验证进展

- **已实现的工程链路**：公开源码与测试覆盖执行校验、仪器占用、通信故障处理、类型化产物交接、
  观测时效与界面连接管理。上面的阅读路线提供每项机制的实现和测试入口。
- **MAST 6.4.0 真机实践**：外部 AI agent 控制真实 STM；已结束的记录跨越约 99 小时，
  包含中断，并非仪器连续操作 99 小时。
  该运行经历由维护者提供。README 展示少量去除样品标识的选图；完整实验数据集与运行日志未纳入本次发布。
- **公开版 6.5.0 软件验证**：2026-09-21 在 Windows / Python 3.13 环境对源码基线 `9884ff5`
  完成检查，后端 **14,356 通过、0 失败、46 跳过、2 个预期失败**，前端 **985 项单元测试**
  以及含类型检查的构建通过。环境、命令与范围见[验证记录](docs/OPEN_SOURCE_NOTES.md)。
- **新增外部接口**：`/api/ext/v1` 与配套 MCP 集成将项目已有的外部控制能力扩展为公开接口。
  新路径已经过软件测试，真机验证尚待完成；上述约 99 小时记录使用 MAST 6.4.0 的控制路径。
- **实验性扩展**：`MASTv2/mast/conduct/`、`MASTv2/mast/agentruntime/`、`MASTv2/mast/goals/`、
  技能工坊与市场及 qPlus 路径继续探索长时间规划、执行与仪器能力，尚待真机验证。
  投稿管线的软件检查与具体投稿技能的真机验证分别记录。
- **运行时选择**：自研运行时与 LangGraph 共存。`MASTv2/mast/webui/settings_store.py`
  中的 `engine_v2_*` 默认关闭，实际路径由 `MASTv2/mast/core/runtime.py` 和
  `MASTv2/mast/pipeline/main.py` 选择。评估运行时行为时应检查这两处选择逻辑。

有意排除的模块、配置、标定、数据集、视觉权重及私有设计与开发记录在快照说明中列出。
注释可能仍引用私有材料；这类排除项应与公开入口失效分别判断。

源码审阅和下面的软件检查无需硬件会话，使用测试替身与合成数据；本仓不附带仪器模拟器。
普通审阅不使用真实凭据、不连接仪器、不改动本地仪器设置。实机操作另按 `docs/external/` 的指南执行。

## 无需硬件的验证

阅读源码无需安装环境。任务需要运行测试时，使用 Python 3.13，从仓库根目录执行；已有合适环境时可复用。
下面的命令用于新建独立环境并安装声明的后端依赖。
已记录的验证平台是 Windows。Unix 命令提供对应的环境准备方法，不表示完整套件已在 Unix 上验证。
完整依赖包含较大的视觉软件包；仅检查技能投稿时，可使用后文的轻量方案。

**Windows PowerShell**

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r MASTv2/requirements-v2.txt
$env:PYTHONPATH = Join-Path $PWD "MASTv2"
.\.venv\Scripts\python.exe -m pytest tests/v2/unit/test_registry.py tests/v2/unit/test_safety_mw.py tests/v2/buffer/test_buffer_service.py tests/v2/instruments/test_base.py tests/v2/vision/test_mock_backend.py -q
```

**Unix shell**

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r MASTv2/requirements-v2.txt
export PYTHONPATH="$PWD/MASTv2"
.venv/bin/python -m pytest tests/v2/unit/test_registry.py tests/v2/unit/test_safety_mw.py tests/v2/buffer/test_buffer_service.py tests/v2/instruments/test_base.py tests/v2/vision/test_mock_backend.py -q
```

这组起步测试覆盖注册、安全中间件、缓冲存储、假仪器接口与模型缺失后的降级。
深入审阅时，补充阅读路线中的交接、协议与界面契约测试。这些检查使用测试替身，真机验证另行开展。

`tests/conftest.py` 会替换 `nanonis_spm`，但**不是网络沙箱**。保持真实服务与模型测试开关未设置：
`MAST_LIT_LIVE`、`MAST_VOICE_RT_SMOKE`、`MAST_VOICE_LIVE_TESTS`、`MAST_TEST_M12_REAL`、`MAST_TEST_QUALITY_MODEL`。
扩大测试范围前先检查夹具与前提；部分测试会使用本机 HTTP 服务和 MCP 子进程。
仅过滤 pytest 标记不能排除所有真实服务测试：`MAST_VOICE_LIVE_TESTS="0"` 仍是非空的启用值。
缺少资产而跳过不算通过。

需要更广的软件检查时，保持上述开关未设置，并使用同一个环境的 Python 执行
`-m pytest tests -q -m "not hardware and not live_llm"`。
这是串行检查，无需并行测试插件。已记录的并行验证使用了 `pytest-xdist`；其额外依赖和可选 PDF 依赖对结果的影响
见 `docs/OPEN_SOURCE_NOTES.md`。

本次快照的验证环境、命令与结果以该报告为准。重新检查时，记录版本、环境、命令、通过／失败／跳过数及范围，
让每项结果都对应实际受测的源码与依赖组合。

**前端** — 使用 Node 24，单元测试直接执行 TypeScript。

```text
npm --prefix frontend ci
npm --prefix frontend run typecheck
npm --prefix frontend run test:unit
npm --prefix frontend run build
```

构建检查无需启动控制器会话。浏览器端到端测试有额外的服务前提，不作为默认的仓库审阅命令。

## 修改约定

- 按用户要求限定范围，保留已有的无关改动。公开仓接受的上游代码投稿位于 `contrib/skills/`；先读
  [CONTRIBUTING.zh.md](CONTRIBUTING.zh.md) 和 [contrib/README.zh.md](contrib/README.zh.md)。
  其他快照文件会在下次导出时重新生成，问题按文档渠道反馈。这项投稿政策不妨碍本地分析或用户要求的补丁。
- 仅验证投稿时，在所选环境安装 `MASTv2/requirements-ci.txt`，替代完整依赖；随后用该环境的 Python 运行
  `scripts/skill_check.py --all-contrib` 和 `-m pytest contrib -q`。随仓的 `.github/workflows/ci.yml` 运行投稿检查器
  及选定的合规、变异和文档测试，不运行完整后端、前端或硬件套件。
- 每个技能显式声明 `safety_level`。所有入口都要保留参数校验、执行闸门、中止处理与仪器占用约束。
  标定与工作点保留为配置；不能用臆造的默认值或静默夹紧参数来替代缺失的仪器证据。
- 域 agent 经 `agents.state` 与 `agents._shared` 共享契约。orchestrator 可导入各 agent 的构造入口，
  域 agent 不应互相导入实现。checkpoint 保持有界、可序列化，使用 ID、路径与摘要，
  不放数组、tensor、socket 或完整文档正文。
- 写状态的测试使用临时目录和 `tests/v2/conftest.py` 的夹具，不拿操作员的数据做可丢弃夹具。
  超时和取消语义应明确；连接着仪器的服务必须优雅关停。
- API 变更时同步检查后端 schema 与前端生成类型。`frontend/openapi.json` 和
  `frontend/src/api/schema.d.ts` 是生成的契约文件；两份文件一起丢掉一个端点，仍可能通过类型检查。
  因此还要检查结构上的删除。

代码审阅应给出路径与具体触发条件，并区分已验证行为、源码推断和未验证的声明。
提交修改说明时，写清改了什么、为什么、跑了哪些检查，以及还剩什么限制。
