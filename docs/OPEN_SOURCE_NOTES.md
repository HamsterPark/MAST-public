# 公开版本说明：保留能力、发布范围与验证记录

本仓是 **MAST 6.5.0** 在 2026-09-21 导出的工作树审查快照。
它是经过清理与本地软件验证的**公开精简源码版**，保留多智能体编排、通用仪器技能与执行约束、
感知和数据处理框架、React 操作界面、外部 agent 接口及相应测试。
这些内容支持从高层任务协作一直读到控制器协议和界面通信的实现；下文记录已有成果、公开取舍及验证方法。

## 真机实践与版本范围

**在 MAST 6.4.0 一轮已结束的实机实验中，外部 AI agent 控制真实 STM，带时间戳的记录从 2026-09-17 23:10:37 至 2026-09-22 02:08:10（北京时间），跨越约 99 小时。**
这是项目已有的长时间仪器控制经历，涉及当时的外部控制路径。该经历由维护者提供；README 仅展示少量去除样品标识的选图，完整实验数据集和运行日志不随本快照发布。

MAST 6.5.0 在已有实践基础上新增 `/api/ext/v1` 及配套 MCP 集成。新接口已有软件测试，尚待真机验证。
项目的真机实践与公开版的软件验证构成两类记录：约 99 小时记录对应 MAST 6.4.0 的实际部署，
下文的测试数字对应本次公开源码；新增接口的真机验证是后续工作。

## 公开范围与取舍

本次采用有选择的公开方式，综合考虑以下因素：

- **第三方版权与许可**：第三方代码、模型、文献、厂商文档和图像各有其版权与许可边界，未统一纳入本次发布。
  具体收录与排除情况见 [`THIRD_PARTY.md`](../THIRD_PARTY.md)。
- **知识产权与商业安排**：为保留后续产品化与商业化空间，部分专用模块、知识资产和交付材料不在本次公开范围内。
- **未发表研究与现场数据保护**：除 README 中少量去除样品标识的选图外，未发表样品的研究资料、完整实验设计与记录、原始数据、仪器标定、站点配置和凭据不随仓提供。

删减既涉及数据与配置，也涉及部分实现模块和工具入口。保留代码中依赖这些内容的功能，可能降级、返回不可用，
或需要使用者补充相应资产与配置；具体处理见下表和公开版补丁清单。本快照因此不能直接复现完整系统的功能与部署环境。

本说明解释的是公开范围，具体许可见 [`LICENSE`](../LICENSE) 及相关第三方声明。
对公开代码的判断仍应以本仓实现与测试为依据；未公开的能力不作为公开版本已验证的证据。

## 一、删减类别与运行影响

| 类别 | 未纳入内容 | 保留代码的处理方式 |
|---|---|---|
| **预存的机器参数** | 标定曲线与阈值、站点运行参数、Z 控制器预设、扫描档位表、锻针规程、光学台与 Nanonis 脚本配置、`config/overrides/` 及其历史、现场调试记录 | 读取这些文件的代码保留；文件缺席时按各自的缺省或「未标定」处理 |
| **知识库** | 十个样品类型知识模块、实验设计 / 故障诊断 / 硬件 profile / 安全参考值 / 噪声知识、词汇表、文献先验、材料覆盖索引、chunk 检索 | `knowledge/lookups.py` 的材料注册表置空（所有查询返回「未知材料」）；其余调用点本来就是惰性 import 且带兜底 |
| **文献库** | **保留通用实现，初始库为空。** 未纳入随仓数据：语料派生的材料覆盖索引与文献先验（`material_coverage.json`、`literature_priors.json`）。建索引的流水线 `scripts/openalex_pipeline/` 保留，语料本身（OpenAlex，CC0）自己拉 | 数据缺失时按实现降级：语义检索报「索引缺失」并指向流水线、语料计数为 0、库注册表不在就建一个空的全局库、先验文件不在返回空。PDF 取全文与 OCR 用的 PyMuPDF（AGPL）移到可选的 `MASTv2/requirements-pdf.txt` |
| **移植自已发表文献的技能** | `skills/paper/` 共 21 个（DeepSPM、Scanbot、gpSTS、AtomAI、ASD-STM 等的移植）、其基准脚本与报告、v1 的 RL 针尖策略 | 数据处理 agent 里桥接它们的 18 个工具一并移除；两处引用其常量的地方就地保留常量 |
| **开发过程记录** | 工作计划、开发计划、设计稿与 RFC、架构决策记录、会话交接与结果报告、审查与修复记录、已知问题清单、发布验证声明、提交历史、私有开发工作流中的编码助手指令与钩子、报告与演示页、由验收记录生成的用户手册 | 保留面向公开读者的说明、provider 参考、agent 拓扑图与技能目录；代码注释里引用 `docs/v2/design/…` 的地方指的是不随仓的设计稿 |

公开代码审阅指南 [`AGENTS.md`](../AGENTS.md) / [`AGENTS.zh.md`](../AGENTS.zh.md)、
引用这套共用指引的 [`CLAUDE.md`](../CLAUDE.md)，以及 [`docs/external/`](external/) 外部 agent 使用指南随仓保留。
它们面向公开快照的读者与使用者，不包含私有现场的开发指令。

其他排除项：凭据与站点坐标（地址、账户、私钥名、证书）、第三方版权材料（Nanonis 手册与协议文档、他人的 STM 图像及派生图）、
v1 时代的文档、调试脚本与设计交付物。

**可识别未发表样品的研究资料**不在本仓：某一样品 campaign 的实验设计及其 conduct 模板、
完整原始实测数据与详细参数，以及 prompt 示例里引用的真实研究假设（改成了不指向具体样品的写法）。
README 中少量去除样品标识的实测选图是这一范围说明的例外，不包含原始数据集。
**数据图库保留通用代码，初始为空**：浏览、标记、索引与通用出图功能随仓，真实采集文件、系列清单、
图库索引、标注和图表不随仓；涉及本站实验的注释和夹具改为技术说明或合成数据，标定量须由使用者提供。
预处理仅提供 `generic-uncommissioned` 示例 profile，不携带任何样品批次的标定声明；起伏标定量仍为空。
真实实验的数组夹具不随仓。三份 Nanonis 格式夹具由随仓的 `tests/v2/fixtures/nanonis/generate.py`
从独立公式生成，内容可逐字节复现；其中日期、坐标和曲线均为合成数据，仅用于解析与分析测试。
线级分析另有合成数据测试。
`vision/` 里以 `sf09` 命名的模型与语料代号指的是一批不随仓发布的真实帧语料，代码只保留消费端。

## 二、公开版补丁（保留代码的适配改动）

每一条都在私有仓的 `scripts/release/public_patches.py` 里有锚点；锚点找不到导出就失败，不会静默跳过。

| 文件 | 改动 |
|---|---|
| `MASTv2/mast/knowledge/lookups.py` | 样品类型知识模块不随仓发布，注册表留空 |
| `MASTv2/mast/knowledge/__init__.py` | chunk_registry / retriever 随知识库一起移除 |
| `MASTv2/mast/agents/data_processing/tools.py` | 移除桥接 skills/paper 的 18 个工具 |
| `MASTv2/mast/skills/composite/prescan_check.py` | line_check 技能不随仓发布，常量就地保留 |
| `MASTv2/mast/vision/corrugation_gate.py` | line_check 技能不随仓发布，常量就地保留 |
| `MASTv2/mast/conduct/templates/__init__.py` | 样品 campaign 模板不随仓发布 |
| `MASTv2/mast/agents/research_director/prompts.py` | 示例假设去掉具体样品与其研究问题 |
| `MASTv2/mast/agents/_shared/campaign_tools.py` | 示例假设去掉具体样品与其研究问题 |
| `MASTv2/mast/api/tool_narration.py` | 解说表去掉随 skills/paper 移除的 compose_montage |
| `tests/v2/unit/conduct/test_api_conducts.py` | 两条依赖 campaign 模板 coord_epoch 槽位的测试 |
| `tests/v2/unit/conduct/test_compiler.py` | 一条依赖 campaign 模板 coord_epoch 槽位的测试 |
| `installer/mast2_setup.iss` | 安装包的项目 URL 误写成了 anthropics 组织 |
| `MASTv2/mast/vision/_legacy_wrapper.py` | training/（移植自论文的模型结构）不随仓，legacy 后端明确报未随仓 |
| `docs/v2/skill-catalog.md` | 技能目录去掉模块不随仓的技能（论文移植包整包被排除），总数与分级计数随之改正 |
| `MASTv2/requirements-v2.txt` | PyMuPDF 移到 requirements-pdf.txt（AGPL） |
| `tests/v2/agents/literature/test_graph_smoke.py` | PyMuPDF 变成可选依赖后，夹具缺它要 skip 不要报错 |
| `tests/v2/agents/literature/test_lit_tools_findings.py` | PyMuPDF 变成可选依赖后，夹具缺它要 skip 不要报错 |
| `MASTv2/mast/gallery/figures/stitch.py` | 移除站点实测 κ；只有调用方明确给值时才估算倍率 |
| `frontend/src/components/gallery/figures/StitchControls.tsx` | κ 输入默认空值；留空仍可拼接，明确给值才请求理论倍率 |
| `MASTv2/mast/gallery/figures/series.py` | 叠加视野缺省采用输入首帧宽度，不沿用实验出图尺寸 |
| `frontend/src/components/gallery/figures/SeriesFigureActions.tsx` | 叠加表单缺省取输入帧宽度 |
| `MASTv2/mast/gallery/figures/series.py` | 相关窗与搜索域限制在实际视野内，小视野也可默认叠加 |
| `tests/v2/unit/gallery/figures/test_figures_stitch.py` | 默认拼接不携带任何实测 κ；显式参数测试仍保留 |
| `mast2.spec` | 打包预导入及隐藏导入列表不再点名已移除的论文技能包 |
| `MASTv2/mast/__init__.py` | 构建元信息不随仓时，公开快照的版本回退为 6.5.0 |
| `MASTv2/mast/agents/_shared/artifacts.py` | 工具产物流图去掉不随仓的论文移植工具映射，保留真实性断言 |
| `tests/v2/unit/agents/test_prompt_provenance_gate.py` | 溯源闸门使用运行时拼出的合成禁止标记，清洗不会消掉变异输入 |
| `tests/v2/unit/agents/test_instrument_control_handoff.py` | 公开交接测试验证确定性派发与维护说明的位置，不依赖私有提交 |
| `tests/v2/unit/test_material_vocab_consistency.py` | 空材料库仍返回显式空候选、接受自由文本且保留显式类型 |
| `MASTv2/mast/vision/scan_prep_thresholds.py` | 移除样品标定 profile，内建阈值改成明确未标定的合成示例 |
| `MASTv2/mast/skills/builtins/scan_prep.py` | 技能参数说明明确默认 profile 未标定，须由使用者另行验证 |
| `tests/v2/unit/vision/test_scan_prep.py` | 合成图测试验证未标定声明及其完整传递，不再依赖私有样品 |
| `tests/v2/unit/skills/builtins/test_scan_prep_skills.py` | 技能与报告测试确认默认 profile 身份、未标定声明及出处保持一致 |
| `MASTv2/mast/skills/builtins/deltaf_curve.py` | 力谱采集的十种命令显式使用字面动词，使静态安全检查可见 |
| `tests/v2/unit/io/test_exp_map_analysis_zero_footprint.py` | 完整零足迹回归集纳入只读谱文件的 FindSpectralPeaks，保留集合相等与行为断言 |

## 三、脱敏

全部文本文件过同一张规则表（私有仓 `scripts/release/scrub_rules.py`）：开发机路径、站点机器名与地址、SSH 账户与私钥名、
明文 key，注释和文档里「某人某天说过某句话」式的现场归属与交互记录，
以及指向开发过程的标记（现场反馈编号、取证编号、审查日期、审查发现编号）。真名作为作者署名保留。

通用技术推理保留；本站未经脱敏的实验记录、样品身份与仪器标定值不作为公开叙事或测试夹具，README 中去除样品标识的选图除外。

导出后全树再过一遍卫生扫描（判据表复用 `tests/v2/unit/test_release_hygiene.py`，外加第三方图像编号、私钥块等几条），
命中必须为零，只允许写明理由的放行：讲 CGNAT 网段本身的 `net/tailscale.py`、用 `100.101.x.x` / `10.0.0.x` 作占位的几个测试。

## 四、import 闭包与测试

移除子系统之后，剩余代码里指向被移除模块的 import 分两档：

- **无兜底**：生产代码里必须为零（靠上面的补丁表）；测试文件里有的话，该测试文件整个不收录。
- **有兜底**（`try / except`）：保留，对应功能在缺席时降级。清单：

- `MASTv2/mast/__init__.py:32` ← mast._buildinfo
- `MASTv2/mast/__init__.py:33` ← mast._buildinfo
- `MASTv2/mast/api/version.py:13` ← mast._buildinfo
- `MASTv2/mast/llm/quickask.py:350` ← mast.knowledge.fault_diagnosis
- `MASTv2/mast/llm/quickask.py:360` ← mast.knowledge.stm_noise
- `MASTv2/mast/llm/quickask.py:395` ← mast.knowledge.reference_index
- `MASTv2/mast/llm/quickask.py:409` ← mast.knowledge.reference_index
- `MASTv2/mast/update/defaults.py:17` ← mast.update._defaults
- `MASTv2/mast/update/defaults.py:23` ← mast.update._defaults
- `MASTv2/mast/vision/_legacy_wrapper.py:71` ← mast.training.models, mast.training.models.architectures
- `MASTv2/mast/api/routes/codex_reference.py:249` ← mast.knowledge.experiment_design
- `MASTv2/mast/api/routes/codex_reference.py:276` ← mast.knowledge.hardware_profile
- `MASTv2/mast/api/routes/codex_reference.py:314` ← mast.knowledge.safety_constraints
- `MASTv2/mast/api/routes/settings_admin_write.py:155` ← mast.knowledge.fault_diagnosis
- `MASTv2/mast/api/routes/settings_admin_write.py:159` ← mast.knowledge.experiment_design
- `MASTv2/mast/api/routes/settings_admin_write.py:169` ← mast.knowledge.hardware_profile
- `MASTv2/mast/api/routes/settings_admin_write.py:180` ← mast.knowledge.image_databases
- `MASTv2/mast/agents/_shared/meta_tools.py:705` ← mast.knowledge.fault_diagnosis
- `MASTv2/mast/agents/_shared/meta_tools.py:719` ← mast.knowledge.stm_noise
- `MASTv2/mast/agents/_shared/meta_tools.py:767` ← mast.knowledge.reference_index
- `MASTv2/mast/agents/_shared/meta_tools.py:781` ← mast.knowledge.reference_index

因 import 了被移除模块而未收录的测试文件：

- `tests/v2/skills/test_paper_numerics.py`
- `tests/v2/skills/test_tip_verdict_skills_safe.py`
- `tests/v2/unit/agents/test_dp_tool_noop_honesty.py`
- `tests/v2/unit/api/test_skills_meta.py`
- `tests/v2/unit/core/test_operating_mode.py`
- `tests/v2/unit/core/test_vacuum_interlock.py`
- `tests/v2/unit/data/test_scan_format_parity.py`
- `tests/v2/unit/skills/builtins/test_load_scan_frame_from_file.py`
- `tests/v2/unit/skills/builtins/test_step_height.py`
- `tests/v2/unit/skills/composite/test_prescan_twin_reads_the_same_metric.py`
- `tests/v2/unit/skills/composite/test_verify_cannot_pass_a_dead_frame.py`
- `tests/v2/unit/skills/test_frame_validity_guard.py`
- `tests/v2/unit/test_misc_safety_findings.py`
- `tests/v2/unit/vision/test_corrugation_gate.py`

以下结果针对本次公开快照；测试环境与范围随结果一并记录。

2026-09-21，在 Windows / Python 3.13 环境中验证本次 MAST 6.5.0 公开快照，未连接仪器。

整树受测源码基线为 `9884ff5`。后续文档、文档生成模板及 agent 提示文案的修订另行检查；本记录不替代修改后源码的整树复测。

- 后端整树：**14,356 通过、0 失败、46 跳过、2 个预期失败**。命令为 `python -m pytest tests -q -n 8 --dist loadfile -m "not hardware and not live_llm"`，`PYTHONPATH` 指向本快照的 `MASTv2`。
- 前端：`npm run build`（含类型检查）通过，`npm run test:unit` **985 / 985 通过**。
- OpenAPI 与公开后端一致：302 条路径、577 个 schema；前端生成类型已同步。
- 全树 Python 语法检查、import 闭包及公开卫生扫描通过；三份 Nanonis 格式夹具可从随仓生成器逐字节复现。

跳过与预期失败按测试自身标记统计，不计为通过。本结果覆盖无需硬件和真实 LLM 的测试范围，不代表真机验证。

### 公开读者文档修订检查

2026-09-21，基于 `9884ff5` 的公开候选工作树完成文档、插件配置说明及 agent 提示文案修订。
Python 文件改动经 AST 比较确认仅涉及文本常量与注释，插件 manifest 仅修改两处字段说明。

- 双语文档结构、指南镜像、API 参考生成、外部概览与 MCP 集成检查：**92 通过、1 跳过**。
  使用所选 Python 3.13 环境运行 `-m pytest tests/v2/unit/docs/test_external_docs_bilingual.py tests/v2/unit/docs/test_external_docs_sync.py tests/v2/unit/api_ext/test_ext_api_reference.py tests/v2/unit/api_ext/test_ext_overview.py tests/v2/unit/integrations -q`。
- 跳过项为嵌入式 Python 的启动测试；公开源码包不含 `MASTv2/pyruntime/python.exe`。
- 最后一处工具说明调整后，另复查工具清单与 schema 测试，**1 项通过**。
- API 文档生成一致性、18 份插件指南镜像、公开文档本地链接与文本扫描通过。

检查使用测试替身、临时文件及本地测试服务，未连接仪器或真实模型服务；未重复运行整树后端与前端套件。

随后将当前项目名称统一为 **MAST**，同步文档、agent 自称与服务展示文字；路径、配置键和兼容标识保持原样。
命名修订后的双语结构与指南镜像检查 **8 项通过**，Python 改动仍仅涉及文字与注释，前端包仅更新描述字段。

### 复现验证时的前提

上述并行命令中的 `-n 8 --dist loadfile` 由 `pytest-xdist` 提供，它不在主依赖清单中。
如需同样的并行方式，先用所选虚拟环境的 Python 执行 `-m pip install pytest-xdist`。
无需并行时，省略这两个参数即可；默认串行命令见根目录 AGENTS。

运行前保持 `MAST_LIT_LIVE`、`MAST_VOICE_RT_SMOKE`、`MAST_VOICE_LIVE_TESTS`、
`MAST_TEST_M12_REAL` 和 `MAST_TEST_QUALITY_MODEL` 未设置。部分真实服务测试由环境变量控制，
单靠 `-m "not hardware and not live_llm"` 不能禁用它们；`MAST_VOICE_LIVE_TESTS="0"` 也会启用对应测试。
这些设置和厂商库 mock 都不构成网络隔离。

通过与跳过数还受依赖组合影响。例如，可选的 `MASTv2/requirements-pdf.txt` 决定部分 PDF 测试能否运行。
历史记录没有附带完整的锁定环境，因此仅安装主依赖不能保证复现完全相同的计数。
重新运行时应记录具体依赖、可选组件及实际结果。GitHub 的投稿 CI 覆盖范围见 `.github/workflows/ci.yml`，
不等同于上述本地整树验证。

## 五、快照如何生成

公开版由自动化导出流程生成：按清单排除内容、应用带锚点的公开版补丁、执行文本脱敏，
再检查 import 闭包与发布内容规则。补丁锚点失配时导出失败；每次导出生成 `EXPORT_REPORT.md`，
记录逐规则命中数、补丁清单、闭包检查、扫描结果与体积。

导出机制位于开发仓的 `scripts/release/`。脚本及报告留在开发侧，避免将脱敏规则中的原始敏感字面量一并发布；
公开侧保留删减清单、相应行为说明与软件验证记录。

## 六、读代码时要知道的

- 代码注释里引用的 `docs/v2/design/*.md`、`HANDOFF`、`PROGRESS` 等文档都不随仓；公开说明保留通用机制，未附带实验记录或私有设计稿。
- `config/composite_skills/` 保留公开的复合技能声明式规格；具体清单以目录内容为准。
- 代码里的通用算法与合成测试不代表任何仪器的标定结果；使用者必须自行提供仪器配置、数据与标定。
