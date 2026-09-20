# 社区技能（`contrib/`）

[English](README.md)

社区技能是扩展 MAST 实验能力的公开入口，支持纯数据分析、仪器读写与多步组合流程。
本目录提供投稿格式、检查规则和本地安装方法；人编写与编码 agent 辅助编写的技能采用相同标准，
在维护者完成真机验证前保留于社区层。

## 两级制

| 级别 | 位置 | 投稿进入该层的要求 |
|---|---|---|
| **社区** | `contrib/skills/<Name>/` | 过了 `scripts/skill_check.py` 与 CI。**没有经过维护者真机验证。** 这里的东西不会被自动加载：每一个技能都要使用者自己安装、显式启用。 |
| **官方** | `MASTv2/mast/skills/builtins/`、`config/composite_skills/` | 社区技能转入官方树前，需经过维护者真机验证。 |

社区技能经维护者真机验证之后**毕业**：移出 `contrib/`、进入官方树，**主版本号 +1**（从此按另一套标准
要求它），[`GRADUATED.md`](GRADUATED.md) 留署名。

这一毕业要求不代表现有 builtin 或复合技能中的每项功能都已通过真机验证；
应按各项功能已说明的验证状态判断。

**优先用 CompositeSpec JSON 组合现有官方技能。** 它复用既有执行逻辑，
安全级与能力标签从子步**继承**（声明只能更严、不能更松），并与工作流编辑器和 agent 使用同一套组合校验。
需要现有技能无法表达的能力时，再提交 Python 实现。

## 目录约定

```
contrib/skills/<Name>/
├── manifest.json
├── skill.py | spec.json
├── test_<name>.py
└── README.md
```

- `manifest.json` —— 必填，见下。
- `skill.py`（Python 技能）**或** `spec.json`（组合 spec）—— 二者恰好其一。
- `test_<name>.py` —— 至少一个测试；文件名在整个 `contrib/` 里唯一。
- `README.md` —— 可选。

`<Name>` 就是技能名：一个普通标识符（`^[A-Za-z][A-Za-z0-9_]*$`），与类名、`metadata().name`（Python）
或 `"name"`（spec）完全一致。安装时只拷 `skill.py` 一个文件，所以它不能 import 同目录或 `contrib`
下的其它模块。

两个范例：

- [`EstimateScanDuration`](skills/EstimateScanDuration/) —— 纯计算的 Python 技能。
- [`ReadJunctionState`](skills/ReadJunctionState/) —— 四个官方只读步骤组成的组合 spec。

## `manifest.json`

| 键 | 必填 | 取值 |
|---|---|---|
| `schema` | 是 | `1` |
| `name` | 是 | 技能名（= 目录名） |
| `kind` | 是 | `"python"` 或 `"spec"` |
| `version` | 是 | `"X.Y.Z"`；Python 技能必须等于 `metadata().version` |
| `summary` | 是 | 一句英文 |
| `summary_zh` | 否 | 一句中文 |
| `safety_level` | 是 | `"auto"` / `"confirm"` / `"dangerous"`；spec 写**生效**安全级（从子步继承来的那个） |
| `footprint` | 是 | `"pure-analysis"` / `"hardware-read-only"` / `"hardware-write"`；必须与检查器从代码推出的一致 |
| `authors` | 是 | `[{"name": "...", "github": "..."}]`，至少一项带 `name` |
| `license` | 是 | `"MIT"` |
| `verification` | 是 | `"unit-tested"`（最低）或 `"contributor-hardware"` |
| `hardware_notes` | `contributor-hardware` 时必填 | 在什么仪器 / 控制器版本上、跑了什么、看到了什么 |
| `tests` | 是 | 本目录里的测试文件 |
| `policy` | 是 | `{"original_work": true, "no_machine_specific_defaults": true, "accepts_inbound_license": true}` |

## 检查一份投稿

从仓库根目录运行。使用 Python 3.13，并在所选环境安装 `MASTv2/requirements-ci.txt`；
下面的 `python` 指该环境的解释器。Windows 与 Unix 的环境准备见
[审阅与开发指南](../AGENTS.zh.md#无需硬件的验证)。以下为 Unix shell 写法。

```bash
export PYTHONPATH="$PWD/MASTv2"
python scripts/skill_check.py contrib/skills/<Name>
python scripts/skill_check.py --all-contrib
python scripts/skill_check.py --all-contrib --json
python -m pytest contrib -q
```

四条命令依次是：检查一份投稿、检查 `contrib/skills/` 下全部、以机器可读的 JSON 输出同样的结果、跑投稿
自带的测试。退出码 `0` = 全部通过，`1` = 至少一条 `FAIL`。`SKIPPED` 行表示那一项**没有检查**（比如没装
`nanonis_spm`，就没法核对 Nanonis 命令名）—— 跳过不等于通过。

| 代号 | 判据 |
|---|---|
| S01 | `safety_level` 显式写出（缺省会静默落成 CONFIRM） |
| S02 | 恰好一个 `BaseSkill` 子类，有 `metadata()` 与 `execute()`；`metadata()` 直接构造 `SkillMetadata(...)` |
| S03 | AST 拒绝名单 —— 与自定义技能加载器在加载时重跑的是同一个检查器 |
| S04 | 名字是普通标识符，且 安装文件名 = 类名 = `metadata().name` |
| S05 | 名字没被已注册技能或内置模板占用 |
| P01 | 每个带量纲参数（写了 `unit` 的）都有 `min_value` 与 `max_value`；报告会写出它落在全局安全包络的哪一行 |
| P02 | 量程整段远离 1 的参数必须写成带 SI 前缀的形式（`'50n'`），所以描述里不能教指数写法（`5e-8`） |
| P03 | 每个前置条件都在已知词表里 |
| P04 | 会打偏压脉冲、会修针的技能，声明了对应的能力标签（`bias_pulse` / `tip_shaping`）—— SAFE / SEMI 操作模式闸只认标签 |
| V01 | 每个 `safe_call` 的动词都是字符串字面量 |
| V02 | 每个动词都在 `nanonis_spm`（或 MAST 的补丁层）里存在 |
| V03 | 硬件足迹能分类，且与 `category`、manifest 一致 |
| V04 | 写进去的值有回读（启发式，只提醒） |
| R01 | `SkillResult(...)` 的每个关键字都是真字段 |
| R02 | 不把 Nanonis 的三段回包原样当数据交出去 |
| X01 | 冒烟执行：用一个回包形状正确的假仪器调 `execute()`，要返回 `SkillResult`（S03 通过后才跑） |
| C01 | spec 过设计期全量校验，且每一步都是官方技能 |
| M01 | `manifest.json` 结构 |
| M02 | manifest 与代码、目录一致 |
| M03 | 政策：原创、默认值里没有某台仪器的参数、接受入站许可；不收已发表论文的移植 |
| E01 | 检查器自己的运行环境（注册表、依赖） |

关于 V03：仪器只能经 `execute()` 收到的那个执行上下文碰到。把上下文交给检查器看不见的代码、把它存起来、
或者 import 一条绕开它的路（厂商库、MAST 的连接层或运行时、串口或 HTTP 客户端）的技能，足迹无法分类，
判为不通过。

检查器读的是本机的可选硬件模块开关与管理员覆写；CI 跑在出厂配置下，所有可选硬件模块都是关的。
所以子步属于可选模块的 spec，在这一级会被拒。

## 在自己的 MAST 上装一个社区技能

`contrib/` 里的东西，不安装、不启用就不会被加载。先读代码：启用一个 Python 技能，等于以 MAST 进程的
全部权限运行它 —— 拒绝名单是防呆，不是沙箱。

### Python 技能

1. 把 `contrib/skills/<Name>/skill.py` 拷到 `config/custom_skills/<Name>.py`（MAST 数据根下）。
2. 把名字加进 `config/custom_skills/enabled.json`：`{"enabled": ["<Name>"]}`。
3. 重启 MAST，在技能列表里确认 `<Name>` 出现了。

如果技能未出现在列表中，检查加载日志及以下条件：

- **拷进去不等于启用。** 没列在 `enabled.json` 里的文件永远不会被执行。
- **启用键是文件名。** `enabled.json` 里写的是文件名（不带 `.py`），注册用的却是 `metadata().name`。
  文件必须恰好叫 `<Name>.py`。
- **一个文件一个技能。** 只注册文件里的第一个 `BaseSkill` 子类。
- **加载时重跑拒绝名单。** 过不了就跳过。

### 组合 spec

在工作流编辑器里导入，或者发给正在运行的 MAST：

```bash
curl -X POST http://127.0.0.1:7862/api/composites \
     -H "Content-Type: application/json" \
     -d "{\"spec\": $(cat contrib/skills/<Name>/spec.json)}"
```

（7862 是启动器的缺省端口；开发时 `python -m mast` 用它自己配置的端口。）
**投稿的 spec 只能引用官方技能。** MAST 启动时按这个顺序注册技能：内置 → 已存的 spec → agent 工具 →
自定义技能 → 覆盖层。spec 注册的那一刻，自定义技能与 agent 工具都还不存在，引用了它们的 spec 会被拒绝
注册，并在日志中记录原因。

## 投稿

规则（入站许可、DCO 签署、不收什么）见 [`CONTRIBUTING.zh.md`](../CONTRIBUTING.zh.md)；技术细节见
[`docs/external/zh/05-authoring-skills.md`](../docs/external/zh/05-authoring-skills.md) 与
[`docs/external/zh/06-contributing-skills.md`](../docs/external/zh/06-contributing-skills.md)。
