# MAST 外部 agent API v1 参考

> 本页由 `scripts/gen_ext_api_reference.py` 从 API 自己的 OpenAPI 生成，请勿手改。

基础地址：`http://127.0.0.1:7862/api/ext/v1`（远程见 [快速上手](01-quickstart.md)）。机器可读的契约在 `GET /openapi.json`，交互式文档在 `/docs`。v1 之内只增不减。

## 通用约定

| 请求头 | 含义 |
|---|---|
| `X-MAST-Actor` | 你的名字（可选）。清洗成小写的 `[\w.-]`，最长 48；空 = `anonymous`。它只用于**归属**：动作、笔记、请求、组合技能都署 `ext:<名字>`。它不是认证，也永远不会是拒绝请求的理由。 |
| `X-MAST-Session` | 会话名（可选，最长 64）。实验记录里的署名变成 `ext:<名字>/<会话>`，用来区分同一个名字的两次接入。 |

认证与主 API 相同：本机回环不需要；局域网模式是 HTTP Basic + TLS。每个响应都带 `X-MAST-Ext-Version: 1`。

外部面只给程序化客户端（MCP server、脚本、`curl`）用：**每个 POST 都必须带 `Content-Type: application/json` 和一个 JSON 请求体**（没有内容就发 `{}`，`/estop` 与 `/jobs/{job_id}/cancel` 也一样），浏览器发起的跨站请求一律拒绝。

除 `/data/file` 与 `/data/frame` 外，响应都是 JSON。错误体的形状固定，`detail` 是给人读的说明（目前是中文）：

```json
{"error": "request_id_conflict", "detail": "...", "existing_job_id": "j_0123456789ab"}
```

| 状态码 | 含义 |
|---|---|
| 200 / 202 | 成功；202 = 新作业已受理。**技能自身失败也是 200**：看作业的 `state` 与 `result.success`。 |
| 403 | 路径不在允许的根目录之内，或扩展名不在白名单里；或 `cross_origin_refused`：浏览器发起的跨站请求（`Origin` 不是本服务、`Sec-Fetch-Site` 是跨站，或经陌生主机名访问 —— DNS 重绑定的形状）。 |
| 404 | 资源不存在（技能、作业、实验、样品、请求、文件、通道、路径）。 |
| 409 | 冲突：`request_id` 已用于另一份内容；仪器正被占用时切作用域。 |
| 415 | `unsupported_media_type`：写请求没有带 `application/json`。 |
| 422 | 请求不合法；技能在本机被关闭。 |
| 429 | 同时在跑的作业达到上限。 |
| 503 | 子系统未接线（带 `missing`），或服务正在关停。 |

## 端点一览

| 方法 | 路径 | 作用 |
|---|---|---|
| GET | `/health` | 外部面自己的健康与接线 |
| GET | `/guide` | 面向 agent 的铁律 |
| GET | `/status` | 快速状态（零硬件 I/O） |
| GET | `/briefing` | 分段简报：开始工作前先读 |
| GET | `/scope` | 当前实验与样品；动作会不会被记录 |
| POST | `/scope` | 切换或新建实验 / 样品 |
| POST | `/estop` | 硬件急停并取消外部作业 |
| POST | `/jobs` | 提交技能作业（立即返回） |
| GET | `/jobs` | 你最近的作业 |
| GET | `/jobs/{job_id}` | 作业视图，可长轮询 |
| POST | `/jobs/{job_id}/cancel` | 协作式取消 |
| GET | `/skills/search` | 按动作 / Nanonis 命令找技能 |
| GET | `/skills/{name}` | 技能卡 |
| POST | `/composites/draft` | 校验组合技能草稿（不保存） |
| POST | `/composites` | 保存并热注册组合技能 |
| POST | `/skills/proposals` | 提议 Python 技能（待人审） |
| GET | `/data/files` | 最近的数据文件 |
| GET | `/data/file` | 原始字节 |
| GET | `/data/frame` | 统一朝向后的 `.sxm` 帧（`.npz`） |
| POST | `/notes` | 写笔记进 MAST 记忆库 |
| GET | `/notes` | 检索笔记（含内部 agent 写的） |
| POST | `/requests` | 向操作员发问 |
| GET | `/requests` | 你的请求与答复 |
| GET | `/requests/{request_id}` | 单条请求与答复 |
| POST | `/handover` | 交接报告存进文档库 |

## 总览、作用域与急停

### `GET /health`

外部面自己的健康：各子系统有没有接上（`runtime`、`registry`、`pool`、`state`、`storage`、`cognition`、`jobs`）。`missing` 非空时，依赖那些子系统的端点会回 503 `not_wired`。不需要仪器在线。

### `GET /guide`

面向 agent 的铁律清单，`lang=zh|en`。返回 `{lang, rules:[{id, title, text}], docs_hint}`。规则的原因与细节见 [操作规则](03-operating-rules.md)。

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|---|
| `lang` | query | `string` |  | `"en"` |  |

### `GET /status`

快速状态，只读内存与缓存快照、**不碰硬件**：`mode`（`safe` / `semi` / `auto`；没人设置过时是 `unknown`，不会被说成某个具体值）、`abort`（`{set, emergency, why}`）、`lock`（仪器锁：`{held:false}` 或 `{held, owner, skill, held_s}`）、`connection`（各 TCP 角色连没连上）、`live`（偏压、电流、设定点、Z、Z 控制、扫描中、是否退针、`stale`）、`scope`、`jobs`（`{running, max}`）、`degraded`（读不到的字段名）。

### `GET /briefing`

分段简报：MAST 自己的 agent 每一轮看到的上下文块，加上「在你之前谁做了什么」。`sections=a,b` 只取某几段（空 = 全部；未知段名 422，回执里有 `available`）。每段 `{ok, text?, data?, error?}`；**任何一段读不到只让那一段 `ok:false` 并进 `degraded`**，其余照给。顶层 `text` 是各段文字拼成的 markdown，截断到约 12000 字符。零硬件 I/O。各段内容见下文 [简报的段](#简报的段)。

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|---|
| `sections` | query | `string` |  | `""` |  |

### `GET /scope`

当前实验、样品与实验记录服务是否可用：`{experiment:{id, name, goal}|null, sample:{id, name, sample_type}|null, recording:{v1, v2, note}}`。没有当前实验时，动作不能归入实验动作记录，但独立作业日志与全局笔记仍可存在。这里描述当前配置；实际写入结果见各作业的 `recorded`。

### `POST /scope`

切换或新建实验 / 样品。给 `id` 就切换到已有的（切回一个实验时恢复它上次用过的样品；切到别的实验下的样品会把实验一起切过去）；给 `name` 就按名字复用，没有才新建（样品在当前实验内查找）。仪器正被占用时默认 409 `instrument_busy`（在跑的动作会记到新作用域下），`force:true` 才切并在 `warnings` 里写明。其他错误：404 `unknown_experiment` / `unknown_sample`、409 `no_experiment`（没有实验就建样品）、422 `empty_request`。回执 = `GET /scope` 的内容 + `changed` + `warnings`。

**请求体** (`ScopeBody`)

| 字段 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|
| `experiment` | `ScopeRef \| null` |  |  |  |
| `sample` | `ScopeRef \| null` |  |  |  |
| `force` | `boolean` |  | `false` |  |
| `reason` | `string` |  | `""` | `maxLength=300` |

其中 `ScopeRef`:

| 字段 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|
| `id` | `string \| null` |  |  | `maxLength=120` |
| `name` | `string \| null` |  |  | `maxLength=200` |
| `goal` | `string` |  | `""` | `maxLength=2000` |
| `description` | `string` |  | `""` | `maxLength=2000` |
| `sample_type` | `string` |  | `""` | `maxLength=120` |

```json
{"experiment": {"name": "Au(111) reconstruction", "goal": "map the herringbone"},
 "sample": {"name": "Au(111) #3", "sample_type": "metal"}}
```

### `POST /estop`

硬件急停：调用与操作员急停按钮相同的流程，先挂中止与急停闩，再尝试停止运动和退针。闩上的原因写成「外部 agent ext:<名字> 触发急停：<reason>」；然后请求取消仍在运行的外部作业。回执含 `why`、`cancelled_jobs` 与急停动作本身的结果（`errors`、`retracted` 等）。闩已置位不代表硬件动作成功，必须检查结果并核实仪器状态。**解闩是操作员的事**，外部面不提供解闩。

**请求体** (`EstopBody`)

| 字段 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|
| `reason` | `string` |  | `""` | `maxLength=300` |

## 作业

### `POST /jobs`

提交一个技能作业，立即返回（新作业 202 + 作业视图）。门口按顺序检查：技能存在（否则 404 `unknown_skill`，带 `did_you_mean`）→ 技能及其声明式子步都没被本机关闭（否则 422 `skill_disabled`）→ `request_id` 幂等（同一调用方同一 id：内容相同回原作业 200 + `idempotent_replay:true`；内容不同 409 `request_id_conflict`）→ 并发上限（429 `too_many_jobs`；停止 / 退针类技能如 `StopScan` 不受上限约束）。带量纲的参数可写 `"5n"` 这类带 SI 前缀的字符串。**技能自身失败不是 HTTP 错误**：作业以终态结束。提交丢了回执时，用**同一个** `request_id` 重发。

**请求体** (`JobSubmit`)

| 字段 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|
| `skill` | `string` | 是 |  | `minLength=1, maxLength=120` |
| `params` | `object` |  |  |  |
| `request_id` | `string \| null` |  |  | `maxLength=128` |
| `note` | `string` |  | `""` | `maxLength=500` |

```bash
curl -s -X POST http://127.0.0.1:7862/api/ext/v1/jobs \
  -H "Content-Type: application/json" -H "X-MAST-Actor: claude-code" \
  -d '{"skill": "GetBias", "params": {}, "request_id": "getbias-001"}'
curl -s "http://127.0.0.1:7862/api/ext/v1/jobs/j_0123456789ab?wait_s=30" \
  -H "X-MAST-Actor: claude-code"
```

### `GET /jobs`

你最近提交的作业（新的在前）：`{count, jobs:[作业视图]}`。`all=true` 看所有调用方的，`state=` 按状态过滤。丢了上下文之后用它找回自己提交过什么。

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|---|
| `all` | query | `boolean` |  | `false` |  |
| `state` | query | `string` |  | `""` |  |
| `limit` | query | `integer` |  | `50` | `min=1, max=300` |

### `GET /jobs/{job_id}`

作业视图（见 [作业视图](#作业视图)）。`wait_s`（0–30）> 0 时等到作业结束或超时再回；服务端等待不占线程。超过保留数量的旧作业会被丢弃（404 `unknown_job`）。

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|---|
| `job_id` | path | `string` | 是 |  |  |
| `wait_s` | query | `number` |  | `0.0` | `min=0.0, max=30.0` |

### `POST /jobs/{job_id}/cancel`

协作式取消：技能在下一次检查中止处停下，原因（谁、为什么）会传到技能与记录里。卡在一条阻塞的 Nanonis 命令里时要等那条命令返回。请求物理停止时，运行不受仪器锁阻挡的停止类技能（例如 `StopScan`），危险时调用 `POST /estop`；检查响应并确认仪器状态。对已结束的作业是空操作。

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|---|
| `job_id` | path | `string` | 是 |  |  |

**请求体** (`CancelBody`)

| 字段 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|
| `reason` | `string` |  | `""` | `maxLength=300` |

## 技能

### `GET /skills/search`

按**动作**找技能：技能按「治什么问题」命名，按名字猜很难猜中。`q` 会同时匹配名字、Nanonis 命令名（如 `Bias_Set`、`Scan_Action`，从技能源码静态读出）、意图词、标签、参数名与描述；官方技能排在前面。每条结果带 `footprint`（`pure-analysis` / `hardware-read-only` / `hardware-write` / `unknown`）、`matched_on`（为什么命中）与 `origin_code`。

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|---|
| `q` | query | `string` | 是 |  | `minLength=1, maxLength=200` |
| `limit` | query | `integer` |  | `20` | `min=1, max=100` |

### `GET /skills/{name}`

技能卡：运行前该读的一切。字段见下文 [技能卡](#技能卡)。未知名字 404，带 `did_you_mean`。

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|---|
| `name` | path | `string` | 是 |  |  |

### `POST /composites/draft`

校验一份组合技能草稿（`CompositeSpec` JSON），**不落盘、不注册**，问题整批返回，外加与已有官方技能的重合提示。`spec: "?"` 返回格式说明。写法见 [编写技能](05-authoring-skills.md)。

**请求体** (`CompositeBody`)

| 字段 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|
| `spec` | `object \| string` | 是 |  |  |
| `base_version` | `integer` |  | `-1` |  |

### `POST /composites`

保存组合技能并热注册，署名 `ext:<名字>`。之后它就是一个普通技能：经 `POST /jobs` 执行，每个子步都过全部安全闸。改自己存过的同名技能要带 `base_version`（乐观锁；缺了回 `base_version_required`）；人做的技能不能被覆盖。业务上的拒绝是 200 + `ok:false` + `error`。

**请求体** (`CompositeBody`)

| 字段 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|
| `spec` | `object \| string` | 是 |  |  |
| `base_version` | `integer` |  | `-1` |  |

### `POST /skills/proposals`

提议一个新的原子技能（Python 源码）—— 组合表达不了时才用。写进本机的自定义技能目录**等人审**：不注册、不执行；要用，操作员审过代码、在启用清单里打开、重启。回执附一份合规报告 `compliance`（与投稿校验器同一套判据）。含禁用写法（如 `import os`）的代码直接 `ok:false`、`error:"rejected"`，不落盘。

**请求体** (`ProposalBody`)

| 字段 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|
| `name` | `string` | 是 |  | `minLength=1, maxLength=80` |
| `code` | `string` | 是 |  | `minLength=1, maxLength=200000` |
| `rationale` | `string` | 是 |  | `minLength=1, maxLength=4000` |

## 数据

### `GET /data/files`

最近的数据文件（mtime 新的在前；同一份测量的多个拷贝折叠成一条，`locations` 列出每一份）：`{files:[{path, name, ext, mtime, size_bytes, kind, copies, locations}], count, has_more, degraded, counts_by_ext}`。只列允许根目录之内的文件；**不接受客户端指定目录**。

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|---|
| `n` | query | `integer` |  | `20` | `min=1, max=200` |
| `ext` | query | `string \| null` |  |  |  |
| `offset` | query | `integer` |  | `0` | `min=0` |

### `GET /data/file`

原始字节（`application/octet-stream`）。路径先解析（`..`、符号链接）再判归属：只许数据搜索目录与实验根目录之内、扩展名在白名单里（`.sxm .dat .3ds .txt .csv .png .npy .npz`）的文件；网络路径一律拒绝。错误：403 `path_not_allowed` / `extension_not_allowed`、404 `not_found`、422 `bad_path`。

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|---|
| `path` | query | `string` | 是 |  |  |

### `GET /data/frame`

一个 `.sxm` 通道的帧，**服务端统一朝向**后打包成 `.npz`：`forward` / `backward` 的第 0 行是图的顶边（`scan_dir=up` 已翻转），`backward` 已去镜像（与 `forward` 同一几何朝向）；只有反向数据的通道，那一块作为 `forward` 给出，`meta_json` 的 `served_block` 如实写明。`meta_json` 另含宽高（nm）、nm/px、偏压、设定点、扫描方向、单位、可用通道；同一份元数据也在响应头 `X-MAST-Frame-Meta`（ASCII JSON）。未知通道 404 `unknown_channel`（带 `channels`）；非 `.sxm` 422 `not_sxm`。

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|---|
| `path` | query | `string` | 是 |  |  |
| `channel` | query | `string` |  | `"Z"` |  |

```python
import io, json, urllib.parse, urllib.request
import numpy as np

BASE = "http://127.0.0.1:7862/api/ext/v1"
q = urllib.parse.urlencode({"path": path, "channel": "Z"})
body = urllib.request.urlopen(f"{BASE}/data/frame?{q}").read()
z = np.load(io.BytesIO(body), allow_pickle=False)
meta = json.loads(str(z["meta_json"]))
forward = z["forward"]
backward = z["backward"] if "backward" in z.files else None
```

## 笔记、发问与交接

### `POST /notes`

写一条笔记进 MAST 的记忆库，**MAST 自己的 agent 会自动召回它**。`scope=experiment` 存进当前实验（没有实验时退到 global 并在 `warnings` 里写明），`global` 跨实验。署名 `ext:<名字>`；同标题同内容重发落在同一条上（路径由内容哈希决定），不会重复。

**请求体** (`NoteBody`)

| 字段 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|
| `title` | `string` | 是 |  | `minLength=1, maxLength=200` |
| `content` | `string` | 是 |  | `minLength=1, maxLength=20000` |
| `kind` | `string` |  | `"note"` | `maxLength=40` |
| `tags` | `array<string>` |  |  |  |
| `scope` | `"experiment" \| "global"` |  | `"experiment"` |  |

```json
{"title": "tip state", "content": "atomic contrast at -1.2 V / 50 pA",
 "kind": "insight", "tags": ["tip"], "scope": "experiment"}
```

### `GET /notes`

检索笔记：`q` 非空时召回（当前实验 ∪ global；语义索引不可用时退回子串匹配），为空时列最近的。MAST 自己的 agent 写下的记忆也在这里（看 `author`）。

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|---|
| `q` | query | `string` |  | `""` | `maxLength=500` |
| `scope` | query | `string` |  | `"both"` |  |
| `limit` | query | `integer` |  | `10` | `min=1, max=50` |

### `POST /requests`

向操作员发一个问题或请求，界面上会亮起来。回执 `{ok, request:{id, status, ...}}`。**不要在这里等** —— 答复是异步的，之后用 `GET /requests/{request_id}` 或简报的 `operator_requests` 段看。同一个未答的问题重发不会重复。

**请求体** (`RequestBody`)

| 字段 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|
| `message` | `string` | 是 |  | `minLength=1, maxLength=4000` |
| `kind` | `string` |  | `"question"` | `maxLength=40` |

### `GET /requests`

你发过的请求与答复（未答的在前）：`{count, requests}`。`status` 过滤：`pending`（未答，同义 `open`）、`done`、`dismissed`；未知值回 422 `unknown_status`。

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|---|
| `status` | query | `string` |  | `""` |  |

### `GET /requests/{request_id}`

单条请求：`note` 是文字答复，`path` 是操作员给的文件或目录路径（两者可能都有）。

| 参数 | 位置 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|---|
| `request_id` | path | `string` | 是 |  |  |

### `POST /handover`

生成交接报告存进 MAST 的文档库（操作员在「报告」页能看到），署名 `ext:<名字>`：你写的 `summary` 与 `next_steps`，你这次提交的作业及结局，实验记录里署你名字的动作，你写的笔记。`since`（ISO 时间）只汇总此后的内容。回执 `{ok, doc_id, version, path, title, jobs, actions, notes}`。会话结束前写一份。

**请求体** (`HandoverBody`)

| 字段 | 类型 | 必填 | 默认 | 约束 |
|---|---|---|---|---|
| `summary` | `string` | 是 |  | `minLength=1, maxLength=20000` |
| `next_steps` | `string \| array<string>` |  |  |  |
| `title` | `string` |  | `""` | `maxLength=200` |
| `since` | `string \| null` |  |  | `maxLength=40` |

## 作业视图

```json
{
  "job_id": "j_0123456789ab", "skill": "GetBias", "params": {}, "params_used": {},
  "actor": "claude-code", "session": "", "request_id": "getbias-001", "note": "",
  "run_id": "ext-j_0123456789ab",
  "state": "succeeded", "terminal": true,
  "created_at": "2026-01-01T12:00:00", "started_at": "2026-01-01T12:00:00",
  "finished_at": "2026-01-01T12:00:01", "elapsed_s": 0.41,
  "cancel_requested": false, "cancel_reason": "",
  "result": {"success": true, "summary": "", "error": "", "data": {},
             "nanonis_calls": 1, "elapsed_s": 0.4},
  "refused_by": null, "busy_holder": null, "abort": null,
  "recorded": {"v1": true, "v2_action_id": "...", "experiment_id": "...", "sample_id": "..."}
}
```

| `state` | 含义 |
|---|---|
| `queued` / `running` | 还没结束。 |
| `succeeded` | 技能报告成功。 |
| `failed` | 技能失败或被拒，详情见 `result.error`。`refused_by` 只标识部分拒绝类别（如 `sample_gate`、`si_parse`、`needs_human_node`），也可能为 null。 |
| `refused_busy` | 仪器正被别人占用，**不排队**；`busy_holder` 说明是谁在开车。 |
| `cancelled` | 已请求取消，且作业未报告成功。仪器最终状态需要另行核实。 |
| `crashed` | 执行线程自己出了意外。 |
| `lost_on_restart` | 服务重启时它还没结束，不知道它在仪器上做到了哪一步。**绝不重放**；用同一个 `request_id` 重发拿到的就是这一条。先读简报再决定下一步。 |

`recorded` 是实验记录写入回执。`v1: false` 表示 v1 写入未成功，可能是缺作用域、存储未接线或写入错误；结合 `experiment_id`、`sample_id`、`v2_action_id` 和可能的 `error` 判断。空回执不代表已记录，独立作业日志也不等于实验动作记录。`abort` 在技能结束时中止事件已置位的情况下给出 `{set, reason}`。`result.data` 过大时会被截断。

## 简报的段

| 段 | 内容 |
|---|---|
| `status` | 与 `GET /status` 相同的内容，写成文字。 |
| `scope` | 当前实验、样品、目标；没有时说明后果。 |
| `resume` | 续工块：这个实验此前做到了哪里（与 MAST 自己的 agent 看到的相同）。 |
| `tip` | 当前针尖的登记信息。 |
| `instrument` | 仪器档案与已学到的标定。 |
| `live` | live 读数（缓存快照）。 |
| `prefs` | 操作员设置的默认参数偏好。 |
| `safety` | 此刻生效的安全包络，以及挂着的闩和解除办法。 |
| `recent_actions` | 本实验最近的动作，带署名（谁做的）。 |
| `recent_files` | 最近的数据文件。 |
| `alarms` | 环境监控的总体状态与最近的告警。 |
| `notes` | 本实验最近的笔记（任何作者）。 |
| `recording` | 你的署名，以及当前作用域是否配置了实验记录。 |
| `jobs` | 你最近的作业。 |
| `operator_requests` | 你发给操作员的请求与答复。 |

## 技能卡

除参数外，技能卡给出：

| 字段 | 内容 |
|---|---|
| `parameters` | 每个参数的类型、单位、必填、默认值、上下限、取值集。 |
| `safety_level / capabilities` | 安全级与能力标签（例如是否属于针尖处理）。 |
| `preconditions` | 运行前必须成立的条件。 |
| `category / tags` | 类别与标签。 |
| `origin / origin_code / official` | 来源：`origin` 是给人读的标签（目前是中文），`origin_code` 是语言中立的代号（`builtin`、`composite`、`paper`、`user_composite`、`custom`、`agent_tool`、`overlay`、`other`）；`official` 只对前三种为真。 |
| `footprint / footprint_basis / footprint_reasons` | 对仪器做什么（`pure-analysis` / `hardware-read-only` / `hardware-write` / `unknown`）。与投稿校验器 `scripts/skill_check.py` 出自同一份静态分析：`footprint_basis` 为 `static` 是从源码读出来的，`declared` 是源码看不透、按类别保守取的，`footprint_reasons` 说明为什么看不透。 |
| `verbs / verbs_unknown` | 会发的 Nanonis 命令；分析不完整时 `verbs_unknown:true`，不把缺失报成空集。 |
| `sub_skills` | 组合技能的子步。 |
| `takes_instrument_token` | 是否占用仪器锁（占用时别人会被 `refused_busy`）。 |
| `requires_sample` | 是否必须先选样品。 |
| `si_params` | 哪些参数接受 `"5n"` 这类写法，以及是否必须带前缀。 |
| `tool_face` | 本机是否关闭了它，以及原因。 |
| `duration` | `{estimated_s, measured:{n, p50_s, p95_s, max_s}\|null, note}` —— `measured` 取自这台仪器上的成功记录。 |
