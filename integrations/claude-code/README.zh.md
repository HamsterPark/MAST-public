# Claude Code 的 MAST 插件

[English](README.md)

将 Claude Code 接入 MAST 的真实仪器工作流：通过 `/api/ext/v1` 获取实验上下文、检索技能、
提交可跟踪的作业、读取数据并与操作员协作。

一轮已结束的实验中，外部 agent 通过 MAST 6.4.0 的上一代控制路径操作真实 STM；
带时间戳的记录从 2026-09-17 23:10:37 至 2026-09-22 02:08:10（北京时间），
跨越约 99 小时。本插件对应 6.5.0 新接口，
已完成软件测试，尚待真机验证。公开源码包含插件与通用接口，完整仪器部署另需配置与配套资产。
仅审阅实现时，从[仓库指南](../../AGENTS.zh.md)开始即可。

## 包含什么

- **`mast` MCP server**：二十个 `mast_*` 工具，覆盖会话简报、技能检索与技能卡、带轮询与取消的作业、
  急停、实验与样品范围、原始数据下载、笔记、向操作员发问、组合技能和交接报告。
- **两个技能**：[mast-operator](skills/mast-operator/SKILL.md)（操作流程与执行规则）和
  [mast-skill-author](skills/mast-skill-author/SKILL.md)（技能复用、组合与开发）。
- **指南资源**：中英双语操作指南，以 `mast://guide/{en|zh}/{file}` 提供。

MCP server 仅使用 Python 标准库，无需额外 Python 包；MAST 服务本身有独立的依赖与部署要求。

## 前提

- 已配置并运行、带外部 agent 接口的 MAST 服务，位于本机或通过 VPN 可达的仪器电脑。
- 用于 MCP server 的 Python 3.10 或更新版本；MAST 后端的源码环境要求为 Python 3.13。
  本公开源码包不附带 Python 运行时。已有完整安装版时，可选用其中的 `MASTv2/pyruntime/python.exe`。

## 安装

在 Claude Code 里，将下面的占位路径替换为本地 MAST 仓库根目录：

```text
/plugin marketplace add <path-to-your-MAST-checkout>
/plugin install mast@mast
```

启用插件时 Claude Code 会询问下面这些选项。

## 配置

| 选项 | 默认值 | 含义 |
|---|---|---|
| `python` | `python` | 运行 server 的解释器（3.10+）：MAST 的 `MASTv2/pyruntime/python.exe` 或已安装的 Python |
| `mast_url` | `http://127.0.0.1:7862` | MAST 根地址；LAN 模式用 `https://` |
| `mast_user` | 空 | HTTP Basic 用户名，仅 LAN 模式 |
| `mast_password` | 空 | HTTP Basic 密码，仅 LAN 模式；以 `sensitive` 字段交由客户端存储 |
| `verify_tls` | `true` | 默认校验证书；仅在确认目标的受控自签环境中按需设为 `false` |
| `allow_remote` | `false` | 允许不在本机的 MAST 地址（只经 VPN） |
| `actor` | `claude-code` | 你的动作在 MAST 记录里的署名 |

### 环境变量

手动运行时，server 从 `MAST_URL`、`MAST_USER`、`MAST_PASSWORD`、`MAST_VERIFY_TLS`、
`MAST_ALLOW_REMOTE`、`MAST_ACTOR` 读同样的设置。`MAST_FETCH_DIR` 指定 `mast_fetch` 的落盘目录，
默认是项目里的 `.mast-fetch/`，它自带一个 `.gitignore`，拉下来的数据不会进 git。`MAST_MCP_LOG`
设日志级别（日志走 stderr）。

### 检查连接

在 Claude Code 里运行 `/mcp`：`mast` 应显示为已连接。如果显示「Failed to connect」并带
「Connection closed」，说明 `python` 选项没有指向能用的解释器（见 Windows 注意事项）。然后让 Claude
调一次 `mast_status`。如果返回错误，错误里会写明该查什么：MAST 是否在运行、http 还是 https、
登录凭据、地址是否不在本机。

## Windows 注意事项

- **Python 路径。** Windows 上裸的 `python` 常常是微软商店的占位程序，跑不了 server。把 `python`
  选项指向真正的解释器，例如默认安装目录下 MAST 自带的那个：

```text
C:\MAST\MASTv2\pyruntime\python.exe
```

- **HTTPS 与自签证书。** LAN 模式的 `mast_url` 使用 `https://`。优先配置受信任的证书；
  已确认目标仪器电脑、且处于受控自签环境时，可按需将 `verify_tls` 设为 `false`，这会关闭证书校验。
- **凭据。** 密码标记为 `sensitive`，存储由 Claude Code 管理；这一标记不等于所有平台都有加密钥匙串。
  平台细节与保护要求见[安全说明](../../SECURITY.zh.md)。
- **默认只认本机。** `127.0.0.1`、`localhost`、`::1` 以外的地址在打开 `allow_remote` 之前一律拒绝，
  因为仪器控制只应发生在本机或 VPN 内（例如经 VPN 访问 `https://192.0.2.10:7862`）。绝不要把 MAST
  暴露到公网。
- **不走代理。** server 从不经系统代理发送 MAST 流量。

## 工具

所有端点都在 `/api/ext/v1` 之下。

| 工具 | 端点 | 用途 |
|---|---|---|
| `mast_briefing` | `GET /briefing` | 一次拿到全部状态；最先读 |
| `mast_status` | `GET /status` | 快速状态检查 |
| `mast_find_skills` | `GET /skills/search` | 按动作找技能 |
| `mast_skill_card` | `GET /skills/{name}` | 参数、单位、安全级、耗时 |
| `mast_run` | `POST /jobs`, `GET /jobs/{id}` | 以作业方式执行技能并等待 |
| `mast_job` | `GET /jobs/{id}` | 轮询单个作业 |
| `mast_jobs` | `GET /jobs` | 列出作业 |
| `mast_cancel` | `POST /jobs/{id}/cancel` | 协作式取消 |
| `mast_emergency_stop` | `POST /estop` | 急停，并取消所有外部作业 |
| `mast_scope` | `GET /scope`, `POST /scope` | 实验与样品 |
| `mast_list_data` | `GET /data/files` | 最近的数据文件 |
| `mast_fetch` | `GET /data/file`, `GET /data/frame` | 把原始文件或一帧存到本地 |
| `mast_note_write` | `POST /notes` | 往 MAST 记忆库写笔记 |
| `mast_note_search` | `GET /notes` | 检索笔记 |
| `mast_ask_operator` | `POST /requests` | 向操作员发问 |
| `mast_operator_reply` | `GET /requests`, `GET /requests/{id}` | 读答复 |
| `mast_composite_draft` | `POST /composites/draft` | 校验组合技能 spec |
| `mast_composite_save` | `POST /composites` | 保存并注册组合技能 |
| `mast_propose_skill` | `POST /skills/proposals` | 提交 Python 代码待人审 |
| `mast_handover` | `POST /handover` | 交接报告 |

server 为单次工具调用设置 55 秒预算，单次作业等待最多 50 秒。
更长的工作在 MAST 上以作业形式继续运行：用 `mast_job` 轮询。
中断工具调用不会取消作业，取消要用 `mast_cancel`。

## 指南

完整的操作指南在 [mast-operator 的 references](skills/mast-operator/references/zh/README.md) 里，
也以 MCP 资源 `mast://guide/zh/...` 提供。

## 开发

在仓库根目录使用[指南中的 Python 3.13 环境](../../AGENTS.zh.md#无需硬件的验证)；
下面的 `python` 指该环境的解释器：

```text
python -m pytest tests/v2/unit/integrations -q
python scripts/sync_external_docs.py --check
```

`skills/mast-operator/references/` 下的指南文件是 `docs/external/` 的拷贝；改原件，再运行
`scripts/sync_external_docs.py`。

## 许可

MIT，与 MAST 相同。
