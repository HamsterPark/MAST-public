# 快速上手

这一篇带你连接已有的 MAST 服务：读简报、选择实际样品、跑一个只读技能、交接。
操作员应先完成仪器配置，启动带 `/api/ext/v1` 的版本，并授权本次会话。公开源码检出本身不等于这套部署。
接下来读 [概念](02-concepts.md) 了解你刚才每一步背后的道理，读
[运行规则](03-operating-rules.md) 了解规则本身与它们各自的理由。

## 三种接入方式

三条路都通向同一个 API（`/api/ext/v1`），受同一套规则约束；按你所在的环境挑一种。

### A. Claude Code 插件

自带的插件给 Claude Code 一组 `mast_*` 工具（每个端点一个）加两个技能：`mast-operator`
（驱动 MAST 的工作流程）与 `mast-skill-author`（写组合技能与 Python 提议，并把它们投稿
上游）。安装：

```text
/plugin marketplace add <path-to-your-MAST-checkout>
/plugin install mast@mast
```

请将占位路径替换为本公开快照在本机的绝对路径。

使用这些选项配置插件：

| 选项 | 默认值 | 含义 |
|---|---|---|
| `python` | `python` | 跑服务器的解释器（3.10+）；Windows 上裸 `python` 若打开微软商店，请指向一个真的解释器 |
| `mast_url` | `http://127.0.0.1:7862` | MAST 的地址；LAN 模式用 `https://` |
| `mast_user` | 空 | HTTP Basic 用户名，只在 LAN 模式下用 |
| `mast_password` | 空 | HTTP Basic 密码，只在 LAN 模式下用 |
| `verify_tls` | `true` | `false` 接受自签名证书 |
| `allow_remote` | `false` | 允许一个不是本机的地址 |
| `actor` | `claude-code` | 你的动作在 MAST 记录里署的名字 |

完整的选项清单与精确默认值在插件自己的包里（就在插件源码旁边的 `README.md`）；这张表
是摘要。接上之后，先让 agent 去读简报 —— 这是 [运行规则](03-operating-rules.md) 的第一条。

这个只用标准库的 MCP 桥接进程只需 Python 3.10+；MAST 后端使用 Python 3.13 及其自身依赖。
客户端机器无需为了桥接进程安装后端的整套环境。

### B. 项目级 MCP 服务器

任何兼容 MCP 的客户端都能直接从一份 MAST 的代码检出跑同一个服务器，不必经过插件市场：
把 MCP 配置指向插件目录里的 `server/run_server.py`。这样跑的话，服务器从环境变量读设置，
而不是插件的那些选项：

| 变量 | 含义 |
|---|---|
| `MAST_URL` | MAST 的地址，与上面的 `mast_url` 相同 |
| `MAST_USER` / `MAST_PASSWORD` | HTTP Basic 登录，只在 LAN 模式下用 |
| `MAST_VERIFY_TLS` | `false` 接受自签名证书 |
| `MAST_ALLOW_REMOTE` | 允许一个不是本机的地址 |
| `MAST_ACTOR` | 你的动作署的名字 |
| `MAST_SESSION` | 会话名；不设置时用一个随机的 |
| `MAST_FETCH_DIR` | 取回的文件存在哪；设了 `CLAUDE_PROJECT_DIR` 时默认在它下面的 `.mast-fetch/`，否则在工作目录下 |

```json
{
  "mcpServers": {
    "mast": {
      "command": "python",
      "args": ["/path/to/MAST/integrations/claude-code/server/run_server.py"],
      "env": {
        "MAST_URL": "http://127.0.0.1:7862",
        "MAST_ACTOR": "my-agent"
      }
    }
  }
}
```

### C. 纯 HTTP

这个 API 根本不需要专门的客户端：任何 HTTP 库，或者 `curl`，都能直接调它。这是最底层
的接入方式，也最适合初次看清 API 到底返回什么。下面的走一遍全程用它，好让你看到每个
答复的形状。

## 远程访问

以上所有方式默认都是在本机、用明文 HTTP、不带任何凭据。要接一台在别的机器上的仪器，
只支持经 VPN，而且要先打开 LAN 模式：MAST 这时会用 `https://`（自签名证书）加 HTTP
Basic 登录来提供服务。不要把 MAST 暴露到公网 —— 没有别的认证层兜底。

插件自己就会强制回环默认值这一点：不是本机的地址会被拒绝，直到打开 `allow_remote`。
插件没有办法核实你打开之后用的那个地址是不是真的在 VPN 里 —— 让它待在那里是你自己的
责任；绝不要把它指向一个公网可达的地址。自签名证书需要把 `verify_tls` 设为 `false`，
或者在本机装上那张证书。还有一个提前知道能省不少事的坑：如果你填的是一个恰好落在浏览器内置 HSTS
预加载名单里的**主机名**，浏览器会连「仍要继续」这个接受自签名证书的选项都不给 ——
这种情况下改用数字 IP 地址。

## 首次会话走一遍

下面的路径均相对于 `/api/ext/v1`。按顺序执行，并重复轮询直到作业进入终态。
这些调用会读取仪器，并写入作用域与会话记录。

1. **健康检查。** `GET /health` —— 确认外部面本身活着，并且在你依赖它们之前先说清哪些
   子系统接了线。`ok: true` 不代表仪器已连通；还应检查 `wired`、`missing`，以及简报里的连接状态与读数新鲜度。
2. **简报。** `GET /briefing` —— 先读它再做别的。里面有针尖、实时读数、操作员的偏好、
   所有人最近的动作，以及你自己尚未得到答复的请求。
3. **作用域。** `POST /scope` —— 选定或新建实验与样品。扫描与谱学需要样品作用域；实验动作入账
   需要当前实验和可用存储，应检查返回的记录状态。
4. **找一个技能。** `GET /skills/search` —— 按你要的动作搜，别去猜名字。
5. **技能卡。** `GET /skills/{name}` —— 任何技能第一次跑之前先读单位、范围、前置条件、
   以及已有时的实测耗时。
6. **跑一个只读作业。** `POST /jobs` —— 把一次无害的读取（比如读偏压）当作业提交；它
   立刻带着 `job_id` 返回。
7. **轮询。** `GET /jobs/{job_id}?wait_s=30` —— 等作业进入终态。
8. **交接。** `POST /handover` —— 用一份总结收尾这次会话，哪怕会话很短；它让下一个接手
   的人或 agent 知道发生过什么。

```bash
BASE=http://127.0.0.1:7862/api/ext/v1
curl -s "$BASE/health"
curl -s "$BASE/briefing" -H "X-MAST-Actor: claude-code"
curl -s -X POST "$BASE/scope" -H "Content-Type: application/json" -H "X-MAST-Actor: claude-code" \
  -d '{"experiment": {"name": "Quickstart"}, "sample": {"name": "Bench sample"}}'
curl -s "$BASE/skills/search?q=bias" -H "X-MAST-Actor: claude-code"
curl -s "$BASE/skills/GetBias" -H "X-MAST-Actor: claude-code"
curl -s -X POST "$BASE/jobs" -H "Content-Type: application/json" -H "X-MAST-Actor: claude-code" \
  -d '{"skill": "GetBias", "params": {}, "request_id": "quickstart-001"}'
curl -s "$BASE/jobs/j_0123456789ab?wait_s=30" -H "X-MAST-Actor: claude-code"
curl -s -X POST "$BASE/handover" -H "Content-Type: application/json" -H "X-MAST-Actor: claude-code" \
  -d '{"summary": "Quickstart walkthrough: read the bias once."}'
```

上面的 shell 示例采用 Bash 语法。实验与样品名称请换成操作员指定的作用域，必须对应实际安装的样品。
轮询那一步里的作业 id 是占位符：换成你自己提交后拿到的那个 `job_id`。每个新的预期动作使用新的
`request_id`，仅在重试该次提交时复用同一个 ID。这些答复里每个
字段的含义见 [作业视图](07-api-reference.md)；你先定了作用域之后到底记下了什么，见
[记录与上下文](04-records-and-context.md)。
