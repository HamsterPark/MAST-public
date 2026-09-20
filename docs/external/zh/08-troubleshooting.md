# 排障

现象、原因、处理办法 —— 遇到什么就从上往下找。这些现象背后的机制见
[概念](02-concepts.md) 与[运行规则](03-operating-rules.md)；一个错误体的精确形状见
[API 参考](07-api-reference.md)。

## 接入

| 现象 | 原因 | 处理办法 |
|---|---|---|
| 对一台你确定在 LAN 模式下跑的 MAST，`curl` 报 `HTTP=000` 或连接错误 | LAN 模式在传输层就是 TLS | 用 `https://`，不是 `http://` |
| `401 Unauthorized` | LAN 模式用 HTTP Basic，要么没发凭据，要么发错了 | 设置登录信息（插件里是 `mast_user`/`mast_password`，环境变量是 `MAST_USER`/`MAST_PASSWORD`） |
| `403`，`cross_origin_refused` | 这次请求看起来像是浏览器发的（带了 `Origin` 或某个 `Sec-Fetch-*` 头），而且要么跨了源，要么是经一个主机名（而不是回环名或字面 IP 地址）到达服务的 | 直接调 API，不要从网页里调；用 `127.0.0.1` 或数字地址而不是主机名 —— curl、MCP server 这类程序化客户端不受影响，因为它们两个头都不带 |
| `415`，`unsupported_media_type` | 一次写请求 —— 包括 `POST /estop` 与 `POST /jobs/{job_id}/cancel` —— 发出时没带 `Content-Type: application/json` 与一个 JSON 请求体 | 带上 `Content-Type: application/json` 和一个 JSON 请求体；没有别的要说就发 `{}` |
| 应返回 JSON 的端点给了 HTML 页面 | 路径前缀写错了，或者这份 MAST 没有外部 agent API | 确认 `/api/ext/v1`。包括 404 在内的 API 错误为 JSON；成功的 `/data/file` 与 `/data/frame` 下载返回二进制数据。 |
| TLS 证书错误 | 仪器机上的证书是自签名的 | 把 `verify_tls` 设为 false，或者在本机装上那张证书 |
| 一个远程地址被拒绝 | 不是本机的地址默认拒绝，除非显式打开了远程访问 | 只在 VPN 内打开它 —— 绝不要把 MAST 暴露到公网 |
| MCP 服务器起不来，或显示「Failed to connect」 | 配置的 Python 解释器是个占位符（Windows 上常是微软商店的那个），不是真的解释器 | 把解释器选项指向一个真的 Python 3.10+ |
| 非 ASCII 文字打印成乱码 | 终端或客户端没有按 UTF-8 读 JSON 响应 | JSON 响应按 UTF-8 解码；文件与帧下载按二进制保存。 |

## 跑技能

| 现象 | 原因 | 处理办法 |
|---|---|---|
| 所有作业立刻全部失败，错误里说 `aborted by operator — refusing to start ...` | 中止标志已置位；若为急停闩，`GET /status` 报告 `abort.emergency: true` 与 `abort.why` | 闩置位期间不要重试作业，停止技能也一样。检查急停响应的 `errors` 与 `retracted`；闩本身不能证明物理停止已成功。请操作员核实仪器状态，满足恢复条件后再解除闩。 |
| `503`、一份 `degraded` 列表、或一份 `missing` 列表 | 所需子系统不可用，可能正在启动、缺少配置，或部署资产有意未提供 | 与操作员核对具体子系统和服务配置。启动期间可稍后重试健康检查；反复重试不能补齐缺少的资产。 |
| `refused_busy` | 另一条链路已经占着仪器 | 等它，或者问操作员；立刻重试不会让它更快结束 |
| `failed`，`refused_by: sample_gate` | 这个技能产数据，需要一个当前样品 | 先用 `POST /scope` 定一个 |
| `failed`，`refused_by: si_parse` | 一个带量纲参数的值解析不出来 —— 往往是一个字符串漏掉了必须要有的 SI 前缀 | 查技能卡的 `si_params`；写 `"5n"` 这样的值，或者直接写 SI 基本单位下的一个数 —— 前缀规则只对字符串生效 |
| `failed`，`refused_by: needs_human_node` | 这个组合技能里有一步只有人能确认，而这条路径上没有人在等着接住那次暂停 | 改到 MAST 自己的界面里跑，或者去问操作员 |
| `422`，`skill_disabled` | 这个技能，或者它用到的某个子步骤，需要本机关闭了的硬件或能力 | 换一个技能，或者请操作员打开它 |
| `429` | 同时在跑的作业太多了 | 等一个跑完，或者取消一个不再需要的作业。被识别为停止/撤回的技能（如 `StopScan`）不受此上限约束，其他提交与执行检查仍然适用。 |
| `503`，`shutting_down` | 服务正在关停，拒绝接新作业 | 等它回来，再重新连接 |
| `lost_on_restart` | 作业还在跑的时候 MAST 进程重启了；它绝不会被重放 | 决定要不要再提交一次之前先读简报 |
| 拒绝里点名 `safe_mode_tip_processing_blocked` | 操作员设了 SAFE，它拒绝自主针尖处理，包括外部作业与组合子步骤 | 不要重试；任务需要时去问操作员 |
| 拒绝描述的是五条硬闸之一，说没有批准会到来 | 你调用了（或者组合技能的某一步调用了）这条路径上无条件拒绝的某个物理危险动作之一 | 别再重试 —— 什么都改变不了这个结局；确实需要的话，得由人在 MAST 自己的界面里去做 |
