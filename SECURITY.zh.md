# 安全策略

[English](SECURITY.md)

MAST 通过 Nanonis 控制器操作真实扫描隧道显微镜。安全设计同时涉及软件、访问权限和物理执行：
漏洞可能影响实验数据，也可能导致针尖、样品或运动机构受损。

## 私密报告漏洞

优先使用 GitHub 仓库的 **Security / Advisories → Report a vulnerability** 私密报告入口。
如果尚未显示该入口，请仅开 issue 请求维护者提供私密联络渠道，不附漏洞细节、利用代码、凭据或现场数据。
本项目未提供专用邮箱。入口条件见 [GitHub 私密报告说明](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/report-privately)。

报告中请提供受影响版本或提交、相关组件、复现前提、预期与实际行为，以及可能影响的仪器操作。
可用替身或合成数据复现时，优先提供这类最小示例。

本仓按源码快照维护。漏洞修复进入主开发线，并随后续公开快照提供；不为旧快照提供回溯补丁。

## 部署与权限边界

- **仪器控制接口限于本机或可信网络。** 服务默认绑定回环地址；LAN 模式使用 HTTP Basic 与 TLS。
  远程操作通过 VPN 访问仪器电脑，不向公网开放端口。
- **MCP 默认仅连接回环地址。** 经 VPN 访问其他机器时，需显式配置 `allow_remote`。
- **凭据由客户端管理。** 插件将密码字段标记为 `sensitive`；Claude Code 在没有受支持钥匙串的平台上
  使用 `~/.claude/.credentials.json` 存储这类值。保护客户端凭据文件，不将密码、令牌或本地配置提交到仓库。
  具体平台行为见 [Claude Code 配置参考](https://code.claude.com/docs/en/plugins-reference#user-configuration)。
- **外部 agent 使用操作员权限。** 当前没有按 agent 分配的委托令牌或独立权限域。
  交接前由操作员设置 SAFE 模式，可限制偏压脉冲与修针；运行模式与执行检查用于约束操作，
  不替代访问控制。开环 Z 粗逼近等操作另有执行限制。
- **社区 Python 技能在 MAST 进程内运行。** 检查器与加载器使用 AST 拒绝名单检查代码，
  不提供隔离沙箱；启用前审阅源码及其声明的硬件足迹。

## 相关说明

[公开版本与验证记录](docs/OPEN_SOURCE_NOTES.md) · [操作规则](docs/external/zh/03-operating-rules.md) ·
[社区技能](contrib/README.zh.md)
