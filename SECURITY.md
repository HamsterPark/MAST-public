# Security policy

[中文](SECURITY.zh.md)

MAST operates a real scanning tunnelling microscope through a Nanonis controller.
Its security design spans software, access control and physical execution:
a vulnerability can affect experimental data, the tip, the sample or instrument motion.

## Reporting privately

Use the repository's GitHub **Security / Advisories → Report a vulnerability** entry.
If it is unavailable, open an issue only to request a private contact channel; leave out
vulnerability details, exploit code, credentials and site data. No dedicated email channel
is provided. See [GitHub's reporting guide](https://docs.github.com/en/code-security/how-tos/report-and-fix-vulnerabilities/report-privately)
for the feature's availability.

Include the affected version or commit, component, reproduction prerequisites, expected
and observed behavior, and possible consequences for instrument operation.
Prefer a minimal example using test doubles or synthetic data when feasible.

This repository is maintained as source snapshots. Fixes enter the main development line
and subsequent public snapshots; backports to older snapshots are not provided.

## Deployment and permissions

- **Limit instrument control to the local machine or trusted networks.** The service
  binds to loopback by default; LAN mode uses HTTP Basic and TLS. For remote operation,
  access the instrument PC through a VPN without opening its ports to the public internet.
- **MCP connects to loopback by default.** Access to another machine over a VPN requires
  an explicit `allow_remote` setting.
- **The client manages credentials.** The plugin marks its password field `sensitive`.
  On platforms without a supported keychain, Claude Code stores these values in
  `~/.claude/.credentials.json`. Protect client credential files and keep passwords,
  tokens and local configuration out of the repository. See the
  [Claude Code configuration reference](https://code.claude.com/docs/en/plugins-reference#user-configuration).
- **External agents use operator privileges.** There are no per-agent delegated tokens
  or separate permission domains. Before handover, the operator can set SAFE mode to
  restrict bias pulses and tip shaping. Operating modes and execution checks constrain
  actions; they do not replace access control. Operations such as open-loop coarse Z
  approach have additional execution restrictions.
- **Community Python skills run inside the MAST process.** The checker and loader
  inspect code using an AST deny-list, without providing an isolation sandbox.
  Review the source and its declared hardware footprint before enabling it.

## Related guidance

[Public scope and validation](docs/OPEN_SOURCE_NOTES.md) ·
[Operating rules](docs/external/en/03-operating-rules.md) · [Community skills](contrib/README.md)
