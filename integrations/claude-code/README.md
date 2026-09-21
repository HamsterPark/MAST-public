# MAST plugin for Claude Code

[中文](README.zh.md)

Connect Claude Code to MAST's real-instrument workflow through `/api/ext/v1`: read experimental
context, discover skills, submit trackable jobs, retrieve data and collaborate with the operator.

In an ongoing experiment, an external agent used MAST 6.4.0's earlier control path to operate a
real STM over an approximately 93-hour window at the recorded snapshot.
This plugin targets the new 6.5.0 interface, which has undergone software testing and awaits hardware
validation. The public source includes the plugin and general API; instrument deployment needs
additional configuration and assets. For source review, start with [the repository guide](../../AGENTS.md).

## What it adds

- **The `mast` MCP server**: twenty `mast_*` tools for the session briefing, skill search and
  skill cards, jobs with polling and cancellation, emergency stop, experiment and sample scope,
  raw data download, notes, operator requests, composite skills and the handover report.
- **Two skills**: [mast-operator](skills/mast-operator/SKILL.md) (working procedure and rules)
  and [mast-skill-author](skills/mast-skill-author/SKILL.md) (skill reuse, composition and development).
- **Guide resources**: the bilingual operator guide as `mast://guide/{en|zh}/{file}`.

The MCP server uses only the Python standard library, with no additional Python packages.
The MAST service has its own dependencies and deployment requirements.

## Requirements

- A configured, running MAST service with the external agent API, on this machine or an instrument
  PC reachable over a VPN.
- Python 3.10 or newer for the MCP server; the MAST backend's source environment requires Python 3.13.
  This public source package does not bundle Python. An existing full installation may provide
  `MASTv2/pyruntime/python.exe`.

## Install

In Claude Code, replace the path placeholder with the root of your local MAST checkout:

```text
/plugin marketplace add <path-to-your-MAST-checkout>
/plugin install mast@mast
```

Claude Code asks for the options below when the plugin is enabled.

## Configuration

| Option | Default | Meaning |
|---|---|---|
| `python` | `python` | Interpreter for the server (3.10+): MAST's `MASTv2/pyruntime/python.exe` or an installed Python |
| `mast_url` | `http://127.0.0.1:7862` | MAST root address; LAN mode uses `https://` |
| `mast_user` | empty | HTTP Basic user, LAN mode only |
| `mast_password` | empty | HTTP Basic password, LAN mode only; stored by the client as a `sensitive` field |
| `verify_tls` | `true` | Verify certificates by default; use `false` only for a confirmed target in a controlled self-signed setup |
| `allow_remote` | `false` | Allow a MAST address that is not this machine (VPN only) |
| `actor` | `claude-code` | Name your actions are attributed to in MAST's records |

### Environment variables

Run by hand, the server reads the same settings from `MAST_URL`, `MAST_USER`, `MAST_PASSWORD`,
`MAST_VERIFY_TLS`, `MAST_ALLOW_REMOTE` and `MAST_ACTOR`. `MAST_FETCH_DIR` sets where `mast_fetch`
saves files; the default is `.mast-fetch/` in the project, which gets its own `.gitignore` so
fetched data stays out of git. `MAST_MCP_LOG` sets the log level (logs go to stderr).

### Checking the connection

Run `/mcp` in Claude Code: `mast` should be listed as connected. If it shows "Failed to connect"
with "Connection closed", the `python` option does not point at a working interpreter (see the
Windows notes). Then ask Claude to call `mast_status`. If the answer is an error, it says what to
check: whether MAST is running, http or https, the login, or an address that is not this machine.

## Windows notes

- **Python path.** The bare `python` on Windows is often the Microsoft Store placeholder, which
  cannot run the server. Point the `python` option at a real interpreter, for example MAST's
  bundled one in the default installation folder:

```text
C:\MAST\MASTv2\pyruntime\python.exe
```

- **HTTPS and self-signed certificates.** Use `https://` for LAN mode. Prefer a trusted
  certificate; for a confirmed instrument PC in a controlled self-signed setup, `verify_tls=false`
  is available but disables certificate verification.
- **Credentials.** Passwords are marked `sensitive` and stored by Claude Code. The flag does
  not imply an encrypted keychain on every platform; see [the security policy](../../SECURITY.md).
- **This machine only, by default.** Addresses other than `127.0.0.1`, `localhost` and `::1` are
  refused until `allow_remote` is on, because instrument control belongs on this machine or
  inside a VPN (for example `https://192.0.2.10:7862` over a VPN). Never expose MAST to the open
  internet.
- **No proxies.** The server never sends MAST traffic through a system proxy.

## Tools

All endpoints are under `/api/ext/v1`.

| Tool | Endpoint | Purpose |
|---|---|---|
| `mast_briefing` | `GET /briefing` | Whole state in one call; read it first |
| `mast_status` | `GET /status` | Quick state check |
| `mast_find_skills` | `GET /skills/search` | Find skills by action |
| `mast_skill_card` | `GET /skills/{name}` | Parameters, units, safety, duration |
| `mast_run` | `POST /jobs`, `GET /jobs/{id}` | Run a skill as a job and wait for it |
| `mast_job` | `GET /jobs/{id}` | Poll one job |
| `mast_jobs` | `GET /jobs` | List jobs |
| `mast_cancel` | `POST /jobs/{id}/cancel` | Cooperative cancel |
| `mast_emergency_stop` | `POST /estop` | E-STOP and cancel every external job |
| `mast_scope` | `GET /scope`, `POST /scope` | Experiment and sample |
| `mast_list_data` | `GET /data/files` | Recent data files |
| `mast_fetch` | `GET /data/file`, `GET /data/frame` | Save a raw file or a frame locally |
| `mast_note_write` | `POST /notes` | Write a note into MAST's memory |
| `mast_note_search` | `GET /notes` | Search notes |
| `mast_ask_operator` | `POST /requests` | Ask the operator |
| `mast_operator_reply` | `GET /requests`, `GET /requests/{id}` | Read the answers |
| `mast_composite_draft` | `POST /composites/draft` | Check a composite spec |
| `mast_composite_save` | `POST /composites` | Save and register a composite |
| `mast_propose_skill` | `POST /skills/proposals` | Propose Python code for review |
| `mast_handover` | `POST /handover` | Handover report |

The server assigns a 55-second budget to a tool call and allows up to 50 seconds for a job wait.
Longer work continues on MAST as a job; poll it with `mast_job`.
Interrupting a tool call does not cancel the job; `mast_cancel` requests cancellation.

## Guide

The full operator guide is in the [mast-operator references](skills/mast-operator/references/en/README.md),
also served as MCP resources `mast://guide/en/...`.

## Development

From the repository root, use [the documented Python 3.13 environment](../../AGENTS.md#validation-without-hardware).
Here, `python` means that environment's interpreter:

```text
python -m pytest tests/v2/unit/integrations -q
python scripts/sync_external_docs.py --check
```

The guide files under `skills/mast-operator/references/` are copies of `docs/external/`; edit
the originals and run `scripts/sync_external_docs.py`.

## License

MIT, like MAST.
