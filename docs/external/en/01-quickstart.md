# Quickstart

This walkthrough connects to an existing MAST service: read the briefing, select the actual
sample, run one read-only skill, and hand over. The operator must first configure the instrument,
start a build with `/api/ext/v1`, and authorize the session. A public-source checkout alone does
not provide that deployment. Read [Concepts](02-concepts.md) next for the ideas
behind what you just did, and [Operating rules](03-operating-rules.md) for the rules with reasons.

## Three ways to connect

All three reach the same API, `/api/ext/v1`, and are bound by the same rules; pick whichever fits
where you are running.

### A. The Claude Code plugin

The bundled plugin gives Claude Code a set of `mast_*` tools, one per endpoint, plus two skills:
`mast-operator` (the working procedure for driving MAST) and `mast-skill-author` (writing
composite skills and Python proposals, and contributing them upstream). Install it with:

```text
/plugin marketplace add <path-to-your-MAST-checkout>
/plugin install mast@mast
```

Replace the path placeholder with the absolute path of your local public-snapshot checkout.

Configure the plugin with these options:

| Option | Default | Meaning |
|---|---|---|
| `python` | `python` | Interpreter that runs the server (3.10+); avoid a bare `python` on Windows if it opens the Microsoft Store — point it at a real interpreter |
| `mast_url` | `http://127.0.0.1:7862` | MAST's address; LAN mode uses `https://` |
| `mast_user` | empty | HTTP Basic user, LAN mode only |
| `mast_password` | empty | HTTP Basic password, LAN mode only |
| `verify_tls` | `true` | `false` accepts a self-signed certificate |
| `allow_remote` | `false` | Allow an address that is not this machine |
| `actor` | `claude-code` | Name your actions are attributed to in MAST's records |

The full option list and its exact defaults live in the plugin's own package (its `README.md`,
right next to the plugin source); this table is a summary. Once connected, ask the agent to read
the briefing — that is the first rule in [Operating rules](03-operating-rules.md).

Python 3.10+ is sufficient for this standard-library MCP bridge; the MAST backend uses Python
3.13 and its own dependencies. The bridge does not require the backend's environment on the
client machine.

### B. A project-level MCP server

Any MCP-compatible client can run the same server directly from a checkout of MAST, without going
through a plugin marketplace: point an MCP configuration at `server/run_server.py` inside the
plugin's directory. Run this way, the server takes its settings from environment variables instead
of the plugin's options:

| Variable | Meaning |
|---|---|
| `MAST_URL` | MAST's address, same as `mast_url` above |
| `MAST_USER` / `MAST_PASSWORD` | HTTP Basic login, LAN mode only |
| `MAST_VERIFY_TLS` | `false` accepts a self-signed certificate |
| `MAST_ALLOW_REMOTE` | Allow an address that is not this machine |
| `MAST_ACTOR` | Name your actions are attributed to |
| `MAST_SESSION` | Session name; a random one is used when this is unset |
| `MAST_FETCH_DIR` | Where a fetched file is saved; defaults to `.mast-fetch/` under `CLAUDE_PROJECT_DIR` when set, otherwise the working directory |

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

### C. Plain HTTP

The API needs no client at all: any HTTP library, or `curl`, can call it directly. This is the
lowest-level way in, and useful for a first look at what the API returns. The walkthrough below
uses it throughout, so you can read the shape of every answer.

## Remote access

By default, everything above talks to MAST on the same machine, over plain HTTP, with no
credentials. Reaching an instrument on another machine is supported only over a VPN, and only with
LAN mode turned on: MAST then serves `https://` (a self-signed certificate) with an HTTP Basic
login. Do not expose MAST to the open internet; there is no other authentication layer.

The plugin enforces the loopback default itself: an address other than this machine is refused
until `allow_remote` is turned on. The plugin has no way to verify that the address you then use is
actually inside a VPN — keeping it there is your own responsibility; never point it at an address
reachable from the open internet. A self-signed certificate needs `verify_tls` set to `false`, or
the certificate installed locally. One more trap worth knowing about in advance: if you type a
**hostname** that happens to sit on a browser's built-in HSTS-preload list, the browser will refuse
to offer a "proceed anyway" option for the self-signed certificate at all — use the numeric IP
address instead in that case.

## First session walkthrough

Paths below are relative to `/api/ext/v1`. Run the steps in order; repeat the poll until the job
reaches a terminal state. These calls read the instrument and write scope and session records.

1. **Health.** `GET /health` — confirms the external surface is up and says which subsystems are
   wired before you rely on them. An `ok: true` response is not proof of a live instrument
   connection; inspect `wired`, `missing`, and the briefing's connection and freshness fields.
2. **Briefing.** `GET /briefing` — read this before anything else. It carries the tip, live
   readings, the operator's preferences, recent actions by anyone, and your own open requests.
3. **Scope.** `POST /scope` — choose or create the experiment and sample. Without a sample, scans
   and spectroscopy require sample scope. Experiment action recording requires an active
   experiment and working storage; check the returned recording status.
4. **Find a skill.** `GET /skills/search` — search by the action you need, not by guessing a name.
5. **Skill card.** `GET /skills/{name}` — read units, bounds, preconditions and measured duration
   when available before the first run of any skill.
6. **Run a read-only job.** `POST /jobs` — submit a harmless read (such as reading the bias) as a
   job; it returns at once with a `job_id`.
7. **Poll.** `GET /jobs/{job_id}?wait_s=30` — wait for the job to reach a terminal state.
8. **Handover.** `POST /handover` — close the session with a summary, even for a short one; it
   teaches the next agent, human or not, what happened.

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

The shell example uses Bash syntax. Replace the example experiment and sample names with the
operator's intended scope; it must describe the sample actually mounted. The job id in the poll
step is a placeholder: use the `job_id` your own submission returned. Use a new `request_id` for
each new intended action, and reuse that same ID only when retrying its submission. See
[the job view](07-api-reference.md) for what every field in these answers means, and
[Records and context](04-records-and-context.md) for what just got written down because you set a
scope first.
