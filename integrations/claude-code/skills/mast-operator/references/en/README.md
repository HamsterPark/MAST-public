# MAST external-agent guide

MAST drives a real scanning tunnelling microscope (STM) through a Nanonis controller, using a
catalog of registered instrument and analysis actions ("skills") behind a shared safety layer.
This guide is for an outside AI agent (Claude Code or another coding agent) or a human operator
who drives MAST through its **external-agent API**, `/api/ext/v1`, rather than through MAST's own
built-in agents. It introduces the API workflow and execution rules; the operator supplies the
instrument configuration, calibration and permission to act.

> **Validation:** in a completed experiment, an external AI agent operated a real STM through
> MAST 6.4.0's earlier control path. The timestamped record spans 2026-09-17 23:10:37 to
> 2026-09-22 02:08:10 CST (about 99 hours including gaps, not continuous instrument operation).
> This guide describes the **new 6.5.0 `/api/ext/v1` and MCP path**:
> unit and local end-to-end tests use real HTTP, a real MCP server process and a mock instrument;
> hardware validation of this new path is pending. The operator should supervise initial hardware
> sessions on the new interface.

The public source edition includes this guide and the client integration, but not a commissioned
instrument deployment. The connection examples assume an operator has already configured and
started a compatible MAST service. Installing the plugin starts the MCP client bridge, not the
MAST backend or the instrument.

## Who this is for

Anyone driving MAST from outside the application: an AI coding agent connected through the
Claude Code plugin or a project-level MCP server, a script talking plain HTTP, or a person typing
`curl` commands by hand. The rules in this guide bind all of them equally — the API does not know
or care what is on the other end of the connection.

## How it works

Every capability MAST exposes — reading a value, running a scan, shaping a tip, saving data — is a
named **skill** with declared parameters, units and a safety level. External jobs use
`api/direct_exec.py` and `ExecutionContext.run`, sharing execution checks with composite substeps:
abort handling, sample scope, operating mode, restricted physical actions, parameter checks,
preconditions and instrument arbitration. Internal agent and manual entry points reuse these
mechanisms through their own wrappers; their approval policies and check order can differ. The external-agent API
lets you search that catalog, read a skill's full card before running it, submit a run as an
asynchronous **job** you poll to completion, pull raw data files, leave notes and questions for the
operator, and write a handover report when you are done. A **briefing** endpoint gives you, in one
call, context drawn from MAST's own runtime: the tip, cached readings, the operator's preferences,
recent actions and your own open requests. Read its freshness and degradation indicators before acting.

## Reading order

| File | Read this for |
|---|---|
| [01-quickstart.md](01-quickstart.md) | Three ways to connect, and a full first session from health check to handover |
| [02-concepts.md](02-concepts.md) | Skills, the execution choke point, safety levels, operating modes, jobs |
| [03-operating-rules.md](03-operating-rules.md) | The rules that keep a session safe, with the reasons behind each one |
| [04-records-and-context.md](04-records-and-context.md) | What MAST records about your actions, and where it goes |
| [05-authoring-skills.md](05-authoring-skills.md) | Building a composite skill, or proposing a new one |
| [06-contributing-skills.md](06-contributing-skills.md) | Submitting a skill to the community contribution tree |
| [07-api-reference.md](07-api-reference.md) | The generated endpoint-by-endpoint reference |
| [08-troubleshooting.md](08-troubleshooting.md) | Symptom, cause and fix, for when something goes wrong |
