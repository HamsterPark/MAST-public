# Troubleshooting

Symptom, cause, fix — read top to bottom for whatever you are seeing. For the mechanisms these
refer to, see [Concepts](02-concepts.md) and [Operating rules](03-operating-rules.md); for the
exact shape of an error body, see [the API reference](07-api-reference.md).

## Connecting

| Symptom | Cause | Fix |
|---|---|---|
| `curl` reports `HTTP=000`, or a connection error, against a MAST you know is running in LAN mode | LAN mode serves TLS at the transport layer | Use `https://`, not `http://` |
| `401 Unauthorized` | LAN mode uses HTTP Basic, and either no credentials were sent or they are wrong | Set the login (`mast_user`/`mast_password` in the plugin, `MAST_USER`/`MAST_PASSWORD` as environment variables) |
| `403`, `cross_origin_refused` | The request looked like it came from a browser (it carried an `Origin` or a `Sec-Fetch-*` header) and either crossed origins or reached the service through a hostname instead of a loopback name or a literal IP address | Call the API directly, not from a web page; use `127.0.0.1` or the numeric address instead of a hostname — programmatic clients such as curl and the MCP server are unaffected, since they send neither header |
| `415`, `unsupported_media_type` | A write request — including `POST /estop` and `POST /jobs/{job_id}/cancel` — was sent without `Content-Type: application/json` and a JSON body | Send `Content-Type: application/json` with a JSON body; send `{}` when there is nothing else to say |
| An HTML page from an endpoint expected to return JSON | The wrong path prefix, or a MAST build without the external-agent API | Check `/api/ext/v1`. API errors are JSON, including 404; successful `/data/file` and `/data/frame` downloads return binary data. |
| A TLS certificate error | The instrument PC's certificate is self-signed | Set `verify_tls` to false, or install that certificate locally |
| A remote address is refused | An address other than this machine is refused unless remote access is explicitly allowed | Turn it on only over a VPN — never expose MAST to the open internet |
| The MCP server does not start, or shows "Failed to connect" | The configured Python interpreter is a placeholder (on Windows, often the Microsoft Store's) rather than a real one | Point the interpreter option at a real Python 3.10+ |
| Non-ASCII text prints as garbled characters | The terminal or client is not reading the JSON response as UTF-8 | Decode JSON responses as UTF-8; save file/frame downloads as binary data. |

## Running skills

| Symptom | Cause | Fix |
|---|---|---|
| Every job fails at once, with an error saying `aborted by operator — refusing to start ...` | An abort is active; an emergency latch is reported by `GET /status` as `abort.emergency: true` with `abort.why` | Do not retry jobs, including stop skills, while the latch is set. Inspect the E-STOP reply's `errors` and `retracted`; the latch alone does not prove a successful physical stop. Have the operator verify instrument state and release the latch only after recovery. |
| `503`, a `degraded` list, or a `missing` list | A required subsystem is unavailable: startup, configuration or intentionally absent deployment assets are possible causes | Check the named subsystem and service configuration with the operator. Retry a health check during startup; repeated retries do not supply missing assets. |
| `refused_busy` | Another driver already holds the instrument | Wait, or ask the operator; retrying at once will not make it finish sooner |
| `failed`, `refused_by: sample_gate` | The skill produces data and needs an active sample | Set one with `POST /scope` first |
| `failed`, `refused_by: si_parse` | A dimensioned parameter's value could not be parsed — usually a string that dropped a required SI prefix | Check `si_params` on the skill card; write a value such as `"5n"`, or a plain number in SI base units — the prefix rule applies only to strings |
| `failed`, `refused_by: needs_human_node` | The composite contains a step only a person may confirm, and nobody is listening for that pause on this path | Run it from MAST's own interface, or ask the operator instead |
| `422`, `skill_disabled` | The skill, or a sub-step it uses, needs hardware or a capability switched off on this installation | Pick a different skill, or ask the operator to enable it |
| `429` | Too many jobs are running at once | Wait for one to finish, or cancel one you no longer need. Recognized stop/retract skills such as `StopScan` are exempt from this limit, but other admission and execution checks still apply. |
| `503`, `shutting_down` | The service is shutting down and refuses new jobs | Wait for it to come back, then reconnect |
| `lost_on_restart` | The job was still running when MAST's process restarted; it is never replayed | Read the briefing before deciding whether to submit it again |
| A refusal naming `safe_mode_tip_processing_blocked` | The operator has set SAFE, which refuses autonomous tip processing, including external jobs and composite substeps | Do not retry; ask the operator if the task needs it |
| A refusal describing one of the five hard gates, saying no approval will come | You called, or a composite step called, one of the physically dangerous actions refused unconditionally on this path | Stop retrying — nothing changes this outcome; if it is genuinely needed, a person has to do it from MAST's own interface |
