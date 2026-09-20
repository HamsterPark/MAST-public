"""The ``mast_*`` tools, one per endpoint of MAST's external agent API.

Every result is one plain-language summary line followed by compact JSON, never
longer than ``TEXT_LIMIT`` characters. Anything that goes wrong comes back as a
tool error (``isError``) with an explanation, not as a protocol error, so the
agent can read it and act on it.

Time: every call returns within ``CALL_BUDGET_S``. Claude Code moves MCP calls
that run for minutes into the background, and a job that outlives its HTTP
request is exactly what the job model is for: submit, then poll.
"""
from __future__ import annotations

import base64
import hashlib
import json
import logging
import os
import re
import secrets
import time
import urllib.parse
from dataclasses import dataclass

from .client import MastClient, MastError

log = logging.getLogger("mast_mcp.tools")

TEXT_LIMIT = 20_000
CALL_BUDGET_S = 55.0        # hard ceiling per tool call, all HTTP included
SERVER_MAX_WAIT_S = 30      # GET /jobs/{id}?wait_s is capped at 30 by MAST
RUN_DEFAULT_WAIT_S = 20
MAX_WAIT_S = 50
DEFAULT_TIMEOUT_S = 30.0
POST_TIMEOUT_S = 20.0

TERMINAL_STATES = frozenset({"succeeded", "failed", "refused_busy", "cancelled",
                             "crashed", "lost_on_restart"})
FAILED_STATES = TERMINAL_STATES - {"succeeded"}

BRIEFING_SECTIONS = ("status", "scope", "resume", "tip", "instrument", "live", "prefs",
                     "safety", "recent_actions", "recent_files", "alarms", "notes",
                     "recording", "jobs", "operator_requests")

INSTRUCTIONS = """\
MAST drives a real scanning tunneling microscope (Nanonis). Rules:
1. Start with mast_briefing; read it again after a restart, a lost_on_restart job or any surprise.
2. Set experiment and sample with mast_scope first: without a sample, scans and spectroscopy are refused; an active experiment enables experiment-linked recording. Inspect the job's recorded fields.
3. Find skills by the action you need (a verb or Nanonis command: "withdraw", "set bias", "scan frame"), not by guessing names. Read mast_skill_card before a skill's first run.
4. Every action is a job: mast_run submits and waits up to wait_s; while it runs, poll mast_job(job_id, wait_s=30). Never resubmit a running action.
5. refused_busy: someone else holds the instrument and your action did not run. Do not retry at once; check mast_status and wait.
6. lost_on_restart: MAST restarted mid-job and never replays it. Read mast_briefing before submitting again.
7. Never approve human-in-the-loop requests for the operator; ask with mast_ask_operator and keep working on what you can.
8. The operating mode (mast_status: safe/semi/auto/unknown) is the operator's choice and you never change it. SAFE refuses autonomous tip processing (bias pulses, tip shaping), including external jobs and composite substeps; if a task needs it, ask with mast_ask_operator.
9. To stop: mast_cancel acts at the job's next check; a stop skill such as StopScan (via mast_run) is never blocked by the instrument lock; mast_emergency_stop is for real danger and only the operator clears its latch.
10. A failed skill is not an API error, and degraded is not empty: read state, result and degraded before concluding.
11. Finish with the instrument idle (no job running, scan stopped) and a mast_handover.
Guide: mast://guide/en/README.md, mast://guide/zh/README.md."""


class ArgError(Exception):
    """The tool was called with arguments it cannot use."""


# ── registry ─────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class Tool:
    name: str
    title: str
    description: str
    input_schema: dict
    annotations: dict
    handler: object

    def as_mcp(self) -> dict:
        return {"name": self.name, "title": self.title, "description": self.description,
                "inputSchema": self.input_schema,
                "annotations": {"title": self.title, **self.annotations}}


TOOLS: dict[str, Tool] = {}


def _tool(name: str, *, title: str, description: str, properties: dict | None = None,
          required: tuple = (), read_only: bool = False, destructive: bool = False,
          idempotent: bool = False, open_world: bool = False):
    schema: dict = {"type": "object", "properties": properties or {}}
    if required:
        schema["required"] = list(required)
    annotations = {"readOnlyHint": read_only, "destructiveHint": destructive,
                   "idempotentHint": idempotent, "openWorldHint": open_world}

    def register(fn):
        TOOLS[name] = Tool(name, title, description, schema, annotations, fn)
        return fn
    return register


def _str_p(description: str, **extra) -> dict:
    return {"type": "string", "description": description, **extra}


def _int_p(description: str, **extra) -> dict:
    return {"type": "integer", "description": description, **extra}


def _num_p(description: str, **extra) -> dict:
    return {"type": "number", "description": description, **extra}


def _list_p(description: str) -> dict:
    return {"type": "array", "items": {"type": "string"}, "description": description}


# ── call context ─────────────────────────────────────────────────────────────
class CallContext:
    """One tool call: the client, its stop signal and its time budget."""

    def __init__(self, client, config, signal, *, clock=time.monotonic, sleeper=None,
                 budget: float = CALL_BUDGET_S):
        self.client = client
        self.config = config
        self.signal = signal
        self.clock = clock
        self._sleeper = sleeper or signal.wait
        self.deadline = clock() + budget

    def remaining(self) -> float:
        return self.deadline - self.clock()

    def timeout(self, preferred: float = DEFAULT_TIMEOUT_S) -> float:
        left = self.remaining() - 1.0
        if left < 1.0:
            raise MastError("budget", "This tool call ran out of its time budget before MAST "
                            "answered. If a job was involved, look it up with mast_jobs.")
        return min(preferred, left)

    def get(self, path: str, query: dict | None = None, *, timeout: float = DEFAULT_TIMEOUT_S):
        return self.client.call("GET", path, query=query, timeout=self.timeout(timeout))

    def post(self, path: str, body: dict, *, timeout: float = DEFAULT_TIMEOUT_S):
        return self.client.call("POST", path, body=body, timeout=self.timeout(timeout))

    def sleep(self, seconds: float) -> bool:
        """Sleep; True when the call should stop instead."""
        return bool(self._sleeper(seconds))


class Toolbox:
    """What ``stdio_rpc`` calls: list the tools, run one."""

    def __init__(self, config, client=None):
        self.config = config
        self.client = client if client is not None else MastClient(config)

    def list_tools(self) -> list[dict]:
        return [t.as_mcp() for t in TOOLS.values()]

    def has_tool(self, name: str) -> bool:
        return name in TOOLS

    def call_tool(self, name: str, arguments: dict, signal, *, clock=time.monotonic,
                  sleeper=None) -> tuple[str, bool]:
        tool = TOOLS[name]
        if self.config.problem:
            return clip(f"{name} did not run. {self.config.problem}"), True
        ctx = CallContext(self.client, self.config, signal, clock=clock, sleeper=sleeper)
        started = time.monotonic()
        try:
            text, is_error = tool.handler(ctx, dict(arguments or {}))
        except ArgError as exc:
            text, is_error = f"{name}: invalid arguments: {exc}", True
        except MastError as exc:
            text, is_error = render(exc.message, exc.as_dict()), True
        except Exception as exc:  # noqa: BLE001 - reported to the agent, logged in full
            log.exception("tool %s failed", name)
            text, is_error = f"{name} hit an internal error in the MAST MCP server: {exc!r}", True
        log.info("%s -> %s in %.1f s", name, "error" if is_error else "ok",
                 time.monotonic() - started)
        return clip(text), bool(is_error)


# ── rendering ────────────────────────────────────────────────────────────────
def compact(obj) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)


def clip(text: str, limit: int = TEXT_LIMIT) -> str:
    if len(text) <= limit:
        return text
    cut = limit - 160
    return (text[:cut] + f"\n...[truncated {len(text) - cut} characters; ask for less: "
            "fewer sections, a smaller limit or a narrower query]")


def render(summary: str, payload=None, *, body: str | None = None) -> str:
    parts = [summary.strip()]
    if body:
        parts.append(body.rstrip())
    if payload is not None:
        parts.append(compact(payload))
    return "\n".join(parts)


def _one_line(value, limit: int = 300) -> str:
    text = re.sub(r"\s+", " ", str(value)).strip()
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _secs(value) -> str:
    try:
        return f"{float(value):.0f} s"
    except (TypeError, ValueError):
        return f"{value} s"


def _size(n: int) -> str:
    if n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.2f} MB"
    if n >= 1024:
        return f"{n / 1024:.1f} kB"
    return f"{n} bytes"


def _rows(data, *keys: str) -> list:
    """The list inside a listing answer, whichever key it came under."""
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in keys:
            if isinstance(data.get(key), list):
                return data[key]
    return []


def _ok_false(data) -> bool:
    return isinstance(data, dict) and data.get("ok") is False


def _problems(data) -> str:
    probs = data.get("problems") if isinstance(data, dict) else None
    if isinstance(probs, list) and probs:
        return "; ".join(_one_line(p, 200) for p in probs[:5])
    return _one_line(data.get("detail") or data.get("error") or "", 300) if isinstance(data, dict) else ""


# ── argument helpers ─────────────────────────────────────────────────────────
def _seg(value) -> str:
    return urllib.parse.quote(str(value), safe="")


def _str(args: dict, key: str, *, required: bool = False) -> str | None:
    value = args.get(key)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value = str(value)                      # ids handed over as numbers
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ArgError(f"'{key}' is required")
        return None
    if not isinstance(value, str):
        raise ArgError(f"'{key}' must be a string")
    return value.strip()


def _text(args: dict, key: str, *, required: bool = False) -> str | None:
    """Like ``_str`` but keeps inner formatting (notes, code, messages)."""
    value = args.get(key)
    if value is None or (isinstance(value, str) and not value.strip()):
        if required:
            raise ArgError(f"'{key}' is required")
        return None
    if not isinstance(value, str):
        raise ArgError(f"'{key}' must be a string")
    return value


def _num(args: dict, key: str, *, default: float, lo: float, hi: float) -> float:
    value = args.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        raise ArgError(f"'{key}' must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ArgError(f"'{key}' must be a number") from None
    if number != number:  # NaN
        raise ArgError(f"'{key}' must be a number")
    return max(lo, min(hi, number))


def _int(args: dict, key: str, *, default: int, lo: int, hi: int) -> int:
    return int(_num(args, key, default=default, lo=lo, hi=hi))


def _bool(args: dict, key: str, default: bool = False) -> bool:
    value = args.get(key)
    if value is None or value == "":
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str) and value.strip().lower() in ("1", "true", "yes", "on"):
        return True
    if isinstance(value, str) and value.strip().lower() in ("0", "false", "no", "off"):
        return False
    raise ArgError(f"'{key}' must be true or false")


def _obj(args: dict, key: str) -> dict | None:
    value = args.get(key)
    if value is None or value == "":
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except ValueError:
            raise ArgError(f"'{key}' must be a JSON object") from None
    if not isinstance(value, dict):
        raise ArgError(f"'{key}' must be an object")
    return value


def _names(args: dict, key: str) -> list[str] | None:
    """A list of short names; a comma-separated string or JSON array works too."""
    value = args.get(key)
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("["):
            try:
                value = json.loads(text)
            except ValueError:
                value = text
        if isinstance(value, str):
            value = value.split(",")
    if not isinstance(value, list):
        raise ArgError(f"'{key}' must be a list of strings")
    out = [str(v).strip() for v in value if str(v).strip()]
    return out or None


def _lines(args: dict, key: str) -> list[str] | None:
    """A list of sentences; a string becomes one item per non-empty line."""
    value = args.get(key)
    if value is None or value == "" or value == []:
        return None
    if isinstance(value, str):
        value = [re.sub(r"^\s*(?:[-*]|\d+[.)])\s+", "", line) for line in value.splitlines()]
    if not isinstance(value, list):
        raise ArgError(f"'{key}' must be a list of strings")
    out = [str(v).strip() for v in value if str(v).strip()]
    return out or None


def _choice(args: dict, key: str, allowed: tuple, default: str | None = None) -> str | None:
    value = _str(args, key)
    if value is None:
        return default
    value = value.lower()
    if value not in allowed:
        raise ArgError(f"'{key}' must be one of {', '.join(allowed)}")
    return value


def _spec(args: dict):
    value = args.get("spec")
    if value is None or (isinstance(value, str) and not value.strip()):
        raise ArgError("'spec' is required (pass '?' to get the syntax)")
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text in ("?", "？", "help"):
            return "?"
        try:
            parsed = json.loads(text)
        except ValueError:
            return text                         # MAST reports exactly what is wrong
        return parsed if isinstance(parsed, dict) else text
    raise ArgError("'spec' must be a JSON object (or '?' for the syntax)")


# ── jobs ─────────────────────────────────────────────────────────────────────
def _terminal(view) -> bool:
    return isinstance(view, dict) and (view.get("terminal") is True
                                       or view.get("state") in TERMINAL_STATES)


def _wait_for_job(ctx: CallContext, job_id: str, wait_s: float, view: dict) -> dict:
    """Long-poll ``GET /jobs/{id}?wait_s`` until the job ends or ``wait_s`` is spent.

    MAST caps one wait at 30 s, so a longer wait takes two requests. Everything
    stays inside the call's budget, with room left to answer.
    """
    end = min(ctx.clock() + wait_s, ctx.deadline - 3.0)
    path = f"/jobs/{_seg(job_id)}"
    while not _terminal(view):
        if ctx.signal.should_stop():
            break
        left = end - ctx.clock()
        if left < 1.0:
            break
        wait = int(min(SERVER_MAX_WAIT_S, left))
        asked_at = ctx.clock()
        view = ctx.get(path, {"wait_s": wait}, timeout=wait + 8)
        # A MAST that ignores wait_s answers at once; do not hammer it.
        if not _terminal(view) and ctx.clock() - asked_at < 0.5:
            if ctx.sleep(min(1.0, max(0.0, end - ctx.clock()))):
                break
    return view


#: What to do when MAST refuses a submission outright (no job was created).
_SUBMIT_REFUSALS = {
    "request_id_conflict": (
        "That request_id was already used for a different skill or different params. Reuse "
        "an id only to resubmit the same action after a timeout; a new action needs a new id."),
    "skill_disabled": (
        "The skill, or one of its declared steps, is switched off on this machine (hardware "
        "module absent or capability not granted). Choose another skill or ask the operator; "
        "wrapping it in a composite does not switch it on."),
}


def _refusal_hint(code: str) -> str:
    c = code.lower()
    if "human" in c:
        return ("the composite contains a step that needs a person; ask the operator to run "
                "it in MAST")
    if "sample" in c:
        return "the skill needs an active sample; set one with mast_scope(sample_name=...)"
    if "si" in c.split("_") or "param" in c or "parse" in c:
        return ("a parameter value could not be read; check units and si_params in the skill "
                "card, and write values such as '5n' where a prefix is required")
    if "mode" in c or "safe" in c:
        return "the current operating mode does not allow this action; ask the operator"
    if "disabled" in c:
        return "this capability is switched off on this machine"
    if "wired" in c:
        return ("MAST could not set up the execution (a subsystem is not wired, often while "
                "starting); check mast_status and tell the operator if it persists")
    return ""


def job_summary(view: dict, *, cancel_request: bool = False, replayed: bool | None = None,
                retried: bool = False) -> tuple[str, bool]:
    """Summary line for a job view, and whether it counts as a failure.

    ``replayed`` carries the submit answer's ``idempotent_replay`` past the
    polls (a poll's view does not repeat it); ``retried`` marks a resubmission
    this server made itself after MAST's first answer was lost.
    """
    state = str(view.get("state") or "unknown")
    job_id = view.get("job_id", "?")
    head = f"Job {job_id} ({view.get('skill', '?')}): {state}"
    elapsed = view.get("elapsed_s")
    if isinstance(elapsed, (int, float)) and not isinstance(elapsed, bool):
        head += f" after {elapsed:.1f} s"
    result = view.get("result") if isinstance(view.get("result"), dict) else {}
    notes: list[str] = []
    if replayed is None:
        replayed = bool(view.get("idempotent_replay"))
    if replayed and retried:
        notes.append("MAST's first answer was lost in transit; resubmitting with the same "
                     "request_id returned this same job, so nothing ran twice")
    elif replayed:
        notes.append("this request_id was used before, so this is the ORIGINAL job; "
                     "nothing new was run")
    if state == "succeeded":
        if result.get("summary"):
            notes.append(_one_line(result["summary"]))
    elif state in ("queued", "running"):
        notes.append(f'still {state}: keep waiting with mast_job(job_id="{job_id}", '
                     "wait_s=30); do not submit it again")
    elif state == "refused_busy":
        holder = view.get("busy_holder")
        if isinstance(holder, dict):
            who = holder.get("owner") or "another user"
            if holder.get("skill"):
                who += f" running {holder['skill']}"
            if holder.get("held_s") is not None:
                who += f" for {_secs(holder['held_s'])}"
        else:
            who = str(holder or "another user")
        notes.append(f"NOT executed and not queued: the instrument is held by {who}. Do not "
                     "retry at once; check mast_status and submit again after that work ends")
    elif state == "failed":
        refused_by = view.get("refused_by")
        if refused_by:
            hint = _refusal_hint(str(refused_by))
            notes.append(f"refused by {refused_by}" + (f" ({hint})" if hint else ""))
        err = result.get("error") or (result.get("summary") if not refused_by else "")
        if err:
            notes.append(_one_line(err, 400))
    elif state == "cancelled":
        reason = view.get("cancel_reason")
        notes.append("cancelled" + (f": {_one_line(reason, 200)}" if reason else ""))
    elif state == "crashed":
        notes.append("the job crashed inside MAST and its outcome is unknown; read "
                     "mast_briefing (alarms, recent actions) before anything else and tell "
                     "the operator")
    elif state == "lost_on_restart":
        notes.append("MAST restarted while this job was in flight; it was NOT replayed and its "
                     "outcome is unknown. Read mast_briefing before deciding to submit it again")
    abort = view.get("abort")
    if isinstance(abort, dict) and abort.get("set"):
        reason = abort.get("reason") or abort.get("why")
        notes.append("the abort flag is set" + (f" ({_one_line(reason, 200)})" if reason else "")
                     + "; write actions are refused until the operator clears it")
    recorded = view.get("recorded")
    if _terminal(view) and isinstance(recorded, dict) and recorded.get("v1") is False:
        notes.append("Legacy experiment record NOT recorded (recorded.v1=false); check "
                     "mast_scope and the other recorded fields for this job")
    summary = head + (". " + "; ".join(notes) if notes else "")
    return summary, (state in FAILED_STATES and not cancel_request)


def _job_result(view, **kw) -> tuple[str, bool]:
    if not isinstance(view, dict):
        return render("MAST returned something that is not a job view.", view), True
    summary, is_error = job_summary(view, **kw)
    return render(summary, view), is_error


# ── tools ────────────────────────────────────────────────────────────────────
@_tool("mast_briefing", title="MAST briefing", read_only=True, idempotent=True, description=(
    "Read this first in every session, and again after any restart or surprise. One call "
    "returns MAST's current state as sectioned text: operating mode and abort flag, instrument "
    "lock, experiment/sample scope and whether actions are being recorded, resume notes, tip "
    "state, instrument profile and learned calibrations, live readings, the operator's "
    "parameter preferences, safety envelope, recent actions and files, alarms, notes, your jobs "
    "and your operator requests. No hardware I/O. Keywords: 简报, 状态, 续工, 针尖, 仪器档案."),
    properties={"sections": _list_p(
        "Optional subset of sections: " + ", ".join(BRIEFING_SECTIONS) + ". Omit for all.")})
def _briefing(ctx: CallContext, args: dict):
    data = ctx.get("/briefing", {"sections": _names(args, "sections")})
    if not isinstance(data, dict):
        return render("MAST briefing", data), False
    sections = data.get("sections")
    degraded = [str(d.get("section") or d) if isinstance(d, dict) else str(d)
                for d in (data.get("degraded") or [])]
    body = data.get("text") or ""
    rest = {k: v for k, v in data.items() if k != "text"}
    if isinstance(sections, dict):
        if not body:
            body = "\n\n".join(f"## {name}\n{sec.get('text')}" for name, sec in sections.items()
                               if isinstance(sec, dict) and sec.get("text"))
        rest["sections"] = {name: ({k: v for k, v in sec.items() if k != "text"}
                                   if isinstance(sec, dict) else sec)
                            for name, sec in sections.items()}
        count = sum(1 for sec in sections.values() if isinstance(sec, dict) and sec.get("ok"))
        summary = f"MAST briefing: {count} of {len(sections)} sections ok"
    else:
        summary = "MAST briefing"
    if degraded:
        summary += f"; degraded: {', '.join(degraded)}"
    return render(summary, rest, body=body), False


def status_summary(d: dict) -> str:
    parts = [f"mode {d.get('mode') or 'unknown'}"]
    abort = d.get("abort") if isinstance(d.get("abort"), dict) else {}
    if "abort" in d and d.get("abort") is None:
        parts.append("abort state UNREADABLE (do not assume it is clear)")
    elif abort.get("set") or abort.get("emergency"):
        why = abort.get("why") or abort.get("reason")
        parts.append("ABORT SET" + (" (emergency)" if abort.get("emergency") else "")
                     + (f": {_one_line(why, 200)}" if why else "")
                     + "; write actions are refused until the operator clears it")
    else:
        parts.append("abort clear")
    lock = d.get("lock") if isinstance(d.get("lock"), dict) else {}
    if lock.get("held"):
        text = f"instrument held by {lock.get('owner') or 'someone'}"
        if lock.get("skill"):
            text += f" running {lock['skill']}"
        if lock.get("held_s") is not None:
            text += f" for {_secs(lock['held_s'])}"
        parts.append(text)
    elif "lock" in d:
        parts.append("instrument free")
    jobs = d.get("jobs") if isinstance(d.get("jobs"), dict) else {}
    if jobs:
        parts.append(f"external jobs running {jobs.get('running', '?')}/{jobs.get('max', '?')}")
    degraded = d.get("degraded")
    if degraded:
        parts.append("degraded: " + ", ".join(str(x) for x in degraded))
    return "MAST status: " + "; ".join(parts)


@_tool("mast_status", title="MAST status", read_only=True, idempotent=True, description=(
    "Quick state check, cheap and without hardware I/O: operating mode (SAFE/SEMI/AUTO), abort "
    "flag and why, who holds the instrument lock and for how long, connection, active scope, "
    "running jobs against the limit, degraded subsystems. Use it before submitting work and "
    "after refused_busy. Keywords: 状态, 锁, 急停, 模式."))
def _status(ctx: CallContext, args: dict):
    data = ctx.get("/status")
    summary = status_summary(data) if isinstance(data, dict) else "MAST status"
    return render(summary, data), False


@_tool("mast_find_skills", title="Find MAST skills", read_only=True, idempotent=True,
       description=(
    "Search MAST's skills (instrument actions and analyses). Search by the ACTION you need, a "
    "verb or Nanonis command such as 'withdraw', 'set bias', 'z controller', 'scan frame', "
    "'bias spectroscopy', not by guessing a name: many skills are named after the problem they "
    "solve. Each hit shows origin (official or not), safety level, footprint (pure-analysis / "
    "hardware-read-only / hardware-write) and why it matched. Prefer official skills. "
    "Keywords: 技能, 查找, 动作."),
    properties={"query": _str_p("What you want to do, as a verb or command."),
                "limit": _int_p("Maximum hits (1-100).", default=20, minimum=1, maximum=100)},
    required=("query",))
def _find_skills(ctx: CallContext, args: dict):
    query = _str(args, "query", required=True)
    data = ctx.get("/skills/search", {"q": query,
                                      "limit": _int(args, "limit", default=20, lo=1, hi=100)})
    hits = _rows(data, "results")
    names = [str(h.get("name")) for h in hits if isinstance(h, dict) and h.get("name")]
    count = data.get("count", len(hits)) if isinstance(data, dict) else len(hits)
    if names:
        shown = ", ".join(names[:10]) + (", ..." if len(names) > 10 else "")
        summary = f'{count} skill(s) match "{query}": {shown}. Read one with mast_skill_card.'
    else:
        summary = (f'No skill matches "{query}". Try another verb, a Nanonis command name, or a '
                   "broader word.")
    return render(summary, data), False


@_tool("mast_skill_card", title="MAST skill card", read_only=True, idempotent=True,
       description=(
    "Full card for one skill: parameters with type, unit, required, default, min/max and "
    "choices (and which accept SI strings like '5n'), preconditions, safety level, footprint, "
    "the Nanonis commands it sends, sub-skills, whether it takes the instrument lock or needs "
    "an active sample, whether it is switched off on this machine, and how long it usually "
    "takes. Read it before a skill's first run. Keywords: 技能卡, 参数, 单位."),
    properties={"name": _str_p("Exact skill name, as returned by mast_find_skills.")},
    required=("name",))
def _skill_card(ctx: CallContext, args: dict):
    name = _str(args, "name", required=True)
    card = ctx.get(f"/skills/{_seg(name)}")
    if not isinstance(card, dict):
        return render(f"Skill {name}", card), False
    bits = [f"Skill {card.get('name', name)}"]
    for key in ("safety_level", "footprint"):
        if card.get(key):
            bits.append(f"{key} {card[key]}")
    if card.get("takes_instrument_token") is True:
        bits.append("takes the instrument lock")
    if card.get("requires_sample"):
        bits.append("needs an active sample (mast_scope)")
    face = card.get("tool_face")
    if isinstance(face, dict) and (face.get("disabled") or face.get("closed")):
        bits.append("SWITCHED OFF on this machine"
                    + (f" ({_one_line(face.get('reason'), 160)})" if face.get("reason") else ""))
    elif isinstance(face, dict) and "disabled" in face and face["disabled"] is None:
        bits.append("whether it is switched off on this machine is UNKNOWN")
    duration = card.get("duration") if isinstance(card.get("duration"), dict) else {}
    measured = duration.get("measured") if isinstance(duration.get("measured"), dict) else None
    if measured and measured.get("p50_s") is not None:
        bits.append(f"typically {_secs(measured['p50_s'])} (p95 {_secs(measured.get('p95_s'))}, "
                    f"n={measured.get('n', '?')})")
    elif duration.get("estimated_s") is not None:
        bits.append(f"estimated {_secs(duration['estimated_s'])}")
    return render("; ".join(bits), card), False


@_tool("mast_run", title="Run a MAST skill", destructive=True, open_world=True, description=(
    "Run a skill on MAST as a job: submit it, then wait up to wait_s seconds (default 20, max "
    "50) for it to end. If it is still running when this returns, keep polling with "
    "mast_job(job_id, wait_s=30); never submit it again. End states: succeeded, failed, "
    "refused_busy (someone else holds the instrument; NOT executed, do not retry at once), "
    "cancelled, crashed, lost_on_restart (MAST restarted; never replayed). The answer also says "
    "the recorded fields for each recording path; experiment-linked records require an "
    "active experiment (see mast_scope). "
    "Safety gates, the operating mode (SAFE refuses tip shaping and pulses) and the abort flag "
    "all apply. Interrupting this call does not cancel the job. "
    "Keywords: 执行, 运行, 作业, 扫描, 偏压, 进针, 退针."),
    properties={
        "skill": _str_p("Exact skill name (from mast_find_skills)."),
        "params": {"type": "object", "additionalProperties": True, "description": (
            "Skill parameters as a JSON object. Units are SI (V, A, m, s); parameters the "
            "skill card lists under si_params also accept strings such as '5n' or '200p' (the "
            "card says where the prefix is required).")},
        "request_id": _str_p(
            "Idempotency key: submitting again with the same request_id returns the original "
            "job instead of running the action twice. Use one whenever you might retry after "
            "a timeout. Generated automatically when omitted."),
        "wait_s": _num_p("Seconds to wait for the job to end (0-50).", default=20, minimum=0,
                         maximum=MAX_WAIT_S),
        "note": _str_p("Optional note stored with the job: why you ran it."),
    },
    required=("skill",))
def _run(ctx: CallContext, args: dict):
    body: dict = {"skill": _str(args, "skill", required=True),
                  "params": _obj(args, "params") or {},
                  "request_id": (_str(args, "request_id")
                                 or f"mcp-{ctx.config.session}-{secrets.token_hex(6)}")}
    note = _text(args, "note")
    if note:
        body["note"] = note
    wait_s = _num(args, "wait_s", default=RUN_DEFAULT_WAIT_S, lo=0, hi=MAX_WAIT_S)
    retried = False
    try:
        try:
            view = ctx.post("/jobs", body, timeout=POST_TIMEOUT_S)
        except MastError as exc:
            # The request may or may not have reached MAST. The request_id makes a second
            # attempt safe: MAST answers with the original job instead of running it twice.
            if not exc.transient or ctx.signal.should_stop() or ctx.remaining() < 15:
                raise
            log.warning("POST /jobs failed (%s); retrying once with request_id %s",
                        exc.kind, body["request_id"])
            view = ctx.post("/jobs", body, timeout=POST_TIMEOUT_S)
            retried = True
    except MastError as exc:
        hint = _SUBMIT_REFUSALS.get(exc.kind)
        if hint:
            raise MastError(exc.kind, f"{exc.message} {hint}", status=exc.status,
                            payload=exc.payload) from None
        raise
    replayed = isinstance(view, dict) and bool(view.get("idempotent_replay"))
    if isinstance(view, dict) and view.get("job_id") and not _terminal(view) and wait_s >= 1:
        view = _wait_for_job(ctx, str(view["job_id"]), wait_s, view)
    return _job_result(view, replayed=replayed, retried=retried)


@_tool("mast_job", title="MAST job status", read_only=True, idempotent=True, description=(
    "Look at one job from mast_run. With wait_s > 0 (max 50) it blocks until the job ends or "
    "the time is up: use that instead of polling in a tight loop. Interrupting this call does "
    "not cancel the job; mast_cancel does. Keywords: 作业, 轮询, 结果."),
    properties={"job_id": _str_p("The job_id from mast_run."),
                "wait_s": _num_p("Seconds to wait for the job to end (0-50).", default=0,
                                 minimum=0, maximum=MAX_WAIT_S)},
    required=("job_id",))
def _job(ctx: CallContext, args: dict):
    job_id = _str(args, "job_id", required=True)
    wait_s = _num(args, "wait_s", default=0, lo=0, hi=MAX_WAIT_S)
    view: dict = {}
    if wait_s >= 1:
        view = _wait_for_job(ctx, job_id, wait_s, view)
    if not view:
        view = ctx.get(f"/jobs/{_seg(job_id)}")
    return _job_result(view)


@_tool("mast_jobs", title="List MAST jobs", read_only=True, idempotent=True, description=(
    "List jobs submitted under this actor name (all=true: every actor's), newest first, "
    "optionally filtered by state: queued, running, succeeded, failed, refused_busy, "
    "cancelled, crashed, lost_on_restart. Keywords: 作业列表."),
    properties={"all": {"type": "boolean", "default": False,
                        "description": "Include other actors' jobs."},
                "state": _str_p("Only jobs in this state."),
                "limit": _int_p("Maximum jobs (1-200).", default=50, minimum=1, maximum=200)})
def _jobs(ctx: CallContext, args: dict):
    data = ctx.get("/jobs", {"all": _bool(args, "all"), "state": _str(args, "state"),
                             "limit": _int(args, "limit", default=50, lo=1, hi=200)})
    jobs = [j for j in _rows(data, "jobs", "items", "results") if isinstance(j, dict)]
    counts: dict[str, int] = {}
    for j in jobs:
        counts[str(j.get("state"))] = counts.get(str(j.get("state")), 0) + 1
    summary = f"{len(jobs)} job(s)"
    if counts:
        summary += ": " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
    return render(summary, data), False


@_tool("mast_cancel", title="Cancel a MAST job", idempotent=True, description=(
    "Ask a running job to stop. Cancellation is cooperative: it takes effect between skills, "
    "between sub-steps and before each write command, not inside one blocking Nanonis command. "
    "To request a stop, run a skill such as StopScan with mast_run; stop skills bypass the "
    "instrument ownership lock. Check the result and instrument feedback; for real danger "
    "use mast_emergency_stop. "
    "Keywords: 取消, 停止作业."),
    properties={"job_id": _str_p("The job to cancel."),
                "reason": _str_p("Why; stored with the job.")},
    required=("job_id",))
def _cancel(ctx: CallContext, args: dict):
    job_id = _str(args, "job_id", required=True)
    reason = _text(args, "reason")
    view = ctx.post(f"/jobs/{_seg(job_id)}/cancel", {"reason": reason} if reason else {})
    if not isinstance(view, dict):
        return render("Cancel requested.", view), False
    summary, _ = job_summary(view, cancel_request=True)
    if not _terminal(view):
        summary = ("Cancel requested; it takes effect at the next step boundary. Check with "
                   "mast_job(wait_s=30). " + summary)
    return render(summary, view), False


@_tool("mast_emergency_stop", title="MAST emergency stop", destructive=True, idempotent=True,
       open_world=True, description=(
    "EMERGENCY STOP: the same action as MAST's E-STOP (stops motion and scanning, raises the "
    "abort flag) plus cancelling every external job. Only for real danger: crash risk, runaway "
    "current, anything that cannot be stopped otherwise. Afterwards write actions stay refused "
    "until the operator clears the abort flag; tell the operator what happened "
    "(mast_ask_operator). Keywords: 急停, 紧急停止, E-STOP."),
    properties={"reason": _str_p("What is going wrong; shown to the operator.")},
    required=("reason",))
def _estop(ctx: CallContext, args: dict):
    data = ctx.post("/estop", {"reason": _text(args, "reason", required=True)})
    cancelled = data.get("cancelled_jobs") if isinstance(data, dict) else None
    summary = "EMERGENCY STOP sent to MAST"
    if isinstance(cancelled, list):
        summary += f"; {len(cancelled)} external job(s) cancelled"
    summary += (". Write actions stay refused while the abort flag is set, and only the "
                "operator clears it: tell them what happened (mast_ask_operator) and check "
                "mast_status.")
    return render(summary, data), _ok_false(data)


def scope_summary(d: dict) -> str:
    scope = d.get("scope") if isinstance(d.get("scope"), dict) and "experiment" not in d else d
    exp, smp = scope.get("experiment"), scope.get("sample")
    rec = scope.get("recording") if isinstance(scope.get("recording"), dict) else {}
    if isinstance(exp, dict):
        e = f'experiment "{exp.get("name", "?")}" (id {exp.get("id", "?")})'
    else:
        e = "no active experiment"
    if isinstance(smp, dict):
        kind = f", {smp['sample_type']}" if smp.get("sample_type") else ""
        s = f'sample "{smp.get("name", "?")}" (id {smp.get("id", "?")}{kind})'
    else:
        s = "no active sample"
    out = f"Scope: {e}; {s}"
    if rec:
        out += (f"; recording v1 {'on' if rec.get('v1') else 'off'}, "
                f"v2 {'on' if rec.get('v2') else 'off'}")
        if rec.get("note"):
            out += f" ({_one_line(rec['note'], 200)})"
    changed = d.get("changed")
    if isinstance(changed, list) and changed:
        out += f"; changed: {', '.join(map(str, changed))}"
    warnings = d.get("warnings")
    if isinstance(warnings, list) and warnings:
        out += f"; warnings: {'; '.join(_one_line(w, 200) for w in warnings)}"
    if not isinstance(exp, dict):
        out += (". Actions are NOT recorded until an experiment is active: "
                "mast_scope(experiment_name=..., goal=..., sample_name=...)")
    return out


@_tool("mast_scope", title="MAST experiment and sample", idempotent=True, description=(
    "Show or set the active experiment and sample. Without arguments: the current scope and "
    "whether actions are being recorded. With arguments: switch to an existing experiment or "
    "sample by id, or open one by name (an open one with the same name is reused; a new sample "
    "needs an experiment, and both can be given in one call). Do this before running actions; "
    "an active experiment is needed for experiment-linked recording. Check each job's "
    "recorded fields. While the instrument is busy MAST "
    "refuses the switch unless force=true. Keywords: 实验, 样品, 记录."),
    properties={
        "experiment_id": _str_p("Switch to this existing experiment."),
        "experiment_name": _str_p("Open (or reuse) an experiment with this name."),
        "goal": _str_p("Goal of the experiment; only with experiment_name."),
        "sample_id": _str_p("Switch to this existing sample."),
        "sample_name": _str_p("Open (or reuse) a sample with this name."),
        "sample_type": _str_p("Kind of sample; only with sample_name."),
        "sample_description": _str_p("Free description; only with sample_name."),
        "force": {"type": "boolean", "default": False, "description": (
            "Switch even though the instrument is busy. The running action is then filed "
            "under the NEW scope; use it only when that is what you want.")},
        "reason": _str_p("Why the scope changes; stored with the switch."),
    })
def _scope(ctx: CallContext, args: dict):
    exp_id, exp_name = _str(args, "experiment_id"), _str(args, "experiment_name")
    goal = _text(args, "goal")
    s_id, s_name = _str(args, "sample_id"), _str(args, "sample_name")
    s_type, s_desc = _str(args, "sample_type"), _text(args, "sample_description")
    force, reason = _bool(args, "force"), _text(args, "reason")
    if exp_id and exp_name:
        raise ArgError("give experiment_id (switch to an existing experiment) or "
                       "experiment_name (open one by name), not both")
    if goal and not exp_name:
        raise ArgError("goal only goes with experiment_name")
    if s_id and s_name:
        raise ArgError("give sample_id or sample_name, not both")
    if (s_type or s_desc) and not s_name:
        raise ArgError("sample_type and sample_description only go with sample_name")
    body: dict = {}
    if exp_id:
        body["experiment"] = {"id": exp_id}
    elif exp_name:
        body["experiment"] = {"name": exp_name, **({"goal": goal} if goal else {})}
    if s_id:
        body["sample"] = {"id": s_id}
    elif s_name:
        body["sample"] = {"name": s_name,
                          **({"sample_type": s_type} if s_type else {}),
                          **({"description": s_desc} if s_desc else {})}
    if not body:
        if force or reason:
            raise ArgError("force and reason only go with a switch (an experiment or a sample)")
        data = ctx.get("/scope")
    else:
        if force:
            body["force"] = True
        if reason:
            body["reason"] = reason
        try:
            data = ctx.post("/scope", body)
        except MastError as exc:
            if exc.kind == "instrument_busy":
                raise MastError(exc.kind, exc.message + (
                    " Switching now would file the running action under the new scope: wait "
                    "until the instrument is free (mast_status), or pass force=true if that "
                    "is intended."), status=exc.status, payload=exc.payload) from None
            raise
    summary = scope_summary(data) if isinstance(data, dict) else "Scope"
    return render(summary, data), _ok_false(data)


@_tool("mast_list_data", title="List MAST data files", read_only=True, idempotent=True,
       description=(
    "List the most recent data files MAST can serve (.sxm images, .dat spectra and so on), "
    "newest first. Pass a returned path to mast_fetch. Keywords: 数据, 文件, 图像, 谱."),
    properties={"n": _int_p("How many files (1-200).", default=20, minimum=1, maximum=200),
                "ext": _list_p("Only these extensions, e.g. [\"sxm\", \"dat\"]."),
                "offset": _int_p("Skip this many of the newest files (paging).", default=0,
                                 minimum=0)})
def _list_data(ctx: CallContext, args: dict):
    ext = _names(args, "ext")
    if ext:
        ext = [e.lstrip(".").lower() for e in ext]
    data = ctx.get("/data/files", {"n": _int(args, "n", default=20, lo=1, hi=200), "ext": ext,
                                   "offset": _int(args, "offset", default=0, lo=0, hi=10 ** 6)})
    files = [f for f in _rows(data, "files", "items", "results") if isinstance(f, dict)]
    names = [re.split(r"[\\/]", str(f.get("path") or f.get("name") or ""))[-1] for f in files]
    summary = f"{len(files)} data file(s)"
    if names:
        summary += ": " + ", ".join(names[:8]) + (", ..." if len(names) > 8 else "")
        summary += ". Download one with mast_fetch(path=...)"
    return render(summary, data), False


_WINDOWS_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)),
                     *(f"lpt{i}" for i in range(1, 10))}


def _safe_part(text: str) -> str:
    return re.sub(r"[^\w.-]+", "_", text).strip("._")


def local_fetch_path(fetch_dir: str, remote: str, mode: str, channel: str | None) -> str:
    """Where a fetched file lands: ``<stem>.<hash6>[.<channel>].<ext>`` in ``fetch_dir``.

    The name is built here and never taken from MAST or the caller as a path,
    so nothing can land outside ``fetch_dir``. The short hash of the remote path
    keeps two ``Topo001.sxm`` from different folders apart; fetching the same
    file again overwrites the same local copy.
    """
    base = re.split(r"[\\/]", remote.strip())[-1]
    stem, dot, suffix = base.rpartition(".")
    if not dot:
        stem, suffix = base, ""
    stem = _safe_part(stem)[:80] or "data"
    if stem.lower() in _WINDOWS_RESERVED:
        stem = "_" + stem
    tag = hashlib.sha1(remote.encode("utf-8")).hexdigest()[:6]
    if mode == "frame":
        name = f"{stem}.{tag}.{_safe_part(channel or 'Z')[:32] or 'Z'}.npz"
    else:
        suffix = _safe_part(suffix)[:12]
        name = f"{stem}.{tag}" + (f".{suffix}" if suffix else "")
    return os.path.join(fetch_dir, name)


def _ensure_fetch_dir(config) -> None:
    existed = os.path.isdir(config.fetch_dir)
    try:
        os.makedirs(config.fetch_dir, exist_ok=True)
    except OSError as exc:
        raise MastError("local_io", f"Cannot create the fetch directory {config.fetch_dir}: "
                        f"{exc}. Set MAST_FETCH_DIR to a writable folder.") from None
    if config.fetch_dir_is_default and not existed:
        # Keep downloaded instrument data out of the project's git history.
        try:
            with open(os.path.join(config.fetch_dir, ".gitignore"), "w", encoding="utf-8") as fh:
                fh.write("# Created by the MAST MCP server: fetched data stays out of git.\n*\n")
        except OSError:
            pass


def _frame_meta(header: str | None):
    if not header:
        return None
    try:
        header = header.encode("latin-1").decode("utf-8")   # http.client decodes as latin-1
    except (UnicodeEncodeError, UnicodeDecodeError):
        pass
    for decode in (lambda h: h, lambda h: base64.b64decode(h, validate=True).decode("utf-8"),
                   urllib.parse.unquote):
        try:
            return json.loads(decode(header))
        except (ValueError, UnicodeDecodeError):
            continue
    return header


@_tool("mast_fetch", title="Download MAST data", description=(
    "Download one data file to this machine and return its local path (the content is not put "
    "into the conversation). mode='raw' saves the original bytes; mode='frame' saves an .npz "
    "with 'forward' and 'backward' arrays for one channel, orientation already normalized by "
    "MAST, plus 'meta_json'. Files go to MAST_FETCH_DIR (default .mast-fetch/ in the project). "
    "Keywords: 下载, 原始数据, 帧, npz."),
    properties={"path": _str_p("A path exactly as returned by mast_list_data."),
                "mode": _str_p("raw (original file) or frame (npz arrays).", enum=["raw", "frame"],
                               default="raw"),
                "channel": _str_p("Channel for mode=frame, e.g. Z or Current.", default="Z")},
    required=("path",))
def _fetch(ctx: CallContext, args: dict):
    remote = _str(args, "path", required=True)
    mode = _choice(args, "mode", ("raw", "frame"), default="raw")
    channel = (_str(args, "channel") or "Z") if mode == "frame" else None
    dest = local_fetch_path(ctx.config.fetch_dir, remote, mode, channel)
    _ensure_fetch_dir(ctx.config)
    endpoint = "/data/file" if mode == "raw" else "/data/frame"
    query = {"path": remote, "channel": channel}
    info = ctx.client.download(endpoint, query, dest, timeout=ctx.timeout(DEFAULT_TIMEOUT_S),
                               should_stop=ctx.signal.should_stop, deadline=ctx.deadline - 1.0,
                               clock=ctx.clock)
    payload: dict = {"saved_to": dest, "bytes": info["bytes"], "sha256": info["sha256"],
                     "content_type": info["content_type"], "remote_path": remote, "mode": mode}
    remote_name = re.split(r"[\\/]", remote)[-1]
    if mode == "frame":
        payload["channel"] = channel
        payload["meta"] = _frame_meta(info["headers"].get("X-MAST-Frame-Meta"))
        summary = (f"Saved channel {channel} of {remote_name} ({_size(info['bytes'])}) to {dest}. "
                   "numpy.load(path) gives 'forward', 'backward' (when the channel has one) and "
                   "'meta_json'; row 0 is the top edge and backward is already un-mirrored.")
    else:
        summary = f"Saved {remote_name} ({_size(info['bytes'])}) to {dest}."
    return render(summary, payload), False


@_tool("mast_note_write", title="Write a MAST note", description=(
    "Write a note into MAST's memory. MAST's own agents recall these on their own, so record "
    "findings, calibrations and caveats that others should know. scope='experiment' (default) "
    "ties the note to the active experiment; 'global' keeps it across experiments. "
    "Keywords: 笔记, 记忆, 记录发现."),
    properties={"title": _str_p("Short title."),
                "content": _str_p("The note itself (markdown is fine)."),
                "kind": _str_p("note (default), insight, summary, hypothesis or protocol."),
                "tags": _list_p("Optional tags."),
                "scope": _str_p("experiment (default) or global.", enum=["experiment", "global"])},
    required=("title", "content"))
def _note_write(ctx: CallContext, args: dict):
    title = _str(args, "title", required=True)
    body: dict = {"title": title, "content": _text(args, "content", required=True)}
    kind = _str(args, "kind")
    tags = _names(args, "tags")
    scope = _choice(args, "scope", ("experiment", "global"))
    if kind:
        body["kind"] = kind
    if tags:
        body["tags"] = tags
    if scope:
        body["scope"] = scope
    data = ctx.post("/notes", body)
    if _ok_false(data):
        return render(f"Note not saved: {_problems(data)}", data), True
    ref, warnings = None, []
    if isinstance(data, dict):
        ref = data.get("id") or data.get("path")
        warnings = [str(w) for w in (data.get("warnings") or [])]
    summary = (f'Note "{title}" saved' + (f" ({ref})" if ref else "")
               + (f", namespace {data['namespace']}" if isinstance(data, dict)
                  and data.get("namespace") else "")
               + ". MAST's agents will recall it.")
    if warnings:
        summary += " Warnings: " + "; ".join(_one_line(w, 200) for w in warnings)
    return render(summary, data), False


@_tool("mast_note_search", title="Search MAST notes", read_only=True, idempotent=True,
       description=(
    "Search MAST's notes (meaning and substring). An empty query lists the most recent ones. "
    "Keywords: 笔记, 检索, 回忆."),
    properties={"query": _str_p("What to look for; empty for the most recent notes."),
                "scope": _str_p("both (default), experiment or global.",
                                enum=["both", "experiment", "global"]),
                "limit": _int_p("Maximum notes (1-50).", default=10, minimum=1, maximum=50)})
def _note_search(ctx: CallContext, args: dict):
    query = _str(args, "query")
    data = ctx.get("/notes", {"q": query,
                              "scope": _choice(args, "scope", ("both", "experiment", "global"),
                                               default="both"),
                              "limit": _int(args, "limit", default=10, lo=1, hi=50)})
    notes = _rows(data, "notes", "results", "items")
    summary = f"{len(notes)} note(s)" + (f' for "{query}"' if query else ", most recent first")
    return render(summary, data), False


@_tool("mast_ask_operator", title="Ask the MAST operator", description=(
    "Send a question or request to the human operator; it appears in MAST's request list. It "
    "does not block: keep working on what you can and read the answer later with "
    "mast_operator_reply. Use it for anything that needs a person: approvals, sample changes, "
    "actions the current mode refuses. Never approve human-in-the-loop requests yourself. "
    "Keywords: 问操作员, 请求, 人工."),
    properties={"message": _str_p("The question or request, self-contained."),
                "kind": _str_p("question (default), action or info.")},
    required=("message",))
def _ask_operator(ctx: CallContext, args: dict):
    body: dict = {"message": _text(args, "message", required=True)}
    kind = _str(args, "kind")
    if kind:
        body["kind"] = kind
    data = ctx.post("/requests", body)
    if _ok_false(data) or (isinstance(data, dict) and data.get("error")):
        return render(f"The request was not posted: {_problems(data)}", data), True
    rec = data.get("request") if isinstance(data, dict) and isinstance(
        data.get("request"), dict) else data
    rid = (rec.get("id") or rec.get("request_id")) if isinstance(rec, dict) else None
    status = rec.get("status", "pending") if isinstance(rec, dict) else "pending"
    summary = (f"Request {rid or '(no id)'} is with the operator (status {status}). It does not "
               "block: carry on with what you can and read the answer later with "
               f'mast_operator_reply(request_id="{rid}").')
    return render(summary, data), False


@_tool("mast_operator_reply", title="Read operator replies", read_only=True, idempotent=True,
       description=(
    "Read the operator's answers. With request_id: that request, including the reply note or "
    "attached path. Without: every request from this actor name and its status. "
    "Keywords: 答复, 回复."),
    properties={"request_id": _str_p("A request id from mast_ask_operator; omit for all.")})
def _operator_reply(ctx: CallContext, args: dict):
    rid = _str(args, "request_id")
    if rid:
        data = ctx.get(f"/requests/{_seg(rid)}")
        rec = data.get("request") if isinstance(data, dict) and isinstance(
            data.get("request"), dict) else data
        status = rec.get("status", "?") if isinstance(rec, dict) else "?"
        if status == "pending":
            summary = f"Request {rid} is still pending: no answer yet."
        else:
            summary = f"Request {rid}: {status}"
            if isinstance(rec, dict) and rec.get("note"):
                summary += f"; operator's note: {_one_line(rec['note'], 600)}"
            if isinstance(rec, dict) and rec.get("path"):
                summary += f"; attached: {rec['path']}"
        return render(summary, data), False
    data = ctx.get("/requests")
    rows = [r for r in _rows(data, "requests", "items", "results") if isinstance(r, dict)]
    counts: dict[str, int] = {}
    for r in rows:
        counts[str(r.get("status"))] = counts.get(str(r.get("status")), 0) + 1
    summary = f"{len(rows)} request(s)"
    if counts:
        summary += ": " + ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
    return render(summary, data), False


def _split_syntax(data):
    """Move a long syntax text out of the JSON so its line breaks stay readable."""
    if isinstance(data, dict) and isinstance(data.get("syntax"), str):
        rest = {k: v for k, v in data.items() if k != "syntax"}
        return rest, data["syntax"]
    return data, None


@_tool("mast_composite_draft", title="Check a composite skill", read_only=True,
       idempotent=True, description=(
    "Validate a composite-skill spec (a JSON graph built from existing skills) without saving "
    "it. All problems come back at once, plus hints such as 'an official skill already does "
    "this'. Pass spec='?' to get the spec syntax first. Composites are the normal way to add a "
    "capability: build from existing skills before proposing Python code. "
    "Keywords: 组合技能, 草稿, 校验."),
    properties={"spec": {"type": ["object", "string"],
                         "description": "The spec as a JSON object, or '?' for the syntax."}},
    required=("spec",))
def _composite_draft(ctx: CallContext, args: dict):
    spec = _spec(args)
    data = ctx.post("/composites/draft", {"spec": spec})
    rest, syntax = _split_syntax(data)
    if spec == "?":
        summary = "Composite spec syntax, from MAST:"
    elif isinstance(data, dict) and data.get("ok"):
        summary = "The draft is valid. Save it with mast_composite_save(spec=...)."
        hints = data.get("hints")
        if hints:
            summary += " Hints: " + "; ".join(_one_line(h, 200) for h in list(hints)[:3])
    else:
        summary = f"The draft has problems: {_problems(data) or 'see below'}"
    return render(summary, rest, body=syntax), False


@_tool("mast_composite_save", title="Save a composite skill", description=(
    "Save a composite skill and hot-register it: it can be run at once with mast_run, under the "
    "same safety gates as official skills. Check it with mast_composite_draft first. To "
    "overwrite a composite you saved before, pass base_version = the version you last got "
    "(optimistic lock). Composites made by a person are never overwritten; save under a new "
    "name. Keywords: 组合技能, 保存."),
    properties={"spec": {"type": ["object", "string"],
                         "description": "The validated spec, as a JSON object."},
                "base_version": _int_p("Version you are replacing (only when overwriting).")},
    required=("spec",))
def _composite_save(ctx: CallContext, args: dict):
    spec = _spec(args)
    if spec == "?":
        raise ArgError("for the syntax call mast_composite_draft(spec='?')")
    body: dict = {"spec": spec}
    if args.get("base_version") not in (None, ""):
        body["base_version"] = _int(args, "base_version", default=-1, lo=-1, hi=10 ** 9)
    data = ctx.post("/composites", body)
    if not isinstance(data, dict) or not data.get("ok"):
        err = data.get("error") if isinstance(data, dict) else None
        return render(f"Not saved ({err or 'refused'}): {_problems(data)}", data), True
    name = data.get("name", "?")
    summary = f"Saved composite {name} v{data.get('version', '?')}"
    if data.get("unchanged"):
        summary += " (identical to the stored version; nothing new was written)"
    if data.get("hot_registered") is False:
        summary += f", but registering it failed: {_one_line(data.get('registration_error'), 300)}"
    else:
        summary += f'. Run it with mast_run(skill="{name}", params=...).'
    return render(summary, data), False


@_tool("mast_propose_skill", title="Propose a Python skill", description=(
    "Propose a NEW atomic skill as Python source, only when official skills and composites "
    "cannot express it (for example a Nanonis command nobody wraps yet). The code is written "
    "to disk for a person to review; it is NOT registered or run, and only a person can enable "
    "it. Carry on with existing skills meanwhile. code defines a BaseSkill subclass "
    "(metadata() and execute(context, params)); rationale is what the reviewer judges. "
    "Keywords: 新技能, 提议, Python."),
    properties={"name": _str_p("Skill class/file name for the proposal."),
                "code": _str_p("Complete Python source of the skill."),
                "rationale": _str_p("Why existing skills and composites are not enough.")},
    required=("name", "code", "rationale"))
def _propose_skill(ctx: CallContext, args: dict):
    name = _str(args, "name", required=True)
    data = ctx.post("/skills/proposals", {"name": name,
                                          "code": _text(args, "code", required=True),
                                          "rationale": _text(args, "rationale", required=True)})
    if not isinstance(data, dict) or not data.get("ok"):
        err = data.get("error") if isinstance(data, dict) else None
        return render(f"Proposal not accepted ({err or 'refused'}): {_problems(data)}", data), True
    where = data.get("path") or data.get("file")
    summary = (f"Proposal for {name} written" + (f" to {where}" if where else "")
               + ". It is NOT enabled: a person has to review the code and switch it on. "
               "Carry on with existing skills.")
    return render(summary, data), False


@_tool("mast_handover", title="Write a MAST handover", description=(
    "Finish a work session: MAST writes a handover report (what was done, taken from its "
    "records, plus your summary and next steps) into its document library for the operator "
    "and the next agent. Do this last, once the instrument is idle. "
    "Keywords: 交接, 报告, 收尾."),
    properties={"summary": _str_p("What you did, what you found, what state you leave."),
                "next_steps": _list_p("Suggested next steps, one per item."),
                "title": _str_p("Optional report title."),
                "since": _str_p("Optional ISO time the report should start from.")},
    required=("summary",))
def _handover(ctx: CallContext, args: dict):
    body: dict = {"summary": _text(args, "summary", required=True)}
    steps = _lines(args, "next_steps")
    title = _str(args, "title")
    since = _str(args, "since")
    if steps:
        body["next_steps"] = steps
    if title:
        body["title"] = title
    if since:
        body["since"] = since
    data = ctx.post("/handover", body)
    if _ok_false(data):
        return render(f"Handover not stored: {_problems(data)}", data), True
    ref = None
    if isinstance(data, dict):
        ref = data.get("doc_id") or data.get("document_id") or data.get("id") or data.get("path")
    summary = ("Handover report stored in MAST's document library" + (f" ({ref})" if ref else "")
               + ". The operator and the next agent will find it there.")
    return render(summary, data), False
