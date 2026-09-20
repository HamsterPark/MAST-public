"""MCP over stdio: JSON-RPC 2.0 framing and the MCP method set.

This is the only module that knows the wire protocol. Tools and resources are
handed in as plain providers (see ``serve``), so replacing this file with an
official MCP SDK leaves the rest of the package untouched.

Framing rules (MCP 2025-06-18, stdio transport):

* one message per line, UTF-8, no embedded newline;
* **stdout carries protocol messages and nothing else** -- logs go to stderr;
* bytes are read from ``sys.stdin.buffer`` and written to ``sys.stdout.buffer``
  so that neither the console code page of a Chinese Windows nor text-mode
  ``\\r\\n`` translation can touch the stream.

Threading: the reader thread answers everything inline except ``tools/call``,
which runs on its own daemon thread. A 50-second job wait therefore never
delays a ``ping`` or a ``notifications/cancelled``.
"""
from __future__ import annotations

import json
import logging
import sys
import threading
import time

log = logging.getLogger("mast_mcp.rpc")

SUPPORTED_PROTOCOL_VERSIONS = ("2025-06-18", "2025-03-26", "2024-11-05")
LATEST_PROTOCOL_VERSION = SUPPORTED_PROTOCOL_VERSIONS[0]

PARSE_ERROR = -32700
INVALID_REQUEST = -32600
METHOD_NOT_FOUND = -32601
INVALID_PARAMS = -32602
INTERNAL_ERROR = -32603
RESOURCE_NOT_FOUND = -32002

#: Tool calls allowed to run at once. More than this is almost certainly a
#: client stuck in a loop; the extra calls get a tool error instead of a thread.
MAX_CONCURRENT_CALLS = 8

#: After stdin closes, how long in-flight tool calls get to finish and answer.
DRAIN_TIMEOUT_S = 5.0


class RpcError(Exception):
    """A JSON-RPC error response (protocol level, not a failed tool)."""

    def __init__(self, code: int, message: str, data=None):
        super().__init__(message)
        self.code = code
        self.message = message
        self.data = data


class CallSignal:
    """Why a running tool call should stop waiting early.

    * ``cancelled`` -- the client sent ``notifications/cancelled``; the call's
      response is dropped, as the protocol asks.
    * draining -- stdin closed; the call should wrap up quickly but still answer.
    """

    def __init__(self, draining: threading.Event):
        self.cancelled = threading.Event()
        self._draining = draining

    def should_stop(self) -> bool:
        return self.cancelled.is_set() or self._draining.is_set()

    def wait(self, seconds: float) -> bool:
        """Sleep up to ``seconds``; return True as soon as the call should stop."""
        end = time.monotonic() + max(0.0, seconds)
        while not self.should_stop():
            left = end - time.monotonic()
            if left <= 0:
                return False
            self.cancelled.wait(min(left, 0.25))
        return True


def encode_message(obj) -> bytes:
    """One JSON-RPC message as one UTF-8 line.

    ``json.dumps`` escapes control characters inside strings and the compact
    separators emit no whitespace outside them, so the only newline is the
    terminator. U+2028/U+2029 are legal raw inside JSON strings but some line
    readers split on them, so they are escaped too. A lone surrogate (only
    possible if a client sent one) cannot be encoded as UTF-8; that message
    falls back to pure-ASCII escapes.
    """
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str)
    text = text.replace("\u2028", "\\u2028").replace("\u2029", "\\u2029")
    try:
        data = text.encode("utf-8")
    except UnicodeEncodeError:
        data = json.dumps(obj, ensure_ascii=True, separators=(",", ":"),
                          default=str).encode("ascii")
    return data + b"\n"


def _id_key(mid) -> str:
    # JSON-RPC ids are strings or numbers, and 1 and "1" are different ids.
    return json.dumps(mid, sort_keys=True)


def _error(mid, code: int, message: str, data=None) -> dict:
    err = {"code": code, "message": message}
    if data is not None:
        err["data"] = data
    return {"jsonrpc": "2.0", "id": mid, "error": err}


def _tool_result(text: str, is_error: bool) -> dict:
    return {"content": [{"type": "text", "text": text}], "isError": bool(is_error)}


class McpServer:
    """Reads requests from ``instream`` and writes responses to ``outstream``.

    ``tools`` provides ``list_tools()``, ``has_tool(name)`` and
    ``call_tool(name, arguments, signal) -> (text, is_error)``.
    ``resources`` provides ``list_resources()``, ``list_templates()`` and
    ``read_resource(uri) -> contents item`` (``KeyError`` when unknown).
    """

    def __init__(self, *, server_info: dict, instructions: str, tools, resources,
                 instream=None, outstream=None):
        self._server_info = dict(server_info)
        self._instructions = instructions
        self._tools = tools
        self._resources = resources
        self._in = instream if instream is not None else sys.stdin.buffer
        self._out = outstream if outstream is not None else sys.stdout.buffer
        self._write_lock = threading.Lock()
        self._inflight: dict[str, CallSignal] = {}
        self._inflight_lock = threading.Lock()
        self._draining = threading.Event()
        self._slots = threading.BoundedSemaphore(MAX_CONCURRENT_CALLS)
        self._threads: list[threading.Thread] = []
        self.protocol_version: str | None = None
        self.initialized = False

    # ── transport ────────────────────────────────────────────────────────
    def serve(self) -> int:
        """Answer messages until stdin closes; return a process exit code."""
        for raw in iter(self._in.readline, b""):
            line = raw.strip()
            if line:
                self._handle_line(line)
        self._drain()
        return 0

    def _send(self, obj) -> None:
        data = encode_message(obj)
        with self._write_lock:
            try:
                self._out.write(data)
                self._out.flush()
            except (BrokenPipeError, ValueError, OSError) as exc:
                # The client is gone; nothing useful can be done with the answer.
                log.warning("could not write a response (%s)", exc)

    def _drain(self) -> None:
        self._draining.set()
        deadline = time.monotonic() + DRAIN_TIMEOUT_S
        for t in list(self._threads):
            left = deadline - time.monotonic()
            if left <= 0:
                break
            t.join(left)

    # ── dispatch ─────────────────────────────────────────────────────────
    def _handle_line(self, line: bytes) -> None:
        try:
            msg = json.loads(line.decode("utf-8"))
        except (UnicodeDecodeError, ValueError) as exc:
            self._send(_error(None, PARSE_ERROR, f"Parse error: {exc}"))
            return
        if isinstance(msg, list):
            self._handle_batch(msg)
        else:
            self._dispatch(msg, self._send, batch=False)

    def _handle_batch(self, items: list) -> None:
        # Batches were dropped in 2025-06-18; older clients may still send one.
        if not items:
            self._send(_error(None, INVALID_REQUEST, "Invalid request: empty batch"))
            return
        replies: list = []
        for item in items:
            self._dispatch(item, replies.append, batch=True)
        if replies:
            self._send(replies)

    def _dispatch(self, msg, reply, *, batch: bool) -> None:
        if not isinstance(msg, dict):
            reply(_error(None, INVALID_REQUEST, "Invalid request: not a JSON object"))
            return
        method = msg.get("method")
        has_id = "id" in msg
        mid = msg.get("id")
        if method is None:
            return  # a response to a request this server never sends
        if not isinstance(method, str):
            if has_id:
                reply(_error(mid, INVALID_REQUEST, "Invalid request: method must be a string"))
            return
        params = msg.get("params")
        if params is None:
            params = {}
        if not isinstance(params, dict):
            if has_id:
                reply(_error(mid, INVALID_PARAMS, "Invalid params: params must be an object"))
            return
        if not has_id:
            self._notification(method, params)
            return
        if method == "tools/call" and not batch:
            self._start_call(mid, params, reply)
            return
        try:
            result = self._request(method, params)
        except RpcError as exc:
            reply(_error(mid, exc.code, exc.message, exc.data))
            return
        except Exception as exc:  # noqa: BLE001 - a bug here must not kill the server
            log.exception("internal error while handling %s", method)
            reply(_error(mid, INTERNAL_ERROR, f"Internal error: {exc}"))
            return
        reply({"jsonrpc": "2.0", "id": mid, "result": result})

    def _request(self, method: str, params: dict):
        if method == "initialize":
            return self._initialize(params)
        if method == "ping":
            return {}
        if method == "tools/list":
            return {"tools": self._tools.list_tools()}
        if method == "tools/call":          # only reached from a batch
            return self._call_inline(params)
        if method == "resources/list":
            return {"resources": self._resources.list_resources()}
        if method == "resources/templates/list":
            return {"resourceTemplates": self._resources.list_templates()}
        if method == "resources/read":
            uri = params.get("uri")
            if not isinstance(uri, str) or not uri:
                raise RpcError(INVALID_PARAMS, "resources/read needs a uri")
            try:
                item = self._resources.read_resource(uri)
            except KeyError:
                raise RpcError(RESOURCE_NOT_FOUND, f"Resource not found: {uri}",
                               {"uri": uri}) from None
            return {"contents": [item]}
        raise RpcError(METHOD_NOT_FOUND, f"Method not found: {method}")

    def _initialize(self, params: dict) -> dict:
        requested = params.get("protocolVersion")
        version = (requested if requested in SUPPORTED_PROTOCOL_VERSIONS
                   else LATEST_PROTOCOL_VERSION)
        self.protocol_version = version
        client = params.get("clientInfo") if isinstance(params.get("clientInfo"), dict) else {}
        log.info("initialize: client=%s %s, protocol %s (requested %s)",
                 client.get("name", "?"), client.get("version", ""), version, requested)
        return {
            "protocolVersion": version,
            "capabilities": {
                "tools": {"listChanged": False},
                "resources": {"subscribe": False, "listChanged": False},
            },
            "serverInfo": self._server_info,
            "instructions": self._instructions,
        }

    def _notification(self, method: str, params: dict) -> None:
        if method == "notifications/initialized":
            self.initialized = True
        elif method == "notifications/cancelled":
            key = _id_key(params.get("requestId"))
            with self._inflight_lock:
                signal = self._inflight.get(key)
            if signal is not None:
                signal.cancelled.set()
                log.info("request %s cancelled by the client (%s)", key,
                         params.get("reason") or "no reason given")
        else:
            log.debug("ignoring notification %s", method)

    # ── tools/call ───────────────────────────────────────────────────────
    def _parse_call(self, params: dict) -> tuple[str, dict]:
        name = params.get("name")
        if not isinstance(name, str) or not name:
            raise RpcError(INVALID_PARAMS, "tools/call needs a tool name")
        if not self._tools.has_tool(name):
            raise RpcError(INVALID_PARAMS, f"Unknown tool: {name}")
        arguments = params.get("arguments")
        if arguments is None:
            arguments = {}
        if not isinstance(arguments, dict):
            raise RpcError(INVALID_PARAMS, "tools/call arguments must be an object")
        return name, arguments

    def _call_inline(self, params: dict) -> dict:
        name, arguments = self._parse_call(params)
        text, is_error = self._tools.call_tool(name, arguments, CallSignal(self._draining))
        return _tool_result(text, is_error)

    def _start_call(self, mid, params: dict, reply) -> None:
        try:
            name, arguments = self._parse_call(params)
        except RpcError as exc:
            reply(_error(mid, exc.code, exc.message, exc.data))
            return
        if not self._slots.acquire(blocking=False):
            reply({"jsonrpc": "2.0", "id": mid, "result": _tool_result(
                f"The MAST MCP server is already running {MAX_CONCURRENT_CALLS} tool "
                "calls; wait for one of them to finish and try again.", True)})
            return
        signal = CallSignal(self._draining)
        key = _id_key(mid)
        with self._inflight_lock:
            self._inflight[key] = signal
        thread = threading.Thread(target=self._run_call,
                                  args=(mid, key, name, arguments, signal, reply),
                                  name=f"mcp-{name}", daemon=True)
        self._threads = [t for t in self._threads if t.is_alive()]
        self._threads.append(thread)
        thread.start()

    def _run_call(self, mid, key: str, name: str, arguments: dict,
                  signal: CallSignal, reply) -> None:
        try:
            text, is_error = self._tools.call_tool(name, arguments, signal)
            result = _tool_result(text, is_error)
        except Exception as exc:  # noqa: BLE001 - becomes a tool error, never a crash
            log.exception("tool %s raised", name)
            result = _tool_result(f"Internal error in the MAST MCP server while running "
                                  f"{name}: {exc!r}", True)
        finally:
            with self._inflight_lock:
                self._inflight.pop(key, None)
            self._slots.release()
        if signal.cancelled.is_set():
            log.info("dropping the response to cancelled request %s", key)
            return
        reply({"jsonrpc": "2.0", "id": mid, "result": result})


def serve(*, server_info: dict, instructions: str, tools, resources,
          instream=None, outstream=None) -> int:
    """Run an MCP server on stdio (or the given binary streams) until EOF."""
    return McpServer(server_info=server_info, instructions=instructions, tools=tools,
                     resources=resources, instream=instream, outstream=outstream).serve()
