// ════════════════════════════════════════════════════════════════════════════
// sse.ts — the one SSE reader every streaming view uses.
//
// This module fixes a class of bug where a network change on the operator's machine leaves the conversation looking stuck.
// The operator drives this box over Tailscale; a roam / re-route / sleep drops
// the TCP connection under a live SSE stream. Four views each hand-rolled the
// same reader loop, and all four had the same two holes:
//
//   1. **EOF was treated as success.** `if (done) break;` cannot tell "the server
//      finished" from "the socket died", because nothing tracked whether a
//      terminal frame ever arrived. A dropped connection therefore looked
//      IDENTICAL to a completed turn: the spinner stopped, no error appeared,
//      no reconnect banner appeared, and the conversation simply ended
//      mid-sentence with zero signal. That is the "对话卡住" the operator sees —
//      not a frozen screen, but a screen that lies about being finished.
//
//   2. **A half-open socket hung forever.** A TCP connection killed by a NAT
//      rebind is not closed, it just stops delivering; `reader.read()` then
//      never settles and the view waits for eternity with no timeout.
//
// This module fixes both, and is the client half of the backend keep-alive in
// `MASTv2/mast/api/sse.py`: because every stream now emits `: ping` every ~15 s,
// prolonged silence is real evidence of a dead link rather than a slow model, so
// an idle watchdog can fire without ever cutting a healthy long-running run.
//
// Invariant it serves ("UI 绝不冻结"): a stream can end four different ways and
// every one of them must leave the UI in a state the operator can act on.
// ════════════════════════════════════════════════════════════════════════════

/* eslint-disable @typescript-eslint/no-explicit-any */

/** How long a stream may deliver NOTHING before we call it dead. Generous
 *  multiple of the server's 15 s keep-alive so beat jitter, a paused laptop or
 *  a slow first token can never trip it — only genuine silence does. */
export const SSE_IDLE_MS = 75_000;

export type SseEnd =
  /** Server sent its terminal frame — the turn really finished. */
  | { reason: "done" }
  /** Caller aborted (stop button, switching conversations). Not an error. */
  | { reason: "aborted" }
  /** Body ended with no terminal frame — the connection was cut mid-run. */
  | { reason: "truncated" }
  /** No bytes at all for `idleMs`, keep-alives included — link presumed dead. */
  | { reason: "idle" }
  /** fetch/read threw (DNS, refused, TLS, offline…). */
  | { reason: "error"; message: string };

/** True when the stream did NOT reach its terminal frame — i.e. the view must
 *  tell the operator something instead of silently looking finished. */
export function sseBroke(end: SseEnd): boolean {
  return end.reason === "truncated" || end.reason === "idle" || end.reason === "error";
}

/**
 * Operator-facing CAUSE of a broken stream, or null when nothing is wrong.
 *
 * Cause only, deliberately: what happened to the CONNECTION and what is true of
 * the RUN are different facts, and the backend keeps working after the stream
 * drops (see mast/api/sse.py) — so "连接断了" must never be read as "任务没了".
 * Each view appends its own consequence sentence once it has reconciled with the
 * server, which is the only honest way to state it.
 */
export function sseEndMessage(end: SseEnd): string | null {
  const offline =
    typeof navigator !== "undefined" && navigator.onLine === false
      ? "（浏览器显示当前离线）"
      : "";
  switch (end.reason) {
    case "done":
    case "aborted":
      return null;
    case "truncated":
      return `连接中断${offline}：网络变化（切换 Wi-Fi / Tailscale 重连 / 休眠）会掐断这条流。`;
    case "idle":
      return `连接已静默 ${Math.round(SSE_IDLE_MS / 1000)} 秒${offline}：服务端每 15 秒有一次心跳，收不到说明链路已断。`;
    case "error":
      return `连接出错${offline}：${end.message}`;
  }
}

export interface ReadSseOptions {
  /** Called once per parsed `data:` frame, in arrival order. */
  onFrame: (frame: any) => void;
  /**
   * Marks the frame that means "the server is finished". Without this every
   * stream end looks clean — the exact bug this module exists to kill.
   * Default: `frame.kind === "done"`, the shape all MAST streams use.
   */
  isTerminal?: (frame: any) => boolean;
  /** Silence budget before declaring the link dead. */
  idleMs?: number;
  /**
   * The caller's AbortController. On an idle timeout we abort it so the dead
   * fetch is actually released rather than left dangling on a half-open socket.
   */
  controller?: AbortController;
}

/**
 * Drain an SSE response body, reporting HOW it ended.
 *
 * Never throws: every failure is returned as an `SseEnd` so callers can't
 * accidentally leave a spinner running in a forgotten catch.
 */
export async function readSseStream(
  resp: Response,
  opts: ReadSseOptions,
): Promise<SseEnd> {
  const {
    onFrame,
    isTerminal = (f: any) => f?.kind === "done",
    idleMs = SSE_IDLE_MS,
    controller,
  } = opts;

  if (!resp.body) return { reason: "error", message: `HTTP ${resp.status}（无响应体）` };

  const reader = resp.body.getReader();
  const decoder = new TextDecoder();
  let buf = "";
  let sawTerminal = false;

  // Sentinel for the idle race. A plain symbol keeps it distinguishable from
  // any legitimate read result.
  const IDLE = Symbol("sse-idle");

  try {
    for (;;) {
      let timer: ReturnType<typeof setTimeout> | undefined;
      const idleGuard = new Promise<typeof IDLE>((resolve) => {
        timer = setTimeout(() => resolve(IDLE), idleMs);
      });
      let res: ReadableStreamReadResult<Uint8Array> | typeof IDLE;
      try {
        res = await Promise.race([reader.read(), idleGuard]);
      } finally {
        if (timer !== undefined) clearTimeout(timer);
      }

      if (res === IDLE) {
        // Half-open socket: reader.read() would never settle. Abort the fetch so
        // the connection is released, then report it honestly.
        try {
          controller?.abort();
        } catch {
          /* already aborted */
        }
        return { reason: "idle" };
      }

      const { value, done } = res;
      if (done) {
        // THE fix: an ended body is only a success if the server said so.
        return sawTerminal ? { reason: "done" } : { reason: "truncated" };
      }

      buf += decoder.decode(value, { stream: true });
      let sep: number;
      while ((sep = buf.indexOf("\n\n")) !== -1) {
        const rawFrame = buf.slice(0, sep);
        buf = buf.slice(sep + 2);
        // `: ping` keep-alive comments carry no data: line → empty payload → skipped.
        const payload = rawFrame
          .split("\n")
          .filter((l) => l.startsWith("data:"))
          .map((l) => l.slice(5).trim())
          .join("");
        if (!payload) continue;
        let frame: any;
        try {
          frame = JSON.parse(payload);
        } catch {
          continue;
        }
        if (isTerminal(frame)) sawTerminal = true;
        onFrame(frame);
      }
    }
  } catch (err) {
    if ((err as Error)?.name === "AbortError") return { reason: "aborted" };
    return { reason: "error", message: String((err as Error)?.message ?? err) };
  } finally {
    try {
      reader.cancel();
    } catch {
      /* already released */
    }
  }
}
