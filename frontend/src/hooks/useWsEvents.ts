// ════════════════════════════════════════════════════════════════════════════
// useWsEvents — React bindings for the single /ws/events socket (lib/ws.ts).
//
// Split from the transport on purpose: lib/ws.ts has no React import, so the
// reconnect/backoff/fallback logic can be exercised in plain Node without a DOM
// (frontend/test/ws.reconnect.test.ts). Everything React-shaped lives here.
// ════════════════════════════════════════════════════════════════════════════

import { useEffect, useRef, useSyncExternalStore } from "react";
import {
  getStatus,
  shouldPoll,
  shouldWarn,
  subscribe,
  subscribeStatus,
  type WsEvent,
  type WsStatus,
} from "@/lib/ws";

/**
 * Current connection state, re-rendering the caller on every change.
 *
 * `useSyncExternalStore` rather than a `useState` + effect: the status can
 * change between render and effect (a socket opens fast on localhost) and the
 * effect version would show a stale "connecting" until the next change.
 */
export function useWsStatus(): WsStatus {
  return useSyncExternalStore(subscribeStatus, getStatus, () => "idle" as WsStatus);
}

/** Convenience: everything a consumer needs to decide how to behave. */
export function useWsConnection(): {
  status: WsStatus;
  /** True while the consumer must keep its HTTP polling running. */
  polling: boolean;
  /** True when the operator should be told push is unavailable. */
  warn: boolean;
} {
  const status = useWsStatus();
  return { status, polling: shouldPoll(status), warn: shouldWarn(status) };
}

/**
 * Subscribe to one event type for the lifetime of the component.
 *
 * The handler is held in a ref so a caller can pass an inline arrow without
 * re-subscribing on every render — re-subscribing is not merely wasteful here,
 * it drops the refcount to zero between the cleanup and the re-subscribe and
 * would tear the shared socket down and back up on each render.
 *
 * `enabled: false` unsubscribes entirely, which is how a consumer opts out
 * without unmounting (and, if it is the only subscriber, closes the socket).
 */
export function useWsEvent(
  type: string,
  handler: (event: WsEvent) => void,
  enabled = true,
): void {
  const ref = useRef(handler);
  ref.current = handler;

  useEffect(() => {
    if (!enabled) return;
    return subscribe(type, (event) => ref.current(event));
  }, [type, enabled]);
}
