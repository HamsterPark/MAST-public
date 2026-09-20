// ════════════════════════════════════════════════════════════════════════════
// pollRates.ts — how often the UI asks for live instrument readings, and WHY
// that number is what it is.
//
// Whether the update rate could go faster, closer to real-time, once conditions
// allow it. The answer is yes, but only up to a hard ceiling that lives on
// the other side of the API — so this constant exists mainly to stop the next
// person from "just" cranking it to 100 ms and thinking that helps.
//
// THE CHAIN (measured 2026-07-28, see tests/v2/unit/api/test_live_readings_
// no_hardware_io.py):
//
//   Nanonis ──TCP(monitor port)──▶ InstrumentState.refresh()  every ~1 s
//                                        │  10 calls per cycle
//                                        ▼
//                                  InstrumentState._cache
//                                        │  pure dict read, ZERO TCP
//                                        ▼
//                        GET /api/hardware/live-readings ◀── this poll
//
// So the producer runs at ~1 Hz and the endpoint is free. The UI used to poll
// every 2000 ms — HALF the producer's rate — which meant it threw away every
// other reading the instrument had already given us, and a value could be up to
// ~3 s old on screen (1 s producer period + 2 s poll period).
//
// 500 ms fixes exactly that, at zero cost to the instrument: the request never
// touches the TCP port, so polling faster adds no hardware load whatsoever. It
// guarantees we pick up every producer update within half its period, so the
// numbers now tick once a second like a real instrument readout instead of
// stuttering every two seconds and skipping values.
//
// WHY NOT FASTER — the ceiling is the producer, not this number.
// Nothing the client does can make a reading fresher than the ~1 s refresh.
// Going to 250 ms buys ~0.25 s of worst-case age for twice the requests over
// what is often a Tailscale tunnel; not worth it.
//
// WHY WE DID NOT SPEED UP THE PRODUCER INSTEAD. The monitor TCP port is shared,
// serialised behind one per-role lock (core/connection.py), and already busy:
//   · SafetyWatchdog      0.5 s × 1 call   =  2 calls/s  ← tip-crash → 紧急退针
//   · InstrumentState     1.0 s × 10 calls = ~10 calls/s
//   · ScanMonitor         0.5 s × ≤4 calls = ≤8 calls/s  (only while scanning)
// ≈20 transactions/s during a scan — which is precisely when the operator most
// wants live numbers. Halving the InstrumentState interval would take lock time
// away from the tip-crash watchdog, on a TCP port the project treats as fragile
// by rule. That trade needs a real-rig latency measurement we do not have, so
// the producer stays at 1 s.
// ════════════════════════════════════════════════════════════════════════════

/**
 * Poll interval for `GET /api/hardware/live-readings`.
 *
 * Safe to lower toward the producer's period; pointless below it. Raising the
 * PRODUCER's rate is the thing that costs instrument time — see above.
 */
export const LIVE_READINGS_POLL_MS = 500;

/**
 * How old a readings payload may get before the UI must stop presenting it as
 * live. Generous multiple of the poll interval so one dropped request or a GC
 * pause never flashes a false "stale"; short enough that a dead monitor link is
 * called out long before the operator acts on a frozen number.
 */
export const LIVE_READINGS_STALE_AFTER_MS = 15_000;

/**
 * Poll interval to fall back to while the /ws/events push channel is OPEN.
 *
 * Not zero, on purpose. A WebSocket killed by a NAT rebind stays open-looking
 * until the idle watchdog fires (WS_IDLE_MS = 45 s in lib/ws.ts), so a consumer
 * that stops polling the instant the socket connects can show frozen numbers for
 * most of a minute with nothing on screen admitting it. This keeps a slow pulse
 * underneath push so the display can never outlive the link by more than one
 * interval — and, because the endpoint is a cached read that never touches the
 * Nanonis TCP port, it costs the instrument exactly nothing.
 *
 * 10 s ≈ 1/20th of the traffic of the 500 ms poll while push is healthy, which
 * is the actual win being banked here.
 */
export const LIVE_READINGS_WS_KEEPALIVE_POLL_MS = 10_000;

// ════════════════════════════════════════════════════════════════════════════
// 电流监控 (pages/MonitoringPage.tsx) — a DIFFERENT budget from the one above,
// and it is worth saying why, because the numbers look inconsistent otherwise.
//
// The chain has no instrument in it at all:
//
//   Nanonis ──Osci1T──▶ monitor daemon ──▶ SQLite store (mast.monitoring)
//                                              │  read-only, no TCP
//                                              ▼
//                          GET /api/monitoring/{status,live-trace,…} ◀── poll
//
// The daemon owns the ONE oscilloscope subscription and writes a segment per
// second. Every endpoint this page uses reads that store — routes/monitoring.py
// is explicitly zero-TCP. So unlike live-readings, polling here cannot take
// lock time from the tip-crash watchdog no matter what number we pick; the only
// cost is bytes on the wire, which matters solely because remote access runs
// over a Tailscale tunnel .
//
// That is what the WS keep-alive rates are buying: while push is healthy the
// polls drop to a slow floor and save the tunnel ~5× the traffic. They do not
// stop, for the same reason the top bar's do not — a socket killed by a NAT
// rebind looks open until WS_IDLE_MS (45 s), and a chart that keeps drawing a
// frozen line for most of a minute is the failure 「UI 绝不冻结」 forbids.
// ════════════════════════════════════════════════════════════════════════════

/**
 * Poll interval for `GET /api/monitoring/live-trace`.
 *
 * The producer emits one segment per second, so 2 s is deliberately UNDER-rated
 * relative to it: the trace is an envelope band over a 30 s–5 min window, where
 * one segment is 1/30th of the narrowest view — arriving a second late is
 * invisible. A WS `segment` event invalidates this query anyway (throttled to
 * ≤1/s), so in practice the poll is the floor, not the mechanism.
 */
export const MONITORING_TRACE_POLL_MS = 2_000;

/** Trace floor while /ws/events is OPEN — push does the real work. */
export const MONITORING_TRACE_WS_KEEPALIVE_POLL_MS = 10_000;

/**
 * Poll interval for `GET /api/monitoring/status`.
 *
 * Slower than the trace because most of what it carries barely moves: daemon
 * state, storage totals, retention. The one fast-moving part (`latest`) is
 * patched straight into this cache by the WS `segment` event, so the poll is
 * there to correct drift and to notice a daemon that died quietly.
 */
export const MONITORING_STATUS_POLL_MS = 3_000;

/** Status floor while /ws/events is OPEN. Still well under WS_IDLE_MS (45 s),
 *  so a silently dead socket is caught by data, not only by the watchdog. */
export const MONITORING_STATUS_WS_KEEPALIVE_POLL_MS = 15_000;

/**
 * How old the newest segment may get before the page stops presenting it as
 * live — the verdict banner greys out to 「无数据」 and the chart says so.
 *
 * 15 s ≈ 15 segments. Long enough that a GC pause, a retention sweep or one
 * dropped request never flashes a false stale; short enough that a stalled
 * daemon is called out long before an operator would act on the reading. The
 * threshold intentionally matches LIVE_READINGS_STALE_AFTER_MS: two panels
 * disagreeing about what "live" means is its own bug report.
 */
export const MONITORING_STALE_AFTER_MS = 15_000;

/**
 * Poll interval for the 环境历史 series/spectra queries.
 *
 * Deliberately slow, and NOT because the endpoint is expensive — it reads
 * SQLite and issues zero TCP. It is slow because the DATA is: the underlying
 * buckets are one minute wide, so refetching every 30 s already over-samples
 * the source by 2×. Anything faster would redraw the same curve.
 *
 * There is also no WebSocket for this page, on purpose. The bus replays only
 * the last 100 events and carries live instrument telemetry; a history view
 * that redraws twice a minute has no business competing for that budget.
 */
export const ENV_HISTORY_SERIES_POLL_MS = 30_000;

/**
 * Poll interval for `GET /api/env-history/status`.
 *
 * Faster than the series because this is the panel that answers "is it actually
 * recording?" — a sink that self-disabled after a write failure, or a sweep
 * that started deleting, should surface within a few seconds rather than half a
 * minute. It is still a pure in-memory + SQLite read.
 */
export const ENV_HISTORY_STATUS_POLL_MS = 10_000;
