/* Node harness for the gui_jsx review fixes (findings #24 / #25).
 *
 * Exercises the REAL transpiled agents-ui.jsx code — not a mock of it — to
 * prove:
 *   #24  AGInterruptHost consumes snapshot.pending_interrupts and POSTs the
 *        operator verdict to /interrupts/<id>/resolve (the live control path).
 *   #25  AGBackendBadge renders an always-on degraded indicator (it does NOT
 *        gate on standalone mode the way AGHeader does).
 *
 * It does this by transpiling the JSX with the vendored Babel, loading the
 * module under a tiny single-pass React stub, rendering the two components
 * with a controlled AGLiveCtx value, and capturing the actual fetch() the
 * component issues. fetch / React are the only stubbed boundaries; the
 * component logic under test runs for real.
 *
 * Usage:  node _gui_jsx_interrupt_harness.cjs <repo_root>
 * Exit 0 + "ALL_OK" on success; non-zero with a FAIL line otherwise.
 */
"use strict";
const fs = require("fs");
const path = require("path");
const vm = require("vm");

const repoRoot = process.argv[2];
if (!repoRoot) { console.error("FAIL: repo root arg missing"); process.exit(2); }
const mastv2 = path.join(repoRoot, "MASTv2");
const Babel = require(path.join(mastv2, "scripts", "_agents_build", "babel.standalone.js"));
const SRC = path.join(mastv2, "mast", "gui", "static", "agents", "agents-ui.jsx");

const src = fs.readFileSync(SRC, "utf8");
const transpiled = Babel.transform(src, {
  presets: ["react"], filename: "agents-ui.jsx", compact: false,
}).code;

// ── Minimal single-pass React stub ────────────────────────────────────
// Enough to render a function component tree ONCE and run its hooks. Each
// render gets a fresh hook cursor; effects run synchronously after render.
function makeReact() {
  let hooks = [];
  let cursor = 0;
  let ctxStack = new Map(); // context object -> current value
  const effects = [];

  function renderComponent(type, props) {
    const prevHooks = hooks, prevCursor = cursor;
    hooks = []; cursor = 0; effects.length = 0;
    const out = type(props || {});
    for (const fn of effects) { const c = fn(); if (typeof c === "function") { /* ignore cleanup */ } }
    hooks = prevHooks; cursor = prevCursor;
    return out;
  }

  const React = {
    Fragment: Symbol("Fragment"),
    createElement(type, props, ...children) {
      // Resolve function components eagerly so nested components (e.g.
      // AGDangerousModal inside AGInterruptHost) also execute and we can walk
      // the tree. Host elements / stub components keep their raw shape.
      const kids = children.flat().filter(c => c != null && c !== false);
      if (typeof type === "function") {
        return renderComponent(type, Object.assign({}, props, kids.length ? { children: kids } : {}));
      }
      return { type, props: props || {}, children: kids };
    },
    createContext(def) {
      const ctx = { _default: def, Provider: null };
      ctx.Provider = function Provider(p) {
        ctxStack.set(ctx, p.value);
        return p.children;
      };
      return ctx;
    },
    useContext(ctx) { return ctxStack.has(ctx) ? ctxStack.get(ctx) : ctx._default; },
    useState(init) {
      const i = cursor++;
      if (!(i in hooks)) hooks[i] = (typeof init === "function") ? init() : init;
      const set = (v) => { hooks[i] = (typeof v === "function") ? v(hooks[i]) : v; };
      return [hooks[i], set];
    },
    useEffect(fn) { effects.push(fn); },
    useMemo(fn) { return fn(); },
    useRef(init) { const i = cursor++; if (!(i in hooks)) hooks[i] = { current: init }; return hooks[i]; },
    useCallback(fn) { return fn; },
  };
  React._setContext = (ctx, val) => ctxStack.set(ctx, val);
  React._render = renderComponent;
  return React;
}

// ── fetch spy ─────────────────────────────────────────────────────────
const fetchCalls = [];
function fetchSpy(url, opts) {
  fetchCalls.push({ url, opts });
  return Promise.resolve({
    ok: true, status: 200,
    json: () => Promise.resolve({ ok: true }),
  });
}

// ── Sandbox ───────────────────────────────────────────────────────────
const React = makeReact();
const sandbox = {
  React,
  ReactDOM: { createRoot: () => ({ render() {} }) },
  window: { MAST_RUNTIME: { mode: "live", version: "test" } },
  document: { getElementById: () => null },  // skip auto-mount
  fetch: fetchSpy,
  setTimeout: (fn) => 0, clearTimeout: () => {},
  setInterval: () => 0, clearInterval: () => {},
  console,
  JSON, Math, Object, Array, String, Number, Boolean, Symbol,
  encodeURIComponent,
};
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(transpiled, sandbox, { filename: "agents-ui.js" });

// Expose internals: the module assigns views to window via Object.assign, but
// the helper components we need aren't on window. Re-run a probe that re-reads
// them from the module scope by re-declaring against the same context.
// Simpler: re-evaluate a tail snippet that captures the symbols onto window.
vm.runInContext(
  "window.__T = { AGInterruptHost, AGBackendBadge, AGLiveCtx, normalizeInterrupt, useAGBackend };",
  sandbox, { filename: "probe.js" });
const T = sandbox.window.__T;

function fail(msg) { console.error("FAIL: " + msg); process.exit(1); }

// Walk a rendered tree and collect every onClick handler keyed by the visible
// text of its element (so we can find e.g. the "Approve" button).
function collectButtons(node, acc) {
  acc = acc || [];
  (function walk(n) {
    if (n == null || n === false) return;
    if (Array.isArray(n)) { n.forEach(walk); return; }
    if (typeof n !== "object") return;
    // Only treat <button> host elements as clickable controls — NOT the modal
    // backdrop <div onClick={onClose}>, whose text spans the whole dialog and
    // would otherwise shadow the real action buttons.
    if (n.type === "button" && n.props && typeof n.props.onClick === "function") {
      acc.push({ onClick: n.props.onClick, text: collectText(n).join("").trim() });
    }
    if (n.children) walk(n.children);
    if (n.props && n.props.children) walk(n.props.children);
  })(node);
  return acc;
}
// Find a button by exact (trimmed) visible label.
function findButton(buttons, label) {
  return buttons.find(b => b.text === label);
}

// ── Test #24: pending interrupt → modal renders, Approve POSTs to resolve ──
// This is the FULL operator path: snapshot.pending_interrupts → AGInterruptHost
// mounts the real AGDangerousModal → operator clicks Approve → live POST to
// /interrupts/<id>/resolve. No mocking of the components under test; only the
// fetch network boundary is spied.
(function testInterruptApproveEndToEnd() {
  const live = {
    backend: "orchestrator",
    capabilities: { hold: true, interject: true, model_switch_live: true },
    pending_interrupts: [
      { event_id: "intr-77", agent_id: "instrument_control",
        skill: "TipPulse", params: { bias_v: 8.0 },
        rationale: "tip needs a pulse", kind: "dangerous" },
    ],
  };
  React._setContext(T.AGLiveCtx, live);
  fetchCalls.length = 0;
  const tree = React._render(T.AGInterruptHost, {});
  if (fetchCalls.length !== 0) fail("merely rendering must not fire the network");

  const buttons = collectButtons(tree);
  const approve = findButton(buttons, "Approve");
  if (!approve) fail("Approve button not found in rendered DANGEROUS modal; buttons=" +
    JSON.stringify(buttons.map(b => b.text)));

  // Operator clicks Approve — the real onResolve('approve', {reason}) fires,
  // which must POST to /interrupts/<id>/resolve.
  approve.onClick();
  if (fetchCalls.length !== 1) fail("Approve did not issue exactly one fetch (got " +
    fetchCalls.length + ")");
  const c = fetchCalls[0];
  if (!/\/interrupts\/intr-77\/resolve$/.test(c.url)) fail("Approve hit wrong url: " + c.url);
  if (!c.opts || c.opts.method !== "POST") fail("resolve must be a POST");
  const body = JSON.parse(c.opts.body);
  if (body.verdict !== "approve") fail("verdict not forwarded: " + c.opts.body);
  console.log("APPROVE_POST_OK " + c.url + " " + c.opts.body);
})();

// ── Test #24b: id is url-encoded so weird event ids can't break the path ──
(function testInterruptIdEncoded() {
  const live = {
    backend: "orchestrator", capabilities: {},
    pending_interrupts: [{ event_id: "intr 9/x", agent_id: "instrument_control",
      skill: "MotorMove", params: {}, kind: "dangerous" }],
  };
  React._setContext(T.AGLiveCtx, live);
  fetchCalls.length = 0;
  const tree = React._render(T.AGInterruptHost, {});
  const approve = findButton(collectButtons(tree), "Approve");
  approve.onClick();
  const c = fetchCalls[0];
  if (!/\/interrupts\/intr%209%2Fx\/resolve$/.test(c.url)) fail("id not encoded: " + c.url);
  console.log("ENCODE_OK " + c.url);
})();

// ── Test #24c: no pending interrupts → host renders nothing actionable ──
(function testNoPending() {
  React._setContext(T.AGLiveCtx, { backend: "orchestrator", capabilities: {}, pending_interrupts: [] });
  fetchCalls.length = 0;
  const tree = React._render(T.AGInterruptHost, {});
  const buttons = collectButtons(tree).filter(b => /Approve|Reject|接受当前草稿/.test(b.text));
  if (buttons.length !== 0) fail("no modal should render with empty pending list");
  console.log("EMPTY_OK no modal when nothing pending");
})();

// ── Test #24d: ESCALATE-kind interrupt routes to AGEscalateModal + resolves ──
(function testEscalateRoute() {
  React._setContext(T.AGLiveCtx, {
    backend: "orchestrator", capabilities: {},
    pending_interrupts: [{ event_id: "esc-3", kind: "escalate_to_human" }],
  });
  fetchCalls.length = 0;
  const tree = React._render(T.AGInterruptHost, {});
  const btn = findButton(collectButtons(tree), "接受当前草稿");
  if (!btn) fail("escalate modal accept button not found; buttons=" +
    JSON.stringify(collectButtons(tree).map(b => b.text)));
  btn.onClick();
  if (fetchCalls.length !== 1) fail("escalate resolve did not POST once");
  const c = fetchCalls[0];
  if (!/\/interrupts\/esc-3\/resolve$/.test(c.url)) fail("escalate hit wrong url: " + c.url);
  const body = JSON.parse(c.opts.body);
  if (body.verdict !== "accept") fail("escalate verdict not forwarded: " + c.opts.body);
  console.log("ESCALATE_OK " + c.url + " " + c.opts.body);
})();

// ── Test #25a: AGBackendBadge renders degraded label when mission_planner ──
(function testDegradedBadge() {
  React._setContext(T.AGLiveCtx, { backend: "mission_planner", capabilities: {} });
  const tree = React._render(T.AGBackendBadge, {});
  const flat = JSON.stringify(collectText(tree));
  if (!/降级/.test(flat)) fail("degraded badge missing '降级' label: " + flat);
  console.log("BADGE_DEGRADED_OK");
})();

// ── Test #25b: AGBackendBadge renders 'online' when orchestrator ──────
(function testOnlineBadge() {
  React._setContext(T.AGLiveCtx, { backend: "orchestrator", capabilities: {} });
  const tree = React._render(T.AGBackendBadge, {});
  const flat = JSON.stringify(collectText(tree));
  if (!/编排器在线/.test(flat)) fail("online badge missing label: " + flat);
  console.log("BADGE_ONLINE_OK");
})();

// ── Test #25c: AGBackendBadge renders NOTHING before first snapshot ───
(function testNoBackendNoBadge() {
  React._setContext(T.AGLiveCtx, null);
  const tree = React._render(T.AGBackendBadge, {});
  if (tree != null) fail("badge should render null when backend unknown");
  console.log("BADGE_NULL_OK");
})();

// ── Test #25d: useAGBackend marks degraded + closes capabilities ──────
(function testUseBackendDegraded() {
  React._setContext(T.AGLiveCtx, { backend: "mission_planner",
    capabilities: { hold: false, model_switch_live: false } });
  const got = React._render(() => T.useAGBackend(), {});
  if (got.degraded !== true) fail("useAGBackend.degraded should be true");
  if (got.cap("hold") !== false) fail("hold capability should be closed");
  if (got.cap("model_switch_live") !== false) fail("model switch should be closed");
  console.log("USEBACKEND_OK");
})();

function collectText(node) {
  const acc = [];
  (function walk(n) {
    if (n == null || n === false) return;
    if (typeof n === "string" || typeof n === "number") { acc.push(String(n)); return; }
    if (Array.isArray(n)) { n.forEach(walk); return; }
    if (n.children) walk(n.children);
    if (n.props && n.props.children) walk(n.props.children);
  })(node);
  return acc;
}

console.log("ALL_OK");
