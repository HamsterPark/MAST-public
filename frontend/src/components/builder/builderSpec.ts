// Pure CompositeSpec tree operations for the 技能构建器 (composite skill builder).
// Ports the spec-manipulation core of the old builder-ui.jsx (git 8bef1a1) to
// typed TS. The document format = the backend CompositeSpec JSON (single source
// of truth, skills/composite/spec.py). Sequential order == execution order;
// if/loop/llm/human/agent are containers whose branch bodies live under
// then/else (if), body (loop), routes.<name>/on_error (llm/human/agent).
//
// Every mutation returns a NEW spec (deep-cloned) — never mutates in place.

export type NodeKind =
  | "step"
  | "if"
  | "loop"
  | "set"
  | "llm"
  | "human"
  | "agent"
  | "try"
  | "break"
  | "continue"
  | "succeed"
  | "fail";

export type SpecNode = Record<string, unknown> & { type?: string; id?: string };

export interface Spec {
  name: string;
  description: string;
  version: number;
  safety_level: "auto" | "confirm" | "dangerous" | string;
  params: SpecParam[];
  nodes: SpecNode[];
  outputs: SpecOutput[];
  success_when?: string;
  fail_message?: string;
  estimated_duration_s?: number;
  tags: string[];
  author?: string;
  notes?: string;
  [k: string]: unknown;
}

export interface SpecParam {
  name: string;
  type: "int" | "float" | "str" | "bool" | string;
  default: unknown;
  description?: string;
  required?: boolean;
}

export interface SpecOutput {
  name: string;
  expr: string;
}

// addr identifies an insertion list within the tree.
export interface Addr {
  containerId: string | null;
  slot: "root" | "then" | "else" | "body" | "on_error" | string; // route:<name>
}

export const NAME_RE = /^[A-Za-z0-9_一-鿿][A-Za-z0-9_\-一-鿿]{0,80}$/;
export const ROUTE_RE = /^[A-Za-z_][A-Za-z0-9_]{0,40}$/;

export const deep = <T,>(o: T): T => JSON.parse(JSON.stringify(o));

export const isExpr = (v: unknown): v is { $expr: string } =>
  !!v &&
  typeof v === "object" &&
  Object.keys(v as object).length === 1 &&
  "$expr" in (v as object);

export function emptySpec(name = ""): Spec {
  return {
    name,
    description: "",
    version: 0,
    safety_level: "confirm",
    params: [],
    nodes: [],
    outputs: [],
    success_when: "",
    fail_message: "",
    estimated_duration_s: 0,
    tags: [],
    author: "",
    notes: "",
  };
}

const CONTAINER_ROUTE_KINDS = ["llm", "human", "agent"];

interface FindHit {
  node: SpecNode;
  list: SpecNode[];
  index: number;
  owner: { node: SpecNode; slot: string } | null;
}

export function findIn(
  list: SpecNode[] | undefined,
  id: string,
  owner: { node: SpecNode; slot: string } | null,
): FindHit | null {
  const arr = list || [];
  for (let i = 0; i < arr.length; i++) {
    const n = arr[i]!;
    if (n.id === id) return { node: n, list: arr, index: i, owner };
    if (n.type === "if") {
      const r =
        findIn(n.then as SpecNode[], id, { node: n, slot: "then" }) ||
        findIn(n.else as SpecNode[], id, { node: n, slot: "else" });
      if (r) return r;
    } else if (n.type === "loop") {
      const r = findIn(n.body as SpecNode[], id, { node: n, slot: "body" });
      if (r) return r;
    } else if (n.type === "try") {
      const r =
        findIn(n.body as SpecNode[], id, { node: n, slot: "body" }) ||
        findIn(n.finally as SpecNode[], id, { node: n, slot: "finally" });
      if (r) return r;
    } else if (CONTAINER_ROUTE_KINDS.includes(String(n.type))) {
      const routes = (n.routes as Record<string, SpecNode[]>) || {};
      for (const rname of Object.keys(routes)) {
        const r = findIn(routes[rname], id, { node: n, slot: `route:${rname}` });
        if (r) return r;
      }
      const r2 = findIn(n.on_error as SpecNode[], id, { node: n, slot: "on_error" });
      if (r2) return r2;
    }
  }
  return null;
}

export function listByAddr(spec: Spec, addr: Addr | null): SpecNode[] | null {
  if (!addr || !addr.containerId) return spec.nodes;
  const hit = findIn(spec.nodes, addr.containerId, null);
  if (!hit) return null;
  const n = hit.node;
  if (addr.slot && addr.slot.startsWith("route:")) {
    const rname = addr.slot.slice(6);
    if (!n.routes) n.routes = {};
    const routes = n.routes as Record<string, SpecNode[]>;
    if (!Array.isArray(routes[rname])) routes[rname] = [];
    return routes[rname];
  }
  if (!Array.isArray(n[addr.slot])) n[addr.slot] = [];
  return n[addr.slot] as SpecNode[];
}

export function allIds(list: SpecNode[] | undefined, out?: Set<string>): Set<string> {
  out = out || new Set();
  for (const n of list || []) {
    if (n.id) out.add(n.id);
    if (n.type === "if") {
      allIds(n.then as SpecNode[], out);
      allIds(n.else as SpecNode[], out);
    }
    if (n.type === "loop") allIds(n.body as SpecNode[], out);
    if (n.type === "try") {
      allIds(n.body as SpecNode[], out);
      allIds(n.finally as SpecNode[], out);
    }
    if (CONTAINER_ROUTE_KINDS.includes(String(n.type))) {
      for (const rl of Object.values((n.routes as Record<string, SpecNode[]>) || {}))
        allIds(rl, out);
      allIds(n.on_error as SpecNode[], out);
    }
  }
  return out;
}

export function genId(spec: Spec, base?: string): string {
  const ids = allIds(spec.nodes);
  let stem = (base || "node")
    .replace(/[^A-Za-z0-9_一-鿿]/g, "_")
    .toLowerCase();
  if (!stem) stem = "node";
  if (!ids.has(stem)) return stem;
  for (let i = 2; ; i++) if (!ids.has(`${stem}_${i}`)) return `${stem}_${i}`;
}

export function newNode(spec: Spec, kind: NodeKind, skillName?: string): SpecNode {
  if (kind === "step")
    return { type: "step", id: genId(spec, skillName), skill: skillName, params: {} };
  if (kind === "if")
    return { type: "if", id: genId(spec, "if"), cond: "True", then: [], else: [] };
  if (kind === "loop")
    return {
      type: "loop",
      id: genId(spec, "loop"),
      mode: "repeat",
      count: "3",
      max_iter: 100,
      body: [],
    };
  if (kind === "llm")
    return {
      type: "llm",
      id: genId(spec, "decide"),
      mode: "route",
      responsibility: "（写一句话职责）",
      inputs: {},
      routes: { proceed: [], escalate: [] },
      route_descriptions: { proceed: "继续", escalate: "升级人工" },
      escape: "escalate",
    };
  if (kind === "human")
    return {
      type: "human",
      id: genId(spec, "ask"),
      message: "需要人工决策：{原因}",
      inputs: {},
      routes: { continue: [], abort: [] },
    };
  if (kind === "agent")
    return {
      type: "agent",
      id: genId(spec, "delegate"),
      agent: "literature",
      task: "（描述要委托的任务，可用 {名} 插值 inputs）",
      inputs: {},
      max_model_calls: 8,
      timeout_s: 600,
      on_error: [],
    };
  if (kind === "try")
    return { type: "try", id: genId(spec, "try"), body: [], finally: [] };
  if (kind === "break") return { type: "break", id: genId(spec, "break") };
  if (kind === "continue") return { type: "continue", id: genId(spec, "continue") };
  if (kind === "succeed")
    return { type: "succeed", id: genId(spec, "succeed"), reason: "'done'" };
  if (kind === "fail")
    return { type: "fail", id: genId(spec, "fail"), reason: "'failed'" };
  return { type: "set", id: genId(spec, "set"), var: "x", value: "0" };
}

export function specInsert(
  spec: Spec,
  addr: Addr,
  index: number | null,
  node: SpecNode,
): Spec {
  const s = deep(spec);
  const list = listByAddr(s, addr);
  if (!list) return spec;
  const i =
    index === null || index === undefined || index > list.length
      ? list.length
      : index;
  list.splice(i, 0, node);
  return s;
}

export function specRemove(spec: Spec, id: string): Spec {
  const s = deep(spec);
  const hit = findIn(s.nodes, id, null);
  if (!hit) return spec;
  hit.list.splice(hit.index, 1);
  return s;
}

export function specUpdate(spec: Spec, id: string, patch: Record<string, unknown>): Spec {
  const s = deep(spec);
  const hit = findIn(s.nodes, id, null);
  if (!hit) return spec;
  Object.assign(hit.node, patch);
  return s;
}

export function specMove(spec: Spec, id: string, delta: number): Spec {
  const s = deep(spec);
  const hit = findIn(s.nodes, id, null);
  if (!hit) return spec;
  const j = hit.index + delta;
  if (j < 0 || j >= hit.list.length) return spec;
  const [n] = hit.list.splice(hit.index, 1);
  if (n) hit.list.splice(j, 0, n);
  return s;
}

// Node id rename cascades all $expr / cond / count / iterable / value refs
// (otherwise a dangling reference only blows up at runtime safe_eval).
export function renameRefs(spec: Spec, oldId: string, newId: string): Spec {
  const s = deep(spec);
  const esc = oldId.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  const re = new RegExp(
    `(?<![A-Za-z0-9_\\u4e00-\\u9fff])${esc}(?![A-Za-z0-9_\\u4e00-\\u9fff])`,
    "g",
  );
  const fix = (str: unknown) => String(str).replace(re, newId);
  const walk = (list: SpecNode[] | undefined) => {
    for (const n of list || []) {
      if (typeof n.cond === "string") n.cond = fix(n.cond);
      if (typeof n.count === "string") n.count = fix(n.count);
      if (typeof n.iterable === "string") n.iterable = fix(n.iterable);
      if (typeof n.value === "string") n.value = fix(n.value);
      if (typeof n.reason === "string") n.reason = fix(n.reason);
      const params = (n.params as Record<string, unknown>) || {};
      for (const k of Object.keys(params)) {
        const v = params[k];
        if (isExpr(v)) params[k] = { $expr: fix(v.$expr) };
      }
      const inputs = (n.inputs as Record<string, unknown>) || {};
      for (const k of Object.keys(inputs)) {
        const v = inputs[k];
        if (isExpr(v)) inputs[k] = { $expr: fix(v.$expr) };
      }
      if (n.type === "if") {
        walk(n.then as SpecNode[]);
        walk(n.else as SpecNode[]);
      } else if (n.type === "loop") {
        walk(n.body as SpecNode[]);
      } else if (n.type === "try") {
        walk(n.body as SpecNode[]);
        walk(n.finally as SpecNode[]);
      } else if (CONTAINER_ROUTE_KINDS.includes(String(n.type))) {
        for (const rl of Object.values((n.routes as Record<string, SpecNode[]>) || {}))
          walk(rl);
        walk(n.on_error as SpecNode[]);
      }
    }
  };
  walk(s.nodes);
  return s;
}

// ── local drafts (localStorage; per-browser, never written to the server) ──

const DRAFT_PREFIX = "mast_builder_draft_";
export const draftKey = (name: string) => DRAFT_PREFIX + (name || "__new__");

export interface Draft {
  key: string;
  name: string;
  spec: Spec;
  baseVersion: number;
  staging: string[];
  ts: number;
  nodes: number;
}

export function listDrafts(): Draft[] {
  const out: Draft[] = [];
  try {
    for (let i = 0; i < localStorage.length; i++) {
      const k = localStorage.key(i);
      if (!k || k.indexOf(DRAFT_PREFIX) !== 0) continue;
      let d: { spec?: Spec; baseVersion?: number; staging?: string[]; ts?: number } | null = null;
      try {
        d = JSON.parse(localStorage.getItem(k) || "null");
      } catch {
        continue;
      }
      if (!d || !d.spec) continue;
      out.push({
        key: k,
        name: k.slice(DRAFT_PREFIX.length),
        spec: d.spec,
        baseVersion: d.baseVersion || 0,
        staging: d.staging || [],
        ts: d.ts || 0,
        nodes: (d.spec.nodes || []).length,
      });
    }
  } catch {
    /* localStorage unavailable */
  }
  out.sort((a, b) => b.ts - a.ts);
  return out;
}

export function writeDraft(spec: Spec, baseVersion: number, staging: string[]): void {
  try {
    localStorage.setItem(
      draftKey(spec.name),
      JSON.stringify({ spec, baseVersion, staging, ts: Date.now() }),
    );
  } catch {
    /* quota — ignore */
  }
}

export function removeDraft(key: string): void {
  try {
    localStorage.removeItem(key);
  } catch {
    /* ignore */
  }
}
