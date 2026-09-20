// Flatten a CompositeSpec control-flow tree (the recursive nodes/body/then/else
// shape from skills/composite/spec.py to_dict) into a read-only react-flow graph.
// The spec is NOT a flat nodes/edges graph; it is a sequential tree of
// step | if | loop | set | llm | human | agent nodes, where branch bodies live
// under `body` (loop), `then`/`else` (if). We walk it sequentially, emit one
// react-flow node per spec node, and connect them in execution order. Unknown
// shapes degrade gracefully (callers fall back to a node list).

import type { Edge, Node } from "@xyflow/react";

export type SpecNode = Record<string, unknown>;

export interface FlowResult {
  nodes: Node[];
  edges: Edge[];
}

const X_STEP = 260; // horizontal indent per nesting depth
const Y_STEP = 92; // vertical gap between sequential nodes

function nodeLabel(n: SpecNode): { title: string; sub: string; kind: string } {
  const kind = String(n.type ?? "step");
  const id = String(n.id ?? "?");
  switch (kind) {
    case "step":
      return { title: String(n.skill ?? "step"), sub: `step · ${id}`, kind };
    case "if":
      return { title: `if ${String(n.cond ?? "")}`, sub: `if · ${id}`, kind };
    case "loop":
      return {
        title: `loop (${String(n.mode ?? "repeat")})`,
        sub: `loop · ${id}`,
        kind,
      };
    case "set":
      return {
        title: `${String(n.var ?? "?")} = ${String(n.value ?? "")}`,
        sub: `set · ${id}`,
        kind,
      };
    case "llm":
      return { title: `llm (${String(n.mode ?? "")})`, sub: `llm · ${id}`, kind };
    case "human":
      return { title: "human", sub: `human · ${id}`, kind };
    case "agent":
      return { title: String(n.agent ?? "agent"), sub: `agent · ${id}`, kind };
    case "try":
      return { title: "try / finally", sub: `try · ${id}`, kind };
    case "succeed":
      return { title: `succeed ${String(n.reason ?? "")}`.trim(), sub: `succeed · ${id}`, kind };
    case "fail":
      return { title: `fail ${String(n.reason ?? "")}`.trim(), sub: `fail · ${id}`, kind };
    case "break":
    case "continue":
      return { title: kind, sub: `${kind} · ${id}`, kind };
    default:
      return { title: kind, sub: id, kind };
  }
}

export const KIND_TONE: Record<string, string> = {
  step: "#38bdf8",
  if: "#fbbf24",
  loop: "#a78bfa",
  set: "#94a3b8",
  llm: "#34d399",
  human: "#f87171",
  agent: "#22d3ee",
  try: "#c084fc",
  break: "#fb923c",
  continue: "#fb923c",
  succeed: "#4ade80",
  fail: "#f87171",
};

/** Convert a spec.nodes tree into react-flow nodes + edges (best-effort). */
export function specToFlow(spec: SpecNode | null | undefined): FlowResult {
  const nodes: Node[] = [];
  const edges: Edge[] = [];
  if (!spec || !Array.isArray(spec.nodes)) return { nodes, edges };

  let row = 0; // running vertical position (in Y_STEP units)
  let auto = 0; // fallback id counter

  const addEdge = (from: string, to: string, label?: string) => {
    edges.push({
      id: `e-${from}-${to}-${edges.length}`,
      source: from,
      target: to,
      label,
      animated: false,
      style: { stroke: "#475569" },
      labelStyle: { fill: "#94a3b8", fontSize: 10 },
    });
  };

  // Walk a sequence; return {first, last} ids so the parent can chain.
  const walk = (
    seq: SpecNode[],
    depth: number,
    prev: string | null,
  ): { first: string | null; last: string | null } => {
    let first: string | null = null;
    let last: string | null = prev;
    for (const raw of seq) {
      if (!raw || typeof raw !== "object") continue;
      const n = raw as SpecNode;
      const nid = String(n.id ?? `__n${auto++}`);
      const { title, sub, kind } = nodeLabel(n);
      nodes.push({
        id: nid,
        position: { x: depth * X_STEP, y: row * Y_STEP },
        data: { label: title, sub, kind },
        type: "default",
        style: {
          background: "#0f172a",
          color: "#e2e8f0",
          border: `1px solid ${KIND_TONE[kind] ?? "#475569"}`,
          borderRadius: 8,
          fontSize: 12,
          width: 220,
          padding: 6,
        },
      });
      row += 1;
      if (first === null) first = nid;
      if (last) addEdge(last, nid);

      // Recurse into branch bodies, chaining their entry to this node.
      if (kind === "if") {
        const thenSeq = Array.isArray(n.then) ? (n.then as SpecNode[]) : [];
        const elseSeq = Array.isArray(n.else) ? (n.else as SpecNode[]) : [];
        const t = walk(thenSeq, depth + 1, null);
        if (t.first) addEdge(nid, t.first, "then");
        const e = walk(elseSeq, depth + 1, null);
        if (e.first) addEdge(nid, e.first, "else");
        // After an if, the next sequential node continues from the if node.
        last = nid;
      } else if (kind === "loop") {
        const bodySeq = Array.isArray(n.body) ? (n.body as SpecNode[]) : [];
        const b = walk(bodySeq, depth + 1, null);
        if (b.first) addEdge(nid, b.first, "body");
        if (b.last && b.last !== nid) addEdge(b.last, nid, "↺");
        last = nid;
      } else if (kind === "try") {
        const bodySeq = Array.isArray(n.body) ? (n.body as SpecNode[]) : [];
        const finSeq = Array.isArray(n.finally) ? (n.finally as SpecNode[]) : [];
        const b = walk(bodySeq, depth + 1, null);
        if (b.first) addEdge(nid, b.first, "try");
        const f = walk(finSeq, depth + 1, null);
        if (f.first) addEdge(nid, f.first, "finally");
        last = nid;
      } else {
        last = nid;
      }
    }
    return { first, last };
  };

  walk(spec.nodes as SpecNode[], 0, null);
  return { nodes, edges };
}

/** Flatten the tree into ordered rows for the degraded list fallback. */
export function specToRows(
  spec: SpecNode | null | undefined,
): { depth: number; kind: string; title: string; sub: string }[] {
  const out: { depth: number; kind: string; title: string; sub: string }[] = [];
  if (!spec || !Array.isArray(spec.nodes)) return out;
  const walk = (seq: SpecNode[], depth: number) => {
    for (const raw of seq) {
      if (!raw || typeof raw !== "object") continue;
      const n = raw as SpecNode;
      const { title, sub, kind } = nodeLabel(n);
      out.push({ depth, kind, title, sub });
      if (Array.isArray(n.then)) walk(n.then as SpecNode[], depth + 1);
      if (Array.isArray(n.else)) walk(n.else as SpecNode[], depth + 1);
      if (Array.isArray(n.body)) walk(n.body as SpecNode[], depth + 1);
      if (Array.isArray(n.finally)) walk(n.finally as SpecNode[], depth + 1);
    }
  };
  walk(spec.nodes as SpecNode[], 0);
  return out;
}
