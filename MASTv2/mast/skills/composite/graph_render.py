"""Render a CompositeSpec (or a static Python composite's plan) as a flow graph.

Two outputs:

  * :func:`build_graph` — a structured ``{"nodes": [...], "edges": [...]}`` graph
    (decision diamonds for ``if``, back-edges for ``loop``). Pure + testable.
  * :func:`to_mermaid` — Mermaid ``flowchart`` source built from that graph, so
    the GUI can draw a real flowchart with conditional branches and loop
    bodies for any composite that has them.

A static (Python) composite that only declares a flat ``plan()`` is rendered as
a straight-line sequence via :func:`steps_to_graph`, so existing composites are
visualised too (just without branches/loops they don't express declaratively).
"""

from __future__ import annotations

from typing import Any

from mast.skills.composite.spec import CompositeSpec


class _Builder:
    def __init__(self) -> None:
        self.nodes: list[dict] = []
        self.edges: list[dict] = []
        self._n = 0

    def node(self, kind: str, label: str, **extra) -> str:
        nid = f"n{self._n}"
        self._n += 1
        self.nodes.append({"id": nid, "kind": kind, "label": label, **extra})
        return nid

    def edge(self, src: str, dst: str, label: str = "") -> None:
        if src and dst:
            self.edges.append({"from": src, "to": dst, "label": label})

    # Walk a node list. Returns (entry_id, exit_ids) where exit_ids are the
    # dangling nodes that the following node should connect from.
    def walk(self, nodes: list[dict], incoming: list[str]) -> list[str]:
        cur = incoming
        for node in nodes:
            cur = self._emit(node, cur)
        return cur

    def _emit(self, node: dict, incoming: list[str]) -> list[str]:
        ntype = node.get("type")
        nid_label = node.get("id", "")
        if ntype == "step":
            label = node.get("skill", "?")
            if nid_label:
                label = f"{label}\n[{nid_label}]"
            n = self.node("step", label, optional=bool(node.get("optional")))
            for src in incoming:
                self.edge(src, n)
            return [n]

        if ntype == "set":
            n = self.node("set", f"{node.get('var', '?')} = {node.get('value', '')}")
            for src in incoming:
                self.edge(src, n)
            return [n]

        if ntype == "if":
            dec = self.node("if", node.get("cond", "?"))
            for src in incoming:
                self.edge(src, dec)
            then_exits = self.walk(node.get("then") or [], [dec])
            # label the first 'then' edge
            self._label_first_edge(dec, "是")
            else_nodes = node.get("else") or []
            if else_nodes:
                else_exits = self.walk(else_nodes, [dec])
                self._label_edge(dec, else_exits and self._first_target(dec, exclude="是"), "否")
            else:
                else_exits = [dec]  # false path falls through
            return then_exits + else_exits

        if ntype == "loop":
            mode = node.get("mode", "loop")
            bound = (node.get("count") or node.get("iterable")
                     or node.get("cond") or "")
            loop = self.node("loop", f"{mode} {bound}".strip(), mode=mode)
            for src in incoming:
                self.edge(src, loop)
            body_exits = self.walk(node.get("body") or [], [loop])
            self._label_first_edge(loop, "每次")
            # back-edge body → loop
            for ex in body_exits:
                self.edge(ex, loop, "循环")
            return [loop]  # exit on loop-done

        if ntype == "agent":
            dec = self.node("if", f"🤝 {node.get('agent', '?')}: "
                                  f"{node.get('task', '')}"[:60])
            for src in incoming:
                self.edge(src, dec)
            err_exits = self.walk(node.get("on_error") or [], [dec])
            if node.get("on_error"):
                self._label_first_edge(dec, "失败")
            return list(dict.fromkeys([dec, *err_exits]))

        if ntype == "human":
            dec = self.node("if", f"✋ {node.get('message', '人工决策')}"[:60])
            for src in incoming:
                self.edge(src, dec)
            routes = node.get("routes") or {}
            if not routes:
                return [dec]
            exits: list[str] = []
            for rname, rlist in routes.items():
                branch_exits = self.walk(rlist or [], [dec])
                self._label_first_edge(dec, rname)
                exits.extend(branch_exits)
            return list(dict.fromkeys(exits)) or [dec]

        if ntype == "llm":
            mode = node.get("mode", "route")
            dec = self.node("if", f"🤖 {node.get('responsibility', 'LLM 决策')}"[:60])
            for src in incoming:
                self.edge(src, dec)
            if mode == "route":
                exits: list[str] = []
                esc = node.get("escape")
                for rname, rlist in (node.get("routes") or {}).items():
                    label = f"{rname}{' (escape)' if rname == esc else ''}"
                    branch_exits = self.walk(rlist or [], [dec])
                    self._label_first_edge(dec, label)
                    exits.extend(branch_exits)
                # dedupe，保持顺序
                return list(dict.fromkeys(exits)) or [dec]
            err_exits = self.walk(node.get("on_error") or [], [dec])
            self._label_first_edge(dec, "失败")
            return list({*err_exits, dec})

        # unknown
        n = self.node("note", f"? {ntype}")
        for src in incoming:
            self.edge(src, n)
        return [n]

    # -- small helpers for edge labelling --
    def _first_target(self, src: str, exclude: str) -> str | None:
        for e in self.edges:
            if e["from"] == src and e["label"] != exclude:
                return e["to"]
        return None

    def _label_first_edge(self, src: str, label: str) -> None:
        for e in self.edges:
            if e["from"] == src and not e["label"]:
                e["label"] = label
                return

    def _label_edge(self, src: str, dst: str | None, label: str) -> None:
        if dst is None:
            return
        for e in self.edges:
            if e["from"] == src and e["to"] == dst and not e["label"]:
                e["label"] = label
                return


def build_graph(spec: CompositeSpec) -> dict:
    """Structured flow graph for *spec*: {start, end, nodes, edges}."""
    b = _Builder()
    start = b.node("start", "开始")
    exits = b.walk(spec.nodes, [start])
    end = b.node("end", "完成")
    for ex in exits:
        b.edge(ex, end)
    return {"start": start, "end": end, "nodes": b.nodes, "edges": b.edges}


def steps_to_graph(steps: list) -> dict:
    """Straight-line graph from a static plan() list of CompositeStep."""
    b = _Builder()
    start = b.node("start", "开始")
    cur = [start]
    for st in steps:
        label = getattr(st, "skill_name", "?")
        sid = getattr(st, "step_id", "")
        if sid:
            label = f"{label}\n[{sid}]"
        n = b.node("step", label, optional=getattr(st, "optional", False))
        for src in cur:
            b.edge(src, n)
        cur = [n]
    end = b.node("end", "完成")
    for src in cur:
        b.edge(src, end)
    return {"start": start, "end": end, "nodes": b.nodes, "edges": b.edges}


_SHAPE = {
    "start": ("([", "])"), "end": ("([", "])"),
    "step": ("[", "]"), "set": ("[/", "/]"),
    "if": ("{", "}"), "loop": ("[[", "]]"), "note": ("(", ")"),
}


def _mm_escape(text: str) -> str:
    return (str(text).replace('"', "'").replace("\n", "<br/>")
            .replace("[", "(").replace("]", ")"))


def to_mermaid(graph: dict) -> str:
    """Mermaid ``flowchart TD`` source for a :func:`build_graph` result."""
    lines = ["flowchart TD"]
    for n in graph["nodes"]:
        o, c = _SHAPE.get(n["kind"], ("[", "]"))
        lines.append(f'    {n["id"]}{o}"{_mm_escape(n["label"])}"{c}')
    for e in graph["edges"]:
        if e.get("label"):
            lines.append(f'    {e["from"]} -->|{_mm_escape(e["label"])}| {e["to"]}')
        else:
            lines.append(f'    {e["from"]} --> {e["to"]}')
    return "\n".join(lines)


def spec_to_mermaid(spec: CompositeSpec) -> str:
    return to_mermaid(build_graph(spec))


# ─────────────────────────────────────────────────────────────────────────
# Offline HTML flowchart (no external mermaid.js — works in the frozen binary)
# ─────────────────────────────────────────────────────────────────────────

def _esc(s: Any) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


_PALETTE = {
    "step": ("#0d9488", "#ecfdf5"),    # teal
    "set": ("#6366f1", "#eef2ff"),     # indigo
    "if": ("#b45309", "#fffbeb"),      # amber
    "loop": ("#7c3aed", "#f5f3ff"),    # violet
}


def _param_summary(params: dict) -> str:
    if not params:
        return ""
    bits = []
    for k, v in list(params.items())[:6]:
        if isinstance(v, dict) and set(v.keys()) == {"$expr"}:
            bits.append(f"{k}=<i>{_esc(v['$expr'])}</i>")
        else:
            bits.append(f"{k}={_esc(v)}")
    return ", ".join(bits)


def _render_nodes_html(nodes: list, depth: int = 0) -> str:
    out: list[str] = []
    for node in (nodes or []):
        t = node.get("type")
        nid = node.get("id", "")
        if t == "step":
            border, bg = _PALETTE["step"]
            params = _param_summary(node.get("params") or {})
            opt = ' · <span style="color:#b45309">可选</span>' if node.get("optional") else ""
            out.append(
                f'<div style="margin:4px 0;padding:6px 10px;border-left:3px solid {border};'
                f'background:{bg};border-radius:4px;">'
                f'<b>{_esc(node.get("skill","?"))}</b>'
                f'<span style="color:#64748b;font-size:0.85em;"> [{_esc(nid)}]{opt}</span>'
                + (f'<div style="color:#475569;font-size:0.85em;font-family:monospace;">{params}</div>'
                   if params else "")
                + "</div>")
        elif t == "set":
            border, bg = _PALETTE["set"]
            out.append(
                f'<div style="margin:3px 0;padding:3px 10px;border-left:3px solid {border};'
                f'background:{bg};border-radius:4px;font-family:monospace;font-size:0.88em;">'
                f'{_esc(node.get("var","?"))} = {_esc(node.get("value",""))}</div>')
        elif t == "if":
            border, bg = _PALETTE["if"]
            out.append(
                f'<div style="margin:5px 0;padding:6px 10px;border:1px solid {border};'
                f'background:{bg};border-radius:6px;">'
                f'<div style="font-weight:600;color:{border};">◆ 条件: '
                f'<span style="font-family:monospace;">{_esc(node.get("cond","?"))}</span></div>'
                f'<div style="margin-left:14px;border-left:2px dashed #16a34a;padding-left:10px;margin-top:4px;">'
                f'<div style="color:#16a34a;font-size:0.8em;font-weight:600;">是 ↓</div>'
                f'{_render_nodes_html(node.get("then") or [], depth+1) or "<i style=color:#94a3b8>（空）</i>"}</div>'
                + (f'<div style="margin-left:14px;border-left:2px dashed #dc2626;padding-left:10px;margin-top:4px;">'
                   f'<div style="color:#dc2626;font-size:0.8em;font-weight:600;">否 ↓</div>'
                   f'{_render_nodes_html(node.get("else") or [], depth+1)}</div>'
                   if node.get("else") else "")
                + "</div>")
        elif t == "loop":
            border, bg = _PALETTE["loop"]
            mode = node.get("mode", "loop")
            bound = (node.get("count") or node.get("iterable") or node.get("cond") or "")
            label = {"repeat": f"重复 {bound} 次", "foreach": f"遍历 {bound}",
                     "while": f"当 {bound} 时"}.get(mode, f"{mode} {bound}")
            out.append(
                f'<div style="margin:5px 0;padding:6px 10px;border:1px solid {border};'
                f'background:{bg};border-radius:6px;">'
                f'<div style="font-weight:600;color:{border};">↻ 循环: {_esc(label)} '
                f'<span style="color:#64748b;font-size:0.8em;">(变量 {_esc(node.get("var","i"))})</span></div>'
                f'<div style="margin-left:14px;border-left:2px solid {border};padding-left:10px;margin-top:4px;">'
                f'{_render_nodes_html(node.get("body") or [], depth+1) or "<i style=color:#94a3b8>（空）</i>"}</div>'
                "</div>")
    return "\n".join(out)


def spec_to_html(spec: CompositeSpec) -> str:
    """Self-contained HTML flowchart for *spec* (offline; no mermaid.js).

    Renders the control-flow tree as nested boxes: steps, ``set`` pills,
    ``if`` branches (是/否), and ``loop`` blocks — so conditionals and loops are
    visible at a glance inside the frozen binary too."""
    body = _render_nodes_html(spec.nodes) or '<i style="color:#94a3b8">（无步骤）</i>'
    return (
        '<div style="font-size:0.92em;line-height:1.4;">'
        '<div style="text-align:center;color:#64748b;margin-bottom:4px;">▼ 开始 ▼</div>'
        f'{body}'
        '<div style="text-align:center;color:#64748b;margin-top:4px;">▼ 完成 ▼</div>'
        '</div>')


__all__ = ["build_graph", "steps_to_graph", "to_mermaid", "spec_to_mermaid",
           "spec_to_html"]
