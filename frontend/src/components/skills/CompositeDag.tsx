import { useMemo } from "react";
import { ReactFlow, Background, Controls, type Node } from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import {
  specToFlow,
  specToRows,
  KIND_TONE,
  type SpecNode,
} from "./compositeGraph";

/** Read-only DAG visualization of a composite spec's control-flow tree.
 *  Renders with @xyflow/react when the spec parses into a graph; otherwise
 *  (no nodes / unexpected shape) degrades to an ordered, indented node list. */
export function CompositeDag({ spec }: { spec: SpecNode | null | undefined }) {
  const { nodes, edges } = useMemo(() => specToFlow(spec), [spec]);
  const rows = useMemo(() => specToRows(spec), [spec]);

  if (!rows.length) {
    return <p className="text-sm text-mast-muted">该工作流没有可视化节点。</p>;
  }

  // If react-flow produced a graph, show it. (It always does when rows exist,
  // but we keep the list as a guaranteed fallback for unusual shapes.)
  if (nodes.length) {
    const rfNodes: Node[] = nodes.map((n) => ({
      ...n,
      data: {
        ...n.data,
        label: (
          <div className="text-left leading-tight">
            <div className="truncate font-medium">{String((n.data as any).label)}</div>
            <div className="truncate text-[10px] opacity-60">
              {String((n.data as any).sub)}
            </div>
          </div>
        ),
      },
    }));
    return (
      <div className="h-[460px] w-full overflow-hidden rounded-lg border border-mast-border bg-mast-bg">
        <ReactFlow
          nodes={rfNodes}
          edges={edges}
          fitView
          nodesDraggable={false}
          nodesConnectable={false}
          elementsSelectable={false}
          proOptions={{ hideAttribution: true }}
        >
          <Background color="#1e293b" gap={18} />
          <Controls showInteractive={false} />
        </ReactFlow>
      </div>
    );
  }

  // Degraded fallback: indented ordered list.
  return <CompositeNodeList spec={spec} />;
}

/** Pure-DOM ordered node list (no react-flow) — always works. */
export function CompositeNodeList({ spec }: { spec: SpecNode | null | undefined }) {
  const rows = useMemo(() => specToRows(spec), [spec]);
  if (!rows.length) {
    return <p className="text-sm text-mast-muted">该工作流没有节点。</p>;
  }
  return (
    <ul className="space-y-1 text-sm">
      {rows.map((r, i) => (
        <li
          key={i}
          className="flex items-center gap-2"
          style={{ paddingLeft: `${r.depth * 18}px` }}
        >
          <span
            className="inline-block h-2 w-2 shrink-0 rounded-full"
            style={{ background: KIND_TONE[r.kind] ?? "#475569" }}
          />
          <span className="font-medium text-mast-text">{r.title}</span>
          <span className="text-xs text-mast-muted">{r.sub}</span>
        </li>
      ))}
    </ul>
  );
}
