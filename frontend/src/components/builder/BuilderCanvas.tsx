import clsx from "clsx";
import { KIND_TONE } from "@/components/skills/compositeGraph";
import type { Spec, SpecNode, Addr, NodeKind } from "./builderSpec";

// Structured, ordered tree canvas for the builder. Sequential order == execution
// order; if/loop/llm/human/agent are containers with branch slots. Replaces the
// old react-flow free canvas with an explicit nested-list editor (clearer for
// per-node edit, and never relies on drag for correctness). Each node row has:
// select · move up/down · delete · "add node here". Container slots render their
// children recursively and an "add into <slot>" button.

const KIND_LABEL: Record<string, string> = {
  step: "step",
  if: "if",
  loop: "loop",
  set: "set",
  llm: "llm",
  human: "human",
  agent: "agent",
  try: "try",
  break: "break",
  continue: "continue",
  succeed: "succeed",
  fail: "fail",
};

function nodeTitle(n: SpecNode): string {
  const kind = String(n.type ?? "step");
  switch (kind) {
    case "step":
      return String(n.skill ?? "step");
    case "if":
      return `if ${String(n.cond ?? "")}`;
    case "loop":
      return `loop (${String(n.mode ?? "repeat")})`;
    case "set":
      return `${String(n.var ?? "?")} = ${String(n.value ?? "")}`;
    case "llm":
      return `llm (${String(n.mode ?? "route")})`;
    case "human":
      return "human";
    case "agent":
      return String(n.agent ?? "agent");
    case "try":
      return "try / finally";
    case "break":
      return "break · 跳出循环";
    case "continue":
      return "continue · 下一轮";
    case "succeed":
      return `succeed ${String(n.reason ?? "")}`.trim();
    case "fail":
      return `fail ${String(n.reason ?? "")}`.trim();
    default:
      return kind;
  }
}

const ROUTE_KINDS = ["llm", "human", "agent"];

// Human-readable slot labels — mirror the old builder-ui.jsx slot captions
// ("then 为真" / "else 为假" / "body 循环体" / "on_error 失败时") so block nesting
// reads like a code editor rather than bare keys.
const SLOT_LABEL: Record<string, string> = {
  then: "then · 为真",
  else: "else · 为假",
  body: "body · 主体",
  finally: "finally · 总会执行",
  on_error: "on_error · 失败时",
};
function slotLabel(slot: string): string {
  if (SLOT_LABEL[slot]) return SLOT_LABEL[slot];
  if (slot.startsWith("route:")) return `route · ${slot.slice(6)}`;
  return slot;
}

export function BuilderCanvas({
  spec,
  selId,
  errorsById,
  onSelect,
  onMove,
  onRemove,
  onAddInto,
  onAddAfter,
  onDropSkill,
  onDropKind,
}: {
  spec: Spec;
  selId: string | null;
  errorsById: Record<string, { errors: string[]; warnings: string[] }>;
  onSelect: (id: string) => void;
  onMove: (id: string, delta: number) => void;
  onRemove: (id: string) => void;
  onAddInto: (addr: Addr) => void;
  onAddAfter: (id: string) => void;
  onDropSkill: (name: string, addr: Addr) => void;
  onDropKind?: (kind: NodeKind, addr: Addr) => void;
}) {
  // a drop may carry a skill ("text/skill") or a control-flow node kind
  // ("text/nodekind"); route to the right handler.
  const handleDrop = (e: React.DragEvent, addr: Addr) => {
    const kind = e.dataTransfer.getData("text/nodekind");
    if (kind && onDropKind) {
      onDropKind(kind as NodeKind, addr);
      return;
    }
    const name = e.dataTransfer.getData("text/skill");
    if (name) onDropSkill(name, addr);
  };
  // A slot = one nested block (then/else/body/route:*/on_error). It renders an
  // INDENTED column with a colored guide line (tinted by the parent container's
  // kind) so nesting depth is visually obvious. `tone` = parent container color.
  const renderSlot = (
    slot: string,
    list: SpecNode[] | undefined,
    addr: Addr,
    tone: string,
    depth: number,
  ) => {
    const items = list || [];
    return (
      <div
        key={slot}
        className="relative ml-4 pl-3"
        onDragOver={(e) => e.preventDefault()}
        onDrop={(e) => {
          e.preventDefault();
          e.stopPropagation();
          handleDrop(e, addr);
        }}
      >
        {/* connecting guide line, colored by the owning container's kind */}
        <span
          aria-hidden
          className="absolute bottom-2 left-0 top-1.5 w-px rounded"
          style={{ background: tone, opacity: 0.5 }}
        />
        {/* slot caption badge */}
        <div className="flex items-center gap-1.5 py-0.5">
          <span
            aria-hidden
            className="h-px w-2.5 shrink-0"
            style={{ background: tone, opacity: 0.6 }}
          />
          <span
            className="rounded px-1.5 py-px text-[10px] font-medium tracking-wide text-mast-muted"
            style={{ background: `color-mix(in srgb, ${tone} 13%, transparent)`, color: tone }}
          >
            {slotLabel(slot)}
          </span>
          {!items.length && <span className="text-[10px] text-mast-muted/70">（空）</span>}
        </div>
        {items.map((n) => renderNode(n, depth + 1))}
        <button
          onClick={() => onAddInto(addr)}
          className="my-1 rounded border border-dashed border-mast-border px-2 py-0.5 text-xs text-mast-muted hover:border-mast-accent hover:text-mast-accent"
        >
          ＋ 在此添加节点
        </button>
      </div>
    );
  };

  const renderNode = (n: SpecNode, depth = 0) => {
    const id = String(n.id ?? "?");
    const kind = String(n.type ?? "step");
    const sel = selId === id;
    const rep = errorsById[id];
    const hasErr = rep && rep.errors.length > 0;
    const hasWarn = rep && rep.warnings.length > 0;
    const tone = KIND_TONE[kind] ?? "var(--mast-ag-sup)";
    const isContainer =
      kind === "if" || kind === "loop" || kind === "try" || ROUTE_KINDS.includes(kind);
    return (
      <div key={id} className="my-1">
        <div
          onClick={(e) => {
            e.stopPropagation();
            onSelect(id);
          }}
          className={clsx(
            "flex cursor-pointer items-center gap-2 rounded-md border px-2 py-1.5",
            sel ? "border-mast-accent bg-mast-accent/10" : "border-mast-border bg-mast-panel hover:bg-mast-bg",
            hasErr && "ring-1 ring-mast-danger-border",
          )}
          // containers carry a colored left edge so the header that "opens" a
          // nested block is easy to spot at any depth
          style={isContainer ? { borderLeft: `3px solid ${tone}` } : undefined}
        >
          <span className="inline-block h-2.5 w-2.5 shrink-0 rounded-full" style={{ background: tone }} />
          <span className="shrink-0 rounded bg-mast-bg px-1 text-[10px] uppercase text-mast-muted">
            {KIND_LABEL[kind] ?? kind}
          </span>
          <span className="flex-1 truncate text-sm text-mast-text" title={nodeTitle(n)}>
            {nodeTitle(n)}
          </span>
          <span className="shrink-0 font-mono text-[10px] text-mast-muted">{id}</span>
          {hasErr && <span title={rep.errors.join("; ")} className="text-mast-danger">⛔</span>}
          {!hasErr && hasWarn && <span title={rep.warnings.join("; ")} className="text-mast-warn">⚠</span>}
          <span className="flex shrink-0 items-center gap-0.5">
            <IconBtn title="上移" onClick={() => onMove(id, -1)}>▲</IconBtn>
            <IconBtn title="下移" onClick={() => onMove(id, +1)}>▼</IconBtn>
            <IconBtn title="在此之后添加" onClick={() => onAddAfter(id)}>＋</IconBtn>
            <IconBtn title="删除" onClick={() => onRemove(id)} danger>✕</IconBtn>
          </span>
        </div>

        {/* container slots — each renders an indented, guide-lined nested block */}
        {kind === "if" && (
          <div className="mt-1 space-y-1">
            {renderSlot("then", n.then as SpecNode[], { containerId: id, slot: "then" }, tone, depth)}
            {renderSlot("else", n.else as SpecNode[], { containerId: id, slot: "else" }, tone, depth)}
          </div>
        )}
        {kind === "loop" && (
          <div className="mt-1">
            {renderSlot("body", n.body as SpecNode[], { containerId: id, slot: "body" }, tone, depth)}
          </div>
        )}
        {kind === "try" && (
          <div className="mt-1 space-y-1">
            {renderSlot("body", n.body as SpecNode[], { containerId: id, slot: "body" }, tone, depth)}
            {renderSlot("finally", n.finally as SpecNode[], { containerId: id, slot: "finally" }, tone, depth)}
          </div>
        )}
        {ROUTE_KINDS.includes(kind) && (
          <div className="mt-1 space-y-1">
            {Object.keys((n.routes as Record<string, SpecNode[]>) || {}).map((rname) =>
              renderSlot(
                `route:${rname}`,
                (n.routes as Record<string, SpecNode[]>)[rname],
                { containerId: id, slot: `route:${rname}` },
                tone,
                depth,
              ),
            )}
            {(kind === "llm" || kind === "agent") &&
              renderSlot("on_error", n.on_error as SpecNode[], { containerId: id, slot: "on_error" }, tone, depth)}
          </div>
        )}
      </div>
    );
  };

  return (
    <div
      className="h-full overflow-auto rounded-lg border border-mast-border bg-mast-bg p-3"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        handleDrop(e, { containerId: null, slot: "root" });
      }}
      onClick={() => onSelect("")}
    >
      {!spec.nodes.length ? (
        <div className="flex h-full min-h-[200px] flex-col items-center justify-center gap-1 text-center text-mast-muted">
          <div className="text-sm">从左侧双击 / 拖入技能开始组合</div>
          <div className="text-xs">或用下方“＋ 添加节点”加入 if / loop / set / llm 等节点</div>
        </div>
      ) : (
        spec.nodes.map((n) => renderNode(n))
      )}
      <button
        onClick={(e) => {
          e.stopPropagation();
          onAddInto({ containerId: null, slot: "root" });
        }}
        className="mt-2 w-full rounded border border-dashed border-mast-border px-2 py-1 text-xs text-mast-muted hover:border-mast-accent hover:text-mast-accent"
      >
        ＋ 添加节点到末尾
      </button>
    </div>
  );
}

function IconBtn({
  children,
  onClick,
  title,
  danger,
}: {
  children: React.ReactNode;
  onClick: () => void;
  title: string;
  danger?: boolean;
}) {
  return (
    <button
      title={title}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
      className={clsx(
        "rounded px-1 text-xs text-mast-muted hover:bg-mast-bg",
        danger ? "hover:text-mast-danger" : "hover:text-mast-accent",
      )}
    >
      {children}
    </button>
  );
}

export type AddKind = NodeKind;
