import { useMemo, useState } from "react";
import clsx from "clsx";
import type { NodeKind } from "./builderSpec";
import { paletteHiddenCount, paletteHides } from "@/lib/skillMarket";

// Control-flow nodes are FIRST-CLASS in the palette (not buried in a modal) so
// it is immediately obvious the builder supports 判断/循环/变量/决策, not just
// skills. Each is click-to-add (at end) AND draggable into any container slot
// (dataTransfer "text/nodekind", mirrored by BuilderCanvas' drop handlers).
const CONTROL_FLOW: { kind: NodeKind; label: string; hint: string }[] = [
  { kind: "if", label: "⑂ if 分支", hint: "条件为真/假走不同分支（then / else）" },
  { kind: "loop", label: "↻ loop 循环", hint: "重复 N 次 / 遍历序列 / while 条件循环（body）" },
  { kind: "set", label: "≔ set 变量", hint: "设置/更新一个变量，供后续表达式引用" },
  { kind: "llm", label: "🤖 llm 决策", hint: "受限路由 / 结构化输出的 LLM 决策节点" },
  { kind: "human", label: "✋ human 人工", hint: "暂停等待用户决策" },
  { kind: "agent", label: "🤝 agent 委托", hint: "委托给文献/分析/写作等子 agent" },
];

// Skill catalog palette for the builder. Ports the old builder-ui.jsx Palette:
// fuzzy search · category/source facet chips · grouped-by-domain collapsible list ·
// favorites (★, local-only) · staging basket (🧺, per-draft) · double-click /
// drag to add. Backed by GET /api/builder/catalog (richer than /api/skills/catalog:
// carries zh / category / level / tags — exactly the fields the old Palette
// displayed and searched over: name + 中文名, fuzzy over name/zh/domain/tags).

export interface CatalogEntry {
  name: string;
  zh?: string;
  domain: string;
  source: string;
  // builder/catalog uses `safety`; skills/catalog used `safety_level`. Accept both.
  safety?: string;
  safety_level?: string;
  category?: string;
  level?: number;
  tags?: string[];
  composition_level?: string | null;
  summary?: string | null;
  /** 在用户的订阅面上吗。**字段缺席时不过滤**（fail-open）—— 一个旧的/降级的
   *  响应不该让 palette 空掉。 */
  subscribed?: boolean;
  /** 必装项（不可退订）。这里只用来显示，判据真源在后端。 */
  mandatory?: boolean;
}

function safetyOf(e: CatalogEntry): string {
  return String(e.safety ?? e.safety_level ?? "");
}

const SAFETY_DOT: Record<string, string> = {
  AUTO: "var(--mast-auto)",
  auto: "var(--mast-auto)",
  WARN: "var(--mast-warn)",
  confirm: "var(--mast-warn)",
  CONFIRM: "var(--mast-warn)",
  DANGEROUS: "var(--mast-danger)",
  dangerous: "var(--mast-danger)",
};

// OLD builder-ui.jsx Palette facets (verbatim): category row 读/写/组合/分析,
// then source row 内置/论文/用户组合. (No safety facet row in the old palette.)
const CATEGORY_FACETS: { val: string; label: string }[] = [
  { val: "READ", label: "读" },
  { val: "WRITE", label: "写" },
  { val: "COMPOSITE", label: "组合" },
  { val: "ANALYSIS", label: "分析" },
];

const SOURCE_FACETS: { val: string; label: string }[] = [
  { val: "builtin", label: "内置" },
  { val: "paper", label: "论文" },
  { val: "user_composite", label: "用户组合" },
];

export function BuilderPalette({
  catalog,
  favorites,
  onToggleFav,
  staging,
  setStaging,
  onAddSkill,
  onAddAtEnd,
  onAddKind,
  onHover,
  loading,
}: {
  catalog: CatalogEntry[];
  favorites: string[];
  onToggleFav: (name: string) => void;
  staging: string[];
  setStaging: (next: string[]) => void;
  onAddSkill: (name: string) => void;
  onAddAtEnd?: () => void;
  onAddKind?: (kind: NodeKind) => void;
  onHover: (name: string | null, ev?: React.MouseEvent) => void;
  loading: boolean;
}) {
  const [q, setQ] = useState("");
  const [cat, setCat] = useState("");
  const [src, setSrc] = useState("");
  const [open, setOpen] = useState<Record<string, boolean>>({});
  // 默认只看订阅 —— palette 是「他要用的那些」，市场是「本机有的全部」。
  // 未定制时（出厂态）每一条都是已订阅，所以这个默认对新机器是恒真的，不改变行为。
  const [subOnly, setSubOnly] = useState(true);

  /** 被订阅过滤挡掉的条数。**必须显示出来** —— 静默截断会被读成「市场里只有这些」，
   *  而这里的后果更实在：他会以为本机没有那个技能，转头去写一个重复的。 */
  const hiddenBySub = useMemo(
    () => paletteHiddenCount(catalog, subOnly),
    [catalog, subOnly],
  );

  const filtered = useMemo(() => {
    const ql = q.trim().toLowerCase();
    const qt = q.trim();
    return catalog.filter((e) => {
      // 判据在 lib/skillMarket（可被 node --test 覆盖），不在这里现写一遍。
      if (paletteHides(e, subOnly)) return false;
      if (cat && e.category !== cat) return false;
      if (src && e.source !== src) return false;
      if (!ql) return true;
      return (
        e.name.toLowerCase().includes(ql) ||
        (e.zh || "").includes(qt) ||
        (e.domain || "").includes(qt) ||
        (e.tags || []).some((t) => t.toLowerCase().includes(ql)) ||
        (e.summary || "").toLowerCase().includes(ql)
      );
    });
  }, [catalog, q, cat, src, subOnly]);

  const groups = useMemo(() => {
    const g: Record<string, CatalogEntry[]> = {};
    for (const e of filtered) (g[e.domain] = g[e.domain] || []).push(e);
    return Object.entries(g).sort((a, b) => b[1].length - a[1].length);
  }, [filtered]);

  const favEntries = catalog.filter((e) => favorites.includes(e.name));

  const Row = (e: CatalogEntry) => (
    <div
      key={e.name}
      draggable
      onDragStart={(ev) => ev.dataTransfer.setData("text/skill", e.name)}
      onDoubleClick={() => onAddSkill(e.name)}
      onMouseEnter={(ev) => onHover(e.name, ev)}
      onMouseLeave={() => onHover(null)}
      title="双击添加到工作流末尾；拖到画布同效"
      className="group flex cursor-grab items-center gap-2 rounded px-2 py-1 text-sm hover:bg-mast-bg"
    >
      <span
        className="inline-block h-2 w-2 shrink-0 rounded-full"
        style={{ background: SAFETY_DOT[safetyOf(e)] ?? "var(--mast-ag-sup)" }}
      />
      <span className="flex-1 truncate text-xs">
        <span className={clsx("font-mono", e.subscribed === false
          ? "text-mast-muted" : "text-mast-text")}>{e.name}</span>
        {e.zh ? <span className="ml-1 text-mast-muted">{e.zh}</span> : null}
      </span>
      {/* 未订阅的仍然可以用在工作流里 —— composite 的子步走 ExecutionContext，
          那条路不看订阅。所以这里是一个**标注**，不是一个禁用。 */}
      {e.subscribed === false && (
        <span
          className="shrink-0 rounded border border-mast-border px-1 text-[10px] text-mast-muted"
          title="不在你的订阅面上（agent 看不见它，但工作流仍可调用）"
        >
          未订阅
        </span>
      )}
      <span className="shrink-0 text-[10px] text-mast-muted">L{e.level ?? e.composition_level ?? 0}</span>
      <button
        title="加入本草稿候选"
        onClick={(ev) => {
          ev.stopPropagation();
          if (!staging.includes(e.name)) setStaging([...staging, e.name]);
        }}
        className="shrink-0 text-xs opacity-0 group-hover:opacity-100"
      >
        🧺
      </button>
      <button
        onClick={(ev) => {
          ev.stopPropagation();
          onToggleFav(e.name);
        }}
        className={clsx(
          "shrink-0 text-xs",
          favorites.includes(e.name) ? "text-mast-warn" : "text-mast-muted opacity-0 group-hover:opacity-100",
        )}
        title={favorites.includes(e.name) ? "取消收藏" : "收藏"}
      >
        ★
      </button>
    </div>
  );

  const Chip = (
    label: string,
    val: string,
    cur: string,
    set: (v: string) => void,
  ) => (
    <button
      key={label + val}
      onClick={() => set(cur === val ? "" : val)}
      className={clsx(
        "rounded-full border px-2 py-0.5 text-xs",
        cur === val
          ? "border-mast-accent bg-mast-accent/15 text-mast-accent"
          : "border-mast-border text-mast-muted hover:text-mast-text",
      )}
    >
      {label}
    </button>
  );

  return (
    <div className="flex h-full flex-col gap-2">
      <input
        value={q}
        onChange={(e) => setQ(e.target.value)}
        placeholder={`搜索 ${catalog.length} 个技能…（名称/中文/标签/域）`}
        className="rounded-md border border-mast-border bg-mast-bg px-2.5 py-1.5 text-sm text-mast-text outline-none focus:border-mast-accent"
      />
      {onAddAtEnd && (
        <button
          onClick={onAddAtEnd}
          className="rounded-md border border-mast-accent/60 bg-mast-accent/10 px-2.5 py-1.5 text-sm font-medium text-mast-accent hover:bg-mast-accent/20"
        >
          ＋ 添加节点到末尾
        </button>
      )}

      {/* 控制流节点 — first-class, so 判断/循环 are obviously available */}
      {onAddKind && (
        <div className="rounded-md border border-mast-border bg-mast-bg p-2">
          <div className="mb-1.5 text-[11px] font-medium text-mast-muted">
            控制流 · 判断 / 循环 / 变量（点击加到末尾，或拖入分支槽）
          </div>
          <div className="flex flex-wrap gap-1">
            {CONTROL_FLOW.map((c) => (
              <button
                key={c.kind}
                draggable
                onDragStart={(ev) => ev.dataTransfer.setData("text/nodekind", c.kind)}
                onClick={() => onAddKind(c.kind)}
                title={c.hint + "（双击/拖拽同效）"}
                className="cursor-grab rounded border border-mast-border-strong bg-mast-panel px-2 py-1 text-xs text-mast-text hover:border-mast-accent hover:text-mast-accent"
              >
                {c.label}
              </button>
            ))}
          </div>
        </div>
      )}
      <div className="flex flex-wrap gap-1">
        {CATEGORY_FACETS.map((f) => Chip(f.label, f.val, cat, setCat))}
      </div>
      <div className="flex flex-wrap gap-1">
        {SOURCE_FACETS.map((f) => Chip(f.label, f.val, src, setSrc))}
      </div>

      {/* 订阅面 vs 全市场。隐藏了多少**必须说出来** —— 不然他会以为本机没有那个
          技能，转头去写一个重复的。点这一行就切到全市场。 */}
      <div className="flex flex-wrap items-center gap-2 text-xs">
        <button
          type="button"
          onClick={() => setSubOnly(!subOnly)}
          className={clsx(
            "rounded border px-1.5 py-0.5",
            subOnly
              ? "border-mast-border-strong bg-mast-panel text-mast-text"
              : "border-mast-border text-mast-muted",
          )}
        >
          {subOnly ? "只看订阅" : "全市场"}
        </button>
        {subOnly && hiddenBySub > 0 && (
          <button
            type="button"
            onClick={() => setSubOnly(false)}
            className="text-mast-accent underline decoration-dotted"
            title="订阅列表在「技能 → 市场」维护"
          >
            另有 {hiddenBySub} 个未订阅（点此显示）
          </button>
        )}
      </div>

      {staging.length > 0 && (
        <div className="rounded border border-mast-border bg-mast-bg p-2">
          <div className="mb-1 text-xs text-mast-muted">🧺 本草稿候选（点击添加）</div>
          <div className="flex flex-wrap gap-1">
            {staging.map((s) => (
              <span
                key={s}
                onClick={() => onAddSkill(s)}
                className="flex cursor-pointer items-center gap-1 rounded bg-mast-panel px-1.5 py-0.5 text-xs text-mast-text"
              >
                {s}
                <span
                  onClick={(e) => {
                    e.stopPropagation();
                    setStaging(staging.filter((x) => x !== s));
                  }}
                  className="text-mast-muted hover:text-mast-danger"
                >
                  ✕
                </span>
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="min-h-0 flex-1 overflow-auto pr-1">
        {loading && <p className="px-2 py-1 text-xs text-mast-muted">加载技能目录…</p>}
        {!loading && !catalog.length && (
          <p className="px-2 py-1 text-xs text-mast-muted">技能目录为空（内核未连接？）。</p>
        )}
        {favEntries.length > 0 && (
          <div className="mb-2">
            <div className="px-2 py-1 text-xs font-medium text-mast-warn">
              ★ 收藏 <span className="text-mast-muted">{favEntries.length}</span>
            </div>
            {favEntries.map(Row)}
          </div>
        )}
        {groups.map(([dom, list]) => {
          const collapsed = open[dom] === false;
          return (
            <div key={dom} className="mb-1.5">
              <button
                onClick={() => setOpen({ ...open, [dom]: !collapsed ? false : true })}
                className="flex w-full items-center gap-1 px-2 py-1 text-left text-xs font-medium text-mast-text hover:text-mast-accent"
              >
                <span>{collapsed ? "▸" : "▾"}</span>
                <span className="flex-1 truncate">{dom}</span>
                <span className="text-mast-muted">{list.length}</span>
              </button>
              {!collapsed && list.map(Row)}
            </div>
          );
        })}
      </div>
      <p className="px-1 text-[11px] text-mast-muted">双击/拖拽添加 · ★收藏 · 🧺存入草稿候选</p>
    </div>
  );
}
