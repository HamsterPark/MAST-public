import { useMemo, useState, type ReactNode, type ReactElement } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import {
  Card,
  Badge,
  Spinner,
  ErrorNote,
  DegradedNote,
  EmptyNote,
} from "@/components/ui";
import { SubTabs } from "@/components/controls";

// ── 技能 → IC 参考 / Codex (read-only) ──────────────────────────────────────
// Restores the old Gradio "IC Workflows / Guide / Advisor / 实验设计 / 硬件参考"
// reference sub-views. Each is the structured JSON returned by
//   GET /api/knowledge/codex/{view}
// (view ∈ workflows/decision_trees/guide/advisor/experiment_strategies/
//  hardware_reference/intent_map). Pure read-only; renders tables/cards.
// Backing schemas: api/schemas_codex_reference.py (single source of types).

type CodexView =
  | "workflows"
  | "decision_trees"
  | "guide"
  | "advisor"
  | "experiment_strategies"
  | "hardware_reference"
  | "intent_map";

// The OpenAPI body for this endpoint is `unknown` (the backend uses
// response_model=None over a union of per-view models), so we mirror the
// Pydantic shapes locally. These match schemas_codex_reference.py exactly.
type CodexBase = { view: string; degraded?: boolean; detail?: string | null };

type WorkflowRecipe = { name: string; desc: string; chain: string[]; params: string };
type WorkflowsResponse = CodexBase & {
  recipes: WorkflowRecipe[];
  composite_hierarchy: Record<string, string[]>;
};

type DecisionOption = { label: string; skill: string };
type DecisionNode = { q: string; options: DecisionOption[] };
type DecisionTree = { id: string; title: string; nodes: DecisionNode[] };
type DecisionTreesResponse = CodexBase & {
  trees: DecisionTree[];
  composite_hierarchy: Record<string, string[]>;
};

type IntentEntry = { keywords: string; skill: string; note: string };
type GuideResponse = CodexBase & {
  intent_mapping: IntentEntry[];
  decision_trees: DecisionTree[];
  composite_hierarchy: Record<string, string[]>;
};

type AdvisorCategory = {
  id: string;
  name: string;
  name_en: string;
  description: string;
  completeness: string;
  n_phases: number;
  materials: string[];
  raw: Record<string, unknown>;
};
type AdvisorResponse = CodexBase & { categories: AdvisorCategory[] };

type ExperimentStrategiesResponse = CodexBase & {
  measurement_strategies: Record<string, unknown>;
  reference_experiments: Record<string, unknown>;
  anomaly_response: Record<string, unknown>;
};

type HardwareReferenceResponse = CodexBase & {
  piezo_materials: Record<string, unknown>;
  preamplifiers: Record<string, unknown>;
  controllers: Record<string, unknown>;
  motor_types: Record<string, unknown>;
  noise_formulas: Record<string, unknown>;
};

type IntentMapResponse = CodexBase & {
  intent_mapping: IntentEntry[];
  safety_constraints: Record<string, unknown>[];
  scan_speed_rule: Record<string, unknown>;
};

const VIEW_TABS: { id: CodexView; label: string; what: string }[] = [
  { id: "workflows", label: "工作流配方", what: "工作流配方" },
  { id: "decision_trees", label: "决策树", what: "决策树" },
  { id: "guide", label: "决策指南", what: "决策指南" },
  { id: "advisor", label: "样品顾问", what: "样品顾问" },
  { id: "experiment_strategies", label: "实验设计", what: "实验设计参考" },
  { id: "hardware_reference", label: "硬件参考", what: "硬件参考" },
  { id: "intent_map", label: "意图映射", what: "意图映射" },
];

// Read one codex view. The endpoint is degrade-safe (200 + degraded=true) and
// never freezes; the typed client returns the raw JSON we cast per view.
function useCodexView<T extends CodexBase>(view: CodexView) {
  return useQuery({
    queryKey: ["codex", view],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/knowledge/codex/{view}", {
        params: { path: { view } },
      });
      if (error) throw error;
      return data as unknown as T;
    },
  });
}

// ── shared renderers ─────────────────────────────────────────────────────────

function SearchBox({
  value,
  onChange,
  placeholder,
  count,
}: {
  value: string;
  onChange: (v: string) => void;
  placeholder: string;
  count?: string;
}) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <input
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        className="w-64 rounded border border-mast-border bg-mast-bg px-3 py-1.5 text-sm text-mast-text placeholder:text-mast-muted"
      />
      {count && <span className="text-xs text-mast-muted">{count}</span>}
    </div>
  );
}

function CompositeHierarchy({ map }: { map: Record<string, string[]> }) {
  const entries = Object.entries(map ?? {});
  if (!entries.length) return null;
  return (
    <Card className="space-y-2">
      <h3 className="text-sm font-semibold text-mast-text">复合技能层级</h3>
      <div className="space-y-1.5">
        {entries.map(([parent, children]) => (
          <div key={parent} className="flex flex-wrap items-center gap-1 text-xs">
            <span className="font-mono text-mast-accent">{parent}</span>
            <span className="text-mast-muted">→</span>
            {(children ?? []).map((c) => (
              <span key={c} className="rounded bg-mast-bg px-1.5 py-0.5 text-mast-muted">{c}</span>
            ))}
          </div>
        ))}
      </div>
    </Card>
  );
}

function DecisionTreeCard({ tree }: { tree: DecisionTree }) {
  return (
    <Card className="space-y-3">
      <h3 className="text-sm font-semibold text-mast-text">{tree.title}</h3>
      <div className="space-y-3">
        {tree.nodes.map((node, i) => (
          <div key={i} className="space-y-1">
            <p className="text-xs font-medium text-mast-text">Q{i + 1}. {node.q}</p>
            <div className="ml-3 space-y-1">
              {node.options.map((o, j) => (
                <div key={j} className="flex flex-wrap items-center gap-1 text-xs">
                  <span className="text-mast-muted">{o.label}</span>
                  <span className="text-mast-muted">→</span>
                  <span className="font-mono text-mast-accent">{o.skill || "—"}</span>
                </div>
              ))}
            </div>
          </div>
        ))}
        {!tree.nodes.length && <p className="text-xs text-mast-muted">（无节点）</p>}
      </div>
    </Card>
  );
}

function IntentTable({ rows }: { rows: IntentEntry[] }) {
  return (
    <Card>
      <div className="overflow-auto rounded border border-mast-border">
        <table className="w-full text-sm">
          <thead className="bg-mast-bg text-mast-muted">
            <tr>
              <th className="px-3 py-2 text-left">意图关键词</th>
              <th className="px-3 py-2 text-left">技能 / 配方</th>
              <th className="px-3 py-2 text-left">说明</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((m, i) => {
              const dangerous = (m.note ?? "").toUpperCase().includes("DANGEROUS");
              return (
                <tr key={i} className="border-t border-mast-border">
                  <td className="px-3 py-2 text-mast-text">{m.keywords}</td>
                  <td className="px-3 py-2 font-mono text-xs text-mast-accent">{m.skill}</td>
                  <td className="px-3 py-2 text-xs">
                    {dangerous ? (
                      <Badge tone="DANGEROUS">{m.note}</Badge>
                    ) : (
                      <span className="text-mast-muted">{m.note || "—"}</span>
                    )}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>
    </Card>
  );
}

// Permissive renderer for the hand-authored reference dicts (experiment design /
// hardware): a section of cards, one per top-level key, dumping nested values
// readably without assuming a fixed schema.
function renderValue(v: unknown): ReactNode {
  if (v === null || v === undefined) return <span className="text-mast-muted">—</span>;
  if (Array.isArray(v)) {
    return (
      <ul className="list-disc space-y-0.5 pl-4">
        {v.map((item, i) => (
          <li key={i}>{renderValue(item)}</li>
        ))}
      </ul>
    );
  }
  if (typeof v === "object") {
    return (
      <div className="space-y-1">
        {Object.entries(v as Record<string, unknown>).map(([k, val]) => (
          <div key={k} className="text-xs">
            <span className="font-medium text-mast-text">{k}</span>:{" "}
            <span className="text-mast-muted">{renderValue(val)}</span>
          </div>
        ))}
      </div>
    );
  }
  return <span className="text-mast-muted">{String(v)}</span>;
}

function DictSection({
  title,
  data,
}: {
  title: string;
  data: Record<string, unknown>;
}) {
  const entries = Object.entries(data ?? {});
  if (!entries.length) return null;
  return (
    <div className="space-y-2">
      <h3 className="text-sm font-semibold text-mast-text">{title}</h3>
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {entries.map(([k, v]) => (
          <Card key={k} className="space-y-1.5">
            <h4 className="text-xs font-semibold text-mast-accent">{k}</h4>
            <div className="text-xs">{renderValue(v)}</div>
          </Card>
        ))}
      </div>
    </div>
  );
}

// ── per-view panels ──────────────────────────────────────────────────────────

function WorkflowsView() {
  const q = useCodexView<WorkflowsResponse>("workflows");
  const [text, setText] = useState("");

  const recipes = q.data?.recipes ?? [];
  const filtered = useMemo(() => {
    const t = text.trim().toLowerCase();
    if (!t) return recipes;
    return recipes.filter(
      (r) =>
        r.name.toLowerCase().includes(t) ||
        r.desc.toLowerCase().includes(t) ||
        r.chain.some((s) => s.toLowerCase().includes(t)),
    );
  }, [recipes, text]);

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;
  if (q.data?.degraded) return <DegradedNote what="工作流配方" />;
  if (!recipes.length && !Object.keys(q.data?.composite_hierarchy ?? {}).length)
    return <EmptyNote label="暂无工作流数据" />;

  return (
    <div className="space-y-4">
      <SearchBox
        value={text}
        onChange={setText}
        placeholder="搜索配方名 / 描述 / 技能链…"
        count={`共 ${filtered.length} / ${recipes.length} 个配方`}
      />
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {filtered.map((r) => (
          <Card key={r.name} className="space-y-2">
            <h3 className="text-sm font-semibold text-mast-text">{r.name}</h3>
            {r.desc && <p className="text-xs text-mast-muted">{r.desc}</p>}
            {!!r.chain.length && (
              <div className="flex flex-wrap items-center gap-1">
                {r.chain.map((s, i) => (
                  <span key={i} className="flex items-center gap-1">
                    <span className="rounded bg-mast-bg px-1.5 py-0.5 font-mono text-xs text-mast-accent">{s}</span>
                    {i < r.chain.length - 1 && <span className="text-mast-muted">→</span>}
                  </span>
                ))}
              </div>
            )}
            {r.params && <p className="text-[11px] text-mast-muted/70">参数：{r.params}</p>}
          </Card>
        ))}
      </div>
      <CompositeHierarchy map={q.data?.composite_hierarchy ?? {}} />
    </div>
  );
}

function DecisionTreesView() {
  const q = useCodexView<DecisionTreesResponse>("decision_trees");

  const trees = q.data?.trees ?? [];

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;
  if (q.data?.degraded) return <DegradedNote what="决策树" />;
  if (!trees.length && !Object.keys(q.data?.composite_hierarchy ?? {}).length)
    return <EmptyNote label="暂无决策树数据" />;

  return (
    <div className="space-y-4">
      <span className="text-xs text-mast-muted">共 {trees.length} 棵决策树</span>
      <div className="grid gap-3 lg:grid-cols-2">
        {trees.map((t) => (
          <DecisionTreeCard key={t.id} tree={t} />
        ))}
      </div>
      <CompositeHierarchy map={q.data?.composite_hierarchy ?? {}} />
    </div>
  );
}

function GuideView() {
  const q = useCodexView<GuideResponse>("guide");
  const [text, setText] = useState("");

  const intents = q.data?.intent_mapping ?? [];
  const trees = q.data?.decision_trees ?? [];

  const filteredIntents = useMemo(() => {
    const t = text.trim().toLowerCase();
    if (!t) return intents;
    return intents.filter(
      (m) =>
        m.keywords.toLowerCase().includes(t) ||
        m.skill.toLowerCase().includes(t) ||
        (m.note ?? "").toLowerCase().includes(t),
    );
  }, [intents, text]);

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;
  if (q.data?.degraded) return <DegradedNote what="决策指南" />;
  if (!intents.length && !trees.length) return <EmptyNote label="暂无决策指南数据" />;

  return (
    <div className="space-y-4">
      <p className="text-xs text-mast-muted">
        LLM 决策指南：意图→技能映射 + 决策树 + 复合层级，一处汇总。
      </p>
      {!!intents.length && (
        <div className="space-y-2">
          <SearchBox
            value={text}
            onChange={setText}
            placeholder="搜索意图关键词 / 技能…"
            count={`共 ${filteredIntents.length} 条映射`}
          />
          <IntentTable rows={filteredIntents} />
        </div>
      )}
      {!!trees.length && (
        <div className="grid gap-3 lg:grid-cols-2">
          {trees.map((t) => (
            <DecisionTreeCard key={t.id} tree={t} />
          ))}
        </div>
      )}
      <CompositeHierarchy map={q.data?.composite_hierarchy ?? {}} />
    </div>
  );
}

const COMPLETENESS_TONE: Record<string, string> = {
  full: "AUTO",
  partial: "WARN",
  stub: "INFO",
};

function AdvisorView() {
  const q = useCodexView<AdvisorResponse>("advisor");
  const [text, setText] = useState("");

  const cats = q.data?.categories ?? [];
  const filtered = useMemo(() => {
    const t = text.trim().toLowerCase();
    if (!t) return cats;
    return cats.filter(
      (c) =>
        c.name.toLowerCase().includes(t) ||
        c.name_en.toLowerCase().includes(t) ||
        c.description.toLowerCase().includes(t) ||
        c.materials.some((m) => m.toLowerCase().includes(t)),
    );
  }, [cats, text]);

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;
  if (q.data?.degraded) return <DegradedNote what="样品顾问" />;
  if (!cats.length) return <EmptyNote label="暂无样品类别数据" />;

  return (
    <div className="space-y-4">
      <SearchBox
        value={text}
        onChange={setText}
        placeholder="搜索类别 / 材料…"
        count={`共 ${filtered.length} / ${cats.length} 个类别`}
      />
      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {filtered.map((c) => (
          <Card key={c.id} className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-sm font-semibold text-mast-text">
                {c.name}
                {c.name_en && <span className="ml-1 text-xs text-mast-muted">{c.name_en}</span>}
              </h3>
              <Badge tone={COMPLETENESS_TONE[c.completeness] ?? "default"}>{c.completeness}</Badge>
            </div>
            {c.description && <p className="text-xs text-mast-muted">{c.description}</p>}
            <p className="text-[11px] text-mast-muted/70">
              {c.n_phases} 个相 · {c.materials.length} 种材料
            </p>
            {!!c.materials.length && (
              <div className="flex flex-wrap gap-1">
                {c.materials.map((m) => (
                  <span key={m} className="rounded bg-mast-bg px-1.5 py-0.5 text-xs text-mast-muted">{m}</span>
                ))}
              </div>
            )}
          </Card>
        ))}
      </div>
    </div>
  );
}

function ExperimentStrategiesView() {
  const q = useCodexView<ExperimentStrategiesResponse>("experiment_strategies");

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;
  if (q.data?.degraded) return <DegradedNote what="实验设计参考" />;

  const ms = q.data?.measurement_strategies ?? {};
  const re = q.data?.reference_experiments ?? {};
  const ar = q.data?.anomaly_response ?? {};
  if (!Object.keys(ms).length && !Object.keys(re).length && !Object.keys(ar).length)
    return <EmptyNote label="暂无实验设计数据" />;

  return (
    <div className="space-y-6">
      <DictSection title="测量策略" data={ms} />
      <DictSection title="参考实验" data={re} />
      <DictSection title="异常响应" data={ar} />
    </div>
  );
}

function HardwareReferenceView() {
  const q = useCodexView<HardwareReferenceResponse>("hardware_reference");

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;
  if (q.data?.degraded) return <DegradedNote what="硬件参考" />;

  const piezo = q.data?.piezo_materials ?? {};
  const preamp = q.data?.preamplifiers ?? {};
  const ctrl = q.data?.controllers ?? {};
  const motors = q.data?.motor_types ?? {};
  const noise = q.data?.noise_formulas ?? {};
  if (![piezo, preamp, ctrl, motors, noise].some((d) => Object.keys(d).length))
    return <EmptyNote label="暂无硬件参考数据" />;

  return (
    <div className="space-y-6">
      <DictSection title="压电材料" data={piezo} />
      <DictSection title="前置放大器" data={preamp} />
      <DictSection title="控制器" data={ctrl} />
      <DictSection title="马达类型" data={motors} />
      <DictSection title="噪声公式" data={noise} />
    </div>
  );
}

function IntentMapView() {
  const q = useCodexView<IntentMapResponse>("intent_map");
  const [text, setText] = useState("");

  const intents = q.data?.intent_mapping ?? [];
  const constraints = q.data?.safety_constraints ?? [];
  const scanRule = q.data?.scan_speed_rule ?? {};

  const filtered = useMemo(() => {
    const t = text.trim().toLowerCase();
    if (!t) return intents;
    return intents.filter(
      (m) =>
        m.keywords.toLowerCase().includes(t) ||
        m.skill.toLowerCase().includes(t) ||
        (m.note ?? "").toLowerCase().includes(t),
    );
  }, [intents, text]);

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;
  if (q.data?.degraded) return <DegradedNote what="意图映射" />;
  if (!intents.length && !constraints.length && !Object.keys(scanRule).length)
    return <EmptyNote label="暂无意图映射数据" />;

  return (
    <div className="space-y-4">
      {!!intents.length && (
        <div className="space-y-2">
          <SearchBox
            value={text}
            onChange={setText}
            placeholder="搜索意图关键词 / 技能…"
            count={`共 ${filtered.length} / ${intents.length} 条映射`}
          />
          <IntentTable rows={filtered} />
        </div>
      )}
      {!!constraints.length && <DictSection title="安全约束" data={Object.fromEntries(constraints.map((c, i) => [String(c.name ?? c.id ?? i), c]))} />}
      {!!Object.keys(scanRule).length && (
        <Card className="space-y-1.5">
          <h3 className="text-sm font-semibold text-mast-text">扫描速度规则</h3>
          <div className="text-xs">{renderValue(scanRule)}</div>
        </Card>
      )}
    </div>
  );
}

// ── public component ─────────────────────────────────────────────────────────

const VIEW_COMPONENT: Record<CodexView, () => ReactElement> = {
  workflows: WorkflowsView,
  decision_trees: DecisionTreesView,
  guide: GuideView,
  advisor: AdvisorView,
  experiment_strategies: ExperimentStrategiesView,
  hardware_reference: HardwareReferenceView,
  intent_map: IntentMapView,
};

export function CodexReference() {
  const [view, setView] = useState<CodexView>("workflows");
  const Active = VIEW_COMPONENT[view];
  return (
    <div className="space-y-2">
      <SubTabs
        tabs={VIEW_TABS.map((t) => ({ id: t.id, label: t.label }))}
        value={view}
        onChange={setView}
      />
      <Active />
    </div>
  );
}
