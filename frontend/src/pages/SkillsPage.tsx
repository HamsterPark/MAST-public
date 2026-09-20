import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { type ColumnDef } from "@tanstack/react-table";
import { api } from "@/api/client";
import { DataTable } from "@/components/DataTable";
import {
  Section,
  Badge,
  Spinner,
  ErrorNote,
  DegradedNote,
} from "@/components/ui";
import { SubTabs, Modal } from "@/components/controls";
import { SkillCardDetail } from "@/components/skills/SkillCardDetail";
import { SkillOverrideEditor } from "@/components/skills/SkillOverrideEditor";
import { CompositesManager } from "@/components/skills/CompositesManager";
import { EncyclopediaView } from "@/components/skills/EncyclopediaView";
import { CodexReference } from "@/components/skills/CodexReference";
import { SkillOverlayManager } from "@/components/skills/SkillOverlayManager";
import { MarketView } from "@/components/skills/MarketView";
import { useStickyTab } from "@/hooks/useStickyTab";

// ── 技能 (full parity) ───────────────────────────────────────────────────────
// SubTabs reproduce the old nested Gradio surfaces (flat React state, no freeze):
//   技能目录   — catalog table + 来源/领域/安全 筛选 + 客户端搜索 + 行点详情
//   技能详情编辑 — 参数/前后置/时长/回滚/安全级 编辑, 经 admin overrides 写
//   复合技能   — 列表 + react-flow DAG + 版本史 + clone/restore/validate
//   覆盖层     — 用 <data>/config/skill_overlay 里的 .py 覆盖内置技能,
//                重载即生效(不重启); 首要职责是回答「现在生效了吗」
//   百科       — 领域分组 + intent 映射 / 配方
//   参考/Codex — 只读 IC 工作流/决策树/指南/顾问/实验设计/硬件参考/意图映射

type View =
  | "catalog"
  | "market"
  | "editor"
  | "composites"
  | "overlay"
  | "encyclopedia"
  | "codex";

// 一份数据同时喂 tab 条和 useStickyTab 的合法名单。
const VIEW_TABS: { id: View; label: string }[] = [
  { id: "catalog", label: "技能目录" },
  // 市场：全体技能是市场，订阅列表才是 agent 日常的装载面。紧挨「技能目录」——
  // 目录回答「有什么」，市场回答「他要用哪些」，是同一件事的两半。
  { id: "market", label: "市场" },
  { id: "editor", label: "技能详情编辑" },
  { id: "composites", label: "复合技能" },
  // 覆盖层：改 skill 不用发新版本。放在「复合技能」之后、「百科」之前 ——
  // 它和上面三个一样是**改东西**的地方，而后两个是查资料的地方。
  { id: "overlay", label: "覆盖层" },
  { id: "encyclopedia", label: "百科" },
  { id: "codex", label: "参考/Codex" },
];

const SAFETY_TONE: Record<string, string> = {
  AUTO: "AUTO",
  INFO: "INFO",
  WARN: "WARN",
  DANGEROUS: "DANGEROUS",
  auto: "AUTO",
  confirm: "WARN",
  dangerous: "DANGEROUS",
};

function useCatalog() {
  return useQuery({
    queryKey: ["skills", "catalog"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skills/catalog");
      if (error) throw error;
      return data;
    },
  });
}

type SkillRow = {
  name: string;
  domain: string;
  source: string;
  safety_level: string;
  composition_level?: string | null;
  summary?: string | null;
};

function FilterSelect({
  label,
  value,
  setValue,
  options,
}: {
  label: string;
  value: string;
  setValue: (v: string) => void;
  options: string[];
}) {
  return (
    <label className="flex items-center gap-1 text-xs text-mast-muted">
      {label}
      <select
        value={value}
        onChange={(e) => setValue(e.target.value)}
        className="rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text"
      >
        <option value="all">全部</option>
        {options.map((o) => (
          <option key={o} value={o}>{o}</option>
        ))}
      </select>
    </label>
  );
}

// ── 技能目录 ─────────────────────────────────────────────────────────────────
function CatalogView() {
  const q = useCatalog();
  const [text, setText] = useState("");
  const [source, setSource] = useState("all");
  const [domain, setDomain] = useState("all");
  const [safety, setSafety] = useState("all");
  const [level, setLevel] = useState("all");
  const [selected, setSelected] = useState<string | null>(null);

  const rows = (q.data?.index ?? []) as SkillRow[];

  const sources = useMemo(() => Array.from(new Set(rows.map((r) => r.source))).sort(), [rows]);
  const domains = useMemo(() => Array.from(new Set(rows.map((r) => r.domain))).sort(), [rows]);
  const safeties = useMemo(() => Array.from(new Set(rows.map((r) => r.safety_level))).sort(), [rows]);
  const levels = useMemo(
    () => Array.from(new Set(rows.map((r) => r.composition_level).filter(Boolean) as string[])).sort(),
    [rows],
  );

  const filtered = useMemo(
    () =>
      rows.filter(
        (r) =>
          (source === "all" || r.source === source) &&
          (domain === "all" || r.domain === domain) &&
          (safety === "all" || r.safety_level === safety) &&
          (level === "all" || r.composition_level === level),
      ),
    [rows, source, domain, safety, level],
  );

  const columns = useMemo<ColumnDef<SkillRow, any>[]>(
    () => [
      {
        accessorKey: "name",
        header: "名称",
        cell: (c) => <span className="font-medium text-mast-text">{c.getValue() as string}</span>,
      },
      { accessorKey: "domain", header: "领域" },
      { accessorKey: "source", header: "来源" },
      {
        accessorKey: "safety_level",
        header: "安全等级",
        cell: (c) => (
          <Badge tone={SAFETY_TONE[c.getValue() as string] ?? "default"}>{c.getValue() as string}</Badge>
        ),
      },
      { id: "composition_level", header: "层级", accessorFn: (r) => r.composition_level ?? "—" },
      {
        accessorKey: "summary",
        header: "摘要",
        cell: (c) => <span className="text-mast-muted">{(c.getValue() as string) || "—"}</span>,
      },
      {
        id: "open",
        header: "",
        cell: (c) => (
          <button
            onClick={() => setSelected(c.row.original.name)}
            className="rounded border border-mast-border px-2 py-0.5 text-xs text-mast-accent hover:bg-mast-accent/10"
          >
            查看
          </button>
        ),
      },
    ],
    [],
  );

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;
  if (q.data?.degraded) return <DegradedNote what="技能目录" />;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="搜索技能名 / 领域 / 摘要…"
          className="w-64 rounded border border-mast-border bg-mast-bg px-3 py-1.5 text-sm text-mast-text placeholder:text-mast-muted"
        />
        <FilterSelect label="来源" value={source} setValue={setSource} options={sources} />
        <FilterSelect label="领域" value={domain} setValue={setDomain} options={domains} />
        <FilterSelect label="安全" value={safety} setValue={setSafety} options={safeties} />
        {levels.length > 0 && (
          <FilterSelect label="层级" value={level} setValue={setLevel} options={levels} />
        )}
        <span className="text-xs text-mast-muted">共 {filtered.length} / {rows.length}</span>
      </div>

      <DataTable data={filtered} columns={columns} globalFilter={text} empty="无匹配技能" />

      {/* Skill detail opens in a modal instead of expanding
          inline below the table. Modal returns null while closed, so
          SkillCardDetail (and its on-demand query) only mounts when a row is
          selected — same lifecycle as the old inline guard. */}
      <Modal
        open={!!selected}
        onClose={() => setSelected(null)}
        title={selected ? `技能详情：${selected}` : "技能详情"}
        wide
      >
        {selected && <SkillCardDetail name={selected} />}
      </Modal>
    </div>
  );
}

// ── 技能详情编辑 wrapper (provides the skill-name list from the catalog) ──────
function EditorView() {
  const q = useCatalog();
  const rows = (q.data?.index ?? []) as SkillRow[];
  const names = useMemo(() => rows.map((r) => r.name).sort(), [rows]);

  if (q.isError) return <ErrorNote error={q.error} />;
  if (q.data?.degraded) return <DegradedNote what="技能目录" />;

  return <SkillOverrideEditor names={names} loadingNames={q.isPending} />;
}

export default function SkillsPage() {
  // 五个子页里「参考/Codex」和「百科」都是要翻着看的长文档,
  // 每次回来被打回「技能目录」等于每次重新找一遍。
  const [view, setView] = useStickyTab<View>(
    "skills", VIEW_TABS.map((t) => t.id), "catalog");

  return (
    <div>
      <Section title="技能">
        <SubTabs tabs={VIEW_TABS} value={view} onChange={setView} />
        {view === "catalog" && <CatalogView />}
        {view === "market" && <MarketView />}
        {view === "editor" && <EditorView />}
        {view === "composites" && <CompositesManager />}
        {view === "overlay" && <SkillOverlayManager />}
        {view === "encyclopedia" && <EncyclopediaView />}
        {view === "codex" && <CodexReference />}
      </Section>
    </div>
  );
}
