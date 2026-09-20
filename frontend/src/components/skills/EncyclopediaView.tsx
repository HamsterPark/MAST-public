import { useMemo, useState } from "react";
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

// ── 百科 — domain groups + intent→skill mapping (decision-tree/recipe view) ──
// Reproduces encyclopedia.py surfaces:
//   领域分组（17 domains × skills, owning-agent badge）·
//   intent 映射（keywords → skill 配方 + note）+ 客户端搜索过滤.

type EncyView = "domains" | "intent";

const AGENT_LABEL: Record<string, string> = {
  instrument_control: "仪器控制 IC",
  data_processing: "数据处理 DP",
  literature: "文献 LIT",
  experiment_design: "实验设计 XD",
  paper_writing: "论文写作 PW",
  paper_review: "论文审稿 PR",
};

function useDomains() {
  return useQuery({
    queryKey: ["encyclopedia", "domains"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/encyclopedia/domains");
      if (error) throw error;
      return data;
    },
  });
}

function useIntentMapping() {
  return useQuery({
    queryKey: ["encyclopedia", "intent-mapping"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/encyclopedia/intent-mapping");
      if (error) throw error;
      return data;
    },
  });
}

function DomainsView() {
  const q = useDomains();
  const [agent, setAgent] = useState("all");
  const [text, setText] = useState("");

  const domains = q.data?.domains ?? [];

  const agents = useMemo(
    () => Array.from(new Set(domains.map((d) => d.agent).filter(Boolean) as string[])).sort(),
    [domains],
  );

  const filtered = useMemo(() => {
    const t = text.trim().toLowerCase();
    return domains.filter(
      (d) =>
        (agent === "all" || d.agent === agent) &&
        (!t ||
          d.name.toLowerCase().includes(t) ||
          (d.desc ?? "").toLowerCase().includes(t) ||
          (d.skills ?? []).some((s) => s.toLowerCase().includes(t))),
    );
  }, [domains, agent, text]);

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;
  if (q.data?.degraded) return <DegradedNote what="百科领域" />;
  if (!domains.length) return <EmptyNote label="暂无领域数据" />;

  const totalSkills = filtered.reduce((n, d) => n + (d.skills?.length ?? 0), 0);

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="搜索领域 / 技能…"
          className="w-64 rounded border border-mast-border bg-mast-bg px-3 py-1.5 text-sm text-mast-text placeholder:text-mast-muted"
        />
        <label className="flex items-center gap-1 text-xs text-mast-muted">
          归属 agent
          <select
            value={agent}
            onChange={(e) => setAgent(e.target.value)}
            className="rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text"
          >
            <option value="all">全部</option>
            {agents.map((a) => (
              <option key={a} value={a}>{AGENT_LABEL[a] ?? a}</option>
            ))}
          </select>
        </label>
        <span className="text-xs text-mast-muted">
          {filtered.length} 个领域 · {totalSkills} 个技能
        </span>
      </div>

      <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-3">
        {filtered.map((d) => (
          <Card key={d.id} className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <h3 className="text-sm font-semibold text-mast-text">{d.name}</h3>
              {d.agent && <Badge tone="INFO">{AGENT_LABEL[d.agent] ?? d.agent}</Badge>}
            </div>
            {d.desc && <p className="text-xs text-mast-muted">{d.desc}</p>}
            {!!d.skills?.length && (
              <>
                <div className="flex flex-wrap gap-1">
                  {d.skills.map((s) => (
                    <span key={s} className="rounded bg-mast-bg px-1.5 py-0.5 text-xs text-mast-muted">{s}</span>
                  ))}
                </div>
                <p className="text-[11px] text-mast-muted/70">{d.skills.length} 个技能</p>
              </>
            )}
          </Card>
        ))}
      </div>
    </div>
  );
}

function IntentView() {
  const q = useIntentMapping();
  const [text, setText] = useState("");

  const mapping = q.data?.mapping ?? [];

  const filtered = useMemo(() => {
    const t = text.trim().toLowerCase();
    if (!t) return mapping;
    return mapping.filter(
      (m) =>
        m.keywords.toLowerCase().includes(t) ||
        m.skill.toLowerCase().includes(t) ||
        (m.note ?? "").toLowerCase().includes(t),
    );
  }, [mapping, text]);

  if (q.isPending) return <Spinner />;
  if (q.isError) return <ErrorNote error={q.error} />;
  if (q.data?.degraded) return <DegradedNote what="意图映射" />;
  if (!mapping.length) return <EmptyNote label="暂无意图映射数据" />;

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2">
        <input
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="搜索意图关键词 / 技能…"
          className="w-64 rounded border border-mast-border bg-mast-bg px-3 py-1.5 text-sm text-mast-text placeholder:text-mast-muted"
        />
        <span className="text-xs text-mast-muted">共 {filtered.length} 条配方</span>
      </div>

      <Card>
        <p className="mb-3 text-xs text-mast-muted">
          意图（自然语言关键词）→ 推荐技能/配方。这是技能选择的决策映射，"-&gt;"
          表示分级升级或多步组合。
        </p>
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
              {filtered.map((m, i) => {
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
    </div>
  );
}

export function EncyclopediaView() {
  const [view, setView] = useState<EncyView>("domains");
  return (
    <div className="space-y-2">
      <SubTabs
        tabs={[
          { id: "domains", label: "领域分组" },
          { id: "intent", label: "意图映射 / 配方" },
        ]}
        value={view}
        onChange={setView}
      />
      {view === "domains" ? <DomainsView /> : <IntentView />}
    </div>
  );
}
