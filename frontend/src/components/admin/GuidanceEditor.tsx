import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Badge, Card, DegradedNote, EmptyNote, ErrorNote, Spinner } from "@/components/ui";
import { Button, Field, SubTabs, TextField, useToast } from "@/components/controls";
import { JsonSectionEditor } from "@/components/admin/JsonSectionEditor";

/** 技能指导 — per-skill expert annotations (SKILL_EXTRA), mirroring the old
 *  guidance/annotations.py sub-tab: when_use / when_not / related / nanonis /
 *  example. Read+write via GET/POST /api/admin/guidance/{skill}. The skill list
 *  comes from /api/skills/catalog. Empty payload ⇒ reset to code default.
 *
 *  Plus the three whole-section guidance kinds the old guidance sub-tabs edited —
 *  decision_trees / recipes / templates — via GET/POST
 *  /api/admin/guidance-extra/{kind} (effective default⊕override + override JSON
 *  editor, reusing JsonSectionEditor). A top mode-switch picks between the
 *  per-skill annotations and the section editors. */

type GuidanceFields = {
  when_use: string;
  when_not: string;
  related: string;
  nanonis: string;
  example: string;
};

type GuidanceMode = "annotations" | "extra";

const GUIDANCE_MODES: { id: GuidanceMode; label: string }[] = [
  { id: "annotations", label: "逐技能标注" },
  { id: "extra", label: "决策树 / 配方 / 模板" },
];

const EXTRA_KINDS: { id: string; label: string; desc: string }[] = [
  { id: "decision_trees", label: "决策树 (decision_trees)", desc: "分诊决策树（DECISION_TREES）——按 kind 整段覆盖；留空=恢复代码默认。" },
  { id: "recipes", label: "配方 (recipes)", desc: "实验配方列表（RECIPES）——按 kind 整段覆盖；留空=恢复代码默认。" },
  { id: "templates", label: "模板 (templates)", desc: "指导模板（TEMPLATES）——按 kind 整段覆盖；留空=恢复代码默认。" },
];

function TextArea({ value, onChange, rows = 3, mono }: { value: string; onChange: (v: string) => void; rows?: number; mono?: boolean }) {
  return (
    <textarea
      value={value}
      rows={rows}
      spellCheck={false}
      onChange={(e) => onChange(e.target.value)}
      className={
        "w-full resize-y rounded-md border border-mast-border bg-mast-bg px-2.5 py-1.5 text-sm text-mast-text outline-none focus:border-mast-accent " +
        (mono ? "font-mono text-xs" : "")
      }
    />
  );
}

export function GuidanceEditor() {
  const [mode, setMode] = useState<GuidanceMode>("annotations");
  return (
    <div className="space-y-4">
      <SubTabs tabs={GUIDANCE_MODES} value={mode} onChange={setMode} />
      {mode === "annotations" && <AnnotationsEditor />}
      {mode === "extra" && <GuidanceExtraEditor />}
    </div>
  );
}

// ── 决策树 / 配方 / 模板 — whole-section override editors ─────────────────────
function GuidanceExtraEditor() {
  const [kind, setKind] = useState(EXTRA_KINDS[0]!.id);
  const cur = EXTRA_KINDS.find((k) => k.id === kind)!;
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-2">
        {EXTRA_KINDS.map((k) => (
          <button
            key={k.id}
            onClick={() => setKind(k.id)}
            className={
              "rounded-md border px-3 py-1.5 text-sm " +
              (kind === k.id
                ? "border-mast-accent bg-mast-accent/15 text-mast-accent"
                : "border-mast-border text-mast-muted hover:text-mast-text")
            }
          >
            {k.label}
          </button>
        ))}
      </div>
      <JsonSectionEditor
        key={kind}
        title={`技能指导 · ${cur.label}`}
        description={cur.desc}
        getPath="/api/admin/guidance-extra/{kind}"
        postPath="/api/admin/guidance-extra/{kind}"
        pathParam={{ kind }}
        queryKey={["guidance-extra", kind]}
      />
    </div>
  );
}

// ── 逐技能专家标注 (SKILL_EXTRA) ─────────────────────────────────────────────
function AnnotationsEditor() {
  const queryClient = useQueryClient();
  const { toast, node } = useToast();
  const [filter, setFilter] = useState("");
  const [skill, setSkill] = useState<string | null>(null);

  const catQ = useQuery({
    queryKey: ["skills", "catalog"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skills/catalog");
      if (error) throw error;
      return data;
    },
  });

  const guidQ = useQuery({
    queryKey: ["guidance", skill],
    enabled: !!skill,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/admin/guidance/{skill}", {
        params: { path: { skill: skill! } },
      });
      if (error) throw error;
      return data;
    },
  });

  const names = useMemo(() => {
    const f = filter.trim().toLowerCase();
    const all = (catQ.data?.index ?? []).map((s) => s.name).sort();
    return f ? all.filter((n) => n.toLowerCase().includes(f)) : all;
  }, [catQ.data, filter]);

  const effective = (guidQ.data?.data ?? {}) as Partial<GuidanceFields>;
  const [draft, setDraft] = useState<GuidanceFields | null>(null);
  useEffect(() => { setDraft(null); }, [guidQ.data, skill]);

  const view: GuidanceFields = draft ?? {
    when_use: effective.when_use ?? "",
    when_not: effective.when_not ?? "",
    related: effective.related ?? "",
    nanonis: effective.nanonis ?? "",
    example: effective.example ?? "",
  };

  const save = useMutation({
    mutationFn: async (payload: GuidanceFields | Record<string, never>) => {
      const { data, error } = await api.POST("/api/admin/guidance/{skill}", {
        params: { path: { skill: skill! } },
        body: { data: payload } as never,
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      setDraft(null);
      queryClient.invalidateQueries({ queryKey: ["guidance", skill] });
      res?.degraded ? toast("写入未生效（内核未接入）。", "err") : toast("已保存技能指导。", "ok");
    },
    onError: (e) => toast(`保存失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[280px_1fr]">
      <Card>
        <div className="mb-2 text-sm font-medium text-mast-text">选择技能</div>
        <TextField value={filter} onChange={setFilter} placeholder="搜索技能…" />
        {catQ.isPending && <Spinner />}
        {catQ.error && <ErrorNote error={catQ.error} />}
        {catQ.data?.degraded && <DegradedNote what="技能目录" />}
        {catQ.data && !catQ.data.degraded && (
          names.length === 0 ? <EmptyNote label="无匹配技能" /> : (
            <ul className="mt-2 max-h-[60vh] space-y-0.5 overflow-auto pr-1">
              {names.map((n) => (
                <li key={n}>
                  <button
                    onClick={() => setSkill(n)}
                    className={
                      "w-full truncate rounded-md px-2 py-1.5 text-left font-mono text-xs " +
                      (skill === n ? "bg-mast-accent/15 text-mast-accent" : "text-mast-text hover:bg-mast-bg/60")
                    }
                  >
                    {n}
                  </button>
                </li>
              ))}
            </ul>
          )
        )}
      </Card>

      <Card>
        {!skill && <EmptyNote label="从左侧选择一个技能编辑其指导信息。" />}
        {skill && guidQ.isPending && <Spinner />}
        {skill && guidQ.error && <ErrorNote error={guidQ.error} />}
        {skill && guidQ.data?.degraded && <DegradedNote what="技能指导" />}
        {skill && guidQ.data && !guidQ.data.degraded && (
          <div className="space-y-3">
            <div className="flex items-center gap-2">
              <h3 className="font-mono text-sm font-semibold text-mast-text">{skill}</h3>
              {guidQ.data.has_override ? <Badge tone="INFO">已覆盖</Badge> : <Badge tone="AUTO">默认</Badge>}
            </div>
            <Field label="何时使用 (when_use)">
              <TextArea value={view.when_use} onChange={(v) => setDraft({ ...view, when_use: v })} />
            </Field>
            <Field label="何时不使用 (when_not)">
              <TextArea value={view.when_not} onChange={(v) => setDraft({ ...view, when_not: v })} />
            </Field>
            <Field label="关联技能 (related)">
              <TextField value={view.related} onChange={(v) => setDraft({ ...view, related: v })} />
            </Field>
            <Field label="底层 Nanonis TCP 方法 (nanonis)">
              <TextField value={view.nanonis} mono onChange={(v) => setDraft({ ...view, nanonis: v })} />
            </Field>
            <Field label="示例 JSON 调用 (example)">
              <TextArea value={view.example} rows={5} mono onChange={(v) => setDraft({ ...view, example: v })} />
            </Field>
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="primary" disabled={save.isPending} onClick={() => save.mutate(view)}>
                {save.isPending ? "保存中…" : "保存"}
              </Button>
              {draft && <Button variant="ghost" onClick={() => setDraft(null)}>撤销编辑</Button>}
              <Button variant="danger" disabled={save.isPending} onClick={() => save.mutate({})}>
                重置为默认值
              </Button>
            </div>
          </div>
        )}
      </Card>
      {node}
    </div>
  );
}
