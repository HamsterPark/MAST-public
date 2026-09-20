import { useEffect, useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { components } from "@/api/schema";
import { Badge, Card, DegradedNote, EmptyNote, ErrorNote, Spinner } from "@/components/ui";
import { Button, Field, SelectField, TextField, useToast } from "@/components/controls";

/** 技能管理 — catalog browser + per-skill metadata editor (full parity with the
 *  old admin/tabs/skills_tab.py 技能编辑 sub-tab): safety_level / estimated
 *  duration / rollback skill / preconditions / postconditions / per-parameter
 *  fields, persisted via POST /api/skills/{name}/override (whole-file
 *  skill_overrides.json relay → core merge + hot-reload). Reset clears the
 *  override (empty payload ⇒ revert to code defaults). NO Dataframe — params are
 *  an HTML table + a per-parameter form (matching the freeze-safe Gradio rebuild).
 *
 *  Catalog: GET /api/skills/catalog (index). Detail: GET /api/skills/{name}
 *  (full card; already merged with any active override at the registry). */

type SkillOverrideRequest = components["schemas"]["SkillOverrideRequest"];

const SAFETY_TONE: Record<string, string> = {
  AUTO: "AUTO", auto: "AUTO",
  CONFIRM: "WARN", confirm: "WARN",
  DANGEROUS: "DANGEROUS", dangerous: "DANGEROUS",
};
const LEVELS = ["auto", "confirm", "dangerous"] as const;

export function SkillsManager() {
  const queryClient = useQueryClient();
  const { toast, node } = useToast();
  const [filter, setFilter] = useState("");
  const [selected, setSelected] = useState<string | null>(null);

  const catQ = useQuery({
    queryKey: ["skills", "catalog"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skills/catalog");
      if (error) throw error;
      return data;
    },
  });

  const ovrQ = useQuery({
    queryKey: ["override", "skill"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/admin/overrides/{category}", {
        params: { path: { category: "skill" } },
      });
      if (error) throw error;
      return data;
    },
  });

  const detailQ = useQuery({
    queryKey: ["skills", "detail", selected],
    enabled: !!selected,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skills/{name}", {
        params: { path: { name: selected! } },
      });
      if (error) throw error;
      return data;
    },
  });

  const index = catQ.data?.index ?? [];
  const filtered = useMemo(() => {
    const f = filter.trim().toLowerCase();
    const rows = f ? index.filter((s) => s.name.toLowerCase().includes(f) || (s.summary ?? "").toLowerCase().includes(f)) : index;
    return [...rows].sort((a, b) => a.name.localeCompare(b.name));
  }, [index, filter]);

  const skillOvr = (ovrQ.data?.data ?? {}) as Record<string, { safety_level?: string }>;

  // Whole-skill metadata override write (params/conditions/duration/rollback/
  // safety_level) via the dedicated relay. Empty payload ⇒ revert to defaults.
  const saveOverride = useMutation({
    mutationFn: async ({ name, body }: { name: string; body: SkillOverrideRequest }) => {
      const { data, error } = await api.POST("/api/skills/{name}/override", {
        params: { path: { name } },
        body,
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["override", "skill"] });
      queryClient.invalidateQueries({ queryKey: ["skills", "detail", selected] });
      if (res?.degraded) toast("写入未生效（内核未接入）。", "err");
      else toast(`已保存覆盖（${(res?.fields ?? []).length} 个字段），重启主服务后对运行中的 agent 生效。`, "ok");
    },
    onError: (e) => toast(`保存失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const resetOverride = useMutation({
    mutationFn: async (name: string) => {
      const { data, error } = await api.POST("/api/skills/{name}/override", {
        params: { path: { name } },
        body: {},
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (res) => {
      queryClient.invalidateQueries({ queryKey: ["override", "skill"] });
      queryClient.invalidateQueries({ queryKey: ["skills", "detail", selected] });
      res?.degraded ? toast("重置未生效（内核未接入）。", "err") : toast("已恢复为代码默认值。", "ok");
    },
    onError: (e) => toast(`重置失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  return (
    <div className="grid grid-cols-1 gap-4 lg:grid-cols-[320px_1fr]">
      {/* ── catalog list ── */}
      <Card>
        <div className="mb-2 flex items-center gap-2">
          <span className="text-sm font-medium text-mast-text">技能目录</span>
          <Badge tone="INFO">{index.length}</Badge>
        </div>
        <TextField value={filter} onChange={setFilter} placeholder="搜索技能…" />
        {catQ.isPending && <Spinner />}
        {catQ.error && <ErrorNote error={catQ.error} />}
        {catQ.data?.degraded && <DegradedNote what="技能目录" />}
        {catQ.data && !catQ.data.degraded && (
          filtered.length === 0 ? <EmptyNote label="无匹配技能" /> : (
            <ul className="mt-2 max-h-[60vh] space-y-0.5 overflow-auto pr-1">
              {filtered.map((s) => {
                const ovrLevel = skillOvr[s.name]?.safety_level;
                const lvl = (ovrLevel ?? s.safety_level ?? "AUTO").toString();
                return (
                  <li key={s.name}>
                    <button
                      onClick={() => setSelected(s.name)}
                      className={
                        "flex w-full items-center justify-between gap-2 rounded-md px-2 py-1.5 text-left text-sm " +
                        (selected === s.name ? "bg-mast-accent/15 text-mast-accent" : "text-mast-text hover:bg-mast-bg/60")
                      }
                    >
                      <span className="truncate font-mono text-xs">{s.name}</span>
                      <Badge tone={SAFETY_TONE[lvl] ?? "INFO"}>{lvl.toUpperCase()}</Badge>
                    </button>
                  </li>
                );
              })}
            </ul>
          )
        )}
      </Card>

      {/* ── detail ── */}
      <Card>
        {!selected && <EmptyNote label="从左侧选择一个技能查看详情。" />}
        {selected && detailQ.isPending && <Spinner />}
        {selected && detailQ.error && <ErrorNote error={detailQ.error} />}
        {selected && detailQ.data?.degraded && <DegradedNote what="技能详情" />}
        {selected && detailQ.data && !detailQ.data.degraded && (
          detailQ.data.found === false ? (
            <EmptyNote label={`未找到技能：${selected}`} />
          ) : (
            <SkillDetail
              card={detailQ.data}
              hasOverride={selected in skillOvr}
              rollbackChoices={["", ...index.map((s) => s.name).sort()]}
              onSave={(body) => saveOverride.mutate({ name: selected, body })}
              onReset={() => resetOverride.mutate(selected)}
              saving={saveOverride.isPending}
              resetting={resetOverride.isPending}
            />
          )
        )}
      </Card>
      {node}
    </div>
  );
}

// ── per-parameter editable row model ──────────────────────────────────────────
type ParamRow = {
  name: string;
  type: string;
  unit: string;
  required: boolean;
  default: string;
  min: string;
  max: string;
  allowed_values: string; // JSON array text
  description: string;
};

function cardParamsToRows(params: any[]): ParamRow[] {
  return (params ?? []).map((p) => ({
    name: String(p.name ?? ""),
    type: String(p.type ?? ""),
    unit: p.unit == null ? "" : String(p.unit),
    required: !!p.required,
    default: p.default == null ? "" : String(p.default),
    min: p.min == null ? "" : String(p.min),
    max: p.max == null ? "" : String(p.max),
    allowed_values: p.allowed_values == null ? "" : JSON.stringify(p.allowed_values),
    description: String(p.description ?? ""),
  }));
}

/** Diff edited rows against the card's current values; emit only changed fields
 *  per param, mapping min/max → min_value/max_value (the override store's keys,
 *  mirroring admin/tabs/skills_data.py:table_to_param_overrides). */
function rowsToParamOverride(
  rows: ParamRow[],
  base: any[],
): { override: Record<string, Record<string, unknown>>; error?: string } {
  const baseByName = new Map<string, any>((base ?? []).map((p) => [String(p.name), p]));
  const out: Record<string, Record<string, unknown>> = {};
  for (const r of rows) {
    const spec = baseByName.get(r.name);
    if (!spec) continue; // can only override existing code params
    const diff: Record<string, unknown> = {};
    if (r.type.trim() && r.type.trim() !== String(spec.type ?? "")) diff.type = r.type.trim();
    if (r.unit.trim() !== String(spec.unit ?? "")) diff.unit = r.unit.trim();
    if (r.required !== !!spec.required) diff.required = r.required;
    const baseDefault = spec.default == null ? "" : String(spec.default);
    if (r.default.trim() !== baseDefault) diff.default = r.default.trim() === "" ? null : r.default.trim();
    const baseMin = spec.min == null ? "" : String(spec.min);
    if (r.min.trim() !== baseMin) diff.min_value = r.min.trim() === "" ? null : Number(r.min);
    const baseMax = spec.max == null ? "" : String(spec.max);
    if (r.max.trim() !== baseMax) diff.max_value = r.max.trim() === "" ? null : Number(r.max);
    const baseAllowed = spec.allowed_values == null ? "" : JSON.stringify(spec.allowed_values);
    if (r.allowed_values.trim() !== baseAllowed) {
      if (r.allowed_values.trim() === "") {
        diff.allowed_values = null;
      } else {
        try {
          const parsed = JSON.parse(r.allowed_values);
          if (!Array.isArray(parsed)) return { override: {}, error: `参数「${r.name}」的 allowed_values 必须是 JSON 数组` };
          diff.allowed_values = parsed;
        } catch {
          return { override: {}, error: `参数「${r.name}」的 allowed_values JSON 解析失败` };
        }
      }
    }
    if (r.description.trim() !== String(spec.description ?? "")) diff.description = r.description.trim();
    if (Object.keys(diff).length) out[r.name] = diff;
  }
  return { override: out };
}

function SkillDetail({
  card, hasOverride, rollbackChoices, onSave, onReset, saving, resetting,
}: {
  card: any;
  hasOverride: boolean;
  rollbackChoices: string[];
  onSave: (body: SkillOverrideRequest) => void;
  onReset: () => void;
  saving: boolean;
  resetting: boolean;
}) {
  const baseParams: any[] = card.parameters ?? [];

  // Editable draft seeded from the (effective) card. Re-seed whenever the card
  // identity/content changes (selection switch or post-save invalidation).
  const seed = useMemo(
    () => ({
      level: (card.safety ?? "auto").toString().toLowerCase(),
      duration: card.estimated_duration_s == null ? "" : String(card.estimated_duration_s),
      rollback: card.rollback_skill ?? "",
      preconds: (card.preconditions ?? []).join("\n"),
      postconds: (card.postconditions ?? []).join("\n"),
      params: cardParamsToRows(baseParams),
    }),
    [card],
  );

  const [level, setLevel] = useState(seed.level);
  const [duration, setDuration] = useState(seed.duration);
  const [rollback, setRollback] = useState(seed.rollback);
  const [preconds, setPreconds] = useState(seed.preconds);
  const [postconds, setPostconds] = useState(seed.postconds);
  const [params, setParams] = useState<ParamRow[]>(seed.params);
  const [selParam, setSelParam] = useState<string>("");
  const [paramErr, setParamErr] = useState<string | null>(null);

  useEffect(() => {
    setLevel(seed.level);
    setDuration(seed.duration);
    setRollback(seed.rollback);
    setPreconds(seed.preconds);
    setPostconds(seed.postconds);
    setParams(seed.params);
    setSelParam("");
    setParamErr(null);
  }, [seed]);

  const cur = params.find((p) => p.name === selParam) ?? null;
  const patchCur = (patch: Partial<ParamRow>) =>
    setParams((rows) => rows.map((p) => (p.name === selParam ? { ...p, ...patch } : p)));

  const submit = () => {
    setParamErr(null);
    const { override: paramOvr, error } = rowsToParamOverride(params, baseParams);
    if (error) { setParamErr(error); return; }
    const body: SkillOverrideRequest = {};
    if (level && level !== (card.safety ?? "auto").toString().toLowerCase()) body.safety_level = level;
    const baseDur = card.estimated_duration_s == null ? "" : String(card.estimated_duration_s);
    if (duration.trim() !== baseDur) body.estimated_duration_s = duration.trim() === "" ? null : Number(duration);
    if ((rollback || "") !== (card.rollback_skill ?? "")) body.rollback_skill = rollback.trim() === "" ? null : rollback;
    const newPre = preconds.split("\n").map((s: string) => s.trim()).filter(Boolean);
    if (JSON.stringify(newPre) !== JSON.stringify(card.preconditions ?? [])) body.preconditions = newPre;
    const newPost = postconds.split("\n").map((s: string) => s.trim()).filter(Boolean);
    if (JSON.stringify(newPost) !== JSON.stringify(card.postconditions ?? [])) body.postconditions = newPost;
    if (Object.keys(paramOvr).length) body.parameters = paramOvr;
    onSave(body);
  };

  const baseLevel = (card.safety ?? "auto").toString();

  return (
    <div className="space-y-4">
      <div>
        <div className="flex flex-wrap items-center gap-2">
          <h3 className="font-mono text-sm font-semibold text-mast-text">{card.name}</h3>
          {card.zh && <span className="text-sm text-mast-muted">{card.zh}</span>}
          {card.category && <Badge tone="INFO">{card.category}</Badge>}
          {card.domain && <Badge>{card.domain}</Badge>}
          {card.version && <span className="text-xs text-mast-muted">v{card.version}</span>}
          {hasOverride ? <Badge tone="WARN">已覆盖</Badge> : <Badge tone="AUTO">代码默认</Badge>}
        </div>
        {(card.description_zh || card.description) && (
          <p className="mt-1 text-sm text-mast-muted">{card.description_zh || card.description}</p>
        )}
      </div>

      {/* top-level governed fields */}
      <div className="grid grid-cols-1 gap-3 sm:grid-cols-3">
        <Field label="安全级别">
          <SelectField value={level} onChange={setLevel} options={LEVELS.map((l) => ({ value: l, label: l.toUpperCase() }))} />
          <span className="mt-1 block text-xs text-mast-muted">
            生效 <Badge tone={SAFETY_TONE[baseLevel] ?? "INFO"}>{baseLevel.toUpperCase()}</Badge>
          </span>
        </Field>
        <Field label="预计时长 (秒)">
          <TextField value={duration} onChange={setDuration} placeholder="—" />
        </Field>
        <Field label="回滚技能">
          <SelectField
            value={rollback}
            onChange={setRollback}
            options={rollbackChoices.map((n) => ({ value: n, label: n === "" ? "（无）" : n }))}
          />
        </Field>
      </div>

      {/* parameters: read-only table + per-parameter form */}
      <div>
        <h4 className="mb-1 text-sm font-medium text-mast-text">参数</h4>
        {params.length === 0 ? (
          <EmptyNote label="无参数" />
        ) : (
          <>
            <div className="overflow-auto rounded-md border border-mast-border">
              <table className="w-full text-xs">
                <thead className="text-mast-muted">
                  <tr className="border-b border-mast-border">
                    <th className="px-2 py-1 text-left">名称</th>
                    <th className="px-2 py-1 text-left">类型</th>
                    <th className="px-2 py-1 text-left">单位</th>
                    <th className="px-2 py-1 text-center">必填</th>
                    <th className="px-2 py-1 text-left">默认</th>
                    <th className="px-2 py-1 text-left">范围</th>
                    <th className="px-2 py-1 text-left">允许值</th>
                  </tr>
                </thead>
                <tbody>
                  {params.map((p, i) => (
                    <tr
                      key={i}
                      onClick={() => setSelParam(p.name)}
                      className={
                        "cursor-pointer border-b border-mast-border/50 " +
                        (selParam === p.name ? "bg-mast-accent/10" : "hover:bg-mast-bg/50")
                      }
                    >
                      <td className="px-2 py-1 font-mono">{p.name}</td>
                      <td className="px-2 py-1">{p.type || "—"}</td>
                      <td className="px-2 py-1">{p.unit || "—"}</td>
                      <td className="px-2 py-1 text-center">{p.required ? "✓" : ""}</td>
                      <td className="px-2 py-1 font-mono">{p.default || "—"}</td>
                      <td className="px-2 py-1 tabular-nums">
                        {p.min || p.max ? `${p.min || "−∞"} ~ ${p.max || "+∞"}` : "—"}
                      </td>
                      <td className="px-2 py-1 font-mono text-mast-muted">{p.allowed_values || "—"}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>

            <div className="mt-2 rounded-md border border-mast-border p-3">
              <div className="mb-2 w-56">
                <SelectField
                  value={selParam}
                  onChange={setSelParam}
                  options={[{ value: "", label: "选择参数编辑…" }, ...params.map((p) => ({ value: p.name, label: p.name }))]}
                />
              </div>
              {cur && (
                <div className="space-y-2">
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                    <Field label="type"><TextField value={cur.type} onChange={(v) => patchCur({ type: v })} /></Field>
                    <Field label="unit"><TextField value={cur.unit} onChange={(v) => patchCur({ unit: v })} /></Field>
                    <Field label="required">
                      <SelectField
                        value={cur.required ? "true" : "false"}
                        onChange={(v) => patchCur({ required: v === "true" })}
                        options={[{ value: "true", label: "True" }, { value: "false", label: "False" }]}
                      />
                    </Field>
                  </div>
                  <div className="grid grid-cols-1 gap-2 sm:grid-cols-3">
                    <Field label="default"><TextField value={cur.default} onChange={(v) => patchCur({ default: v })} /></Field>
                    <Field label="min"><TextField value={cur.min} onChange={(v) => patchCur({ min: v })} /></Field>
                    <Field label="max"><TextField value={cur.max} onChange={(v) => patchCur({ max: v })} /></Field>
                  </div>
                  <Field label="allowed_values (JSON 数组)">
                    <TextField value={cur.allowed_values} mono onChange={(v) => patchCur({ allowed_values: v })} />
                  </Field>
                  <Field label="说明">
                    <TextField value={cur.description} onChange={(v) => patchCur({ description: v })} />
                  </Field>
                </div>
              )}
            </div>
          </>
        )}
      </div>

      {/* preconditions / postconditions */}
      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
        <Field label="前置条件（每行一个）">
          <textarea
            value={preconds}
            rows={4}
            spellCheck={false}
            onChange={(e) => setPreconds(e.target.value)}
            className="w-full resize-y rounded-md border border-mast-border bg-mast-bg px-2.5 py-1.5 font-mono text-xs text-mast-text outline-none focus:border-mast-accent"
          />
        </Field>
        <Field label="后置条件（每行一个）">
          <textarea
            value={postconds}
            rows={4}
            spellCheck={false}
            onChange={(e) => setPostconds(e.target.value)}
            className="w-full resize-y rounded-md border border-mast-border bg-mast-bg px-2.5 py-1.5 font-mono text-xs text-mast-text outline-none focus:border-mast-accent"
          />
        </Field>
      </div>

      {paramErr && <p className="text-xs text-mast-danger">{paramErr}</p>}

      <div className="flex flex-wrap items-center gap-2">
        <Button variant="primary" disabled={saving} onClick={submit}>
          {saving ? "保存中…" : "保存覆盖"}
        </Button>
        <Button variant="danger" disabled={resetting || !hasOverride} onClick={onReset}>
          {resetting ? "恢复中…" : "恢复默认"}
        </Button>
      </div>
      <p className="text-xs text-mast-muted">
        注：保存只持久化与代码默认不同的字段（经 skill_overrides.json 热重载）；参数仅可覆盖已存在的代码参数，不能新增/删除。
      </p>
    </div>
  );
}
