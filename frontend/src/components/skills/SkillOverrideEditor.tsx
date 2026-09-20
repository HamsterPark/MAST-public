import { useEffect, useMemo, useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import {
  Card,
  Badge,
  Spinner,
  ErrorNote,
  DegradedNote,
  EmptyNote,
} from "@/components/ui";
import {
  Field,
  TextField,
  NumberField,
  SelectField,
  Button,
  useToast,
} from "@/components/controls";

// ── 技能详情编辑 — edit skill metadata via the override seam ──────────────────
// Mirrors the old admin/tabs/skills_tab.py right-detail panel control-by-control:
//   安全等级 dropdown · 预计时长 number · 回滚技能 dropdown ·
//   参数表 (read-only) + 逐参数编辑表单 (type/unit/required/default/min/max/
//   allowed_values) · 前置条件 textarea · 后置条件 textarea ·
//   保存覆盖 / 恢复默认 buttons + 变更预览(diff-ish).
// Writes whole-file via POST /api/admin/overrides/skill (category="skill"),
// a dict keyed by skill name — read current file via GET first so we merge,
// not clobber.

const SAFETY_OPTIONS: { value: string; label: string }[] = [
  { value: "AUTO", label: "AUTO 自动" },
  { value: "INFO", label: "INFO 信息" },
  { value: "WARN", label: "WARN 警告" },
  { value: "DANGEROUS", label: "DANGEROUS 危险" },
];

const REQUIRED_OPTIONS: { value: string; label: string }[] = [
  { value: "", label: "—（不变）" },
  { value: "True", label: "True 必填" },
  { value: "False", label: "False 可选" },
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

type ParamRow = {
  name: string;
  type: string;
  unit: string;
  required: string; // "", "True", "False"
  default: string;
  min: string;
  max: string;
  allowed_values: string; // JSON array text
};

function useOverrideFile() {
  return useQuery({
    queryKey: ["admin", "overrides", "skill"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/admin/overrides/{category}", {
        params: { path: { category: "skill" } },
      });
      if (error) throw error;
      return data;
    },
  });
}

function useSkillCard(name: string | null) {
  return useQuery({
    enabled: !!name,
    queryKey: ["skill", name],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skills/{name}", {
        params: { path: { name: name! } },
      });
      if (error) throw error;
      return data;
    },
  });
}

function toRow(p: {
  name: string;
  type?: string;
  unit?: string | null;
  required?: boolean;
  default?: unknown;
  min?: number | null;
  max?: number | null;
  allowed_values?: unknown[] | null;
}): ParamRow {
  return {
    name: p.name,
    type: p.type ?? "",
    unit: p.unit == null ? "" : String(p.unit),
    required: p.required ? "True" : "False",
    default: p.default == null ? "" : String(p.default),
    min: p.min == null ? "" : String(p.min),
    max: p.max == null ? "" : String(p.max),
    allowed_values: p.allowed_values?.length
      ? JSON.stringify(p.allowed_values)
      : "",
  };
}

export function SkillOverrideEditor({
  names,
  loadingNames,
}: {
  names: string[];
  loadingNames?: boolean;
}) {
  const qc = useQueryClient();
  const { toast, node: toastNode } = useToast();

  const [selected, setSelected] = useState<string | null>(null);
  const [search, setSearch] = useState("");

  const card = useSkillCard(selected);
  const file = useOverrideFile();

  // editable state
  const [safety, setSafety] = useState("AUTO");
  const [duration, setDuration] = useState("");
  const [rollback, setRollback] = useState("None");
  const [preconds, setPreconds] = useState("");
  const [postconds, setPostconds] = useState("");
  const [rows, setRows] = useState<ParamRow[]>([]);

  // per-parameter edit form
  const [pSel, setPSel] = useState<string>("");
  const [pType, setPType] = useState("");
  const [pUnit, setPUnit] = useState("");
  const [pRequired, setPRequired] = useState("");
  const [pDefault, setPDefault] = useState("");
  const [pMin, setPMin] = useState("");
  const [pMax, setPMax] = useState("");
  const [pAllowed, setPAllowed] = useState("");

  // hydrate editable state from the fetched card whenever it loads
  const cardData = card.data;
  useEffect(() => {
    if (!cardData?.found) return;
    setSafety(cardData.safety || "AUTO");
    setDuration(
      cardData.estimated_duration_s == null
        ? ""
        : String(cardData.estimated_duration_s),
    );
    setRollback(cardData.rollback_skill || "None");
    setPreconds((cardData.preconditions ?? []).join("\n"));
    setPostconds((cardData.postconditions ?? []).join("\n"));
    setRows((cardData.parameters ?? []).map(toRow));
    setPSel("");
  }, [cardData]);

  const filteredNames = useMemo(() => {
    const t = search.trim().toLowerCase();
    if (!t) return names;
    return names.filter((n) => n.toLowerCase().includes(t));
  }, [names, search]);

  const hasOverride =
    selected != null &&
    !!(file.data?.data as Record<string, unknown> | undefined)?.[selected];

  // ── per-parameter form fill on select ─────────────────────────────────────
  function selectParam(pname: string) {
    setPSel(pname);
    const r = rows.find((x) => x.name === pname);
    if (!r) {
      setPType("");
      setPUnit("");
      setPRequired("");
      setPDefault("");
      setPMin("");
      setPMax("");
      setPAllowed("");
      return;
    }
    setPType(r.type);
    setPUnit(r.unit);
    setPRequired(REQUIRED_OPTIONS.some((o) => o.value === r.required) ? r.required : "");
    setPDefault(r.default);
    setPMin(r.min);
    setPMax(r.max);
    setPAllowed(r.allowed_values);
  }

  function applyParam() {
    if (!pSel) return;
    setRows((prev) =>
      prev.map((r) =>
        r.name === pSel
          ? {
              ...r,
              type: pType.trim(),
              unit: pUnit.trim(),
              required: pRequired.trim(),
              default: pDefault.trim(),
              min: pMin.trim(),
              max: pMax.trim(),
              allowed_values: pAllowed.trim(),
            }
          : r,
      ),
    );
    toast(`已写入参数 ${pSel}（未保存）`, "ok");
  }

  // ── build override dict (only differences vs code card) ────────────────────
  function buildOverride(): Record<string, unknown> | { __error: string } {
    if (!cardData?.found) return { __error: "技能未加载" };
    const ovr: Record<string, unknown> = {};

    if (safety && safety !== cardData.safety) ovr.safety_level = safety.toLowerCase();

    const durNum = duration.trim() === "" ? null : Number(duration);
    if (durNum != null && !Number.isNaN(durNum) && durNum !== cardData.estimated_duration_s) {
      ovr.estimated_duration_s = durNum;
    }

    const rb = rollback === "None" ? null : rollback;
    if (rb !== (cardData.rollback_skill ?? null)) ovr.rollback_skill = rb;

    const preList = preconds.split("\n").map((s) => s.trim()).filter(Boolean);
    if (JSON.stringify(preList) !== JSON.stringify(cardData.preconditions ?? [])) {
      ovr.preconditions = preList;
    }
    const postList = postconds.split("\n").map((s) => s.trim()).filter(Boolean);
    if (JSON.stringify(postList) !== JSON.stringify(cardData.postconditions ?? [])) {
      ovr.postconditions = postList;
    }

    // parameter overrides — only cells that differ from the code card
    const codeRows = new Map((cardData.parameters ?? []).map((p) => [p.name, toRow(p)]));
    const paramOvr: Record<string, Record<string, unknown>> = {};
    for (const r of rows) {
      const code = codeRows.get(r.name);
      if (!code) continue;
      const diff: Record<string, unknown> = {};
      if (r.type !== code.type) diff.type = r.type;
      if (r.unit !== code.unit) diff.unit = r.unit || null;
      if (r.required !== code.required && r.required !== "")
        diff.required = r.required === "True";
      if (r.default !== code.default) diff.default = r.default === "" ? null : r.default;
      if (r.min !== code.min) diff.min = r.min === "" ? null : Number(r.min);
      if (r.max !== code.max) diff.max = r.max === "" ? null : Number(r.max);
      if (r.allowed_values !== code.allowed_values) {
        if (r.allowed_values.trim() === "") {
          diff.allowed_values = null;
        } else {
          try {
            diff.allowed_values = JSON.parse(r.allowed_values);
          } catch {
            return { __error: `参数 ${r.name} 的 allowed_values 不是合法 JSON 数组` };
          }
        }
      }
      if (Object.keys(diff).length) paramOvr[r.name] = diff;
    }
    if (Object.keys(paramOvr).length) ovr.parameters = paramOvr;

    return ovr;
  }

  const preview = useMemo(() => {
    if (!cardData?.found) return null;
    const o = buildOverride();
    if ("__error" in o) return { error: o.__error as string, ovr: null };
    return { error: null, ovr: o };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [cardData, safety, duration, rollback, preconds, postconds, rows]);

  // ── save / reset mutations ─────────────────────────────────────────────────
  const saveMut = useMutation({
    mutationFn: async () => {
      const o = buildOverride();
      if ("__error" in o) throw new Error(o.__error as string);
      if (!Object.keys(o).length) throw new Error("无变更，未保存");
      const existing = { ...((file.data?.data as Record<string, unknown>) ?? {}) };
      existing[selected!] = o;
      const { data, error } = await api.POST("/api/admin/overrides/{category}", {
        params: { path: { category: "skill" } },
        body: { data: existing },
      });
      if (error) throw error;
      if (!data?.ok) throw new Error("保存失败（后端未生效）");
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["admin", "overrides", "skill"] });
      qc.invalidateQueries({ queryKey: ["skill", selected] });
      qc.invalidateQueries({ queryKey: ["skills", "catalog"] });
      toast("已保存覆盖，重启主服务后对运行中的 agent 生效", "ok");
    },
    onError: (e) => toast(e instanceof Error ? e.message : "保存失败", "err"),
  });

  const resetMut = useMutation({
    mutationFn: async () => {
      const existing = { ...((file.data?.data as Record<string, unknown>) ?? {}) };
      if (!(selected! in existing)) throw new Error(`${selected} 无覆盖配置`);
      delete existing[selected!];
      const { data, error } = await api.POST("/api/admin/overrides/{category}", {
        params: { path: { category: "skill" } },
        body: { data: existing },
      });
      if (error) throw error;
      if (!data?.ok) throw new Error("恢复失败（后端未生效）");
      return data;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["admin", "overrides", "skill"] });
      qc.invalidateQueries({ queryKey: ["skill", selected] });
      qc.invalidateQueries({ queryKey: ["skills", "catalog"] });
      toast("已恢复为代码默认值", "ok");
    },
    onError: (e) => toast(e instanceof Error ? e.message : "恢复失败", "err"),
  });

  return (
    <div className="grid gap-4 lg:grid-cols-[280px_1fr]">
      {toastNode}

      {/* ── LEFT: skill list ── */}
      <Card className="space-y-2 self-start">
        <h3 className="text-sm font-semibold text-mast-text">技能列表</h3>
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="搜索技能名…"
          className="w-full rounded border border-mast-border bg-mast-bg px-2.5 py-1.5 text-sm text-mast-text placeholder:text-mast-muted"
        />
        {loadingNames ? (
          <Spinner />
        ) : (
          <div className="max-h-[520px] overflow-auto rounded border border-mast-border">
            {filteredNames.length === 0 ? (
              <EmptyNote label="无匹配技能" />
            ) : (
              <ul className="divide-y divide-mast-border text-sm">
                {filteredNames.map((n) => {
                  const ov = !!(file.data?.data as Record<string, unknown> | undefined)?.[n];
                  return (
                    <li key={n}>
                      <button
                        onClick={() => setSelected(n)}
                        className={
                          "flex w-full items-center justify-between px-2.5 py-1.5 text-left hover:bg-mast-bg/60 " +
                          (selected === n ? "bg-mast-accent/10 text-mast-accent" : "text-mast-text")
                        }
                      >
                        <span className="truncate">{n}</span>
                        {ov && <span className="ml-2 h-1.5 w-1.5 shrink-0 rounded-full bg-mast-warn" title="已覆盖" />}
                      </button>
                    </li>
                  );
                })}
              </ul>
            )}
          </div>
        )}
        <p className="text-xs text-mast-muted">共 {filteredNames.length} / {names.length}</p>
      </Card>

      {/* ── RIGHT: detail editor ── */}
      <div className="space-y-4">
        {!selected ? (
          <Card><EmptyNote label="从左侧列表选择一个技能以编辑其覆盖配置" /></Card>
        ) : card.isPending ? (
          <Card><Spinner /></Card>
        ) : card.isError ? (
          <Card><ErrorNote error={card.error} /></Card>
        ) : card.data?.degraded ? (
          <DegradedNote what="技能详情" />
        ) : !card.data?.found ? (
          <Card><EmptyNote label={`未找到技能：${selected}`} /></Card>
        ) : (
          <>
            <Card className="space-y-4">
              <div className="flex flex-wrap items-center gap-2">
                <h3 className="text-base font-semibold text-mast-text">{card.data.name}</h3>
                {card.data.zh && <span className="text-sm text-mast-muted">{card.data.zh}</span>}
                <Badge tone={SAFETY_TONE[card.data.safety] ?? "default"}>{card.data.safety || "—"}</Badge>
                <Badge tone="INFO">L{card.data.level}</Badge>
                {hasOverride && <Badge tone="WARN">已覆盖</Badge>}
              </div>
              {(card.data.description_zh || card.data.description) && (
                <p className="text-sm text-mast-text/90">{card.data.description_zh || card.data.description}</p>
              )}

              {/* safety / duration / rollback */}
              <div className="grid gap-3 sm:grid-cols-3">
                <Field label="安全等级">
                  <SelectField value={safety} onChange={setSafety} options={SAFETY_OPTIONS} />
                </Field>
                <Field label="预计时长 (秒)">
                  <NumberField value={duration} onChange={setDuration} step="0.1" />
                </Field>
                <Field label="回滚技能">
                  <SelectField
                    value={rollback}
                    onChange={setRollback}
                    options={[
                      { value: "None", label: "None（无）" },
                      ...names.map((n) => ({ value: n, label: n })),
                    ]}
                  />
                </Field>
              </div>
            </Card>

            {/* parameters table + per-param edit form */}
            <Card className="space-y-3">
              <h4 className="text-sm font-medium text-mast-text">参数</h4>
              {rows.length === 0 ? (
                <EmptyNote label="该技能无参数" />
              ) : (
                <div className="overflow-auto rounded border border-mast-border">
                  <table className="w-full text-xs">
                    <thead className="bg-mast-bg text-mast-muted">
                      <tr>
                        {["名称", "类型", "单位", "必填", "默认", "min", "max", "允许值"].map((h) => (
                          <th key={h} className="px-2 py-1.5 text-left">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {rows.map((r) => {
                        const code = (card.data!.parameters ?? []).find((p) => p.name === r.name);
                        const changed = code ? JSON.stringify(toRow(code)) !== JSON.stringify(r) : false;
                        return (
                          <tr
                            key={r.name}
                            onClick={() => selectParam(r.name)}
                            className={
                              "cursor-pointer border-t border-mast-border hover:bg-mast-bg/60 " +
                              (pSel === r.name ? "bg-mast-accent/10" : "")
                            }
                          >
                            <td className="px-2 py-1.5 font-mono">
                              {r.name}
                              {changed && <span className="ml-1 text-mast-warn">●</span>}
                            </td>
                            <td className="px-2 py-1.5 font-mono text-mast-muted">{r.type || "—"}</td>
                            <td className="px-2 py-1.5 font-mono text-mast-muted">{r.unit || "—"}</td>
                            <td className="px-2 py-1.5 font-mono text-mast-muted">{r.required || "—"}</td>
                            <td className="px-2 py-1.5 font-mono text-mast-muted">{r.default || "—"}</td>
                            <td className="px-2 py-1.5 font-mono text-mast-muted tabular-nums">{r.min || "—"}</td>
                            <td className="px-2 py-1.5 font-mono text-mast-muted tabular-nums">{r.max || "—"}</td>
                            <td className="px-2 py-1.5 font-mono text-mast-muted">{r.allowed_values || "—"}</td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>
              )}

              {pSel && (
                <div className="rounded border border-mast-border bg-mast-bg/40 p-3">
                  <p className="mb-2 text-xs text-mast-muted">编辑参数：<span className="font-mono text-mast-text">{pSel}</span></p>
                  <div className="grid gap-3 sm:grid-cols-3">
                    <Field label="type"><TextField value={pType} onChange={setPType} mono /></Field>
                    <Field label="unit"><TextField value={pUnit} onChange={setPUnit} mono /></Field>
                    <Field label="required">
                      <SelectField value={pRequired} onChange={setPRequired} options={REQUIRED_OPTIONS} />
                    </Field>
                    <Field label="default"><TextField value={pDefault} onChange={setPDefault} mono /></Field>
                    <Field label="min"><NumberField value={pMin} onChange={setPMin} /></Field>
                    <Field label="max"><NumberField value={pMax} onChange={setPMax} /></Field>
                  </div>
                  <div className="mt-3">
                    <Field label="allowed_values (JSON 数组，例如 [1,2,3] 或 [&quot;a&quot;,&quot;b&quot;])">
                      <TextField value={pAllowed} onChange={setPAllowed} mono />
                    </Field>
                  </div>
                  <div className="mt-3">
                    <Button variant="default" onClick={applyParam}>应用到该参数</Button>
                  </div>
                </div>
              )}
            </Card>

            {/* preconditions / postconditions */}
            <Card className="grid gap-4 sm:grid-cols-2">
              <Field label="前置条件（每行一个）">
                <textarea
                  value={preconds}
                  onChange={(e) => setPreconds(e.target.value)}
                  rows={4}
                  placeholder={"z_controller_on\nbias_v != 0"}
                  className="rounded-md border border-mast-border bg-mast-bg px-2.5 py-1.5 font-mono text-xs text-mast-text"
                />
              </Field>
              <Field label="后置条件（每行一个）">
                <textarea
                  value={postconds}
                  onChange={(e) => setPostconds(e.target.value)}
                  rows={4}
                  placeholder={"scan_running"}
                  className="rounded-md border border-mast-border bg-mast-bg px-2.5 py-1.5 font-mono text-xs text-mast-text"
                />
              </Field>
            </Card>

            {/* change preview */}
            <Card className="space-y-2">
              <h4 className="text-sm font-medium text-mast-text">变更预览</h4>
              {preview?.error ? (
                <p className="text-xs text-mast-danger">{preview.error}</p>
              ) : preview?.ovr && Object.keys(preview.ovr).length ? (
                <pre className="max-h-64 overflow-auto rounded border border-mast-border bg-mast-bg p-2 text-xs text-mast-muted">
                  {JSON.stringify(preview.ovr, null, 2)}
                </pre>
              ) : (
                <p className="text-xs text-mast-muted">无变更</p>
              )}
              <div className="flex gap-2 pt-1">
                <Button
                  variant="primary"
                  disabled={saveMut.isPending}
                  onClick={() => saveMut.mutate()}
                >
                  {saveMut.isPending ? "保存中…" : "保存覆盖"}
                </Button>
                <Button
                  variant="danger"
                  disabled={resetMut.isPending || !hasOverride}
                  onClick={() => resetMut.mutate()}
                >
                  {resetMut.isPending ? "恢复中…" : "恢复默认"}
                </Button>
              </div>
              {file.data?.degraded && (
                <p className="text-xs text-mast-warn">覆盖存储未接入（degraded）— 保存可能不生效。</p>
              )}
            </Card>
          </>
        )}
      </div>
    </div>
  );
}
