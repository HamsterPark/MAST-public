import { useMemo, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import type { ColumnDef } from "@tanstack/react-table";
import { api } from "@/api/client";
import type { components } from "@/api/schema";
import { useUiStore } from "@/store";
import {
  Badge,
  Card,
  DegradedNote,
  ErrorNote,
  Section,
  Spinner,
} from "@/components/ui";
import { DataTable } from "@/components/DataTable";
import { CapabilityGates } from "@/components/admin/CapabilityGates";
import {
  useToast, RadioGroup, SubTabs } from "@/components/controls";
import { PinGate } from "@/components/admin/PinGate";
import { ChecksEditor, ConstraintsEditor, SkillLevelsEditor } from "@/components/admin/SafetySections";
import { SkillsManager } from "@/components/admin/SkillsManager";
import { GuidanceEditor } from "@/components/admin/GuidanceEditor";
import { QuickPromptsEditor } from "@/components/admin/QuickPromptsEditor";
import { HardwareManager } from "@/components/admin/HardwareManager";
import { OverrideHistory } from "@/components/admin/OverrideHistory";
import { JsonSectionEditor } from "@/components/admin/JsonSectionEditor";
import { ReferenceTables } from "@/components/admin/ReferenceTables";
import { LightAgentPanel } from "@/components/admin/LightAgentPanel";
import { PromptInspector } from "@/components/admin/PromptInspector";
import { useStickyTab } from "@/hooks/useStickyTab";

// Domain E — 高级管理 (round-2 rebuild → PER-AGENT, matching the OLD Gradio
// admin_panel.py). The user: "从前高级管理是分agent的，现在没了，大量东西丢了"。
// Top-level is a per-agent selector (RadioGroup), 6 agents in scientific order
// (encyclopedia.AGENT_ORDER minus _supervisor/buffer_summarizer) + a final 系统/全局
// entry:  文献 LIT / 实验设计 XD / 仪器控制 IC / 数据处理 DP / 论文写作 PW /
//         论文审稿 PR / 系统 / 全局
//   • IC = the FULL governance (the former function-tabs, moved UNDER IC):
//       安全系统 / 技能管理 / 知识库 / 技能指导 / 百科配置 / 参考规则
//   • LIT/XD/DP/PW/PR = a LIGHT panel (model+thinking display + read-only AUTO
//     tool codex) — see LightAgentPanel.
//   • 系统/全局 = the surfaces that were NEVER per-agent (hardware was in 设置,
//     system-check / overrides / quick-prompts are MAST additions).
// PIN gate up front (shared zustand flag). Everything is forms + tables; NO
// Dataframe anywhere. Every read renders loading / error / degraded / empty.
// Nothing was deleted — only reorganized PER-AGENT.

type SafetyLimits = components["schemas"]["SafetyLimits"];
type SystemCheckItem = components["schemas"]["SystemCheckItem"];

// ── Safety limits (read + edit as a FORM, never a Dataframe) ──────────────────
// Mirrors the OLD admin/tabs/safety/limits.py 5-column "Lab Console" grid: 4
// paired rows (偏压/Z/XY/电流设定值), each split into 最小/最大 (8 fields).
// scan_size_* is intentionally NOT shown (OLD _LIMIT_ROWS omits it).
const LIMIT_ROWS: {
  label: string;
  unit: string;
  minKey: keyof SafetyLimits;
  maxKey: keyof SafetyLimits;
}[] = [
  { label: "偏压 (Bias)", unit: "V", minKey: "bias_min_v", maxKey: "bias_max_v" },
  { label: "Z 位置", unit: "m", minKey: "z_min_m", maxKey: "z_max_m" },
  { label: "XY 位置", unit: "m", minKey: "xy_min_m", maxKey: "xy_max_m" },
  { label: "电流设定值 (Setpoint)", unit: "A", minKey: "setpoint_min_a", maxKey: "setpoint_max_a" },
];
const LIMIT_FIELDS: { key: keyof SafetyLimits; label: string; unit: string }[] =
  LIMIT_ROWS.flatMap((r) => [
    { key: r.minKey, label: `${r.label} · 最小`, unit: r.unit },
    { key: r.maxKey, label: `${r.label} · 最大`, unit: r.unit },
  ]);

// SI prefix formatter — verbatim port of admin/parsers.py:fmt_sci (DEFAULT cell).
function fmtSci(value: number, unit: string): string {
  if (!Number.isFinite(value)) return `${value} ${unit}`;
  const absVal = Math.abs(value);
  if (absVal === 0) return `0 ${unit}`;
  const prefixes: [number, string][] = [
    [1e-15, "f"], [1e-12, "p"], [1e-9, "n"], [1e-6, "μ"],
    [1e-3, "m"], [1, ""], [1e3, "k"], [1e6, "M"],
  ];
  for (const [scale, prefix] of prefixes) {
    if (absVal < scale * 1000) {
      const scaled = value / scale;
      if (scaled === Math.trunc(scaled)) return `${Math.trunc(scaled)} ${prefix}${unit}`;
      return `${scaled} ${prefix}${unit}`;
    }
  }
  return `${value} ${unit}`;
}

// SafetyLimits code defaults (verbatim from the schema @default annotations) —
// the OLD grid's DEFAULT column shows these in SI-prefixed form.
const LIMIT_DEFAULTS: Record<string, number> = {
  bias_min_v: -10, bias_max_v: 10,
  z_min_m: 0, z_max_m: 1.5e-6,
  xy_min_m: -1.5e-6, xy_max_m: 1.5e-6,
  setpoint_min_a: 1e-12, setpoint_max_a: 1e-7,
};

function SafetyLimitsSection() {
  const queryClient = useQueryClient();
  const { data, isPending, error } = useQuery({
    queryKey: ["safety", "limits"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/safety/limits");
      if (error) throw error;
      return data;
    },
  });
  // Override layer — drives the per-field STATUS pill (override / default).
  const ovrQ = useQuery({
    queryKey: ["override", "safety_limits"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/admin/overrides/{category}", {
        params: { path: { category: "safety_limits" } },
      });
      if (error) throw error;
      return data;
    },
  });
  const overrideMap = (ovrQ.data?.data ?? {}) as Record<string, unknown>;

  const [draft, setDraft] = useState<Record<string, string> | null>(null);
  const view = useMemo<Record<string, string> | null>(() => {
    if (!data) return null;
    if (draft) return draft;
    const m: Record<string, string> = {};
    for (const f of LIMIT_FIELDS) m[f.key as string] = String(data[f.key]);
    return m;
  }, [data, draft]);

  const save = useMutation({
    mutationFn: async (payload: Record<string, number>) => {
      const { data, error } = await api.POST("/api/admin/overrides/{category}", {
        params: { path: { category: "safety_limits" } },
        body: { data: payload },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: () => {
      setDraft(null);
      queryClient.invalidateQueries({ queryKey: ["safety", "limits"] });
      queryClient.invalidateQueries({ queryKey: ["override", "safety_limits"] });
    },
  });

  const isOverridden = (key: string): boolean =>
    overrideMap[key] != null && overrideMap[key] !== "";

  return (
    <Section title="全局安全限制">
      {isPending && <Spinner />}
      {error && <ErrorNote error={error} />}
      {data && view && (
        <Card>
          <p className="mb-3 text-xs text-mast-muted">
            硬件参数的最小/最大边界。空值 = 使用代码默认。保存后立即生效，主 GUI 的 SafetyGuard 随下次操作拾取新值。
          </p>
          <form
            onSubmit={(e) => {
              e.preventDefault();
              const payload: Record<string, number> = {};
              for (const f of LIMIT_FIELDS) {
                const raw = (view[f.key as string] ?? "").trim();
                if (raw === "") continue;
                const n = Number(raw);
                if (!Number.isNaN(n)) payload[f.key as string] = n;
              }
              save.mutate(payload);
            }}
          >
            {/* 5-column grid header — LIMIT / OVERRIDE / UNIT / DEFAULT / STATUS */}
            <div className="grid grid-cols-[1.6fr_1.2fr_0.5fr_1fr_0.8fr] items-center gap-2 border-b border-mast-border px-1 pb-1 text-[0.7rem] font-medium uppercase tracking-wide text-mast-muted">
              <div>LIMIT</div>
              <div>OVERRIDE</div>
              <div>UNIT</div>
              <div>DEFAULT</div>
              <div>STATUS</div>
            </div>
            {LIMIT_FIELDS.map((f) => {
              const key = f.key as string;
              const ovr = isOverridden(key);
              return (
                <div
                  key={key}
                  className="grid grid-cols-[1.6fr_1.2fr_0.5fr_1fr_0.8fr] items-center gap-2 border-b border-mast-border/40 px-1 py-1.5"
                >
                  <div className="leading-tight">
                    <div className="text-sm text-mast-text">{f.label}</div>
                    <div className="font-mono text-[0.7rem] text-mast-muted">{key}</div>
                  </div>
                  <input
                    type="text"
                    inputMode="decimal"
                    value={view[key] ?? ""}
                    onChange={(e) =>
                      setDraft({ ...(view as Record<string, string>), [key]: e.target.value })
                    }
                    className="rounded-md border border-mast-border bg-mast-bg px-2 py-1.5 font-mono text-sm text-mast-text outline-none focus:border-mast-accent"
                  />
                  <div className="text-xs text-mast-muted">{f.unit}</div>
                  <div className="font-mono text-xs text-mast-muted">
                    {fmtSci(LIMIT_DEFAULTS[key] ?? 0, f.unit)}
                  </div>
                  <div>
                    <Badge tone={ovr ? "WARN" : "AUTO"}>{ovr ? "override" : "default"}</Badge>
                  </div>
                </div>
              );
            })}
            <div className="mt-3 flex flex-wrap items-center gap-3">
              <button
                type="submit"
                disabled={save.isPending}
                className="rounded-md bg-mast-accent/20 px-4 py-2 text-sm font-medium text-mast-accent hover:bg-mast-accent/30 disabled:opacity-50"
              >
                {save.isPending ? "保存中…" : "Save override"}
              </button>
              <button
                type="button"
                onClick={() => { setDraft(null); save.mutate({}); }}
                className="rounded-md border border-mast-border px-4 py-2 text-sm text-mast-muted hover:text-mast-text"
              >
                Reset to defaults
              </button>
              {save.isError && <ErrorNote error={save.error} />}
              {save.data?.degraded && (
                <span className="text-sm text-mast-warn">写入未生效（内核未接入）。</span>
              )}
              {save.data?.ok && !save.data?.degraded && (
                <span className="text-sm text-mast-auto">
                  已保存{save.data.reloaded ? "并热重载" : ""}。
                </span>
              )}
            </div>
          </form>
        </Card>
      )}
    </Section>
  );
}

// ── System self-check ────────────────────────────────────────────────────────
const CHECK_TONE: Record<string, string> = {
  ok: "AUTO",
  warning: "WARN",
  error: "DANGEROUS",
  unavailable: "INFO",
};
const CHECK_LABEL: Record<string, string> = {
  ok: "正常",
  warning: "警告",
  error: "错误",
  unavailable: "不可用",
};

function SystemCheckSection() {
  const [enabled, setEnabled] = useState(false);
  const checkQ = useQuery({
    queryKey: ["system", "check"],
    enabled,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/system/check");
      if (error) throw error;
      return data;
    },
  });

  const columns: ColumnDef<SystemCheckItem, any>[] = [
    { accessorKey: "name", header: "检查项" },
    {
      id: "status",
      header: "状态",
      accessorFn: (r) => r.status,
      cell: (c) => (
        <Badge tone={CHECK_TONE[c.row.original.status] ?? "INFO"}>
          {CHECK_LABEL[c.row.original.status] ?? c.row.original.status}
        </Badge>
      ),
    },
    { accessorKey: "detail", header: "详情" },
  ];

  return (
    <Section
      title="系统自检"
      actions={
        <button
          onClick={() => {
            setEnabled(true);
            checkQ.refetch();
          }}
          disabled={checkQ.isFetching}
          className="rounded-md bg-mast-accent/20 px-3 py-1.5 text-sm font-medium text-mast-accent hover:bg-mast-accent/30 disabled:opacity-50"
        >
          {checkQ.isFetching ? "检查中…" : "运行自检"}
        </button>
      }
    >
      {!enabled && (
        <p className="text-sm text-mast-muted">点击「运行自检」检查 Nanonis / 存储 / LLM / 传感器。</p>
      )}
      {enabled && checkQ.isPending && <Spinner label="检查中…" />}
      {enabled && checkQ.error && <ErrorNote error={checkQ.error} />}
      {enabled && checkQ.data?.degraded && <DegradedNote what="系统自检" />}
      {enabled && checkQ.data && !checkQ.data.degraded && (
        <DataTable data={checkQ.data.items ?? []} columns={columns} empty="无检查结果" />
      )}
    </Section>
  );
}

// ── 安全系统 sub-tab: 4 sub-nav sections (mirrors OLD safety_tab Radio) ────────
type SafetySub = "limits" | "checks" | "constraints" | "skill_levels";
const SAFETY_SUBS: { id: SafetySub; label: string }[] = [
  { id: "limits", label: "全局安全限制" },
  { id: "checks", label: "全局检查规则" },
  { id: "constraints", label: "材料安全约束" },
  { id: "skill_levels", label: "技能安全级别" },
];

function SafetyTab() {
  const [sub, setSub] = useStickyTab<SafetySub>(
    "admin.ic.safety", SAFETY_SUBS.map((t) => t.id), "limits");
  return (
    <div className="space-y-4">
      <SubTabs tabs={SAFETY_SUBS} value={sub} onChange={setSub} />
      {sub === "limits" && <SafetyLimitsSection />}
      {sub === "checks" && <ChecksEditor />}
      {sub === "constraints" && <ConstraintsEditor />}
      {sub === "skill_levels" && <SkillLevelsEditor />}
    </div>
  );
}

// ── 知识库 sub-tab: 5 fine-grained ktype editors ──────────────────────────────
const KNOWLEDGE_TYPES: { id: string; label: string; desc: string }[] = [
  { id: "experiment_design", label: "实验设计", desc: "测量策略 / 参考实验 / 异常响应协议（experiment_design）。" },
  { id: "fault_diagnosis", label: "故障诊断", desc: "故障分类与诊断条目（fault_diagnosis）。" },
  { id: "hardware_profile", label: "硬件画像", desc: "仪器硬件画像与能力（hardware_profile）。" },
  { id: "image_databases", label: "图像数据库", desc: "参考图像数据库（image_databases）。" },
  { id: "workflows", label: "工作流", desc: "工作流分类与配方（workflows）。" },
];

function KnowledgeTab() {
  const [ktype, setKtype] = useState(KNOWLEDGE_TYPES[0]!.id);
  const cur = KNOWLEDGE_TYPES.find((k) => k.id === ktype)!;
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-2">
        {KNOWLEDGE_TYPES.map((k) => (
          <button
            key={k.id}
            onClick={() => setKtype(k.id)}
            className={
              "rounded-md border px-3 py-1.5 text-sm " +
              (ktype === k.id
                ? "border-mast-accent bg-mast-accent/15 text-mast-accent"
                : "border-mast-border text-mast-muted hover:text-mast-text")
            }
          >
            {k.label}
          </button>
        ))}
      </div>
      <JsonSectionEditor
        key={ktype}
        title={`知识库 · ${cur.label}`}
        description={cur.desc}
        getPath="/api/admin/knowledge/{ktype}"
        postPath="/api/admin/knowledge/{ktype}"
        pathParam={{ ktype }}
        queryKey={["knowledge", ktype]}
      />
    </div>
  );
}

// ── 百科配置 sub-tab: domains / intents / hierarchy / verification ────────────
const ENC_SECTIONS: { id: string; label: string; desc: string }[] = [
  { id: "domains", label: "领域 (domains)", desc: "技能领域分组定义（DOMAINS）。" },
  { id: "intents", label: "意图映射 (intents)", desc: "意图到技能/agent 的路由映射（INTENT_MAPPING）。" },
  { id: "hierarchy", label: "复合层级 (hierarchy)", desc: "复合技能层级（COMPOSITE_HIERARCHY）。" },
  { id: "verification", label: "验证 (verification)", desc: "技能验证规则（SKILL_VERIFICATION）。" },
];

function EncyclopediaTab() {
  const [section, setSection] = useState(ENC_SECTIONS[0]!.id);
  const cur = ENC_SECTIONS.find((s) => s.id === section)!;
  return (
    <div>
      <div className="mb-3 flex flex-wrap gap-2">
        {ENC_SECTIONS.map((s) => (
          <button
            key={s.id}
            onClick={() => setSection(s.id)}
            className={
              "rounded-md border px-3 py-1.5 text-sm " +
              (section === s.id
                ? "border-mast-accent bg-mast-accent/15 text-mast-accent"
                : "border-mast-border text-mast-muted hover:text-mast-text")
            }
          >
            {s.label}
          </button>
        ))}
      </div>
      <JsonSectionEditor
        key={section}
        title={`百科配置 · ${cur.label}`}
        description={cur.desc}
        getPath="/api/encyclopedia/config/{section}"
        postPath="/api/encyclopedia/config/{section}"
        pathParam={{ section }}
        queryKey={["encyclopedia", section]}
      />
    </div>
  );
}

// ── IC (仪器控制) FULL governance — the former function-tabs, now nested UNDER
// the IC agent. Order mirrors the OLD IC admin sub-tabs (安全系统 / 技能管理 /
// 知识库 / 技能指导 / 百科配置), then 参考规则. 复杂技能 was consolidated into the
// dedicated 技能构建器 top-tab (composites version/clone/rollback live there) so
// it is intentionally not duplicated here.
type IcSub = "safety" | "skills" | "knowledge" | "guidance" | "encyclopedia" | "reference";
const IC_SUB_TABS: { id: IcSub; label: string }[] = [
  { id: "safety", label: "安全系统" },
  { id: "skills", label: "技能管理" },
  { id: "knowledge", label: "知识库" },
  { id: "guidance", label: "技能指导" },
  { id: "encyclopedia", label: "百科配置" },
  { id: "reference", label: "参考/规则" },
];

function IcGovernanceTabs() {
  const [sub, setSub] = useStickyTab<IcSub>(
    "admin.ic", IC_SUB_TABS.map((t) => t.id), "safety");
  return (
    <div className="space-y-3">
      <p className="text-xs text-mast-muted">
        <strong className="text-mast-text">仪器控制 IC</strong> — 拥有全部 ~250 个 Nanonis V5e
        硬件技能，以下为完整治理面板（安全限值 / 技能元数据 / 知识库 / 技能指导 / 百科配置）。
        ⚠️ <strong className="text-mast-text">改动会影响真实运行的 agent</strong>（经 ConfigOverrideRegistry 持久化并热重载）。
      </p>
      <SubTabs tabs={IC_SUB_TABS} value={sub} onChange={setSub} />
      {sub === "safety" && <SafetyTab />}
      {sub === "skills" && (
        <Section title="技能管理">
          <SkillsManager />
        </Section>
      )}
      {sub === "knowledge" && (
        <Section title="知识库编辑">
          <KnowledgeTab />
        </Section>
      )}
      {sub === "guidance" && (
        <Section title="技能指导（专家标注）">
          <GuidanceEditor />
        </Section>
      )}
      {sub === "encyclopedia" && (
        <Section title="百科配置">
          <EncyclopediaTab />
        </Section>
      )}
      {sub === "reference" && (
        <Section title="参考/规则（只读）">
          <ReferenceTables />
        </Section>
      )}
    </div>
  );
}

// ── 系统 / 全局 — surfaces that were NEVER per-agent in the old admin (hardware
// lived in 设置; system-check / override-history / quick-prompts are MAST
// additions). Grouped here so they stay reachable off the per-agent selector.
type SysSub = "capabilities" | "hardware" | "system" | "history" | "prompts" | "context";
const SYS_SUB_TABS: { id: SysSub; label: string }[] = [
  { id: "capabilities", label: "能力开关" },
  { id: "hardware", label: "硬件连接" },
  { id: "system", label: "系统自检" },
  { id: "history", label: "覆盖历史" },
  { id: "prompts", label: "快速提示" },
  // 上下文注入 — the assembled prompt surface (inventory + override) plus, behind
  // a collapsed section, the real captured requests. NOT the default sub-tab:
  // 一般用户不让他们关心这个。
  { id: "context", label: "上下文注入" },
];

function SystemGlobalTabs() {
  const [sub, setSub] = useStickyTab<SysSub>(
    "admin.system", SYS_SUB_TABS.map((t) => t.id), "capabilities");
  const { toast, node: toastNode } = useToast();
  return (
    <div className="space-y-3">
      {toastNode}
      <p className="text-xs text-mast-muted">
        <strong className="text-mast-text">系统 / 全局</strong> — 非按-agent 的全局管理面（
        能力开关 / 硬件连接 / 系统自检 / 覆盖历史 / 快速提示）。
      </p>
      <SubTabs tabs={SYS_SUB_TABS} value={sub} onChange={setSub} />
      {sub === "capabilities" && (
        <Section title="能力开关 — 授予 agent 默认没有的能力（需管理 PIN）">
          <CapabilityGates toast={toast} />
        </Section>
      )}
      {sub === "hardware" && (
        <Section title="硬件连接">
          <HardwareManager />
        </Section>
      )}
      {sub === "system" && <SystemCheckSection />}
      {sub === "history" && (
        <Section title="覆盖历史 · 查看 + 恢复">
          <OverrideHistory />
        </Section>
      )}
      {sub === "prompts" && (
        <Section title="快速提示编辑">
          <QuickPromptsEditor />
        </Section>
      )}
      {sub === "context" && <PromptInspector />}
    </div>
  );
}

// ── per-agent selector ─────────────────────────────────────────────────────────
// 6 core agents in canonical scientific-workflow order (encyclopedia.AGENT_ORDER
// minus _supervisor / buffer_summarizer), then a final 系统 / 全局 entry. NOT
// instrument-control-first — IC owns the full governance but is no longer listed
// first (mirrors the OLD _admin_agents()).
type AgentSel =
  | "literature"
  | "experiment_design"
  | "instrument_control"
  | "data_processing"
  | "paper_writing"
  | "paper_review"
  | "__system__";

const AGENT_SELECTOR: { value: AgentSel; label: string }[] = [
  { value: "literature", label: "文献 LIT" },
  { value: "experiment_design", label: "实验设计 XD" },
  { value: "instrument_control", label: "仪器控制 IC" },
  { value: "data_processing", label: "数据处理 DP" },
  { value: "paper_writing", label: "论文写作 PW" },
  { value: "paper_review", label: "论文审稿 PR" },
  { value: "__system__", label: "系统 / 全局" },
];

export default function AdminPage() {
  const pinUnlocked = useUiStore((s) => s.pinUnlocked);
  const setPinUnlocked = useUiStore((s) => s.setPinUnlocked);
  // Default to IC — its panel holds the most content (the full governance).
  // 这一页有四层选择器,每层都被复位的话,回到一个改到一半的
  // 「上下文注入」要重新点三下 —— 而中间那几下每一下都换掉整块面板。
  const [agent, setAgent] = useStickyTab<AgentSel>(
    "admin.agent", AGENT_SELECTOR.map((a) => a.value), "instrument_control");

  if (!pinUnlocked) return <PinGate />;

  const cur = AGENT_SELECTOR.find((a) => a.value === agent)!;

  return (
    <div className="space-y-3">
      <div className="mb-2 flex items-center justify-between">
        <h1 className="text-lg font-semibold text-mast-text">高级管理</h1>
        <button
          onClick={() => setPinUnlocked(false)}
          className="rounded-md border border-mast-border px-3 py-1.5 text-sm text-mast-muted hover:text-mast-text"
        >
          锁定
        </button>
      </div>
      <p className="mb-2 text-xs text-mast-muted">
        按 agent 分组的系统配置。<strong className="text-mast-text">仪器控制 IC</strong> 拥有完整治理面板；
        其余 5 个工具型 agent 为只读（工具均 AUTO 级，无安全门控）；模型 / thinking 在「设置」调整。
      </p>

      <RadioGroup value={agent} onChange={setAgent} options={AGENT_SELECTOR} />

      <div className="pt-1">
        {agent === "instrument_control" && <IcGovernanceTabs />}
        {agent === "__system__" && <SystemGlobalTabs />}
        {agent !== "instrument_control" && agent !== "__system__" && (
          <LightAgentPanel agentId={agent} label={cur.label} />
        )}
      </div>
    </div>
  );
}
