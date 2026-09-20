import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "../api/client";
import { Badge, Card, ErrorNote, Section, Spinner } from "../components/ui";
import { Accordion, useToast } from "../components/controls";

// 用量·花销 — records the cost of every paid API call (LLM turns across all
// agents + main chat, voice TTS/ASR, document OCR) and aggregates the spend.
// Data comes from the mast.billing ledger via /api/usage/*. Prices are an
// EDITABLE estimate (see the 价格表 section) — the ledger stores whether each
// row was authoritatively priced, surfaced here with a ≈ badge.

const RANGES = [
  { value: "today", label: "今日" },
  { value: "7d", label: "近 7 天" },
  { value: "30d", label: "近 30 天" },
  { value: "all", label: "全部" },
];

const KIND_LABEL: Record<string, string> = {
  llm: "对话 LLM",
  tts: "语音合成 TTS",
  asr: "语音识别 ASR",
  ocr: "文献 OCR",
  embedding: "向量 Embedding",
};

const SOURCE_LABEL: Record<string, string> = {
  orchestrator: "编排 Orchestrator",
  instrument_control: "仪器控制 IC",
  experiment_design: "实验设计 XD",
  data_processing: "数据处理 DP",
  literature: "文献 Lit",
  paper_writing: "论文写作",
  paper_review: "论文评审",
  chat: "主聊天",
  quickask: "查询助手",
  voice: "语音",
  ocr: "OCR",
  adhoc: "临时调用",
};

function money(cost: number, currency: string): string {
  const sym = currency === "USD" ? "$" : "¥";
  const digits = Math.abs(cost) < 1 ? 4 : 2;
  return `${sym}${cost.toFixed(digits)}`;
}

function fmtInt(n: number): string {
  return (n || 0).toLocaleString("en-US");
}

// ── total cards ────────────────────────────────────────────────────────────
function Totals({ data }: { data: any }) {
  const byCur = data.by_currency ?? {};
  const currencies = Object.keys(byCur);
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-4">
      {currencies.length === 0 && (
        <Card>
          <div className="text-sm text-mast-muted">本区间暂无花销</div>
        </Card>
      )}
      {currencies.map((cur) => {
        const t = byCur[cur];
        return (
          <Card key={cur}>
            <div className="text-xs text-mast-muted">{cur} 花销</div>
            <div className="mt-1 font-mono text-2xl font-semibold text-mast-text">
              {money(t.cost, cur)}
            </div>
            <div className="mt-1 text-xs text-mast-muted">
              {fmtInt(t.count)} 次调用
              {t.estimated_cost > 0 && (
                <span className="ml-1 text-mast-warn">· 含估算 {money(t.estimated_cost, cur)}</span>
              )}
            </div>
          </Card>
        );
      })}
      {data.combined_cny != null && currencies.length > 1 && (
        <Card>
          <div className="text-xs text-mast-muted">折算合计 ≈¥</div>
          <div className="mt-1 font-mono text-2xl font-semibold text-mast-accent">
            ¥{Number(data.combined_cny).toFixed(2)}
          </div>
          <div className="mt-1 text-xs text-mast-muted">按 1 USD ≈ {data.usd_to_cny} CNY 折算</div>
        </Card>
      )}
    </div>
  );
}

// ── one breakdown table ──────────────────────────────────────────────────────
function Breakdown({
  title,
  rows,
  labelMap,
}: {
  title: string;
  rows: any[];
  labelMap?: Record<string, string>;
}) {
  return (
    <Card>
      <div className="mb-2 text-sm font-medium text-mast-text">{title}</div>
      {rows.length === 0 ? (
        <div className="text-xs text-mast-muted">—</div>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-left text-xs text-mast-muted">
                <th className="py-1 pr-2 font-normal">名称</th>
                <th className="py-1 pr-2 text-right font-normal">花销</th>
                <th className="py-1 pr-2 text-right font-normal">次数</th>
                <th className="py-1 text-right font-normal">tokens (in/out)</th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr key={`${r.key}-${r.currency}-${i}`} className="border-t border-mast-border">
                  <td className="py-1 pr-2 text-mast-text">
                    {labelMap?.[r.key] ?? r.key}
                    {!r.all_priced && (
                      <span className="ml-1 align-middle">
                        <Badge tone="WARN">≈</Badge>
                      </span>
                    )}
                  </td>
                  <td className="py-1 pr-2 text-right font-mono text-mast-text">
                    {money(r.cost, r.currency)}
                  </td>
                  <td className="py-1 pr-2 text-right font-mono text-mast-muted">{fmtInt(r.count)}</td>
                  <td className="py-1 text-right font-mono text-xs text-mast-muted">
                    {r.input_tokens || r.output_tokens
                      ? `${fmtInt(r.input_tokens)} / ${fmtInt(r.output_tokens)}`
                      : "—"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

// ── recent calls ─────────────────────────────────────────────────────────────
function Recent({ events }: { events: any[] }) {
  if (events.length === 0) return <div className="text-xs text-mast-muted">暂无调用记录</div>;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-left text-xs text-mast-muted">
            <th className="py-1 pr-3 font-normal">时间</th>
            <th className="py-1 pr-3 font-normal">类型</th>
            <th className="py-1 pr-3 font-normal">模型</th>
            <th className="py-1 pr-3 font-normal">来源</th>
            <th className="py-1 pr-3 text-right font-normal">用量</th>
            <th className="py-1 text-right font-normal">花销</th>
          </tr>
        </thead>
        <tbody>
          {events.map((e, i) => (
            <tr key={i} className="border-t border-mast-border">
              <td className="whitespace-nowrap py-1 pr-3 text-xs text-mast-muted">
                {new Date(e.ts * 1000).toLocaleString("zh-CN", { hour12: false })}
              </td>
              <td className="py-1 pr-3 text-xs text-mast-text">{KIND_LABEL[e.kind] ?? e.kind}</td>
              <td className="py-1 pr-3 font-mono text-xs text-mast-text">{e.model}</td>
              <td className="py-1 pr-3 text-xs text-mast-muted">{SOURCE_LABEL[e.source] ?? e.source}</td>
              <td className="py-1 pr-3 text-right font-mono text-xs text-mast-muted">
                {e.kind === "tts"
                  ? `${fmtInt(e.chars)} 字`
                  : e.kind === "asr"
                    ? `${Math.round(e.seconds)} 秒`
                    : `${fmtInt(e.input_tokens)}/${fmtInt(e.output_tokens)}`}
              </td>
              <td className="py-1 text-right font-mono text-xs text-mast-text">
                {money(e.cost, e.currency)}
                {!e.cost_known && <span className="ml-0.5 text-mast-warn">≈</span>}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

// ── editable price book ──────────────────────────────────────────────────────
function PricingEditor() {
  const qc = useQueryClient();
  const { toast, node } = useToast();
  const pricing = useQuery({
    queryKey: ["usage", "pricing"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/usage/pricing");
      if (error) throw error;
      return data;
    },
  });
  const [rows, setRows] = useState<any[]>([]);
  useEffect(() => {
    if (pricing.data?.models) setRows(pricing.data.models.map((m: any) => ({ ...m })));
  }, [pricing.data]);

  const save = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/usage/pricing", { body: { models: rows } });
      if (error) throw error;
      return data;
    },
    onSuccess: () => {
      toast("价格已保存（后续调用按新价计费）", "ok");
      qc.invalidateQueries({ queryKey: ["usage"] });
    },
    onError: (e) => toast(`保存失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const setField = (i: number, field: string, val: string) =>
    setRows((rs) => rs.map((r, j) => (j === i ? { ...r, [field]: parseFloat(val) || 0 } : r)));

  if (pricing.isPending) return <Spinner />;
  if (pricing.isError) return <ErrorNote error={pricing.error} />;

  return (
    <div>
      <p className="mb-2 text-xs text-mast-muted">
        单价均为<strong>估算默认值</strong>（各家随档位 / 上下文 / 缓存变动）。你付账，你改价——改后仅影响<strong>之后</strong>的调用记账。LLM/OCR 按每百万 token；TTS 按每百万字符；ASR 按每分钟。
      </p>
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="text-left text-xs text-mast-muted">
              <th className="py-1 pr-2 font-normal">模型</th>
              <th className="py-1 pr-2 font-normal">币种</th>
              <th className="py-1 pr-2 text-right font-normal">输入/1M</th>
              <th className="py-1 pr-2 text-right font-normal">输出/1M</th>
              <th className="py-1 pr-2 text-right font-normal">字符/1M</th>
              <th className="py-1 text-right font-normal">每分钟</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((r, i) => (
              <tr key={r.model} className="border-t border-mast-border">
                <td className="py-1 pr-2 font-mono text-xs text-mast-text">{r.model}</td>
                <td className="py-1 pr-2 text-xs text-mast-muted">{r.currency}</td>
                {(["input_per_m", "output_per_m", "char_per_m", "per_minute"] as const).map((f) => (
                  <td key={f} className="py-1 pr-2 text-right">
                    <input
                      type="number"
                      step="0.01"
                      value={r[f]}
                      onChange={(e) => setField(i, f, e.target.value)}
                      className="w-20 rounded border border-mast-border bg-mast-bg/40 px-1 py-0.5 text-right font-mono text-xs text-mast-text"
                    />
                  </td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="mt-3">
        <button
          onClick={() => save.mutate()}
          disabled={save.isPending}
          className="rounded-md border border-mast-accent bg-mast-accent-soft px-3 py-1.5 text-sm font-medium text-mast-accent hover:opacity-90 disabled:opacity-50"
        >
          {save.isPending ? "保存中…" : "保存价格表"}
        </button>
      </div>
      {node}
    </div>
  );
}

// ── page ─────────────────────────────────────────────────────────────────────
export default function UsagePage() {
  const [range, setRange] = useState("7d");
  const [confirmReset, setConfirmReset] = useState(false);
  const qc = useQueryClient();
  const { toast, node } = useToast();

  const summary = useQuery({
    queryKey: ["usage", "summary", range],
    refetchInterval: 15000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/usage/summary", { params: { query: { range } } });
      if (error) throw error;
      return data;
    },
  });
  const recent = useQuery({
    queryKey: ["usage", "recent"],
    refetchInterval: 15000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/usage/recent", { params: { query: { limit: 60 } } });
      if (error) throw error;
      return data;
    },
  });
  const reset = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/usage/reset");
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      toast(`已清空 ${d?.deleted ?? 0} 条记录`, "ok");
      setConfirmReset(false);
      qc.invalidateQueries({ queryKey: ["usage"] });
    },
    onError: (e) => toast(`重置失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  return (
    <div>
      <Section title="用量·花销">
        <p className="mb-4 text-sm text-mast-muted">
          <strong>接口花销记录器</strong> — 统计每一次付费接口调用的花销：全 Agent + 主聊天的 LLM、语音 TTS/ASR、文献 OCR。
          金额按<strong>可编辑的价格表</strong>估算，不同币种分开合计（¥ 与 $）。带 <Badge tone="WARN">≈</Badge> 表示该项用了兜底/未定价，数字仅供参考。
        </p>

        {/* range selector */}
        <div className="mb-4 flex items-center gap-2">
          <div className="inline-flex overflow-hidden rounded-md border border-mast-border">
            {RANGES.map((r) => (
              <button
                key={r.value}
                onClick={() => setRange(r.value)}
                className={
                  "px-3 py-1.5 text-sm " +
                  (range === r.value
                    ? "bg-mast-accent-soft font-medium text-mast-accent"
                    : "text-mast-muted hover:text-mast-text")
                }
              >
                {r.label}
              </button>
            ))}
          </div>
          <div className="ml-auto">
            {confirmReset ? (
              <span className="flex items-center gap-2 text-sm">
                <span className="text-mast-warn">确认清空全部记录？</span>
                <button
                  onClick={() => reset.mutate()}
                  disabled={reset.isPending}
                  className="rounded-md border border-mast-danger-border bg-mast-danger-bg px-2 py-1 text-xs text-mast-danger"
                >
                  确认重置
                </button>
                <button
                  onClick={() => setConfirmReset(false)}
                  className="rounded-md border border-mast-border px-2 py-1 text-xs text-mast-muted"
                >
                  取消
                </button>
              </span>
            ) : (
              <button
                onClick={() => setConfirmReset(true)}
                className="rounded-md border border-mast-border px-2 py-1 text-xs text-mast-muted hover:text-mast-text"
              >
                清空记录
              </button>
            )}
          </div>
        </div>

        {summary.isPending ? (
          <Spinner />
        ) : summary.isError ? (
          <ErrorNote error={summary.error} />
        ) : (
          <div className="space-y-4">
            <Totals data={summary.data} />

            <div className="grid grid-cols-1 gap-4 lg:grid-cols-2">
              <Breakdown title="按供应商 Provider" rows={summary.data.by_provider ?? []} />
              <Breakdown title="按类型 Kind" rows={summary.data.by_kind ?? []} labelMap={KIND_LABEL} />
              <Breakdown title="按模型 Model" rows={summary.data.by_model ?? []} />
              <Breakdown title="按来源 Agent/来源" rows={summary.data.by_source ?? []} labelMap={SOURCE_LABEL} />
            </div>

            <Card>
              <div className="mb-2 text-sm font-medium text-mast-text">最近调用</div>
              {recent.isPending ? <Spinner /> : recent.isError ? (
                <ErrorNote error={recent.error} />
              ) : (
                <Recent events={recent.data?.events ?? []} />
              )}
            </Card>

            <Accordion title="价格表 Pricing（可编辑 · 估算默认值）" defaultOpen={false}>
              <PricingEditor />
            </Accordion>
          </div>
        )}
      </Section>
      {node}
    </div>
  );
}
