import { useEffect, useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import { Field } from "../controls";
import { ErrorNote, Spinner } from "../ui";
import type { SettingsPatch } from "@/lib/settingsWrite";

// ── 扫描参数档位表(按尺度)────────────────────────────────────────────────────
//
// 「这个尺度下该扫多快、取多少像素」是用户按仪器、按样品积累出来的知识,在这
// 之前系统里没有存放它的地方 —— LLM 只能每次现编一个数字。这一节就是那张表。
//
// 档数**可变**(1..8):不同人对「几个尺度」的划分不一样,固定四档只是
// 一种假设,不一定贴合真实工作方式。
//
// bias 刻意不在表里:它决定探测的电子态与成像对比,是物理意图参数,不是尺度的
// 函数。它留在「实验默认参数」那一节。
//
// 整表替换语义:SettingsStore 不做 merge,所以保存必须回传完整的 tiers 列表。
// 后端在**持久化之前**做结构校验(区间单调 / 恰好一个兜底档 / 档数上限),
// 结构非法整表拒绝 —— 半张表比没有表更危险。

type Tier = {
  name?: string | null;
  upper_size_m?: number | null;
  pixels?: number | null;
  line_time_s?: number | null;
  setpoint_a?: number | null;
  p_gain?: number | null;
  time_constant_s?: number | null;
  source?: string | null;
  _factory_filled?: string[] | null;
};

type PolicyPayload = {
  tiers: Tier[];
  stored: Tier[];
  factory: Tier[];
  customised: boolean;
  min_tiers: number;
  max_tiers: number;
};

function useScanPolicy() {
  return useQuery({
    queryKey: ["settings", "scan-policy"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/settings/scan-policy");
      if (error) throw error;
      return data as unknown as PolicyPayload;
    },
  });
}

function usePreview(sizeNm: number) {
  return useQuery({
    queryKey: ["settings", "scan-policy", "preview", sizeNm],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/settings/scan-policy/preview", {
        params: { query: { size_nm: sizeNm } },
      });
      if (error) throw error;
      return data as unknown as {
        ok: boolean;
        error?: string;
        tier_name?: string;
        pixels?: number;
        line_time_s?: number;
        estimated_scan_s?: number;
        summary?: string[];
        warnings?: string[];
      };
    },
    enabled: Number.isFinite(sizeNm) && sizeNm > 0,
  });
}

const nmOf = (m: number | null | undefined): string =>
  m === null || m === undefined ? "" : String(Math.round(m * 1e9 * 1e6) / 1e6);

const paOf = (a: number | null | undefined): string =>
  a === null || a === undefined ? "" : String(Math.round(a * 1e12 * 1e6) / 1e6);

const numOr = (s: string): number | null => {
  const t = s.trim();
  if (t === "") return null;
  const n = Number.parseFloat(t);
  return Number.isFinite(n) ? n : null;
};

const fmtDuration = (s: number | undefined): string => {
  if (!s || s <= 0) return "—";
  if (s < 90) return `${Math.round(s)} 秒`;
  return `${(s / 60).toFixed(1)} 分钟`;
};

/** 一行档位。全部字段以人类单位显示(nm / pA),提交时转 SI。 */
function TierRow({
  tier,
  index,
  isLast,
  factory,
  disabled,
  onChange,
  onRemove,
}: {
  tier: Tier;
  index: number;
  isLast: boolean;
  factory: Tier | undefined;
  disabled: boolean;
  onChange: (patch: Partial<Tier>) => void;
  onRemove: () => void;
}) {
  return (
    <div className="rounded border border-mast-border p-2">
      <div className="mb-2 flex items-center gap-2">
        <input
          type="text"
          value={tier.name ?? ""}
          disabled={disabled}
          placeholder={factory?.name ?? `档 ${index + 1}`}
          onChange={(e) => onChange({ name: e.target.value })}
          className="w-28 rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text disabled:opacity-50"
        />
        <span className="text-xs text-mast-muted">边长 ≤</span>
        {isLast ? (
          <span className="text-xs text-mast-muted">
            (兜底档 — 比上一档更大的尺寸全部落这里)
          </span>
        ) : (
          <>
            <input
              type="text"
              inputMode="decimal"
              value={nmOf(tier.upper_size_m)}
              disabled={disabled}
              placeholder="如 100"
              onChange={(e) =>
                onChange({
                  upper_size_m: (() => {
                    const n = numOr(e.target.value);
                    return n === null ? null : n * 1e-9;
                  })(),
                })
              }
              className="w-24 rounded border border-mast-border bg-mast-bg px-2 py-1 font-mono text-sm tabular-nums text-mast-text disabled:opacity-50"
            />
            <span className="text-xs text-mast-muted">nm</span>
          </>
        )}
        <button
          type="button"
          disabled={disabled}
          onClick={onRemove}
          className="ml-auto rounded border border-mast-border px-2 py-0.5 text-xs text-mast-muted hover:text-mast-text disabled:opacity-50"
        >
          删除
        </button>
      </div>

      <div className="grid grid-cols-2 gap-2 md:grid-cols-5">
        <label className="text-xs text-mast-muted">
          像素 / 线数
          <input
            type="text"
            inputMode="numeric"
            value={tier.pixels ?? ""}
            disabled={disabled}
            placeholder={String(factory?.pixels ?? 256)}
            onChange={(e) => onChange({ pixels: numOr(e.target.value) })}
            className="mt-0.5 w-full rounded border border-mast-border bg-mast-bg px-2 py-1 font-mono text-sm tabular-nums text-mast-text disabled:opacity-50"
          />
        </label>
        <label className="text-xs text-mast-muted">
          每线时间 (s)
          <input
            type="text"
            inputMode="decimal"
            value={tier.line_time_s ?? ""}
            disabled={disabled}
            placeholder={String(factory?.line_time_s ?? 1)}
            onChange={(e) => onChange({ line_time_s: numOr(e.target.value) })}
            className="mt-0.5 w-full rounded border border-mast-border bg-mast-bg px-2 py-1 font-mono text-sm tabular-nums text-mast-text disabled:opacity-50"
          />
        </label>
        <label className="text-xs text-mast-muted">
          setpoint (pA)
          <input
            type="text"
            inputMode="decimal"
            value={paOf(tier.setpoint_a)}
            disabled={disabled}
            placeholder="留空 = 不改"
            onChange={(e) =>
              onChange({
                setpoint_a: (() => {
                  const n = numOr(e.target.value);
                  return n === null ? null : n * 1e-12;
                })(),
              })
            }
            className="mt-0.5 w-full rounded border border-mast-border bg-mast-bg px-2 py-1 font-mono text-sm tabular-nums text-mast-text disabled:opacity-50"
          />
        </label>
        <label className="text-xs text-mast-muted">
          Z 反馈 P
          <input
            type="text"
            inputMode="decimal"
            value={tier.p_gain ?? ""}
            disabled={disabled}
            placeholder="留空 = 不改"
            onChange={(e) => onChange({ p_gain: numOr(e.target.value) })}
            className="mt-0.5 w-full rounded border border-mast-border bg-mast-bg px-2 py-1 font-mono text-sm tabular-nums text-mast-text disabled:opacity-50"
          />
        </label>
        <label className="text-xs text-mast-muted">
          Z 反馈 T (s)
          <input
            type="text"
            inputMode="decimal"
            value={tier.time_constant_s ?? ""}
            disabled={disabled}
            placeholder="留空 = 不改"
            onChange={(e) => onChange({ time_constant_s: numOr(e.target.value) })}
            className="mt-0.5 w-full rounded border border-mast-border bg-mast-bg px-2 py-1 font-mono text-sm tabular-nums text-mast-text disabled:opacity-50"
          />
        </label>
      </div>
      {(tier.p_gain === null || tier.p_gain === undefined) !==
        (tier.time_constant_s === null || tier.time_constant_s === undefined) && (
        <p className="mt-1 text-xs text-amber-500">
          Z 反馈的 P 与时间常数必须成对填写(积分增益 I = P/T 由两者导出),
          只填一个时本档不会下发 PI 设置。
        </p>
      )}
    </div>
  );
}

export function ScanPolicySection({
  saving,
  save,
}: {
  saving: boolean;
  save: (patch: SettingsPatch) => void;
}) {
  const policy = useScanPolicy();
  const [draft, setDraft] = useState<Tier[] | null>(null);
  const [previewNm, setPreviewNm] = useState(100);
  const preview = usePreview(previewNm);

  // 服务端是权威:每次拉到新数据就重置草稿,避免编辑器停留在一份陈旧的表上。
  useEffect(() => {
    if (policy.data) setDraft(policy.data.tiers.map((t) => ({ ...t })));
  }, [policy.data]);

  const factoryByIndex = useMemo(() => policy.data?.factory ?? [], [policy.data]);

  if (policy.isLoading) return <Spinner />;
  if (policy.error) return <ErrorNote error={policy.error} />;
  if (!policy.data || !draft) return null;

  const { min_tiers: minTiers, max_tiers: maxTiers, customised } = policy.data;

  const commit = (next: Tier[]) => {
    // 整表替换:store 不 merge,少送一档就等于删掉它。
    const payload = next.map((t, i) => ({
      name: (t.name ?? "").trim() || `tier${i + 1}`,
      // 最后一档永远是兜底档(upper 留空);中间档必须有边界。
      upper_size_m: i === next.length - 1 ? null : t.upper_size_m ?? null,
      pixels: t.pixels ?? null,
      line_time_s: t.line_time_s ?? null,
      setpoint_a: t.setpoint_a ?? null,
      p_gain: t.p_gain ?? null,
      time_constant_s: t.time_constant_s ?? null,
    }));
    save({ scan_policy: { tiers: payload } });
  };

  const updateRow = (index: number, patch: Partial<Tier>) => {
    const next = draft.map((t, i) => (i === index ? { ...t, ...patch } : t));
    setDraft(next);
  };

  const addRow = () => {
    if (draft.length >= maxTiers) return;
    // 新档插在倒数第二位:最后一档必须留给兜底档。边界默认取前一档的 2 倍,
    // 保证插入后仍然单调递增(否则后端会整表拒绝,而用户不知道为什么)。
    const prevBound = draft.length >= 2 ? draft[draft.length - 2]?.upper_size_m : null;
    const seed = prevBound ? prevBound * 2 : 1e-7;
    const next = [...draft];
    next.splice(next.length - 1, 0, {
      name: `tier${draft.length}`,
      upper_size_m: seed,
      pixels: draft[draft.length - 1]?.pixels ?? 256,
      line_time_s: draft[draft.length - 1]?.line_time_s ?? 1.0,
      setpoint_a: null,
      p_gain: null,
      time_constant_s: null,
    });
    setDraft(next);
  };

  const removeRow = (index: number) => {
    if (draft.length <= minTiers) return;
    setDraft(draft.filter((_, i) => i !== index));
  };

  return (
    <div className="max-w-4xl space-y-3">
      <p className="text-xs text-mast-muted">
        每个尺度下你惯用的<b>扫描速度与分辨率</b>。扫图时 agent 只说「扫哪、多大」,
        速度 / 像素 / 反馈增益由这张表<b>确定性地</b>决定 —— 那些量取决于这台仪器、
        这个针尖、这种样品,不该让模型去猜。
        <br />
        表按<b>扫描边长</b>查,取第一个「边长 ≤ 上界」的档;最后一档是兜底档,
        接住所有更大的尺寸。
        <br />
        <b>留空的 setpoint / PI 表示不改动硬件当前值</b>(不是设成 0)。
        <b>偏压不在这张表里</b> —— 它决定探测的电子态,是物理决策,不是尺度的函数。
      </p>

      {!customised && (
        <p className="rounded border border-mast-border bg-mast-bg px-2 py-1 text-xs text-mast-muted">
          当前用的是<b>出厂参考值</b>(取自渐进缩放协议的四级建议)。这些数字随仪器和
          样品变化,<b>请按本机实际情况校准</b>后再依赖它们。
        </p>
      )}

      <div className="space-y-2">
        {draft.map((tier, i) => (
          <TierRow
            key={i}
            tier={tier}
            index={i}
            isLast={i === draft.length - 1}
            factory={factoryByIndex[Math.min(i, factoryByIndex.length - 1)]}
            disabled={saving}
            onChange={(patch) => updateRow(i, patch)}
            onRemove={() => removeRow(i)}
          />
        ))}
      </div>

      <div className="flex flex-wrap items-center gap-3 pt-1">
        <button
          type="button"
          disabled={saving || draft.length >= maxTiers}
          onClick={addRow}
          className="rounded border border-mast-border px-3 py-1 text-sm text-mast-text disabled:opacity-50"
        >
          + 增加一档
        </button>
        <button
          type="button"
          disabled={saving}
          onClick={() => commit(draft)}
          className="rounded border border-mast-accent bg-mast-accent/10 px-3 py-1 text-sm text-mast-text disabled:opacity-50"
        >
          保存档位表
        </button>
        <button
          type="button"
          disabled={saving || !customised}
          onClick={() => save({ scan_policy: { tiers: [] } })}
          className="rounded border border-mast-border px-3 py-1 text-sm text-mast-muted disabled:opacity-50"
        >
          恢复出厂表
        </button>
        <span className="text-xs text-mast-muted">
          {draft.length} / {maxTiers} 档
        </span>
      </div>

      {/* 预览:改完表立刻看到「一张 X nm 的图会用什么参数」,不必真的扫一张去验证 */}
      <div className="mt-3 rounded border border-mast-border p-2">
        <Field
          label="预览"
          hint="输入一个扫描边长,看这张表会给出什么参数(零硬件成本)。"
        >
          <div className="flex items-center gap-2">
            <input
              type="text"
              inputMode="decimal"
              defaultValue={String(previewNm)}
              onBlur={(e) => {
                const n = numOr(e.target.value);
                if (n !== null && n > 0) setPreviewNm(n);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter") (e.currentTarget as HTMLInputElement).blur();
              }}
              className="w-24 rounded border border-mast-border bg-mast-bg px-2 py-1 font-mono text-sm tabular-nums text-mast-text"
            />
            <span className="text-xs text-mast-muted">nm</span>
          </div>
        </Field>
        {preview.data?.ok && (
          <div className="mt-1 space-y-1 text-xs text-mast-muted">
            <div>
              档位 <b className="text-mast-text">{preview.data.tier_name}</b>
              {" · "}
              {preview.data.pixels}px{" · "}
              {preview.data.line_time_s}s/线{" · "}
              预计 <b className="text-mast-text">{fmtDuration(preview.data.estimated_scan_s)}</b>
            </div>
            {(preview.data.warnings ?? []).map((w, i) => (
              <div key={i} className="text-amber-500">
                {w}
              </div>
            ))}
            <details>
              <summary className="cursor-pointer">每个参数的来源</summary>
              <ul className="mt-1 space-y-0.5 font-mono">
                {(preview.data.summary ?? []).map((line, i) => (
                  <li key={i}>{line}</li>
                ))}
              </ul>
            </details>
          </div>
        )}
        {preview.data && !preview.data.ok && (
          <p className="mt-1 text-xs text-red-500">{preview.data.error}</p>
        )}
      </div>
    </div>
  );
}
