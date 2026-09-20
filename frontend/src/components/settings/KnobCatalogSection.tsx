import { useEffect, useState, type ReactNode } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import { ErrorNote, Spinner } from "../ui";
import { Field, Toggle } from "../controls";
import { fmtCurrent, fmtLength } from "../../lib/units";
import { buildKnobPayload, clampKnob, fmtKnobValue, type KnobLike } from "../../lib/knobs";
import type { SettingsPatch } from "@/lib/settingsWrite";

// 设置 → 一个后端旋钮目录 — generic renderer for any config endpoint that ships
// a knob catalogue (电流监控, 环境历史, …).
//
// The rows come from the catalogue rather than a hard-coded list, so adding a
// knob to the backend's EDITABLE_KEYS makes it appear here with no frontend
// change. Generic rather than copied per subsystem because the whole-replace
// guard is subtle and a copied guard is one that drifts — env_history would
// have been that copy.
//
// Everything commits through buildKnobPayload; the guard is documented in
// lib/knobs.ts. No PIN: these are thresholds and retention settings, the same
// class as the classical tip knobs, not a dangerous-capability switch.

/** A knob as the backend catalogues describe it. Structurally identical across
 *  `ThresholdKnob` and `EnvHistoryKnob`, so one row component serves both. */
export interface CatalogKnob {
  key: string;
  label_zh: string;
  hint_zh: string;
  min: number;
  max: number;
  step: number;
  default: number;
  value: number;
  is_bool: boolean;
  /** `current` | `aux` on the monitoring catalogue; absent elsewhere. */
  group?: string;
  /** 有限档位（今天只有 conduct 的 `cd_autonomy`）。空/缺席 = 连续量，用数字框。
   *
   *  档位名的真源在后端（`autonomy.describe`）—— 在这里再写一份中文档位名，
   *  就是那个名字的第二处定义，而两处定义岔开时没有任何东西会报错：界面上
   *  写着「半自主」，存进去的却是别的档。 */
  choices?: Array<{ value: number; label_zh: string }>;
}

/** Amperes get an engineering-notation echo next to the box. Typing 1.2531
 *  where 1.2531e-6 was meant is a real incident in this project's history, and
 *  a value the operator can read as "20.0 pA" is the cheapest guard there is.
 *
 *  ORDER MATTERS. `_m_per_s` has to be tested before `_s`, or a Z drift rate of
 *  5e-11 m/s echoes back as "5e-11 秒" — an echo that says the wrong unit is
 *  worse than no echo, because this box exists to be believed. Same for `_m`
 *  before `_a`-style suffixes: metres and amperes both read as tiny numbers and
 *  the picoprefix hides which one you are looking at. */
function unitEcho(key: string, value: number): string | null {
  if (key.endsWith("_m_per_s")) return `${fmtLength(value)}/s`;
  if (key.endsWith("_a")) return fmtCurrent(value, { placeholder: "—" });
  if (key.endsWith("_m")) return fmtLength(value);
  if (key.endsWith("_hz")) return value > 0 ? `${value} Hz` : "自动";
  if (key.endsWith("_frac") || key.endsWith("_sd_warn")) {
    return `${(value * 100).toFixed(1)} %`;
  }
  if (key.endsWith("_s")) return `${value} 秒`;
  if (key.endsWith("_gb")) return `${value} GB`;
  if (key.endsWith("_hours")) return `${value} 小时`;
  if (key.endsWith("_days")) return `${value} 天`;
  if (key.endsWith("_bins") || key.endsWith("_segments")) return `${value} 个`;
  if (key.endsWith("_k")) return `${value} σ`;
  return null;
}

/** Section heading per group, plus the sentence the operator needs BEFORE
 *  touching those rows. `undefined` means the catalogue has no groups (环境历史)
 *  and everything renders as one flat list, exactly as before. */
const GROUP_META: Record<string, { title: string; note: string }> = {
  current: {
    title: "隧道电流",
    note: "示波器采下来的 2 kHz 电流。阈值单位是安培，按这台机器的真实底噪标定。",
  },
  aux: {
    title: "辅助通道（Z 位置 / qPlus 振幅）",
    note:
      "1 Hz 低速旁路，不占示波器。阈值单位是米与赫兹——" +
      "与上面那组毫无关系，不要照抄。告警出厂是关的：" +
      "先跑 python -m mast.monitoring.commission 看【辅助通道】那段的建议值。",
  },
};

function KnobRow({
  knob,
  value,
  disabled,
  onCommit,
}: {
  knob: CatalogKnob;
  value: number;
  disabled: boolean;
  onCommit: (v: number) => void;
}) {
  const [text, setText] = useState(() => fmtKnobValue(value));
  useEffect(() => {
    setText(fmtKnobValue(value));
  }, [value]);

  const commit = () => {
    const t = text.trim();
    const n = Number.parseFloat(t);
    if (t === "" || Number.isNaN(n)) {
      setText(fmtKnobValue(value)); // revert junk rather than commit a NaN
      return;
    }
    const clamped = clampKnob(knob as KnobLike, n);
    setText(fmtKnobValue(clamped));
    if (clamped !== value) onCommit(clamped);
  };

  const choices = knob.choices ?? [];
  if (choices.length > 0) {
    // 有限档位：分段控件，不是数字框。让用户在一个框里敲「2」，等于要求他
    // 记住那三个数分别是什么 —— 而记错一档在 conduct 上就是「谁能点头」记错。
    const name = knob.label_zh || knob.key;
    return (
      <div className={"py-1" + (disabled ? " opacity-50" : "")}>
        <div className="text-sm text-mast-text">{name}</div>
        {knob.hint_zh && <div className="text-xs text-mast-faint">{knob.hint_zh}</div>}
        <div className="mt-1 flex flex-wrap gap-1" role="radiogroup" aria-label={name}>
          {choices.map((c) => {
            const on = Math.abs(c.value - value) < 1e-9;
            return (
              <button
                key={c.value}
                type="button"
                role="radio"
                aria-checked={on}
                disabled={disabled}
                onClick={() => !disabled && !on && onCommit(c.value)}
                className={
                  "rounded-mast-ctl border px-2 py-1 text-xs " +
                  (on
                    ? "border-mast-accent bg-mast-accent/10 text-mast-accent"
                    : "border-mast-border text-mast-muted hover:text-mast-text")
                }
              >
                {c.label_zh || String(c.value)}
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  if (knob.is_bool) {
    const on = value >= 0.5;
    const name = knob.label_zh || knob.key;
    return (
      <div className={"flex items-center justify-between gap-3 py-1" + (disabled ? " opacity-50" : "")}>
        <div>
          <div className="text-sm text-mast-text">{name}</div>
          {knob.hint_zh && <div className="text-xs text-mast-faint">{knob.hint_zh}</div>}
        </div>
        <span className="flex shrink-0 items-center gap-2">
          {/* Toggle's `label` is only an aria-label, so the state gets a visible
              word too — the knob position and the accent color must not be the
              only thing telling the operator whether 常开采集 is on. */}
          <span className={"text-xs " + (on ? "text-mast-accent" : "text-mast-muted")}>
            {on ? "开" : "关"}
          </span>
          <Toggle
            checked={on}
            onChange={(v) => !disabled && onCommit(v ? 1 : 0)}
            label={`${name}：${on ? "开" : "关"}`}
          />
        </span>
      </div>
    );
  }

  const echo = unitEcho(knob.key, value);
  const bounds =
    knob.max > 0 ? `范围 ${fmtKnobValue(knob.min)} – ${fmtKnobValue(knob.max)}` : "";
  const hint = [knob.hint_zh, bounds, echo && `当前 ≈ ${echo}`]
    .filter(Boolean)
    .join(" · ");

  return (
    <Field label={knob.label_zh || knob.key} hint={hint}>
      <input
        type="text"
        inputMode="decimal"
        value={text}
        disabled={disabled}
        placeholder={fmtKnobValue(knob.default)}
        onChange={(e) => setText(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.currentTarget as HTMLInputElement).blur();
        }}
        className="w-36 rounded border border-mast-border bg-mast-bg px-2 py-1 font-mono text-sm tabular-nums text-mast-text disabled:opacity-50"
      />
    </Field>
  );
}

export function KnobCatalogSection({
  configPath,
  queryKey,
  settingsKey,
  blurb,
  unavailableNote,
  settings,
  saving,
  save,
}: {
  /** The config endpoint that ships the catalogue. */
  configPath:
    | "/api/monitoring/config"
    | "/api/env-history/config"
    | "/api/conducts/config";
  queryKey: readonly unknown[];
  /** The settings-store dict this catalogue writes to, WHOLE. */
  settingsKey: "current_monitor" | "env_history" | "conduct";
  blurb: ReactNode;
  unavailableNote: string;
  settings: { data?: Record<string, unknown> | null };
  saving: boolean;
  save: (patch: SettingsPatch) => void;
}) {
  const config = useQuery({
    queryKey,
    // The catalogue is shape, not telemetry — it only changes when the backend
    // gains a knob. The effective `value`s ride along, but the persisted
    // settings outrank them in buildKnobPayload, so a stale minute is harmless.
    staleTime: 60_000,
    queryFn: async () => {
      const { data, error } = await api.GET(configPath);
      if (error) throw error;
      // conduct 的降级理由字段叫 `reason`（monitoring / env-history 叫
      // `detail`）。少认一个名字的后果不是报错，是那句「为什么读不到」静默
      // 消失 —— 而它正是这个分支存在的理由。
      return data as {
        knobs?: CatalogKnob[];
        degraded?: boolean;
        detail?: string;
        reason?: string;
      };
    },
  });

  if (config.isPending) return <Spinner />;
  if (config.isError) return <ErrorNote error={config.error} />;

  const knobs = config.data?.knobs ?? [];
  const persisted = (settings.data?.[settingsKey] ?? {}) as Record<string, number>;

  if (config.data?.degraded || knobs.length === 0) {
    return (
      <p className="text-sm text-mast-muted">
        {unavailableNote}
        {(() => {
          const why = config.data?.detail || config.data?.reason;
          return why ? `（${why}）` : "";
        })()}
      </p>
    );
  }

  const effective = (k: CatalogKnob): number => {
    const p = persisted[k.key];
    return typeof p === "number" && Number.isFinite(p) ? p : k.value;
  };

  const commit = (key: string, value: number) =>
    save({ [settingsKey]: buildKnobPayload(knobs as KnobLike[], persisted, { key, value }) });

  const resetDefaults = () => save({ [settingsKey]: {} });

  // Group in CATALOGUE ORDER, so the backend's EDITABLE_KEYS stays the single
  // place that decides what appears and in what sequence. A catalogue with no
  // `group` (环境历史) collapses to one unlabelled section — same page as before.
  const sections: Array<{ id: string | undefined; rows: CatalogKnob[] }> = [];
  for (const k of knobs) {
    const last = sections[sections.length - 1];
    if (last && last.id === k.group) last.rows.push(k);
    else sections.push({ id: k.group, rows: [k] });
  }
  const labelled = sections.some((s) => s.id && GROUP_META[s.id]);

  return (
    <div className="max-w-lg space-y-3">
      <p className="text-xs text-mast-muted">{blurb}</p>
      {sections.map((sec) => {
        const meta = sec.id ? GROUP_META[sec.id] : undefined;
        return (
          <div key={sec.id ?? "_"} className={labelled ? "space-y-3 pt-1" : "space-y-3"}>
            {meta && (
              <div className="border-t border-mast-border pt-3">
                <div className="text-sm font-medium text-mast-text">{meta.title}</div>
                <div className="mt-0.5 text-xs leading-snug text-mast-faint">
                  {meta.note}
                </div>
              </div>
            )}
            {sec.rows.map((k) => (
              <KnobRow
                key={k.key}
                knob={k}
                value={effective(k)}
                disabled={saving}
                onCommit={(v) => commit(k.key, v)}
              />
            ))}
          </div>
        );
      })}
      <div className="pt-1">
        <button
          type="button"
          onClick={resetDefaults}
          disabled={saving}
          className="rounded border border-mast-border bg-mast-panel px-3 py-1 text-xs text-mast-muted hover:text-mast-text disabled:opacity-50"
        >
          恢复默认
        </button>
      </div>
      {saving && <p className="text-xs text-mast-muted">保存中…</p>}
    </div>
  );
}
