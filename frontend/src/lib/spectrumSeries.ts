// STS 曲线的数据整形 —— GET /api/scans/spectrum → uPlot。
//
// A point spectrum used to reach this app as a small PNG of the file's first two
// columns: no axes, no units, no zoom, and every other channel (the backward
// sweep, the lock-in that IS the dI/dV) discarded before the wire. This turns the
// numbers into aligned columns for the chart.
//
// The one distinction that must survive intact: **dI/dV has three states.** A
// measured lock-in channel, a numeric derivative of I(V), and nothing. The
// middle one is a different measurement from the first — a derivative of a noisy
// current is not a lock-in trace — so it is labelled wherever it is drawn.

export interface SpectrumSeriesLike {
  id: string;
  name: string;
  /** Optional to match the generated schema: Pydantic's `default_factory=list`
   *  makes the field non-required, so the wire type is `values?`. */
  values?: (number | null)[];
  source: string;
}

export interface SpectrumResponseLike {
  kind?: string;
  sweep_name?: string;
  sweep?: (number | null)[];
  series?: SpectrumSeriesLike[];
  didv_source?: string | null;
  columns?: string[];
}

export type SpectrumMode = "current" | "didv" | "other";

export interface SpectrumModeOption {
  id: SpectrumMode;
  label: string;
}

/** Which curves this file can actually show. Never offers an empty chart. */
export function availableModes(resp: SpectrumResponseLike | null | undefined): SpectrumModeOption[] {
  const series = resp?.series ?? [];
  const out: SpectrumModeOption[] = [];
  if (series.some((s) => s.id.startsWith("current"))) {
    // I(z) sweeps Z, not bias, so calling that tab "I-V" would mislabel the
    // physics. `kind` comes from which column is actually sweeping, not from the
    // file header — 字段标签会说谎.
    out.push({ id: "current", label: resp?.kind === "iz" ? "I-z" : "I-V" });
  }
  if (series.some((s) => s.id.startsWith("didv"))) {
    out.push({ id: "didv", label: "dI/dV" });
  }
  if (series.some((s) => s.id.startsWith("col"))) {
    out.push({ id: "other", label: "全部通道" });
  }
  return out;
}

/** The mode to show, keeping the operator's choice when this file supports it. */
export function resolveMode(
  resp: SpectrumResponseLike | null | undefined,
  preferred: SpectrumMode | null,
): SpectrumMode | null {
  const modes = availableModes(resp);
  if (!modes.length) return null;
  if (preferred && modes.some((m) => m.id === preferred)) return preferred;
  return modes[0]!.id;
}

export interface PreparedSpectrum {
  /** uPlot AlignedData: [x, ...ys]. */
  data: (number | null)[][];
  series: SpectrumSeriesLike[];
  xLabel: string;
  yLabel: string;
  /** True when any drawn curve is a numeric derivative rather than a measurement. */
  hasNumeric: boolean;
}

/**
 * Shape one mode's curves into aligned columns.
 *
 * Every series is emitted at the sweep's length. A series of a different length
 * would silently pair the wrong x with the wrong y — uPlot does not check, it
 * just draws — so a mismatch is padded with nulls, which BREAK the line rather
 * than inventing points. (图上空档不能连线: a gap in a spectrum is missing data,
 * and drawing a straight segment across it asserts a measurement nobody made.)
 */
export function buildSpectrumData(
  resp: SpectrumResponseLike | null | undefined,
  mode: SpectrumMode | null,
): PreparedSpectrum | null {
  if (!resp || !mode) return null;
  const sweep = resp.sweep ?? [];
  if (!sweep.length) return null;

  const prefix = mode === "other" ? "col" : mode;
  const picked = (resp.series ?? []).filter((s) => s.id.startsWith(prefix));
  if (!picked.length) return null;

  const cols: (number | null)[][] = [sweep];
  for (const s of picked) {
    const v = s.values ?? [];
    cols.push(
      v.length === sweep.length
        ? v
        : Array.from({ length: sweep.length }, (_, i) => (i < v.length ? v[i]! : null)),
    );
  }

  return {
    data: cols,
    series: picked,
    xLabel: resp.sweep_name || "",
    yLabel: yLabelFor(mode, picked),
    hasNumeric: picked.some((s) => s.source === "numeric"),
  };
}

function yLabelFor(mode: SpectrumMode, picked: SpectrumSeriesLike[]): string {
  if (mode === "current") return "I (A)";
  if (mode === "didv") return "dI/dV (A/V)";
  // Unnamed channels: the real column names ARE the label, because we have no
  // idea what they mean and guessing is how a previous bug came to label every
  // trace "I (A)".
  return picked.map((s) => s.name).join(" · ");
}

/** 数值微分的说明。null when the dI/dV on screen was actually measured. */
export function numericDidvNote(resp: SpectrumResponseLike | null | undefined): string | null {
  if (resp?.didv_source !== "numeric") return null;
  return "dI/dV 由 I(V) 数值微分得到（文件里没有 lock-in 通道），不是实测调制信号";
}
