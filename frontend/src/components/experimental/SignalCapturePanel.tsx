import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import type { components } from "@/api/schema";
import { Card, ErrorNote, EmptyNote, Spinner } from "@/components/ui";
import {
  Button,
  Field,
  RadioGroup,
  SelectField,
  TextField,
  Toggle,
  useToast,
} from "@/components/controls";
import { FftChart, type FftSeries } from "@/components/vision/FftChart";
import { TimeTraceChart } from "@/components/experimental/TimeTraceChart";

type SignalChannel = components["schemas"]["SignalChannel"];
type FFT = components["schemas"]["FFTResponse"];
type WindowFn = "hann" | "hamming" | "rect";
type OutputKind = "magnitude" | "power";

// ─────────────────────────────────────────────────────────────────────────────
// 信号捕获 + FFT — ITEM 6 (fuller redesign of gui/exp_capture).
//
// The live hardware capture (Osci1T 20 kHz / 轮询 fast-path) runs in the desktop
// core and streams huge sample arrays that NEVER cross the typed HTTP wire — so
// there is no read-only capture endpoint. What IS exposed:
//   · GET  /api/experimental/signals — channels + units + Osci timebases + enums
//   · POST /api/experimental/fft     — pure one-sided rfft over supplied samples
//
// Added capability on top of the first redesign:
//   1. 多通道选择 / 通道对比 — pick 2+ channels, paste a trace each, overlay spectra
//   2. 平均 (N 次) — average N FFT computes to pull down the noise floor + 计数器
//   3. 峰值读出表 — top-N peaks per channel (DataTable in FftChart)
//   4. 频率轴控制 — min/max Hz 缩放 + 线性/对数 X（外加既有对数 Y）
//   5. 实时连续模式 — 按间隔自动重算 vs 单次
//   6. 单位/标度读出 — |FFT| (V) / PSD (V/√Hz) 标注
//   7. 预设保存/加载 (localStorage) + 时域 trace & 频谱各导 CSV + PNG
// Degrade-safe: no live signals → default channel list (degraded flag); live
// capture needs hardware → clear "需连接 Nanonis" note; never freezes.
// ─────────────────────────────────────────────────────────────────────────────

const PRESET_KEY = "mast.exp.capture.presets.v1";

interface CapturePreset {
  name: string;
  windowFn: WindowFn;
  output: OutputKind;
  logY: boolean;
  logX: boolean;
  dropDc: boolean;
  showPeaks: boolean;
  topN: number;
  fMinText: string;
  fMaxText: string;
  avgN: number;
  manualFsText: string;
  nPointsText: string;
}

function parseSamples(text: string): number[] {
  return text
    .split(/[\s,;]+/)
    .map((s) => s.trim())
    .filter((s) => s.length > 0)
    .map(Number)
    .filter((n) => Number.isFinite(n));
}

/** Build a representative demo time-domain trace so the FFT path can be exercised
 *  WITHOUT hardware: a couple of sine components on a small noise floor. `seed`
 *  decorrelates the noise/phase per channel so an overlay compare looks distinct.
 *  Mirrors the kind of trace gui/exp_capture captured from Osci1T (a few tones). */
function demoTrace(n = 1024, fsHz = 20000, seed = 0): number[] {
  // Two tones placed well inside Nyquist, scaled to a typical pA-ish magnitude.
  const f1 = 1000 + seed * 250; // Hz
  const f2 = 3500 + seed * 500; // Hz
  // Cheap deterministic PRNG so the demo is stable across re-renders.
  let s = (seed + 1) * 1234567;
  const rand = () => {
    s = (s * 1103515245 + 12345) & 0x7fffffff;
    return s / 0x7fffffff - 0.5;
  };
  const fs = fsHz > 0 ? fsHz : 20000;
  const out: number[] = new Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / fs;
    out[i] =
      1.0 * Math.sin(2 * Math.PI * f1 * t + seed) +
      0.5 * Math.sin(2 * Math.PI * f2 * t) +
      0.15 * rand();
  }
  return out;
}

/** Format a number array back into a compact pasteable string. */
function formatSamples(arr: number[]): string {
  return arr.map((v) => v.toFixed(4)).join(", ");
}

function downloadText(filename: string, text: string, mime = "text/csv;charset=utf-8") {
  const blob = new Blob([text], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}

/** Export the on-screen uPlot canvas as a PNG. Best-effort: if the canvas is not
 *  found we no-op (a dead export must never freeze the page). */
function exportCanvasPng(container: HTMLElement | null, filename: string): boolean {
  const canvas = container?.querySelector("canvas");
  if (!canvas) return false;
  try {
    const url = (canvas as HTMLCanvasElement).toDataURL("image/png");
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    return true;
  } catch {
    return false;
  }
}

function loadPresets(): CapturePreset[] {
  try {
    const raw = localStorage.getItem(PRESET_KEY);
    if (!raw) return [];
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? (parsed as CapturePreset[]) : [];
  } catch {
    return [];
  }
}

/** Average a list of spectra bin-wise (element-wise mean, clamped to the shortest
 *  length). Returns a synthetic FFTResponse carrying the mean spectrum. Used for
 *  the N-capture averaging that pulls the noise floor down. */
function averageSpectra(specs: FFT[]): FFT | null {
  const valid = specs.filter((s) => s.ok && (s.spectrum?.length ?? 0) > 0);
  if (!valid.length) return null;
  const n = Math.min(...valid.map((s) => s.spectrum!.length));
  const acc = new Array<number>(n).fill(0);
  for (const s of valid) {
    const sp = s.spectrum!;
    for (let i = 0; i < n; i++) acc[i]! += sp[i]!;
  }
  for (let i = 0; i < n; i++) acc[i]! /= valid.length;
  const base = valid[0]!;
  return {
    ...base,
    spectrum: acc,
    freqs_hz: (base.freqs_hz ?? []).slice(0, n),
  };
}

/** One selected channel's editing state (its pasted trace + last computed FFT). */
interface ChannelState {
  channel: SignalChannel;
  samplesText: string;
  fft: FFT | null;
  trace: { ts: number[]; ys: number[] } | null;
  // Accumulated spectra for the N-capture average + the running counter.
  acc: FFT[];
}

export function SignalCapturePanel() {
  const { toast, node } = useToast();

  // ── /signals metadata (channels, timebases, enums) ──
  const signals = useQuery({
    queryKey: ["experimental", "signals"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/experimental/signals");
      if (error) throw error;
      return data;
    },
    staleTime: 30_000,
  });

  const meta = signals.data;
  const channels: SignalChannel[] = meta?.channels ?? [];
  const timebases = meta?.timebases ?? [];

  // ── Channel picker (MULTI-select for channel compare) ──
  const [search, setSearch] = useState("");
  const [selectedIdx, setSelectedIdx] = useState<number[]>([]);

  // Default-select the first current channel once metadata arrives.
  const didDefaultRef = useRef(false);
  useEffect(() => {
    if (didDefaultRef.current || channels.length === 0) return;
    const curIdx = meta?.current_indices?.[0] ?? channels[0]?.index;
    if (curIdx != null) {
      setSelectedIdx([curIdx]);
      didDefaultRef.current = true;
    }
  }, [channels, meta]);

  const filteredChannels = useMemo(() => {
    const q = search.trim().toLowerCase();
    if (!q) return channels;
    return channels.filter(
      (c) => c.name.toLowerCase().includes(q) || String(c.index).includes(q),
    );
  }, [channels, search]);

  const selectedChannels = useMemo(
    () => selectedIdx.map((i) => channels.find((c) => c.index === i)).filter(Boolean) as SignalChannel[],
    [selectedIdx, channels],
  );

  const toggleChannel = (idx: number) =>
    setSelectedIdx((prev) => (prev.includes(idx) ? prev.filter((i) => i !== idx) : [...prev, idx]));

  // ── Acquisition controls ──
  const [timebaseIdx, setTimebaseIdx] = useState<string>("");
  const [nPointsText, setNPointsText] = useState("4096");
  const selectedTimebase = useMemo(
    () => timebases.find((t) => String(t.index) === timebaseIdx) ?? null,
    [timebases, timebaseIdx],
  );
  const [manualFsText, setManualFsText] = useState("20000");
  const fsHz = selectedTimebase?.fs_hz ?? Number(manualFsText);
  const nPoints = Math.max(0, Math.floor(Number(nPointsText) || 0));
  const durationS = fsHz > 0 ? nPoints / fsHz : 0;
  const nyquist = fsHz > 0 ? fsHz / 2 : 0;

  // ── FFT options ──
  const [windowFn, setWindowFn] = useState<WindowFn>("hann");
  const [output, setOutput] = useState<OutputKind>("magnitude");
  const [logY, setLogY] = useState(true);
  const [logX, setLogX] = useState(false);
  const [dropDc, setDropDc] = useState(true);
  const [showPeaks, setShowPeaks] = useState(true);
  const [topN, setTopN] = useState(5);

  // ── Frequency-axis zoom window ──
  const [fMinText, setFMinText] = useState("");
  const [fMaxText, setFMaxText] = useState("");
  const fMin = fMinText.trim() === "" ? null : Number(fMinText);
  const fMax = fMaxText.trim() === "" ? null : Number(fMaxText);

  // ── Averaging (N captures) + live continuous mode ──
  const [avgN, setAvgN] = useState(1);
  const [liveMode, setLiveMode] = useState(false);
  const [liveIntervalText, setLiveIntervalText] = useState("2");

  // ── Per-channel editing state (pasted trace + last spectrum + average accum) ──
  const [chanStates, setChanStates] = useState<Record<number, ChannelState>>({});
  // Collapse the raw paste textareas by default — the one-click 示例 → FFT path is
  // the default experience; manual pasting is an advanced/opt-in area.
  const [showManualPaste, setShowManualPaste] = useState(false);

  // Keep a ChannelState for every selected channel; drop deselected ones.
  useEffect(() => {
    setChanStates((prev) => {
      const next: Record<number, ChannelState> = {};
      for (const ch of selectedChannels) {
        const existing = prev[ch.index];
        // refresh the channel ref in case metadata reloaded; keep pasted trace.
        next[ch.index] = existing
          ? { ...existing, channel: ch }
          : { channel: ch, samplesText: "", fft: null, trace: null, acc: [] };
      }
      return next;
    });
  }, [selectedChannels]);

  const setSamplesFor = (idx: number, text: string) =>
    setChanStates((prev) => ({ ...prev, [idx]: { ...prev[idx]!, samplesText: text } }));

  // Fill one channel's textarea with a representative demo trace (sine+noise) so
  // the FFT can be tried WITHOUT hardware. `seed` keeps channels distinct in the
  // overlay compare. Uses the current N / fs so the spectrum lands sensibly.
  const fillDemoFor = (idx: number, seed: number) => {
    const n = nPoints > 0 ? Math.min(nPoints, 4096) : 1024;
    const fs = Number.isFinite(fsHz) && fsHz > 0 ? fsHz : 20000;
    setSamplesFor(idx, formatSamples(demoTrace(n, fs, seed)));
  };

  // Fill EVERY selected channel at once (one-click demo → compute → FFT).
  const fillDemoAll = () => {
    const n = nPoints > 0 ? Math.min(nPoints, 4096) : 1024;
    const fs = Number.isFinite(fsHz) && fsHz > 0 ? fsHz : 20000;
    setChanStates((prev) => {
      const next = { ...prev };
      selectedChannels.forEach((ch, i) => {
        if (next[ch.index]) next[ch.index] = { ...next[ch.index]!, samplesText: formatSamples(demoTrace(n, fs, i)) };
      });
      return next;
    });
    toast(`已为 ${selectedChannels.length} 个通道填入示例数据`);
  };

  const windowOptions: { value: WindowFn; label: string }[] = meta?.windows?.length
    ? meta.windows.map((w) => ({ value: w.value as WindowFn, label: w.label }))
    : [
        { value: "hann", label: "Hann" },
        { value: "hamming", label: "Hamming" },
        { value: "rect", label: "矩形/无窗 (Rect)" },
      ];
  const outputOptions: { value: OutputKind; label: string }[] = meta?.output_modes?.length
    ? meta.output_modes.map((o) => ({ value: o.value as OutputKind, label: o.label }))
    : [
        { value: "magnitude", label: "幅度 |FFT|" },
        { value: "power", label: "功率谱 PSD" },
      ];

  // ── Compute one FFT for one channel via POST /fft ──
  const computeOne = useCallback(
    async (st: ChannelState): Promise<FFT> => {
      const samples = parseSamples(st.samplesText);
      const { data, error } = await api.POST("/api/experimental/fft", {
        body: {
          samples,
          fs_hz: Number.isFinite(fsHz) && fsHz > 0 ? fsHz : null,
          window: windowFn,
          output,
          detrend: dropDc,
          unit: st.channel.unit ?? "A",
          channel_name: st.channel.name,
        },
      });
      if (error) throw error;
      return data as FFT;
    },
    [fsHz, windowFn, output, dropDc],
  );

  // ── Compute (single shot or one live tick): for each selected channel run the
  //    FFT, fold into the N-capture average, keep last `avgN` spectra. ──
  const [computing, setComputing] = useState(false);
  const [computeError, setComputeError] = useState<unknown>(null);

  const runCompute = useCallback(async () => {
    if (selectedChannels.length === 0) return;
    setComputing(true);
    setComputeError(null);
    try {
      const updates: Record<number, ChannelState> = {};
      for (const ch of selectedChannels) {
        const st = chanStates[ch.index];
        if (!st) continue;
        const samples = parseSamples(st.samplesText);
        if (samples.length < 4) {
          updates[ch.index] = st; // leave untouched; flagged in UI
          continue;
        }
        const fft = await computeOne(st);
        const acc = [...st.acc, fft].slice(-Math.max(1, avgN));
        const ts = fsHz > 0 ? samples.map((_, i) => i / fsHz) : samples.map((_, i) => i);
        updates[ch.index] = { ...st, fft, acc, trace: { ts, ys: samples } };
      }
      setChanStates((prev) => ({ ...prev, ...updates }));
    } catch (e) {
      setComputeError(e);
      toast(String((e as Error)?.message ?? e), "err");
    } finally {
      setComputing(false);
    }
  }, [selectedChannels, chanStates, computeOne, avgN, fsHz, toast]);

  // ── Live continuous mode: re-run the compute on an interval. Never blocks —
  //    a tick only POSTs the already-pasted traces. ──
  const liveTimer = useRef<number | null>(null);
  useEffect(() => {
    if (!liveMode) {
      if (liveTimer.current != null) {
        window.clearInterval(liveTimer.current);
        liveTimer.current = null;
      }
      return;
    }
    const ms = Math.max(250, (Number(liveIntervalText) || 2) * 1000);
    liveTimer.current = window.setInterval(() => {
      void runCompute();
    }, ms);
    return () => {
      if (liveTimer.current != null) {
        window.clearInterval(liveTimer.current);
        liveTimer.current = null;
      }
    };
  }, [liveMode, liveIntervalText, runCompute]);

  // ── The overlay series fed to FftChart: per channel, the averaged spectrum
  //    (if avgN > 1 and we have ≥1 accumulated) else the last single FFT. ──
  const series: FftSeries[] = useMemo(() => {
    const out: FftSeries[] = [];
    for (const ch of selectedChannels) {
      const st = chanStates[ch.index];
      if (!st) continue;
      const eff = avgN > 1 && st.acc.length > 0 ? averageSpectra(st.acc) : st.fft;
      if (eff && eff.ok) {
        const suffix = avgN > 1 && st.acc.length > 1 ? ` (avg×${st.acc.length})` : "";
        out.push({ fft: eff, label: `${ch.name}${suffix}` });
      }
    }
    return out;
  }, [selectedChannels, chanStates, avgN]);

  const captureCount = useMemo(
    () => selectedChannels.reduce((m, ch) => Math.max(m, chanStates[ch.index]?.acc.length ?? 0), 0),
    [selectedChannels, chanStates],
  );

  const resetAverage = () =>
    setChanStates((prev) => {
      const next = { ...prev };
      for (const ch of selectedChannels) if (next[ch.index]) next[ch.index] = { ...next[ch.index]!, acc: [], fft: null, trace: null };
      return next;
    });

  // ── Presets (localStorage) ──
  const [presets, setPresets] = useState<CapturePreset[]>(() => loadPresets());
  const [presetName, setPresetName] = useState("");

  const persistPresets = (list: CapturePreset[]) => {
    setPresets(list);
    try {
      localStorage.setItem(PRESET_KEY, JSON.stringify(list));
    } catch {
      /* localStorage may be unavailable; never freeze */
    }
  };

  const savePreset = () => {
    const name = presetName.trim() || `预设 ${presets.length + 1}`;
    const p: CapturePreset = {
      name,
      windowFn,
      output,
      logY,
      logX,
      dropDc,
      showPeaks,
      topN,
      fMinText,
      fMaxText,
      avgN,
      manualFsText,
      nPointsText,
    };
    const list = [...presets.filter((x) => x.name !== name), p];
    persistPresets(list);
    setPresetName("");
    toast(`已保存预设「${name}」`);
  };

  const applyPreset = (name: string) => {
    const p = presets.find((x) => x.name === name);
    if (!p) return;
    setWindowFn(p.windowFn);
    setOutput(p.output);
    setLogY(p.logY);
    setLogX(p.logX);
    setDropDc(p.dropDc);
    setShowPeaks(p.showPeaks);
    setTopN(p.topN);
    setFMinText(p.fMinText);
    setFMaxText(p.fMaxText);
    setAvgN(p.avgN);
    setManualFsText(p.manualFsText);
    setNPointsText(p.nPointsText);
    toast(`已加载预设「${name}」`);
  };

  const deletePreset = (name: string) => {
    persistPresets(presets.filter((x) => x.name !== name));
    toast(`已删除预设「${name}」`);
  };

  // ── Exports (spectrum: wide CSV with one column per channel) ──
  const onExportSpectrumCsv = () => {
    if (series.length === 0) return;
    // Align on the first series' freq axis; index-align the rest.
    const freqs = series[0]!.fft.freqs_hz ?? [];
    const col = series[0]!.fft.output === "power" ? "psd" : "magnitude";
    const header = ["freq_hz", ...series.map((s) => `${s.label.replace(/,/g, " ")}_${col}`)];
    const lines = [header.join(",")];
    const n = Math.min(...series.map((s) => s.fft.spectrum?.length ?? 0), freqs.length);
    for (let i = 0; i < n; i++) {
      const row = [freqs[i], ...series.map((s) => s.fft.spectrum?.[i] ?? "")];
      lines.push(row.join(","));
    }
    downloadText(`fft_${Date.now()}.csv`, lines.join("\n"));
    toast("频谱 CSV 已导出");
  };

  const onExportTraceCsv = () => {
    const withTrace = selectedChannels
      .map((ch) => chanStates[ch.index])
      .filter((st): st is ChannelState => !!st?.trace && st.trace.ys.length > 0);
    if (withTrace.length === 0) return;
    const base = withTrace[0]!.trace!;
    const header = ["t_s", ...withTrace.map((st) => st.channel.name.replace(/,/g, " "))];
    const lines = [header.join(",")];
    const n = Math.min(...withTrace.map((st) => st.trace!.ys.length));
    for (let i = 0; i < n; i++) {
      const row = [base.ts[i], ...withTrace.map((st) => st.trace!.ys[i])];
      lines.push(row.join(","));
    }
    downloadText(`trace_${Date.now()}.csv`, lines.join("\n"));
    toast("时域 trace CSV 已导出");
  };

  const onExportSpectrumPng = () => {
    const ok = exportCanvasPng(document.getElementById("fft-spectrum-chart"), `fft_${Date.now()}.png`);
    toast(ok ? "频谱 PNG 已导出" : "无可导出的频谱图", ok ? "ok" : "err");
  };
  const onExportTracePng = () => {
    const ok = exportCanvasPng(document.getElementById("fft-trace-chart"), `trace_${Date.now()}.png`);
    toast(ok ? "时域 PNG 已导出" : "无可导出的时域图", ok ? "ok" : "err");
  };

  const totalSamples = selectedChannels.reduce(
    (sum, ch) => sum + parseSamples(chanStates[ch.index]?.samplesText ?? "").length,
    0,
  );
  const anyReady = selectedChannels.some(
    (ch) => parseSamples(chanStates[ch.index]?.samplesText ?? "").length >= 4,
  );
  const canCompute = anyReady && !computing;

  // The trace chart shows the FIRST selected channel with a trace (overlay of
  // raw time series across channels is rarely meaningful — they differ in unit).
  const traceChannel = selectedChannels.find((ch) => chanStates[ch.index]?.trace?.ys.length);
  const traceState = traceChannel ? chanStates[traceChannel.index] : null;

  return (
    <div className="space-y-3">
      {node}
      <p className="text-xs text-mast-muted">
        <strong className="text-mast-text">信号捕获 + FFT</strong> —
        多通道选择 / 通道对比、N 次平均降噪、峰值读出表、频率轴缩放（线性/对数 X + 对数 Y）、实时连续模式、单位标度，并支持时域
        trace 与频谱的 CSV / PNG 导出与采集预设保存。
      </p>

      {/* Hardware-capture seam note: live acquisition (Osci1T 20 kHz / 轮询 fast-path)
          runs in the desktop core and streams large arrays that never cross the
          typed wire. The web seam exposes the channel/timebase metadata + the pure
          rfft compute; paste an already-captured trace to get the full spectrum. */}
      {/* 删掉「大样本数组永不过线 / 桌面内核 / 轮询 fast-path」——
          那是这条缝怎么实现的，不是用户要知道的。留下的两句都是他要做的动作：
          波形从哪来，以及没接硬件时怎么试。 */}
      <div className="rounded-md border border-mast-warn-border bg-mast-warn-bg px-3 py-2 text-xs text-mast-text">
        <strong className="text-mast-warn">需连接 Nanonis</strong>：本页不直接采集波形，
        对<strong className="text-mast-text">已采集的时域 trace</strong> 做单边 rfft 并按所选采样率标注。
        <span className="text-mast-muted">未接硬件时可点各通道的「示例」按钮填入演示数据。</span>
      </div>

      {signals.isError && <ErrorNote error={signals.error} />}

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-3">
        {/* ── LEFT: controls ── */}
        <div className="space-y-3 lg:col-span-1">
          {/* 1. Channel picker (multi-select) */}
          <Card className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="flex items-center gap-1.5 text-sm font-semibold text-mast-text">
                信号通道（可多选对比）
                {selectedChannels.length > 0 && (
                  <span className="rounded-full bg-mast-accent px-1.5 py-0.5 text-[10px] font-semibold leading-none text-mast-accent-ink">
                    已选 {selectedChannels.length}
                  </span>
                )}
              </span>
              <span className="shrink-0 text-xs text-mast-muted">
                {meta ? `共 ${meta.n_channels} 个` : ""}
                {meta?.degraded && <span className="ml-1 text-mast-warn">默认列表</span>}
              </span>
            </div>
            {signals.isLoading ? (
              <Spinner label="读取信号通道…" />
            ) : channels.length === 0 ? (
              <EmptyNote label="无可用信号通道。" />
            ) : (
              <>
                <TextField value={search} onChange={setSearch} placeholder="搜索通道名 / 索引…" />
                <div className="max-h-56 overflow-auto rounded-md border border-mast-border">
                  {filteredChannels.length === 0 ? (
                    <p className="p-2 text-xs text-mast-muted">无匹配通道。</p>
                  ) : (
                    filteredChannels.map((c) => {
                      const active = selectedIdx.includes(c.index);
                      return (
                        <button
                          key={c.index}
                          type="button"
                          aria-pressed={active}
                          onClick={() => toggleChannel(c.index)}
                          className={`flex w-full items-center justify-between gap-2 border-l-2 px-2.5 py-1.5 text-left text-xs transition-colors ${
                            active
                              ? "border-mast-accent bg-mast-accent/15 font-medium text-mast-text"
                              : "border-transparent text-mast-text hover:bg-mast-accent/5"
                          }`}
                        >
                          <span className="flex min-w-0 items-center gap-2">
                            <span
                              className={`flex h-4 w-4 shrink-0 items-center justify-center rounded border text-mast-accent-ink ${
                                active
                                  ? "border-mast-accent bg-mast-accent"
                                  : "border-mast-border bg-transparent"
                              }`}
                            >
                              {active && (
                                <svg viewBox="0 0 12 12" className="h-3 w-3" fill="none" stroke="currentColor" strokeWidth="2.2">
                                  <path d="M2.5 6.5L5 9l4.5-5" strokeLinecap="round" strokeLinejoin="round" />
                                </svg>
                              )}
                            </span>
                            <span className="truncate">
                              <span className="mr-1.5 font-mono text-mast-muted">{c.index}</span>
                              {c.name}
                              {c.is_current && (
                                <span className="ml-1.5 rounded bg-mast-auto-bg px-1 text-[10px] text-mast-auto">
                                  电流
                                </span>
                              )}
                            </span>
                          </span>
                          {c.unit && (
                            <span className={`font-mono ${active ? "text-mast-accent" : "text-mast-muted"}`}>{c.unit}</span>
                          )}
                        </button>
                      );
                    })
                  )}
                </div>
                <p className="text-xs text-mast-muted">
                  已选 <span className="text-mast-text">{selectedChannels.length}</span> 个：
                  {selectedChannels.length === 0 ? (
                    <span className="ml-1 text-mast-warn">请至少选一个通道</span>
                  ) : (
                    <span className="ml-1 text-mast-text">
                      {selectedChannels.map((c) => c.name).join("、")}
                    </span>
                  )}
                </p>
              </>
            )}
          </Card>

          {/* 2. Acquisition params */}
          <Card className="space-y-2">
            <span className="text-sm font-semibold text-mast-text">采集参数</span>
            {timebases.length > 0 ? (
              <Field label="Osci 时基 / 采样率">
                <SelectField<string>
                  value={timebaseIdx || String(timebases[0]?.index ?? "")}
                  onChange={setTimebaseIdx}
                  options={timebases.map((t) => ({
                    value: String(t.index),
                    label: `${t.fs_hz.toFixed(0)} Hz  (dt=${(t.dt_s * 1e6).toFixed(1)} µs)`,
                  }))}
                />
              </Field>
            ) : (
              <Field label="采样率 fs (Hz)" hint={meta?.osci_available ? undefined : "Osci1T 时基不可用，手动填写采样率"}>
                <TextField value={manualFsText} onChange={setManualFsText} mono />
              </Field>
            )}
            <Field label="采样点数 N" hint={fsHz > 0 ? `时长 ≈ ${(durationS * 1e3).toFixed(2)} ms · fs=${fsHz.toFixed(0)} Hz` : undefined}>
              <TextField value={nPointsText} onChange={setNPointsText} mono />
            </Field>
            <p className="text-xs text-mast-muted">
              Nyquist ≈ {fsHz > 0 ? nyquist.toFixed(0) : "—"} Hz · Δf ≈{" "}
              {fsHz > 0 && nPoints > 0 ? (fsHz / nPoints).toFixed(3) : "—"} Hz
            </p>
          </Card>

          {/* 3. FFT + axis options */}
          <Card className="space-y-2">
            <span className="text-sm font-semibold text-mast-text">FFT 选项</span>
            <Field label="窗函数">
              <SelectField<WindowFn> value={windowFn} onChange={setWindowFn} options={windowOptions} />
            </Field>
            <Field label="输出模式">
              <SelectField<OutputKind> value={output} onChange={setOutput} options={outputOptions} />
            </Field>
            <div className="grid grid-cols-2 gap-2">
              <label className="flex items-center justify-between text-sm">
                <span className="text-mast-muted">对数 Y</span>
                <Toggle checked={logY} onChange={setLogY} label="对数 Y" />
              </label>
              <label className="flex items-center justify-between text-sm">
                <span className="text-mast-muted">对数 X</span>
                <Toggle checked={logX} onChange={setLogX} label="对数 X" />
              </label>
              <label className="flex items-center justify-between text-sm">
                <span className="text-mast-muted">去 DC</span>
                <Toggle checked={dropDc} onChange={setDropDc} label="去 DC" />
              </label>
              <label className="flex items-center justify-between text-sm">
                <span className="text-mast-muted">峰值</span>
                <Toggle checked={showPeaks} onChange={setShowPeaks} label="峰值" />
              </label>
            </div>
            <Field label="峰值个数 (Top-N)">
              <SelectField<string>
                value={String(topN)}
                onChange={(v) => setTopN(Math.max(1, Number(v) || 5))}
                options={[3, 5, 8, 12].map((n) => ({ value: String(n), label: `${n}` }))}
              />
            </Field>
          </Card>

          {/* 4. Frequency-axis zoom */}
          <Card className="space-y-2">
            <span className="text-sm font-semibold text-mast-text">频率轴缩放</span>
            <div className="grid grid-cols-2 gap-2">
              <Field label="最小 f (Hz)" hint="留空=自动">
                <TextField value={fMinText} onChange={setFMinText} mono placeholder="auto" />
              </Field>
              <Field label="最大 f (Hz)" hint={`≤ Nyquist ${nyquist > 0 ? nyquist.toFixed(0) : "—"}`}>
                <TextField value={fMaxText} onChange={setFMaxText} mono placeholder="auto" />
              </Field>
            </div>
            {fMin != null && fMax != null && fMin >= fMax && (
              <span className="text-xs text-mast-warn">最小 f 应小于最大 f。</span>
            )}
          </Card>

          {/* 5. Averaging + live mode */}
          <Card className="space-y-2">
            <span className="text-sm font-semibold text-mast-text">平均 / 实时</span>
            <Field label="平均次数 N" hint="保留最近 N 次频谱做平均以降噪">
              <SelectField<string>
                value={String(avgN)}
                onChange={(v) => setAvgN(Math.max(1, Number(v) || 1))}
                options={[1, 2, 4, 8, 16, 32].map((n) => ({ value: String(n), label: `${n}×` }))}
              />
            </Field>
            <div className="flex items-center justify-between text-sm">
              <span className="text-mast-muted">已累积捕获</span>
              <span className="font-mono text-mast-text">
                {captureCount} / {avgN}
              </span>
            </div>
            <label className="flex items-center justify-between text-sm">
              <span className="text-mast-muted">实时连续模式</span>
              <Toggle checked={liveMode} onChange={setLiveMode} label="实时" />
            </label>
            {liveMode && (
              <Field label="刷新间隔 (s)">
                <TextField value={liveIntervalText} onChange={setLiveIntervalText} mono />
              </Field>
            )}
            <Button variant="ghost" onClick={resetAverage}>
              清空平均累积
            </Button>
          </Card>

          {/* 7. Presets */}
          <Card className="space-y-2">
            <span className="text-sm font-semibold text-mast-text">采集预设</span>
            <div className="flex gap-1.5">
              <div className="min-w-0 flex-1">
                <TextField value={presetName} onChange={setPresetName} placeholder="预设名称…" />
              </div>
              <Button variant="primary" onClick={savePreset}>
                保存
              </Button>
            </div>
            {presets.length === 0 ? (
              <p className="text-xs text-mast-muted">尚无预设。当前 FFT / 轴 / 平均设置可保存复用（存于本地浏览器）。</p>
            ) : (
              <div className="space-y-1">
                {presets.map((p) => (
                  <div key={p.name} className="flex items-center justify-between gap-2 rounded border border-mast-border px-2 py-1 text-xs">
                    <span className="truncate text-mast-text">{p.name}</span>
                    <span className="flex shrink-0 gap-1">
                      <Button variant="ghost" onClick={() => applyPreset(p.name)}>
                        加载
                      </Button>
                      <Button variant="ghost" onClick={() => deletePreset(p.name)}>
                        删除
                      </Button>
                    </span>
                  </div>
                ))}
              </div>
            )}
          </Card>
        </div>

        {/* ── RIGHT: trace inputs, plots, exports ── */}
        <div className="space-y-3 lg:col-span-2">
          {/* Per-channel trace input */}
          <Card className="space-y-2">
            <div className="flex items-center justify-between gap-2">
              <span className="text-sm font-semibold text-mast-text">时域 trace 数据</span>
              <span className="shrink-0 text-xs text-mast-muted">共识别 {totalSamples} 个样本</span>
            </div>
            <p className="text-xs text-mast-muted">
              粘贴一段时域采样值（空格 / 逗号 / 换行分隔），或点「示例」自动填入演示数据（正弦 + 噪声），即可在无硬件时看到 FFT 效果。
            </p>
            {selectedChannels.length === 0 ? (
              <EmptyNote label="请先在左侧选择至少一个信号通道。" />
            ) : (
              <div className="space-y-2">
                {/* One-click: fill demo data into every selected channel. */}
                <div className="flex flex-wrap items-center gap-2">
                  <Button variant="primary" onClick={fillDemoAll}>
                    填入示例数据（全部通道）
                  </Button>
                  <button
                    type="button"
                    onClick={() => setShowManualPaste((v) => !v)}
                    className="text-xs text-mast-accent hover:underline"
                  >
                    {showManualPaste ? "▾ 收起手动粘贴" : "▸ 手动粘贴（高级）"}
                  </button>
                </div>

                {/* Per-channel rows: a compact status line always; the raw paste
                    textarea only when the advanced area is expanded. */}
                <div className="space-y-2">
                  {selectedChannels.map((ch, i) => {
                    const st = chanStates[ch.index];
                    const cnt = parseSamples(st?.samplesText ?? "").length;
                    return (
                      <div key={ch.index} className="space-y-1 rounded-md border border-mast-border px-2 py-1.5">
                        <div className="flex items-center justify-between gap-2 text-xs">
                          <span className="min-w-0 truncate text-mast-text">
                            <span className="mr-1.5 font-mono text-mast-muted">{ch.index}</span>
                            {ch.name}
                            {ch.unit && <span className="ml-1 font-mono text-mast-muted">({ch.unit})</span>}
                          </span>
                          <span className="flex shrink-0 items-center gap-2">
                            <span className={cnt === 0 ? "text-mast-muted" : cnt < 4 ? "text-mast-warn" : "text-mast-auto"}>
                              {cnt === 0 ? "无数据" : `${cnt} 样本${cnt < 4 ? "（至少 4）" : ""}`}
                            </span>
                            <Button variant="ghost" onClick={() => fillDemoFor(ch.index, i)}>
                              示例
                            </Button>
                          </span>
                        </div>
                        {showManualPaste && (
                          <textarea
                            className="h-16 w-full rounded border border-mast-border bg-mast-bg px-2 py-1.5 font-mono text-xs text-mast-text outline-none focus:border-mast-accent"
                            placeholder="0.0, 0.12, 0.23, …（空格 / 逗号 / 换行分隔），或点「示例」自动填入"
                            value={st?.samplesText ?? ""}
                            onChange={(e) => setSamplesFor(ch.index, e.target.value)}
                          />
                        )}
                      </div>
                    );
                  })}
                </div>
              </div>
            )}
            <div className="flex flex-wrap items-center gap-2">
              <Button variant="primary" onClick={() => void runCompute()} disabled={!canCompute}>
                {/* No read-only hardware capture endpoint exists — this computes
                    the FFT of the PASTED/demo trace, so don't label it "采集". */}
                {computing ? "计算中…" : liveMode ? "立即计算一次" : "计算 FFT（粘贴/演示数据）"}
              </Button>
              {liveMode && (
                <RadioGroup<string>
                  value="on"
                  onChange={() => setLiveMode(false)}
                  options={[
                    { value: "on", label: "实时中" },
                    { value: "off", label: "停止" },
                  ]}
                />
              )}
              {selectedChannels.length > 0 && !anyReady && (
                <span className="text-xs text-mast-warn">每通道至少需要 4 个样本。</span>
              )}
            </div>
          </Card>

          {computeError != null && <ErrorNote error={computeError} />}

          {/* Time-domain trace (first channel that has one) */}
          {traceState?.trace && traceState.trace.ys.length > 0 && (
            <Card className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-mast-text">
                  时域 trace · {traceChannel?.name}
                </span>
                <div className="flex gap-1.5">
                  <Button variant="ghost" onClick={onExportTraceCsv}>
                    CSV
                  </Button>
                  <Button variant="ghost" onClick={onExportTracePng}>
                    PNG
                  </Button>
                </div>
              </div>
              <div id="fft-trace-chart">
                <TimeTraceChart
                  timestampsS={traceState.trace.ts}
                  samples={traceState.trace.ys}
                  unit={traceChannel?.unit ?? "A"}
                  channelName={traceChannel?.name ?? "signal"}
                />
              </div>
              {selectedChannels.length > 1 && (
                <p className="text-xs text-mast-muted">
                  多通道时域单位各异，仅显示首个通道波形；CSV 导出含全部通道。
                </p>
              )}
            </Card>
          )}

          {/* Spectrum (overlay of all channels) */}
          {series.length > 0 && (
            <Card className="space-y-2">
              <div className="flex items-center justify-between">
                <span className="text-sm font-semibold text-mast-text">
                  FFT 频谱{series.length > 1 ? `（${series.length} 通道对比）` : ""}
                </span>
                <div className="flex gap-1.5">
                  <Button variant="ghost" onClick={onExportSpectrumCsv}>
                    CSV
                  </Button>
                  <Button variant="ghost" onClick={onExportSpectrumPng}>
                    PNG
                  </Button>
                </div>
              </div>
              <div className="flex flex-wrap gap-x-6 gap-y-1 text-xs text-mast-muted">
                <span>N={series[0]!.fft.n_samples}</span>
                <span>fs={series[0]!.fft.fs_hz.toFixed(2)} Hz</span>
                <span>Nyquist={series[0]!.fft.nyquist_hz.toFixed(2)} Hz</span>
                <span>Δf={series[0]!.fft.df_hz.toFixed(3)} Hz</span>
                <span>窗={series[0]!.fft.window}</span>
                <span>输出={series[0]!.fft.output === "power" ? "PSD" : "|FFT|"}</span>
                {avgN > 1 && <span className="text-mast-accent">平均×{captureCount}/{avgN}</span>}
                {liveMode && <span className="text-mast-auto">● 实时</span>}
              </div>
              <div id="fft-spectrum-chart">
                <FftChart
                  series={series}
                  logY={logY}
                  logX={logX}
                  dropDc={dropDc}
                  showPeaks={showPeaks}
                  topN={topN}
                  fMin={fMin}
                  fMax={fMax}
                />
              </div>
            </Card>
          )}

          {series.length === 0 && selectedChannels.length > 0 && anyReady && !computing && (
            <EmptyNote label="点「采集 / 计算 FFT」生成频谱。" />
          )}
        </div>
      </div>
    </div>
  );
}
