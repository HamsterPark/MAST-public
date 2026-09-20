// 高级 → 能力开关.
//
// Two lists, one gate. Both grant the agent capability it does not have by default,
// and both are written through the SAME PIN-checked path (POST /api/settings with
// admin_pin). They live here rather than in 设置 for the reason the PIN exists at all:
// getting the model or the voice wrong costs you quality; getting these wrong costs
// you a tip, a sample, or the meaning of a vetted script slot.
//
//   高级能力  — powers that step around a protection (script-file I/O, quitting
//               Nanonis, a scan config MAST cannot read, holding the main connection).
//   硬件模块  — hardware you may not own. MOVED here from 设置 (2026-07-13): switching
//               one ON hands the agent DANGEROUS skills — a laser, an RF amplifier,
//               probes that can collide into each other.
//
// Two things this UI must not get wrong:
//
//   1. WHOLE-REPLACE. SettingsStore REPLACES each dict wholesale rather than merging
//      it. A save must carry the state of EVERY entry plus the one being toggled, or
//      the others silently switch off. (The vision thresholds hit this first.)
//   2. PERSISTED ≠ LIVE. The agent's tool list is frozen at graph-build time. The
//      backend returns an honest `rebuild_note` saying what will actually happen —
//      show it verbatim instead of a bare "已保存".
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { useState } from "react";

import { api } from "../../api/client";
import { settingsWriteProblem, type SettingsPatch } from "@/lib/settingsWrite";

type Toast = (msg: string, kind?: "ok" | "err") => void;

function usePinStatus() {
  return useQuery({
    queryKey: ["admin", "pin-status"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/admin/pin-status");
      if (error) throw error;
      return data;
    },
  });
}

function useAdvancedCapabilities() {
  return useQuery({
    queryKey: ["settings", "advanced-capabilities"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/settings/advanced-capabilities");
      if (error) throw error;
      return data;
    },
  });
}

function useHardwareModules() {
  return useQuery({
    queryKey: ["settings", "hardware-modules"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/settings/hardware-modules");
      if (error) throw error;
      return data;
    },
  });
}

// ── 粗动驱动电压：本机耐压声明 ────────────────────────────────────────────────
//
// This is NOT a capability. The other two lists hand the AGENT a power; this one
// records a CLAIM ABOUT THE HARDWARE that nothing in the system can verify — the
// controller will happily output a voltage the piezo stack does not survive
// (some go to 400 V; some stacks die at 300), and no reading tells you which rig
// you are on. Get it wrong and the stack is gone: no retry, no rollback.
//
// It sits behind the same PIN for the same practical reason — a guard against a
// stray hand, not against the model, which has no path to this API at all — and
// inherits the fail-closed rule: with nothing declared, EVERY drive write and
// every coarse move is refused. That is deliberate. MAST will not pick a
// voltage for a machine it has never been told about.
function CoarseDriveDeclaration({
  save,
  busy,
}: {
  save: { mutate: (b: Record<string, unknown>) => void };
  busy: boolean;
}) {
  const settings = useQuery({
    queryKey: ["settings"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/settings");
      if (error) throw error;
      return data;
    },
  });
  const stored = (settings.data?.coarse_drive ?? {}) as Record<string, unknown>;
  const [amp, setAmp] = useState("");
  const [freq, setFreq] = useState("");
  const [by, setBy] = useState("");

  const declaredAmp =
    typeof stored.max_amplitude_v === "number" ? stored.max_amplitude_v : null;
  const declaredFreq =
    typeof stored.expected_frequency_hz === "number"
      ? stored.expected_frequency_hz
      : null;

  return (
    <section className="space-y-3">
      <div>
        <h3 className="text-sm font-semibold text-mast-text">粗动驱动电压（本机耐压声明）</h3>
        <p className="mt-1 text-sm text-mast-muted">
          控制器<b>能输出</b>的电压 ≠ 这台机器的压电叠堆<b>能承受</b>的电压。
          有些 Nanonis 控制器支持 400 V，而有的叠堆 300 V 就烧了——
          <b>没有任何读数会告诉你是哪一种</b>，设错了不可恢复。
          这个数只有你知道。
        </p>
        <p className="mt-1 text-sm text-mast-muted">
          <b>不填就等于全部拒绝</b>：未声明时任何设置驱动电压的尝试、以及任何粗动移动
          都会被拒。这是刻意的——MAST 不替一台它没被告知过的机器猜电压。
          每次粗动前还会读回当前驱动值与这里比对，<b>读不到也拒绝</b>
          （电压可能是别人在 Nanonis 界面里改的）。
        </p>
      </div>

      <div className="rounded border border-mast-border bg-mast-panel p-3 text-sm">
        <div className="mb-2 text-xs text-mast-muted">
          当前声明：
          {declaredAmp == null ? (
            <b className="text-mast-warn">未声明（粗动被禁止）</b>
          ) : (
            <b className="text-mast-text">
              {declaredAmp} V
              {declaredFreq != null ? ` @ ${declaredFreq} Hz` : ""}
              {typeof stored.declared_by === "string" && stored.declared_by
                ? `（${stored.declared_by}）`
                : ""}
            </b>
          )}
        </div>
        <div className="flex flex-wrap items-center gap-2">
          <input
            type="number"
            min={0}
            max={400}
            step={5}
            value={amp}
            onChange={(e) => setAmp(e.target.value)}
            placeholder="本机上限 V（≤400）"
            className="w-40 rounded border border-mast-border bg-transparent px-2 py-1"
          />
          <input
            type="number"
            min={0}
            max={20000}
            step={10}
            value={freq}
            onChange={(e) => setFreq(e.target.value)}
            placeholder="期望频率 Hz（可空）"
            className="w-44 rounded border border-mast-border bg-transparent px-2 py-1"
          />
          <input
            value={by}
            onChange={(e) => setBy(e.target.value)}
            placeholder="署名"
            className="w-28 rounded border border-mast-border bg-transparent px-2 py-1"
          />
          <button
            type="button"
            disabled={busy || !amp}
            onClick={() =>
              save.mutate({
                coarse_drive: {
                  max_amplitude_v: Number(amp),
                  ...(freq ? { expected_frequency_hz: Number(freq) } : {}),
                  ...(by ? { declared_by: by } : {}),
                },
              })
            }
            className="rounded border border-mast-border bg-mast-bg px-3 py-1 text-mast-text disabled:opacity-50"
          >
            声明
          </button>
        </div>
        <p className="mt-2 text-xs text-mast-muted">
          超过声明值的写入会被<b>直接拒绝，而不是降到上限</b>——
          悄悄降下来会让调用方以为自己设的是另一个值，那是「成功了但不是你要的」，
          比失败更危险。
        </p>
      </div>
    </section>
  );
}

export function CapabilityGates({ toast }: { toast: Toast }) {
  const pin = usePinStatus();
  const caps = useAdvancedCapabilities();
  const mods = useHardwareModules();
  const queryClient = useQueryClient();

  // The PIN is held in component state for the length of the session and never
  // stored. It goes on the wire once per save and the backend drops it — it is
  // compared against a SHA-256 in config/admin_pin.txt and never persisted.
  const [enteredPin, setEnteredPin] = useState("");

  const save = useMutation({
    mutationFn: async (body: SettingsPatch) => {
      const { data, error } = await api.POST("/api/settings", {
        body: { ...body, admin_pin: enteredPin },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      // PIN 与 degraded 这两条本来就查了；缺的是 `ok:false` 的第三种拒绝
      // （结构非法 / 取值越界 —— 端点是共用的）。三条现在由同一个判据回答。
      const problem = settingsWriteProblem(data);
      if (problem) {
        toast(problem, "err");
        return;
      }
      const note = (data as { rebuild_note?: string })?.rebuild_note;
      toast(note ? `已保存${note}` : "已保存", "ok");
      queryClient.invalidateQueries({ queryKey: ["settings"] });
    },
    onError: (e) => toast(`保存失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  if (pin.isLoading) return <p className="text-sm text-mast-muted">读取中…</p>;

  if (!pin.data?.pin_is_set) {
    return <SetPinPrompt toast={toast} onDone={() => pin.refetch()} />;
  }

  const capList = caps.data?.capabilities ?? [];
  const modList = mods.data?.modules ?? [];

  const toggleCapability = (id: string, enabled: boolean) => {
    const next: Record<string, boolean> = {};
    for (const c of capList) next[c.id] = c.id === id ? enabled : c.enabled;
    save.mutate({ advanced_capabilities: next });
  };

  const toggleModule = (id: string, enabled: boolean) => {
    const next: Record<string, boolean> = {};
    for (const m of modList) next[m.id] = m.id === id ? enabled : m.enabled;
    save.mutate({ hardware_modules: next });
  };

  const busy = save.isPending;

  return (
    <div className="space-y-6">
      <div className="rounded border border-mast-border bg-mast-panel p-3">
        <label className="flex flex-wrap items-center gap-2 text-sm">
          <span className="text-mast-text">管理 PIN</span>
          <input
            type="password"
            value={enteredPin}
            onChange={(e) => setEnteredPin(e.target.value)}
            placeholder="改动下面任一开关都需要"
            className="min-w-[14rem] flex-1 rounded border border-mast-border bg-transparent px-2 py-1 text-mast-text"
          />
        </label>
        <p className="mt-1 text-xs text-mast-muted">
          PIN 只在保存时随请求发一次，<b>不会被存下来</b>（后端拿它跟 config/admin_pin.txt
          里的 SHA-256 比对后即丢弃）。忘记了就删掉那个文件重设——这是台仪器，不是银行，
          把用户锁在自己的显微镜外面是更糟的失败。
        </p>
      </div>

      {/* ── 高级能力 ───────────────────────────────────────────────────────── */}
      <section className="space-y-3">
        <div>
          <h3 className="text-sm font-semibold text-mast-text">高级能力</h3>
          <p className="mt-1 text-sm text-mast-muted">
            这些不是「你没有这个硬件」，而是「这个能力能<b>绕过某道保护</b>」。
            全部默认关闭；关闭时它们的 skill <b>根本不进 agent 的工具表</b>，
            调不到不存在的东西。
          </p>
          {(caps.data?.gated_skill_count ?? 0) > 0 && (
            <p className="mt-1 text-xs text-mast-muted">
              当前有 <b className="text-mast-text">{caps.data?.gated_skill_count}</b> 个 skill
              因高级能力关闭而未挂载。
            </p>
          )}
        </div>
        {capList.map((c) => (
          <div
            key={c.id}
            className={`rounded border px-3 py-2 ${
              c.enabled
                ? "border-red-500/50 bg-red-500/5"
                : "border-mast-border bg-mast-panel"
            }`}
          >
            <label className="flex cursor-pointer items-start gap-3">
              <input
                type="checkbox"
                checked={c.enabled}
                disabled={busy}
                onChange={(e) => toggleCapability(c.id, e.target.checked)}
                className="mt-1 shrink-0 accent-mast-accent disabled:opacity-50"
              />
              <span className="min-w-0 flex-1">
                <span className="flex flex-wrap items-baseline gap-x-2">
                  <b className="text-sm text-mast-text">{c.name}</b>
                  <span className="text-xs text-mast-muted">{c.skill_count} 个 skill</span>
                  {c.enabled && (
                    <span className="rounded bg-red-500/15 px-1.5 py-0.5 text-[11px] text-red-400">
                      已授予
                    </span>
                  )}
                </span>
                <span className="mt-1 block text-xs text-red-400/90">⚠ {c.risk}</span>
                <span className="mt-1 block text-xs text-mast-muted">{c.description}</span>
              </span>
            </label>
          </div>
        ))}
      </section>

      {/* ── 硬件模块 ───────────────────────────────────────────────────────── */}
      <section className="space-y-3">
        <div>
          <h3 className="text-sm font-semibold text-mast-text">
            硬件模块
            <span className="ml-2 text-xs font-normal text-mast-muted">
              （{mods.data?.enabled_count ?? 0} / {modList.length} 已启用）
            </span>
          </h3>
          <p className="mt-1 text-sm text-mast-muted">
            Nanonis 的选装模块——授权都有，硬件不一定装。默认全关。
            <b>放在这里而不是设置页</b>：打开一个模块就等于把 DANGEROUS skill 交给 agent
            （激光、射频功放、会互撞的多探针）。
            关闭时它们的 skill 不进工具表，也不会白白挤占工具表（仪器控制已有约 300 个工具，
            工具越多路由越容易选错）。
          </p>
          {(mods.data?.gated_skill_count ?? 0) > 0 && (
            <p className="mt-1 text-xs text-mast-muted">
              当前有 <b className="text-mast-text">{mods.data?.gated_skill_count}</b> 个 skill
              因硬件模块关闭而未挂载。
            </p>
          )}
        </div>
        {modList.map((m) => (
          <div
            key={m.id}
            className={`rounded border px-3 py-2 ${
              m.enabled
                ? "border-mast-accent/50 bg-mast-accent/5"
                : "border-mast-border bg-mast-panel"
            }`}
          >
            <label className="flex cursor-pointer items-start gap-3">
              <input
                type="checkbox"
                checked={m.enabled}
                disabled={busy}
                onChange={(e) => toggleModule(m.id, e.target.checked)}
                className="mt-1 shrink-0 accent-mast-accent disabled:opacity-50"
              />
              <span className="min-w-0 flex-1">
                <span className="flex flex-wrap items-baseline gap-x-2">
                  <b className="text-sm text-mast-text">{m.name}</b>
                  <span className="text-xs text-mast-muted">{m.skill_count} 个 skill</span>
                </span>
                <span className="mt-0.5 block text-xs text-mast-muted">{m.description}</span>
                <span className="mt-0.5 block text-xs text-mast-muted">
                  需要硬件：<span className="text-mast-text">{m.hardware}</span>
                </span>
              </span>
            </label>
          </div>
        ))}
      </section>

      <CoarseDriveDeclaration save={save} busy={busy} />

      <p className="text-xs text-mast-muted">
        改动<b>立即持久化</b>，但 agent 的工具表是在建图时冻结的——保存后会在后台重建
        （数秒生效）。<b>任务运行中不会重建</b>（否则会在运行中途换掉图），需等任务结束或重启。
      </p>
      {busy && <p className="text-xs text-mast-muted">保存中…</p>}
    </div>
  );
}

// No PIN on this machine yet. The capability toggles are refused outright until one
// exists — fail-closed, and deliberately: handing an agent the ability to overwrite a
// vetted script slot should not happen on a machine where nobody has yet claimed to be
// the administrator.
function SetPinPrompt({ toast, onDone }: { toast: Toast; onDone: () => void }) {
  const [pin, setPin] = useState("");
  const [again, setAgain] = useState("");

  const setPinMut = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/admin/set-pin", {
        body: { new_pin: pin },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      if (data?.ok) {
        toast("管理 PIN 已设置", "ok");
        onDone();
      } else {
        toast(data?.message || "设置失败", "err");
      }
    },
    onError: (e) => toast(`设置失败：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const mismatch = again.length > 0 && pin !== again;
  const tooShort = pin.length > 0 && pin.length < 4;

  return (
    <div className="max-w-lg space-y-3">
      <div className="rounded border border-mast-border bg-mast-panel p-3">
        <h3 className="text-sm font-semibold text-mast-text">先设置一个管理 PIN</h3>
        <p className="mt-2 text-sm text-mast-muted">
          这一页的开关会把 agent 默认没有的能力交给它——覆盖已审的脚本槽位、退出 Nanonis、
          打开激光和射频。<b>在还没有人认领管理员的机器上，这些开关一律拒绝改动</b>
          （不是「随便改」，是<b>拒绝</b>）。
        </p>
        <p className="mt-2 text-xs text-mast-muted">
          只有 SHA-256 会写到磁盘（config/admin_pin.txt），原始 PIN 不落盘。忘了就删掉那个
          文件重设。
        </p>
      </div>
      <input
        type="password"
        value={pin}
        onChange={(e) => setPin(e.target.value)}
        placeholder="新 PIN（至少 4 位）"
        className="w-full rounded border border-mast-border bg-transparent px-2 py-1 text-mast-text"
      />
      <input
        type="password"
        value={again}
        onChange={(e) => setAgain(e.target.value)}
        placeholder="再输一次"
        className="w-full rounded border border-mast-border bg-transparent px-2 py-1 text-mast-text"
      />
      {tooShort && <p className="text-xs text-red-400">PIN 至少 4 位。</p>}
      {mismatch && <p className="text-xs text-red-400">两次输入不一致。</p>}
      <button
        type="button"
        disabled={setPinMut.isPending || pin.length < 4 || pin !== again}
        onClick={() => setPinMut.mutate()}
        className="rounded border border-mast-border bg-mast-panel px-3 py-1 text-sm text-mast-text hover:border-mast-accent disabled:opacity-40"
      >
        设置 PIN
      </button>
    </div>
  );
}
