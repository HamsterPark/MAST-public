import { useEffect } from "react";
import { useQuery } from "@tanstack/react-query";
import clsx from "clsx";
import { api } from "@/api/client";
import { useVoiceSession, type VoiceMode, type VoiceStatus } from "./voiceSessionStore";

// VoiceDock — the full-duplex voice control surface (replaces VoiceControls).
//   · 🎙 connect/disconnect the /ws/voice channel
//   · mode: 按住说话 (PTT) · 唤醒词 (wake) · 全双工 (duplex)
//   · live ASR caption + streaming reply + execution narration
//   · barge-in / stop while speaking
// The heavy lifting lives in voiceSessionStore (module-scoped: survives tab
// switches). This component is a thin, never-freezing view over it.

const STATUS_LABEL: Record<VoiceStatus, string> = {
  offline: "未连接",
  idle: "就绪",
  listening: "聆听中…",
  transcribing: "识别中…",
  thinking: "思考中…",
  speaking: "播报中…",
  degraded: "已降级",
};

const STATUS_DOT: Record<VoiceStatus, string> = {
  offline: "bg-mast-muted",
  idle: "bg-mast-accent",
  listening: "bg-mast-accent animate-pulse",
  transcribing: "bg-mast-warn animate-pulse",
  thinking: "bg-mast-warn animate-pulse",
  speaking: "bg-mast-accent animate-pulse",
  degraded: "bg-mast-danger",
};

const MODES: { id: VoiceMode; label: string }[] = [
  { id: "ptt", label: "按住说话" },
  { id: "wake", label: "唤醒词" },
  { id: "duplex", label: "全双工" },
];

export function VoiceDock({ conversationId }: { conversationId: string | null }) {
  const s = useVoiceSession();

  // Seed the session defaults (mode / voice / narrate) from persisted settings —
  // only while not connected, so an active session is never disturbed. The
  // module-scoped session survives tab switches, so we never disconnect on unmount.
  const settings = useQuery({
    queryKey: ["voice", "settings"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/settings");
      if (error) throw error;
      return data;
    },
  });
  useEffect(() => {
    const d = settings.data;
    if (!d || useVoiceSession.getState().connected) return;
    const patch: Partial<{ mode: VoiceMode; voice: string; narrate: boolean }> = {};
    if (d.voice_mode === "ptt" || d.voice_mode === "wake" || d.voice_mode === "duplex") {
      patch.mode = d.voice_mode;
    }
    if (d.voice && d.voice !== "Off") patch.voice = d.voice;
    if (typeof d.voice_narrate === "boolean") patch.narrate = d.voice_narrate;
    if (Object.keys(patch).length) useVoiceSession.setState(patch);
  }, [settings.data]);

  const speaking = s.status === "speaking";
  const listening = s.status === "listening";

  return (
    <div className="rounded-lg border border-mast-border bg-mast-panel p-2.5">
      {/* header: connect toggle + status */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={() =>
            s.connected ? s.disconnect() : s.connect(conversationId, s.mode)
          }
          className={clsx(
            "shrink-0 rounded-md border px-3 py-1.5 text-sm font-medium",
            s.connected
              ? "border-mast-accent bg-mast-accent/10 text-mast-accent"
              : "border-mast-border text-mast-muted hover:border-mast-accent hover:text-mast-accent",
          )}
          title={s.connected ? "断开语音" : "连接语音"}
        >
          🎙 {s.connected ? "语音已开" : "语音"}
        </button>

        <span className="flex items-center gap-1.5 text-xs text-mast-muted">
          <span className={clsx("h-2 w-2 rounded-full", STATUS_DOT[s.status])} />
          {STATUS_LABEL[s.status]}
          {s.connected && !s.degraded && (
            <span className="ml-1 rounded bg-mast-bg px-1 text-[10px] text-mast-faint">
              {s.streaming ? "流式" : "批量"} · {s.voice}
            </span>
          )}
        </span>

        {/* mode selector */}
        <div className="ml-auto flex overflow-hidden rounded-md border border-mast-border">
          {MODES.map((m) => (
            <button
              key={m.id}
              type="button"
              disabled={!s.connected}
              onClick={() => s.setMode(m.id)}
              className={clsx(
                "px-2 py-1 text-xs",
                s.mode === m.id
                  ? "bg-mast-accent text-mast-accent-ink"
                  : "text-mast-muted hover:bg-mast-bg disabled:opacity-40",
              )}
            >
              {m.label}
            </button>
          ))}
        </div>
      </div>

      {/* main control row */}
      {s.connected && !s.degraded && (
        <div className="mt-2 flex items-center gap-2">
          {s.mode === "ptt" && (
            <button
              type="button"
              onPointerDown={() => void s.pttDown()}
              onPointerUp={() => s.pttUp()}
              onPointerLeave={() => listening && s.pttUp()}
              onPointerCancel={() => s.pttUp()}
              className={clsx(
                "flex-1 select-none rounded-md border px-3 py-2 text-sm font-medium",
                listening
                  ? "border-mast-danger-border bg-mast-danger-bg text-mast-danger"
                  : "border-mast-accent bg-mast-accent/10 text-mast-accent hover:bg-mast-accent/20",
              )}
            >
              {listening ? "● 松开结束" : "按住说话"}
            </button>
          )}

          {s.mode === "duplex" && (
            <div className="flex-1 rounded-md border border-mast-accent/40 bg-mast-accent/5 px-3 py-2 text-center text-sm text-mast-accent">
              {listening ? "🎧 持续聆听中" : "全双工待命"}
            </div>
          )}

          {s.mode === "wake" && (
            <div
              className={clsx(
                "flex-1 rounded-md border px-3 py-2 text-center text-sm",
                s.armed
                  ? "border-mast-accent bg-mast-accent/10 text-mast-accent"
                  : "border-mast-border bg-mast-bg text-mast-muted",
              )}
            >
              {s.armed ? "🟢 已唤醒，请讲指令…" : "🎧 待命 · 说「MAST」唤醒"}
            </div>
          )}

          {speaking && (
            <button
              type="button"
              onClick={() => s.bargeIn()}
              className="shrink-0 rounded-md border border-mast-danger-border bg-mast-danger-bg px-3 py-2 text-sm text-mast-danger"
              title="打断播报"
            >
              打断
            </button>
          )}
          {(s.status === "thinking" || s.status === "transcribing") && (
            <button
              type="button"
              onClick={() => s.stop()}
              className="shrink-0 rounded-md border border-mast-border px-3 py-2 text-sm text-mast-muted hover:border-mast-danger-border hover:text-mast-danger"
            >
              停止
            </button>
          )}
        </div>
      )}

      {/* live caption + reply + narration */}
      {(s.caption || s.reply || s.narrations.length > 0) && (
        <div className="mt-2 space-y-1.5">
          {s.caption && (
            <div
              className={clsx(
                "text-sm",
                s.captionFinal ? "text-mast-text" : "italic text-mast-muted",
              )}
            >
              <span className="mr-1 text-mast-faint">你：</span>
              {s.caption}
            </div>
          )}
          {s.reply && (
            <div className="text-sm text-mast-text">
              <span className="mr-1 text-mast-faint">MAST：</span>
              {s.reply}
            </div>
          )}
          {s.narrations.length > 0 && (
            <div className="flex flex-wrap gap-1">
              {s.narrations.map((n, i) => (
                <span
                  key={i}
                  className="rounded bg-mast-bg px-1.5 py-0.5 text-[11px] text-mast-muted"
                >
                  {n}
                </span>
              ))}
            </div>
          )}
        </div>
      )}

      {s.degraded && (
        <p className="mt-2 rounded border border-mast-warn-border bg-mast-warn-bg px-2 py-1 text-xs text-mast-warn">
          语音子系统已降级（{s.degradedReason || "缺少 DashScope 密钥或对话引擎"}）。可继续用文字输入。
        </p>
      )}
      {s.error && <p className="mt-2 text-xs text-mast-danger">{s.error}</p>}
    </div>
  );
}
