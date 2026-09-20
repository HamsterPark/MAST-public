import { create } from "zustand";
import { MicCapture } from "@/lib/audio/mic";
import { PcmPlayer } from "@/lib/audio/player";
import { EnergyVad } from "@/lib/voice/vad";

// ════════════════════════════════════════════════════════════════════════════
// voiceSessionStore — the full-duplex voice session, hoisted OUT of React into a
// module-scoped zustand store (same rationale as runTaskStore): switching tabs
// must NOT drop the mic stream / WS / playback. The WebSocket + AudioContext +
// VAD live as module-level handles; the store holds only serializable UI state.
//
// Transport: browser ⇄ (WS /ws/voice, raw PCM16) ⇄ FastAPI ⇄ DashScope. Mic goes
// up @16 kHz; TTS comes down @24 kHz.
//
// Modes:
//   · ptt    — hold to talk; release commits the utterance.
//   · wake   — (P4) wake word arms one PTT-like turn.
//   · duplex — continuous listening; a local energy VAD auto-commits on
//              speech→silence and barge-ins (stops playback) if the user speaks
//              while the assistant is talking. Mic audio is only streamed to the
//              server while VAD says speech is active, so idle/echo never lands in
//              the utterance buffer.
// Everything degrades to a note, never hangs.
// ════════════════════════════════════════════════════════════════════════════

export type VoiceStatus =
  | "offline"
  | "idle"
  | "listening"
  | "transcribing"
  | "thinking"
  | "speaking"
  | "degraded";

export type VoiceMode = "ptt" | "wake" | "duplex";

// non-serializable session handles (survive component unmount)
let _ws: WebSocket | null = null;
let _mic: MicCapture | null = null;
let _player: PcmPlayer | null = null;
let _vad: EnergyVad | null = null;
let _convId: string | null = null;
let _sendAudio = false; // gate mic→ws: true only while an utterance is active

function wsUrl(): string {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/voice`;
}

function send(obj: unknown): void {
  if (_ws && _ws.readyState === WebSocket.OPEN) _ws.send(JSON.stringify(obj));
}

interface VoiceState {
  status: VoiceStatus;
  mode: VoiceMode;
  connected: boolean;
  degraded: boolean;
  degradedReason: string;
  streaming: boolean; // realtime vs batch fallback (from hello_ack)
  voice: string;
  caption: string; // live ASR (partial → final)
  captionFinal: boolean;
  reply: string; // accumulating assistant reply text
  narrations: string[]; // execution narration ("扫描完成" …)
  armed: boolean; // wake mode: heard the wake word, awaiting the command
  narrate: boolean; // speak agent tool executions (seeded from settings)
  error: string | null;

  connect: (conversationId?: string | null, mode?: VoiceMode) => void;
  disconnect: () => void;
  setMode: (m: VoiceMode) => void;
  pttDown: () => Promise<void>;
  pttUp: () => void;
  sendText: (t: string) => void;
  bargeIn: () => void;
  stop: () => void;
}

export const useVoiceSession = create<VoiceState>((set, get) => {
  // ── mic lifecycle ────────────────────────────────────────────────────────
  const startMic = async (useVad: boolean): Promise<boolean> => {
    if (_mic) return true;
    _mic = new MicCapture(
      (buf) => {
        if (_sendAudio && _ws && _ws.readyState === WebSocket.OPEN) _ws.send(buf);
      },
      16000,
      useVad ? (rms) => _vad?.push(rms) : undefined,
    );
    try {
      await _mic.start();
      return true;
    } catch (e) {
      set({ error: `无法访问麦克风：${String((e as Error)?.message ?? e)}` });
      _mic = null;
      return false;
    }
  };
  const stopMic = async (): Promise<void> => {
    await _mic?.stop();
    _mic = null;
  };

  // ── duplex VAD: auto endpoint + barge-in ─────────────────────────────────
  const setupVad = () => {
    if (!_vad) _vad = new EnergyVad({ threshold: 0.02, silenceMs: 700 });
    _vad.reset();
    _vad.onSpeechStart = () => {
      const st = get().status;
      if (st === "speaking" || st === "thinking" || st === "transcribing") {
        _player?.clear();
        send({ type: "barge_in" });
      }
      _sendAudio = true;
      set({ caption: "", captionFinal: false, reply: "", narrations: [], status: "listening" });
    };
    _vad.onSpeechEnd = () => {
      if (_sendAudio) {
        _sendAudio = false;
        send({ type: "commit" });
      }
    };
  };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const handleFrame = (f: any) => {
    switch (f?.type) {
      case "hello_ack":
        set({
          mode: f.mode ?? get().mode,
          voice: f.voice ?? get().voice,
          streaming: !!f.streaming,
          degraded: !!f.degraded,
          status: f.degraded ? "degraded" : "idle",
        });
        break;
      case "state":
        set({ status: (f.state as VoiceStatus) ?? "idle" });
        break;
      case "asr_partial":
        set({ caption: String(f.text ?? ""), captionFinal: false });
        break;
      case "asr_final":
        set({ caption: String(f.text ?? ""), captionFinal: true });
        break;
      case "reply_delta":
        set((s) => ({ reply: s.reply + String(f.text ?? "") }));
        break;
      case "narration":
        set((s) => ({ narrations: [...s.narrations, String(f.text ?? "")].slice(-8) }));
        break;
      case "interrupt":
        set((s) => ({
          narrations: [...s.narrations, `⚠ ${String(f.spoken ?? "需要人工确认")}`].slice(-8),
        }));
        break;
      case "tts_begin":
        if (!_player) _player = new PcmPlayer(f.sample_rate ?? 24000);
        break;
      case "wake":
        set({ armed: !!f.armed });
        break;
      case "tts_done":
        break;
      case "degraded":
        set({ degraded: true, degradedReason: String(f.reason ?? ""), status: "degraded" });
        break;
      case "error":
        set({ error: String(f.message ?? "语音出错") });
        break;
      default:
        break;
    }
  };

  return {
    status: "offline",
    mode: "ptt",
    connected: false,
    degraded: false,
    degradedReason: "",
    streaming: true,
    voice: "Cherry",
    caption: "",
    captionFinal: false,
    reply: "",
    narrations: [],
    armed: false,
    narrate: true,
    error: null,

    connect: (conversationId, mode) => {
      if (_ws && (_ws.readyState === WebSocket.OPEN || _ws.readyState === WebSocket.CONNECTING)) {
        return;
      }
      _convId = conversationId ?? _convId;
      if (!_player) _player = new PcmPlayer(24000);
      _player.prime(); // resume AudioContext inside the connect click (user gesture)

      const ws = new WebSocket(wsUrl());
      ws.binaryType = "arraybuffer";
      _ws = ws;
      set({ error: null });

      ws.onopen = () => {
        set({ connected: true });
        send({
          type: "hello",
          mode: mode ?? get().mode,
          conversation_id: _convId,
          voice: get().voice,
          narrate: get().narrate,
        });
      };
      ws.onmessage = (ev: MessageEvent) => {
        if (typeof ev.data !== "string") {
          _player?.enqueue(ev.data as ArrayBuffer);
          return;
        }
        try {
          handleFrame(JSON.parse(ev.data));
        } catch {
          /* ignore malformed frame */
        }
      };
      ws.onclose = () => {
        set({ connected: false, status: "offline" });
        _ws = null;
      };
      ws.onerror = () => {
        set({ error: "语音连接出错" });
      };
    },

    disconnect: () => {
      send({ type: "bye" });
      _sendAudio = false;
      void stopMic();
      _vad?.reset();
      void _player?.close();
      _player = null;
      try {
        _ws?.close();
      } catch {
        /* ignore */
      }
      _ws = null;
      set({
        connected: false, status: "offline", caption: "", reply: "",
        narrations: [], armed: false,
      });
    },

    setMode: (m) => {
      set({ mode: m, armed: false });
      send({ type: "set_mode", mode: m });
      if (m === "duplex" || m === "wake") {
        setupVad(); // continuous listening; VAD auto-commits + barges in
        void startMic(true);
      } else {
        _sendAudio = false;
        void stopMic();
      }
    },

    pttDown: async () => {
      if (!get().connected || get().mode !== "ptt") return; // wake/duplex are hands-free
      set({ reply: "", narrations: [], caption: "", captionFinal: false, error: null });
      const ok = await startMic(false);
      if (ok) {
        _sendAudio = true;
        set({ status: "listening" });
      }
    },

    pttUp: () => {
      if (get().mode !== "ptt") return; // wake/duplex are VAD-driven
      _sendAudio = false;
      void stopMic();
      send({ type: "commit" });
    },

    sendText: (t) => {
      const text = t.trim();
      if (!text) return;
      set({ reply: "", narrations: [] });
      send({ type: "text", content: text });
    },

    bargeIn: () => {
      _player?.clear();
      send({ type: "barge_in" });
    },

    stop: () => {
      _player?.clear();
      _sendAudio = false;
      if (get().mode === "ptt") void stopMic(); // keep listening in wake/duplex
      send({ type: "stop" });
    },
  };
});
