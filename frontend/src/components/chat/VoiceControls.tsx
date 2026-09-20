import { useRef } from "react";
import { useMutation } from "@tanstack/react-query";
import { api } from "@/api/client";

// Batch text→speech for the chat 朗读 (🔊) action on a finished reply.
//
// NOTE: the live voice INPUT path (record → transcribe) was replaced by the
// full-duplex VoiceDock (/ws/voice streaming ASR/TTS). Only this one-shot "read
// the latest reply aloud" helper remains here — it uses the batch
// /api/voice/synthesize endpoint (kept as the graceful-degradation TTS backend).

/** Play a TTS reply: POST /api/voice/synthesize then autoplay. Returns a
 *  speak() trigger usable from the chat pane (the 🔊 button on a reply). */
export function useVoiceOutput(onError?: (msg: string) => void) {
  const audioRef = useRef<HTMLAudioElement | null>(null);
  const synth = useMutation({
    mutationFn: async (text: string) => {
      const { data, error } = await api.POST("/api/voice/synthesize", {
        body: { text, voice: null },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      if (data?.ok && data.audio_b64) {
        const src = `data:${data.mime || "audio/wav"};base64,${data.audio_b64}`;
        const a = audioRef.current ?? new Audio();
        audioRef.current = a;
        a.src = src;
        void a.play().catch(() => {});
      } else {
        onError?.(data?.error || "语音合成失败");
      }
    },
    onError: (e) => onError?.(String((e as Error)?.message ?? e)),
  });
  return {
    speak: (text: string) => {
      const t = (text || "").trim();
      if (t) synth.mutate(t);
    },
    speaking: synth.isPending,
  };
}
