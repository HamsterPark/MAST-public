# DashScope 实时语音（realtime ASR + TTS）— 协议参考与参数审计

MAST 语音层用 DashScope **realtime**（流式）ASR/TTS，两条独立 WebSocket = "分离式流式"。
实现见 `MASTv2/mast/voice/realtime/dashscope_rt.py`；上层编排 `MASTv2/mast/voice/session.py`。

## 0. 结论（已实测验证到握手层）

- **端点（两者共用）**：`wss://dashscope.aliyuncs.com/api-ws/v1/realtime?model=<model>`
- **鉴权**：WS 握手 HTTP header `Authorization: Bearer <DASHSCOPE_API_KEY>`（不是 query 参数）。
- **模型**：ASR = `qwen3-asr-flash-realtime`，TTS = `qwen3-tts-flash-realtime`（音色与批量版同：Cherry/Ethan/Chelsie/Serena/Dylan…）。
- **协议族**：OpenAI-Realtime 风格（JSON 控制帧 + base64 音频）。**注意**别和 run-task 族 `/api-ws/v1/inference`（Paraformer/CosyVoice，二进制音频）混用——事件名与音频编码完全不同。



### ⚠️ 两处真实协议修正（实测，与初版调研不符）
1. **TTS `response_format` 枚举是 `{mp3, wav, pcm, opus}`**，不是 `PCM_24000HZ_MONO_16BIT`（后者报 `invalid_value`）。用 `response_format:"pcm"` + 独立 `sample_rate:24000`。
2. **ASR 流式中间结果在 `stash` 字段**（`text` 恒为空），最终结果才在 `completed` 事件的 `transcript`。

### 网络注意
网络链路可能延迟握手；`open()` 的 `connect_timeout` 默认 20 s，应按实际网络条件调整。

## 1. 流式 ASR（qwen3-asr-flash-realtime）

事件序列（server_vad 自动断句）：
```jsonc
// C→S 配置
{ "type":"session.update", "session":{
    "modalities":["text"], "input_audio_format":"pcm", "sample_rate":16000,
    "input_audio_transcription":{ "language":"zh" },           // 省略=自动检测
    "turn_detection":{ "type":"server_vad", "threshold":0.2, "silence_duration_ms":800 } } }
// C→S 循环推音频（~100ms/片 = 3200B pcm16@16k）
{ "type":"input_audio_buffer.append", "audio":"<base64 pcm16>" }
// S→C
{ "type":"input_audio_buffer.speech_started" }
// partial：累积文本在 stash（text 恒空），language/emotion 常带
{ "type":"conversation.item.input_audio_transcription.text", "text":"", "stash":"今天天气", "language":"zh", "emotion":"neutral" }
{ "type":"input_audio_buffer.committed" }
{ "type":"conversation.item.input_audio_transcription.completed", "transcript":"今天天气不错" }          // final
```
手动断句（无 VAD）：`turn_detection:null` + `{ "type":"input_audio_buffer.commit" }`。
音频要求：**16 kHz、PCM16、单声道**。

## 2. 流式 TTS（qwen3-tts-flash-realtime）

```jsonc
// C→S  （response_format 枚举 {mp3,wav,pcm,opus}；PCM 用 "pcm" + sample_rate）
{ "type":"session.update", "session":{
    "voice":"Cherry", "response_format":"pcm", "sample_rate":24000, "mode":"server_commit" } }
{ "type":"input_text_buffer.append", "text":"扫描已" }   // 可多次（LLM 逐句喂）
{ "type":"input_text_buffer.append", "text":"完成，发现台阶。" }
{ "type":"input_text_buffer.commit" }
{ "type":"session.finish" }
// S→C
{ "type":"session.created", "session":{ "object":"realtime.session", "mode":"server_commit", … } }  // 实测已确认
{ "type":"response.audio.delta", "delta":"<base64 pcm24k>" }   // 多次
{ "type":"response.done" }  /  { "type":"session.finished" }
```
输出：**24 kHz、PCM16、单声道**。barge-in：`{ "type":"response.cancel" }` 后关连接。

## 3. 浏览器音频（前端管线，见 `frontend/src/lib/audio/`）
- **弃用 MediaRecorder/webm**：realtime 要裸 PCM。用 AudioWorklet 采集（48k Float32 → 降采样 16k → Int16）。
- 播放：24k PCM16 → 调度式 `AudioBufferSourceNode`（声明 24k 缓冲，Web Audio 自动重采样到硬件率），barge-in 时 `stop()` 全部。
- **裸 PCM 走 MAST 自有 WS `/ws/voice`**（key 只在 FastAPI，绝不下发浏览器）；FastAPI 侧再 base64 转发给 DashScope。

## 4. VAD / 唤醒词（客户端，离线）
- 现用**能量 VAD**（`frontend/src/lib/voice/vad.ts`，RMS+挂起）驱动全双工断句与 barge-in——依赖 `echoCancellation` 抑制 TTS 回授。无依赖、离线友好。
- 升级路径：Silero（`@ricky0123/vad-web`，WASM+ONNX）——精度更高但需打包 wasm/onnx 资产（LAN 离线场景待处理），接口已按可替换设计。
- 唤醒词"MAST"：**服务端关键词门控**（VAD 分段→批量转写→匹配 `_WAKE_WORDS`），无新依赖/无 key。可选升级 Porcupine（需 Picovoice 付费 key，纯前端）。

## 5. 备选协议族（未采用，备查）
run-task 族 `/api-ws/v1/inference`：ASR=Paraformer-realtime-v2（仅北京地域，二进制 PCM 帧，`is_sentence_end`）；TTS=CosyVoice-v2（音色另一套 longxiaochun 等，支持 SSML）。融合式 omni-realtime（`qwen3-omni-flash-realtime`）一条连接音入音出，但挡在"每步安全门控 + 自有多智能体"前，故不用。

## 官方文档
- 实时 ASR：https://www.alibabacloud.com/help/en/model-studio/qwen-real-time-speech-recognition
- 实时 TTS：https://help.aliyun.com/en/model-studio/qwen-tts-realtime
- 音色表：https://www.alibabacloud.com/help/en/model-studio/qwen-tts-voice-list
