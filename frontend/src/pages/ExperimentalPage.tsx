import { Section } from "@/components/ui";
import { SubTabs } from "@/components/controls";
import { SignalCapturePanel } from "@/components/experimental/SignalCapturePanel";
import { MonitorPanel } from "@/components/vision/MonitorPanel";
import { MosaicPanel } from "@/components/experimental/MosaicPanel";
import { useStickyTab } from "@/hooks/useStickyTab";

// 实验性功能 — full-parity rebuild of the old Gradio 实验性功能 tab
// (gui/exp_capture.build_experimental_tab + gui/exp_puzzle). Three sub-functions
// dispatched by flat in-page SubTabs (mirrors the old Radio + Group dispatcher —
// NEVER nested gr.Tabs, which froze the page):
//   信号捕获 + FFT — GET /api/experimental/signals (channels + units + Osci
//                    timebases + FFT enums) + POST /api/experimental/fft. Fuller
//                    panel: 多通道对比叠加 · N 次平均降噪 + 计数器 · 峰值读出表 ·
//                    频率轴缩放 (线性/对数 X + 对数 Y) · 实时连续模式 · 单位标度 ·
//                    采集预设 (localStorage) + 时域 / 频谱 CSV·PNG 导出
//   长期监控      — POST monitor/start|stop, GET monitor/status (reused MonitorPanel)
//   图像拼接      — POST /api/experimental/mosaic (big-canvas overview PNG)
// Direct hardware tools (no chat agent in the loop); large sample/ndarray data
// is rendered/exported by the core and never crosses the wire.

type SubTab = "capture" | "monitor" | "mosaic";

// 一份数据同时喂 tab 条和 useStickyTab 的合法名单 —— 抄第二份就会漂开,
// 而漂开的症状是「某个子页记不住」,没人会去查存储键。
const SUB_TABS: { id: SubTab; label: string }[] = [
  { id: "capture", label: "信号捕获 + FFT" },
  { id: "monitor", label: "长期监控" },
  { id: "mosaic", label: "图像拼接" },
];

export default function ExperimentalPage() {
  const [tab, setTab] = useStickyTab<SubTab>(
    "experimental", SUB_TABS.map((t) => t.id), "capture");
  return (
    <Section title="实验性功能">
      <p className="mb-3 text-xs text-mast-muted">
        实验性功能 / Experimental — 直接调用仪器 skill 的图形化工具，不经过对话 agent。⚠️ 直连 Nanonis
        硬件，操作前请确认仪器状态。
      </p>
      <SubTabs<SubTab> value={tab} onChange={setTab} tabs={SUB_TABS} />
      {tab === "capture" && <SignalCapturePanel />}
      {tab === "monitor" && <MonitorPanel />}
      {tab === "mosaic" && <MosaicPanel />}
    </Section>
  );
}
