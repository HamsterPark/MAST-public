import { Section } from "@/components/ui";
import { QaPanel } from "@/components/qa/QaPanel";

// 查询助手 — single-turn, read-only knowledge Q&A (parity rebuild of the old
// Gradio 查询助手 tab). All UI/state lives in QaPanel; see its header comment for
// why the backend is flagged missing rather than wired to the stateful chat graph.
export default function QaPage() {
  return (
    <Section title="查询助手">
      <QaPanel />
    </Section>
  );
}
