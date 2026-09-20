import { Badge } from "../ui";

// Read-only model-capability table (设置 → 模型 → model_info). Mirrors the old
// Gradio build_model_info_html: every MODEL_PRESETS row + derived limits.

const THINKING_BADGE: Record<string, { label: string; tone: string }> = {
  none: { label: "无", tone: "WARN" },
  fixed: { label: "固定", tone: "INFO" },
  tunable: { label: "可调", tone: "AUTO" },
};

export type ModelRow = {
  alias: string;
  provider: string;
  model_id: string;
  description: string;
  thinking_mode: string;
  output_limit: number;
  input_context: number;
  default_max_tokens: number;
  is_default: boolean;
};

export function ModelCapabilityTable({ models }: { models: ModelRow[] }) {
  return (
    <div className="overflow-x-auto rounded-lg border border-mast-border">
      <table className="w-full text-sm">
        <thead className="bg-mast-bg/60 text-mast-muted">
          <tr>
            <th className="px-3 py-2 text-left">别名</th>
            <th className="px-3 py-2 text-left">Provider</th>
            <th className="px-3 py-2 text-left">模型 ID</th>
            <th className="px-3 py-2 text-left">思考</th>
            <th className="px-3 py-2 text-right">输出上限</th>
            <th className="px-3 py-2 text-right">输入上下文</th>
            <th className="px-3 py-2 text-left">说明</th>
          </tr>
        </thead>
        <tbody>
          {models.map((m) => {
            const tb = THINKING_BADGE[m.thinking_mode] ?? { label: m.thinking_mode, tone: "default" };
            return (
              <tr key={m.alias} className="border-t border-mast-border align-top">
                <td className="px-3 py-2 font-medium">
                  {m.alias}
                  {m.is_default && (
                    <span className="ml-2 rounded bg-mast-accent/20 px-1.5 py-0.5 text-xs text-mast-accent">
                      默认
                    </span>
                  )}
                </td>
                <td className="px-3 py-2 text-mast-muted">{m.provider}</td>
                <td className="px-3 py-2 font-mono text-xs text-mast-muted">{m.model_id}</td>
                <td className="px-3 py-2">
                  <Badge tone={tb.tone}>{tb.label}</Badge>
                </td>
                <td className="px-3 py-2 text-right tabular-nums">{m.output_limit.toLocaleString()}</td>
                <td className="px-3 py-2 text-right tabular-nums">{m.input_context.toLocaleString()}</td>
                <td className="px-3 py-2 text-mast-muted">{m.description}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
