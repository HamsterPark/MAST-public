import { Link } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";

// Model + thinking quick-display — a read-only strip showing the global default
// model and the IC agent's model + thinking level, with a link to the settings
// page where they are actually changed. Mirrors the old Gradio 全局/IC 模型 +
// thinking 快显 (which in the old UI were dropdowns; here they are read-only per
// the brief — "只读链接到设置").

const AGENT_ID = "instrument_control";

const THINKING_LABEL: Record<string, string> = {
  none: "无",
  fixed: "固定",
  off: "关闭",
  low: "低",
  medium: "中",
  high: "高",
  max: "最高",
};

function Pill({ k, v, mono }: { k: string; v: string; mono?: boolean }) {
  return (
    <span className="inline-flex items-center gap-1 rounded-md border border-mast-border bg-mast-panel/60 px-2 py-1 text-xs">
      <span className="text-mast-muted">{k}</span>
      <span className={mono ? "font-mono text-mast-text" : "text-mast-text"}>{v}</span>
    </span>
  );
}

export function ModelStrip() {
  const models = useQuery({
    queryKey: ["chat", "config-models"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/config/models");
      if (error) throw error;
      return data;
    },
    staleTime: 60_000,
  });

  const agentModels = useQuery({
    queryKey: ["chat", "agents-models"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/agents/models");
      if (error) throw error;
      return data;
    },
    refetchInterval: 15_000,
  });

  const globalAlias = models.data?.default_alias ?? "—";
  const ic = (agentModels.data?.agents ?? []).find((a) => a.agent_id === AGENT_ID);
  const icModel = ic?.model ?? globalAlias;
  const icThinking = ic?.thinking ? (THINKING_LABEL[ic.thinking] ?? ic.thinking) : "—";

  return (
    <div className="flex flex-wrap items-center gap-2">
      <Pill k="全局模型" v={globalAlias} mono />
      <Pill k="IC 模型" v={icModel} mono />
      <Pill k="思考" v={icThinking} />
      {agentModels.data?.degraded && (
        <span className="text-xs text-mast-warn">（模型信息降级）</span>
      )}
      <Link
        to="/settings/general"
        className="ml-auto rounded-md border border-mast-border px-2 py-1 text-xs text-mast-accent hover:border-mast-accent"
      >
        在设置中修改 →
      </Link>
    </div>
  );
}
