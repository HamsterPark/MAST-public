import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import type { components } from "../../api/schema";
import { Card, DegradedNote, EmptyNote, ErrorNote, Section, Spinner } from "../ui";
import { Field, JsonBlock, StatusPill, fmtTime } from "./shared";

type ActionDetail = components["schemas"]["ActionDetail"];

function useActionDetail(id: string) {
  return useQuery({
    queryKey: ["records", "action", id],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/records/actions/{action_id}", {
        params: { path: { action_id: id } },
      });
      if (error) throw error;
      return data;
    },
  });
}

export function ActionDetailPane({
  actionId,
  onBack,
}: {
  actionId: string;
  onBack: () => void;
}) {
  const q = useActionDetail(actionId);

  return (
    <div className="space-y-6">
      <button onClick={onBack} className="text-sm text-mast-accent hover:underline">
        ← 返回
      </button>

      {q.isPending && <Spinner />}
      {q.isError && <ErrorNote error={q.error} />}
      {q.data && <Body data={q.data} />}
    </div>
  );
}

function Body({ data }: { data: ActionDetail }) {
  if (data.degraded) return <DegradedNote what="动作详情" />;
  if (data.found === false) return <EmptyNote label={`未找到动作 ${data.id}`} />;

  return (
    <Section title={`动作 ${data.action_type || data.skill_name || data.id}`}>
      <Card className="space-y-1">
        <Field label="ID">{data.id}</Field>
        <Field label="实验">{data.experiment_id ?? "—"}</Field>
        <Field label="样品">{data.sample_id ?? "—"}</Field>
        <Field label="父动作">{data.parent_action_id ?? "—"}</Field>
        <Field label="Agent">{data.agent_id ?? "—"}</Field>
        <Field label="动作类型">{data.action_type ?? "—"}</Field>
        <Field label="技能">{data.skill_name ?? "—"}</Field>
        <Field label="状态">
          <StatusPill status={data.status} />
        </Field>
        <Field label="HLC">{data.hlc ?? "—"}</Field>
        <Field label="时间">{fmtTime(data.timestamp)}</Field>
        <Field label="耗时(ms)">{data.duration_ms ?? "—"}</Field>
        <Field label="耗时(s)">{data.duration_s ?? "—"}</Field>
        {data.error && (
          <Field label="错误">
            <span className="text-mast-danger">{data.error}</span>
          </Field>
        )}
        <Field label="参数">
          <JsonBlock value={data.params} />
        </Field>
        <Field label="状态变更">
          <JsonBlock value={data.state_delta} />
        </Field>
      </Card>
    </Section>
  );
}
