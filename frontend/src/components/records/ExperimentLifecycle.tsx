import { useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api } from "../../api/client";
import { Button, Field, Modal, SelectField, TextField, useToast } from "../controls";

// Experiment lifecycle controls reused by the 实验 tab:
//   - 新建实验  POST /api/experiments {name, goal}
//   - 结束实验  POST /api/experiments/{id}/end {status}
// Both invalidate the experiments list so the table refreshes.

export function StartExperimentButton() {
  const qc = useQueryClient();
  const { toast, node } = useToast();
  const [open, setOpen] = useState(false);
  const [name, setName] = useState("");
  const [goal, setGoal] = useState("");

  const m = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/experiments", {
        body: { name, goal, thread_id: null },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      if (d.ok) {
        toast("实验已创建");
        setOpen(false);
        setName("");
        setGoal("");
        qc.invalidateQueries({ queryKey: ["experiments"] });
      } else {
        toast(d.degraded ? "记录后端未连接（降级）" : "创建失败", "err");
      }
    },
    onError: () => toast("创建失败", "err"),
  });

  return (
    <>
      <Button variant="primary" onClick={() => setOpen(true)}>
        新建实验
      </Button>
      <Modal open={open} onClose={() => setOpen(false)} title="新建实验">
        <div className="space-y-4">
          <Field label="名称">
            <TextField value={name} onChange={setName} placeholder="实验名称" />
          </Field>
          <Field label="目标">
            <TextField value={goal} onChange={setGoal} placeholder="实验目标（可选）" />
          </Field>
          <div className="flex justify-end gap-2">
            <Button onClick={() => setOpen(false)}>取消</Button>
            <Button
              variant="primary"
              onClick={() => m.mutate()}
              disabled={!name || m.isPending}
            >
              {m.isPending ? "创建中…" : "创建"}
            </Button>
          </div>
        </div>
      </Modal>
      {node}
    </>
  );
}

// EndExperimentButton 与 STATUS_OPTIONS 已删除 (2026-07-28)。
//
// 实验没有「结束」这个动作 —— 判据：「没必要做归档。有的实验可能过了
// 十年重启。何必归档呢？如果说要给一个实验写总结，写报告，并不必以归档为
// 前提。」记录页的主动作因此换成了「切换到此实验」(RecordsPage)。
//
// 后端的 POST /api/experiments/{id}/end 仍在（标了 deprecated，行为改成
// 「仅在它是当前作用域时清指针」），只是 UI 不再提供入口。

const RATING_OPTIONS = [
  { value: "up", label: "👍 好" },
  { value: "down", label: "👎 差" },
  { value: "neutral", label: "😐 中性" },
];

export function FeedbackButton({ experimentId }: { experimentId?: string | null }) {
  const qc = useQueryClient();
  const { toast, node } = useToast();
  const [open, setOpen] = useState(false);
  const [rating, setRating] = useState("up");
  const [comment, setComment] = useState("");

  const m = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/feedback", {
        body: {
          rating,
          comment,
          experiment_id: experimentId ?? null,
          sample_id: null,
          conversation_id: null,
          agent: "",
        },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      if (d.ok) {
        toast("反馈已记录");
        setOpen(false);
        setComment("");
        if (experimentId) qc.invalidateQueries({ queryKey: ["experiment", experimentId] });
      } else {
        toast("记录失败", "err");
      }
    },
    onError: () => toast("记录失败", "err"),
  });

  return (
    <>
      <Button onClick={() => setOpen(true)}>提交反馈</Button>
      <Modal open={open} onClose={() => setOpen(false)} title="用户反馈">
        <div className="space-y-4">
          <Field label="评分">
            <SelectField value={rating} onChange={setRating} options={RATING_OPTIONS} />
          </Field>
          <Field label="评论">
            <TextField value={comment} onChange={setComment} placeholder="可选评论" />
          </Field>
          <div className="flex justify-end gap-2">
            <Button onClick={() => setOpen(false)}>取消</Button>
            <Button variant="primary" onClick={() => m.mutate()} disabled={m.isPending}>
              {m.isPending ? "提交中…" : "提交"}
            </Button>
          </div>
        </div>
      </Modal>
      {node}
    </>
  );
}
