// 实验 / 样品选择器 —— 「做了一个月这个又回去做那个」的那一次点击。
//
// 排序按 last_active_at 倒序（不是创建时间）：要回到的那个实验可能是上个月建的，
// 但昨天才动过。/api/experiments/recent 已经按这个排好了。

import { useState } from "react";
import { Modal, TextField } from "../controls";
import { Spinner } from "../ui";
import { useRecentExperiments, useSamplesOf } from "@/api/scope";

function relTime(iso?: string | null): string {
  if (!iso) return "从未活动";
  const t = Date.parse(iso);
  if (Number.isNaN(t)) return "";
  const s = Math.max(0, (Date.now() - t) / 1000);
  if (s < 90) return "刚刚";
  if (s < 3600) return `${Math.round(s / 60)} 分钟前`;
  if (s < 86400) return `${Math.round(s / 3600)} 小时前`;
  return `${Math.round(s / 86400)} 天前`;
}

const rowCls =
  "flex w-full items-center justify-between gap-3 rounded-mast-ctl px-3 py-2 text-left " +
  "hover:bg-mast-panel-2 disabled:cursor-not-allowed disabled:opacity-60";

export function ExperimentPicker({
  open,
  onClose,
  onPick,
  onCreate,
  currentId,
}: {
  open: boolean;
  onClose: () => void;
  onPick: (id: string, name: string) => void;
  onCreate: () => void;
  currentId?: string | null;
}) {
  const [q, setQ] = useState("");
  const list = useRecentExperiments(q, open);

  return (
    <Modal open={open} onClose={onClose} title="切换实验">
      <div className="space-y-3">
        <TextField value={q} onChange={setQ} placeholder="按名称筛选…" />
        <button
          className={rowCls + " border border-dashed border-mast-border-strong text-mast-accent"}
          onClick={onCreate}
        >
          ＋ 新建实验
        </button>
        <div className="max-h-[46vh] space-y-1 overflow-auto">
          {list.isLoading && <Spinner />}
          {list.data?.experiments?.length === 0 && (
            <p className="px-3 py-6 text-center text-sm text-mast-muted">没有匹配的实验</p>
          )}
          {list.data?.experiments?.map((e) => {
            const isCur = e.id === currentId;
            return (
              <button
                key={e.id}
                className={rowCls + (isCur ? " bg-mast-accent-soft" : "")}
                onClick={() => onPick(e.id, e.name)}
                disabled={isCur}
                title={e.id}
              >
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-mast-text">
                    {e.name || "(未命名)"}
                    {isCur && <span className="ml-2 text-xs text-mast-accent">当前</span>}
                  </span>
                  <span className="block truncate text-xs text-mast-faint">
                    {e.sample_count} 个样品 · {e.action_count} 条动作 ·{" "}
                    {relTime(e.last_active_at)}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
        <p className="text-xs text-mast-faint">
          切换实验不会结束任何实验 —— 实验永久存在，随时可以再切回来。
        </p>
      </div>
    </Modal>
  );
}

export function SamplePicker({
  open,
  onClose,
  onPick,
  onCreate,
  experimentId,
  experimentName,
  currentId,
}: {
  open: boolean;
  onClose: () => void;
  onPick: (id: string, name: string) => void;
  onCreate: () => void;
  experimentId?: string | null;
  experimentName?: string;
  currentId?: string | null;
}) {
  const list = useSamplesOf(experimentId, open);

  return (
    <Modal open={open} onClose={onClose} title={`选择样品 — ${experimentName || "当前实验"}`}>
      <div className="space-y-3">
        <button
          className={rowCls + " border border-dashed border-mast-border-strong text-mast-accent"}
          onClick={onCreate}
        >
          ＋ 新建样品
        </button>
        <div className="max-h-[46vh] space-y-1 overflow-auto">
          {list.isLoading && <Spinner />}
          {list.data?.samples?.length === 0 && (
            <p className="px-3 py-6 text-center text-sm text-mast-muted">
              这个实验下还没有样品
            </p>
          )}
          {list.data?.samples?.map((s) => {
            const isCur = s.id === currentId;
            return (
              <button
                key={s.id}
                className={rowCls + (isCur ? " bg-mast-accent-soft" : "")}
                onClick={() => onPick(s.id, s.name)}
                disabled={isCur}
                title={s.id}
              >
                <span className="min-w-0 flex-1">
                  <span className="block truncate text-sm text-mast-text">
                    {s.index ? `S${String(s.index).padStart(2, "0")} · ` : ""}
                    {s.name || "(未命名)"}
                    {isCur && <span className="ml-2 text-xs text-mast-accent">当前</span>}
                  </span>
                  <span className="block truncate text-xs text-mast-faint">
                    {s.sample_type || "未分类"} · {relTime(s.last_active_at)}
                  </span>
                </span>
              </button>
            );
          })}
        </div>
        <p className="text-xs text-mast-faint">
          样品可以来回切换 —— 换样品不会结束上一个，之前那块随时能切回来。
        </p>
      </div>
    </Modal>
  );
}
