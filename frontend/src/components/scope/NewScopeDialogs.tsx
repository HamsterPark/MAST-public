// 新建实验 / 新建样品的严谨确认弹窗。
//
// 提交前展示操作后果及已有同名对象，避免重复创建。
//
// 范式参照 admin/DeviceScanner：标题即行动指令、明说会发生什么、主按钮自述后果。

import { useState } from "react";
import { Field, SelectField, TextField } from "../controls";
import { ConfirmDialog } from "./ConfirmDialog";
import {
  useCreateExperiment,
  useCreateSample,
  usePreflight,
  useSwitchExperiment,
} from "@/api/scope";

export const SAMPLE_TYPES = [
  "",
  "metal",
  "semiconductor",
  "molecule",
  "2D",
  "oxide",
  "superconductor",
  "other",
] as const;

function errText(e: unknown): string {
  if (!e) return "";
  if (e instanceof Error) return e.message;
  return typeof e === "string" ? e : JSON.stringify(e);
}

export function NewExperimentDialog({
  open,
  onClose,
  currentName,
}: {
  open: boolean;
  onClose: () => void;
  currentName?: string;
}) {
  const [name, setName] = useState("");
  const [goal, setGoal] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const create = useCreateExperiment();
  const switchTo = useSwitchExperiment();
  const pre = usePreflight(name, open);
  const existing = pre.data?.existing;

  function reset() {
    setName("");
    setGoal("");
    setErr(null);
    onClose();
  }

  async function doCreate() {
    setErr(null);
    try {
      await create.mutateAsync({ name: name.trim(), goal: goal.trim() });
      reset();
    } catch (e) {
      // 失败时弹窗保持打开 + 就地显示原因。静默失败是这里最贵的失败模式。
      setErr(errText(e) || "新建失败——请重试或检查服务状态");
    }
  }

  async function doReuse() {
    if (!existing) return;
    setErr(null);
    try {
      await switchTo.mutateAsync({ experimentId: existing.id });
      reset();
    } catch (e) {
      setErr(errText(e) || "切换失败");
    }
  }

  return (
    <ConfirmDialog
      open={open}
      onClose={reset}
      title="新建实验 — 请确认这是一个新项目"
      confirmLabel="创建实验并切换过去"
      busy={create.isPending}
      busyLabel="创建中…"
      disabled={name.trim().length === 0}
      error={err}
      onConfirm={doCreate}
    >
      <Field label="实验名称">
        <TextField value={name} onChange={setName} placeholder="例如：NiI2 质量表征" />
      </Field>
      <Field label="目标（可选）">
        <TextField value={goal} onChange={setGoal} placeholder="想验证什么" />
      </Field>

      {existing && (
        <div className="rounded border border-mast-warn/40 bg-mast-warn/10 px-3 py-2 text-sm">
          <p className="text-mast-text">
            已存在实验「{existing.name}」—— {existing.sample_count} 个样品 ·{" "}
            {existing.action_count} 条动作
          </p>
          <p className="mt-1 text-xs text-mast-muted">
            要继续做这个实验，还是确实要另开一个新项目？
          </p>
          <button
            className="mt-2 rounded-mast-ctl bg-mast-accent px-3 py-1.5 text-sm font-semibold text-mast-accent-ink hover:opacity-90"
            onClick={doReuse}
            disabled={switchTo.isPending}
          >
            {switchTo.isPending ? "切换中…" : "继续该实验"}
          </button>
        </div>
      )}

      <p className="text-xs text-mast-faint">
        {currentName ? `当前实验「${currentName}」不会被关闭，你随时可以切回去——实验永远可以继续。` : "新建后，之后的扫描/谱学都会记到这个实验下。"}
        {" 不会结束任何实验，也不会影响已有的对话。"}
      </p>
    </ConfirmDialog>
  );
}

export function NewSampleDialog({
  open,
  onClose,
  experimentId,
  experimentName,
  currentSampleName,
}: {
  open: boolean;
  onClose: () => void;
  experimentId?: string | null;
  experimentName?: string;
  currentSampleName?: string;
}) {
  const [name, setName] = useState("");
  const [type, setType] = useState<string>("");
  const [desc, setDesc] = useState("");
  const [err, setErr] = useState<string | null>(null);
  const create = useCreateSample();

  function reset() {
    setName("");
    setType("");
    setDesc("");
    setErr(null);
    onClose();
  }

  async function doCreate() {
    if (!experimentId) {
      setErr("没有当前实验——请先新建或选择一个实验");
      return;
    }
    setErr(null);
    try {
      await create.mutateAsync({
        experimentId,
        name: name.trim(),
        description: desc.trim(),
        sample_type: type,
      });
      reset();
    } catch (e) {
      setErr(errText(e) || "新建样品失败——请重试");
    }
  }

  return (
    <ConfirmDialog
      open={open}
      onClose={reset}
      title={`在实验「${experimentName || "—"}」下新建样品 — 请确认`}
      confirmLabel="创建样品并切换过去"
      busy={create.isPending}
      busyLabel="创建中…"
      disabled={name.trim().length === 0 || !experimentId}
      error={err}
      onConfirm={doCreate}
    >
      <Field label="样品名称">
        <TextField value={name} onChange={setName} placeholder="例如：film-A" />
      </Field>
      <Field label="类型">
        <SelectField
          value={type}
          onChange={setType}
          options={SAMPLE_TYPES.map((t) => ({ value: t, label: t || "（未分类）" }))}
        />
      </Field>
      <Field label="描述（可选）">
        <TextField value={desc} onChange={setDesc} placeholder="制备方法 / 来源" />
      </Field>
      <p className="text-xs text-mast-faint">
        {currentSampleName
          ? `当前样品「${currentSampleName}」不会被结束，可随时切回。`
          : "创建后它将成为当前样品。"}
        {" 新样品的数据会存进实验文件夹下它自己的目录，扫描文件名也会随之改变。"}
      </p>
    </ConfirmDialog>
  );
}
