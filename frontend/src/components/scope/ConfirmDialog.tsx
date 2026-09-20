// 确认弹窗基元。controls.tsx 只有 Modal，每个调用方都在手写「取消 / 确认」。
//
// 核心契约（取自 RunTaskPanel 的注释）：**失败时弹窗不关闭**，把服务端返回的
// 错误就地红字渲染。toast 不够 —— 它会飘走，而「原先以为成功了」是这类操作最贵的
// 失败模式。一个没说自己失败了的确认框，比没有确认框更糟。

import type { ReactNode } from "react";
import { Button, Modal } from "../controls";

export function ConfirmDialog({
  open,
  title,
  children,
  confirmLabel = "确认",
  cancelLabel = "取消",
  tone = "primary",
  busy = false,
  busyLabel,
  error,
  disabled = false,
  wide = false,
  onConfirm,
  onClose,
  extraAction,
}: {
  open: boolean;
  title: string;
  children?: ReactNode;
  confirmLabel?: string;
  cancelLabel?: string;
  tone?: "primary" | "danger";
  busy?: boolean;
  busyLabel?: string;
  /** 服务端原话。非空时红字就地显示，且弹窗保持打开。 */
  error?: string | null;
  disabled?: boolean;
  wide?: boolean;
  onConfirm: () => void;
  onClose: () => void;
  /** 可选的第三个动作（如「先中止任务」）。 */
  extraAction?: ReactNode;
}) {
  return (
    <Modal open={open} onClose={onClose} title={title} wide={wide}>
      <div className="space-y-4">
        {children}
        {error ? (
          <div
            role="alert"
            className="rounded border border-mast-danger/40 bg-mast-danger/10 px-3 py-2 text-sm text-mast-danger"
          >
            {error}
          </div>
        ) : null}
        <div className="flex items-center justify-end gap-2">
          {extraAction}
          <Button onClick={onClose} disabled={busy}>
            {cancelLabel}
          </Button>
          <Button
            variant={tone}
            onClick={onConfirm}
            loading={busy}
            disabled={busy || disabled}
          >
            {busy ? (busyLabel ?? "处理中…") : confirmLabel}
          </Button>
        </div>
      </div>
    </Modal>
  );
}
