import { useState } from "react";
import { Button, Field, Modal, TextField, Toggle } from "@/components/controls";
import { opEnabled, opLabel, type ConductOp } from "@/lib/conduct";

// ════════════════════════════════════════════════════════════════════════════
// 五枚操作按钮 —— pause / resume / abort / takeover / attended。
//
// ── 两条纪律 ────────────────────────────────────────────────────────────────
//
// **一、灰按钮必须说得出为什么灰。** 后端对一个当前状态下没有意义的意图会
// 「显式拒绝 + 记 op_rejected」,不静默 no-op —— 但用户看到的仍然是「我按了,
// 什么都没发生」,而且要去翻审计流才知道为什么。所以判据在这一侧也有一份
// (`lib/conduct.opEnabled`,与 director.OP_VALID_STATUSES 对着 parity 测试),
// 灰按钮把理由挂在 title 上。
//
// **二、UI 绝不冻结。** 每个按钮最坏的结果是「无反应 + 一句提示」:意图入队失败
// 只弹 toast,不锁死界面、不转圈等一个不会来的回包。abort 尤其如此 —— 它是
// 「能停不能解」的反面,必须从任何非终态都按得下去。
//
// ── abort 为什么要一个理由 ───────────────────────────────────────────────────
//
// 后端 schema 把 reason 定成必填。一个没有理由的中止,事后没人答得上「为什么那
// 一晚停了」——而那句话正是第二天早上唯一有用的东西。所以这里用一个小对话框收它,
// 而不是让请求带一个空串过去被 422 挡回来。
// ════════════════════════════════════════════════════════════════════════════

export function OpBar({
  status,
  attended,
  busy,
  onOp,
  onAbort,
  onAttended,
}: {
  status: string;
  attended: boolean;
  busy: boolean;
  // 走通用「一个 op 名 = 一条路由」的那几个。剩下的各有各的形状,不能混进来:
  // abort 要 reason、ack 要 wait_id、waive 要 wait_id+reason、set_attended 要一个
  // 布尔,而 **override_decision 要 decision_id + reason**(它的卡是 DecisionCard)。
  // 把它们塞进这个按钮条,参数就得靠调用方猜 —— 而猜出来的那个多半是空串。
  onOp: (
    op: Exclude<
      ConductOp,
      "abort" | "ack" | "waive_condition" | "set_attended" | "override_decision"
    >,
  ) => void;
  onAbort: (reason: string) => void;
  onAttended: (next: boolean) => void;
}) {
  const [abortOpen, setAbortOpen] = useState(false);
  const [reason, setReason] = useState("");

  const btn = (op: "pause" | "resume" | "takeover") => {
    const { enabled, why } = opEnabled(op, status);
    return (
      <span title={why || undefined}>
        <Button variant="default" disabled={!enabled || busy} onClick={() => onOp(op)}>
          {opLabel(op)}
        </Button>
      </span>
    );
  };

  const abortGate = opEnabled("abort", status);
  const attendedGate = opEnabled("set_attended", status);

  return (
    <div className="flex flex-wrap items-center gap-2">
      {btn("pause")}
      {btn("resume")}
      {btn("takeover")}
      <span title={abortGate.why || undefined}>
        <Button
          variant="danger"
          disabled={!abortGate.enabled || busy}
          onClick={() => setAbortOpen(true)}
        >
          中止
        </Button>
      </span>

      <span className="ml-2 border-l border-mast-border pl-3" title={attendedGate.why || undefined}>
        <Toggle
          checked={attended}
          onChange={(v) => attendedGate.enabled && !busy && onAttended(v)}
          label={attended ? "有人值守" : "无人值守"}
        />
      </span>
      <span className="text-xs text-mast-faint">
        {attended
          ? "判不了的时候会问你"
          : "判不了的时候直接走保守分支（不问人——无人窗口里问了也必然超时）"}
      </span>

      <Modal open={abortOpen} onClose={() => setAbortOpen(false)} title="中止这份 conduct">
        <p className="text-sm text-mast-muted">
          中止会**立刻**给当前这一步发停止信号（不等下一个 tick——指挥线程可能正卡在一次长动作里），
          然后走确认式退针。等待态里针已经退了，直接结束。
        </p>
        <p className="mt-2 text-sm text-mast-muted">
          这是终态：**要接着做请新建一份。**
        </p>
        <div className="mt-4">
          <Field label="理由（必填）" hint="第二天早上唯一有用的东西就是这句话。">
            <TextField
              value={reason}
              onChange={setReason}
              placeholder="例：针尖在 S2 第二个偏压上明显变钝，先停下来手工修"
            />
          </Field>
        </div>
        <div className="mt-4 flex justify-end gap-2">
          <Button variant="ghost" onClick={() => setAbortOpen(false)}>
            取消
          </Button>
          <Button
            variant="danger"
            disabled={!reason.trim() || busy}
            onClick={() => {
              onAbort(reason.trim());
              setAbortOpen(false);
              setReason("");
            }}
          >
            确认中止
          </Button>
        </div>
      </Modal>
    </div>
  );
}
