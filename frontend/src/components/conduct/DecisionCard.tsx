import { useState } from "react";
import { Badge, Card } from "@/components/ui";
import { Button, Field, Modal, TextField } from "@/components/controls";
import type { components } from "@/api/schema";

type PendingDecision = components["schemas"]["PendingDecisionView"];

// ════════════════════════════════════════════════════════════════════════════
// 判定大卡 —— 一道闸门停下来之后,人**看得见**它停在哪一次判定上,并且能说「继续」。
//
// 在这块卡存在之前,一份被闸门停住的 conduct 只有两条出路:中止,或者永久接管。
// 也就是说凌晨闸门停下、早上人看了觉得没问题,唯一的选择是**放弃这份 conduct**。
// 那是「能停不能解 = 死锁」的第二次(第一次是 2026-08-13 的急停闩:针和仪器完好、
// 机器锁死到重启,因为解闩的函数是死代码)。
//
// ── 三个显示纪律(与 WaitCard 同源,但答的是另一个问题)──────────────────
//
// **一、它跟等待卡不是一回事。** 同一个 `waiting_operator` 有两个来源:一个 `wait`
// 步(等 ack + 物理条件,可轮询)和一次裁决转人(等人的判断,没有可轮询的闸)。
// 前者是 WaitCard,后者是这里。两张卡不会同时出现 —— 后端的 `pending_decision`
// 在有 `active_wait` 时就是 `null`。
//
// **二、闸门判的那句话原样摆着。** 人推翻的是**它**,所以屏幕上必须能读到被推翻的
// 是什么。把它折叠成一句「等人处理」,事后对账就看不出这一次放行推翻了什么。
//
// **三、它解的是这一次判定,不是这道闸。** 弹窗里逐字写着,因为这正是最容易被
// 当成「跳过这道闸」的按钮 —— 而下一次走到同一道闸,照样重新判。
// ════════════════════════════════════════════════════════════════════════════

const WHICH_LABEL: Record<string, string> = {
  entry: "阶段入口闸",
  step: "步后闸",
  exit: "阶段出口闸",
};

export function DecisionCard({
  pending,
  onOverride,
  busy,
}: {
  pending: PendingDecision;
  onOverride: (reason: string) => void;
  busy: boolean;
}) {
  const [open, setOpen] = useState(false);
  const [reason, setReason] = useState("");
  const where = WHICH_LABEL[pending.which] ?? pending.which;

  return (
    <Card className="border-mast-warn-border bg-mast-warn-bg/30">
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-[17px] font-semibold text-mast-text">
            等你看一眼：闸门停在这里了
          </h3>
          <p className="mt-1 text-sm text-mast-muted">
            {pending.stage_id} 的 {where}
            <span className="ml-1 font-mono text-xs">{pending.gate_id}</span>
          </p>
        </div>
        <Badge tone="WARN">等人判断</Badge>
      </div>

      {/* 闸门当时判了什么、为什么 —— 人推翻的就是这一句,所以它原样摆着。 */}
      <div className="mt-4 rounded-mast-ctl border border-mast-border bg-mast-panel px-3 py-2">
        <p className="text-xs text-mast-faint">闸门的裁决</p>
        <p className="mt-0.5 font-mono text-sm text-mast-text">{pending.verdict}</p>
        <p className="mt-1 whitespace-pre-wrap text-sm text-mast-text">{pending.reason}</p>
      </div>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Button variant="primary" onClick={() => setOpen(true)} disabled={busy}>
          我看过了，继续
        </Button>
        <span className="text-xs text-mast-faint">
          判定编号 #{pending.decision_id}
          　·　继续 = 只放行这一次；下一次走到这道闸，还会重新判
        </span>
      </div>

      <Modal open={open} onClose={() => setOpen(false)} title="放行这一次闸门判定">
        <p className="text-sm text-mast-muted">
          这**不是**让这道闸失效：你放行的是**第 #{pending.decision_id} 次判定**这一次。
          下一次走到 <span className="font-mono">{pending.gate_id}</span>，它照样重新判。
        </p>
        <p className="mt-2 text-sm text-mast-muted">
          闸门那条判定记录**原样留着**（它说的是闸门当时判了什么），你的决定另起一条。
          两者分开记，否则事后对账会看到一道从不判 fail 的闸——而那正是最该复核的那种记录。
        </p>
        <div className="mt-4">
          <Field
            label="理由（必填）"
            hint="写给三天后的自己看：当时凭什么认为可以继续。"
          >
            <TextField
              value={reason}
              onChange={setReason}
              placeholder="例：n_keep=0 是因为这一批谱存到了另一个目录，人工核对过 12 条都可用"
            />
          </Field>
        </div>
        <div className="mt-4 flex justify-end gap-2">
          <Button variant="ghost" onClick={() => setOpen(false)}>
            取消
          </Button>
          <Button
            variant="primary"
            disabled={!reason.trim() || busy}
            onClick={() => {
              onOverride(reason.trim());
              setOpen(false);
              setReason("");
            }}
          >
            放行这一次并留痕
          </Button>
        </div>
      </Modal>
    </Card>
  );
}
