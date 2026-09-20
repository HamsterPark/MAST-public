import { useState } from "react";
import { Badge, Card } from "@/components/ui";
import { Button, Field, Modal, TextField } from "@/components/controls";
import type { components } from "@/api/schema";
import { fmtClock, fmtDuration, waitGateRows } from "@/lib/conduct";

type ActiveWait = components["schemas"]["ActiveWaitView"];

// ════════════════════════════════════════════════════════════════════════════
// 等待大卡 —— 面板上唯一一块「现在轮到你了」。
//
// 它要回答的**不是**「在等吗」,而是「**还缺哪一个**」。设计 §3-5 的原话:双闸
// = 人的 ack AND 物理条件,两个证据回答两个问题、互不替代(人确认了不等于降到温,
// 温度到位不等于样品换好了)。所以两个闸各占一行、各带各的话,而不是一个
// 「还差 1 项」的计数 —— 计数答不出「差的是哪一个」,而那正是用户要做的下一件事。
//
// ── 三个显示纪律 ────────────────────────────────────────────────────────────
//
// **一、stale 不是「没到」。** 温度读数过期是**判不了**:干等下去是错的。它用
// 「读不到」的措辞和自己的颜色,并且直接把下一步动作写出来(温度采集程序还在跑吗)。
//
// **二、waive 标记持续显示。** 那个闸是人拿一条留痕的命令放过去的,报告里带着它,
// 屏幕上也必须一直带着 —— 不是按下那一刻闪一次。
//
// **三、ack 按钮按下去不代表就走了。** 它只是把意图入队,下一 tick 才生效,而且
// 另一个闸可能还缺。按钮回来的那句话说的是「还缺什么」,不是「已放行」。
// ════════════════════════════════════════════════════════════════════════════

export function WaitCard({
  wait,
  onAck,
  onWaive,
  busy,
}: {
  wait: ActiveWait;
  onAck: (note: string) => void;
  onWaive: (reason: string) => void;
  busy: boolean;
}) {
  const [waiveOpen, setWaiveOpen] = useState(false);
  const [reason, setReason] = useState("");
  const rows = waitGateRows(wait);
  const lacking = rows.filter((r) => !r.ok);
  const cond = wait.condition;
  const ackDone = wait.ack?.at != null;

  return (
    <Card className="border-mast-warn-border bg-mast-warn-bg/30">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h3 className="text-[17px] font-semibold text-mast-text">
            {lacking.length ? `等你：还缺 ${lacking.map((r) => r.label).join(" + ")}` : "两个闸都齐了，正在放行"}
          </h3>
          <p className="mt-1 whitespace-pre-wrap text-sm text-mast-text">{wait.message}</p>
        </div>
        <Badge tone="WARN">等待中</Badge>
      </div>

      <ul className="mt-4 space-y-2">
        {rows.map((r) => (
          <li
            key={r.label}
            className="flex items-start gap-2.5 rounded-mast-ctl border border-mast-border bg-mast-panel px-3 py-2"
          >
            <span
              className={
                r.ok
                  ? "mt-0.5 text-mast-auto"
                  : r.unreadable
                    ? "mt-0.5 text-mast-warn"
                    : "mt-0.5 text-mast-muted"
              }
              aria-hidden="true"
            >
              {r.ok ? "✓" : r.unreadable ? "?" : "○"}
            </span>
            <div className="min-w-0">
              <p className="text-sm font-medium text-mast-text">
                {r.label}
                {/* 「读不到」自己一个词,不混进「没过」—— 两者的下一步动作不同。 */}
                {r.unreadable && <span className="ml-2 text-xs text-mast-warn">判不了</span>}
              </p>
              <p className="mt-0.5 text-xs text-mast-muted">{r.detail}</p>
            </div>
          </li>
        ))}
      </ul>

      <div className="mt-4 flex flex-wrap items-center gap-2">
        <Button variant="primary" onClick={() => onAck("")} disabled={busy || ackDone}>
          {ackDone ? "已确认" : "我已确认"}
        </Button>
        {/* waive 只在真有条件闸、且还没放行过的时候出现:一个不适用的按钮
            比没有按钮更让人犹豫。 */}
        {cond && !cond.waived && (
          <Button variant="default" onClick={() => setWaiveOpen(true)} disabled={busy}>
            条件读不到，由我提供证据
          </Button>
        )}
        <span className="text-xs text-mast-faint">
          进入于 {fmtClock(wait.entered_at)}
          {wait.last_notified_at != null && `　·　上次通知 ${fmtClock(wait.last_notified_at)}`}
          {wait.request_id && `　·　心愿单 ${wait.request_id}`}
        </span>
      </div>

      {cond?.stale_after_s != null && (
        <p className="mt-2 text-xs text-mast-faint">
          读数超过 {fmtDuration(cond.stale_after_s)} 就算读不到（那时这条等待会降级为「要人来看」，而不是继续干等）。
        </p>
      )}

      <Modal open={waiveOpen} onClose={() => setWaiveOpen(false)} title="由人提供证据放行条件闸">
        <p className="text-sm text-mast-muted">
          这**不是**默默放行：谁放的、为什么放，会进审计流，面板上会**一直**显示这个标记，报告里也带着它。
        </p>
        <p className="mt-2 text-sm text-mast-muted">
          它存在的理由是「能停不能解 = 死锁」——传感器读不到时这个闸会永远 stale，
          人必须有一条显式、留痕的解锁路。
        </p>
        <div className="mt-4">
          <Field label="理由（必填）" hint="写给三天后的自己看：当时凭什么认为条件其实满足了。">
            <TextField
              value={reason}
              onChange={setReason}
              placeholder="例：温度计 COM13 被 另一个串口程序 占着，手工读数 4.3 K"
            />
          </Field>
        </div>
        <div className="mt-4 flex justify-end gap-2">
          <Button variant="ghost" onClick={() => setWaiveOpen(false)}>
            取消
          </Button>
          <Button
            variant="primary"
            disabled={!reason.trim() || busy}
            onClick={() => {
              onWaive(reason.trim());
              setWaiveOpen(false);
              setReason("");
            }}
          >
            确认放行并留痕
          </Button>
        </div>
      </Modal>
    </Card>
  );
}
