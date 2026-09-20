import { useState } from "react";
import { Card } from "@/components/ui";
import { Button, Field, TextField } from "@/components/controls";
import { approveBlockReason } from "@/lib/conduct";

// ════════════════════════════════════════════════════════════════════════════
// 批准一份草稿 —— **这一页是唯一的入口**。
//
// 设计 §7 写死了「approve 只接受 UI 来源,本路由永不注册为任何 agent 工具」:
// 批准一份要连跑三天、要动针的流程是人的动作,做成工具等于把「要不要做这个实验」
// 交给一次采样。
//
// 那条规矩的**推论**是这个组件必须存在:如果面板不给按钮,approve 就只剩 curl
// 一条路,而屏幕上会有一份永远停在「草稿」的 conduct,没有任何东西说明为什么
// 推不动。2026-08-04 仪器初始化那次就是这个形状 —— 功能全都接好了、就是没有
// 一个人能点的地方,报上来的话是「这个页面只能进去一次」。
//
// ── 批不下去的时候,这里是最该说清楚的地方 ──────────────────────────────────
//
// 后端只看 `approvable = ok ∧ complete`:「没发现错误」不等于「该跑的检查都跑了」。
// 一次注册表读不到会让规则③整条没跑,而那时 `ok` 仍然是 true —— 所以 409 的回包
// 里 findings(发现了什么)与 checks_skipped(什么没检查)是**两个列表**,
// 这里也分两块显示。把它们并成一句「校验失败」,就把这个区分又抹掉了。
// ════════════════════════════════════════════════════════════════════════════

export interface ApproveOutcome {
  ok?: boolean;
  validation_ok?: boolean;
  validation_complete?: boolean;
  findings?: string[];
  checks_skipped?: string[];
  errors?: string[];
  spec_doc_path?: string;
}

export function ApproveCard({
  onApprove,
  busy,
  outcome,
}: {
  onApprove: (by: string) => void;
  busy: boolean;
  outcome: ApproveOutcome | null;
}) {
  const [by, setBy] = useState("");
  const blocked = outcome && !outcome.ok;

  return (
    <Card className="border-mast-accent">
      <h3 className="text-[17px] font-semibold text-mast-text">这份 conduct 还是草稿</h3>
      <p className="mt-1 text-sm text-mast-muted">
        批准会**冻结参数**、渲染一份人读快照到实验文件夹，然后指挥线程才会接手。
        批准之后参数不能再改——要改就新建一份。
      </p>
      <div className="mt-3 max-w-sm">
        <Field label="批准人（必填）" hint="进审计流。这是「谁让它开始跑的」那一栏。">
          <TextField value={by} onChange={setBy} placeholder="你的名字" />
        </Field>
      </div>
      <div className="mt-3">
        <Button variant="primary" disabled={!by.trim() || busy} onClick={() => onApprove(by.trim())}>
          批准并交给指挥线程
        </Button>
      </div>

      {blocked && (
        <div className="mt-4 rounded-mast-ctl border border-mast-warn-border bg-mast-warn-bg px-3 py-2.5 text-sm text-mast-warn">
          <p className="font-semibold">批不下去</p>
          <p className="mt-0.5 text-xs">{approveBlockReason(outcome)}</p>

          {/* 「发现了什么」与「什么没检查」分开显示。并成一句「校验失败」，
              就把「没检查」和「检查通过」之间那个区分又抹掉了。 */}
          {outcome.findings?.length ? (
            <>
              <p className="mt-2 text-xs text-mast-faint">发现：</p>
              <ul className="ml-4 list-disc text-xs">
                {outcome.findings.map((f) => (
                  <li key={f}>{f}</li>
                ))}
              </ul>
            </>
          ) : null}

          {outcome.checks_skipped?.length ? (
            <>
              <p className="mt-2 text-xs text-mast-faint">这些检查**没能跑**：</p>
              <ul className="ml-4 list-disc text-xs">
                {outcome.checks_skipped.map((s) => (
                  <li key={s}>{s}</li>
                ))}
              </ul>
            </>
          ) : null}

          {outcome.errors?.length ? (
            <ul className="ml-4 mt-2 list-disc text-xs">
              {outcome.errors.map((e) => (
                <li key={e}>{e}</li>
              ))}
            </ul>
          ) : null}
        </div>
      )}
    </Card>
  );
}
