import { Badge, Card, EmptyNote } from "@/components/ui";
import type { components } from "@/api/schema";
import {
  badgeTone,
  budgetText,
  fmtClock,
  fmtDuration,
  heartbeatBanner,
  ignitionText,
  statusLabel,
  statusTone,
  timelineProgress,
} from "@/lib/conduct";

type Detail = components["schemas"]["ConductDetail"];

// ════════════════════════════════════════════════════════════════════════════
// 面板的其余部分:状态头 / 停滞告警条 / 时间线 / 当前卡 / 闸门史 / 预算 / 文件夹。
//
// 全部来自 `GET /api/conducts/{id}` 这**一个**端点。这不是省事,是防一类具体的
// 缺陷:面板由 N 个端点拼起来时,每个端点各有各的时刻,于是「状态是 RUNNING」和
// 「等待卡还亮着」可以同时显示,而两者都各自没错。一个端点 = 一个时刻。
// ════════════════════════════════════════════════════════════════════════════

/** 状态头:一句「现在什么样」+ **一句为什么**。 */
export function StatusHeader({ d }: { d: Detail }) {
  return (
    <Card>
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div className="min-w-0">
          <div className="flex flex-wrap items-center gap-2">
            <Badge tone={badgeTone(statusTone(d.status))}>{statusLabel(d.status)}</Badge>
            {d.detour?.active && (
              // detour 是**正交标志不是状态** —— 绕道中同样有 RUNNING/WAITING
              // 子状态,画成第二个状态徽章会造成双重身份。
              <Badge tone="WARN">绕道中（回 {d.detour.return_stage}）</Badge>
            )}
            {!d.attended && <Badge tone="INFO">无人值守</Badge>}
            {d.director_running ? null : <Badge tone="default">指挥线程未在跑</Badge>}
          </div>
          {/* status_reason 是用户唯一能看的那句话。后端保证异常态永不空着它。 */}
          {d.status_reason && (
            <p className="mt-2 whitespace-pre-wrap text-sm text-mast-text">{d.status_reason}</p>
          )}
          <p className="mt-1 text-xs text-mast-faint">
            {d.title || d.spec_id}　·　模板 {d.spec_id} v{d.spec_version}
            　·　实验 {d.experiment_id || "—"}
            　·　证据代次 {d.evidence_epoch}
            {d.approved_by && `　·　${d.approved_by} 批于 ${d.approved_at}`}
          </p>
        </div>
      </div>
      {ignitionText(d.ignition) && (
        <p className="mt-3 rounded-mast-ctl border border-mast-warn-border bg-mast-warn-bg px-3 py-2 text-xs text-mast-warn">
          {ignitionText(d.ignition)}
        </p>
      )}
      {d.detour?.active && d.detour.reason && (
        <p className="mt-3 rounded-mast-ctl border border-mast-warn-border bg-mast-warn-bg px-3 py-2 text-xs text-mast-warn">
          进绕道的理由：{d.detour.reason}
          　—— 进绕道即作废此前采的针尖/仪器证据（代次已 +1），闸门只认当前代次。
        </p>
      )}
    </Card>
  );
}

/** 停滞告警条。不该报的时候**不画** —— 报久了就没人看。 */
export function StallBanner({ d }: { d: Detail }) {
  const b = heartbeatBanner(d.heartbeat, d.status);
  if (!b) return null;
  const crit = b.tone === "crit";
  return (
    <div
      className={
        crit
          ? "rounded-mast-ctl border border-mast-danger-border bg-mast-danger-bg px-4 py-3 text-sm text-mast-danger"
          : "rounded-mast-ctl border border-mast-warn-border bg-mast-warn-bg px-4 py-3 text-sm text-mast-warn"
      }
    >
      <p className="font-semibold">{b.title}</p>
      <p className="mt-1 whitespace-pre-wrap">{b.detail}</p>
    </div>
  );
}

/** 时间线条:每个阶段一行,当前那个高亮。 */
export function Timeline({ d }: { d: Detail }) {
  const rows = d.timeline ?? [];
  if (!rows.length) {
    return (
      <EmptyNote label={(d.not_available?.length ? d.not_available[0] : "这份 conduct 没有阶段")} />
    );
  }
  return (
    <ol className="space-y-1.5">
      {rows.map((row) => {
        const cur = row.status === "current";
        const done = row.status === "done";
        const pct = row.steps_total ? (row.steps_done / row.steps_total) * 100 : 0;
        return (
          <li
            key={row.stage_id}
            className={
              cur
                ? "rounded-mast-ctl border border-mast-accent bg-mast-accent-soft px-3 py-2"
                : "rounded-mast-ctl border border-mast-border px-3 py-2"
            }
          >
            <div className="flex items-center justify-between gap-3 text-sm">
              <span className={cur ? "font-semibold text-mast-text" : "text-mast-muted"}>
                {done && "✓ "}
                {row.stage_id}　{row.title}
              </span>
              <span className="shrink-0 font-mono text-xs tabular-nums text-mast-faint">
                {timelineProgress(row.steps_done, row.steps_total)}
              </span>
            </div>
            <div className="mt-1.5 h-1 w-full overflow-hidden rounded bg-mast-border">
              <div
                className={done ? "h-full bg-mast-auto" : "h-full bg-mast-accent"}
                style={{ width: `${pct}%` }}
              />
            </div>
          </li>
        );
      })}
    </ol>
  );
}

/** 当前卡:在哪一段、哪一步、跑了多久。 */
export function CurrentCard({ d }: { d: Detail }) {
  const hb = d.heartbeat;
  // 降级响应里这几块可能整个缺席（schema 里它们都是非必填）。缺了就画「—」，
  // 而不是让整页崩在一个 undefined 上 —— 「UI 绝不冻结」在这一层就是这个意思。
  const stage = d.stage;
  const step = d.step;
  return (
    <Card>
      <dl className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm sm:grid-cols-4">
        <Cell label="阶段" value={stage?.id ? `${stage.id}　${stage.title}` : "—"} />
        <Cell label="步" value={step?.id || "—"} sub={step?.kind} />
        <Cell
          label="开始于"
          value={fmtClock(step?.started_at)}
          // 不在步里时后端**不给**开始时刻(而不是拿一个旧的顶上),这里如实说。
          sub={d.active_run_id ? `已跑 ${fmtDuration(hb?.step_elapsed_s)}` : "当前不在步里"}
        />
        <Cell
          label="心跳"
          value={hb?.age_s != null ? `${fmtDuration(hb.age_s)}前` : "—"}
          sub={hb?.in_step ? "执行长步时心跳不更新，那是设计" : hb?.reason || ""}
        />
      </dl>
    </Card>
  );
}

function Cell({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <div className="min-w-0">
      <dt className="text-xs text-mast-faint">{label}</dt>
      <dd className="truncate text-mast-text" title={value}>
        {value}
      </dd>
      {sub && <p className="truncate text-xs text-mast-faint" title={sub}>{sub}</p>}
    </div>
  );
}

/**
 * 一条闸门判定的**来路**:是规则判的还是模型判的,模型是哪一个。
 *
 * 少了这一行,一条 LLM 判决与一条 rule 判定在表上长得一模一样 —— 而这两者事后
 * 要做的核对完全不同(一个查判据,一个查那次判决本身:哪个模型答的、怎么解析
 * 出来的、这一段的唤醒预算还剩几次)。判决**没发生**时(超时 / 建不出模型 /
 * 返回值不在闭集里)也要说出来,那正是最该看见的一类。
 */
function GateProvenance({ g }: { g: components["schemas"]["GateHistoryRow"] }) {
  if (g.kind !== "llm") return null;
  const bits: string[] = [];
  // 模型名读不到 ⇒ 说「读不到」,不留空。空着会被读成「不是模型判的」。
  bits.push(`模型 ${g.llm_model || "(没记到)"}`);
  if (g.llm_parse_path) bits.push(g.llm_parse_path);
  if (g.llm_wakes_max != null) bits.push(`本段唤醒 ${g.llm_wakes_used ?? "?"}/${g.llm_wakes_max}`);
  return (
    <p className="mt-0.5 text-xs text-mast-faint">
      <span className="mr-1 rounded bg-mast-border px-1 py-0.5 font-mono text-[10px]">LLM</span>
      {bits.join("　·　")}
      {g.llm_unavailable && (
        <span className="ml-1 text-mast-warn">　判决没发生:{g.llm_unavailable}</span>
      )}
      {g.escaped && !g.llm_unavailable && <span className="ml-1 text-mast-warn">　模型弃权</span>}
    </p>
  );
}

/** 闸门史表(近 20 条)。 */
export function GateHistory({ d }: { d: Detail }) {
  const gates = d.gates_history ?? [];
  if (!gates.length) return <EmptyNote label="还没有闸门判定" />;
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-left text-sm">
        <thead className="text-xs text-mast-faint">
          <tr>
            <th className="py-1 pr-3 font-normal">时刻</th>
            <th className="py-1 pr-3 font-normal">阶段</th>
            <th className="py-1 pr-3 font-normal">闸门</th>
            <th className="py-1 pr-3 font-normal">裁决</th>
            <th className="py-1 font-normal">理由</th>
          </tr>
        </thead>
        <tbody>
          {[...gates].reverse().map((g, i) => (
            <tr key={`${g.ts}-${g.gate_id}-${i}`} className="border-t border-mast-border">
              <td className="whitespace-nowrap py-1.5 pr-3 font-mono text-xs text-mast-muted">
                {g.ts}
              </td>
              <td className="py-1.5 pr-3 text-mast-muted">{g.stage_id}</td>
              <td className="py-1.5 pr-3 text-mast-text">{g.gate_id}</td>
              <td className="py-1.5 pr-3">
                <Badge tone={g.verdict === "pass" ? "AUTO" : "WARN"}>{g.verdict}</Badge>
              </td>
              <td className="py-1.5 text-xs text-mast-muted">
                {g.note}
                <GateProvenance g={g} />
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** 预算 + LLM 唤醒 + 实验文件夹产物。 */
export function Vitals({ d }: { d: Detail }) {
  const budget = budgetText(d.budget);
  const wakes = Object.entries(d.llm_wakes ?? {});
  const folder = d.folder;
  return (
    <div className="grid gap-3 sm:grid-cols-3">
      <Card>
        <p className="text-xs text-mast-faint">花销</p>
        <p
          className={
            budget.unreadable
              ? "mt-1 text-lg text-mast-warn"
              : "mt-1 text-lg tabular-nums text-mast-text"
          }
        >
          {budget.text}
        </p>
        {/* 「读不到」要写出来。一个空着的预算条会被读成「没花钱」。 */}
        {budget.hint && <p className="mt-1 text-xs text-mast-faint">{budget.hint}</p>}
      </Card>

      <Card>
        <p className="text-xs text-mast-faint">LLM 唤醒（每阶段）</p>
        {wakes.length ? (
          <ul className="mt-1 space-y-0.5 text-sm text-mast-text">
            {wakes.map(([stage, n]) => (
              <li key={stage} className="tabular-nums">
                {stage}：{n} 次
              </li>
            ))}
          </ul>
        ) : (
          <p className="mt-1 text-sm text-mast-muted">还没有唤醒过</p>
        )}
        <p className="mt-1 text-xs text-mast-faint">
          闸门里的 LLM **只选路不给数**，每条路的动作参数都是 spec 预写的。
        </p>
      </Card>

      <Card>
        <p className="text-xs text-mast-faint">实验文件夹</p>
        {folder?.path ? (
          <>
            <p className="mt-1 break-all font-mono text-xs text-mast-muted">{folder.path}</p>
            <p className="mt-1 text-sm text-mast-text">
              {folder.spec_doc || "还没有快照"}
              　·　progress{" "}
              {folder.progress_lines == null ? "读不到" : `${folder.progress_lines} 行`}
            </p>
          </>
        ) : (
          <p className="mt-1 text-sm text-mast-warn">{folder?.reason || "没有落点"}</p>
        )}
        <p className="mt-1 text-xs text-mast-faint">
          人读副本，不是真源；恢复流程不读它。
        </p>
      </Card>
    </div>
  );
}

/** 本次响应算不出来的东西。**空着不画**,有就必须显示 —— 不当没这回事。 */
export function NotAvailable({ d }: { d: Detail }) {
  if (!d.not_available?.length) return null;
  return (
    <div className="rounded-mast-ctl border border-dashed border-mast-border-strong px-3 py-2 text-xs text-mast-muted">
      <span className="text-mast-faint">这次没能算出来：</span>
      {d.not_available.join("；")}
    </div>
  );
}
