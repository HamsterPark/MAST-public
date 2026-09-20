import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import clsx from "clsx";
import { api } from "@/api/client";
import type { components } from "@/api/schema";
import { Badge, ErrorNote, Section, Spinner } from "@/components/ui";
import { Accordion, Button, Field, SelectField, TextField, useToast } from "@/components/controls";
import { SignalIndexField } from "@/components/signals/SignalIndexField";
import { SIGNAL_AUTO_VALUE, SIGNAL_INDEX_KEYS } from "@/lib/signalChannels";
import { INSTRUMENT_INIT_KEY, useInstrumentInit } from "@/hooks/useInstrumentInit";
import { splitBold } from "@/lib/inlineBold";
import {
  groupsToOpen, isOutstanding, visibleItems, type InitItemLike,
} from "@/lib/initFilter";

// ── 新仪器初始化 ──────────────────────────────────────────────────────────────
// 设计文档：docs/v2/design/new_instrument_initialization.md
//
// 这一页回答一个问题：**装到一台新机器上，还差哪些数。**
//
// 形态刻意是「单页可滚动 + 分组」而不是 next/back 向导：向导对第一次装机友好，
// 对「我只想改一项破坏半径」是刑罚，而这一页两种用途都要担。
//
// ⚠️ 这一页填的数字**不经过任何 LLM**。人在输入框里打字 → HTTP JSON →
// Python 校验 → 既有 store 写入 → Python 回读 → Python 比对。
// 2026-08-03 有一次工具调用把 3e-12 发成 3（错 10¹² 倍），随后同一个模型又把
// 读回的 3.0 解释成「= 3e-12，浮点精度而已」。所以这里既没有「让 AI 帮你填」，
// 也没有把两个数丢给模型问「一致吗」——比对结果由后端算好，逐行显示在下面。
//
// 目录、区间、枚举选项全部由后端从真源现取（GET /api/instrument-init 的
// items[].input），**这个文件里一个区间都不抄** —— 抄一份就会漂。

type InitItem = components["schemas"]["InitItemModel"];
type InitGroup = components["schemas"]["InitGroupModel"];
type InputSpec = components["schemas"]["InputSpec"];
type Verdict = components["schemas"]["ValueVerdict"];

const FALLBACK_INPUT: InputSpec = { type: "float", min: null, max: null, choices: [] };

/** 输入规格来自后端真源；缺席时退成一个自由文本框，而不是猜一套区间。 */
const inputOf = (item: InitItem): InputSpec => item.input ?? FALLBACK_INPUT;

// tone 用的是 ui.tsx 的 BADGE_TONE 键名（AUTO / INFO / WARN / DANGEROUS）——
// 写错会静默退回 default，一个红色的「必填」就变成中性蓝，而没有任何报错。
const SEVERITY_META: Record<string, { label: string; tone: string; blurb: string }> = {
  required: {
    label: "必填",
    tone: "DANGEROUS",
    blurb: "缺了会让某条安全网失效，或者让一整类物理量整体错。",
  },
  recommended: {
    label: "推荐",
    tone: "WARN",
    blurb: "缺了功能会降级（并且会明说），但不会造成危险。",
  },
  optional: {
    label: "可跳过",
    tone: "INFO",
    blurb: "缺了只是少一句注释。系统不会因此发明一个数。",
  },
};

/** 值的显示形式。科学计数法照原样给出——这一页的读者要的就是量级。 */
function fmtValue(v: unknown): string {
  if (v === null || v === undefined || v === "") return "（未设置）";
  if (typeof v === "number") {
    if (!Number.isFinite(v)) return String(v);
    if (v !== 0 && (Math.abs(v) < 1e-3 || Math.abs(v) >= 1e5)) return v.toExponential(4);
    return String(v);
  }
  if (Array.isArray(v)) return `（${v.length} 项）`;
  if (typeof v === "object") return "（已填写）";
  return String(v);
}

/** 输入框里的数字：允许 `3e-12` 这类写法原样打进去，不做任何自动改写。 */
function parseNum(raw: string): number | null {
  const t = raw.trim();
  if (!t) return null;
  const n = Number(t);
  return Number.isFinite(n) ? n : null;
}

/** 「这个数比典型值大/小几个数量级」——只提示，不阻止。判据是后端给的区间。 */
function magnitudeHint(item: InitItem, n: number | null): string {
  if (n === null || n === 0) return "";
  const { min, max } = inputOf(item);
  if (typeof max === "number" && max > 0 && n > max) {
    const d = Math.log10(n / max);
    return `⚠ 比允许的上限大${d >= 1 ? ` 约 ${d.toFixed(1)} 个数量级` : ""}——保存时会被夹到上限，不会是你填的这个数。`;
  }
  if (typeof min === "number" && min > 0 && n < min) {
    const d = Math.log10(min / n);
    return `⚠ 比允许的下限小${d >= 1 ? ` 约 ${d.toFixed(1)} 个数量级` : ""}——保存时会被夹到下限，不会是你填的这个数。`;
  }
  return "";
}

function SeverityBadge({ severity }: { severity: string }) {
  const meta = SEVERITY_META[severity] ?? SEVERITY_META.recommended!;
  return <Badge tone={meta.tone}>{meta.label}</Badge>;
}

/** 后端文案用 `**强调**` 写重点；不渲染就是一行里夹着星号（#59 的一部分）。 */
function Bold({ text }: { text: string | null | undefined }) {
  return (
    <>
      {splitBold(text).map((s, i) =>
        s.bold ? <b key={i} className="text-mast-text">{s.text}</b> : <span key={i}>{s.text}</span>,
      )}
    </>
  );
}

function StatusDot({ item }: { item: InitItem }) {
  const tone =
    item.status === "set" ? "bg-mast-auto"
      : item.status === "acknowledged" ? "bg-mast-accent"
        : item.status === "n/a" ? "bg-mast-border"
          : item.severity === "required" ? "bg-mast-danger"
            : "bg-mast-warn";
  const title =
    item.status === "set" ? "你填过一个明确的值"
      : item.status === "acknowledged" ? "你核对过：出厂值就是对的"
        : item.status === "n/a" ? "这台机器不适用"
          : item.status === "missing" ? "没有出厂默认，也没填——硬缺口"
            : "还在用出厂默认，你没核对过";
  return <span className={clsx("inline-block h-2 w-2 shrink-0 rounded-full", tone)} title={title} />;
}

/**
 * 「填错会怎样」——**只有 `safety_critical` 的项有，而且常显**。
 *
 * 反复迭代之后剩下的规则只有这一条。其余项的
 * 后果一个字都不在页面上，原文躺在 `core/instrument_init.py` 各项上方的注释里。
 *
 * 常显而不是收进折叠区：填这一项的人多半是第一次装机，也就最不可能主动去点开。
 * `retract_motor_dir` 的「搞反了，一次退针 3000 步就是往样品里送 3000 步」
 * （2026-08-03 实测一步几百纳米）不该藏在一次点击后面。
 */
function ConsequenceLine({ item }: { item: InitItem }) {
  // Muted body, one warn-coloured marker. Painting fifteen required rows red
  // would be a different kind of noise than the one #59 is about — and the
  // catalog puts its `**` on the REASSURING half as often as the scary one
  // (retract_motor_dir emphasises 「运行时有兜底」), so emphasis must not be
  // wired to a danger colour. Severity already has a badge.
  return (
    <p className="mt-1.5 text-xs leading-relaxed text-mast-muted">
      <span className="mr-1 text-mast-warn">⚠</span>
      <Bold text={item.consequence} />
    </p>
  );
}

// 一行提示 —— 这一项在页面上的**全部**说明。
//
// 从前这里是一张卡片，四个带标签的段落：「这是什么 / 哪里找 / 填错会怎样 /
// 可对账」，冗余到有人会问这些说明为什么还没清理。
//
// 单纯删字不够 —— 因为**结构还在**：
// 有四个格子，就总有理由把每个格子填满，而「加起来太长了」不属于任何一次改动。
// 真正要删的是格子：后端合成了一个 `hint` 字段，这里就只剩一行。
//
// 「可对账」那一行是同一句话说第二遍：`probe` 和 `hint` 指的是同一个读法
// （`GetPiezoConfig.range` 的 Z 分量 ↔ GetPiezoConfig.range（Z 分量））。
// 它现在只喂上面那个「从仪器读一次」按钮，不上屏。
function HintLine({ item }: { item: InitItem }) {
  if (!item.hint) return null;
  return (
    <p className="mt-1 text-xs leading-relaxed text-mast-muted">
      <Bold text={item.hint} />
    </p>
  );
}

// ── 一行 ─────────────────────────────────────────────────────────────────────
function ItemRow({
  item, draft, onDraft, verdict,
}: {
  item: InitItem;
  draft: string | undefined;
  onDraft: (v: string | undefined) => void;
  verdict?: Verdict;
}) {
  // 「说明」这个折叠区整个没了（#39 / #40）。它是四段冗余说明的容器，而只要容器在，
  // 就总有人往里加一段 —— 前三轮删字全部长了回来。现在页面上一项只有：
  // 标签 ·（单位）· 级别 · 一行 hint ·（安全项才有的）方向性警示 · 输入框。
  //
  // 顺带少一次点击：从前「哪里读这个数」藏在折叠里，而那恰恰是填表时要看的东西。
  const showConsequence = item.safety_critical;
  const spec = inputOf(item);
  const kind = spec.type;
  const current = draft ?? (item.value === null || item.value === undefined ? "" : String(item.value));
  const hint = kind === "float" || kind === "int" ? magnitudeHint(item, parseNum(current)) : "";

  return (
    <div className={clsx(
      "border-t border-mast-border py-3 first:border-t-0",
      item.status === "n/a" && "opacity-50",
    )}>
      <div className="flex flex-wrap items-center gap-2">
        <StatusDot item={item} />
        <span className="text-sm font-medium text-mast-text">{item.label}</span>
        {item.unit && <span className="text-xs text-mast-faint">（{item.unit}）</span>}
        <SeverityBadge severity={item.severity} />
      </div>

      {item.status !== "n/a" && <HintLine item={item} />}
      {showConsequence && item.status !== "n/a" && <ConsequenceLine item={item} />}

      {item.status === "n/a" ? (
        // 「上面某一项」把读者留在一道 48 项的谜题前：是哪一项？改了它这一项会
        // 不会回来？后端现在指名道姓（instrument_init.na_reason），前端照说。
        <p className="mt-1 text-xs text-mast-faint">
          {item.na_reason || "这台机器不适用。"}
        </p>
      ) : kind === "table" ? (
        <p className="mt-2 text-xs text-mast-muted">
          这一项在<Link to="/settings/general" className="text-mast-accent hover:underline"> 设置 → 扫描参数档位 </Link>
          里编辑（那里是它的真源，`ScanAt` 每帧都从那里下发）。
          当前：{item.value ? `已自定义 ${(item.value as unknown[]).length} 档` : "在用出厂 4 档模板"}。
        </p>
      ) : kind === "modules" ? (
        <p className="mt-2 text-xs text-mast-muted">
          这一项在<Link to="/settings/general" className="text-mast-accent hover:underline"> 设置 → 硬件模块 </Link>
          里勾选（需要管理员 PIN）。
          当前：{item.value ? "已声明过" : "从未声明 —— 全部按「没装」处理"}。
        </p>
      ) : SIGNAL_INDEX_KEYS.has(item.key) ? (
        // 信号索引不让人填裸数字。这里按 key 分派而不是按
        // InputSpec 的 type —— 选项是**活的**（从仪器读回来的通道名单），不是
        // _CHOICE_SPEC 里的静态枚举，所以走不了 choice 那条路。
        <div className="mt-2 max-w-md">
          <SignalIndexField
            value={current}
            onCommit={(v) => onDraft(v)}
            autoValue={SIGNAL_AUTO_VALUE[item.key]}
            placeholder={item.factory !== null && item.factory !== undefined
              ? String(fmtValue(item.factory)) : "例如 86"}
          />
        </div>
      ) : kind === "choice" ? (
        <div className="mt-2 max-w-md">
          <SelectField
            value={current || String(item.factory ?? "")}
            onChange={(v) => onDraft(v)}
            options={[
              { value: "", label: "— 未选择 —" },
              ...(spec.choices ?? []).map((c) => ({ value: c.value ?? "", label: c.label ?? "" })),
            ]}
          />
        </div>
      ) : (
        <div className="mt-2 max-w-md">
          {/* 输入框收**自由文本**而不是 <input type=number>：科学计数法要能原样
              打进去（`3e-12`），而 number 输入在某些浏览器/输入法下会重写它。
              这一页的整个要点就是量级不能在传递途中被改写。 */}
          <TextField
            value={current}
            onChange={(v) => onDraft(v)}
            placeholder={item.factory !== null && item.factory !== undefined
              ? String(fmtValue(item.factory)) : "例如 1e9"}
            mono
          />
          <p className="mt-1 text-xs text-mast-faint">
            {item.factory !== null && item.factory !== undefined
              ? `留空 = 用出厂默认 ${fmtValue(item.factory)}`
              : "没有出厂默认——留空就是「未设置」，依赖它的功能会明说自己不可用。"}
          </p>
          {hint && <p className="mt-1 text-xs text-mast-danger">{hint}</p>}
        </div>
      )}

      {verdict && (
        <p className={clsx(
          "mt-2 rounded-mast-ctl border px-2.5 py-1.5 text-xs",
          verdict.verdict === "match"
            ? "border-mast-auto-border bg-mast-auto-bg text-mast-auto"
            : "border-mast-danger-border bg-mast-danger-bg text-mast-danger",
        )}>
          {verdict.verdict === "match"
            ? `已写入并回读确认：${fmtValue(verdict.stored)}`
            : verdict.note}
        </p>
      )}
    </div>
  );
}

// ── 一组 ─────────────────────────────────────────────────────────────────────
function GroupCard({
  group, items, groupItems, drafts, setDraft, verdicts, onSave, onAcknowledge, busy,
  open, onOpenChange,
}: {
  group: InitGroup;
  /** The rows to DRAW — already filtered by 待办/全部 and the search box. */
  items: InitItem[];
  /** Every item in this group, filter or no filter. */
  groupItems: InitItem[];
  drafts: Record<string, string>;
  setDraft: (id: string, v: string | undefined) => void;
  verdicts: Record<string, Verdict>;
  onSave: (items: InitItem[]) => void;
  onAcknowledge: (items: InitItem[]) => void;
  busy: boolean;
  open: boolean;
  onOpenChange: (open: boolean) => void;
}) {
  const [openIntro, setOpenIntro] = useState(false);
  // Counts and the SAVE SET come from the whole group, never the filtered view:
  // a draft typed before the operator changed the filter must still be written,
  // and a badge that shrinks with the filter is describing the filter, not the
  // machine.
  const outstanding = groupItems.filter((i) => !i.complete && i.severity === "required");
  const editable = groupItems.filter(
    (i) => inputOf(i).type !== "table" && inputOf(i).type !== "modules");
  const dirty = editable.some((i) => drafts[i.id] !== undefined);
  const defaulted = groupItems.filter((i) => i.status === "default" && i.severity !== "optional");
  const answered = groupItems.filter((i) => i.status === "set" || i.status === "acknowledged");
  const hidden = groupItems.length - items.length;

  // The intro is 2–6 lines of背景 per group. Show its first line and put the
  // rest behind a toggle: it explains WHY the group exists, which matters once
  // and then never again .
  const introLines = (group.intro || "").split("\n").filter((l) => l.trim());
  const introHead = introLines[0] ?? "";
  const introRest = introLines.slice(1);

  return (
    <Accordion
      open={open}
      onOpenChange={onOpenChange}
      title={
        <span className="flex flex-wrap items-center gap-2">
          <span>{group.title}</span>
          {outstanding.length > 0
            ? <Badge tone="DANGEROUS">{outstanding.length} 项必填未完成</Badge>
            : <Badge tone="AUTO">完成</Badge>}
          {/* A finished group collapses, and its contents then look like they do
              not exist — 2026-08-04 the operator could not find 退针方向 to
              change it and had to have it edited over the API. Say what is
              inside, on the closed header. */}
          {answered.length > 0 && (
            <span className="text-xs font-normal text-mast-muted">
              已填 {answered.length} 项，可随时改
            </span>
          )}
        </span>
      }
    >
      <p className="mb-1 text-xs leading-relaxed text-mast-muted">
        <Bold text={introHead} />
        {introRest.length > 0 && (
          <button
            type="button"
            onClick={() => setOpenIntro((v) => !v)}
            className="ml-2 text-mast-accent hover:underline"
          >
            {openIntro ? "收起" : "更多"}
          </button>
        )}
      </p>
      {openIntro && introRest.length > 0 && (
        <p className="mb-3 whitespace-pre-line text-xs leading-relaxed text-mast-muted">
          {introRest.map((l, i) => (
            <span key={i} className="block">
              <Bold text={l} />
            </span>
          ))}
        </p>
      )}
      {hidden > 0 && (
        <p className="mb-2 text-xs text-mast-faint">
          当前筛选下显示 {items.length} / {groupItems.length} 项。
        </p>
      )}
      <div className="mb-3" />
      <div>
        {items.map((it) => (
          <ItemRow
            key={it.id}
            item={it}
            draft={drafts[it.id]}
            onDraft={(v) => setDraft(it.id, v)}
            verdict={verdicts[it.id]}
          />
        ))}
      </div>
      <div className="mt-4 flex flex-wrap items-center gap-2 border-t border-mast-border pt-3">
        <Button variant="primary" disabled={!dirty || busy} loading={busy}
          onClick={() => onSave(editable)}>
          保存这一组并回读比对
        </Button>
        {defaulted.length > 0 && (
          <Button disabled={busy} onClick={() => onAcknowledge(defaulted)}>
            这一组我核对过了（出厂值就对）
          </Button>
        )}
        {defaulted.length > 0 && (
          <span className="text-xs text-mast-faint">
            「核对过」标记 {defaulted.length} 项还在用出厂默认的项。有默认的项，
            「没填」和「填的正好等于默认」在存储里是同一件事，程序分不出来——所以要你按一下。
          </span>
        )}
      </div>
    </Accordion>
  );
}

// ── 页面 ─────────────────────────────────────────────────────────────────────
export default function SetupPage() {
  const q = useInstrumentInit();
  const qc = useQueryClient();
  const { toast, node: toastNode } = useToast();
  const [drafts, setDrafts] = useState<Record<string, string>>({});
  const [verdicts, setVerdicts] = useState<Record<string, Verdict>>({});
  const [pin, setPin] = useState("");
  const [openAbout, setOpenAbout] = useState(false);
  // 待办 / 全部 + 搜索 —— 这一页要担两种用途（第一次装机 / 回来改一项），
  // 而这两种用途以前只有前一种好用：填过的项归进已完成的组、组默认收起，
  // 「退针方向」这类项因此找不到改的入口，只能靠 API 改。
  const [scope, setScope] = useState<"todo" | "all" | null>(null);
  const [query, setQuery] = useState("");
  const [openGroups, setOpenGroups] = useState<Record<string, boolean>>({});
  const [probeOut, setProbeOut] = useState<Record<string, unknown> | null>(null);
  const [imported, setImported] = useState<Record<string, unknown> | null>(null);
  const fileRef = useRef<HTMLInputElement | null>(null);

  const setDraft = (id: string, v: string | undefined) =>
    setDrafts((d) => {
      const next = { ...d };
      if (v === undefined) delete next[id];
      else next[id] = v;
      return next;
    });

  const invalidate = () => qc.invalidateQueries({ queryKey: INSTRUMENT_INIT_KEY });

  const applyMut = useMutation({
    mutationFn: async (payload: { store: string; values: Record<string, unknown> }) => {
      const { data, error } = await api.POST("/api/instrument-init/apply", {
        body: { ...payload, admin_pin: pin || null },
      });
      if (error) throw error;
      return data;
    },
  });

  const ackMut = useMutation({
    mutationFn: async (ids: string[]) => {
      const { data, error } = await api.POST("/api/instrument-init/acknowledge", {
        body: { item_ids: ids, undo: false },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: () => { invalidate(); toast("已记下：这些项你核对过了。"); },
  });

  const completeMut = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/instrument-init/complete", {
        body: { completed_by: "", rig_label: "" },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: () => { invalidate(); toast("已盖完成戳。换机器或重做标定后会自动重新提醒。"); },
  });

  // 「重新提醒」在 设置 页也有一个，但那正是最容易被忽略的入口。
  // 清掉完成戳，**一个数值都不动** —— 必填的判据从来不看这个戳。
  const reopenMut = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/instrument-init/reopen", {});
      if (error) throw error;
      return data;
    },
    onSuccess: () => { invalidate(); toast("完成戳已清除：提醒横幅会回来。数值一个都没动。"); },
    onError: () => toast("操作失败。", "err"),
  });

  const probeMut = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/instrument-init/probe", {});
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      setProbeOut(d as unknown as Record<string, unknown>);
      invalidate();
      if (!d?.ok) toast(d?.message || "没能从仪器读到任何量。", "err");
    },
    onError: () => toast("对账失败——检查 Nanonis 连接。", "err"),
  });

  const importMut = useMutation({
    mutationFn: async (bundle: Record<string, unknown>) => {
      const { data, error } = await api.POST("/api/instrument-init/import", {
        body: { bundle },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      if (!d?.ok) { toast(d?.rejected || "导入被拒绝。", "err"); return; }
      setImported(d.stores as Record<string, unknown>);
      // 导入 = 把数字填进表单，不是写进硬件。逐项进 draft，等用户逐组确认。
      const next: Record<string, string> = {};
      for (const [store, blob] of Object.entries(d.stores ?? {})) {
        if (store === "coarse_drive") continue;       // 必须重新签一次
        if (!blob || typeof blob !== "object") continue;
        for (const [k, v] of Object.entries(blob as Record<string, unknown>)) {
          if (v === null || v === undefined || typeof v === "object") continue;
          next[`${store}.${k}`] = String(v);
        }
      }
      setDrafts((cur) => ({ ...next, ...cur }));
      toast(`已载入待复核（${Object.keys(next).length} 项）。逐组确认后才会写下去。`);
    },
    onError: () => toast("导入失败。", "err"),
  });

  const data = q.data;
  const allItems = useMemo(() => (data?.items ?? []) as InitItem[], [data]);

  // 默认视图跟着这台机器的状态走：还有必填缺口 → 待办；已经装完 → 全部
  // （回来改一项的人打开页面就该看见全部，而不是一片空白）。
  const effScope = scope ?? (data?.needs_setup ? "todo" : "all");
  const searching = query.trim().length > 0;

  const visible = useMemo(
    () => visibleItems(allItems as unknown as InitItemLike[], { scope: effScope, query }),
    [allItems, effScope, query],
  );
  const autoOpen = useMemo(() => groupsToOpen(visible, { query }), [visible, query]);

  const byGroup = useMemo(() => {
    const m: Record<string, InitItem[]> = {};
    for (const it of visible as unknown as InitItem[]) (m[it.group] ??= []).push(it);
    return m;
  }, [visible]);

  const allByGroup = useMemo(() => {
    const m: Record<string, InitItem[]> = {};
    for (const it of allItems) (m[it.group] ??= []).push(it);
    return m;
  }, [allItems]);

  const outstandingCount = allItems.filter((i) =>
    isOutstanding(i as unknown as InitItemLike)).length;

  // Manual open/close is per-view. Without this a group the operator collapsed
  // once would stay shut when a later search puts its only hit inside it — the
  // search would look broken, which is the exact complaint being fixed.
  useEffect(() => { setOpenGroups({}); }, [query, effScope]);

  const saveGroup = async (items: InitItem[]) => {
    const byStore: Record<string, Record<string, unknown>> = {};
    for (const it of items) {
      const raw = drafts[it.id];
      if (raw === undefined) continue;
      const t = raw.trim();
      // 空 = 清掉这个键。后端 sanitize 对空值就是「不收」，语义一致。
      let val: unknown = t;
      const t2 = inputOf(it).type;
      if (t2 === "float" || t2 === "int") {
        const n = parseNum(t);
        if (t && n === null) { toast(`${it.label}：这不是一个数。`, "err"); return; }
        val = n;
      }
      if (t === "") val = null;
      (byStore[it.store] ??= {})[it.key] = val;
    }
    if (!Object.keys(byStore).length) return;

    const nextVerdicts: Record<string, Verdict> = {};
    let anyBad = false;
    let restart = false;
    for (const [store, values] of Object.entries(byStore)) {
      try {
        const res = await applyMut.mutateAsync({ store, values });
        if (res?.pin_required) {
          toast(res.message || "这一组需要管理员 PIN。", "err");
          return;
        }
        restart = restart || Boolean(res?.restart_required);
        for (const v of res?.verdicts ?? []) {
          nextVerdicts[`${store}.${v.key}`] = v as Verdict;
          if (v.verdict !== "match") anyBad = true;
        }
        if (!res?.ok && !res?.verdicts?.length) {
          anyBad = true;
          toast(res?.message || `${store} 写入失败。`, "err");
        }
      } catch {
        anyBad = true;
        toast(`${store} 写入失败。`, "err");
      }
    }
    setVerdicts((v) => ({ ...v, ...nextVerdicts }));
    if (!anyBad) {
      setDrafts((d) => {
        const next = { ...d };
        for (const it of items) delete next[it.id];
        return next;
      });
      toast(restart
        ? "已保存。安全包络覆写要重启才生效——本进程仍在用旧值。"
        : "已保存并回读确认。");
    } else {
      toast("有值回读回来和你填的不一样——看每一行下面的红字。", "err");
    }
    invalidate();
  };

  const doExport = async () => {
    const { data: d, error } = await api.GET("/api/instrument-init/export");
    if (error || !d?.ok) { toast("导出失败。", "err"); return; }
    const blob = new Blob([JSON.stringify(d.bundle, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `mast-instrument-init-${new Date().toISOString().slice(0, 10)}.json`;
    a.click();
    URL.revokeObjectURL(url);
    toast(d.stripped?.length
      ? `已导出。剥掉了 ${d.stripped.length} 个学习量（它们绑这根针/这台机器，搬过去就是错值）。`
      : "已导出。");
  };

  const onPickFile = (f: File | null) => {
    if (!f) return;
    const r = new FileReader();
    r.onload = () => {
      try {
        importMut.mutate(JSON.parse(String(r.result)));
      } catch {
        toast("这个文件不是合法的 JSON。", "err");
      }
    };
    r.readAsText(f);
  };

  if (q.isLoading) return <Section title="新仪器初始化"><Spinner /></Section>;
  if (q.error) return <Section title="新仪器初始化"><ErrorNote error={q.error} /></Section>;
  if (!data) return null;

  const req = data.counts?.required ?? { total: 0, complete: 0 };
  const rec = data.counts?.recommended ?? { total: 0, complete: 0 };
  const busy = applyMut.isPending;

  return (
    <Section title="新仪器初始化">
      {toastNode}

      {/* ── 顶部：进度 + 全局动作 ── */}
      <div className="sticky top-0 z-10 -mx-1 mb-4 rounded-mast-card border border-mast-border bg-mast-panel px-4 py-3 shadow-mast">
        <div className="flex flex-wrap items-center gap-3">
          <span className="text-sm">
            <b className={req.complete < req.total ? "text-mast-danger" : "text-mast-auto"}>
              必填 {req.complete}/{req.total}
            </b>
            <span className="ml-3 text-mast-muted">推荐 {rec.complete}/{rec.total}</span>
          </span>
          <div className="ml-auto flex flex-wrap gap-2">
            <Button onClick={() => probeMut.mutate()} loading={probeMut.isPending}>
              从仪器读一次对账
            </Button>
            <Button onClick={doExport}>导出</Button>
            <Button onClick={() => fileRef.current?.click()} loading={importMut.isPending}>
              导入
            </Button>
            <input ref={fileRef} type="file" accept="application/json,.json" className="hidden"
              onChange={(e) => { onPickFile(e.target.files?.[0] ?? null); e.target.value = ""; }} />
            {data.completed_at ? (
              <Button loading={reopenMut.isPending} onClick={() => reopenMut.mutate()}>
                让提醒横幅回来
              </Button>
            ) : (
              <Button variant="primary" disabled={data.needs_setup}
                loading={completeMut.isPending}
                onClick={() => completeMut.mutate()}>
                全部完成，别再提醒
              </Button>
            )}
          </div>
        </div>
        {data.needs_setup && (
          <p className="mt-2 text-xs text-mast-danger">
            还有 {(data.outstanding_required ?? []).length} 项必填没有答案。
            <b>「全部完成」按钮只抑制提醒，不豁免必填</b>——必填的判据永远从值本身算，
            所以它现在是禁用的。
          </p>
        )}
        {/* 完成之后这一页原来什么都不说，看起来就像「这里没事了」——而顶部横幅
            同时也消失了，用户的结论是「初始化页只能进去一次」（2026-08-04）。
            所以完成状态要**留在页面上**：谁盖的、什么时候、哪台机器，以及它
            抑制的到底是什么。 */}
        {!data.needs_setup && data.completed_at && (
          <p className="mt-2 text-xs text-mast-muted">
            <span className="text-mast-auto">已盖完成戳</span>
            <span className="ml-1">
              （{new Date(data.completed_at * 1000).toLocaleString("zh-CN", { hour12: false })}
              {data.rig_label ? ` · ${data.rig_label}` : ""}
              {data.rig_fingerprint ? ` · 指纹 ${data.rig_fingerprint}` : ""}）。
            </span>
            <b className="ml-1 text-mast-text">戳只抑制横幅，不锁任何一项</b>
            ——下面每一项随时可以改，改完立刻回读比对。
          </p>
        )}
        {data.fingerprint_changed && (
          <p className="mt-2 text-xs text-mast-warn">
            硬件指纹与上次完成初始化时不同：要么换了机器，要么有人重做了标定。
            压电量程 / Z 限位 / 前放增益值得重新过一遍。
          </p>
        )}
        {data.degraded && (
          <p className="mt-2 text-xs text-mast-warn">
            设置存储未接线——这份清单是只读的，填了不会保存。
          </p>
        )}
      </div>

      {/* ── 这一页是干什么的 ──
          原来这里是三段散文，头两句之后全是背景。留一行结论 +
          一行分级图例（那是图例，不是散文），其余进折叠。 */}
      <div className="mb-4 rounded-mast-card border border-mast-border bg-mast-panel-2 p-4 text-xs leading-relaxed text-mast-muted">
        <p className="flex flex-wrap items-center gap-x-1 gap-y-1">
          <span>
            下面每一项都是<b className="text-mast-text">「换一台仪器就会变成错的」</b>的数。
          </span>
          {Object.entries(SEVERITY_META).map(([k, m]) => (
            <span key={k} className="ml-1 inline-flex items-center gap-1" title={m.blurb}>
              <Badge tone={m.tone}>{m.label}</Badge>
            </span>
          ))}
          <button
            type="button"
            onClick={() => setOpenAbout((v) => !v)}
            className="ml-1 text-mast-accent hover:underline"
          >
            {openAbout ? "收起" : "分级与设计说明"}
          </button>
        </p>
        {openAbout && (
          <div className="mt-2 space-y-2 border-t border-mast-border pt-2">
            <ul className="space-y-0.5">
              {Object.entries(SEVERITY_META).map(([k, m]) => (
                <li key={k}>
                  <Badge tone={m.tone}>{m.label}</Badge>
                  <span className="ml-1">{m.blurb}</span>
                </li>
              ))}
            </ul>
            <p>
              这些数散在 6 个不同的存储里，默认值有三种完全不同的性质
              （出厂猜测 / 保守起点 / 根本没有默认）——这份清单就是把它们收在一处。
            </p>
            <p>
              <b className="text-mast-text">这一页填的数字不经过任何 AI。</b>
              你打的字直接进代码、代码直接写、写完立刻读回来比对，结果逐行显示在下面。
              （2026-08-03 有一次工具调用把 <code>3e-12</code> 发成了 <code>3</code>，
              而同一个模型随后把读回的 <code>3.0</code> 说成「等于 3e-12，浮点精度而已」
              ——所以比对绝不交给会犯错的那一方。）
            </p>
          </div>
        )}
      </div>

      {/* ── 前放交叉核对 ── */}
      {data.preamp_check?.consistent === false && (
        <div className="mb-4 rounded-mast-card border border-mast-danger-border bg-mast-danger-bg p-3 text-xs text-mast-danger">
          {data.preamp_check.note}
        </div>
      )}
      {(data.derived ?? []).length > 0 && (
        <div className="mb-4 rounded-mast-card border border-mast-border bg-mast-panel p-4 text-xs">
          <p className="font-semibold text-mast-text">由前放满量程导出的两条下游线</p>
          <p className="mt-1 text-mast-faint">
            这些是<b>建议值，不会自动写入</b>。一条被程序悄悄改过的安全上限，
            就不再是你声明过的那条线——所以按不按由你。
          </p>
          <ul className="mt-2 space-y-2">
            {(data.derived ?? []).map((d) => (
              <li key={d.target} className="flex flex-wrap items-center gap-2">
                <code className="text-mast-muted">{d.target}</code>
                <span>建议 <b className="font-mono">{fmtValue(d.value)}</b></span>
                <span className={clsx(d.current_is_looser ? "text-mast-danger" : "text-mast-faint")}>
                  当前 {fmtValue(d.current)}
                  {d.current_is_looser && "（比物理量程还宽——这是危险方向）"}
                </span>
                <Button
                  disabled={busy}
                  onClick={() => {
                    const dot = d.target.indexOf(".");
                    if (dot < 0) return;
                    void saveGroupDirect(d.target.slice(0, dot),
                      { [d.target.slice(dot + 1)]: d.value });
                  }}
                >
                  按建议收紧
                </Button>
                <span className="w-full text-mast-faint">{d.why}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* ── 对账结果 ── */}
      {probeOut && <ProbePanel probe={probeOut} />}

      {imported && (
        <div className="mb-4 rounded-mast-card border border-mast-warn-border bg-mast-warn-bg p-3 text-xs text-mast-warn">
          已载入一份导入包，值填在下面的输入框里但<b>还没有写下去</b>。
          请逐组核对后按「保存这一组」。
          {Object.keys(imported).includes("coarse_drive") && (
            <b className="block mt-1">
              粗动驱动电压没有被载入——那是唯一一个填错了叠堆当场报废的数，
              同型号也不等于同一台，必须重新签一次。
            </b>
          )}
        </div>
      )}

      {/* ── PIN ── */}
      <div className="mb-4 max-w-sm">
        <Field label="管理员 PIN（只有粗动驱动电压 / 硬件模块需要）"
          hint="没设 PIN 时这两项一律拒写（fail-closed）——不知道叠堆能承受多少，是不动的理由。">
          <input type="password" value={pin} onChange={(e) => setPin(e.target.value)}
            className="w-full rounded-mast-ctl border border-mast-border bg-mast-panel-2 px-3 py-2 text-sm" />
        </Field>
      </div>

      {/* ── 视图：待办 / 全部 + 搜索 ──
          「填过之后就找不到修改入口」的正面解法（2026-08-04）。搜索**不受
          待办/全部影响**：要找的那一项按定义已经填过了。 */}
      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="inline-flex overflow-hidden rounded-mast-ctl border border-mast-border">
          {([
            { id: "todo", label: `待办 (${outstandingCount})` },
            { id: "all", label: `全部 (${allItems.length})` },
          ] as const).map((t) => (
            <button
              key={t.id}
              type="button"
              onClick={() => setScope(t.id)}
              className={clsx(
                "px-2.5 py-1 text-xs",
                effScope === t.id && !searching
                  ? "bg-mast-accent-soft font-medium text-mast-accent"
                  : "text-mast-muted hover:text-mast-text",
              )}
            >
              {t.label}
            </button>
          ))}
        </div>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="搜索：退针 / preamp / z_extend_sign …"
          className="min-w-[13rem] flex-1 rounded-mast-ctl border border-mast-border bg-mast-panel-2 px-3 py-1.5 text-sm text-mast-text placeholder:text-mast-faint"
        />
        {searching && (
          <>
            <span className="text-xs text-mast-muted">
              全部 {allItems.length} 项中命中 {visible.length} 项（含已填）
            </span>
            <button
              type="button"
              onClick={() => setQuery("")}
              className="text-xs text-mast-accent hover:underline"
            >
              清除
            </button>
          </>
        )}
      </div>

      {/* ── 分组 ── */}
      {(data.groups ?? []).map((g) => {
        const items = byGroup[g.id] ?? [];
        // A group with nothing left after the filter is not drawn — but never
        // while searching with zero hits overall, which gets its own message.
        if (items.length === 0) return null;
        return (
          <GroupCard
            key={g.id}
            group={g}
            items={items}
            groupItems={allByGroup[g.id] ?? []}
            drafts={drafts}
            setDraft={setDraft}
            verdicts={verdicts}
            onSave={saveGroup}
            onAcknowledge={(its) => ackMut.mutate(its.map((i) => i.id))}
            busy={busy}
            open={openGroups[g.id] ?? autoOpen.has(g.id)}
            onOpenChange={(o) => setOpenGroups((s) => ({ ...s, [g.id]: o }))}
          />
        );
      })}

      {visible.length === 0 && (
        <p className="rounded-mast-card border border-mast-border bg-mast-panel-2 p-4 text-xs text-mast-muted">
          {searching
            ? `没有匹配「${query.trim()}」的项。搜索会查标签、存储键名和说明文字。`
            : "没有待办项——这台机器该填的都填了。切到「全部」可以复查或修改任何一项。"}
        </p>
      )}
    </Section>
  );

  // 单点写入（「按建议收紧」用）。走同一条 apply 通道，同样回读比对。
  async function saveGroupDirect(store: string, values: Record<string, unknown>) {
    try {
      const res = await applyMut.mutateAsync({ store, values });
      const next: Record<string, Verdict> = {};
      for (const v of res?.verdicts ?? []) next[`${store}.${v.key}`] = v as Verdict;
      setVerdicts((cur) => ({ ...cur, ...next }));
      toast(res?.message || (res?.ok ? "已保存并回读确认。" : "写入失败。"),
        res?.ok ? "ok" : "err");
      invalidate();
    } catch {
      toast("写入失败。", "err");
    }
  }
}

// ── 对账结果面板 ─────────────────────────────────────────────────────────────
function ProbePanel({ probe }: { probe: Record<string, unknown> }) {
  const fields = (probe.fields ?? []) as {
    key: string; label: string; read: unknown; configured: unknown;
    verdict: string; note: string;
  }[];
  const warnings = (probe.warnings ?? []) as string[];
  return (
    <div className="mb-4 rounded-mast-card border border-mast-border bg-mast-panel p-4 text-xs">
      <p className="font-semibold text-mast-text">从仪器读回来的对账结果</p>
      <p className="mt-1 text-mast-faint">
        <b>只报告不一致，什么都没改。</b>哪个数对由你定；若要改，只往收紧的方向改是安全的。
      </p>
      {probe.rig_fingerprint ? (
        <p className="mt-1 text-mast-faint">
          硬件指纹 <code>{String(probe.rig_fingerprint)}</code>
          {probe.fingerprint_changed ? "（与上次完成初始化时不同）" : ""}
        </p>
      ) : null}
      {fields.length > 0 && (
        <div className="mt-2 overflow-x-auto">
          <table className="w-full min-w-[520px] text-left">
            <thead className="text-mast-faint">
              <tr><th className="py-1">量</th><th>仪器上读到</th><th>MAST 登记</th><th>结论</th></tr>
            </thead>
            <tbody className="font-mono tabular-nums">
              {fields.map((f) => (
                <tr key={f.key} className="border-t border-mast-border">
                  <td className="py-1 font-sans">{f.label}</td>
                  <td>{fmtValue(f.read)}</td>
                  <td>{fmtValue(f.configured)}</td>
                  <td className={clsx("font-sans",
                    f.verdict === "match" ? "text-mast-auto"
                      : f.verdict === "mismatch" ? "text-mast-danger" : "text-mast-muted")}>
                    {f.verdict === "match" ? "一致"
                      : f.verdict === "mismatch" ? "不一致"
                        : f.verdict === "not_configured" ? "还没登记" : "没读到"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {fields.some((f) => f.verdict === "mismatch") && (
        <ul className="mt-2 space-y-1 text-mast-danger">
          {fields.filter((f) => f.verdict !== "match" && f.note).map((f) => (
            <li key={f.key}>{f.label}：{f.note}</li>
          ))}
        </ul>
      )}
      {warnings.map((w, i) => (
        <p key={i} className="mt-2 text-mast-warn">{w}</p>
      ))}
    </div>
  );
}
