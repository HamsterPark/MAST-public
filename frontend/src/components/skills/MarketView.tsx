import { useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { type ColumnDef } from "@tanstack/react-table";
import { api } from "@/api/client";
import { DataTable } from "@/components/DataTable";
import {
  Section,
  Card,
  Badge,
  Spinner,
  ErrorNote,
  DegradedNote,
  EmptyNote,
} from "@/components/ui";
import {
  Button,
  Modal,
  RadioGroup,
  SelectField,
  TextField,
  Toggle,
  useToast,
} from "@/components/controls";
import { useWsEvent } from "@/hooks/useWsEvents";
import { Field } from "@/components/controls";
import { PendingRecommendations } from "./PendingRecommendations";
import {
  canUnsubscribe,
  diffManifestAgainstCatalog,
  liveBadge,
  marketWriteProblem,
  parseSubscriptionManifest,
  pendingBadgeCount,
  locallyAuthoredUnsubscribed,
  type MarketWriteResult,
  type SubscriptionManifest,
} from "@/lib/skillMarket";

// ── 技能市场 —— 全体技能是市场，订阅列表是他要用的那一份 ─────────────────────
//
// 这个页面的第一职责和覆盖层面板一样：**回答「现在生效了吗」**。改订阅只改了一个
// holder，而 agent 的工具表是建图时冻结的 —— 一次成功的写入之后，模型手上那张表
// 可能还是旧的（任务在跑 ⇒ 排队；重建失败；没有活的运行时）。三条路径都返回
// ok:true，所以这里按 agent_path_pending / fingerprint_matches 显示，**不按 ok**，
// 并且把后端那句 rebuild_note **逐字**摆出来（三种情况的措辞不一样，前端改写会把
// 它们压成一种）。
//
// 第二职责是把「订阅不是安全机制」这件事说清楚：未订阅的技能仍然能在技能直调页
// 手动执行、仍然能被 composite 子步与 conduct 调到。界面上不说，用户就会把它
// 当成一道权限门来用。

type MarketRow = {
  name: string;
  zh: string;
  category: string;
  safety: string;
  level: number;
  source: string;
  source_zh: string;
  tags: string[];
  domain: string;
  subscribed: boolean;
  mandatory: boolean;
  pending_rec_id: string;
};

type Status = {
  customised: boolean;
  subscribed_count: number;
  market_total: number;
  mandatory: string[];
  missing_entries: string[];
  unreadable: string;
  pending_count: number;
  store_path: string;
  agent_path_pending: boolean | null;
  fingerprint_matches: boolean | null;
  degraded: boolean;
  reason: string;
};

type Rec = {
  id: string;
  skill: string;
  by_agent: string;
  reason: string;
  at: string;
  status: string;
  resolved_at: string | null;
};

const SAFETY_TONE: Record<string, string> = {
  auto: "AUTO",
  confirm: "WARN",
  dangerous: "DANGEROUS",
};

function useStatus() {
  return useQuery({
    queryKey: ["skill-market", "status"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skill-market/status");
      if (error) throw error;
      return data as unknown as Status;
    },
  });
}

function useMarket(view: "all" | "on" | "off") {
  return useQuery({
    queryKey: ["skill-market", "catalog", view],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skill-market/catalog", {
        params: { query: { subscribed: view === "all" ? "" : view === "on" ? "1" : "0" } },
      });
      if (error) throw error;
      return data as unknown as {
        total: number;
        skills: MarketRow[];
        customised: boolean;
        degraded: boolean;
        reason: string;
      };
    },
  });
}

function useRecommendations() {
  return useQuery({
    queryKey: ["skill-market", "recommendations"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skill-market/recommendations");
      if (error) throw error;
      return data as unknown as { pending: Rec[]; resolved: Rec[]; degraded: boolean };
    },
  });
}

/** 生效状态条 —— 权威的三态显示。toast 只有两档，装不下它。 */
function EffectBanner({ st }: { st: Status }) {
  const live = liveBadge(st.fingerprint_matches);
  const tone = st.agent_path_pending ? "WARN" : live.tone === "ok" ? "AUTO" : "INFO";
  return (
    <Card>
      <div className="flex flex-wrap items-center gap-3 text-sm">
        <Badge tone={tone}>
          {st.agent_path_pending ? "agent 侧尚未跟上" : `agent 工具表：${live.text}`}
        </Badge>
        <span className="text-mast-muted">
          {st.customised
            ? `已定制：${st.subscribed_count} / ${st.market_total} 个技能在装载面上`
            : `未定制：全部 ${st.market_total} 个技能都在装载面上（出厂默认）`}
        </span>
        {st.missing_entries.length > 0 && (
          <Badge tone="INFO">
            {st.missing_entries.length} 个订阅条目在本机找不到对应技能
          </Badge>
        )}
      </div>
      {st.unreadable && (
        <p className="mt-2 text-sm text-mast-warn">
          ⚠ 订阅文件读不出来（{st.unreadable}）—— 现在按**全订阅**在跑，不是空订阅。
          修好 {st.store_path} 后重载即可。
        </p>
      )}
      <p className="mt-2 text-xs text-mast-muted">
        订阅决定的是 agent 工具表里有什么，<b>不是权限</b>：未订阅的技能仍可在
        「技能直调」手动执行，也仍会被复合技能的子步与 conduct 调用。
      </p>
    </Card>
  );
}

/** 已裁决的推荐 —— **拒绝也留在这里**。留痕的意思是能看见「他拒过」。 */
function ResolvedRecommendations({ recs }: { recs: Rec[] }) {
  if (recs.length === 0) return null;
  return (
    <Section title={`已处理的推荐（${recs.length}）`}>
      <div className="flex flex-col gap-1 text-sm">
        {recs.slice().reverse().map((r) => (
          <div key={r.id} className="flex flex-wrap items-center gap-2">
            <Badge tone={r.status === "accepted" ? "AUTO" : "default"}>
              {r.status === "accepted" ? "已接受" : "已拒绝"}
            </Badge>
            <code className="text-xs">{r.skill}</code>
            <span className="text-xs text-mast-muted">
              {r.by_agent || "agent"} · {r.resolved_at || r.at}
            </span>
            {r.reason && <span className="text-xs text-mast-faint">{r.reason}</span>}
          </div>
        ))}
      </div>
    </Section>
  );
}

/** 订阅面的变更史。
 *
 * 这个组件是审计流**唯一的消费方**。第一版把 audit 写进了盘、也写了往返测试，
 * 然后没有任何东西读它 —— 一条只写不读的日志（[[producer_wired_consumer_absent]]，
 * 而且是在引用着那条纪律的同一天犯的）。
 */
const AUDIT_ZH: Record<string, string> = {
  materialise: "固化为明确名单",
  subscribe: "加入",
  unsubscribe: "移出",
  set: "整体替换",
  reset: "恢复全订阅",
  accepted: "接受推荐",
  rejected: "拒绝推荐",
};

function AuditTrail() {
  const [open, setOpen] = useState(false);
  const q = useQuery({
    queryKey: ["skill-market", "audit"],
    enabled: open,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skill-market/audit");
      if (error) throw error;
      return data as unknown as {
        entries: { at: string; action: string; skills: string[]; via: string }[];
        degraded: boolean;
        reason: string;
      };
    },
  });

  return (
    <Section
      title="变更史"
      subtitle="谁在什么时候动了订阅面、经的哪条路。reset 会清掉定制，但不清这份记录。"
      actions={<Button onClick={() => setOpen(!open)}>{open ? "收起" : "查看"}</Button>}
    >
      {open && (
        <Card>
          {q.isPending && <Spinner />}
          {q.error && <ErrorNote error={q.error} />}
          {q.data?.degraded && <DegradedNote what="变更史" />}
          {q.data && !q.data.degraded && q.data.entries.length === 0 && (
            <EmptyNote label="还没有改动过订阅面" />
          )}
          <div className="flex flex-col gap-1 text-xs">
            {(q.data?.entries ?? []).slice().reverse().map((a, i) => (
              <div key={`${a.at}-${i}`} className="flex flex-wrap items-center gap-2">
                <span className="text-mast-faint">{a.at}</span>
                <Badge tone="default">{AUDIT_ZH[a.action] ?? a.action}</Badge>
                {a.skills.length > 0 && (
                  <span className="text-mast-muted">
                    {a.skills.slice(0, 6).join("、")}
                    {a.skills.length > 6 ? ` …共 ${a.skills.length} 个` : ""}
                  </span>
                )}
                <span className="flex-1" />
                <span className="text-mast-faint">{a.via}</span>
              </div>
            ))}
          </div>
        </Card>
      )}
    </Section>
  );
}

/** 实验室中心索引：把自己的订阅列表发上去，或看看别人发了什么。 */
function LabShare({ onFetched }: { onFetched: (m: SubscriptionManifest) => void }) {
  const { toast, node: toastNode } = useToast();
  const [label, setLabel] = useState("");
  const [note, setNote] = useState("");
  const [browsing, setBrowsing] = useState(false);

  const index = useQuery({
    queryKey: ["skill-market", "lab-index"],
    enabled: browsing,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/skill-market/lab-index");
      if (error) throw error;
      return data as unknown as {
        subscriptions: {
          id: string; label: string; note: string; machine: string;
          skill_count: number; embedded_specs: number; status: string; ts_server: string;
        }[];
        degraded: boolean;
        reason: string;
      };
    },
  });

  const publish = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/skill-market/share/publish", {
        body: { label, note },
      });
      if (error) throw error;
      return data as unknown as { ok: boolean; reason: string; id: string; skill_count: number };
    },
    onSuccess: (d) => {
      // 服务器的拒绝理由**原样**显示 ——「代码不随单走」那条红线的报文就在里面，
      // 压成一句「发布失败」会让人下次还这么发。
      if (!d.ok) { toast(d.reason || "发布失败", "err"); return; }
      toast(`已发布（${d.id}，${d.skill_count} 个技能），等管理员审核`, "ok");
      setLabel(""); setNote("");
    },
    onError: (e: any) => toast(String(e?.message || e), "err"),
  });

  const fetchOne = useMutation({
    mutationFn: async (id: string) => {
      const { data, error } = await api.GET("/api/skill-market/lab-fetch/{sub_id}", {
        params: { path: { sub_id: id } },
      });
      if (error) throw error;
      return data as unknown as {
        ok: boolean; reason: string; manifest: SubscriptionManifest | null;
      };
    },
    onSuccess: (d) => {
      // 失败有自己的字段 —— 不靠「manifest 里某个位空不空」来推断成没成。
      if (!d.ok || !d.manifest) { toast(d.reason || "取回失败", "err"); return; }
      // 取回 ≠ 导入：换掉工作面要他自己按一次（同覆盖层「写清单是一次编辑，
      // 让它生效是一次决定」）。所以这里只把它放进导入预览。
      onFetched(d.manifest);
    },
    onError: (e: any) => toast(String(e?.message || e), "err"),
  });

  return (
    <Section
      title="实验室中心索引"
      subtitle="把你的订阅列表分享给同组的人，或看看别人在用哪些技能"
      actions={
        <Button onClick={() => setBrowsing(!browsing)}>
          {browsing ? "收起" : "浏览别人的"}
        </Button>
      }
    >
      {toastNode}
      <Card>
        <div className="flex flex-wrap items-end gap-2">
          <div className="w-48">
            <Field label="名字"><TextField value={label} onChange={setLabel}
              placeholder="如「qPlus 日常」" /></Field>
          </div>
          <div className="min-w-0 flex-1">
            <Field label="说明"><TextField value={note} onChange={setNote}
              placeholder="一句话说明这份列表适合谁" /></Field>
          </div>
          <Button variant="primary" loading={publish.isPending}
                  onClick={() => publish.mutate()}>发布我的订阅</Button>
        </div>
        <p className="mt-2 text-xs text-mast-muted">
          发布的是<b>本机导出的那一份</b>（与「导出订阅」逐字节相同）。
          代码（.py）不随单走 —— 只有用户组合技能会带上它的纯数据 spec。
        </p>
      </Card>

      {browsing && (
        <div className="mt-3">
          {index.isPending && <Spinner />}
          {index.error && <ErrorNote error={index.error} />}
          {index.data?.degraded && (
            <Card><p className="text-sm text-mast-warn">{index.data.reason}</p></Card>
          )}
          {index.data && !index.data.degraded && index.data.subscriptions.length === 0 && (
            <EmptyNote label="中心索引里还没有订阅列表" />
          )}
          <div className="flex flex-col gap-2">
            {(index.data?.subscriptions ?? []).slice().reverse().map((s) => (
              <Card key={s.id}>
                <div className="flex flex-wrap items-center justify-between gap-2">
                  <div className="min-w-0">
                    <div className="flex items-center gap-2">
                      <b className="text-sm">{s.label || s.id}</b>
                      <Badge tone="INFO">{s.skill_count} 个技能</Badge>
                      {s.embedded_specs > 0 && (
                        <Badge tone="default">含 {s.embedded_specs} 个组合</Badge>
                      )}
                    </div>
                    <p className="mt-0.5 text-xs text-mast-muted">
                      {s.note || "（没写说明）"} · {s.machine} · {s.ts_server}
                    </p>
                  </div>
                  <Button disabled={fetchOne.isPending}
                          onClick={() => fetchOne.mutate(s.id)}>取回并预览</Button>
                </div>
              </Card>
            ))}
          </div>
        </div>
      )}
    </Section>
  );
}

export function MarketView() {
  const qc = useQueryClient();
  const { toast, node: toastNode } = useToast();
  const [view, setView] = useState<"all" | "on" | "off">("all");
  const [search, setSearch] = useState("");
  const [source, setSource] = useState("");
  const [note, setNote] = useState("");
  const [confirmReset, setConfirmReset] = useState(false);
  const [importPreview, setImportPreview] = useState<SubscriptionManifest | null>(null);
  const [importMode, setImportMode] = useState<"replace" | "merge">("replace");
  const fileRef = useRef<HTMLInputElement | null>(null);

  const st = useStatus();
  const cat = useMarket(view);
  const recs = useRecommendations();

  // 帧只做触发：收到就 refetch，不按帧累积列表（总线只重放 100 条）。
  useWsEvent("skill_recommendation", () => {
    qc.invalidateQueries({ queryKey: ["skill-market", "recommendations"] });
    qc.invalidateQueries({ queryKey: ["skill-market", "catalog"] });
  });

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["skill-market"] });
    qc.invalidateQueries({ queryKey: ["skills", "catalog"] });
    qc.invalidateQueries({ queryKey: ["agents", "tools"] });
  };

  /** 一次写入之后统一走这里 —— 判据只有一份（lib/skillMarket）。 */
  const settle = (d: MarketWriteResult | undefined) => {
    const problem = marketWriteProblem(d);
    // 后端那句人话逐字显示。三种「没生效」的措辞不一样，改写会把它们压成一种。
    setNote(d?.rebuild_note || problem?.message || "");
    refresh();
    if (!problem) {
      toast(d?.rebuild_note || "已生效", "ok");
      return;
    }
    // toast 只有 ok/err 两档，装不下三态。pending / unsure 一律走 err：宁可看
    // 起来重一点，也不能让「已安排」被读成「已完成」—— 用户会基于「已生效」
    // 去做下一件事。权威的三态在上面的 EffectBanner。
    toast(problem.message, "err");
  };

  const write = useMutation({
    mutationFn: async (v: { subscribe?: string[]; unsubscribe?: string[] }) => {
      const { data, error } = await api.POST("/api/skill-market/subscription", {
        body: { subscribe: v.subscribe ?? [], unsubscribe: v.unsubscribe ?? [] },
      });
      if (error) throw error;
      return data as unknown as MarketWriteResult;
    },
    onSuccess: settle,
    onError: (e: any) => toast(String(e?.message || e), "err"),
  });

  const reset = useMutation({
    mutationFn: async () => {
      const { data, error } = await api.POST("/api/skill-market/subscription/reset", {});
      if (error) throw error;
      return data as unknown as MarketWriteResult;
    },
    onSuccess: (d) => {
      setConfirmReset(false);
      settle(d);
    },
    onError: (e: any) => toast(String(e?.message || e), "err"),
  });

  // 裁决推荐这件事**只有一份实现**（`PendingRecommendations`），聊天页与这里共用。
  // 这个页面此前有自己的一份 —— 两份各自 toast、各自失效缓存，迟早只有一份是对的。

  const doImport = useMutation({
    mutationFn: async (v: { manifest: SubscriptionManifest; dry: boolean }) => {
      const { data, error } = await api.POST("/api/skill-market/import", {
        body: { manifest: v.manifest as any, mode: importMode, dry_run: v.dry },
      });
      if (error) throw error;
      return data as unknown as MarketWriteResult & { report?: any; dry_run?: boolean };
    },
    onSuccess: (d) => {
      if (d?.dry_run) {
        const r = d.report ?? {};
        toast(`预览：可用 ${r.matched?.length ?? 0} 个，缺 ${r.missing?.length ?? 0} 个`, "ok");
        return;
      }
      setImportPreview(null);
      settle(d);
    },
    onError: (e: any) => toast(String(e?.message || e), "err"),
  });

  const onPickFile = async (f: File | null) => {
    if (!f) return;
    const parsed = parseSubscriptionManifest(await f.text());
    if ("error" in parsed) {
      toast(parsed.error, "err");
      return;
    }
    setImportPreview(parsed.manifest);
  };

  const exportNow = async () => {
    const { data, error } = await api.GET("/api/skill-market/export");
    if (error) {
      toast("导出失败", "err");
      return;
    }
    // 沙箱里 <a download> 是惰性的，但这是本地 SPA，不是 artifact —— 正常可用。
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "mast-skill-subscription.json";
    a.click();
    URL.revokeObjectURL(url);
  };

  const rows = useMemo(() => {
    const all = cat.data?.skills ?? [];
    if (!source) return all;
    return all.filter((r) => r.source === source);
  }, [cat.data, source]);

  const sources = useMemo(
    () => Array.from(new Set((cat.data?.skills ?? []).map((r) => r.source))).sort(),
    [cat.data],
  );

  const columns = useMemo<ColumnDef<MarketRow, any>[]>(
    () => [
      {
        header: "订阅",
        accessorKey: "subscribed",
        cell: ({ row }) => {
          const r = row.original;
          const locked = !canUnsubscribe(r);
          return (
            <div className="flex items-center gap-2">
              <Toggle
                checked={r.subscribed}
                label={`订阅 ${r.name}`}
                onChange={(v) => {
                  if (locked && !v) {
                    toast(`${r.name} 是必装技能，不能退订 —— 它保证工具面上永远有「停下来 / 退针」可用`, "err");
                    return;
                  }
                  write.mutate(v ? { subscribe: [r.name] } : { unsubscribe: [r.name] });
                }}
              />
              {locked && <Badge tone="INFO">必装</Badge>}
            </div>
          );
        },
      },
      { header: "技能", accessorKey: "name",
        cell: ({ row }) => (
          <div className="min-w-0">
            <code className="text-sm">{row.original.name}</code>
            {row.original.zh && (
              <div className="text-xs text-mast-muted">{row.original.zh}</div>
            )}
          </div>
        ) },
      { header: "领域", accessorKey: "domain" },
      { header: "来源", accessorKey: "source_zh" },
      {
        header: "安全级",
        accessorKey: "safety",
        cell: ({ getValue }) => {
          const v = String(getValue() ?? "");
          return <Badge tone={SAFETY_TONE[v] ?? "default"}>{v.toUpperCase() || "?"}</Badge>;
        },
      },
      {
        header: "推荐",
        accessorKey: "pending_rec_id",
        cell: ({ getValue }) =>
          getValue() ? <Badge tone="WARN">待确认</Badge> : null,
      },
    ],
    [write, toast],
  );

  if (st.isLoading) return <Spinner />;
  if (st.error) return <ErrorNote error={st.error} />;
  if (st.data?.degraded) return <DegradedNote what="技能市场" />;

  const pendingRecs = recs.data?.pending ?? [];
  // 派生视图：本机长出来、却不在工具面上的技能（判据在 lib，可被 node --test 覆盖）。
  const localNew = useMemo(
    () => locallyAuthoredUnsubscribed(cat.data?.skills ?? []),
    [cat.data],
  );
  const missing = st.data?.missing_entries ?? [];

  return (
    <div className="flex flex-col gap-4">
      {toastNode}
      {st.data && <EffectBanner st={st.data} />}
      {note && (
        <Card>
          <p className="text-sm">
            <span className="text-mast-muted">上一次操作：</span>
            {note}
          </p>
        </Card>
      )}

      {/* 本机长出来、却不在工具面上的技能。定制过订阅之后，agent 用技能工坊新造的
          组合技能只进市场不进工具面 —— 而没有任何东西会告诉用户它出现了。
          这里是一个**派生视图**，不是一道闸：真正缺的是可见性（闸会把工具面重新
          变成安全边界，2026-08-20 已经拆过一次）。 */}
      {localNew.length > 0 && (
        <Section
          title={`本机新出现、尚未在工具面上（${localNew.length}）`}
          subtitle="用户/agent 在这台机器上造的技能。它们不在 agent 的工具表里，但仍可手动执行、也仍可被工作流调用。"
          actions={
            <Button
              variant="primary"
              disabled={write.isPending}
              onClick={() => write.mutate({ subscribe: localNew.map((r) => r.name) })}
            >
              全部加入工具面
            </Button>
          }
        >
          <Card>
            <div className="flex flex-col gap-1 text-sm">
              {localNew.map((r) => (
                <div key={r.name} className="flex flex-wrap items-center gap-2">
                  <code className="text-xs">{r.name}</code>
                  {r.zh && <span className="text-xs text-mast-muted">{r.zh}</span>}
                  <Badge tone="INFO">{r.source_zh || r.source}</Badge>
                  <span className="flex-1" />
                  <Button disabled={write.isPending}
                          onClick={() => write.mutate({ subscribe: [r.name] })}>
                    加入
                  </Button>
                </div>
              ))}
            </div>
          </Card>
        </Section>
      )}

      {missing.length > 0 && (
        <Section
          title={`订阅里有、本机却没有的条目（${missing.length}）`}
          subtitle="技能集合是动态的（覆盖层卸载 / 包被移除 / 自建技能停用），所以这些名字刻意保留不剔 —— 但要让你看得见是哪几个。"
          actions={
            <Button
              disabled={write.isPending}
              onClick={() => write.mutate({ unsubscribe: missing })}
            >
              从订阅里移除这些
            </Button>
          }
        >
          <Card>
            <div className="flex flex-wrap gap-2 text-xs">
              {missing.map((n) => <code key={n}>{n}</code>)}
            </div>
          </Card>
        </Section>
      )}

      {pendingRecs.length > 0 && (
        <Section title={`待确认的推荐（${pendingRecs.length}）`}>
          <PendingRecommendations />
        </Section>
      )}
      <ResolvedRecommendations recs={recs.data?.resolved ?? []} />
      <AuditTrail />
      <LabShare onFetched={(m) => { setImportMode("replace"); setImportPreview(m); }} />

      <Section
        title={`市场（${cat.data?.total ?? 0}）`}
        actions={
          <div className="flex flex-wrap items-center gap-2">
            <Button onClick={exportNow}>导出订阅</Button>
            <Button onClick={() => fileRef.current?.click()}>导入订阅</Button>
            <Button variant="danger" onClick={() => setConfirmReset(true)}
                    disabled={!st.data?.customised}>
              恢复全订阅
            </Button>
          </div>
        }
      >
        <input
          ref={fileRef}
          type="file"
          accept="application/json,.json"
          className="hidden"
          onChange={(e) => {
            void onPickFile(e.target.files?.[0] ?? null);
            e.target.value = "";
          }}
        />
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <RadioGroup
            value={view}
            onChange={setView}
            options={[
              { value: "all", label: "全市场" },
              { value: "on", label: "已订阅" },
              { value: "off", label: "未订阅" },
            ]}
          />
          <div className="w-56">
            <TextField value={search} onChange={setSearch} placeholder="搜索技能名 / 中文名…" />
          </div>
          <div className="w-40">
            <SelectField
              value={source}
              onChange={setSource}
              options={[{ value: "", label: "全部来源" },
                        ...sources.map((s) => ({ value: s, label: s }))]}
            />
          </div>
          {pendingBadgeCount(pendingRecs) > 0 && (
            <Badge tone="WARN">{pendingBadgeCount(pendingRecs)} 条推荐待确认</Badge>
          )}
        </div>
        {cat.isLoading && <Spinner />}
        {cat.error && <ErrorNote error={cat.error} />}
        {cat.data?.degraded && <DegradedNote what="技能目录" />}
        {cat.data && !cat.data.degraded && rows.length === 0 && <EmptyNote />}
        {cat.data && !cat.data.degraded && rows.length > 0 && (
          <DataTable data={rows} columns={columns} globalFilter={search} />
        )}
      </Section>

      <Modal open={confirmReset} onClose={() => setConfirmReset(false)} title="恢复全订阅">
        <p className="text-sm">
          把订阅列表恢复成出厂态：<b>全部 {st.data?.market_total ?? 0} 个技能</b>都回到
          agent 的装载面上，你此前的定制会被丢弃（审计记录保留）。
        </p>
        <div className="mt-4 flex justify-end gap-2">
          <Button onClick={() => setConfirmReset(false)}>取消</Button>
          <Button variant="danger" loading={reset.isPending} onClick={() => reset.mutate()}>
            恢复
          </Button>
        </div>
      </Modal>

      <Modal open={!!importPreview} onClose={() => setImportPreview(null)}
             title="导入订阅列表" wide>
        {importPreview && (
          <ImportPreview
            manifest={importPreview}
            known={new Set((cat.data?.skills ?? []).map((r) => r.name))}
            mode={importMode}
            onMode={setImportMode}
            busy={doImport.isPending}
            onDryRun={() => doImport.mutate({ manifest: importPreview, dry: true })}
            onConfirm={() => doImport.mutate({ manifest: importPreview, dry: false })}
          />
        )}
      </Modal>
    </div>
  );
}

function ImportPreview({
  manifest, known, mode, onMode, busy, onDryRun, onConfirm,
}: {
  manifest: SubscriptionManifest;
  known: Set<string>;
  mode: "replace" | "merge";
  onMode: (v: "replace" | "merge") => void;
  busy: boolean;
  onDryRun: () => void;
  onConfirm: () => void;
}) {
  const diff = diffManifestAgainstCatalog(manifest, known);
  return (
    <div className="flex flex-col gap-3 text-sm">
      <div className="text-mast-muted">
        来自 <b>{manifest.machine || "未知机器"}</b>
        {manifest.exported_at ? `，导出于 ${manifest.exported_at}` : ""}
        ，共 {manifest.entries.length} 条。
      </div>
      <div className="flex flex-wrap gap-2">
        <Badge tone="AUTO">本机可用 {diff.matched.length}</Badge>
        <Badge tone={diff.missing.length ? "WARN" : "default"}>
          本机缺 {diff.missing.length}
        </Badge>
        {diff.embedded > 0 && <Badge tone="INFO">内嵌组合技能 {diff.embedded}</Badge>}
      </div>
      {diff.missing.length > 0 && (
        <div>
          <div className="mb-1 text-mast-muted">本机没有的（导入后会被列在报告里，不会写进订阅）：</div>
          <div className="max-h-40 overflow-auto rounded-mast-ctl border border-mast-border p-2">
            {diff.missing.slice(0, 50).map((m) => (
              <div key={m.name}>
                <code>{m.name}</code>{" "}
                <span className="text-xs text-mast-muted">{m.source}</span>
              </div>
            ))}
            {diff.missing.length > 50 && (
              <div className="text-xs text-mast-muted">…还有 {diff.missing.length - 50} 条</div>
            )}
          </div>
        </div>
      )}
      <p className="text-xs text-mast-muted">
        以上是<b>本地预览</b>：内嵌的组合技能会在导入时先落地，所以后端 dry-run 的
        数字可能比这里多。代码（.py）不随订阅单走 —— 覆盖层与自建技能需要单独获取。
      </p>
      <RadioGroup
        value={mode}
        onChange={onMode}
        options={[
          { value: "replace", label: "替换我的订阅" },
          { value: "merge", label: "并入我的订阅" },
        ]}
      />
      <div className="flex justify-end gap-2">
        <Button onClick={onDryRun} disabled={busy}>后端预览（不写入）</Button>
        <Button variant="primary" loading={busy} onClick={onConfirm}>导入</Button>
      </div>
    </div>
  );
}
