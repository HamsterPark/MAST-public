import { useCallback, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { Card, EmptyNote, ErrorNote, Section, Spinner } from "@/components/ui";
import { Button, SelectField, useToast } from "@/components/controls";
import { useWsConnection, useWsEvent } from "@/hooks/useWsEvents";
import {
  CONDUCT_FRAMES,
  CONDUCT_SETTINGS_TITLE,
  frameAction,
  statusLabel,
} from "@/lib/conduct";
import { DecisionCard } from "@/components/conduct/DecisionCard";
import { WaitCard } from "@/components/conduct/WaitCard";
import { OpBar } from "@/components/conduct/OpBar";
import { ApproveCard, type ApproveOutcome } from "@/components/conduct/ApproveCard";
import { NewConductCard } from "@/components/conduct/NewConductCard";
import {
  CurrentCard,
  GateHistory,
  NotAvailable,
  StallBanner,
  StatusHeader,
  Timeline,
  Vitals,
} from "@/components/conduct/PanelBody";

// ════════════════════════════════════════════════════════════════════════════
// 多天 conduct 面板 —— 用户盯一次跑三天的实验时看的那一页。
//
// ── 一个端点 = 一个时刻 ─────────────────────────────────────────────────────
//
// 整个面板由 `GET /api/conducts/{id}` 一次响应渲染(设计 §7)。这不是省事:面板由
// N 个端点拼起来时每个端点各有各的时刻,于是「状态是 RUNNING」和「等待卡还亮着」
// 可以同时显示,而两者都各自没错。
//
// ── WS 帧只做触发,不做增量状态源 ────────────────────────────────────────────
//
// 收到 conduct 帧 ⇒ **invalidate 那个 query**,一个字段都不从帧里取。理由是这条
// 总线只重放最后 100 条:一个跑三天的 conduct 必然丢帧,而按帧累积状态的客户端会
// 带着一个**错的**状态一直显示下去 —— 比没有实时推送糟得多。
//
// 这与电流监控页刻意相反(那边 `patchStatusFromWsEvent` 真的把帧并进缓存)。那边
// 的帧是 1 Hz 的连续标量流,丢一帧下一帧就补上;这边的帧是状态**转移**,丢一帧
// 就永远缺一块。同一个总线,两种正确的用法,差别写在这里免得有人「统一」它们。
//
// ── 引擎关着 ≠ 没有 conduct ────────────────────────────────────────────────
//
// `conduct.cd_enabled` 默认关。关着时读端点回 200 + degraded + 一句为什么,
// 这一页把那句话显示出来并给出去哪里开 —— **不是一片空白**。空白看起来像
// 「这个功能坏了」,而真相是「它还没被打开」。
//
// ── UI 绝不冻结 ─────────────────────────────────────────────────────────────
//
// 每个按钮最坏的结果是「无反应 + 一句 toast」。意图入队失败不锁界面、不转圈等
// 一个不会来的回包;没有活跃 conduct 时整页照常渲染(空态),而不是卡在 loading。
// ════════════════════════════════════════════════════════════════════════════

const LIST_KEY = ["conduct", "list"] as const;
const CONFIG_KEY = ["conduct", "config"] as const;

/** 面板轮询周期。WS 活着时放慢到保活档 —— 帧会负责及时性。 */
const DETAIL_POLL_MS = 5_000;
const DETAIL_WS_KEEPALIVE_POLL_MS = 30_000;

export default function ConductPage() {
  const ws = useWsConnection();
  const qc = useQueryClient();
  const { toast, node } = useToast();
  // 选中项走 URL 而不是纯 state:这样它可以被收藏、被转发给下一个班的人,
  // 而心愿单里那条 `conduct:<id>` 的请求也就有了一个能点的去处。
  // 同 nav.ts 里「二级页是真的路由段」那条的理由 —— 一个存在组件 state 里的
  // 选择回答不了「这个链接指向哪一份 conduct」。
  const [params, setParams] = useSearchParams();
  const picked = params.get("id") ?? "";
  const setPicked = (id: string) => setParams(id ? { id } : {}, { replace: true });

  // 新建表单开着没有。**不进 URL**:`?id=` 指的是「看哪一份 conduct」,而一张
  // 填了一半的表不是一份 conduct —— 把它塞进同一个参数里,一条转发出去的链接
  // 会在别人那边打开一张空表,而收到链接的人以为自己看到的是那份 conduct。
  const [newOpen, setNewOpen] = useState(false);

  // ── 引擎开没开 ────────────────────────────────────────────────────────────
  const config = useQuery({
    queryKey: CONFIG_KEY,
    refetchInterval: 60_000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/conducts/config");
      if (error) throw error;
      return data;
    },
  });

  // ── 列表(单活跃不变式 ⇒ 多数时候只有一个) ────────────────────────────────
  const list = useQuery({
    queryKey: LIST_KEY,
    refetchInterval: ws.polling ? 15_000 : 60_000,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/conducts");
      if (error) throw error;
      return data;
    },
  });

  // 选中谁:用户点过的优先,否则那个未了结的,再否则最近一份。
  // 这里**不写进 state 的默认值**——写进去之后 conduct 结束、新建一份时,
  // 面板会固执地停在旧的那个上,而屏幕上没有任何东西说明它为什么不动。
  const activeId = list.data?.active_conduct_id || "";
  const rows = list.data?.conducts ?? [];
  const currentId = picked || activeId || rows[0]?.conduct_id || "";

  // ── 面板唯一数据源 ────────────────────────────────────────────────────────
  const detail = useQuery({
    queryKey: ["conduct", "detail", currentId],
    enabled: Boolean(currentId),
    refetchInterval: ws.polling ? DETAIL_POLL_MS : DETAIL_WS_KEEPALIVE_POLL_MS,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/conducts/{conduct_id}", {
        params: { path: { conduct_id: currentId } },
      });
      if (error) throw error;
      return data;
    },
  });

  // ── WS:只触发 refetch ────────────────────────────────────────────────────
  const onFrame = useCallback(
    (event: { type?: string; data: unknown }) => {
      if (frameAction(event.type ?? "", event.data, currentId) !== "refetch") return;
      qc.invalidateQueries({ queryKey: ["conduct", "detail", currentId] });
      // 列表里的状态列也会变(而且新建/中止会改 active_conduct_id)。
      qc.invalidateQueries({ queryKey: LIST_KEY });
    },
    [qc, currentId],
  );
  // 三种帧同一个处理器 —— 它们的区别在后端(谁触发的),对面板都是「去重读一次」。
  useWsEvent(CONDUCT_FRAMES[0], onFrame);
  useWsEvent(CONDUCT_FRAMES[1], onFrame);
  useWsEvent(CONDUCT_FRAMES[2], onFrame);

  // ── 意图 ──────────────────────────────────────────────────────────────────
  //
  // 全部只是**入队**:状态列的单写者是 Director,下一 tick 生效。所以成功的提示
  // 是「已排队」而不是「已暂停」—— 后者是一句还没发生的事。
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["conduct", "detail", currentId] });
    qc.invalidateQueries({ queryKey: LIST_KEY });
  };

  const op = useMutation({
    mutationFn: async (v: { path: "pause" | "resume" | "takeover"; by: string }) => {
      // 三条路径**逐字写出来**,不是拼字符串再 `as` 回去。拼出来的那种写法
      // 通不过类型检查,只能靠一个 cast —— 而那个 cast 会把「路径拼错了」
      // 从编译期错误降级成运行时 404,而 404 在这一页的表现是「按了没反应」。
      const body = { by: v.by, note: "" };
      const params = { path: { conduct_id: currentId } };
      const r =
        v.path === "pause"
          ? await api.POST("/api/conducts/{conduct_id}/pause", { params, body })
          : v.path === "resume"
            ? await api.POST("/api/conducts/{conduct_id}/resume", { params, body })
            : await api.POST("/api/conducts/{conduct_id}/takeover", { params, body });
      if (r.error) throw r.error;
      return r.data;
    },
    onSuccess: (d) => {
      toast(d?.queued ? "已排队，下一拍生效" : d?.errors?.[0] || "没有排上", d?.queued ? "ok" : "err");
      refresh();
    },
    onError: (e) => toast(`发不出去：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const abort = useMutation({
    mutationFn: async (reason: string) => {
      const { data, error } = await api.POST("/api/conducts/{conduct_id}/abort", {
        params: { path: { conduct_id: currentId } },
        body: { by: "operator", reason },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      // `abort_signalled` 说的是「当前那一步**已经**收到停止信号」——不等 tick。
      // 它和 `queued` 是两件事,分开报:只 queued 意味着此刻没有在跑的步。
      toast(
        d?.abort_signalled
          ? "已中止：当前这一步已收到停止信号"
          : d?.queued
            ? "已排队中止（当前没有在跑的步）"
            : d?.errors?.[0] || "没有排上",
        d?.queued ? "ok" : "err",
      );
      refresh();
    },
    onError: (e) => toast(`中止发不出去：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const ack = useMutation({
    mutationFn: async (v: { waitId: string; note: string }) => {
      const { data, error } = await api.POST("/api/conducts/{conduct_id}/ack", {
        params: { path: { conduct_id: currentId } },
        body: { wait_id: v.waitId, by: "operator", note: v.note },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      // 「还缺什么」当场回给按按钮的人 —— 双闸的另一半可能还没齐。
      const still = d?.wait_echo?.length ? `　还缺：${d.wait_echo.join("、")}` : "";
      toast(d?.ok ? `已确认。${still}` : d?.errors?.[0] || "确认没被接受", d?.ok ? "ok" : "err");
      refresh();
    },
    onError: (e) => toast(`确认发不出去：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const waive = useMutation({
    mutationFn: async (v: { waitId: string; reason: string }) => {
      const { data, error } = await api.POST("/api/conducts/{conduct_id}/waive-condition", {
        params: { path: { conduct_id: currentId } },
        body: { wait_id: v.waitId, by: "operator", reason: v.reason },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      toast(d?.queued ? "已记下：条件闸由人提供证据（留痕）" : d?.errors?.[0] || "没有排上",
        d?.queued ? "ok" : "err");
      refresh();
    },
    onError: (e) => toast(`发不出去：${String((e as Error)?.message ?? e)}`, "err"),
  });

  // 闸门判定的放行。与 waive 同一条纪律(留痕 + 持续显示),答的是另一个问题:
  // waive 说「这个物理条件的证据由我提供」,这个说「这一次裁决我看过了,继续」。
  //
  // 在它存在之前,被闸门停住的 conduct 只剩中止与接管两条路 —— 凌晨停下、
  // 早上人看了觉得没问题,唯一的选择是放弃这份 conduct。
  const override = useMutation({
    mutationFn: async (v: { decisionId: number; reason: string }) => {
      const { data, error } = await api.POST("/api/conducts/{conduct_id}/override-decision", {
        params: { path: { conduct_id: currentId } },
        body: { decision_id: v.decisionId, by: "operator", reason: v.reason },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      toast(
        d?.queued
          ? "已记下：这一次判定由人放行（留痕；下一次到这道闸还会重新判）"
          : d?.errors?.[0] || "没有排上",
        d?.queued ? "ok" : "err",
      );
      refresh();
    },
    onError: (e) => toast(`发不出去：${String((e as Error)?.message ?? e)}`, "err"),
  });

  // approve 是**人的动作**,而且这一页是唯一的入口(设计 §7:永不注册为 agent
  // 工具)。所以 409 不是异常路径,是这里最该说清楚的东西 —— 回包里的 findings
  // 与 checks_skipped 原样交给卡片显示,不并成一句「校验失败」。
  const [approveOutcome, setApproveOutcome] = useState<ApproveOutcome | null>(null);
  const approve = useMutation({
    mutationFn: async (by: string) => {
      const { data, error, response } = await api.POST(
        "/api/conducts/{conduct_id}/approve",
        {
          params: { path: { conduct_id: currentId } },
          body: { approved_by: by },
        },
      );
      // 409 的**回包体**才是有用的那部分(缺什么),所以这里不 throw,
      // 把它当成一个结果往下传。throw 掉的话屏幕上只剩一句「加载失败」。
      if (error && response.status !== 409) throw error;
      return (data ?? (error as ApproveOutcome)) as ApproveOutcome;
    },
    onSuccess: (d) => {
      setApproveOutcome(d);
      toast(
        d?.ok ? `已批准。快照：${d.spec_doc_path || "（未渲染）"}` : "批不下去，见下方原因",
        d?.ok ? "ok" : "err",
      );
      refresh();
    },
    onError: (e) => toast(`批准发不出去：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const attended = useMutation({
    mutationFn: async (next: boolean) => {
      const { data, error } = await api.POST("/api/conducts/{conduct_id}/attended", {
        params: { path: { conduct_id: currentId } },
        body: { attended: next, by: "operator" },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (d) => {
      toast(d?.queued ? "值守模式已排队" : d?.errors?.[0] || "没有排上", d?.queued ? "ok" : "err");
      refresh();
    },
    onError: (e) => toast(`发不出去：${String((e as Error)?.message ?? e)}`, "err"),
  });

  const busy = op.isPending || abort.isPending || ack.isPending || waive.isPending
    || attended.isPending || approve.isPending;

  // ── 引擎关着:说清楚,并给出去哪里开 ──────────────────────────────────────
  const engineOff = config.data && !config.data.enabled;
  const d = detail.data;

  return (
    <div className="mx-auto max-w-5xl px-4 py-6">
      <Section
        title="多天 conduct"
        subtitle="一次跑几天的实验：现在在哪一步、卡在什么上、要不要你确认"
        actions={
          <div className="flex items-center gap-2">
            {rows.length > 1 && (
              <div className="w-64">
                <SelectField
                  value={currentId}
                  onChange={setPicked}
                  options={rows.map((r) => ({
                    value: r.conduct_id,
                    label: `${r.title || r.spec_id}（${statusLabel(r.status)}）`,
                  }))}
                />
              </div>
            )}
            {/* 「新建」在有 conduct 的时候也要够得着 —— 只在空态给入口的话,
                跑完一份之后就再也建不了下一份了(而空态那时并不空)。 */}
            {!newOpen && (
              <Button onClick={() => setNewOpen(true)}>新建 conduct</Button>
            )}
          </div>
        }
      >
        {engineOff && (
          <div className="mb-4 rounded-mast-ctl border border-mast-warn-border bg-mast-warn-bg px-4 py-3 text-sm text-mast-warn">
            <p className="font-semibold">conduct 指挥线程未启用</p>
            <p className="mt-1">
              这不是故障，是<b>默认关</b>：打开之前系统里根本没有这条常驻线程。
              去「设置 → {CONDUCT_SETTINGS_TITLE}」打开<b>启用 conduct 指挥线程</b>。
              已有 conduct 的状态现在仍然读得到，中止也仍然按得下。
            </p>
          </div>
        )}

        {list.isPending && <Spinner label="载入 conduct 列表…" />}
        {list.isError && <ErrorNote error={list.error} />}

        {newOpen && (
          <div className="mb-4">
            <NewConductCard
              activeConductId={activeId}
              onClose={() => setNewOpen(false)}
              onCreated={(id) => {
                setNewOpen(false);
                // 建完就把面板切到新那一份上:它是**草稿**,下一步是批准,
                // 而批准卡就在那一页上。不切的话用户刚建完的东西不在眼前。
                setPicked(id);
                refresh();
              }}
            />
          </div>
        )}

        {!list.isPending && !currentId && (
          <EmptyNote
            label={
              list.data?.degraded
                ? list.data.reason || "引擎未启用"
                : newOpen
                  ? "还没有 conduct —— 上面那张表填完就有了。"
                  : "还没有 conduct。点右上角「新建 conduct」。"
            }
          />
        )}

        {currentId && detail.isPending && <Spinner label="载入面板…" />}
        {currentId && detail.isError && <ErrorNote error={detail.error} />}

        {d && d.degraded && !d.ok && (
          <div className="mb-4 rounded-mast-ctl border border-mast-warn-border bg-mast-warn-bg px-4 py-3 text-sm text-mast-warn">
            {d.reason || "面板数据暂时读不到"}
          </div>
        )}

        {d?.ok && (
          <div className="space-y-4">
            <StallBanner d={d} />
            <StatusHeader d={d} />

            {/* 草稿的唯一出路。没有它,屏幕上会有一份永远停在「草稿」的
                conduct,而没有任何东西说明为什么推不动。 */}
            {d.status === "draft" && (
              <ApproveCard
                busy={busy}
                outcome={approveOutcome}
                onApprove={(by) => approve.mutate(by)}
              />
            )}

            {/* 等待大卡排在时间线之前:用户打开这一页时,「要不要我做点什么」
                是第一个问题。 */}
            {d.active_wait && (
              <WaitCard
                wait={d.active_wait}
                busy={busy}
                onAck={(note) =>
                  ack.mutate({ waitId: d.active_wait?.wait_id ?? "", note })
                }
                onWaive={(reason) =>
                  waive.mutate({ waitId: d.active_wait?.wait_id ?? "", reason })
                }
              />
            )}

            {/* 判定大卡:一次裁决转人。与等待大卡**互斥** —— 后端的
                pending_decision 在有 active_wait 时就是 null,所以这里不会
                两张卡同时出现。 */}
            {d.pending_decision && (
              <DecisionCard
                pending={d.pending_decision}
                busy={busy}
                onOverride={(reason) =>
                  override.mutate({
                    decisionId: d.pending_decision?.decision_id ?? 0,
                    reason,
                  })
                }
              />
            )}

            <CurrentCard d={d} />

            <Card>
              <OpBar
                status={d.status}
                attended={d.attended}
                busy={busy}
                onOp={(path) => op.mutate({ path, by: "operator" })}
                onAbort={(reason) => abort.mutate(reason)}
                onAttended={(next) => attended.mutate(next)}
              />
            </Card>

            <NotAvailable d={d} />

            <div>
              <h3 className="mb-2 text-sm font-semibold text-mast-muted">阶段进度</h3>
              <Timeline d={d} />
            </div>

            <Vitals d={d} />

            <div>
              <h3 className="mb-2 text-sm font-semibold text-mast-muted">闸门史（近 20 条）</h3>
              <Card>
                <GateHistory d={d} />
              </Card>
            </div>
          </div>
        )}
      </Section>

      {/* 手动刷新:WS 断了、轮询也被浏览器背景标签页压住的时候,总得有一条人能按的路。 */}
      <div className="flex items-center gap-3">
        <Button variant="ghost" onClick={refresh}>
          刷新
        </Button>
        <span className="text-xs text-mast-faint">
          {ws.warn ? "实时推送不可用，正在轮询" : "实时推送已连接（帧只用来触发重读）"}
        </span>
      </div>
      {node}
    </div>
  );
}
