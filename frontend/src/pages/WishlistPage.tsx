import { useState } from "react";
import { Link } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api } from "@/api/client";
import { conductIdFromAgentId, conductPanelHref } from "@/lib/conduct";
import {
  Section,
  Card,
  Badge,
  Spinner,
  ErrorNote,
  DegradedNote,
  EmptyNote,
} from "@/components/ui";
import type { components } from "@/api/schema";

// Domain H — 心愿单 (Wishlist). Two boards over the typed FastAPI seam:
//   1) 用户愿望/反馈：提交表单 + 列表（queued/sent/failed）→ 上报更新服务器。
//   2) Agent→用户请求：agent 需要配合时的请求，可标记 已完成 / 已忽略。
// All field names + enums are type-checked against the generated OpenAPI schema.

type Wish = components["schemas"]["Wish"];
type AgentRequest = components["schemas"]["AgentRequest"];

// ── status → (label, tailwind tone) — mirrors wishlist_panel.py palettes ────
const WISH_STATUS: Record<Wish["status"], { label: string; cls: string }> = {
  queued: { label: "待发送", cls: "bg-mast-warn-bg text-mast-warn" },
  sent: { label: "已上报", cls: "bg-mast-auto-bg text-mast-auto" },
  failed: { label: "上报失败", cls: "bg-mast-danger-bg text-mast-danger" },
};

const REQ_STATUS: Record<AgentRequest["status"], { label: string; cls: string }> = {
  pending: { label: "待处理", cls: "bg-mast-warn-bg text-mast-warn" },
  done: { label: "已完成", cls: "bg-mast-auto-bg text-mast-auto" },
  dismissed: { label: "已忽略", cls: "bg-mast-border text-mast-muted" },
};

// Exact parity with the old Gradio gr.Radio choices (wishlist tab, app.py).
const WISH_CATEGORIES = ["功能建议", "bug反馈", "体验改进", "其它"];

function fmtTs(iso?: string | null): string {
  if (!iso) return "";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return String(iso).slice(0, 16);
  const p = (n: number) => String(n).padStart(2, "0");
  return `${p(d.getMonth() + 1)}-${p(d.getDate())} ${p(d.getHours())}:${p(d.getMinutes())}`;
}

function StatusPill({ label, cls }: { label: string; cls: string }) {
  return <span className={`rounded-full px-2 py-0.5 text-xs ${cls}`}>{label}</span>;
}

// ── data hook (polls so async relay status flips show up) ───────────────────
function useWishlist() {
  return useQuery({
    queryKey: ["wishlist"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/wishlist");
      if (error) throw error;
      return data;
    },
    refetchInterval: 4000,
  });
}

export default function WishlistPage() {
  const qc = useQueryClient();
  const board = useWishlist();

  // Tag each wish with the running app version so the admin sees which build a
  // wish came from (was hardcoded "" — the server recorded empty client_version).
  const health = useQuery({
    queryKey: ["health"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/health");
      if (error) throw error;
      return data;
    },
    staleTime: 5 * 60_000,
  });
  const clientVersion = health.data?.version ?? "";

  // ── submit-wish form state ──
  const [text, setText] = useState("");
  const [category, setCategory] = useState(WISH_CATEGORIES[0]!);
  const [formMsg, setFormMsg] = useState<{ ok: boolean; text: string } | null>(null);

  const submitWish = useMutation({
    mutationFn: async (payload: { text: string; category: string }) => {
      const { data, error } = await api.POST("/api/wishlist/wishes", {
        body: { text: payload.text, category: payload.category, client_version: clientVersion },
      });
      if (error) throw error;
      return data;
    },
    onSuccess: (data) => {
      setFormMsg({ ok: data.ok, text: data.message || (data.ok ? "已提交" : "提交失败") });
      if (data.ok) setText("");
      qc.invalidateQueries({ queryKey: ["wishlist"] });
    },
    onError: (err) => setFormMsg({ ok: false, text: String((err as Error)?.message ?? err) }),
  });

  // ── resolve-request mutation ──
  const resolveReq = useMutation({
    mutationFn: async (args: {
      id: string;
      action: "done" | "dismissed";
      note: string;
      path: string;
    }) => {
      const { data, error } = await api.POST(
        "/api/wishlist/requests/{request_id}/resolve",
        {
          params: { path: { request_id: args.id } },
          body: { action: args.action, note: args.note, path: args.path },
        },
      );
      if (error) throw error;
      return data;
    },
    onSuccess: () => qc.invalidateQueries({ queryKey: ["wishlist"] }),
  });

  const onSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    const t = text.trim();
    if (!t) {
      setFormMsg({ ok: false, text: "请先填写内容" });
      return;
    }
    submitWish.mutate({ text: t, category });
  };

  // ── render guards ──
  if (board.isPending) return <Spinner />;
  if (board.isError) return <ErrorNote error={board.error} />;

  const data = board.data!;
  const wishes = data.wishes ?? [];
  const requests = data.requests ?? [];

  return (
    <div className="space-y-2">
      {/* Intro markdown — verbatim parity with the old Gradio 心愿单 header
          (app.py gr.Markdown "### 心愿单 / 反馈 …"). */}
      <Section title="心愿单 / 反馈">
        <p className="text-sm text-mast-muted">
          提交愿望与反馈，会<strong className="text-mast-text">自动上报到更新服务器</strong>供开发者改进。
          下方「Agent 请求」是各 agent 需要你配合的事项（如文献 agent 请你上传全文、仪器 agent
          请你操作硬件），处理完点标记。
        </p>
      </Section>

      {/* ── 提交愿望 / 反馈 (OLD order: submit FIRST) ────────────────────── */}
      <Section title="你的愿望 / 反馈">
        <Card>
          <form onSubmit={onSubmit} className="space-y-3">
            <textarea
              value={text}
              onChange={(e) => setText(e.target.value)}
              rows={3}
              placeholder="例如：希望支持批量 STS 自动采集 / 某处用起来卡…"
              className="w-full resize-y rounded border border-mast-border bg-mast-bg px-3 py-2 text-sm text-mast-text placeholder:text-mast-muted focus:border-mast-accent focus:outline-none"
            />
            {/* 类别 — OLD was a gr.Radio (4 inline options, default 功能建议). */}
            <div className="flex flex-wrap items-center gap-x-4 gap-y-1">
              <span className="text-xs text-mast-muted">类别</span>
              {WISH_CATEGORIES.map((c) => (
                <label key={c} className="flex items-center gap-1.5 text-sm text-mast-text">
                  <input
                    type="radio"
                    name="wish-category"
                    value={c}
                    checked={category === c}
                    onChange={() => setCategory(c)}
                  />
                  {c}
                </label>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-3">
              <button
                type="submit"
                disabled={submitWish.isPending}
                className="rounded bg-mast-accent/20 px-4 py-1.5 text-sm text-mast-accent hover:bg-mast-accent/30 disabled:opacity-50"
              >
                {submitWish.isPending ? "提交中…" : "提交并上报"}
              </button>
              {/* Old Gradio 心愿单 had an explicit 刷新 button beside submit;
                  the board also polls every 4 s, but keep the manual control
                  for parity (一个不落). */}
              <button
                type="button"
                disabled={board.isFetching}
                onClick={() => qc.invalidateQueries({ queryKey: ["wishlist"] })}
                className="rounded border border-mast-border px-4 py-1.5 text-sm text-mast-muted hover:text-mast-text disabled:opacity-50"
              >
                {board.isFetching ? "刷新中…" : "刷新"}
              </button>
              {formMsg && (
                <span className={`text-sm ${formMsg.ok ? "text-mast-auto" : "text-mast-danger"}`}>
                  {formMsg.text}
                </span>
              )}
            </div>
          </form>
        </Card>
      </Section>

      {/* ── 我的愿望 列表 (OLD: "#### 我的愿望", before Agent 请求) ───────── */}
      <Section title="我的愿望">
        {!data.degraded && wishes.length === 0 && (
          <EmptyNote label="尚无愿望/反馈。在上方提交，会自动上报到更新服务器。" />
        )}
        <div className="space-y-2">
          {wishes.map((w) => (
            <WishRow key={w.id} wish={w} />
          ))}
        </div>
      </Section>

      {/* ── Agent → 用户 请求 (OLD: "#### Agent 请求(需你处理)", LAST) ───── */}
      <Section
        title="Agent 请求（需你处理）"
        actions={
          data.pending_count > 0 ? (
            <Badge tone="WARN">{data.pending_count} 待处理</Badge>
          ) : undefined
        }
      >
        {data.degraded && <DegradedNote what="心愿单" />}
        {!data.degraded && requests.length === 0 && (
          <EmptyNote label="暂无来自 agent 的请求。agent 需要你配合时(如上传全文、操作硬件),会出现在这里。" />
        )}
        <div className="space-y-2">
          {requests.map((r) => (
            <RequestRow
              key={r.id}
              req={r}
              busy={resolveReq.isPending}
              onResolve={(action, note, path) =>
                resolveReq.mutate({ id: r.id, action, note, path })
              }
            />
          ))}
        </div>
        {resolveReq.isError && (
          <p className="mt-2 text-sm text-mast-danger">
            处理失败：{String((resolveReq.error as Error)?.message ?? resolveReq.error)}
          </p>
        )}
      </Section>
    </div>
  );
}

// ── single wish card ───────────────────────────────────────────────────────
function WishRow({ wish }: { wish: Wish }) {
  const st = WISH_STATUS[wish.status] ?? { label: wish.status, cls: "bg-mast-border text-mast-muted" };
  return (
    <Card>
      <div className="flex items-center gap-2 text-xs text-mast-muted">
        <StatusPill label={st.label} cls={st.cls} />
        <span>{wish.category}</span>
        <span className="ml-auto">{fmtTs(wish.created_at)}</span>
      </div>
      <p className="mt-2 whitespace-pre-wrap text-sm text-mast-text">{wish.text}</p>
      {wish.server_ack && (
        <p className="mt-1 text-xs text-mast-muted">服务器回执：{wish.server_ack}</p>
      )}
      {wish.error && <p className="mt-1 text-xs text-mast-danger">{wish.error}</p>}
    </Card>
  );
}

// ── single agent-request card (with resolve / dismiss actions) ─────────────
function RequestRow({
  req,
  busy,
  onResolve,
}: {
  req: AgentRequest;
  busy: boolean;
  onResolve: (action: "done" | "dismissed", note: string, path: string) => void;
}) {
  const st = REQ_STATUS[req.status] ?? { label: req.status, cls: "bg-mast-border text-mast-muted" };
  const [note, setNote] = useState("");
  const [path, setPath] = useState("");
  const open = req.status === "pending";

  return (
    <Card>
      <div className="flex items-center gap-2 text-xs text-mast-muted">
        <StatusPill label={st.label} cls={st.cls} />
        <b className="text-mast-accent">{req.agent_id}</b>
        <span>{req.kind}</span>
        {/* conduct 发的请求给一条能点的出路。在此之前它是一段**没有出口的
            文字**:想去处理它，得先知道 conduct 面板在「实验记录」下面一层。
            双闸的另一半（物理条件）也只有在那一页看得见。 */}
        {conductIdFromAgentId(req.agent_id) && (
          <Link
            to={conductPanelHref(conductIdFromAgentId(req.agent_id))}
            className="text-mast-accent underline hover:opacity-80"
          >
            去 conduct 面板
          </Link>
        )}
        {req.experiment_id && <span>· 实验 {req.experiment_id}</span>}
        <span className="ml-auto">
          #{req.id} · {fmtTs(req.created_at)}
        </span>
      </div>
      <p className="mt-2 whitespace-pre-wrap text-sm text-mast-text">{req.message}</p>
      {req.path && (
        <p className="mt-1 font-mono text-xs text-mast-accent">路径：{req.path}</p>
      )}
      {req.note && <p className="mt-1 text-xs text-mast-auto">备注：{req.note}</p>}
      {open && (
        <div className="mt-3 space-y-2">
          {/* 路径 is its OWN field now . The agent's most common ask is
              "where is the file?", and the only way to answer was to type it into
              a free-text note — which the agent had to guess at, and which it
              never even received unless a run happened to be streaming at that
              moment. A path typed into prose is not a channel. */}
          <label className="flex flex-wrap items-center gap-2">
            <span className="shrink-0 text-xs font-medium text-mast-accent">路径</span>
            <input
              value={path}
              onChange={(e) => setPath(e.target.value)}
              spellCheck={false}
              placeholder={"智能体索要的文件/目录，例如 D:\\Data\\fsSTM\\Au111_001.sxm"}
              className="min-w-0 flex-1 rounded border border-mast-accent/40 bg-mast-bg px-3 py-1.5 font-mono text-xs text-mast-text placeholder:text-mast-muted focus:border-mast-accent focus:outline-none"
            />
          </label>
          <div className="flex flex-wrap items-center gap-2">
            <input
              value={note}
              onChange={(e) => setNote(e.target.value)}
              placeholder="备注（可选）——说明你做了什么 / 还有什么要注意的"
              className="min-w-0 flex-1 rounded border border-mast-border bg-mast-bg px-3 py-1.5 text-sm text-mast-text placeholder:text-mast-muted focus:border-mast-accent focus:outline-none"
            />
          <button
            type="button"
            disabled={busy}
            onClick={() => onResolve("done", note, path)}
            className="rounded bg-mast-auto-bg px-3 py-1.5 text-sm text-mast-auto hover:bg-mast-auto-border disabled:opacity-50"
          >
            标记完成
          </button>
          <button
            type="button"
            disabled={busy}
            onClick={() => onResolve("dismissed", note, path)}
            className="rounded bg-mast-border px-3 py-1.5 text-sm text-mast-muted hover:bg-mast-border/70 disabled:opacity-50"
          >
            忽略
          </button>
          </div>
          <p className="text-[11px] text-mast-muted">
            答复会实时转发给正在运行的智能体；若当前没有运行中的任务，它会
            <strong className="text-mast-text">保存下来</strong>，智能体下次运行时自己读回
            （<code className="font-mono">check_my_requests</code>）——你不必再手动复述一遍。
          </p>
        </div>
      )}
      {req.resolved_at && (
        <p className="mt-2 text-xs text-mast-muted">处理于 {fmtTs(req.resolved_at)}</p>
      )}
    </Card>
  );
}
