import { useMemo, useState } from "react";
import { useQuery } from "@tanstack/react-query";

import { api } from "@/api/client";
import { Avatar } from "@/components/agents/registry";
import { ContextInjectionMatrix } from "@/components/agents/ContextInjectionMatrix";
import { ShareBar } from "@/components/agents/ShareBar";
import { Button } from "@/components/controls";
import {
  Badge,
  Card,
  DegradedNote,
  EmptyNote,
  ErrorNote,
  Spinner,
} from "@/components/ui";
import {
  type InjectionBlock,
  ALL_AGENTS,
  availabilityLabel,
  canShowText,
  captureEmptyMessage,
  composeShares,
  fmtAge,
  fmtChars,
  positionLabel,
  segmentAnchor,
  splitByBlocks,
  splitMessages,
  toBackendAgentId,
  whenLabel,
} from "@/lib/contextInjection";

// ════════════════════════════════════════════════════════════════════════
// 「上下文注入」子页 —— 只读。
//
// 它回答两个问题，而这两个在 2026-08-24 之前都没有地方能答：
//   1. 不同角色有针对性的注入吗？        → 「全部」视图的矩阵 + 逐 agent 足迹
//   2. 这一次调用到底由什么构成？        → 单 agent 视图的分段快照 + 占比条
//
// 编辑仍然留在 高级管理 → 上下文注入（PIN 门后面）。这里一个写请求都没有 ——
// 它不在 PIN 后面，能写的话那道 PIN 就形同虚设。
// ════════════════════════════════════════════════════════════════════════

export const ALL_VIEW = "__all__";

// ── 取数 ────────────────────────────────────────────────────────────────

function useMatrix() {
  return useQuery({
    // 挂在 ["admin","prompts"] 前缀下：PromptInspector 保存覆写后会
    // invalidate 这个前缀，矩阵跟着失效，那边零改动。
    queryKey: ["admin", "prompts", "manifest"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/prompts/manifest");
      if (error) throw error;
      return data;
    },
    staleTime: 30_000,
  });
}

function useAgentManifest(backendId: string, enabled: boolean) {
  return useQuery({
    queryKey: ["admin", "prompts", "manifest", backendId],
    enabled,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/prompts/manifest/{agent}", {
        params: { path: { agent: backendId } },
      });
      if (error) throw error;
      return data;
    },
    staleTime: 30_000,
  });
}

function useLatestCapture(backendId: string, enabled: boolean) {
  return useQuery({
    // staleTime 不设 Infinity：latest 会变（每来一次调用就换一条），
    // 设了就等于「刷新」按钮永远不刷。
    queryKey: ["prompt-capture", "latest", backendId],
    enabled,
    queryFn: async () => {
      const { data, error, response } = await api.GET(
        "/api/prompts/capture/latest/{agent}",
        { params: { path: { agent: backendId } } },
      );
      // 404 = 「还没有这个 agent 的快照」，是**空态**不是错误。
      // openapi-fetch 把非 2xx 的 body 放进 error，直接 throw 会让「还没跑过」
      // 画成一条红色的「加载失败」，而它们要的动作完全不同。
      // 后端在 404 上返回的仍然是 LatestCaptureResponse（found=false + 原因），
      // 而 openapi-fetch 把非 2xx 的 body 一律归到 error 并按 HTTPValidationError
      // 打类型。经 unknown 转一次是这里唯一诚实的写法。
      if (response.status === 404 && error) {
        return error as unknown as typeof data;
      }
      if (error) throw error;
      return data;
    },
  });
}

// ── 单块 ────────────────────────────────────────────────────────────────

function BlockRow({
  block,
  onJump,
  presentInCapture,
}: {
  block: InjectionBlock;
  onJump: (id: string) => void;
  presentInCapture: boolean;
}) {
  const [open, setOpen] = useState(false);
  const showable = canShowText(block);
  const detail = useQuery({
    // 与 PromptInspector 同一个 key，共享缓存。
    queryKey: ["admin", "prompt", block.id],
    enabled: open && showable,
    queryFn: async () => {
      const { data, error } = await api.GET("/api/admin/prompts/{prompt_id}", {
        params: { path: { prompt_id: block.id } },
      });
      if (error) throw error;
      return data;
    },
  });

  return (
    <li className="border-b border-mast-border last:border-b-0">
      <button
        type="button"
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-start gap-3 px-3 py-2.5 text-left hover:bg-mast-accent-soft"
      >
        <span className="mt-0.5 w-3 shrink-0 text-mast-faint">{open ? "▾" : "▸"}</span>
        <span className="min-w-0 flex-1">
          <span className="flex flex-wrap items-center gap-2">
            <span className="text-sm text-mast-text">{block.label}</span>
            {block.overridden && <Badge tone="WARN">已覆写</Badge>}
            {block.exclusive && <Badge tone="INFO">专属</Badge>}
            {(block.agents ?? []).includes(ALL_AGENTS) && <Badge>全员</Badge>}
          </span>
          <span className="mt-1 flex flex-wrap gap-x-3 gap-y-0.5 text-xs text-mast-muted">
            <span>{whenLabel(block.when ?? "always")}</span>
            <span>落在 {positionLabel(block.position ?? "system")}</span>
            <span>{availabilityLabel(block.availability)}</span>
            <span className="font-mono text-mast-faint">{block.id}</span>
          </span>
        </span>
        <span className="shrink-0 text-xs text-mast-muted">
          {showable ? fmtChars(block.effective_chars) : "—"}
        </span>
      </button>

      {open && (
        <div className="px-3 pb-3 pl-9">
          {block.note && (
            <p className="mb-2 whitespace-pre-wrap text-xs text-mast-muted">{block.note}</p>
          )}
          {showable ? (
            <>
              {detail.isPending && <Spinner />}
              {detail.error != null && <ErrorNote error={detail.error} />}
              {detail.data?.degraded && <DegradedNote what="这一块的正文" />}
              {detail.data && !detail.data.degraded && (
                <pre className="max-h-80 overflow-auto whitespace-pre-wrap break-words rounded-mast-ctl border border-mast-border bg-mast-bg p-2.5 font-mono text-xs text-mast-text">
                  {detail.data.effective_text || "（这一块当前是空的）"}
                </pre>
              )}
            </>
          ) : (
            // 诚实性铁律：拿不到就说拿不到，**绝不填示例文本**。
            <div className="rounded-mast-ctl border border-mast-warn-border bg-mast-warn-bg px-3 py-2 text-xs text-mast-warn">
              <p>{block.unavailable_reason || "这一块只在真实请求里才有内容。"}</p>
              {presentInCapture && (
                <button
                  type="button"
                  className="mt-1.5 underline"
                  onClick={() => onJump(block.id)}
                >
                  ↓ 到下面那次真实请求里看这一段
                </button>
              )}
            </div>
          )}
          <p className="mt-2 font-mono text-[11px] text-mast-faint">{block.source}</p>
        </div>
      )}
    </li>
  );
}

// ── 单 agent 视图 ───────────────────────────────────────────────────────

function AgentView({ uiId }: { uiId: string }) {
  const backendId = toBackendAgentId(uiId);
  const manifest = useAgentManifest(backendId, true);
  const capture = useLatestCapture(backendId, true);

  const cap = capture.data;
  const hasCapture = Boolean(cap?.found);
  const { system, history } = useMemo(
    () => splitMessages(cap?.messages), [cap?.messages]);

  const first = system[0];
  const { segments, issues } = useMemo(
    () => splitByBlocks(first?.content ?? "", first?.chars ?? 0,
                        (first?.blocks ?? null) as never),
    [first?.content, first?.chars, first?.blocks],
  );

  const labelById = useMemo(() => {
    const m = new Map<string, string>();
    for (const b of manifest.data?.blocks ?? []) m.set(b.id, b.label);
    return m;
  }, [manifest.data?.blocks]);

  const shares = useMemo(
    () => composeShares({
      segments,
      history,
      toolsChars: cap?.tools_chars ?? null,
      labelFor: (id) => labelById.get(id) ?? id,
    }),
    [segments, history, cap?.tools_chars, labelById],
  );

  const capturedIds = useMemo(
    () => new Set(segments.map((s) => s.id).filter(Boolean) as string[]),
    [segments],
  );

  const jump = (id: string) => {
    document.getElementById(segmentAnchor(id))?.scrollIntoView({ block: "center" });
  };

  const surface = manifest.data?.tool_surface;

  return (
    <div className="space-y-4">
      {/* ── 组成 ── */}
      <Card>
        <div className="mb-2 flex items-center justify-between">
          <h3 className="text-sm font-semibold text-mast-text">
            每次调用收到的块（按挂载顺序）
          </h3>
          <span className="text-xs text-mast-faint">
            {manifest.data?.order_source === "build"
              ? "顺序来自这个进程真的建过的那次图"
              : "顺序来自登记表的声明（这个进程还没建过它的图）"}
          </span>
        </div>
        {manifest.isPending && <Spinner />}
        {manifest.error != null && <ErrorNote error={manifest.error} />}
        {manifest.data?.degraded && <DegradedNote what="注入清单" />}
        {manifest.data && !manifest.data.degraded && (
          <ul className="rounded-mast-ctl border border-mast-border">
            {(manifest.data.blocks ?? []).map((b) => (
              <BlockRow key={b.id} block={b} onJump={jump}
                        presentInCapture={capturedIds.has(b.id)} />
            ))}
          </ul>
        )}
      </Card>

      {/* ── 工具面 ── */}
      <Card>
        <h3 className="mb-2 text-sm font-semibold text-mast-text">
          工具面 —— 一次调用里最大的一块，而它不在消息列表里
        </h3>
        {manifest.isPending && <Spinner />}
        {!manifest.isPending && !surface && (
          <EmptyNote label={manifest.data?.tool_surface_note || "没有工具面数据。"} />
        )}
        {surface && (
          <div className="space-y-2 text-sm text-mast-text">
            <p>
              {surface.count} 个工具，schema 合计{" "}
              <strong>{fmtChars(surface.schema_chars)}</strong>
              {surface.core_tools !== null && surface.core_tools !== undefined && (
                <>
                  {" "}—— 按需加载开着，这一次只发核心的{" "}
                  <strong>{surface.core_tools}</strong> 个（
                  {fmtChars(surface.core_chars)}，省{" "}
                  {surface.schema_chars > 0
                    ? Math.round(100 * (1 - (surface.core_chars ?? 0) / surface.schema_chars))
                    : 0}
                  %）
                </>
              )}
            </p>
            {(surface.packs ?? []).length > 0 && (
              <ul className="flex flex-wrap gap-2">
                {(surface.packs ?? []).map((p) => (
                  <li key={p.name}
                      className="rounded-mast-badge border border-mast-border px-2 py-0.5 text-xs text-mast-muted">
                    <span className="text-mast-text">{p.name}</span> {p.label} ·{" "}
                    {p.tools} 个 · {fmtChars(p.chars)}
                  </li>
                ))}
              </ul>
            )}
            {(surface.top ?? []).length > 0 && (
              <details>
                <summary className="cursor-pointer text-xs text-mast-muted">
                  最大的几个工具
                </summary>
                <ul className="mt-1.5 space-y-0.5 text-xs text-mast-muted">
                  {(surface.top ?? []).map((t) => (
                    <li key={String(t.name)}>
                      <span className="font-mono text-mast-text">{String(t.name)}</span>{" "}
                      {fmtChars(Number(t.chars))}
                    </li>
                  ))}
                </ul>
              </details>
            )}
          </div>
        )}
      </Card>

      {/* ── 最近一次真实请求 ── */}
      <Card>
        <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
          <h3 className="text-sm font-semibold text-mast-text">最近一次真实请求</h3>
          <Button onClick={() => capture.refetch()}>刷新</Button>
        </div>
        {capture.isPending && <Spinner />}
        {capture.error != null && <ErrorNote error={capture.error} />}
        {cap?.degraded && <DegradedNote what="注入记录" />}
        {cap && !cap.degraded && !hasCapture && (
          <EmptyNote label={captureEmptyMessage(cap.reason_code ?? "", cap.reason ?? "")} />
        )}
        {cap && hasCapture && (
          <div className="space-y-3">
            <p className="flex flex-wrap gap-x-4 gap-y-1 text-xs text-mast-muted">
              <span>#{cap.seq}</span>
              <span>{fmtAge(cap.age_s)}</span>
              <span>{cap.model_id} · {cap.provider}</span>
              {cap.node && <span>节点 {cap.node}</span>}
              <span>
                输入 token：
                {cap.tokens_source === "provider" && cap.input_tokens !== null
                  ? cap.input_tokens
                  : "provider 未上报"}
              </span>
              {cap.cache_read_tokens !== null && cap.cache_read_tokens !== undefined && (
                <span>缓存命中 {cap.cache_read_tokens}</span>
              )}
            </p>
            {cap.blocks_source === "none" && (
              <p className="text-xs text-mast-warn">
                这一次的分段归属查不到（多半是这条消息没经过注入 helper）——
                下面按整段显示，不猜。
              </p>
            )}
            {issues.length > 0 && (
              <ul className="space-y-0.5 text-xs text-mast-warn">
                {issues.map((it, i) => <li key={i}>{it.detail}</li>)}
              </ul>
            )}

            <ShareBar
              shares={shares}
              agentId={uiId}
              caption="条按**字符**分；token 由 provider 上报，两者不可互换。带斜纹的那段是估算。"
            />

            <div>
              <h4 className="mb-1.5 text-xs font-semibold text-mast-muted">
                系统消息，按来源切分
              </h4>
              <ul className="space-y-1.5">
                {segments.map((seg, i) => (
                  <li key={`${seg.id ?? "none"}-${i}`}
                      id={seg.id ? segmentAnchor(seg.id) : undefined}>
                    <details className="rounded-mast-ctl border border-mast-border">
                      <summary className="cursor-pointer px-2.5 py-1.5 text-xs text-mast-muted">
                        <span className="text-mast-text">
                          {seg.id ? (labelById.get(seg.id) ?? seg.id) : "未归属"}
                        </span>{" "}
                        · {fmtChars(seg.chars)}
                        {seg.clipped && <span className="text-mast-warn">（原文被截断）</span>}
                      </summary>
                      <pre className="max-h-72 overflow-auto whitespace-pre-wrap break-words border-t border-mast-border p-2.5 font-mono text-xs text-mast-text">
                        {seg.text}
                      </pre>
                    </details>
                  </li>
                ))}
              </ul>
            </div>

            <details>
              <summary className="cursor-pointer text-xs text-mast-muted">
                对话历史 {history.length} 条（这些不是注入，屏幕上本来就看得见）
              </summary>
              <ul className="mt-1.5 space-y-1">
                {history.map((m, i) => (
                  <li key={i} className="text-xs text-mast-muted">
                    <span className="font-mono text-mast-text">{m.role}</span> ·{" "}
                    {fmtChars(m.chars)}
                    {m.truncated && <span className="text-mast-warn">（已截断）</span>}
                  </li>
                ))}
              </ul>
            </details>
          </div>
        )}
      </Card>
    </div>
  );
}

// ── 主体 ────────────────────────────────────────────────────────────────

export function ContextInjectionView({
  picks,
  pick,
  onPick,
}: {
  picks: { id: string; label: string }[];
  pick: string;
  onPick: (id: string) => void;
}) {
  const matrix = useMatrix();
  const known = useMemo(
    () => new Set((matrix.data?.agents ?? []).map((a) => a.id)),
    [matrix.data?.agents],
  );

  return (
    <div className="space-y-4">
      <p className="text-xs text-mast-muted">
        这一页是**只读**的：它回答「不同角色有针对性的注入吗」与「这一次调用由什么
        构成」。要改话术请去{" "}
        <a className="underline" href="/settings/admin">高级管理 → 上下文注入</a>。
      </p>

      <div className="flex flex-wrap gap-1.5">
        {picks.map((p) => {
          const isAll = p.id === ALL_VIEW;
          const active = pick === p.id;
          return (
            <button
              key={p.id}
              type="button"
              onClick={() => onPick(p.id)}
              className={
                "flex items-center gap-1.5 rounded-mast-badge border px-2 py-1 text-xs " +
                (active
                  ? "border-mast-accent bg-mast-accent-soft text-mast-text"
                  : "border-mast-border text-mast-muted hover:bg-mast-accent-soft")
              }
            >
              {!isAll && <Avatar id={p.id} size={16} />}
              <span>{p.label}</span>
              {!isAll && matrix.data && !known.has(toBackendAgentId(p.id)) && (
                <span className="text-mast-warn" title="后端的注入清单里没有这个 agent">
                  ·未登记
                </span>
              )}
            </button>
          );
        })}
      </div>

      {matrix.isPending && <Spinner />}
      {matrix.error != null && <ErrorNote error={matrix.error} />}
      {matrix.data?.degraded && <DegradedNote what="注入清单" />}

      {matrix.data && !matrix.data.degraded && pick === ALL_VIEW && (
        <ContextInjectionMatrix matrix={matrix.data} onPick={onPick} />
      )}
      {pick !== ALL_VIEW && <AgentView key={pick} uiId={pick} />}
    </div>
  );
}
