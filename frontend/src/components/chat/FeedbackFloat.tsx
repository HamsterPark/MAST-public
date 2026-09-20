import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { useLocation } from "react-router-dom";
import { useMutation } from "@tanstack/react-query";
import clsx from "clsx";
import { api } from "@/api/client";
// 路由 → 「这条反馈在说哪一页」。判断住在 lib/ 里(纯函数、无 JSX),
// 因为 node --test 剥不了 .tsx —— 而这几条判断错了没有任何症状,
// 只有一张内容是错的反馈表(#77 那 108 行)。
import { subjectForPath } from "@/lib/feedbackSubject";

// Draggable, always-visible conversation-feedback floating widget. Mirrors the
// old Gradio 对话反馈悬浮窗 (build_feedback_panel + feedback_float_assets).
//
//   ratings + free comment → POST /api/feedback (rating/comment + agent + conv id)
//   draggable by its title bar via DOCUMENT-LEVEL pointer event delegation
//   (the proven approach from MEMORY feedback_gradio_drag_delegation —
//   re-find the panel on each pointerdown, never bind to a node that React
//   may rebuild). React owns the DOM here, but document-level delegation is
//   still the robust pattern and survives re-renders.

const RATINGS: { value: string; label: string }[] = [
  { value: "good", label: "👍 很好" },
  { value: "ok", label: "🙂 不错" },
  { value: "neutral", label: "😐 一般" },
  { value: "poor", label: "👎 较差" },
  { value: "problem", label: "⚠️ 有问题" },
];

// ITEM 2 — fighting-game combo window: a same-rating tap within this many ms of
// the previous one bumps the ×N multiplier; otherwise the counter resets to 1.
const COMBO_IDLE_MS = 1200;

export function FeedbackFloat({ conversationId }: { conversationId: string | null }) {
  const panelRef = useRef<HTMLDivElement>(null);
  const [flash, setFlash] = useState<{ text: string; ok: boolean } | null>(null);
  const [comment, setComment] = useState("");
  // ITEM 2 — combo state. `combo` is the live multiplier shown as a floating
  // ×N badge over the tapped button; it fades + clears after COMBO_IDLE_MS idle.
  // `bump` retriggers the pop animation on every tap (key changes).
  const [combo, setCombo] = useState<{ value: string; count: number; bump: number } | null>(null);
  const comboTimer = useRef<ReturnType<typeof setTimeout> | null>(null);
  const lastRating = useRef<{ value: string; t: number; count: number }>({ value: "", t: 0, count: 0 });
  // null position = use CSS default (bottom-right). Once dragged, becomes {x,y}.
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);
  const dragState = useRef<{
    dragging: boolean;
    sx: number;
    sy: number;
    ox: number;
    oy: number;
  }>({ dragging: false, sx: 0, sy: 0, ox: 0, oy: 0 });

  // Document-level pointer delegation for dragging by the handle.
  useEffect(() => {
    function onDown(e: PointerEvent) {
      const panel = panelRef.current;
      const target = e.target as HTMLElement | null;
      if (!panel || !target?.closest) return;
      const handle = target.closest(".mast-fb-handle");
      if (!handle || !panel.contains(handle)) return;
      const r = panel.getBoundingClientRect();
      dragState.current = {
        dragging: true,
        sx: e.clientX,
        sy: e.clientY,
        ox: r.left,
        oy: r.top,
      };
      setPos({ x: r.left, y: r.top });
      panel.classList.add("mast-fb-dragging");
      e.preventDefault();
    }
    function onMove(e: PointerEvent) {
      const d = dragState.current;
      const panel = panelRef.current;
      if (!d.dragging || !panel) return;
      let nx = d.ox + (e.clientX - d.sx);
      let ny = d.oy + (e.clientY - d.sy);
      nx = Math.max(0, Math.min(nx, window.innerWidth - panel.offsetWidth));
      ny = Math.max(0, Math.min(ny, window.innerHeight - panel.offsetHeight));
      setPos({ x: nx, y: ny });
    }
    function onUp() {
      if (!dragState.current.dragging) return;
      dragState.current.dragging = false;
      panelRef.current?.classList.remove("mast-fb-dragging");
    }
    document.addEventListener("pointerdown", onDown, true);
    document.addEventListener("pointermove", onMove, true);
    document.addEventListener("pointerup", onUp, true);
    document.addEventListener("pointercancel", onUp, true);
    return () => {
      document.removeEventListener("pointerdown", onDown, true);
      document.removeEventListener("pointermove", onMove, true);
      document.removeEventListener("pointerup", onUp, true);
      document.removeEventListener("pointercancel", onUp, true);
    };
  }, []);

  useEffect(() => {
    if (!flash) return;
    const t = setTimeout(() => setFlash(null), 1500);
    return () => clearTimeout(t);
  }, [flash]);

  // ITEM 2 — clear the pending combo timer when the widget unmounts.
  useEffect(() => () => {
    if (comboTimer.current) clearTimeout(comboTimer.current);
  }, []);

  // Record WHERE the operator gave the feedback (// "对话反馈知道用户是在哪个界面反馈的吗？" — it didn't: agent was hardcoded
  // and meta carried nothing, so all 108 rows looked like instrument_control
  // chat feedback). useLocation re-renders on route change, so the pathname
  // is always the page the widget was on at submit time.
  const { pathname } = useLocation();

  const post = useMutation({
    mutationFn: async (body: { rating?: string; comment?: string; meta?: Record<string, unknown> }) => {
      const { data, error } = await api.POST("/api/feedback", {
        body: {
          rating: body.rating ?? "",
          comment: body.comment ?? "",
          agent: subjectForPath(pathname),
          // A conversation id only means something on the chat surfaces; on
          // 设置/记录/… it would falsely bind the note to whatever chat was
          // last open.
          conversation_id:
            pathname === "/" || pathname.startsWith("/agents") ? conversationId : null,
          experiment_id: null,
          sample_id: null,
          meta: { page: pathname, ...(body.meta ?? {}) },
        },
      });
      if (error) throw error;
      return data;
    },
  });

  function rate(value: string, label: string) {
    // ITEM 2 — combo bookkeeping: same rating within the idle window bumps ×N.
    const now = Date.now();
    const prev = lastRating.current;
    const count = prev.value === value && now - prev.t <= COMBO_IDLE_MS ? prev.count + 1 : 1;
    lastRating.current = { value, t: now, count };
    setCombo((c) => ({ value, count, bump: (c?.bump ?? 0) + 1 }));
    if (comboTimer.current) clearTimeout(comboTimer.current);
    comboTimer.current = setTimeout(() => {
      setCombo(null);
      lastRating.current = { value: "", t: 0, count: 0 };
    }, COMBO_IDLE_MS);

    // Still POST one feedback per tap; ride the running combo count in meta.
    post.mutate(
      { rating: value, meta: { label, combo: count } },
      {
        onSuccess: (d) =>
          setFlash(
            d?.ok ? { text: `✓ 已记录 ${label}`, ok: true } : { text: "未记录（记录库不可用）", ok: false },
          ),
        onError: () => setFlash({ text: "记录失败", ok: false }),
      },
    );
  }

  function submitComment() {
    const text = comment.trim();
    if (!text) {
      setFlash({ text: "评论为空", ok: false });
      return;
    }
    post.mutate(
      { comment: text },
      {
        onSuccess: (d) => {
          if (d?.ok) {
            setComment("");
            setFlash({ text: "✓ 评论已记录", ok: true });
          } else {
            setFlash({ text: "未记录（记录库不可用）", ok: false });
          }
        },
        onError: () => setFlash({ text: "记录失败", ok: false }),
      },
    );
  }

  const style: React.CSSProperties = pos
    ? { left: pos.x, top: pos.y, right: "auto", bottom: "auto" }
    : { right: 22, bottom: 22 };

  // Portal to <body> so the fixed widget can NEVER be clipped / re-anchored by an
  // ancestor's overflow:hidden or transform/filter/will-change (the classic reason
  // a position:fixed floating panel silently vanishes). This guarantees it always
  // floats over the viewport regardless of where in the tree it's mounted.
  return createPortal(
    <div
      ref={panelRef}
      style={{ position: "fixed", zIndex: 99998, width: 236, maxWidth: "62vw", ...style }}
      className="rounded-2xl border border-mast-border bg-mast-panel/95 p-3 shadow-2xl backdrop-blur"
    >
      <div className="mast-fb-handle flex cursor-grab touch-none select-none items-center justify-between border-b border-mast-border pb-2 active:cursor-grabbing">
        <span className="text-sm font-semibold text-mast-text">💬 对话反馈</span>
        <span className="text-[10px] text-mast-muted/60">⠿ 拖动</span>
      </div>
      {/* ITEM 2 — combo badge animation (scoped). */}
      <style>{`
        @keyframes mastComboPop {
          0%   { transform: translate(50%, 0) scale(0.6); opacity: 0; }
          25%  { transform: translate(50%, -6px) scale(1.25); opacity: 1; }
          70%  { transform: translate(50%, -10px) scale(1); opacity: 1; }
          100% { transform: translate(50%, -18px) scale(0.9); opacity: 0; }
        }
        .mast-combo-badge { animation: mastComboPop 1.2s ease-out forwards; }
      `}</style>
      <div className="mt-2 flex flex-wrap gap-1">
        {RATINGS.map((r) => {
          const showCombo = combo && combo.value === r.value && combo.count > 1;
          return (
            <button
              key={r.value}
              type="button"
              disabled={post.isPending}
              onClick={() => rate(r.value, r.label)}
              className="relative rounded-md border border-mast-border px-2 py-1 text-xs text-mast-text hover:border-mast-accent disabled:opacity-50"
            >
              {r.label}
              {showCombo && (
                <span
                  key={combo.bump}
                  aria-hidden
                  className="mast-combo-badge pointer-events-none absolute -top-2 right-0 select-none rounded-full bg-mast-warn px-1.5 py-0.5 text-[11px] font-extrabold leading-none text-mast-bg shadow-lg"
                >
                  ×{combo.count}
                </span>
              )}
            </button>
          );
        })}
      </div>
      <div className="min-h-[18px] py-1 text-xs">
        {flash && (
          <span className={clsx("font-semibold", flash.ok ? "text-mast-auto" : "text-mast-danger")}>
            {flash.text}
          </span>
        )}
      </div>
      <textarea
        value={comment}
        onChange={(e) => setComment(e.target.value)}
        placeholder="写点评论…（随时）"
        rows={2}
        className="w-full resize-none rounded-md border border-mast-border bg-mast-bg px-2 py-1.5 text-xs text-mast-text outline-none focus:border-mast-accent"
      />
      <button
        type="button"
        onClick={submitComment}
        disabled={post.isPending}
        className="mt-2 w-full rounded-md bg-mast-accent/20 px-2 py-1.5 text-xs font-medium text-mast-accent hover:bg-mast-accent/30 disabled:opacity-50"
      >
        提交评论
      </button>
    </div>,
    document.body,
  );
}
