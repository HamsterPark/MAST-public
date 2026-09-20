import type { ReactNode, SVGProps } from "react";
import clsx from "clsx";

/** Shared, dependency-light UI primitives every panel reuses. Replaces the old
 *  Gradio gr.HTML/gr.Markdown table hacks with plain, typed components.
 *  Restyled onto the MAST Lab Console design system — all color/radius/shadow
 *  comes from the design tokens (never hardcoded hex), so both themes + density
 *  inherit automatically. */

/* ── Inline icons (no icon dependency installed; tiny lucide-shaped SVGs) ──── */
function Icon({ className, children, ...rest }: SVGProps<SVGSVGElement>) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
      {...rest}
    >
      {children}
    </svg>
  );
}

function IconLoader(props: SVGProps<SVGSVGElement>) {
  // lucide loader-circle — single arc; spin is applied by the caller via class
  return (
    <Icon {...props}>
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </Icon>
  );
}

function IconInbox(props: SVGProps<SVGSVGElement>) {
  return (
    <Icon {...props}>
      <path d="M22 12h-6l-2 3h-4l-2-3H2" />
      <path d="M5.45 5.11 2 12v6a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2v-6l-3.45-6.89A2 2 0 0 0 16.76 4H7.24a2 2 0 0 0-1.79 1.11z" />
    </Icon>
  );
}

function IconCloudOff(props: SVGProps<SVGSVGElement>) {
  return (
    <Icon {...props}>
      <path d="m2 2 20 20" />
      <path d="M5.782 5.782A7 7 0 0 0 9 19h8.5a4.5 4.5 0 0 0 1.307-.193" />
      <path d="M21.532 16.5A4.5 4.5 0 0 0 17.5 10h-1.79A7.008 7.008 0 0 0 10 5.07" />
    </Icon>
  );
}

function IconAlertCircle(props: SVGProps<SVGSVGElement>) {
  return (
    <Icon {...props}>
      <circle cx="12" cy="12" r="10" />
      <line x1="12" x2="12" y1="8" y2="12" />
      <line x1="12" x2="12.01" y1="16" y2="16" />
    </Icon>
  );
}

export function Section({
  title, subtitle, actions, children,
}: { title: string; subtitle?: string; actions?: ReactNode; children: ReactNode }) {
  return (
    <section className="mb-6">
      <div className="mb-3 flex items-start justify-between">
        <div className="flex items-center gap-2.5">
          <span className="h-[18px] w-[3px] shrink-0 rounded bg-mast-accent" aria-hidden="true" />
          <div>
            <h2 className="text-[18px] font-semibold leading-tight text-mast-text">{title}</h2>
            {subtitle && <p className="mt-0.5 text-xs text-mast-faint">{subtitle}</p>}
          </div>
        </div>
        {actions}
      </div>
      {children}
    </section>
  );
}

export function Card({ className, children }: { className?: string; children: ReactNode }) {
  return (
    <div
      className={clsx(
        "rounded-mast-card border border-mast-border bg-mast-panel p-4 shadow-mast",
        className,
      )}
    >
      {children}
    </div>
  );
}

// Semantic safety tones — single source mapping to the design-system tokens
// (fg + bg + border each). AUTO=自主/安全 · INFO=信息/只读 · WARN=需确认 ·
// DANGEROUS=停/失败. Color is never the sole signal (callers pair with a label).
const BADGE_TONE: Record<string, string> = {
  AUTO: "text-mast-auto bg-mast-auto-bg border border-mast-auto-border",
  INFO: "text-mast-info bg-mast-info-bg border border-mast-info-border",
  WARN: "text-mast-warn bg-mast-warn-bg border border-mast-warn-border",
  DANGEROUS: "text-mast-danger bg-mast-danger-bg border border-mast-danger-border",
  default: "text-mast-accent bg-mast-accent-soft border border-mast-border",
};

export function Badge({ tone, children }: { tone?: string; children: ReactNode }) {
  return (
    <span
      className={clsx(
        "inline-flex items-center gap-1 rounded-mast-badge px-1.5 py-0.5 text-xs",
        BADGE_TONE[tone ?? "default"] ?? BADGE_TONE.default,
      )}
    >
      {children}
    </span>
  );
}

export function Spinner({ label = "加载中…" }: { label?: string }) {
  return (
    <p className="flex items-center gap-2 text-sm text-mast-muted">
      <IconLoader className="h-4 w-4 animate-spin text-mast-accent" />
      {label}
    </p>
  );
}

export function ErrorNote({ error, label = "加载失败" }: { error: unknown; label?: string }) {
  // ``label`` lets a caller use a context-appropriate prefix — a mid-run stream
  // error is NOT a "加载失败" (load failed); RunTaskPanel/AgentChatPanel pass
  // "运行出错" so a recursion-limit/abort message reads correctly.
  return (
    <div className="flex items-center gap-2 rounded-mast-ctl border border-mast-danger-border bg-mast-danger-bg px-3 py-2.5 text-sm text-mast-danger">
      <IconAlertCircle className="h-[15px] w-[15px] shrink-0" />
      <span>
        {label}：{String((error as Error)?.message ?? error)}
      </span>
    </div>
  );
}

export function DegradedNote({ what = "此功能" }: { what?: string }) {
  return (
    <div className="flex items-center gap-2 rounded-mast-ctl border border-mast-warn-border bg-mast-warn-bg px-3 py-2.5 text-sm text-mast-warn">
      <IconCloudOff className="h-[15px] w-[15px] shrink-0" />
      <span>{what}当前未连接到运行中的内核（独立开发模式）。接入实时服务后将显示真实数据。</span>
    </div>
  );
}

export function EmptyNote({ label = "暂无数据" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 rounded-mast-ctl border border-dashed border-mast-border-strong px-3 py-2.5 text-sm text-mast-muted">
      <IconInbox className="h-[15px] w-[15px] shrink-0" />
      <span>{label}</span>
    </div>
  );
}
