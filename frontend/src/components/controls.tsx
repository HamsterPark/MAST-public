import { type ReactNode, type SVGProps, useEffect, useState } from "react";
import clsx from "clsx";

/** Shared interactive primitives for the parity rebuild — sub-tab navigation,
 *  form controls, modal. In-page sub-tabs replace the old Gradio nested Tabs
 *  (which froze); here they are plain React state, no freeze class.
 *  Restyled onto the MAST Lab Console design system — color/radius/shadow come
 *  from design tokens only, so both themes + density inherit automatically. */

/* Inline loader-circle (no icon dependency installed); spin via caller class. */
function IconLoader({ className, ...rest }: SVGProps<SVGSVGElement>) {
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
      <path d="M21 12a9 9 0 1 1-6.219-8.56" />
    </svg>
  );
}

/** Sub-tab navigation — segmented pill rail matching the console SubTabs
 *  pattern (panel-2 rail, accent active segment). Same props/handlers. */
export function SubTabs<T extends string>({
  tabs, value, onChange,
}: {
  tabs: { id: T; label: string; badge?: number | string }[];
  value: T;
  onChange: (id: T) => void;
}) {
  return (
    <div className="mb-4 inline-flex max-w-full flex-wrap gap-[3px] rounded-mast-ctl border border-mast-border bg-mast-panel-2 p-[3px]">
      {tabs.map((t) => {
        const active = value === t.id;
        return (
          <button
            key={t.id}
            onClick={() => onChange(t.id)}
            className={clsx(
              "inline-flex items-center gap-1.5 rounded-md px-3 py-1.5 text-sm transition-colors",
              active
                ? "bg-mast-accent font-semibold text-mast-accent-ink"
                : "text-mast-muted hover:text-mast-text",
            )}
            aria-pressed={active}
          >
            {t.label}
            {t.badge != null && (
              <span
                className={clsx(
                  "inline-flex h-[18px] min-w-[18px] items-center justify-center rounded-full px-1 font-mono text-[10.5px] tabular-nums",
                  // On the active (accent-filled) tab the count pill INVERTS —
                  // solid ink behind accent text. It used to be ink at 20% with
                  // ink text, i.e. white-on-pale-cyan in light mode (~2.8:1 at
                  // 10.5px). Same misuse of `accent-ink` as that token
                  // is the foreground for a SOLID accent fill, never for a tint
                  // of itself.
                  active
                    ? "bg-mast-accent-ink text-mast-accent"
                    : "bg-mast-accent text-mast-accent-ink",
                )}
              >
                {t.badge}
              </span>
            )}
          </button>
        );
      })}
    </div>
  );
}

const inputCls =
  "rounded-mast-ctl border border-mast-border-strong bg-mast-panel px-2.5 py-2 text-sm text-mast-text outline-none focus:border-mast-accent";

export function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-mast-muted">{label}</span>
      {children}
      {hint && <span className="text-xs text-mast-faint">{hint}</span>}
    </label>
  );
}

export function TextField({
  value, onChange, placeholder, mono,
}: { value: string; onChange: (v: string) => void; placeholder?: string; mono?: boolean }) {
  return (
    <input
      type="text"
      value={value}
      placeholder={placeholder}
      onChange={(e) => onChange(e.target.value)}
      className={clsx(inputCls, mono && "font-mono")}
    />
  );
}

export function NumberField({
  value, onChange, step,
}: { value: string; onChange: (v: string) => void; step?: string }) {
  return (
    <input type="text" inputMode="decimal" value={value} step={step}
      onChange={(e) => onChange(e.target.value)} className={clsx(inputCls, "font-mono tabular-nums")} />
  );
}

export function SelectField<T extends string>({
  value, onChange, options,
}: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[] }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value as T)} className={inputCls}>
      {options.map((o) => (
        <option key={o.value} value={o.value}>{o.label}</option>
      ))}
    </select>
  );
}

/** Segmented radio group — visual parity with the old gr.Radio (one row of
 *  pill choices, the selected one highlighted). Used for 字体大小 / 主题 /
 *  知识注入模式 / Thinking where the old GUI used gr.Radio, not a dropdown. */
export function RadioGroup<T extends string>({
  value, onChange, options,
}: { value: T; onChange: (v: T) => void; options: { value: T; label: string }[] }) {
  return (
    <div
      className="inline-flex flex-wrap gap-[3px] rounded-mast-ctl border border-mast-border bg-mast-panel-2 p-[3px]"
      role="radiogroup"
    >
      {options.map((o) => (
        <button
          key={o.value}
          type="button"
          onClick={() => onChange(o.value)}
          className={clsx(
            "rounded-md px-3 py-1.5 text-sm transition-colors",
            value === o.value
              ? "bg-mast-accent font-semibold text-mast-accent-ink"
              : "text-mast-muted hover:text-mast-text",
          )}
          role="radio"
          aria-checked={value === o.value}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

export function Checkbox({
  checked, onChange, label,
}: { checked: boolean; onChange: (v: boolean) => void; label: ReactNode }) {
  return (
    <label className="inline-flex cursor-pointer items-center gap-2 text-sm text-mast-text">
      <span className="relative inline-flex h-4 w-4 shrink-0 items-center justify-center">
        <input
          type="checkbox"
          checked={checked}
          onChange={(e) => onChange(e.target.checked)}
          className="peer absolute inset-0 cursor-pointer opacity-0"
        />
        <span
          className={clsx(
            "pointer-events-none inline-flex h-4 w-4 items-center justify-center rounded-[4px] transition-colors",
            checked
              ? "bg-mast-accent text-mast-accent-ink"
              : "border border-mast-border-strong",
          )}
        >
          {checked && (
            <svg
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth={3}
              strokeLinecap="round"
              strokeLinejoin="round"
              className="h-3 w-3"
              aria-hidden="true"
            >
              <path d="M20 6 9 17l-5-5" />
            </svg>
          )}
        </span>
      </span>
      <span>{label}</span>
    </label>
  );
}

export type ChoiceOption = { label: string; description?: string };

/** Stacked choice cards for an agent's ask_user question.
 *
 *  Distinct from RadioGroup (a one-row pill segment for settings): each option
 *  here carries a description, because what the operator is choosing between is
 *  a trade-off — "B 区 / 平坦台面，适合先做 STS 基线" is the actual decision, and
 *  a bare label makes them guess it. Handles single-select, multi-select, and
 *  the "其他" free-text escape in one control, so the three approval surfaces
 *  render an identical question.
 *
 *  With no options at all this degrades to just the text box — an open question.
 *  That is why `allowCustom` cannot be turned off in that case upstream: a
 *  question with neither options nor a text box is unanswerable. */
export function OptionGroup({
  options, multi, allowCustom, value, onChange, disabled,
}: {
  options: ChoiceOption[];
  multi: boolean;
  allowCustom: boolean;
  value: { selected: string[]; customText: string };
  onChange: (v: { selected: string[]; customText: string }) => void;
  disabled?: boolean;
}) {
  const customOpen = allowCustom && (value.customText.length > 0 || options.length === 0);
  const toggle = (label: string) => {
    if (disabled) return;
    if (multi) {
      const next = value.selected.includes(label)
        ? value.selected.filter((s) => s !== label)
        : [...value.selected, label];
      onChange({ ...value, selected: next });
    } else {
      // Single-select: picking again clears, so a mis-click is undoable without
      // a separate "clear" affordance.
      onChange({
        ...value,
        selected: value.selected[0] === label ? [] : [label],
      });
    }
  };
  return (
    <div className="flex flex-col gap-1.5">
      {options.map((o) => {
        const on = value.selected.includes(o.label);
        return (
          <button
            key={o.label}
            type="button"
            disabled={disabled}
            onClick={() => toggle(o.label)}
            role={multi ? "checkbox" : "radio"}
            aria-checked={on}
            className={clsx(
              "flex w-full items-start gap-2.5 rounded-mast-ctl border px-3 py-2 text-left transition-colors",
              disabled && "cursor-not-allowed opacity-60",
              on
                ? "border-mast-accent bg-mast-accent/10"
                : "border-mast-border bg-mast-panel-2 hover:border-mast-border-strong",
            )}
          >
            <span
              className={clsx(
                "mt-0.5 inline-flex h-4 w-4 shrink-0 items-center justify-center transition-colors",
                multi ? "rounded-[4px]" : "rounded-full",
                on ? "bg-mast-accent text-mast-accent-ink" : "border border-mast-border-strong",
              )}
              aria-hidden="true"
            >
              {on && (
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={3}
                  strokeLinecap="round" strokeLinejoin="round" className="h-3 w-3">
                  <path d="M20 6 9 17l-5-5" />
                </svg>
              )}
            </span>
            <span className="min-w-0 flex-1">
              <span className="block text-sm text-mast-text">{o.label}</span>
              {o.description && (
                <span className="mt-0.5 block text-xs text-mast-muted">{o.description}</span>
              )}
            </span>
          </button>
        );
      })}
      {allowCustom && (
        <div className="flex flex-col gap-1">
          {options.length > 0 && !customOpen && (
            <button
              type="button"
              disabled={disabled}
              onClick={() => onChange({ ...value, customText: " " })}
              className={clsx(
                "w-full rounded-mast-ctl border border-dashed border-mast-border px-3 py-2 text-left text-sm text-mast-muted transition-colors hover:border-mast-border-strong hover:text-mast-text",
                disabled && "cursor-not-allowed opacity-60",
              )}
            >
              其他 / 自己写一个答案…
            </button>
          )}
          {customOpen && (
            <textarea
              value={value.customText}
              disabled={disabled}
              autoFocus={options.length > 0}
              onChange={(e) => onChange({ ...value, customText: e.target.value })}
              rows={options.length === 0 ? 4 : 2}
              placeholder={options.length === 0 ? "写下你的回答…" : "补充说明，或写一个上面没有的选项…"}
              className="w-full resize-y rounded-md border border-mast-border bg-mast-bg px-3 py-2 text-sm text-mast-text outline-none focus:border-mast-accent disabled:opacity-60"
            />
          )}
        </div>
      )}
    </div>
  );
}

/** Collapsible accordion — parity with gr.Accordion (clickable title row with a
 *  ▸/▾ caret, open/closed). Defaults open. Replaces the page's SubTabs so the
 *  设置 / 记忆 surfaces stack as collapsible sections like the old Gradio build
 *  (no nested gr.Tabs — and no freeze, since it is plain React state). */
/**
 * Collapsible panel. Uncontrolled by default; pass `open` + `onOpenChange` to
 * drive it from outside — 新仪器初始化 does that so a search hit can open the
 * group holding it (an item the operator cannot find is an item they cannot fix,
 * which was the 2026-08-04 complaint about 退针方向).
 */
export function Accordion({
  title, defaultOpen = true, open: controlled, onOpenChange, children,
}: {
  title: ReactNode;
  defaultOpen?: boolean;
  open?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: ReactNode;
}) {
  const [uncontrolled, setUncontrolled] = useState(defaultOpen);
  const open = controlled ?? uncontrolled;
  const toggle = () => {
    if (controlled === undefined) setUncontrolled((v) => !v);
    onOpenChange?.(!open);
  };
  return (
    <div className="mb-3 overflow-hidden rounded-mast-card border border-mast-border bg-mast-panel shadow-mast">
      <button
        type="button"
        onClick={toggle}
        aria-expanded={open}
        className="flex w-full items-center justify-between px-4 py-3 text-left text-sm font-semibold text-mast-text hover:bg-mast-panel-2"
      >
        <span>{title}</span>
        <span className="text-mast-muted">{open ? "▾" : "▸"}</span>
      </button>
      {open && <div className="border-t border-mast-border p-4">{children}</div>}
    </div>
  );
}

export function Toggle({ checked, onChange, label }: { checked: boolean; onChange: (v: boolean) => void; label?: string }) {
  return (
    <button
      type="button"
      onClick={() => onChange(!checked)}
      className={clsx(
        "inline-flex h-5 w-9 items-center rounded-full transition-colors",
        checked ? "bg-mast-accent" : "bg-mast-border",
      )}
      role="switch"
      aria-checked={checked}
      aria-label={label}
    >
      <span className={clsx("inline-block h-4 w-4 transform rounded-full bg-white transition-transform",
        checked ? "translate-x-4" : "translate-x-0.5")} />
    </button>
  );
}

export function Button({
  children, onClick, variant = "default", disabled, loading, type = "button",
}: {
  children: ReactNode; onClick?: () => void; disabled?: boolean; loading?: boolean;
  type?: "button" | "submit"; variant?: "default" | "primary" | "danger" | "ghost";
}) {
  // default === the spec's "secondary" (panel + strong border); primary fills
  // the accent; danger uses the danger triple; ghost is borderless accent text.
  const v = {
    default: "bg-mast-panel text-mast-text border border-mast-border-strong hover:bg-mast-panel-2",
    primary: "bg-mast-accent text-mast-accent-ink font-semibold hover:opacity-90",
    danger: "bg-mast-danger-bg text-mast-danger border border-mast-danger-border font-semibold hover:opacity-90",
    ghost: "text-mast-accent hover:text-mast-accent hover:bg-mast-accent-soft",
  }[variant];
  return (
    <button
      type={type}
      onClick={onClick}
      disabled={disabled || loading}
      aria-busy={loading || undefined}
      className={clsx(
        "inline-flex items-center justify-center gap-1.5 rounded-mast-ctl px-3.5 py-2 text-sm disabled:cursor-not-allowed disabled:opacity-50",
        v,
        loading && "opacity-85",
      )}
    >
      {loading && <IconLoader className="h-3.5 w-3.5 animate-spin" />}
      {children}
    </button>
  );
}

export function Modal({ open, onClose, title, children, wide }: {
  open: boolean; onClose: () => void; title: string; children: ReactNode; wide?: boolean;
}) {
  useEffect(() => {
    if (!open) return;
    const h = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    document.addEventListener("keydown", h);
    return () => document.removeEventListener("keydown", h);
  }, [open, onClose]);
  if (!open) return null;
  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-auto bg-black/60 p-6 backdrop-blur-sm"
      onClick={onClose}>
      <div
        className={clsx("mt-10 w-full rounded-mast-card border border-mast-border bg-mast-panel shadow-mast",
          wide ? "max-w-4xl" : "max-w-xl")}
        onClick={(e) => e.stopPropagation()}
      >
        <div className="flex items-center justify-between border-b border-mast-border bg-mast-panel-2 px-5 py-3">
          <h3 className="text-base font-semibold text-mast-text">{title}</h3>
          <button onClick={onClose} className="text-mast-muted hover:text-mast-text">✕</button>
        </div>
        <div className="p-5">{children}</div>
      </div>
    </div>
  );
}

/** Lightweight transient toast (no provider needed for simple cases). */
export function useToast() {
  const [msg, setMsg] = useState<{ text: string; tone: "ok" | "err" } | null>(null);
  useEffect(() => {
    if (!msg) return;
    const t = setTimeout(() => setMsg(null), 3500);
    return () => clearTimeout(t);
  }, [msg]);
  const node = msg ? (
    <div className={clsx(
      "fixed bottom-6 right-6 z-50 rounded-mast-ctl border px-4 py-2.5 text-sm shadow-mast",
      msg.tone === "ok" ? "border-mast-auto-border bg-mast-auto-bg text-mast-auto"
        : "border-mast-danger-border bg-mast-danger-bg text-mast-danger")}>
      {msg.text}
    </div>
  ) : null;
  return { toast: (text: string, tone: "ok" | "err" = "ok") => setMsg({ text, tone }), node };
}
