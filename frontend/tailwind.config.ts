import type { Config } from "tailwindcss";

// Dark mode via a class on <html> (the store toggles `dark`). Colors map to the
// --mast-* "Lab Console" design tokens defined in src/index.css (landed from the
// Claude-design output). Every key here becomes a Tailwind utility, e.g.
// text-mast-auto / bg-mast-panel-2 / border-mast-info-border / text-mast-ag-ic.
export default {
  darkMode: "class",
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        mast: {
          // base scale
          bg: "var(--mast-bg)",
          panel: "var(--mast-panel)",
          "panel-2": "var(--mast-panel-2)",
          border: "var(--mast-border)",
          "border-strong": "var(--mast-border-strong)",
          text: "var(--mast-text)",
          muted: "var(--mast-muted)",
          faint: "var(--mast-faint)",
          "code-bg": "var(--mast-code-bg)",
          // accent (electric cyan)
          accent: "var(--mast-accent)",
          "accent-soft": "var(--mast-accent-soft)",
          "accent-line": "var(--mast-accent-line)",
          "accent-ink": "var(--mast-accent-ink)",
          // semantic safety colors (single source — fg / bg / border each)
          auto: "var(--mast-auto)",
          "auto-bg": "var(--mast-auto-bg)",
          "auto-border": "var(--mast-auto-border)",
          info: "var(--mast-info)",
          "info-bg": "var(--mast-info-bg)",
          "info-border": "var(--mast-info-border)",
          warn: "var(--mast-warn)",
          "warn-bg": "var(--mast-warn-bg)",
          "warn-border": "var(--mast-warn-border)",
          danger: "var(--mast-danger)",
          "danger-bg": "var(--mast-danger-bg)",
          "danger-border": "var(--mast-danger-border)",
          // special-purpose
          dream: "var(--mast-dream)",
          brainstorm: "var(--mast-brainstorm)",
          // per-agent palette
          "ag-sup": "var(--mast-ag-sup)",
          "ag-rd": "var(--mast-ag-rd)",
          "ag-lit": "var(--mast-ag-lit)",
          "ag-xd": "var(--mast-ag-xd)",
          "ag-ic": "var(--mast-ag-ic)",
          "ag-dp": "var(--mast-ag-dp)",
          "ag-pw": "var(--mast-ag-pw)",
          "ag-pr": "var(--mast-ag-pr)",
        },
      },
      borderRadius: {
        "mast-card": "var(--mast-r-card)",
        "mast-ctl": "var(--mast-r-ctl)",
        "mast-badge": "var(--mast-r-badge)",
      },
      boxShadow: {
        mast: "var(--mast-shadow)",
      },
      backgroundImage: {
        "mast-topbar": "var(--mast-topbar)",
      },
    },
  },
  plugins: [],
} satisfies Config;
