// ════════════════════════════════════════════════════════════════════════════
// 设计令牌闭环 — every `*-mast-<token>` class a component writes must exist in
// tailwind.config.ts, and every `var(--mast-…)` the config maps to must exist in
// index.css.
//
//     cd frontend && npm run test:unit
//
// Why this is a TEST and not a lint rule: an undefined Tailwind class is not an
// error anywhere. Tailwind simply emits no rule for it, the element keeps
// whatever it inherited, and the page still renders — so the ONLY symptom is
// that it becomes hard to read. That has now happened three times:
//
//   · ① — a batch of components used `--fg-muted` / `--accent` /
//     `--border` / `--bg-elevated`, none of which exist (the prefix is --mast-*).
//   · — 文档筛选标签 in light mode. Not an undefined token this time
//     but the same silence: `text-mast-accent-ink` (white, meant for a SOLID
//     accent fill) over `bg-mast-accent-soft` (a 9% tint) = white on white.
//   · found while sweeping for #61 — `text-mast-ok` (TipControls) and
//     `hover:text-mast-fg` (VisionPage) had been dead since they were written.
//
//   · — the SAME 文档筛选标签, reported again against the #61 fix. The
//     class was right by then; the token was not. `--mast-accent` was 4.6:1 on
//     white, and a selected chip never sits on white — it sits on a 9% tint of
//     the accent, where the same ink is 3.5:1.
//
// That last one is why the paragraph this replaces ("the contrast half cannot be
// checked from source text") was wrong, and expensively so: it read as a closed
// question, so nobody re-opened it, and the fix shipped one layer too shallow.
// Both halves ARE checkable — the tokens are literal hex/rgba in index.css and
// compositing a known alpha over a known backdrop is arithmetic. See the
// "token contrast" block below.
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const SRC = join(HERE, "..", "src");
const CONFIG = join(HERE, "..", "tailwind.config.ts");
const INDEX_CSS = join(SRC, "index.css");

function sourceFiles(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) sourceFiles(p, out);
    else if (/\.tsx?$/.test(name)) out.push(p);
  }
  return out;
}

/** Keys of one object literal in the config, e.g. the `mast: { … }` colour map. */
function objectKeys(config: string, openerRe: RegExp): Set<string> {
  const m = openerRe.exec(config);
  assert.ok(m, `tailwind.config.ts no longer contains ${openerRe}`);
  let depth = 0;
  let i = m.index + m[0].length - 1; // sits on the `{`
  const start = i;
  for (; i < config.length; i++) {
    if (config[i] === "{") depth++;
    else if (config[i] === "}") {
      depth--;
      if (depth === 0) break;
    }
  }
  // Strip `//` comments first: the config groups its tokens under section
  // comments, and a key sitting on the line after one is not reachable from the
  // preceding comma.
  const body = config.slice(start + 1, i).replace(/\/\/[^\n]*/g, "");
  const keys = new Set<string>();
  for (const km of body.matchAll(/^\s*"?([a-zA-Z][a-zA-Z0-9-]*)"?\s*:/gm)) {
    keys.add(km[1]!);
  }
  return keys;
}

const config = readFileSync(CONFIG, "utf8");
const colorTokens = objectKeys(config, /\bmast:\s*\{/);
const radiusTokens = objectKeys(config, /\bborderRadius:\s*\{/);
const shadowTokens = objectKeys(config, /\bboxShadow:\s*\{/);
const bgImageTokens = objectKeys(config, /\bbackgroundImage:\s*\{/);

// Utilities that resolve against theme.colors. `shadow-` and `rounded-` resolve
// against their own scales, so they are checked separately below.
const COLOR_UTILS =
  "text|bg|border|fill|stroke|ring|outline|divide|placeholder|caret|decoration|from|to|via|accent|shadow";

describe("tailwind mast-* class names all resolve", () => {
  it("every colour-ish utility names a token in tailwind.config.ts", () => {
    const bad: string[] = [];
    for (const file of sourceFiles(SRC)) {
      const text = readFileSync(file, "utf8");
      for (const m of text.matchAll(
        new RegExp(String.raw`\b(${COLOR_UTILS})-mast-([a-z0-9-]+)`, "g"),
      )) {
        const util = m[1]!;
        const token = m[2]!;
        // `bg-mast-topbar` is a backgroundImage, and `shadow-mast` a boxShadow —
        // both live in their own scale, not in colors.mast.
        if (util === "bg" && bgImageTokens.has(`mast-${token}`)) continue;
        if (util === "shadow" && shadowTokens.has(`mast-${token}`)) continue;
        if (!colorTokens.has(token)) bad.push(`${file}: ${m[0]}`);
      }
    }
    assert.deepEqual(bad, [], `undefined mast-* colour tokens:\n${bad.join("\n")}`);
  });

  it("every rounded-mast-* names a borderRadius token", () => {
    const bad: string[] = [];
    for (const file of sourceFiles(SRC)) {
      const text = readFileSync(file, "utf8");
      for (const m of text.matchAll(/\brounded-(mast-[a-z0-9-]+)/g)) {
        if (!radiusTokens.has(m[1]!)) bad.push(`${file}: ${m[0]}`);
      }
    }
    assert.deepEqual(bad, [], `undefined rounded-mast-* tokens:\n${bad.join("\n")}`);
  });
});

describe("CSS custom properties all resolve", () => {
  const css = readFileSync(INDEX_CSS, "utf8");
  // index.css packs three declarations onto one line (the semantic triples and
  // the agent palette), so this must NOT be line-anchored. The lookbehind drops
  // `var(--x)` READS — index.css reads --mast-accent-line in the focus ring, and
  // counting a read as a definition would make this check vacuous.
  const defined = new Set(
    [...css.matchAll(/(?<!var\()(--[a-z0-9-]+)\s*:/g)].map((m) => m[1]!),
  );

  it("index.css defines the light and dark halves of every token", () => {
    // A token defined only under html.dark is invisible in light mode and vice
    // versa — the shape of #61, one layer down.
    const root = /:root\s*\{([\s\S]*?)\n\}/.exec(css);
    const dark = /html\.dark\s*\{([\s\S]*?)\n\}/.exec(css);
    assert.ok(root && dark, "index.css lost its :root / html.dark blocks");
    const names = (block: string) =>
      new Set([...block.matchAll(/(--mast-[a-z0-9-]+)\s*:/g)].map((m) => m[1]!));
    const lightNames = names(root[1]!);
    const darkNames = names(dark[1]!);
    // Geometry tokens (radii, paddings) are theme-independent by design and are
    // declared once, in :root. Only the PAINT has to exist in both.
    const paint = (n: string) => !/^--mast-(r-|row-|card-pad|sec-gap)/.test(n);
    const missingInDark = [...lightNames].filter((n) => paint(n) && !darkNames.has(n));
    assert.deepEqual(missingInDark, [], "defined for light but not dark");
    const missingInLight = [...darkNames].filter((n) => !lightNames.has(n));
    assert.deepEqual(missingInLight, [], "defined for dark but not light");
  });

  it("tailwind.config.ts only maps vars that index.css defines", () => {
    const bad = [...config.matchAll(/var\((--[a-z0-9-]+)\)/g)]
      .map((m) => m[1]!)
      .filter((v) => !defined.has(v));
    assert.deepEqual(bad, [], `tailwind maps undefined vars: ${bad.join(", ")}`);
  });

  it("no component references a var index.css does not define", () => {
    const bad: string[] = [];
    for (const file of sourceFiles(SRC)) {
      const text = readFileSync(file, "utf8");
      for (const m of text.matchAll(/var\((--[a-z0-9-]+)\)/g)) {
        if (!defined.has(m[1]!)) bad.push(`${file}: ${m[0]}`);
      }
    }
    assert.deepEqual(bad, [], `undefined CSS vars:\n${bad.join("\n")}`);
  });
});

// ── token contrast (the half #61 declared uncheckable, and #13 disproved) ──

type Rgb = readonly [number, number, number];

function parseColor(v: string): Rgb | null {
  const hex = /^#([0-9a-f]{6})$/i.exec(v.trim());
  if (hex) {
    const n = parseInt(hex[1]!, 16);
    return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
  }
  const rgba = /^rgba?\(\s*([\d.]+)[,\s]+([\d.]+)[,\s]+([\d.]+)/i.exec(v.trim());
  if (rgba) return [+rgba[1]!, +rgba[2]!, +rgba[3]!];
  return null;
}

/** Alpha of an rgba() literal; 1 for anything opaque. */
function parseAlpha(v: string): number {
  const m = /^rgba\(\s*[\d.]+[,\s]+[\d.]+[,\s]+[\d.]+[,\s/]+([\d.]+)\s*\)/i.exec(v.trim());
  return m ? +m[1]! : 1;
}

function over(fg: Rgb, alpha: number, bg: Rgb): Rgb {
  return [0, 1, 2].map((i) => bg[i]! + (fg[i]! - bg[i]!) * alpha) as unknown as Rgb;
}

/** WCAG 2.x relative luminance. */
function luminance([r, g, b]: Rgb): number {
  const lin = (c: number) => {
    const s = c / 255;
    return s <= 0.03928 ? s / 12.92 : ((s + 0.055) / 1.055) ** 2.4;
  };
  return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b);
}

function contrast(a: Rgb, b: Rgb): number {
  const [hi, lo] = [luminance(a), luminance(b)].sort((x, y) => y - x);
  return (hi! + 0.05) / (lo! + 0.05);
}

describe("token contrast holds in BOTH themes", () => {
  const css = readFileSync(INDEX_CSS, "utf8");

  /** Raw declared value of one token inside one theme block. */
  function tokensOf(selector: RegExp): Map<string, string> {
    const m = selector.exec(css);
    assert.ok(m, `index.css lost the ${selector} block`);
    const out = new Map<string, string>();
    for (const d of m[1]!.matchAll(/(--mast-[a-z0-9-]+)\s*:\s*([^;]+);/g)) {
      out.set(d[1]!, d[2]!.trim());
    }
    return out;
  }

  const THEMES = [
    ["light", tokensOf(/:root\s*\{([\s\S]*?)\n\}/)],
    ["dark", tokensOf(/html\.dark\s*\{([\s\S]*?)\n\}/)],
  ] as const;

  // AA for normal text. Chips render at 10.5–12px, i.e. never "large text", so
  // the 3:1 large-text allowance does not apply to any of these.
  const AA = 4.5;

  for (const [theme, tok] of THEMES) {
    it(`${theme}: text-mast-accent is legible on bg-mast-accent-soft`, () => {
      // The selected-chip recipe used by DocumentsPane, ScanDataPane, AppLayout's
      // top nav, LiveCurrentChart's window picker, Badge's default tone, … — the
      // single most repeated colour pair in the app.
      const ink = parseColor(tok.get("--mast-accent")!)!;
      const tintRgb = parseColor(tok.get("--mast-accent-soft")!)!;
      const alpha = parseAlpha(tok.get("--mast-accent-soft")!);
      const panel = parseColor(tok.get("--mast-panel")!)!;
      const ratio = contrast(ink, over(tintRgb, alpha, panel));
      assert.ok(
        ratio >= AA,
        `--mast-accent on --mast-accent-soft over --mast-panel is ${ratio.toFixed(2)}:1 in ${theme}, need ${AA}:1`,
      );
    });

    it(`${theme}: text-mast-accent is legible directly on the panel`, () => {
      const ratio = contrast(
        parseColor(tok.get("--mast-accent")!)!,
        parseColor(tok.get("--mast-panel")!)!,
      );
      assert.ok(ratio >= AA, `--mast-accent on --mast-panel is ${ratio.toFixed(2)}:1 in ${theme}`);
    });

    it(`${theme}: accent-ink is legible on a SOLID accent fill`, () => {
      // The other half of #61: accent-ink is only ever valid over solid accent.
      const ratio = contrast(
        parseColor(tok.get("--mast-accent-ink")!)!,
        parseColor(tok.get("--mast-accent")!)!,
      );
      assert.ok(ratio >= AA, `--mast-accent-ink on --mast-accent is ${ratio.toFixed(2)}:1 in ${theme}`);
    });

    it(`${theme}: body text and muted text clear AA on the page background`, () => {
      for (const name of ["--mast-text", "--mast-muted"]) {
        const ratio = contrast(parseColor(tok.get(name)!)!, parseColor(tok.get("--mast-bg")!)!);
        assert.ok(ratio >= AA, `${name} on --mast-bg is ${ratio.toFixed(2)}:1 in ${theme}`);
      }
    });
  }

  it("the checker itself can fail", () => {
    // Without this, a broken parser would make every assertion above vacuously
    // true — the shape that let #61 ship twice.
    assert.ok(contrast([255, 255, 255], [254, 254, 254]) < AA);
    assert.ok(contrast([0, 0, 0], [255, 255, 255]) > 20);
    assert.deepEqual(over([0, 0, 0], 0.5, [255, 255, 255]), [127.5, 127.5, 127.5]);
    assert.equal(parseAlpha("rgba(11, 111, 131, 0.09)"), 0.09);
    assert.equal(parseAlpha("#0b6f83"), 1);
  });
});
