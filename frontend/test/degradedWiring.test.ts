// ════════════════════════════════════════════════════════════════════════════
// 降级态接线闭环 — a panel's `degraded` must come from the SAME query that
// supplies its data.
//
//     cd frontend && npm run test:unit
//
// ## 为什么这是一道闸，而不是一次扫一遍
//
// 2026-08-14，MonitoringPage 把 **status** 端点的 `degraded` 传给了告警表，而
// 告警来自 `/api/monitoring/alerts` —— 另一个端点，有它自己的 `degraded`。
// 后端那一侧当天刚改诚实（`alerts_query` 查询失败改抛 `StoreQueryFailed`，不再
// 回 `([], 0)`），可面板照样画「暂无告警」：
//
//     <AlertsTable alerts={alerts.data?.alerts ?? []} degraded={degraded} />
//                         ^^^^^^ alerts 端点          ^^^^^^^^ status 端点
//
// 空列表 + 一个说「一切正常」的旗子 = 一句**正面断言**：「我查过了，这段时间零条
// 告警」。而真相是一条都没读到。同一页上另外三块都写对了（`degraded || (…
// data?.degraded)`），只有这一块没写 —— 这正是人眼扫一遍抓不住的形状：四个几乎
// 一样的表达式里少了半句。
//
// 折叠可以发生在任何一层。一个诚实的后端配一个接错旗子的前端，合起来还是在撒谎，
// 所以后端修好并不让这条闸变得多余。
//
// ## 判据
//
// 对每个带 `degraded={…}` 的 JSX 元素：它别的 prop 里出现的每一个查询
// （`Q.data…` 这个写法），都必须在 `degraded={…}` 那个表达式里被提到
// （`Q.data`）。多个数据源就多 OR 几个。
//
// 不带数据 prop 的元素不受约束 —— `<SegmentBrowser degraded={degraded} />`
// 自己在内部再 OR 一次它自己那个查询的（见该文件），这条规则管不到也不该管。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { readFileSync, readdirSync, statSync } from "node:fs";
import { join, relative } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const SRC = join(HERE, "..", "src");

function sourceFiles(dir: string, out: string[] = []): string[] {
  for (const name of readdirSync(dir)) {
    const p = join(dir, name);
    if (statSync(p).isDirectory()) sourceFiles(p, out);
    else if (/\.tsx$/.test(name)) out.push(p);
  }
  return out;
}

/**
 * The text between `<Component` and the `>` that closes its opening tag.
 *
 * Brace- and string-aware: a prop value like `{a > b ? x : y}` contains a `>`
 * that does NOT end the tag, and a string prop may contain either. Only
 * capitalised tags are scanned — a lowercase HTML element never takes a
 * `degraded` prop, and `<` on lowercase is also where TSX generics live.
 */
function openingTags(src: string): { tag: string; attrs: string; index: number }[] {
  const out: { tag: string; attrs: string; index: number }[] = [];
  for (const m of src.matchAll(/<([A-Z][\w.]*)/g)) {
    const start = m.index + m[0].length;
    let depth = 0;
    let quote = "";
    let i = start;
    for (; i < src.length; i++) {
      const c = src[i]!;
      if (quote) {
        if (c === "\\") i++;
        else if (c === quote) quote = "";
        continue;
      }
      if (c === '"' || c === "'" || c === "`") quote = c;
      else if (c === "{") depth++;
      else if (c === "}") depth--;
      else if (c === ">" && depth === 0) break;
    }
    if (i < src.length) out.push({ tag: m[1]!, attrs: src.slice(start, i), index: m.index });
  }
  return out;
}

/** The balanced `{…}` expression of `name={…}`, or null when the prop is absent. */
function propExpr(attrs: string, name: string): string | null {
  const re = new RegExp(`\\b${name}\\s*=\\s*\\{`);
  const m = re.exec(attrs);
  if (!m) return null;
  let depth = 0;
  const open = m.index + m[0].length - 1;
  for (let i = open; i < attrs.length; i++) {
    const c = attrs[i]!;
    if (c === "{") depth++;
    else if (c === "}") {
      depth--;
      if (depth === 0) return attrs.slice(open + 1, i);
    }
  }
  return null;
}

/** Query identifiers used as `Q.data` in a chunk of JSX attribute text. */
function queriesIn(text: string): Set<string> {
  const out = new Set<string>();
  for (const m of text.matchAll(/\b([A-Za-z_$][\w$]*)\s*\.\s*data\b/g)) out.add(m[1]!);
  return out;
}

/**
 * Splice in the definition of every bare identifier the expression names.
 *
 * A page-level `const degraded = status.data?.degraded ?? false` IS a reading of
 * the status query, just through a name — MonitoringPage writes it exactly that
 * way and it is correct. Without this the gate would demand `status.data` be
 * repeated inline, i.e. it would push people toward noise to silence it, and a
 * gate you satisfy by making the code worse gets turned off. One level only:
 * chains deeper than that are rare enough to be worth spelling out.
 */
function expandAliases(expr: string, src: string): string {
  let out = expr;
  for (const m of expr.matchAll(/\b([A-Za-z_$][\w$]*)\b(?!\s*[.(])/g)) {
    const def = new RegExp(`\\bconst\\s+${m[1]!}\\s*=([^;]*);`).exec(src);
    if (def) out += ` ${def[1]}`;
  }
  return out;
}

type Violation = { tag: string; missing: string[]; line: number };

/** Every element whose `degraded` ignores a query it renders data from. */
function violations(src: string): Violation[] {
  const out: Violation[] = [];
  for (const el of openingTags(src)) {
    const degraded = propExpr(el.attrs, "degraded");
    if (degraded === null) continue;
    const fromDegraded = queriesIn(expandAliases(degraded, src));
    // Everything EXCEPT the degraded expression itself.
    const rest = el.attrs.replace(degraded, " ");
    const missing = [...queriesIn(rest)].filter((q) => !fromDegraded.has(q));
    if (missing.length) {
      out.push({
        tag: el.tag,
        missing,
        line: src.slice(0, el.index).split("\n").length,
      });
    }
  }
  return out;
}

describe("degraded 与它所渲染的数据必须同源", () => {
  // ── the checker proves it can see the bug ─────────────────────────────────
  // A structural gate that cannot detect the defect it was written for is
  // decoration. Both halves of the 2026-08-14 fix are replayed here as text, so
  // a future refactor of the scanner cannot quietly stop catching it.
  it("认得出 2026-08-14 那一行（修之前）", () => {
    const before = `<AlertsTable alerts={alerts.data?.alerts ?? []} degraded={degraded} />`;
    const found = violations(before);
    assert.equal(found.length, 1, "扫描器没抓到已知的那个缺陷");
    assert.deepEqual(found[0]!.missing, ["alerts"]);
  });

  it("别名展开没有把闸门放松掉", () => {
    // The real file HAS `const degraded = status.data?.degraded ?? false`, and
    // the bug lived right next to it. Expanding the alias must resolve it to the
    // STATUS query — not to "some query, close enough".
    const withAlias =
      `const degraded = status.data?.degraded ?? false;\n` +
      `<AlertsTable alerts={alerts.data?.alerts ?? []} degraded={degraded} />`;
    const found = violations(withAlias);
    assert.equal(found.length, 1, "别名展开之后就抓不到那个缺陷了");
    assert.deepEqual(found[0]!.missing, ["alerts"]);
  });

  it("别名指到同一个查询时放行", () => {
    const sameQuery =
      `const degraded = status.data?.degraded ?? false;\n` +
      `<AuxChannels aux={status.data?.aux ?? null} degraded={degraded} />`;
    assert.deepEqual(violations(sameQuery), []);
  });

  it("修之后不再报", () => {
    const after =
      `<AlertsTable alerts={alerts.data?.alerts ?? []} ` +
      `degraded={degraded || (alerts.data?.degraded ?? false)} detail={alerts.data?.detail} />`;
    assert.deepEqual(violations(after), []);
  });

  it("没有数据 prop 的元素不受约束", () => {
    assert.deepEqual(violations(`<SegmentBrowser degraded={degraded} />`), []);
  });

  it("两个数据源就要两个都提到", () => {
    const twoSources =
      `<Panel rows={a.data?.rows ?? []} extra={b.data?.x} degraded={a.data?.degraded ?? false} />`;
    const found = violations(twoSources);
    assert.equal(found.length, 1);
    assert.deepEqual(found[0]!.missing, ["b"]);
  });

  // ── the repo itself ──────────────────────────────────────────────────────
  it("src 里没有接错端点的 degraded", () => {
    const bad: string[] = [];
    for (const file of sourceFiles(SRC)) {
      for (const v of violations(readFileSync(file, "utf8"))) {
        bad.push(
          `${relative(SRC, file)}:${v.line} <${v.tag}> 渲染 ` +
            `${v.missing.map((q) => `${q}.data`).join(" / ")} 的数据，` +
            `但 degraded 没有读它自己的 —— 那个查询失败时这块会画成「空 = 正常」。` +
            `写成 degraded={degraded || (${v.missing[0]}.data?.degraded ?? false)}。`,
        );
      }
    }
    assert.deepEqual(bad, [], `\n${bad.join("\n")}\n`);
  });
});
