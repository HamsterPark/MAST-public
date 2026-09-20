// ════════════════════════════════════════════════════════════════════════════
// 拓扑图画布必须装得下所有成员 —— 结构闸门。
//
//     cd frontend && npm run test:unit
//
// Why this is a TEST and not a code review note: an SVG silently declines to
// paint anything outside its viewBox. No error, no warning, no console line.
// When research_director was added to PIPELINE the canvas width
// was still the literal `1120` that had fitted exactly six modules, so
// paper_review's box ran from x=1112 to x=1244 and **just wasn't there**.
// The only symptom was that part of the topology diagram looked covered up.
//
// 这道闸问的是两件不同的事，缺一不可：
//
//   ① 按当前成员数算，最后一个模块的右沿在画布里 —— 「现在是对的」。
//   ② 画布宽度是**推导**出来的，不是一个字面量 —— 「下一个人加第八个 agent
//      时它还会是对的」。只有 ① 的话，这道闸会在真正需要它的那一天绿着。
//
// 修复项预判到了这个形态（「三个成员不会
// 挤坏布局」），但没人在浏览器里看过 —— 于是它照常发布了。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { describe, it } from "node:test";

const HERE = fileURLToPath(new URL(".", import.meta.url));
const AGENTS = join(HERE, "..", "src", "components", "agents");
const GRAPH = readFileSync(join(AGENTS, "TopologyGraph.tsx"), "utf8");
const REGISTRY = readFileSync(join(AGENTS, "registry.tsx"), "utf8");

/** `const NAME = <number>;` at module scope. */
function num(src: string, name: string): number {
  const m = new RegExp(`^const ${name} = (-?\\d+(?:\\.\\d+)?);`, "m").exec(src);
  assert.ok(m, `TopologyGraph.tsx 里找不到 const ${name} = <数字>`);
  return Number(m![1]);
}

function pipelineMembers(): string[] {
  const m = /export const PIPELINE = \[([\s\S]*?)\];/.exec(REGISTRY);
  assert.ok(m, "registry.tsx 里找不到 export const PIPELINE = [...]");
  return [...m![1].matchAll(/"([a-z_]+)"/g)].map((x) => x[1]);
}

describe("拓扑图画布", () => {
  it("装得下 PIPELINE 里的每一个模块", () => {
    const members = pipelineMembers();
    assert.ok(members.length >= 6, `PIPELINE 只解析出 ${members.length} 个成员——正则该修了`);

    const ROW_X = num(GRAPH, "ROW_X");
    const MOD_W = num(GRAPH, "MOD_W");
    const MOD_GAP = num(GRAPH, "MOD_GAP");
    const RIGHT_PAD = num(GRAPH, "RIGHT_PAD");

    // 与源码里 modX() 同一条式子；这里重算是为了不 import TSX
    const lastRight = ROW_X + (members.length - 1) * (MOD_W + MOD_GAP) + MOD_W;

    const wExpr = /^const W = (.+);$/m.exec(GRAPH);
    assert.ok(wExpr, "TopologyGraph.tsx 里找不到 const W = ...");
    const floor = /Math\.max\((\d+),/.exec(wExpr![1]);
    assert.ok(floor, "W 应当带一个 Math.max(<下限>, …) 的最小画布宽");

    const W = Math.max(Number(floor![1]), lastRight + RIGHT_PAD);

    assert.ok(
      lastRight <= W,
      `${members.length} 个模块的最右沿是 ${lastRight}，画布只有 ${W} —— ` +
        `最后 ${members.length ? members[members.length - 1] : "?"} 会被 viewBox 裁掉 ` +
        `${lastRight - W} px（不报错、不告警，只是不画）`,
    );
  });

  it("画布宽度是推导出来的，不是写死的数字", () => {
    const m = /^const W = (.+);$/m.exec(GRAPH);
    assert.ok(m, "TopologyGraph.tsx 里找不到 const W = ...");
    const expr = m![1];

    assert.ok(
      /PIPELINE\.length/.test(expr),
      `const W = ${expr} —— 它没有引用 PIPELINE.length。` +
        `写死画布宽度会让下一个加进 PIPELINE 的 agent 被无声裁掉，` +
        `而上面那条测试届时**照样绿**（它按同一个字面量算）。`,
    );
    assert.ok(
      !/^\s*\d+\s*$/.test(expr),
      `const W = ${expr} —— 又变回字面量了`,
    );
  });

  it("总线画到画布右缘之内", () => {
    const m = /^const BUS_X1 = (.+);$/m.exec(GRAPH);
    assert.ok(m, "找不到 const BUS_X1");
    assert.ok(
      /W\s*-/.test(m![1]),
      `const BUS_X1 = ${m![1]} —— 总线的右端要跟着 W 走，` +
        `否则画布变宽之后总线停在半路，最右边几个模块挂在一条断掉的轨上`,
    );
  });
});
