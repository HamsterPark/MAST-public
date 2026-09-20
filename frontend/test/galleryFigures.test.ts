// ════════════════════════════════════════════════════════════════════════════
// 数据图库 → 出图 的纯逻辑 — src/lib/gallery/figures.ts
//
//     cd frontend && npm run test:unit
//
// 使用合成系列验证跨批次、跨线和区组分组；断言逐项核对系列 ID。
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  CATEGORY_ORDER,
  allEntries,
  badFromToForm,
  blockLabel,
  blockNumber,
  commonLineName,
  describeStart,
  fileLinks,
  findPrefill,
  fmtBytes,
  fmtNumber,
  fmtSummaryValue,
  formFromOptions,
  formToBadFrom,
  fullImageUrl,
  groupFor,
  lineKey,
  normaliseLineName,
  optionsText,
  parseKappa,
  parsePositiveInt,
  previewUrl,
  rowsToStationMarks,
  sameSet,
  singleSpectraByDir,
  stationMarksToRows,
  stsLinesOptions,
  suggestLineGroups,
  summaryChips,
  withAllCategories,
} from "../src/lib/gallery/figures.ts";

// eslint-disable-next-line @typescript-eslint/no-explicit-any
type Any = any;

const S = (name: string, k: string, ids: string[] = []): Any => ({ name, k, ids, r: 0, tags: [], note: "" });

/** 合成的多批次系列；不含实验记录。 */
const SYNTHETIC_SERIES: Record<string, Any> = {
  fixture_series_00: S("2001-01-01 0001–0003 · 3 帧", "f"),
  fixture_series_01: S("批次A · 区组0 · L1 线A", "s"),
  fixture_series_02: S("批次A · 区组0 · L3 线C", "s"),
  fixture_series_03: S("批次A · 区组0 · L2 线B", "s"),
  fixture_series_04: S("批次A · 区组0 · L4 线D", "s"),
  fixture_series_05: S("批次A · 区组1 · L4 线D", "s"),
  fixture_series_06: S("批次A · 区组1 · L2 线B", "s"),
  fixture_series_07: S("批次A · 区组1 · L3 线C", "s"),
  fixture_series_08: S("批次A · 区组1 · L1 线A", "s"),
  fixture_series_09: S("批次A · 区组2 · L1 线A", "s"),
  fixture_series_10: S("批次A · 区组2 · L3 线C", "s"),
  fixture_series_11: S("批次A · 区组2 · L2 线B", "s"),
  fixture_series_12: S("批次A · 区组2 · L4 线D", "s"),
  fixture_series_13: S("批次B · 区组0 · 主线", "s"),
  fixture_series_14: S("批次B · 区组0 · Y线", "s"),
  fixture_series_15: S("批次B · 区组0 · X线", "s"),
  fixture_series_16: S("批次B · 区组1 · X线", "s"),
  fixture_series_17: S("批次B · 区组1 · Y线", "s"),
  fixture_series_18: S("批次B · 区组1 · 主线", "s"),
  fixture_series_19: S("批次B · 区组2 · 主线（中断）", "s"),
};

describe("线名归一化", () => {
  it("去掉区组记号与它两边的分隔", () => {
    assert.equal(normaliseLineName("批次A · 区组0 · L1 线A"), "批次A · L1 线A");
    assert.equal(normaliseLineName("区组1 · L4 线D"), "L4 线D");
    assert.equal(normaliseLineName("L2 线B · 区组2"), "L2 线B");
  });

  it("末尾中英文括号注释一起去掉", () => {
    assert.equal(normaliseLineName("批次B · 区组2 · 主线（中断）"), "批次B · 主线");
    assert.equal(normaliseLineName("line A · block 2 (retry)"), "line A");
  });

  it("组 / block / blk 的写法都认，大小写不敏感", () => {
    assert.equal(normaliseLineName("组 4 · 线"), "线");
    assert.equal(normaliseLineName("BLK3 · a"), "a");
    assert.equal(normaliseLineName("Block 12 · run7"), "run7");
  });

  it("没有记号的名字原样；只剩记号时分组键退回原名", () => {
    assert.equal(normaliseLineName("2001-01-01 0001–0003 · 3 帧"), "2001-01-01 0001–0003 · 3 帧");
    assert.equal(normaliseLineName("区组0"), "");
    assert.equal(lineKey("区组0"), "区组0");
  });

  it("区组号与区组记号", () => {
    assert.equal(blockNumber("批次A · 区组2 · L1 线A"), 2);
    assert.equal(blockNumber("no token"), Number.POSITIVE_INFINITY);
    assert.equal(blockLabel("批次B · 区组1 · 主线"), "区组1");
    assert.equal(blockLabel("主线"), "主线");
  });
});

describe("同一条线的建议分组（合成系列名）", () => {
  const groups = suggestLineGroups(SYNTHETIC_SERIES);

  it("两批合成谱线按区组归并；合成帧系列不参与", () => {
    assert.deepEqual(
      groups.map((g) => [g.lineName, g.sids]),
      [
        ["批次A · L1 线A", ["fixture_series_01", "fixture_series_08", "fixture_series_09"]],
        ["批次A · L2 线B", ["fixture_series_03", "fixture_series_06", "fixture_series_11"]],
        ["批次A · L3 线C", ["fixture_series_02", "fixture_series_07", "fixture_series_10"]],
        ["批次A · L4 线D", ["fixture_series_04", "fixture_series_05", "fixture_series_12"]],
        ["批次B · X线", ["fixture_series_15", "fixture_series_16"]],
        ["批次B · Y线", ["fixture_series_14", "fixture_series_17"]],
        ["批次B · 主线", ["fixture_series_13", "fixture_series_18", "fixture_series_19"]],
      ],
    );
    assert.ok(!groups.some((g) => g.sids.includes("fixture_series_00")));
  });

  it("「主线（中断）」和另外两个主线区组在一组", () => {
    assert.deepEqual(groupFor(groups, "fixture_series_19").sids, ["fixture_series_13", "fixture_series_18", "fixture_series_19"]);
  });

  it("找不到的系列自己成一组", () => {
    assert.deepEqual(groupFor(groups, "fixture_series_00"), { lineName: "", sids: ["fixture_series_00"] });
  });

  it("参与与否可以由调用方判定（例如按索引里的成员种类）", () => {
    const only = suggestLineGroups(SYNTHETIC_SERIES, (sid) => sid === "fixture_series_13" || sid === "fixture_series_18");
    assert.deepEqual(only.map((g) => g.sids), [["fixture_series_13", "fixture_series_18"]]);
  });

  it("默认线名：同一组取归一化名；不同线取公共前缀", () => {
    assert.equal(commonLineName(SYNTHETIC_SERIES, ["fixture_series_19", "fixture_series_13"]), "批次B · 主线");
    assert.equal(commonLineName(SYNTHETIC_SERIES, ["fixture_series_01", "fixture_series_03"]), "批次A · L");
    assert.equal(commonLineName(SYNTHETIC_SERIES, []), "");
  });
});

describe("拉线谱表单 ↔ options", () => {
  it("站位标注往返；站位号规整、非法行丢掉、按站位排", () => {
    const rows = stationMarksToRows({ "12": "Q₋", "5": "V" });
    assert.deepEqual(rows, [
      { station: "5", label: "V" },
      { station: "12", label: "Q₋" },
    ]);
    assert.deepEqual(rowsToStationMarks(rows), { "5": "V", "12": "Q₋" });
    assert.deepEqual(
      rowsToStationMarks([
        { station: "07", label: " S " },
        { station: "-1", label: "x" },
        { station: "3.5", label: "y" },
        { station: "4", label: "  " },
      ]),
      { "7": "S" },
    );
    assert.deepEqual(stationMarksToRows(null), []);
    assert.deepEqual(stationMarksToRows([1, 2]), []);
  });

  it("bad_from 只收勾选了的系列与非负整数", () => {
    assert.deepEqual(formToBadFrom({ A: "20", B: "", C: "x", D: "3" }, ["A", "B", "C"]), { A: 20 });
    assert.deepEqual(badFromToForm({ A: 20, B: -1, C: "7", D: 1.5 }), { A: "20", C: "7" });
  });

  it("表单 → options → 表单", () => {
    const form = { lineName: "  批次B · 主线 ", marks: [{ station: "6", label: "D" }], badFrom: { W: "0" }, excludeRejected: false };
    const opts = stsLinesOptions(form, ["W"]);
    assert.deepEqual(opts, { station_marks: { "6": "D" }, bad_from: { W: 0 }, exclude_rejected: false, line_name: "批次B · 主线" });
    const back = formFromOptions(opts, "默认");
    assert.equal(back.lineName, "批次B · 主线");
    assert.deepEqual(back.marks, [{ station: "6", label: "D" }]);
    assert.deepEqual(back.badFrom, { W: "0" });
    assert.equal(back.excludeRejected, false);
  });

  it("没写的选项取默认：线名用默认值、剔除排除评级默认开", () => {
    const f = formFromOptions({}, "批次A · L2 线B");
    assert.equal(f.lineName, "批次A · L2 线B");
    assert.equal(f.excludeRejected, true);
    assert.deepEqual(f.marks, []);
    assert.equal("line_name" in stsLinesOptions({ ...f, lineName: " " }, []), false);
  });
});

describe("预填", () => {
  const entry = (kind: string, series: string[], created: string, options: Any = {}): Any => ({
    key: `sts_lines/${created}`, category: "sts_lines", kind, title: "", base: created, created, series, options,
  });
  const entries = [
    entry("sts_lines", ["A", "B", "C"], "2001-09-14 10:00", { line_name: "旧" }),
    entry("sts_lines", ["C", "B", "A"], "2001-09-14 11:00", { line_name: "新" }),
    entry("sts_stitch", ["A", "B", "C"], "2001-09-14 12:00"),
    entry("sts_lines", ["A", "B"], "2001-09-14 13:00"),
  ];

  it("集合相等、与顺序无关", () => {
    assert.equal(sameSet(["A", "B"], ["B", "A"]), true);
    assert.equal(sameSet(["A", "B"], ["A", "B", "C"]), false);
    assert.equal(sameSet([], []), true);
  });

  it("取同一组系列最近的一次拉线谱，别的种类与别的集合不算", () => {
    assert.equal(findPrefill(entries, ["B", "C", "A"])?.options?.line_name, "新");
    assert.equal(findPrefill(entries, ["A", "B"])?.base, "2001-09-14 13:00");
    assert.equal(findPrefill(entries, ["A"]), null);
    assert.equal(findPrefill(entries, []), null);
  });

  it("跨类别拍平", () => {
    assert.equal(allEntries([{ key: "sts_lines", title: "", figures: entries }, { key: "frames", title: "" }] as Any).length, 4);
    assert.deepEqual(allEntries(undefined), []);
  });
});

describe("类别", () => {
  it("补齐五个类别并按设计 D16 排序；认不出的排最后", () => {
    const cats = withAllCategories([
      { key: "series", title: "x", figures: [] },
      { key: "weird" as Any, title: "怪", figures: [] },
      { key: "frames", title: "", figures: [] },
    ]);
    assert.deepEqual(cats.map((c) => c.key), [...CATEGORY_ORDER, "weird"]);
    assert.deepEqual(cats.map((c) => c.title), ["标记帧", "网格谱", "拉线谱", "单根谱拼接", "旋转系列", "怪"]);
  });
});

describe("产物文字", () => {
  it("数字位数", () => {
    assert.equal(fmtNumber(3), "3");
    assert.equal(fmtNumber(27.4321), "27.4");
    assert.equal(fmtNumber(0.98765), "0.988");
    assert.equal(fmtNumber(-1.2345), "-1.23");
    assert.equal(fmtNumber(1234.56), "1235");
    assert.equal(fmtNumber(0.000123456), "0.000123");
  });

  it("summary 小标签：数字、布尔、短数组；嵌套对象与空值跳过；有上限", () => {
    assert.equal(fmtSummaryValue(true), "是");
    assert.equal(fmtSummaryValue([1.04, 0.9876]), "1.04 / 0.988");
    assert.equal(fmtSummaryValue([1, 2, 3, 4, 5]), "5 项");
    assert.equal(fmtSummaryValue({ a: 1 }), null);
    assert.equal(fmtSummaryValue("x".repeat(50))?.length, 40);
    assert.deepEqual(
      summaryChips({ span_pm: 27.43, r: 0.99, nested: { a: 1 }, none: null, sv: [1.0393, 1.0096] }),
      [
        { key: "span_pm", text: "span_pm 27.4" },
        { key: "r", text: "r 0.99" },
        { key: "sv", text: "sv 1.04 / 1.01" },
      ],
    );
    assert.equal(summaryChips({ a: 1, b: 2, c: 3 }, 2).length, 2);
  });

  it("options 摘要", () => {
    assert.equal(
      optionsText({ anchor: "darkest", fov_nm: 7.6, station_marks: { "5": "V" }, bad_from: {}, empty: "" }),
      'anchor=darkest · fov_nm=7.6 · station_marks={"5":"V"}',
    );
    assert.equal(optionsText(undefined), "");
  });

  it("下载链接 png → jpg → csv → npy → json；主图优先 <base>.png", () => {
    const files: Any[] = [
      { name: "旋转系列_叠加_配准表.csv", ext: "csv", url: "/c", size: 23825, mtime: 0 },
      { name: "旋转系列_叠加_主图_2x.png", ext: "png", url: "/p2", size: 378046, mtime: 0, preview_url: "/pv2" },
      { name: "旋转系列_叠加.png", ext: "png", url: "/p1", size: 397568, mtime: 0, preview_url: "/pv1" },
      { name: "旋转系列_叠加_刚性平均.npy", ext: "npy", url: "/n", size: 1210696, mtime: 0 },
    ];
    assert.deepEqual(fileLinks(files).map((f) => f.label), ["PNG 388 KB", "PNG 369 KB", "CSV 23 KB", "NPY 1.2 MB"]);
    const entry = { base: "旋转系列_叠加", files };
    assert.equal(fullImageUrl(entry), "/p1");
    assert.equal(previewUrl(entry), "/pv1");
    assert.equal(fullImageUrl({ base: "x", files: [{ name: "x.jpg", ext: "jpg", url: "/j", size: 1, mtime: 0 }] }), "/j");
    assert.equal(fullImageUrl({ base: "x", files: [] }), null);
    assert.equal(fmtBytes(900), "900 B");
  });
});

describe("不属于任何系列的单根谱", () => {
  const it0 = (id: string, d: string, k: string, t: number): Any => ({ id, d, k, fn: id.split("/").pop(), t, p: "", pf: "", ad: "", th: "" });
  const items = [
    it0("R/0907/rep18.dat", "R/20010907", "s", 30),
    it0("R/0907/rep06.dat", "R/20010907", "s", 10),
    it0("R/0911/rep01.dat", "R/20010911", "s", 5),
    it0("R/0910/rep02.dat", "R/20010910", "s", 1),
    it0("R/0907/f.sxm", "R/20010907", "f", 2),
  ];
  const byId = new Map(items.map((x) => [x.id, x]));
  const marks: Any = {
    "R/0907/rep18.dat": { r: 2 },
    "R/0907/rep06.dat": { r: 0 },
    "R/0911/rep01.dat": { r: 1 },
    "R/0910/rep02.dat": { r: 1 },
    "R/0907/f.sxm": { r: 2 },
    "R/gone/rep99.dat": { r: 1 },
  };
  const series: Any = { S1: S("批次A · 区组0 · L1", "s", ["R/0910/rep02.dat"]) };

  it("按目录分组、组内按时间；系列成员、帧、索引里没有的都不算", () => {
    assert.deepEqual(singleSpectraByDir(marks, series, byId), [
      { d: "R/20010907", ids: ["R/0907/rep06.dat", "R/0907/rep18.dat"] },
      { d: "R/20010911", ids: ["R/0911/rep01.dat"] },
    ]);
  });
});

describe("发起出图的文案", () => {
  const st = (o: Any): Any => ({ running: false, phase: "idle", done: 0, total: 0, message: "", degraded: false, ...o });

  it("后端不可用、已有任务在跑、别的种类在跑、出错、正常", () => {
    assert.deepEqual(describeStart(undefined, st({ degraded: true, detail: "no module" }), "frame_sheet"), {
      ok: false, text: "出图没有开始：no module",
    });
    assert.equal(describeStart(st({ running: true, kind: "sts_lines" }), st({ running: true, kind: "sts_lines" }), "frame_sheet").ok, false);
    assert.match(describeStart(st({}), st({ running: true, kind: "series_stack" }), "frame_sheet").text, /旋转叠加/);
    assert.equal(describeStart(st({}), st({ phase: "error", message: "boom" }), "grid_sheets").text, "出图出错：boom");
    assert.deepEqual(describeStart(st({}), st({ running: true, kind: "grid_sheets" }), "grid_sheets"), { ok: true, text: "已开始：网格逐层图" });
    assert.equal(describeStart(st({}), undefined, "grid_sheets").ok, false);
  });

  it("κ 与正整数输入", () => {
    assert.equal(parseKappa("5.0"), 5.0);
    assert.equal(parseKappa(" .5 "), 0.5);
    assert.equal(parseKappa("0"), null);
    assert.equal(parseKappa("-3"), null);
    assert.equal(parseKappa("1e3"), null);
    assert.equal(parsePositiveInt("100"), 100);
    assert.equal(parsePositiveInt("0"), null);
    assert.equal(parsePositiveInt("2.5"), null);
  });
});
