// ════════════════════════════════════════════════════════════════════════════
// 按实验 / 样品分组浏览的数据整形 — src/lib/scanAttribution.ts
//
// The flat listing walks the disk, and a path alone cannot say which experiment
// claimed a file. `/api/scans/attribution` reads the v2 `file_locations` table,
// which is that record; this shapes its rows into the experiment→sample→files
// tree the browser renders.
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  displayPath,
  fileName,
  groupBySample,
  groupSummary,
  pathKeys,
} from "../src/lib/scanAttribution.ts";

const row = (over: Record<string, unknown> = {}) => ({
  sha256: "a".repeat(64),
  rel_path: "samples/S01__x__aaa/raw/sxm/scan.sxm",
  abs_path: "D:/MAST-Data/experiments/E1/samples/S01__x__aaa/raw/sxm/scan.sxm",
  origin_path: "D:/sessions/20260318/scan.sxm",
  root_kind: "experiment_folder",
  sample_id: "sam-1",
  source: "skill",
  size_bytes: 1024,
  status: "ok",
  ingested_at: "2026-08-01T00:00:00Z",
  ...over,
});

describe("displayPath", () => {
  it("优先用重建出来的实验文件夹路径", () => {
    assert.equal(displayPath(row()), row().abs_path);
  });

  it("重建不出来时退回 origin_path", () => {
    // abs_path is null when the experiment folder cannot be located (root moved
    // or renamed). The origin the ingest recorded is then the only path we have.
    assert.equal(displayPath(row({ abs_path: null })), row().origin_path);
  });

  it("两个都没有就承认没有路径", () => {
    assert.equal(displayPath(row({ abs_path: null, origin_path: null })), null);
  });
});

describe("fileName", () => {
  it("认得 Windows 与 POSIX 两种分隔符", () => {
    assert.equal(fileName(row({ abs_path: "D:\\a\\b\\scan_01.sxm" })), "scan_01.sxm");
    assert.equal(fileName(row({ abs_path: "/data/b/scan_02.sxm" })), "scan_02.sxm");
  });

  it("没有路径时退回 rel_path 的末段", () => {
    assert.equal(fileName(row({ abs_path: null, origin_path: null })), "scan.sxm");
  });
});

describe("groupBySample", () => {
  it("同一样品的文件归到一组", () => {
    const groups = groupBySample([
      row({ sample_id: "s1", rel_path: "a" }),
      row({ sample_id: "s2", rel_path: "b" }),
      row({ sample_id: "s1", rel_path: "c" }),
    ]);
    assert.equal(groups.length, 2);
    assert.deepEqual(groups[0]!.files.map((f) => f.rel_path), ["a", "c"]);
  });

  it("没有样品的文件不丢，只是排到最后", () => {
    // Files ingested before the sample layer existed, or with no active sample,
    // are real files. Dropping them would make this view show less than the
    // experiment holds.
    const groups = groupBySample([
      row({ sample_id: null, rel_path: "orphan" }),
      row({ sample_id: "s1", rel_path: "a" }),
    ]);
    assert.equal(groups.length, 2);
    assert.equal(groups[groups.length - 1]!.sampleId, null);
    assert.equal(groups[groups.length - 1]!.files[0]!.rel_path, "orphan");
  });

  it("一个文件都不会消失", () => {
    const rows = [
      row({ sample_id: "s1", rel_path: "a" }),
      row({ sample_id: null, rel_path: "b" }),
      row({ sample_id: "s2", rel_path: "c" }),
      row({ sample_id: "s1", rel_path: "d" }),
    ];
    const seen = groupBySample(rows).flatMap((g) => g.files);
    assert.equal(seen.length, rows.length);
  });

  it("累计字节数", () => {
    const groups = groupBySample([
      row({ sample_id: "s1", size_bytes: 100 }),
      row({ sample_id: "s1", size_bytes: 200 }),
    ]);
    assert.equal(groups[0]!.bytes, 300);
  });

  it("空输入是空分组", () => {
    assert.deepEqual(groupBySample([]), []);
  });
});

describe("pathKeys", () => {
  it("大小写与分隔符都归一 —— 同一个文件在两侧拼写不同", () => {
    // One spelling comes from config, the other from the Nanonis session path;
    // matching a disk listing against these rows has to compare normalised.
    const keys = pathKeys(row({ abs_path: "D:\\MAST-Data\\A.SXM", origin_path: "D:/s/A.sxm" }));
    assert.deepEqual(keys, ["d:/mast-data/a.sxm", "d:/s/a.sxm"]);
  });

  it("缺路径就少一个键，而不是造一个空键", () => {
    assert.deepEqual(pathKeys(row({ abs_path: null, origin_path: null })), []);
  });
});

describe("groupSummary", () => {
  it("MB / KB 按大小切换", () => {
    assert.ok(groupSummary({ files: [1, 2], bytes: 5 * 1024 * 1024 }).includes("MB"));
    assert.ok(groupSummary({ files: [1], bytes: 4096 }).includes("KB"));
  });

  it("文件数是真实条数", () => {
    assert.ok(groupSummary({ files: [1, 2, 3], bytes: 0 }).startsWith("3 个文件"));
  });
});
