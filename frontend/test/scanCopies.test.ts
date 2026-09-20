// ════════════════════════════════════════════════════════════════════════════
// 自动拷贝副本的显示 — src/lib/scanCopies.ts
//
// MAST copies every scan into the experiment folder and leaves the original
// where Nanonis wrote it. The 数据 tab therefore showed one measurement as
// several cards with an identical name, size and timestamp.
//
// The server folds them. These tests pin the DISCLOSURE half: a fold the
// operator cannot see is indistinguishable from files going missing, which is
// the failure this repo already recorded twice from the other direction
// (#53/#60 — a listing quietly showing less than the disk holds).
// ════════════════════════════════════════════════════════════════════════════

import assert from "node:assert/strict";
import { describe, it } from "node:test";

import {
  copyBadge,
  copyLocations,
  isUnattributed,
  locationKindLabel,
  parentFolder,
} from "../src/lib/scanCopies.ts";

describe("parentFolder", () => {
  // The OTHER repetition: not copies of one measurement, but different
  // measurements sharing a Nanonis auto-name. Measured on this machine: 36
  // basenames occur in more than one session folder, every time with a different
  // size and timestamp. Those must NOT be folded — so the cards have to be
  // tellable apart instead.
  it("给出会话目录，让同名的两张卡片分得开", () => {
    assert.equal(
      parentFolder("D:\\MAST\\working-sessions\\20260327\\unnamed0033.sxm"),
      "20260327",
    );
    assert.equal(
      parentFolder("D:\\MAST\\working-sessions\\20260319\\unnamed0033.sxm"),
      "20260319",
    );
  });

  it("POSIX 分隔符一样处理", () => {
    assert.equal(parentFolder("/data/20260319/a.sxm"), "20260319");
  });

  it("没有父目录时给空串，而不是编一个", () => {
    assert.equal(parentFolder("a.sxm"), "");
    assert.equal(parentFolder(""), "");
  });
});

describe("copyBadge", () => {
  it("只有真的有多份时才出现", () => {
    assert.equal(copyBadge({ path: "a", copies: 1 }), null);
    assert.equal(copyBadge({ path: "a" }), null);          // 字段缺失 = 一份
    assert.equal(copyBadge({ path: "a", copies: 3 }), "×3");
  });
});

describe("copyLocations", () => {
  it("多份时按 原件 → 实验文件夹 → 隔离区 排", () => {
    const locs = copyLocations({
      path: "x",
      copies: 3,
      locations: [
        { path: "q", kind: "quarantine" },
        { path: "e", kind: "experiment" },
        { path: "o", kind: "origin" },
      ],
    });
    assert.deepEqual(locs.map((l) => l.path), ["o", "e", "q"]);
  });

  it("单份也要给得出位置", () => {
    // The server omits `locations` for single-copy entries to keep the payload
    // small. A caller asking "where is this file" must not get an empty answer
    // for a file that plainly exists.
    const locs = copyLocations({ path: "D:/s/a.sxm", kind: "origin" });
    assert.deepEqual(locs, [{ path: "D:/s/a.sxm", kind: "origin" }]);
  });

  it("列出的份数与徽章一致", () => {
    const entry = {
      path: "o",
      copies: 2,
      locations: [
        { path: "o", kind: "origin" },
        { path: "e", kind: "experiment" },
      ],
    };
    assert.equal(copyLocations(entry).length, Number(copyBadge(entry)!.slice(1)));
  });
});

describe("locationKindLabel", () => {
  it("认得服务端的三种 root_kind", () => {
    assert.equal(locationKindLabel("origin"), "原始位置");
    assert.equal(locationKindLabel("experiment"), "实验文件夹");
    assert.equal(locationKindLabel("experiment_folder"), "实验文件夹");
    assert.equal(locationKindLabel("quarantine"), "隔离区");
  });

  it("没见过的类型原样显示，不说成「未知」", () => {
    // A kind we have not been taught is a NEW kind, not a lost file — calling it
    // 未知位置 would read as "we cannot find it".
    assert.equal(locationKindLabel("archive"), "archive");
    assert.equal(locationKindLabel(undefined), "位置");
  });
});

describe("isUnattributed", () => {
  it("有实验文件夹副本的就是已归属", () => {
    assert.equal(
      isUnattributed({
        path: "o",
        copies: 2,
        locations: [
          { path: "o", kind: "origin" },
          { path: "e", kind: "experiment" },
        ],
      }),
      false,
    );
  });

  it("只有原件的算未归属", () => {
    assert.equal(isUnattributed({ path: "o", kind: "origin", copies: 1 }), true);
  });

  it("单份但本身就在实验文件夹里的不算未归属", () => {
    // copies===1 with no location list still has to be judged — this is the case
    // where the ingest moved rather than copied, or the original is gone.
    assert.equal(isUnattributed({ path: "e", kind: "experiment", copies: 1 }), false);
  });

  it("隔离区不算「已归属某个实验」", () => {
    // Quarantine is where the ingest puts files it could NOT place. Counting it
    // as attributed would hide exactly the files that need attention.
    assert.equal(isUnattributed({ path: "q", kind: "quarantine", copies: 1 }), true);
  });
});
