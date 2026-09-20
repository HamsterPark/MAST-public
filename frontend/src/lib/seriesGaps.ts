// ════════════════════════════════════════════════════════════════════════════
// seriesGaps.ts — 一条时间序列在哪里**不再连续**。
//
// 已知问题：辅助通道在数据中断的时候不会自动正确显示中断，而是强行连线。
// 一条直线是「测量过」的形状。守护停了二十分钟，两端连起来之后，那二十分钟看上去
// 和一段平稳的信号一模一样 —— 而这两件事要采取的行动完全相反。
//
// 规则只有一条，而且只写一处。环境历史页早就在做这件事（`envHistory.toAlignedData`
// 内联了一份），监控页两张图一份都没有。两处各写一份容易出现不一致，
// 所以这里把它抽出来，环境历史改成调用它。
//
// ── 实现依据（uPlot 1.6.32，读过源码，不是猜的） ──────────────────────────
// · 序列值为 `null` 处会断开 —— `spanGaps` 默认 false，`findGaps` 把 null 段收集成
//   `_paths.gaps`，再由 `clipGaps` 变成一条把空洞挖掉的 clip path。
// · **band 填充也断得掉**：`fillStroke`/`strokeFill` 在画 band 时同时套用上边缘的
//   `gapsClip` 与下边缘的 `gapsClip2`。所以两条边缘都填 null，中间那片阴影一起断。
//   （这一点必须核实过才敢用：只断线不断填充的话，那片色块照样在说「这里有数据」。）
// ════════════════════════════════════════════════════════════════════════════

/**
 * 相邻时间戳间隔的**中位数** —— 「这条曲线本来多密」的自标定答案。
 *
 * 用中位数不用均值：均值会被它要找的那个空档本身拉大，于是空档越大越不像空档。
 *
 * 有服务端申报步长时使用申报值。辅助通道采用机会式采样，设置只是节流上限；
 * 服务端还可能按 max_points 抽稀，因此应从返回时间戳估计显示点距。
 *
 * 取不到（点数不足 / 时间戳非递增）返回 0，调用方按「不划断点」处理。
 */
export function medianStep(t: readonly number[]): number {
  const d: number[] = [];
  for (let i = 1; i < t.length; i += 1) {
    const a = t[i - 1];
    const b = t[i];
    if (typeof a === "number" && typeof b === "number" && b > a) d.push(b - a);
  }
  if (!d.length) return 0;
  d.sort((x, y) => x - y);
  const m = d.length >> 1;
  return d.length % 2 ? d[m]! : ((d[m - 1]! + d[m]!) / 2);
}

/**
 * 间隔超过步长的这个倍数就算**中断**。
 *
 * 3 而不是 1.x：采样是机会式的（角色锁被扫描占着就跳过这一次），偶尔漏一两拍是
 * 正常运行，不是中断 —— 面板本来就在另一处如实报「因通道被占跳过 N 次」。
 * 1.33 s 的节奏下这条线落在 4 s：漏一拍(2.7 s)、漏两拍(4.0 s)都不断，漏三拍才断。
 *
 * 另一侧的代价是短于 4 s 的真中断画不出来。这是**刻意选的方向**：
 * 划错一道断点会让人去查一个没坏的东西，而它换来的信息只有「这 4 秒没测」。
 */
export const GAP_FACTOR = 3;

function num(v: unknown): number | null {
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

/**
 * 把 `null` 行插进每一处采集中断，交给 uPlot 画成空洞。
 *
 * `step` **由调用方给，不在这里兜底猜**。给 0 或负数 = 「我不知道这条曲线该多密」
 * ⇒ 一个断点都不划。这一条是被测试钉住的（环境历史在桶宽为 0 时不能凭空造断点），
 * 而且方向是对的：不知道正常间隔是多少的时候，任何断点判定都是在编。
 * 想要自标定的调用方自己传 {@link medianStep}(t) —— 让它是一个**看得见的选择**。
 *
 * 断点的时间戳取 `上一点 + step`，紧贴在最后一个真实样本之后，而不是空档正中：
 * 空洞的左边缘就该是数据停下来的那一刻。
 */
export function spliceGaps(
  t: readonly number[],
  cols: ReadonlyArray<ReadonlyArray<number | null | undefined>>,
  opts: { step: number; factor?: number },
): { t: number[]; cols: (number | null)[][]; gaps: number } {
  const factor = Number.isFinite(opts.factor) ? (opts.factor as number) : GAP_FACTOR;
  const step = Number.isFinite(opts.step) && opts.step > 0 ? opts.step : 0;
  const outT: number[] = [];
  const outCols: (number | null)[][] = cols.map(() => []);
  let gaps = 0;

  for (let i = 0; i < t.length; i += 1) {
    const ts = num(t[i]);
    if (ts == null) continue;
    const prev = outT.length ? outT[outT.length - 1]! : null;
    if (prev != null && step > 0 && ts - prev > step * factor) {
      // 断点必须严格落在两个真实样本之间，否则 x 轴不再单调递增，uPlot 会画出
      // 一条往回走的线。factor ≥ 1 时 prev+step 天然在中间，但 factor 是参数，
      // 所以这里不靠调用方守规矩。
      outT.push(Math.min(prev + step, (prev + ts) / 2));
      for (const c of outCols) c.push(null);
      gaps += 1;
    }
    outT.push(ts);
    for (let k = 0; k < cols.length; k += 1) outCols[k]!.push(num(cols[k]![i]));
  }

  return { t: outT, cols: outCols, gaps };
}
