// ════════════════════════════════════════════════════════════════════════════
// envPanel — 右栏 ENVIRONMENT 要画哪些行的**纯**判断。
//
// 与 `components/shell/RightPanel.tsx` 的关系同 `lib/transcriptRefresh.ts` 与
// 它的 hook：这里没有 React import，所以每一条判断都能在 node --test 里直接跑。
//
// 抠出来的理由：这些判断全是「什么时候**不**画」。漏一条的症状是面板上多一行
// 长得很正常的读数 —— 不会崩、typecheck 管不着、截图看着也对。
// ════════════════════════════════════════════════════════════════════════════

/** `/api/environment/readings` 的 `sensors[]` 里我们要看的字段。 */
export interface EnvSensor {
  name: string;
  type?: string | null;
  kind?: string | null;
  placeholder?: boolean | null;
  value?: number | null;
  unit?: string | null;
  status?: string | null;
  connected?: boolean | null;
}

/**
 * 镜像 `InstrumentState` 的那一类传感器（`tunnel_current` 是其中一个）。
 *
 * 它整个类的存在理由就是「零 TCP 地把仪器状态当成传感器再报一遍」，所以按构造
 * 它报的每一个量，上面那一栏 INSTRUMENT 已经画过了 —— 同一个数在同一条右栏里
 * 出现两次。ENVIRONMENT 里的 tunnel_current 可以不要。
 *
 * 判据取**驱动类**而不是名字 `tunnel_current`：这个类可以带别的 attr 起别的名字
 * （`name=` / `attr=` 都是构造参数），按名字过滤只挡得住今天这一个。
 *
 * 只改展示。传感器本身照常采集、照常进环境历史、照常参与判据。
 */
export const INSTRUMENT_MIRROR_TYPE = "InstrumentStateSensor";

export function mirrorsInstrumentPanel(s: EnvSensor): boolean {
  return (s.type ?? "") === INSTRUMENT_MIRROR_TYPE;
}

/**
 * 占位实现 —— 不是「读不到」，是「MAST 里根本没有这个量的驱动」。
 *
 * 两者今天在 `status` 上长得一模一样（都是 `unavailable`），而它们是不同的事实：
 * 一台 COM 口拔掉的真空计插回去就好，`NoiseSensor` 插什么都没用 —— 它从来没有
 * 返回过一个数字，也没有任何发现流程能把它填上。
 *
 * 「不知道」和「没有」必须是两句话（Noise Level 一直是
 * unavailable，看起来像坏了）。所以占位不进读数行，只在末尾报一次「未接入」。
 *
 * 判据来自后端 `SensorEntry.placeholder`（那边是 isinstance，不是类名字符串）。
 */
export function isStub(s: EnvSensor): boolean {
  return s.placeholder === true;
}

/** 面板真正要画成一行读数的传感器。 */
export function readingRows(sensors: readonly EnvSensor[]): EnvSensor[] {
  return sensors.filter((s) => !mirrorsInstrumentPanel(s) && !isStub(s));
}

/**
 * 某个 headline 位（vacuum / temperature / …）后面**活着的**传感器。
 *
 * 空 = 这一位没有实时读数。此时还要问一句「是没接，还是接了读不到」——
 * 见 {@link stubKinds}。
 */
export function headlineSensors(sensors: readonly EnvSensor[], kind: string): EnvSensor[] {
  return readingRows(sensors).filter((s) => (s.kind ?? "") === kind && s.connected === true);
}

/**
 * 只有占位撑着的那些 headline 位 —— 面板末尾那行「未接入」列的就是它们。
 *
 * 有真传感器在（哪怕此刻读不到）的位不算：那一位该继续显示 N/A 并等着它回来，
 * 把它说成「未接入」是另一个方向的假话。
 */
export function stubKinds(sensors: readonly EnvSensor[], kinds: readonly string[]): string[] {
  const live = new Set(readingRows(sensors).map((s) => s.kind ?? ""));
  const stubbed = new Set(sensors.filter(isStub).map((s) => s.kind ?? ""));
  return kinds.filter((k) => stubbed.has(k) && !live.has(k));
}
