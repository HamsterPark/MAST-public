// ════════════════════════════════════════════════════════════════════════════
// 数据图库 —— 选中（旧版兼容格式 series.js 的 pickClick / pickKey / selRange）。
//
// 卡片勾选框：单击选/取消，按住 Shift 点另一张 = 从上次点的那张连选到这一张。
// 大图键盘：S 选/取消当前，[ 定起点（并选中），] 从起点（没有起点就从上次点的那张）
// 选到当前。连选只**加**不减——原版如此，免得一次手滑把选好的一段清掉。
// ════════════════════════════════════════════════════════════════════════════

export interface HasId {
  id: string;
}

/** `fromId` 所在位置到 `toIndex` 之间（含两端）的全部 id；`fromId` 不在列表里时只含 toIndex。 */
export function rangeIds(list: readonly HasId[], fromId: string | null, toIndex: number): string[] {
  const a0 = fromId != null ? list.findIndex((x) => x.id === fromId) : -1;
  const a = a0 < 0 ? toIndex : a0;
  const out: string[] = [];
  for (let k = Math.min(a, toIndex); k <= Math.max(a, toIndex); k++) {
    const it = list[k];
    if (it) out.push(it.id);
  }
  return out;
}

export interface SelectionState {
  ids: ReadonlySet<string>;
  lastId: string | null;
  startId: string | null;
}

/** 卡片勾选框被点。 */
export function pickClick(
  st: SelectionState,
  list: readonly HasId[],
  j: number,
  shift: boolean,
  checked: boolean,
): SelectionState {
  const it = list[j];
  if (!it) return st;
  const ids = new Set(st.ids);
  if (shift && st.lastId != null) for (const id of rangeIds(list, st.lastId, j)) ids.add(id);
  else if (checked) ids.add(it.id);
  else ids.delete(it.id);
  return { ...st, ids, lastId: it.id };
}

/** 大图里按 S / [ / ]。 */
export function pickKey(
  st: SelectionState,
  list: readonly HasId[],
  j: number,
  code: "KeyS" | "BracketLeft" | "BracketRight",
): SelectionState {
  const it = list[j];
  if (!it) return st;
  const ids = new Set(st.ids);
  let startId = st.startId;
  if (code === "KeyS") {
    if (ids.has(it.id)) ids.delete(it.id);
    else ids.add(it.id);
  } else if (code === "BracketLeft") {
    startId = it.id;
    ids.add(it.id);
  } else {
    for (const id of rangeIds(list, startId ?? st.lastId, j)) ids.add(id);
  }
  return { ids, lastId: it.id, startId };
}
