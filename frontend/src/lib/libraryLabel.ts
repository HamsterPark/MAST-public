/** Resolve a readable library label without creating a library on a read path.
 * An experiment library may have a derived id before a registry row exists;
 * in that case use the current experiment name as the label.
 */

export interface LibraryRowLike {
  library_id: string;
  name?: string | null;
  experiment_name?: string | null;
}

export interface LibraryLabel {
  /** 给人看的名字。永远非空 —— 最差也是那个裸 id。 */
  text: string;
  /**
   * 这个库还没被建出来（懒创建，首次收录时才落盘）。为真时界面应当说清楚，
   * 不要假装它已经在那儿了 —— 名字是推导出来的，库不是。
   */
  pending: boolean;
}

/**
 * @param libraryId       有效库 id（或面板显式选中的库 id）
 * @param source          "experiment" | "manual" | "fallback" | "picked"
 * @param libraries       `/api/literature/libraries` 的 libraries 数组
 * @param experimentName  当前实验名（`/api/experiments/current`），可空
 */
export function libraryLabel(
  libraryId: string,
  source: string,
  libraries: readonly LibraryRowLike[],
  experimentName?: string | null,
): LibraryLabel {
  const id = (libraryId || "").trim();
  if (!id) return { text: "", pending: false };

  const row = libraries.find((l) => l.library_id === id) ?? null;
  if (row) {
    // 实验库优先用实验名 —— 一个裸 exp_1a2b3c4d 说明不了任何事。
    const exp = (row.experiment_name || "").trim();
    if (exp) return { text: `《${exp}》的文献库`, pending: false };
    const name = (row.name || "").trim();
    if (name) return { text: name, pending: false };
    return { text: id, pending: false };
  }

  // 库不在列表里。只有一种情况下这不是错误：实验库还没被懒创建出来。此时名字
  // 可以从当前实验推出来，但必须说明它还没落盘。
  const exp = (experimentName || "").trim();
  if (source === "experiment" && exp) {
    return { text: `《${exp}》的文献库`, pending: true };
  }
  return { text: id, pending: source === "experiment" };
}
