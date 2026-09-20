// 数据根与构建（设计 D2 / D8 / D15；旧版兼容格式没有这一页——它的根写死在脚本里，更新靠双击 bat）。
//
// 数据根**显式配置**：打开图库不会遍历任何目录。「建议」只是建议，点一下才加进去。
// 构建在服务端后台线程里跑，这里只发起、取消、看进度；完成的那一刻 GalleryApp 让索引
// 与标记重新拉取。

import { useEffect, useState } from "react";
import { useQueryClient } from "@tanstack/react-query";
import clsx from "clsx";
import { api } from "@/api/client";
import type { GalleryBuildStatus } from "@/lib/gallery/types";
import { ErrorNote, Spinner } from "@/components/ui";
import { GALLERY_KEYS, useGalleryConfig, useGalleryStatus } from "./useGalleryData";
import { ImportMarksForm } from "./MarkedView";
import { toast } from "./toast";
import { SMALL_BTN, SMALL_SELECT } from "./bits";
import { H1, Sub } from "./ListHeads";

interface RootRow {
  key: number;
  name: string;
  path: string;
  enabled: boolean;
  exists?: boolean;
}

const PHASE: Record<string, string> = {
  idle: "空闲",
  inventory: "清点文件",
  render: "渲染缩略图与自动判据",
  dups: "查重复保存",
  index: "写索引",
  done: "完成",
  cancelled: "已取消",
  error: "出错",
};

let rowKey = 0;

export function SetupView() {
  const qc = useQueryClient();
  const cfg = useGalleryConfig();
  const status = useGalleryStatus();
  const [rows, setRows] = useState<RootRow[] | null>(null);
  const [workers, setWorkers] = useState("");
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState<{ ok: boolean; text: string } | null>(null);

  useEffect(() => {
    if (!cfg.data || rows) return;
    setRows((cfg.data.roots ?? []).map((r) => ({ key: ++rowKey, ...r })));
    setWorkers(String(cfg.data.workers ?? ""));
  }, [cfg.data, rows]);

  const st: GalleryBuildStatus | undefined = status.data;
  const running = !!st?.running;

  const save = async () => {
    if (!rows) return;
    setSaving(true);
    setSaveMsg(null);
    try {
      const w = Number(workers);
      const { data, error } = await api.POST("/api/gallery/config", {
        body: {
          roots: rows.map((r) => ({ name: r.name.trim() || null, path: r.path.trim(), enabled: r.enabled })),
          workers: Number.isFinite(w) && w >= 1 ? Math.round(w) : null,
        },
      });
      if (error || !data) throw new Error("请求失败");
      if (!data.ok) {
        setSaveMsg({ ok: false, text: data.detail || "没有保存" });
        return;
      }
      qc.setQueryData(GALLERY_KEYS.config, data);
      setRows((data.roots ?? []).map((r) => ({ key: ++rowKey, ...r })));
      setWorkers(String(data.workers ?? ""));
      setSaveMsg({ ok: true, text: "已保存。改了数据根之后点「增量更新」。" });
    } catch (e) {
      setSaveMsg({ ok: false, text: (e as Error).message });
    } finally {
      setSaving(false);
    }
  };

  const build = async (force: boolean) => {
    try {
      const { data, error } = await api.POST("/api/gallery/build", { body: { force } });
      if (error || !data) throw new Error("请求失败");
      qc.setQueryData(GALLERY_KEYS.status, data);
      if (data.degraded) toast(`构建没有开始：${data.detail || "后端不可用"}`, "err");
    } catch (e) {
      toast(`构建没有开始：${(e as Error).message}`, "err");
    }
  };

  const cancel = async () => {
    try {
      const { data } = await api.POST("/api/gallery/build/cancel");
      if (data) qc.setQueryData(GALLERY_KEYS.status, data);
    } catch {
      /* 下一次轮询会看到真实状态 */
    }
  };

  const cell = "border-b border-mast-border px-1.5 py-1";
  const pct = st && st.total ? Math.round((100 * st.done) / st.total) : 0;

  return (
    <div className="max-w-5xl">
      <H1>数据根与构建</H1>
      <Sub>
        图库只处理这里列出的目录（递归找 .sxm / .dat / .3ds）。根名是每个条目 id 的第一段——数据搬了家，改路径不改名，
        标记就跟着走。缓存、缩略图、索引与标记都在状态目录：
        <span className="ml-1 font-mono">{cfg.data?.state_dir || "—"}</span>
      </Sub>
      {cfg.isPending && <Spinner />}
      {cfg.isError && <ErrorNote error={cfg.error} />}
      {cfg.data?.degraded && (
        <div className="mb-2 rounded-mast-ctl border border-mast-warn-border bg-mast-warn-bg px-3 py-2 text-sm text-mast-warn">
          图库后端不可用：{cfg.data.detail || "未知原因"}
        </div>
      )}

      <h2 className="mb-1.5 mt-3 text-[1.05rem] font-semibold">数据根</h2>
      <div className="overflow-x-auto">
        <table className="w-full border-collapse text-[13px]">
          <thead>
            <tr className="text-left">
              {["启用", "根名", "路径", "存在", ""].map((h, i) => (
                <th key={i} className="border-b border-mast-border bg-mast-bg px-1.5 py-1 font-semibold">
                  {h}
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {(rows ?? []).map((r) => (
              <tr key={r.key}>
                <td className={cell}>
                  <input
                    type="checkbox"
                    checked={r.enabled}
                    onChange={(e) => setRows((rs) => rs!.map((x) => (x.key === r.key ? { ...x, enabled: e.target.checked } : x)))}
                  />
                </td>
                <td className={cell}>
                  <input
                    className={`${SMALL_SELECT} w-32 font-mono`}
                    value={r.name}
                    placeholder="留空＝用目录名"
                    onChange={(e) => setRows((rs) => rs!.map((x) => (x.key === r.key ? { ...x, name: e.target.value } : x)))}
                  />
                </td>
                <td className={cell}>
                  <input
                    className={`${SMALL_SELECT} w-full min-w-[20rem] font-mono`}
                    value={r.path}
                    placeholder="例如 D:\Experimental Data\SPM"
                    onChange={(e) => setRows((rs) => rs!.map((x) => (x.key === r.key ? { ...x, path: e.target.value } : x)))}
                  />
                </td>
                <td className={clsx(cell, r.exists === false && "text-mast-danger")}>
                  {r.exists === undefined ? "—" : r.exists ? "✓" : "不存在"}
                </td>
                <td className={cell}>
                  <button type="button" className={SMALL_BTN} onClick={() => setRows((rs) => rs!.filter((x) => x.key !== r.key))}>
                    删除
                  </button>
                </td>
              </tr>
            ))}
            {rows && !rows.length && (
              <tr>
                <td colSpan={5} className={`${cell} text-mast-muted`}>
                  还没有数据根。在下面加一个，或者从「建议」里挑。
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
      <div className="mt-2 flex flex-wrap items-center gap-2 text-[13px]">
        <button
          type="button"
          className={SMALL_BTN}
          onClick={() => setRows((rs) => [...(rs ?? []), { key: ++rowKey, name: "", path: "", enabled: true }])}
        >
          + 加一个根
        </button>
        <label className="flex items-center gap-1" title="构建时并行处理文件的线程数（1–16）">
          线程
          <input className={`${SMALL_SELECT} w-14`} value={workers} onChange={(e) => setWorkers(e.target.value)} />
        </label>
        <button type="button" className={clsx(SMALL_BTN, "!border-mast-accent !bg-mast-accent !text-mast-accent-ink")} disabled={saving || !rows} onClick={save}>
          {saving ? "保存中…" : "保存数据根"}
        </button>
        {saveMsg && <span className={saveMsg.ok ? "text-mast-auto" : "text-mast-danger"}>{saveMsg.text}</span>}
      </div>

      {(cfg.data?.suggestions?.length ?? 0) > 0 && (
        <>
          <h3 className="mb-1 mt-3 text-sm font-semibold">建议</h3>
          <ul className="space-y-1 text-[13px]">
            {cfg.data!.suggestions!.map((sg) => {
              const already = (rows ?? []).some((r) => r.path.trim().toLowerCase() === sg.path.toLowerCase());
              return (
                <li key={sg.path} className="flex flex-wrap items-center gap-2">
                  <span className="font-mono">{sg.path}</span>
                  <span className="text-mast-muted">{sg.why}</span>
                  <button
                    type="button"
                    className={SMALL_BTN}
                    disabled={already}
                    onClick={() => setRows((rs) => [...(rs ?? []), { key: ++rowKey, name: "", path: sg.path, enabled: true }])}
                  >
                    {already ? "已在列表里" : "加入"}
                  </button>
                </li>
              );
            })}
          </ul>
        </>
      )}

      <h2 className="mb-1.5 mt-5 text-[1.05rem] font-semibold">构建</h2>
      <Sub>
        增量更新只处理新文件与大小/修改时刻变了的文件；「全部重出」忽略渲染与判据缓存。构建在后台跑，关掉这一页不影响。
      </Sub>
      <div className="flex flex-wrap items-center gap-2 text-[13px]">
        <button type="button" className={clsx(SMALL_BTN, "!border-mast-accent !bg-mast-accent !text-mast-accent-ink")} disabled={running} onClick={() => build(false)}>
          增量更新
        </button>
        <button type="button" className={SMALL_BTN} disabled={running} onClick={() => build(true)} title="缩略图与自动判据全部重新生成（慢）">
          全部重出
        </button>
        <button type="button" className={SMALL_BTN} disabled={!running} onClick={cancel}>
          取消
        </button>
      </div>
      {status.isError && <ErrorNote error={status.error} />}
      {st && (
        <div className="mt-2.5 rounded-[3px] border border-mast-border bg-mast-panel px-3 py-2.5 text-[13px]">
          {st.degraded && <div className="mb-1 text-mast-warn">图库后端不可用：{st.detail || "未知原因"}</div>}
          <div>
            状态：<b>{PHASE[st.phase] ?? st.phase}</b>
            {running ? "（进行中）" : ""}
            {st.started && ` · 开始 ${st.started}`}
            {st.finished && ` · 结束 ${st.finished}`}
          </div>
          {st.total > 0 && (
            <div className="mt-1.5">
              <div className="h-1.5 overflow-hidden rounded bg-mast-panel-2">
                <div className="h-full bg-mast-accent transition-[width]" style={{ width: `${pct}%` }} />
              </div>
              <div className="mt-0.5 font-mono text-xs text-mast-muted">
                {st.done} / {st.total}（{pct}%）
              </div>
            </div>
          )}
          <div className="mt-1 font-mono text-xs text-mast-muted">
            条目 {st.n_files} · 新 {st.n_new} · 变动 {st.n_changed} · 渲染 {st.n_render} · 判据 {st.n_analysis} · 失败 {st.n_failed}
          </div>
          {st.message && <div className="mt-1">{st.message}</div>}
          {(st.errors?.length ?? 0) > 0 && (
            <details className="mt-1.5">
              <summary className="cursor-pointer text-mast-danger">失败 {st.errors!.length} 个</summary>
              <ul className="mt-1 max-h-48 overflow-auto font-mono text-xs">
                {st.errors!.map((e) => (
                  <li key={e.id}>
                    {e.id} — {e.why}
                  </li>
                ))}
              </ul>
            </details>
          )}
          {(st.log?.length ?? 0) > 0 && (
            <pre className="mt-1.5 max-h-56 overflow-auto whitespace-pre-wrap rounded bg-mast-code-bg p-2 font-mono text-xs text-mast-text">
              {st.log!.join("\n")}
            </pre>
          )}
        </div>
      )}

      <h2 className="mb-1.5 mt-5 text-[1.05rem] font-semibold">导入旧版兼容格式的标记</h2>
      <Sub>
        旧版兼容格式的 marks.json 以相对它数据根的路径为键（例如 2001/200101/20010101/example.sxm）。键前缀填这里对应的根名加斜杠，
        导入后单条、系列、系定与目录备注都按时间新者胜合并，标签表取并集；当前索引里找不到的键照样导入，不丢。
      </Sub>
      <ImportMarksForm defaultPrefix={cfg.data?.roots?.[0] ? `${cfg.data.roots[0].name}/` : ""} />
    </div>
  );
}
