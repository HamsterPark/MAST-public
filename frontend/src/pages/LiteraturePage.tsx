import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "@/api/client";
import { useCurrentScope } from "@/api/scope";
import { libraryLabel } from "@/lib/libraryLabel";
import { Section } from "@/components/ui";
import { SubTabs } from "@/components/controls";
import { LibraryManager } from "@/components/literature/LibraryManager";
import { SearchPanel, AbstractView } from "@/components/literature/SearchPanel";
import { AddMembers } from "@/components/literature/AddMembers";
import {
  IngestReadiness,
  IngestPanel,
  FetchPanel,
} from "@/components/literature/IngestFetch";
import { FetchBoard } from "@/components/literature/FetchBoard";
import { PdfUploadPanel, ManualEntryPanel } from "@/components/literature/UploadManual";
import { useStickyTab } from "@/hooks/useStickyTab";

// 文献库 — Domain G. Full-parity port of the Gradio 文献库 tab
// (mast.gui.literature_panel) over the typed FastAPI routes. Flat SubTabs (no
// nested gr.Tabs → no freeze):
//   1. 库管理     — list / create / rename / delete pointer-set libraries + add members
//   2. 语义检索   — search the big library (optionally scoped to a library)
//   3. 摘要查询   — fetch a full abstract directly by work_id
//   4. 摄取与取文 — PDF ingest + DOI/URL fetch + readiness probe
//   5. 取文请求板 — agent-asks-user full-text request board
// 大库是真的库；其他库都是大库的指针。所有论文都进大库。

type View = "libraries" | "search" | "abstract" | "ingest" | "board";

const VIEW_TABS: { id: View; label: string }[] = [
  { id: "libraries", label: "库管理" },
  { id: "search", label: "语义检索" },
  { id: "abstract", label: "摘要查询" },
  { id: "ingest", label: "摄取与取文" },
  { id: "board", label: "取文请求板" },
];

export default function LiteraturePage() {
  // 「取文请求板」上有等着处理的请求(顶栏那个角标数的就是它),
  // 每次回来被打回「库管理」等于每次重新点一遍去看那个角标指的东西。
  const [view, setView] = useStickyTab<View>(
    "literature", VIEW_TABS.map((t) => t.id), "libraries");
  // selected library id scopes 语义检索 / 摄取 / 加入成员 to that library.
  //
  // null = 还没选过 → 用后端算出的**有效库**（当前实验的专属库；没有活跃实验时
  // 是手动指针，再兜底到 reading）。后端的 schema 一直写着「active_library_id is
  // the id the UI should pre-select」，旧 Gradio 版也确实预选了，但 React 版这里
  // 硬编码成 null 且从不读它 —— 于是「后端的活动库」和「前端选中的库」是两个互不
  // 相通的概念，AddMembers 只会甩一句「先在上方选择一个库」。
  const [pickedLibraryId, setPickedLibraryId] = useState<string | null>(null);
  const [abstractInput, setAbstractInput] = useState("");
  const [abstractWorkId, setAbstractWorkId] = useState<string | null>(null);
  // a fetched PDF path can flow into the ingest panel.
  const [fetchedPath, setFetchedPath] = useState<string | null>(null);
  // 当前实验 —— 用来给还没建出来的实验专属库取名（见下方 libraryLabel）。
  const scope = useCurrentScope();

  // Pending full-text-request badge — auto-surfaces (poll 5 s) at the top of the
  // tab regardless of which sub-view is open, mirroring the OLD Gradio
  // render_fetch_badge_html banner (so the user sees agent requests without
  // opening 取文请求板). Read-only, cheap, safe to poll.
  const boardQ = useQuery({
    queryKey: ["literature", "fetch-board", ""],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/literature/fetch-board", {
        params: { query: { status: null } },
      });
      if (error) throw error;
      return data;
    },
    refetchInterval: 5000,
  });
  const pending = boardQ.data?.pending_count ?? 0;

  // 有效库：一实验一专属库，切实验就跟着变（后端每次现读 active_scope 算出来）。
  const libsQ = useQuery({
    queryKey: ["literature", "libraries", "effective"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/literature/libraries");
      if (error) throw error;
      return data;
    },
    refetchInterval: 30_000,
  });
  const effectiveId = libsQ.data?.effective_library_id ?? null;
  const effectiveSource = libsQ.data?.effective_source ?? "";
  const selectedLibraryId = pickedLibraryId ?? effectiveId;
  const usingEffective = pickedLibraryId === null && !!effectiveId;
  // 名字，不是编号。实验库是懒创建的，所以在这个实验第一次收录文献之前，
  // effective_library_id 指向一个还没进 registry 的库 —— `find()` 落空，从前就
  // 直接把 `exp_example` 摆给用户看。当前实验名走的是 TopBar 已经在
  // 轮询的那个 queryKey，不额外发请求。
  const label = libraryLabel(
    effectiveId ?? "",
    effectiveSource,
    libsQ.data?.libraries ?? [],
    scope.data?.experiment?.name,
  );

  return (
    <div>
      {/* OLD intro markdown (app.py gr.Markdown). */}
      <p className="mb-3 text-sm text-mast-muted">
        <strong className="text-mast-text">文献库</strong> —
        大库是唯一真实的库；其他库都是大库的指针(work_id)。用户添加的论文都促进进大库，库只持有引用。
      </p>

      {/* OLD render_fetch_badge_html — auto-surfacing pending-request banner. */}
      {pending > 0 && (
        <button
          onClick={() => setView("board")}
          className="mb-3 block w-full rounded-lg border border-mast-warn-border bg-mast-warn-bg px-3 py-2 text-left text-sm text-mast-warn hover:bg-mast-warn-border"
        >
          📥 文献 agent 有 <b>{pending}</b> 条全文请求待你应答 —— 见「取文请求板」，上传 PDF / 取文即可满足。
        </button>
      )}

      {/* 当前生效的库 —— 用户在任何子标签里都该看得见文献会落到哪儿。 */}
      {effectiveId && (
        <p className="mb-3 text-xs text-mast-muted">
          当前生效的库：
          <b className="text-mast-text" title={effectiveId}>
            {label.text}
          </b>
          {effectiveSource === "experiment"
            ? "（当前实验的专属库）"
            : effectiveSource === "manual"
              ? "（手动指定；当前没有活跃实验）"
              : "（默认库）"}
          {/* 名字是从实验推出来的，库本身还没落盘 —— 别让它看起来已经在那儿了。 */}
          {label.pending && "，首次收录文献时自动建立"}
          {!usingEffective && (
            <>
              {" · 你已手动选择了别的库 "}
              <button
                className="underline decoration-dotted"
                onClick={() => setPickedLibraryId(null)}
              >
                恢复为生效库
              </button>
            </>
          )}
        </p>
      )}

      <SubTabs<View> value={view} onChange={setView} tabs={VIEW_TABS} />

      {view === "libraries" && (
        <Section title="文献库管理">
          <div className="space-y-4">
            <LibraryManager
              selectedId={selectedLibraryId}
              onSelect={(id) => setPickedLibraryId(id)}
            />
            <AddMembers libraryId={selectedLibraryId} />
            {/* #125 直接管理条目：手动新增一个不在 OpenAlex、也无 PDF 的条目。 */}
            <ManualEntryPanel libraryId={selectedLibraryId} />
          </div>
        </Section>
      )}

      {view === "search" && (
        <Section title="语义检索（大库）">
          <SearchPanel libraryId={selectedLibraryId} />
        </Section>
      )}

      {view === "abstract" && (
        <Section title="摘要查询（按 work_id）">
          <div className="space-y-4">
            <div className="flex items-end gap-2">
              <label className="flex flex-1 flex-col text-xs text-mast-muted">
                work_id
                <input
                  value={abstractInput}
                  onChange={(e) => setAbstractInput(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter" && abstractInput.trim())
                      setAbstractWorkId(abstractInput.trim());
                  }}
                  placeholder="例如：W2912345678"
                  className="mt-1 w-full rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text"
                />
              </label>
              <button
                onClick={() => setAbstractWorkId(abstractInput.trim() || null)}
                disabled={!abstractInput.trim()}
                className="rounded bg-mast-accent/20 px-3 py-1.5 text-sm text-mast-accent hover:bg-mast-accent/30 disabled:opacity-40"
              >
                查询摘要
              </button>
            </div>
            {abstractWorkId && <AbstractView workId={abstractWorkId} />}
          </div>
        </Section>
      )}

      {view === "ingest" && (
        <Section title="摄取与取文">
          <div className="space-y-4">
            <IngestReadiness />
            {/* #125 附加 PDF：浏览器直接上传 PDF（主路径）。 */}
            <PdfUploadPanel libraryId={selectedLibraryId} />
            <FetchPanel onFetchedPath={(p) => setFetchedPath(p)} />
            <IngestPanel
              key={fetchedPath ?? "ingest"}
              libraryId={selectedLibraryId}
            />
            {fetchedPath && (
              <p className="text-xs text-mast-muted">
                上次取到的 PDF：<code>{fetchedPath}</code>（可复制到上方摄取路径）
              </p>
            )}
          </div>
        </Section>
      )}

      {view === "board" && (
        <Section title="取文请求板（agent 请你取全文）">
          <FetchBoard />
        </Section>
      )}
    </div>
  );
}
