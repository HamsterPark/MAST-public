import { useState } from "react";
import { useMutation, useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import { Card, Spinner, ErrorNote, DegradedNote, EmptyNote } from "../ui";
import type { components } from "../../api/schema";

type SearchHit = components["schemas"]["SearchHit"];
type SearchResponse = components["schemas"]["SearchResponse"];

/** 语义检索 over the big library, optionally scoped to a library's member set.
 *  Click a hit → load its full abstract (lazy GET /literature/abstract/{id}). */
export function SearchPanel({ libraryId }: { libraryId: string | null }) {
  const [query, setQuery] = useState("");
  const [material, setMaterial] = useState("");
  const [k, setK] = useState(8);
  const [openWorkId, setOpenWorkId] = useState<string | null>(null);

  const searchM = useMutation({
    mutationFn: async (): Promise<SearchResponse> => {
      const { data, error } = await api.POST("/api/literature/search", {
        body: {
          query: query.trim(),
          k,
          material: material.trim(),
          library_id: libraryId,
        },
      });
      if (error) throw error;
      return data;
    },
  });

  const res = searchM.data;
  const hits: SearchHit[] = res?.hits ?? [];

  return (
    <div className="space-y-4">
      <Card>
        <div className="flex flex-wrap items-end gap-2">
          <label className="flex flex-1 flex-col text-xs text-mast-muted">
            查询（语义）
            <input
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && query.trim()) searchM.mutate();
              }}
              placeholder="例如：CO tip functionalization on Cu(111)"
              className="mt-1 w-full rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text"
            />
          </label>
          <label className="flex flex-col text-xs text-mast-muted">
            材料过滤
            <input
              value={material}
              onChange={(e) => setMaterial(e.target.value)}
              placeholder="可选"
              className="mt-1 w-32 rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text"
            />
          </label>
          <label className="flex flex-col text-xs text-mast-muted">
            返回数
            <input
              type="number"
              min={1}
              max={200}
              value={k}
              onChange={(e) => setK(Math.max(1, Math.min(200, Number(e.target.value) || 1)))}
              className="mt-1 w-20 rounded border border-mast-border bg-mast-bg px-2 py-1 text-sm text-mast-text"
            />
          </label>
          <button
            onClick={() => searchM.mutate()}
            disabled={!query.trim() || searchM.isPending}
            className="rounded bg-mast-accent/20 px-3 py-1.5 text-sm text-mast-accent hover:bg-mast-accent/30 disabled:opacity-40"
          >
            {searchM.isPending ? "检索中…" : "检索"}
          </button>
        </div>
        <p className="mt-2 text-xs text-mast-muted">
          {libraryId
            ? `检索范围：库 ${libraryId}（大库检索 → 成员过滤）`
            : "检索范围：全大库"}
        </p>
      </Card>

      {searchM.isPending && <Spinner label="检索中…" />}
      {searchM.isError && <ErrorNote error={searchM.error} />}
      {res?.degraded && (
        <Card className="border-mast-warn-border">
          <p className="text-sm text-mast-warn">
            语义检索暂不可用（缺 DashScope key / 网络不通 / 后端未接入），结果可能已降级为关键词匹配或为空。
          </p>
        </Card>
      )}
      {res && !res.degraded && hits.length === 0 && searchM.isSuccess && (
        <EmptyNote label="无匹配结果。" />
      )}

      {hits.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs text-mast-muted">{res?.count ?? hits.length} 条结果</p>
          {hits.map((h) => (
            <Card key={h.work_id || h.title}>
              <div className="flex items-start justify-between gap-3">
                <div className="min-w-0">
                  <div className="font-medium">{h.title || "(无标题)"}</div>
                  <div className="mt-0.5 text-xs text-mast-muted">
                    {h.year || "?"}
                    {h.journal && ` · ${h.journal}`}
                    {` · 被引 ${h.cited}`}
                    {h.source && ` · ${h.source}`}
                    {` · ${h.retrieval}`}
                    {h.score > 0 && ` · score ${h.score.toFixed(3)}`}
                  </div>
                  <div className="mt-0.5 text-xs text-mast-muted">
                    <code>{h.work_id}</code>
                    {h.doi && (
                      <>
                        {" · "}
                        <a
                          href={`https://doi.org/${h.doi}`}
                          target="_blank"
                          rel="noreferrer"
                          className="text-mast-accent hover:underline"
                        >
                          {h.doi}
                        </a>
                      </>
                    )}
                  </div>
                  {h.abstract_excerpt && (
                    <p className="mt-2 text-sm text-mast-text/90">{h.abstract_excerpt}</p>
                  )}
                </div>
                <button
                  onClick={() =>
                    setOpenWorkId(openWorkId === h.work_id ? null : h.work_id)
                  }
                  disabled={!h.work_id}
                  className="shrink-0 rounded px-2 py-1 text-xs text-mast-accent hover:bg-mast-accent/20 disabled:opacity-30"
                >
                  {openWorkId === h.work_id ? "收起摘要" : "查看摘要"}
                </button>
              </div>
              {openWorkId === h.work_id && <AbstractView workId={h.work_id} />}
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}

/** Full abstract for one work — lazy GET /literature/abstract/{work_id}. */
export function AbstractView({ workId }: { workId: string }) {
  const q = useQuery({
    queryKey: ["literature", "abstract", workId],
    queryFn: async () => {
      const { data, error } = await api.GET(
        "/api/literature/abstract/{work_id}",
        { params: { path: { work_id: workId } } },
      );
      if (error) throw error;
      return data;
    },
    enabled: !!workId,
  });

  const rec = q.data;

  return (
    <div className="mt-3 rounded border border-mast-border bg-mast-bg/50 p-3">
      {q.isPending && <Spinner label="加载摘要…" />}
      {q.isError && <ErrorNote error={q.error} />}
      {rec?.degraded && <DegradedNote what="摘要服务" />}
      {rec && !rec.degraded && !rec.found && (
        <EmptyNote label={`未找到摘要。${rec.note || ""}`} />
      )}
      {rec && rec.found && (
        <div className="space-y-2">
          <div className="font-medium">{rec.title || workId}</div>
          <div className="text-xs text-mast-muted">
            {(rec.first_author || rec.authors) && (
              <>{rec.first_author || rec.authors} · </>
            )}
            {rec.year || "?"}
            {rec.journal && ` · ${rec.journal}`}
            {rec.source && ` · ${rec.source}`}
            {rec.cited_by_count > 0 && ` · 被引 ${rec.cited_by_count}`}
          </div>
          {rec.doi && (
            <div className="text-xs">
              <a
                href={`https://doi.org/${rec.doi}`}
                target="_blank"
                rel="noreferrer"
                className="text-mast-accent hover:underline"
              >
                {rec.doi}
              </a>
            </div>
          )}
          <pre className="whitespace-pre-wrap rounded bg-mast-panel p-2 text-sm text-mast-text/90">
            {rec.abstract || "(无摘要)"}
          </pre>
          {rec.user_abstract && (
            <div>
              <div className="text-xs text-mast-muted">用户摘要 ({rec.abstract_provenance || "user"})</div>
              <pre className="whitespace-pre-wrap rounded bg-mast-panel p-2 text-sm text-mast-text/90">
                {rec.user_abstract}
              </pre>
            </div>
          )}
        </div>
      )}
    </div>
  );
}
