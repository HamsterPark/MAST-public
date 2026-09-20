import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { api } from "../../api/client";
import type { components } from "../../api/schema";
import { Card, DegradedNote, EmptyNote, ErrorNote, Section, Spinner } from "../ui";
import { Button } from "../controls";
import { fmtTime } from "./shared";

// Tip-Shape 记录 sub-tab — reproduces tip_shape_records.py's producer output:
//   the post-skill hook renders tip-shape z/current dual-curve PNGs into the
//   artifacts dir (tip_shape_*.png) and registers them as scan_files. There is
//   no dedicated tip-shape endpoint, so we surface those PNGs via the shared
//   scan discovery + preview endpoints, filtered to tip_shape_*.png.
//   GET /api/scans/latest  +  GET /api/scans/preview

type ScanFileEntry = components["schemas"]["ScanFileEntry"];

function useLatestScans() {
  return useQuery({
    queryKey: ["scans", "latest"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/scans/latest");
      if (error) throw error;
      return data;
    },
  });
}

function useScanPreview(path: string | null) {
  return useQuery({
    enabled: !!path,
    queryKey: ["scans", "preview", path],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/scans/preview", {
        params: { query: { path: path as string } },
      });
      if (error) throw error;
      return data;
    },
  });
}

function isTipShape(s: ScanFileEntry): boolean {
  return s.name.toLowerCase().startsWith("tip_shape") && s.ext.toLowerCase() === ".png";
}

export function TipShapePane() {
  const q = useLatestScans();
  const [selected, setSelected] = useState<string | null>(null);

  const all: ScanFileEntry[] = q.data?.scans ?? [];
  const tips = all.filter(isTipShape);
  const effective = selected || tips[0]?.path || null;
  const prev = useScanPreview(effective);

  return (
    <div className="space-y-4">
      <Section
        title={`Tip-Shape 记录${tips.length ? ` (${tips.length})` : ""}`}
        actions={
          <Button variant="primary" onClick={() => q.refetch()}>
            刷新
          </Button>
        }
      >
        {q.isPending && <Spinner />}
        {q.isError && <ErrorNote error={q.error} />}
        {q.data?.degraded && <DegradedNote what="Tip-Shape 记录" />}
        {q.data && !q.data.degraded && tips.length === 0 && (
          <EmptyNote label="暂无 Tip-Shape 记录（TipShapeWithReadback 运行后生成 z/电流双曲线 PNG）" />
        )}

        {tips.length > 0 && (
          <div className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
            {tips.map((s) => (
              <button
                key={s.path}
                onClick={() => setSelected(s.path)}
                className={
                  "rounded-md border p-2 text-left text-xs transition-colors " +
                  (effective === s.path
                    ? "border-mast-accent bg-mast-accent/10"
                    : "border-mast-border hover:bg-mast-bg/60")
                }
              >
                <div className="truncate font-medium text-mast-text" title={s.name}>
                  {s.name}
                </div>
                <div className="mt-1 text-mast-muted tabular-nums">
                  {s.mtime ? fmtTime(new Date(s.mtime * 1000).toISOString()) : "—"}
                </div>
              </button>
            ))}
          </div>
        )}
      </Section>

      {effective && (
        <Section title="预览（z / 电流 双曲线）">
          <Card>
            <div className="mb-2 break-all font-mono text-xs text-mast-muted">{effective}</div>
            {prev.isPending && <Spinner />}
            {prev.isError && <ErrorNote error={prev.error} />}
            {prev.data?.degraded && <DegradedNote what="Tip-Shape 预览" />}
            {prev.data?.image && (
              <img
                src={prev.data.image}
                alt={effective}
                className="max-h-[480px] w-auto rounded border border-mast-border bg-white"
              />
            )}
          </Card>
        </Section>
      )}
    </div>
  );
}
