import { useQuery } from "@tanstack/react-query";
import { useNavigate } from "react-router-dom";
import { api } from "@/api/client";
import { writeStickyTab } from "@/lib/stickyTab";

// LATEST DATA strip — faithful to the old Gradio build_data_strip_html: a row of
// up to 4 most-recent .sxm scan THUMBNAILS + a "+" tile, headed by
// "LATEST DATA · live feed" with an "open data browser →" link. NOT a row of
// live readings (those live in the header). Thumbnails come from
// GET /api/scans/latest (mtime-desc) rendered via GET /api/scans/preview.

type ScanEntry = {
  path: string;
  name: string;
  ext: string;
  mtime?: number | null;
};

function ThumbTile({ scan, onOpen }: { scan: ScanEntry; onOpen: () => void }) {
  const preview = useQuery({
    queryKey: ["chat", "scan-preview", scan.path],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/scans/preview", {
        params: { query: { path: scan.path, size: 96 } },
      });
      if (error) throw error;
      return data;
    },
    staleTime: 5 * 60_000,
  });

  const img = preview.data?.image ?? null;
  const t = scan.mtime
    ? new Date(scan.mtime * 1000).toLocaleTimeString("zh-CN", { hour: "2-digit", minute: "2-digit", hour12: false })
    : "";
  const label = scan.name.replace(/\.[^.]+$/, "").slice(0, 16);

  return (
    <button
      type="button"
      onClick={onOpen}
      title={scan.name}
      className="flex w-24 shrink-0 cursor-pointer flex-col items-stretch rounded-md border border-mast-border bg-mast-bg p-1 text-left hover:border-mast-accent"
    >
      <div className="h-[72px] w-full overflow-hidden rounded bg-black/40">
        {img ? (
          <img src={img} alt="" className="h-full w-full object-cover" />
        ) : (
          <div className="h-full w-full" />
        )}
      </div>
      <div className="mt-0.5 truncate text-[11px] text-mast-text">{label}</div>
      <div className="flex items-center justify-between text-[10px] text-mast-muted">
        <span>{scan.ext}</span>
        <span>{t}</span>
      </div>
    </button>
  );
}

export function DataStrip() {
  const navigate = useNavigate();
  const q = useQuery({
    queryKey: ["chat", "scans-latest"],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/scans/latest", {
        params: { query: { n: 4 } },
      });
      if (error) throw error;
      return data;
    },
    refetchInterval: 10_000,
  });

  const scans = (q.data?.scans ?? []).slice(0, 4) as ScanEntry[];
  // Land on the 数据 tab, not on wherever the operator last was in 实验记录.
  // The route alone was not enough: `useStickyTab` restores the remembered
  // sub-tab on arrival, so "打开数据浏览器" opened the records page and then
  // navigated away from the thing it promised — a link that works and still
  // does not take you there.
  const openData = () => {
    writeStickyTab("records", "data");
    navigate("/records/log");
  };

  return (
    <div className="rounded-lg border border-mast-border bg-mast-panel/60 px-3 py-2">
      <div className="mb-1.5 flex items-center gap-2">
        <span className="text-[10px] font-semibold uppercase tracking-wider text-mast-muted">
          LATEST DATA
        </span>
        <span className="text-[10px] text-mast-muted">
          {scans.length ? "· live feed" : "· no scans yet"}
        </span>
        <button
          type="button"
          onClick={openData}
          title="打开数据浏览器"
          className="ml-auto cursor-pointer text-xs text-mast-accent hover:underline"
        >
          open data browser →
        </button>
      </div>
      <div className="flex items-stretch gap-2 overflow-x-auto">
        {scans.map((s) => (
          <ThumbTile key={s.path} scan={s} onOpen={openData} />
        ))}
        <button
          type="button"
          onClick={openData}
          title="打开数据浏览器"
          className="flex h-[96px] w-24 shrink-0 items-center justify-center rounded-md border border-dashed border-mast-border text-xl text-mast-muted hover:border-mast-accent hover:text-mast-accent"
        >
          +
        </button>
      </div>
    </div>
  );
}
