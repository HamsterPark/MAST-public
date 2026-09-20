import { useQuery } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { api } from "@/api/client";
import { fmtCurrent } from "@/lib/units";
import { CURVE_BOX, curveGeometry } from "@/lib/curvePath";

// 电流监控事件的缩略曲线 — the waveform an alert was actually judged on.
//
// a `tip_quality_drop` from the current monitor showed「无图像」.
// That was CORRECT and not a bug — the event comes from a 1 kHz current segment
// and there is no scan frame anywhere near it. But it is also not the whole
// truth: the segment IS stored, and drawing it is the one picture that belongs
// next to this judgement. (Borrowing a nearby scan image, which an earlier build
// did for a different event type, is the falsified history of #76/#78 and stays
// forbidden.)
//
// The segment id rides in `cause_ref` as `current_monitor#<seg_id>`, and that id
// is the same one /api/monitoring/segments/{seg_id}/data takes — both come from
// MonitoringService's `seg_id`, verified 2026-08-04.
//
// Sizing: an SVG with a viewBox scales to its container by itself, so this needs
// no ResizeObserver — that machinery (hooks/useElementWidth + lib/chartSize)
// exists for <canvas>, which cannot honour `width: 100%`. Nothing here is in px.

/** Enough shape to see a transient; far below the 4000 the detail view asks for. */
const THUMB_POINTS = 400;

/** The expanded alert row is a wide strip, so it can carry more of the trace. */
const WIDE_POINTS = 1200;

function Frame({
  children,
  title,
  wide,
}: {
  children: React.ReactNode;
  title: string;
  wide?: boolean;
}) {
  return (
    <div
      title={title}
      className={
        "flex w-full flex-col justify-center rounded border border-mast-border bg-mast-bg p-1.5 " +
        (wide ? "aspect-[4/1]" : "aspect-square")
      }
    >
      {children}
    </div>
  );
}

function Caption({ children }: { children: React.ReactNode }) {
  return <div className="mt-1 truncate text-center text-[10px] text-mast-muted">{children}</div>;
}

export function SegmentCurveThumb({
  segId,
  hint,
  wide,
}: {
  segId: number;
  hint: string;
  /** Wide strip for the expanded alert row, instead of the square tile. */
  wide?: boolean;
}) {
  const maxPoints = wide ? WIDE_POINTS : THUMB_POINTS;
  const q = useQuery({
    // A written segment is immutable; the only thing that changes is whether the
    // retention sweep has taken the .npy, and that is slow. Same staleTime the
    // detail pane uses.
    staleTime: Infinity,
    queryKey: ["monitoring", "segment-data", segId, maxPoints],
    queryFn: async () => {
      const { data, error } = await api.GET("/api/monitoring/segments/{seg_id}/data", {
        params: { path: { seg_id: segId }, query: { max_points: maxPoints } },
      });
      if (error) throw error;
      return data;
    },
  });

  if (q.isPending) {
    return (
      <Frame title={hint} wide={wide}>
        <div className="text-center text-[10px] text-mast-faint">载入第 {segId} 段波形…</div>
      </Frame>
    );
  }

  const d = q.data;
  if (q.isError || !d?.ok) {
    // Say which of the two it is. "波形不可用" over a swept segment and over a
    // broken backend read the same, and only one of them is worth chasing.
    return (
      <Frame title={hint} wide={wide}>
        <div className="px-1 text-center text-[10px] leading-snug text-mast-muted">
          电流监控 · 第 {segId} 段
          <div className="mt-0.5 text-mast-faint">
            {q.isError ? "波形读取失败" : d?.detail || "该段波形已不可用"}
          </div>
        </div>
      </Frame>
    );
  }

  const geo = curveGeometry(d.i_a ?? []);
  const envelope = d.source === "envelope";

  if (!geo.points) {
    return (
      <Frame title={hint} wide={wide}>
        <div className="text-center text-[10px] text-mast-muted">
          第 {segId} 段只有 {geo.n} 个采样点，画不出曲线
        </div>
      </Frame>
    );
  }

  return (
    <Link
      // 指到**具体那一段**，不是组根。2026-08-06(#34)合并之后 `/monitoring`
      // 只是个中转，它会把人送到「上次停的那一段」—— 如果那是环境历史，这个
      // 链接就带着一个 `?seg=` 落在一张读不懂它的页面上。链接照样打得开，
      // 所以没人会意识到定位丢了。
      to={`/monitoring/current?seg=${segId}`}
      title={`${hint}\n点击打开第 ${segId} 段的完整波形与频谱`}
      className="block"
    >
      <Frame title="" wide={wide}>
        <svg
          viewBox={`0 0 ${CURVE_BOX} ${CURVE_BOX}`}
          preserveAspectRatio="none"
          className="h-full w-full"
          role="img"
          aria-label={`第 ${segId} 段电流波形`}
        >
          <polyline
            points={geo.points}
            fill="none"
            stroke="var(--mast-danger)"
            strokeWidth={1.2}
            vectorEffect="non-scaling-stroke"
            strokeLinejoin="round"
          />
        </svg>
      </Frame>
      <Caption>
        第 {segId} 段 · {fmtCurrent(geo.min)} ～ {fmtCurrent(geo.max)}
        {envelope ? " · 包络" : ""}
      </Caption>
    </Link>
  );
}
