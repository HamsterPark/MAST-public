// Inline SVG sparkline — typed replacement for gui/status_panel._svg_sparkline
// and the vision_buffer events/s sparkline. Pure presentational.
//
// ── why `fill` defaults to off ─────────────────────────────────────────────
// then #10: 「不宜图线下方填色」. Shading the area under a line
// claims the area means something, and here it cannot: the y scale is
// auto-fitted to [min, max] of THIS series (see `vmin`/`rng` below), so the
// shaded region is measured from the window's own minimum, not from zero. A
// 0.1 pA ripple and a 100 pA collapse therefore paint the same block. The line
// alone carries every bit of information the fill did, without the claim.
//
// It stays a prop rather than being deleted because a series that genuinely is
// a quantity-above-zero (a count, a duration) can honestly be filled — but the
// caller has to say so.

export function Sparkline({
  values,
  color = "var(--mast-accent)",
  width = 70,
  height = 22,
  fill = false,
}: {
  values: number[];
  color?: string;
  width?: number;
  height?: number;
  fill?: boolean;
}) {
  if (!values || values.length < 2) {
    return (
      <svg width={width} height={height} className="block">
        <line
          x1={0}
          y1={height - 1}
          x2={width}
          y2={height - 1}
          stroke="var(--mast-border)"
          strokeWidth={1}
        />
      </svg>
    );
  }
  const vmin = Math.min(...values);
  const vmax = Math.max(...values);
  let rng = vmax - vmin;
  if (rng === 0) rng = 1;
  const n = values.length;
  const pts = values.map((v, i) => {
    const x = (i * width) / (n - 1);
    const y = height - ((v - vmin) / rng) * (height - 2) - 1;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  });
  const poly = pts.join(" ");
  const fillPts = `0,${height} ${poly} ${width},${height}`;
  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      preserveAspectRatio="none"
      className="block"
    >
      {fill && <polygon points={fillPts} fill={color} opacity={0.12} />}
      <polyline
        points={poly}
        fill="none"
        stroke={color}
        strokeWidth={1.3}
        strokeLinejoin="round"
        strokeLinecap="round"
      />
    </svg>
  );
}
