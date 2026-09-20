import { useCallback, useEffect, useRef, useState } from "react";

// Measure the width a container actually got, and keep it current.
//
// uPlot and Konva both paint to <canvas>, which needs a concrete pixel width —
// neither honours `width: 100%`. So "responsive" here means: measure the box CSS
// gave us, hand the number to the chart, remeasure when it changes.
//
// Returns a CALLBACK ref rather than an object ref on purpose: an object ref is
// still null on the first render, so the observer would only attach after an
// extra effect pass and the chart would paint once at its fallback size. The
// callback fires the moment React attaches the node.

export function useElementWidth<T extends HTMLElement = HTMLDivElement>(): [
  (node: T | null) => void,
  number | null,
] {
  const [width, setWidth] = useState<number | null>(null);
  const observer = useRef<ResizeObserver | null>(null);

  const ref = useCallback((node: T | null) => {
    observer.current?.disconnect();
    observer.current = null;
    if (!node) return;

    const publish = (w: number) => {
      // Sub-pixel churn from flexbox would otherwise re-run uPlot's setSize on
      // every scroll-driven reflow.
      setWidth((prev) => (prev != null && Math.abs(prev - w) < 1 ? prev : w));
    };
    publish(node.getBoundingClientRect().width);

    // jsdom and older webviews have no ResizeObserver. One measurement is still
    // better than a hardcoded 860 — degrade, never throw.
    if (typeof ResizeObserver === "undefined") return;
    const ro = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (!entry) return;
      // contentRect excludes padding/border, which is what the canvas may use.
      publish(entry.contentRect.width);
    });
    ro.observe(node);
    observer.current = ro;
  }, []);

  useEffect(() => () => observer.current?.disconnect(), []);

  return [ref, width];
}

/** Viewport height, tracked so a chart can stay above the fold. 0 until mounted. */
export function useViewportHeight(): number | null {
  const [h, setH] = useState<number | null>(
    typeof window === "undefined" ? null : window.innerHeight,
  );
  useEffect(() => {
    if (typeof window === "undefined") return;
    const on = () => setH(window.innerHeight);
    on();
    window.addEventListener("resize", on);
    return () => window.removeEventListener("resize", on);
  }, []);
  return h;
}
