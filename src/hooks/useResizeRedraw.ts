import { useEffect, type RefObject } from "react";

/**
 * Runs `draw(el)` once the container has a non-zero width, and again on
 * every resize. `draw` is expected to fully clear and repaint `el` each
 * call (e.g. `el.innerHTML = ""` then rebuild an SVG).
 */
export function useResizeRedraw(
  ref: RefObject<HTMLElement | null>,
  draw: (el: HTMLElement) => void,
  deps: unknown[],
) {
  useEffect(() => {
    const el = ref.current;
    if (!el) return;

    const run = () => {
      if (el.clientWidth > 0) draw(el);
    };

    run();
    const ro = new ResizeObserver(run);
    ro.observe(el);
    return () => ro.disconnect();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);
}
