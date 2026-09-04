import { useEffect, useRef } from "react";

/**
 * Fades + rises each direct child of the returned ref's element as it
 * scrolls into view, staggered by index. Re-binds whenever `deps` change
 * (e.g. on route change) so newly-mounted content animates in too.
 */
export function useScrollReveal<T extends HTMLElement>(deps: unknown[] = []) {
  const ref = useRef<T>(null);

  useEffect(() => {
    const scope = ref.current;
    if (!scope || typeof IntersectionObserver === "undefined") return;

    const blocks = Array.from(scope.children).filter((n): n is HTMLElement => n.nodeType === 1);

    blocks.forEach((el, i) => {
      el.style.willChange = "opacity, transform";
      el.style.transition = `opacity 0.6s cubic-bezier(.22,.61,.36,1) ${i * 55}ms, transform 0.7s cubic-bezier(.22,.61,.36,1) ${i * 55}ms`;
      el.style.opacity = "0";
      el.style.transform = "translateY(26px) scale(0.995)";
    });

    let observer: IntersectionObserver | undefined;
    const raf = requestAnimationFrame(() => {
      observer = new IntersectionObserver(
        (entries) => {
          entries.forEach((entry) => {
            const on = entry.isIntersecting;
            const el = entry.target as HTMLElement;
            el.style.opacity = on ? "1" : "0.12";
            el.style.transform = on ? "translateY(0) scale(1)" : "translateY(18px) scale(0.995)";
          });
        },
        { threshold: 0.06, rootMargin: "0px 0px -8% 0px" },
      );
      blocks.forEach((b) => observer!.observe(b));
    });

    return () => {
      cancelAnimationFrame(raf);
      observer?.disconnect();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps);

  return ref;
}
