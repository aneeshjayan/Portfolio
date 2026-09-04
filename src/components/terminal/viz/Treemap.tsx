import { useRef } from "react";
import * as d3 from "d3";
import { SKILL_GROUPS } from "@/data/portfolio";
import { useResizeRedraw } from "@/hooks/useResizeRedraw";

interface LeafDatum {
  name: string;
  color: string;
  items: string;
  value: number;
}

type HierarchyDatum = { children: LeafDatum[] } | LeafDatum;

export function Treemap() {
  const ref = useRef<HTMLDivElement>(null);

  useResizeRedraw(
    ref,
    (el) => {
      el.innerHTML = "";
      const W = el.clientWidth;
      const H = el.clientHeight;

      const root = d3
        .hierarchy<HierarchyDatum>({
          children: SKILL_GROUPS.map((g) => ({
            name: g.name,
            color: g.color,
            items: g.items,
            value: g.items.split(",").length,
          })),
        })
        .sum((d) => ("value" in d ? d.value : 0))
        .sort((a, b) => (b.value ?? 0) - (a.value ?? 0));

      const laidOut = d3.treemap<HierarchyDatum>().size([W, H]).paddingInner(3).paddingOuter(4)(
        root,
      );

      const svg = d3
        .select(el)
        .append("svg")
        .attr("width", W)
        .attr("height", H)
        .style("display", "block");
      const cell = svg
        .selectAll("g")
        .data(laidOut.leaves() as d3.HierarchyRectangularNode<LeafDatum>[])
        .join("g")
        .attr("transform", (d) => `translate(${d.x0},${d.y0})`);

      cell
        .append("rect")
        .attr("width", (d) => d.x1 - d.x0)
        .attr("height", (d) => d.y1 - d.y0)
        .attr("fill", (d) => `color-mix(in oklab, ${d.data.color} 16%, oklch(0.1 0.012 235))`)
        .attr("stroke", (d) => `color-mix(in oklab, ${d.data.color} 45%, transparent)`)
        .style("transition", "fill 0.2s")
        .on("mouseenter", function (_e, d) {
          d3.select(this).attr(
            "fill",
            `color-mix(in oklab, ${d.data.color} 30%, oklch(0.1 0.012 235))`,
          );
        })
        .on("mouseleave", function (_e, d) {
          d3.select(this).attr(
            "fill",
            `color-mix(in oklab, ${d.data.color} 16%, oklch(0.1 0.012 235))`,
          );
        });

      cell
        .append("foreignObject")
        .attr("x", 9)
        .attr("y", 7)
        .attr("width", (d) => Math.max(0, d.x1 - d.x0 - 18))
        .attr("height", (d) => Math.max(0, Math.min(46, (d.y1 - d.y0) * 0.42)))
        .style("pointer-events", "none")
        .append("xhtml:div")
        .style("font-family", "'JetBrains Mono', monospace")
        .style("font-size", (d) => `${d.x1 - d.x0 < 150 ? 10 : 11.5}px`)
        .style("font-weight", "700")
        .style("line-height", "1.3")
        .style("overflow", "hidden")
        .style("word-break", "break-word")
        .style("color", (d) => d.data.color)
        .text((d) => d.data.name);

      cell
        .filter((d) => d.y1 - d.y0 > 74)
        .append("text")
        .attr("x", 10)
        .attr("y", (d) => Math.min(46, (d.y1 - d.y0) * 0.42) + 20)
        .text((d) => `${d.data.value} tools`)
        .attr("fill", "oklch(0.55 0.02 200)")
        .style("font-family", "'JetBrains Mono', monospace")
        .style("font-size", "10px")
        .style("pointer-events", "none");

      cell
        .filter((d) => d.x1 - d.x0 > 150 && d.y1 - d.y0 > 130)
        .append("foreignObject")
        .attr("x", 10)
        .attr("y", (d) => Math.min(46, (d.y1 - d.y0) * 0.42) + 30)
        .attr("width", (d) => Math.max(0, d.x1 - d.x0 - 20))
        .attr("height", (d) => Math.max(0, d.y1 - d.y0 - (Math.min(46, (d.y1 - d.y0) * 0.42) + 40)))
        .style("pointer-events", "none")
        .append("xhtml:div")
        .style("font-family", "'JetBrains Mono', monospace")
        .style("font-size", "9.5px")
        .style("line-height", "1.65")
        .style("color", "oklch(0.6 0.02 200)")
        .style("overflow", "hidden")
        .style("word-break", "keep-all")
        .style("overflow-wrap", "normal")
        .text((d) => d.data.items);
    },
    [],
  );

  return <div ref={ref} id="viz-treemap" style={{ width: "100%", height: "100%" }} />;
}
