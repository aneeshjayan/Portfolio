import { useRef } from "react";
import * as d3 from "d3";
import { ROLES } from "@/data/portfolio";
import { useResizeRedraw } from "@/hooks/useResizeRedraw";

export function Timeline() {
  const ref = useRef<HTMLDivElement>(null);

  useResizeRedraw(
    ref,
    (el) => {
      el.innerHTML = "";
      const W = el.clientWidth;
      const H = el.clientHeight;
      const m = { top: 26, right: 24, bottom: 30, left: 24 };
      const svg = d3
        .select(el)
        .append("svg")
        .attr("width", W)
        .attr("height", H)
        .style("display", "block");

      const x = d3
        .scaleLinear()
        .domain([2023.1, 2026.9])
        .range([m.left, W - m.right]);
      svg
        .append("g")
        .attr("transform", `translate(0,${H - m.bottom})`)
        .call(
          d3
            .axisBottom(x)
            .tickFormat(d3.format("d"))
            .ticks(5)
            .tickSize(-(H - m.top - m.bottom)),
        )
        .call((g) => {
          g.select(".domain").attr("stroke", "oklch(0.32 0.03 190 / 0.5)");
          g.selectAll(".tick line").attr("stroke", "oklch(0.32 0.03 190 / 0.25)");
          g.selectAll("text")
            .attr("fill", "oklch(0.55 0.02 200)")
            .style("font-family", "'JetBrains Mono', monospace")
            .style("font-size", "10px");
        });

      const rowH = (H - m.top - m.bottom) / ROLES.length;
      const rows = svg
        .selectAll("g.role")
        .data(ROLES.slice().reverse())
        .join("g")
        .attr("class", "role")
        .attr("transform", (_d, i) => `translate(0,${m.top + i * rowH})`);

      rows
        .append("rect")
        .attr("x", (d) => x(d.start))
        .attr("y", rowH * 0.18)
        .attr("height", rowH * 0.42)
        .attr("width", 0)
        .attr("fill", (d) => `color-mix(in oklab, ${d.color} 30%, transparent)`)
        .attr("stroke", (d) => d.color)
        .attr("stroke-width", 1)
        .transition()
        .duration(750)
        .ease(d3.easeCubicOut)
        .attr("width", (d) => Math.max(4, x(d.end) - x(d.start)));

      rows
        .append("text")
        .attr("x", (d) => x(d.start) + 8)
        .attr("y", rowH * 0.18 - 6)
        .text((d) => d.company)
        .attr("fill", (d) => d.color)
        .style("font-family", "'JetBrains Mono', monospace")
        .style("font-size", "10.5px");
    },
    [],
  );

  return <div ref={ref} id="viz-timeline" style={{ width: "100%", height: "100%" }} />;
}
