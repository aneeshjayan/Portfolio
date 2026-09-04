import { useRef } from "react";
import * as d3 from "d3";
import { useResizeRedraw } from "@/hooks/useResizeRedraw";

const DATA = [
  { label: "inference latency", before: 850, after: 240, note: "−72%" },
  { label: "serving throughput", before: 100, after: 320, note: "3.2x" },
  { label: "inference cost", before: 100, after: 53, note: "−47%" },
  { label: "claims turnaround", before: 100, after: 20, note: "−80%" },
  { label: "hallucination rate", before: 100, after: 72, note: "−28%" },
  { label: "manual escalations", before: 100, after: 59, note: "−41%" },
];

export function ImpactChart() {
  const ref = useRef<HTMLDivElement>(null);

  useResizeRedraw(
    ref,
    (el) => {
      el.innerHTML = "";
      const W = el.clientWidth;
      const H = el.clientHeight;
      const m = { top: 26, right: 90, bottom: 24, left: 172 };
      const svg = d3
        .select(el)
        .append("svg")
        .attr("width", W)
        .attr("height", H)
        .style("display", "block");

      const y = d3
        .scaleBand()
        .domain(DATA.map((d) => d.label))
        .range([m.top, H - m.bottom])
        .padding(0.42);
      const x = d3
        .scaleLinear()
        .domain([0, d3.max(DATA, (d) => Math.max(d.before, d.after))!])
        .range([m.left, W - m.right]);

      const rows = svg.selectAll("g.r").data(DATA).join("g").attr("class", "r");

      rows
        .append("text")
        .attr("x", m.left - 12)
        .attr("y", (d) => y(d.label)! + y.bandwidth() * 0.66)
        .attr("text-anchor", "end")
        .text((d) => d.label)
        .attr("fill", "oklch(0.68 0.02 200)")
        .style("font-family", "'JetBrains Mono', monospace")
        .style("font-size", "11px");

      rows
        .append("rect")
        .attr("x", m.left)
        .attr("y", (d) => y(d.label)!)
        .attr("height", y.bandwidth() * 0.4)
        .attr("width", (d) => x(d.before) - m.left)
        .attr("fill", "oklch(0.34 0.02 240)");

      rows
        .append("rect")
        .attr("x", m.left)
        .attr("y", (d) => y(d.label)! + y.bandwidth() * 0.52)
        .attr("height", y.bandwidth() * 0.4)
        .attr("width", 0)
        .attr("fill", "oklch(0.85 0.19 145 / 0.75)")
        .transition()
        .duration(850)
        .delay((_d, i) => i * 70)
        .ease(d3.easeCubicOut)
        .attr("width", (d) => Math.max(2, x(d.after) - m.left));

      rows
        .append("text")
        .attr("x", W - m.right + 12)
        .attr("y", (d) => y(d.label)! + y.bandwidth() * 0.66)
        .text((d) => d.note)
        .attr("fill", "oklch(0.85 0.19 145)")
        .style("font-family", "'JetBrains Mono', monospace")
        .style("font-size", "12px")
        .style("font-weight", 700);

      svg
        .append("text")
        .attr("x", m.left)
        .attr("y", 14)
        .text("baseline ▪ vs shipped ▪")
        .attr("fill", "oklch(0.5 0.02 200)")
        .style("font-family", "'JetBrains Mono', monospace")
        .style("font-size", "10px");
    },
    [],
  );

  return <div ref={ref} id="viz-impact" style={{ width: "100%", height: "100%" }} />;
}
