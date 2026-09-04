import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { ACCENT, PROJECTS, type Category } from "@/data/portfolio";
import { useResizeRedraw } from "@/hooks/useResizeRedraw";

type NodeType = "core" | "cat" | "proj";

interface GraphNode extends d3.SimulationNodeDatum {
  id: string;
  type: NodeType;
  label: string;
  r: number;
  color: string;
  cat?: Category;
  idx?: number;
}

interface GraphLink {
  source: string | GraphNode;
  target: string | GraphNode;
}

const CATS: Category[] = ["agents", "inference", "security", "applied", "research"];

interface Props {
  filter: "all" | Category;
  selectedIndex: number;
  onSelectProject: (idx: number) => void;
  onSetFilter: (cat: "all" | Category) => void;
}

export function ForceGraph({ filter, selectedIndex, onSelectProject, onSetFilter }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const simRef = useRef<d3.Simulation<GraphNode, GraphLink> | null>(null);
  const selRef = useRef<{
    node: d3.Selection<SVGGElement, GraphNode, SVGGElement, unknown>;
    link: d3.Selection<SVGLineElement, GraphLink, SVGGElement, unknown>;
  } | null>(null);
  // Keep latest callbacks reachable from the click handler bound once at draw time.
  const callbacksRef = useRef({ onSelectProject, onSetFilter, filter });
  callbacksRef.current = { onSelectProject, onSetFilter, filter };

  useResizeRedraw(
    ref,
    (el) => {
      if (simRef.current) simRef.current.stop();
      el.innerHTML = "";
      const W = el.clientWidth;
      const H = el.clientHeight;

      const svg = d3
        .select(el)
        .append("svg")
        .attr("width", W)
        .attr("height", H)
        .style("display", "block");
      const defs = svg.append("defs");
      const glow = defs
        .append("filter")
        .attr("id", "nglow")
        .attr("x", "-80%")
        .attr("y", "-80%")
        .attr("width", "260%")
        .attr("height", "260%");
      glow.append("feGaussianBlur").attr("stdDeviation", 5).attr("result", "b");
      const fm = glow.append("feMerge");
      fm.append("feMergeNode").attr("in", "b");
      fm.append("feMergeNode").attr("in", "SourceGraphic");

      const zoomG = svg.append("g");
      svg.call(
        d3
          .zoom<SVGSVGElement, unknown>()
          .scaleExtent([0.55, 3])
          .on("zoom", (e) => zoomG.attr("transform", e.transform)),
      );

      const nodes: GraphNode[] = [
        {
          id: "core",
          type: "core",
          label: "aneesh",
          r: 26,
          color: ACCENT.research,
          fx: W / 2,
          fy: H / 2,
        },
      ];
      CATS.forEach((c) =>
        nodes.push({ id: c, type: "cat", label: "--" + c, r: 13, color: ACCENT[c], cat: c }),
      );
      PROJECTS.forEach((p, i) =>
        nodes.push({
          id: p.slug,
          type: "proj",
          label: p.name,
          r: 9,
          color: p.color,
          cat: p.category,
          idx: i,
        }),
      );

      const links: GraphLink[] = [
        ...CATS.map((c): GraphLink => ({ source: "core", target: c })),
        ...PROJECTS.map((p): GraphLink => ({ source: p.category, target: p.slug })),
      ];

      const sim = d3
        .forceSimulation(nodes)
        .force(
          "link",
          d3
            .forceLink<GraphNode, GraphLink>(links)
            .id((d) => d.id)
            .distance((l) => {
              const s = l.source as GraphNode;
              return s.type === "core" ? 130 : 92;
            })
            .strength(0.55),
        )
        .force(
          "charge",
          d3.forceManyBody().strength((d) => ((d as GraphNode).type === "proj" ? -320 : -680)),
        )
        .force("center", d3.forceCenter(W / 2, H / 2))
        .force(
          "collide",
          d3.forceCollide((d) => (d as GraphNode).r + 22),
        );
      simRef.current = sim;

      const link = zoomG
        .append("g")
        .selectAll<SVGLineElement, GraphLink>("line")
        .data(links)
        .join("line")
        .attr("stroke", (l) =>
          typeof l.target === "object" ? l.target.color : ACCENT[l.target as Category] || "#888",
        )
        .attr("stroke-width", 1)
        .attr("stroke-dasharray", "3 5")
        .style("animation", "dc-dash 1.4s linear infinite");

      const node = zoomG
        .append("g")
        .selectAll<SVGGElement, GraphNode>("g")
        .data(nodes)
        .join("g")
        .style("cursor", "pointer")
        .on("click", (_e, d) => {
          const cb = callbacksRef.current;
          if (d.type === "proj" && d.idx !== undefined) cb.onSelectProject(d.idx);
          else if (d.type === "cat" && d.cat) cb.onSetFilter(cb.filter === d.cat ? "all" : d.cat);
          else cb.onSetFilter("all");
        });

      node
        .append("circle")
        .attr("r", (d) => d.r)
        .attr("fill", (d) =>
          d.type === "core"
            ? d.color
            : `color-mix(in oklab, ${d.color} 22%, oklch(0.12 0.012 235))`,
        )
        .attr("stroke", (d) => d.color)
        .attr("stroke-width", (d) => (d.type === "proj" ? 1.2 : 1.6))
        .attr("filter", (d) => (d.type === "core" ? "url(#nglow)" : null));

      node
        .filter((d) => d.type === "core")
        .append("text")
        .text("AJP")
        .attr("text-anchor", "middle")
        .attr("dy", 4)
        .style("font-family", "'JetBrains Mono', monospace")
        .style("font-size", "11px")
        .style("font-weight", 700)
        .attr("fill", "#04070a");

      node
        .filter((d) => d.type !== "core")
        .append("text")
        .text((d) => d.label)
        .attr("text-anchor", "middle")
        .attr("dy", (d) => -d.r - 7)
        .style("font-family", "'JetBrains Mono', monospace")
        .style("font-size", (d) => (d.type === "cat" ? "11px" : "10px"))
        .attr("fill", (d) => d.color);

      node.call(
        d3
          .drag<SVGGElement, GraphNode>()
          .on("start", (e, d) => {
            if (!e.active) sim.alphaTarget(0.25).restart();
            d.fx = d.x;
            d.fy = d.y;
          })
          .on("drag", (e, d) => {
            d.fx = e.x;
            d.fy = e.y;
          })
          .on("end", (e, d) => {
            if (!e.active) sim.alphaTarget(0);
            if (d.type !== "core") {
              d.fx = null;
              d.fy = null;
            }
          }),
      );

      sim.on("tick", () => {
        link
          .attr("x1", (l) => (l.source as GraphNode).x!)
          .attr("y1", (l) => (l.source as GraphNode).y!)
          .attr("x2", (l) => (l.target as GraphNode).x!)
          .attr("y2", (l) => (l.target as GraphNode).y!);
        node.attr("transform", (d) => `translate(${d.x},${d.y})`);
      });

      selRef.current = { node, link };
    },
    [],
  );

  // Re-apply dim/highlight styling whenever filter or selection changes, without re-running the simulation.
  useEffect(() => {
    const sel = selRef.current;
    if (!sel) return;
    const { node, link } = sel;
    const dim = (d: GraphNode) => filter !== "all" && d.cat && d.cat !== filter;

    node.style("opacity", (d) => (dim(d) ? 0.16 : 1));
    node
      .select("circle")
      .attr("r", (d) => (d.type === "proj" && d.idx === selectedIndex ? d.r + 4 : d.r))
      .attr("stroke-width", (d) =>
        d.type === "proj" && d.idx === selectedIndex ? 2.4 : d.type === "proj" ? 1.2 : 1.6,
      )
      .attr("filter", (d) =>
        d.type === "core" || (d.type === "proj" && d.idx === selectedIndex) ? "url(#nglow)" : null,
      );

    link.style("opacity", (l) => {
      const t = l.target as GraphNode;
      const s = l.source as GraphNode;
      if (filter === "all") return 0.45;
      const cat = t?.cat || s?.cat;
      return cat === filter ? 0.6 : 0.07;
    });
  }, [filter, selectedIndex]);

  useEffect(() => {
    return () => {
      simRef.current?.stop();
    };
  }, []);

  return <div ref={ref} id="viz-force" style={{ width: "100%", height: "100%" }} />;
}
