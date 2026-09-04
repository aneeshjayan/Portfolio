import { useEffect, useRef } from "react";
import * as d3 from "d3";
import { ACCENT } from "@/data/portfolio";
import { useResizeRedraw } from "@/hooks/useResizeRedraw";

const TERMS: [string, string][] = [
  ["LangGraph", ACCENT.agents],
  ["vLLM", ACCENT.inference],
  ["SGLang", ACCENT.inference],
  ["TensorRT", ACCENT.inference],
  ["MCP", ACCENT.agents],
  ["Agent SDK", ACCENT.agents],
  ["PyTorch", ACCENT.research],
  ["ONNX", ACCENT.inference],
  ["Triton", ACCENT.inference],
  ["CUDA", ACCENT.inference],
  ["QLoRA", ACCENT.research],
  ["PPO", ACCENT.security],
  ["GRPO", ACCENT.security],
  ["RAG", ACCENT.agents],
  ["FAISS", ACCENT.applied],
  ["Neo4j", ACCENT.applied],
  ["Kafka", ACCENT.applied],
  ["Redis", ACCENT.applied],
  ["Airflow", ACCENT.applied],
  ["PySpark", ACCENT.applied],
  ["FastAPI", ACCENT.inference],
  ["React", ACCENT.inference],
  ["Next.js", ACCENT.inference],
  ["gRPC", ACCENT.inference],
  ["Docker", ACCENT.research],
  ["Kubernetes", ACCENT.research],
  ["Terraform", ACCENT.research],
  ["Bedrock", ACCENT.agents],
  ["Vertex AI", ACCENT.agents],
  ["Claude", ACCENT.agents],
  ["Prometheus", ACCENT.applied],
  ["Grafana", ACCENT.applied],
  ["pytest", ACCENT.security],
  ["Guardrails", ACCENT.security],
  ["INT8", ACCENT.inference],
  ["Swin", ACCENT.research],
  ["XGBoost", ACCENT.applied],
  ["MLflow", ACCENT.inference],
  ["Linux", ACCENT.research],
  ["Postgres", ACCENT.applied],
];

interface Point {
  label: string;
  color: string;
  x: number;
  y: number;
  z: number;
}

export function Sphere3D() {
  const ref = useRef<HTMLDivElement>(null);
  const rafRef = useRef<number | null>(null);

  useResizeRedraw(
    ref,
    (el) => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
      el.innerHTML = "";
      const W = el.clientWidth;
      const H = el.clientHeight;
      const svg = d3
        .select(el)
        .append("svg")
        .attr("width", W)
        .attr("height", H)
        .style("display", "block")
        .style("cursor", "grab");
      const g = svg.append("g").attr("transform", `translate(${W / 2},${H / 2})`);

      const R = Math.min(W, H) * 0.36;
      const pts: Point[] = TERMS.map((t, i) => {
        const phi = Math.acos(1 - (2 * (i + 0.5)) / TERMS.length);
        const theta = Math.PI * (1 + Math.sqrt(5)) * (i + 0.5);
        return {
          label: t[0],
          color: t[1],
          x: R * Math.sin(phi) * Math.cos(theta),
          y: R * Math.cos(phi),
          z: R * Math.sin(phi) * Math.sin(theta),
        };
      });

      const wire = g.append("g");
      [0.42, 0.72, 1].forEach((k) => {
        wire
          .append("circle")
          .attr("r", R * k)
          .attr("fill", "none")
          .attr("stroke", "oklch(0.85 0.19 145 / 0.09)")
          .attr("stroke-dasharray", "2 4");
      });

      const nodes = g
        .selectAll<SVGGElement, Point>("g.term")
        .data(pts)
        .join("g")
        .attr("class", "term");
      nodes.append("circle").attr("r", 2);
      nodes
        .append("text")
        .attr("text-anchor", "middle")
        .attr("dy", -6)
        .style("font-family", "'JetBrains Mono', monospace")
        .style("font-weight", 500);

      let rotY = 0;
      let rotX = -0.22;
      const vY = 0.0035;
      let dragging = false;

      const render = () => {
        const cy = Math.cos(rotY),
          sy = Math.sin(rotY),
          cx = Math.cos(rotX),
          sx = Math.sin(rotX);
        nodes.each(function (p) {
          const x1 = p.x * cy - p.z * sy,
            z1 = p.x * sy + p.z * cy;
          const y2 = p.y * cx - z1 * sx,
            z2 = p.y * sx + z1 * cx;
          const scale = 340 / (340 + z2);
          const sel = d3.select(this);
          sel
            .attr("transform", `translate(${x1 * scale},${y2 * scale})`)
            .style("opacity", 0.25 + 0.75 * ((z2 + R) / (2 * R)));
          sel
            .select("circle")
            .attr("fill", p.color)
            .attr("r", 1.4 * scale);
          sel
            .select("text")
            .attr("fill", p.color)
            .style("font-size", `${10.5 * scale}px`)
            .text(p.label);
        });
      };

      const loop = () => {
        if (!dragging) rotY += vY;
        render();
        rafRef.current = requestAnimationFrame(loop);
      };

      svg.call(
        d3
          .drag<SVGSVGElement, unknown>()
          .on("start", () => {
            dragging = true;
            svg.style("cursor", "grabbing");
          })
          .on("drag", (e) => {
            rotY += e.dx * 0.006;
            rotX = Math.max(-1.1, Math.min(1.1, rotX + e.dy * 0.005));
          })
          .on("end", () => {
            dragging = false;
            svg.style("cursor", "grab");
          }),
      );

      loop();
    },
    [],
  );

  useEffect(
    () => () => {
      if (rafRef.current) cancelAnimationFrame(rafRef.current);
    },
    [],
  );

  return <div ref={ref} id="viz-sphere" style={{ width: "100%", height: "100%" }} />;
}
