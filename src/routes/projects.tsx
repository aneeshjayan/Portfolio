import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { FILTERS, PROJECTS, type Category } from "@/data/portfolio";
import { ForceGraph } from "@/components/terminal/viz/ForceGraph";
import { useScrollReveal } from "@/hooks/useScrollReveal";

export const Route = createFileRoute("/projects")({
  head: () => ({
    meta: [
      { title: "Projects — Aneesh Jayan Prabhu" },
      {
        name: "description",
        content:
          "14 projects spanning agentic systems, inference optimization, AI security, and applied ML research.",
      },
      { property: "og:title", content: "Projects — Aneesh Jayan Prabhu" },
    ],
  }),
  component: ProjectsPage,
});

const ACCENT = "oklch(0.85 0.19 145)";

function ProjectsPage() {
  const ref = useScrollReveal<HTMLDivElement>([]);
  const [filter, setFilter] = useState<"all" | Category>("all");
  const [selectedIndex, setSelectedIndex] = useState(0);

  const total = PROJECTS.length;
  const selected = PROJECTS[selectedIndex];
  const index = `${String(selectedIndex + 1).padStart(2, "0")} / ${String(total).padStart(2, "0")}`;
  const shown = filter === "all" ? total : PROJECTS.filter((p) => p.category === filter).length;
  const prev = (selectedIndex - 1 + total) % total;
  const next = (selectedIndex + 1) % total;

  return (
    <div ref={ref} style={{ animation: "dc-boot-in 0.4s ease-out both" }}>
      <p style={{ margin: 0, fontSize: 12, color: "oklch(0.5 0.02 200)" }}>
        <span style={{ color: ACCENT }}>$</span> graph ~/projects --force-directed
      </p>
      <h2
        style={{
          margin: "14px 0 4px",
          fontSize: 26,
          fontWeight: 700,
          color: "oklch(0.94 0.02 160)",
        }}
      >
        // projects
      </h2>
      <p style={{ margin: 0, fontSize: 12.5, color: "oklch(0.6 0.02 200)" }}>
        Live force simulation. Drag nodes, scroll to zoom, click any node to inspect. Category nodes
        filter the graph.
      </p>

      <div style={{ marginTop: 18, display: "flex", flexWrap: "wrap", gap: 6 }}>
        {FILTERS.map((f) => {
          const on = filter === f.key;
          return (
            <button
              key={f.key}
              onClick={() => setFilter(f.key)}
              style={{
                border: `1px solid ${on ? `color-mix(in oklab, ${f.color} 50%, transparent)` : "oklch(0.32 0.03 190 / 0.4)"}`,
                background: on ? `color-mix(in oklab, ${f.color} 14%, transparent)` : "transparent",
                padding: "5px 11px",
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 11.5,
                color: on ? f.color : "oklch(0.55 0.02 200)",
                cursor: "pointer",
              }}
            >
              {f.label}
            </button>
          );
        })}
      </div>

      <div className="term-two-col" style={{ marginTop: 14 }}>
        <div
          style={{
            border: "1px solid oklch(0.32 0.03 190 / 0.4)",
            background: "oklch(0.075 0.012 235 / 0.7)",
            position: "relative",
          }}
        >
          <div style={{ width: "100%", height: 560 }}>
            <ForceGraph
              filter={filter}
              selectedIndex={selectedIndex}
              onSelectProject={setSelectedIndex}
              onSetFilter={setFilter}
            />
          </div>
          <span
            style={{
              position: "absolute",
              left: 12,
              bottom: 10,
              fontSize: 10,
              color: "oklch(0.48 0.02 200)",
            }}
          >
            {shown} of {total} projects · 5 categories · drag / zoom / click
          </span>
        </div>

        <div
          style={{
            border: "1px solid oklch(0.32 0.03 190 / 0.4)",
            background: "oklch(0.09 0.012 235 / 0.7)",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              borderBottom: "1px solid oklch(0.32 0.03 190 / 0.3)",
              padding: "8px 14px",
              fontSize: 10.5,
              color: "oklch(0.5 0.02 200)",
            }}
          >
            <span>~/projects/{selected.slug}.md</span>
            <span>{index}</span>
          </div>
          <div style={{ padding: 18 }}>
            <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
              <span
                style={{
                  height: 9,
                  width: 9,
                  borderRadius: 999,
                  background: selected.color,
                  boxShadow: `0 0 12px ${selected.color}`,
                }}
              />
              <span style={{ fontSize: 10.5, color: selected.color }}>--{selected.category}</span>
              <span style={{ marginLeft: "auto", fontSize: 10.5, color: "oklch(0.5 0.02 200)" }}>
                {selected.date}
              </span>
            </div>
            <h3
              style={{
                margin: "10px 0 2px",
                fontSize: 19,
                fontWeight: 700,
                color: "oklch(0.94 0.02 160)",
              }}
            >
              {selected.name}
            </h3>
            <p style={{ margin: 0, fontSize: 11.5, color: "oklch(0.6 0.02 200)" }}>
              {selected.blurb}
            </p>

            <div
              style={{
                marginTop: 16,
                display: "grid",
                gridTemplateColumns: "repeat(3, 1fr)",
                gap: 7,
              }}
            >
              {selected.metrics.map((m) => (
                <div
                  key={m.label}
                  style={{
                    border: "1px solid oklch(0.32 0.03 190 / 0.35)",
                    background: "oklch(0.06 0.01 235 / 0.6)",
                    padding: "9px 7px",
                    textAlign: "center",
                  }}
                >
                  <div style={{ fontSize: 13.5, fontWeight: 700, color: "oklch(0.88 0.02 160)" }}>
                    {m.value}
                  </div>
                  <div
                    style={{
                      marginTop: 2,
                      fontSize: 8.5,
                      textTransform: "uppercase",
                      letterSpacing: "0.06em",
                      color: "oklch(0.55 0.02 200)",
                    }}
                  >
                    {m.label}
                  </div>
                </div>
              ))}
            </div>

            <p style={{ margin: "16px 0 0", fontSize: 10.5, color: "oklch(0.5 0.02 200)" }}>
              ## problem
            </p>
            <p
              style={{
                margin: "5px 0 0",
                fontSize: 12.5,
                lineHeight: 1.75,
                color: "oklch(0.7 0.02 200)",
              }}
            >
              {selected.problem}
            </p>

            <p style={{ margin: "14px 0 0", fontSize: 10.5, color: "oklch(0.5 0.02 200)" }}>
              ## architecture
            </p>
            <pre
              style={{
                margin: "5px 0 0",
                overflowX: "auto",
                borderLeft: `2px solid ${selected.color}`,
                padding: "8px 0 8px 12px",
                fontSize: 10.5,
                lineHeight: 1.75,
                color: "oklch(0.68 0.02 200)",
              }}
            >
              {selected.architecture}
            </pre>

            <p style={{ margin: "14px 0 0", fontSize: 10.5, color: "oklch(0.5 0.02 200)" }}>
              ## stack
            </p>
            <div style={{ marginTop: 6, display: "flex", flexWrap: "wrap", gap: 5 }}>
              {selected.tech.map((t) => (
                <span
                  key={t}
                  style={{
                    border: "1px solid oklch(0.32 0.03 190 / 0.4)",
                    padding: "2px 7px",
                    fontSize: 10.5,
                    color: "oklch(0.66 0.02 200)",
                  }}
                >
                  {t}
                </span>
              ))}
            </div>

            <div style={{ marginTop: 16, display: "flex", flexWrap: "wrap", gap: 7 }}>
              {selected.repo && (
                <a
                  href={selected.repo}
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    display: "inline-block",
                    border: "1px solid oklch(0.32 0.03 190 / 0.5)",
                    padding: "7px 12px",
                    fontSize: 11.5,
                    color: "oklch(0.78 0.02 200)",
                    textDecoration: "none",
                  }}
                >
                  source ↗
                </a>
              )}
              {selected.link && (
                <a
                  href={selected.link}
                  target="_blank"
                  rel="noreferrer"
                  style={{
                    display: "inline-block",
                    border: "1px solid oklch(0.85 0.19 145 / 0.35)",
                    background: "oklch(0.85 0.19 145 / 0.08)",
                    padding: "7px 12px",
                    fontSize: 11.5,
                    color: ACCENT,
                    textDecoration: "none",
                  }}
                >
                  live demo ↗
                </a>
              )}
            </div>

            <div
              style={{
                marginTop: 18,
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                borderTop: "1px solid oklch(0.32 0.03 190 / 0.3)",
                paddingTop: 12,
              }}
            >
              <button
                onClick={() => setSelectedIndex(prev)}
                style={{
                  border: "none",
                  background: "none",
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 11,
                  color: "oklch(0.6 0.02 200)",
                  cursor: "pointer",
                }}
              >
                ← {String(prev + 1).padStart(2, "0")}
              </button>
              <button
                onClick={() => setSelectedIndex(next)}
                style={{
                  border: "none",
                  background: "none",
                  fontFamily: "'JetBrains Mono', monospace",
                  fontSize: 11,
                  color: "oklch(0.6 0.02 200)",
                  cursor: "pointer",
                }}
              >
                {String(next + 1).padStart(2, "0")} →
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
