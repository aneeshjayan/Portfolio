import { createFileRoute } from "@tanstack/react-router";
import { SKILL_GROUPS } from "@/data/portfolio";
import { IconGlyph } from "@/components/terminal/IconGlyph";
import { Treemap } from "@/components/terminal/viz/Treemap";
import { useScrollReveal } from "@/hooks/useScrollReveal";

export const Route = createFileRoute("/skills")({
  head: () => ({
    meta: [
      { title: "Skills — Aneesh Jayan Prabhu" },
      {
        name: "description",
        content:
          "Agentic AI, ML inference optimization, backend engineering, and cloud infrastructure — sized by depth of use.",
      },
      { property: "og:title", content: "Skills — Aneesh Jayan Prabhu" },
    ],
  }),
  component: SkillsPage,
});

const ACCENT = "oklch(0.85 0.19 145)";

function SkillsPage() {
  const ref = useScrollReveal<HTMLDivElement>([]);

  return (
    <div ref={ref} style={{ animation: "dc-boot-in 0.4s ease-out both" }}>
      <p style={{ margin: 0, fontSize: 12, color: "oklch(0.5 0.02 200)" }}>
        <span style={{ color: ACCENT }}>$</span> ls -R ~/stack | viz --treemap
      </p>
      <h2
        style={{
          margin: "14px 0 4px",
          fontSize: 26,
          fontWeight: 700,
          color: "oklch(0.94 0.02 160)",
        }}
      >
        // stack
      </h2>
      <p style={{ margin: 0, fontSize: 12.5, color: "oklch(0.6 0.02 200)" }}>
        Sized by depth of use, not by how it reads on a résumé. Hover any block for the full list.
      </p>

      <div
        style={{
          marginTop: 18,
          border: "1px solid oklch(0.32 0.03 190 / 0.4)",
          background: "oklch(0.075 0.012 235 / 0.7)",
        }}
      >
        <div style={{ width: "100%", height: 460 }}>
          <Treemap />
        </div>
      </div>

      <div
        style={{
          marginTop: 22,
          display: "grid",
          gridTemplateColumns: "repeat(auto-fill, minmax(300px, 1fr))",
          gap: 14,
        }}
      >
        {SKILL_GROUPS.map((g, i) => (
          <div
            key={g.name}
            style={{
              border: "1px solid oklch(0.32 0.03 190 / 0.4)",
              background: "oklch(0.09 0.012 235 / 0.6)",
              padding: "14px 16px",
            }}
          >
            <div style={{ display: "flex", alignItems: "center", gap: 9 }}>
              <span
                style={{
                  display: "flex",
                  height: 28,
                  width: 28,
                  flexShrink: 0,
                  alignItems: "center",
                  justifyContent: "center",
                  border: `1px solid color-mix(in oklab, ${g.color} 35%, transparent)`,
                  background: `color-mix(in oklab, ${g.color} 12%, transparent)`,
                  color: g.color,
                }}
              >
                <IconGlyph icon={g.icon} />
              </span>
              <span style={{ fontSize: 10.5, color: "oklch(0.45 0.02 200)" }}>
                {String(i + 1).padStart(2, "0")}
              </span>
              <span style={{ fontSize: 12.5, fontWeight: 700, color: g.color }}>{g.name}</span>
            </div>
            <p
              style={{
                margin: "8px 0 0",
                fontSize: 11.5,
                lineHeight: 1.85,
                color: "oklch(0.66 0.02 200)",
              }}
            >
              {g.items}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
