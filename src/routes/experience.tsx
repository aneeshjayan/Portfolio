import { createFileRoute } from "@tanstack/react-router";
import { ROLES } from "@/data/portfolio";
import { Timeline } from "@/components/terminal/viz/Timeline";
import { ImpactChart } from "@/components/terminal/viz/ImpactChart";
import { useScrollReveal } from "@/hooks/useScrollReveal";

export const Route = createFileRoute("/experience")({
  head: () => ({
    meta: [
      { title: "Experience — Aneesh Jayan Prabhu" },
      {
        name: "description",
        content:
          "Forward-deployed AI engineering at Revmo AI, data science at Wolters Kluwer, and biomedical research at VIT.",
      },
      { property: "og:title", content: "Experience — Aneesh Jayan Prabhu" },
    ],
  }),
  component: ExperiencePage,
});

const ACCENT = "oklch(0.85 0.19 145)";

function ExperiencePage() {
  const ref = useScrollReveal<HTMLDivElement>([]);

  return (
    <div ref={ref} style={{ animation: "dc-boot-in 0.4s ease-out both" }}>
      <p style={{ margin: 0, fontSize: 12, color: "oklch(0.5 0.02 200)" }}>
        <span style={{ color: ACCENT }}>$</span> cat ~/experience --with-impact
      </p>
      <h2
        style={{
          margin: "14px 0 4px",
          fontSize: 26,
          fontWeight: 700,
          color: "oklch(0.94 0.02 160)",
        }}
      >
        // experience
      </h2>
      <p style={{ margin: 0, fontSize: 12.5, color: "oklch(0.6 0.02 200)" }}>
        Three roles, one thread: take a messy real-world problem and put a measurable system in
        production.
      </p>

      <div
        style={{
          marginTop: 18,
          border: "1px solid oklch(0.32 0.03 190 / 0.4)",
          background: "oklch(0.075 0.012 235 / 0.7)",
        }}
      >
        <div style={{ width: "100%", height: 190 }}>
          <Timeline />
        </div>
      </div>

      <div style={{ marginTop: 22, display: "flex", flexDirection: "column", gap: 16 }}>
        {ROLES.map((r) => (
          <div
            key={r.company}
            style={{
              border: "1px solid oklch(0.32 0.03 190 / 0.4)",
              background: "oklch(0.09 0.012 235 / 0.6)",
              borderLeft: `2px solid ${r.color}`,
            }}
          >
            <div
              style={{
                display: "flex",
                flexWrap: "wrap",
                alignItems: "center",
                gap: 12,
                borderBottom: "1px solid oklch(0.32 0.03 190 / 0.25)",
                padding: "14px 18px",
              }}
            >
              <div
                style={{
                  display: "flex",
                  height: 34,
                  width: 34,
                  alignItems: "center",
                  justifyContent: "center",
                  border: "1px solid oklch(0.32 0.03 190 / 0.4)",
                  background: "oklch(0.14 0.012 235)",
                  overflow: "hidden",
                }}
              >
                {r.logo ? (
                  <img
                    src={r.logo}
                    alt={r.company}
                    style={{ height: "100%", width: "100%", objectFit: "contain", padding: 3 }}
                  />
                ) : (
                  <span style={{ fontSize: 13, fontWeight: 700, color: r.color }}>{r.glyph}</span>
                )}
              </div>
              <div style={{ flex: "1 1 240px" }}>
                <div style={{ fontSize: 14.5, fontWeight: 700, color: "oklch(0.94 0.02 160)" }}>
                  {r.title}
                </div>
                <div style={{ fontSize: 11.5, color: r.color }}>{r.company}</div>
              </div>
              <div
                style={{
                  textAlign: "right",
                  fontSize: 10.5,
                  lineHeight: 1.6,
                  color: "oklch(0.55 0.02 200)",
                }}
              >
                <div>{r.period}</div>
                <div>{r.location}</div>
              </div>
            </div>
            <ul
              style={{
                margin: 0,
                padding: "14px 18px",
                listStyle: "none",
                display: "flex",
                flexDirection: "column",
                gap: 9,
              }}
            >
              {r.bullets.map((b) => (
                <li
                  key={b}
                  style={{
                    display: "flex",
                    gap: 10,
                    fontSize: 12.5,
                    lineHeight: 1.75,
                    color: "oklch(0.7 0.02 200)",
                  }}
                >
                  <span style={{ color: r.color }}>▸</span>
                  <span>{b}</span>
                </li>
              ))}
            </ul>
            <div style={{ display: "flex", flexWrap: "wrap", gap: 5, padding: "0 18px 16px" }}>
              {r.stack.map((t) => (
                <span
                  key={t}
                  style={{
                    border: "1px solid oklch(0.32 0.03 190 / 0.4)",
                    padding: "2px 7px",
                    fontSize: 10.5,
                    color: "oklch(0.62 0.02 200)",
                  }}
                >
                  {t}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>

      <div style={{ marginTop: 26 }}>
        <p style={{ margin: "0 0 10px", fontSize: 12, color: "oklch(0.5 0.02 200)" }}>
          <span style={{ color: ACCENT }}>$</span> bench --before-after
        </p>
        <div
          style={{
            border: "1px solid oklch(0.32 0.03 190 / 0.4)",
            background: "oklch(0.075 0.012 235 / 0.7)",
          }}
        >
          <div style={{ width: "100%", height: 320 }}>
            <ImpactChart />
          </div>
        </div>
      </div>
    </div>
  );
}
