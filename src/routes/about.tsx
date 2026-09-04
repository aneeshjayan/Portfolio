import { createFileRoute } from "@tanstack/react-router";
import { ABOUT_NUMBERS, EDUCATION } from "@/data/portfolio";
import { useScrollReveal } from "@/hooks/useScrollReveal";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "About — Aneesh Jayan Prabhu" },
      {
        name: "description",
        content:
          "Bio of Aneesh Jayan Prabhu — MS Data Science at ASU, AI/ML engineer open to ML Engineer, Data Scientist, AI Engineer, and Forward Deployed Engineer roles.",
      },
      { property: "og:title", content: "About — Aneesh Jayan Prabhu" },
    ],
  }),
  component: AboutPage,
});

const ACCENT = "oklch(0.85 0.19 145)";

function AboutPage() {
  const ref = useScrollReveal<HTMLDivElement>([]);

  return (
    <div ref={ref} style={{ animation: "dc-boot-in 0.4s ease-out both" }}>
      <p style={{ margin: 0, fontSize: 12, color: "oklch(0.5 0.02 200)" }}>
        <span style={{ color: ACCENT }}>$</span> cat ~/about/README.md
      </p>
      <h2
        style={{
          margin: "14px 0 4px",
          fontSize: 26,
          fontWeight: 700,
          color: "oklch(0.94 0.02 160)",
        }}
      >
        // bio
      </h2>

      <div className="term-two-col" style={{ marginTop: 16 }}>
        <div
          style={{
            border: "1px solid oklch(0.32 0.03 190 / 0.4)",
            background: "oklch(0.09 0.012 235 / 0.6)",
            padding: 20,
          }}
        >
          <p style={{ margin: 0, fontSize: 13, lineHeight: 1.9, color: "oklch(0.72 0.02 200)" }}>
            I started in electronics and signal processing at VIT — EEG and fMRI pipelines,
            quantum-hybrid architectures for autism detection — and found I cared less about the
            model and more about whether it survived contact with real data.
          </p>
          <p
            style={{
              margin: "14px 0 0",
              fontSize: 13,
              lineHeight: 1.9,
              color: "oklch(0.72 0.02 200)",
            }}
          >
            That took me to Wolters Kluwer, building agentic reporting over enterprise legal
            documents, and now to Revmo AI as a forward-deployed engineer: sitting with clients,
            running discovery, writing the spec, then shipping the multi-agent system that solves it
            — with the benchmarks, guardrails, and rollback story to back it up.
          </p>
          <p
            style={{
              margin: "14px 0 0",
              fontSize: 13,
              lineHeight: 1.9,
              color: "oklch(0.72 0.02 200)",
            }}
          >
            The work I like best lives at the inference layer: making a model cheap and fast enough
            that a business can actually afford to run it.
          </p>
        </div>

        <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
          <div
            style={{
              border: "1px solid oklch(0.32 0.03 190 / 0.4)",
              background: "oklch(0.09 0.012 235 / 0.6)",
              padding: 16,
            }}
          >
            <p style={{ margin: "0 0 12px", fontSize: 11, color: "oklch(0.5 0.02 200)" }}>
              // by the numbers
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 10 }}>
              {ABOUT_NUMBERS.map((n) => (
                <div key={n.label} style={{ display: "flex", alignItems: "baseline", gap: 10 }}>
                  <span style={{ fontSize: 17, fontWeight: 700, color: n.color, minWidth: 74 }}>
                    {n.value}
                  </span>
                  <span style={{ fontSize: 11, lineHeight: 1.5, color: "oklch(0.62 0.02 200)" }}>
                    {n.label}
                  </span>
                </div>
              ))}
            </div>
          </div>

          <div
            style={{
              border: "1px solid oklch(0.32 0.03 190 / 0.4)",
              background: "oklch(0.09 0.012 235 / 0.6)",
              padding: 16,
            }}
          >
            <p style={{ margin: "0 0 12px", fontSize: 11, color: "oklch(0.5 0.02 200)" }}>
              // education
            </p>
            <div style={{ display: "flex", flexDirection: "column", gap: 14 }}>
              {EDUCATION.map((d) => (
                <div key={d.school} style={{ display: "flex", alignItems: "center", gap: 12 }}>
                  <div
                    style={{
                      flexShrink: 0,
                      display: "flex",
                      height: 42,
                      width: 42,
                      alignItems: "center",
                      justifyContent: "center",
                      border: "1px solid oklch(0.32 0.03 190 / 0.4)",
                      background: "oklch(0.98 0.005 240)",
                      overflow: "hidden",
                    }}
                  >
                    <img
                      src={d.logo}
                      alt={d.school}
                      style={{ height: "100%", width: "100%", objectFit: "contain", padding: 2 }}
                    />
                  </div>
                  <div>
                    <div style={{ fontSize: 12.5, fontWeight: 700, color: "oklch(0.9 0.02 160)" }}>
                      {d.degree}
                    </div>
                    <div style={{ fontSize: 11, color: "oklch(0.6 0.02 200)" }}>{d.period}</div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
