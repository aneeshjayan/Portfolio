import { createFileRoute } from "@tanstack/react-router";
import { HEADLINE_METRICS } from "@/data/portfolio";
import { Sphere3D } from "@/components/terminal/viz/Sphere3D";
import { useScrollReveal } from "@/hooks/useScrollReveal";

export const Route = createFileRoute("/")({
  head: () => ({
    meta: [
      { title: "Aneesh Jayan Prabhu — AI/ML Engineer" },
      {
        name: "description",
        content:
          "AI/ML engineer building agentic systems, RAG pipelines, and production ML — from research to shipped products.",
      },
      { property: "og:title", content: "Aneesh Jayan Prabhu — AI/ML Engineer" },
      {
        property: "og:description",
        content: "AI/ML engineer building agentic systems, RAG pipelines, and production ML.",
      },
    ],
  }),
  component: HomePage,
});

const ACCENT = "oklch(0.85 0.19 145)";

function HomePage() {
  const ref = useScrollReveal<HTMLDivElement>([]);

  return (
    <div ref={ref} style={{ animation: "dc-boot-in 0.4s ease-out both" }}>
      <p style={{ margin: 0, fontSize: 12, color: "oklch(0.5 0.02 200)" }}>
        <span style={{ color: ACCENT }}>$</span> ./boot --profile aneesh
      </p>

      <div
        style={{
          marginTop: 22,
          display: "flex",
          gap: 34,
          flexWrap: "wrap",
          alignItems: "flex-start",
        }}
      >
        <div
          style={{
            flexShrink: 0,
            width: 300,
            border: "1px solid oklch(0.32 0.03 190 / 0.5)",
            background: "oklch(0.09 0.012 235 / 0.85)",
            boxShadow: "0 0 40px -12px oklch(0.85 0.19 145 / 0.25), 0 24px 50px -30px #000",
          }}
        >
          <div
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              borderBottom: "1px solid oklch(0.32 0.03 190 / 0.35)",
              padding: "7px 11px",
              fontSize: 10,
              color: "oklch(0.55 0.02 200)",
            }}
          >
            <span>~/profile.jpg</span>
            <span style={{ display: "flex", alignItems: "center", gap: 5, color: ACCENT }}>
              <span
                style={{
                  height: 5,
                  width: 5,
                  borderRadius: 999,
                  background: ACCENT,
                  animation: "dc-glow-pulse 2s ease-in-out infinite",
                }}
              />
              live
            </span>
          </div>
          <div style={{ position: "relative", overflow: "hidden", height: 300 }}>
            <div
              style={{
                position: "absolute",
                inset: 0,
                background:
                  "radial-gradient(circle at 50% 60%, oklch(0.85 0.19 145 / 0.16), transparent 65%)",
              }}
            />
            <img
              src="/profile.png"
              alt="Aneesh Jayan Prabhu"
              style={{
                position: "absolute",
                inset: 0,
                height: "100%",
                width: "100%",
                objectFit: "cover",
              }}
            />
            <div
              style={{
                position: "absolute",
                inset: 0,
                background:
                  "linear-gradient(160deg, oklch(0.85 0.19 145 / 0.22), transparent 45%, oklch(0.72 0.2 330 / 0.2))",
                mixBlendMode: "overlay",
                pointerEvents: "none",
              }}
            />
            <div
              style={{
                position: "absolute",
                inset: 0,
                background: "linear-gradient(to top, oklch(0.09 0.012 235) 2%, transparent 42%)",
                pointerEvents: "none",
              }}
            />
            <div
              style={{
                position: "absolute",
                inset: 0,
                backgroundImage:
                  "repeating-linear-gradient(oklch(0.85 0.19 145 / 0.07) 0 1px, transparent 1px 4px)",
                pointerEvents: "none",
              }}
            />
            <div
              style={{
                position: "absolute",
                inset: 14,
                border: "1px solid oklch(0.85 0.19 145 / 0.18)",
                pointerEvents: "none",
              }}
            />
            <span
              style={{
                position: "absolute",
                top: 12,
                left: 12,
                width: 12,
                height: 12,
                borderTop: "1px solid oklch(0.85 0.19 145 / 0.7)",
                borderLeft: "1px solid oklch(0.85 0.19 145 / 0.7)",
                pointerEvents: "none",
              }}
            />
            <span
              style={{
                position: "absolute",
                top: 12,
                right: 12,
                width: 12,
                height: 12,
                borderTop: "1px solid oklch(0.85 0.19 145 / 0.7)",
                borderRight: "1px solid oklch(0.85 0.19 145 / 0.7)",
                pointerEvents: "none",
              }}
            />
            <span
              style={{
                position: "absolute",
                bottom: 12,
                left: 12,
                width: 12,
                height: 12,
                borderBottom: "1px solid oklch(0.85 0.19 145 / 0.7)",
                borderLeft: "1px solid oklch(0.85 0.19 145 / 0.7)",
                pointerEvents: "none",
              }}
            />
            <span
              style={{
                position: "absolute",
                bottom: 12,
                right: 12,
                width: 12,
                height: 12,
                borderBottom: "1px solid oklch(0.85 0.19 145 / 0.7)",
                borderRight: "1px solid oklch(0.85 0.19 145 / 0.7)",
                pointerEvents: "none",
              }}
            />
          </div>
          <div
            style={{
              borderTop: "1px solid oklch(0.32 0.03 190 / 0.35)",
              padding: 11,
              display: "flex",
              flexDirection: "column",
              gap: 4,
              fontSize: 10.5,
              lineHeight: 1.6,
            }}
          >
            <div style={{ color: "oklch(0.88 0.02 160)" }}>Aneesh Jayan Prabhu</div>
            <div style={{ color: "oklch(0.6 0.02 200)" }}>Tempe, AZ · ASU '26</div>
            <div style={{ color: ACCENT }}>revmo-ai · forward-deployed</div>
          </div>
        </div>

        <div style={{ flex: "1 1 420px", minWidth: 320 }}>
          <h1
            style={{
              margin: 0,
              fontSize: "clamp(28px, 3.4vw, 44px)",
              fontWeight: 700,
              lineHeight: 1.1,
              letterSpacing: "-0.02em",
              color: "oklch(0.95 0.02 160)",
            }}
          >
            Building <span style={{ color: ACCENT }}>agentic AI</span>
            <br />
            that survives
            <br />
            production.
          </h1>
          <p
            style={{
              margin: "18px 0 0",
              maxWidth: 560,
              fontSize: 13.5,
              lineHeight: 1.85,
              color: "oklch(0.68 0.02 200)",
            }}
          >
            Forward-deployed AI engineer at Revmo AI. I sit with clients, translate their pain into
            specs, and ship multi-agent systems end to end — orchestration, inference optimization,
            evaluation, CI/CD, the whole path to production.
          </p>

          <div
            style={{
              marginTop: 22,
              border: "1px solid oklch(0.32 0.03 190 / 0.4)",
              background: "oklch(0.09 0.012 235 / 0.8)",
            }}
          >
            <div
              style={{
                borderBottom: "1px solid oklch(0.32 0.03 190 / 0.3)",
                padding: "7px 12px",
                fontSize: 10.5,
                color: "oklch(0.5 0.02 200)",
              }}
            >
              ~/boot.log
            </div>
            <pre
              style={{
                margin: 0,
                padding: 14,
                fontSize: 11.5,
                lineHeight: 1.9,
                color: "oklch(0.72 0.02 200)",
                whiteSpace: "pre-wrap",
              }}
            >
              <span style={{ color: ACCENT }}>[ ok ]</span> mounted /experience revmo-ai ·
              wolters-kluwer · vit{"\n"}
              <span style={{ color: ACCENT }}>[ ok ]</span> loaded /projects 11 systems · 5
              categories{"\n"}
              <span style={{ color: ACCENT }}>[ ok ]</span> served 8+ agentic systems 50,000+
              calls/mo{"\n"}
              <span style={{ color: ACCENT }}>[ ok ]</span> vllm vs sglang bench 3.2x throughput
              850→240ms{"\n"}
              <span style={{ color: ACCENT }}>[ ok ]</span> quantized qwen2.5-7b int8/fp16 −47% cost
              {"\n"}
              <span style={{ color: "oklch(0.85 0.16 85)" }}>[warn]</span> inbox unread —{" "}
              <span style={{ color: "oklch(0.82 0.15 200)" }}>say hi below</span>
              <span
                className="animate-blink"
                style={{
                  display: "inline-block",
                  width: 7,
                  height: 12,
                  background: ACCENT,
                  transform: "translateY(2px)",
                }}
              />
            </pre>
          </div>

          <div
            style={{
              marginTop: 20,
              display: "grid",
              gridTemplateColumns: "repeat(auto-fit, minmax(126px, 1fr))",
              gap: 8,
            }}
          >
            {HEADLINE_METRICS.map((m) => (
              <div
                key={m.label}
                style={{
                  border: "1px solid oklch(0.32 0.03 190 / 0.4)",
                  background: "oklch(0.09 0.012 235 / 0.6)",
                  padding: 12,
                }}
              >
                <div style={{ fontSize: 20, fontWeight: 700, color: m.color }}>{m.value}</div>
                <div
                  style={{
                    marginTop: 3,
                    fontSize: 10,
                    lineHeight: 1.4,
                    color: "oklch(0.58 0.02 200)",
                  }}
                >
                  {m.label}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div style={{ marginTop: 34 }}>
        <p style={{ margin: "0 0 10px", fontSize: 12, color: "oklch(0.5 0.02 200)" }}>
          <span style={{ color: ACCENT }}>$</span> render --stack-sphere --rotate
        </p>
        <div
          style={{
            border: "1px solid oklch(0.32 0.03 190 / 0.4)",
            background: "oklch(0.075 0.012 235 / 0.7)",
            position: "relative",
          }}
        >
          <div style={{ width: "100%", height: 420 }}>
            <Sphere3D />
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
            drag to rotate · 3D projection of the stack
          </span>
        </div>
      </div>
    </div>
  );
}
