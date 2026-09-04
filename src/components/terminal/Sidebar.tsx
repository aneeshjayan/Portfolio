import { Link, useRouterState } from "@tanstack/react-router";
import { EMAIL, GITHUB_URL, LINKEDIN_URL, PROJECTS, RESUME_URL } from "@/data/portfolio";

const ROUTES = [
  { to: "/" as const, label: "index.sh", count: "", branch: "├─", glyph: "▚" },
  { to: "/about" as const, label: "about.md", count: "", branch: "├─", glyph: "❯" },
  {
    to: "/projects" as const,
    label: "projects/",
    count: String(PROJECTS.length),
    branch: "├─",
    glyph: "◈",
  },
  { to: "/experience" as const, label: "experience/", count: "3", branch: "├─", glyph: "▤" },
  { to: "/skills" as const, label: "stack/", count: "9", branch: "├─", glyph: "▦" },
  { to: "/contact" as const, label: "contact.sh", count: "", branch: "└─", glyph: "✉" },
];

const ACCENT = "oklch(0.85 0.19 145)";

export function Sidebar() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });

  return (
    <aside
      style={{
        borderRight: "1px solid oklch(0.32 0.03 190 / 0.35)",
        background: "oklch(0.1 0.012 235 / 0.7)",
        padding: "20px 16px",
        display: "flex",
        flexDirection: "column",
        gap: 22,
      }}
    >
      <div>
        <p style={{ margin: 0, fontSize: 11, color: "oklch(0.5 0.02 200)" }}>
          aneesh@portfolio:~$ whoami
        </p>
        <p
          style={{
            margin: "6px 0 0",
            fontSize: 15,
            fontWeight: 700,
            color: "oklch(0.92 0.02 160)",
          }}
        >
          Aneesh Jayan Prabhu
        </p>
        <p
          style={{ margin: "3px 0 0", fontSize: 11, lineHeight: 1.6, color: "oklch(0.6 0.02 200)" }}
        >
          AI Software Engineer
          <br />
          ML · Data Science · Data Eng
        </p>
        <div
          style={{
            marginTop: 10,
            display: "inline-flex",
            alignItems: "center",
            gap: 6,
            border: `1px solid ${ACCENT.replace(")", " / 0.35)")}`,
            background: `${ACCENT.replace(")", " / 0.08)")}`,
            padding: "3px 8px",
            fontSize: 10,
            color: ACCENT,
          }}
        >
          <span
            style={{
              height: 6,
              width: 6,
              borderRadius: 999,
              background: ACCENT,
              animation: "dc-glow-pulse 2s ease-in-out infinite",
            }}
          />
          available for new roles
        </div>
      </div>

      <nav>
        <p style={{ margin: "0 0 8px", fontSize: 11, color: "oklch(0.5 0.02 200)" }}>
          $ tree ~/portfolio
        </p>
        <ul
          style={{
            listStyle: "none",
            margin: 0,
            padding: 0,
            display: "flex",
            flexDirection: "column",
            gap: 1,
          }}
        >
          {ROUTES.map((r) => {
            const on = r.to === "/" ? pathname === "/" : pathname.startsWith(r.to);
            return (
              <li key={r.to}>
                <Link
                  to={r.to}
                  style={{
                    width: "100%",
                    display: "flex",
                    alignItems: "center",
                    gap: 8,
                    border: "none",
                    borderLeft: `2px solid ${on ? ACCENT : "transparent"}`,
                    background: on ? "oklch(0.19 0.02 200 / 0.8)" : "transparent",
                    padding: "7px 10px",
                    fontFamily: "'JetBrains Mono', monospace",
                    fontSize: 12.5,
                    color: on ? "oklch(0.92 0.02 160)" : "oklch(0.66 0.02 200)",
                    cursor: "pointer",
                    textAlign: "left",
                    textDecoration: "none",
                  }}
                >
                  <span style={{ color: "oklch(0.42 0.02 200)", fontSize: 11 }}>{r.branch}</span>
                  <span
                    style={{ color: on ? ACCENT : "oklch(0.42 0.02 200)", fontSize: 11, width: 12 }}
                  >
                    {r.glyph}
                  </span>
                  <span>{r.label}</span>
                  <span style={{ marginLeft: "auto", fontSize: 10, color: "oklch(0.42 0.02 200)" }}>
                    {r.count}
                  </span>
                </Link>
              </li>
            );
          })}
        </ul>
      </nav>

      <div>
        <p style={{ margin: "0 0 8px", fontSize: 11, color: "oklch(0.5 0.02 200)" }}>
          $ ls ~/links
        </p>
        <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
          <a
            href={RESUME_URL}
            target="_blank"
            rel="noreferrer"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              border: `1px solid ${ACCENT.replace(")", " / 0.3)")}`,
              background: `${ACCENT.replace(")", " / 0.07)")}`,
              padding: "8px 10px",
              fontSize: 11.5,
              color: ACCENT,
              textDecoration: "none",
            }}
          >
            resume.pdf<span>↓</span>
          </a>
          <a
            href={GITHUB_URL}
            target="_blank"
            rel="noreferrer"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              border: "1px solid oklch(0.32 0.03 190 / 0.5)",
              padding: "8px 10px",
              fontSize: 11.5,
              color: "oklch(0.75 0.02 200)",
              textDecoration: "none",
            }}
          >
            github<span>↗</span>
          </a>
          <a
            href={LINKEDIN_URL}
            target="_blank"
            rel="noreferrer"
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              border: "1px solid oklch(0.32 0.03 190 / 0.5)",
              padding: "8px 10px",
              fontSize: 11.5,
              color: "oklch(0.75 0.02 200)",
              textDecoration: "none",
            }}
          >
            linkedin<span>↗</span>
          </a>
          <a
            href={`mailto:${EMAIL}`}
            style={{
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between",
              border: "1px solid oklch(0.32 0.03 190 / 0.5)",
              padding: "8px 10px",
              fontSize: 11.5,
              color: "oklch(0.75 0.02 200)",
              textDecoration: "none",
            }}
          >
            email<span>↗</span>
          </a>
        </div>
      </div>

      <div
        style={{
          marginTop: "auto",
          borderTop: "1px solid oklch(0.32 0.03 190 / 0.3)",
          paddingTop: 14,
          fontSize: 10.5,
          lineHeight: 1.8,
          color: "oklch(0.5 0.02 200)",
        }}
      >
        <div>MS Data Science · ASU '26</div>
        <div>Tempe, Arizona · (602) 768-6622</div>
        <div style={{ color: ACCENT }}>uptime 99.2% · 50k calls/mo</div>
      </div>
    </aside>
  );
}
