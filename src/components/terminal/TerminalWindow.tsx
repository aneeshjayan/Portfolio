import { Outlet, useRouterState } from "@tanstack/react-router";
import { Sidebar } from "@/components/terminal/Sidebar";
import { useClock } from "@/hooks/useClock";
import "@/styles/terminal.css";

export function TerminalWindow() {
  const pathname = useRouterState({ select: (s) => s.location.pathname });
  const clock = useClock();
  const routeSlug = pathname === "/" ? "" : pathname.slice(1);

  return (
    <div
      style={{
        minHeight: "100vh",
        background:
          "radial-gradient(ellipse at 20% 0%, oklch(0.19 0.03 200 / 0.55), transparent 55%), radial-gradient(ellipse at 90% 100%, oklch(0.2 0.04 320 / 0.4), transparent 55%), #04070a",
        color: "oklch(0.86 0.02 160)",
        fontFamily: "'JetBrains Mono', ui-monospace, monospace",
        padding: 18,
        boxSizing: "border-box",
        position: "relative",
      }}
    >
      <div
        aria-hidden="true"
        style={{
          position: "fixed",
          inset: 0,
          zIndex: 60,
          pointerEvents: "none",
          backgroundImage:
            "repeating-linear-gradient(oklch(1 0 0 / 0.05) 0 1px, transparent 1px 3px)",
          animation: "dc-flicker 4s ease-in-out infinite",
        }}
      />

      <div
        style={{
          position: "relative",
          margin: "0 auto",
          maxWidth: 1440,
          border: "1px solid oklch(0.32 0.03 190 / 0.5)",
          borderRadius: 10,
          overflow: "hidden",
          background: "oklch(0.115 0.012 235 / 0.92)",
          boxShadow:
            "0 0 0 1px oklch(0.85 0.19 145 / 0.06), 0 40px 90px -40px #000, inset 0 1px 0 oklch(1 0 0 / 0.04)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            borderBottom: "1px solid oklch(0.32 0.03 190 / 0.45)",
            background: "linear-gradient(oklch(0.2 0.015 235), oklch(0.16 0.012 235))",
            padding: "9px 14px",
          }}
        >
          <span style={{ height: 11, width: 11, borderRadius: 999, background: "#ff5f57" }} />
          <span style={{ height: 11, width: 11, borderRadius: 999, background: "#febc2e" }} />
          <span style={{ height: 11, width: 11, borderRadius: 999, background: "#28c840" }} />
          <span style={{ marginLeft: 10, fontSize: 12, color: "oklch(0.65 0.02 200)" }}>
            aneesh@portfolio: ~/{routeSlug} — zsh — 148×44
          </span>
          <span style={{ marginLeft: "auto", fontSize: 11, color: "oklch(0.55 0.02 200)" }}>
            {clock}
          </span>
        </div>

        <div className="term-shell-grid">
          <Sidebar />
          <main id="term-main" style={{ padding: "28px 32px 40px", minWidth: 0 }}>
            <Outlet />
          </main>
        </div>

        <div
          style={{
            borderTop: "1px solid oklch(0.32 0.03 190 / 0.4)",
            background: "oklch(0.14 0.012 235 / 0.8)",
            padding: "7px 14px",
            display: "flex",
            flexWrap: "wrap",
            gap: 14,
            fontSize: 10.5,
            color: "oklch(0.5 0.02 200)",
          }}
        >
          <span style={{ color: "oklch(0.85 0.19 145)" }}>● ready</span>
          <span>~/{routeSlug}</span>
          <span style={{ marginLeft: "auto" }}>d3 v7 · force + treemap + 3d projection</span>
        </div>
      </div>
    </div>
  );
}
