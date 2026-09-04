import { Link, createRootRoute } from "@tanstack/react-router";
import { TerminalWindow } from "@/components/terminal/TerminalWindow";

function NotFoundComponent() {
  return (
    <div style={{ animation: "dc-boot-in 0.4s ease-out both" }}>
      <p style={{ margin: 0, fontSize: 12, color: "oklch(0.5 0.02 200)" }}>
        <span style={{ color: "oklch(0.85 0.19 145)" }}>$</span> cat ~/404
      </p>
      <h1
        style={{
          margin: "14px 0 4px",
          fontSize: 26,
          fontWeight: 700,
          color: "oklch(0.94 0.02 160)",
        }}
      >
        404: not found
      </h1>
      <p style={{ margin: 0, fontSize: 12.5, color: "oklch(0.6 0.02 200)" }}>
        That path doesn't exist on this filesystem.
      </p>
      <Link
        to="/"
        style={{
          marginTop: 20,
          display: "inline-block",
          border: "1px solid oklch(0.85 0.19 145 / 0.4)",
          background: "oklch(0.85 0.19 145 / 0.1)",
          padding: "8px 14px",
          fontFamily: "'JetBrains Mono', monospace",
          fontSize: 12.5,
          color: "oklch(0.85 0.19 145)",
          textDecoration: "none",
        }}
      >
        cd ~/
      </Link>
    </div>
  );
}

export const Route = createRootRoute({
  component: TerminalWindow,
  notFoundComponent: NotFoundComponent,
});
