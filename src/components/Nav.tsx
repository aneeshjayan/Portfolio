import { Link } from "@tanstack/react-router";
import { useState } from "react";
import { Menu, X } from "lucide-react";

const links = [
  { to: "/about", label: "about" },
  { to: "/projects", label: "projects" },
  { to: "/experience", label: "experience" },
  { to: "/skills", label: "skills" },
  { to: "/contact", label: "contact" },
] as const;

export function Nav() {
  const [open, setOpen] = useState(false);

  return (
    <header className="sticky top-0 z-40 border-b border-border/60 bg-background/70 backdrop-blur-xl">
      <nav className="mx-auto flex max-w-6xl items-center justify-between px-6 py-4">
        <Link
          to="/"
          className="font-mono text-sm font-semibold tracking-tight"
          onClick={() => setOpen(false)}
        >
          <span className="text-muted-foreground">~/</span>
          <span className="text-gradient">aneeshjayan</span>
          <span className="ml-1 inline-block h-3 w-1.5 translate-y-0.5 bg-primary animate-blink" />
        </Link>

        {/* Desktop */}
        <ul className="hidden items-center gap-1 md:flex">
          {links.map((l) => (
            <li key={l.to}>
              <Link
                to={l.to}
                className="rounded-md px-3 py-1.5 font-mono text-sm text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
                activeProps={{ className: "text-foreground bg-secondary" }}
              >
                {l.label}
              </Link>
            </li>
          ))}
        </ul>

        <button
          aria-label="Toggle menu"
          className="rounded-md p-2 text-muted-foreground hover:bg-secondary hover:text-foreground md:hidden"
          onClick={() => setOpen((v) => !v)}
        >
          {open ? <X className="size-5" /> : <Menu className="size-5" />}
        </button>
      </nav>

      {open && (
        <div className="border-t border-border/60 md:hidden">
          <ul className="mx-auto flex max-w-6xl flex-col px-6 py-2">
            {links.map((l) => (
              <li key={l.to}>
                <Link
                  to={l.to}
                  onClick={() => setOpen(false)}
                  className="block rounded-md px-3 py-2 font-mono text-sm text-muted-foreground hover:bg-secondary hover:text-foreground"
                  activeProps={{ className: "text-foreground" }}
                >
                  {l.label}
                </Link>
              </li>
            ))}
          </ul>
        </div>
      )}
    </header>
  );
}
