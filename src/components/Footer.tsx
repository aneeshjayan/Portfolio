import { Github, Linkedin, Mail } from "lucide-react";

export function Footer() {
  return (
    <footer className="mt-24 border-t border-border/60">
      <div className="mx-auto flex max-w-6xl flex-col items-start justify-between gap-4 px-6 py-8 sm:flex-row sm:items-center">
        <p className="font-mono text-xs text-muted-foreground">
          © {new Date().getFullYear()} Aneesh Jayan Prabhu · built with React + TanStack
        </p>
        <div className="flex items-center gap-4">
          <a
            href="https://github.com/aneeshjayan"
            target="_blank"
            rel="noreferrer"
            aria-label="GitHub"
            className="text-muted-foreground transition-colors hover:text-foreground"
          >
            <Github className="size-4" />
          </a>
          <a
            href="https://www.linkedin.com/in/aneeshjayan/"
            target="_blank"
            rel="noreferrer"
            aria-label="LinkedIn"
            className="text-muted-foreground transition-colors hover:text-foreground"
          >
            <Linkedin className="size-4" />
          </a>
          <a
            href="mailto:aneeshjayan9@gmail.com"
            aria-label="Email"
            className="text-muted-foreground transition-colors hover:text-foreground"
          >
            <Mail className="size-4" />
          </a>
        </div>
      </div>
    </footer>
  );
}
