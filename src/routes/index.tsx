import { createFileRoute, Link } from "@tanstack/react-router";
import { ArrowRight, Github, Linkedin, ChevronDown } from "lucide-react";

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
        content:
          "AI/ML engineer building agentic systems, RAG pipelines, and production ML.",
      },
    ],
  }),
  component: HomePage,
});

function HomePage() {
  return (
    <section className="relative">
      <div className="mx-auto flex min-h-[calc(100vh-65px)] max-w-6xl flex-col justify-center px-6 py-20">
        <div className="animate-fade-in-up">
          <div className="mb-6 inline-flex items-center gap-2 rounded-full border border-border bg-card/50 px-3 py-1 backdrop-blur">
            <span className="relative flex h-2 w-2">
              <span className="absolute inline-flex h-full w-full animate-pulse-dot rounded-full bg-emerald-400 opacity-75" />
              <span className="relative inline-flex h-2 w-2 rounded-full bg-emerald-400" />
            </span>
            <span className="font-mono text-xs text-muted-foreground">
              Available for new projects
            </span>
          </div>

          <p className="mb-3 font-mono text-sm text-muted-foreground">
            <span className="text-primary">$</span> whoami
          </p>
          <h1 className="text-5xl font-semibold leading-[1.05] tracking-tight sm:text-7xl lg:text-8xl">
            Aneesh{" "}
            <span className="text-gradient">Jayan Prabhu</span>
          </h1>

          <p className="mt-6 max-w-2xl text-lg text-muted-foreground sm:text-xl">
            AI/ML engineer building{" "}
            <span className="text-foreground">agentic systems</span>,{" "}
            <span className="text-foreground">RAG pipelines</span>, and{" "}
            <span className="text-foreground">production ML</span> — from research to
            shipped products.
          </p>

          <div className="mt-10 flex flex-wrap items-center gap-3">
            <Link
              to="/projects"
              className="group inline-flex items-center gap-2 rounded-lg bg-gradient-brand px-5 py-3 text-sm font-medium text-background transition-transform hover:-translate-y-0.5 glow"
            >
              View Projects
              <ArrowRight className="size-4 transition-transform group-hover:translate-x-0.5" />
            </Link>
            <Link
              to="/contact"
              className="inline-flex items-center gap-2 rounded-lg border border-border bg-card/50 px-5 py-3 text-sm font-medium backdrop-blur transition-colors hover:bg-secondary"
            >
              Get in Touch
            </Link>
            <div className="ml-2 flex items-center gap-3">
              <a
                href="https://github.com/aneeshjayan"
                target="_blank"
                rel="noreferrer"
                aria-label="GitHub"
                className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              >
                <Github className="size-5" />
              </a>
              <a
                href="https://www.linkedin.com/in/aneeshjayan/"
                target="_blank"
                rel="noreferrer"
                aria-label="LinkedIn"
                className="rounded-md p-2 text-muted-foreground transition-colors hover:bg-secondary hover:text-foreground"
              >
                <Linkedin className="size-5" />
              </a>
            </div>
          </div>

          {/* Code preview card */}
          <div className="mt-16 max-w-2xl overflow-hidden rounded-xl border border-border bg-card/70 backdrop-blur card-elevated">
            <div className="flex items-center gap-2 border-b border-border bg-secondary/40 px-4 py-2.5">
              <span className="size-2.5 rounded-full bg-red-500/70" />
              <span className="size-2.5 rounded-full bg-yellow-500/70" />
              <span className="size-2.5 rounded-full bg-green-500/70" />
              <span className="ml-2 font-mono text-xs text-muted-foreground">
                ~/portfolio — zsh
              </span>
            </div>
            <pre className="overflow-x-auto p-5 font-mono text-[13px] leading-relaxed">
{`> import { Aneesh } from "@/engineer";

> const me = new Aneesh({
    role:    "AI/ML Engineer",
    focus:   ["LLMs", "RAG", "Agents", "GNNs"],
    edu:     "MS Data Science · ASU",
    located: "Tempe, AZ",
  });

> me.ship();`}
              <span className="ml-1 inline-block h-3.5 w-2 translate-y-0.5 bg-primary animate-blink" />
            </pre>
          </div>
        </div>

        <div className="mt-16 flex justify-center">
          <ChevronDown className="size-5 animate-float text-muted-foreground" />
        </div>
      </div>
    </section>
  );
}
