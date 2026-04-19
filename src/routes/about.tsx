import { createFileRoute } from "@tanstack/react-router";
import { SectionTag } from "@/components/SectionTag";

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
      {
        property: "og:description",
        content: "Bio, focus areas, and stats for Aneesh Jayan Prabhu.",
      },
    ],
  }),
  component: AboutPage,
});

const stats = [
  { label: "Models trained", value: "30+" },
  { label: "Production systems", value: "5" },
  { label: "Research papers", value: "2" },
  { label: "Best F1 score", value: "0.91" },
];

const openRoles = [
  "ML Engineer",
  "Data Scientist",
  "AI Engineer",
  "Forward Deployed Engineer",
  "Software Engineer",
  "AI/ML Research Engineer",
];

const interests = [
  { emoji: "⚖️", title: "AI for Legal & Finance", desc: "Applying LLMs and agentic systems to legal-tech and fintech to make AI accessible to all." },
  { emoji: "🔐", title: "Agent Security & Infrastructure", desc: "Building secure, scalable infrastructure for agentic systems — guardrails, injection defense, and audit pipelines." },
  { emoji: "⚡", title: "Optimal & Scalable Systems", desc: "Engineering inference pipelines and ML systems optimized for latency, cost, and real-world reliability." },
];

function AboutPage() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-16">

      {/* ── Profile hero ── */}
      <div className="mb-16 flex flex-col items-center gap-5 text-center animate-fade-in-up">
        <div className="relative">
          <div
            className="h-40 w-40 overflow-hidden rounded-full border-2 border-primary/50"
            style={{ boxShadow: "0 0 48px oklch(0.82 0.15 200 / 0.3), 0 0 0 6px oklch(0.82 0.15 200 / 0.08)" }}
          >
            <img
              src="/profile.png"
              alt="Aneesh Jayan Prabhu"
              className="h-full w-full object-cover"
            />
          </div>
          <div className="absolute -bottom-1 left-1/2 -translate-x-1/2 flex items-center gap-1.5 whitespace-nowrap rounded-full border border-border bg-card/90 px-3 py-1 font-mono text-[10px] text-emerald-400 backdrop-blur">
            <span className="size-1.5 rounded-full bg-emerald-400 animate-pulse-dot" />
            open to new roles
          </div>
        </div>

        <div>
          <h1 className="text-4xl font-bold tracking-tight text-gradient">Aneesh Jayan Prabhu</h1>
          <p className="mt-2 font-mono text-sm text-muted-foreground">
            AI/ML Engineer · MS Data Science, Analytics & Engineering @ Arizona State University
          </p>
        </div>

        <div className="flex flex-wrap justify-center gap-2">
          {["LLMs", "RAG", "Agents", "GNNs", "RL / RLHF", "Production ML", "AI Security"].map((tag) => (
            <span
              key={tag}
              className="rounded-full border border-primary/30 bg-primary/10 px-3 py-0.5 font-mono text-xs text-primary"
            >
              {tag}
            </span>
          ))}
        </div>
      </div>

      {/* ── Bio + Terminal ── */}
      <div className="grid gap-10 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <SectionTag>bio</SectionTag>
          <div className="space-y-4 text-base leading-relaxed text-muted-foreground">
            <p>
              I build at the intersection of{" "}
              <span className="text-foreground font-medium">large language models</span>,{" "}
              <span className="text-foreground font-medium">agentic AI systems</span>, and{" "}
              <span className="text-foreground font-medium">production ML infrastructure</span>.
              My work spans multi-agent pipelines with LangGraph, retrieval-augmented generation,
              graph neural networks for fraud detection, and AI security middleware for voice agents.
            </p>
            <p>
              At <span className="text-foreground font-medium">Wolters Kluwer's Legal & Regulatory Division</span>{" "}
              I shipped an agentic AI reporting system, RAG pipelines for legal documents, and FastAPI
              microservices into enterprise-grade products. Before that, at{" "}
              <span className="text-foreground font-medium">VIT's Centre for Cyber-Physical Systems</span>{" "}
              I led research on hybrid deep learning–quantum models for biomedical diagnostics, achieving
              98.17% accuracy on ABIDE I fMRI autism detection datasets.
            </p>
            <p>
              I'm deeply interested in{" "}
              <span className="text-foreground font-medium">applying AI to finance and legal sectors</span>{" "}
              — making intelligent systems accessible to everyone, not just those with technical resources.
              I care about building{" "}
              <span className="text-foreground font-medium">infrastructure for agents that is secure, scalable, and optimal</span>{" "}
              — from guardrails and injection defense to low-latency inference pipelines that survive production.
            </p>
          </div>

          {/* Quick-facts */}
          <div className="mt-8 grid grid-cols-2 gap-3">
            {[
              { label: "Location", value: "Tempe, AZ" },
              { label: "Education", value: "MS Data Science · ASU" },
              { label: "Focus", value: "LLMs, Agents, RAG, RL" },
              { label: "Status", value: "Open to roles · 2025–26" },
            ].map((f) => (
              <div
                key={f.label}
                className="rounded-lg border border-border/60 bg-card/40 px-4 py-3 backdrop-blur"
              >
                <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{f.label}</div>
                <div className="mt-0.5 text-sm font-medium text-foreground">{f.value}</div>
              </div>
            ))}
          </div>
        </div>

        <div className="lg:col-span-2">
          <SectionTag>terminal</SectionTag>
          <div className="overflow-hidden rounded-xl border border-border bg-card/70 backdrop-blur card-elevated">
            <div className="flex items-center gap-2 border-b border-border bg-secondary/40 px-4 py-2.5">
              <span className="size-2.5 rounded-full bg-red-500/70" />
              <span className="size-2.5 rounded-full bg-yellow-500/70" />
              <span className="size-2.5 rounded-full bg-green-500/70" />
              <span className="ml-2 font-mono text-xs text-muted-foreground">~/aneesh — zsh</span>
            </div>
            <pre className="overflow-x-auto p-5 font-mono text-[12.5px] leading-relaxed">
              <span className="text-primary">$</span> whoami{"\n"}
              aneesh — ai/ml engineer{"\n\n"}
              <span className="text-primary">$</span> cat stack.txt{"\n"}
              python · pytorch · langchain{"\n"}
              langgraph · rag · agents{"\n"}
              gnns · rl · cuda · fastapi{"\n\n"}
              <span className="text-primary">$</span> cat open_to.txt{"\n"}
              <span className="text-emerald-400">ML Engineer</span>{"\n"}
              <span className="text-emerald-400">Data Scientist</span>{"\n"}
              <span className="text-emerald-400">AI Engineer</span>{"\n"}
              <span className="text-emerald-400">Fwd Deployed Engineer</span>{"\n"}
              <span className="text-emerald-400">Software Engineer (AI)</span>{"\n\n"}
              <span className="text-primary">$</span> echo $STATUS{"\n"}
              <span className="text-emerald-400">building & shipping</span>
              <span className="ml-1 inline-block h-3 w-2 translate-y-0.5 bg-primary animate-blink" />
            </pre>
          </div>
        </div>
      </div>

      {/* ── Interests ── */}
      <div className="mt-16">
        <SectionTag>interests & mission</SectionTag>
        <div className="grid gap-4 sm:grid-cols-3">
          {interests.map((item, i) => (
            <div
              key={item.title}
              className="rounded-xl border border-border bg-card/50 p-5 backdrop-blur card-elevated animate-fade-in-up hover:-translate-y-0.5 transition-transform duration-300"
              style={{ animationDelay: `${i * 80}ms` }}
            >
              <div className="mb-3 text-2xl">{item.emoji}</div>
              <div className="mb-1 font-semibold text-foreground">{item.title}</div>
              <div className="text-sm leading-relaxed text-muted-foreground">{item.desc}</div>
            </div>
          ))}
        </div>
      </div>

      {/* ── Open to roles ── */}
      <div className="mt-12">
        <SectionTag>open to</SectionTag>
        <div className="flex flex-wrap gap-2">
          {openRoles.map((role) => (
            <span
              key={role}
              className="rounded-lg border border-primary/30 bg-primary/10 px-4 py-2 font-mono text-sm text-primary hover:border-primary/60 hover:bg-primary/20 transition-colors cursor-default"
            >
              {role}
            </span>
          ))}
        </div>
        <p className="mt-3 text-sm text-muted-foreground">
          Graduating May 2026 · Open to full-time positions and co-ops in the US.
        </p>
      </div>

      {/* ── Stats ── */}
      <div className="mt-14">
        <SectionTag>by the numbers</SectionTag>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {stats.map((s, i) => (
            <div
              key={s.label}
              className="rounded-xl border border-border bg-card/50 p-6 backdrop-blur transition-all duration-300 hover:-translate-y-1 hover:border-primary/40 card-elevated animate-fade-in-up"
              style={{ animationDelay: `${i * 80}ms` }}
            >
              <div className="font-mono text-4xl font-semibold text-gradient">{s.value}</div>
              <div className="mt-2 text-sm text-muted-foreground">{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
