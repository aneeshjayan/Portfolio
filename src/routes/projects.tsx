import { createFileRoute } from "@tanstack/react-router";
import { Github, ExternalLink } from "lucide-react";
import { PageHeader } from "@/components/SectionTag";

export const Route = createFileRoute("/projects")({
  head: () => ({
    meta: [
      { title: "Projects — Aneesh Jayan Prabhu" },
      {
        name: "description",
        content:
          "Selected AI/ML projects: Optimal-SLM (small language model RL), FraudGNN (graph fraud detection), TrustMedAI (trust-calibrated medical AI).",
      },
      { property: "og:title", content: "Projects — Aneesh Jayan Prabhu" },
      {
        property: "og:description",
        content:
          "Optimal-SLM, FraudGNN, TrustMedAI — selected work in LLMs, GNNs, and trustworthy AI.",
      },
    ],
  }),
  component: ProjectsPage,
});

type Project = {
  name: string;
  blurb: string;
  description: string;
  tech: string[];
  metrics: { label: string; value: string }[];
  github?: string;
  link?: string;
};

const projects: Project[] = [
  {
    name: "Optimal-SLM",
    blurb: "RL-tuned small language model for reasoning",
    description:
      "Trained a compact (<1B param) language model with reinforcement learning from preference data to match much larger models on multi-step reasoning benchmarks. Custom reward modeling, PPO loop, and evaluation harness.",
    tech: ["PyTorch", "TRL", "PPO", "LoRA", "vLLM", "WandB"],
    metrics: [
      { label: "Params", value: "<1B" },
      { label: "Reasoning lift", value: "+18%" },
      { label: "Inference", value: "vLLM" },
    ],
    github: "https://github.com/aneeshjayan",
  },
  {
    name: "FraudGNN",
    blurb: "Graph neural network for transaction fraud",
    description:
      "Built a heterogeneous graph neural network over millions of card-merchant-device edges to detect fraud rings invisible to tabular models. Productionized with mini-batch neighbor sampling and online inference.",
    tech: ["PyTorch Geometric", "GraphSAGE", "Neo4j", "FastAPI", "Docker"],
    metrics: [
      { label: "F1", value: "0.91" },
      { label: "Recall@1%", value: "0.84" },
      { label: "Latency", value: "<60ms" },
    ],
    github: "https://github.com/aneeshjayan",
  },
  {
    name: "TrustMedAI",
    blurb: "Calibrated diagnostic assistant for clinicians",
    description:
      "Multimodal medical AI assistant that grounds answers in retrieved clinical literature and reports calibrated uncertainty alongside every prediction. Designed with clinician-in-the-loop evaluation.",
    tech: ["LangChain", "RAG", "Llama 3", "FAISS", "Streamlit", "AWS"],
    metrics: [
      { label: "ECE", value: "0.04" },
      { label: "Hallucination ↓", value: "-37%" },
      { label: "Sources", value: "PubMed" },
    ],
    github: "https://github.com/aneeshjayan",
  },
];

function ProjectsPage() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-16">
      <PageHeader
        tag="projects"
        title="Selected Work"
        description="A few systems I've designed, trained, and shipped. Each is a real attempt at solving a hard problem — not a tutorial rebuild."
      />

      <div className="mt-12 grid gap-6 md:grid-cols-2 lg:grid-cols-3">
        {projects.map((p, i) => (
          <article
            key={p.name}
            className="group relative flex flex-col overflow-hidden rounded-xl border border-border bg-card/60 p-6 backdrop-blur transition-all duration-300 hover:-translate-y-1 hover:border-primary/40 hover:bg-card/80 card-elevated animate-fade-in-up"
            style={{ animationDelay: `${i * 80}ms` }}
          >
            <div className="mb-3 flex items-center justify-between">
              <span className="font-mono text-xs text-muted-foreground">
                0{i + 1} / 0{projects.length}
              </span>
              <div className="flex items-center gap-2">
                {p.github && (
                  <a
                    href={p.github}
                    target="_blank"
                    rel="noreferrer"
                    aria-label={`${p.name} on GitHub`}
                    className="text-muted-foreground transition-colors hover:text-foreground"
                  >
                    <Github className="size-4" />
                  </a>
                )}
                {p.link && (
                  <a
                    href={p.link}
                    target="_blank"
                    rel="noreferrer"
                    aria-label={`${p.name} live`}
                    className="text-muted-foreground transition-colors hover:text-foreground"
                  >
                    <ExternalLink className="size-4" />
                  </a>
                )}
              </div>
            </div>

            <h2 className="text-xl font-semibold tracking-tight">
              <span className="text-gradient">{p.name}</span>
            </h2>
            <p className="mt-1 font-mono text-xs text-muted-foreground">{p.blurb}</p>

            <p className="mt-4 text-sm leading-relaxed text-muted-foreground">
              {p.description}
            </p>

            <div className="mt-5 grid grid-cols-3 gap-2">
              {p.metrics.map((m) => (
                <div
                  key={m.label}
                  className="rounded-md border border-border/60 bg-background/40 p-2 text-center"
                >
                  <div className="font-mono text-sm font-semibold text-foreground">
                    {m.value}
                  </div>
                  <div className="mt-0.5 text-[10px] uppercase tracking-wider text-muted-foreground">
                    {m.label}
                  </div>
                </div>
              ))}
            </div>

            <div className="mt-5 flex flex-wrap gap-1.5">
              {p.tech.map((t) => (
                <span
                  key={t}
                  className="rounded-md border border-border/60 bg-secondary/50 px-2 py-0.5 font-mono text-[11px] text-muted-foreground"
                >
                  {t}
                </span>
              ))}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
}
