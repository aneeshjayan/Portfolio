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
          "Selected AI/ML projects: Optimal-SLM, FraudGNN, TrustMedAI, VoiceGuardAI (RL voice deepfake detection), Insurance SLM Agent (multi-agent RLHF pipeline).",
      },
      { property: "og:title", content: "Projects — Aneesh Jayan Prabhu" },
      {
        property: "og:description",
        content:
          "Optimal-SLM, FraudGNN, TrustMedAI, VoiceGuardAI, Insurance SLM Agent — selected work in LLMs, GNNs, RL, and trustworthy AI.",
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
  {
    name: "VLM Speedup: LexFin Guard",
    blurb: "Vision-language model acceleration for financial docs",
    description:
      "Accelerates Vision-Language Models for financial document processing through intelligent routing and early-exit strategies, cutting cost by 96% while preserving accuracy.",
    tech: ["Python", "PyTorch", "MoE", "VLM", "Early Exit", "Streamlit"],
    metrics: [
      { label: "Throughput", value: "3.5x" },
      { label: "Cost reduction", value: "96%" },
      { label: "Latency", value: "~250ms" },
    ],
    github: "https://github.com/aneeshjayan",
  },
  {
    name: "LLM Probing",
    blurb: "Mechanistic interpretability across transformer layers",
    description:
      "Mechanistic interpretability study examining how a 3B-parameter language model encodes behavioral instructions across its layers using linear probes and PCA-based analysis.",
    tech: ["PyTorch", "HuggingFace", "StableLM", "PCA", "Python", "NLP"],
    metrics: [
      { label: "Probe accuracy", value: "~100%" },
      { label: "Model size", value: "3B" },
      { label: "Method", value: "Linear" },
    ],
    github: "https://github.com/aneeshjayan",
  },
  {
    name: "AI Video Editor Agent",
    blurb: "Multi-agent video editing via natural language",
    description:
      "Multi-agent system automating video editing through natural language instructions. Specialized agents handle audio transcription, scene analysis, and FFmpeg orchestration across three pipeline modes.",
    tech: ["CrewAI", "GPT-4o", "Whisper", "FFmpeg", "FastAPI", "Ollama"],
    metrics: [
      { label: "Agents", value: "6" },
      { label: "Pipeline modes", value: "3" },
      { label: "Interface", value: "NL" },
    ],
    github: "https://github.com/aneeshjayan",
  },
  {
    name: "FinSLM",
    blurb: "Domain-adapted financial language model",
    description:
      "Mistral-7B fine-tuned on SEC filings and financial news via QLoRA for consumer-grade deployment. Benchmarked against full fine-tuning across 20+ companies with W&B tracking.",
    tech: ["Mistral-7B", "LoRA", "QLoRA", "SEC EDGAR", "W&B", "Python"],
    metrics: [
      { label: "LoRA vs full", value: "98%" },
      { label: "Min VRAM", value: "2–3 GB" },
      { label: "Companies", value: "20+" },
    ],
    github: "https://github.com/aneeshjayan",
  },
  {
    name: "FocusMate",
    blurb: "ADHD-focused AI co-pilot for executive function",
    description:
      "AI productivity co-pilot that ingests Gmail, Google Calendar, and voice notes via OAuth2, then surfaces prioritized tasks and daily schedules — built for users with executive function challenges.",
    tech: ["FastAPI", "React", "Vite", "Expo", "Gmail API", "Google OAuth2"],
    metrics: [
      { label: "REST endpoints", value: "6+" },
      { label: "Services", value: "2" },
      { label: "Platforms", value: "Web+Mobile" },
    ],
    github: "https://github.com/aneeshjayan",
  },
  {
    name: "VoiceGuardAI",
    blurb: "RL-powered voice deepfake & spoofing detector",
    description:
      "Real-time voice authenticity system trained with reinforcement learning — reward signals shaped by false-accept rate and adversarial robustness. Detects TTS spoofing, voice conversion, and replay attacks across noisy environments with sub-50ms latency.",
    tech: ["PyTorch", "PPO", "RLHF", "Wav2Vec2", "ONNX", "FastAPI"],
    metrics: [
      { label: "EER", value: "1.8%" },
      { label: "Latency", value: "<50ms" },
      { label: "Attack types", value: "6" },
    ],
    github: "https://github.com/aneeshjayan",
  },
  {
    name: "Insurance SLM Agent",
    blurb: "Production-grade insurance agent with full RLHF pipeline",
    description:
      "Multi-agent insurance assistant built on a domain-fine-tuned SLM with a complete RLHF training pipeline (SFT → Bradley-Terry Reward Model → GRPO → DPO). Handles FNOL filing, policy Q&A via RAG, competitor research via web agents, and human escalation — all routed through a LangGraph state machine with compliance guardrails.",
    tech: ["LangGraph", "GRPO", "DPO", "LoRA", "RAG", "FastAPI", "Tavily"],
    metrics: [
      { label: "Agents", value: "5" },
      { label: "RLHF phases", value: "4" },
      { label: "VRAM", value: "~40GB" },
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
                {String(i + 1).padStart(2, "0")} / {String(projects.length).padStart(2, "0")}
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
