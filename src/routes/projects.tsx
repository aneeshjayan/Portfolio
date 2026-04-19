import { createFileRoute } from "@tanstack/react-router";
import { Github, ExternalLink, Brain, Network, Heart, Zap, Search, Video, TrendingUp, CalendarCheck, Mic, FileText } from "lucide-react";
import { PageHeader } from "@/components/SectionTag";
import type { LucideIcon } from "lucide-react";

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
  icon: LucideIcon;
  iconColor: string;
  github?: string;
  link?: string;
};

const projects: Project[] = [
  {
    name: "Optimal-SLM",
    icon: Brain,
    iconColor: "oklch(0.82 0.15 200)",
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
    icon: Network,
    iconColor: "oklch(0.72 0.18 295)",
    blurb: "Production graph-based fraud detection system",
    description:
      "Production-grade fraud detection pipeline on Elliptic Bitcoin and IEEE-CIS datasets combining GraphSAGE/GAT on Neo4j transaction graphs, LSTM temporal modeling, and XGBoost ensemble with SMOTE balancing. Containerized with Docker and SHAP explainability for auditable per-decision outputs.",
    tech: ["GraphSAGE", "GAT", "Neo4j", "LSTM", "XGBoost", "SHAP", "Docker"],
    metrics: [
      { label: "F1", value: "0.91" },
      { label: "AUC-ROC", value: "0.97" },
      { label: "FP reduction", value: "30%" },
    ],
    github: "https://github.com/aneeshjayan",
  },
  {
    name: "TrustMedAI",
    icon: Heart,
    iconColor: "oklch(0.75 0.19 15)",
    blurb: "Medical conversational agent for Type-2 Diabetes",
    description:
      "Semantic retrieval system processing 500+ forum threads and 16,000 lines of clinical guidelines from ADA, Mayo Clinic, and NIH using MiniLM embeddings. Production RAG pipeline with FAISS achieving 0.970 faithfulness. React-based multimodal interface with speech-to-text and citation-backed responses.",
    tech: ["RAG", "FAISS", "MiniLM", "React", "Speech-to-Text", "Python"],
    metrics: [
      { label: "Faithfulness", value: "0.970" },
      { label: "Precision", value: "0.950" },
      { label: "BERTScore", value: "0.930" },
    ],
    github: "https://github.com/aneeshjayan",
  },
  {
    name: "VLM Speedup: LexFin Guard",
    icon: Zap,
    iconColor: "oklch(0.82 0.18 85)",
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
    icon: Search,
    iconColor: "oklch(0.78 0.17 160)",
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
    icon: Video,
    iconColor: "oklch(0.72 0.18 295)",
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
    icon: TrendingUp,
    iconColor: "oklch(0.78 0.17 145)",
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
    icon: CalendarCheck,
    iconColor: "oklch(0.82 0.15 200)",
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
    icon: Mic,
    iconColor: "oklch(0.75 0.19 15)",
    blurb: "Prompt injection security middleware for voice agents",
    description:
      "4-layer real-time injection detection pipeline: Aho-Corasick rule engine (p99 <1ms), FAISS semantic classifier, PPO routing policy, and async GRPO reasoner — achieving 88% attack recall across 6 threat classes including jailbreak, tool hijacking, and data exfiltration. Deployed on AWS EC2 at voiceguardlm.store.",
    tech: ["PPO", "GRPO", "FAISS", "Redis", "FastAPI", "Docker", "AWS EC2"],
    metrics: [
      { label: "Attack recall", value: "88%" },
      { label: "Latency p99", value: "<50ms" },
      { label: "Threat classes", value: "6" },
    ],
    github: "https://github.com/aneeshjayan",
    link: "https://voiceguardlm.store",
  },
  {
    name: "Insurance SLM Agent",
    icon: FileText,
    iconColor: "oklch(0.82 0.18 85)",
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
            {/* Icon + links row */}
            <div className="mb-4 flex items-start justify-between">
              <div
                className="rounded-lg p-2.5"
                style={{ background: `color-mix(in oklab, ${p.iconColor} 15%, transparent)`, border: `1px solid color-mix(in oklab, ${p.iconColor} 30%, transparent)` }}
              >
                <p.icon className="size-5" style={{ color: p.iconColor }} />
              </div>
              <div className="flex items-center gap-2 pt-0.5">
                <span className="font-mono text-xs text-muted-foreground">
                  {String(i + 1).padStart(2, "0")} / {String(projects.length).padStart(2, "0")}
                </span>
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
