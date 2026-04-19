import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/SectionTag";

export const Route = createFileRoute("/experience")({
  head: () => ({
    meta: [
      { title: "Experience — Aneesh Jayan Prabhu" },
      {
        name: "description",
        content:
          "Data Scientist at Wolters Kluwer (Legal & Regulatory); Research Scientist in Biomedical & Neuroinformatics at VIT.",
      },
      { property: "og:title", content: "Experience — Aneesh Jayan Prabhu" },
      {
        property: "og:description",
        content: "Roles, achievements, and shipped work across industry and research.",
      },
    ],
  }),
  component: ExperiencePage,
});

type Role = {
  company: string;
  logo: string;
  title: string;
  location: string;
  period: string;
  type: string;
  bullets: string[];
  stack: string[];
};

const roles: Role[] = [
  {
    company: "Wolters Kluwer — Legal & Regulatory Division",
    logo: "/wk.jpg",
    title: "Data Scientist",
    location: "New York, USA",
    period: "May 2025 – Dec 2025",
    type: "Industry · Internship",
    bullets: [
      "Designed and deployed a multi-agent agentic AI reporting system using LangGraph pipelines, translating complex legal data into conversational dashboard insights — boosting reporting efficiency by 22%.",
      "Built and validated RAG preprocessing pipelines for unstructured legal documents, applying flowchart and table transcription strategies; benchmarked BERT, RoBERTa, and T5 transformers, improving factual accuracy by 85%.",
      "Engineered FastAPI microservices integrated with PySpark and MongoDB-backed workflows with CI/CD via Azure DevOps, reducing manual escalations by 41% and enabling scalable enterprise-grade inference pipelines.",
      "Automated SMTP-to-OneDrive data ingestion and email-triggered workflows via Microsoft Graph API — achieving 95% improvement in ingestion reliability and 42% latency reduction with production monitoring via alerts.",
      "Participated in sprint planning, design reviews, and architecture sessions; collaborated with cross-functional partners to align ML system performance with enterprise product goals.",
    ],
    stack: ["LangGraph", "RAG", "FastAPI", "PySpark", "MongoDB", "Azure DevOps", "Microsoft Graph API", "CI/CD"],
  },
  {
    company: "Centre for Cyber-Physical Systems, VIT",
    logo: "/vit.jpg",
    title: "Research Scientist — Biomedical & Neuroinformatics",
    location: "Chennai, India",
    period: "May 2023 – May 2024",
    type: "Research",
    bullets: [
      "Engineered EEG and fMRI preprocessing pipelines using Spatio-Spectral Decomposition (SSD) in MATLAB and custom deep learning architectures for neuro-signal analysis.",
      "Built a hybrid deep learning–quantum framework for Autism detection from fMRI, integrating Swin Transformers, CNNs, and Quantum SVM/QNN models.",
      "Achieved 98.17% accuracy on ABIDE I and 96.2% on ABIDE II with a 25% reduction in computation time, demonstrating scalable AI-driven diagnostics.",
    ],
    stack: ["MATLAB", "PyTorch", "Swin Transformers", "Quantum SVM", "Python", "fMRI/EEG", "LaTeX"],
  },
];

function ExperiencePage() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-16">
      <PageHeader
        tag="experience"
        title="Experience"
        description="Where I've shipped, researched, and learned."
      />

      <ol className="relative mt-12 space-y-10 border-l border-border/60 pl-8">
        {roles.map((r, i) => (
          <li
            key={r.company}
            className="relative animate-fade-in-up"
            style={{ animationDelay: `${i * 120}ms` }}
          >
            {/* Timeline node */}
            <span className="absolute -left-[37px] top-5 flex size-5 items-center justify-center rounded-full border border-primary/40 bg-background">
              <span className="size-2 rounded-full bg-gradient-brand" />
            </span>

            <div className="rounded-xl border border-border bg-card/60 p-6 backdrop-blur card-elevated transition-all duration-300 hover:border-primary/30 hover:bg-card/80">
              {/* Header */}
              <div className="flex flex-wrap items-start gap-4">
                {/* Logo */}
                <div className="shrink-0 overflow-hidden rounded-lg border border-border bg-secondary/40 p-1">
                  <img
                    src={r.logo}
                    alt={r.company}
                    className="h-12 w-12 object-contain"
                    onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                  />
                </div>

                <div className="flex-1">
                  <div className="flex flex-wrap items-start justify-between gap-2">
                    <div>
                      <h2 className="text-lg font-semibold">
                        <span className="text-gradient">{r.company}</span>
                      </h2>
                      <p className="mt-0.5 text-sm font-medium text-foreground/90">{r.title}</p>
                      <p className="font-mono text-xs text-muted-foreground">{r.location}</p>
                    </div>
                    <div className="flex flex-col items-end gap-1.5">
                      <span className="rounded-md border border-border bg-secondary/50 px-2.5 py-0.5 font-mono text-xs text-muted-foreground">
                        {r.period}
                      </span>
                      <span className="rounded-md border border-primary/30 bg-primary/10 px-2.5 py-0.5 font-mono text-[10px] text-primary">
                        {r.type}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              {/* Bullets */}
              <ul className="mt-5 space-y-2.5">
                {r.bullets.map((b, j) => (
                  <li key={j} className="flex gap-3 text-sm leading-relaxed text-muted-foreground">
                    <span className="mt-2 size-1.5 shrink-0 rounded-full bg-primary/70" />
                    <span>{b}</span>
                  </li>
                ))}
              </ul>

              {/* Stack */}
              <div className="mt-5 flex flex-wrap gap-1.5">
                {r.stack.map((t) => (
                  <span
                    key={t}
                    className="rounded-md border border-border/60 bg-secondary/50 px-2 py-0.5 font-mono text-[11px] text-muted-foreground"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </div>
          </li>
        ))}
      </ol>
    </section>
  );
}
