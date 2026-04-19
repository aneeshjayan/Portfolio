import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/SectionTag";

export const Route = createFileRoute("/experience")({
  head: () => ({
    meta: [
      { title: "Experience — Aneesh Jayan Prabhu" },
      {
        name: "description",
        content:
          "Experience: Data Science Intern at Wolters Kluwer; Research Intern at VIT Centre for Cyber-Physical Systems.",
      },
      { property: "og:title", content: "Experience — Aneesh Jayan Prabhu" },
      {
        property: "og:description",
        content:
          "Roles, achievements, and shipped work across industry and research.",
      },
    ],
  }),
  component: ExperiencePage,
});

type Role = {
  company: string;
  title: string;
  location: string;
  period: string;
  bullets: string[];
  stack: string[];
};

const roles: Role[] = [
  {
    company: "Wolters Kluwer",
    title: "Data Science Intern",
    location: "Remote / India",
    period: "2024",
    bullets: [
      "Shipped ML-powered features into a regulated legal/tax product used by enterprise customers, owning the pipeline from data prep to evaluation.",
      "Built retrieval and ranking models over large legal document corpora, improving top-k relevance over the production baseline.",
      "Designed evaluation harnesses for hallucination, faithfulness, and latency to gate releases.",
      "Collaborated across product, engineering, and legal SMEs to translate domain requirements into measurable ML objectives.",
    ],
    stack: ["Python", "PyTorch", "Transformers", "AWS", "Airflow", "SQL"],
  },
  {
    company: "VIT — Centre for Cyber-Physical Systems",
    title: "Research Intern",
    location: "Vellore, India",
    period: "2023 – 2024",
    bullets: [
      "Led a research project on graph neural networks for financial fraud detection over heterogeneous transaction graphs.",
      "Co-authored peer-reviewed publications on graph-based anomaly detection and trustworthy AI.",
      "Built reproducible training pipelines and ablation suites; presented findings to faculty and external reviewers.",
    ],
    stack: ["PyTorch Geometric", "NetworkX", "Neo4j", "Pandas", "LaTeX"],
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

      <ol className="relative mt-12 space-y-10 border-l border-border/60 pl-6">
        {roles.map((r, i) => (
          <li
            key={r.company}
            className="relative animate-fade-in-up"
            style={{ animationDelay: `${i * 100}ms` }}
          >
            <span className="absolute -left-[33px] top-2 flex size-4 items-center justify-center rounded-full border border-border bg-background">
              <span className="size-1.5 rounded-full bg-gradient-brand" />
            </span>

            <div className="rounded-xl border border-border bg-card/60 p-6 backdrop-blur card-elevated">
              <div className="flex flex-wrap items-start justify-between gap-2">
                <div>
                  <h2 className="text-lg font-semibold">
                    {r.title} ·{" "}
                    <span className="text-gradient">{r.company}</span>
                  </h2>
                  <p className="font-mono text-xs text-muted-foreground">
                    {r.location}
                  </p>
                </div>
                <span className="rounded-md border border-border bg-secondary/50 px-2 py-0.5 font-mono text-xs text-muted-foreground">
                  {r.period}
                </span>
              </div>

              <ul className="mt-4 space-y-2">
                {r.bullets.map((b, j) => (
                  <li
                    key={j}
                    className="flex gap-3 text-sm leading-relaxed text-muted-foreground"
                  >
                    <span className="mt-2 size-1.5 shrink-0 rounded-full bg-primary/70" />
                    <span>{b}</span>
                  </li>
                ))}
              </ul>

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
