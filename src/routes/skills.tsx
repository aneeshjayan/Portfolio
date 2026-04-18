import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/SectionTag";

export const Route = createFileRoute("/skills")({
  head: () => ({
    meta: [
      { title: "Skills — Aneesh Jayan Prabhu" },
      {
        name: "description",
        content:
          "Technical skills: Python, PyTorch, LLMs, RAG, agents, GNNs, AWS, Docker, Postgres, and more.",
      },
      { property: "og:title", content: "Skills — Aneesh Jayan Prabhu" },
      {
        property: "og:description",
        content:
          "A grouped overview of languages, ML frameworks, AI/GenAI, infra, data, cloud, and visualization tools.",
      },
    ],
  }),
  component: SkillsPage,
});

const groups: { title: string; items: string[] }[] = [
  {
    title: "Languages",
    items: ["Python", "TypeScript", "SQL", "C++", "Bash", "R"],
  },
  {
    title: "ML Frameworks",
    items: ["PyTorch", "TensorFlow", "scikit-learn", "XGBoost", "PyTorch Geometric", "Hugging Face"],
  },
  {
    title: "AI / GenAI",
    items: ["LLMs", "RAG", "Agents", "LangChain", "LlamaIndex", "vLLM", "LoRA / PEFT", "TRL / RLHF"],
  },
  {
    title: "Infrastructure",
    items: ["Docker", "Kubernetes", "FastAPI", "Airflow", "MLflow", "Ray"],
  },
  {
    title: "Data Engineering",
    items: ["Spark", "Pandas", "Polars", "dbt", "Kafka"],
  },
  {
    title: "Cloud",
    items: ["AWS", "GCP", "Azure", "Cloudflare"],
  },
  {
    title: "Databases",
    items: ["PostgreSQL", "MongoDB", "Neo4j", "Redis", "Pinecone", "FAISS"],
  },
  {
    title: "Visualization",
    items: ["Plotly", "Matplotlib", "Seaborn", "Tableau", "Power BI"],
  },
];

function SkillsPage() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-16">
      <PageHeader
        tag="skills"
        title="Skills"
        description="Tools I reach for when building, shipping, and measuring intelligent systems."
      />

      <div className="mt-12 grid gap-6 md:grid-cols-2">
        {groups.map((g, i) => (
          <div
            key={g.title}
            className="rounded-xl border border-border bg-card/60 p-6 backdrop-blur card-elevated animate-fade-in-up"
            style={{ animationDelay: `${i * 60}ms` }}
          >
            <div className="mb-4 flex items-center gap-2">
              <span className="font-mono text-xs text-muted-foreground">
                {String(i + 1).padStart(2, "0")}
              </span>
              <h2 className="font-semibold">
                <span className="text-gradient">{g.title}</span>
              </h2>
            </div>
            <div className="flex flex-wrap gap-2">
              {g.items.map((it) => (
                <span
                  key={it}
                  className="rounded-md border border-border/60 bg-secondary/40 px-2.5 py-1 font-mono text-xs text-foreground/90 transition-colors hover:border-primary/50 hover:text-foreground"
                >
                  {it}
                </span>
              ))}
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
