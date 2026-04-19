import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/SectionTag";

export const Route = createFileRoute("/skills")({
  head: () => ({
    meta: [
      { title: "Skills — Aneesh Jayan Prabhu" },
      {
        name: "description",
        content:
          "Full technical skill set: Python, PyTorch, LangGraph, RAG, Agents, GNNs, CUDA, FastAPI, AWS, and more.",
      },
      { property: "og:title", content: "Skills — Aneesh Jayan Prabhu" },
      {
        property: "og:description",
        content:
          "Languages, ML frameworks, AI/GenAI, infrastructure, data engineering, cloud, databases, and visualization tools.",
      },
    ],
  }),
  component: SkillsPage,
});

const groups: { title: string; items: string[] }[] = [
  {
    title: "Languages",
    items: ["Python", "SQL", "R", "C", "C++", "MATLAB"],
  },
  {
    title: "ML Frameworks & Tools",
    items: [
      "PyTorch", "TensorFlow", "Scikit-Learn", "OpenCV",
      "LoRA / QLoRA", "PEFT", "Hugging Face Transformers",
      "MLflow", "Dataiku", "XGBoost",
    ],
  },
  {
    title: "AI & GenAI",
    items: [
      "LangChain", "LangGraph", "Model Context Protocol (MCP)",
      "Agentic AI Systems", "RAG Pipelines", "FAISS", "Neo4j Vector",
      "Prompt Engineering", "Guardrails & Inference Pipelines",
      "Semantic Search", "TRL / RLHF", "vLLM",
    ],
  },
  {
    title: "AI Infrastructure",
    items: [
      "CUDA", "TensorRT", "Docker", "AWS SageMaker",
      "Databricks", "GPU Workload Optimization",
      "CI/CD (Azure DevOps, GitHub Actions)", "MLOps",
    ],
  },
  {
    title: "Data Engineering",
    items: [
      "ETL", "PySpark", "Kafka", "Airflow", "Hadoop",
      "Data Ingestion Pipelines", "Benchmarking & Observability", "Grafana",
    ],
  },
  {
    title: "Cloud & DevOps",
    items: [
      "AWS (Lambda, S3, SageMaker)", "Azure DevOps",
      "Google Cloud SQL", "Docker", "Git", "CI/CD",
    ],
  },
  {
    title: "APIs & Services",
    items: [
      "FastAPI", "Microsoft Graph API", "Exchange API",
      "Power Automate", "Google OAuth2", "REST APIs",
    ],
  },
  {
    title: "Databases",
    items: ["MySQL", "PostgreSQL", "MongoDB", "Neo4j", "FAISS", "Redis"],
  },
  {
    title: "Visualization & Frontend",
    items: ["React", "Streamlit", "Power BI", "Matplotlib", "Seaborn", "Plotly"],
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

      <div className="mt-12 grid gap-5 md:grid-cols-2 lg:grid-cols-3">
        {groups.map((g, i) => (
          <div
            key={g.title}
            className="rounded-xl border border-border bg-card/60 p-6 backdrop-blur card-elevated animate-fade-in-up transition-all duration-300 hover:border-primary/30"
            style={{ animationDelay: `${i * 55}ms` }}
          >
            <div className="mb-4 flex items-center gap-2">
              <span className="font-mono text-xs text-muted-foreground">
                {String(i + 1).padStart(2, "0")}
              </span>
              <h2 className="font-semibold">
                <span className="text-gradient">{g.title}</span>
              </h2>
            </div>
            <div className="flex flex-wrap gap-1.5">
              {g.items.map((it) => (
                <span
                  key={it}
                  className="rounded-md border border-border/60 bg-secondary/40 px-2.5 py-1 font-mono text-xs text-foreground/90 transition-colors hover:border-primary/50 hover:bg-primary/10 hover:text-primary"
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
