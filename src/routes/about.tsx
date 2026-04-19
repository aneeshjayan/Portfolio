import { createFileRoute } from "@tanstack/react-router";
import { PageHeader, SectionTag } from "@/components/SectionTag";

export const Route = createFileRoute("/about")({
  head: () => ({
    meta: [
      { title: "About — Aneesh Jayan Prabhu" },
      {
        name: "description",
        content:
          "Bio of Aneesh Jayan Prabhu — MS Data Science at ASU, AI/ML engineer focused on agents, RAG, and production ML.",
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

function AboutPage() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-16">
      <PageHeader
        tag="about"
        title="About"
        description="I'm Aneesh — an AI/ML engineer focused on shipping intelligent systems that actually work in production. Currently pursuing my MS in Data Science at Arizona State University."
      />

      <div className="mt-12 grid gap-10 lg:grid-cols-5">
        <div className="lg:col-span-3">
          <SectionTag>bio</SectionTag>
          <div className="space-y-4 text-base leading-relaxed text-muted-foreground">
            <p>
              I build at the intersection of <span className="text-foreground">large language models</span>,
              {" "}
              <span className="text-foreground">graph neural networks</span>, and
              {" "}
              <span className="text-foreground">applied data engineering</span>. My
              work spans agentic LLM systems, retrieval-augmented generation,
              fraud detection on transaction graphs, and trust-calibrated medical AI.
            </p>
            <p>
              At <span className="text-foreground">Wolters Kluwer</span> I shipped data
              science features into a regulated legal-tech product. Before that, at
              <span className="text-foreground"> VIT's Centre for Cyber-Physical Systems</span>{" "}
              I led research on graph-based fraud detection and contributed to
              peer-reviewed publications.
            </p>
            <p>
              I care about clean abstractions, evaluation that survives
              distribution shift, and models that earn user trust — not just leaderboard wins.
            </p>
          </div>
        </div>

        <div className="lg:col-span-2">
          <SectionTag>terminal</SectionTag>
          <div className="overflow-hidden rounded-xl border border-border bg-card/70 backdrop-blur card-elevated">
            <div className="flex items-center gap-2 border-b border-border bg-secondary/40 px-4 py-2.5">
              <span className="size-2.5 rounded-full bg-red-500/70" />
              <span className="size-2.5 rounded-full bg-yellow-500/70" />
              <span className="size-2.5 rounded-full bg-green-500/70" />
            </div>
            <pre className="overflow-x-auto p-5 font-mono text-[13px] leading-relaxed">
              <span className="text-primary">$</span> whoami{"\n"}
              aneesh — ai/ml engineer{"\n\n"}
              <span className="text-primary">$</span> cat skills.txt{"\n"}
              python · pytorch · langchain{"\n"}
              llms · rag · agents · gnns{"\n"}
              aws · docker · postgres{"\n\n"}
              <span className="text-primary">$</span> echo $STATUS{"\n"}
              <span className="text-emerald-400">building & shipping</span>
              <span className="ml-1 inline-block h-3 w-2 translate-y-0.5 bg-primary animate-blink" />
            </pre>
          </div>
        </div>
      </div>

      <div className="mt-16">
        <SectionTag>by the numbers</SectionTag>
        <div className="grid grid-cols-2 gap-4 sm:grid-cols-4">
          {stats.map((s) => (
            <div
              key={s.label}
              className="rounded-xl border border-border bg-card/50 p-5 backdrop-blur transition-transform hover:-translate-y-0.5"
            >
              <div className="font-mono text-3xl font-semibold text-gradient">
                {s.value}
              </div>
              <div className="mt-1 text-sm text-muted-foreground">{s.label}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
