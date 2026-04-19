import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/SectionTag";

export const Route = createFileRoute("/education")({
  head: () => ({
    meta: [
      { title: "Education — Aneesh Jayan Prabhu" },
      {
        name: "description",
        content:
          "MS Data Science at Arizona State University; B.Tech ECE at VIT.",
      },
      { property: "og:title", content: "Education — Aneesh Jayan Prabhu" },
      {
        property: "og:description",
        content: "Degrees and coursework at ASU and VIT.",
      },
    ],
  }),
  component: EducationPage,
});

const degrees = [
  {
    school: "Arizona State University",
    shortName: "ASU",
    logo: "/asu.jpg",
    degree: "M.S. Data Science, Analytics & Engineering",
    period: "2024 – 2026",
    location: "Tempe, AZ",
    gpa: "In Progress",
    highlight: "Ira A. Fulton Schools of Engineering",
    courses: [
      "Statistics for Data Analysts",
      "Data Processing At Scale",
      "Statistical Machine Learning",
      "Data Mining",
      "Knowledge Representation & Reasoning",
      "Computing for Data-Driven Optimization",
      "Semantic Web Mining",
    ],
  },
  {
    school: "Vellore Institute of Technology",
    shortName: "VIT",
    logo: "/vit.jpg",
    degree: "B.Tech Electronics & Communication Engineering",
    period: "2020 – 2024",
    location: "Chennai, India",
    gpa: "8.5 / 10",
    highlight: "School of Electronics Engineering",
    courses: [
      "Probability & Statistics",
      "Linear Algebra",
      "Signals & Systems",
      "Machine Learning",
      "Data Structures & Algorithms",
      "Embedded Systems",
    ],
  },
];

function EducationPage() {
  return (
    <section className="mx-auto max-w-6xl px-6 py-16">
      <PageHeader
        tag="education"
        title="Education"
        description="Where I trained the models in my head."
      />

      <div className="mt-12 grid gap-6 md:grid-cols-2">
        {degrees.map((d, i) => (
          <div
            key={d.school}
            className="flex flex-col overflow-hidden rounded-xl border border-border bg-card/60 backdrop-blur card-elevated animate-fade-in-up transition-all duration-300 hover:border-primary/30 hover:bg-card/80"
            style={{ animationDelay: `${i * 100}ms` }}
          >
            {/* Logo banner */}
            <div className="flex items-center gap-4 border-b border-border/60 bg-secondary/30 px-6 py-5">
              <div className="shrink-0 overflow-hidden rounded-lg border border-border bg-background/60 p-1.5">
                <img
                  src={d.logo}
                  alt={d.school}
                  className="h-14 w-14 object-contain"
                  onError={(e) => { (e.target as HTMLImageElement).style.display = "none"; }}
                />
              </div>
              <div className="min-w-0">
                <h2 className="text-lg font-semibold leading-tight">
                  <span className="text-gradient">{d.school}</span>
                </h2>
                <p className="text-sm text-foreground/80">{d.degree}</p>
                <p className="font-mono text-xs text-muted-foreground">{d.highlight}</p>
              </div>
            </div>

            {/* Details */}
            <div className="flex-1 p-6">
              <div className="mb-5 flex flex-wrap gap-2">
                <span className="flex items-center gap-1.5 rounded-md border border-border bg-secondary/50 px-2.5 py-1 font-mono text-xs text-muted-foreground">
                  📅 {d.period}
                </span>
                <span className="flex items-center gap-1.5 rounded-md border border-border bg-secondary/50 px-2.5 py-1 font-mono text-xs text-muted-foreground">
                  📍 {d.location}
                </span>
                <span className="flex items-center gap-1.5 rounded-md border border-primary/30 bg-primary/10 px-2.5 py-1 font-mono text-xs text-primary">
                  GPA {d.gpa}
                </span>
              </div>

              <p className="mb-2.5 font-mono text-xs uppercase tracking-wider text-muted-foreground">
                relevant coursework
              </p>
              <div className="flex flex-wrap gap-1.5">
                {d.courses.map((c) => (
                  <span
                    key={c}
                    className="rounded-md border border-border/60 bg-secondary/40 px-2 py-0.5 font-mono text-[11px] text-muted-foreground"
                  >
                    {c}
                  </span>
                ))}
              </div>
            </div>
          </div>
        ))}
      </div>
    </section>
  );
}
