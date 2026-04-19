import { createFileRoute } from "@tanstack/react-router";
import { PageHeader } from "@/components/SectionTag";
import { GraduationCap } from "lucide-react";

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
    degree: "M.S. Data Science",
    period: "2024 – 2026 · Tempe, AZ",
    courses: [
      "Statistical Machine Learning",
      "Deep Learning",
      "Natural Language Processing",
      "Data Mining",
      "Knowledge Representation",
      "Big Data Systems",
    ],
  },
  {
    school: "Vellore Institute of Technology",
    degree: "B.Tech Electronics & Communication Engineering",
    period: "2020 – 2024 · Vellore, India",
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
            className="rounded-xl border border-border bg-card/60 p-6 backdrop-blur card-elevated animate-fade-in-up"
            style={{ animationDelay: `${i * 80}ms` }}
          >
            <div className="flex items-start gap-3">
              <div className="rounded-lg border border-border bg-secondary/60 p-2.5">
                <GraduationCap className="size-5 text-primary" />
              </div>
              <div>
                <h2 className="text-lg font-semibold">
                  <span className="text-gradient">{d.school}</span>
                </h2>
                <p className="text-sm text-foreground/90">{d.degree}</p>
                <p className="font-mono text-xs text-muted-foreground">
                  {d.period}
                </p>
              </div>
            </div>

            <div className="mt-5">
              <p className="mb-2 font-mono text-xs uppercase tracking-wider text-muted-foreground">
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
