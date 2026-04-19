import { createFileRoute } from "@tanstack/react-router";
import { useState } from "react";
import { Mail, MapPin, Phone, Github, Linkedin, Send } from "lucide-react";
import { PageHeader, SectionTag } from "@/components/SectionTag";

export const Route = createFileRoute("/contact")({
  head: () => ({
    meta: [
      { title: "Contact — Aneesh Jayan Prabhu" },
      {
        name: "description",
        content:
          "Reach Aneesh Jayan Prabhu — email, phone, LinkedIn, GitHub. Based in Tempe, Arizona.",
      },
      { property: "og:title", content: "Contact — Aneesh Jayan Prabhu" },
      {
        property: "og:description",
        content: "Get in touch via email, LinkedIn, or the contact form.",
      },
    ],
  }),
  component: ContactPage,
});

const channels = [
  {
    icon: Mail,
    label: "Email",
    value: "aneeshjayan11@gmail.com",
    href: "mailto:aneeshjayan11@gmail.com",
  },
  {
    icon: Phone,
    label: "Phone",
    value: "+1 (602) 768-6622",
    href: "tel:+16027686622",
  },
  {
    icon: Linkedin,
    label: "LinkedIn",
    value: "linkedin.com/in/aneeshjayan",
    href: "https://www.linkedin.com/in/aneeshjayan/",
  },
  {
    icon: Github,
    label: "GitHub",
    value: "github.com/aneeshjayan",
    href: "https://github.com/aneeshjayan",
  },
  {
    icon: MapPin,
    label: "Location",
    value: "Tempe, Arizona, USA",
  },
];

function ContactPage() {
  const [sent, setSent] = useState(false);

  function onSubmit(e: React.FormEvent<HTMLFormElement>) {
    e.preventDefault();
    const data = new FormData(e.currentTarget);
    const subject = encodeURIComponent(`Portfolio contact — ${data.get("name")}`);
    const body = encodeURIComponent(
      `${data.get("message")}\n\n— ${data.get("name")} (${data.get("email")})`,
    );
    window.location.href = `mailto:aneeshjayan11@gmail.com?subject=${subject}&body=${body}`;
    setSent(true);
  }

  return (
    <section className="mx-auto max-w-6xl px-6 py-16">
      <PageHeader
        tag="contact"
        title="Get in touch"
        description="I'm open to AI/ML roles, collaborations, and interesting research conversations."
      />

      <div className="mt-12 grid gap-8 lg:grid-cols-5">
        {/* Channels */}
        <div className="lg:col-span-2">
          <SectionTag>channels</SectionTag>
          <ul className="space-y-2">
            {channels.map(({ icon: Icon, label, value, href }) => {
              const inner = (
                <div className="flex items-center gap-3 rounded-xl border border-border bg-card/60 p-4 backdrop-blur transition-colors hover:border-primary/40 hover:bg-card/80">
                  <div className="rounded-md border border-border bg-secondary/60 p-2">
                    <Icon className="size-4 text-primary" />
                  </div>
                  <div className="min-w-0">
                    <div className="font-mono text-xs uppercase tracking-wider text-muted-foreground">
                      {label}
                    </div>
                    <div className="truncate text-sm">{value}</div>
                  </div>
                </div>
              );
              return (
                <li key={label}>
                  {href ? (
                    <a href={href} target="_blank" rel="noreferrer">
                      {inner}
                    </a>
                  ) : (
                    inner
                  )}
                </li>
              );
            })}
          </ul>
        </div>

        {/* Form */}
        <div className="lg:col-span-3">
          <SectionTag>send a message</SectionTag>
          <form
            onSubmit={onSubmit}
            className="space-y-4 rounded-xl border border-border bg-card/60 p-6 backdrop-blur card-elevated"
          >
            <div className="grid gap-4 sm:grid-cols-2">
              <Field name="name" label="Name" placeholder="Ada Lovelace" required />
              <Field
                name="email"
                label="Email"
                type="email"
                placeholder="ada@example.com"
                required
              />
            </div>
            <div>
              <label className="mb-1.5 block font-mono text-xs uppercase tracking-wider text-muted-foreground">
                Message
              </label>
              <textarea
                name="message"
                required
                rows={6}
                placeholder="What are you building?"
                className="w-full resize-none rounded-md border border-border bg-background/60 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/60 focus:border-primary focus:outline-none focus:ring-1 focus:ring-ring"
              />
            </div>

            <div className="flex items-center justify-between gap-4">
              <p className="font-mono text-xs text-muted-foreground">
                {sent ? (
                  <span className="text-emerald-400">$ message dispatched ✓</span>
                ) : (
                  <>$ awaiting input...</>
                )}
              </p>
              <button
                type="submit"
                className="inline-flex items-center gap-2 rounded-lg bg-gradient-brand px-4 py-2 text-sm font-medium text-background transition-transform hover:-translate-y-0.5 glow"
              >
                Send
                <Send className="size-4" />
              </button>
            </div>
          </form>
        </div>
      </div>
    </section>
  );
}

function Field({
  name,
  label,
  type = "text",
  placeholder,
  required,
}: {
  name: string;
  label: string;
  type?: string;
  placeholder?: string;
  required?: boolean;
}) {
  return (
    <div>
      <label className="mb-1.5 block font-mono text-xs uppercase tracking-wider text-muted-foreground">
        {label}
      </label>
      <input
        name={name}
        type={type}
        placeholder={placeholder}
        required={required}
        className="w-full rounded-md border border-border bg-background/60 px-3 py-2 text-sm text-foreground placeholder:text-muted-foreground/60 focus:border-primary focus:outline-none focus:ring-1 focus:ring-ring"
      />
    </div>
  );
}
