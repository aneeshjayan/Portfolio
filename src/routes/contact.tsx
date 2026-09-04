import { useState } from "react";
import { createFileRoute } from "@tanstack/react-router";
import { EMAIL } from "@/data/portfolio";
import { useScrollReveal } from "@/hooks/useScrollReveal";

export const Route = createFileRoute("/contact")({
  head: () => ({
    meta: [
      { title: "Contact — Aneesh Jayan Prabhu" },
      {
        name: "description",
        content: "Get in touch with Aneesh Jayan Prabhu — email, LinkedIn, or GitHub.",
      },
      { property: "og:title", content: "Contact — Aneesh Jayan Prabhu" },
    ],
  }),
  component: ContactPage,
});

const ACCENT = "oklch(0.85 0.19 145)";

function ContactPage() {
  const ref = useScrollReveal<HTMLDivElement>([]);
  const [replyTo, setReplyTo] = useState("");
  const [message, setMessage] = useState("");

  const sendMail = () => {
    const from = replyTo ? `From: ${replyTo}\n\n` : "";
    window.location.href = `mailto:${EMAIL}?subject=${encodeURIComponent("Portfolio contact")}&body=${encodeURIComponent(from + message)}`;
  };

  return (
    <div ref={ref} style={{ animation: "dc-boot-in 0.4s ease-out both" }}>
      <p style={{ margin: 0, fontSize: 12, color: "oklch(0.5 0.02 200)" }}>
        <span style={{ color: ACCENT }}>$</span> mail -s "hello" aneesh
      </p>
      <h2
        style={{
          margin: "14px 0 4px",
          fontSize: 26,
          fontWeight: 700,
          color: "oklch(0.94 0.02 160)",
        }}
      >
        // contact
      </h2>
      <p style={{ margin: 0, fontSize: 12.5, color: "oklch(0.6 0.02 200)" }}>
        Write below and hit send — it opens your mail client with the message addressed to me.
      </p>

      <div
        style={{
          marginTop: 20,
          maxWidth: 620,
          border: "1px solid oklch(0.32 0.03 190 / 0.4)",
          background: "oklch(0.09 0.012 235 / 0.7)",
        }}
      >
        <div
          style={{
            borderBottom: "1px solid oklch(0.32 0.03 190 / 0.3)",
            padding: "8px 14px",
            fontSize: 10.5,
            color: "oklch(0.5 0.02 200)",
          }}
        >
          compose — {EMAIL}
        </div>
        <div style={{ padding: 18, display: "flex", flexDirection: "column", gap: 14 }}>
          <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <span style={{ fontSize: 10.5, color: "oklch(0.55 0.02 200)" }}>from:</span>
            <input
              type="email"
              value={replyTo}
              onChange={(e) => setReplyTo(e.target.value)}
              placeholder="you@company.com"
              style={{
                border: "1px solid oklch(0.32 0.03 190 / 0.45)",
                background: "#04070a",
                padding: "9px 11px",
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 12.5,
                color: "oklch(0.9 0.02 160)",
                outline: "none",
              }}
            />
          </label>
          <label style={{ display: "flex", flexDirection: "column", gap: 5 }}>
            <span style={{ fontSize: 10.5, color: "oklch(0.55 0.02 200)" }}>body:</span>
            <textarea
              value={message}
              onChange={(e) => setMessage(e.target.value)}
              rows={6}
              placeholder="what are you building?"
              style={{
                resize: "vertical",
                border: "1px solid oklch(0.32 0.03 190 / 0.45)",
                background: "#04070a",
                padding: "9px 11px",
                fontFamily: "'JetBrains Mono', monospace",
                fontSize: 12.5,
                lineHeight: 1.7,
                color: "oklch(0.9 0.02 160)",
                outline: "none",
              }}
            />
          </label>
          <button
            onClick={sendMail}
            style={{
              alignSelf: "flex-start",
              border: "1px solid oklch(0.85 0.19 145 / 0.4)",
              background: "oklch(0.85 0.19 145 / 0.12)",
              padding: "10px 18px",
              fontFamily: "'JetBrains Mono', monospace",
              fontSize: 12.5,
              fontWeight: 600,
              color: ACCENT,
              cursor: "pointer",
            }}
          >
            send ↵
          </button>
        </div>
      </div>

      <div
        style={{
          marginTop: 20,
          display: "flex",
          flexWrap: "wrap",
          gap: 10,
          fontSize: 11.5,
          color: "oklch(0.6 0.02 200)",
        }}
      >
        <span>{EMAIL}</span>
        <span style={{ color: "oklch(0.4 0.02 200)" }}>·</span>
        <span>(602) 768-6622</span>
        <span style={{ color: "oklch(0.4 0.02 200)" }}>·</span>
        <span>Tempe, Arizona</span>
      </div>
    </div>
  );
}
