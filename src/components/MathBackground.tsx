/**
 * Tiled background of ML/DL/LLM mathematics rendered as a faint SVG overlay.
 * Sits fixed behind all content. Pointer-events disabled.
 */
export function MathBackground() {
  return (
    <div
      aria-hidden
      className="pointer-events-none fixed inset-0 -z-10 overflow-hidden"
    >
      {/* Grid */}
      <div className="absolute inset-0 grid-bg opacity-60" />

      {/* Radial glow */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at 20% 0%, color-mix(in oklab, var(--cyan) 12%, transparent), transparent 50%), radial-gradient(ellipse at 80% 100%, color-mix(in oklab, var(--violet) 12%, transparent), transparent 50%)",
        }}
      />

      {/* Math equations layer */}
      <svg
        className="absolute inset-0 h-full w-full opacity-[0.07]"
        xmlns="http://www.w3.org/2000/svg"
      >
        <defs>
          <pattern
            id="math-tile"
            x="0"
            y="0"
            width="640"
            height="480"
            patternUnits="userSpaceOnUse"
          >
            <g
              fill="currentColor"
              fontFamily="JetBrains Mono, monospace"
              fontSize="13"
              className="text-foreground"
            >
              <text x="20" y="40">Attention(Q,K,V) = softmax(QKᵀ / √dₖ)·V</text>
              <text x="40" y="78">∂L/∂θ = -∑ y·log(ŷ)</text>
              <text x="20" y="118">W = W₀ + B·A    # LoRA</text>
              <text x="60" y="158">σ(x) = 1 / (1 + e⁻ˣ)</text>
              <text x="20" y="198">ReLU(x) = max(0, x)</text>
              <text x="40" y="238">θₜ₊₁ = θₜ − η·∇L(θₜ)</text>
              <text x="20" y="278">H(p,q) = −∑ p(x)·log q(x)</text>
              <text x="60" y="318">P(y|x) = softmax(Wx + b)</text>
              <text x="20" y="358">KL(P‖Q) = ∑ P(x)·log(P(x)/Q(x))</text>
              <text x="40" y="398">y = β₀ + β₁x + ε</text>
              <text x="20" y="438">FFN(x) = max(0, xW₁+b₁)W₂+b₂</text>
              <text x="60" y="468">RAG: retrieve → augment → generate</text>
            </g>
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#math-tile)" />
      </svg>

      {/* Vignette */}
      <div
        className="absolute inset-0"
        style={{
          background:
            "radial-gradient(ellipse at center, transparent 50%, var(--background) 100%)",
        }}
      />
    </div>
  );
}
