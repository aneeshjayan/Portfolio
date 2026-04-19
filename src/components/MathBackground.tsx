const NN_LAYERS = [
  { x: 120, ys: [175, 262, 350, 437, 525] },
  { x: 330, ys: [100, 171, 243, 314, 386, 457, 529, 600] },
  { x: 540, ys: [100, 171, 243, 314, 386, 457, 529, 600] },
  { x: 750, ys: [143, 229, 314, 400, 486, 571] },
  { x: 960, ys: [257, 350, 443] },
];

export function MathBackground() {
  return (
    <div aria-hidden className="pointer-events-none fixed inset-0 -z-10 overflow-hidden">
      {/* Grid */}
      <div className="absolute inset-0 grid-bg opacity-35" />

      {/* Gradient orbs */}
      <div
        className="absolute -top-48 -left-24 w-[700px] h-[700px] rounded-full"
        style={{
          background:
            "radial-gradient(circle, color-mix(in oklab, oklch(0.82 0.15 200) 20%, transparent), transparent 65%)",
        }}
      />
      <div
        className="absolute -bottom-48 -right-24 w-[800px] h-[800px] rounded-full"
        style={{
          background:
            "radial-gradient(circle, color-mix(in oklab, oklch(0.72 0.18 295) 18%, transparent), transparent 65%)",
        }}
      />
      <div
        className="absolute top-[38%] left-[48%] w-[500px] h-[500px] rounded-full"
        style={{
          background:
            "radial-gradient(circle, color-mix(in oklab, oklch(0.78 0.17 160) 9%, transparent), transparent 65%)",
        }}
      />

      {/* Neural network diagram */}
      <svg
        className="absolute inset-0 w-full h-full opacity-[0.065]"
        viewBox="0 0 1200 720"
        preserveAspectRatio="xMidYMid slice"
        xmlns="http://www.w3.org/2000/svg"
      >
        <g className="text-foreground" stroke="currentColor" fill="currentColor">
          {NN_LAYERS.slice(0, -1).map((layer, li) =>
            layer.ys.flatMap((y1) =>
              NN_LAYERS[li + 1].ys.map((y2) => (
                <line
                  key={`e${li}-${y1}-${y2}`}
                  x1={layer.x}
                  y1={y1}
                  x2={NN_LAYERS[li + 1].x}
                  y2={y2}
                  strokeWidth="0.45"
                  fill="none"
                />
              ))
            )
          )}
          {NN_LAYERS.map((layer, li) =>
            layer.ys.map((y) => (
              <circle
                key={`n${li}-${y}`}
                cx={layer.x}
                cy={y}
                r="6"
                strokeWidth="1.5"
                fillOpacity="0.45"
              />
            ))
          )}
          {/* layer labels */}
          <text x="100" y="660" fontFamily="JetBrains Mono,monospace" fontSize="10" textAnchor="middle" fillOpacity="0.5">input</text>
          <text x="330" y="660" fontFamily="JetBrains Mono,monospace" fontSize="10" textAnchor="middle" fillOpacity="0.5">hidden₁</text>
          <text x="540" y="660" fontFamily="JetBrains Mono,monospace" fontSize="10" textAnchor="middle" fillOpacity="0.5">hidden₂</text>
          <text x="750" y="660" fontFamily="JetBrains Mono,monospace" fontSize="10" textAnchor="middle" fillOpacity="0.5">hidden₃</text>
          <text x="960" y="660" fontFamily="JetBrains Mono,monospace" fontSize="10" textAnchor="middle" fillOpacity="0.5">output</text>
        </g>
      </svg>

      {/* Math + DSA + Cloud equations */}
      <svg className="absolute inset-0 h-full w-full opacity-[0.07]" xmlns="http://www.w3.org/2000/svg">
        <defs>
          <pattern
            id="math-tile"
            x="0"
            y="0"
            width="820"
            height="600"
            patternUnits="userSpaceOnUse"
          >
            <g
              fill="currentColor"
              fontFamily="JetBrains Mono, monospace"
              fontSize="12"
              className="text-foreground"
            >
              {/* Transformer / LLM */}
              <text x="20" y="42">Attention(Q,K,V) = softmax(QKᵀ/√dₖ)·V</text>
              <text x="20" y="76">∂L/∂θ = −∑ yᵢ·log(ŷᵢ)   # cross-entropy loss</text>
              <text x="20" y="110">W = W₀ + BA   rank(BA) ≪ rank(W₀)   # LoRA</text>
              {/* Activations + Optimiser */}
              <text x="20" y="144">σ(x) = 1/(1+e⁻ˣ)     ReLU(x) = max(0, x)</text>
              <text x="20" y="178">θₜ₊₁ = θₜ − η·∇L(θₜ)   # SGD update rule</text>
              <text x="20" y="212">Adam: mₜ = β₁mₜ₋₁ + (1−β₁)gₜ   vₜ = β₂vₜ₋₁ + (1−β₂)gₜ²</text>
              {/* CNN + Normalisation */}
              <text x="20" y="246">Conv2d out: ⌊(W−F+2P)/S⌋ + 1</text>
              <text x="20" y="280">BatchNorm: x̂ = (x−μ)/√(σ²+ε)  →  γx̂ + β</text>
              <text x="20" y="314">Dropout: keep prob = 1−p   # regularisation</text>
              {/* Probability + Information theory */}
              <text x="20" y="348">P(y|x) = softmax(Wx+b)   KL(P‖Q) = Σ P·log(P/Q)</text>
              <text x="20" y="382">H(X) = −Σ p(x)·log₂p(x)   # Shannon entropy</text>
              {/* DSA */}
              <text x="20" y="416">T(n) = 2T(n/2) + O(n)   → O(n log n)  # merge sort</text>
              <text x="20" y="450">Dijkstra: O((V+E)·log V)    BFS/DFS: O(V+E)</text>
              <text x="20" y="484">DP: optimal substructure + overlapping subproblems</text>
              {/* Cloud / Distributed */}
              <text x="20" y="518">CAP: consistency ∪ availability ∪ partition tolerance</text>
              <text x="20" y="552">MapReduce: map(k,v) → shuffle → reduce(k, [v])</text>
              {/* Feed-forward / RLHF */}
              <text x="20" y="586">FFN(x) = max(0, xW₁+b₁)W₂+b₂   # transformer FFN</text>
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
            "radial-gradient(ellipse at center, transparent 30%, var(--background) 85%)",
        }}
      />
    </div>
  );
}
