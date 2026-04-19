

## Portfolio for Aneesh Jayan Prabhu

A dark, terminal-inspired portfolio styled after aaabadcode.com — featuring a tiled background of **real ML/DL equations and LLM mathematics** (softmax, attention, cross-entropy, gradient descent, backprop, transformer math) rendered as faint SVG/LaTeX overlays.

### Design
- **Theme:** Pure black background, faint white/grey math overlays, cyan (#22d3ee) → violet (#a78bfa) gradient accents
- **Typography:** Geist Sans for body/headings, JetBrains Mono for code/tags/terminal
- **Section labels:** `// about`, `// projects`, `// experience` mono tags
- **Effects:** Smooth scroll, fade-in on scroll, subtle hover lifts, "Available for new projects" pulse dot
- **Background math layer:** Tiled SVG containing equations like attention `Attention(Q,K,V)=softmax(QKᵀ/√dₖ)V`, cross-entropy, ReLU/sigmoid plots, gradient descent, LoRA decomposition `W = W₀ + BA`, RAG diagram fragments

### Sections (separate routes for SEO)

1. **Home (`/`)** — Hero with name "Aneesh Jayan Prabhu" gradient-accented, tagline *"AI/ML engineer building agentic systems, RAG pipelines, and production ML — from research to shipped products."*, "Available for new projects" pill, View Projects / Get in Touch CTAs, GitHub + LinkedIn icons, scroll indicator
2. **About (`/about`)** — Bio + terminal block (`whoami`, `cat skills.txt`, `echo $STATUS`), stats (Models trained, Production systems, Research papers, F1 best score 0.91)
3. **Projects (`/projects`)** — Cards for: Optimal-SLM, FraudGNN, TrustMedAI (with metrics, tech tags, GitHub links)
4. **Experience (`/experience`)** — Wolters Kluwer (Data Science Intern), VIT Centre for Cyber-Physical Systems (Research Intern) — timeline layout with bullet achievements
5. **Skills (`/skills`)** — Grouped chip grid: Languages, ML Frameworks, AI/GenAI, Infrastructure, Data Engineering, Cloud, Databases, Visualization
6. **Education (`/education`)** — ASU (MS Data Science 2026) + VIT (B.Tech ECE 2024) with coursework
7. **Contact (`/contact`)** — Email, phone, LinkedIn, GitHub, location (Tempe, AZ); simple contact form

### Navigation
Sticky top nav: `aneeshjayan` logo (left) + About / Projects / Experience / Skills / Contact (right). Mobile hamburger.

### Explicitly excluded
The Legal AI (CIPS Lab) volunteer and SoDA Mentor entries will **not** appear anywhere.

### SEO
Per-route `head()` with unique title/description/og tags. Root meta updated to "Aneesh Jayan Prabhu — AI/ML Engineer".

