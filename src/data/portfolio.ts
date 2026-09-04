export type Category = "agents" | "inference" | "security" | "applied" | "research";

export const ACCENT: Record<Category, string> = {
  agents: "oklch(0.72 0.2 330)",
  inference: "oklch(0.82 0.15 200)",
  security: "oklch(0.75 0.2 25)",
  applied: "oklch(0.85 0.16 85)",
  research: "oklch(0.85 0.19 145)",
};

export interface ProjectMetric {
  value: string;
  label: string;
}

export interface Project {
  name: string;
  slug: string;
  repo: string | null;
  category: Category;
  color: string;
  date: string;
  blurb: string;
  problem: string;
  architecture: string;
  metrics: ProjectMetric[];
  tech: string[];
  link: string | null;
}

export const PROJECTS: Project[] = [
  {
    name: "TerraInfer",
    slug: "terrainfer",
    repo: null,
    category: "inference",
    color: ACCENT.inference,
    date: "May 2026",
    blurb: "3D LLM inference pipeline for geospatial survey data",
    problem:
      "Outdoor LiDAR scans are enormous and 3D-LLM baselines burn GPU hours re-encoding the same terrain on every query, which makes kilometre-scale survey reasoning too expensive to run.",
    architecture:
      "LiDAR tile\n  └─ PointNet++ encoder (TensorRT)\n      └─ GPS-keyed feature cache (Redis + MLflow)\n          └─ cross-attn compressor 8192→128 tok\n              └─ Qwen-0.5B AWQ INT4 → CAD/JSON",
    metrics: [
      { value: "20-50x", label: "cost cut" },
      { value: "99.75%", label: "attn saved" },
      { value: "34ms", label: "per tile" },
    ],
    tech: ["TensorRT", "PointNet++", "Redis", "MLflow", "FastAPI", "Docker", "AWS EC2 G4dn"],
    link: null,
  },
  {
    name: "VoiceGuard AI",
    slug: "voiceguard-ai",
    repo: "https://github.com/aneeshjayan/voiceguard-ai",
    category: "security",
    color: ACCENT.security,
    date: "Apr 2026",
    blurb: "AI guardrails & prompt-injection defense for agents",
    problem:
      "Voice agents with tool access are one crafted utterance away from data exfiltration, and generic moderation APIs are too slow and too blunt to sit in a real-time call path.",
    architecture:
      "utterance\n  ├─ L1 Aho-Corasick rules   (p99 <1ms)\n  ├─ L2 FAISS semantic classifier\n  ├─ L3 PPO routing policy\n  └─ L4 async GRPO reasoner\n      → verdict + audit log (<50ms)",
    metrics: [
      { value: "0.90", label: "pipeline F1" },
      { value: "<30ms", label: "p99 L1-L3" },
      { value: "200+", label: "rule phrases" },
    ],
    tech: ["PPO", "GRPO", "FAISS", "Redis", "FastAPI", "PostgreSQL", "Nginx"],
    link: "https://voiceguardlm.store",
  },
  {
    name: "TrustMedAI",
    slug: "trustmedai",
    repo: "https://github.com/aneeshjayan/TrustMedAI",
    category: "agents",
    color: ACCENT.agents,
    date: "Dec 2025",
    blurb: "Retrieval-augmented conversational health agent",
    problem:
      "Patients asking about Type-2 diabetes get either forum folklore or unreadable clinical PDFs, and an ungrounded chatbot in that gap is actively dangerous.",
    architecture:
      "500+ threads + 16k lines of guidelines\n  └─ MiniLM embeddings → FAISS\n      └─ grounded RAG + citation binder\n          └─ React/TS multimodal UI\n              └─ eval harness (precision/faithfulness)",
    metrics: [
      { value: "0.970", label: "faithfulness" },
      { value: "0.950", label: "precision" },
      { value: "0.930", label: "BERTScore" },
    ],
    tech: ["FAISS", "MiniLM", "RAG", "React", "TypeScript", "Python"],
    link: null,
  },
  {
    name: "Insurance SLM Agent",
    slug: "insurance-slm-agent",
    repo: null,
    category: "agents",
    color: ACCENT.agents,
    date: "Mar 2026",
    blurb: "Multi-agent insurance assistant with full RLHF pipeline",
    problem:
      "Insurance front-office work is templated but high-stakes: FNOL intake, policy Q&A, and escalation all need the same domain fluency with an auditable trail.",
    architecture:
      "SFT → Bradley-Terry reward model → GRPO → DPO\n  └─ LangGraph state machine\n      ├─ FNOL intake agent\n      ├─ policy RAG agent\n      ├─ competitor research (Tavily)\n      └─ human escalation + guardrails",
    metrics: [
      { value: "5", label: "agents" },
      { value: "4", label: "RLHF phases" },
      { value: "~40GB", label: "train VRAM" },
    ],
    tech: ["LangGraph", "GRPO", "DPO", "LoRA", "RAG", "FastAPI", "Tavily"],
    link: null,
  },
  {
    name: "AI Video Editor Agent",
    slug: "ai-video-editor-agent",
    repo: "https://github.com/aneeshjayan/Video_editing_agent",
    category: "agents",
    color: ACCENT.agents,
    date: "Feb 2026",
    blurb: "Natural-language video editing via a multi-agent crew",
    problem:
      "Editing is a chain of mechanical decisions — find the quiet parts, cut the filler, match the beat — that is easy to describe in a sentence and tedious to execute by hand.",
    architecture:
      "prompt\n  └─ CrewAI planner\n      ├─ Whisper transcription agent\n      ├─ scene analysis agent\n      └─ FFmpeg orchestration agent\n          → 3 pipeline modes (fast/balanced/deep)",
    metrics: [
      { value: "6", label: "agents" },
      { value: "3", label: "pipelines" },
      { value: "6", label: "platform presets" },
    ],
    tech: ["CrewAI", "GPT-4o", "Whisper", "FFmpeg", "FastAPI", "Ollama"],
    link: null,
  },
  {
    name: "FocusMate",
    slug: "focusmate",
    repo: "https://github.com/aneeshjayan/FocusMate_AI_Co_Pilot_for_ADHD",
    category: "agents",
    color: ACCENT.agents,
    date: "Jan 2026",
    blurb: "Executive-function co-pilot for ADHD workflows",
    problem:
      "Task tools assume you can already prioritize. For executive-function challenges the hard part is deciding what matters today, not storing the list.",
    architecture:
      "Gmail + Calendar + voice notes (OAuth2)\n  └─ extraction & dedupe service\n      └─ priority scoring model\n          └─ daily schedule generator\n              └─ React web + Expo mobile",
    metrics: [
      { value: "6+", label: "endpoints" },
      { value: "2", label: "services" },
      { value: "web+app", label: "surfaces" },
    ],
    tech: ["FastAPI", "React", "Vite", "Expo", "Gmail API", "OAuth2"],
    link: null,
  },
  {
    name: "Optimal-SLM",
    slug: "optimal-slm",
    repo: "https://github.com/aneeshjayan/Prompt_optimizer",
    category: "inference",
    color: ACCENT.inference,
    date: "Nov 2025",
    blurb: "Dual-SLM reasoning + prompt optimization on 6GB",
    problem:
      "Verbose speech-to-text prompts waste tokens on every call, but running a frontier model just to compress them defeats the point — and most GPUs on hand have 6GB.",
    architecture:
      "utterance\n  └─ Qwen2-1.5B reasoner (4-bit, ~900MB)\n      └─ CoT intent + entity extraction\n          └─ Phi-3.5-Mini optimizer (~2.4GB)\n              └─ A2A protocol: cross-validate + score",
    metrics: [
      { value: "20-50%", label: "token cut" },
      { value: "6GB", label: "VRAM budget" },
      { value: "<2s", label: "per optimize" },
    ],
    tech: ["Qwen2-1.5B", "Phi-3.5-Mini", "LoRA", "4-bit quant", "A2A", "PyTorch"],
    link: null,
  },
  {
    name: "VLM Speedup — LexFin Guard",
    slug: "lexfin-guard",
    repo: "https://github.com/aneeshjayan/VLM-Speedup",
    category: "inference",
    color: ACCENT.inference,
    date: "Oct 2025",
    blurb: "Vision-language acceleration for financial documents",
    problem:
      "Most pages in a financial filing are trivial; running a full VLM on every one of them means paying frontier prices to read a header.",
    architecture:
      "page batch\n  └─ complexity router\n      ├─ cheap path: layout heuristics\n      └─ deep path: VLM w/ early-exit heads\n          └─ confidence gate → escalate",
    metrics: [
      { value: "3.5x", label: "throughput" },
      { value: "96%", label: "cost cut" },
      { value: "~250ms", label: "latency" },
    ],
    tech: ["PyTorch", "VLM", "MoE", "Early Exit", "Streamlit"],
    link: null,
  },
  {
    name: "FinSLM",
    slug: "finslm",
    repo: "https://github.com/aneeshjayan/Harvey-FinLM",
    category: "inference",
    color: ACCENT.inference,
    date: "Sep 2025",
    blurb: "Domain-adapted financial language model on consumer GPUs",
    problem:
      "Finance-specific language understanding usually implies a hosted frontier model, which is a non-starter when the data cannot leave the building.",
    architecture:
      "SEC EDGAR filings + financial news\n  └─ QLoRA fine-tune on Mistral-7B\n      └─ benchmark vs. full fine-tune\n          └─ 2-3GB VRAM deployment",
    metrics: [
      { value: "98%", label: "of full FT" },
      { value: "2-3GB", label: "min VRAM" },
      { value: "20+", label: "companies" },
    ],
    tech: ["Mistral-7B", "QLoRA", "PEFT", "SEC EDGAR", "W&B"],
    link: null,
  },
  {
    name: "LLM Probing",
    slug: "llm-probing",
    repo: null,
    category: "security",
    color: ACCENT.security,
    date: "Aug 2025",
    blurb: "Mechanistic interpretability of instruction encoding",
    problem:
      "We ask models to follow behavioural instructions without knowing where in the network that instruction actually lives — which makes failures unexplainable.",
    architecture:
      "StableLM-3B activations\n  └─ per-layer linear probes\n      └─ PCA of instruction subspace\n          └─ layer-wise accuracy curve",
    metrics: [
      { value: "~100%", label: "probe acc" },
      { value: "3B", label: "model" },
      { value: "linear", label: "method" },
    ],
    tech: ["PyTorch", "HuggingFace", "StableLM", "PCA", "NLP"],
    link: null,
  },
  {
    name: "FraudGNN",
    slug: "fraudgnn",
    repo: "https://github.com/aneeshjayan/Graph-Based-RAG-for-Cryptocurrency-Trend-Analysis",
    category: "applied",
    color: ACCENT.applied,
    date: "Jul 2025",
    blurb: "Graph + temporal ensemble for transaction fraud",
    problem:
      "Fraud is a property of the neighbourhood around a transaction, not the row itself, and flat tabular models keep flagging honest customers to catch it.",
    architecture:
      "Elliptic BTC + IEEE-CIS\n  └─ Neo4j transaction graph\n      ├─ GraphSAGE / GAT embeddings\n      ├─ LSTM temporal signal\n      └─ XGBoost ensemble + SMOTE\n          └─ SHAP per-decision audit",
    metrics: [
      { value: "0.91", label: "F1" },
      { value: "0.97", label: "AUC-ROC" },
      { value: "-30%", label: "false pos" },
    ],
    tech: ["GraphSAGE", "GAT", "Neo4j", "LSTM", "XGBoost", "SHAP", "Docker"],
    link: null,
  },
  {
    name: "LexFin",
    slug: "lexfin",
    repo: "https://github.com/aneeshjayan/LexFin",
    category: "agents",
    color: ACCENT.agents,
    date: "Feb 2026",
    blurb: "Vendor & contract risk monitoring with agent memory",
    problem:
      "Vendor risk lives in two disconnected places — payment data and signed contracts — so a spend spike and a missing liability cap never get seen as one problem until it is expensive.",
    architecture:
      "Kaggle transactions + CUAD contracts\n  └─ AutoDB warehouse (NL → schema)\n      ├─ Financial agent: spend anomalies\n      ├─ Legal agent: RAG clause checks\n      └─ shared memory (tryclean.ai)\n          └─ risk engine: fin×0.6 + legal×0.4",
    metrics: [
      { value: "2", label: "agents" },
      { value: "6", label: "risk flags" },
      { value: "41", label: "clause types" },
    ],
    tech: ["Prefect", "AutoDB", "tryclean.ai", "DuckDB", "PyArrow", "CUAD"],
    link: null,
  },
  {
    name: "Wildfire Multimodal Fusion",
    slug: "wildfire-fusion",
    repo: "https://github.com/aneeshjayan/Wildfire-detection",
    category: "applied",
    color: ACCENT.applied,
    date: "Mar 2025",
    blurb: "RGB + thermal fusion with vision-LLM explanations",
    problem:
      "Smoke fools RGB models and thermal alone has no context, so single-modality wildfire detection either misses early ignition or floods responders with false alarms.",
    architecture:
      "RGB + thermal frames\n  ├─ late fusion: independent CNN branches\n  └─ early fusion: 4-channel CrossViT\n      └─ KOSMOS-2 vision-LLM\n          └─ natural-language incident report",
    metrics: [
      { value: "2", label: "fusion modes" },
      { value: "4-ch", label: "input stack" },
      { value: "CrossViT", label: "backbone" },
    ],
    tech: ["CrossViT", "KOSMOS-2", "PyTorch", "OpenCV", "Streamlit"],
    link: null,
  },
  {
    name: "Autism Detection (fMRI)",
    slug: "autism-fmri",
    repo: "https://github.com/aneeshjayan/Dual-Block-Feature-Fusion-Network-for-Autism-detection-",
    category: "research",
    color: ACCENT.research,
    date: "May 2024",
    blurb: "Hybrid deep-learning / quantum architectures on ABIDE",
    problem:
      "Autism biomarkers in fMRI are subtle and the datasets are small, so heavy architectures overfit while cheap ones miss the signal entirely.",
    architecture:
      "EEG/fMRI + SSD preprocessing (MATLAB)\n  ├─ Swin Transformer / CNN branch\n  └─ Quantum SVM / QNN branch\n      └─ fused classifier → ABIDE I/II",
    metrics: [
      { value: "98.17%", label: "ABIDE I" },
      { value: "96.2%", label: "ABIDE II" },
      { value: "-25%", label: "compute" },
    ],
    tech: ["MATLAB", "Swin Transformer", "CNN", "Quantum SVM", "QNN"],
    link: null,
  },
];

export interface Role {
  company: string;
  title: string;
  location: string;
  period: string;
  color: string;
  glyph: string;
  start: number;
  end: number;
  bullets: string[];
  stack: string[];
}

export const ROLES: Role[] = [
  {
    company: "Revmo AI",
    title: "AI Software Engineer (Forward-Deployed)",
    location: "Phoenix, AZ",
    period: "May 2026 — Present",
    color: ACCENT.inference,
    glyph: "R",
    start: 2026.33,
    end: 2026.75,
    bullets: [
      "Embedded as the forward-deployed engineer across client engagements — ran discovery with business stakeholders, turned pain points into technical specs, and delivered agentic systems from design through production.",
      "Designed a multi-agent claims analysis system on LangGraph with specialized evidence-review, compliance-validation, and decision-letter agents plus human-in-the-loop approval — cutting review turnaround 80% and lifting client claims revenue 92%.",
      "Built a taxation-client automation platform where scheduled agents pull server-side data, push structured outputs to SharePoint, and auto-generate recurring transcript reports, replacing a fully manual cycle.",
      "Converted high-volume unstructured client data into ETL pipelines feeding a React 3D visualization portal for non-technical stakeholders.",
      "Shipped 8+ production agentic systems with tool-calling into POS, CRM, and payment platforms via AWS Bedrock — 50,000+ calls/month at 99.2% uptime, released through GitHub Actions CI/CD with canary and blue-green rollout, automated rollback, and distributed tracing.",
      "Benchmarked vLLM vs. SGLang on throughput, latency, and cost: 3.2x throughput, 850ms → 240ms; TensorRT/ONNX INT8/FP16 quantization on Qwen2.5-7B for 47% cost reduction at under 2% accuracy loss.",
    ],
    stack: [
      "LangGraph",
      "LangChain",
      "vLLM",
      "SGLang",
      "TensorRT",
      "AWS Bedrock",
      "GitHub Actions",
      "React",
    ],
  },
  {
    company: "Wolters Kluwer — Legal & Regulatory",
    title: "Data Scientist",
    location: "New York, NY",
    period: "May 2025 — Dec 2025",
    color: ACCENT.agents,
    glyph: "WK",
    start: 2025.33,
    end: 2025.99,
    bullets: [
      "Designed and deployed a multi-agent reporting system on LangGraph pipelines, turning enterprise legal data into conversational dashboard insights and boosting reporting efficiency 22%.",
      "Built and validated RAG preprocessing and retrieval over unstructured legal documents; benchmarked BERT, RoBERTa, and T5 to improve factual accuracy 85%.",
      "Engineered containerized FastAPI microservices with PySpark and MongoDB workflows and Azure DevOps CI/CD, reducing manual escalations 41%.",
      "Automated SMTP-to-OneDrive ingestion and event-triggered workflows via Microsoft Graph API: 95% better ingestion reliability, 42% latency reduction, with production monitoring and alerting.",
    ],
    stack: ["LangGraph", "RAG", "FastAPI", "PySpark", "MongoDB", "Azure DevOps"],
  },
  {
    company: "Centre for Cyber-Physical Systems, VIT",
    title: "Research Scientist — Biomedical & Neuroinformatics",
    location: "Chennai, India",
    period: "May 2023 — May 2024",
    color: ACCENT.research,
    glyph: "VIT",
    start: 2023.33,
    end: 2024.33,
    bullets: [
      "Engineered EEG and fMRI preprocessing pipelines using Spatio-Spectral Decomposition in MATLAB with custom deep learning architectures.",
      "Evaluated hybrid deep-learning/quantum architectures (Swin Transformers, CNNs, Quantum SVM/QNN) for autism detection from fMRI: 98.17% on ABIDE I, 96.2% on ABIDE II, with 25% less compute time.",
    ],
    stack: ["MATLAB", "PyTorch", "Swin Transformer", "Quantum SVM", "SSD"],
  },
];

export interface SkillGroup {
  name: string;
  color: string;
  icon: keyof typeof ICONS;
  items: string;
}

type IconShape =
  | { t: "rect"; x: number; y: number; w: number; h: number; rx: number }
  | { t: "circle"; cx: number; cy: number; r: number }
  | { t: "line"; x1: number; y1: number; x2: number; y2: number }
  | { t: "polyline"; points: string }
  | { t: "path"; d: string };

export const ICONS = {
  agents: [
    { t: "circle", cx: 12, cy: 6, r: 2.4 },
    { t: "circle", cx: 6, cy: 17, r: 2.4 },
    { t: "circle", cx: 18, cy: 17, r: 2.4 },
    { t: "line", x1: 12, y1: 8.4, x2: 7, y2: 15 },
    { t: "line", x1: 12, y1: 8.4, x2: 17, y2: 15 },
    { t: "line", x1: 8.4, y1: 17, x2: 15.6, y2: 17 },
  ],
  chip: [
    { t: "rect", x: 7, y: 7, w: 10, h: 10, rx: 2 },
    { t: "line", x1: 12, y1: 3, x2: 12, y2: 7 },
    { t: "line", x1: 12, y1: 17, x2: 12, y2: 21 },
    { t: "line", x1: 3, y1: 12, x2: 7, y2: 12 },
    { t: "line", x1: 17, y1: 12, x2: 21, y2: 12 },
  ],
  cloudai: [
    { t: "path", d: "M7 18h10a3.5 3.5 0 0 0 .3-6.98A5 5 0 0 0 7.6 11.2A3.4 3.4 0 0 0 7 18z" },
    { t: "circle", cx: 12, cy: 14, r: 1.4 },
  ],
  code: [
    { t: "polyline", points: "8,8 4,12 8,16" },
    { t: "polyline", points: "16,8 20,12 16,16" },
    { t: "line", x1: 13.5, y1: 6, x2: 10.5, y2: 18 },
  ],
  shield: [
    { t: "path", d: "M12 3l7 3.2v5.3c0 4.4-3 7.5-7 8.5-4-1-7-4.1-7-8.5V6.2L12 3z" },
    { t: "polyline", points: "9,12 11.4,14.4 15,10.6" },
  ],
  pipeline: [
    { t: "circle", cx: 6, cy: 7, r: 2.2 },
    { t: "circle", cx: 18, cy: 7, r: 2.2 },
    { t: "circle", cx: 12, cy: 18, r: 2.2 },
    { t: "path", d: "M6 9.2v2.3a2.5 2.5 0 0 0 2.5 2.5h1" },
    { t: "path", d: "M18 9.2v2.3a2.5 2.5 0 0 1-2.5 2.5h-1" },
  ],
  api: [
    { t: "rect", x: 3, y: 5, w: 18, h: 14, rx: 2 },
    { t: "line", x1: 3, y1: 9.5, x2: 21, y2: 9.5 },
    { t: "circle", cx: 6.5, cy: 7.2, r: 0.8 },
    { t: "polyline", points: "8,14 10,16 8,18" },
    { t: "line", x1: 12, y1: 18, x2: 16, y2: 18 },
  ],
  db: [
    { t: "path", d: "M4 6.5c0-1.4 3.6-2.5 8-2.5s8 1.1 8 2.5-3.6 2.5-8 2.5-8-1.1-8-2.5z" },
    { t: "path", d: "M4 6.5v11c0 1.4 3.6 2.5 8 2.5s8-1.1 8-2.5v-11" },
    { t: "path", d: "M4 12c0 1.4 3.6 2.5 8 2.5s8-1.1 8-2.5" },
  ],
  monitor: [
    { t: "rect", x: 3, y: 4, w: 18, h: 13, rx: 2 },
    { t: "polyline", points: "7,12 10,9 12.5,11.5 17,7" },
    { t: "line", x1: 9, y1: 20, x2: 15, y2: 20 },
  ],
} satisfies Record<string, IconShape[]>;

export function shapePath(shape: IconShape, key: number) {
  switch (shape.t) {
    case "rect":
      return {
        key,
        tag: "rect" as const,
        props: { x: shape.x, y: shape.y, width: shape.w, height: shape.h, rx: shape.rx },
      };
    case "circle":
      return { key, tag: "circle" as const, props: { cx: shape.cx, cy: shape.cy, r: shape.r } };
    case "line":
      return {
        key,
        tag: "line" as const,
        props: { x1: shape.x1, y1: shape.y1, x2: shape.x2, y2: shape.y2 },
      };
    case "polyline":
      return { key, tag: "polyline" as const, props: { points: shape.points } };
    case "path":
      return { key, tag: "path" as const, props: { d: shape.d } };
  }
}

export const SKILL_GROUPS: SkillGroup[] = [
  {
    name: "Agentic AI & Agent Skills",
    color: ACCENT.agents,
    icon: "agents",
    items:
      "Agent Skills, MCP, Agent SDK, LangChain, LangGraph, multi-agent systems, autonomous agents, copilots, orchestration (routing, planning, task execution), tool/function calling, ReAct, planner-executor, CoT, RAG, prompt engineering (Jinja2, few-shot)",
  },
  {
    name: "ML & Inference Optimization",
    color: ACCENT.inference,
    icon: "chip",
    items:
      "PyTorch, TensorFlow, Scikit-Learn, XGBoost, HF Transformers, Vision/Swin Transformers, CNNs, MLflow, vLLM, SGLang, TensorRT, Triton, ONNX Runtime, CUDA, INT4/INT8/FP16 quantization",
  },
  {
    name: "AI Services & Foundations",
    color: ACCENT.agents,
    icon: "cloudai",
    items:
      "Anthropic Claude, OpenAI, AWS Bedrock, Vertex AI & Gemini, Azure OpenAI, transformers, embeddings, deep learning, NLP, computer vision, RL (PPO, GRPO), hallucination mitigation, fine-tuning (LoRA/QLoRA, PEFT)",
  },
  {
    name: "Languages & Engineering",
    color: ACCENT.research,
    icon: "code",
    items:
      "Python, TypeScript, JavaScript, Node.js, SQL, C++, C, R, MATLAB, Git, Linux, system design & tradeoff analysis, code review, technical specifications",
  },
  {
    name: "Quality & Eval-Driven Dev",
    color: ACCENT.security,
    icon: "shield",
    items:
      "Unit & integration testing, AI-assisted test generation, pytest, regression automation, agent evaluation frameworks, guardrails, benchmarking, A/B testing, root-cause analysis",
  },
  {
    name: "CI/CD & Developer Platform",
    color: ACCENT.applied,
    icon: "pipeline",
    items:
      "GitHub Actions, Azure DevOps, deployment tooling, self-service infrastructure, templates & scaffolding, Terraform & IaC, canary/blue-green/shadow deploys, automated rollback",
  },
  {
    name: "Backend, APIs & Front End",
    color: ACCENT.inference,
    icon: "api",
    items:
      "FastAPI, Django, REST API design, GraphQL, gRPC, WebSockets, microservices, React.js, Next.js, Tailwind, Streamlit, full-stack delivery",
  },
  {
    name: "Data Schemas & Databases",
    color: ACCENT.applied,
    icon: "db",
    items:
      "PostgreSQL, MySQL, Redis, Kafka, MongoDB, Neo4j, FAISS, schema design, indexing & query optimization, ETL, PySpark, Airflow, Databricks",
  },
  {
    name: "Cloud, Containers & Monitoring",
    color: ACCENT.research,
    icon: "monitor",
    items:
      "AWS (Bedrock, SageMaker, EC2, EKS/ECS, Fargate, Lambda, S3, SQS), GCP (Vertex AI, GKE, Cloud SQL), Docker, Kubernetes, serverless, autoscaling, Prometheus, Grafana, CloudWatch, distributed tracing",
  },
];

export const HEADLINE_METRICS = [
  { value: "3.2x", label: "LLM serving throughput (vLLM vs SGLang)", color: ACCENT.inference },
  { value: "−47%", label: "inference cost after INT8/FP16 quantization", color: ACCENT.research },
  { value: "50k+", label: "agent calls / month at 99.2% uptime", color: ACCENT.agents },
  { value: "88%", label: "prompt-injection attack recall", color: ACCENT.security },
];

export const ABOUT_NUMBERS = [
  { value: "3.2x", label: "serving throughput lift, 850ms → 240ms", color: ACCENT.inference },
  {
    value: "−47%",
    label: "inference cost via quantization at <2% accuracy loss",
    color: ACCENT.research,
  },
  { value: "50k+", label: "production agent calls per month", color: ACCENT.agents },
  { value: "88%", label: "attack recall on injection guardrails", color: ACCENT.security },
  { value: "98.17%", label: "ABIDE I accuracy, autism detection research", color: ACCENT.applied },
];

export const EDUCATION = [
  {
    school: "Arizona State University",
    logo: "/asu-seal.png",
    degree: "MS Data Science, Analytics & Engineering",
    period: "Arizona State University · Tempe, AZ · May 2026",
  },
  {
    school: "Vellore Institute of Technology",
    logo: "/vit-seal.png",
    degree: "B.Tech Electronics & Communication",
    period: "Vellore Institute of Technology · Chennai · May 2024",
  },
];

export const RESUME_URL = "/resume_new.pdf";
export const GITHUB_URL = "https://github.com/aneeshjayan";
export const LINKEDIN_URL = "https://www.linkedin.com/in/aneeshjayan/";
export const EMAIL = "aneeshjayan11@gmail.com";

export const FILTERS: { key: "all" | Category; label: string; color: string }[] = [
  { key: "all", label: "--all", color: ACCENT.research },
  { key: "agents", label: "--agents", color: ACCENT.agents },
  { key: "inference", label: "--inference", color: ACCENT.inference },
  { key: "security", label: "--security", color: ACCENT.security },
  { key: "applied", label: "--applied", color: ACCENT.applied },
  { key: "research", label: "--research", color: ACCENT.research },
];
