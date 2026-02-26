"""
Portfolio Knowledge Base — chunked documents for RAG retrieval.
Each chunk covers one semantic topic about Aneesh Jayan Prabhu's portfolio.
"""

DOCUMENTS = [
    {
        "id": "personal_info",
        "category": "personal",
        "text": (
            "Aneesh Jayan Prabhu is an AI/ML Engineer and Data Scientist based in Phoenix, Arizona, USA. "
            "He is currently open to AI/ML Engineering and Data Science opportunities. "
            "Contact: Email aneeshjayan11@gmail.com, Phone (602) 768-6622. "
            "LinkedIn: linkedin.com/in/aneeshjayan | GitHub: github.com/aneeshjayan. "
            "He specialises in Generative AI, RAG Pipelines, LLM fine-tuning, and Production ML systems."
        ),
    },
    {
        "id": "about_summary",
        "category": "about",
        "text": (
            "Aneesh Jayan Prabhu is a passionate AI/ML engineer with hands-on experience building "
            "production-grade Generative AI systems, RAG pipelines, and LLM-powered applications. "
            "He combines deep academic grounding (MS Data Science at ASU) with real-world industry "
            "experience (Data Science Intern at Wolters Kluwer). His core strengths include RAG, "
            "fine-tuning (LoRA, QLoRA, PEFT), mechanistic interpretability, MoE architectures, and "
            "end-to-end ML deployment on AWS and Azure. He is equally comfortable with healthcare AI, "
            "FinTech AI, and general-purpose LLM applications."
        ),
    },
    {
        "id": "education_asu",
        "category": "education",
        "text": (
            "Arizona State University (ASU) — Master of Science in Data Science, Analytics and Engineering. "
            "Graduating May 2026, Tempe, Arizona. "
            "Relevant courses: Statistics for Data Analysis, Data Processing at Scale, Statistical Machine Learning, "
            "Semantic Web Mining, Data Mining, Knowledge Representation, Computing for Data-Driven Optimization. "
            "ASU is a top-ranked research university in the US known for innovation and engineering programs."
        ),
    },
    {
        "id": "education_vit",
        "category": "education",
        "text": (
            "Vellore Institute of Technology (VIT) — B.Tech in Electronics and Communication Engineering. "
            "Graduated May 2024, Chennai, India. "
            "Key achievement: 98.17% model accuracy in fMRI-based Autism detection research, "
            "published in peer-reviewed conferences. Research used hybrid deep learning and quantum "
            "computing frameworks. CGPA reflects strong academic foundation in signal processing and ML."
        ),
    },
    {
        "id": "experience_cips_lab",
        "category": "experience",
        "text": (
            "Project Volunteer — Legal AI | CIPS Lab, Arizona State University (Feb 2026 – May 2026, Tempe AZ). "
            "Building an AI-powered Legal Assistant that predicts case victory probability, surfaces "
            "Explainable AI (XAI)-backed reasoning, implements a RAG pipeline over statutes and case law, "
            "and provides an act/law interpreter for plain-language legal explanations. "
            "Stack: RAG, Explainable AI (XAI), LLMs, Python, LangChain, FAISS."
        ),
    },
    {
        "id": "experience_soda_mentor",
        "category": "experience",
        "text": (
            "Mentor | Software Developers Association (SoDA), ASU (Feb 2026 – May 2026). "
            "Mentoring students on AI/ML projects, software architecture design, code review, and career preparation. "
            "Running Python, ML, and deployment workshops to help students break into the tech industry."
        ),
    },
    {
        "id": "experience_wolters_kluwer",
        "category": "experience",
        "text": (
            "Data Science Intern | Wolters Kluwer — Legal & Regulatory Division (May 2025 – Dec 2025, New York). "
            "Built LangGraph pipelines for semantic dashboard insights processing 50K+ daily records with 95% reliability. "
            "Reduced manual escalations by 41% via FastAPI microservices integrating SQL, MongoDB, and Azure DevOps CI/CD. "
            "Developed RAG preprocessing for unstructured legal documents including flowchart and table transcription. "
            "Benchmarked BERT, RoBERTa, and T5 models in RAG pipelines, achieving 85% factual accuracy improvement. "
            "Achieved 22% efficiency boost via automated SMTP alerts and 42% latency reduction through optimization. "
            "Stack: LangGraph, FastAPI, Azure DevOps, RAG, BERT, RoBERTa, T5, MongoDB, SQL, Python."
        ),
    },
    {
        "id": "experience_vit_research",
        "category": "experience",
        "text": (
            "Research Intern — Biomedical & Neuroinformatics | Centre for Cyber-Physical Systems, VIT (May 2023 – May 2024, Chennai). "
            "Developed a hybrid deep learning and quantum computing framework for Autism Spectrum Disorder detection from fMRI brain scans. "
            "Achieved 98.17% accuracy on ABIDE I dataset and 96.2% accuracy on ABIDE II dataset. "
            "Reduced computation by 25% using Swin Transformers, CNNs, Quantum SVM and QNN approaches. "
            "EEG and fMRI signal preprocessing using Source Space Decomposition (SSD) in MATLAB. "
            "Stack: PyTorch, MATLAB, CNN, Transformers, Quantum ML, fMRI, EEG, Signal Processing."
        ),
    },
    {
        "id": "project_trustmedai",
        "category": "project",
        "text": (
            "TrustMedAI: Medical Conversational Agent. "
            "A RAG-based medical Q&A system for Type-2 Diabetes information, making clinical knowledge accessible. "
            "Processed 500+ patient forum threads and 16,000+ lines of clinical guidelines from ADA, Mayo Clinic, and NIH. "
            "Uses MiniLM embeddings and FAISS vector database for retrieval. "
            "Multimodal interface with speech-to-text and text-to-speech (TTS) for accessibility. "
            "Metrics: Precision 0.950, Recall 0.920, Faithfulness 0.970, Semantic Similarity 0.888. "
            "Stack: Python, FAISS, React, MiniLM, RAG, Healthcare AI, NLP."
        ),
    },
    {
        "id": "project_vlm_speedup",
        "category": "project",
        "text": (
            "VLM Speedup: LexFin Guard — Vision-Language Model acceleration for financial document processing. "
            "Uses Mixture-of-Experts (MoE) routing directing image regions to specialized Table and OCR experts. "
            "Confidence-based early exit at Layer 10 or Layer 17 for simple document layouts, saving ~50% compute. "
            "Auto-validates extracted financial data by verifying Subtotal + Tax == Total. "
            "Results: 3.5x throughput (14 docs/sec vs 4 docs/sec baseline), 96% cost reduction, 94% accuracy, ~250ms latency per doc. "
            "Stack: Python, MoE, PyTorch, Streamlit, HuggingFace, FinTech."
        ),
    },
    {
        "id": "project_optimal_slm",
        "category": "project",
        "text": (
            "Optimal-SLM: Reasoning and Prompt Optimization Framework. "
            "Dual-agent system using Qwen2-1.5B and Phi-3.5-Mini models, PEFT fine-tuned on Alpaca and OpenOrca datasets. "
            "Implements chain-of-thought reasoning, agent-to-agent (A2A) coordination, and confidence scoring. "
            "Achieves 76% token reduction while maintaining quality above 0.8. "
            "Deployed with Docker and CI/CD on AWS SageMaker. "
            "Stack: Python, AWS SageMaker, Docker, PEFT, LoRA, LLMs."
        ),
    },
    {
        "id": "project_llm_probing",
        "category": "project",
        "text": (
            "LLM Probing: Mechanistic Interpretability Study of Language Models. "
            "Investigated how StableLM-Tuned-Alpha-3B (3 billion parameters) encodes honest vs dishonest framing. "
            "Performed layer-wise hidden state extraction and trained logistic regression probes achieving ~100% accuracy on deep layers. "
            "Used PCA visualization and cosine similarity analysis for interpretability. "
            "Key finding: early transformer layers encode honest and dishonest framing identically; "
            "deeper layers show asymmetric divergence indicating where truthfulness is represented. "
            "Stack: Python, PyTorch, HuggingFace, StableLM, PCA, AI Safety."
        ),
    },
    {
        "id": "project_ai_video_editor",
        "category": "project",
        "text": (
            "AI Video Editor Agent — Multi-agent system for editing videos using plain English commands. "
            "Built with CrewAI framework, contains 6 specialized agents: "
            "Audio Intelligence, Scene Detection, Clip Trimming, Narrative Structuring, "
            "Subtitle Generation (using Whisper), and Platform Adaptation. "
            "Smart pipeline routing: uses 2-agent, 3-agent, or 6-agent pipelines based on task complexity. "
            "Fully offline operation via Ollama (no cloud API required). "
            "Stack: Python, CrewAI, GPT-4o, Whisper, FFmpeg, FastAPI, Ollama."
        ),
    },
    {
        "id": "project_finslm",
        "category": "project",
        "text": (
            "FinSLM: Financial Small Language Model fine-tuned for financial analysis. "
            "Fine-tuned Mistral-7B on SEC EDGAR 10-K and 10-Q filings from 20+ companies, "
            "sentiment-labeled financial news, and yfinance market fundamental data. "
            "Uses QLoRA quantization: requires minimum 2-3 GB VRAM while achieving ~98% of full fine-tuning quality. "
            "Covers financial concepts: P/E ratios, DCF valuation, EPS analysis, SEC filing interpretation. "
            "Stack: Python, Mistral-7B, LoRA, QLoRA, SEC EDGAR, Weights & Biases (W&B), FinTech."
        ),
    },
    {
        "id": "project_focusmate",
        "category": "project",
        "text": (
            "FocusMate – AI Co-Pilot for Executive Function (ADHD support app). "
            "Built for the Spark Challenge Hackathon, November 2025. "
            "Converts Gmail emails, Google Calendar events, and voice notes into prioritized tasks, "
            "email summaries, and time-blocked daily plans with 90-minute focus blocks for ADHD users. "
            "Features React (Vite) web dashboard, Expo mobile app, 6+ REST API endpoints, and 2 FastAPI microservices. "
            "Stack: Python, FastAPI, React, Google OAuth2, Gmail API, Google Calendar API, LLMs."
        ),
    },
    {
        "id": "skills",
        "category": "skills",
        "text": (
            "Aneesh's technical skills: "
            "Programming Languages: Python (Expert), SQL (Advanced), JavaScript (Intermediate), R (Advanced), C++ (Intermediate), MATLAB (Advanced). "
            "ML/DL Frameworks: PyTorch (Expert), TensorFlow (Advanced), Scikit-Learn, HuggingFace Transformers, LangChain (Expert), LangGraph (Expert). "
            "AI Techniques: RAG (Retrieval-Augmented Generation), Fine-tuning with LoRA/QLoRA/PEFT, Mechanistic Interpretability, Mixture-of-Experts (MoE), Model Optimization, Chain-of-Thought Reasoning. "
            "Web Frameworks: FastAPI (Advanced), React (Intermediate), CrewAI, Streamlit. "
            "Cloud Platforms: Azure (Azure ML, Azure DevOps), AWS (SageMaker, Lambda, S3). "
            "DevOps: Docker, Kubernetes, CI/CD pipelines, Git. "
            "Databases: PostgreSQL, MongoDB, Neo4j, FAISS vector database. "
            "Domain Expertise: Generative AI, Production ML Systems, Healthcare AI, FinTech AI, Distributed Systems, Model Optimization."
        ),
    },
    {
        "id": "contact",
        "category": "contact",
        "text": (
            "How to contact Aneesh Jayan Prabhu: "
            "Email: aneeshjayan11@gmail.com. "
            "Phone: (602) 768-6622. "
            "LinkedIn: linkedin.com/in/aneeshjayan. "
            "GitHub: github.com/aneeshjayan. "
            "Location: Phoenix, Arizona, USA. "
            "He is currently open to full-time AI/ML Engineering and Data Science roles. "
            "You can also use the contact form on this portfolio website to send him a message directly."
        ),
    },
]
