// ===================================
// CONFIGURATION — FILL IN YOUR KEYS
// ===================================

// 1. EmailJS: Create account at https://emailjs.com → Email Services → Add Service
//    Then Manage → Email Templates → Create Template with variables:
//    {{from_name}}, {{reply_to}}, {{subject}}, {{message}}, {{to_name}}
const EMAILJS_CONFIG = {
    publicKey:  'LHc4kEBq7ngX9uO-D',   // Account → General → Public Key
    serviceId:  'service_2ium49l',            // Email Services tab
    templateId: 'template_hjyx26k'           // Email Templates tab
};

// 2. Gemini API: Get free key at https://aistudio.google.com/app/apikey
const GEMINI_API_KEY = 'AIzaSyBlzWgEZc21i9N3pILlMrH1sgVlBd6rUC0';

// ===================================
// PORTFOLIO CONTEXT (for RAG bot)
// ===================================
const PORTFOLIO_CONTEXT = `
You are the AI assistant on Aneesh Jayan Prabhu's portfolio website. Your job is to answer questions about Aneesh in a helpful, concise, and professional tone. Use HTML formatting like <strong>, <br>, bullet points (•) when it improves readability.

=== PERSONAL INFO ===
Name: Aneesh Jayan Prabhu
Email: aneeshjayan11@gmail.com | Phone: (602) 768-6622
Location: Phoenix, Arizona, USA
LinkedIn: linkedin.com/in/aneeshjayan | GitHub: github.com/aneeshjayan
Status: Currently open to AI/ML Engineering opportunities

=== EDUCATION ===
1. Arizona State University (ASU) — MS in Data Science, Analytics and Engineering (Graduating May 2026, Tempe AZ)
   Courses: Statistics for Data Analysis, Data Processing At Scale, Statistical Machine Learning, Semantic Web Mining, Data Mining, Knowledge Representation, Computing for Data-Driven Optimization

2. Vellore Institute of Technology (VIT) — B.Tech in Electronics and Communication Engineering (May 2024, Chennai India)
   Achievements: 98.17% model accuracy in fMRI Autism detection research, published in peer-reviewed conferences

=== EXPERIENCE ===
1. Project Volunteer — Legal AI | CIPS Lab, Arizona State University (Feb 2026 – May 2026)
   Building Legal AI assistant: predicts case victory probability, surfaces XAI-backed reasoning, RAG pipeline over statutes/case law, act/law interpreter for plain language explanations.
   Stack: RAG, Explainable AI (XAI), LLMs, Python, LangChain, FAISS

2. Mentor | Software Developers Association (SoDA), ASU (Feb 2026 – May 2026)
   Mentoring students on AI/ML projects, architecture design, code review, career prep, Python/ML/deployment workshops.

3. Data Science Intern | Wolters Kluwer — Legal & Regulatory (May 2025 – Dec 2025, New York)
   - LangGraph pipelines for semantic dashboard insights, 50K+ daily records, 95% reliability
   - Reduced manual escalations 41% via FastAPI microservices (SQL + MongoDB + Azure DevOps CI/CD)
   - RAG preprocessing for unstructured legal documents (flowchart/table transcription)
   - Benchmarked BERT/RoBERTa/T5 in RAG → 85% factual accuracy improvement
   - 22% efficiency boost via automated SMTP alerts; 42% latency reduction through optimization
   Stack: LangGraph, FastAPI, Azure DevOps, RAG, BERT, RoBERTa, T5, MongoDB, SQL, Python

4. Research Intern — Biomedical & Neuroinformatics | Centre for Cyber-Physical Systems, VIT (May 2023 – May 2024, Chennai)
   - Hybrid deep learning-quantum framework for Autism detection from fMRI
   - 98.17% accuracy (ABIDE I), 96.2% accuracy (ABIDE II), 25% computation reduction
   - Used Swin Transformers, CNNs, Quantum SVM/QNN; EEG/fMRI preprocessing with SSD in MATLAB
   Stack: PyTorch, MATLAB, CNN, Transformers, Quantum ML, fMRI, EEG, Signal Processing

=== PROJECTS ===
1. TrustMedAI: Medical Conversational Agent
   RAG-based medical Q&A for Type-2 Diabetes. Processed 500+ forum threads, 16K+ lines of clinical guidelines (ADA, Mayo Clinic, NIH). MiniLM embeddings + FAISS. Multimodal (speech-to-text + TTS).
   Metrics: Precision 0.950, Recall 0.920, Faithfulness 0.970, Similarity 0.888
   Stack: Python, FAISS, React, MiniLM, RAG, Healthcare AI, NLP

2. VLM Speedup: LexFin Guard
   Accelerates Vision-Language Models for financial document processing. MoE routing (Table/OCR experts) + confidence-based early exit (exits at Layer 10/17 for simple layouts, ~50% compute saved). Auto-validates extracted data (Subtotal + Tax == Total).
   Metrics: 3.5x throughput (14 docs/sec vs 4 baseline), 96% cost reduction, 94% accuracy, ~250ms latency/doc
   Stack: Python, MoE, PyTorch, Streamlit, HuggingFace, FinTech

3. Optimal-SLM: Reasoning and Prompt Optimization Framework
   Dual-agent (Qwen2-1.5B + Phi-3.5-Mini) PEFT fine-tuned on Alpaca + OpenOrca. Chain-of-thought reasoning, A2A coordination, confidence scoring. 76% token reduction, >0.8 quality. Docker + CI/CD on AWS SageMaker.
   Stack: Python, AWS SageMaker, Docker, PEFT, LoRA, LLMs

4. LLM Probing: Mechanistic Interpretability Study
   Investigated how StableLM-Tuned-Alpha-3B (3B params) encodes "honest" vs "dishonest" framing. Layer-wise hidden states, logistic regression probes (~100% accuracy on deep layers), PCA visualization, cosine similarity. Findings: early layers identical, deeper layers show asymmetric divergence.
   Stack: Python, PyTorch, HuggingFace, StableLM, PCA, AI Safety

5. AI Video Editor Agent
   Multi-agent system (CrewAI) for editing videos via plain English. 6 specialized agents: Audio Intelligence, Scene Detection, Clip Trimming, Narrative Structuring, Subtitle Generation (Whisper), Platform Adaptation. Smart 2/3/6-agent pipeline routing. Fully offline via Ollama.
   Stack: Python, CrewAI, GPT-4o, Whisper, FFmpeg, FastAPI, Ollama

6. FinSLM: Financial Small Language Model
   Fine-tuned Mistral-7B on SEC EDGAR 10-K/10-Q filings (20+ companies), sentiment-labeled financial news, yfinance market fundamentals. QLoRA: min 2-3 GB VRAM, ~98% of full FT quality. Covers P/E, DCF, EPS, SEC interpretation.
   Stack: Python, Mistral-7B, LoRA, QLoRA, SEC EDGAR, W&B, FinTech

7. FocusMate – AI Co-Pilot for Executive Function
   ADHD co-pilot (spark Challenge Hackathon, Nov 2025). Converts Gmail + Google Calendar + voice notes into prioritized tasks, email summaries, time-blocked daily plans (90-min focus blocks). React (Vite) dashboard + Expo mobile + 6+ REST endpoints + 2 FastAPI services.
   Stack: Python, FastAPI, React, Google OAuth2, Gmail API, Calendar API, LLMs

=== SKILLS ===
Languages: Python (Expert), SQL (Advanced), JavaScript (Intermediate), R (Advanced), C++ (Intermediate), MATLAB (Advanced)
ML/DL: PyTorch (Expert), TensorFlow (Advanced), Scikit-Learn, HuggingFace, LangChain (Expert), LangGraph (Expert)
Techniques: RAG, Fine-tuning (LoRA, QLoRA, PEFT), Mechanistic Interpretability, MoE, Model Optimization
Frameworks: FastAPI (Advanced), React (Intermediate), CrewAI, Streamlit
Cloud: Azure (Azure ML, Azure DevOps), AWS (SageMaker, Lambda, S3)
DevOps: Docker, Kubernetes, CI/CD, Git
Databases: PostgreSQL, MongoDB, Neo4j, FAISS
Domain expertise: Generative AI, Production ML, Healthcare AI, FinTech AI, Distributed Systems, Model Optimization

Keep answers focused and helpful. If asked something unrelated to Aneesh, politely redirect to portfolio topics.
`;

// ===================================
// NAVIGATION & SCROLL
// ===================================

const navbar = document.getElementById('navbar');
const navLinks = document.querySelectorAll('.nav-link');
const mobileMenuToggle = document.getElementById('mobileMenuToggle');
const navMenu = document.getElementById('navMenu');

// Navbar scroll effect
window.addEventListener('scroll', () => {
    if (window.scrollY > 50) {
        navbar.classList.add('scrolled');
    } else {
        navbar.classList.remove('scrolled');
    }
    
    // Update active nav link
    updateActiveNavLink();
});

// Mobile menu toggle
mobileMenuToggle.addEventListener('click', () => {
    navMenu.classList.toggle('active');
});

// Close mobile menu when clicking a link
navLinks.forEach(link => {
    link.addEventListener('click', () => {
        navMenu.classList.remove('active');
    });
});

// Update active nav link based on scroll position
function updateActiveNavLink() {
    const sections = document.querySelectorAll('.section');
    const scrollPos = window.scrollY + 100;
    
    sections.forEach(section => {
        const sectionTop = section.offsetTop;
        const sectionHeight = section.offsetHeight;
        const sectionId = section.getAttribute('id');
        
        if (scrollPos >= sectionTop && scrollPos < sectionTop + sectionHeight) {
            navLinks.forEach(link => {
                link.classList.remove('active');
                if (link.getAttribute('href') === `#${sectionId}`) {
                    link.classList.add('active');
                }
            });
        }
    });
}

// ===================================
// THEME TOGGLE
// ===================================

const themeToggle = document.getElementById('themeToggle');
const body = document.body;

// Check for saved theme preference
const savedTheme = localStorage.getItem('theme');
if (savedTheme === 'light') {
    body.classList.add('light-mode');
    themeToggle.textContent = '☀️';
}

themeToggle.addEventListener('click', () => {
    body.classList.toggle('light-mode');
    const isLight = body.classList.contains('light-mode');
    themeToggle.textContent = isLight ? '☀️' : '🌙';
    localStorage.setItem('theme', isLight ? 'light' : 'dark');
});

// ===================================
// TYPING EFFECT
// ===================================

const typingText = document.getElementById('typingText');
const skills = [
    'Generative AI',
    'RAG Pipelines',
    'Production ML',
    'LLMs',
    'Fine-tuning',
    'Prompt Engineering',
    'Distributed Systems',
    'Model Optimization'
];

let skillIndex = 0;
let charIndex = 0;
let isDeleting = false;

function typeSkill() {
    const currentSkill = skills[skillIndex];
    
    if (isDeleting) {
        typingText.textContent = currentSkill.substring(0, charIndex - 1);
        charIndex--;
    } else {
        typingText.textContent = currentSkill.substring(0, charIndex + 1);
        charIndex++;
    }
    
    let typeSpeed = isDeleting ? 50 : 100;
    
    if (!isDeleting && charIndex === currentSkill.length) {
        typeSpeed = 2000;
        isDeleting = true;
    } else if (isDeleting && charIndex === 0) {
        isDeleting = false;
        skillIndex = (skillIndex + 1) % skills.length;
        typeSpeed = 500;
    }
    
    setTimeout(typeSkill, typeSpeed);
}

// Start typing effect
typeSkill();

// ===================================
// STATS COUNTER ANIMATION
// ===================================

const statNumbers = document.querySelectorAll('.stat-number');
let statsAnimated = false;

function animateStats() {
    if (statsAnimated) return;
    
    const aboutSection = document.getElementById('about');
    const aboutPosition = aboutSection.getBoundingClientRect().top;
    const screenPosition = window.innerHeight / 1.3;
    
    if (aboutPosition < screenPosition) {
        statsAnimated = true;
        
        statNumbers.forEach(stat => {
            const target = parseFloat(stat.getAttribute('data-target'));
            const increment = target / 100;
            let current = 0;
            
            const updateCounter = () => {
                if (current < target) {
                    current += increment;
                    if (target % 1 !== 0) {
                        stat.textContent = current.toFixed(2);
                    } else {
                        stat.textContent = Math.ceil(current);
                    }
                    setTimeout(updateCounter, 20);
                } else {
                    if (target % 1 !== 0) {
                        stat.textContent = target.toFixed(2);
                    } else {
                        stat.textContent = target;
                    }
                }
            };
            
            updateCounter();
        });
    }
}

window.addEventListener('scroll', animateStats);

// ===================================
// TIMELINE EXPANSION
// ===================================

function toggleTimeline(button) {
    const timelineItem = button.closest('.timeline-item');
    timelineItem.classList.toggle('expanded');
    button.textContent = timelineItem.classList.contains('expanded') ? 'Show Less' : 'Show More';
}

// ===================================
// PROJECT FILTERS
// ===================================

const filterBtns = document.querySelectorAll('.filter-btn');
const projectCards = document.querySelectorAll('.project-card');

filterBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active class from all buttons
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        
        const filter = btn.getAttribute('data-filter');
        
        projectCards.forEach(card => {
            if (filter === 'all') {
                card.style.display = 'block';
                card.classList.add('fade-in');
            } else {
                const categories = card.getAttribute('data-category');
                if (categories.includes(filter)) {
                    card.style.display = 'block';
                    card.classList.add('fade-in');
                } else {
                    card.style.display = 'none';
                }
            }
        });
    });
});

// ===================================
// SKILLS TABS
// ===================================

const tabBtns = document.querySelectorAll('.tab-btn');
const tabContents = document.querySelectorAll('.tab-content');

tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        // Remove active class from all buttons and contents
        tabBtns.forEach(b => b.classList.remove('active'));
        tabContents.forEach(c => c.classList.remove('active'));
        
        // Add active class to clicked button
        btn.classList.add('active');
        
        // Show corresponding content
        const tabId = btn.getAttribute('data-tab');
        document.getElementById(tabId).classList.add('active');
    });
});

// ===================================
// PROJECT MODAL
// ===================================

const projectModal = document.getElementById('projectModal');
const modalBody = document.getElementById('modalBody');
const modalClose = document.getElementById('modalClose');

const projectDetails = {
    trustmedai: {
        title: 'TrustMedAI: Medical Conversational Agent for Type-2 Diabetes',
        description: `
            <h3>Overview</h3>
            <p>TrustMedAI is a production-ready medical conversational agent designed to make Type-2 Diabetes information more accessible and understandable. The system bridges the gap between patient forums and authoritative medical literature.</p>
            
            <h3>Technical Architecture</h3>
            <ul>
                <li><strong>Data Processing:</strong> Processed 500+ forum threads and 16,000 lines of clinical guidelines from ADA, Mayo Clinic, and NIH</li>
                <li><strong>Semantic Retrieval:</strong> Built using FAISS vector database with MiniLM embeddings for high-precision retrieval</li>
                <li><strong>RAG Pipeline:</strong> Developed production RAG pipeline with regex normalization, deduplication, and Key Recurring Themes (KRT) identification</li>
                <li><strong>Frontend:</strong> React-based multimodal interface with speech-to-text and text-to-speech capabilities</li>
            </ul>
            
            <h3>Key Results</h3>
            <ul>
                <li>Content Precision: <strong>0.950</strong></li>
                <li>Recall: <strong>0.920</strong></li>
                <li>Faithfulness: <strong>0.970</strong></li>
                <li>Semantic Similarity: <strong>0.888</strong></li>
            </ul>
            
            <h3>Impact</h3>
            <p>The system significantly improves healthcare information accessibility for diabetes patients, providing accurate, citation-backed responses that maintain medical accuracy while being understandable to non-experts.</p>
            
            <h3>Technologies</h3>
            <p>Python, FAISS, React, MiniLM Embeddings, RAG, NLP, Healthcare AI</p>
        `
    },
    vlmspeedup: {
        title: 'VLM Speedup: LexFin Guard — High-Performance Invoice Extraction',
        description: `
            <h3>Overview</h3>
            <p>LexFin Guard tackles the challenge of running large Vision-Language Models on high-volume financial documents (invoices, receipts). By replacing a standard dense model with a specialized MoE architecture and confidence-based early exits, the system achieves 3.5x throughput and 96% cost reduction.</p>

            <h3>Core Architecture</h3>
            <ul>
                <li><strong>MoE Routing:</strong> Dynamically routes image regions to specialized experts — Table Expert for dense rows, OCR Expert for sparse text — based on content type</li>
                <li><strong>Early Exit Strategy:</strong> Checks confidence at intermediate layers (Layer 10, 17). Simple layouts exit early, saving ~50% of compute. Only complex documents reach full depth (Layer 24)</li>
                <li><strong>Validation & Reconciliation:</strong> Automatically verifies extracted data (Subtotal + Tax == Total) for financial accuracy</li>
                <li><strong>Streamlit Dashboard:</strong> Interactive app for live invoice extraction demos</li>
            </ul>

            <h3>Performance Benchmarks</h3>
            <ul>
                <li>Throughput: <strong>~14 docs/sec</strong> vs 4 docs/sec baseline (3.5x)</li>
                <li>Latency: <strong>~250ms/doc</strong> vs ~800ms baseline</li>
                <li>Cost Reduction: <strong>96%</strong></li>
                <li>Accuracy: <strong>94%</strong> (vs 92% baseline)</li>
            </ul>

            <h3>Extracted Fields</h3>
            <p>Vendor, Total, Tax, Subtotal, Line Items — with automated reconciliation.</p>

            <h3>Technologies</h3>
            <p>Python, PyTorch, HuggingFace, MoE, Streamlit, PEFT/LoRA, FinTech</p>
        `
    },
    optimalslm: {
        title: 'Optimal-SLM: Dual-Model Prompt Optimizer',
        description: `
            <h3>Overview</h3>
            <p>Designed a dual-model SLM system pairing Qwen2-1.5B (reasoning) with Phi-3.5-Mini (optimization) in a coordinated Agent-to-Agent (A2A) pipeline, purpose-built to run within a strict 6 GB VRAM constraint.</p>

            <h3>Technical Architecture</h3>
            <ul>
                <li><strong>Dual-Agent A2A Pipeline:</strong> Qwen2-1.5B handles Chain-of-Thought intent reasoning; Phi-3.5-Mini handles prompt optimization — coordinated via cross-model quality assessment and confidence scoring</li>
                <li><strong>Fine-tuning:</strong> Both models fine-tuned with LoRA adapters on Alpaca, OpenOrca, and custom intent datasets</li>
                <li><strong>Intent Classification:</strong> 5 query types with >90% accuracy; key entity extraction and redundancy detection</li>
                <li><strong>Speech-to-Text:</strong> Preprocessing for verbose spoken input patterns</li>
                <li><strong>Production Deployment:</strong> Docker + CI/CD on AWS SageMaker; Streamlit dashboard for intent analysis and metrics</li>
            </ul>

            <h3>Key Results</h3>
            <ul>
                <li>Intent Classification Accuracy: <strong>>90%</strong></li>
                <li>Semantic Preservation: <strong>>85%</strong></li>
                <li>Token Reduction (speech): <strong>20–50%</strong></li>
                <li>Token Reduction (text): <strong>15–30%</strong></li>
                <li>Processing Window: <strong>&lt;2 seconds</strong></li>
                <li>Quality Score: <strong>&gt;0.8</strong></li>
            </ul>

            <h3>Technologies</h3>
            <p>Python, Qwen2-1.5B, Phi-3.5-Mini, LoRA, 4-bit Quantization (BitsAndBytes), PyTorch, asyncio, AWS SageMaker, Docker, Streamlit</p>
        `
    },
    probing: {
        title: 'LLM Probing: Mechanistic Interpretability Study',
        description: `
            <h3>Overview</h3>
            <p>Investigated how LLMs internally encode high-level contextual instructions — specifically the distinction between "honest" and "dishonest" behavioral framing — using interpretability techniques on StableLM-Tuned-Alpha-3B.</p>

            <h3>Methodology</h3>
            <ul>
                <li><strong>Controlled Dataset:</strong> Designed prompts where the underlying question was fixed but contextual framing varied (honest vs. dishonest)</li>
                <li><strong>Layer-wise Analysis:</strong> Extracted hidden state activations and attention maps across all network layers</li>
                <li><strong>Linear Probing:</strong> Logistic regression probes to test linear separability of the honesty signal</li>
                <li><strong>PCA Visualization:</strong> Visualized geometric structure of activation spaces across layers</li>
                <li><strong>Cosine Similarity:</strong> Quantified representational divergence between honest/dishonest contexts</li>
            </ul>

            <h3>Key Findings</h3>
            <ul>
                <li>Early layers showed <strong>nearly identical</strong> activations for both contexts</li>
                <li>Deeper layers exhibited <strong>strong asymmetric divergence</strong> — dishonest framings produced significantly higher representation variance</li>
                <li>Linear probes on deeper layers achieved <strong>near-perfect accuracy</strong>, confirming honesty signals become linearly decodable with depth</li>
            </ul>

            <h3>Impact</h3>
            <p>Contributes to mechanistic interpretability by demonstrating that behavioral alignment in LLMs leaves measurable geometric traces — relevant to AI safety, prompt engineering, and model auditing.</p>

            <h3>Technologies</h3>
            <p>Python, PyTorch, HuggingFace Transformers, StableLM-Tuned-Alpha-3B, Logistic Regression, PCA, Cosine Similarity</p>
        `
    },
    videoeditor: {
        title: 'AI Video Editor Agent',
        description: `
            <h3>Overview</h3>
            <p>Built a multi-agent AI-powered video editing system that lets users edit videos using plain English instructions — no video editing experience required. The system intelligently routes tasks through specialized agent pipelines depending on complexity.</p>

            <h3>Agent Architecture</h3>
            <ul>
                <li><strong>6 Specialized Agents:</strong> Audio intelligence, scene detection, clip trimming, narrative structuring, subtitle generation, and platform adaptation</li>
                <li><strong>Smart Pipeline Routing:</strong> Selects between 2-agent, 3-agent, or 6-agent workflows based on task complexity</li>
                <li><strong>Whisper Integration:</strong> Auto-subtitle generation and filler word detection</li>
                <li><strong>OpenCV Scene Detection:</strong> Automatic scene boundary detection</li>
                <li><strong>FFmpeg Engine:</strong> Core video processing for trimming, speed adjustment, subtitle burning, and reframing</li>
            </ul>

            <h3>Key Features</h3>
            <ul>
                <li>Natural language editing: trimming, silence removal, speed adjustment, subtitle burning, highlight reels, platform reframing</li>
                <li>Human-readable edit log after every operation</li>
                <li>Streamlit web UI for interactive use</li>
                <li>FastAPI REST backend with async job queuing</li>
                <li>Full local-first support via Ollama — runs offline without API keys</li>
            </ul>

            <h3>Technologies</h3>
            <p>Python, CrewAI, OpenAI GPT-4o, Ollama (llama3.2), OpenAI Whisper, FFmpeg, OpenCV, Streamlit, FastAPI, Pydantic</p>
        `
    },
    finslm: {
        title: 'FinSLM: Financial Small Language Model',
        description: `
            <h3>Overview</h3>
            <p>Fine-tuned Mistral-7B on a curated financial domain corpus to create a specialized language model capable of reasoning over SEC filings, answering financial questions, analyzing market data, and interpreting news sentiment.</p>

            <h3>Data Pipeline</h3>
            <ul>
                <li><strong>SEC EDGAR:</strong> 10-K and 10-Q filings for 20+ companies</li>
                <li><strong>Financial News:</strong> Scraped articles with sentiment labels</li>
                <li><strong>Market Fundamentals:</strong> yfinance data including P/E ratios, EPS, DCF analysis inputs</li>
            </ul>

            <h3>Training Framework</h3>
            <ul>
                <li><strong>4 Strategies:</strong> Full fine-tuning, LoRA, 8-bit quantization, and QLoRA — accommodating 2 GB to 40+ GB VRAM budgets</li>
                <li><strong>LoRA Adapters:</strong> Targeting attention projection layers</li>
                <li><strong>Monitoring:</strong> Weights & Biases for training metrics</li>
                <li><strong>Evaluation:</strong> Held-out test split covering P/E ratios, DCF, EPS, and SEC filing interpretation</li>
            </ul>

            <h3>Key Results</h3>
            <ul>
                <li>LoRA achieved <strong>~98%</strong> of full fine-tuning quality at a fraction of compute cost</li>
                <li>QLoRA enables training on consumer hardware with <strong>as little as 2–3 GB VRAM</strong></li>
                <li>Model deployable on commodity hardware — no cloud dependency required</li>
            </ul>

            <h3>Technologies</h3>
            <p>Python, Mistral-7B, HuggingFace Transformers, LoRA / QLoRA, BitsAndBytes, SEC EDGAR API, yfinance, Weights & Biases, PyTorch</p>
        `
    },
    focusmate: {
        title: 'FocusMate – AI Co-Pilot for Executive Function',
        description: `
            <h3>Overview</h3>
            <p>ADHD-focused AI co-pilot built for the spark Challenge Hackathon. Converts Gmail, Google Calendar, and voice notes into prioritized tasks, email summaries, and time-blocked daily plans with 90-minute break scheduling to reduce overwhelm and time-blindness.</p>

            <h3>System Architecture</h3>
            <ul>
                <li><strong>2 FastAPI Microservices</strong> (ports 8000/8001) with Google OAuth2 and Gmail/Calendar APIs</li>
                <li><strong>Email Intelligence:</strong> Ingests and caches emails, classifies action vs. informational, extracts due-dated next steps, enables natural-language inbox search</li>
                <li><strong>Calendar Integration:</strong> Google Calendar sync with 90-minute focus block scheduling and automatic break insertion</li>
                <li><strong>Voice Pipeline:</strong> Voice-to-task pipeline with speech-to-text preprocessing</li>
            </ul>

            <h3>Frontend</h3>
            <ul>
                <li>React (Vite) planning dashboard</li>
                <li>Optional Expo mobile client</li>
                <li>6+ REST endpoints: refresh, search, summaries, STT, task triage, planning</li>
            </ul>

            <h3>Impact</h3>
            <p>Demonstrates how LLM-powered productivity tools can be tailored to neurodivergent users — reducing cognitive overhead through intelligent prioritization, structured scheduling, and multimodal input.</p>

            <h3>Technologies</h3>
            <p>Python, FastAPI, React (Vite), Expo, Google OAuth2, Gmail API, Calendar API, LLMs, Speech-to-Text</p>
        `
    }
};

function openProjectModal(projectId) {
    const project = projectDetails[projectId];
    if (project) {
        modalBody.innerHTML = `
            <h2>${project.title}</h2>
            ${project.description}
        `;
        projectModal.classList.add('open');
    }
}

modalClose.addEventListener('click', () => {
    projectModal.classList.remove('open');
});

projectModal.addEventListener('click', (e) => {
    if (e.target === projectModal) {
        projectModal.classList.remove('open');
    }
});

// ===================================
// CONTACT FORM
// ===================================

const contactForm   = document.getElementById('contactForm');
const submitBtn     = document.getElementById('formSubmitBtn');
const btnText       = submitBtn?.querySelector('.btn-text');
const btnSpinner    = submitBtn?.querySelector('.btn-spinner');
const formSuccess   = document.getElementById('formSuccess');
const formError     = document.getElementById('formError');

function setFormLoading(loading) {
    submitBtn.disabled = loading;
    btnText.style.display    = loading ? 'none' : 'inline';
    btnSpinner.style.display = loading ? 'inline' : 'none';
}

contactForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    formSuccess.style.display = 'none';
    formError.style.display   = 'none';
    setFormLoading(true);

    const useEmailJS = EMAILJS_CONFIG.publicKey  !== 'YOUR_EMAILJS_PUBLIC_KEY' &&
                       EMAILJS_CONFIG.templateId !== 'YOUR_TEMPLATE_ID';

    const fd      = new FormData(contactForm);
    const name    = fd.get('from_name') || '';
    const reply   = fd.get('reply_to')  || '';
    const subject = fd.get('subject')   || '';
    const msg     = fd.get('message')   || '';

    if (useEmailJS) {
        // EmailJS path — explicit params are more reliable than sendForm
        try {
            await emailjs.send(
                EMAILJS_CONFIG.serviceId,
                EMAILJS_CONFIG.templateId,
                {
                    from_name: name,
                    reply_to:  reply,
                    subject:   subject,
                    message:   msg,
                    to_name:   'Aneesh',
                }
            );
            formSuccess.style.display = 'flex';
            contactForm.reset();
        } catch (err) {
            console.error('EmailJS error:', err);
            formError.style.display = 'flex';
        }
    } else {
        // Fallback: open mailto with prefilled content
        const body = `From: ${name} (${reply})\n\n${msg}`;
        window.open(
            `mailto:aneeshjayan11@gmail.com?subject=${encodeURIComponent(subject)}&body=${encodeURIComponent(body)}`,
            '_blank'
        );
        formSuccess.style.display = 'flex';
        contactForm.reset();
    }

    setFormLoading(false);
});

// ===================================
// AI CHATBOT
// ===================================

const chatToggle = document.getElementById('chatToggle');
const chatWindow = document.getElementById('chatWindow');
const chatClose = document.getElementById('chatClose');
const chatMinimize = document.getElementById('chatMinimize');
const chatMessages = document.getElementById('chatMessages');
const chatInput = document.getElementById('chatInput');
const chatSend = document.getElementById('chatSend');
const quickActions = document.getElementById('quickActions');
const quickActionBtns = document.querySelectorAll('.quick-action-btn');

// Toggle chat window
chatToggle.addEventListener('click', () => {
    chatWindow.classList.toggle('open');
    if (chatWindow.classList.contains('open')) {
        chatInput.focus();
    }
});

chatClose.addEventListener('click', () => {
    chatWindow.classList.remove('open');
});

chatMinimize.addEventListener('click', () => {
    chatWindow.classList.remove('open');
});

// Quick action buttons
quickActionBtns.forEach(btn => {
    btn.addEventListener('click', () => {
        const question = btn.getAttribute('data-question');
        sendMessage(question);
        quickActions.style.display = 'none';
    });
});

// Send message
chatSend.addEventListener('click', () => {
    const message = chatInput.value.trim();
    if (message) {
        sendMessage(message);
        chatInput.value = '';
    }
});

chatInput.addEventListener('keypress', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        chatSend.click();
    }
});

// AI Knowledge Base
const aiKnowledge = {
    projects: {
        trustmedai: {
            name: 'TrustMedAI',
            purpose: 'Medical conversational agent for Type-2 Diabetes',
            tech: 'Python, FAISS, React, MiniLM embeddings, RAG',
            scale: '500+ forum threads, 16,000 clinical guidelines',
            metrics: '0.950 precision, 0.920 recall, 0.970 faithfulness',
            features: 'Semantic retrieval, multimodal (speech-to-text, TTS)',
            impact: 'Improves diabetes health information accessibility'
        },
        vlmspeedup: {
            name: 'VLM Speedup: LexFin Guard',
            purpose: 'High-performance invoice extraction using MoE + early exit for VLMs',
            tech: 'Python, PyTorch, MoE, HuggingFace, Streamlit',
            results: '3.5x throughput, 96% cost reduction, 94% accuracy, ~250ms latency',
            innovation: 'MoE routing to specialized experts + confidence-based early exit at layers 10/17'
        },
        optimalslm: {
            name: 'Optimal-SLM: Dual-Model Prompt Optimizer',
            purpose: 'A2A dual-model SLM pipeline for intent reasoning and prompt optimization',
            tech: 'Python, Qwen2-1.5B, Phi-3.5-Mini, LoRA, 4-bit Quant, AWS SageMaker, Docker',
            results: '>90% intent accuracy, 20-50% token reduction, <2s processing, >0.8 quality',
            features: 'Chain-of-Thought reasoning, A2A coordination, fallback mechanisms, voice STT'
        },
        probing: {
            name: 'LLM Probing: Mechanistic Interpretability',
            purpose: 'Probing LLM hidden states to decode behavioral framing (honesty vs dishonesty)',
            tech: 'Python, PyTorch, HuggingFace, StableLM-3B, Logistic Regression, PCA',
            results: 'Near-perfect probe accuracy on deeper layers; confirmed linear decodability of honesty signals',
            innovation: 'Layer-wise divergence analysis showing dishonest framings produce higher activation variance'
        },
        videoeditor: {
            name: 'AI Video Editor Agent',
            purpose: 'Multi-agent video editing via plain English — no video editing experience needed',
            tech: 'Python, CrewAI, GPT-4o, Ollama, Whisper, FFmpeg, OpenCV, FastAPI, Streamlit',
            results: '6 specialized agents, 3 pipeline modes (2/3/6-agent), full local offline support',
            features: 'Trimming, silence removal, subtitles, highlight reels, platform reframing, edit logs'
        },
        finslm: {
            name: 'FinSLM: Financial Small Language Model',
            purpose: 'Domain-adapted Mistral-7B for SEC filings, financial Q&A, and market analysis',
            tech: 'Python, Mistral-7B, LoRA/QLoRA, BitsAndBytes, SEC EDGAR, yfinance, W&B',
            results: 'LoRA ~98% of full FT quality; QLoRA runs on 2-3 GB VRAM; 20+ companies covered',
            innovation: 'Consumer-hardware deployable financial reasoning model'
        },
        focusmate: {
            name: 'FocusMate – AI Co-Pilot for Executive Function',
            purpose: 'ADHD-focused AI that converts Gmail/Calendar/voice into structured daily plans',
            tech: 'Python, FastAPI, React (Vite), Expo, Google OAuth2, Gmail/Calendar APIs, LLMs',
            results: '6+ REST endpoints, 2 microservices, voice-to-task pipeline, hackathon project',
            impact: 'Reduces cognitive overwhelm via intelligent prioritization and 90-min block scheduling'
        }
    },
    experience: {
        cips: {
            role: 'Project Volunteer — Legal AI',
            company: 'CIPS Lab, Arizona State University',
            duration: 'Feb 2026 – May 2026',
            focus: 'Legal AI assistant predicting case victory and explaining acts/laws via XAI + RAG',
            tech: 'RAG, Explainable AI, LLMs, Python, LangChain, FAISS',
            highlights: 'Case outcome prediction, citation-backed explanations, legal act interpretation'
        },
        soda: {
            role: 'Mentor',
            company: 'Software Developers Association (SoDA), ASU',
            duration: 'Feb 2026 – May 2026',
            focus: 'Mentoring students on AI/ML projects, software engineering, and careers',
            tech: 'AI/ML, Python, Software Engineering',
            highlights: 'AI/ML guidance, code review, career prep'
        },
        wolters: {
            role: 'Data Science Intern',
            company: 'Wolters Kluwer',
            duration: 'May 2025 - Dec 2025',
            achievements: '22% efficiency boost, 41% reduction in escalations, 85% accuracy improvement',
            tech: 'LangGraph, FastAPI, Azure DevOps, RAG, transformers',
            impact: 'Production AI system processing 50K+ daily records'
        },
        vit: {
            role: 'Research Intern - Biomedical AI',
            company: 'VIT Centre for Cyber-Physical Systems',
            duration: 'May 2023 - May 2024',
            achievement: '98.17% accuracy in autism detection from fMRI',
            tech: 'CNN, Transformers, Quantum ML',
            innovation: 'Hybrid deep learning-quantum framework'
        }
    },
    skills: {
        languages: 'Python (expert), SQL, C++, JavaScript, R, MATLAB',
        mlai: 'LLMs, RAG pipelines, PyTorch, TensorFlow, Hugging Face, Fine-tuning (LoRA, QLoRA)',
        cloud: 'Azure (Azure ML, DevOps), AWS (SageMaker, Lambda, S3), Docker, Kubernetes',
        specializations: 'Generative AI, Production ML, Distributed Systems, Model Optimization'
    },
    education: {
        current: 'MS in Data Science at Arizona State University (2024-2026)',
        previous: 'B.Tech in ECE from VIT (2020-2024)',
        coursework: 'Statistical ML, Data Mining, Data Processing At Scale'
    },
    contact: {
        email: 'aneeshjayan11@gmail.com',
        phone: '(602) 768-6622',
        linkedin: 'linkedin.com/in/aneeshjayan',
        github: 'github.com/aneeshjayan',
        location: 'Phoenix, Arizona, USA',
        availability: 'Currently open to opportunities in AI/ML Engineering'
    }
};

// Initialize EmailJS
(function () {
    if (EMAILJS_CONFIG.publicKey !== 'YOUR_EMAILJS_PUBLIC_KEY') {
        emailjs.init({ publicKey: EMAILJS_CONFIG.publicKey });
    }
})();

// Conversation history for RAG context (stores {role, content} pairs)
const conversationHistory = [];

const RAG_API_URL = 'http://localhost:8000/chat';

async function callRAGAPI(userMessage) {
    const response = await fetch(RAG_API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            message: userMessage,
            history: conversationHistory   // [{role, content}] array
        })
    });

    if (!response.ok) throw new Error(`RAG API error ${response.status}`);
    const data = await response.json();
    return data.answer || 'Sorry, I could not generate a response.';
}

async function sendMessage(userMessage) {
    // Add user message to UI
    addMessageToChat(userMessage, 'user');
    showTypingIndicator();

    let aiResponse;

    try {
        aiResponse = await callRAGAPI(userMessage);
        // Convert Markdown-style bold/italic and newlines to HTML
        aiResponse = aiResponse
            .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
            .replace(/\*(.*?)\*/g, '<em>$1</em>')
            .replace(/\n/g, '<br>');
        // Store turn in history for multi-turn context
        conversationHistory.push({ role: 'user',      content: userMessage });
        conversationHistory.push({ role: 'assistant', content: aiResponse });
        // Keep last 12 messages (6 turns) to avoid unbounded growth
        if (conversationHistory.length > 12) conversationHistory.splice(0, 2);
    } catch (err) {
        console.warn('RAG backend unreachable, falling back to keyword matching:', err);
        aiResponse = generateAIResponse(userMessage);
    }

    hideTypingIndicator();
    addMessageToChat(aiResponse, 'ai');
}

function addMessageToChat(message, sender) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `chat-message ${sender}`;
    
    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = sender === 'user' ? '👤' : '🤖';
    
    const bubble = document.createElement('div');
    bubble.className = 'message-bubble';
    bubble.innerHTML = message;
    
    messageDiv.appendChild(avatar);
    messageDiv.appendChild(bubble);
    
    chatMessages.appendChild(messageDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function showTypingIndicator() {
    const typingDiv = document.createElement('div');
    typingDiv.className = 'chat-message typing-indicator-message';
    typingDiv.innerHTML = `
        <div class="message-avatar">🤖</div>
        <div class="message-bubble typing-indicator">
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
            <span class="typing-dot"></span>
        </div>
    `;
    chatMessages.appendChild(typingDiv);
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function hideTypingIndicator() {
    const typingIndicator = document.querySelector('.typing-indicator-message');
    if (typingIndicator) {
        typingIndicator.remove();
    }
}

function generateAIResponse(userMessage) {
    const msg = userMessage.toLowerCase();
    
    // TrustMedAI questions
    if (msg.includes('trustmedai') || msg.includes('trust med') || msg.includes('medical') || msg.includes('diabetes')) {
        return `TrustMedAI is one of Aneesh's most impressive projects! 🏥<br><br>
                It's a medical conversational agent designed to make Type-2 Diabetes information more accessible. Here's what makes it special:<br><br>
                📊 <strong>Scale:</strong> Processed 500+ forum threads and 16,000 lines of clinical guidelines from authoritative sources like the ADA and Mayo Clinic<br><br>
                🎯 <strong>Performance:</strong> Achieved 0.950 precision, 0.970 faithfulness, and 0.888 semantic similarity<br><br>
                🔧 <strong>Tech:</strong> Built using Python, FAISS vector database, React, and MiniLM embeddings with a RAG architecture<br><br>
                🎤 <strong>Features:</strong> Includes multimodal interface with speech-to-text and text-to-speech for accessibility<br><br>
                The system bridges the gap between patient forums and medical literature, making healthcare information both accurate and understandable!<br><br>
                Want to know more about the technical architecture or see other projects?`;
    }
    
    // VLM Speedup / LexFin Guard questions
    if (msg.includes('vlm') || msg.includes('lexfin') || msg.includes('invoice') || msg.includes('speedup') || msg.includes('moe') || msg.includes('early exit')) {
        return `VLM Speedup: LexFin Guard is a high-impact FinTech AI project! ⚡<br><br>
                <strong>Purpose:</strong> Accelerate Vision-Language Models for high-volume financial document processing<br><br>
                🏗️ <strong>Architecture:</strong><br>
                • <strong>MoE Routing</strong> — routes image regions to specialized experts (Table Expert, OCR Expert)<br>
                • <strong>Early Exit</strong> — exits at Layer 10 or 17 for simple layouts, saving ~50% compute<br>
                • <strong>Validation</strong> — auto-verifies extracted data (Subtotal + Tax == Total)<br><br>
                📊 <strong>Results:</strong><br>
                • 3.5x throughput (~14 docs/sec vs 4 docs/sec)<br>
                • 96% cost reduction<br>
                • 94% accuracy (vs 92% baseline)<br>
                • ~250ms latency per document<br><br>
                🔧 <strong>Tech Stack:</strong> Python, PyTorch, MoE, HuggingFace, Streamlit<br><br>
                This project showcases Aneesh's ability to optimize AI inference for production financial workloads!<br><br>
                Want to explore other projects or his experience?`;
    }

    // Optimal-SLM questions
    if (msg.includes('optimal') || msg.includes('slm') || msg.includes('prompt') || (msg.includes('llm') && !msg.includes('probing'))) {
        return `Optimal-SLM is a sophisticated dual-model prompt optimization system! 🚀<br><br>
                <strong>Purpose:</strong> Agent-to-Agent pipeline pairing Qwen2-1.5B (reasoning) + Phi-3.5-Mini (optimization) within 6 GB VRAM<br><br>
                🔧 <strong>Architecture:</strong><br>
                • Chain-of-Thought reasoning → intent classification across 5 query types<br>
                • A2A coordination with confidence scoring and fallback mechanisms<br>
                • Speech-to-text preprocessing for voice inputs<br><br>
                📊 <strong>Results:</strong><br>
                • >90% intent classification accuracy<br>
                • >85% semantic preservation<br>
                • 20–50% token reduction (speech), 15–30% (text)<br>
                • Sub-2-second processing window<br><br>
                💡 <strong>Deployment:</strong> Docker + CI/CD on AWS SageMaker with Streamlit dashboard<br><br>
                Want to learn about his other projects or experience?`;
    }

    // LLM Probing / Interpretability questions
    if (msg.includes('probing') || msg.includes('interpretability') || msg.includes('mechanistic') || msg.includes('honesty') || msg.includes('hidden state')) {
        return `The LLM Probing project dives deep into AI interpretability! 🧠<br><br>
                <strong>Purpose:</strong> Reveal how LLMs internally encode behavioral framing (honest vs. dishonest) using probing techniques<br><br>
                🔬 <strong>Methods:</strong><br>
                • Extracted layer-wise hidden states and attention maps from StableLM-Tuned-Alpha-3B<br>
                • Applied logistic regression probes for linear separability testing<br>
                • Used PCA to visualize geometric structure of activation spaces<br>
                • Measured cosine similarity to quantify representational divergence<br><br>
                🎯 <strong>Key Findings:</strong><br>
                • Early layers: nearly identical activations for both contexts<br>
                • Deeper layers: strong asymmetric divergence — dishonest framings show higher variance<br>
                • Linear probes on deep layers: near-perfect accuracy<br><br>
                📌 <strong>Impact:</strong> Relevant to AI safety, prompt engineering, and model auditing<br><br>
                Interested in his other research or projects?`;
    }

    // AI Video Editor questions
    if (msg.includes('video') || msg.includes('editor') || msg.includes('crewai') || msg.includes('ffmpeg')) {
        return `The AI Video Editor Agent makes professional editing accessible to everyone! 🎬<br><br>
                <strong>Purpose:</strong> Edit videos using plain English — no video editing experience required<br><br>
                🤖 <strong>Architecture (6 Agents):</strong><br>
                • Audio Intelligence • Scene Detection • Clip Trimming<br>
                • Narrative Structuring • Subtitle Generation • Platform Adaptation<br><br>
                🔀 <strong>Smart Routing:</strong> 2-agent, 3-agent, or 6-agent pipelines based on task complexity<br><br>
                ⚙️ <strong>Capabilities:</strong><br>
                • Trimming, silence removal, speed adjustment<br>
                • Auto-subtitle burning via Whisper<br>
                • Highlight reel generation, platform reframing<br>
                • Human-readable edit logs after every operation<br><br>
                🏠 <strong>Fully Local:</strong> Runs offline via Ollama — no API keys needed<br><br>
                Want to know about other projects or how to reach Aneesh?`;
    }

    // FinSLM questions
    if (msg.includes('finslm') || msg.includes('financial') || msg.includes('sec') || msg.includes('mistral') || msg.includes('qlora')) {
        return `FinSLM is a domain-adapted financial reasoning model! 💰<br><br>
                <strong>Purpose:</strong> Fine-tune Mistral-7B on financial data for SEC filings, market analysis, and financial Q&A<br><br>
                📊 <strong>Training Data:</strong><br>
                • SEC EDGAR 10-K/10-Q filings — 20+ companies<br>
                • Financial news with sentiment labels<br>
                • Market fundamentals via yfinance<br><br>
                🔧 <strong>Training Strategies:</strong> Full FT, LoRA, 8-bit quantization, QLoRA (2–40+ GB VRAM)<br><br>
                📈 <strong>Results:</strong><br>
                • LoRA achieves ~98% of full fine-tuning quality at a fraction of compute<br>
                • QLoRA runs on consumer hardware with just 2–3 GB VRAM<br>
                • Covers P/E, DCF, EPS, and SEC filing interpretation<br><br>
                💡 Bridges general-purpose LLMs and precision financial analysis on commodity hardware!<br><br>
                Interested in his other projects or technical skills?`;
    }

    // FocusMate questions
    if (msg.includes('focusmate') || msg.includes('adhd') || msg.includes('hackathon') || msg.includes('spark') || msg.includes('calendar') || msg.includes('gmail')) {
        return `FocusMate is Aneesh's Hackathon project tackling executive function! 🧩<br><br>
                <strong>Purpose:</strong> ADHD co-pilot converting Gmail, Calendar, and voice notes into structured daily plans<br><br>
                🏗️ <strong>System:</strong><br>
                • 2 FastAPI microservices with Google OAuth2, Gmail & Calendar APIs<br>
                • Classifies emails: action vs. informational; extracts due-dated next steps<br>
                • Natural-language inbox search<br>
                • Time-blocked plans with 90-minute focus blocks + break scheduling<br><br>
                🖥️ <strong>Frontend:</strong><br>
                • React (Vite) planning dashboard<br>
                • Optional Expo mobile client<br>
                • Voice-to-task pipeline (STT)<br>
                • 6+ REST endpoints<br><br>
                📌 <strong>spark Challenge Hackathon</strong> — November 2025<br><br>
                Want to know about more of his projects or contact him?`;
    }
    
    // CIPS Lab questions
    if (msg.includes('cips') || msg.includes('legal ai') || msg.includes('case victory') || msg.includes('explainable') || msg.includes('xai')) {
        return `Aneesh is currently volunteering at ASU's CIPS Lab on a cutting-edge Legal AI project! ⚖️<br><br>
                <strong>CIPS Lab — Project Volunteer (Feb 2026 – May 2026):</strong><br><br>
                🎯 <strong>Focus:</strong> Building a Legal AI assistant that predicts case outcomes and explains legal reasoning<br><br>
                🔧 <strong>What he's building:</strong><br>
                • <strong>Case outcome predictor</strong> — estimates victory probability with evidence from prior case law<br>
                • <strong>XAI reasoning</strong> — citation-backed, interpretable explanations for each prediction<br>
                • <strong>RAG pipeline</strong> over statutes, case law, and legal acts for grounded answer generation<br>
                • <strong>Act/law interpreter</strong> — translates complex legal provisions into plain language<br><br>
                🔬 <strong>Tech:</strong> RAG, Explainable AI, LLMs, Python, LangChain, FAISS<br><br>
                This sits at the intersection of legal tech, AI safety, and NLP — a very exciting research area!<br><br>
                Want to know about his other experience or projects?`;
    }

    // SoDA Mentor questions
    if (msg.includes('soda') || msg.includes('mentor') || msg.includes('software developers association')) {
        return `Aneesh is also a Mentor at ASU's Software Developers Association! 🎓<br><br>
                <strong>SoDA Mentor (Feb 2026 – May 2026):</strong><br><br>
                📌 Mentoring student developers on AI/ML projects, software engineering, and career growth<br><br>
                🔧 <strong>Activities:</strong><br>
                • AI/ML project guidance — model selection, architecture, and evaluation<br>
                • Code review and debugging workshops<br>
                • Technical interview preparation<br>
                • Python, ML frameworks, and deployment sessions<br><br>
                This reflects his passion for knowledge-sharing and building the next generation of AI engineers!<br><br>
                Want to learn about his research or projects?`;
    }

    // Experience questions
    if (msg.includes('experience') || msg.includes('work') || msg.includes('wolters') || msg.includes('internship')) {
        return `Aneesh has a strong and diverse professional background! 💼<br><br>
                <strong>CIPS Lab, ASU (Feb 2026 – May 2026):</strong><br>
                ⚖️ Building Legal AI assistant — case outcome prediction + XAI + RAG over legal corpora<br><br>
                <strong>SoDA Mentor, ASU (Feb 2026 – May 2026):</strong><br>
                🎓 Mentoring students on AI/ML, software engineering, and career development<br><br>
                <strong>At Wolters Kluwer (May 2025 – Dec 2025):</strong><br>
                ✅ Built production AI systems processing 50K+ daily records<br>
                ✅ Reduced manual escalations by 41% via FastAPI microservices<br>
                ✅ Improved factual accuracy by 85% using transformer models<br>
                ✅ Reduced latency by 42% through optimization<br><br>
                <strong>VIT Research (May 2023 – May 2024):</strong><br>
                🧠 98.17% accuracy in autism detection from fMRI<br>
                🔬 Hybrid deep learning-quantum framework<br><br>
                Would you like to know about specific roles or his projects?`;
    }
    
    // Skills questions
    if (msg.includes('skill') || msg.includes('technology') || msg.includes('tech stack') || msg.includes('expertise')) {
        return `Aneesh has comprehensive AI/ML expertise! 🎯<br><br>
                <strong>Languages:</strong><br>
                • Python (Expert) - Primary language for ML/AI<br>
                • SQL, JavaScript, R, C++, MATLAB<br><br>
                <strong>ML/AI:</strong><br>
                • LLMs & Generative AI (Expert)<br>
                • RAG Pipelines, Fine-tuning (LoRA, QLoRA)<br>
                • PyTorch, TensorFlow, Hugging Face<br>
                • LangChain, LangGraph<br><br>
                <strong>Cloud & DevOps:</strong><br>
                • Azure (Azure ML, DevOps)<br>
                • AWS (SageMaker, Lambda, S3)<br>
                • Docker, Kubernetes, CI/CD<br><br>
                <strong>Specializations:</strong><br>
                🤖 Production ML Systems<br>
                🏥 Healthcare AI<br>
                ⚡ Model Optimization<br>
                🌐 Distributed Systems<br><br>
                Want to know more about specific projects or how to contact him?`;
    }
    
    // Contact questions
    if (msg.includes('contact') || msg.includes('reach') || msg.includes('email') || msg.includes('hire')) {
        return `Great! Here's how you can reach Aneesh: 📧<br><br>
                📧 <strong>Email:</strong> aneeshjayan11@gmail.com<br>
                📱 <strong>Phone:</strong> (602) 768-6622<br>
                💼 <strong>LinkedIn:</strong> linkedin.com/in/aneeshjayan<br>
                💻 <strong>GitHub:</strong> github.com/aneeshjayan<br>
                📍 <strong>Location:</strong> Phoenix, Arizona, USA<br><br>
                🌟 He's currently open to opportunities in AI/ML Engineering!<br><br>
                You can also use the contact form on this page, and he'll get back to you promptly.<br><br>
                Is there anything specific you'd like to discuss with him?`;
    }
    
    // Education questions
    if (msg.includes('education') || msg.includes('degree') || msg.includes('university') || msg.includes('asu') || msg.includes('vit')) {
        return `Aneesh has a strong educational background! 🎓<br><br>
                <strong>Current:</strong><br>
                🎓 MS in Data Science, Analytics and Engineering<br>
                📍 Arizona State University (2024-2026)<br>
                📚 Coursework: Statistical ML, Data Mining, Data Processing At Scale, Semantic Web Mining<br><br>
                <strong>Previous:</strong><br>
                🎓 B.Tech in Electronics and Communication Engineering<br>
                📍 Vellore Institute of Technology (2020-2024)<br>
                🏆 Research Intern - Biomedical AI<br><br>
                His education combines strong theoretical foundations with hands-on research and industry experience!<br><br>
                Want to know about his projects or work experience?`;
    }
    
    // Default response
    return `I can help you learn about Aneesh's:<br><br>
            🚀 <strong>Projects:</strong><br>
            • VLM Speedup: LexFin Guard (MoE + Early Exit, FinTech)<br>
            • TrustMedAI (Medical RAG, Healthcare AI)<br>
            • Optimal-SLM (Dual-model prompt optimizer)<br>
            • LLM Probing (Mechanistic interpretability)<br>
            • AI Video Editor Agent (CrewAI multi-agent)<br>
            • FinSLM (Financial Mistral-7B fine-tuning)<br>
            • FocusMate (ADHD co-pilot, Hackathon)<br><br>
            💼 <strong>Experience:</strong> CIPS Lab ASU, SoDA Mentor, Wolters Kluwer, VIT Research<br>
            🎯 <strong>Skills:</strong> Python, ML/AI, Cloud, LLMs, RAG, LoRA<br>
            🎓 <strong>Education:</strong> MS at ASU, B.Tech from VIT<br>
            📧 <strong>Contact:</strong> How to reach him<br><br>
            What would you like to know more about?`;
}

// ===================================
// DOWNLOAD RESUME
// ===================================

const downloadResume = document.getElementById('downloadResume');
downloadResume.addEventListener('click', (e) => {
    e.preventDefault();
    alert('Resume download functionality will be implemented. Please contact aneeshjayan11@gmail.com for the latest resume.');
});

// ===================================
// CHAT WITH AI BUTTON
// ===================================

const chatWithAI = document.getElementById('chatWithAI');
chatWithAI.addEventListener('click', () => {
    chatWindow.classList.add('open');
    chatInput.focus();
});

// ===================================
// FOOTER YEAR
// ===================================

document.getElementById('currentYear').textContent = new Date().getFullYear();

// ===================================
// SCROLL ANIMATIONS
// ===================================

const observerOptions = {
    threshold: 0.1,
    rootMargin: '0px 0px -100px 0px'
};

const observer = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            entry.target.classList.add('fade-in');
        }
    });
}, observerOptions);

// Observe all glass cards
document.querySelectorAll('.glass-card').forEach(card => {
    observer.observe(card);
});
