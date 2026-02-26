"""
Portfolio RAG Backend — FastAPI server implementing traditional RAG:
  1. Embed all knowledge-base chunks at startup (sentence-transformers)
  2. Upsert embeddings into Qdrant Cloud at startup
  3. On each /chat request: embed query → Qdrant vector search → retrieve top-k chunks
  4. Build context-rich prompt → call Gemini 2.0 Flash REST API → return answer
"""

import os
import logging
import requests
from typing import List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from knowledge_base import DOCUMENTS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

GEMINI_API_KEY  = os.getenv("GEMINI_API_KEY",  "AIzaSyBlzWgEZc21i9N3pILlMrH1sgVlBd6rUC0")
QDRANT_URL      = os.getenv("QDRANT_URL",      "https://ffe3896b-ed98-4150-8241-128b25fba331.eu-central-1-0.aws.cloud.qdrant.io:6333")
QDRANT_API_KEY  = os.getenv("QDRANT_API_KEY",  "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.u-Z1UhffjrC_Ux1IjNTgyiQY66BkUndIci31cXq3fIc")
GEMINI_URL      = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    f"gemini-2.0-flash:generateContent?key={GEMINI_API_KEY}"
)

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"   # 384-dim, fast, great quality
COLLECTION_NAME  = "portfolio_chunks"
VECTOR_SIZE      = 384
TOP_K            = 4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App & CORS
# ---------------------------------------------------------------------------

app = FastAPI(title="Portfolio RAG API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Startup: embed chunks → upsert into Qdrant
# ---------------------------------------------------------------------------

embed_model: SentenceTransformer = None
qdrant: QdrantClient = None


@app.on_event("startup")
def startup():
    global embed_model, qdrant

    # 1. Load embedding model
    logger.info("Loading sentence-transformer model: %s", EMBED_MODEL_NAME)
    embed_model = SentenceTransformer(EMBED_MODEL_NAME)

    # 2. Connect to Qdrant Cloud
    logger.info("Connecting to Qdrant Cloud: %s", QDRANT_URL)
    qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

    # 3. Recreate collection (ensures fresh index on every server start)
    collections = [c.name for c in qdrant.get_collections().collections]
    if COLLECTION_NAME in collections:
        logger.info("Deleting existing collection '%s' for fresh index …", COLLECTION_NAME)
        qdrant.delete_collection(COLLECTION_NAME)

    qdrant.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
    )
    logger.info("Created Qdrant collection '%s' (dim=%d, cosine)", COLLECTION_NAME, VECTOR_SIZE)

    # 4. Encode all chunks
    texts = [doc["text"] for doc in DOCUMENTS]
    ids   = [doc["id"]   for doc in DOCUMENTS]
    logger.info("Encoding %d knowledge-base chunks …", len(texts))
    vectors = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=True).tolist()

    # 5. Upsert into Qdrant
    points = [
        PointStruct(
            id=idx,
            vector=vectors[idx],
            payload={"doc_id": ids[idx], "text": texts[idx]},
        )
        for idx in range(len(DOCUMENTS))
    ]
    qdrant.upsert(collection_name=COLLECTION_NAME, points=points)
    logger.info("Upserted %d points into Qdrant. RAG index ready.", len(points))


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class ChatMessage(BaseModel):
    role: str       # "user" | "assistant"
    content: str


class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []


class ChatResponse(BaseModel):
    answer: str
    retrieved_chunks: List[str] = []   # chunk IDs (for debugging)


# ---------------------------------------------------------------------------
# Retrieval — Qdrant vector search
# ---------------------------------------------------------------------------

def retrieve(query: str, top_k: int = TOP_K) -> List[dict]:
    """Embed query, search Qdrant, return top-k chunks."""
    q_vec = embed_model.encode([query], convert_to_numpy=True)[0].tolist()

    results = qdrant.search(
        collection_name=COLLECTION_NAME,
        query_vector=q_vec,
        limit=top_k,
        with_payload=True,
    )

    return [
        {
            "id":    hit.payload["doc_id"],
            "text":  hit.payload["text"],
            "score": hit.score,
        }
        for hit in results
    ]


# ---------------------------------------------------------------------------
# Gemini generation
# ---------------------------------------------------------------------------

def build_system_prompt(retrieved_chunks: List[dict]) -> str:
    context_block = "\n\n".join(
        f"[{i+1}] {chunk['text']}" for i, chunk in enumerate(retrieved_chunks)
    )
    return (
        "You are the AI assistant on Aneesh Jayan Prabhu's portfolio website. "
        "Answer questions about Aneesh using ONLY the context below. "
        "Be helpful, concise, and professional. "
        "Use HTML formatting (<strong>, <br>, bullet points •) when it improves readability. "
        "If the context doesn't contain the answer, say so politely and suggest visiting the portfolio. "
        "Never make up information not present in the context.\n\n"
        "=== RETRIEVED CONTEXT ===\n"
        f"{context_block}\n"
        "========================="
    )


def call_gemini(system_prompt: str, history: List[ChatMessage], user_message: str) -> str:
    contents = []

    # First turn always carries the system prompt + RAG context
    first_user_text = history[0].content if history else user_message
    contents.append({
        "role": "user",
        "parts": [{"text": system_prompt + "\n\nUser question: " + first_user_text}]
    })

    if history:
        if len(history) >= 2:
            contents.append({"role": "model", "parts": [{"text": history[1].content}]})
            for turn in history[2:]:
                role = "user" if turn.role == "user" else "model"
                contents.append({"role": role, "parts": [{"text": turn.content}]})
        contents.append({"role": "user", "parts": [{"text": user_message}]})

    payload = {
        "contents": contents,
        "generationConfig": {"temperature": 0.6, "maxOutputTokens": 700},
    }

    resp = requests.post(GEMINI_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    try:
        return data["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError) as exc:
        logger.error("Unexpected Gemini response: %s", data)
        raise ValueError("Could not parse Gemini response") from exc


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    collections = [c.name for c in qdrant.get_collections().collections] if qdrant else []
    return {
        "status": "ok",
        "qdrant_collection": COLLECTION_NAME,
        "collection_exists": COLLECTION_NAME in collections,
        "model": EMBED_MODEL_NAME,
        "chunks": len(DOCUMENTS),
    }


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    if embed_model is None or qdrant is None:
        raise HTTPException(status_code=503, detail="Server not ready yet")

    # 1. Retrieve relevant chunks via Qdrant
    chunks = retrieve(req.message, top_k=TOP_K)
    logger.info(
        "Query: %r | Top chunks: %s",
        req.message[:80],
        [f"{c['id']}({c['score']:.2f})" for c in chunks],
    )

    # 2. Build context-grounded system prompt
    system_prompt = build_system_prompt(chunks)

    # 3. Generate with Gemini
    try:
        answer = call_gemini(system_prompt, req.history, req.message)
    except requests.HTTPError as exc:
        logger.error("Gemini HTTP error: %s", exc)
        raise HTTPException(status_code=502, detail=f"Gemini API error: {exc.response.status_code}")
    except Exception as exc:
        logger.error("Generation error: %s", exc)
        raise HTTPException(status_code=500, detail=str(exc))

    return ChatResponse(
        answer=answer,
        retrieved_chunks=[c["id"] for c in chunks],
    )
