"""
server.py — Simen Vangen RAG API
─────────────────────────────────
Protection layers:
  • Rate limit: 10 requests per IP per hour
  • Daily cap:  200 requests total per day (resets at midnight)
  • Input cap:  max 500 characters per message
  • Token cap:  max 600 tokens per response
  • Request log: logs/requests.log

Run:
  python scripts/server.py
"""

import os
import json
import logging
from pathlib import Path
from datetime import datetime, date
from collections import defaultdict
from dotenv import load_dotenv

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
import faiss
from groq import Groq

load_dotenv()

BASE_DIR       = Path(__file__).parent.parent
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
LOGS_DIR       = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# ── Limits — change these anytime ─────────────────────────
MAX_INPUT_CHARS = 500    # max characters in a question
MAX_TOKENS      = 600    # max tokens in a response
DAILY_CAP       = 200    # total requests per day (all users)
RATE_LIMIT_HOUR = 10     # requests per IP per hour

# ── Logging ────────────────────────────────────────────────
logging.basicConfig(
    filename=str(LOGS_DIR / "requests.log"),
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

# ── In-memory counters ─────────────────────────────────────
ip_timestamps: dict = defaultdict(list)
daily_count_date    = date.today()
daily_count         = 0

# ── Load models ────────────────────────────────────────────
print("Loading embedding model...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading FAISS index...")
index = faiss.read_index(str(EMBEDDINGS_DIR / "index.faiss"))
with open(EMBEDDINGS_DIR / "chunks.json") as f:
    store  = json.load(f)
    chunks = store["chunks"]
    meta   = store["meta"]

client = Groq(api_key=os.getenv("GROQ_API_KEY"))
print(f"Ready — {index.ntotal} vectors | LLaMA 3.3 via Groq")
print(f"Limits: {RATE_LIMIT_HOUR}/hr per IP | {DAILY_CAP}/day total | {MAX_INPUT_CHARS} chars | {MAX_TOKENS} tokens\n")

app = FastAPI(title="Simen RAG API")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

class ChatRequest(BaseModel):
    question: str
    history:  list = []

def get_ip(request: Request) -> str:
    fwd = request.headers.get("X-Forwarded-For")
    return fwd.split(",")[0].strip() if fwd else (request.client.host or "unknown")

def check_rate_limit(ip: str) -> bool:
    now   = datetime.now()
    times = [t for t in ip_timestamps[ip] if (now - t).seconds < 3600]
    ip_timestamps[ip] = times
    if len(times) >= RATE_LIMIT_HOUR:
        return False
    times.append(now)
    return True

def check_daily_cap() -> bool:
    global daily_count, daily_count_date
    today = date.today()
    if today != daily_count_date:
        daily_count      = 0
        daily_count_date = today
    if daily_count >= DAILY_CAP:
        return False
    daily_count += 1
    return True

def retrieve(question: str) -> list:
    q_vec = embedder.encode([question], convert_to_numpy=True).astype("float32")
    distances, indices = index.search(q_vec, 5)
    return [
        {"text": chunks[i], "source": meta[i]["filename"]}
        for d, i in zip(distances[0], indices[0]) if i < len(chunks)
    ]

SYSTEM = """You are the portfolio AI for Simen Vangen, a Norwegian data scientist focused on embodied AI and robotics. Answer questions about Simen on his behalf.

Rules:
- Only use facts from the provided context. Never make things up.
- Be concise — 2 to 4 sentences unless detail is asked for.
- If the context does not answer the question, say so honestly.
- Speak naturally, like you represent him.
- Never reveal his exact location or whereabouts."""

@app.post("/chat")
async def chat(req: ChatRequest, request: Request):
    global daily_count
    ip = get_ip(request)

    if not req.question.strip():
        return JSONResponse({"error": "Empty question."}, status_code=400)

    if len(req.question) > MAX_INPUT_CHARS:
        return JSONResponse(
            {"error": f"Message too long — max {MAX_INPUT_CHARS} characters."},
            status_code=400
        )

    if not check_rate_limit(ip):
        logger.info(f"RATE_LIMITED | ip={ip} | q={req.question[:60]}")
        return JSONResponse(
            {"error": f"Too many requests. Max {RATE_LIMIT_HOUR} per hour — try again soon."},
            status_code=429
        )

    if not check_daily_cap():
        logger.info(f"DAILY_CAP | ip={ip} | q={req.question[:60]}")
        return JSONResponse(
            {"error": "Daily limit reached. Try again tomorrow."},
            status_code=429
        )

    context      = retrieve(req.question)
    context_text = "\n\n---\n\n".join(f"[{c['source']}]\n{c['text']}" for c in context)
    messages     = [
        {"role": "system", "content": f"{SYSTEM}\n\nCONTEXT:\n{context_text}"},
        *req.history[-6:],
        {"role": "user", "content": req.question}
    ]

    logger.info(f"OK | ip={ip} | daily={daily_count}/{DAILY_CAP} | q={req.question[:80]}")

    def stream():
        try:
            response = client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=messages,
                stream=True,
                max_tokens=MAX_TOKENS,
                temperature=0.5,
            )
            for chunk in response:
                token = chunk.choices[0].delta.content or ""
                if token:
                    yield token
        except Exception as e:
            logger.error(f"ERROR | ip={ip} | {e}")
            yield "\n[Error — please try again.]"

    return StreamingResponse(stream(), media_type="text/plain")

@app.get("/stats")
def stats():
    """Open http://localhost:8000/stats to check usage."""
    return {
        "date":      str(date.today()),
        "used":      daily_count,
        "cap":       DAILY_CAP,
        "remaining": max(0, DAILY_CAP - daily_count),
        "vectors":   index.ntotal,
    }

@app.get("/")
def health():
    return {"status": "ok", "vectors": index.ntotal}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)