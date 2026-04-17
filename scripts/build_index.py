"""
build_index.py
Run this once to build the vector index from your /data folder.
Re-run whenever you add new files to /data.

Usage:
  python scripts/build_index.py
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

BASE_DIR       = Path(__file__).parent.parent
DATA_DIR       = BASE_DIR / "data"
EMBEDDINGS_DIR = BASE_DIR / "embeddings"
EMBEDDINGS_DIR.mkdir(exist_ok=True)

CHUNK_SIZE    = 400
CHUNK_OVERLAP = 80

def load_documents():
    docs = []
    for path in DATA_DIR.glob("**/*"):
        if path.suffix in (".txt", ".md") and not path.name.startswith("PROJECT_TEMPLATE"):
            text = path.read_text(encoding="utf-8")
            docs.append({"filename": path.name, "text": text})
            print(f"  loaded: {path.name} ({len(text)} chars)")
    return docs

def chunk_text(text):
    chunks = []
    start = 0
    while start < len(text):
        end = start + CHUNK_SIZE
        chunk = text[start:end].strip()
        if len(chunk) > 40:
            chunks.append(chunk)
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks

def build():
    print("\n── Loading documents ──")
    docs = load_documents()
    if not docs:
        print("No documents found in /data. Add .txt or .md files and re-run.")
        return

    print("\n── Chunking ──")
    chunks, meta = [], []
    for doc in docs:
        doc_chunks = chunk_text(doc["text"])
        for i, chunk in enumerate(doc_chunks):
            chunks.append(chunk)
            meta.append({"filename": doc["filename"], "chunk_index": i})
        print(f"  {doc['filename']}: {len(doc_chunks)} chunks")
    print(f"\n  Total: {len(chunks)} chunks")

    print("\n── Embedding (may take a minute) ──")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks, show_progress_bar=True, convert_to_numpy=True).astype("float32")

    print("\n── Building FAISS index ──")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, str(EMBEDDINGS_DIR / "index.faiss"))

    with open(EMBEDDINGS_DIR / "chunks.json", "w") as f:
        json.dump({"chunks": chunks, "meta": meta}, f, indent=2)

    print(f"\n  Done — {index.ntotal} vectors saved to /embeddings")
    print("  Now run: python scripts/server.py\n")

if __name__ == "__main__":
    build()