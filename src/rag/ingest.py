"""
RAG Ingestion Pipeline — Embed medical flashcards into ChromaDB.

Reads rag_documents.jsonl, converts each Q&A pair into a 384-dim
vector using sentence-transformers, and stores them in a persistent
ChromaDB vector database for semantic retrieval.
"""

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "rag_documents.jsonl"
CHROMA_DIR = PROJECT_ROOT / "models" / "chromadb"
COLLECTION_NAME = "medical_flashcards"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BATCH_SIZE = 256  # How many documents to embed at once


def load_documents(path: Path) -> list[dict]:
    """Load JSONL documents into a list of dicts."""
    docs = []
    with open(path, "r") as f:
        for line in f:
            docs.append(json.loads(line))
    print(f"  Loaded {len(docs)} documents from {path.name}")
    return docs


def ingest():
    print("=" * 60)
    print("RAG INGESTION PIPELINE")
    print("=" * 60)

    # 1. Load documents
    print("\n[1/3] Loading documents...")
    documents = load_documents(DATA_PATH)

    # 2. Initialize embedding model
    print(f"\n[2/3] Loading embedding model: {EMBEDDING_MODEL}")
    embedder = SentenceTransformer(EMBEDDING_MODEL)

    # 3. Initialize ChromaDB (persistent storage on disk)
    print(f"\n[3/3] Building ChromaDB at {CHROMA_DIR}")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    
    # Delete existing collection if re-running (idempotent)
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
    
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"}  # Use cosine similarity
    )

    # 4. Process in batches
    total = len(documents)
    for i in range(0, total, BATCH_SIZE):
        batch = documents[i : i + BATCH_SIZE]
        
        # The document text is already combined in 'content'
        texts = [doc["content"] for doc in batch]
        
        # Create unique IDs
        ids = [f"doc_{i + j}" for j in range(len(batch))]
        
        # Store the original metadata for retrieval later
        metadatas = [doc["metadata"] for doc in batch]

        # Embed the texts
        embeddings = embedder.encode(texts, show_progress_bar=False).tolist()
        
        # Insert into ChromaDB
        collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=texts,
            metadatas=metadatas
        )
        
        print(f"  Indexed {min(i + BATCH_SIZE, total)}/{total} documents")

    print(f"\n✅ Ingestion complete! {total} documents indexed in ChromaDB.")
    print(f"   Database location: {CHROMA_DIR}")


if __name__ == "__main__":
    ingest()
