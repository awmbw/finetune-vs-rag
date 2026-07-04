"""
RAG Retrieval — Semantic search over the ChromaDB vector store.

Given a user query, finds the top-K most similar medical flashcards
using cosine similarity on sentence-transformer embeddings.
"""

from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

# --- Configuration ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CHROMA_DIR = PROJECT_ROOT / "models" / "chromadb"
COLLECTION_NAME = "medical_flashcards"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def get_retriever():
    """Initialize and return the ChromaDB collection and embedding model."""
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection = client.get_collection(name=COLLECTION_NAME)
    return collection, embedder


def retrieve(query: str, collection, embedder, top_k: int = 3) -> list[dict]:
    """
    Retrieve the top-K most relevant flashcards for a given query.
    
    Returns a list of dicts with 'question', 'document', and 'distance' keys.
    """
    # Embed the query into the same 384-dim space as our documents
    query_embedding = embedder.encode([query]).tolist()
    
    # Search ChromaDB using cosine similarity
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )
    
    # Format the results
    retrieved = []
    for i in range(len(results["ids"][0])):
        retrieved.append({
            "question": results["metadatas"][0][i].get("question", ""),
            "document": results["documents"][0][i],
            "distance": results["distances"][0][i]
        })
    
    return retrieved


if __name__ == "__main__":
    # Quick test — search for a medical question
    collection, embedder = get_retriever()
    
    test_query = "What causes type 2 diabetes?"
    print(f"Query: {test_query}\n")
    
    results = retrieve(test_query, collection, embedder, top_k=3)
    
    for i, r in enumerate(results, 1):
        print(f"--- Result {i} (distance: {r['distance']:.4f}) ---")
        print(f"  {r['document'][:200]}")
        print()
