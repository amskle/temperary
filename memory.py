"""Vector memory – store and retrieve past report generations.

Uses Chroma DB with sentence-transformers embeddings (with optional DeepSeek
embedding fallback).
"""

import hashlib
from typing import Any

import chromadb
from chromadb.config import Settings

from config import config


_client: Any = None
_collection: Any = None


def _get_client() -> Any:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=config.chroma_db_path,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_embedding(text: str) -> list[float]:
    """Embed text using sentence-transformers, imported lazily."""
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(config.embedding_model)
    return model.encode(text, show_progress_bar=False).tolist()


def init_memory() -> Any:
    """Initialize (or get) the Chroma collection for report memories."""
    global _collection
    client = _get_client()

    try:
        _collection = client.get_collection("report_memories")
    except Exception:
        _collection = client.create_collection(
            name="report_memories",
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def add_memory(
    requirement: str,
    generated_content: dict[str, str],
    rating: int,
) -> None:
    """Store a successful generation into the vector DB (rating >= threshold).

    Only stores when rating >= config.memory_store_threshold (default 4).
    The embedding is computed from the requirement text for retrieval.
    """
    if rating < config.memory_store_threshold:
        return

    col = init_memory()
    embedding = _get_embedding(requirement)

    # Flatten content for storage
    content_str = "; ".join(
        f"{k}: {v[:300]}" for k, v in generated_content.items()
    )
    doc_id = hashlib.md5(requirement.encode()).hexdigest()[:16]

    col.upsert(
        ids=[doc_id],
        embeddings=[embedding],
        documents=[content_str],
        metadatas=[{"requirement": requirement, "rating": rating}],
    )
    print(f"  [Memory] Stored case '{doc_id}' (rating={rating}).")


def retrieve_similar(requirement: str, k: int | None = None) -> list[dict]:
    """Retrieve top-k similar past cases from the vector DB.

    Returns a list of dicts with keys: id, requirement, content, distance.
    """
    if k is None:
        k = config.retrieval_top_k

    col = init_memory()

    try:
        count = col.count()
    except Exception:
        count = 0

    if count == 0:
        return []

    query_emb = _get_embedding(requirement)
    results = col.query(query_embeddings=[query_emb], n_results=min(k, count))

    examples: list[dict] = []
    for i in range(len(results["ids"][0])):
        examples.append(
            {
                "id": results["ids"][0][i],
                "requirement": results["metadatas"][0][i].get("requirement", ""),
                "content": results["documents"][0][i],
                "distance": (
                    results["distances"][0][i] if results.get("distances") else None
                ),
            }
        )
    return examples
