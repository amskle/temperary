"""Vector memory – store and retrieve past report generations.

Uses Chroma DB with sentence-transformers embeddings (with optional DeepSeek
embedding fallback).
"""

import hashlib
import logging
from typing import Any

import chromadb
from chromadb.config import Settings

from config import config

logger = logging.getLogger(__name__)

_client: Any = None
_collection: Any = None
_embedding_model: Any = None


def _get_client() -> Any:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(
            path=config.chroma_db_path,
            settings=Settings(anonymized_telemetry=False),
        )
    return _client


def _get_embedding(text: str) -> list[float]:
    """Embed text using sentence-transformers, with model caching."""
    global _embedding_model
    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer(config.embedding_model)
    return _embedding_model.encode(text, show_progress_bar=False).tolist()


def _build_embedding_text(requirement: str, template_headers: str) -> str:
    """Build the text used for embedding — combines template structure with requirement.

    This ensures vector search retrieves few-shot examples that match both the
    subject matter AND the specific report structure, not just the requirement.
    """
    if template_headers:
        return f"Template Theme: {template_headers} | Requirement: {requirement}"
    return requirement


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
    template_headers: str = "",
) -> None:
    """Store a successful generation into the vector DB (rating >= threshold).

    Only stores when rating >= config.memory_store_threshold (default 4).
    The embedding text combines template headers with the requirement so
    vector search matches both subject AND report structure.
    """
    if rating < config.memory_store_threshold:
        return

    col = init_memory()
    embedding_text = _build_embedding_text(requirement, template_headers)
    embedding = _get_embedding(embedding_text)

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
    logger.info("  [Memory] Stored case '%s' (rating=%d).", doc_id, rating)


def retrieve_similar(
    requirement: str, k: int | None = None, template_headers: str = ""
) -> list[dict]:
    """Retrieve top-k similar past cases from the vector DB.

    The query embedding combines template headers with the requirement
    so retrieval matches both subject matter and report structure.

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

    query_text = _build_embedding_text(requirement, template_headers)
    query_emb = _get_embedding(query_text)
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
