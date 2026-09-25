from __future__ import annotations

from typing import Any

from qa.embeddings import batch_embed_texts
from qa.vectorstore import PersistentVectorStore


def retrieve_top_k(
    query: str,
    store: PersistentVectorStore,
    top_k: int = 4,
    threshold: float | None = None,
) -> list[dict[str, Any]]:
    """Embed a query and retrieve top-k relevant chunks from the FAISS index."""
    if not query or not query.strip():
        return []

    query_vector = batch_embed_texts([query.strip()], batch_size=1)[0]
    if store.index is not None and len(query_vector) != store.index.d:
        return []
    results = store.search(query_vector, top_k=top_k, threshold=threshold)
    return results
