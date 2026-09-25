from __future__ import annotations

from typing import Any

import os

from qa.embeddings import batch_embed_texts
from qa.debug import log
from qa.vectorstore import PersistentVectorStore


def retrieve_top_k(
    query: str,
    store: PersistentVectorStore,
    top_k: int = 4,
) -> list[dict[str, Any]]:
    """Embed a query and retrieve top-k relevant chunks from the FAISS index."""
    if not query or not query.strip():
        return []

    query_vector = batch_embed_texts([query.strip()], batch_size=1)[0]
    log("[QUERY EMBEDDING]")
    log(f"Model: {os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')}")
    log(f"Dimension: {len(query_vector)}")
    log("Status: SUCCESS")
    if store.index is not None and len(query_vector) != store.index.d:
        log("    Status: FAILED (embedding dimension does not match the FAISS index)")
        return []
    # Always return TOP_K FAISS matches. The grounded LLM decides whether the
    # retrieved context supports an answer; no fixed score cutoff is applied.
    results = store.search(query_vector, top_k=top_k)
    return results
