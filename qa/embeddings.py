from __future__ import annotations

from typing import Any

from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def get_embedding_model(model_name: str = EMBEDDING_MODEL) -> SentenceTransformer:
    """Initialize the cached sentence-transformer model used for embeddings.

    The application already persists its FAISS index locally.  Keeping the
    embedder local prevents an unrelated Hugging Face network check from
    blocking a live `/ask/` request before it reaches FAISS.
    """
    return SentenceTransformer(model_name, local_files_only=True)


def batch_embed_texts(texts: list[str], batch_size: int = 32, model_name: str = EMBEDDING_MODEL) -> list[list[float]]:
    """Generate embeddings for a list of texts in batches and return vectors."""
    if not texts:
        return []
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than 0")

    model = get_embedding_model(model_name)
    embeddings = model.encode(texts, batch_size=batch_size, convert_to_numpy=True, show_progress_bar=False)
    return embeddings.astype(float).tolist()


def generate_chunk_embeddings(chunks: list[dict[str, Any]], batch_size: int = 32, model_name: str = EMBEDDING_MODEL) -> list[dict[str, Any]]:
    """Attach embeddings to chunk metadata while preserving source identity."""
    if not chunks:
        return []

    texts = [chunk.get("text", "") for chunk in chunks]
    vectors = batch_embed_texts(texts, batch_size=batch_size, model_name=model_name)

    embedded_chunks: list[dict[str, Any]] = []
    for chunk, vector in zip(chunks, vectors):
        embedded_chunk = dict(chunk)
        embedded_chunk["embedding"] = vector
        embedded_chunks.append(embedded_chunk)

    return embedded_chunks
