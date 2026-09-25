from __future__ import annotations

import re
from typing import Any


DEFAULT_CHUNK_SIZE = 500
DEFAULT_CHUNK_OVERLAP = 100


def clean_text(text: str) -> str:
    """Normalize whitespace in extracted document text."""
    return re.sub(r"\s+", " ", text or "").strip()


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks while preserving readability."""
    cleaned = clean_text(text)
    if not cleaned:
        return []

    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    chunks: list[str] = []
    start = 0
    step = chunk_size - overlap

    while start < len(cleaned):
        end = min(start + chunk_size, len(cleaned))
        chunk = cleaned[start:end]
        chunks.append(chunk.strip())
        if end == len(cleaned):
            break
        start += step

    return [chunk for chunk in chunks if chunk]


def chunk_documents(pages: list[dict[str, Any]], chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[dict[str, Any]]:
    """Convert page-level document text into chunk metadata with page and filename info."""
    chunks: list[dict[str, Any]] = []

    for page in pages:
        text = page.get("text", "")
        for index, chunk_text_value in enumerate(chunk_text(text, chunk_size=chunk_size, overlap=overlap), start=1):
            chunks.append(
                {
                    "filename": page.get("filename"),
                    "page_number": page.get("page_number"),
                    "chunk_id": f"{page.get('filename')}::p{page.get('page_number')}::c{index}",
                    "text": chunk_text_value,
                }
            )

    return chunks
