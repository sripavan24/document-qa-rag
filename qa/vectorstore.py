from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import faiss
import numpy as np


DEFAULT_INDEX_PATH = Path(__file__).resolve().parent.parent / "vectorstore" / "faiss_index.index"
DEFAULT_METADATA_PATH = Path(__file__).resolve().parent.parent / "vectorstore" / "metadata.json"


class PersistentVectorStore:
    """Simple FAISS-backed vector store with persisted metadata."""

    def __init__(self, index_path: str | Path = DEFAULT_INDEX_PATH, metadata_path: str | Path = DEFAULT_METADATA_PATH):
        self.index_path = Path(index_path)
        self.metadata_path = Path(metadata_path)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        self.metadata_path.parent.mkdir(parents=True, exist_ok=True)
        self.index: faiss.Index | None = None
        self.metadata: list[dict[str, Any]] = []
        self._load_if_exists()

    def _load_if_exists(self) -> None:
        if self.index_path.exists():
            self.index = faiss.read_index(str(self.index_path))
        else:
            self.index = None

        if self.metadata_path.exists():
            try:
                with self.metadata_path.open("r", encoding="utf-8") as handle:
                    self.metadata = json.load(handle)
            except (json.JSONDecodeError, OSError):
                self.metadata = []

    def add_vectors(self, vectors: list[list[float]], metadata: list[dict[str, Any]]) -> None:
        if not vectors:
            return

        array = np.asarray(vectors, dtype="float32")
        dimension = array.shape[1]

        if self.index is None:
            self.index = faiss.IndexFlatIP(dimension)
        elif self.index.d != dimension:
            raise ValueError(f"Vector dimension mismatch: index expects {self.index.d}, got {dimension}")

        self.index.add(array)
        self.metadata.extend(metadata)
        self._save()

    def search(self, query_vector: list[float], top_k: int = 4, threshold: float | None = None) -> list[dict[str, Any]]:
        if self.index is None or self.index.ntotal == 0:
            return []

        query = np.asarray([query_vector], dtype="float32")
        if query.shape[1] != self.index.d:
            raise ValueError(f"Query dimension mismatch: index expects {self.index.d}, got {query.shape[1]}")

        scores, indices = self.index.search(query, min(top_k, self.index.ntotal))
        results: list[dict[str, Any]] = []

        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or idx >= len(self.metadata):
                continue
            if threshold is not None and float(score) < float(threshold):
                continue
            item = dict(self.metadata[int(idx)])
            item["score"] = float(score)
            results.append(item)

        return results

    def _save(self) -> None:
        if self.index is not None:
            faiss.write_index(self.index, str(self.index_path))
        with self.metadata_path.open("w", encoding="utf-8") as handle:
            json.dump(self.metadata, handle, ensure_ascii=False)

    def reset(self) -> None:
        self.index = None
        self.metadata = []
        self._save()
