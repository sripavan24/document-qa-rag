"""Opt-in, terminal-friendly diagnostics for the RAG pipeline."""
from __future__ import annotations

import os
import sys
from typing import Any


def enabled() -> bool:
    """Keep request-level RAG diagnostics visible in the Django terminal by default."""
    return os.getenv("RAG_DEBUG", "True").strip().lower() in {"1", "true", "yes", "on"}


def log(message: str = "") -> None:
    """Print diagnostics only when RAG_DEBUG is enabled."""
    if enabled():
        # Windows terminals commonly use cp1252 while PDF text can contain
        # Unicode bullets and symbols.  Diagnostics must never abort `/ask/`.
        encoding = sys.stdout.encoding or "utf-8"
        safe_message = str(message).encode(encoding, errors="replace").decode(encoding)
        print(safe_message, flush=True)


def rule(title: str) -> None:
    if enabled():
        print("\n" + "=" * 50, flush=True)
        print(title, flush=True)
        print("=" * 50, flush=True)


def preview(text: Any, limit: int = 180) -> str:
    compact = " ".join(str(text or "").split())
    return compact if len(compact) <= limit else f"{compact[:limit - 3]}..."
