from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz

SUPPORTED_EXTENSIONS = {".pdf"}


def discover_documents(data_dir: str | Path) -> list[Path]:
    """Discover supported document files from the data directory."""
    base_dir = Path(data_dir)
    if not base_dir.exists():
        return []

    return sorted(
        path
        for path in base_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def validate_document(file_path: str | Path) -> bool:
    """Validate that a document is a supported PDF file."""
    path = Path(file_path)
    if not path.exists() or not path.is_file():
        return False
    return path.suffix.lower() in SUPPORTED_EXTENSIONS


def extract_pdf_pages(file_path: str | Path) -> list[dict[str, Any]]:
    """Extract text from each PDF page while preserving filename and page metadata."""
    path = Path(file_path)
    if not validate_document(path):
        return []

    extracted_pages: list[dict[str, Any]] = []
    document = fitz.open(path)
    try:
        for page_number in range(len(document)):
            page = document[page_number]
            text = page.get_text("text")
            extracted_pages.append(
                {
                    "filename": path.name,
                    "page_number": page_number + 1,
                    "text": text.strip(),
                }
            )
    finally:
        document.close()

    return extracted_pages


def load_documents(data_dir: str | Path) -> list[dict[str, Any]]:
    """Load all valid PDF documents and their pages into a simple metadata structure."""
    pages: list[dict[str, Any]] = []
    for pdf_path in discover_documents(data_dir):
        for page in extract_pdf_pages(pdf_path):
            pages.append({"source_file": str(pdf_path), "filename": page["filename"], "page_number": page["page_number"], "text": page["text"]})
    return pages
