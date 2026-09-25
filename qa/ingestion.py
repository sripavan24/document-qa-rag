from __future__ import annotations

from pathlib import Path
from typing import Any

import fitz
from docx import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx"}


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
    """Validate that a document has a supported extension."""
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


def _single_page_document(path: Path, text: str) -> list[dict[str, Any]]:
    """Represent a text-based file as one source page for RAG metadata."""
    return [{"filename": path.name, "page_number": 1, "text": text.strip()}] if text.strip() else []


def extract_txt_pages(file_path: str | Path) -> list[dict[str, Any]]:
    """Extract UTF-8 plain text as a single source page."""
    path = Path(file_path)
    if not validate_document(path) or path.suffix.lower() != ".txt":
        return []
    return _single_page_document(path, path.read_text(encoding="utf-8", errors="replace"))


def extract_docx_pages(file_path: str | Path) -> list[dict[str, Any]]:
    """Extract paragraph text from a DOCX file as a single source page."""
    path = Path(file_path)
    if not validate_document(path) or path.suffix.lower() != ".docx":
        return []
    document = Document(path)
    return _single_page_document(path, "\n".join(paragraph.text for paragraph in document.paragraphs))


def extract_document_pages(file_path: str | Path) -> list[dict[str, Any]]:
    """Extract pages from any supported document type."""
    path = Path(file_path)
    extractors = {
        ".pdf": extract_pdf_pages,
        ".txt": extract_txt_pages,
        ".docx": extract_docx_pages,
    }
    extractor = extractors.get(path.suffix.lower())
    return extractor(path) if extractor else []


def load_documents(data_dir: str | Path) -> list[dict[str, Any]]:
    """Load all valid supported documents into a simple metadata structure."""
    pages: list[dict[str, Any]] = []
    for document_path in discover_documents(data_dir):
        for page in extract_document_pages(document_path):
            pages.append({"source_file": str(document_path), "filename": page["filename"], "page_number": page["page_number"], "text": page["text"]})
    return pages
