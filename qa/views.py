import json
import os
from pathlib import Path

from django.http import JsonResponse
from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_http_methods

from qa.chunking import chunk_documents
from qa.embeddings import generate_chunk_embeddings
from qa.debug import log, rule
from qa.generator import NO_ANSWER, generate_answer
from qa.ingestion import SUPPORTED_EXTENSIONS, discover_documents, load_documents
from qa.retriever import retrieve_top_k
from qa.vectorstore import PersistentVectorStore

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
PERSISTENT_INDEX_PATH = BASE_DIR / "vectorstore" / "faiss_index.index"
PERSISTENT_METADATA_PATH = BASE_DIR / "vectorstore" / "metadata.json"
DOCUMENT_STATE_PATH = BASE_DIR / "vectorstore" / "document_state.json"


def home(request):
    """Serve the document Q&A chat interface."""
    pdf_files = discover_documents(DATA_DIR)
    return render(
        request,
        "qa/home.html",
        {"pdf_files": [path.name for path in pdf_files]},
    )


def get_document_signature(data_dir: str | Path) -> list[dict[str, object]]:
    base_dir = Path(data_dir)
    if not base_dir.exists():
        return []

    files: list[dict[str, object]] = []
    for pdf_path in sorted(base_dir.rglob("*")):
        if not pdf_path.is_file() or pdf_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        stat = pdf_path.stat()
        files.append(
            {
                "name": pdf_path.name,
                "path": str(pdf_path),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
            }
        )
    return files


def build_document_store(
    data_dir: str | Path = DATA_DIR,
    index_path: str | Path = PERSISTENT_INDEX_PATH,
    metadata_path: str | Path = PERSISTENT_METADATA_PATH,
    document_state_path: str | Path = DOCUMENT_STATE_PATH,
) -> PersistentVectorStore:
    store = PersistentVectorStore(index_path=index_path, metadata_path=metadata_path)
    signature = get_document_signature(data_dir)
    pdf_paths = discover_documents(data_dir)
    rule("DOCUMENT INDEXING")
    log(f"Documents found: {len(pdf_paths)}")
    state_path = Path(document_state_path)
    state_path.parent.mkdir(parents=True, exist_ok=True)

    if state_path.exists():
        try:
            previous_signature = json.loads(state_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            previous_signature = None
        if previous_signature == signature and store.index is not None and store.index.ntotal > 0:
            _log_index_summary(pdf_paths, store.metadata, store.index.d, store.index_path, "LOADED")
            rule("INDEXING COMPLETE")
            return store

    docs = load_documents(data_dir)
    if not docs:
        state_path.write_text("[]", encoding="utf-8")
        return store

    chunks = chunk_documents(docs, chunk_size=500, overlap=100)
    if not chunks:
        state_path.write_text(json.dumps(signature, ensure_ascii=False), encoding="utf-8")
        return store

    embedded_chunks = generate_chunk_embeddings(chunks, batch_size=32)
    embedding_dimension = len(embedded_chunks[0]["embedding"]) if embedded_chunks else 0
    if not embedded_chunks:
        state_path.write_text(json.dumps(signature, ensure_ascii=False), encoding="utf-8")
        return store

    vectors = [item["embedding"] for item in embedded_chunks]
    metadata = [{key: value for key, value in item.items() if key != "embedding"} for item in embedded_chunks]

    store.reset()
    store.add_vectors(vectors, metadata)
    state_path.write_text(json.dumps(signature, ensure_ascii=False), encoding="utf-8")
    _log_index_summary(pdf_paths, metadata, embedding_dimension, store.index_path, "CREATED")
    rule("INDEXING COMPLETE")
    return store


def _log_index_summary(pdf_paths, metadata, embedding_dimension, index_path, status):
    """Emit indexing diagnostics to the backend terminal only."""
    total_pages = 0
    for position, pdf_path in enumerate(pdf_paths, start=1):
        file_chunks = [item for item in metadata if item.get("filename") == pdf_path.name]
        pages = {item.get("page_number") for item in file_chunks}
        total_pages += len(pages)
        log(f"\nDocument {position}:")
        log(f"  Name: {pdf_path.name}")
        log(f"  Pages: {len(pages)}")
        log(f"  Chunks: {len(file_chunks)}")
    log(f"\nTotal documents: {len(pdf_paths)}")
    log(f"Total pages: {total_pages}")
    log(f"Total chunks: {len(metadata)}")
    log(f"\nEmbedding model: {os.getenv('EMBEDDING_MODEL', 'sentence-transformers/all-MiniLM-L6-v2')}")
    log(f"Embedding dimension: {embedding_dimension}")
    log("\nFAISS index:")
    log(f"  Path: {index_path}")
    log(f"  Vectors: {len(metadata)}")
    log(f"  Metadata records: {len(metadata)}")
    log(f"  Status: {status}")


@csrf_exempt
@require_http_methods(["GET", "POST"])
def ask_question(request):
    """Answer the question carried by this request; no request state is cached."""
    if request.method == "GET":
        question = (request.GET.get("question") or "").strip()
    else:
        raw_body = request.body.decode("utf-8") if request.body else ""
        payload: dict[str, object] = {}

        if raw_body:
            try:
                payload = json.loads(raw_body)
            except json.JSONDecodeError:
                payload = {}

        # `payload` is created for this POST only.  Do not fall back to a
        # previous request's data, answer, contexts, or retrieval results.
        question = str(payload.get("question", request.POST.get("question", "")) or "").strip()

    if not question:
        return JsonResponse({"error": "Question is required."}, status=400)

    rule("RAG QUERY")
    log(f"Question: {question}")
    pdf_files = discover_documents(DATA_DIR)
    store = build_document_store()
    top_k = int(os.getenv("TOP_K", "4"))
    contexts = retrieve_top_k(question, store, top_k=top_k)
    log("\n[FAISS SIMILARITY SEARCH]")
    log(f"TOP_K: {top_k}")
    log(f"Retrieved chunks: {len(contexts)}")
    if not contexts:
        log("    None")
    for rank, context in enumerate(contexts, start=1):
        log(f"\n[CHUNK {rank}]")
        log(f"Rank: {rank}")
        log(f"Score: {context.get('score', 'unknown')}")
        log(f"File: {context.get('filename', 'unknown')}")
        log(f"Page: {context.get('page_number', 'unknown')}")
        log(f"Chunk ID: {context.get('chunk_id', 'unknown')}")
        log(f"Text: {context.get('text', '')}")
    result = generate_answer(question, contexts)
    log("\n[FINAL ANSWER]")
    log(result.get("answer", NO_ANSWER))
    log("\n[SOURCES]")
    for source in result.get("sources", []):
        log(f"{source.get('filename', 'unknown')} — Page {source.get('page_number', 'unknown')}")
    if not result.get("sources"):
        log("None")
    rule("QUERY COMPLETE")

    return JsonResponse(
        {
            "question": question,
            "answer": result.get("answer", NO_ANSWER),
        }
    )
