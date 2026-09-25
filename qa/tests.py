from pathlib import Path
from tempfile import TemporaryDirectory

import fitz
from django.test import SimpleTestCase

from qa.chunking import chunk_documents, chunk_text
from qa.embeddings import batch_embed_texts, generate_chunk_embeddings
from qa.generator import build_grounded_prompt, generate_answer
from qa.ingestion import discover_documents, extract_pdf_pages, validate_document
from qa.retriever import retrieve_top_k
from qa.vectorstore import PersistentVectorStore


class IngestionTests(SimpleTestCase):
    def test_discover_documents_finds_pdfs_only(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "keep.pdf").write_bytes(b"pdf")
            (root / "ignore.txt").write_text("not a pdf")
            nested_dir = root / "nested"
            nested_dir.mkdir()
            (nested_dir / "also.pdf").write_bytes(b"pdf")

            docs = discover_documents(root)

            self.assertEqual(len(docs), 2)
            self.assertTrue(all(path.suffix.lower() == ".pdf" for path in docs))

    def test_validate_document_accepts_pdf_and_rejects_unsupported_files(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            pdf_path.write_bytes(b"pdf")
            txt_path = Path(temp_dir) / "sample.txt"
            txt_path.write_text("plain text")

            self.assertTrue(validate_document(pdf_path))
            self.assertFalse(validate_document(txt_path))

    def test_extract_pdf_pages_keeps_filename_and_page_number(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "This is a test page.")
            document.save(pdf_path)
            document.close()

            pages = extract_pdf_pages(pdf_path)

            self.assertTrue(pages)
            first = pages[0]
            self.assertEqual(first["filename"], pdf_path.name)
            self.assertIsInstance(first["page_number"], int)
            self.assertIn("This is a test page", first["text"])


class ChunkingTests(SimpleTestCase):
    def test_chunk_text_adds_overlap_and_keeps_text(self):
        text = (
            "The project uses Django for the application shell and a simple retrieval pipeline. "
            "Each document is parsed page by page so the source metadata stays accurate. "
            "Chunks are intentionally overlapped to preserve context at boundaries. "
            "This approach keeps answers grounded in the user-provided documents. "
            "The system stores chunk metadata for later source citations."
        )
        chunks = chunk_text(text, chunk_size=120, overlap=30)

        self.assertTrue(chunks)
        self.assertTrue(len(chunks) > 1)
        self.assertTrue(all(len(chunk) <= 120 for chunk in chunks))
        self.assertNotEqual(chunks[0], chunks[1])

    def test_chunk_documents_preserves_metadata(self):
        pages = [{"filename": "doc.pdf", "page_number": 1, "text": "This is a sample page with enough text to chunk into pieces."}]
        chunks = chunk_documents(pages, chunk_size=30, overlap=10)

        self.assertTrue(chunks)
        self.assertEqual(chunks[0]["filename"], "doc.pdf")
        self.assertEqual(chunks[0]["page_number"], 1)
        self.assertIn("chunk_id", chunks[0])

    def test_chunk_text_handles_empty_input(self):
        self.assertEqual(chunk_text("   ", chunk_size=30, overlap=5), [])


class EmbeddingTests(SimpleTestCase):
    def test_batch_embed_texts_returns_embeddings_with_consistent_shape(self):
        texts = ["This is a test sentence.", "This is a second sentence."]
        embeddings = batch_embed_texts(texts, batch_size=2)

        self.assertEqual(len(embeddings), 2)
        self.assertTrue(all(len(vector) > 0 for vector in embeddings))
        self.assertEqual(len(embeddings[0]), len(embeddings[1]))

    def test_generate_chunk_embeddings_keeps_chunk_metadata(self):
        chunks = [
            {"chunk_id": "doc.pdf::p1::c1", "text": "Python and Django are used for the backend."},
            {"chunk_id": "doc.pdf::p1::c2", "text": "RAG uses vector similarity to retrieve relevant context."},
        ]

        embedded = generate_chunk_embeddings(chunks, batch_size=1)

        self.assertEqual(len(embedded), 2)
        self.assertIn("embedding", embedded[0])
        self.assertIn("chunk_id", embedded[0])
        self.assertTrue(len(embedded[0]["embedding"]) > 0)


class VectorStoreTests(SimpleTestCase):
    def test_vector_store_persists_and_loads_index(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = root / "faiss.index"
            metadata_path = root / "metadata.json"
            store = PersistentVectorStore(index_path=index_path, metadata_path=metadata_path)

            vectors = batch_embed_texts(["alpha", "beta"], batch_size=2)
            metadata = [{"chunk_id": "a", "text": "alpha"}, {"chunk_id": "b", "text": "beta"}]
            store.add_vectors(vectors, metadata)

            reloaded = PersistentVectorStore(index_path=index_path, metadata_path=metadata_path)
            self.assertEqual(len(reloaded.metadata), 2)
            self.assertTrue(reloaded.index is not None)
            self.assertEqual(reloaded.index.ntotal, 2)

    def test_vector_store_search_returns_metadata_and_score(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            index_path = root / "faiss.index"
            metadata_path = root / "metadata.json"
            store = PersistentVectorStore(index_path=index_path, metadata_path=metadata_path)
            vectors = batch_embed_texts(["alpha", "beta"], batch_size=2)
            store.add_vectors(vectors, [{"chunk_id": "one", "text": "alpha"}, {"chunk_id": "two", "text": "beta"}])

            results = store.search(vectors[0], top_k=1, threshold=0.0)

            self.assertTrue(results)
            self.assertIn("score", results[0])
            self.assertIn("chunk_id", results[0])


class RetrievalTests(SimpleTestCase):
    def test_retrieve_top_k_returns_results_and_score(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            store = PersistentVectorStore(index_path=root / "faiss.index", metadata_path=root / "metadata.json")
            vectors = batch_embed_texts(
                [
                    "Django is a Python web framework.",
                    "FAISS stores vectors for search.",
                ],
                batch_size=2,
            )
            store.add_vectors(
                vectors,
                [{"chunk_id": "one", "text": "Django is a Python web framework."}, {"chunk_id": "two", "text": "FAISS stores vectors for search."}],
            )

            results = retrieve_top_k("Django", store, top_k=2, threshold=0.0)

            self.assertTrue(results)
            self.assertIn("score", results[0])
            self.assertIn("text", results[0])

    def test_retrieve_top_k_ignores_empty_query(self):
        with TemporaryDirectory() as temp_dir:
            store = PersistentVectorStore(index_path=Path(temp_dir) / "faiss.index", metadata_path=Path(temp_dir) / "metadata.json")
            self.assertEqual(retrieve_top_k("   ", store), [])


class GeneratorTests(SimpleTestCase):
    def test_build_grounded_prompt_uses_context_only(self):
        contexts = [{"filename": "doc.pdf", "page_number": 2, "text": "Django uses a model-view-template architecture."}]
        prompt = build_grounded_prompt("What architecture does Django use?", contexts)

        self.assertIn("doc.pdf", prompt)
        self.assertIn("Page 2", prompt)
        self.assertIn("Django uses a model-view-template architecture", prompt)

    def test_generate_answer_handles_unanswerable_questions(self):
        result = generate_answer("What is the capital of France?", [])
        self.assertEqual(result["answer"], "I couldn't find the answer in the provided documents.")

    def test_generate_answer_returns_safe_fallback_without_api_key(self):
        contexts = [{"filename": "sample.pdf", "page_number": 1, "text": "The project stores files in the data directory."}]
        result = generate_answer("Where are the files stored?", contexts)
        self.assertIn("sample.pdf", str(result["sources"]))
