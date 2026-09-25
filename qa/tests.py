import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import MagicMock, patch

import fitz
from docx import Document
from django.test import Client, SimpleTestCase

from qa.chunking import chunk_documents, chunk_text
from qa.debug import log
from qa.embeddings import batch_embed_texts, generate_chunk_embeddings
from qa.generator import NO_ANSWER, build_grounded_prompt, generate_answer
from qa.ingestion import discover_documents, extract_docx_pages, extract_pdf_pages, extract_txt_pages, validate_document
from qa.retriever import retrieve_top_k
from qa.vectorstore import PersistentVectorStore
from qa.views import build_document_store


class IngestionTests(SimpleTestCase):
    def test_discover_documents_finds_supported_document_types(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            (root / "keep.pdf").write_bytes(b"pdf")
            (root / "keep.txt").write_text("plain text")
            (root / "ignore.csv").write_text("not supported")
            nested_dir = root / "nested"
            nested_dir.mkdir()
            (nested_dir / "also.pdf").write_bytes(b"pdf")
            document = Document()
            document.add_paragraph("DOCX content")
            document.save(root / "keep.docx")

            docs = discover_documents(root)

            self.assertEqual(len(docs), 4)
            self.assertEqual({path.suffix.lower() for path in docs}, {".pdf", ".txt", ".docx"})

    def test_validate_document_accepts_supported_types_and_rejects_unsupported_files(self):
        with TemporaryDirectory() as temp_dir:
            pdf_path = Path(temp_dir) / "sample.pdf"
            pdf_path.write_bytes(b"pdf")
            txt_path = Path(temp_dir) / "sample.txt"
            txt_path.write_text("plain text")
            docx_path = Path(temp_dir) / "sample.docx"
            Document().save(docx_path)
            csv_path = Path(temp_dir) / "sample.csv"
            csv_path.write_text("unsupported")

            self.assertTrue(validate_document(pdf_path))
            self.assertTrue(validate_document(txt_path))
            self.assertTrue(validate_document(docx_path))
            self.assertFalse(validate_document(csv_path))

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

    def test_extract_txt_and_docx_as_single_source_page(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            txt_path = root / "sample.txt"
            txt_path.write_text("Text document content.", encoding="utf-8")
            docx_path = root / "sample.docx"
            document = Document()
            document.add_paragraph("DOCX document content.")
            document.save(docx_path)

            txt_pages = extract_txt_pages(txt_path)
            docx_pages = extract_docx_pages(docx_path)

            self.assertEqual(txt_pages, [{"filename": "sample.txt", "page_number": 1, "text": "Text document content."}])
            self.assertEqual(docx_pages, [{"filename": "sample.docx", "page_number": 1, "text": "DOCX document content."}])


class DebugTests(SimpleTestCase):
    def test_rag_debug_false_suppresses_detailed_terminal_output(self):
        with patch.dict("os.environ", {"RAG_DEBUG": "False"}), patch("builtins.print") as terminal_print:
            log("debug-only message")

        terminal_print.assert_not_called()


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

            results = store.search(vectors[0], top_k=1)

            self.assertTrue(results)
            self.assertIn("score", results[0])
            self.assertIn("chunk_id", results[0])


class RetrievalTests(SimpleTestCase):
    def test_retrieve_top_k_embeds_the_current_question(self):
        class RecordingStore:
            index = None

            def __init__(self):
                self.arguments = None

            def search(self, vector, top_k):
                self.arguments = (vector, top_k)
                return []

        store = RecordingStore()
        with patch("qa.retriever.batch_embed_texts", return_value=[[0.1, 0.2]]) as embed:
            retrieve_top_k("  What is ML?  ", store, top_k=3)

        embed.assert_called_once_with(["What is ML?"], batch_size=1)
        self.assertEqual(store.arguments, ([0.1, 0.2], 3))

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

            results = retrieve_top_k("Django", store, top_k=2)

            self.assertTrue(results)
            self.assertIn("score", results[0])
            self.assertIn("text", results[0])

    def test_retrieve_top_k_ignores_empty_query(self):
        with TemporaryDirectory() as temp_dir:
            store = PersistentVectorStore(index_path=Path(temp_dir) / "faiss.index", metadata_path=Path(temp_dir) / "metadata.json")
            self.assertEqual(retrieve_top_k("   ", store), [])


class ApiEndpointTests(SimpleTestCase):
    def test_sequential_posts_use_only_their_current_question_and_retrieval(self):
        contexts_by_question = {
            "What is ML?": [
                {"filename": "ML FINAL.pdf", "page_number": 1, "score": 0.91, "text": "Machine learning (ML) is a field of study."},
            ],
            "What is supervised machine learning?": [
                {"filename": "ML FINAL.pdf", "page_number": 9, "score": 0.88, "text": "Several algorithms have been developed for supervised learning."},
                {"filename": "ML FINAL.pdf", "page_number": 64, "score": 0.86, "text": "In classification, the algorithm assigns labels to data based on predefined features. This is an example of supervised learning."},
            ],
            "What is quantum computing?": [],
        }

        def retrieve_for_current_question(question, store, top_k):
            return contexts_by_question[question]

        with (
            patch("qa.views.build_document_store", return_value=object()),
            patch("qa.views.discover_documents", return_value=[Path("ML FINAL.pdf")]),
            patch("qa.views.retrieve_top_k", side_effect=retrieve_for_current_question) as retrieve,
            patch("qa.views.generate_answer", side_effect=lambda question, contexts: {"answer": f"Natural answer for: {question}", "sources": contexts}),
            patch.dict("os.environ", {"GROQ_API_KEY": "", "TOP_K": "2"}),
        ):
            responses = [
                Client().post("/ask/", data=json.dumps({"question": question}), content_type="application/json").json()
                for question in contexts_by_question
            ]

        self.assertEqual([response["question"] for response in responses], list(contexts_by_question))
        self.assertEqual(responses[0]["answer"], "Natural answer for: What is ML?")
        self.assertEqual(responses[1]["answer"], "Natural answer for: What is supervised machine learning?")
        self.assertEqual(responses[2]["answer"], "Natural answer for: What is quantum computing?")
        self.assertTrue(all(set(response) == {"question", "answer"} for response in responses))
        self.assertEqual([call.args[0] for call in retrieve.call_args_list], list(contexts_by_question))
        self.assertTrue(all(call.kwargs == {"top_k": 2} for call in retrieve.call_args_list))


class CacheOptimizationTests(SimpleTestCase):
    def test_build_document_store_reuses_existing_index_for_unchanged_documents(self):
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            data_dir.mkdir()
            pdf_path = data_dir / "sample.pdf"
            document = fitz.open()
            page = document.new_page()
            page.insert_text((72, 72), "Django is a Python web framework.")
            document.save(pdf_path)
            document.close()

            vector_dir = root / "vectorstore"
            vector_dir.mkdir()
            state_path = vector_dir / "document_state.json"

            with patch("qa.views.DATA_DIR", data_dir), patch("qa.views.PERSISTENT_INDEX_PATH", vector_dir / "faiss.index"), patch("qa.views.PERSISTENT_METADATA_PATH", vector_dir / "metadata.json"), patch("qa.views.DOCUMENT_STATE_PATH", state_path):
                first = build_document_store()
                self.assertTrue(first.index is not None)

                with patch("qa.views.generate_chunk_embeddings", side_effect=AssertionError("rebuild should not happen")):
                    second = build_document_store()

                self.assertTrue(second.index is not None)
                self.assertEqual(second.index.ntotal, first.index.ntotal)


class GeneratorTests(SimpleTestCase):
    def test_build_grounded_prompt_uses_context_only(self):
        contexts = [{"filename": "doc.pdf", "page_number": 2, "text": "Django uses a model-view-template architecture."}]
        prompt = build_grounded_prompt("What architecture does Django use?", contexts)

        self.assertIn("doc.pdf", prompt)
        self.assertIn("Page 2", prompt)
        self.assertIn("Django uses a model-view-template architecture", prompt)

    def test_generate_answer_handles_unanswerable_questions(self):
        result = generate_answer("What is the capital of France?", [])
        self.assertEqual(result["answer"], NO_ANSWER)

    def test_generate_answer_uses_groq_output_not_raw_context(self):
        contexts = [{"filename": "network.pdf", "page_number": 2, "text": "A computer network connects computers, software, and hardware."}]
        completion = MagicMock()
        completion.choices[0].message.content = "A computer network connects computers, software, and hardware so they can communicate."
        client = MagicMock()
        client.chat.completions.create.return_value = completion

        with patch.dict("os.environ", {"GROQ_API_KEY": "test-key"}), patch("qa.generator.Groq", return_value=client):
            result = generate_answer("What is a computer network?", contexts)

        self.assertEqual(result["answer"], completion.choices[0].message.content)
        self.assertNotEqual(result["answer"], contexts[0]["text"])
        prompt = client.chat.completions.create.call_args.kwargs["messages"][1]["content"]
        self.assertIn("What is a computer network?", prompt)
        self.assertIn(contexts[0]["text"], prompt)

    def test_generate_answer_handles_supervised_machine_learning_from_context(self):
        contexts = [
            {"filename": "ML FINAL.pdf", "page_number": 9, "text": "Supervised Learning is where the AI really began its journey. This technique was applied successfully in several cases. You have used this model while doing the hand-written recognition on your machine. Several algorithms have been developed for supervised learning."},
            {"filename": "ML FINAL.pdf", "page_number": 64, "text": "Classification. In classification, the algorithm assigns labels to data based on the predefined features. This is an example of supervised learning. Clustering. An algorithm splits data into a number of clusters based on the similarity of features. This is an example of unsupervised learning."},
        ]
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
            result = generate_answer("What is supervised machine learning?", contexts)

        self.assertEqual(result["answer"], NO_ANSWER)
        self.assertEqual(
            [(source["filename"], source["page_number"]) for source in result["sources"]],
            [("ML FINAL.pdf", 9), ("ML FINAL.pdf", 64)],
        )

    def test_generate_answer_prefers_the_supervised_learning_definition(self):
        contexts = [
            {
                "filename": "Notes.pdf",
                "page_number": 4,
                "score": 0.71,
                "text": (
                    "Machine Learning systems can be classified according to the amount and type of supervision they get during training. "
                    "There are four major categories. In supervised learning, the training data you feed to the algorithm includes the desired solutions, called labels. "
                    "A typical supervised learning task is classification."
                ),
            }
        ]
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
            result = generate_answer("What is supervised machine learning?", contexts)

        self.assertEqual(result["answer"], NO_ANSWER)

    def test_generate_answer_returns_safe_fallback_without_api_key(self):
        contexts = [{"filename": "sample.pdf", "page_number": 1, "text": "The project stores files in the data directory."}]
        with patch.dict("os.environ", {"GROQ_API_KEY": ""}):
            result = generate_answer("Where are the files stored?", contexts)
        self.assertIn("sample.pdf", str(result["sources"]))

    def test_generate_answer_falls_back_for_unrelated_question(self):
        contexts = [{"filename": "ML FINAL.pdf", "page_number": 64, "text": "Classification. In classification, the algorithm assigns labels to data based on the predefined features. This is an example of supervised learning."}]
        result = generate_answer("What is the capital of France?", contexts)
        self.assertEqual(result["answer"], NO_ANSWER)
