"""Ask the document RAG pipeline directly from the terminal."""
from __future__ import annotations

import os

from django.core.management.base import BaseCommand, CommandError

from qa.debug import log, rule
from qa.generator import NO_ANSWER, generate_answer
from qa.retriever import retrieve_top_k
from qa.views import build_document_store


class Command(BaseCommand):
    help = "Ask a question against the indexed documents without using Postman."

    def add_arguments(self, parser):
        parser.add_argument("question", nargs="*", help="Question to ask. Omit it to enter one interactively.")
        parser.add_argument("--debug", action="store_true", help="Enable detailed RAG terminal logs for this command.")
        parser.add_argument("--top-k", type=int, default=None, help="Override TOP_K for this query.")

    def handle(self, *args, **options):
        if options["debug"]:
            os.environ["RAG_DEBUG"] = "True"

        question = " ".join(options["question"]).strip()
        top_k = options["top_k"] if options["top_k"] is not None else int(os.getenv("TOP_K", "4"))
        if top_k <= 0:
            raise CommandError("--top-k must be greater than zero.")

        if question:
            self._answer(question, top_k)
            return

        self.stdout.write("Ask questions about your documents. Type 'exit' or 'quit' to stop.")
        while True:
            try:
                question = input("\nQuestion: ").strip()
            except (EOFError, KeyboardInterrupt):
                self.stdout.write("\nGoodbye.")
                return

            if question.lower() in {"exit", "quit"}:
                self.stdout.write("Goodbye.")
                return
            if not question:
                self.stdout.write("Please enter a question, or type 'exit' to stop.")
                continue
            self._answer(question, top_k)

    def _answer(self, question: str, top_k: int) -> None:
        """Run one independent RAG question without retaining prior results."""

        rule("RAG QUERY")
        log(f"Question: {question}")
        store = build_document_store()
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
            log(f"{source.get('filename', 'unknown')} - Page {source.get('page_number', 'unknown')}")
        rule("QUERY COMPLETE")

        self.stdout.write("\n" + "=" * 50)
        self.stdout.write("RAG TERMINAL RESULT")
        self.stdout.write("=" * 50)
        self.stdout.write(f"Question: {question}")
        self.stdout.write(f"Retrieved chunks: {len(contexts)}")
        self.stdout.write(f"Answer: {result.get('answer', NO_ANSWER)}")
        self.stdout.write("Sources:")
        sources = result.get("sources", [])
        if sources:
            for source in sources:
                self.stdout.write(f"  - {source.get('filename', 'unknown')} (page {source.get('page_number', 'unknown')})")
        else:
            self.stdout.write("  - None")
