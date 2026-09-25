from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq
from qa.debug import log

load_dotenv()


DEFAULT_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b")
FALLBACK_MODELS = [
    DEFAULT_MODEL,
    "openai/gpt-oss-20b",
    "openai/gpt-oss-120b",
]
NO_ANSWER = "I don't know. I couldn't find the answer in the provided documents."
def _sources(contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"filename": context.get("filename"), "page_number": context.get("page_number"), "text": context.get("text")}
        for context in contexts
    ]


def build_document_context(contexts: list[dict[str, Any]]) -> str:
    """Format the exact retrieved evidence supplied to Groq."""
    return "\n\n".join(
        f"Source: {context.get('filename', 'unknown')} (Page {context.get('page_number', 'unknown')})\n"
        f"{context.get('text', '')}"
        for context in contexts
    )


def build_grounded_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    """Build a grounded prompt using only the retrieved document context."""
    if not contexts:
        return (
            "Answer only from the provided documents. "
            "If the answer is not in the documents, respond exactly: "
            + NO_ANSWER
        )

    context_text = build_document_context(contexts)

    return (
        "You are a document Q&A assistant.\n\n"
        "Answer the user's current question using ONLY the provided document context.\n\n"
        f"Question:\n{question}\n\nContext:\n{context_text}\n\n"
        "Rules:\n"
        "- Answer naturally and directly.\n"
        "- Use only information supported by the context.\n"
        "- You may combine multiple retrieved chunks.\n"
        "- Do not copy raw document text unnecessarily.\n"
        "- Do not use outside knowledge.\n"
        "- Do not invent information.\n"
        f"- If the context does not contain enough information to answer the question, respond exactly: {NO_ANSWER}\n\n"
        "Return ONLY the final answer."
    )


def generate_answer(question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a grounded answer using Groq if an API key is available."""
    if not question or not question.strip():
        log("\n[GROQ]")
        log(f"Model: {DEFAULT_MODEL}")
        log("Status: SKIPPED (empty question)")
        return {"answer": NO_ANSWER, "sources": []}

    if not contexts:
        log("\n[CONTEXT SENT TO GROQ]")
        log("Number of chunks: 0")
        log("Documents used: None")
        log("Pages used: None")
        log("\n[GROQ]")
        log(f"Model: {DEFAULT_MODEL}")
        log("Status: SKIPPED (no relevant context)")
        return {"answer": NO_ANSWER, "sources": []}

    sources = _sources(contexts)
    documents_used = sorted({str(context.get("filename", "unknown")) for context in contexts})
    pages_used = sorted({str(context.get("page_number", "unknown")) for context in contexts})
    log("\n[CONTEXT SENT TO GROQ]")
    log(f"Number of chunks: {len(contexts)}")
    log(f"Documents used: {', '.join(documents_used)}")
    log(f"Pages used: {', '.join(pages_used)}")
    log("Exact context:")
    log(build_document_context(contexts))

    api_key = (os.getenv("GROQ_API_KEY") or "").strip()
    if not api_key or api_key.lower().startswith("your_"):
        log("\n[GROQ]")
        log(f"Model: {DEFAULT_MODEL}")
        log("Status: FAILED (API key unavailable)")
        return {"answer": NO_ANSWER, "sources": sources}

    try:
        client = Groq(api_key=api_key)
        prompt = build_grounded_prompt(question, contexts)

        attempt_errors: list[tuple[str, Exception]] = []
        for model_name in list(dict.fromkeys(model for model in FALLBACK_MODELS if model)):
            try:
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": "Answer only using the provided document context."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.1,
                    max_tokens=400,
                )
                answer_text = (response.choices[0].message.content or "").strip()
                answer = answer_text or NO_ANSWER
                log("\n[GROQ]")
                log(f"Model: {model_name}")
                log("Status: SUCCESS")
                break
            except Exception as exc:  # pragma: no cover - depends on external API availability
                attempt_errors.append((model_name, exc))
                continue
        else:
            if attempt_errors:
                attempted_models = ", ".join(model for model, _ in attempt_errors)
                last_model, last_error = attempt_errors[-1]
                raise RuntimeError(
                    f"Groq failed for models: {attempted_models}. "
                    f"Last failure ({last_model}): {type(last_error).__name__}: {last_error}"
                ) from last_error
            answer = NO_ANSWER
    except Exception as exc:
        answer = NO_ANSWER
        log("\n[GROQ]")
        log(f"Model: {DEFAULT_MODEL}")
        log("Status: FAILED")
        log(f"Error type: {type(exc).__name__}")
        log(f"Error: {exc}")

    return {"answer": answer, "sources": sources}
