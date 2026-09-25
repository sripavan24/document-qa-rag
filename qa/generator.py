from __future__ import annotations

import os
from typing import Any

from dotenv import load_dotenv
from groq import Groq

load_dotenv()


DEFAULT_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")


def build_grounded_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    """Build a grounded prompt using only the retrieved document context."""
    if not contexts:
        return (
            "Answer only from the provided documents. "
            "If the answer is not in the documents, respond exactly: "
            "I couldn't find the answer in the provided documents."
        )

    context_text = "\n\n".join(
        f"Source: {context.get('filename', 'unknown')} (Page {context.get('page_number', 'unknown')})\n{context.get('text', '')}"
        for context in contexts
    )

    return (
        "You are a careful assistant. Answer ONLY from the provided document context. "
        "If the information is not present in the context, respond exactly: "
        "I couldn't find the answer in the provided documents. "
        "Never invent facts or citations. "
        "When you do answer, include source names and page numbers based only on the provided context.\n\n"
        f"Question: {question}\n\nContext:\n{context_text}"
    )


def generate_answer(question: str, contexts: list[dict[str, Any]]) -> dict[str, Any]:
    """Generate a grounded answer using Groq if an API key is available."""
    if not question or not question.strip():
        return {"answer": "I couldn't find the answer in the provided documents.", "sources": []}

    if not contexts:
        return {"answer": "I couldn't find the answer in the provided documents.", "sources": []}

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        return {
            "answer": "I couldn't find the answer in the provided documents.",
            "sources": [
                {"filename": context.get("filename"), "page_number": context.get("page_number")}
                for context in contexts
            ],
        }

    client = Groq(api_key=api_key)
    prompt = build_grounded_prompt(question, contexts)

    response = client.chat.completions.create(
        model=DEFAULT_MODEL,
        messages=[
            {"role": "system", "content": "Answer only using the provided document context."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.1,
        max_tokens=400,
    )

    answer_text = response.choices[0].message.content.strip()
    answer = answer_text if answer_text else "I couldn't find the answer in the provided documents."

    sources = [
        {"filename": context.get("filename"), "page_number": context.get("page_number"), "text": context.get("text")}
        for context in contexts
    ]

    return {"answer": answer, "sources": sources}
