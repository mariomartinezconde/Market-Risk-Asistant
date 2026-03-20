"""System prompt and context injection for the RAG pipeline."""
from __future__ import annotations

from app.ingestion.chunker import Chunk

SYSTEM_PROMPT = """You are a professional AI assistant specialised in market risk regulation and documentation for a regulated banking institution.

You serve Risk Officers, Compliance Officers, Model Validators, and Internal Auditors.

STRICT RULES:
1. Answer ONLY from the DOCUMENT CONTEXT provided. Never use outside knowledge.
2. Cite every factual claim with [Document Name, Section] inline.
3. If documents contradict each other, flag it explicitly.
4. If context is insufficient, say clearly: "The available documentation does not provide sufficient basis to answer this question."
5. Never fabricate regulatory references, figures, or dates.
6. When evidence is weak, recommend escalation to a subject matter expert.
7. Be precise, professional, and concise. Avoid padding.
"""

NO_CONTEXT_RESPONSE = (
    "The available documentation does not contain sufficient information to answer this question. "
    "Please ensure the relevant documents have been uploaded and indexed, or consult a subject matter expert directly."
)


def build_prompt(
    query: str,
    chunks: list[tuple[Chunk, float]],
    conflict_hint: str | None = None,
) -> str:
    if not chunks:
        return f"[NO DOCUMENTS AVAILABLE]\n\nQuestion: {query}"

    blocks = []
    for i, (chunk, score) in enumerate(chunks, 1):
        blocks.append(
            f"--- Source {i} ---\n"
            f"Document: {chunk.doc_id}\n"
            f"Section: {chunk.section_title or 'N/A'}\n"
            f"Relevance: {score:.2f}\n\n"
            f"{chunk.text.strip()}"
        )

    context = "\n\n".join(blocks)
    conflict = f"\n\n⚠ CONFLICT DETECTED: {conflict_hint}\nYou MUST address this in your answer." if conflict_hint else ""

    return f"[DOCUMENT CONTEXT]\n\n{context}{conflict}\n\n[QUESTION]\n{query}"
