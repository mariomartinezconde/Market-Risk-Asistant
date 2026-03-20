"""
Prompt Orchestration
====================
Defines the system prompt and builds the user message with injected context.

Governance rules enforced here:
  G1 – No answer without context: if no chunks were retrieved, the prompt
       instructs the model to say so explicitly.
  G2 – Mandatory citations: the model is instructed to cite [DOC_NAME] inline.
  G3 – Conflict flag: if conflict_hint is passed, the model is told to surface
       the contradiction in its answer.
  G4 – No hallucination: the model is explicitly prohibited from using
       knowledge not present in the provided context.
  G5 – Escalation: the model is asked to recommend escalation when evidence is
       insufficient or contradictory.
"""
from __future__ import annotations

from app.models import RetrievedChunk


SYSTEM_PROMPT = """
You are an expert AI assistant specialised in market risk regulation and documentation.
You serve Risk Officers, Compliance Officers, Model Validators, and Internal Auditors
at a regulated banking institution.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
STRICT OPERATING RULES (must be followed in every response)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. CONTEXT-ONLY ANSWERS
   You MUST answer exclusively from the DOCUMENT CONTEXT provided below.
   Do NOT use general knowledge, training data, or inference beyond
   what is explicitly stated in the context.
   If the context does not contain sufficient information, say so clearly.

2. MANDATORY CITATIONS
   Every factual statement in your answer must include a citation in the
   format: [DOC_NAME, Section/Article if available].
   Example: "The capital ratio must exceed 8% [CRR_Article_92, Article 92(1)]."
   Never make an uncited claim.

3. CONFLICT DETECTION
   If retrieved chunks contradict each other (e.g., two documents specify
   different values for the same parameter), you MUST:
   a) Surface the contradiction explicitly.
   b) Identify which documents disagree.
   c) Recommend that the user verify with the source documents.

4. NO HALLUCINATION
   Do not invent figures, thresholds, dates, or regulatory references.
   If you are uncertain, say "The context does not confirm this."

5. ESCALATION
   If you cannot fully answer the question from the context, or a material
   conflict exists, end your answer with:
   "⚠️ ESCALATION RECOMMENDED: [brief reason]"

6. TONE & FORMAT
   - Be precise and professional.
   - Use bullet points or numbered lists for multi-part answers.
   - Do not add disclaimers unrelated to the question.
   - Do not pad answers with generic regulatory boilerplate.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
"""

NO_CONTEXT_ANSWER = (
    "I was unable to find relevant information in the knowledge base to answer "
    "this question. The corpus does not contain context that addresses your query.\n\n"
    "⚠️ ESCALATION RECOMMENDED: Please consult the source documents or a subject "
    "matter expert directly."
)


def build_user_message(
    query: str,
    chunks: list[RetrievedChunk],
    conflict_hint: str | None = None,
) -> str:
    """
    Build the user turn of the Claude conversation.

    Structure:
      [DOCUMENT CONTEXT]
        <chunk 1> …
        <chunk 2> …
      [CONFLICT ALERT]  ← only if conflict_hint is set
      [USER QUESTION]
    """
    if not chunks:
        return f"[NO CONTEXT AVAILABLE]\n\n[USER QUESTION]\n{query}"

    context_blocks: list[str] = []
    for i, rc in enumerate(chunks, start=1):
        m = rc.chunk.metadata
        header = (
            f"--- DOCUMENT {i} ---\n"
            f"Source: {m.doc_name}\n"
            f"Type: {m.doc_type.value}\n"
            f"Version: {m.version or 'N/A'}\n"
            f"Section: {m.section_title or 'N/A'}\n"
            f"Hierarchy: {m.hierarchy or 'N/A'}\n"
            f"Similarity: {rc.score:.3f}\n"
        )
        context_blocks.append(f"{header}\n{rc.chunk.text.strip()}")

    context_section = "\n\n".join(context_blocks)

    conflict_section = ""
    if conflict_hint:
        conflict_section = (
            f"\n\n[⚠️ CONFLICT ALERT]\n"
            f"The retrieved documents appear to contain contradictory information:\n"
            f"{conflict_hint}\n"
            f"You MUST address this conflict in your answer.\n"
        )

    return (
        f"[DOCUMENT CONTEXT]\n\n"
        f"{context_section}"
        f"{conflict_section}\n\n"
        f"[USER QUESTION]\n{query}"
    )
