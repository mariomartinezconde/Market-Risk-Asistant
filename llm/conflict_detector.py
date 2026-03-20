"""
Conflict Detector
=================
Heuristic detection of contradictions between retrieved chunks.

Approach (pre-LLM, lightweight):
  1. Numeric conflict: same numeric metric appears with different values in
     different documents (e.g., "8%" in CRR vs "10%" in internal policy).
  2. Date conflict: same effective-date field differs across docs.
  3. Version conflict: two chunks from the same doc name but different versions.

A more sophisticated approach (LLM-as-judge) is a future enhancement.
The current approach is intentionally conservative (low false-positive rate)
to avoid flooding users with spurious conflict warnings.
"""
from __future__ import annotations

import re
from collections import defaultdict

from app.models import RetrievedChunk


_PERCENTAGE_PATTERN = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_THRESHOLD_KEYWORDS = re.compile(
    r"(minimum|maximum|must\s+not\s+exceed|shall\s+be|is\s+set\s+at|threshold|limit|ratio)\s+"
    r"(of\s+)?(\d+(?:\.\d+)?)\s*(%|percent|basis\s+points?|bps)",
    re.IGNORECASE,
)


def detect_conflicts(chunks: list[RetrievedChunk]) -> tuple[bool, str | None]:
    """
    Analyse retrieved chunks for contradictions.

    Returns:
        (conflict_detected: bool, conflict_description: str | None)
    """
    if len(chunks) < 2:
        return False, None

    conflicts: list[str] = []

    # 1. Version conflict (same doc_name, different version)
    version_by_doc: dict[str, set[str]] = defaultdict(set)
    for rc in chunks:
        m = rc.chunk.metadata
        if m.version:
            version_by_doc[m.doc_name].add(m.version)
    for doc_name, versions in version_by_doc.items():
        if len(versions) > 1:
            conflicts.append(
                f"Document '{doc_name}' has multiple versions in context: "
                f"{', '.join(sorted(versions))}. Ensure the correct version is used."
            )

    # 2. Numeric threshold conflict across different source documents
    threshold_map: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    # threshold_map[keyword_context][percentage_value] -> {doc_name, ...}
    for rc in chunks:
        for match in _THRESHOLD_KEYWORDS.finditer(rc.chunk.text):
            keyword = match.group(1).strip().lower()
            value = f"{match.group(3)}{match.group(4)}"
            threshold_map[keyword][value].add(rc.chunk.metadata.doc_name)

    for keyword, value_docs in threshold_map.items():
        if len(value_docs) > 1:
            # Multiple different values for the same keyword
            summary_parts = [f"{val} (in: {', '.join(sorted(docs))})" for val, docs in value_docs.items()]
            conflicts.append(
                f"Conflicting threshold for '{keyword}': {' vs '.join(summary_parts)}."
            )

    if conflicts:
        return True, "\n".join(f"• {c}" for c in conflicts)
    return False, None
