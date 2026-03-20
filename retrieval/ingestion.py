"""
Document Ingestion Pipeline
===========================
Supports PDF, DOCX, and TXT.

Pipeline:
  1. Discover files in the data directory.
  2. Extract raw text + page/section metadata.
  3. Classify each document (regulation, internal policy, procedure…).
  4. Split text into overlapping chunks.
  5. Return DocumentChunk objects ready for embedding.

Assumptions (documented):
  - Filenames encode document type when the document body does not.
    Pattern: <type>_<name>.<ext>  e.g. regulation_CRR_Article325.pdf
    Type token: "regulation", "policy", "procedure", "guidance".
  - If the filename pattern is absent the type defaults to UNKNOWN.
  - Version information is extracted from a "Version:" line in the first page.
"""
from __future__ import annotations

import hashlib
import re
import uuid
from pathlib import Path
from typing import Iterator

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.logger import get_logger
from app.models import DocumentChunk, DocumentMetadata, DocumentType

logger = get_logger(__name__)

# ── Text extractors ──────────────────────────────────────────────────────────

def _extract_pdf(path: Path) -> list[tuple[int, str]]:
    """Return list of (page_number, text) tuples."""
    from pypdf import PdfReader
    reader = PdfReader(str(path))
    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append((i, text))
    return pages


def _extract_docx(path: Path) -> list[tuple[int, str]]:
    """DOCX has no pages; we treat the whole doc as page 1."""
    from docx import Document as DocxDocument
    doc = DocxDocument(str(path))
    full_text = "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    return [(1, full_text)]


def _extract_txt(path: Path) -> list[tuple[int, str]]:
    return [(1, path.read_text(encoding="utf-8", errors="replace"))]


_EXTRACTORS = {
    ".pdf": _extract_pdf,
    ".docx": _extract_docx,
    ".txt": _extract_txt,
}

# ── Document type classification ─────────────────────────────────────────────

_TYPE_KEYWORDS: dict[DocumentType, list[str]] = {
    DocumentType.REGULATION: ["crr", "rts", "bis", "regulation", "directive", "article", "basel"],
    DocumentType.INTERNAL_POLICY: ["policy", "internal", "framework"],
    DocumentType.PROCEDURE: ["procedure", "process", "workflow", "sop"],
    DocumentType.GUIDANCE: ["guidance", "guideline", "note"],
}


def _classify_document(name_lower: str, first_500_chars: str) -> DocumentType:
    text = name_lower + " " + first_500_chars.lower()
    scores: dict[DocumentType, int] = {dt: 0 for dt in DocumentType}
    for doc_type, keywords in _TYPE_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[doc_type] += 1
    best = max(scores, key=lambda dt: scores[dt])
    return best if scores[best] > 0 else DocumentType.UNKNOWN


def _extract_version(text: str) -> str | None:
    """Try to find a version string like 'Version: 2.1' in the document."""
    m = re.search(r"[Vv]ersion\s*[:\-]?\s*([0-9]+(?:\.[0-9]+)*)", text)
    return m.group(1) if m else None


def _extract_section_title(chunk_text: str) -> str | None:
    """Heuristic: first line that looks like a heading (short, capitalised)."""
    for line in chunk_text.split("\n"):
        line = line.strip()
        if 5 < len(line) < 120 and (line.isupper() or re.match(r"^(Article|Section|§|Chapter)\s+", line, re.I)):
            return line
    return None


# ── Main ingestion function ──────────────────────────────────────────────────

def ingest_directory(
    directory: str | Path,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> Iterator[DocumentChunk]:
    """
    Walk *directory* and yield DocumentChunk objects for every supported file.
    """
    directory = Path(directory)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    supported = set(_EXTRACTORS.keys())
    files = [f for f in directory.rglob("*") if f.suffix.lower() in supported]

    if not files:
        logger.warning("ingestion.no_files_found", directory=str(directory))
        return

    for file_path in files:
        try:
            yield from _ingest_file(file_path, splitter)
        except Exception as exc:
            logger.error("ingestion.file_failed", file=str(file_path), error=str(exc))


def _ingest_file(
    file_path: Path,
    splitter: RecursiveCharacterTextSplitter,
) -> Iterator[DocumentChunk]:
    ext = file_path.suffix.lower()
    extractor = _EXTRACTORS[ext]
    pages: list[tuple[int, str]] = extractor(file_path)

    full_text = " ".join(text for _, text in pages)
    doc_type = _classify_document(file_path.stem.lower(), full_text[:500])
    version = _extract_version(full_text[:2000])

    doc_id = hashlib.md5(str(file_path).encode()).hexdigest()[:12]

    chunk_idx = 0
    for page_num, page_text in pages:
        if not page_text.strip():
            continue
        raw_chunks = splitter.split_text(page_text)
        for raw_chunk in raw_chunks:
            if not raw_chunk.strip():
                continue
            chunk_id = f"{doc_id}_{chunk_idx}"
            metadata = DocumentMetadata(
                doc_id=doc_id,
                doc_name=file_path.stem,
                doc_type=doc_type,
                version=version,
                source_file=str(file_path),
                chunk_index=chunk_idx,
                page_number=page_num,
                section_title=_extract_section_title(raw_chunk),
                hierarchy=_build_hierarchy(file_path.stem, raw_chunk),
            )
            yield DocumentChunk(
                chunk_id=chunk_id,
                text=raw_chunk,
                metadata=metadata,
            )
            chunk_idx += 1

    logger.info(
        "ingestion.file_done",
        file=file_path.name,
        doc_type=doc_type.value,
        chunks=chunk_idx,
    )


def _build_hierarchy(doc_name: str, chunk_text: str) -> str | None:
    """
    Build a simple hierarchy string for display in citations.
    E.g. "CRR > Article 325 > Capital requirements"
    """
    article_match = re.search(r"(Article|Section|§)\s+(\w+)", chunk_text, re.I)
    section = _extract_section_title(chunk_text)
    parts = [doc_name]
    if article_match:
        parts.append(article_match.group(0))
    if section and section != article_match.group(0) if article_match else section:
        parts.append(section[:60])
    return " > ".join(parts) if len(parts) > 1 else None
