"""Split document text into overlapping chunks."""
from __future__ import annotations

import re
import hashlib
from dataclasses import dataclass, field
from typing import Optional

from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.core.config import get_settings


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    page_number: int
    chunk_index: int
    section_title: Optional[str] = None
    hierarchy: Optional[str] = None


def make_chunks(
    doc_id: str,
    doc_name: str,
    pages: list[tuple[int, str]],
) -> list[Chunk]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks: list[Chunk] = []
    idx = 0
    for page_num, text in pages:
        if not text.strip():
            continue
        for raw in splitter.split_text(text):
            if not raw.strip():
                continue
            chunk_id = hashlib.md5(f"{doc_id}_{idx}".encode()).hexdigest()[:16]
            section = _extract_section(raw)
            hierarchy = _build_hierarchy(doc_name, raw, section)
            chunks.append(Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                text=raw,
                page_number=page_num,
                chunk_index=idx,
                section_title=section,
                hierarchy=hierarchy,
            ))
            idx += 1
    return chunks


def _extract_section(text: str) -> Optional[str]:
    for line in text.split("\n"):
        line = line.strip()
        if 5 < len(line) < 120 and re.match(r"^(Article|Section|§|Chapter|\d+\.)\s+", line, re.I):
            return line
    return None


def _build_hierarchy(doc_name: str, text: str, section: Optional[str]) -> Optional[str]:
    parts = [doc_name]
    m = re.search(r"(Article|Section)\s+\w+", text, re.I)
    if m:
        parts.append(m.group(0))
    if section and section != (m.group(0) if m else None):
        parts.append(section[:60])
    return " > ".join(parts) if len(parts) > 1 else None
