#!/usr/bin/env python
"""
CLI Ingestion Script
====================
Run this before starting the API server if you want to pre-build the index.

Usage:
    python scripts/ingest_cli.py
    python scripts/ingest_cli.py --data-dir /path/to/docs
    python scripts/ingest_cli.py --data-dir /path/to/docs --chunk-size 512 --chunk-overlap 64
"""
from __future__ import annotations

import argparse
import sys
import os

# Allow running from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.logger import configure_logging, get_logger
from config import get_settings
from retrieval.ingestion import ingest_directory
from retrieval.vector_store import get_vector_store

configure_logging("INFO")
logger = get_logger("ingest_cli")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents and build FAISS index.")
    parser.add_argument("--data-dir", default=None, help="Directory containing documents.")
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--chunk-overlap", type=int, default=None)
    args = parser.parse_args()

    settings = get_settings()
    data_dir = args.data_dir or settings.DATA_DIR
    chunk_size = args.chunk_size or settings.CHUNK_SIZE
    chunk_overlap = args.chunk_overlap or settings.CHUNK_OVERLAP

    logger.info("cli.start", data_dir=data_dir, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    chunks = list(ingest_directory(data_dir, chunk_size, chunk_overlap))

    if not chunks:
        logger.error("cli.no_chunks", data_dir=data_dir)
        print(f"\n❌ No documents found in '{data_dir}'. Place PDF/DOCX/TXT files there first.")
        sys.exit(1)

    store = get_vector_store()
    store.build(chunks)

    doc_names = sorted({c.metadata.doc_name for c in chunks})
    print(f"\n✅ Ingestion complete.")
    print(f"   Documents : {len(doc_names)}")
    print(f"   Chunks    : {len(chunks)}")
    print(f"   Index path: {settings.FAISS_INDEX_PATH}")
    print("\n   Documents indexed:")
    for name in doc_names:
        print(f"     • {name}")


if __name__ == "__main__":
    main()
