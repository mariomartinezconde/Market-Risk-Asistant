#!/usr/bin/env python
"""
Query CLI — test the RAG pipeline from the command line without the API server.

Usage:
    python scripts/query_cli.py "What are the capital requirements under CRR Article 92?"
    python scripts/query_cli.py --filter-type regulation "What is the minimum capital ratio?"
"""
from __future__ import annotations

import argparse
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.logger import configure_logging
from app.models import DocumentType, QueryRequest
from app.orchestrator import answer_query
from config import get_settings
from retrieval.vector_store import get_vector_store

configure_logging("WARNING")  # quiet for CLI


def main():
    parser = argparse.ArgumentParser(description="Query the Market Risk AI Assistant from CLI.")
    parser.add_argument("query", help="Your question.")
    parser.add_argument("--filter-type", choices=[dt.value for dt in DocumentType], default=None)
    parser.add_argument("--filter-doc", default=None, help="Filter by document name substring.")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--json", action="store_true", help="Output raw JSON.")
    args = parser.parse_args()

    settings = get_settings()
    store = get_vector_store()
    if not store.load():
        print("❌ No vector index found. Run: python scripts/ingest_cli.py")
        sys.exit(1)

    request = QueryRequest(
        query=args.query,
        filter_doc_type=DocumentType(args.filter_type) if args.filter_type else None,
        filter_doc_name=args.filter_doc,
        top_k=args.top_k,
    )

    print(f"\n🔍 Query: {args.query}\n")
    response = answer_query(request)

    if args.json:
        print(json.dumps(response.model_dump(), indent=2, default=str))
        return

    # Pretty print
    print("━" * 70)
    print(f"📋 ANSWER\n")
    print(response.answer)
    print()

    print("━" * 70)
    print(f"📚 SOURCES ({len(response.sources)} retrieved)\n")
    for i, src in enumerate(response.sources, 1):
        print(f"  [{i}] {src.doc_name} (v{src.version or 'N/A'}) — score: {src.similarity_score:.3f}")
        if src.section_title:
            print(f"      Section: {src.section_title}")
        print(f"      Excerpt: {src.chunk_excerpt[:120]}…")
        print()

    print("━" * 70)
    conf_icon = {"high": "🟢", "medium": "🟡", "low": "🟠", "insufficient": "🔴"}.get(response.confidence.value, "⚪")
    print(f"{conf_icon} Confidence       : {response.confidence.value.upper()}")
    print(f"{'⚠️' if response.conflict_detected else '✅'} Conflict detected : {response.conflict_detected}")
    if response.conflict_details:
        print(f"   {response.conflict_details}")
    print(f"{'⚠️' if response.escalation_recommended else '✅'} Escalation needed : {response.escalation_recommended}")
    if response.escalation_reason:
        print(f"   Reason: {response.escalation_reason}")
    print(f"\n   Query ID: {response.query_id}")
    print("━" * 70)


if __name__ == "__main__":
    main()
