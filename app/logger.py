"""
Structured, audit-grade logging for the Market Risk AI Assistant.

Every query and its full context (retrieved chunks, model response, governance
flags) is written to an append-only JSONL audit log file so that any answer
can be reconstructed and reviewed post-hoc.
"""
import json
import os
import uuid
import structlog
from datetime import datetime
from pathlib import Path
from typing import Any


def configure_logging(log_level: str = "INFO") -> None:
    """Configure structlog for structured JSON output."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(__import__("logging"), log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
    )


def get_logger(name: str) -> structlog.BoundLogger:
    return structlog.get_logger(name)


class AuditLogger:
    """
    Append-only JSONL audit trail.

    Each record contains:
    - query_id, timestamp
    - raw query
    - retrieved chunks (with scores)
    - prompt sent to the LLM
    - raw LLM response
    - governance flags (conflict, confidence, escalation)
    - final answer delivered to the user
    """

    def __init__(self, log_path: str) -> None:
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, event: dict[str, Any]) -> None:
        record = {
            "audit_id": str(uuid.uuid4()),
            "timestamp": datetime.utcnow().isoformat(),
            **event,
        }
        with open(self.log_path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")


# Singleton – created once at startup
_audit_logger: AuditLogger | None = None


def get_audit_logger(log_path: str | None = None) -> AuditLogger:
    global _audit_logger
    if _audit_logger is None:
        if log_path is None:
            log_path = os.getenv("AUDIT_LOG_PATH", "logs/audit.jsonl")
        _audit_logger = AuditLogger(log_path)
    return _audit_logger
