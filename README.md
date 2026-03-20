# Market Risk AI Assistant

Production-grade Retrieval-Augmented Generation (RAG) system for market risk regulation queries in regulated banking environments.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        CLIENT (HTTP)                                │
└──────────────────────────────┬──────────────────────────────────────┘
                               │ POST /query
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     FastAPI Application                             │
│  ┌───────────────┐  ┌─────────────────┐  ┌──────────────────────┐  │
│  │  API Key Auth │  │ Request Logging  │  │   CORS Middleware    │  │
│  └───────────────┘  └─────────────────┘  └──────────────────────┘  │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      Query Orchestrator                             │
│                                                                     │
│  1. Retrieve top-k chunks (FAISS + metadata filters)               │
│  2. Governance gate  → refuse if no context                        │
│  3. Conflict detection (version mismatch, threshold disagreement)  │
│  4. Build prompt (system + injected context)                       │
│  5. Call Claude API                                                │
│  6. Format structured response                                     │
│  7. Write audit record (append-only JSONL)                        │
└──────┬────────────────────────────────────────────┬────────────────┘
       │                                            │
       ▼                                            ▼
┌──────────────────┐                  ┌─────────────────────────────┐
│  FAISS Vector DB │                  │     Anthropic Claude API    │
│                  │                  │  (claude-opus-4-5, T=0.0)  │
│ • IndexFlatIP    │                  │                             │
│ • JSON metadata  │                  │  System prompt:             │
│ • Cosine sim.    │                  │  • Context-only answers     │
│                  │                  │  • Mandatory citations      │
└──────────────────┘                  │  • Conflict surfacing       │
                                      │  • No hallucination         │
                                      │  • Escalation guidance      │
                                      └─────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                     Document Ingestion Pipeline                     │
│                                                                     │
│  PDF / DOCX / TXT → Text Extraction → Classification → Chunking    │
│  → Embedding (sentence-transformers) → FAISS Index + JSON metadata │
└─────────────────────────────────────────────────────────────────────┘
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| `IndexFlatIP` (exact search) | No recall trade-off; corpus < 100k chunks |
| `temperature=0.0` | Determinism required for audit |
| Cosine similarity via L2-normalised vectors | Standard for semantic search |
| Append-only JSONL audit log | Tamper-evident; meets audit requirements |
| Pre-LLM conflict detection | Fast, cheap; LLM-as-judge is a future enhancement |
| Metadata in JSON sidecar | FAISS doesn't natively store arbitrary metadata |

---

## Repository Structure

```
market-risk-rag/
├── main.py                     # FastAPI app factory + lifespan
├── requirements.txt
├── Procfile                    # Railway / Heroku process file
├── railway.toml                # Railway deployment configuration
├── pytest.ini
├── .env.example                # Environment variable template
├── .gitignore
│
├── app/
│   ├── models.py               # All Pydantic data models
│   ├── orchestrator.py         # End-to-end query pipeline
│   └── logger.py               # Structured logging + audit trail
│
├── retrieval/
│   ├── ingestion.py            # Document loading, chunking, classification
│   ├── embeddings.py           # sentence-transformers wrapper
│   └── vector_store.py         # FAISS index + metadata filtering
│
├── llm/
│   ├── prompts.py              # System prompt + context injection
│   ├── claude_client.py        # Anthropic SDK wrapper + retry
│   ├── conflict_detector.py    # Contradiction detection (pre-LLM)
│   └── answer_formatter.py     # Confidence scoring + response assembly
│
├── api/
│   ├── routes.py               # FastAPI route handlers
│   └── middleware.py           # Auth, CORS, request logging
│
├── config/
│   └── settings.py             # Pydantic-settings configuration
│
├── data/
│   └── documents/              # Place your PDF/DOCX/TXT files here
│
├── scripts/
│   ├── ingest_cli.py           # CLI: build FAISS index from documents
│   └── query_cli.py            # CLI: test queries without API server
│
└── tests/
    ├── conftest.py             # Shared fixtures (no real API key needed)
    ├── test_ingestion.py
    ├── test_conflict_detection.py
    ├── test_answer_formatter.py
    ├── test_api_routes.py
    └── test_governance.py
```

---

## Governance Rules

The system enforces five governance rules in every response:

| Rule | Enforcement Point |
|---|---|
| **G1**: No answer without context | `orchestrator.py` — returns `INSUFFICIENT` before calling LLM |
| **G2**: Mandatory citations | `prompts.py` — system prompt enforces `[DOC_NAME]` citation format |
| **G3**: Conflict detection | `conflict_detector.py` — checks version mismatches and threshold contradictions |
| **G4**: No hallucination | `prompts.py` — model explicitly prohibited from using knowledge outside context |
| **G5**: Escalation recommendation | `answer_formatter.py` — auto-set when confidence is LOW/INSUFFICIENT or conflict detected |

Every query is logged in full (prompt, raw LLM output, retrieved chunks, governance flags) to an append-only JSONL audit file.

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- An Anthropic API key ([get one here](https://console.anthropic.com/))
- Git

### 1. Clone the repository

```bash
git clone https://github.com/YOUR_ORG/market-risk-rag.git
cd market-risk-rag
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and set ANTHROPIC_API_KEY (and optionally API_KEY for endpoint protection)
```

### 5. Add your documents

Place your PDF, DOCX, and/or TXT files in `data/documents/`:

```bash
cp /path/to/your/CRR_Article_92.pdf       data/documents/
cp /path/to/your/Internal_Risk_Policy.docx data/documents/
cp /path/to/your/MR-001_Procedure.docx    data/documents/
```

**Document naming convention** (helps with auto-classification):

| Prefix | Classified as |
|---|---|
| `regulation_*` or contains "crr", "rts", "bis", "article" | `regulation` |
| `policy_*` or contains "policy", "framework" | `internal_policy` |
| `procedure_*` or contains "procedure", "sop" | `procedure` |
| `guidance_*` or contains "guidance", "guideline" | `guidance` |

### 6. Build the vector index

```bash
python scripts/ingest_cli.py
```

Expected output:
```
✅ Ingestion complete.
   Documents : 5
   Chunks    : 847
   Index path: data/faiss_index
   Documents indexed:
     • CRR_Article_92
     • Internal_Market_Risk_Policy
     • MR-001_Market_Risk_Reporting
     ...
```

### 7. Start the API server

```bash
python main.py
# Or with uvicorn directly:
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

The API will be available at `http://localhost:8000`.
Interactive docs: `http://localhost:8000/docs`

---

## Usage Examples

### Query via cURL

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the minimum Total Capital Ratio under CRR Article 92?"}'
```

**Response:**
```json
{
  "answer": "Under CRR Article 92(1)(c), institutions must maintain a Total Capital Ratio of at least 8% at all times [CRR_Article_92, Article 92(1)(c)]. This comprises: Common Equity Tier 1 of at least 4.5% [CRR_Article_92, Article 92(1)(a)] and Tier 1 Capital of at least 6% [CRR_Article_92, Article 92(1)(b)].",
  "sources": [
    {
      "doc_name": "CRR_Article_92",
      "doc_type": "regulation",
      "version": "2.1",
      "section_title": "Article 92 – Own funds requirements",
      "chunk_excerpt": "Article 92 of the Capital Requirements Regulation (CRR) requires institutions to maintain a Total Capital Ratio...",
      "similarity_score": 0.9142,
      "hierarchy": "CRR > Article 92 > Own funds requirements"
    }
  ],
  "confidence": "high",
  "conflict_detected": false,
  "conflict_details": null,
  "escalation_recommended": false,
  "escalation_reason": null,
  "query_id": "3f4a1b2c-...",
  "processed_at": "2025-01-15T09:32:11.412Z"
}
```

### Query via CLI (no API server needed)

```bash
python scripts/query_cli.py "What are the VaR reporting requirements?"
python scripts/query_cli.py --filter-type regulation "What approaches exist for market risk capital?"
python scripts/query_cli.py --json "Compare CRR vs internal policy on capital targets"
```

### With API key protection

```bash
# Set API_KEY=mysecretkey in .env, then:
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -H "X-API-Key: mysecretkey" \
  -d '{"query": "What is the VaR reporting frequency?"}'
```

### Filter by document type or name

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is the capital buffer target?",
    "filter_doc_type": "internal_policy",
    "filter_doc_name": "Market_Risk_Policy",
    "top_k": 5
  }'
```

### Trigger ingestion via API

```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"directory": "data/documents"}'
```

### List indexed documents

```bash
curl http://localhost:8000/documents
```

### Check system health

```bash
curl http://localhost:8000/health
```

---

## Deployment on Railway

### Step 1: Push to GitHub

```bash
git init
git add .
git commit -m "Initial commit: Market Risk AI Assistant"
git remote add origin https://github.com/YOUR_ORG/market-risk-rag.git
git push -u origin main
```

### Step 2: Create a Railway project

1. Go to [railway.app](https://railway.app) and sign in.
2. Click **New Project → Deploy from GitHub repo**.
3. Select your repository.

### Step 3: Set environment variables

In the Railway dashboard → your service → **Variables**, set:

| Variable | Value |
|---|---|
| `ANTHROPIC_API_KEY` | `sk-ant-...` |
| `API_KEY` | A strong random string (to protect endpoints) |
| `APP_ENV` | `production` |
| `DEBUG` | `false` |

Railway will inject `PORT` automatically. The `Procfile` / `railway.toml` uses `$PORT`.

### Step 4: Add persistent storage for documents and index

Railway volumes are required to persist the FAISS index and audit logs across deploys:

1. In Railway: **New → Volume** → attach to your service.
2. Mount path: `/data`
3. Update environment variables:
   - `DATA_DIR=/data/documents`
   - `FAISS_INDEX_PATH=/data/faiss_index`
   - `FAISS_METADATA_PATH=/data/faiss_metadata.json`
   - `AUDIT_LOG_PATH=/data/logs/audit.jsonl`

### Step 5: Upload documents and trigger ingestion

After deployment, upload your documents to the Railway volume (via Railway CLI or SSH), then:

```bash
curl -X POST https://your-app.railway.app/ingest \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d '{"directory": "/data/documents"}'
```

### Step 6: Verify

```bash
curl https://your-app.railway.app/health
```

---

## Running Tests

```bash
# All unit tests (fast, no API key required)
pytest tests/test_ingestion.py tests/test_conflict_detection.py \
       tests/test_answer_formatter.py tests/test_governance.py -v

# API route tests (require embedding model download ~80MB first run)
pytest tests/test_api_routes.py -v

# All tests
pytest -v

# Skip slow integration tests
pytest -v -m "not integration and not llm"
```

---

## Supported Use Cases

| Use Case | Example Query |
|---|---|
| **UC01** – Explain regulatory requirement | "What are the capital requirements under CRR Article 92?" |
| **UC02** – Compare frameworks | "How does CRR differ from our internal policy on capital buffers?" |
| **UC03** – Detect inconsistency | "Are there any contradictions in capital ratio definitions across our documents?" |
| **UC04** – Locate document section | "Where is the VaR reporting procedure defined?" |
| **UC05** – Handle insufficient evidence | "What is the required LCR for ring-fenced entities?" (if not in corpus) |

---

## Audit Trail

Every query is logged to `AUDIT_LOG_PATH` (default: `logs/audit.jsonl`) as a JSONL record containing:

```json
{
  "audit_id": "uuid",
  "timestamp": "ISO-8601",
  "query_id": "uuid",
  "query": "user's question",
  "filter_doc_type": null,
  "filter_doc_name": null,
  "chunks_retrieved": [
    {"chunk_id": "...", "doc_name": "...", "score": 0.91, "excerpt": "..."}
  ],
  "prompt_user_message": "full prompt sent to LLM",
  "raw_llm_answer": "raw text from Claude",
  "confidence": "high",
  "conflict_detected": false,
  "conflict_details": null,
  "escalation_recommended": false,
  "escalation_reason": null
}
```

This log is append-only and must be stored in a tamper-evident system (WORM storage, SIEM) for regulated environments.

---

## Environment Variables Reference

| Variable | Required | Default | Description |
|---|---|---|---|
| `ANTHROPIC_API_KEY` | ✅ | — | Anthropic API key |
| `CLAUDE_MODEL` | | `claude-opus-4-5` | Model ID |
| `CLAUDE_MAX_TOKENS` | | `2048` | Max tokens in response |
| `CLAUDE_TEMPERATURE` | | `0.0` | Keep at 0.0 for auditability |
| `DATA_DIR` | | `data/documents` | Document directory |
| `FAISS_INDEX_PATH` | | `data/faiss_index` | FAISS index file |
| `FAISS_METADATA_PATH` | | `data/faiss_metadata.json` | Metadata sidecar |
| `TOP_K_RETRIEVAL` | | `8` | Chunks to retrieve per query |
| `MIN_SIMILARITY_SCORE` | | `0.35` | Minimum cosine similarity |
| `CHUNK_SIZE` | | `512` | Characters per chunk |
| `CHUNK_OVERLAP` | | `64` | Overlap between chunks |
| `AUDIT_LOG_PATH` | | `logs/audit.jsonl` | Audit log location |
| `API_KEY` | | (disabled) | Bearer token for API protection |
| `APP_ENV` | | `development` | `development` or `production` |
| `DEBUG` | | `false` | Enable debug logging |

---

## Assumptions & Limitations

1. **Document classification** is heuristic (keyword-based). For higher accuracy, add document type as an explicit metadata field in the filename or a companion YAML manifest.

2. **Conflict detection** is pre-LLM and heuristic. It catches version mismatches and numeric threshold disagreements but will not catch all semantic contradictions. A future LLM-as-judge pass would improve recall.

3. **FAISS IndexFlatIP** is exact (brute-force) search. For corpora > 500k chunks, switch to `IndexIVFFlat` with an appropriate number of centroids.

4. **Embeddings** use `all-MiniLM-L6-v2` (fast, 384 dimensions). For higher semantic accuracy in legal/regulatory text, consider `legal-bert-base-uncased` or a domain-fine-tuned model.

5. **No multi-turn memory**: each query is stateless. Conversation history is not passed to the LLM.

6. **Railway persistence**: Railway's ephemeral file system means the FAISS index is lost on redeploy unless a Volume is mounted.

---

## License

Proprietary — for internal use within the institution. All interactions are subject to the institution's data governance and AI usage policies.
