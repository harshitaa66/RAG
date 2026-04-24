# RAG — Modular Retrieval-Augmented Generation System

A production-ready, modular RAG chatbot with hybrid retrieval, live web crawling fallback, image/OCR processing, and a React chat interface. Built with FastAPI and designed to outperform both vanilla LLMs and standard RAG pipelines on quality and latency.

---

## Architecture Overview

The system implements a **three-tier answer chain**:

```
User Query
    │
    ▼
Hybrid Search (BM25 + Cosine Similarity)
    │
    ├─ Score ≥ 0.35 → call_rag() → GPT-3.5-turbo with retrieved context
    │                      │
    │                      └─ "Not in context" → fallback()
    │
    └─ Score < 0.35 → fallback()
                           │
                           ├─ Domain: products  → Amazon / eBay scraper
                           ├─ Domain: academic  → arXiv / Semantic Scholar
                           ├─ Domain: general   → DuckDuckGo (via ddgs)
                           │
                           └─ Crawler fails / timeout → Pure LLM answer
```

### System Comparison

| System | Context Precision | Answer Relevancy | Faithfulness | Latency |
|---|---|---|---|---|
| Vanilla LLM | 0.580 | 0.620 | 0.640 | 2800 ms |
| Standard RAG | 0.650 | 0.800 | 0.720 | 1500 ms |
| Proposed Modular RAG | 0.677 | 0.833 | 0.667 | 1160 ms |
| **Improved Modular RAG** | **0.750** | **0.900** | **0.850** | **1280 ms** |

The Improved system adds **hybrid BM25 + cosine retrieval** and **crawler escalation**, beating Standard RAG by +10.8% on Answer Relevancy and +18.1% on Faithfulness while remaining faster than Standard RAG.

---

## Features

- **Hybrid Retrieval** — Combines FAISS cosine similarity (dense, semantic) with BM25 (sparse, keyword) at a 70/30 weighting for superior document matching
- **Live Crawler Fallback** — When the knowledge base can't answer, the system automatically fetches live web data via domain-specific crawlers
- **Domain Detection** — Automatically routes queries to the right crawler: product queries → e-commerce, academic queries → research databases, everything else → DuckDuckGo
- **Image & OCR Support** — Upload images; text is extracted via Tesseract OCR and fed into the RAG pipeline
- **Batch File Upload** — Upload PDFs, DOCX, and images in bulk; documents are chunked and indexed automatically
- **JWT Authentication** — Secure token-based auth with login/logout flow
- **React Chat Interface** — Clean frontend with text query, file upload, and batch upload components
- **PostgreSQL Storage** — Documents and user data stored in Postgres via psycopg2

---

## Project Structure

```
Production/
├── Backend/
│   ├── main.py          # Core RAG engine: hybrid search, crawler integration, LLM calls
│   ├── Api.py           # FastAPI routes: /query, /file, /batch_files, /token
│   ├── DB.py            # PostgreSQL document and auth database layer
│   ├── logs.py          # Logging configuration
│   ├── Hash.py          # Password hashing utility
│   ├── Crawl.py         # Advanced web crawler (Crawl4AI + Tavily integration)
│   ├── ingest_data.py   # Bulk document ingestion pipeline
│   ├── evaluate.py      # RAG evaluation suite (RAGAS metrics)
│   ├── my_rag_figures.py# Performance comparison bar chart generator
│   └── start.bat        # Windows launcher (activates Rag_env + starts uvicorn)
│
└── chat_interface/      # React frontend
    └── src/
        ├── components/
        │   ├── TextQuery.js    # Multi-query text input
        │   ├── FileUpload.js   # Single file upload
        │   └── BatchUpload.js  # Bulk file upload
        └── Services/
            └── api.js          # Axios API client with JWT interceptor
```

---

## Tech Stack

**Backend**
- Python 3.10
- FastAPI + Uvicorn
- FAISS (`IndexFlatIP` with normalized vectors = cosine similarity)
- `rank_bm25` — BM25Okapi sparse index
- `sentence-transformers` — `all-MiniLM-L6-v2` embedding model
- OpenAI `gpt-3.5-turbo` (async client)
- `ddgs` — DuckDuckGo search (bot-detection bypass)
- Tesseract OCR + OpenCV
- PostgreSQL + psycopg2
- passlib + bcrypt — password hashing
- PyJWT — token auth

**Frontend**
- React 18
- Axios with request/response interceptors

---

## Setup

### Prerequisites
- Conda environment `Rag_env` with Python 3.10
- PostgreSQL running locally
- Tesseract OCR installed
- OpenAI API key

### Install dependencies

```bash
conda activate Rag_env
pip install fastapi uvicorn sentence-transformers faiss-cpu rank-bm25 openai \
            passlib[bcrypt] bcrypt==3.2.2 PyJWT python-dotenv psycopg2-binary \
            pypdf python-docx pytesseract opencv-python pillow ddgs
```

### Environment variables

Create `Backend/.env`:

```env
OPENAI_API_KEY=your_openai_key
POSTGRES=your_postgres_connection_string
POSTGRES_DSN=your_postgres_dsn
SECRET_KEY=your_jwt_secret_key
```

### Run the backend

```bash
# Windows — double-click or run from terminal:
Backend\start.bat

# Or manually:
conda activate Rag_env
cd Production/Backend
uvicorn Api:app --reload --host 127.0.0.1 --port 8000
```

### Run the frontend

```bash
cd Production/chat_interface
npm install
npm start
```

The app will be at `http://localhost:3000`, API at `http://localhost:8000`.

---

## API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| POST | `/token` | Login — returns JWT access token |
| POST | `/query` | Submit text queries → RAG/crawler answer |
| POST | `/file` | Upload single file (PDF, DOCX, image) |
| POST | `/batch_files` | Upload multiple files at once |

---

## How Hybrid Retrieval Works

1. **Dense search** — query is embedded with `all-MiniLM-L6-v2`, then matched against document embeddings using FAISS cosine similarity (`IndexFlatIP` on L2-normalized vectors)
2. **Sparse search** — BM25Okapi scores all documents for keyword overlap
3. **Fusion** — scores are combined: `0.7 × cosine_norm + 0.3 × bm25_norm`
4. **Threshold** — if the top combined score ≥ 0.35, the top-3 documents are passed to GPT-3.5-turbo; otherwise the crawler fallback is triggered

This hybrid approach captures both semantic similarity (cosine) and exact keyword matches (BM25), outperforming either method alone.
