# Document Audio Generation Service

A standalone FastAPI microservice that processes uploaded documents, chunks content semantically, generates embeddings, and converts educational responses into audio format.

## Features

- **Document Processing** — Upload PDF, DOCX, TXT, and Markdown files with automatic text extraction
- **Semantic Chunking** — Split documents into meaningful chunks respecting paragraph/sentence boundaries
- **Embedding Generation** — Create vector embeddings using open-source sentence-transformers
- **Similarity Search** — Find relevant content using cosine similarity over DuckDB + Parquet
- **Two Response Modes:**
  - **Explain** — Clear, simple explanation for beginners
  - **Teach** — Structured lesson with intro, concepts, examples, and takeaways
- **Audio Generation** — Convert responses to speech using Edge TTS (Microsoft Neural voices)
- **Predefined Content** — Generate teaching audio for predefined topics and subtopics

## Quick Start

```bash
# 1. Clone / navigate to project
cd document-audio-generation-service

# 2. Create virtual environment
python -m venv venv
venv\Scripts\activate   # Windows
# source venv/bin/activate  # Linux/Mac

# 3. Install dependencies
pip install -r requirements.txt

# 4. Copy environment config
copy .env.example .env   # Windows
# cp .env.example .env   # Linux/Mac

# 5. Start the server
python -m uvicorn app.main:app --reload --port 8001
```

Open **http://localhost:8001/docs** for interactive API documentation.

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/v1/documents/upload` | Upload and process a document |
| GET | `/api/v1/documents` | List all documents |
| GET | `/api/v1/documents/{id}` | Get document details |
| DELETE | `/api/v1/documents/{id}` | Delete a document |
| GET | `/api/v1/chunks` | List chunks (optionally by document) |
| POST | `/api/v1/chunks/search` | Semantic search across chunks |
| POST | `/api/v1/query` | Query with explain/teach mode |
| POST | `/api/v1/query/explain` | Explain mode shortcut |
| POST | `/api/v1/query/teach` | Teach mode shortcut |
| POST | `/api/v1/query/predefined` | Predefined topic teaching |
| POST | `/api/v1/audio/generate` | Generate audio from a response |
| POST | `/api/v1/audio/synthesize` | Direct text-to-audio |
| GET | `/api/v1/audio/{id}/download` | Download audio file |
| GET | `/health` | Health check |

## Architecture

```
app/
├── main.py                     # FastAPI app + lifespan
├── core/                       # Config, constants
├── models/                     # Pydantic request/response schemas
├── endpoints/                  # API route handlers
├── services/
│   ├── document_processing/   # Upload, parse, chunk
│   ├── embedding_generation/  # Embeddings, vectors, search
│   ├── content_analysis/      # Query processing, response generation
│   └── audio_generation/      # TTS, audio processing
├── infrastructure/
│   ├── data_access/           # DuckDB, Parquet, document store
│   ├── external_apis/         # LLM client, model manager
│   └── cache/                 # Embedding + response caching
├── prompts/                   # System prompts (explain/teach)
└── utils/                     # File, text, error utilities
```

## Tech Stack

- **Framework:** FastAPI + Uvicorn
- **Embeddings:** sentence-transformers (all-MiniLM-L6-v2)
- **TTS:** edge-tts (Microsoft Edge Neural voices)
- **LLM:** Ollama (local) or Azure OpenAI
- **Storage:** DuckDB + Apache Parquet
- **Document Parsing:** PyPDF2, python-docx

## Configuration

Set `APP_ENVIRONMENT` to `DEV`, `QA`, or `PROD` for environment-specific defaults. See `.env.example` for all available settings.

## License

Internal use only.
