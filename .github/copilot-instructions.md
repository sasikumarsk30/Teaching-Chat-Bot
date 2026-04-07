# AI Agent Instructions for Document Audio Generation Service

## 🎯 Architecture Overview

This is a **standalone Document Processing + Audio Generation Service** built with FastAPI. It processes uploaded documents, chunks content semantically, generates embeddings, and converts teaching/explanatory responses into audio format using open-source models.

### Core Components

```
┌──────────────────────────────────────────────────────────────┐
│                 FastAPI Main App (app/main.py)                │
│  ├─ /api/v1/documents/*  - Document Upload & Management      │
│  ├─ /api/v1/chunks/*     - Chunk Listing & Search            │
│  ├─ /api/v1/query/*      - Query, Explain, Teach             │
│  ├─ /api/v1/audio/*      - Audio Generation & Download       │
│  └─ Lifespan Manager (startup/shutdown)                      │
└──────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐     ┌────────────────┐    ┌──────────────┐
│   Document   │     │   Embedding    │    │   Content    │
│   Processing │     │   Generation   │    │   Analysis   │
│  (Chunking,  │────▶│  (Vectors,     │◀───│  (Retrieve,  │
│   Parsing)   │     │   Parquet)     │    │   Generate)  │
└──────────────┘     └────────────────┘    └──────────────┘
        │                     │                     │
        ▼                     ▼                     ▼
┌──────────────┐     ┌────────────────┐    ┌──────────────┐
│  DuckDB      │     │  Parquet       │    │  Audio       │
│  (Metadata   │     │  (Chunks +     │    │  Generation  │
│   + Index)   │     │   Embeddings)  │    │  (TTS)       │
└──────────────┘     └────────────────┘    └──────────────┘
```

### Service Organization

**Services under `app/services/`:**

- **`document_processing/`** - Document upload, parsing, semantic chunking, metadata management
- **`embedding_generation/`** - Embedding creation, vector storage (Parquet), similarity search (DuckDB)
- **`content_analysis/`** - User query processing, content retrieval, LLM response generation
- **`audio_generation/`** - Text-to-speech conversion, audio processing, speech style management

---

## 🔄 Data Flow & Processing Pipeline

### Full Pipeline Flow

```
User Upload → DocumentIngestionService.upload_document()
    ├─ Validate file type (PDF, DOCX, TXT, MD)
    ├─ Parse content
    ├─ Store original in /data/documents/
    └─ Store metadata in DuckDB
    ↓
ChunkingService.chunk_document()
    ├─ Analyze structure (paragraphs, sections, headings)
    ├─ Apply semantic chunking (respect boundaries)
    ├─ Generate chunk metadata (source, position, context)
    └─ Store chunks in Parquet (chunks_metadata.parquet)
    ↓
EmbeddingService.generate_embeddings_batch()
    ├─ Load sentence-transformers model
    ├─ Batch process chunks
    ├─ Generate vectors (384-dim)
    └─ Store in Parquet (chunks_vectors.parquet)
    ↓
ContentRetriever.retrieve_relevant_chunks()
    ├─ Semantic search via DuckDB (cosine similarity)
    ├─ Rank by relevance (top-k)
    └─ Return chunk content with context
    ↓
ResponseGenerator.generate_response()
    ├─ Build system prompt (explain or teach mode)
    ├─ Call LLM with context chunks
    └─ Format for audio delivery
    ↓
TTSService.synthesize_speech()
    ├─ Load TTS model
    ├─ Generate audio with appropriate tone
    ├─ Post-process (normalize, compress)
    └─ Save to /data/audio/
```

---

## ⚙️ Critical Patterns & Conventions

### 1. **Service Initialization Pattern**

All services follow this consistent pattern:

```python
class MyService:
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.config = get_app_config()
        self._initialize_resources()

    def _initialize_resources(self):
        """Load models, connections, etc."""
        pass

    async def process(self, input):
        try:
            result = await self._do_work(input)
            self.logger.info(f"Processed {input.id}: success")
            return result
        except SpecificError as e:
            self.logger.error(f"Processing {input.id}: {e}")
            raise
        except Exception as e:
            self.logger.error(f"Unexpected error processing {input.id}: {e}")
            raise
```

### 2. **Environment & Configuration**

**Three-tier environment setup (DEV/QA/PROD):**

```python
# app/core/config.py
APP_ENVIRONMENT = os.getenv('APP_ENVIRONMENT', 'DEV').upper()

config = {
    "DEV": { "chunk_size": 1000, "embedding_batch": 16, ... },
    "QA":  { "chunk_size": 2000, "embedding_batch": 32, ... },
    "PROD": { "chunk_size": 3000, "embedding_batch": 64, ... },
}
```

### 3. **Chunking Strategy**

| Mode | Chunk Size | Overlap | Use Case |
|------|-----------|---------|----------|
| Semantic | Dynamic (by section) | 200 chars | Best for structured docs |
| Fixed | 1000-3000 chars | 200 chars | Fallback for unstructured text |
| Paragraph | Dynamic | 100 chars | Natural document flow |

**Rule:** Always respect semantic boundaries (sentences/paragraphs) before fixed sizes.

### 4. **Query Modes**

- **Explain Mode:** Simple, clear explanation for learners
- **Teach Mode:** Structured lesson with intro, concepts, examples, takeaways

### 5. **Storage Architecture**

```
DuckDB (document_index.duckdb)
├── documents table     - Document metadata
├── chunks table        - Chunk content + metadata
├── chunk_vectors table - Embedding vectors
└── response_cache table - Generated response cache

Parquet Files (/data/)
├── chunks/chunks_metadata.parquet  - Chunk content for bulk reads
├── chunks/chunks_vectors.parquet   - Dense vectors for search
└── embeddings/embeddings.parquet   - Document-level embeddings
```

### 6. **Async Patterns**

- All FastAPI endpoints use `async def`
- Services use `async` for I/O operations
- Use thread pool executor for CPU-bound work (embedding generation, TTS)
- DuckDB operations are synchronous (thread-safe via connection pool)

---

## 🛠️ Key Technologies

```python
# Web Framework
fastapi, uvicorn, pydantic

# Data Processing
duckdb, pyarrow, pandas

# Embeddings
sentence-transformers (all-MiniLM-L6-v2)

# TTS (Text-to-Speech)
TTS (Coqui TTS) or edge-tts

# LLM
ollama (local models) or langchain

# Document Parsing
PyPDF2, python-docx, markdown

# Audio Processing
pydub, soundfile
```

---

## 📋 Common Development Tasks

### Running the App

```bash
cd document-audio-generation-service
python -m uvicorn app.main:app --reload --port 8001
```

### Adding a New Service

1. Create service class in `app/services/{category}/`
2. Follow initialization pattern (constructor, logger, config)
3. Wire through endpoint in `app/endpoints/{category}_endpoints.py`
4. Register in `app/endpoints/router.py`

### Testing

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
```

---

## 🚨 Important Rules

### ✅ DO:
- **Always log with context:** `logger.error(f"Processing doc {doc_id}: {e}")`
- **Validate file types** before processing
- **Respect semantic boundaries** in chunking (paragraphs, sentences)
- **Use async/await** for all I/O operations
- **Cache embeddings** - don't regenerate for same content
- **Return structured responses** with success flag, data, timing info
- **Handle model loading gracefully** - download on first use if missing

### ❌ DON'T:
- **Don't hardcode file paths** - use config.py constants
- **Don't skip overlap in chunking** - context preservation is critical
- **Don't mix async/sync** in endpoint handlers
- **Don't block event loop** with CPU-bound work (use thread executor)
- **Don't ignore audio format** - normalize before serving
- **Don't lose error context** - always include what was being processed

---

## 📦 Key Files to Reference

```
app/
├── main.py                                    # FastAPI app, lifespan manager
├── core/
│   ├── config.py                              # Environment configuration
│   ├── constants.py                           # Global constants
│   └── logging_config.py                      # Logging setup
├── models/
│   ├── request_models.py                      # Pydantic input schemas
│   └── response_models.py                     # Pydantic output schemas
├── endpoints/
│   ├── router.py                              # Main router composition
│   ├── document_endpoints.py                  # Document CRUD
│   ├── chunk_endpoints.py                     # Chunk operations
│   ├── query_endpoints.py                     # Query/search
│   └── audio_endpoints.py                     # Audio generation
├── services/
│   ├── document_processing/
│   │   ├── document_ingestion_service.py      # Upload, parse, validate
│   │   ├── chunking_service.py                # Semantic chunking
│   │   └── metadata_manager.py                # Document metadata
│   ├── embedding_generation/
│   │   ├── embedding_service.py               # Generate embeddings
│   │   ├── vector_store_service.py            # Parquet storage
│   │   └── similarity_search.py               # DuckDB search
│   ├── content_analysis/
│   │   ├── prompt_processor.py                # Parse user queries
│   │   ├── content_retriever.py               # Retrieve relevant chunks
│   │   └── response_generator.py              # Generate teaching responses
│   └── audio_generation/
│       ├── tts_service.py                     # Text-to-speech
│       ├── audio_processor.py                 # Audio optimization
│       └── speech_style_manager.py            # Tone management
├── infrastructure/
│   ├── data_access/
│   │   ├── parquet_manager.py                 # Parquet I/O
│   │   ├── duckdb_manager.py                  # DuckDB operations
│   │   ├── document_store.py                  # Document persistence
│   │   └── staging_manager.py                 # Batch processing
│   ├── external_apis/
│   │   ├── llm_client.py                      # LLM API calls
│   │   └── model_manager.py                   # Model downloads
│   └── cache/
│       ├── embedding_cache.py                 # Embedding cache
│       └── response_cache.py                  # Response/audio cache
├── prompts/
│   └── system_prompts.py                      # System prompts (explain/teach)
└── utils/
    ├── file_utils.py                          # File operations
    ├── text_utils.py                          # Text cleaning
    ├── error_handlers.py                      # Custom exceptions
    └── validators.py                          # Input validation
```

---

## 💡 Quick Reference

| Pattern | Meaning |
|---------|---------|
| `mode='explain'` | Simple clear explanation |
| `mode='teach'` | Structured lesson format |
| `chunk_size=1000` in DEV | Smaller chunks for testing |
| `embedding_batch_size=16` | Batch size for vector generation |
| `top_k=5` in retrieval | Number of relevant chunks to return |
| `@lru_cache` on config | Load config once, reuse |

---

**Python:** 3.9+ | **FastAPI:** 0.104+ | **DuckDB:** 0.9+
