"""
Test fixtures and shared configuration for pytest.

Provides fixtures for:
- Sample texts and documents
- Temporary directories and DuckDB instances
- Mocked services (LLM, TTS, embedding model)
- FastAPI test client
"""

import os
import sys
import uuid
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock

import pytest
import numpy as np

# Ensure the app module is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ── Environment Setup ────────────────────────────────────────
# Force DEV environment for all tests
os.environ["APP_ENVIRONMENT"] = "DEV"
os.environ["DEBUG"] = "True"
os.environ["LOG_LEVEL"] = "WARNING"


# ── Text Fixtures ────────────────────────────────────────────

@pytest.fixture
def sample_text():
    """Sample text for chunking tests (5 paragraphs)."""
    return (
        "Machine learning is a subset of artificial intelligence. "
        "It allows computers to learn from data without being explicitly programmed.\n\n"
        "There are three main types of machine learning: supervised learning, "
        "unsupervised learning, and reinforcement learning.\n\n"
        "Supervised learning uses labeled datasets to train algorithms. "
        "The model learns to map inputs to known outputs.\n\n"
        "Unsupervised learning finds hidden patterns in unlabeled data. "
        "Clustering and dimensionality reduction are common techniques.\n\n"
        "Reinforcement learning teaches agents through rewards and penalties. "
        "It is widely used in robotics and game playing."
    )


@pytest.fixture
def long_sample_text():
    """Longer sample text for testing chunk splitting across boundaries."""
    paragraphs = []
    topics = [
        ("Neural Networks", "Neural networks are computing systems inspired by biological neural networks. "
         "They consist of layers of interconnected nodes that process information. "
         "Each node applies a mathematical function to its inputs and passes the result forward. "
         "Deep learning uses neural networks with many hidden layers to learn complex patterns."),
        ("Convolutional Networks", "Convolutional Neural Networks (CNNs) are particularly effective for image processing. "
         "They use convolutional layers that scan input data with small filters. "
         "Pooling layers reduce the spatial dimensions while preserving important features. "
         "CNNs have revolutionized computer vision tasks like object detection and image classification."),
        ("Recurrent Networks", "Recurrent Neural Networks (RNNs) are designed for sequential data processing. "
         "They maintain a hidden state that captures information from previous time steps. "
         "Long Short-Term Memory (LSTM) networks address the vanishing gradient problem. "
         "Transformers have largely replaced RNNs for natural language processing tasks."),
        ("Training Process", "Training a neural network involves feeding it data and adjusting its weights. "
         "The loss function measures how far the model's predictions are from actual values. "
         "Backpropagation computes gradients used to update the weights. "
         "Optimization algorithms like Adam and SGD control how weights change during training."),
        ("Regularization", "Regularization techniques prevent overfitting during training. "
         "Dropout randomly disables neurons during training to reduce co-adaptation. "
         "L1 and L2 regularization add penalty terms to the loss function. "
         "Early stopping monitors validation performance and halts training when it degrades."),
    ]
    for title, content in topics:
        paragraphs.append(f"{title}\n\n{content}")
    return "\n\n".join(paragraphs)


@pytest.fixture
def sample_query():
    """Sample user query."""
    return "Explain the different types of machine learning"


@pytest.fixture
def sample_chunks(sample_text):
    """Pre-built chunk dicts for testing (simulates ChunkingService output)."""
    paragraphs = sample_text.split("\n\n")
    chunks = []
    pos = 0
    for i, para in enumerate(paragraphs):
        start = sample_text.find(para, pos)
        chunks.append({
            "id": str(uuid.uuid4()),
            "document_id": "test-doc-001",
            "sequence": i,
            "content": para,
            "chunk_size": len(para),
            "start_char": start,
            "end_char": start + len(para),
            "created_at": datetime.utcnow(),
            "metadata": {"strategy": "semantic", "overlap": 200},
        })
        pos = start + len(para)
    return chunks


@pytest.fixture
def sample_embeddings():
    """Fake 384-dim embeddings for 5 chunks."""
    np.random.seed(42)
    return [np.random.randn(384).astype(np.float32) for _ in range(5)]


@pytest.fixture
def sample_query_embedding():
    """A single fake query embedding."""
    np.random.seed(99)
    return np.random.randn(384).astype(np.float32)


# ── Document Fixtures ────────────────────────────────────────

@pytest.fixture
def sample_txt_content():
    """Raw bytes for a .txt upload."""
    text = (
        "Introduction to Data Science\n\n"
        "Data science combines statistics, programming, and domain knowledge "
        "to extract insights from data. It involves collecting, cleaning, "
        "analyzing, and visualizing data to support decision-making.\n\n"
        "Key tools include Python, R, SQL, and various machine learning frameworks."
    )
    return text.encode("utf-8")


@pytest.fixture
def sample_md_content():
    """Raw bytes for a .md upload."""
    text = (
        "# Python Programming\n\n"
        "Python is a versatile programming language.\n\n"
        "## Features\n\n"
        "Python offers dynamic typing, automatic memory management, "
        "and a comprehensive standard library.\n\n"
        "## Applications\n\n"
        "Python is used in web development, data science, artificial intelligence, "
        "and automation scripting."
    )
    return text.encode("utf-8")


@pytest.fixture
def sample_document_metadata():
    """A document metadata dict as returned by DocumentStore."""
    return {
        "id": "test-doc-001",
        "filename": "test_document.txt",
        "title": "Test Document",
        "description": "A test document for unit testing",
        "file_type": ".txt",
        "file_size_bytes": 512,
        "original_path": "/tmp/test_document.txt",
        "tags": ["test", "sample"],
        "upload_date": datetime.utcnow(),
        "total_chunks": 5,
        "embeddings_generated": True,
    }


# ── Temporary Directory Fixtures ─────────────────────────────

@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory structure."""
    dirs = {
        "documents": tmp_path / "documents",
        "chunks": tmp_path / "chunks",
        "embeddings": tmp_path / "embeddings",
        "audio": tmp_path / "audio",
        "duckdb": tmp_path / "duckdb",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs


@pytest.fixture
def tmp_duckdb_path(tmp_data_dir):
    """Path for a test-only DuckDB instance."""
    return str(tmp_data_dir["duckdb"] / "test.duckdb")


# ── DuckDB Test Instance ─────────────────────────────────────

@pytest.fixture
def test_duckdb(tmp_duckdb_path):
    """
    Provide a fresh DuckDB connection with schema initialized.
    Does NOT use the singleton so tests are isolated.
    """
    import duckdb
    conn = duckdb.connect(tmp_duckdb_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id VARCHAR PRIMARY KEY,
            filename VARCHAR NOT NULL,
            title VARCHAR NOT NULL,
            description VARCHAR,
            file_type VARCHAR NOT NULL,
            file_size_bytes INTEGER,
            original_path VARCHAR,
            total_chunks INTEGER DEFAULT 0,
            embeddings_generated BOOLEAN DEFAULT FALSE,
            tags VARCHAR[],
            upload_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id VARCHAR PRIMARY KEY,
            document_id VARCHAR NOT NULL,
            sequence INTEGER NOT NULL,
            content TEXT NOT NULL,
            chunk_size INTEGER,
            start_char INTEGER,
            end_char INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            metadata JSON
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunk_vectors (
            id VARCHAR PRIMARY KEY,
            chunk_id VARCHAR NOT NULL,
            embedding FLOAT[],
            vector_dim INTEGER,
            model_name VARCHAR,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS response_cache (
            id VARCHAR PRIMARY KEY,
            query TEXT NOT NULL,
            mode VARCHAR NOT NULL,
            response_text TEXT NOT NULL,
            source_chunk_ids VARCHAR[],
            audio_path VARCHAR,
            audio_duration_seconds FLOAT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            expires_at TIMESTAMP
        )
    """)

    yield conn
    conn.close()


# ── Mock Fixtures ────────────────────────────────────────────

@pytest.fixture
def mock_embedding_model():
    """Mock sentence-transformers model that returns deterministic embeddings."""
    model = MagicMock()

    def fake_encode(texts, show_progress_bar=False, normalize_embeddings=True):
        np.random.seed(42)
        return [np.random.randn(384).astype(np.float32) for _ in texts]

    model.encode = fake_encode
    return model


@pytest.fixture
def mock_llm_client():
    """Mock LLM client that returns canned responses."""
    client = AsyncMock()
    client.generate = AsyncMock(return_value=(
        "Machine learning is a fascinating field of artificial intelligence "
        "that enables computers to learn from data. There are three main types: "
        "supervised learning, unsupervised learning, and reinforcement learning. "
        "Each approach has distinct strengths and use cases."
    ))
    client.health_check = AsyncMock(return_value={
        "status": "healthy", "provider": "mock"
    })
    return client


@pytest.fixture
def mock_tts_service(tmp_data_dir):
    """Mock TTS service that creates a dummy audio file."""
    async def fake_synthesize(text, mode="explain", voice=None, rate=None, output_format="mp3"):
        audio_id = str(uuid.uuid4())
        filename = f"{audio_id}.{output_format}"
        file_path = tmp_data_dir["audio"] / filename
        file_path.write_bytes(b"\x00" * 1024)
        return {
            "audio_id": audio_id,
            "file_path": str(file_path),
            "filename": filename,
            "format": output_format,
            "duration_seconds": len(text.split()) / 150 * 60,
            "file_size_bytes": 1024,
            "voice": voice or "en-US-AriaNeural",
            "mode": mode,
        }

    service = AsyncMock()
    service.synthesize = AsyncMock(side_effect=fake_synthesize)
    return service
