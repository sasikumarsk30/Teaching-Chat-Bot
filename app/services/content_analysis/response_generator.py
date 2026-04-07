"""
Response Generator

Generates educational responses (explain / teach mode)
by combining retrieved content with LLM-powered synthesis.
"""

import logging
import time
from typing import Optional

from app.core.config import get_settings
from app.core.constants import QUERY_MODE_EXPLAIN, QUERY_MODE_TEACH
from app.infrastructure.external_apis.llm_client import get_llm_client
from app.infrastructure.cache.response_cache import get_response_cache
from app.services.content_analysis.content_retriever import get_content_retriever
from app.services.content_analysis.prompt_processor import get_prompt_processor
from app.prompts.system_prompts import get_system_prompt, build_user_prompt

logger = logging.getLogger(__name__)


class ResponseGenerator:
    """Generates educational text responses using LLM + retrieved context."""

    def __init__(self):
        self.settings = get_settings()
        self.llm = get_llm_client()
        self.cache = get_response_cache()
        self.retriever = get_content_retriever()
        self.prompt_processor = get_prompt_processor()
        logger.info("ResponseGenerator initialized")

    async def generate(
        self,
        question: str,
        mode: str = QUERY_MODE_EXPLAIN,
        document_id: Optional[str] = None,
        top_k: Optional[int] = None,
    ) -> dict:
        """
        Full pipeline: parse query → retrieve context → generate response.

        Args:
            question: User's question or topic.
            mode: 'explain' or 'teach'.
            document_id: Scope to a specific document.
            top_k: Number of chunks to retrieve.

        Returns:
            Dict with response_id, response_text, source_chunks, timing.
        """
        start_time = time.time()

        # 1. Check cache
        cached = self.cache.get(question, mode)
        if cached:
            return {
                "response_id": cached["id"],
                "response_text": cached["response_text"],
                "source_chunks": [],
                "mode": mode,
                "from_cache": True,
                "processing_time_ms": (time.time() - start_time) * 1000,
            }

        # 2. Parse query
        parsed = self.prompt_processor.parse_query(question, mode)

        # 3. Retrieve relevant chunks
        chunks = await self.retriever.retrieve(
            query=question,
            top_k=top_k,
            document_id=document_id,
        )

        if not chunks:
            return self._no_content_response(question, mode, start_time)

        # 4. Build prompt with context
        context = self.retriever.build_context_string(chunks)
        system_prompt = get_system_prompt(mode)
        user_prompt = build_user_prompt(question, context, mode)

        # 5. Generate LLM response
        logger.info(
            f"Generating {mode} response | question='{question[:60]}...' "
            f"chunks={len(chunks)}"
        )

        try:
            response_text = await self.llm.generate(
                prompt=user_prompt,
                system_prompt=system_prompt,
            )
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise

        # 6. Cache the response
        source_chunk_ids = [c.get("chunk_id", "") for c in chunks]
        response_id = self.cache.put(
            query=question,
            mode=mode,
            response_text=response_text,
            source_chunk_ids=source_chunk_ids,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        logger.info(
            f"Response generated | id={response_id} mode={mode} "
            f"chars={len(response_text)} time={elapsed_ms:.0f}ms"
        )

        return {
            "response_id": response_id,
            "response_text": response_text,
            "source_chunks": self._format_source_chunks(chunks),
            "mode": mode,
            "from_cache": False,
            "processing_time_ms": elapsed_ms,
        }

    async def generate_predefined(
        self,
        topic: str,
        subtopics: list[str],
        mode: str = QUERY_MODE_TEACH,
        document_ids: Optional[list[str]] = None,
        max_chunks_per_subtopic: int = 3,
    ) -> dict:
        """
        Generate a structured response for predefined topic + subtopics.

        Searches for each subtopic, aggregates content, and generates
        a comprehensive teaching response.
        """
        start_time = time.time()

        # Retrieve chunks for all subtopics
        chunks = await self.retriever.retrieve_for_subtopics(
            subtopics=subtopics,
            max_per_subtopic=max_chunks_per_subtopic,
            document_ids=document_ids,
        )

        if not chunks:
            return self._no_content_response(topic, mode, start_time)

        # Build a comprehensive prompt
        context = self.retriever.build_context_string(chunks)

        subtopics_text = ", ".join(subtopics)
        full_question = (
            f"Topic: {topic}\n"
            f"Subtopics to cover: {subtopics_text}\n\n"
            f"Please create a comprehensive {mode} response covering "
            f"all the listed subtopics based on the provided content."
        )

        from app.prompts.system_prompts import PREDEFINED_CONTENT_SYSTEM_PROMPT

        user_prompt = build_user_prompt(full_question, context, mode)

        response_text = await self.llm.generate(
            prompt=user_prompt,
            system_prompt=PREDEFINED_CONTENT_SYSTEM_PROMPT,
        )

        source_chunk_ids = [c.get("chunk_id", "") for c in chunks]
        response_id = self.cache.put(
            query=f"predefined:{topic}",
            mode=mode,
            response_text=response_text,
            source_chunk_ids=source_chunk_ids,
        )

        elapsed_ms = (time.time() - start_time) * 1000

        covered_subtopics = list({c.get("subtopic", "") for c in chunks if c.get("subtopic")})

        return {
            "response_id": response_id,
            "topic": topic,
            "subtopics_covered": covered_subtopics,
            "response_text": response_text,
            "source_chunks": self._format_source_chunks(chunks),
            "mode": mode,
            "from_cache": False,
            "processing_time_ms": elapsed_ms,
        }

    def _no_content_response(self, question: str, mode: str, start_time: float) -> dict:
        """Return a fallback when no relevant content is found."""
        elapsed_ms = (time.time() - start_time) * 1000
        return {
            "response_id": "",
            "response_text": (
                "I couldn't find relevant content in the uploaded documents "
                "to address your question. Please upload documents related "
                "to this topic and try again."
            ),
            "source_chunks": [],
            "mode": mode,
            "from_cache": False,
            "processing_time_ms": elapsed_ms,
        }

    @staticmethod
    def _format_source_chunks(chunks: list[dict]) -> list[dict]:
        """Format chunk data for the response model."""
        formatted = []
        for c in chunks:
            formatted.append({
                "chunk_id": c.get("chunk_id", ""),
                "document_id": c.get("document_id", ""),
                "document_title": c.get("document_title", "Unknown"),
                "sequence": c.get("sequence", 0),
                "similarity_score": c.get("similarity_score", 0.0),
                "snippet": c.get("content", "")[:200],
            })
        return formatted


# ── Module-level factory ─────────────────────────────────────

_response_generator: Optional[ResponseGenerator] = None


def get_response_generator() -> ResponseGenerator:
    global _response_generator
    if _response_generator is None:
        _response_generator = ResponseGenerator()
    return _response_generator
