"""
Prompt Processor

Parses user queries to extract intent, topic, and constraints.
Optionally uses LLM for advanced query understanding.
"""

import logging
import json
from typing import Optional

from app.core.constants import QUERY_MODE_EXPLAIN, QUERY_MODE_TEACH
from app.utils.validators import validate_query_mode

logger = logging.getLogger(__name__)


class PromptProcessor:
    """Processes and analyzes user prompts to extract structured information."""

    def __init__(self):
        logger.info("PromptProcessor initialized")

    def parse_query(self, query: str, mode: str = QUERY_MODE_EXPLAIN) -> dict:
        """
        Parse a user query into structured components.

        Args:
            query: Raw user question or topic.
            mode: Desired response mode (explain / teach).

        Returns:
            Dict with topic, mode, search_terms, and constraints.
        """
        mode = validate_query_mode(mode)

        # Basic extraction via heuristics
        topic = self._extract_topic(query)
        search_terms = self._generate_search_terms(query)

        parsed = {
            "original_query": query,
            "topic": topic,
            "mode": mode,
            "search_terms": search_terms,
            "constraints": self._detect_constraints(query),
        }

        logger.info(
            f"Query parsed | topic='{topic}' mode={mode} "
            f"terms={search_terms}"
        )
        return parsed

    def _extract_topic(self, query: str) -> str:
        """
        Extract the main topic from the query using heuristic rules.

        Strips common question prefixes to identify the core subject.
        """
        # Remove common question prefixes
        prefixes = [
            "can you explain",
            "please explain",
            "explain to me",
            "teach me about",
            "teach me",
            "tell me about",
            "what is",
            "what are",
            "how does",
            "how do",
            "how to",
            "why is",
            "why do",
            "describe",
            "explain",
            "help me understand",
            "i want to learn about",
            "i want to understand",
        ]

        lower = query.lower().strip()
        for prefix in prefixes:
            if lower.startswith(prefix):
                topic = query[len(prefix):].strip().strip("?").strip()
                return topic if topic else query

        return query.strip().strip("?").strip()

    def _generate_search_terms(self, query: str) -> list[str]:
        """
        Generate search terms from the query.

        Splits query into meaningful words, removing stop words.
        """
        stop_words = {
            "a", "an", "the", "is", "are", "was", "were", "be", "been",
            "being", "have", "has", "had", "do", "does", "did", "will",
            "would", "could", "should", "may", "might", "can", "shall",
            "to", "of", "in", "for", "on", "with", "at", "by", "from",
            "as", "into", "through", "during", "before", "after",
            "about", "between", "out", "above", "below", "up", "down",
            "and", "but", "or", "nor", "not", "so", "yet", "both",
            "each", "few", "more", "most", "other", "some", "such",
            "than", "too", "very", "just", "because", "if", "when",
            "what", "which", "who", "whom", "this", "that", "these",
            "those", "i", "me", "my", "we", "our", "you", "your",
            "he", "him", "she", "her", "it", "they", "them",
            "explain", "teach", "tell", "describe", "help", "understand",
            "learn", "please", "want",
        }

        words = query.lower().split()
        terms = [
            w.strip("?.,!;:'\"()[]{}") for w in words
            if w.strip("?.,!;:'\"()[]{}") not in stop_words
            and len(w.strip("?.,!;:'\"()[]{}")) > 2
        ]

        # Also include the full query as a search term for semantic search
        return list(dict.fromkeys([query] + terms))  # preserve order, dedupe

    def _detect_constraints(self, query: str) -> dict:
        """Detect any constraints in the query (e.g., 'briefly', 'in detail')."""
        constraints = {}
        lower = query.lower()

        if any(w in lower for w in ["briefly", "short", "quick", "summary"]):
            constraints["detail_level"] = "brief"
        elif any(w in lower for w in ["detail", "thorough", "comprehensive", "deep"]):
            constraints["detail_level"] = "detailed"
        else:
            constraints["detail_level"] = "standard"

        return constraints

    async def parse_query_with_llm(
        self, query: str, llm_client
    ) -> dict:
        """
        Use an LLM for advanced query understanding.

        Falls back to heuristic parsing if LLM is unavailable.
        """
        from app.prompts.system_prompts import QUERY_UNDERSTANDING_PROMPT

        prompt = QUERY_UNDERSTANDING_PROMPT.format(query=query)

        try:
            response = await llm_client.generate(prompt)
            parsed = json.loads(response)
            parsed["original_query"] = query
            logger.info(f"LLM query parsing succeeded | topic='{parsed.get('topic')}'")
            return parsed
        except (json.JSONDecodeError, Exception) as e:
            logger.warning(f"LLM query parsing failed, using heuristics: {e}")
            return self.parse_query(query)


# ── Module-level factory ─────────────────────────────────────

_prompt_processor: Optional[PromptProcessor] = None


def get_prompt_processor() -> PromptProcessor:
    global _prompt_processor
    if _prompt_processor is None:
        _prompt_processor = PromptProcessor()
    return _prompt_processor
