"""
LLM Client

Abstraction layer for interacting with Large Language Models.
Supports local models via Ollama and optionally Azure OpenAI.
"""

import logging
import httpx
from typing import Optional

from app.core.config import get_settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Unified client for LLM response generation."""

    def __init__(self):
        self.settings = get_settings()
        self.provider = self.settings.llm_provider.lower()
        self.model_name = self.settings.llm_model_name
        self.base_url = self.settings.llm_base_url
        self.temperature = self.settings.llm_temperature
        self.max_tokens = self.settings.llm_max_tokens
        logger.info(
            f"LLMClient initialized | provider={self.provider} "
            f"model={self.model_name}"
        )

    async def generate(
        self,
        prompt: str,
        system_prompt: str = "",
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate a text response from the LLM.

        Args:
            prompt: The user prompt / question with context.
            system_prompt: System-level instruction.
            temperature: Override default temperature.
            max_tokens: Override default max tokens.

        Returns:
            Generated text response.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "ollama":
            return await self._generate_ollama(prompt, system_prompt, temp, tokens)
        elif self.provider == "azure_openai":
            return await self._generate_azure_openai(
                prompt, system_prompt, temp, tokens
            )
        else:
            raise ValueError(f"Unsupported LLM provider: {self.provider}")

    # ── Ollama ───────────────────────────────────────────────

    async def _generate_ollama(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate response using a local Ollama model."""
        url = f"{self.base_url}/api/chat"

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data.get("message", {}).get("content", "")
                logger.info(
                    f"Ollama response generated | model={self.model_name} "
                    f"chars={len(content)}"
                )
                return content
        except httpx.HTTPStatusError as e:
            logger.error(f"Ollama HTTP error: {e.response.status_code} {e}")
            raise
        except httpx.ConnectError:
            logger.error(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Is Ollama running?"
            )
            raise RuntimeError(
                f"LLM service unavailable. Ensure Ollama is running at "
                f"{self.base_url}"
            )
        except Exception as e:
            logger.error(f"Ollama generation failed: {e}")
            raise

    # ── Azure OpenAI ─────────────────────────────────────────

    async def _generate_azure_openai(
        self,
        prompt: str,
        system_prompt: str,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate response using Azure OpenAI."""
        settings = self.settings

        if not settings.uses_azure_openai:
            raise ValueError(
                "Azure OpenAI is not configured. Set AZURE_OPENAI_ENDPOINT "
                "and AZURE_OPENAI_API_KEY."
            )

        url = (
            f"{settings.azure_openai_endpoint}/openai/deployments/"
            f"{settings.azure_openai_deployment}/chat/completions"
            f"?api-version={settings.azure_openai_api_version}"
        )

        headers = {
            "Content-Type": "application/json",
            "api-key": settings.azure_openai_api_key,
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(url, headers=headers, json=payload)
                response.raise_for_status()
                data = response.json()
                content = data["choices"][0]["message"]["content"]
                logger.info(
                    f"Azure OpenAI response generated | "
                    f"deployment={settings.azure_openai_deployment} "
                    f"chars={len(content)}"
                )
                return content
        except Exception as e:
            logger.error(f"Azure OpenAI generation failed: {e}")
            raise

    # ── Health Check ─────────────────────────────────────────

    async def health_check(self) -> dict:
        """Check if the LLM service is reachable."""
        if self.provider == "ollama":
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    resp = await client.get(f"{self.base_url}/api/tags")
                    resp.raise_for_status()
                    models = [
                        m["name"] for m in resp.json().get("models", [])
                    ]
                    return {
                        "status": "healthy",
                        "provider": "ollama",
                        "available_models": models,
                    }
            except Exception as e:
                return {"status": "unhealthy", "provider": "ollama", "error": str(e)}
        else:
            return {"status": "configured", "provider": self.provider}


# ── Module-level factory ─────────────────────────────────────

_llm_client: Optional[LLMClient] = None


def get_llm_client() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client
