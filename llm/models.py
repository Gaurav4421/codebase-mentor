"""
Thin LLM client abstraction. Every other module (code_intelligence, memory,
llm.prompts consumers) depends on the `LLMClient` protocol, not on
`google-genai` directly -- this is what makes it possible to unit test
anything downstream with a fake client, and to swap providers later without
touching call sites.
"""
import logging
import os
from typing import Protocol

logger = logging.getLogger(__name__)


class LLMClient(Protocol):
    def generate(self, prompt: str) -> str:
        """Send a single prompt, return the model's text response."""
        ...


DEFAULT_MODEL = "gemini-2.5-flash"


class GeminiClient:
    """Wraps google-genai's client behind the LLMClient protocol."""

    def __init__(self, api_key: str = None, model: str = None):
        from google import genai

        api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GEMINI_API_KEY not set. Copy .env.example to .env and fill it in, "
                "export it in your shell, or pass api_key= explicitly."
            )

        self._client = genai.Client(api_key=api_key)
        self.model = model or os.environ.get("GEMINI_MODEL", DEFAULT_MODEL)

    def generate(self, prompt: str) -> str:
        try:
            response = self._client.models.generate_content(model=self.model, contents=prompt)
            return response.text or ""
        except Exception:
            logger.exception("Gemini generate_content failed")
            raise


class FakeLLMClient:
    """Deterministic stand-in for tests / evaluation runs that shouldn't hit
    the network. Returns a fixed or callable-derived response."""

    def __init__(self, response: str = "", responder=None):
        self.response = response
        self.responder = responder
        self.calls = []

    def generate(self, prompt: str) -> str:
        self.calls.append(prompt)
        if self.responder:
            return self.responder(prompt)
        return self.response
