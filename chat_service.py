"""
LLM provider abstraction layer.
Supports Anthropic and OpenAI. Switch by setting LLM_PROVIDER env var.
If the configured provider's API key is missing, get_llm_provider() raises
a clear ServiceUnavailableError — it does NOT crash the whole app at startup.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from app.core.logging import get_logger

logger = get_logger(__name__)


class LLMProvider(ABC):
    @abstractmethod
    def complete(self, system_prompt: str, user_message: str) -> str:
        ...


class AnthropicProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float):
        import anthropic
        self._client = anthropic.Anthropic(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, system_prompt: str, user_message: str) -> str:
        from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
        import anthropic

        @retry(
            retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
            wait=wait_exponential(min=2, max=30),
            stop=stop_after_attempt(3),
            reraise=True,
        )
        def _call():
            resp = self._client.messages.create(
                model=self._model,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=system_prompt,
                messages=[{"role": "user", "content": user_message}],
            )
            return resp.content[0].text

        return _call()


class OpenAIProvider(LLMProvider):
    def __init__(self, api_key: str, model: str, max_tokens: int, temperature: float):
        from openai import OpenAI
        self._client = OpenAI(api_key=api_key)
        self._model = model
        self._max_tokens = max_tokens
        self._temperature = temperature

    def complete(self, system_prompt: str, user_message: str) -> str:
        resp = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            temperature=self._temperature,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        return resp.choices[0].message.content


class LLMNotConfiguredError(Exception):
    pass


_provider: LLMProvider | None = None


def get_llm_provider() -> LLMProvider:
    global _provider
    if _provider is not None:
        return _provider

    from app.core.config import get_settings
    settings = get_settings()

    if not settings.llm_configured:
        raise LLMNotConfiguredError(
            f"LLM provider '{settings.LLM_PROVIDER}' is not configured. "
            f"Set {'ANTHROPIC_API_KEY' if settings.LLM_PROVIDER == 'anthropic' else 'OPENAI_API_KEY'} "
            f"in environment variables."
        )

    if settings.LLM_PROVIDER == "anthropic":
        _provider = AnthropicProvider(
            api_key=settings.ANTHROPIC_API_KEY,
            model=settings.effective_model_name,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )
    else:
        _provider = OpenAIProvider(
            api_key=settings.OPENAI_API_KEY,
            model=settings.effective_model_name,
            max_tokens=settings.LLM_MAX_TOKENS,
            temperature=settings.LLM_TEMPERATURE,
        )

    logger.info("llm.provider_ready", provider=settings.LLM_PROVIDER, model=settings.effective_model_name)
    return _provider
