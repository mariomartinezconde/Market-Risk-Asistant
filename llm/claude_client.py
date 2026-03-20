"""
Claude API Client
=================
Wraps the Anthropic Python SDK with:
  - Configurable model / token limits from settings.
  - Automatic retry with exponential backoff (tenacity).
  - Structured logging of every request/response.
"""
from __future__ import annotations

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from app.logger import get_logger
from config import get_settings

logger = get_logger(__name__)

_client: anthropic.Anthropic | None = None


def get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        settings = get_settings()
        _client = anthropic.Anthropic(api_key=settings.ANTHROPIC_API_KEY)
    return _client


@retry(
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError)),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    stop=stop_after_attempt(3),
    reraise=True,
)
def call_claude(
    system_prompt: str,
    user_message: str,
) -> str:
    """
    Send a message to Claude and return the text response.

    Temperature is hard-coded to 0.0 in settings (determinism for audit).
    """
    settings = get_settings()
    client = get_client()

    logger.info(
        "claude.request",
        model=settings.CLAUDE_MODEL,
        user_msg_len=len(user_message),
    )

    response = client.messages.create(
        model=settings.CLAUDE_MODEL,
        max_tokens=settings.CLAUDE_MAX_TOKENS,
        temperature=settings.CLAUDE_TEMPERATURE,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )

    answer = response.content[0].text
    logger.info(
        "claude.response",
        input_tokens=response.usage.input_tokens,
        output_tokens=response.usage.output_tokens,
        answer_len=len(answer),
    )
    return answer
