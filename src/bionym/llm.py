"""Optional LLM client: candidate enumeration for JEV to score.

Used ONLY to propose claims from free text (dataset descriptions,
publication titles/abstracts) — the LLM enumerates, JEV judges. It never
decides anything on its own; every proposal becomes a JEV question.

Any OpenAI-compatible chat-completions endpoint works: OpenRouter,
OpenAI, Gemini's compatibility endpoint, local vLLM/Ollama. Configure via
env: ``LLM_API_KEY``, ``LLM_BASE_URL`` (default OpenRouter),
``LLM_MODEL``. Disabled when key or model is unset — the resolver runs
fine without it. ``mock=True`` returns a canned proposal for dev/tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"


class LlmError(RuntimeError):
    pass


class LlmClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str | None = None,
        mock: bool = False,
        cache_dir: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.base_url = (
            base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self.model = model or os.environ.get("LLM_MODEL") or ""
        self.mock = mock
        self.timeout = timeout
        self.usage_log: list[dict[str, Any]] = []
        self._api_key = api_key or os.environ.get("LLM_API_KEY") or ""
        self._cache = None
        if cache_dir:
            import diskcache

            self._cache = diskcache.Cache(os.path.join(cache_dir, "llm"))

    @property
    def enabled(self) -> bool:
        """Proposals only run with a configured model — never required."""
        return self.mock or bool(self._api_key and self.model)

    def complete(self, system: str, user: str) -> str:
        """One chat completion -> raw content string (expected JSON)."""
        if self.mock:
            return '{"proposals": [{"claim": "mock proposal", "quote": "mock"}]}'
        body = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0,
        }
        key = hashlib.sha256(
            json.dumps(body, sort_keys=True).encode()
        ).hexdigest()
        if self._cache is not None and key in self._cache:
            return self._cache[key]

        resp = httpx.post(
            f"{self.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise LlmError(f"LLM API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        content = (data.get("choices") or [{}])[0].get("message", {}).get(
            "content", ""
        )
        usage = data.get("usage") or {}
        self.usage_log.append(
            {
                "input_tokens": usage.get("prompt_tokens", 0),
                "output_tokens": usage.get("completion_tokens", 0),
            }
        )
        # Don't cache empty responses: a transient null content (reasoning
        # models occasionally return one) would otherwise poison the cache
        # and suppress retries forever.
        if self._cache is not None and content:
            self._cache[key] = content
        return content
