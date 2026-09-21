"""JEV (TypeSafe systemone) client.

Sends a `state` payload plus a dict of typed questions (choice / score / noul)
to POST https://api.typesafe.ai/v1/systemone and returns typed answers with
probabilities + confidence. Questions are always Python-templated upstream;
JEV judges over the evidence it is handed — it never enumerates candidates.

Uses raw httpx rather than the SDK to keep dependencies minimal and mocking
trivial. Set TYPESAFE_API_KEY, or pass mock=True for offline dev/tests.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from typing import Any

import httpx

log = logging.getLogger(__name__)

API_URL = "https://api.typesafe.ai/v1/systemone"
DEFAULT_MODEL = "jev-latest"


class JevError(RuntimeError):
    pass


class JevClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        mock: bool = False,
        cache_dir: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.mock = mock
        self.timeout = timeout
        self.usage_log: list[dict[str, Any]] = []
        self._api_key = api_key or os.environ.get("TYPESAFE_API_KEY", "")
        self._cache = None
        if cache_dir:
            import diskcache

            self._cache = diskcache.Cache(os.path.join(cache_dir, "jev"))
        if not (self.mock or self._api_key):
            raise JevError(
                "TYPESAFE_API_KEY is not set. Export it, put it in .env, "
                "or run with --mock-jev for offline development."
            )

    def ask(
        self,
        state: dict[str, Any],
        questions: dict[str, dict[str, Any]],
        stage: str = "",
    ) -> dict[str, dict[str, Any]]:
        """Ask a batch of questions against `state`. Returns raw answer dicts."""
        body = {"state": state, "model": self.model, "questions": questions}
        if self.mock:
            answers = self._mock_answers(questions)
            usage = {"input_tokens": 0, "output_tokens": 0}
        else:
            answers, usage = self._call(body)
        self.usage_log.append(
            {
                "stage": stage,
                "n_questions": len(questions),
                "input_tokens": usage.get("input_tokens", 0),
                "output_tokens": usage.get("output_tokens", 0),
            }
        )
        log.info(
            "jev stage=%s questions=%d in=%s out=%s",
            stage,
            len(questions),
            usage.get("input_tokens"),
            usage.get("output_tokens"),
        )
        return answers

    def total_usage(self) -> dict[str, int]:
        return {
            "input_tokens": sum(u["input_tokens"] for u in self.usage_log),
            "output_tokens": sum(u["output_tokens"] for u in self.usage_log),
            "calls": len(self.usage_log),
        }

    # -- internals ---------------------------------------------------------

    def _call(self, body: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
        key = hashlib.sha256(
            json.dumps(body, sort_keys=True, default=str).encode()
        ).hexdigest()
        if self._cache is not None and key in self._cache:
            cached = self._cache[key]
            return cached["answers"], cached["usage"]

        resp = httpx.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            json=body,
            timeout=self.timeout,
        )
        if resp.status_code != 200:
            raise JevError(f"JEV API error {resp.status_code}: {resp.text[:500]}")
        data = resp.json()
        answers, usage = data.get("answers", {}), data.get("usage", {})
        if self._cache is not None:
            self._cache[key] = {"answers": answers, "usage": usage}
        return answers, usage

    @staticmethod
    def _mock_answers(
        questions: dict[str, dict[str, Any]]
    ) -> dict[str, dict[str, Any]]:
        """Deterministic offline answers: first option / midpoint / 0.5."""
        out: dict[str, dict[str, Any]] = {}
        for qid, q in questions.items():
            qtype = q.get("type")
            if qtype == "choice":
                options = list(q.get("criteria", {}).keys())
                pick = options[0] if options else "other"
                out[qid] = {
                    "type": "choice",
                    "choice": pick,
                    "confidence": 0.5,
                    "probabilities": {o: (1.0 if o == pick else 0.0) for o in options},
                }
            elif qtype == "score":
                levels = q.get("criteria", [])
                mid = (len(levels) - 1) / 2 if levels else 0.0
                out[qid] = {
                    "type": "score",
                    "score": mid,
                    "confidence": 0.5,
                    "legend": {str(i): str(l) for i, l in enumerate(levels)},
                    "probabilities": {str(i): 0.0 for i in range(len(levels))},
                }
            elif qtype == "noul":
                out[qid] = {"type": "noul", "noul": 0.5, "confidence": 0.5}
            else:
                raise JevError(f"unknown question type for {qid!r}: {qtype!r}")
        return out
