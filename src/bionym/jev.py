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
import time
from typing import Any

import httpx

log = logging.getLogger(__name__)

API_URL = "https://api.typesafe.ai/v1/systemone"

# JEV_BASE_URL may be an origin ("http://127.0.0.1:8009") or a full
# endpoint ("https://api.laya.studio/v1/systemone") — both accepted.
def _endpoint(base: str) -> str:
    base = base.rstrip("/")
    return base if base.endswith("/v1/systemone") else f"{base}/v1/systemone"
DEFAULT_MODEL = "jev-latest"


class JevError(RuntimeError):
    pass


class JevClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str | None = None,
        mock: bool = False,
        cache_dir: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        self.model = model
        self.mock = mock
        self.timeout = timeout
        # /v1/systemone-compatible backends (Laya, Kev, CLM, ...) —
        # JEV_BASE_URL repoints the client; auth header env is per
        # backend but TYPESAFE_API_KEY works as the generic key slot.
        self.url = _endpoint(
            base_url or os.environ.get("JEV_BASE_URL") or API_URL
        )
        # shared client: keep-alive/connection pooling across calls
        # (and across threads — resolver prefetches run in workers).
        self._http = httpx.Client()
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
            try:
                answers, usage = self._call(body)
            except JevError as e:
                # A batch too large for the model context comes back as a
                # 400 max_tokens_exceeded. Halve the questions and retry
                # against the same state; each sub-ask logs its own usage.
                # A single question that still overflows means the state
                # itself is too big — nothing to split, so re-raise.
                if "max_tokens_exceeded" not in str(e) or len(questions) < 2:
                    raise
                log.warning(
                    "jev stage=%s max_tokens with %d questions — splitting",
                    stage,
                    len(questions),
                )
                ids = list(questions)
                mid = len(ids) // 2
                merged: dict[str, dict[str, Any]] = {}
                for part in (ids[:mid], ids[mid:]):
                    merged.update(
                        self.ask(
                            state,
                            {k: questions[k] for k in part},
                            stage=stage,
                        )
                    )
                return merged
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

        # Transient upstream failures (5xx, 429, timeouts) get a short
        # retry — one bad moment shouldn't kill a minutes-long resolve.
        # 4xx (e.g. max_tokens_exceeded) raises immediately: ask() handles it.
        resp = None
        for attempt in range(3):
            try:
                resp = self._http.post(
                    self.url,
                    headers={
                        "Authorization": f"Bearer {self._api_key}",
                        "Content-Type": "application/json",
                    },
                    json=body,
                    timeout=self.timeout,
                )
            except httpx.TimeoutException:
                if attempt == 2:
                    raise JevError("JEV API timed out after 3 attempts")
                time.sleep(2 * (attempt + 1))
                continue
            if resp.status_code == 200:
                break
            if resp.status_code < 500 and resp.status_code != 429:
                raise JevError(
                    f"JEV API error {resp.status_code}: {resp.text[:500]}"
                )
            if attempt < 2:
                log.warning(
                    "JEV API %s — retrying (attempt %d)",
                    resp.status_code,
                    attempt + 2,
                )
                time.sleep(2 * (attempt + 1))
        else:
            raise JevError(
                f"JEV API error {resp.status_code}: {resp.text[:500]}"
            )
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
