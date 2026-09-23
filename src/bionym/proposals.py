"""LLM proposal prompts + strict parsing.

The LLM's only job is enumeration: given a dataset description or a
publication title, list the claims the text could support. Output is a
fixed JSON shape so it can be worked with programmatically — anything
malformed is dropped, and JEV scores each surviving proposal against the
source text before it becomes an edge.
"""

from __future__ import annotations

import json
import logging
from typing import Any

log = logging.getLogger(__name__)

# Per-document bound lives in the prompt, not a post-hoc cap: the model
# can always generate more, so recall is a prompt-quality problem.
MAX_PROPOSALS = 12

SYSTEM = (
    "You enumerate candidate claims from biomedical text for downstream "
    "verification. Output JSON only, no commentary."
)

_INSTRUCTIONS = {
    "dataset": (
        "List every distinct experimental condition, contrast, or sample "
        "group this dataset description explicitly states or clearly "
        "implies (e.g. a compound treatment, dose, timepoint, life-cycle "
        "stage, strain, tissue)."
    ),
    "publication": (
        "List every distinct claim or finding this publication title "
        "explicitly states or clearly implies about the gene, its "
        "function, or its organism."
    ),
}

_SCHEMA = (
    'Return JSON: {"proposals": [{"claim": "<short noun phrase>", '
    '"quote": "<verbatim span from the text supporting it>"}]}. '
    f"At most {MAX_PROPOSALS} proposals. Rules: every proposal must be "
    "checkable against the text alone — no outside knowledge, no "
    "speculation, no duplicates."
)


def build_prompt(kind: str, text: str) -> tuple[str, str]:
    """(system, user) messages for one enumeration call."""
    user = (
        f"{_INSTRUCTIONS[kind]}\n\n{_SCHEMA}\n\n"
        f"Text:\n---\n{text}\n---"
    )
    return SYSTEM, user


def parse(content: str) -> list[dict[str, str]]:
    """Validate the JSON response -> [{claim, quote}]. Drops anything
    malformed rather than failing the whole batch."""
    try:
        data = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        log.warning("LLM returned non-JSON: %.200s", content)
        return []
    proposals = data.get("proposals") if isinstance(data, dict) else None
    if not isinstance(proposals, list):
        return []
    out = []
    for p in proposals[:MAX_PROPOSALS]:
        if isinstance(p, dict) and isinstance(p.get("claim"), str) and p["claim"].strip():
            out.append(
                {
                    "claim": " ".join(p["claim"].split()),
                    "quote": str(p.get("quote") or ""),
                }
            )
    return out


def build_state(texts: dict[str, str], proposals: list[dict[str, Any]]) -> dict:
    """JEV state: source texts + the proposals to verify against them."""
    return {
        "source_texts": texts,
        "proposals": [
            {"node": p["node"], "claim": p["claim"], "quote": p.get("quote")}
            for p in proposals
        ],
    }


def build_questions(proposals: list[dict[str, Any]]) -> dict:
    return {
        f"prop_{i}": {
            "type": "noul",
            "instructions": (
                f"`proposals[{i}]` was machine-extracted from "
                f"`source_texts` for node '{p['node']}'. Does the source "
                "text support this claim? Judge textual support, not "
                "plausibility — reject claims the text doesn't state or "
                "clearly imply, even if they sound reasonable."
            ),
        }
        for i, p in enumerate(proposals)
    }
