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
        "List every distinct claim or finding this publication's title "
        "and abstract explicitly state or clearly imply about the gene, "
        "its function, or its organism."
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


def _extract_json(content: str) -> dict | None:
    """Find the first decodable JSON object holding a 'proposals' list.

    Models sometimes wrap the payload in stray braces, markdown fences, or
    preamble text (observed: GLM emitting `{"{"proposals": ...}`), so a
    strict json.loads on the whole string is too brittle.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(content):
        if ch != "{":
            continue
        try:
            data, _ = decoder.raw_decode(content, i)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and isinstance(data.get("proposals"), list):
            return data
    return None


def parse(content: str) -> list[dict[str, str]]:
    """Validate the JSON response -> [{claim, quote}]. Drops anything
    malformed rather than failing the whole batch."""
    data = _extract_json(content or "")
    if data is None:
        log.warning("LLM returned no parseable proposals object: %.200s", content)
        return []
    proposals = data["proposals"]
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


# Triage: one JEV call scores every candidate node's likely value before
# any LLM calls are spent. Texts are truncated — triage only needs the
# gist, and full summaries for ~40 datasets overflow JEV's context.
TRIAGE_LEVELS = ["irrelevant", "unlikely", "likely", "very_likely"]
TRIAGE_TEXT_CHARS = 600
# Bound on the text sent to the LLM itself (summaries can be huge).
PROMPT_TEXT_CHARS = 8000


def build_triage_state(
    source_gene: dict[str, Any], items: list[dict[str, Any]]
) -> dict:
    return {
        "source_gene": {
            "symbol": source_gene.get("symbol"),
            "description": source_gene.get("description"),
            "organism": source_gene.get("organism"),
        },
        "items": [
            {"node": it["node"], "kind": it["kind"], "text": it["text"]}
            for it in items
        ],
    }


# Triage criteria differ by kind: a dataset's value is that the gene was
# *measured* under its conditions (the measured_in edge already links it),
# so the question is whether the text names extractable conditions — not
# whether it discusses the gene. A publication's value is whether it
# actually discusses the gene.
_TRIAGE_INSTRUCTIONS = {
    "dataset": (
        "Does `items[{i}]`'s text describe experimental conditions, "
        "contrasts, or sample groups (treatments, doses, timepoints, "
        "life-cycle stages, strains, tissues) that could be enumerated? "
        "Score by the criteria levels."
    ),
    "publication": (
        "How likely is `items[{i}]`'s text to contain information about "
        "what `source_gene` does, its function, or the conditions under "
        "which it is expressed? Score by the criteria levels."
    ),
}


def build_triage_questions(items: list[dict[str, Any]]) -> dict:
    return {
        f"triage_{i}": {
            "type": "score",
            "instructions": _TRIAGE_INSTRUCTIONS[it["kind"]].format(i=i),
            "criteria": TRIAGE_LEVELS,
        }
        for i, it in enumerate(items)
    }


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
