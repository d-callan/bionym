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
        "implies (e.g. a compound treatment, dose, tissue, strain "
        "comparison). Rules: only biologically informative conditions — "
        "never the organism itself (already implied by context), never "
        "technical artifacts (read orientation, unique reads, sequencing "
        "metrics). Aggregate series into one condition: 'time series "
        "post infection' not '7 hpi' and '16 hpi' separately; 'life "
        "cycle stages' not individual stages. Strain/sample counts are "
        "fine ('5 strains investigated')."
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


def _extract_json(content: str, key: str = "proposals") -> dict | None:
    """Find the first decodable JSON object holding a `key` list.

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
        if isinstance(data, dict) and isinstance(data.get(key), list):
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
# Cap on datasets/publications that survive triage — each one costs an
# LLM call + JEV verification, and a deep graph can yield hundreds.
MAX_ITEMS_PER_KIND = 100


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


# -- condition normalization ------------------------------------------------

_NORMALIZE_SYSTEM = (
    "You normalize biomedical condition labels for consistency. "
    "Output JSON only, no commentary."
)


def build_normalize_prompt(claims: list[str]) -> tuple[str, str]:
    """(system, user) to map each claim to a canonical condition label."""
    user = (
        "Below is a list of experimental conditions extracted from dataset "
        "descriptions. Rewrite each as a short canonical label so that "
        "equivalent conditions share identical wording (e.g. 'time series "
        "post infection' and 'post-infection timeseries' -> 'time series "
        "post infection'). Keep each label faithful to its original — "
        "normalize wording, not meaning. Return JSON: {\"normalized\": "
        "[\"<label>\", ...]} with exactly one label per input, same order."
        f"\n\nConditions:\n{json.dumps(claims)}"
    )
    return _NORMALIZE_SYSTEM, user


def parse_normalized(content: str, n: int) -> list[str] | None:
    """Parse the normalization response -> list of n labels, or None."""
    data = _extract_json(content or "", key="normalized")
    if data is None:
        return None
    labels = [
        " ".join(str(x).split()) if str(x).strip() else None
        for x in data["normalized"][:n]
    ]
    if len(labels) != n or any(l is None for l in labels):
        return None
    return labels


# -- dataset data claims ----------------------------------------------------
#
# For datasets with fetched per-gene measurements (VEuPathDB
# ExpressionGraphsDataTable today), the same propose-then-score loop turns
# the numbers into claims: the LLM summarizes biologically relevant claims
# with the dataset's own metadata as context, JEV verifies each against
# the serialized data. The graph stays claim-shaped — no raw-data model.

# Rows sent to the LLM/JEV per dataset — enough to see the expression
# range without blowing the prompt.
DATA_ROWS_MAX = 60

_DATA_SYSTEM = (
    "You summarize per-gene measurements from a functional genomics "
    "dataset as biological claims. Output JSON only, no commentary."
)


def serialize_rows(
    rows: list[dict[str, Any]], max_rows: int = DATA_ROWS_MAX
) -> str:
    """Per-sample rows -> 'sample | value | percentile' text, highest
    percentile first so the signal leads."""

    def _pct(r: dict[str, Any]) -> float:
        try:
            return float(r.get("percentile") or 0)
        except (TypeError, ValueError):
            return 0.0

    ordered = sorted(rows, key=_pct, reverse=True)
    lines = ["sample | value | percentile"]
    for r in ordered[:max_rows]:
        lines.append(
            f"{r.get('sample')} | {r.get('value')} | {r.get('percentile')}"
        )
    if len(ordered) > max_rows:
        lines.append(f"... {len(ordered) - max_rows} more rows")
    return "\n".join(lines)


def build_data_prompt(
    meta: dict[str, Any], data_text: str
) -> tuple[str, str]:
    """(system, user): propose biological claims from per-gene data.

    The dataset's own metadata is included — raw values are
    uninterpretable without knowing what the contrast is.
    """
    user = (
        "Below is a dataset's metadata and this gene's per-sample "
        "measurements in it. Propose concise biological claims about THIS "
        "gene that the data supports — e.g. conditions or stages where it "
        "is highly or lowly expressed, or notable contrasts. Each claim "
        "must cite the supporting sample(s)/value(s) in 'quote'.\n\n"
        f"{_SCHEMA}\n\n"
        f"Dataset: {meta.get('label')}\n"
        f"Summary: {meta.get('summary') or 'n/a'}\n\n"
        f"Data:\n---\n{data_text}\n---"
    )
    return _DATA_SYSTEM, user


def build_data_state(
    meta: dict[str, Any], data_text: str, proposals: list[dict[str, Any]]
) -> dict:
    """JEV state: dataset context + serialized data + proposals."""
    return {
        "dataset": {
            "name": meta.get("label"),
            "summary": meta.get("summary"),
        },
        "data": data_text,
        "proposals": [
            {"node": p["node"], "claim": p["claim"], "quote": p.get("quote")}
            for p in proposals
        ],
    }


def build_data_questions(proposals: list[dict[str, Any]]) -> dict:
    return {
        f"prop_{i}": {
            "type": "noul",
            "instructions": (
                f"`proposals[{i}]` was machine-extracted from the "
                "per-sample measurements in `data` for this gene. Does "
                "the data support this claim? Judge the numbers, not "
                "plausibility — reject claims whose stated direction or "
                "magnitude the values don't support."
            ),
        }
        for i, p in enumerate(proposals)
    }
