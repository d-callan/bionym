"""S3 — orthologs: judge precomputed candidates + absence in related assemblies."""

from __future__ import annotations

from typing import Any


def build_state(
    identifier: str,
    match: dict[str, Any],
    candidates: list[dict[str, Any]],
    related_organisms: list[str],
) -> dict:
    return {
        "input": {"value": identifier},
        "source_gene": {
            "gene_id": match.get("gene_id"),
            "symbol": match.get("symbol"),
            "organism": match.get("organism"),
        },
        "candidates": [
            {
                "id": c.get("canonical_id") or c.get("omaid"),
                "species": c.get("species"),
                "rel_type": c.get("rel_type"),
                "score": c.get("score"),
                "source": c.get("source"),
            }
            for c in candidates
        ],
        "related_organisms": related_organisms,
    }


def build_questions(
    candidates: list[dict[str, Any]], related_organisms: list[str]
) -> dict:
    """Noul per ortholog candidate + Noul per related organism with no
    candidate (absence claim)."""
    questions: dict[str, Any] = {}
    for i in range(len(candidates)):
        questions[f"ortholog_{i}"] = {
            "type": "noul",
            "instructions": (
                f"Is `candidates[{i}]` a 1:1 ortholog of `source_gene`? "
                "rel_type '1:1' is strong evidence; '1:n'/'m:n' imply "
                "duplication and should lower confidence."
            ),
        }
    for j in range(len(related_organisms)):
        questions[f"absent_{j}"] = {
            "type": "noul",
            "instructions": (
                f"No ortholog candidate was found in "
                f"`related_organisms[{j}]`. Is `source_gene` likely genuinely "
                "absent from that organism (rather than a gap in the "
                "ortholog database or an unannotated gene)?"
            ),
        }
    return questions
