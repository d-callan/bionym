"""S5 — expression: JEV scores candidate expression datasets for relevance.

Candidates come from GEO (db=gds, gene-symbol+organism filtered) and
Expression Atlas (fuzzy geneQuery, species-filtered). Both can contain
false positives — GEO matches on annotation text, GXA matches loosely —
so JEV judges each rather than trusting the source query.
"""

from __future__ import annotations

from typing import Any


def build_state(
    identifier: str,
    match: dict[str, Any],
    candidates: list[dict[str, Any]],
) -> dict:
    return {
        "input": {"value": identifier},
        "source_gene": {
            "gene_id": match.get("gene_id"),
            "symbol": match.get("symbol"),
            "description": match.get("description"),
            "organism": match.get("organism"),
        },
        "candidates": [
            {
                "accession": c.get("accession"),
                "title": c.get("title"),
                "source": c.get("source"),
                "type": c.get("type") or c.get("gds_type"),
                "species": c.get("species") or c.get("taxon"),
                "n_samples": c.get("n_samples") or c.get("n_assays"),
                "factors": c.get("factors", []),
                "technology": c.get("technology") or c.get("tech_type"),
            }
            for c in candidates
        ],
    }


def build_questions(candidates: list[dict[str, Any]]) -> dict:
    return {
        f"relevant_{i}": {
            "type": "noul",
            "instructions": (
                f"Is `candidates[{i}]` an expression dataset that plausibly "
                "contains measurements of `source_gene` in its organism? "
                "The source queries are fuzzy — reject datasets whose title, "
                "species, or factors indicate a different gene, organism, or "
                "unrelated context."
            ),
        }
        for i in range(len(candidates))
    }
