"""S1 — entity resolution: which gene record does the input refer to?"""

from __future__ import annotations

from typing import Any


def _candidate_label(rec: dict[str, Any]) -> str:
    parts = [rec.get("gene_id") or "?"]
    if rec.get("symbol"):
        parts.append(str(rec["symbol"]))
    if rec.get("organism"):
        parts.append(f"({rec['organism']})")
    return " ".join(parts)


def build_state(identifier: str, candidates: list[dict[str, Any]]) -> dict:
    """Compact candidate records — keep `state` small to control token cost."""
    slim = [
        {
            "gene_id": c.get("gene_id"),
            "symbol": c.get("symbol"),
            "description": c.get("description"),
            "locus_tag": c.get("locus_tag"),
            "organism": c.get("organism"),
            "tax_id": c.get("tax_id"),
            "assembly_accession": c.get("assembly_accession"),
            "source": c.get("source"),
        }
        for c in candidates
    ]
    return {"input": {"value": identifier}, "candidates": slim}


def build_questions(candidates: list[dict[str, Any]]) -> dict:
    """Single candidate -> Noul confirmation; multiple -> Choice + `none`."""
    if len(candidates) == 1:
        return {
            "gene_match": {
                "type": "noul",
                "instructions": (
                    "Does `candidates[0]` describe the gene that `input.value` "
                    "refers to? Judge whether the record is a genuine match, "
                    "not merely a same-named or partial hit."
                ),
            }
        }
    criteria = {
        str(i): _candidate_label(c) for i, c in enumerate(candidates)
    }
    criteria["none"] = "None of these records is the gene `input.value` refers to."
    return {
        "gene_match": {
            "type": "choice",
            "instructions": (
                "Which record in `candidates` is the gene that `input.value` "
                "refers to? Prefer exact ID/locus-tag matches over loose "
                "symbol or description matches."
            ),
            "criteria": criteria,
        }
    }
