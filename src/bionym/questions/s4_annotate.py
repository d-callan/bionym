"""S4 — functional annotation.

GO/pathway/domain edges get deterministic confidence from evidence codes
(curated assertions don't need a JEV judgment per term). JEV contributes
one Score: how functionally characterized is this gene overall?
"""

from __future__ import annotations

from typing import Any


def build_state(match: dict[str, Any], uniprot: dict[str, Any] | None) -> dict:
    return {
        "source_gene": {
            "gene_id": match.get("gene_id"),
            "symbol": match.get("symbol"),
            "description": match.get("description"),
            "organism": match.get("organism"),
        },
        "uniprot": (
            {
                "accession": uniprot.get("accession"),
                "protein_name": uniprot.get("protein_name"),
                "n_go_terms": len(uniprot.get("go_terms", [])),
                "n_domains": len(uniprot.get("interpro", []))
                + len(uniprot.get("pfam", [])),
                "n_pathways": len(uniprot.get("kegg", [])),
                "keywords": uniprot.get("keywords", []),
            }
            if uniprot
            else None
        ),
    }


def build_pick_state(match: dict[str, Any], records: list[dict[str, Any]]) -> dict:
    return {
        "source_gene": {
            "symbol": match.get("symbol"),
            "locus_tag": match.get("locus_tag"),
            "description": match.get("description"),
            "organism": match.get("organism"),
            "tax_id": match.get("tax_id"),
        },
        "uniprot_candidates": [
            {
                "accession": r.get("accession"),
                "uniprot_id": r.get("uniprot_id"),
                "gene_name": r.get("gene_name"),
                "protein_name": r.get("protein_name"),
                "organism": r.get("organism"),
                "tax_id": r.get("tax_id"),
            }
            for r in records
        ],
    }


def build_pick_questions(records: list[dict[str, Any]]) -> dict:
    """Single hit -> Noul confirmation; multiple -> Choice + `none`."""
    if len(records) == 1:
        return {
            "uniprot_match": {
                "type": "noul",
                "instructions": (
                    "Does `uniprot_candidates[0]` describe the same gene as "
                    "`source_gene`? The search matched on gene name, which "
                    "may be a symbol or a locus tag — judge whether the "
                    "entry is a genuine match, not merely a same-named or "
                    "wrong-organism hit."
                ),
            }
        }
    criteria = {
        str(i): (
            f"{r.get('accession')} — {r.get('protein_name') or '?'} "
            f"({r.get('organism') or '?'})"
        )
        for i, r in enumerate(records)
    }
    criteria["none"] = "None of these entries is `source_gene`."
    return {
        "uniprot_match": {
            "type": "choice",
            "instructions": (
                "Which `uniprot_candidates` entry is the same gene as "
                "`source_gene`? The search matched on gene name, which may "
                "be a symbol or a locus tag — prefer entries whose gene "
                "name, protein name, and organism agree with the source."
            ),
            "criteria": criteria,
        }
    }


def build_questions() -> dict:
    return {
        "characterization": {
            "type": "score",
            "instructions": (
                "How functionally characterized is `source_gene`, on a "
                "spectrum from 0 (completely uncharacterized / hypothetical) "
                "to 1 (well-studied, specific molecular function known)? "
                "Weigh the description, protein name, and breadth of "
                "annotations in `uniprot`."
            ),
            "criteria": [
                "hypothetical protein, no functional data",
                "specific molecular function well established",
            ],
        }
    }
