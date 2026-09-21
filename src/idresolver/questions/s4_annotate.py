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
