"""S6 — cross-assembly presence.

NCBI Remap's public API is retired (410), and UCSC liftOver has thin
euk-pathogen coverage, so remap is done at the annotation level: the
Datasets v2 gene report lists every assembly where NCBI annotated the
gene (deterministic `annotated_in` edges). For related assemblies with
no annotation, JEV judges whether the gene is likely present anyway
(same organism, different assembly version).
"""

from __future__ import annotations

from typing import Any


def build_state(
    match: dict[str, Any],
    annotated: list[str],
    unannotated: list[dict[str, Any]],
) -> dict:
    return {
        "source_gene": {
            "gene_id": match.get("gene_id"),
            "symbol": match.get("symbol"),
            "organism": match.get("organism"),
        },
        "annotated_assemblies": annotated,
        "unannotated_related": [
            {
                "accession": c.get("accession"),
                "name": c.get("name"),
                "organism": c.get("organism"),
                "level": c.get("level"),
                "refseq": c.get("refseq"),
            }
            for c in unannotated
        ],
    }


def build_questions(unannotated: list[dict[str, Any]]) -> dict:
    return {
        f"present_{i}": {
            "type": "noul",
            "instructions": (
                f"`unannotated_related[{i}]` is an assembly of the same "
                "organism as `source_gene`, but NCBI did not annotate the "
                "gene on it. Is the gene likely present in this assembly "
                "anyway? Weigh assembly level and completeness — a gene may "
                "be absent from annotation due to a partial assembly or "
                "annotation pipeline gaps rather than true biological absence."
            ),
        }
        for i in range(len(unannotated))
    }
