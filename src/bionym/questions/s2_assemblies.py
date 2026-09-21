"""S2 — related & newer assemblies.

Taxon normalization: an assembly's taxon is often tagged at strain/isolate
rank, so we walk the NCBI lineage up to species before collecting relatives.
When the resolved taxon sits below species or the lineage contains a
subspecies rank (e.g. T. brucei), we ask JEV which rank to group at.
"""

from __future__ import annotations

from typing import Any

# Ranks worth grouping relatives at, in preference order.
RANKS_OF_INTEREST = ["species", "subspecies", "genus"]


def build_rank_state(taxon: dict[str, Any]) -> dict:
    return {
        "taxon": {
            "tax_id": taxon.get("tax_id"),
            "name": taxon.get("name"),
            "rank": taxon.get("rank"),
            "lineage": [
                {"name": e.get("name"), "rank": e.get("rank")}
                for e in taxon.get("lineage", [])
            ],
        }
    }


def rank_options(taxon: dict[str, Any]) -> dict[str, str]:
    """Choice criteria: ranks present in the lineage (plus the taxon itself)."""
    options: dict[str, str] = {}
    entries = list(taxon.get("lineage", []))
    entries.append(
        {
            "tax_id": taxon.get("tax_id"),
            "name": taxon.get("name"),
            "rank": taxon.get("rank"),
        }
    )
    for e in entries:
        r = e.get("rank")
        if r in RANKS_OF_INTEREST and r not in options:
            options[r] = f"{r}: {e.get('name')} (taxid {e.get('tax_id')})"
    return options


def build_rank_question(taxon: dict[str, Any]) -> dict:
    """Choice over lineage ranks — only worth asking if >1 option exists."""
    options = rank_options(taxon)
    if len(options) < 2:
        return {}
    return {
        "rank": {
            "type": "choice",
            "instructions": (
                "`taxon` is the organism of the source gene's assembly. For "
                "collecting related assemblies suitable for ortholog "
                "comparison, at which taxonomic rank should relatives be "
                "grouped? Species is the usual default; pick subspecies only "
                "when it is the biologically meaningful unit (e.g. "
                "Trypanosoma brucei subspecies), genus only if the species "
                "has too few assemblies to be useful."
            ),
            "criteria": options,
        }
    }


def taxid_for_rank(taxon: dict[str, Any], rank: str) -> int | None:
    if taxon.get("rank") == rank:
        return taxon.get("tax_id")
    for e in taxon.get("lineage", []):
        if e.get("rank") == rank:
            return e.get("tax_id")
    if rank == "species":
        return taxon.get("species_tax_id")
    return None


def needs_rank_question(taxon: dict[str, Any]) -> bool:
    """Ask JEV only when species isn't the obvious answer."""
    if taxon.get("rank") not in (None, "species"):
        return True
    return any(e.get("rank") == "subspecies" for e in taxon.get("lineage", []))


def build_assembly_state(
    match: dict[str, Any], candidates: list[dict[str, Any]]
) -> dict:
    return {
        "source": {
            "gene": match.get("gene_id"),
            "organism": match.get("organism"),
            "assembly_accession": match.get("assembly_accession"),
        },
        "candidates": [
            {
                "accession": c.get("accession"),
                "name": c.get("name"),
                "organism": c.get("organism"),
                "tax_id": c.get("tax_id"),
                "level": c.get("level"),
                "refseq": c.get("refseq"),
                "submission_date": c.get("submission_date"),
            }
            for c in candidates
        ],
    }


def build_assembly_questions(
    candidates: list[dict[str, Any]], has_source: bool
) -> dict:
    questions: dict[str, Any] = {}
    for i in range(len(candidates)):
        questions[f"related_{i}"] = {
            "type": "noul",
            "instructions": (
                f"Is `candidates[{i}]` an assembly of the same organism as "
                "the source gene's organism (`source.organism`)?"
            ),
        }
        if has_source:
            questions[f"supersedes_{i}"] = {
                "type": "noul",
                "instructions": (
                    f"Is `candidates[{i}]` a newer or preferred assembly for "
                    "the same organism, relative to `source.assembly_accession`? "
                    "Weigh submission_date, refseq status, and assembly level."
                ),
            }
    return questions
