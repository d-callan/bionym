"""S0 — ID classification: which namespace does the input belong to?"""

from __future__ import annotations

import re

# Ordered: most specific first. `locus_tag` is a weak fallback and is dropped
# whenever a more specific namespace also matches.
PATTERNS: list[tuple[str, str]] = [
    ("refseq", r"^[A-Z]{2,3}_[0-9]+\.?[0-9]*$"),  # NM_, XM_, NP_, XP_, NC_, GCF_
    ("ensembl", r"^ENS[A-Z]{0,6}G[0-9]+$"),
    ("uniprot", r"^[OPQ][0-9][A-Z0-9]{3}[0-9]$|^[A-NR-Z][0-9][A-Z][A-Z0-9]{2}[0-9]$"),
    ("veupathdb", r"^[A-Za-z][A-Za-z0-9]{1,9}[_.][0-9][\w.]*$"),  # PF3D7_, Tb927., TGME49_
    ("ncbi_gene", r"^[0-9]{1,10}$"),
    ("gene_symbol", r"^[A-Z][A-Z0-9]{2,15}$"),  # BRCA1, TP53 — uppercase symbol-like
    ("locus_tag", r"^[A-Za-z][\w.-]{2,}$"),
]

NAMESPACE_DESCRIPTIONS: dict[str, str] = {
    "ncbi_gene": "NCBI Gene database numeric ID (e.g. 672).",
    "ensembl": "Ensembl gene ID (e.g. ENSG00000139618, ENSPVA...).",
    "veupathdb": "VEuPathDB gene ID (e.g. PF3D7_0710100, TGME49_..., Tb927...).",
    "uniprot": "UniProt accession (e.g. P12345, Q9Y6K9).",
    "refseq": "RefSeq accession (e.g. NM_000000, XP_..., GCF_...).",
    "gene_symbol": "HGNC-style gene symbol (e.g. BRCA1, TP53).",
    "locus_tag": "Generic locus tag / gene identifier not matching a known namespace.",
}


def regex_hints(identifier: str) -> list[str]:
    """Namespaces whose regex matches, most specific first."""
    hits = [ns for ns, pat in PATTERNS if re.match(pat, identifier)]
    if len(hits) > 1 and "locus_tag" in hits:
        hits.remove("locus_tag")
    return hits


def build_state(identifier: str, hints: list[str]) -> dict:
    return {
        "input": {
            "value": identifier,
            "regex_hints": hints,
        }
    }


def build_questions(candidates: list[str]) -> dict:
    """Choice over candidate namespaces + `other` escape."""
    criteria = {ns: NAMESPACE_DESCRIPTIONS[ns] for ns in candidates}
    criteria["other"] = (
        "None of the above — a different identifier namespace or not a "
        "bioinformatics identifier."
    )
    return {
        "id_type": {
            "type": "choice",
            "instructions": (
                "Which identifier namespace does `input.value` most likely "
                "belong to? `input.regex_hints` lists namespaces whose "
                "format pattern matched; weigh them but judge the actual "
                "value."
            ),
            "criteria": criteria,
        }
    }
