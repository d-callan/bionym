"""UniProt REST client: functional annotation for a resolved gene.

One UniProtKB entry yields GO terms (with evidence codes), KEGG pathway
cross-refs, InterPro/Pfam domains, and keywords — the raw material for S4.
No auth required.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..evidence import Evidence

log = logging.getLogger(__name__)

BASE = "https://rest.uniprot.org/uniprotkb/search"
ENTRY = "https://rest.uniprot.org/uniprotkb"

# GO evidence code -> confidence. Deterministic provenance scoring: these
# annotations are already curated assertions, so confidence reflects the
# strength of the underlying evidence type rather than a JEV judgment.
GO_EVIDENCE_CONFIDENCE = {
    "EXP": 0.95, "IDA": 0.95,
    "IPI": 0.85, "IMP": 0.85, "IGI": 0.85, "IEP": 0.85,
    "HTP": 0.85, "HDA": 0.85, "HMP": 0.85, "HGI": 0.85, "HEP": 0.85,
    "TAS": 0.80,
    "ISS": 0.70, "ISO": 0.70, "ISA": 0.70, "ISM": 0.70, "IGC": 0.70,
    "IBA": 0.70, "IBD": 0.70, "IKR": 0.70, "IRD": 0.70, "RCA": 0.70,
    "IC": 0.65, "NAS": 0.60,
    "IEA": 0.50, "ND": 0.40,
}
DEFAULT_GO_CONFIDENCE = 0.60


class UniProtClient:
    def __init__(self, timeout: float = 30.0, cache_dir: str | None = None) -> None:
        self.timeout = timeout
        self._cache = None
        if cache_dir:
            import diskcache

            self._cache = diskcache.Cache(os.path.join(cache_dir, "uniprot"))

    def search_gene(
        self, symbol: str, tax_id: int | str | None = None, size: int = 3
    ) -> tuple[list[dict[str, Any]], list[Evidence]]:
        """Best UniProtKB entries for a gene symbol (+ organism if known)."""
        query = f"(gene:{symbol})"
        if tax_id:
            query += f" AND (organism_id:{tax_id})"
        params = {"query": query, "format": "json", "size": str(size)}
        data = self._get(BASE, params)
        results = data.get("results", [])
        records = [self._normalize_entry(e) for e in results]
        ev = Evidence(
            source="uniprot",
            endpoint=BASE,
            summary=f"uniprotkb {query}: {len(records)} entr(ies)",
            payload={"query": query, "n": len(records)},
        )
        return records, [ev]

    def entry_by_accession(
        self, accession: str
    ) -> tuple[dict[str, Any] | None, list[Evidence]]:
        """Fetch a single UniProtKB entry by accession (e.g. A0A8A4TR43).

        Used to bridge OMA orthologs (whose canonical ids are UniProt
        accessions) to NCBI Gene via the entry's GeneID cross-reference.
        """
        data = self._get(f"{ENTRY}/{accession}", {"format": "json"})
        ev = Evidence(
            source="uniprot",
            endpoint=f"{ENTRY}/{accession}",
            summary=f"uniprotkb accession {accession}",
            payload={"found": bool(data)},
        )
        if not data or not data.get("primaryAccession"):
            return None, [ev]
        return self._normalize_entry(data), [ev]

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _normalize_entry(e: dict[str, Any]) -> dict[str, Any]:
        xrefs = e.get("uniProtKBCrossReferences") or []
        go_terms = []
        kegg, interpro, pfam = [], [], []
        gene_id = None
        for x in xrefs:
            db = x.get("database")
            props = {p["key"]: p["value"] for p in x.get("properties", [])}
            if db == "GeneID":
                gene_id = x.get("id")
            elif db == "GO":
                term = props.get("GoTerm", "")
                go_terms.append(
                    {
                        "id": x.get("id"),
                        "term": term[2:] if term[:2] in ("F:", "P:", "C:") else term,
                        "aspect": term[:1] if term[:2] in ("F:", "P:", "C:") else "",
                        "evidence": props.get("GoEvidenceType", ""),
                    }
                )
            elif db == "KEGG":
                kegg.append(x.get("id"))
            elif db == "InterPro":
                interpro.append({"id": x.get("id"), "name": props.get("EntryName")})
            elif db == "Pfam":
                pfam.append({"id": x.get("id"), "name": props.get("EntryName")})

        genes = e.get("genes") or []
        gene_name = (genes[0].get("geneName") or {}).get("value") if genes else None
        protein_name = (
            (e.get("proteinDescription") or {})
            .get("recommendedName", {})
            .get("fullName", {})
            .get("value")
        )
        organism = e.get("organism") or {}
        return {
            "source": "uniprot",
            "accession": e.get("primaryAccession"),
            "uniprot_id": e.get("uniProtkbId"),
            "gene_name": gene_name,
            "protein_name": protein_name,
            "organism": organism.get("scientificName"),
            "tax_id": organism.get("taxonId"),
            "go_terms": go_terms,
            "gene_id": gene_id,
            "kegg": kegg,
            "interpro": interpro,
            "pfam": pfam,
            "keywords": [k.get("name") for k in (e.get("keywords") or [])],
            "raw": {},
        }

    def _get(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        cache_key = url + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]
        try:
            resp = httpx.get(url, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                log.info("UniProt %s -> %s", url, resp.status_code)
                return {}
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            log.warning("UniProt %s failed: %s", url, e)
            return {}
        if self._cache is not None:
            self._cache[cache_key] = data
        return data
