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
from ..species import same_species

log = logging.getLogger(__name__)

BASE = "https://rest.uniprot.org/uniprotkb/search"
ENTRY = "https://rest.uniprot.org/uniprotkb"

# GO evidence codes (IDA/TAS/IEA/...) are carried as per-assertion
# provenance on has_go_term edges — a code tier, not a confidence tier.


class UniProtClient:
    def __init__(self, timeout: float = 30.0, cache_dir: str | None = None) -> None:
        self.timeout = timeout
        # shared client: keep-alive/connection pooling across calls
        # (and across threads — resolver prefetches run in workers).
        self._http = httpx.Client()
        self._cache = None
        if cache_dir:
            import diskcache

            self._cache = diskcache.Cache(os.path.join(cache_dir, "uniprot"))

    def search_gene(
        self,
        symbol: str,
        tax_id: int | str | None = None,
        size: int = 3,
        organism: str | None = None,
    ) -> tuple[list[dict[str, Any]], list[Evidence]]:
        """Best UniProtKB entries for a gene symbol (+ organism if known)."""
        query = f"(gene:{symbol})"
        results = []
        if tax_id:
            data = self._get(
                BASE,
                {
                    "query": f"{query} AND (organism_id:{tax_id})",
                    "format": "json",
                    "size": str(size),
                },
            )
            results = data.get("results", [])
        if not results:
            # organism_id misses species-indexed entries whenever tax_id
            # is strain-level (36329 vs 5833). Retry unfiltered — but
            # over-fetch, since bare gene-name hits span all species
            # (AMA1 -> yeast, Toxo, mushrooms) and size=3 would drop the
            # right entry before the organism filter below can keep it.
            data = self._get(
                BASE,
                {"query": query, "format": "json", "size": str(max(size, 20))},
            )
            results = data.get("results", [])
        records = [self._normalize_entry(e) for e in results]
        if organism:
            records = [
                r
                for r in records
                if same_species(organism, r.get("organism"))
            ]
        # Prefer the reviewed (Swiss-Prot) entry when several records in
        # this organism share the gene name — curated canonicals carry
        # the annotation S4 emits. Stable sort keeps API relevance order
        # within each group.
        records.sort(
            key=lambda r: "unreviewed" in str(r.get("entry_type") or "").lower()
        )
        records = records[:size]
        ev = Evidence(
            source="uniprot",
            endpoint=BASE,
            summary=f"uniprotkb {query}: {len(records)} entr(ies)",
            payload={"query": query, "n": len(records)},
        )
        return records, [ev]

    def corroborating_evidence(self, record: dict[str, Any]) -> Evidence:
        """Evidence for a same-gene entry that shares an accession's
        same_as edge — the picked record stays the edge object, this
        stays as provenance."""
        return Evidence(
            source="uniprot",
            endpoint=f"{ENTRY}/{record.get('accession')}",
            summary=(
                f"corroborating UniProt entry {record.get('accession')} "
                f"({record.get('entry_type') or 'unknown'})"
            ),
            payload={
                "accession": record.get("accession"),
                "entry_type": record.get("entry_type"),
                "gene_name": record.get("gene_name"),
                "organism": record.get("organism"),
                "role": "corroborating",
            },
        )

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
        aliases = (
            [
                v.get("value")
                for key in ("synonyms", "orderedLocusNames", "orfNames")
                for v in genes[0].get(key, [])
                if v.get("value")
            ]
            if genes
            else []
        )
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
            "entry_type": e.get("entryType"),
            "gene_name": gene_name,
            "aliases": aliases,
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
            resp = self._http.get(url, params=params, timeout=self.timeout)
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
