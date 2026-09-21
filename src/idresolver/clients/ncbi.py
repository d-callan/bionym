"""NCBI client: Datasets v2 + E-utilities, with polite rate limiting.

Covers the M1 needs (gene resolution, taxon lineage) plus
`assemblies_for_taxon` which S2 will consume. All methods return
(normalized_records, Evidence) so claims can cite what was retrieved.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any

import httpx

from ..evidence import Evidence

log = logging.getLogger(__name__)

EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
DATASETS = "https://api.ncbi.nlm.nih.gov/datasets/v2"


class NcbiClient:
    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        cache_dir: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("NCBI_API_KEY") or None
        self.timeout = timeout
        self._last_request = 0.0
        self._min_interval = 0.1 if self.api_key else 0.34  # 10/s vs 3/s
        self._retmax = 20
        self._cache = None
        if cache_dir:
            import diskcache

            self._cache = diskcache.Cache(os.path.join(cache_dir, "ncbi"))

    # -- public API --------------------------------------------------------

    def gene_by_id(self, gene_id: str) -> tuple[list[dict[str, Any]], list[Evidence]]:
        """Datasets v2 gene report for a numeric NCBI Gene ID."""
        url = f"{DATASETS}/gene/id/{gene_id}"
        data = self._get(url)
        reports = data.get("reports", [])
        records = [self._normalize_gene_report(r) for r in reports]
        ev = Evidence(
            source="ncbi_datasets",
            endpoint=url,
            summary=f"gene/id/{gene_id}: {len(reports)} report(s)",
            payload={"n_reports": len(reports)},
        )
        return records, [ev]

    def find_gene(self, term: str) -> tuple[list[dict[str, Any]], list[Evidence]]:
        """E-utilities esearch+esummary on db=gene.

        Tries a fielded `[Gene Name]` search first — a bare All Fields query
        matches any record mentioning the term (37k hits for BRCA1, top hits
        in unrelated species), so symbol-like inputs need the field
        qualifier. Falls back to unqualified for locus tags and other IDs
        that aren't symbols (incl. VEuPathDB-style IDs NCBI indexes).
        """
        search_url = f"{EUTILS}/esearch.fcgi"
        evidence = []
        ids: list[str] = []
        for query in (f"{term}[Gene Name]", term):
            search = self._get(
                search_url,
                params={"db": "gene", "term": query, "retmode": "json", "retmax": 20},
            )
            ids = search.get("esearchresult", {}).get("idlist", [])
            evidence.append(
                Evidence(
                    source="ncbi_eutils",
                    endpoint=search_url,
                    summary=f"esearch db=gene term={query!r}: {len(ids)} hit(s)",
                    payload={"term": query, "idlist": ids},
                )
            )
            if ids:
                break
        if not ids:
            return [], evidence

        summary_url = f"{EUTILS}/esummary.fcgi"
        summ = self._get(
            summary_url,
            params={
                "db": "gene",
                "id": ",".join(ids),
                "retmode": "json",
            },
        )
        result = summ.get("result", {})
        records = [
            self._normalize_gene_summary(result[uid]) for uid in result.get("uids", [])
        ]
        ev2 = Evidence(
            source="ncbi_eutils",
            endpoint=summary_url,
            summary=f"esummary db=gene: {len(records)} record(s)",
            payload={"uids": result.get("uids", [])},
        )
        return records, evidence + [ev2]

    def taxon(self, tax_id: int | str) -> tuple[dict[str, Any] | None, list[Evidence]]:
        """E-utilities efetch db=taxonomy: rank + lineage with ranks/taxids.

        (Datasets v2 taxonomy returns lineage as bare taxids without ranks,
        which can't drive species-rank normalization; efetch LineageEx can.)
        """
        url = f"{EUTILS}/efetch.fcgi"
        text = self._get(
            url,
            params={"db": "taxonomy", "id": str(tax_id), "retmode": "xml"},
            parse="text",
        )
        ev = Evidence(
            source="ncbi_eutils",
            endpoint=url,
            summary=f"efetch db=taxonomy id={tax_id}",
            payload={"tax_id": str(tax_id)},
        )
        if not text:
            return None, [ev]
        return self._parse_taxon_xml(text), [ev]

    def assemblies_for_taxon(
        self, tax_id: int | str
    ) -> tuple[list[dict[str, Any]], list[Evidence]]:
        """All assemblies annotated to a taxid (S2 will consume this)."""
        url = f"{DATASETS}/genome/taxon/{tax_id}/dataset_report"
        data = self._get(url, params={"filters.assembly_source": "all"})
        reports = data.get("reports", [])
        records = [self._normalize_assembly_report(r) for r in reports]
        ev = Evidence(
            source="ncbi_datasets",
            endpoint=url,
            summary=f"genome/taxon/{tax_id}: {len(records)} assemblies",
            payload={"n_reports": len(reports)},
        )
        return records, [ev]

    def geo_datasets_for_gene(
        self, symbol: str, organism: str | None = None
    ) -> tuple[list[dict[str, Any]], list[Evidence]]:
        """GEO DataSets/Series (db=gds) that profile a gene symbol.

        GSE = submitted series, GDS = curated datasets; both are
        dataset-level units (as opposed to per-gene Profiles).
        """
        term = f"{symbol}[Gene Symbol]"
        if organism:
            term += f" AND {organism}[Organism]"
        term += " AND (GSE[ETYP] OR GDS[ETYP])"
        search_url = f"{EUTILS}/esearch.fcgi"
        data = self._get(
            search_url,
            params={
                "db": "gds", "term": term,
                "retmode": "json", "retmax": str(self._retmax),
            },
        )
        result = data.get("esearchresult", {})
        ids = result.get("idlist", [])
        ev = Evidence(
            source="ncbi_eutils",
            endpoint=search_url,
            summary=f"esearch db=gds '{term}': {result.get('count', 0)} hit(s)",
            payload={"term": term, "count": int(result.get("count", 0))},
        )
        if not ids:
            return [], [ev]

        summary_url = f"{EUTILS}/esummary.fcgi"
        data = self._get(
            summary_url,
            params={"db": "gds", "id": ",".join(ids), "retmode": "json"},
        )
        sresult = data.get("result", {})
        records = [
            self._normalize_gds_summary(sresult[uid])
            for uid in sresult.get("uids", [])
            if uid in sresult
        ]
        ev2 = Evidence(
            source="ncbi_eutils",
            endpoint=summary_url,
            summary=f"esummary db=gds: {len(records)} record(s)",
            payload={"uids": sresult.get("uids", [])},
        )
        return records, [ev, ev2]

    # -- normalizers -------------------------------------------------------

    @staticmethod
    def _normalize_gene_report(r: dict[str, Any]) -> dict[str, Any]:
        # Datasets v2 wraps the gene record: report = {"gene": {...}, "query": ...}
        g = r.get("gene") or r
        annotations = g.get("annotations") or []
        assemblies = [
            a["assembly_accession"]
            for a in annotations
            if a.get("assembly_accession")
        ]
        genomic = [
            loc["genomic_accession_version"]
            for a in annotations
            for loc in (a.get("genomic_locations") or [])
            if loc.get("genomic_accession_version")
        ]
        return {
            "source": "ncbi_datasets",
            "gene_id": str(g.get("gene_id", "")),
            "symbol": g.get("symbol"),
            "description": g.get("description"),
            "locus_tag": g.get("locus_tag"),
            "tax_id": g.get("tax_id"),
            "organism": g.get("taxname"),
            "assembly_accession": assemblies[0] if assemblies else None,
            "assembly_accessions": assemblies,
            "genomic_accessions": genomic,
            "raw": r,
        }

    @staticmethod
    def _normalize_gene_summary(r: dict[str, Any]) -> dict[str, Any]:
        genomic = r.get("genomicinfo") or []
        return {
            "source": "ncbi_eutils",
            "gene_id": str(r.get("uid", "")),
            "symbol": r.get("name"),
            "description": r.get("description"),
            "locus_tag": None,
            "aliases": r.get("otheraliases"),
            "tax_id": (r.get("organism") or {}).get("taxid"),
            "organism": (r.get("organism") or {}).get("scientificname"),
            "assembly_accession": None,
            "genomic_accessions": [
                g.get("chraccver") for g in genomic if g.get("chraccver")
            ],
            "raw": r,
        }

    @staticmethod
    def _normalize_gds_summary(r: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": "geo",
            "accession": r.get("accession"),
            "title": r.get("title") or r.get("seriestitle"),
            "taxon": r.get("taxon"),
            "n_samples": r.get("n_samples"),
            "gds_type": r.get("gdstype"),
            "tech_type": r.get("ptechtype"),
            "pubmed_ids": r.get("pubmedids"),
            "summary": r.get("summary"),
            "raw": {},
        }

    @staticmethod
    def _parse_taxon_xml(text: str) -> dict[str, Any] | None:
        import xml.etree.ElementTree as ET

        root = ET.fromstring(text)
        t = root.find(".//Taxon")
        if t is None:
            return None
        lineage = [
            {
                "tax_id": int(e.findtext("TaxId") or 0),
                "name": e.findtext("ScientificName"),
                "rank": (e.findtext("Rank") or "").lower(),
            }
            for e in t.findall(".//LineageEx/Taxon")
        ]
        rank = (t.findtext("Rank") or "").lower()
        tax_id = int(t.findtext("TaxId") or 0)
        name = t.findtext("ScientificName")
        species = next((e for e in lineage if e["rank"] == "species"), None)
        return {
            "tax_id": tax_id,
            "name": name,
            "rank": rank,
            "species_tax_id": (
                species["tax_id"] if species else (tax_id if rank == "species" else None)
            ),
            "species_name": (
                species["name"] if species else (name if rank == "species" else None)
            ),
            "lineage": lineage,
            "raw": {},
        }

    @staticmethod
    def _normalize_assembly_report(r: dict[str, Any]) -> dict[str, Any]:
        info = r.get("assembly_info") or {}
        org = r.get("organism") or {}
        return {
            "accession": r.get("accession"),
            "name": (info.get("assembly_name")) or r.get("assembly_name"),
            "organism": org.get("organism_name"),
            "tax_id": org.get("tax_id"),
            "level": info.get("assembly_level") or r.get("assembly_level"),
            "refseq": bool(info.get("refseq") or r.get("refseq_category")),
            "submission_date": info.get("submission_date"),
            "raw": r,
        }

    # -- transport ---------------------------------------------------------

    def _get(
        self,
        url: str,
        params: dict[str, Any] | None = None,
        parse: str = "json",
    ) -> Any:
        params = dict(params or {})
        if self.api_key:
            params["api_key"] = self.api_key
        cache_key = url + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]

        wait = self._min_interval - (time.monotonic() - self._last_request)
        if wait > 0:
            time.sleep(wait)
        self._last_request = time.monotonic()

        resp = httpx.get(url, params=params, timeout=self.timeout)
        if resp.status_code != 200:
            log.warning("NCBI %s -> %s: %s", url, resp.status_code, resp.text[:200])
            return {} if parse == "json" else ""
        data = resp.text if parse == "text" else resp.json()
        if self._cache is not None:
            self._cache[cache_key] = data
        return data
