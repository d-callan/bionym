"""VEuPathDB client: best-effort gene record lookup via the WDK service API.

The WDK search/report endpoints are project-specific and their parameter
names vary, so this client tries the likely project site(s) and degrades
gracefully — callers should fall back to NCBI (which indexes VEuPathDB
locus tags in db=gene) when this returns nothing.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..evidence import Evidence

log = logging.getLogger(__name__)

# Prefix -> project site. VEuPathDB gene ID conventions are heterogeneous;
# this covers the common eukaryotic-pathogen projects. Extend as needed.
PREFIX_PROJECTS: list[tuple[str, str]] = [
    ("PF3D7_", "plasmodb"), ("PVA", "plasmodb"), ("PBANKA_", "plasmodb"),
    ("PVX_", "plasmodb"), ("PKNH_", "plasmodb"), ("PY17X_", "plasmodb"),
    ("PO", "plasmodb"), ("PC", "plasmodb"),
    ("TGME49_", "toxodb"), ("TGGT1_", "toxodb"),
    ("Tb927", "tritrypdb"), ("Tb", "tritrypdb"), ("Lm", "tritrypdb"),
    ("LmjF", "tritrypdb"), ("Tbr", "tritrypdb"), ("Tc", "tritrypdb"),
    ("Cgd", "cryptodb"), ("cgd", "cryptodb"), ("Chro", "cryptodb"),
    ("EC", "microsporidiadb"),
    ("B9_", "piroplasmadb"), ("BBOV", "piroplasmadb"),
    ("NCLIV", "piroplasmadb"),
    ("EHI_", "amoebadb"), ("EDI_", "amoebadb"),
    ("GL50803_", "giardiadb"), ("GLP", "giardiadb"),
    ("CRE", "fungidb"), ("CNAG_", "fungidb"),
]

# Fallback order when no prefix matches.
DEFAULT_PROJECTS = ["plasmodb", "toxodb", "cryptodb"]

# All VEuPathDB sites accept /a as the webapp path; it auto-resolves to
# the project's real context (plasmo, toxo, tritrypdb, ...).
BASE = "https://{project}.org/a/service/record-types/gene/searches/GenesByGeneId/reports/standard"


class VEuPathDBClient:
    """WDK service client. Since VEuPathDB release 71 the service API
    requires a registered-user API key; set VEUPATHDB_API_KEY (sent as an
    Authorization bearer). Without it, lookups 401 and callers should fall
    back to NCBI, which indexes VEuPathDB locus tags in db=gene."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 30.0,
        cache_dir: str | None = None,
    ) -> None:
        self.api_key = api_key or os.environ.get("VEUPATHDB_API_KEY") or None
        self.timeout = timeout
        self._cache = None
        if cache_dir:
            import diskcache

            self._cache = diskcache.Cache(os.path.join(cache_dir, "veupathdb"))

    def lookup_gene(
        self, gene_id: str
    ) -> tuple[list[dict[str, Any]], list[Evidence]]:
        """Try project sites for a gene record. Returns (records, evidence)."""
        projects = self._candidate_projects(gene_id)
        evidence: list[Evidence] = []
        for project in projects:
            url = BASE.format(project=project)
            body = {
                "searchConfig": {
                    "parameters": {"ds_gene_ids_gene_id": gene_id},
                },
                "reportConfig": {
                    "attributes": [
                        "primary_key", "organism", "gene_name",
                        "product", "gene_type", "sequence_id",
                    ],
                    "tables": [],
                    "attributeFormat": "table",
                },
            }
            data = self._post(url, body)
            evidence.append(
                Evidence(
                    source="veupathdb",
                    endpoint=url,
                    summary=f"{project} GenesByGeneId {gene_id!r}",
                    payload={"project": project, "ok": bool(data)},
                )
            )
            records = self._records_from_report(data, project)
            if records:
                return records, evidence
        return [], evidence

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _candidate_projects(gene_id: str) -> list[str]:
        for prefix, project in PREFIX_PROJECTS:
            if gene_id.startswith(prefix):
                return [project]
        return DEFAULT_PROJECTS

    @staticmethod
    def _records_from_report(
        data: dict[str, Any], project: str
    ) -> list[dict[str, Any]]:
        records = []
        for rec in data.get("records", []):
            attrs = rec.get("attributes", {})
            records.append(
                {
                    "source": "veupathdb",
                    "project": project,
                    "gene_id": attrs.get("primary_key") or rec.get("id"),
                    "symbol": attrs.get("gene_name"),
                    "description": attrs.get("product"),
                    "organism": attrs.get("organism"),
                    "gene_type": attrs.get("gene_type"),
                    "sequence_id": attrs.get("sequence_id"),
                    "raw": rec,
                }
            )
        return records

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
        import json

        cache_key = url + "|" + json.dumps(body, sort_keys=True)
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]
        headers = {}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        try:
            resp = httpx.post(url, json=body, headers=headers, timeout=self.timeout)
            if resp.status_code == 401 and not self.api_key:
                log.info(
                    "VEuPathDB requires an API key (register at "
                    "veupathdb.org user profile); set VEUPATHDB_API_KEY"
                )
                return {}
            if resp.status_code != 200:
                log.warning(
                    "VEuPathDB %s -> %s: %s", url, resp.status_code, resp.text[:200]
                )
                return {}
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            log.warning("VEuPathDB %s failed: %s", url, e)
            return {}
        if self._cache is not None:
            self._cache[cache_key] = data
        return data
