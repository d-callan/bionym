"""OMA Browser REST client: precomputed ortholog calls, no auth needed.

https://omabrowser.org/api/ — resolves many ID namespaces (UniProt, RefSeq,
Ensembl, some locus tags) and returns orthologs with rel_type (1:1, 1:n,
m:1, m:n). Coverage of VEuPathDB-style IDs is spotty; callers should treat
empty results as "no data", not "no orthologs".

Future sources for this stage: Ensembl Compara REST (ensemblgenomes for
protists/fungi), OrthoDB, VEuPathDB OrthoMCL groups (needs VEUPATHDB_API_KEY).
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..evidence import Evidence

log = logging.getLogger(__name__)

BASE = "https://omabrowser.org/api"


class OmaClient:
    def __init__(self, timeout: float = 30.0, cache_dir: str | None = None) -> None:
        self.timeout = timeout
        self._cache = None
        if cache_dir:
            import diskcache

            self._cache = diskcache.Cache(os.path.join(cache_dir, "oma"))

    def protein_info(
        self, identifier: str
    ) -> tuple[dict[str, Any] | None, list[Evidence]]:
        url = f"{BASE}/protein/{identifier}/"
        data = self._get(url)
        ev = Evidence(
            source="oma",
            endpoint=url,
            summary=f"protein/{identifier}",
            payload={"found": bool(data)},
        )
        if not data:
            return None, [ev]
        return self._normalize_entry(data), [ev]

    def xrefs(self, entry_nr: int | str) -> list[dict[str, Any]]:
        """Cross-references for an OMA entry (SourceID, ORF name, UniProt…).

        Used to find a resolvable identifier for an ortholog — OMA returns
        UniProt canonical IDs, which don't resolve well via NCBI gene; the
        SourceID (e.g. a VEuPathDB locus tag) usually does.
        """
        url = f"{BASE}/protein/{entry_nr}/xref/"
        data = self._get(url)
        return data if isinstance(data, list) else []

    def orthologs(
        self, identifier: str
    ) -> tuple[list[dict[str, Any]], list[Evidence]]:
        url = f"{BASE}/protein/{identifier}/orthologs/"
        data = self._get(url)
        entries = data if isinstance(data, list) else data.get("orthologs", [])
        records = [self._normalize_ortholog(e) for e in entries]
        ev = Evidence(
            source="oma",
            endpoint=url,
            summary=f"protein/{identifier}/orthologs: {len(records)} hit(s)",
            payload={"n": len(records)},
        )
        return records, [ev]

    # -- internals ---------------------------------------------------------

    @staticmethod
    def _normalize_entry(e: dict[str, Any]) -> dict[str, Any]:
        species = e.get("species") or {}
        return {
            "source": "oma",
            "entry_nr": e.get("entry_nr"),
            "omaid": e.get("omaid"),
            "canonical_id": e.get("canonicalid"),
            "species": species.get("name"),
            "tax_id": species.get("taxon_id") or species.get("taxid"),
            "raw": e,
        }

    @classmethod
    def _normalize_ortholog(cls, e: dict[str, Any]) -> dict[str, Any]:
        rec = cls._normalize_entry(e)
        rec["rel_type"] = e.get("rel_type")
        rec["distance"] = e.get("distance")
        rec["score"] = e.get("score")
        return rec

    def _get(self, url: str) -> Any:
        if self._cache is not None and url in self._cache:
            return self._cache[url]
        try:
            resp = httpx.get(
                url, headers={"Accept": "application/json"}, timeout=self.timeout
            )
            if resp.status_code != 200:
                log.info("OMA %s -> %s", url, resp.status_code)
                return {}
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            log.warning("OMA %s failed: %s", url, e)
            return {}
        if self._cache is not None:
            self._cache[url] = data
        return data
