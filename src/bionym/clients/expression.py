"""Expression Atlas (GXA) client: experiment-level expression metadata.

/gxa/json/experiments?geneQuery= is fuzzy full-text — it returns
experiments across all species for loose matches. We filter to the
resolved organism (deterministic; cross-species hits are unambiguous
noise) and let JEV score the rest. No auth required.

Note: VEuPathDB remains the primary expression source for euk pathogens
but its service API requires VEUPATHDB_API_KEY (release 71+); GEO (via
NcbiClient.geo_datasets_for_gene) covers much of the same upstream data.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..evidence import Evidence

log = logging.getLogger(__name__)

BASE = "https://www.ebi.ac.uk/gxa/json/experiments"


class ExpressionAtlasClient:
    def __init__(self, timeout: float = 30.0, cache_dir: str | None = None) -> None:
        self.timeout = timeout
        self._cache = None
        if cache_dir:
            import diskcache

            self._cache = diskcache.Cache(os.path.join(cache_dir, "gxa"))

    def experiments_for_gene(
        self, symbol: str, organism: str | None = None
    ) -> tuple[list[dict[str, Any]], list[Evidence]]:
        params = {"geneQuery": symbol}
        data = self._get(BASE, params)
        experiments = data.get("experiments", [])
        if organism:
            key = organism.lower()
            experiments = [
                e for e in experiments
                if (e.get("species") or "").lower() == key
            ]
        records = [self._normalize(e) for e in experiments]
        ev = Evidence(
            source="expression_atlas",
            endpoint=BASE,
            summary=(
                f"gxa experiments geneQuery={symbol}: "
                f"{len(records)} after species filter"
            ),
            payload={"geneQuery": symbol, "organism": organism, "n": len(records)},
        )
        return records, [ev]

    @staticmethod
    def _normalize(e: dict[str, Any]) -> dict[str, Any]:
        return {
            "source": "expression_atlas",
            "accession": e.get("experimentAccession"),
            "title": e.get("experimentDescription"),
            "type": e.get("experimentType"),  # Baseline | Differential
            "species": e.get("species"),
            "factors": e.get("experimentalFactors") or [],
            "n_assays": e.get("numberOfAssays"),
            "technology": e.get("technologyType") or [],
            "raw": {},
        }

    def _get(self, url: str, params: dict[str, str]) -> dict[str, Any]:
        cache_key = url + "?" + "&".join(f"{k}={v}" for k, v in sorted(params.items()))
        if self._cache is not None and cache_key in self._cache:
            return self._cache[cache_key]
        try:
            resp = httpx.get(url, params=params, timeout=self.timeout)
            if resp.status_code != 200:
                log.info("GXA %s -> %s", url, resp.status_code)
                return {}
            data = resp.json()
        except (httpx.HTTPError, ValueError) as e:
            log.warning("GXA %s failed: %s", url, e)
            return {}
        if self._cache is not None:
            self._cache[cache_key] = data
        return data
