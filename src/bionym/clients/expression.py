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
from ..species import same_species

log = logging.getLogger(__name__)

BASE = "https://www.ebi.ac.uk/gxa/json/experiments"
FTP = "https://ftp.ebi.ac.uk/pub/databases/microarray/data/atlas/experiments"


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
            experiments = [
                e for e in experiments
                if same_species(e.get("species"), organism)
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

    def experiment_gene_values(
        self, accession: str, symbol: str, differential: bool = False
    ) -> tuple[list[dict[str, Any]], list[Evidence]]:
        """Per-group values for a gene in one GXA experiment.

        The JSON data endpoints 400 as of 2026-09, so this reads the
        per-experiment TSVs from EBI's FTP mirror: baseline experiments
        expose ``{acc}-tpms.tsv`` (or ``-fpkms.tsv``) keyed by assay
        group, differential ones ``{acc}-analytics.tsv`` keyed by
        contrast. Group/contrast labels come from the experiment JSON's
        columnHeaders.
        """
        labels = self._column_labels(accession)
        if differential:
            rows = self._analytics_values(accession, symbol, labels)
            tried = [f"{accession}-analytics.tsv"]
        else:
            rows, tried = [], []
            for suffix in ("tpms", "fpkms"):
                tried.append(f"{accession}-{suffix}.tsv")
                rows = self._matrix_values(accession, symbol, suffix, labels)
                if rows:
                    break
        ev = Evidence(
            source="expression_atlas",
            endpoint=f"{FTP}/{accession}/",
            summary=f"GXA {accession}: {len(rows)} values for {symbol}",
            payload={
                "accession": accession,
                "symbol": symbol,
                "files": tried,
            },
        )
        return rows, [ev]

    def _column_labels(self, accession: str) -> dict[str, str]:
        """assayGroupId/contrastId -> human label from the experiment JSON."""
        data = self._get(f"{BASE}/{accession}", {})
        labels = {}
        for h in data.get("columnHeaders", []):
            hid = h.get("assayGroupId") or h.get("id")
            label = h.get("factorValue") or h.get("displayName")
            if hid and label:
                labels[hid] = label
        return labels

    def _matrix_values(
        self,
        accession: str,
        symbol: str,
        suffix: str,
        labels: dict[str, str],
    ) -> list[dict[str, Any]]:
        """Baseline TSV: GeneID | Gene Name | g1 | g2 | ... (replicates
        comma-separated). Rows -> {sample: factorValue, value: mean,
        percentile: within-gene rank}."""
        text = self._get_text(f"{FTP}/{accession}/{accession}-{suffix}.tsv")
        if not text:
            return []
        lines = text.splitlines()
        if not lines:
            return []
        header = lines[0].split("\t")
        sym = symbol.lower()
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            if cols[1].strip().lower() != sym and cols[0].strip().lower() != sym:
                continue
            rows = []
            for i, h in enumerate(header[2:], start=2):
                vals = []
                for v in cols[i].split(",") if i < len(cols) else []:
                    try:
                        vals.append(float(v))
                    except ValueError:
                        pass
                if vals:
                    rows.append(
                        {
                            "sample": labels.get(h, h),
                            "value": round(sum(vals) / len(vals), 3),
                        }
                    )
            n = len(rows)
            if n > 1:
                for rank, i in enumerate(
                    sorted(range(n), key=lambda j: rows[j]["value"])
                ):
                    rows[i]["percentile"] = round(100 * rank / (n - 1), 1)
            return rows
        return []

    def _analytics_values(
        self, accession: str, symbol: str, labels: dict[str, str]
    ) -> list[dict[str, Any]]:
        """Differential TSV: Gene ID | Gene Name | {contrast}.p-value |
        {contrast}.log2foldchange ... Rows -> {sample: contrast label,
        value: log2fc, p_value}."""
        text = self._get_text(f"{FTP}/{accession}/{accession}-analytics.tsv")
        if not text:
            return []
        lines = text.splitlines()
        if not lines:
            return []
        header = lines[0].split("\t")
        contrasts = [
            h.rsplit(".", 1)[0]
            for h in header[2:]
            if h.endswith(".log2foldchange")
        ]
        sym = symbol.lower()
        for line in lines[1:]:
            cols = line.split("\t")
            if len(cols) < 3:
                continue
            if cols[1].strip().lower() != sym and cols[0].strip().lower() != sym:
                continue
            rows = []
            for c in contrasts:
                try:
                    fc = float(cols[header.index(f"{c}.log2foldchange")])
                except (ValueError, IndexError):
                    continue
                try:
                    pv = float(cols[header.index(f"{c}.p-value")])
                except (ValueError, IndexError):
                    pv = None
                rows.append(
                    {
                        "sample": labels.get(c, c),
                        "value": fc,
                        "p_value": pv,
                    }
                )
            return rows
        return []

    def _get_text(self, url: str) -> str | None:
        if self._cache is not None and url in self._cache:
            return self._cache[url]
        try:
            resp = httpx.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                log.info("GXA %s -> %s", url, resp.status_code)
                return None
            text = resp.text
        except httpx.HTTPError as e:
            log.warning("GXA %s failed: %s", url, e)
            return None
        if self._cache is not None:
            self._cache[url] = text
        return text
