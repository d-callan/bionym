"""KEGG REST client: pathway membership for a KEGG gene entry.

UniProt's KEGG cross-refs are KEGG GENES entries (org:locus-tag, e.g.
``pfa:PF3D7_0710100``), not pathways — the pathway membership lives one
hop away at ``link/pathway/{gene}``. Names come from
``list/pathway/{org}`` (one call per organism, cached). The REST API
speaks TSV, not JSON. No auth required.
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx

from ..evidence import Evidence

log = logging.getLogger(__name__)

BASE = "https://rest.kegg.jp"


class KeggClient:
    def __init__(self, timeout: float = 30.0, cache_dir: str | None = None) -> None:
        self.timeout = timeout
        self._cache = None
        if cache_dir:
            import diskcache

            self._cache = diskcache.Cache(os.path.join(cache_dir, "kegg"))

    def pathways_for_gene(
        self, kegg_gene_id: str
    ) -> tuple[list[dict[str, Any]], list[Evidence]]:
        """Pathways a KEGG gene entry belongs to.

        ``kegg_gene_id`` is the org-prefixed form UniProt emits
        (``hsa:672``, ``pfa:PF3D7_0710100``). Returns
        ([{pathway_id, name}], evidence); empty list when the gene is in
        no pathway (common for uncharacterized genes).
        """
        link_url = f"{BASE}/link/pathway/{kegg_gene_id}"
        link_text = self._get_text(link_url)
        evidence = [
            Evidence(
                source="kegg",
                endpoint=link_url,
                summary=f"kegg link/pathway {kegg_gene_id}",
                payload={"lines": len(link_text.splitlines())},
            )
        ]
        pathway_ids = [
            line.split("\t")[1].removeprefix("path:")
            for line in link_text.splitlines()
            if "\t" in line
        ]
        if not pathway_ids:
            return [], evidence

        org = kegg_gene_id.split(":", 1)[0]
        names = self._pathway_names(org)
        evidence.append(
            Evidence(
                source="kegg",
                endpoint=f"{BASE}/list/pathway/{org}",
                summary=f"kegg list/pathway {org}: {len(names)} pathway(s)",
                payload={"org": org},
            )
        )
        return [
            {"pathway_id": pid, "name": names.get(pid)} for pid in pathway_ids
        ], evidence

    # -- internals ---------------------------------------------------------

    def _pathway_names(self, org: str) -> dict[str, str]:
        """{pathway_id: name} for an organism, minus the ' - Organism' suffix."""
        text = self._get_text(f"{BASE}/list/pathway/{org}")
        names = {}
        for line in text.splitlines():
            pid, _, name = line.partition("\t")
            if pid and name:
                names[pid.strip()] = name.split(" - ")[0].strip()
        return names

    def _get_text(self, url: str) -> str:
        if self._cache is not None and url in self._cache:
            return self._cache[url]
        try:
            resp = httpx.get(url, timeout=self.timeout)
            if resp.status_code != 200:
                log.info("KEGG %s -> %s", url, resp.status_code)
                return ""
            text = resp.text
        except httpx.HTTPError as e:
            log.warning("KEGG %s failed: %s", url, e)
            return ""
        if self._cache is not None:
            self._cache[url] = text
        return text
