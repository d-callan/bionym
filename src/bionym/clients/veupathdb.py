"""VEuPathDB client: gene record lookup via the WDK service API.

Verified live against the current service API (Sept 2026) with a
registered-user key. Empirical facts that differ from older assumptions:

- The gene-ID lookup search urlSegment is
  ``single_record_question_GeneRecordClasses_GeneRecordClass`` with parameter
  ``primaryKeys`` = ``"<source_id>,<ProjectId>"`` (a comma-separated string,
  not a JSON array). ``GenesByGeneId`` does not exist on current sites.
- ``reportConfig.attributeFormat`` must be ``"text"`` or ``"display"``;
  ``"table"`` is rejected by request-schema validation.
- ``pagination`` belongs inside ``reportConfig`` (not top level, not
  ``searchConfig``).
- A miss returns HTTP 200 with ``{"isValid": false, "errors": ...}`` —
  check for ``records``, not the status code.
- Useful per-gene tables: ``TranscriptionSummary`` (RNA-seq dataset display
  names), ``ExpressionGraphsDataTable`` (``dataset_id`` = ``DS_*`` plus
  per-sample values/percentiles), ``GeneLinkouts`` (external DB links with
  URLs). ``DS_*`` ids resolve via the ``dataset`` record type (attributes:
  ``display_name``, ``description``, ``pmids``, ...).
"""

from __future__ import annotations

import json
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

# project slug -> (site host, WDK project_id used in record primary keys).
# All sites accept /a as the webapp path; it auto-resolves to the project's
# real context (plasmo, toxo, tritrypdb, ...).
PROJECTS: dict[str, tuple[str, str]] = {
    "plasmodb": ("plasmodb.org", "PlasmoDB"),
    "toxodb": ("toxodb.org", "ToxoDB"),
    "cryptodb": ("cryptodb.org", "CryptoDB"),
    "tritrypdb": ("tritrypdb.org", "TriTrypDB"),
    "fungidb": ("fungidb.org", "FungiDB"),
    "piroplasmadb": ("piroplasmadb.org", "PiroplasmaDB"),
    "amoebadb": ("amoebadb.org", "AmoebaDB"),
    "giardiadb": ("giardiadb.org", "GiardiaDB"),
    "microsporidiadb": ("microsporidiadb.org", "MicrosporidiaDB"),
}

GENE_ID_SEARCH = "single_record_question_GeneRecordClasses_GeneRecordClass"
DATASET_ID_SEARCH = "single_record_question_DatasetRecordClasses_DatasetRecordClass"

# orthomcl.org runs the same WDK service API; group records key on the
# group name that gene records expose as the `orthomcl_name` attribute.
ORTHOMCL_GROUP_URL = (
    "https://orthomcl.org/a/service/record-types/group/searches/"
    "single_record_question_GroupRecordClasses_GroupRecordClass/reports/standard"
)

# Attributes verified available on the single-record gene question.
DEFAULT_ATTRIBUTES = [
    "primary_key", "organism", "name", "product", "gene_type",
    "sequence_id", "ncbi_tax_id", "rnaseq_dataset_count", "link",
    "orthomcl_name", "location_text",
]


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
        record, project, evidence = self._fetch_gene_record(
            gene_id, DEFAULT_ATTRIBUTES, tables=["Alias"]
        )
        if record is None or project is None:
            return [], evidence
        attrs = record.get("attributes", {})
        host = PROJECTS[project][0]
        link = attrs.get("link") or ""
        return [
            {
                "source": "veupathdb",
                "project": project,
                "gene_id": attrs.get("primary_key") or gene_id,
                "symbol": attrs.get("name"),
                "aliases": [
                    {
                        "db": r.get("db_name"),
                        "alias": r.get("alias"),
                        "type": r.get("id_type"),
                    }
                    for r in record.get("tables", {}).get("Alias", [])
                ],
                "description": attrs.get("product"),
                "organism": attrs.get("organism"),
                "gene_type": attrs.get("gene_type"),
                "sequence_id": attrs.get("sequence_id"),
                "ncbi_tax_id": attrs.get("ncbi_tax_id"),
                # downstream stages key on `tax_id` (NCBI record shape)
                "tax_id": attrs.get("ncbi_tax_id"),
                "rnaseq_dataset_count": attrs.get("rnaseq_dataset_count"),
                "orthomcl_name": attrs.get("orthomcl_name"),
                "location": attrs.get("location_text"),
                # `link` is relative to the webapp root; /a auto-resolves
                # to the project's real context (verified: bare /app 404s).
                "url": f"https://{host}/a/{link}" if link else None,
                "raw": record,
            }
        ], evidence

    def gene_datasets(
        self, gene_id: str
    ) -> tuple[dict[str, Any], list[Evidence]]:
        """Transcriptomics datasets that measured this gene.

        Returns ({"project", "datasets", "dataset_names"}, evidence).
        ``datasets`` holds {"dataset_id", "url"} for each DS_* id in
        ExpressionGraphsDataTable; ``dataset_names`` is the display-name
        list from TranscriptionSummary. The two tables don't join on a
        shared key, so both are returned; use ``dataset_record`` to resolve
        a DS_* id to full metadata (description, pmids, ...).
        """
        record, project, evidence = self._fetch_gene_record(
            gene_id,
            ["primary_key"],
            tables=["TranscriptionSummary", "ExpressionGraphsDataTable"],
        )
        if record is None or project is None:
            return {}, evidence
        host = PROJECTS[project][0]
        tables = record.get("tables", {})
        # ExpressionGraphsDataTable rows are per-sample measurements, so
        # sample names aggregate per dataset for free — they encode the
        # conditions assayed (e.g. "Steady_state ring 0h - unique").
        samples_by_ds: dict[str, set[str]] = {}
        for row in tables.get("ExpressionGraphsDataTable", []):
            dsid, sname = row.get("dataset_id"), row.get("sample_name")
            if dsid and sname:
                samples_by_ds.setdefault(dsid, set()).add(sname)
        names = sorted(
            {
                row.get("display_name")
                for row in tables.get("TranscriptionSummary", [])
                if row.get("display_name")
            }
        )
        return {
            "project": project,
            "datasets": [
                {
                    "dataset_id": dsid,
                    "url": f"https://{host}/a/app/record/dataset/{dsid}",
                    "sample_names": sorted(samples_by_ds.get(dsid, ())),
                }
                for dsid in sorted(samples_by_ds)
            ],
            "dataset_names": names,
        }, evidence

    def gene_orthologs(
        self, gene_id: str
    ) -> tuple[list[dict[str, Any]], list[Evidence]]:
        """OrthoMCL orthologs/paralogs within the project site.

        Returns (orthologs, evidence) where each ortholog has gene_id,
        transcript_id, organism, product, is_syntenic, protein_length, url.
        Note: the ``OrthologsLite`` table (cross-VEuPathDB) was empty on all
        genes probed 2026-09 — ``Orthologs`` (within-site) is the populated
        one, so cross-site orthologs still need NCBI/OrthoDB/OMA.
        """
        rows, project, evidence = self._gene_table(gene_id, "Orthologs")
        host = PROJECTS[project][0] if project else ""
        return [
            {
                "gene_id": r.get("ortho_gene_source_id"),
                "transcript_id": r.get("ortho_source_id"),
                "organism": r.get("organism"),
                "product": r.get("gene_product"),
                "is_syntenic": r.get("is_syntenic") == "yes",
                "protein_length": r.get("protein_length"),
                # `gene` is a site-relative path incl. webapp context.
                "url": f"https://{host}{r['gene']}" if r.get("gene") else None,
            }
            for r in rows
        ], evidence

    def gene_go_terms(
        self, gene_id: str
    ) -> tuple[list[dict[str, Any]], list[Evidence]]:
        """GO annotations. Returns (terms, evidence); each term has go_id,
        name, ontology, evidence_code, source, reference, is_not, url."""
        rows, _, evidence = self._gene_table(gene_id, "GOTerms")
        return [
            {
                "go_id": r.get("go_id"),
                "name": r.get("go_term_name"),
                "ontology": r.get("ontology"),
                "evidence_code": r.get("evidence_code"),
                "source": r.get("source"),
                "reference": r.get("reference"),
                "is_not": r.get("is_not") == "yes",
                "url": r.get("go_id_link"),
            }
            for r in rows
        ], evidence

    def gene_pubmed(
        self, gene_id: str
    ) -> tuple[list[dict[str, Any]], list[Evidence]]:
        """PubMed citations linked to the gene. Returns (citations, evidence);
        each has pubmed_id, title, authors, doi, url."""
        rows, _, evidence = self._gene_table(gene_id, "PubMed")
        return [
            {
                "pubmed_id": r.get("pubmed_id"),
                "title": r.get("title"),
                "authors": r.get("authors"),
                "doi": r.get("doi"),
                "url": r.get("pubmed_link"),
            }
            for r in rows
        ], evidence

    def orthomcl_group(
        self, group_name: str
    ) -> tuple[dict[str, Any], list[Evidence]]:
        """Cross-site ortholog group membership from orthomcl.org.

        Unlike the per-site ``Orthologs`` table (within-project only), the
        OrthoMCL group's ``Sequences`` table spans all VEuPathDB species —
        this is the cross-site ortholog source. Members carry ``full_id``
        (heterogeneous: VEuPathDB transcript ids, GenBank proteins),
        ``organism_name``, ``core_peripheral``, ``description``, and a
        site-relative ``sequence_link``. Returns ({group attrs + members},
        evidence); empty dict on a miss.
        """
        body = {
            "searchConfig": {"parameters": {"primaryKeys": group_name}},
            "reportConfig": {
                "attributes": [
                    "primary_key", "group_name", "group_type",
                    "number_of_members",
                ],
                "tables": ["Sequences"],
                "attributeFormat": "text",
            },
        }
        data = self._post(ORTHOMCL_GROUP_URL, body)
        records = data.get("records") or []
        evidence = [
            Evidence(
                source="orthomcl",
                endpoint=ORTHOMCL_GROUP_URL,
                summary=f"OrthoMCL group {group_name!r}",
                payload={"hits": len(records)},
            )
        ]
        if not records:
            return {}, evidence
        rec = records[0]
        members = [
            {
                "full_id": r.get("full_id"),
                "organism": r.get("organism_name"),
                "description": r.get("description"),
                "core_peripheral": r.get("core_peripheral"),
                "length": r.get("length"),
                "url": (
                    f"https://orthomcl.org{r['sequence_link']}"
                    if r.get("sequence_link")
                    else None
                ),
            }
            for r in rec.get("tables", {}).get("Sequences", [])
        ]
        return {**rec.get("attributes", {}), "members": members}, evidence

    def dataset_record(
        self, dataset_id: str, project: str
    ) -> dict[str, Any] | None:
        """Resolve a DS_* dataset id to its record (display_name, pmids, ...).

        The dataset record type has a single-column primary key, so
        ``primaryKeys`` is just the dataset id — no project_id suffix.
        """
        host = PROJECTS[project][0]
        url = (
            f"https://{host}/a/service/record-types/dataset/searches/"
            f"{DATASET_ID_SEARCH}/reports/standard"
        )
        body = {
            "searchConfig": {"parameters": {"primaryKeys": dataset_id}},
            "reportConfig": {
                "attributes": [
                    "primary_key", "display_name", "type",
                    "description", "pmids",
                ],
                "tables": [],
                "attributeFormat": "text",
            },
        }
        records = self._post(url, body).get("records") or []
        return records[0] if records else None

    def dataset_records(
        self, dataset_ids: list[str], project: str
    ) -> dict[str, dict[str, Any]]:
        """Bulk-fetch dataset records via DatasetsById (one POST for all ids).

        Returns {dataset_id: {display_name, summary, type, pmid, citation}}.
        ``summary``/``description`` carry the experimental design in free
        text; ``Publications`` gives the primary citation.
        """
        if not dataset_ids:
            return {}
        host = PROJECTS[project][0]
        url = (
            f"https://{host}/a/service/record-types/dataset/searches/"
            "DatasetsById/reports/standard"
        )
        body = {
            "searchConfig": {"parameters": {"dataset_id": json.dumps(dataset_ids)}},
            "reportConfig": {
                "attributes": [
                    "primary_key", "dataset_id", "display_name", "type",
                    "summary", "description",
                ],
                "tables": ["Publications"],
                "attributeFormat": "text",
            },
        }
        out = {}
        for rec in self._post(url, body).get("records") or []:
            attrs = rec.get("attributes", {})
            pubs = rec.get("tables", {}).get("Publications", [])
            # dataset records' primary_key is the display name; the DS_*
            # id lives in the dataset_id attribute.
            out[attrs.get("dataset_id") or attrs.get("primary_key")] = {
                "display_name": attrs.get("display_name"),
                "summary": attrs.get("summary"),
                "description": attrs.get("description"),
                "type": attrs.get("type"),
                "pmid": pubs[0].get("pmid") if pubs else None,
                "citation": pubs[0].get("citation") if pubs else None,
            }
        return out

    # -- internals ---------------------------------------------------------

    def _gene_table(
        self, gene_id: str, table: str
    ) -> tuple[list[dict[str, Any]], str | None, list[Evidence]]:
        """Fetch one table off the gene record. Returns (rows, project, ev)."""
        record, project, evidence = self._fetch_gene_record(
            gene_id, ["primary_key"], tables=[table]
        )
        if record is None:
            return [], None, evidence
        return record.get("tables", {}).get(table, []), project, evidence

    @staticmethod
    def _candidate_projects(gene_id: str) -> list[str]:
        for prefix, project in PREFIX_PROJECTS:
            if gene_id.startswith(prefix):
                return [project]
        return DEFAULT_PROJECTS

    def _fetch_gene_record(
        self,
        gene_id: str,
        attributes: list[str],
        tables: list[str] | None = None,
    ) -> tuple[dict[str, Any] | None, str | None, list[Evidence]]:
        """POST the single-record gene search on each candidate project.

        Returns (record|None, project slug|None, evidence). A gene that
        doesn't exist on a project yields HTTP 200 with isValid=false and
        no "records" key, so presence of records is the hit test.
        """
        evidence: list[Evidence] = []
        for project in self._candidate_projects(gene_id):
            host, project_id = PROJECTS[project]
            url = (
                f"https://{host}/a/service/record-types/gene/searches/"
                f"{GENE_ID_SEARCH}/reports/standard"
            )
            body = {
                "searchConfig": {
                    "parameters": {"primaryKeys": f"{gene_id},{project_id}"},
                },
                "reportConfig": {
                    "attributes": list(attributes),
                    "tables": list(tables or []),
                    "attributeFormat": "text",
                },
            }
            data = self._post(url, body)
            records = data.get("records") or []
            evidence.append(
                Evidence(
                    source="veupathdb",
                    endpoint=url,
                    summary=f"{project} gene lookup {gene_id!r}",
                    payload={"project": project, "hits": len(records)},
                )
            )
            if records:
                return records[0], project, evidence
        return None, None, evidence

    def _post(self, url: str, body: dict[str, Any]) -> dict[str, Any]:
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
