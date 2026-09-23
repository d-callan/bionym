"""Stage orchestration: S0 classify -> S1 resolve -> (S2+ planned).

Each stage gathers evidence, builds a compact `state`, asks JEV a batch of
typed questions, and materializes nodes/edges whose confidence comes from
the JEV answers.
"""

from __future__ import annotations

import logging
import re
from typing import Any

from .clients.expression import ExpressionAtlasClient
from .clients.kegg import KeggClient
from .clients.ncbi import NcbiClient
from .clients.oma import OmaClient
from .clients.uniprot import GO_EVIDENCE_CONFIDENCE, DEFAULT_GO_CONFIDENCE, UniProtClient
from .clients.veupathdb import VEuPathDBClient
from .evidence import Evidence
from .graph import Edge, KnowledgeGraph, Node, NodeType
from .jev import JevClient
from .species import same_species, species_key
from .questions import (
    s0_classify,
    s1_resolve,
    s2_assemblies,
    s3_orthologs,
    s4_annotate,
    s5_expression,
    s6_remap,
)

log = logging.getLogger(__name__)


class Resolver:
    def __init__(
        self,
        jev: JevClient,
        ncbi: NcbiClient | None = None,
        veupathdb: VEuPathDBClient | None = None,
        oma: OmaClient | None = None,
        uniprot: UniProtClient | None = None,
        gxa: ExpressionAtlasClient | None = None,
        kegg: KeggClient | None = None,
    ) -> None:
        self.jev = jev
        self.ncbi = ncbi or NcbiClient()
        self.veupathdb = veupathdb or VEuPathDBClient()
        self.oma = oma or OmaClient()
        self.uniprot = uniprot or UniProtClient()
        self.gxa = gxa or ExpressionAtlasClient()
        self.kegg = kegg or KeggClient()

    def resolve(self, identifier: str, depth: int = 1) -> KnowledgeGraph:
        graph = KnowledgeGraph(
            metadata={
                "input": identifier,
                "query": identifier,
                "depth": depth,
                "stages": [],
                "jev_usage": {},
            }
        )
        gene_node = Node(
            id=identifier, type=NodeType.GENE, label=identifier
        )
        graph.add_node(gene_node)

        namespace, ev = self._s0_classify(identifier, graph)
        gene_node.id_namespace = namespace
        graph.metadata["stages"].append("s0_classify")

        match = None
        if depth >= 1:
            match = self._s1_resolve(identifier, namespace, gene_node, graph)
            graph.metadata["stages"].append("s1_resolve")

        if depth >= 2 and match:
            self._s2_assemblies(match, graph)
            graph.metadata["stages"].append("s2_assemblies")

        if depth >= 3 and match:
            self._s3_orthologs(identifier, match, graph)
            graph.metadata["stages"].append("s3_orthologs")

        if depth >= 4 and match:
            self._s4_annotate(identifier, match, graph)
            graph.metadata["stages"].append("s4_annotate")

        if depth >= 5 and match:
            self._s5_expression(identifier, match, graph)
            graph.metadata["stages"].append("s5_expression")

        if depth >= 6 and match:
            self._s6_remap(identifier, match, graph)
            graph.metadata["stages"].append("s6_remap")

        graph.metadata["jev_usage"] = {
            "total": self.jev.total_usage(),
            "by_stage": self.jev.usage_log,
        }
        return graph

    # -- S0 ----------------------------------------------------------------

    def _s0_classify(
        self, identifier: str, graph: KnowledgeGraph
    ) -> tuple[str, list[Evidence]]:
        hints = s0_classify.regex_hints(identifier)
        candidates = hints or ["locus_tag"]
        state = s0_classify.build_state(identifier, hints)
        questions = s0_classify.build_questions(candidates)
        answers = self.jev.ask(state, questions, stage="s0_classify")
        ans = answers["id_type"]

        namespace = ans["choice"]
        graph.add_node(
            Node(
                id=f"idtype:{namespace}",
                type=NodeType.ID_TYPE,
                label=namespace,
            )
        )
        graph.add_edge(
            Edge(
                subject=identifier,
                predicate="is_a",
                object=f"idtype:{namespace}",
                confidence=ans.get("confidence", 0.0),
                probabilities=ans.get("probabilities", {}),
                jev_question_id="id_type",
                evidence=[
                    Evidence(
                        source="regex",
                        endpoint="s0_classify.PATTERNS",
                        summary=f"regex hints: {hints or 'none'}",
                        payload={"hints": hints},
                    )
                ],
            )
        )
        return namespace, []

    # -- S1 ----------------------------------------------------------------

    def _s1_resolve(
        self,
        identifier: str,
        namespace: str,
        gene_node: Node,
        graph: KnowledgeGraph,
    ) -> dict[str, Any] | None:
        candidates, evidence = self._fetch_candidates(identifier, namespace)
        if not candidates:
            log.warning("no candidate gene records for %r", identifier)
            graph.metadata["stages"].append("s1_resolve:no_match")
            return None

        state = s1_resolve.build_state(identifier, candidates)
        questions = s1_resolve.build_questions(candidates)
        answers = self.jev.ask(state, questions, stage="s1_resolve")
        ans = answers["gene_match"]

        if len(candidates) == 1:
            match = candidates[0]
            confidence = ans.get("noul", ans.get("confidence", 0.0))
            probabilities = {"yes": ans.get("noul", 0.0)}
        else:
            pick = ans["choice"]
            if pick == "none":
                log.warning("JEV rejected all candidates for %r", identifier)
                graph.metadata["stages"].append("s1_resolve:no_match")
                return None
            match = candidates[int(pick)]
            confidence = ans.get("confidence", 0.0)
            probabilities = ans.get("probabilities", {})

        # esummary hits lack report fields (locus_tag, assembly_accessions,
        # gene_groups) that the VEuPathDB bridge and S6 need — merge the
        # full Datasets record once here so every downstream consumer sees
        # them. `source` stays ncbi_eutils: the hit came from esearch.
        if match.get("source") == "ncbi_eutils" and match.get("gene_id"):
            reports, rep_ev = self.ncbi.gene_by_id(match["gene_id"])
            evidence += rep_ev
            if reports:
                for k, v in reports[0].items():
                    if k != "source" and match.get(k) is None:
                        match[k] = v

        self._materialize_gene(gene_node, match, confidence, probabilities, evidence, graph)
        self._bridge_veupathdb(gene_node.id, match, evidence, graph)
        self._bridge_ncbi(gene_node.id, match, graph)
        self._enrich_ncbi_gene(gene_node.id, match, evidence, graph)
        return match

    def _bridge_veupathdb(
        self,
        identifier: str,
        match: dict[str, Any],
        evidence: list[Evidence],
        graph: KnowledgeGraph,
    ) -> None:
        """Link the resolved gene to its canonical VEuPathDB record.

        VEuPathDB primary keys are locus tags: previous IDs and aliases
        resolve transparently through the PK lookup, but NCBI GeneIDs do
        not — so NCBI-sourced matches bridge via the report's locus_tag.
        VEuPathDB fields (orthomcl_name, veupathdb_id) are merged into
        `match` so later stages (S3 orthologs, S5 expression) can use them.
        """
        vpd = match if match.get("source") == "veupathdb" else None
        vpd_ev = evidence
        if vpd is None:
            locus_tag = match.get("locus_tag")
            if not locus_tag:
                return
            records, vpd_ev = self.veupathdb.lookup_gene(locus_tag)
            if not records:
                return
            vpd = records[0]
            match["orthomcl_name"] = vpd.get("orthomcl_name")

        vpd_id = vpd.get("gene_id")
        if not vpd_id:
            return
        match["veupathdb_id"] = vpd_id
        node_id = f"veupathdb:{vpd_id}"
        # For VEuPathDB-sourced matches the anchor node + same_as edge
        # already exist (created by _materialize_gene with JEV confidence).
        if node_id not in graph.nodes:
            graph.add_node(
                Node(
                    id=node_id,
                    type=NodeType.GENE,
                    label=vpd.get("symbol") or vpd_id,
                    id_namespace="veupathdb",
                    attrs={
                        "project": vpd.get("project"),
                        "organism": vpd.get("organism"),
                        "orthomcl_name": vpd.get("orthomcl_name"),
                        "aliases": vpd.get("aliases"),
                        "url": vpd.get("url"),
                    },
                )
            )
            graph.add_edge(
                Edge(
                    subject=identifier,
                    predicate="same_as",
                    object=node_id,
                    confidence=0.9,  # NCBI's locus_tag assertion, not VEuPathDB's
                    evidence=vpd_ev,
                )
            )

        # VEuPathDB curates its own citation list (with titles/authors) —
        # complements the bare pmid links from NCBI elink.
        citations, cite_ev = self.veupathdb.gene_pubmed(vpd_id)
        for c in citations:
            pmid = c.get("pubmed_id")
            if not pmid:
                continue
            pm_node = f"pubmed:{pmid}"
            graph.add_node(
                Node(
                    id=pm_node,
                    type=NodeType.PUBLICATION,
                    label=c.get("title") or f"PMID {pmid}",
                    id_namespace="pubmed",
                    attrs={"authors": c.get("authors"), "doi": c.get("doi")},
                )
            )
            graph.add_edge(
                Edge(
                    subject=node_id,
                    predicate="cited_in",
                    object=pm_node,
                    confidence=1.0,
                    evidence=cite_ev,
                )
            )

    def _bridge_ncbi(
        self,
        identifier: str,
        match: dict[str, Any],
        graph: KnowledgeGraph,
    ) -> None:
        """Link a non-NCBI match back to its NCBI Gene record.

        NCBI indexes VEuPathDB locus tags in db=gene, so esearch on the
        canonical id finds the GeneID. A tax_id match decides
        deterministically; otherwise JEV picks among the hits. No-op when
        the match already came from NCBI (the query node is the NCBI gene)
        or no locus-tag-like id is available.
        """
        if match.get("source") in ("ncbi_datasets", "ncbi_eutils"):
            return
        query = match.get("veupathdb_id") or match.get("locus_tag")
        if not query:
            return
        records, evidence = self.ncbi.find_gene(query)
        if not records:
            return
        hit, confidence, qid = self._pick_ncbi_hit(
            query, records, match.get("tax_id")
        )
        if not hit:
            return
        match["ncbi_gene_id"] = hit["gene_id"]
        node_id = self._add_ncbigene_node(graph, hit)
        graph.add_edge(
            Edge(
                subject=identifier,
                predicate="same_as",
                object=node_id,
                confidence=confidence,
                jev_question_id=qid,
                evidence=evidence,
            )
        )

    def _pick_ncbi_hit(
        self, query: str, records: list[dict[str, Any]], tax_id: Any
    ) -> tuple[dict[str, Any] | None, float, str]:
        """Choose which find_gene hit `query` refers to.

        A unique tax_id match is strong deterministic evidence (0.9).
        When it doesn't decide — no tax_id, no matching hit, or several
        same-taxon hits — JEV picks (choice) or confirms (noul), and can
        reject all hits via 'none'.
        """
        matched = [
            r for r in records if tax_id and str(r.get("tax_id")) == str(tax_id)
        ]
        if len(matched) == 1:
            return matched[0], 0.9, ""
        pool = matched or records
        state = s1_resolve.build_state(query, pool)
        questions = s1_resolve.build_questions(pool)
        ans = self.jev.ask(state, questions, stage="ncbi_link")["gene_match"]
        if len(pool) == 1:
            return pool[0], ans.get("noul", ans.get("confidence", 0.0)), "gene_match"
        pick = ans.get("choice")
        if pick == "none":
            return None, 0.0, "gene_match"
        try:
            return pool[int(pick)], ans.get("confidence", 0.0), "gene_match"
        except (TypeError, ValueError, IndexError):
            return None, 0.0, "gene_match"

    @staticmethod
    def _add_ncbigene_node(graph: KnowledgeGraph, hit: dict[str, Any]) -> str:
        node_id = f"ncbigene:{hit['gene_id']}"
        graph.add_node(
            Node(
                id=node_id,
                type=NodeType.GENE,
                label=hit.get("symbol") or hit["gene_id"],
                id_namespace="ncbi_gene",
                attrs={
                    "gene_id": hit.get("gene_id"),
                    "organism": hit.get("organism"),
                    "description": hit.get("description"),
                },
            )
        )
        return node_id

    def _enrich_ncbi_gene(
        self,
        identifier: str,
        match: dict[str, Any],
        evidence: list[Evidence],
        graph: KnowledgeGraph,
    ) -> None:
        """Expand the NCBI gene record into product/citation/sequence nodes.

        Runs for both NCBI-sourced matches and bridged ones — the same_as
        targets are entry points to each resource's full record. All edges
        are DB facts (accessions, elink assertions) -> deterministic conf.
        """
        # match["gene_id"] is the uid only for NCBI-sourced matches;
        # bridged ones get ncbi_gene_id from _bridge_ncbi.
        if match.get("source", "").startswith("ncbi"):
            gene_id = match.get("gene_id")
        else:
            gene_id = match.get("ncbi_gene_id")
        if not gene_id:
            return
        # The ncbigene node always exists now — it's the anchor for
        # NCBI-sourced matches, bridged in for the rest.
        parent = f"ncbigene:{gene_id}"
        if parent not in graph.nodes:
            return

        # Bridged matches lack the Datasets report fields — fetch it for
        # genomic accessions (NC_*) and gene_groups (NCBI Ortholog).
        # locus_tag presence marks a match that already carries report
        # fields (ncbi_datasets, or an eutils hit upgraded in S1).
        genomic = match.get("genomic_accessions") or []
        gene_groups = match.get("gene_groups")
        if match.get("locus_tag") is None:
            reports, rep_ev = self.ncbi.gene_by_id(gene_id)
            if reports:
                rep = reports[0]
                genomic = rep.get("genomic_accessions") or []
                gene_groups = rep.get("gene_groups")
                evidence = evidence + rep_ev
                # enrich the bridged node with report metadata
                node = graph.nodes.get(parent)
                if node is not None:
                    node.attrs["gene_type"] = rep.get("gene_type")
                    node.attrs["gene_groups"] = gene_groups
                # write assembly fields back so S2 can hang related
                # assemblies off the real source assembly
                match.setdefault("assembly_accession", rep.get("assembly_accession"))
                match.setdefault("assembly_accessions", rep.get("assembly_accessions"))

        for acc in genomic:
            acc_node = f"nuccore:{acc}"
            graph.add_node(
                Node(id=acc_node, type=NodeType.TRANSCRIPT, label=acc,
                     id_namespace="ncbi_nuccore", attrs={"kind": "genomic"})
            )
            graph.add_edge(
                Edge(subject=parent, predicate="located_on", object=acc_node,
                     confidence=1.0, evidence=evidence)
            )

        products, prod_ev = self.ncbi.gene_products(gene_id)
        for acc in products.get("transcripts", []):
            acc_node = f"nuccore:{acc}"
            graph.add_node(
                Node(id=acc_node, type=NodeType.TRANSCRIPT, label=acc,
                     id_namespace="ncbi_nuccore", attrs={"kind": "transcript"})
            )
            graph.add_edge(
                Edge(subject=parent, predicate="has_transcript", object=acc_node,
                     confidence=1.0, evidence=prod_ev)
            )
        for acc in products.get("proteins", []):
            acc_node = f"protein:{acc}"
            graph.add_node(
                Node(id=acc_node, type=NodeType.PROTEIN, label=acc,
                     id_namespace="ncbi_protein")
            )
            graph.add_edge(
                Edge(subject=parent, predicate="has_protein", object=acc_node,
                     confidence=1.0, evidence=prod_ev)
            )

        links, pm_ev = self.ncbi.gene_pubmed(gene_id)
        for link in links:
            pmid = link.get("pubmed_id")
            if not pmid:
                continue
            pm_node = f"pubmed:{pmid}"
            graph.add_node(
                Node(id=pm_node, type=NodeType.PUBLICATION, label=f"PMID {pmid}",
                     id_namespace="pubmed", attrs={"kind": link.get("kind")})
            )
            graph.add_edge(
                Edge(subject=parent, predicate="cited_in", object=pm_node,
                     confidence=1.0, evidence=pm_ev)
            )

    def _fetch_candidates(
        self, identifier: str, namespace: str
    ) -> tuple[list[dict[str, Any]], list[Evidence]]:
        if namespace == "ncbi_gene":
            return self.ncbi.gene_by_id(identifier)
        if namespace == "veupathdb":
            records, ev = self.veupathdb.lookup_gene(identifier)
            if records:
                return records, ev
            # NCBI indexes VEuPathDB locus tags in db=gene.
            ncbi_records, ncbi_ev = self.ncbi.find_gene(identifier)
            return ncbi_records, ev + ncbi_ev
        # ensembl / uniprot / refseq / locus_tag / other: try NCBI gene search.
        return self.ncbi.find_gene(identifier)

    def _materialize_gene(
        self,
        gene_node: Node,
        rec: dict[str, Any],
        confidence: float,
        probabilities: dict[str, float],
        evidence: list[Evidence],
        graph: KnowledgeGraph,
    ) -> None:
        """Create the resolved record node (the anchor) and same_as it.

        The query node stays a pure input: it only gets is_a (S0) and
        same_as edges — claims that the identifier names a record in an
        external resource. Everything the record asserts about the gene
        (organism, assembly, products, datasets…) hangs off the anchor.
        """
        if rec.get("source") == "veupathdb":
            anchor = f"veupathdb:{rec.get('gene_id')}"
            ns = "veupathdb"
            attrs = {
                "gene_id": rec.get("gene_id"),
                "project": rec.get("project"),
                "organism": rec.get("organism"),
                "orthomcl_name": rec.get("orthomcl_name"),
                "aliases": rec.get("aliases"),
                "url": rec.get("url"),
            }
        else:
            anchor = f"ncbigene:{rec.get('gene_id')}"
            ns = "ncbi_gene"
            attrs = {
                "gene_id": rec.get("gene_id"),
                "organism": rec.get("organism"),
                "description": rec.get("description"),
                "locus_tag": rec.get("locus_tag"),
                "gene_type": rec.get("gene_type"),
                "gene_groups": rec.get("gene_groups"),
            }
        if not rec.get("gene_id"):
            anchor = gene_node.id  # degenerate record: anchor on the query
        rec["anchor"] = anchor

        if anchor != gene_node.id:
            graph.add_node(
                Node(
                    id=anchor,
                    type=NodeType.GENE,
                    label=rec.get("symbol") or rec.get("description") or anchor,
                    id_namespace=ns,
                    attrs=attrs,
                )
            )
        graph.add_edge(
            Edge(
                subject=gene_node.id,
                predicate="same_as",
                object=anchor,
                confidence=confidence,
                probabilities=probabilities,
                jev_question_id="gene_match",
                evidence=evidence,
            )
        )

        if rec.get("organism") or rec.get("tax_id"):
            org_id = f"taxon:{rec.get('tax_id') or rec['organism']}"
            graph.add_node(
                Node(
                    id=org_id,
                    type=NodeType.ORGANISM,
                    label=rec.get("organism") or "",
                    attrs={"tax_id": rec.get("tax_id")},
                )
            )
            graph.add_edge(
                Edge(
                    subject=anchor,
                    predicate="in_organism",
                    object=org_id,
                    confidence=confidence,
                    jev_question_id="gene_match",
                    evidence=evidence,
                )
            )

        if rec.get("assembly_accession"):
            asm_id = f"assembly:{rec['assembly_accession']}"
            graph.add_node(
                Node(
                    id=asm_id,
                    type=NodeType.ASSEMBLY,
                    label=rec["assembly_accession"],
                    id_namespace="insdc",
                )
            )
            graph.add_edge(
                Edge(
                    subject=anchor,
                    predicate="in_assembly",
                    object=asm_id,
                    confidence=confidence,
                    jev_question_id="gene_match",
                    evidence=evidence,
                )
            )

    # -- S2 ----------------------------------------------------------------

    def _s2_assemblies(
        self, match: dict[str, Any], graph: KnowledgeGraph
    ) -> None:
        tax_id = match.get("tax_id")
        if not tax_id:
            log.warning("no tax_id on resolved gene; skipping S2")
            graph.metadata["stages"].append("s2_assemblies:no_taxon")
            return

        taxon, tax_ev = self.ncbi.taxon(tax_id)
        if not taxon:
            graph.metadata["stages"].append("s2_assemblies:no_taxon")
            return

        # Normalize to species rank; ask JEV only when the lineage makes
        # the grouping rank ambiguous (subspecies present, or the assembly
        # taxon sits below species).
        rank = "species"
        if s2_assemblies.needs_rank_question(taxon):
            questions = s2_assemblies.build_rank_question(taxon)
            if questions:
                state = s2_assemblies.build_rank_state(taxon)
                answers = self.jev.ask(state, questions, stage="s2_rank")
                rank = answers["rank"]["choice"]

        group_taxid = (
            s2_assemblies.taxid_for_rank(taxon, rank)
            or taxon.get("species_tax_id")
            or tax_id
        )
        candidates, asm_ev = self.ncbi.assemblies_for_taxon(group_taxid)
        if not candidates:
            graph.metadata["stages"].append("s2_assemblies:no_candidates")
            return

        source_acc = match.get("assembly_accession")
        state = s2_assemblies.build_assembly_state(match, candidates)
        questions = s2_assemblies.build_assembly_questions(
            candidates, has_source=bool(source_acc)
        )
        answers = self.jev.ask(state, questions, stage="s2_assemblies")

        # Related edges hang off the source assembly when known, else the
        # resolved gene record (the anchor), never the raw query node.
        source_node_id = (
            f"assembly:{source_acc}"
            if source_acc
            else match.get("anchor") or graph.metadata["input"]
        )
        evidence = tax_ev + asm_ev
        for i, c in enumerate(candidates):
            acc = c.get("accession")
            if not acc or acc == source_acc:
                continue
            node_id = f"assembly:{acc}"
            graph.add_node(
                Node(
                    id=node_id,
                    type=NodeType.ASSEMBLY,
                    label=c.get("name") or acc,
                    id_namespace="insdc",
                    attrs={
                        "organism": c.get("organism"),
                        "tax_id": c.get("tax_id"),
                        "level": c.get("level"),
                        "refseq": c.get("refseq"),
                        "submission_date": c.get("submission_date"),
                    },
                )
            )
            rel = answers.get(f"related_{i}", {})
            graph.add_edge(
                Edge(
                    subject=source_node_id,
                    predicate="related_assembly",
                    object=node_id,
                    confidence=rel.get("noul", rel.get("confidence", 0.0)),
                    jev_question_id=f"related_{i}",
                    evidence=evidence,
                )
            )
            if source_acc:
                sup = answers.get(f"supersedes_{i}", {})
                graph.add_edge(
                    Edge(
                        subject=source_node_id,
                        predicate="superseded_by",
                        object=node_id,
                        confidence=sup.get("noul", sup.get("confidence", 0.0)),
                        jev_question_id=f"supersedes_{i}",
                        evidence=evidence,
                    )
                )

    # -- S3 ----------------------------------------------------------------

    def _s3_orthologs(
        self, identifier: str, match: dict[str, Any], graph: KnowledgeGraph
    ) -> None:
        candidates, evidence = self.oma.orthologs(identifier)
        oma_id_used = identifier
        if not candidates:
            # OMA's native index is UniProt accessions — retry with the
            # record's UniProt aliases (reviewed SWISSPROT first), then
            # locus_tag/symbol. Capped: each retry is an API call and a
            # missing entry fails slow (read timeout).
            # Alias shape varies by source: VEuPathDB gives
            # {db, alias, type} dicts; NCBI gives plain strings.
            pairs = [
                (a.get("db") or "", a.get("alias"))
                if isinstance(a, dict)
                else ("", a)
                for a in match.get("aliases") or []
            ]
            retries = (
                [v for db, v in pairs if db == "Uniprot/SWISSPROT"]
                + [
                    v
                    for db, v in pairs
                    if db.startswith("Uniprot/") and db != "Uniprot/SWISSPROT"
                ]
                + [match.get("locus_tag"), match.get("symbol")]
                + [v for db, v in pairs if not db.startswith("Uniprot/")]
            )
            for alt in list(
                dict.fromkeys(r for r in retries if r and r != identifier)
            )[:4]:
                candidates, ev2 = self.oma.orthologs(alt)
                evidence += ev2
                if candidates:
                    oma_id_used = alt
                    break

        anchor = match.get("anchor") or identifier

        # ortholog_of/absent_in are OMA's claims, so they hang off the
        # query gene's own OMA entry — not the anchor (which would
        # attribute OMA data to VEuPathDB/NCBI).
        oma_subject = anchor
        entry, pi_ev = self.oma.protein_info(oma_id_used)
        if entry and entry.get("omaid"):
            evidence += pi_ev
            oma_node = f"oma:{entry['omaid']}"
            graph.add_node(
                Node(
                    id=oma_node,
                    type=NodeType.GENE,
                    label=entry.get("canonical_id") or entry["omaid"],
                    id_namespace="oma",
                    attrs={
                        "entry_nr": entry.get("entry_nr"),
                        "species": entry.get("species"),
                        "tax_id": entry.get("tax_id"),
                    },
                )
            )
            graph.add_edge(
                Edge(
                    subject=identifier,
                    predicate="same_as",
                    object=oma_node,
                    confidence=0.9,
                    evidence=pi_ev,
                )
            )
            oma_subject = oma_node

        # Cross-site orthologs: the gene's OrthoMCL group (bridged onto the
        # match in S1) spans all VEuPathDB species, unlike the per-site
        # Orthologs table. Membership is a DB fact -> deterministic edges;
        # whether a member is a true ortholog vs paralog is left to JEV.
        if match.get("orthomcl_name"):
            # OrthoMCL membership is VEuPathDB data — hang it off the
            # veupathdb node when one exists, else the anchor.
            vpd_node = (
                f"veupathdb:{match['veupathdb_id']}"
                if match.get("veupathdb_id")
                else anchor
            )
            self._add_orthomcl_group(vpd_node, match["orthomcl_name"], graph)

        # Organisms of related assemblies (S2) minus the source organism.
        source_key = species_key(match.get("organism"))
        related_organisms = sorted(
            {
                n.attrs["organism"]
                for n in graph.nodes.values()
                if n.type == NodeType.ASSEMBLY
                and n.attrs.get("organism")
                and species_key(n.attrs["organism"]) != source_key
            }
        )
        covered = {species_key(c.get("species")) for c in candidates}
        uncovered = [
            o for o in related_organisms if species_key(o) not in covered
        ]

        if not candidates and not uncovered:
            graph.metadata["stages"].append("s3_orthologs:no_data")
            return

        state = s3_orthologs.build_state(
            identifier, match, candidates, uncovered
        )
        questions = s3_orthologs.build_questions(candidates, uncovered)
        answers = self.jev.ask(state, questions, stage="s3_orthologs")

        for i, c in enumerate(candidates):
            cid = c.get("canonical_id") or c.get("omaid")
            if not cid:
                continue
            # OMA canonical IDs are UniProt accessions, which resolve poorly
            # via NCBI gene. Fetch cross-references and prefer the SourceID
            # (often a VEuPathDB locus tag) as the resolvable identifier.
            resolve_id = cid
            if c.get("entry_nr"):
                for xr in self.oma.xrefs(c["entry_nr"]):
                    if xr.get("source") == "SourceID":
                        resolve_id = xr["xref"]
                        break
            node_id = f"gene:{cid}"
            graph.add_node(
                Node(
                    id=node_id,
                    type=NodeType.GENE,
                    label=cid,
                    id_namespace="oma",
                    attrs={
                        "species": c.get("species"),
                        "tax_id": c.get("tax_id"),
                        "rel_type": c.get("rel_type"),
                        "resolve_id": resolve_id,
                    },
                )
            )
            ans = answers.get(f"ortholog_{i}", {})
            graph.add_edge(
                Edge(
                    subject=oma_subject,
                    predicate="ortholog_of",
                    object=node_id,
                    confidence=ans.get("noul", ans.get("confidence", 0.0)),
                    jev_question_id=f"ortholog_{i}",
                    evidence=evidence,
                )
            )
            self._link_to_ncbi(graph, node_id, resolve_id, c.get("tax_id"))

        for j, org in enumerate(uncovered):
            org_node = f"organism:{org}"
            graph.add_node(
                Node(id=org_node, type=NodeType.ORGANISM, label=org)
            )
            ans = answers.get(f"absent_{j}", {})
            graph.add_edge(
                Edge(
                    subject=oma_subject,
                    predicate="absent_in",
                    object=org_node,
                    confidence=ans.get("noul", ans.get("confidence", 0.0)),
                    jev_question_id=f"absent_{j}",
                    evidence=evidence,
                )
            )

    def _link_to_ncbi(
        self,
        graph: KnowledgeGraph,
        gene_node: str,
        resolve_id: str,
        tax_id: Any,
    ) -> None:
        """Resolve an OMA ortholog to its NCBI Gene page.

        resolve_id is the SourceID xref (often a locus tag), which
        db=gene indexes well. A verified hit gives the ncbigene node —
        S6 then attaches real annotated_in assembly edges to it.
        """
        records, ev = self.ncbi.find_gene(resolve_id)
        if not records:
            return
        hit, confidence, qid = self._pick_ncbi_hit(resolve_id, records, tax_id)
        if not hit:
            return
        node_id = self._add_ncbigene_node(graph, hit)
        graph.add_edge(
            Edge(
                subject=gene_node,
                predicate="same_as",
                object=node_id,
                confidence=confidence,
                jev_question_id=qid,
                evidence=ev,
            )
        )

    def _add_orthomcl_group(
        self, identifier: str, group_name: str, graph: KnowledgeGraph
    ) -> None:
        group, evidence = self.veupathdb.orthomcl_group(group_name)
        if not group:
            return
        og_node = f"orthogroup:{group_name}"
        graph.add_node(
            Node(
                id=og_node,
                type=NodeType.ORTHOLOG_GROUP,
                label=group_name,
                id_namespace="orthomcl",
                attrs={
                    "group_type": group.get("group_type"),
                    "number_of_members": group.get("number_of_members"),
                },
            )
        )
        graph.add_edge(
            Edge(
                subject=identifier,
                predicate="in_orthogroup",
                object=og_node,
                confidence=1.0,
                evidence=evidence,
            )
        )
        # Order members deterministically: those whose species matches a
        # related assembly (S2) first — the candidate genes for the other
        # assemblies — then Core before peripheral.
        asm_species = {
            species_key(n.attrs["organism"])
            for n in graph.nodes.values()
            if n.type == NodeType.ASSEMBLY and n.attrs.get("organism")
        }
        members = sorted(
            group.get("members", []),
            key=lambda m: (
                species_key(m.get("organism")) not in asm_species,
                m.get("core_peripheral") != "Core",
            ),
        )
        for m in members:
            fid = m.get("full_id")
            if not fid:
                continue
            node_id = f"omclseq:{fid}"
            graph.add_node(
                Node(
                    id=node_id,
                    type=NodeType.PROTEIN,
                    label=fid,
                    id_namespace="orthomcl",
                    attrs={
                        "organism": m.get("organism"),
                        "description": m.get("description"),
                        "core_peripheral": m.get("core_peripheral"),
                        "url": m.get("url"),
                    },
                )
            )
            graph.add_edge(
                Edge(
                    subject=og_node,
                    predicate="has_member",
                    object=node_id,
                    confidence=1.0,
                    evidence=evidence,
                )
            )
            self._link_omclseq_to_veupathdb(graph, node_id, fid)

    def _link_omclseq_to_veupathdb(
        self, graph: KnowledgeGraph, omclseq_node: str, full_id: str
    ) -> None:
        """Resolve an OrthoMCL member to its VEuPathDB gene page.

        full_id is a transcript/protein-level id (e.g. PF3D7_1444800-T1,
        AAEL005766-PB); stripping the suffix gives the gene PK, which
        lookup_gene verifies against the WDK record — a real linkout,
        not an inference. GenBank/piped accessions aren't VEuPathDB ids.
        """
        if "|" in full_id or re.match(r"^[A-Z]{2,4}\d{4,}\.\d+$", full_id):
            return
        # full_id is sequence-level: PF3D7_1444800-T1, PCOAH_00046700-t30_1-p1,
        # PKA1H_120042300.1-p1, AAEL005766-PB. Try the dash-stripped form
        # first (keeps dotted gene ids like Tb927.10.12345 intact), then the
        # raw id, then the dot-stripped form for .N-pN suffixes.
        candidates = dict.fromkeys(
            (full_id.split("-")[0], full_id, full_id.split(".")[0])
        )
        for cand in candidates:
            records, ev = self.veupathdb.lookup_gene(cand)
            if not records:
                continue
            rec = records[0]
            vpd_node = f"veupathdb:{rec['gene_id']}"
            graph.add_node(
                Node(
                    id=vpd_node,
                    type=NodeType.GENE,
                    label=rec.get("symbol") or rec["gene_id"],
                    id_namespace="veupathdb",
                    attrs={
                        "project": rec.get("project"),
                        "organism": rec.get("organism"),
                        "url": rec.get("url"),
                    },
                )
            )
            graph.add_edge(
                Edge(
                    subject=omclseq_node,
                    predicate="same_as",
                    object=vpd_node,
                    confidence=0.9,  # suffix strip verified by the lookup hit
                    evidence=ev,
                )
            )
            # the member gene's locus tag resolves in db=gene — bridge to
            # ncbigene so S6 can attach verified annotated_in edges
            self._link_to_ncbi(graph, vpd_node, rec["gene_id"], rec.get("tax_id"))
            return

    # -- S4 ----------------------------------------------------------------

    def _s4_annotate(
        self, identifier: str, match: dict[str, Any], graph: KnowledgeGraph
    ) -> None:
        symbol = match.get("symbol") or match.get("locus_tag") or identifier
        records, evidence = self.uniprot.search_gene(symbol, match.get("tax_id"))
        if not records and symbol != identifier:
            records, ev2 = self.uniprot.search_gene(identifier, match.get("tax_id"))
            evidence += ev2
        if not records:
            graph.metadata["stages"].append("s4_annotate:no_uniprot")
            return

        # Which UniProt entry is this gene? The search matched on a gene
        # name that may be a symbol or a locus tag, so a deterministic
        # name comparison can't decide — JEV picks (or rejects).
        state = s4_annotate.build_pick_state(match, records)
        questions = s4_annotate.build_pick_questions(records)
        ans = self.jev.ask(state, questions, stage="s4_pick")["uniprot_match"]
        if len(records) == 1:
            rec = records[0]
            confidence = ans.get("noul", ans.get("confidence", 0.0))
        else:
            pick = ans.get("choice")
            if pick == "none":
                graph.metadata["stages"].append("s4_annotate:no_uniprot_match")
                return
            try:
                rec = records[int(pick)]
            except (TypeError, ValueError, IndexError):
                graph.metadata["stages"].append("s4_annotate:no_uniprot_match")
                return
            confidence = ans.get("confidence", 0.0)
        anchor = match.get("anchor") or identifier

        # same_as edge: deterministic confidence from identifier agreement.
        # UniProt annotations (GO, domains) hang off the uniprot node —
        # they're that record's claims about the gene.
        acc = rec.get("accession")
        up_node = f"uniprot:{acc}" if acc else anchor
        if acc:
            graph.add_node(
                Node(
                    id=up_node,
                    type=NodeType.GENE,
                    label=rec.get("uniprot_id") or acc,
                    id_namespace="uniprot",
                    attrs={
                        "protein_name": rec.get("protein_name"),
                        "organism": rec.get("organism"),
                    },
                )
            )
            graph.add_edge(
                Edge(
                    subject=identifier,
                    predicate="same_as",
                    object=up_node,
                    confidence=confidence,
                    jev_question_id="uniprot_match",
                    evidence=evidence,
                )
            )

        for t in rec.get("go_terms", []):
            go_id = t.get("id")
            if not go_id:
                continue
            node_id = f"go:{go_id}"
            graph.add_node(
                Node(
                    id=node_id,
                    type=NodeType.GO_TERM,
                    label=t.get("term") or go_id,
                    id_namespace="go",
                    attrs={"aspect": t.get("aspect")},
                )
            )
            graph.add_edge(
                Edge(
                    subject=up_node,
                    predicate="has_go_term",
                    object=node_id,
                    confidence=GO_EVIDENCE_CONFIDENCE.get(
                        t.get("evidence", ""), DEFAULT_GO_CONFIDENCE
                    ),
                    evidence=evidence,
                )
            )

        # UniProt KEGG xrefs are KEGG GENES entries (org:locus-tag, e.g.
        # pfa:PF3D7_0710100), not pathways — represent as same_as gene
        # nodes, then resolve actual pathway membership via the KEGG API.
        for kegg_id in rec.get("kegg", []):
            node_id = f"kegg:{kegg_id}"
            graph.add_node(
                Node(
                    id=node_id,
                    type=NodeType.GENE,
                    label=kegg_id,
                    id_namespace="kegg",
                )
            )
            graph.add_edge(
                Edge(
                    subject=identifier,
                    predicate="same_as",
                    object=node_id,
                    confidence=0.9,
                    evidence=evidence,
                )
            )
            pathways, pw_ev = self.kegg.pathways_for_gene(kegg_id)
            for pw in pathways:
                pw_node = f"kegg:{pw['pathway_id']}"
                graph.add_node(
                    Node(
                        id=pw_node,
                        type=NodeType.PATHWAY,
                        label=pw.get("name") or pw["pathway_id"],
                        id_namespace="kegg",
                    )
                )
                # pathway membership is asserted by the KEGG gene entry
                graph.add_edge(
                    Edge(
                        subject=node_id,
                        predicate="in_pathway",
                        object=pw_node,
                        confidence=0.9,
                        evidence=pw_ev,
                    )
                )

        for source, domains in (("interpro", rec.get("interpro", [])), ("pfam", rec.get("pfam", []))):
            for d in domains:
                dom_id = d.get("id")
                if not dom_id:
                    continue
                node_id = f"domain:{dom_id}"
                graph.add_node(
                    Node(
                        id=node_id,
                        type=NodeType.DOMAIN,
                        label=f"{source.capitalize()}: {d.get('name') or dom_id}",
                        id_namespace=source,
                    )
                )
                graph.add_edge(
                    Edge(
                        subject=up_node,
                        predicate="has_domain",
                        object=node_id,
                        confidence=0.8,
                        evidence=evidence,
                    )
                )

        state = s4_annotate.build_state(match, rec)
        answers = self.jev.ask(state, s4_annotate.build_questions(), stage="s4_annotate")
        graph.nodes[anchor].attrs["function_characterization"] = answers[
            "characterization"
        ].get("score")

    # -- S5 ----------------------------------------------------------------

    def _s5_expression(
        self, identifier: str, match: dict[str, Any], graph: KnowledgeGraph
    ) -> None:
        symbol = match.get("symbol") or match.get("locus_tag") or identifier
        organism = match.get("organism")

        n_vpd = self._add_veupathdb_datasets(identifier, match, graph)

        geo, geo_ev = self.ncbi.geo_datasets_for_gene(symbol, organism)
        gxa, gxa_ev = self.gxa.experiments_for_gene(symbol, organism)
        evidence = geo_ev + gxa_ev

        # JEV scores every candidate for relevance.
        candidates = geo + gxa
        if not candidates:
            if not n_vpd:
                graph.metadata["stages"].append("s5_expression:no_data")
            return

        state = s5_expression.build_state(identifier, match, candidates)
        questions = s5_expression.build_questions(candidates)
        answers = self.jev.ask(state, questions, stage="s5_expression")

        for i, c in enumerate(candidates):
            acc = c.get("accession")
            if not acc:
                continue
            node_id = f"dataset:{acc}"
            graph.add_node(
                Node(
                    id=node_id,
                    type=NodeType.DATASET,
                    label=c.get("title") or acc,
                    id_namespace=c.get("source", ""),
                    attrs={
                        "type": c.get("type") or c.get("gds_type"),
                        "species": c.get("species") or c.get("taxon"),
                        "n_samples": c.get("n_samples") or c.get("n_assays"),
                    },
                )
            )
            ans = answers.get(f"relevant_{i}", {})
            graph.add_edge(
                Edge(
                    subject=match.get("anchor") or identifier,
                    predicate="measured_in",
                    object=node_id,
                    confidence=ans.get("noul", ans.get("confidence", 0.0)),
                    jev_question_id=f"relevant_{i}",
                    evidence=evidence,
                )
            )
            # Experimental factors -> Condition nodes (deterministic metadata).
            for factor in c.get("factors", []):
                cond_id = f"condition:{factor}"
                graph.add_node(
                    Node(id=cond_id, type=NodeType.CONDITION, label=factor)
                )
                graph.add_edge(
                    Edge(
                        subject=node_id,
                        predicate="has_factor",
                        object=cond_id,
                        confidence=1.0,
                        evidence=evidence,
                    )
                )

    def _add_veupathdb_datasets(
        self, identifier: str, match: dict[str, Any], graph: KnowledgeGraph
    ) -> int:
        """VEuPathDB transcriptomics datasets that measured this gene.

        ExpressionGraphsDataTable rows are per-sample measurements of the
        gene itself — membership is a DB fact, so edges are deterministic
        (unlike the fuzzy GEO/GXA candidates, which JEV scores). Uses the
        canonical veupathdb_id bridged onto the match in S1. Returns the
        number of dataset nodes added.
        """
        vpd_id = match.get("veupathdb_id")
        if not vpd_id:
            return 0
        # VEuPathDB datasets hang off the veupathdb node — its own data.
        subject = f"veupathdb:{vpd_id}"
        if subject not in graph.nodes:
            subject = match.get("anchor") or identifier
        result, evidence = self.veupathdb.gene_datasets(vpd_id)
        datasets = result.get("datasets", [])
        # One bulk call resolves display names, summaries, and citations —
        # the metadata JEV would need to judge condition relevance.
        meta = self.veupathdb.dataset_records(
            [d["dataset_id"] for d in datasets]
        )
        for d in datasets:
            dsid = d["dataset_id"]
            m = meta.get(dsid, {})
            node_id = f"dataset:{dsid}"
            graph.add_node(
                Node(
                    id=node_id,
                    type=NodeType.DATASET,
                    label=m.get("display_name") or dsid,
                    id_namespace="veupathdb",
                    attrs={
                        "project": result.get("project"),
                        "type": m.get("type"),
                        "summary": m.get("summary"),
                        "pmid": m.get("pmid"),
                        "sample_names": d.get("sample_names"),
                    },
                )
            )
            graph.add_edge(
                Edge(
                    subject=subject,
                    predicate="measured_in",
                    object=node_id,
                    confidence=1.0,
                    evidence=evidence,
                )
            )
        return len(datasets)

    # -- S6 ----------------------------------------------------------------

    def _s6_remap(
        self, identifier: str, match: dict[str, Any], graph: KnowledgeGraph
    ) -> None:
        # NCBI annotation data hangs off the ncbigene node when one exists
        # (anchor for NCBI matches, bridged otherwise), else the anchor.
        ncbi_uid = (
            match.get("gene_id")
            if match.get("source", "").startswith("ncbi")
            else match.get("ncbi_gene_id")
        )
        subject = f"ncbigene:{ncbi_uid}" if ncbi_uid else match.get("anchor")
        if not subject or subject not in graph.nodes:
            subject = match.get("anchor") or identifier

        # Assemblies where NCBI annotated this gene (from the S1 report).
        annotated = match.get("assembly_accessions") or (
            [match["assembly_accession"]] if match.get("assembly_accession") else []
        )
        evidence = [
            Evidence(
                source="ncbi_datasets",
                endpoint="gene report annotations",
                summary=f"{len(annotated)} annotated assemblies",
                payload={"annotated": annotated},
            )
        ]
        for acc in annotated:
            node_id = f"assembly:{acc}"
            graph.add_node(
                Node(
                    id=node_id,
                    type=NodeType.ASSEMBLY,
                    label=acc,
                    id_namespace="insdc",
                )
            )
            graph.add_edge(
                Edge(
                    subject=subject,
                    predicate="annotated_in",
                    object=node_id,
                    confidence=0.95,
                    evidence=evidence,
                )
            )

        # Ortholog ncbigene nodes get their real annotated_in edges first —
        # they are the verification for presence in related assemblies.
        self._annotate_ortholog_assemblies(graph)
        self._link_shared_orthologs(graph)

        # Related assemblies (S2, same taxon group) where the query gene
        # itself is not annotated. The gene record is assembly-specific, so
        # the honest claim is about an *ortholog* being present — verified
        # when a resolved ortholog's ncbigene is annotated_in that assembly.
        ortholog_asm = {
            e.object: e.subject
            for e in graph.edges
            if e.predicate == "annotated_in" and e.subject != subject
        }
        # GCA_/GCF_ pairs are the same assembly in two namespaces — match on
        # the accession core so a RefSeq-annotated ortholog also verifies
        # its GenBank twin.
        ortholog_cores = {
            k.split("_", 1)[-1]: v for k, v in ortholog_asm.items()
        }
        related_ids = {
            e.object for e in graph.edges if e.predicate == "related_assembly"
        }
        unannotated = [
            n
            for n in graph.nodes.values()
            if n.id in related_ids
            and n.id.removeprefix("assembly:") not in annotated
        ]
        verified = [
            n
            for n in unannotated
            if n.id in ortholog_asm
            or n.id.split("_", 1)[-1] in ortholog_cores
        ]
        unverified = [n for n in unannotated if n not in verified]

        for n in verified:
            orth = ortholog_asm.get(n.id) or ortholog_cores[
                n.id.split("_", 1)[-1]
            ]
            graph.add_edge(
                Edge(
                    subject=subject,
                    predicate="ortholog_present",
                    object=n.id,
                    confidence=0.9,
                    evidence=[
                        Evidence(
                            source="derived",
                            endpoint="ortholog annotated_in",
                            summary=f"{orth} annotated in {n.id}",
                            payload={"ortholog": orth},
                        )
                    ],
                )
            )

        if unverified:
            # No resolved ortholog verifies presence — JEV judges whether a
            # member of the gene's orthogroup is likely there anyway. The
            # subject is the family-level node (orthogroup, else the OMA
            # entry), not the assembly-specific gene record.
            family = next(
                (
                    n.id
                    for n in graph.nodes.values()
                    if n.id.startswith(("orthogroup:", "oma:"))
                ),
                subject,
            )
            cands = [
                {"accession": n.id.removeprefix("assembly:"), **n.attrs}
                for n in unverified
            ]
            state = s6_remap.build_state(match, annotated, cands)
            questions = s6_remap.build_questions(cands)
            answers = self.jev.ask(state, questions, stage="s6_remap")
            for i, c in enumerate(cands):
                ans = answers.get(f"present_{i}", {})
                graph.add_edge(
                    Edge(
                        subject=family,
                        predicate="likely_present",
                        object=f"assembly:{c['accession']}",
                        confidence=ans.get("noul", ans.get("confidence", 0.0)),
                        jev_question_id=f"present_{i}",
                        evidence=[
                            Evidence(
                                source="jev",
                                endpoint="s6_remap",
                                summary=(
                                    "JEV judged orthogroup-member presence "
                                    f"in {c['accession']} from assembly "
                                    "metadata (no ortholog verified)"
                                ),
                                payload={"assembly": c, "answer": ans},
                            )
                        ],
                    )
                )

        self._link_candidate_genes(graph)

    def _annotate_ortholog_assemblies(self, graph: KnowledgeGraph) -> None:
        """Real assembly links for orthologs resolved to NCBI Gene.

        OMA orthologs that bridged to an ncbigene node (S3) get
        annotated_in edges to the assemblies in that gene's Datasets
        report — actual NCBI annotation evidence, replacing the
        species-name guess for those members.
        """
        for e in list(graph.edges):
            if (
                e.predicate != "same_as"
                or not e.object.startswith("ncbigene:")
                or e.subject == graph.metadata["input"]
            ):
                continue
            uid = e.object.split(":", 1)[1]
            reports, ev = self.ncbi.gene_by_id(uid)
            if not reports:
                continue
            rep = reports[0]
            for acc in rep.get("assembly_accessions") or []:
                asm_id = f"assembly:{acc}"
                graph.add_node(
                    Node(
                        id=asm_id,
                        type=NodeType.ASSEMBLY,
                        label=acc,
                        id_namespace="insdc",
                        attrs={
                            "organism": rep.get("organism"),
                            "tax_id": rep.get("tax_id"),
                        },
                    )
                )
                graph.add_edge(
                    Edge(
                        subject=e.object,
                        predicate="annotated_in",
                        object=asm_id,
                        confidence=0.95,
                        evidence=ev,
                    )
                )

    def _link_shared_orthologs(self, graph: KnowledgeGraph) -> None:
        """Link OMA and OrthoMCL members that resolve to the same record.

        When an omclseq member and an OMA gene both carry same_as to the
        same veupathdb/ncbigene node, the two ortholog sources corroborate
        each other — surface that as a direct same_as between the members.
        """
        by_target: dict[str, list[str]] = {}
        for e in graph.edges:
            if e.predicate == "same_as" and e.subject.startswith(
                ("omclseq:", "gene:")
            ):
                by_target.setdefault(e.object, []).append(e.subject)
        for target, members in by_target.items():
            if len(members) < 2:
                continue
            ev = [
                Evidence(
                    source="derived",
                    endpoint="shared ortholog resolution",
                    summary=f"both resolve to {target}",
                    payload={"target": target, "members": members},
                )
            ]
            for m in members[1:]:
                graph.add_edge(
                    Edge(
                        subject=members[0],
                        predicate="same_as",
                        object=m,
                        confidence=0.9,
                        evidence=ev,
                    )
                )

    def _link_candidate_genes(self, graph: KnowledgeGraph) -> None:
        """Join gene nodes to same-species assemblies: has_candidate_gene.

        Ortholog members (OMA gene:*, OrthoMCL omclseq:*) carry a species
        name; assemblies carry an organism. A species match means that
        assembly plausibly encodes that member — a *suspected* gene id for
        the same gene in the other assembly. The join is deterministic
        (tax_id or species-name match); JEV scores each pair's plausibility.
        """
        assemblies = [
            n for n in graph.nodes.values()
            if n.type == NodeType.ASSEMBLY and n.attrs.get("organism")
        ]
        if not assemblies:
            return
        # Only gene-level nodes join — omclseq members are proteins and
        # drop out here. Excluded: the query's own records (input's same_as
        # objects) and members with a verified ncbigene link (they get real
        # annotated_in edges instead of this species-level guess).
        skip = {
            e.subject
            for e in graph.edges
            if e.predicate == "same_as" and e.object.startswith("ncbigene:")
        } | {
            e.object
            for e in graph.edges
            if e.predicate == "same_as" and e.subject == graph.metadata["input"]
        }
        genes = [
            n for n in graph.nodes.values()
            if n.type == NodeType.GENE
            and n.id_namespace in ("oma", "veupathdb")
            and n.id not in skip
            and (n.attrs.get("species") or n.attrs.get("organism"))
        ]
        pairs = []
        for asm in assemblies:
            for g in genes:
                g_name = g.attrs.get("species") or g.attrs.get("organism")
                same_taxon = (
                    asm.attrs.get("tax_id")
                    and g.attrs.get("tax_id")
                    and str(asm.attrs["tax_id"]) == str(g.attrs["tax_id"])
                )
                if not (same_taxon or same_species(asm.attrs["organism"], g_name)):
                    continue
                pairs.append(
                    {
                        "assembly": asm.id,
                        "assembly_organism": asm.attrs["organism"],
                        "assembly_level": asm.attrs.get("level"),
                        "gene": g.id,
                        "gene_species": g_name,
                    }
                )
        if not pairs:
            return
        state = s6_remap.build_candidate_state(pairs)
        questions = s6_remap.build_candidate_questions(pairs)
        answers = self.jev.ask(state, questions, stage="s6_candidates")
        for i, p in enumerate(pairs):
            ans = answers.get(f"candidate_{i}", {})
            graph.add_edge(
                Edge(
                    subject=p["assembly"],
                    predicate="has_candidate_gene",
                    object=p["gene"],
                    confidence=ans.get("noul", ans.get("confidence", 0.0)),
                    jev_question_id=f"candidate_{i}",
                    evidence=[
                        Evidence(
                            source="bionym",
                            endpoint="species join",
                            summary=(
                                f"assembly organism {p['assembly_organism']!r} "
                                f"matches gene species {p['gene_species']!r}"
                            ),
                            payload={
                                "assembly": p["assembly"],
                                "gene": p["gene"],
                            },
                        )
                    ],
                )
            )
