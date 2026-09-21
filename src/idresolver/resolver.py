"""Stage orchestration: S0 classify -> S1 resolve -> (S2+ planned).

Each stage gathers evidence, builds a compact `state`, asks JEV a batch of
typed questions, and materializes nodes/edges whose confidence comes from
the JEV answers.
"""

from __future__ import annotations

import logging
from typing import Any

from .clients.expression import ExpressionAtlasClient
from .clients.ncbi import NcbiClient
from .clients.oma import OmaClient
from .clients.uniprot import GO_EVIDENCE_CONFIDENCE, DEFAULT_GO_CONFIDENCE, UniProtClient
from .clients.veupathdb import VEuPathDBClient
from .evidence import Evidence
from .graph import Edge, KnowledgeGraph, Node, NodeType
from .jev import JevClient
from .questions import (
    s0_classify,
    s1_resolve,
    s2_assemblies,
    s3_orthologs,
    s4_annotate,
    s5_expression,
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
        max_candidates: int = 20,
    ) -> None:
        self.jev = jev
        self.ncbi = ncbi or NcbiClient()
        self.veupathdb = veupathdb or VEuPathDBClient()
        self.oma = oma or OmaClient()
        self.uniprot = uniprot or UniProtClient()
        self.gxa = gxa or ExpressionAtlasClient()
        self.max_candidates = max_candidates

    def resolve(self, identifier: str, depth: int = 1) -> KnowledgeGraph:
        graph = KnowledgeGraph(
            metadata={
                "input": identifier,
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
        candidates = candidates[: self.max_candidates]
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

        self._materialize_gene(gene_node, match, confidence, probabilities, evidence, graph)
        return match

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
        gene_node.label = rec.get("symbol") or rec.get("description") or gene_node.id
        gene_node.attrs.update(
            {
                "gene_id": rec.get("gene_id"),
                "description": rec.get("description"),
                "locus_tag": rec.get("locus_tag"),
                "source": rec.get("source"),
            }
        )
        graph.add_edge(
            Edge(
                subject=gene_node.id,
                predicate="resolved_to",
                object=gene_node.id,
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
                    subject=gene_node.id,
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
                    subject=gene_node.id,
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
        assemblies, asm_ev = self.ncbi.assemblies_for_taxon(group_taxid)
        candidates = assemblies[: self.max_candidates]
        if not candidates:
            graph.metadata["stages"].append("s2_assemblies:no_candidates")
            return

        source_acc = match.get("assembly_accession")
        state = s2_assemblies.build_assembly_state(match, candidates)
        questions = s2_assemblies.build_assembly_questions(
            candidates, has_source=bool(source_acc)
        )
        answers = self.jev.ask(state, questions, stage="s2_assemblies")

        # Related edges hang off the source assembly when known, else the gene.
        source_node_id = (
            f"assembly:{source_acc}" if source_acc else graph.metadata["input"]
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

    @staticmethod
    def _species_key(name: str | None) -> str:
        """Genus + species epithet, lowercased — ignores strain/isolate."""
        return " ".join((name or "").lower().split()[:2])

    def _s3_orthologs(
        self, identifier: str, match: dict[str, Any], graph: KnowledgeGraph
    ) -> None:
        candidates, evidence = self.oma.orthologs(identifier)
        if not candidates:
            # OMA resolves several namespaces; retry with symbol/locus_tag.
            alt = match.get("locus_tag") or match.get("symbol")
            if alt and alt != identifier:
                candidates, ev2 = self.oma.orthologs(alt)
                evidence += ev2

        candidates = candidates[: self.max_candidates]

        # Organisms of related assemblies (S2) minus the source organism.
        source_key = self._species_key(match.get("organism"))
        related_organisms = sorted(
            {
                n.attrs["organism"]
                for n in graph.nodes.values()
                if n.type == NodeType.ASSEMBLY
                and n.attrs.get("organism")
                and self._species_key(n.attrs["organism"]) != source_key
            }
        )
        covered = {
            self._species_key(c.get("species")) for c in candidates
        }
        uncovered = [
            o for o in related_organisms if self._species_key(o) not in covered
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
                    },
                )
            )
            ans = answers.get(f"ortholog_{i}", {})
            graph.add_edge(
                Edge(
                    subject=identifier,
                    predicate="ortholog_of",
                    object=node_id,
                    confidence=ans.get("noul", ans.get("confidence", 0.0)),
                    jev_question_id=f"ortholog_{i}",
                    evidence=evidence,
                )
            )

        for j, org in enumerate(uncovered):
            org_node = f"organism:{org}"
            graph.add_node(
                Node(id=org_node, type=NodeType.ORGANISM, label=org)
            )
            ans = answers.get(f"absent_{j}", {})
            graph.add_edge(
                Edge(
                    subject=identifier,
                    predicate="absent_in",
                    object=org_node,
                    confidence=ans.get("noul", ans.get("confidence", 0.0)),
                    jev_question_id=f"absent_{j}",
                    evidence=evidence,
                )
            )

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
        rec = records[0]

        # same_as edge: deterministic confidence from identifier agreement.
        acc = rec.get("accession")
        if acc:
            up_node = f"uniprot:{acc}"
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
            name_match = (rec.get("gene_name") or "").lower() == symbol.lower()
            tax_match = not match.get("tax_id") or rec.get("tax_id") == match.get("tax_id")
            graph.add_edge(
                Edge(
                    subject=identifier,
                    predicate="same_as",
                    object=up_node,
                    confidence=0.9 if (name_match and tax_match) else 0.6,
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
                    subject=identifier,
                    predicate="has_go_term",
                    object=node_id,
                    confidence=GO_EVIDENCE_CONFIDENCE.get(
                        t.get("evidence", ""), DEFAULT_GO_CONFIDENCE
                    ),
                    evidence=evidence,
                )
            )

        for kegg_id in rec.get("kegg", []):
            node_id = f"kegg:{kegg_id}"
            graph.add_node(
                Node(
                    id=node_id,
                    type=NodeType.PATHWAY,
                    label=kegg_id,
                    id_namespace="kegg",
                )
            )
            graph.add_edge(
                Edge(
                    subject=identifier,
                    predicate="in_pathway",
                    object=node_id,
                    confidence=0.8,
                    evidence=evidence,
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
                        label=d.get("name") or dom_id,
                        id_namespace=source,
                    )
                )
                graph.add_edge(
                    Edge(
                        subject=identifier,
                        predicate="has_domain",
                        object=node_id,
                        confidence=0.8,
                        evidence=evidence,
                    )
                )

        state = s4_annotate.build_state(match, rec)
        answers = self.jev.ask(state, s4_annotate.build_questions(), stage="s4_annotate")
        graph.nodes[identifier].attrs["function_characterization"] = answers[
            "characterization"
        ].get("score")

    # -- S5 ----------------------------------------------------------------

    def _s5_expression(
        self, identifier: str, match: dict[str, Any], graph: KnowledgeGraph
    ) -> None:
        symbol = match.get("symbol") or match.get("locus_tag") or identifier
        organism = match.get("organism")

        geo, geo_ev = self.ncbi.geo_datasets_for_gene(symbol, organism)
        gxa, gxa_ev = self.gxa.experiments_for_gene(symbol, organism)
        evidence = geo_ev + gxa_ev

        # Cap each source independently so one noisy source can't starve the
        # other; JEV scores every candidate for relevance.
        candidates = geo[: self.max_candidates] + gxa[: self.max_candidates]
        if not candidates:
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
                    subject=identifier,
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
