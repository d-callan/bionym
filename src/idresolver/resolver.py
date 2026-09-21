"""Stage orchestration: S0 classify -> S1 resolve -> (S2+ planned).

Each stage gathers evidence, builds a compact `state`, asks JEV a batch of
typed questions, and materializes nodes/edges whose confidence comes from
the JEV answers.
"""

from __future__ import annotations

import logging
from typing import Any

from .clients.ncbi import NcbiClient
from .clients.veupathdb import VEuPathDBClient
from .evidence import Evidence
from .graph import Edge, KnowledgeGraph, Node, NodeType
from .jev import JevClient
from .questions import s0_classify, s1_resolve

log = logging.getLogger(__name__)


class Resolver:
    def __init__(
        self,
        jev: JevClient,
        ncbi: NcbiClient | None = None,
        veupathdb: VEuPathDBClient | None = None,
        max_candidates: int = 20,
    ) -> None:
        self.jev = jev
        self.ncbi = ncbi or NcbiClient()
        self.veupathdb = veupathdb or VEuPathDBClient()
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

        if depth >= 1:
            self._s1_resolve(identifier, namespace, gene_node, graph)
            graph.metadata["stages"].append("s1_resolve")

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
