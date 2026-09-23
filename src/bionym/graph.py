"""Knowledge graph model: typed nodes, scored edges, evidence provenance."""

from __future__ import annotations

import json
import logging
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, computed_field

from .evidence import Evidence

log = logging.getLogger(__name__)


class NodeType(str, Enum):
    GENE = "Gene"
    ASSEMBLY = "Assembly"
    ORGANISM = "Organism"
    ORTHOLOG_GROUP = "OrthologGroup"
    PATHWAY = "Pathway"
    GO_TERM = "GOTerm"
    DOMAIN = "Domain"
    DATASET = "Dataset"
    CONDITION = "Condition"
    CLAIM = "Claim"
    ID_TYPE = "IdType"
    TRANSCRIPT = "Transcript"
    PROTEIN = "Protein"
    PUBLICATION = "Publication"


class Node(BaseModel):
    id: str
    type: NodeType
    label: str = ""
    id_namespace: str = ""  # e.g. "ncbi_gene", "veupathdb", "uniprot"
    attrs: dict[str, Any] = Field(default_factory=dict)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def url(self) -> str | None:
        """External database page for this entity, derived from id/namespace.

        Single source of truth — flows to CLI JSON, API, web UI and HTML
        report automatically via model_dump.
        """
        # an explicit url recorded on the node wins (e.g. VEuPathDB
        # project-specific gene pages, OrthoMCL sequence_link)
        if self.attrs.get("url"):
            return self.attrs["url"]
        ns, nid = self.id_namespace, self.id
        if ":" in nid:
            prefix, _, rest = nid.partition(":")
        else:
            prefix, rest = "", nid
        if self.type == NodeType.GENE:
            if ns == "uniprot":
                return f"https://www.uniprot.org/uniprotkb/{rest}"
            if ns == "oma":
                return f"https://omabrowser.org/oma/info/{rest}"
            if ns == "veupathdb":
                return f"https://veupathdb.org/veupathdb/app/record/gene/{rest}"
            if ns == "kegg":
                return f"https://www.kegg.jp/entry/{rest}"
            gene_id = self.attrs.get("gene_id") or (rest if rest.isdigit() else None)
            if gene_id:
                return f"https://www.ncbi.nlm.nih.gov/gene/{gene_id}"
            return None
        if prefix == "assembly":
            return f"https://www.ncbi.nlm.nih.gov/datasets/genome/{rest}/"
        if prefix == "taxon":
            return f"https://www.ncbi.nlm.nih.gov/Taxonomy/Browser/wwwtax.cgi?id={rest}"
        if prefix == "go":
            return f"https://www.ebi.ac.uk/QuickGO/term/{rest}"
        if prefix == "kegg":
            return f"https://www.kegg.jp/entry/{rest}"
        if prefix == "domain":
            if ns == "pfam":
                return f"https://www.ebi.ac.uk/interpro/entry/pfam/{rest}"
            return f"https://www.ebi.ac.uk/interpro/entry/InterPro/{rest}"
        if prefix == "orthogroup":
            return f"https://orthomcl.org/orthomcl/app/record/group/{rest}"
        if prefix == "omclseq":
            return f"https://orthomcl.org/orthomcl/app/record/sequence/{rest}"
        if prefix == "dataset":
            if ns == "veupathdb":
                return f"https://veupathdb.org/veupathdb/app/record/dataset/{rest}"
            if rest.startswith("E-"):
                return f"https://www.ebi.ac.uk/gxa/experiments/{rest}"
            return f"https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc={rest}"
        if prefix == "nuccore":
            return f"https://www.ncbi.nlm.nih.gov/nuccore/{rest}"
        if prefix == "protein":
            return f"https://www.ncbi.nlm.nih.gov/protein/{rest}"
        if prefix == "pubmed":
            return f"https://pubmed.ncbi.nlm.nih.gov/{rest}/"
        return None


class Edge(BaseModel):
    """A claim: subject --predicate--> object, with JEV confidence + evidence."""

    subject: str
    predicate: str  # e.g. "is_a", "resolved_to", "in_organism", "in_assembly"
    object: str
    confidence: float = 0.0
    probabilities: dict[str, float] = Field(default_factory=dict)
    jev_question_id: str = ""
    evidence: list[Evidence] = Field(default_factory=list)


class KnowledgeGraph(BaseModel):
    nodes: dict[str, Node] = Field(default_factory=dict)
    edges: list[Edge] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def add_node(self, node: Node) -> Node:
        """Insert or merge a node (later attrs enrich, don't clobber)."""
        existing = self.nodes.get(node.id)
        if existing is None:
            self.nodes[node.id] = node
            return node
        if node.label and not existing.label:
            existing.label = node.label
        if node.id_namespace and not existing.id_namespace:
            existing.id_namespace = node.id_namespace
        existing.attrs.update({k: v for k, v in node.attrs.items() if v is not None})
        return existing

    def add_edge(self, edge: Edge) -> Edge:
        self.edges.append(edge)
        return edge

    def mark_stage(self, name: str) -> None:
        """Record a completed pipeline stage and log it — the only
        progress signal while a resolve is in flight."""
        self.metadata.setdefault("stages", []).append(name)
        log.info("[%s] stage %s", self.metadata.get("input"), name)

    def filter_by_confidence(self, min_confidence: float) -> "KnowledgeGraph":
        """Drop edges below `min_confidence`, then nodes left with no edges.

        The query node (metadata['query']) is always kept so the graph
        retains a root even when everything is filtered out.
        """
        if min_confidence <= 0:
            return self
        keep = {
            x
            for e in self.edges
            if e.confidence >= min_confidence
            for x in (e.subject, e.object)
        }
        keep.add(self.metadata.get("query", ""))
        self.edges = [e for e in self.edges if e.confidence >= min_confidence]
        self.nodes = {k: v for k, v in self.nodes.items() if k in keep}
        return self

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def from_json(cls, path: str | Path) -> "KnowledgeGraph":
        return cls.model_validate(json.loads(Path(path).read_text()))
