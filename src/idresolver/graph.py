"""Knowledge graph model: typed nodes, scored edges, evidence provenance."""

from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .evidence import Evidence


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
    ID_TYPE = "IdType"


class Node(BaseModel):
    id: str
    type: NodeType
    label: str = ""
    id_namespace: str = ""  # e.g. "ncbi_gene", "veupathdb", "uniprot"
    attrs: dict[str, Any] = Field(default_factory=dict)


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

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(self.model_dump_json(indent=2) + "\n")

    @classmethod
    def from_json(cls, path: str | Path) -> "KnowledgeGraph":
        return cls.model_validate(json.loads(Path(path).read_text()))
