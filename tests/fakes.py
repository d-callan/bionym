"""Shared test doubles for ask_graph and backend tests.

Not collected (no test_ prefix) — import as `from fakes import ...`
since pytest prepends the tests/ dir to sys.path.
"""
import json

from bionym.graph import Edge, KnowledgeGraph, Node, NodeType
from bionym.jev import JevError


class FakeJev:
    """Canned Jev answers: high noul on selected classify categories,
    fixed nouls for scoring questions, optional failure."""

    def __init__(self, cats=None, score=0.9, fail_scoring=False):
        self._cats = set(cats or [])
        self._score = score
        self._fail = fail_scoring
        self.stages: list[str] = []

    def ask(self, state, questions, stage=""):
        self.stages.append(stage)
        if stage == "ask_classify":
            return {
                qid: {
                    "type": "noul",
                    "noul": 0.9
                    if qid.removeprefix("cat_") in self._cats
                    else 0.0,
                }
                for qid in questions
            }
        if self._fail:
            raise JevError("judge down")
        return {
            qid: {"type": "noul", "noul": self._score}
            for qid in questions
        }


class FakeLlm:
    enabled = True

    def __init__(self, payload=None):
        self._payload = payload or {
            "answer": "it does X",
            "cited_nodes": [],
        }

    def complete(self, system, user):
        return json.dumps(self._payload)


class DisabledLlm:
    enabled = False


def mini_graph() -> KnowledgeGraph:
    """Gene + one dataset measuring it — enough for the datasets slice."""
    g = KnowledgeGraph(metadata={"input": "PF3D7_0710100"})
    g.add_node(
        Node(
            id="veupathdb:PF3D7_0710100",
            type=NodeType.GENE,
            label="k13",
            id_namespace="veupathdb",
        )
    )
    g.add_node(
        Node(
            id="ds:EXP001",
            type=NodeType.DATASET,
            label="rna-seq",
            id_namespace="veupathdb",
        )
    )
    g.add_edge(
        Edge(
            subject="ds:EXP001",
            predicate="measured_in",
            object="veupathdb:PF3D7_0710100",
            confidence=0.9,
        )
    )
    return g
