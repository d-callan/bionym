from bionym.evidence import Evidence
from bionym.graph import Edge, KnowledgeGraph, Node, NodeType


def test_add_node_merges_attrs():
    g = KnowledgeGraph()
    g.add_node(Node(id="g1", type=NodeType.GENE, label="x", attrs={"a": 1}))
    g.add_node(Node(id="g1", type=NodeType.GENE, attrs={"b": 2}))
    assert g.nodes["g1"].attrs == {"a": 1, "b": 2}
    assert g.nodes["g1"].label == "x"


def test_edge_roundtrip(tmp_path):
    g = KnowledgeGraph()
    g.add_node(Node(id="g1", type=NodeType.GENE))
    g.add_edge(
        Edge(
            subject="g1",
            predicate="is_a",
            object="idtype:ncbi_gene",
            confidence=0.9,
            evidence=[Evidence(source="regex", endpoint="test", summary="t")],
        )
    )
    p = tmp_path / "g.json"
    g.to_json(p)
    g2 = KnowledgeGraph.from_json(p)
    assert g2.edges[0].confidence == 0.9
    assert g2.edges[0].evidence[0].source == "regex"


def test_edge_confidence_bounds():
    import pytest
    from pydantic import ValidationError

    for bad in (-0.1, 1.01, 1.5):
        with pytest.raises(ValidationError):
            Edge(subject="a", predicate="p", object="b", confidence=bad)
    # bounds are inclusive
    for ok in (0.0, 0.5, 1.0):
        Edge(subject="a", predicate="p", object="b", confidence=ok)


def test_add_edge_merges_duplicate_claim():
    g = KnowledgeGraph()
    for nid in ("g1", "p1"):
        g.add_node(Node(id=nid, type=NodeType.GENE))
    g.add_edge(
        Edge(subject="g1", predicate="cited_in", object="p1",
             confidence=0.9,
             evidence=[Evidence(source="a", endpoint="e1", summary="s1")])
    )
    merged = g.add_edge(
        Edge(subject="g1", predicate="cited_in", object="p1",
             confidence=1.0,
             evidence=[Evidence(source="b", endpoint="e2", summary="s2")])
    )
    assert len(g.edges) == 1
    assert merged.confidence == 1.0
    assert {ev.source for ev in merged.evidence} == {"a", "b"}
    # exact-duplicate evidence doesn't accumulate
    g.add_edge(
        Edge(subject="g1", predicate="cited_in", object="p1",
             confidence=1.0,
             evidence=[Evidence(source="b", endpoint="e2", summary="s2")])
    )
    assert len(merged.evidence) == 2
