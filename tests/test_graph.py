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
