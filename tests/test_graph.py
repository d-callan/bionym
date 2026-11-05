import pytest

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


UNSAFE_URLS = [
    "javascript:alert(1)",
    " javascript:alert(1)",
    "JavaScript:alert(1)",
    "http://plasmodb.org/plasmo/app/record/gene/PF3D7_1343700",
    "data:text/html,<script>alert(1)</script>",
    "//plasmodb.org/x",
    "http://[broken",  # urlsplit raises on these: must not crash model_dump
    "https://example.org[1]/x",
    "https://",  # no host: would override a working template url
    "https:foo",
    "https://:443/path",  # netloc but no hostname
    "https://user@/path",
    "https://example.org:invalid/path",  # malformed port
    "https://example.org:99999/path",
    "",
    42,
]


@pytest.mark.parametrize("bad", UNSAFE_URLS)
@pytest.mark.parametrize(
    ("fields", "expected"),
    [
        # rejected url falls back to the namespace template...
        (dict(id="uniprot:Q8IDQ5", type=NodeType.GENE, id_namespace="uniprot"),
         "https://www.uniprot.org/uniprotkb/Q8IDQ5"),
        # ...or to None when the node has no template
        (dict(id="claim:1", type=NodeType.CLAIM), None),
    ],
)
def test_url_ignores_non_https_attr(bad, fields, expected):
    n = Node(**fields, attrs={"url": bad})
    assert n.url == expected
    assert n.model_dump()["url"] == expected


@pytest.mark.parametrize(
    "good",
    [
        "https://plasmodb.org/plasmo/app/record/gene/PF3D7_1343700",
        "HTTPS://plasmodb.org/plasmo/app/record/gene/PF3D7_1343700",
    ],
)
def test_url_keeps_https_attr(good):
    n = Node(id="uniprot:Q8IDQ5", type=NodeType.GENE, id_namespace="uniprot",
             attrs={"url": good})
    assert n.url == good
