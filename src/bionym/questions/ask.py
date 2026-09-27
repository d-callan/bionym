"""Ask-the-graph — free-form Q&A over a resolved knowledge graph.

Two-step: JEV first classifies the question into graph categories
(noul per category — multi-select, since e.g. "which assemblies have
orthologs" is assemblies + orthology). Only the matching slice goes to
the LLM and the JEV judge — whole-graph payloads blow JEV's context.

No category above threshold = `other` (rare): the answer is returned
unscored with a warning, since judging requires a bounded projection.
"""

from __future__ import annotations

from typing import Any

# JEV's state embeds the graph slice inline — same context limit the
# chunked asks exist for. The judge sees up to this many edges, strongest
# first; edges_total tells it how much was withheld.
MAX_EDGES = 300

# Category -> what in the graph answers it. Vocabulary comes straight
# from the predicates/node types the resolver emits — the taxonomy of
# what the pipeline knows is closed under these.
CATEGORIES: dict[str, dict[str, Any]] = {
    "identity": {
        "desc": "what the identifier resolved to — gene symbols, aliases, "
        "namespaces, descriptions, cross-references between records",
        "predicates": {"same_as", "has_alias", "is_a", "superseded_by"},
        "types": {"Gene", "Organism", "IdType"},
    },
    "products": {
        "desc": "the gene's transcripts, proteins, and genomic location",
        "predicates": {"has_transcript", "has_protein", "located_on"},
        "types": {"Transcript", "Protein"},
    },
    "orthology": {
        "desc": "orthologs in other species, orthogroup membership, and "
        "judged presence/absence in specific genomes",
        "predicates": {
            "ortholog_of",
            "in_orthogroup",
            "has_member",
            "absent_in",
            "ortholog_present",
            "likely_present",
            "has_candidate_gene",
        },
        "types": {"OrthologGroup", "Gene", "Organism"},
    },
    "assemblies": {
        "desc": "genome assemblies — which contain the gene or its "
        "orthologs, and related/superseded assemblies",
        "predicates": {
            "annotated_in",
            "in_assembly",
            "related_assembly",
            "ortholog_present",
            "likely_present",
        },
        "types": {"Assembly"},
    },
    "function": {
        "desc": "functional annotation — GO terms, protein domains, and "
        "pathway membership",
        "predicates": {"has_go_term", "has_domain", "in_pathway"},
        "types": {"GOTerm", "Domain", "Pathway"},
    },
    "datasets": {
        "desc": "datasets associated with the gene — expression, "
        "phenotypic, variant, or other measurements — and their "
        "experimental conditions",
        "predicates": {"measured_in", "has_factor"},
        "types": {"Dataset", "Condition"},
    },
    "literature": {
        "desc": "publications linked to the gene",
        "predicates": {"cited_in"},
        "types": {"Publication"},
    },
    "claims": {
        "desc": "AI-proposed claims about the gene or its datasets "
        "and publications, or the generated summary (full analysis "
        "only) — anything LLM-proposed and JEV-verified",
        # `reports` attaches claims to whatever they were proposed
        # about (gene/dataset/publication); `has_condition` attaches
        # proposed conditions to datasets. Including the predicates
        # pulls the claim's subject node into the slice as an
        # endpoint — without them a claims-only question sees orphan
        # Claim nodes and zero edges.
        "predicates": {"reports", "has_condition"},
        "types": {"Claim", "Condition"},
    },
}


def _projection(nodes: dict, edges: list) -> dict[str, Any]:
    edges = sorted(edges, key=lambda e: e[3], reverse=True)
    return {
        "nodes": nodes,
        "edges": edges[:MAX_EDGES],
        "edges_total": len(edges),
    }


def _node_dict(n) -> dict[str, Any]:
    return {
        "type": n.type.value if hasattr(n.type, "value") else n.type,
        "label": n.label,
        **({"organism": n.attrs["organism"]} if n.attrs.get("organism") else {}),
    }


def compact_graph(graph: Any) -> dict[str, Any]:
    """Minimal {nodes, edges} projection for prompt/state embedding."""
    return _projection(
        {n.id: _node_dict(n) for n in graph.nodes.values()},
        [
            [e.subject, e.predicate, e.object, round(e.confidence, 3)]
            for e in graph.edges
        ],
    )


def slice_graph(graph: Any, categories: list[str]) -> dict[str, Any]:
    """Projection of only the subgraph the classified categories cover.

    Uniform rule: an edge is in the slice if its predicate belongs to a
    selected category, or both endpoints survive node selection. Nodes
    survive by category node types, as endpoints of predicate-selected
    edges, or as the query gene itself (always included for grounding).
    """
    preds, types = set(), set()
    for c in categories:
        preds |= CATEGORIES[c]["predicates"]
        types |= CATEGORIES[c]["types"]
    pred_edges = [e for e in graph.edges if e.predicate in preds]
    keep = {
        n.id
        for n in graph.nodes.values()
        if (n.type.value if hasattr(n.type, "value") else n.type) in types
    }
    keep |= {x for e in pred_edges for x in (e.subject, e.object)}
    query = graph.metadata.get("query") or graph.metadata.get("input")
    if query and query in graph.nodes:
        keep.add(query)
    edges = [
        e
        for e in graph.edges
        if e.predicate in preds or (e.subject in keep and e.object in keep)
    ]
    return _projection(
        {i: _node_dict(graph.nodes[i]) for i in keep if i in graph.nodes},
        [
            [e.subject, e.predicate, e.object, round(e.confidence, 3)]
            for e in edges
        ],
    )


def build_classify_state(question: str) -> dict:
    return {
        "question": question,
        "categories": {k: v["desc"] for k, v in CATEGORIES.items()},
    }


def build_classify_questions() -> dict:
    return {
        f"cat_{k}": {
            "type": "noul",
            "instructions": (
                f"Does answering `question` require information of this "
                f"kind: {v['desc']}? Score high if yes, low if unrelated."
            ),
        }
        for k, v in CATEGORIES.items()
    }


def build_answer_messages(
    graph_json: str, question: str, excerpt: bool = False
) -> tuple[str, str]:
    """(system, user) for LlmClient.complete.

    `excerpt` tells the model it's seeing a category-filtered slice, so
    it won't infer "not in graph" for material the filter removed.
    """
    scope = (
        "a filtered excerpt of the knowledge graph — it contains only the "
        "edges relevant to the question's categories, so don't assume "
        "absent material doesn't exist in the full graph"
        if excerpt
        else "the knowledge graph"
    )
    system = (
        f"You answer questions about a resolved identifier using only "
        f"{scope} in the user message — never outside knowledge. "
        "The graph maps the identifier to gene records, assemblies, "
        "orthologs, annotations, and expression datasets; every edge has a "
        "confidence score and provenance. If the graph doesn't contain the "
        "answer, say so. Respond as JSON: "
        '{"answer": "<plain prose>", "cited_nodes": ["<node id>", ...]}'
    )
    user = (
        "Knowledge graph (JSON):\n" f"{graph_json}\n\n" f"Question: {question}"
    )
    return system, user


def build_state(
    question: str, answer: str, projection: dict, categories: list[str]
) -> dict:
    return {
        "question": question,
        "answer": answer,
        "judged_categories": categories,
        "graph": projection,
    }


def build_questions() -> dict:
    return {
        "accuracy": {
            "type": "noul",
            "instructions": (
                "How accurate is `answer` relative to `graph`? Score high "
                "only if every claim is supported by a node or edge there; "
                "score low for claims absent from or contradicting the "
                "graph, and for invented node ids."
            ),
        },
        "completeness": {
            "type": "noul",
            "instructions": (
                "How completely does `answer` cover what `graph` contains "
                "relevant to `question`? `graph` is a category-filtered "
                "slice (see `judged_categories`) — judge completeness only "
                "within the slice, not the whole resolved graph. `edges` "
                f"is capped at the top {MAX_EDGES} by confidence "
                "(edges_total tells you the full size) — don't penalize "
                "material that falls below the cap."
            ),
        },
    }
