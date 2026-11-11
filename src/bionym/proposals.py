"""LLM proposal prompts + strict parsing.

The LLM's only job is enumeration: given a dataset description or a
publication title, list the claims the text could support. Output is a
fixed JSON shape so it can be worked with programmatically — anything
malformed is dropped, and JEV scores each surviving proposal against the
source text before it becomes an edge.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any

log = logging.getLogger(__name__)

# Per-document bound lives in the prompt, not a post-hoc cap: the model
# can always generate more, so recall is a prompt-quality problem.
MAX_PROPOSALS = 12

SYSTEM = (
    "You enumerate candidate claims from biomedical text for downstream "
    "verification. Output JSON only, no commentary."
)

# A source can yield two output types in one ask: conditions are
# noun-phrase labels for what the gene was measured under (they
# normalize into shared condition: nodes), claims are falsifiable
# assertions about the gene. The verify wording branches on this.
_TYPE_RULES = (
    "Two proposal types: 'condition' — a short noun phrase naming an "
    "experimental condition the gene was measured under, a label not "
    "a sentence ('intraerythrocytic time course'); and 'claim' — a "
    "falsifiable assertion about the gene that could be true or "
    "false ('essential for erythrocyte invasion')."
)

_INSTRUCTIONS = {
    "dataset": (
        "This text describes a dataset the gene was measured in. List "
        "every distinct experimental condition, contrast, or sample "
        "group it explicitly states or clearly implies (type "
        "'condition'), plus any distinct assertion about the gene it "
        "makes (type 'claim'). Rules: only conditions relevant to the "
        "gene measured and biologically informative — never the "
        "organism itself (already implied by context), never technical "
        "artifacts (read orientation, unique reads, sequencing metrics, "
        "replicate counts). Aggregate series into one condition: 'time "
        "series post infection' not '7 hpi' and '16 hpi' separately; "
        "'life cycle stages' not individual stages. Strain/sample "
        "counts are fine ('5 strains investigated')."
    ),
    "publication": (
        "This text is a publication's title and abstract. List every "
        "distinct falsifiable assertion it explicitly states or clearly "
        "implies about the gene, its function, or its organism (type "
        "'claim'), plus any experimental condition the gene was "
        "measured under (type 'condition')."
    ),
}

_SCHEMA = (
    'Return JSON: {"proposals": [{"claim": "<condition label or '
    'assertion text>", "type": "condition|claim", '
    '"citations": [<numbers of the units above that support it>]}]}. '
    f"At most {MAX_PROPOSALS} proposals. Rules: every proposal must be "
    "checkable against the text alone — no outside knowledge, no "
    "speculation, no duplicates. Cite unit numbers only, never rewrite "
    "or splice the source's words."
)


# Sentences (prose sources) or lines (rows, serialized edges) are the
# citable units: the model names unit numbers and the quote is copied
# verbatim from those units, so a quote can never drift from its
# source (issue #8). The split skips boundaries before a lowercase
# token so initials, 'e.g.'/'i.e.', and binomials ('P. falciparum')
# stay one unit — verbatim correctness doesn't need perfect
# boundaries, but fragment units make useless quotes. Residual: an
# abbreviation followed by a capitalized word ('et al. Johnson')
# still splits.
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[^\sa-z])|\n+")


def _numbered(units: list[str]) -> tuple[str, list[str]]:
    display = "\n".join(f"[{i}] {u}" for i, u in enumerate(units, 1))
    return display, units


def numbered_sentences(text: str) -> tuple[str, list[str]]:
    """Prose -> ('[1] s1\\n[2] s2', [s1, s2]). Source for text claims."""
    return _numbered([s.strip() for s in _SENT_SPLIT.split(text) if s.strip()])


def numbered_lines(text: str) -> tuple[str, list[str]]:
    """Line-structured source (data rows, serialized graph) -> same."""
    return _numbered([ln for ln in text.splitlines() if ln.strip()])


def _valid_citations(raw: Any, n_units: int) -> list[int] | None:
    """Citation field -> 0-based unit indices that exist.

    None = no citation field at all; [] = the list was empty or every
    index missed the unit range. parse() drops the claim either way —
    citations are required (PR #10). Nested groups like [[1,2],5] are
    flattened — they carry no information the indices don't (adjacency
    is derivable), and small models may improvise them."""
    if raw is None:
        return None
    if not isinstance(raw, list):
        return []
    out = []
    for n in raw:
        if isinstance(n, list):  # flatten improvised grouping
            for m in n:
                try:
                    i = int(m) - 1
                except (TypeError, ValueError):
                    continue
                if 0 <= i < n_units:
                    out.append(i)
            continue
        try:
            i = int(n) - 1  # the display numbers are 1-based
        except (TypeError, ValueError):
            continue
        if 0 <= i < n_units:
            out.append(i)
    return out


def _verbatim_quote(units: list[str], citations: list[int]) -> str:
    """Assemble the quote from cited units in SOURCE order: sorted and
    deduplicated, so '[3, 2]' or '[1, 2, 1]' can't read as reordered or
    repeated source text (PR #10). Adjacent units join as one span; a
    gap inserts ' … ' so disjoint sentences never read as contiguous
    source text."""
    quote = ""
    prev = None
    for i in sorted(set(citations)):
        if prev is not None:
            quote += " " if i == prev + 1 else " … "
        quote += units[i]
        prev = i
    return quote


def build_prompt(
    kind: str, text: str, gene: str = ""
) -> tuple[str, str]:
    """(system, user) messages for one enumeration call."""
    user = (
        f"Gene: {gene or 'the source gene'}\n\n"
        f"{_TYPE_RULES}\n\n{_INSTRUCTIONS[kind]}\n\n{_SCHEMA}\n\n"
        f"Text (numbered units):\n---\n{text}\n---"
    )
    return SYSTEM, user


def _extract_json(content: str, key: str = "proposals") -> dict | None:
    """Find the first decodable JSON object holding a `key` list.

    Models sometimes wrap the payload in stray braces, markdown fences, or
    preamble text (observed: GLM emitting `{"{"proposals": ...}`), so a
    strict json.loads on the whole string is too brittle.
    """
    decoder = json.JSONDecoder()
    for i, ch in enumerate(content):
        if ch != "{":
            continue
        try:
            data, _ = decoder.raw_decode(content, i)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        if isinstance(data.get(key), list):
            return data
        # The model sometimes misspells the payload key (observed:
        # "propososals", "propositals") — accept a key sharing the
        # expected prefix and re-key it so callers see the canonical
        # shape.
        for k, v in data.items():
            if k.startswith(key[:6]) and isinstance(v, list):
                return {key: v}
    return None


def parse(
    content: str, units: list[str], default: str = "claim"
) -> list[dict[str, Any]]:
    """Validate the JSON response -> [{claim, type, quote, citations}].

    `units` is the numbered source the model cited: the quote is
    assembled verbatim from the cited units, so it always appears word
    for word in the source. A freeform 'quote' field is ignored — it
    can't be verified as verbatim. Citations are required: a missing
    field, an empty list, or indices all out of range all drop the
    claim — an unquoted claim can't be shown as verbatim-verified
    (PR #10). units is required — without it every claim would be
    dropped silently. `type` is 'condition'|'claim'; a missing or
    invalid value falls back to `default` — sources that only ever
    produce one type (data rows, summaries) don't emit the field."""
    data = _extract_json(content or "")
    if data is None:
        log.warning("LLM returned no parseable proposals object: %.200s", content)
        return []
    proposals = data["proposals"]
    out = []
    for p in proposals[:MAX_PROPOSALS]:
        if not (
            isinstance(p, dict)
            and isinstance(p.get("claim"), str)
            and p["claim"].strip()
        ):
            continue
        citations = _valid_citations(p.get("citations"), len(units))
        if not citations:
            continue
        out.append(
            {
                "claim": " ".join(p["claim"].split()),
                "type": p.get("type")
                if p.get("type") in ("condition", "claim")
                else default,
                "quote": _verbatim_quote(units, citations),
                "citations": citations,
            }
        )
    return out


# Triage: one JEV call scores every candidate node's likely value before
# any LLM calls are spent. Texts are truncated — triage only needs the
# gist, and full summaries for ~40 datasets overflow JEV's context.
TRIAGE_LEVELS = ["irrelevant", "unlikely", "likely", "very_likely"]
TRIAGE_TEXT_CHARS = 600
# Bound on the text sent to the LLM itself (summaries can be huge).
PROMPT_TEXT_CHARS = 8000
# Cap on datasets/publications that survive triage — each one costs an
# LLM call + JEV verification, and a deep graph can yield hundreds.
MAX_ITEMS_PER_KIND = 100
# Every triage item is embedded in JEV's state, so a single ask over a
# big pool overflows the context (BRCA1 → ~7.7k publications: recursive
# question-splitting can't shrink the state, only multiply it). Asks are
# chunked, and the pool is pre-capped per kind — TRIAGE_PER_KIND keeps a
# 4x headroom over the post-triage survivor cap so the judge still has
# selection room.
TRIAGE_CHUNK = 50
TRIAGE_PER_KIND = MAX_ITEMS_PER_KIND * 4


def _gene_ctx(match: dict[str, Any]) -> dict[str, Any]:
    """The gene fields a verifier needs to judge claim relevance."""
    return {
        "symbol": match.get("symbol"),
        "description": match.get("description"),
        "organism": match.get("organism"),
    }


def build_triage_state(
    source_gene: dict[str, Any], items: list[dict[str, Any]]
) -> dict:
    return {
        "source_gene": _gene_ctx(source_gene),
        "items": [
            {"node": it["node"], "kind": it["kind"], "text": it["text"]}
            for it in items
        ],
    }


# Triage criteria differ by kind: a dataset's value is that the gene was
# *measured* under its conditions (the measured_in edge already links it),
# so the question is whether the text names extractable conditions — not
# whether it discusses the gene. A publication's value is whether it
# actually discusses the gene.
_TRIAGE_INSTRUCTIONS = {
    "dataset": (
        "Does `items[{i}]`'s text describe experimental conditions, "
        "contrasts, or sample groups (treatments, doses, timepoints, "
        "life-cycle stages, strains, tissues) that could be enumerated? "
        "Score by the criteria levels."
    ),
    "publication": (
        "How likely is `items[{i}]`'s text to contain information about "
        "what `source_gene` does, its function, or the conditions under "
        "which it is expressed? Score by the criteria levels."
    ),
}


def build_triage_questions(items: list[dict[str, Any]]) -> dict:
    return {
        f"triage_{i}": {
            "type": "score",
            "instructions": _TRIAGE_INSTRUCTIONS[it["kind"]].format(i=i),
            "criteria": TRIAGE_LEVELS,
        }
        for i, it in enumerate(items)
    }


def build_state(
    texts: dict[str, str],
    proposals: list[dict[str, Any]],
    match: dict[str, Any] | None = None,
) -> dict:
    """JEV state: gene context + source texts + proposals to verify."""
    return {
        "source_gene": _gene_ctx(match or {}),
        "source_texts": texts,
        "proposals": [
            {
                "node": p["node"],
                "type": p.get("type"),
                "claim": p["claim"],
                "quote": p.get("quote"),
            }
            for p in proposals
        ],
    }


# Verify wording differs by output type: conditions are labels judged on
# biological informativeness; claims stay assertions, so the bar adds
# falsifiability. Both must be relevant to `source_gene` — textual
# support alone let trivia like 'duplicate samples' through (verbatim
# support doesn't imply a claim worth graphing).
_VERIFY_INSTRUCTIONS = {
    "condition": (
        "`proposals[{i}]` is a condition machine-extracted from "
        "`source_texts` for node '{node}'. Accept it only if the source "
        "text states or clearly implies it, it is biologically "
        "informative about how `source_gene` was measured — not a "
        "technical artifact, metadata, or the organism itself — and "
        "it is relevant to `source_gene`."
    ),
    "claim": (
        "`proposals[{i}]` was machine-extracted from `source_texts` for "
        "node '{node}'. Accept it only if the source text supports it "
        "— judge textual support, not plausibility — and it is a "
        "falsifiable assertion relevant to `source_gene`, not a bare "
        "label or fragment."
    ),
}


def build_questions(proposals: list[dict[str, Any]]) -> dict:
    return {
        f"prop_{i}": {
            "type": "noul",
            "instructions": _VERIFY_INSTRUCTIONS.get(
                p.get("type"), _VERIFY_INSTRUCTIONS["claim"]
            ).format(i=i, node=p["node"]),
        }
        for i, p in enumerate(proposals)
    }


# -- claim categorization ---------------------------------------------------
#
# Text-path claims get a JEV-assigned category on top of their generated
# label, drawn from a caller-supplied vocabulary. 'other' is always
# offered so the judge is never forced into a wrong bucket.
OTHER_CATEGORY = "other"

# Used when the caller doesn't supply a vocabulary. Descriptions are
# judge hints, not user-facing text — near-neighbor pairs (drug
# resistance/sensitivity/target; interactions/host interactions) need
# the hint to be told apart.
DEFAULT_CLAIM_CATEGORIES: dict[str, str] = {
    "molecular function": "biochemical activity of the gene product",
    "structure and domains": "protein structure, domains, motifs",
    "modification and processing":
        "post-translational modifications, cleavage, folding",
    "interactions and complexes":
        "binding partners or complex membership within the organism",
    "location in the cell": "subcellular localization",
    "expression by stage or condition":
        "where, when or how strongly it is expressed",
    "how the gene is regulated":
        "upstream regulators, promoters, feedback on this gene",
    "biological process or pathway":
        "pathway or process the gene participates in",
    "effects on other genes":
        "downstream effects on other genes' expression or products",
    "essentiality and mutant phenotype":
        "fitness or phenotype of knockout/mutant",
    "life-cycle role": "role in developmental or life-cycle stages",
    "virulence and pathogenesis":
        "contribution to infectivity or disease severity",
    "interaction with the host":
        "host factors engaged or host responses modulated",
    "antigen and vaccine": "antigenicity, epitopes, vaccine candidacy",
    "drug resistance": "established resistance mechanisms or mutations",
    "drug target and mode of action":
        "the gene is a drug target; how a drug acts on it",
    "changed drug sensitivity":
        "perturbations altering drug response without established "
        "resistance",
    "variants and where they occur":
        "SNPs, indels, population-level variation",
    "gene family and evolution":
        "paralogs, orthologs, family membership, phylogenetics",
    "diagnostic or surveillance marker":
        "used as a detection, typing or epidemiological marker",
    "reagents and tools":
        "experimental methods, reagents, tools (tagging, transfection)",
    "naming and annotation history":
        "synonyms, prior annotation, naming conventions",
}


def parse_categories(spec: str) -> dict[str, str]:
    """Category spec -> {name: ""}. A spec naming an existing file is
    read one category per line (blanks and # comments skipped);
    anything else is parsed as a comma-separated list. Empty -> {}."""
    spec = spec.strip()
    if not spec:
        return {}
    path = Path(spec).expanduser()
    if path.is_file():
        return {
            line.strip(): ""
            for line in path.read_text().splitlines()
            if line.strip() and not line.startswith("#")
        }
    if "," not in spec and "." in spec:
        # a lone dotted token is almost always a mistyped file path
        log.warning(
            "claim category spec %r looks like a path but no such file "
            "exists; treating it as a category name",
            spec,
        )
    return {c.strip(): "" for c in spec.split(",") if c.strip()}


def build_category_questions(
    proposals: list[dict[str, Any]], categories: dict[str, str]
) -> dict:
    """cat_{i} choice questions classifying each claim proposal.

    Only proposals tagged type="claim" get categorized — conditions are
    self-describing. (claim-type-split formalizes this same type field
    upstream.)"""
    criteria = dict(categories)
    criteria.setdefault(
        OTHER_CATEGORY, "none of the categories describe this claim"
    )
    return {
        f"cat_{i}": {
            "type": "choice",
            "instructions": (
                f"`proposals[{i}]` is a claim machine-extracted from "
                f"`source_texts` for node '{p['node']}'. Pick the "
                "category that best describes the kind of knowledge it "
                f"asserts; pick '{OTHER_CATEGORY}' if none fit."
            ),
            "criteria": criteria,
        }
        for i, p in enumerate(proposals)
        if p.get("type") == "claim"
    }


# -- condition normalization ------------------------------------------------

_NORMALIZE_SYSTEM = (
    "You normalize biomedical condition labels for consistency. "
    "Output JSON only, no commentary."
)


def build_normalize_prompt(claims: list[str]) -> tuple[str, str]:
    """(system, user) to map each claim to a canonical condition label."""
    user = (
        "Below is a list of experimental conditions extracted from dataset "
        "descriptions. Rewrite each as a short canonical label so that "
        "equivalent conditions share identical wording (e.g. 'time series "
        "post infection' and 'post-infection timeseries' -> 'time series "
        "post infection'). Keep each label faithful to its original — "
        "normalize wording, not meaning. Return JSON: {\"normalized\": "
        "[\"<label>\", ...]} with exactly one label per input, same order."
        f"\n\nConditions:\n{json.dumps(claims)}"
    )
    return _NORMALIZE_SYSTEM, user


def parse_normalized(content: str, n: int) -> list[str] | None:
    """Parse the normalization response -> list of n labels, or None."""
    data = _extract_json(content or "", key="normalized")
    if data is None:
        return None
    labels = [
        " ".join(str(x).split()) if str(x).strip() else None
        for x in data["normalized"][:n]
    ]
    if len(labels) != n or any(l is None for l in labels):
        return None
    return labels


# -- dataset data claims ----------------------------------------------------
#
# For datasets with fetched per-gene measurements (VEuPathDB
# ExpressionGraphsDataTable today), the same propose-then-score loop turns
# the numbers into claims: the LLM summarizes biologically relevant claims
# with the dataset's own metadata as context, JEV verifies each against
# the serialized data. The graph stays claim-shaped — no raw-data model.

# Rows sent to the LLM/JEV per dataset — enough to see the expression
# range without blowing the prompt.
DATA_ROWS_MAX = 60

_DATA_SYSTEM = (
    "You summarize per-gene measurements from a functional genomics "
    "dataset as biological claims. Output JSON only, no commentary."
)


def serialize_rows(
    rows: list[dict[str, Any]], max_rows: int = DATA_ROWS_MAX
) -> str:
    """Per-sample rows -> 'sample | <cols>' text, strongest signal first.

    Columns are whichever of value/percentile/p_value the rows carry —
    differential rows (GXA analytics) have p_value and no percentile.
    Ranking: percentile desc when present, else |value| desc so the
    biggest fold-changes lead."""

    def _rank(r: dict[str, Any]) -> float:
        try:
            if r.get("percentile") is not None:
                return float(r["percentile"])
            return abs(float(r.get("value") or 0))
        except (TypeError, ValueError):
            return 0.0

    ordered = sorted(rows, key=_rank, reverse=True)
    keys = [
        k
        for k in ("value", "percentile", "p_value")
        if any(r.get(k) is not None for r in ordered)
    ]
    lines = ["sample | " + " | ".join(keys)]
    for r in ordered[:max_rows]:
        lines.append(
            str(r.get("sample"))
            + " | "
            + " | ".join(str(r.get(k)) for k in keys)
        )
    if len(ordered) > max_rows:
        lines.append(f"... {len(ordered) - max_rows} more rows")
    return "\n".join(lines)


def build_data_prompt(
    meta: dict[str, Any], data_text: str
) -> tuple[str, str]:
    """(system, user): propose biological claims from per-gene data.

    The dataset's own metadata is included — raw values are
    uninterpretable without knowing what the contrast is.
    """
    gene = meta.get("gene") or "the gene"
    user = (
        f"Below is a dataset's metadata and {gene}'s per-sample "
        "measurements in it, one numbered row per sample. Propose "
        "concise falsifiable biological claims about "
        f"{gene} that the data "
        "supports — e.g. conditions or stages where it is highly or "
        "lowly expressed, or notable contrasts. Each claim must cite "
        "the numbers of the supporting row(s) in 'citations'.\n\n"
        f"{_SCHEMA}\n\n"
        f"Dataset: {meta.get('label')}\n"
        f"Summary: {meta.get('summary') or 'n/a'}\n\n"
        f"Data:\n---\n{data_text}\n---"
    )
    return _DATA_SYSTEM, user


def build_data_state(
    meta: dict[str, Any], data_text: str, proposals: list[dict[str, Any]]
) -> dict:
    """JEV state: gene + dataset context + serialized data + proposals."""
    return {
        "gene": meta.get("gene"),
        "dataset": {
            "name": meta.get("label"),
            "summary": meta.get("summary"),
        },
        "data": data_text,
        "proposals": [
            {"node": p["node"], "claim": p["claim"], "quote": p.get("quote")}
            for p in proposals
        ],
    }


def build_data_questions(proposals: list[dict[str, Any]]) -> dict:
    return {
        f"prop_{i}": {
            "type": "noul",
            "instructions": (
                f"`proposals[{i}]` was machine-extracted from the "
                "per-sample measurements in `data`. Accept it only if "
                "the numbers support it — judge the values, not "
                "plausibility — and it is a falsifiable assertion "
                "about `gene`, not a bare label."
            ),
        }
        for i, p in enumerate(proposals)
    }


# -- summary ---------------------------------------------------------------

SUMMARY_EDGES_MAX = 80

_SUMMARY_SYSTEM = (
    "You write concise structured summaries of gene knowledge graphs. "
    "Output JSON only, no commentary."
)


def _node_label(graph: Any, node_id: str) -> str:
    n = graph.nodes.get(node_id)
    return (n.label or node_id) if n else node_id


def serialize_graph(graph: Any, max_edges: int = SUMMARY_EDGES_MAX) -> str:
    """Graph -> 'subject -predicate-> object (conf)' lines grouped by
    predicate, highest-confidence first. This is the 'source text' the
    summary LLM synthesizes from and JEV verifies against."""
    by_pred: dict[str, list[Any]] = {}
    for e in graph.edges:
        by_pred.setdefault(e.predicate, []).append(e)
    lines = []
    for pred in sorted(by_pred):
        edges = sorted(by_pred[pred], key=lambda e: e.confidence, reverse=True)
        lines.append(f"[{pred}]")
        for e in edges[:max_edges]:
            s = _node_label(graph, e.subject)
            o = _node_label(graph, e.object)
            lines.append(f"  {s} -{pred}-> {o} ({e.confidence:.2f})")
        if len(edges) > max_edges:
            lines.append(f"  ... {len(edges) - max_edges} more")
    return "\n".join(lines)


def build_summary_prompt(
    match: dict[str, Any], graph_text: str
) -> tuple[str, str]:
    """(system, user): propose typed summary claims over the subgraph."""
    gene = match.get("symbol") or match.get("locus_tag") or "this gene"
    organism = match.get("organism") or "unknown organism"
    user = (
        f"Gene: {gene} ({organism})\n\n"
        "Knowledge graph edges:\n"
        f"{graph_text}\n\n"
        "Propose a concise summary of this gene as typed claims: what it "
        "is, what it does, where/when it's expressed, what's known vs "
        "unknown. Each claim must be a falsifiable assertion about the "
        "gene, directly supported by the edges above. Gaps are valid "
        "claims ('no known function', 'no orthologs found').\n"
        'Return JSON: {"proposals": [{"claim": "...", "citations": '
        '[<numbers of the edges above that support it>]}]}'
    )
    return _SUMMARY_SYSTEM, user


def build_summary_state(
    match: dict[str, Any], graph_text: str, props: list[dict[str, Any]]
) -> dict:
    """JEV state: gene context + serialized graph + proposed claims."""
    return {
        "gene": match.get("symbol") or match.get("locus_tag"),
        "graph": graph_text,
        "proposals": [
            {"claim": p["claim"], "quote": p.get("quote")} for p in props
        ],
    }


def build_summary_questions(props: list[dict[str, Any]]) -> dict:
    return {
        f"prop_{i}": {
            "type": "noul",
            "instructions": (
                f"`proposals[{i}]` is a candidate summary claim about "
                "`gene`. Accept it only if it is supported by the edges "
                "in `graph`, is a falsifiable assertion about `gene` — "
                "not a bare label — and is important enough for a "
                "concise summary. Reject trivia."
            ),
        }
        for i, p in enumerate(props)
    }
