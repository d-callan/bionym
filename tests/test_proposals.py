"""Unit tests for bionym.proposals parsing."""

import logging

from bionym import proposals


def test_parse_tolerates_misspelled_proposals_key():
    # Observed in the wild: the model emits "propososals"/"propositals".
    # The parser accepts keys sharing the expected prefix rather than
    # dropping the whole batch.
    content = (
        '{"propososals": [{"claim": "BRCA1 is a tumor suppressor", '
        '"citations": [2]}]}'
    )
    out = proposals.parse(content, units=["u1", "BRCA1 unit", "u3"])
    assert out == [
        {
            "claim": "BRCA1 is a tumor suppressor",
            "type": "claim",
            "quote": "BRCA1 unit",
            "citations": [1],
        }
    ]


def test_parse_quote_is_verbatim_from_cited_units():
    # issue #8: quotes are assembled verbatim from cited units — the
    # model can only name unit numbers, never reword the source
    text = (
        "Apical membrane antigen-1 (AMA1) is a merozoite protein. "
        "It is essential for invasion."
    )
    display, units = proposals.numbered_sentences(text)
    assert "[1] Apical membrane antigen-1 (AMA1)" in display
    content = (
        '{"proposals": [{"claim": "AMA1 is a merozoite protein", '
        '"citations": [1]}]}'
    )
    out = proposals.parse(content, units)
    assert out[0]["quote"] == units[0]
    # adjacent units join as one span; disjoint units get an ellipsis so
    # they can't read as contiguous source text
    content = (
        '{"proposals": [{"claim": "c", "citations": [1, 2], '
        '"quote": "fabricated"}]}'
    )
    out = proposals.parse(content, units)
    assert out[0]["quote"] == f"{units[0]} {units[1]}"
    # citation order follows the source, not the model — [2, 1] emits
    # unit 1 then unit 2 (PR #10)
    content = (
        '{"proposals": [{"claim": "c", "citations": [2, 1]}]}'
    )
    out = proposals.parse(content, units)
    assert out[0]["quote"] == f"{units[0]} {units[1]}"
    # a repeated citation quotes the unit once, in source position
    content = (
        '{"proposals": [{"claim": "c", "citations": [1, 2, 1]}]}'
    )
    out = proposals.parse(content, units)
    assert out[0]["quote"] == f"{units[0]} {units[1]}"
    # improvised nested groups flatten — same claim either way
    content = (
        '{"proposals": [{"claim": "c", "citations": [[1, 2]]}]}'
    )
    out = proposals.parse(content, units)
    assert out[0]["quote"] == f"{units[0]} {units[1]}"


def test_numbered_sentences_keeps_abbreviations_whole():
    # periods that aren't sentence ends (initials, 'e.g.', binomials)
    # must not fragment a sentence into unusable cite units
    _, units = proposals.numbered_sentences(
        "We sampled P. falciparum isolates. Each was sequenced."
    )
    assert units == [
        "We sampled P. falciparum isolates.",
        "Each was sequenced.",
    ]
    _, units = proposals.numbered_sentences(
        "Targets, e.g. proteases, were ranked. Five were tested."
    )
    assert units[0] == "Targets, e.g. proteases, were ranked."
    assert units[1] == "Five were tested."


def test_parse_drops_claims_citing_nonexistent_units():
    units = ["u1", "u2"]
    # every cited index out of range -> claim dropped
    assert proposals.parse(
        '{"proposals": [{"claim": "c", "citations": [9]}]}', units
    ) == []
    # partial overlap keeps the claim on the valid subset
    out = proposals.parse(
        '{"proposals": [{"claim": "c", "citations": [2, 99]}]}', units
    )
    assert out[0]["quote"] == "u2"
    # no citation field -> dropped like an empty list: citations are
    # required (PR #10) — an unquoted claim can't be verbatim-verified
    assert proposals.parse('{"proposals": [{"claim": "c"}]}', units) == []
    assert proposals.parse(
        '{"proposals": [{"claim": "c", "citations": []}]}', units
    ) == []


def test_parse_normalized_tolerates_misspelled_key():
    content = '{"normalised": ["time series post infection", "heat shock"]}'
    out = proposals.parse_normalized(content, 2)
    assert out == ["time series post infection", "heat shock"]


def test_parse_type_field():
    # 'type' survives when valid, falls back to the caller's default
    # when missing or invalid — sources that only emit one kind don't
    # send the field at all
    units = ["u1", "u2", "u3"]
    content = (
        '{"proposals": ['
        '{"claim": "heat shock", "type": "condition", "citations": [1]},'
        ' {"claim": "essential", "citations": [2]},'
        ' {"claim": "weird", "type": "bogus", "citations": [3]}]}'
    )
    out = proposals.parse(content, units)
    assert [p["type"] for p in out] == ["condition", "claim", "claim"]
    out = proposals.parse(content, units, default="condition")
    assert [p["type"] for p in out] == ["condition"] * 3


def test_build_questions_split_by_type():
    # verify wording branches on the proposal's output type, not the
    # source's kind — a publication condition is still a condition
    props = [
        {"node": "x", "type": "condition", "claim": "heat shock"},
        {"node": "y", "type": "claim", "claim": "essential"},
    ]
    q = proposals.build_questions(props)
    cond, claim = q["prop_0"]["instructions"], q["prop_1"]["instructions"]
    assert "biologically informative" in cond
    assert "falsifiable" not in cond
    assert "falsifiable assertion" in claim
    assert "source_gene" in cond and "source_gene" in claim


def test_build_state_has_gene_and_type():
    # JEV judges gene relevance — the state must carry the gene and
    # each proposal's type
    state = proposals.build_state(
        {"n": "text"},
        [{"node": "n", "type": "claim", "claim": "c", "quote": "q"}],
        {"symbol": "BRCA1"},
    )
    assert state["source_gene"]["symbol"] == "BRCA1"
    assert state["proposals"][0]["type"] == "claim"
def test_parse_categories(tmp_path, caplog):
    assert proposals.parse_categories("a, b ,c") == {"a": "", "b": "", "c": ""}
    assert proposals.parse_categories("") == {}
    f = tmp_path / "cats.txt"
    f.write_text("# a comment\nmolecular function\n\nexpression\n")
    assert proposals.parse_categories(str(f)) == {
        "molecular function": "",
        "expression": "",
    }
    # a spec that isn't a file falls back to comma parsing, even if it
    # looks like a path — but a lone dotted token warns, since it's
    # almost always a mistyped filename
    with caplog.at_level(logging.WARNING, logger="bionym.proposals"):
        assert proposals.parse_categories("no_such_file.txt") == {
            "no_such_file.txt": ""
        }
    assert "no such file" in caplog.text


def test_build_category_questions():
    # claim-typed proposals get a cat_i choice question; condition-
    # typed ones don't — they're self-describing
    props = [
        {"node": "pubmed:1", "type": "claim", "claim": "a"},
        {"node": "ds:1", "type": "condition", "claim": "b"},
    ]
    q = proposals.build_category_questions(
        props, {"molecular function": "what it does"}
    )
    assert list(q) == ["cat_0"]
    assert q["cat_0"]["type"] == "choice"
    assert set(q["cat_0"]["criteria"]) == {"molecular function", "other"}
    # a caller-supplied 'other' keeps its own description
    q = proposals.build_category_questions(props, {"other": "custom"})
    assert list(q["cat_0"]["criteria"]) == ["other"]
    assert q["cat_0"]["criteria"]["other"] == "custom"
