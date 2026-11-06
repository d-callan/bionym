"""Unit tests for bionym.proposals parsing."""

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
