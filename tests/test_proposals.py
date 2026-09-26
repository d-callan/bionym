"""Unit tests for bionym.proposals parsing."""

from bionym import proposals


def test_parse_tolerates_misspelled_proposals_key():
    # Observed in the wild: the model emits "propososals"/"propositals".
    # The parser accepts keys sharing the expected prefix rather than
    # dropping the whole batch.
    content = (
        '{"propososals": [{"claim": "BRCA1 is a tumor suppressor", '
        '"quote": "tumor suppressor BRCA1"}]}'
    )
    out = proposals.parse(content)
    assert out == [
        {"claim": "BRCA1 is a tumor suppressor", "quote": "tumor suppressor BRCA1"}
    ]


def test_parse_normalized_tolerates_misspelled_key():
    content = '{"normalised": ["time series post infection", "heat shock"]}'
    out = proposals.parse_normalized(content, 2)
    assert out == ["time series post infection", "heat shock"]
