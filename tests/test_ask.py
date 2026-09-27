"""ask_graph paths: scored, 'other' unscored, JEV scoring failure,
citation fidelity, disabled LLM. All offline — FakeJev/FakeLlm."""
import pytest

from bionym.ask import ask_graph
from bionym.llm import LlmError

from fakes import DisabledLlm, FakeJev, FakeLlm, mini_graph


def test_scored_path_returns_scores_and_no_warning():
    res = ask_graph(
        FakeJev(cats=["datasets"]),
        FakeLlm({"answer": "k13 has expression data",
                 "cited_nodes": ["ds:EXP001"]}),
        "what datasets exist?",
        mini_graph(),
    )
    assert res["scored"] is True
    assert res["warning"] is None
    assert res["categories"] == ["datasets"]
    # classify nouls flow through as category_scores
    assert res["category_scores"]["datasets"] == 0.9
    # citation fidelity + one score per scoring question
    assert res["scores"]["citation_fidelity"] == 1.0
    assert len(res["scores"]) > 1
    assert res["invalid_cited_nodes"] == []


def test_citation_fidelity_flags_unknown_node_ids():
    res = ask_graph(
        FakeJev(cats=["datasets"]),
        FakeLlm({"answer": "a",
                 "cited_nodes": ["ds:EXP001", "bogus:1"]}),
        "q",
        mini_graph(),
    )
    assert res["invalid_cited_nodes"] == ["bogus:1"]
    assert res["scores"]["citation_fidelity"] == 0.5


def test_no_category_returns_unscored_with_warning():
    res = ask_graph(
        FakeJev(cats=[]),  # every cat noul below CAT_THRESHOLD
        FakeLlm(),
        "what color is the graph?",
        mini_graph(),
    )
    assert res["scored"] is False
    assert res["categories"] == []
    assert res["scores"] == {}
    assert res["warning"] and "category" in res["warning"]
    assert res["answer"] == "it does X"  # LLM still answered


def test_jev_scoring_failure_keeps_answer_unscored():
    res = ask_graph(
        FakeJev(cats=["datasets"], fail_scoring=True),
        FakeLlm({"answer": "a", "cited_nodes": ["ds:EXP001"]}),
        "q",
        mini_graph(),
    )
    assert res["scored"] is False
    assert "JEV scoring failed" in res["warning"]
    # fidelity still computed — it needs no judge
    assert res["scores"]["citation_fidelity"] == 0.5 or \
        res["scores"]["citation_fidelity"] == 1.0
    assert res["categories"] == ["datasets"]


def test_disabled_llm_raises():
    with pytest.raises(LlmError):
        ask_graph(FakeJev(), DisabledLlm(), "q", mini_graph())


def test_prose_answer_survives_broken_json():
    """LLM ignoring the JSON contract -> raw prose, empty cites."""

    class ProseLlm(FakeLlm):
        def complete(self, system, user):
            return "plain prose, not JSON"

    res = ask_graph(
        FakeJev(cats=[]), ProseLlm(), "q", mini_graph()
    )
    assert res["answer"] == "plain prose, not JSON"
    assert res["cited_nodes"] == []
