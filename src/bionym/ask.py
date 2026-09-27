"""Ask-the-graph orchestration: classify -> slice -> LLM answers -> JEV scores.

The question is first bucketed into graph categories (JEV noul flags —
see questions/ask.CATEGORIES); the LLM and the judging JEV ask then see
only that slice, because whole-graph states overflow JEV's context.
When no category matches (`other`), the LLM still answers over a compact
projection but nothing is scored — a warning says so.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from .graph import KnowledgeGraph
from .jev import JevClient, JevError
from .llm import LlmClient, LlmError
from .questions import ask as ask_q

log = logging.getLogger(__name__)

# Category noul >= this counts the question as "about" that category.
CAT_THRESHOLD = 0.5


def _categories(
    answers: dict[str, Any],
) -> tuple[list[str], dict[str, float]]:
    flags = {
        qid.removeprefix("cat_"): (v.get("noul") if isinstance(v, dict) else v)
        for qid, v in answers.items()
    }
    picked = [k for k, v in flags.items() if (v or 0) >= CAT_THRESHOLD]
    # stable order = CATEGORIES declaration order
    picked.sort(key=list(ask_q.CATEGORIES).index)
    return picked, flags


def _llm_answer(
    llm: LlmClient, projection: dict, question: str, excerpt: bool
) -> tuple[str, list]:
    system, user = ask_q.build_answer_messages(
        json.dumps(projection), question, excerpt=excerpt
    )
    raw = llm.complete(system, user)
    try:
        payload = json.loads(raw)
        return payload.get("answer") or "", payload.get("cited_nodes") or []
    except json.JSONDecodeError:
        # model ignored the JSON contract — keep the prose, flag no cites
        return raw, []


def ask_graph(
    jev: JevClient,
    llm: LlmClient,
    question: str,
    graph: KnowledgeGraph,
) -> dict[str, Any]:
    """Answer `question` over `graph`; return answer + JEV scores."""
    if not llm.enabled:
        raise LlmError("LLM not configured (LLM_API_KEY/LLM_MODEL)")

    cls = jev.ask(
        ask_q.build_classify_state(question),
        ask_q.build_classify_questions(),
        stage="ask_classify",
    )
    categories, flags = _categories(cls)

    if not categories:
        # 'other': no bounded slice exists to judge against, so we answer
        # over the capped full projection and return the answer unscored.
        answer, cited = _llm_answer(
            llm, ask_q.compact_graph(graph), question, excerpt=False
        )
        return {
            "question": question,
            "answer": answer,
            "cited_nodes": cited,
            "invalid_cited_nodes": [c for c in cited if str(c) not in graph.nodes],
            "categories": [],
            "category_scores": flags,
            "scored": False,
            "warning": (
                "Question didn't map to any graph category — answer "
                "returned without JEV scoring. It may be outside what the "
                "graph can answer."
            ),
            "scores": {},
        }

    proj = ask_q.slice_graph(graph, categories)
    answer, cited = _llm_answer(llm, proj, question, excerpt=True)
    # Citation fidelity is checked deterministically against the FULL
    # graph — a cite to a real node outside the slice is still a real id.
    invalid = [c for c in cited if str(c) not in graph.nodes]
    fidelity = (len(cited) - len(invalid)) / len(cited) if cited else None
    try:
        answers = jev.ask(
            ask_q.build_state(question, answer, proj, categories),
            ask_q.build_questions(),
            stage="ask",
        )
    except JevError as e:
        # Judge failed after the answer was generated — same contract as
        # 'other': keep the answer, flag it unscored.
        return {
            "question": question,
            "answer": answer,
            "cited_nodes": cited,
            "invalid_cited_nodes": invalid,
            "categories": categories,
            "category_scores": flags,
            "scored": False,
            "warning": f"JEV scoring failed ({e}) — answer returned unscored.",
            "scores": {"citation_fidelity": fidelity},
        }
    return {
        "question": question,
        "answer": answer,
        "cited_nodes": cited,
        "invalid_cited_nodes": invalid,
        "categories": categories,
        "category_scores": flags,
        "scored": True,
        "warning": None,
        "scores": {
            "citation_fidelity": fidelity,
            **{
                k: (v.get("noul") if isinstance(v, dict) else v)
                for k, v in answers.items()
            },
        },
    }
