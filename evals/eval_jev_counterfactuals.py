#!/usr/bin/env python3
"""Suite 1B — evidence-ladder counterfactual probe (JEV + replayed judges).

  python evals/eval_jev_counterfactuals.py            # JEV run + metrics
  python evals/eval_jev_counterfactuals.py --dry      # asks only, no spend
  python evals/eval_jev_counterfactuals.py --mock     # plumbing smoke test
  python evals/eval_jev_counterfactuals.py \
      --diffs evals/results/*counterfactual*.diffs.jsonl

30 families x 4 evidence rungs (L3 direct > L2 hedged > L1 wrong-species
> L0 unsupported) built through the REAL production question builders —
s1_resolve.gene_match, s5_expression.relevant_0, proposals.prop_0 — so
the probe measures JEV's score response to evidence quality on the same
question shapes the pipeline actually emits.

Deviation from plan §1B: per-rung asks are isolated (120 asks) rather
than batched (~30) — isolating rungs keeps the counterfactual causal
(no cross-candidate contrast effects) and preserves production question
shapes (single-candidate noul). States are tiny; total cost < ~0.1M
input tokens.

The asks file uses the dump_asks {gene, stage, state, questions,
jev_answers} schema, so replay_llm.py can judge the identical 120 asks
through any chat LLM; --diffs then scores that judge on the same
metrics (join via ask_i -> family/rung).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from bionym import proposals  # noqa: E402
from bionym.jev import JevClient, JevQuotaError  # noqa: E402
from bionym.questions import s1_resolve, s5_expression  # noqa: E402

DATA = Path(__file__).parent / "data" / "jev_counterfactuals.yaml"
ASKS_OUT = Path(__file__).parent / "data" / "jev_counterfactuals_asks.jsonl"
RESULTS = Path(__file__).parent / "results"
EVAL_CACHE = Path(__file__).parent / ".cache"
EXIT_QUOTA = 75

RUNGS = ["L3", "L2", "L1", "L0"]  # expected score order: L3 > L2 > L1 > L0


def build_ask(family: dict, rung_key: str) -> dict:
    """One family+rung -> {gene, stage, rung, qid, state, questions}."""
    task = family["task"]
    rung = family["rungs"][rung_key]
    if task == "s1_resolve":
        cand = dict(rung)
        state = s1_resolve.build_state(family["input"], [cand])
        questions = s1_resolve.build_questions([cand])
        qid = "gene_match"
    elif task == "s5_expression":
        cand = dict(rung)
        state = s5_expression.build_state(
            family["input"], family["match"], [cand])
        questions = s5_expression.build_questions([cand])
        qid = "relevant_0"
    elif task == "proposals":
        props = [{"node": family["node"], "type": "claim",
                  "claim": family["claim"], "quote": rung["quote"]}]
        state = proposals.build_state(
            {family["node"]: rung["text"]}, props, family["match"])
        questions = proposals.build_questions(props)
        qid = "prop_0"
    else:
        raise ValueError(f"unknown task {task!r}")
    return {"gene": family["family"], "stage": task, "rung": rung_key,
            "qid": qid, "state": state, "questions": questions}


def load_families() -> list[dict]:
    return yaml.safe_load(DATA.read_text())["families"]


# -- metrics ---------------------------------------------------------------

def _kendall_tau(scores: list[float]) -> float:
    """tau-a between rung order (descending expected) and observed score."""
    n = len(scores)
    concord = discord = 0
    for i in range(n):
        for j in range(i + 1, n):
            # i<j means rung i is stronger; expect scores[i] > scores[j]
            if scores[i] > scores[j]:
                concord += 1
            elif scores[i] < scores[j]:
                discord += 1
    total = n * (n - 1) // 2
    return round((concord - discord) / total, 4) if total else 0.0


def _auroc(pos: list[float], neg: list[float]) -> float | None:
    """Rank-based AUROC: P(pos > neg) with ties at 0.5."""
    if not pos or not neg:
        return None
    wins = sum(1.0 if p > n else 0.5 if p == n else 0.0
               for p in pos for n in neg)
    return round(wins / (len(pos) * len(neg)), 4)


def ladder_metrics(scores: dict[str, dict[str, float]]) -> dict:
    """scores: family -> rung -> noul. Monotone/tau/gaps/accept-rates."""
    fam_scores = {f: [s.get(r) for r in RUNGS] for f, s in scores.items()}
    fam_scores = {f: v for f, v in fam_scores.items()
                  if all(v is not None for v in v)}
    n_fam = len(fam_scores)
    if not n_fam:
        return {"n_families": 0}
    strict = sum(
        1 for v in fam_scores.values() if v[0] > v[1] > v[2] > v[3])
    taus = [_kendall_tau(v) for v in fam_scores.values()]
    gap_l3_l1 = sorted(v[0] - v[2] for v in fam_scores.values())
    gap_l2_l0 = sorted(v[1] - v[3] for v in fam_scores.values())
    accept = {
        f"accept@{t}": {
            r: round(sum(1 for v in fam_scores.values()
                         if v[i] >= t) / n_fam, 4)
            for i, r in enumerate(RUNGS)
        }
        for t in (0.5, 0.85)
    }
    l3 = [v[0] for v in fam_scores.values()]
    l1_l0 = [v[i] for v in fam_scores.values() for i in (2, 3)]
    return {
        "n_families": n_fam,
        "strict_monotone_rate": round(strict / n_fam, 4),
        "mean_kendall_tau": round(sum(taus) / len(taus), 4),
        "median_gap_L3_minus_L1": round(
            gap_l3_l1[len(gap_l3_l1) // 2], 4),
        "median_gap_L2_minus_L0": round(
            gap_l2_l0[len(gap_l2_l0) // 2], 4),
        "auroc_L3_vs_L1L0": _auroc(l3, l1_l0),
        **accept,
    }


def _scores_from_answers(asks: list[dict], key: str) -> dict:
    """family -> rung -> noul, reading answers under `key`."""
    scores: dict[str, dict[str, float]] = {}
    for ask in asks:
        a = (ask.get(key) or {}).get(ask["qid"]) or {}
        if isinstance(a.get("noul"), (int, float)):
            scores.setdefault(ask["gene"], {})[ask["rung"]] = a["noul"]
    return scores


# -- runner ----------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asks", default=str(ASKS_OUT),
                    help="asks jsonl (written on run/dry; read by --diffs)")
    ap.add_argument("--dry", action="store_true",
                    help="write the asks file only — no JEV spend")
    ap.add_argument("--mock", action="store_true",
                    help="mock JEV answers for a plumbing smoke test")
    ap.add_argument("--diffs", nargs="*", default=None,
                    help="per-question replay diffs to score on the same "
                         "ladder metrics (from replay_llm.py --asks)")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    families = load_families()
    asks = [build_ask(f, r) for f in families for r in RUNGS]
    asks_path = Path(args.asks)

    if args.diffs is not None:
        # Rejoin replayed answers by ask_i and score the same metrics.
        diffs_rows = [
            json.loads(l)
            for p in args.diffs
            for l in Path(p).read_text().splitlines()
            if l.strip()
        ]
        scores: dict[str, dict[str, float]] = {}
        for r in diffs_rows:
            if "qid" not in r or "error" in r:
                continue
            ask = asks[r["ask_i"]] if r.get("ask_i") is not None else None
            if ask is None:
                continue
            v = r.get("laya")
            if isinstance(v, (int, float)):
                scores.setdefault(ask["gene"], {})[ask["rung"]] = v
        print(json.dumps(ladder_metrics(scores), indent=1))
        return

    if not args.dry:
        jev = JevClient(mock=args.mock, cache_dir=str(EVAL_CACHE))
        for i, ask in enumerate(asks):
            try:
                ask["jev_answers"] = jev.ask(
                    ask["state"], ask["questions"],
                    stage=f"cf_{ask['stage']}",
                )
            except JevQuotaError as e:
                # Cached asks replay free on re-run — just resume.
                asks_path.parent.mkdir(parents=True, exist_ok=True)
                asks_path.write_text(
                    "".join(json.dumps(a) + "\n" for a in asks))
                print(f"\n[quota] JEV exhausted at ask {i}/{len(asks)} "
                      f"— answered asks are cached, re-run to resume.\n"
                      f"        {e}", flush=True)
                sys.exit(EXIT_QUOTA)
            if (i + 1) % 20 == 0 or i + 1 == len(asks):
                print(f"[{i + 1}/{len(asks)}]", flush=True)
    else:
        for ask in asks:
            ask["jev_answers"] = {}

    asks_path.parent.mkdir(parents=True, exist_ok=True)
    asks_path.write_text("".join(json.dumps(a) + "\n" for a in asks))
    print(f"wrote {asks_path} ({len(asks)} asks)")

    if args.dry:
        return
    scores = _scores_from_answers(asks, "jev_answers")
    metrics = ladder_metrics(scores)
    tag = args.tag or "suite1b-jev-counterfactuals"
    from datetime import date
    out_path = RESULTS / f"{date.today():%Y%m%d}-{tag}.json"
    out = {"suite": "1b-jev-counterfactuals", "judge": "jev-latest",
           "asks": str(asks_path.relative_to(ROOT)),
           "metrics": metrics, "scores": scores}
    out_path.write_text(json.dumps(out, indent=1))
    print(json.dumps(metrics, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
