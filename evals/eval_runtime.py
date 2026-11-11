#!/usr/bin/env python3
"""Suite 5A — runtime & token breakdown (offline, $0).

  python evals/eval_runtime.py

Compiles a unified performance table from the recorded artifacts:

- Judge latency: per-question `ms` in *claimsplit*diffs*.jsonl replays
  + JEV's own timing results (*-jev-timing.results.json).
- Judge errors: rows carrying "error" in each diffs file.
- Generator throughput: *-gen-results.json (wall_s / n_asks).
- End-to-end resolve breakdown: rows in runner outputs (run.py /
  eval_tree_of_life.py results) — wall_s, stage_times, jev/llm usage
  per mode (quick vs full).
"""
from __future__ import annotations

import json
from collections import defaultdict
from datetime import date
from pathlib import Path

RESULTS = Path(__file__).resolve().parent / "results"
OUT = RESULTS / f"{date.today():%Y%m%d}-suite5a-runtime.json"


def _pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    return round(s[min(len(s) - 1, max(0, round(p / 100 * (len(s) - 1))))], 1)


def _median(vals: list[float]) -> float | None:
    return _pct(vals, 50)


def _tag(path: Path) -> str:
    name = path.name
    for pat in ("-claimsplit.diffs.jsonl", "-claimsplit-diffs.jsonl"):
        name = name.replace(pat, "")
    return name.split("-", 1)[-1]


def judge_table() -> dict:
    judges = {}
    for path in sorted(RESULTS.glob("*claimsplit*diffs*.jsonl")):
        rows = [json.loads(l) for l in path.read_text().splitlines()
                if l.strip()]
        ms = [r["ms"] for r in rows if isinstance(r.get("ms"), (int, float))]
        errs = sum(1 for r in rows if "error" in r)
        judges[_tag(path)] = {
            "questions": len([r for r in rows if "qid" in r]),
            "ask_errors": errs,
            "ms_p50": _pct(ms, 50),
            "ms_p90": _pct(ms, 90),
            "ms_max": max(ms) if ms else None,
        }
    # JEV's own recorded timing
    for path in sorted(RESULTS.glob("*jev-timing.results.json")):
        d = json.loads(path.read_text())
        judges["jev-latest"] = {
            "questions": d.get("reproduced_answers"),
            "asks": d.get("n_asks"),
            "ask_errors": d.get("n_errors"),
            "ms_p50": d.get("ms_p50"),
            "ms_p90": d.get("ms_p90"),
            "ms_max": d.get("ms_max"),
            "in_tokens_total": d.get("in_tokens_total"),
            "out_tokens_total": d.get("out_tokens_total"),
            "by_stage_ms": {s: v.get("ms_p50") for s, v in
                            (d.get("by_stage") or {}).items()
                            if isinstance(v, dict) and "ms_p50" in v},
        }
    return judges


def generator_table() -> dict:
    gens = {}
    for path in sorted(RESULTS.glob("*-gen-results.json")):
        d = json.loads(path.read_text())
        n = d.get("n_asks") or 0
        gens[d.get("model", path.stem)] = {
            "asks": n,
            "errors": d.get("n_errors"),
            "wall_s": d.get("wall_s"),
            "mean_s_per_ask": round(d["wall_s"] / n, 2) if n else None,
            "by_stage": {s: {"asks": v.get("asks")}
                         for s, v in (d.get("by_stage") or {}).items()},
        }
    return gens


def resolve_table() -> dict:
    """Aggregate stage_times/usage from runner output rows files."""
    out = {}
    for path in sorted(RESULTS.glob("*.jsonl")):
        if "diffs" in path.name or "raw" in path.name:
            continue
        rows = []
        try:
            rows = [json.loads(l)
                    for l in path.read_text().splitlines() if l.strip()]
        except json.JSONDecodeError:
            continue
        cases = [r for r in rows
                 if (r.get("kind") == "case" or
                     ("id" in r and "wall_s" in r))]
        if not cases:
            continue
        stage_ts: dict[str, list] = defaultdict(list)
        for r in cases:
            for s, t in (r.get("stage_times") or {}).items():
                if isinstance(t, (int, float)):
                    stage_ts[s].append(t)
        walls = [r["wall_s"] for r in cases
                 if isinstance(r.get("wall_s"), (int, float))]
        jev_in = [(r.get("jev_usage") or {}).get("input_tokens")
                  for r in cases]
        llm_in = [(r.get("llm_usage") or {}).get("input_tokens")
                  for r in cases]
        modes = {"full" if r.get("full") else
                 "quick" if r.get("quick", True) else "full"
                 for r in cases}
        out[path.name] = {
            "n": len(cases),
            "modes": sorted(modes),
            "wall_s_median": _median(walls),
            "stage_s_median": {s: _median(v) for s, v in
                               sorted(stage_ts.items())},
            "jev_input_tokens_median": _median(
                [x for x in jev_in if isinstance(x, (int, float))]),
            "llm_input_tokens_median": _median(
                [x for x in llm_in if isinstance(x, (int, float))]),
        }
    return out


def main() -> None:
    report = {
        "suite": "5a-runtime-token-breakdown",
        "judges": judge_table(),
        "generators": generator_table(),
        "resolves": resolve_table(),
    }
    OUT.write_text(json.dumps(report, indent=1))
    for section in ("judges", "generators", "resolves"):
        print(f"== {section} ==")
        for name, m in report[section].items():
            print(f"  {name}: {json.dumps(m)}")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
