#!/usr/bin/env python3
"""Diff two eval result files: outcome flips, assertion regressions,
wall/token deltas per case.

  python evals/compare.py 20260927-120000-baseline.jsonl cand.jsonl

Args may be filenames in evals/results/ or bare run names
(--name used at run time) — resolved by glob.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

RESULTS = Path(__file__).parent / "results"


def load(ref: str) -> tuple[dict, dict[str, dict]]:
    """Return (config row, {case name: result row})."""
    path = Path(ref)
    if not path.exists():
        hits = sorted(RESULTS.glob(f"*{ref.removesuffix('.jsonl')}*.jsonl"))
        if not hits:
            sys.exit(f"no results file matches {ref!r}")
        path = hits[-1]
    config, cases = {}, {}
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if row.get("kind") == "config":
            config = row
        elif row.get("kind") == "case":
            cases[row["case"]] = row
    config["_file"] = path.name
    return config, cases


def tokens(row: dict) -> int:
    u = row.get("jev_usage") or {}
    return u.get("input_tokens", 0) + u.get("output_tokens", 0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("a", help="baseline run file/name")
    ap.add_argument("b", help="candidate run file/name")
    args = ap.parse_args()

    cfg_a, a = load(args.a)
    cfg_b, b = load(args.b)

    print(f"A: {cfg_a['_file']}  llm={cfg_a.get('llm_model')} "
          f"jev={cfg_a.get('jev_model')} git={cfg_a.get('git')}")
    print(f"B: {cfg_b['_file']}  llm={cfg_b.get('llm_model')} "
          f"jev={cfg_b.get('jev_model')} git={cfg_b.get('git')}")
    print()

    rows = []
    n_reg = n_flip = 0
    for name in sorted(set(a) | set(b)):
        ra, rb = a.get(name), b.get(name)
        if ra is None or rb is None:
            rows.append((name, "MISSING", "", "", ""))
            continue
        flip = ra["outcome"] != rb["outcome"]
        reg = ra["ok"] and not rb["ok"]
        n_flip += flip
        n_reg += reg
        oc = (f"{ra['outcome']}→{rb['outcome']}"
              if flip else ra["outcome"])
        ok = ""
        if reg:
            ok = "PASS→FAIL"
        elif not ra["ok"] and rb["ok"]:
            ok = "fail→pass"
        dt = rb["wall_s"] - ra["wall_s"]
        dtok = tokens(rb) - tokens(ra)
        rows.append((name, oc, ok, f"{dt:+.1f}s", f"{dtok:+d}"))
        if reg:
            # show which assertions broke
            new_fail = {c["check"]: c["detail"] for c in rb["assertions"] if not c["ok"]}
            rows.append(("", "", f"   └ {new_fail}", "", ""))

    w = max(len(r[0]) for r in rows)
    hdr = ("case".ljust(w), "outcome", "", "Δwall", "Δjev tokens")
    print("  ".join(hdr))
    for r in rows:
        print("  ".join([r[0].ljust(w), r[1], r[2], r[3], r[4]]).rstrip())

    ta = sum(r["wall_s"] for r in a.values())
    tb = sum(r["wall_s"] for r in b.values())
    ka = sum(tokens(r) for r in a.values())
    kb = sum(tokens(r) for r in b.values())
    print()
    print(f"{n_flip} outcome flips, {n_reg} regressions "
          f"({sum(1 for r in b.values() if r['ok'])}/{len(b)} ok in B)")
    print(f"totals: wall {ta:.0f}→{tb:.0f}s ({tb-ta:+.0f}s)  "
          f"jev tokens {ka}→{kb} ({kb-ka:+d})")


if __name__ == "__main__":
    main()
