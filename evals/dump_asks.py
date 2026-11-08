#!/usr/bin/env python3
"""Record every JEV ask during real resolves -> asks.jsonl.

  python evals/dump_asks.py --out evals/asks.jsonl [--depth 6]

One line per ask: {gene, stage, state, questions, jev_answers}.
Replaying the file against another /v1/systemone backend (e.g. a
Laya kernel) answers "how does that judge score THE SAME asks" —
byte-identical states, no TypeSafe key needed at the far end.

Uses real clients + a real LLM for proposals, so the recorded states
are the ones a production resolve actually sends. If LlmClient is
disabled the proposal/triage stages simply emit no asks — the
deterministic pick/link asks are still recorded.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from bionym.jev import JevClient  # noqa: E402
from bionym.llm import LlmClient  # noqa: E402
from bionym.resolver import Resolver  # noqa: E402

# the three genes from the v2 demo graphs — ama1, crt, kelch13
GENES = ["PF3D7_1133400", "PF3D7_0709000", "PF3D7_1343700"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="evals/asks.jsonl")
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--gene", action="append",
                    help="resolve this id instead of the default three")
    ap.add_argument("--graphs", type=Path, default=None,
                    help="also write <dir>/<gene>.json per resolved graph")
    args = ap.parse_args()
    if args.graphs:
        args.graphs.mkdir(parents=True, exist_ok=True)

    jev = JevClient()
    llm = LlmClient()
    if not llm.enabled:
        print("[warn] LLM not configured — proposals/triage asks "
              "won't be recorded", file=sys.stderr)
    resolver = Resolver(jev=jev, llm=llm)
    orig_ask = jev.ask

    lock = threading.Lock()
    out = Path(args.out)
    n = 0
    with out.open("w") as fh:
        for gene in args.gene or GENES:
            def tap(state, questions, stage="", _gene=gene,
                    _fh=fh):
                nonlocal n
                answers = orig_ask(state, questions, stage=stage)
                with lock:
                    _fh.write(json.dumps({
                        "gene": _gene, "stage": stage,
                        "state": state, "questions": questions,
                        "jev_answers": answers,
                    }) + "\n")
                    n += 1
                return answers

            jev.ask = tap
            print(f"[resolve] {gene} depth={args.depth}", flush=True)
            before = n
            g = resolver.resolve(gene, depth=args.depth, summarize=True)
            if args.graphs:
                g.to_json(args.graphs / f"{gene}.json")
            print(f"  {n - before} asks", flush=True)

    print(f"wrote {n} asks -> {out}")


if __name__ == "__main__":
    main()
