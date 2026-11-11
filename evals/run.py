#!/usr/bin/env python3
"""Eval runner: real resolves over cases.yaml -> results/*.jsonl.

  python evals/run.py --tier t1 --name baseline
  python evals/run.py --case deep-k13 --name k13-check
  python evals/run.py --tier t1 --name mock-smoke --mock   # JEV/LLM mocked

Every row records the configured LLM and systemone backend (model +
URL) at run time, so `compare.py a.jsonl b.jsonl` reads as
"config delta -> outcome delta".
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
# evals always exercise the working tree, not an installed snapshot
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from bionym.clients.ncbi import NcbiClient  # noqa: E402
from bionym.clients.veupathdb import VEuPathDBClient  # noqa: E402
from bionym.jev import JevClient, JevQuotaError  # noqa: E402
from bionym.llm import LlmClient  # noqa: E402
from bionym.resolver import Resolver  # noqa: E402

CASES = Path(__file__).parent / "cases.yaml"
RESULTS = Path(__file__).parent / "results"
# Shared call cache: a --resume run replays already-paid JEV/LLM/NCBI
# calls for free, so a quota stop mid-case loses no work.
EVAL_CACHE = Path(__file__).parent / ".cache"
# sysexits EX_TEMPFAIL — "refill, then --resume"
EXIT_QUOTA = 75


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=5,
        ).stdout.strip()
    except Exception:
        return "unknown"


def assertions(graph, expect: dict) -> list[dict]:
    """Each expect key -> {check, ok, detail}. Tolerates unknown keys."""
    out = []
    meta = graph.metadata
    if "outcome" in expect:
        actual = meta.get("outcome", "?")
        out.append({"check": "outcome", "ok": actual == expect["outcome"],
                    "detail": f"{actual} != {expect['outcome']}" if actual != expect["outcome"] else ""})
    if "anchor_ns" in expect:
        ns = expect["anchor_ns"]
        hit = any(nid.startswith(f"{ns}:") for nid in graph.nodes)
        hit = hit or any(
            e.predicate == "same_as" and e.object.startswith(f"{ns}:")
            for e in graph.edges
        )
        out.append({"check": "anchor_ns", "ok": hit,
                    "detail": "" if hit else f"no node/same_as with {ns}:"})
    if "min_nodes" in expect:
        n = len(graph.nodes)
        out.append({"check": "min_nodes", "ok": n >= expect["min_nodes"],
                    "detail": f"{n} < {expect['min_nodes']}" if n < expect["min_nodes"] else ""})
    if "predicates" in expect:
        have = {e.predicate for e in graph.edges}
        missing = set(expect["predicates"]) - have
        out.append({"check": "predicates", "ok": not missing,
                    "detail": f"missing {sorted(missing)}" if missing else ""})
    return out


def _write_row(fh, row: dict) -> None:
    """Append a JSONL row durably — a crash must not lose completed work."""
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def _completed_cases(path: Path) -> set[str]:
    return {
        json.loads(l)["case"]
        for l in path.read_text().splitlines()
        if l.strip() and json.loads(l).get("kind") == "case"
    }


def run_case(resolver: Resolver, case: dict) -> dict:
    """One resolve -> one result row. Exceptions become outcome=error."""
    t0 = time.monotonic()
    error = ""
    try:
        graph = resolver.resolve(
            case["id"],
            depth=6,
            propose=not case.get("quick", True),
            summarize=not case.get("quick", True),
        )
    except JevQuotaError:
        raise  # never record quota exhaustion as a case failure
    except Exception as e:  # noqa: BLE001 — the row IS the record
        graph, error = None, f"{type(e).__name__}: {e}"
    wall = round(time.monotonic() - t0, 1)

    meta = graph.metadata if graph else {}
    checks = assertions(graph, case.get("expect", {})) if graph else []
    if error:
        checks.append({"check": "outcome", "ok": False, "detail": error})
    return {
        "case": case["name"],
        "id": case["id"],
        "tier": case["tier"],
        "quick": case.get("quick", True),
        "outcome": meta.get("outcome", "error"),
        "assertions": checks,
        "ok": all(c["ok"] for c in checks),
        "wall_s": wall,
        "stage_times": meta.get("stage_times", {}),
        "jev_usage": meta.get("jev_usage", {}).get("total", {}),
        "llm_usage": meta.get("llm_usage", {}).get("total", {}),
        "nodes": len(graph.nodes) if graph else 0,
        "edges": len(graph.edges) if graph else 0,
        "predicates": sorted({e.predicate for e in graph.edges}) if graph else [],
        "error": error,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--tier", choices=["t1", "t2", "t3"], help="run only this tier")
    ap.add_argument("--case", help="run a single case by name")
    ap.add_argument("--name", default="run", help="run label in the filename")
    ap.add_argument("--mock", action="store_true", help="mock JEV+LLM (data APIs still real)")
    ap.add_argument(
        "--resume", metavar="FILE",
        help="append to an existing results file, skipping cases it "
             "already completed — use after a quota stop",
    )
    args = ap.parse_args()

    cases = yaml.safe_load(CASES.read_text())
    if args.tier:
        cases = [c for c in cases if c["tier"] == args.tier]
    if args.case:
        cases = [c for c in cases if c["name"] == args.case or c["id"] == args.case]
    if not cases:
        sys.exit("no cases matched")

    cache = str(EVAL_CACHE)
    jev = JevClient(mock=args.mock, cache_dir=cache)
    llm = LlmClient(mock=args.mock, cache_dir=cache)
    resolver = Resolver(
        jev=jev,
        ncbi=NcbiClient(cache_dir=cache),
        veupathdb=VEuPathDBClient(cache_dir=cache),
        llm=llm,
    )

    config = {
        "llm_model": llm.model, "llm_base": llm.base_url,
        "jev_model": jev.model, "jev_url": jev.url,
        "mock": args.mock, "git": git_sha(),
    }
    print(f"[config] llm={config['llm_model']}@{config['llm_base']} "
          f"jev={config['jev_model']}@{config['jev_url']} "
          f"git={config['git']} mock={args.mock}")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.resume:
        out_path = Path(args.resume)
        if not out_path.exists():
            sys.exit(f"--resume file not found: {out_path}")
        done_cases = _completed_cases(out_path)
        mode = "a"
    else:
        out_path = RESULTS / f"{stamp}-{args.name}.jsonl"
        done_cases = set()
        mode = "w"
    with out_path.open(mode) as fh:
        if args.resume:
            _write_row(fh, {"kind": "resumed", "ts": stamp,
                            "skipped": sorted(done_cases)})
        else:
            _write_row(fh, {"kind": "config", **config,
                            "ts": stamp, "n_cases": len(cases)})
        for case in cases:
            if case["name"] in done_cases:
                print(f"[skip] {case['name']} — already in {out_path.name}",
                      flush=True)
                continue
            print(f"[case] {case['name']} ({case['id']}) …", flush=True)
            try:
                row = run_case(resolver, case)
            except JevQuotaError as e:
                _write_row(fh, {"kind": "stopped", "reason": str(e),
                                "case": case["name"], "ts": stamp})
                print(
                    f"\n[quota] JEV balance exhausted at "
                    f"{case['name']} — progress is durable.\n"
                    f"        Refill the account, then resume with:\n"
                    f"        python evals/run.py --resume {out_path}",
                    flush=True,
                )
                sys.exit(EXIT_QUOTA)
            _write_row(fh, {"kind": "case", **row})
            mark = "ok" if row["ok"] else "FAIL"
            fails = [c["detail"] for c in row["assertions"] if not c["ok"]]
            print(f"  -> {row['outcome']} {row['wall_s']}s {mark}"
                  + (f"  {'; '.join(fails)}" if fails else ""), flush=True)

    done = out_path.stat().st_size
    print(f"[done] {out_path} ({done}b) — compare: "
          f"python evals/compare.py {out_path.name} <other>")


if __name__ == "__main__":
    main()
