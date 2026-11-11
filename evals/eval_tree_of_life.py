#!/usr/bin/env python3
"""Suite 3 — Tree-of-Life x Information-Density matrix.

  python evals/eval_tree_of_life.py --stage 3a            # 63 quick scans
  python evals/eval_tree_of_life.py --stage 3b            # 21 full graphs
  python evals/eval_tree_of_life.py --mock                # plumbing smoke
  python evals/eval_tree_of_life.py --stage 3a --resume results/<f>.jsonl

Stage 3A: all 63 corpus genes in Quick Scan (propose/summarize off) —
tests S0-S6 across clades and info tiers; records referent accuracy
(taxid), structural richness per facet, wall/usage per gene.

Stage 3B: the deep_subset (7 clades x 3 tiers x 1) in Full mode —
adds proposal funnel counts (claims >=0.50 / >=0.85 accepted, conditions)
and the 100% verbatim-quote audit (every quote segment must appear in
some fetched source text node).

Rows are fsynced per gene; JevQuotaError writes a 'stopped' row and
exits 75 — --resume FILE picks up where it left off.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from bionym.clients.ncbi import NcbiClient  # noqa: E402
from bionym.clients.veupathdb import VEuPathDBClient  # noqa: E402
from bionym.jev import JevClient, JevQuotaError  # noqa: E402
from bionym.llm import LlmClient  # noqa: E402
from bionym.resolver import Resolver  # noqa: E402

CORPUS = Path(__file__).parent / "data" / "tree_of_life_corpus.yaml"
RESULTS = Path(__file__).parent / "results"
EVAL_CACHE = Path(__file__).parent / ".cache"
EXIT_QUOTA = 75

ASSEMBLY_PREDS = {"in_assembly", "related_assembly", "superseded_by"}
ORTHO_PREDS = {"ortholog_of", "in_orthogroup", "has_member"}
TERM_PREDS = {"has_go_term", "has_domain", "in_pathway"}
DATA_PREDS = {"measured_in", "has_factor", "cited_in"}


def facet_counts(g) -> dict:
    types = Counter(n.type for n in g.nodes.values())
    preds = Counter(e.predicate for e in g.edges)
    return {
        "n_nodes": len(g.nodes),
        "n_edges": len(g.edges),
        "assemblies": sum(preds.get(p, 0) for p in ASSEMBLY_PREDS),
        "ortholog_edges": sum(preds.get(p, 0) for p in ORTHO_PREDS),
        "orthogroups": types.get("OrthologGroup", 0),
        "go_terms": types.get("GOTerm", 0),
        "domains": types.get("Domain", 0),
        "pathways": types.get("Pathway", 0),
        "datasets": types.get("Dataset", 0),
        "publications": types.get("Publication", 0),
        "claims": types.get("Claim", 0),
        "conditions": types.get("Condition", 0),
    }


def referent_ok(g, entry: dict) -> dict:
    """Check the resolved anchor hits the registered taxid."""
    want_taxid = entry.get("expected_taxid")
    got = set()
    for e in g.edges:
        if e.predicate == "in_organism":
            got.add(e.object.rsplit(":", 1)[-1])
    got |= {
        str(n.attrs.get("tax_id"))
        for n in g.nodes.values()
        if n.attrs.get("tax_id") is not None
    }
    taxid_hit = want_taxid is None or str(want_taxid) in got
    return {"expected_taxid": want_taxid, "taxid_hit": taxid_hit,
            "taxids_seen": sorted(got)[:6]}


def _quotes_in(obj) -> list[str]:
    """Collect every 'quote' string anywhere under an edge's payload."""
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "quote" and isinstance(v, str):
                out.append(v)
            else:
                out += _quotes_in(v)
    elif isinstance(obj, list):
        for v in obj:
            out += _quotes_in(v)
    return out


def _source_texts(g) -> list[str]:
    """All text-bearing node attrs (abstracts/descriptions/titles)."""
    out = []
    for n in g.nodes.values():
        for k in ("text", "abstract", "description", "title", "summary"):
            v = n.attrs.get(k)
            if isinstance(v, str) and len(v) > 40:
                out.append(v)
    return out


def quote_audit(g) -> dict:
    """Every proposal quote segment must appear verbatim in a fetched
    source text (the upstream abstract/table the LLM cited)."""
    corpus = _source_texts(g)
    n_q = n_ok = 0
    misses = []
    for e in g.edges:
        if e.predicate not in ("reports", "has_condition"):
            continue
        for quote in _quotes_in(e.model_dump()):
            for seg in [s.strip() for s in quote.split(" … ") if s.strip()]:
                n_q += 1
                if any(seg in t for t in corpus):
                    n_ok += 1
                else:
                    misses.append(seg[:120])
    return {
        "quote_segments": n_q,
        "verbatim_ok": n_ok,
        "verbatim_rate": round(n_ok / n_q, 4) if n_q else None,
        "unverifiable_if_zero_corpus": not corpus,
        "misses": misses[:10],
    }


def run_gene(resolver: Resolver, entry: dict, full: bool) -> dict:
    t0 = time.monotonic()
    error = ""
    try:
        g = resolver.resolve(
            str(entry["id"]), depth=6,
            propose=full, summarize=full,
        )
    except JevQuotaError:
        raise
    except Exception as e:  # noqa: BLE001 — row is the record
        g, error = None, f"{type(e).__name__}: {e}"
    wall = round(time.monotonic() - t0, 1)
    meta = g.metadata if g else {}
    row = {
        "id": entry["id"],
        "clade": entry["clade"],
        "info_tier": entry["info_tier"],
        "full": full,
        "outcome": meta.get("outcome", "error"),
        "wall_s": wall,
        "stage_times": meta.get("stage_times", {}),
        "jev_usage": meta.get("jev_usage", {}).get("total", {}),
        "llm_usage": meta.get("llm_usage", {}).get("total", {}),
        "error": error,
    }
    if g:
        row["referent"] = referent_ok(g, entry)
        row.update(facet_counts(g))
        if full:
            row["quote_audit"] = quote_audit(g)
    return row


def _write_row(fh, row: dict) -> None:
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def _median(xs):
    xs = sorted(x for x in xs if x is not None)
    return xs[len(xs) // 2] if xs else None


def summarize(rows: list[dict], stage: str) -> dict:
    done = [r for r in rows if r.get("kind") == "case" or "id" in r]
    by_tier: dict = {}
    for r in done:
        if not r.get("outcome"):
            continue
        t = r["info_tier"]
        by_tier.setdefault(t, []).append(r)
    tiers = {}
    for tier, rs in sorted(by_tier.items()):
        tiers[tier] = {
            "n": len(rs),
            "resolved": sum(r["outcome"] == "resolved" for r in rs),
            "refused": sum(r["outcome"] == "refused" for r in rs),
            "errors": sum(r["outcome"] == "error" or r.get("error")
                          for r in rs),
            "taxid_hit_rate": round(
                sum((r.get("referent") or {}).get("taxid_hit") is True
                    for r in rs) / len(rs), 4),
            "median_nodes": _median(r.get("n_nodes") for r in rs),
            "median_edges": _median(r.get("n_edges") for r in rs),
            "median_datasets": _median(r.get("datasets") for r in rs),
            "median_go_terms": _median(r.get("go_terms") for r in rs),
            "median_orthogroups": _median(
                r.get("orthogroups") for r in rs),
            "median_wall_s": _median(r.get("wall_s") for r in rs),
            "median_jev_input_tokens": _median(
                (r.get("jev_usage") or {}).get("input_tokens")
                for r in rs),
        }
    if stage == "3b":
        for tier, rs in tiers.items():
            rows_t = by_tier[tier]
            rs["claims_ge_0.5"] = [
                r.get("claims", 0) for r in rows_t]
            rs["median_claims"] = _median(r.get("claims") for r in rows_t)
            rs["median_conditions"] = _median(
                r.get("conditions") for r in rows_t)
            audits = [r["quote_audit"] for r in rows_t
                      if r.get("quote_audit") and
                      r["quote_audit"]["quote_segments"]]
            rs["verbatim_rate"] = round(
                sum(a["verbatim_ok"] for a in audits)
                / max(1, sum(a["quote_segments"] for a in audits)), 4)
    return {"stage": stage, "n": len(done), "by_tier": tiers}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--stage", choices=["3a", "3b"], default="3a")
    ap.add_argument("--name", default="")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--resume", metavar="FILE",
                    help="append to a partial rows file, skipping done ids")
    ap.add_argument("--id", help="run a single corpus id")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    corpus = yaml.safe_load(CORPUS.read_text())["corpus"]
    full = args.stage == "3b"
    if full:
        corpus = [c for c in corpus if c.get("deep_subset")]
    if args.id:
        corpus = [c for c in corpus if str(c["id"]) == args.id]
    if not corpus:
        sys.exit("no corpus entries matched")

    cache = str(EVAL_CACHE)
    jev = JevClient(mock=args.mock, cache_dir=cache)
    llm = LlmClient(mock=args.mock, cache_dir=cache)
    resolver = Resolver(
        jev=jev,
        ncbi=NcbiClient(cache_dir=cache),
        veupathdb=VEuPathDBClient(cache_dir=cache),
        llm=llm,
    )
    print(f"[config] stage={args.stage} n={len(corpus)} "
          f"jev={jev.model} llm={llm.model or 'off'} mock={args.mock}")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.resume:
        out_path = Path(args.resume)
        if not out_path.exists():
            sys.exit(f"--resume file not found: {out_path}")
        done_ids = {
            json.loads(l)["id"]
            for l in out_path.read_text().splitlines()
            if l.strip() and json.loads(l).get("kind") == "case"
        }
        mode = "a"
    else:
        name = args.name or f"suite{args.stage}-tol"
        out_path = Path(args.out) if args.out else (
            RESULTS / f"{stamp}-{name}.jsonl")
        done_ids = set()
        mode = "w"

    rows = []
    with out_path.open(mode) as fh:
        if not args.resume:
            _write_row(fh, {"kind": "config", "stage": args.stage,
                            "ts": stamp, "n_cases": len(corpus),
                            "mock": args.mock})
        else:
            _write_row(fh, {"kind": "resumed", "ts": stamp,
                            "skipped": sorted(done_ids)})
        for i, entry in enumerate(corpus):
            if str(entry["id"]) in done_ids:
                print(f"[skip] {entry['id']}", flush=True)
                continue
            print(f"[{i + 1}/{len(corpus)}] {entry['id']} "
                  f"({entry['clade']}/{entry['info_tier']}) …", flush=True)
            try:
                row = run_gene(resolver, entry, full)
            except JevQuotaError as e:
                _write_row(fh, {"kind": "stopped", "reason": str(e),
                                "id": entry["id"], "ts": stamp})
                print(f"\n[quota] JEV exhausted at {entry['id']} — "
                      f"progress is durable.\n        Refill, then:\n"
                      f"        python evals/eval_tree_of_life.py "
                      f"--stage {args.stage} --resume {out_path}",
                      flush=True)
                sys.exit(EXIT_QUOTA)
            row["kind"] = "case"
            rows.append(row)
            _write_row(fh, row)
            ref = row.get("referent") or {}
            print(f"  -> {row['outcome']} taxid_hit={ref.get('taxid_hit')} "
                  f"nodes={row.get('n_nodes')} {row['wall_s']}s", flush=True)

    summary = summarize(rows, args.stage)
    print(json.dumps(summary, indent=1))
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
