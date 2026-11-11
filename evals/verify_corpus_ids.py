#!/usr/bin/env python3
"""Verify corpus IDs are real BEFORE any JEV spend (public API reads only).

  python evals/verify_corpus_ids.py                     # all corpora
  python evals/verify_corpus_ids.py --file evals/data/tree_of_life_corpus.yaml
  python evals/verify_corpus_ids.py --id PF3D7_1343700

Checks each identifier against the light clients (NCBI Datasets/E-utils,
VEuPathDB, UniProt — whatever the S0 regex hints suggest) and reports:
  ok     record exists and (when checkable) matches expected_taxid /
         expected_symbol_or_locus
  warn   record exists but a registered expectation didn't match, or the
         record lacks fields to check it against
  FAIL   no record found — fix the corpus entry before running evals

Exit 1 on any FAIL. Results cache in evals/.cache so re-runs are free.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from bionym.clients.ncbi import NcbiClient  # noqa: E402
from bionym.clients.uniprot import UniProtClient  # noqa: E402
from bionym.clients.veupathdb import VEuPathDBClient  # noqa: E402
from bionym.questions.s0_classify import regex_hints  # noqa: E402

EVAL_CACHE = Path(__file__).parent / ".cache"
DEFAULT_FILES = [
    Path(__file__).parent / "data" / "tree_of_life_corpus.yaml",
]


def _expected_ok(rec: dict, entry: dict) -> list[str]:
    """Expectation mismatches — empty list = all good/uncheckable."""
    problems = []
    want_taxid = entry.get("expected_taxid")
    if want_taxid is not None:
        got = rec.get("tax_id") or rec.get("organism_taxid")
        try:
            if got is not None and int(got) != int(want_taxid):
                problems.append(f"taxid {got} != {want_taxid}")
        except (TypeError, ValueError):
            pass
    want_name = (entry.get("expected_symbol_or_locus") or "").lower()
    if want_name:
        candidates = {
            str(rec.get(k) or "").lower()
            for k in ("symbol", "locus_tag", "gene_id", "accession",
                      "primary_symbol")
        }
        candidates |= {
            str(a.get("alias") if isinstance(a, dict) else a).lower()
            for a in (rec.get("aliases") or [])
        }
        if want_name not in candidates:
            problems.append(
                f"symbol/locus {want_name!r} not in {sorted(candidates)[:6]}")
    return problems


def _verify_one(clients: dict, entry: dict) -> tuple[str, str]:
    """-> (status, detail). status: ok | warn | fail"""
    ident = entry["id"]
    hints = regex_hints(str(ident))
    records: list[dict] = []
    if "ncbi_gene" in hints:
        records, _ = clients["ncbi"].gene_by_id(str(ident))
    elif "veupathdb" in hints:
        records, _ = clients["veupathdb"].lookup_gene(str(ident))
    elif "uniprot" in hints:
        records, _ = clients["uniprot"].search_gene(str(ident))
    else:
        # symbol / locus_tag / ensembl / anything else -> NCBI Gene search
        records, _ = clients["ncbi"].find_gene(str(ident))
    if not records:
        return "fail", f"{ident} ({hints or ['other']}): no record found"
    rec = records[0]
    problems = _expected_ok(rec, entry)
    label = rec.get("symbol") or rec.get("gene_id") or rec.get("accession")
    org = rec.get("organism") or rec.get("organism_name") or "?"
    if problems:
        return "warn", f"{ident} -> {label} ({org}): {'; '.join(problems)}"
    return "ok", f"{ident} -> {label} ({org})"


def _iter_entries(path: Path):
    data = yaml.safe_load(path.read_text())
    if isinstance(data, dict) and "corpus" in data:
        yield from data["corpus"]
    elif isinstance(data, list):
        yield from data
    else:
        raise ValueError(f"unrecognized corpus format: {path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--file", action="append", default=None,
                    help="corpus yaml to verify (repeatable); default "
                         "tree_of_life_corpus.yaml")
    ap.add_argument("--id", help="verify a single corpus entry id")
    args = ap.parse_args()

    files = [Path(f) for f in (args.file or DEFAULT_FILES)]
    cache = str(EVAL_CACHE)
    clients = {
        "ncbi": NcbiClient(cache_dir=cache),
        "veupathdb": VEuPathDBClient(cache_dir=cache),
        "uniprot": UniProtClient(cache_dir=cache),
    }

    n_ok = n_warn = n_fail = 0
    for path in files:
        for entry in _iter_entries(path):
            if args.id and str(entry["id"]) != args.id:
                continue
            status, detail = _verify_one(clients, entry)
            n_ok += status == "ok"
            n_warn += status == "warn"
            n_fail += status == "fail"
            mark = {"ok": "ok  ", "warn": "WARN", "fail": "FAIL"}[status]
            print(f"[{mark}] {detail}", flush=True)
    print(f"[done] {n_ok} ok, {n_warn} warnings, {n_fail} failures")
    sys.exit(1 if n_fail else 0)


if __name__ == "__main__":
    main()
