#!/usr/bin/env python3
"""Suite 4 — grounded `bionym ask` vs. bare LLM, replicated.

  python evals/eval_grounded_ask_vs_bare.py --regime t0 --rep 1
  python evals/eval_grounded_ask_vs_bare.py --analyze          # stability
  python evals/eval_grounded_ask_vs_bare.py --mock --rep 1     # smoke

12 targets x 4 questions x {grounded, bare} x K reps x 2 regimes
(t0 = temperature 0.0, t07 = 0.7).

Cache discipline (plan §4.2): graphs resolve under the SHARED
evals/.cache so Suite 3B's JEV/DB calls are reused and the graph is
identical across reps. Ask generations use a per-(regime, rep)
ISOLATED LlmClient cache (evals/.cache/suite4_<regime>_rep<k>) so K=3
runs are genuinely independent. Anchor targets (one per stratum) also
get a full end-to-end resolve under the rep cache to measure upstream
construction repeatability.

Refused targets (orf1): the refusal is recorded as the grounded
outcome, and ask_graph still runs over the refused graph (plan §4.1
D3: the pipeline must say it could not resolve, never fabricate).
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from dotenv import load_dotenv  # noqa: E402

load_dotenv(ROOT / ".env")

from bionym.ask import ask_graph  # noqa: E402
from bionym.clients.ncbi import NcbiClient  # noqa: E402
from bionym.clients.veupathdb import VEuPathDBClient  # noqa: E402
from bionym.jev import JevClient, JevQuotaError  # noqa: E402
from bionym.llm import LlmClient  # noqa: E402
from bionym.resolver import Resolver  # noqa: E402

PANEL = Path(__file__).parent / "data" / "ask_stratified_panel.yaml"
RESULTS = Path(__file__).parent / "results"
EVAL_CACHE = Path(__file__).parent / ".cache"
EXIT_QUOTA = 75
REGIMES = {"t0": 0.0, "t07": 0.7}
# one full-rebuild anchor per stratum (plan §4.2 R1)
ANCHORS = {"PF3D7_1343700", "PVX_081515", "cgd4_440", "ama1"}

Q1 = ("Which exact gene and organism does {id} refer to, what aliases "
      "or cross-database accessions map to it, and in which genome "
      "assemblies is it annotated?")
Q2 = ("What conserved protein domains, functional annotations, and "
      "orthologs across other species are established for {id}?")
Q3 = ("In which public transcriptomic or functional-genomics datasets "
      "and experimental conditions has {id} been measured, and what "
      "findings were reported?")

BARE_SYSTEM = (
    "You are a bioinformatics expert. Answer concisely and factually "
    "about the gene identifier asked, citing specific database "
    "accessions where you know them. If the evidence does not exist or "
    "you are unsure, say 'unknown' or 'insufficient evidence' rather "
    "than guessing."
)

# accession-ish tokens — dataset/assembly/protein/locus patterns
ACCESSION_RE = re.compile(
    r"\b(GCA_\d+(?:\.\d+)?|GCF_\d+(?:\.\d+)?|GSE\d+|GDS\d+|GSM\d+"
    r"|SR[RPX]\d+|ERR\d+|DRR\d+|PRJ[NDE][AB]\d+"
    r"|E-MTAB-\d+|E-GEOD-\d+|E-MEXP-\d+"
    r"|PF3D7_\d+|PVX_\d+|TGME49_\d+|TGGT1_\d+|AGAP\d+|AAEL\d+"
    r"|A[AF]UA_\w+|B9J08_\w+|ISCW\w+|Smp_\w+|cgd\d_\d+|BMR1_\w+|BPSL\w+"
    r"|Rv\d{4}[a-z]?|Tb927\.[\d.]+|LmjF[\w.]+|CNAG_\d+|CIMG_\w+"
    r"|ENS[GTP]\d+|NM_\d+(?:\.\d+)?|N[MPXY]P_\d+(?:\.\d+)?"
    r"|[OPQ][0-9][A-Z0-9]{3}[0-9]|A0A[A-Z0-9]{7}"
    r"|[1-9][a-z0-9]{3}(?=\s|,|\.|\)|:)|pmid:?\s?\d{5,}"
    r"|AT[1-5]G\d{5}|WBGene\d{8})\b")

ABSTAIN_RE = re.compile(
    r"\b(unknown|insufficient evidence|not established|no evidence"
    r"|not known|cannot determine|no such|not been reported"
    r"|not reported|not available|uncharacterized)\b", re.I)


def _accessions(text: str) -> set[str]:
    return {m.group(0).rstrip(",.):") for m in
            ACCESSION_RE.finditer(text or "")}


def _abstains(text: str) -> bool:
    return bool(ABSTAIN_RE.search(text or ""))


def _parse_answer(raw: str) -> dict:
    """LLM JSON {answer, cited_nodes} or prose fallback."""
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return {"answer": str(data.get("answer", raw)),
                    "cited_nodes": data.get("cited_nodes") or []}
    except (json.JSONDecodeError, TypeError):
        pass
    return {"answer": raw, "cited_nodes": []}


def _ask_once(resolver, jev, llm, target, qkey, question,
              graphs: dict) -> dict:
    """One (target, question) under one arm. Grounded resolves lazily."""
    ident = target["id"]
    graph = graphs.get(ident)
    if graph is None:
        graph = resolver.resolve(
            ident, depth=6,
            propose=not target.get("refuse_expected"),
            summarize=not target.get("refuse_expected"))
        graphs[ident] = graph
    refused = graph.metadata.get("outcome") == "refused"
    try:
        out = ask_graph(jev, llm, question, graph)
    except JevQuotaError:
        raise
    except Exception as e:  # noqa: BLE001 — row is the record
        return {"refused": refused, "error": f"{type(e).__name__}: {e}"}
    return {"refused": refused, **out}


def _bare_once(llm, ident, question) -> dict:
    system, user = BARE_SYSTEM, f"Gene identifier: {ident}\n\n{question}"
    try:
        raw = llm.complete(system, user)
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    return _parse_answer(raw)


def _write_row(fh, row: dict) -> None:
    fh.write(json.dumps(row) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def _done_keys(path: Path) -> set[str]:
    keys = set()
    for l in path.read_text().splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("kind") == "qa":
            keys.add("|".join(str(r[k]) for k in
                              ("id", "q", "regime", "rep", "arm")))
    return keys


# -- analysis ---------------------------------------------------------------

def _jaccard(a: set, b: set) -> float | None:
    if not a and not b:
        return None
    union = a | b
    return round(len(a & b) / len(union), 4) if union else None


def analyze(rows_path: Path) -> dict:
    rows = [json.loads(l) for l in rows_path.read_text().splitlines()
            if l.strip() and '"kind": "qa"' in l]
    groups = defaultdict(list)
    for r in rows:
        groups[(r["id"], r["q"], r["regime"], r["arm"])].append(r)

    per_group = {}
    for (ident, q, regime, arm), rs in sorted(groups.items()):
        answers = [r.get("answer_text", "") for r in rs]
        accs = [_accessions(a) for a in answers]
        nodes = [set(r.get("cited_nodes") or []) for r in rs]
        acc_j = [_jaccard(a, b) for a, b in
                 itertools.combinations(accs, 2)]
        acc_j = [x for x in acc_j if x is not None]
        node_j = [_jaccard(a, b) for a, b in
                  itertools.combinations(nodes, 2)]
        node_j = [x for x in node_j if x is not None]
        per_group["|".join([ident, q, regime, arm])] = {
            "n": len(rs),
            "accession_jaccard": round(sum(acc_j) / len(acc_j), 4)
            if acc_j else None,
            "cited_nodes_jaccard": round(sum(node_j) / len(node_j), 4)
            if node_j else None,
            "abstain_rate": round(
                sum(_abstains(a) for a in answers) / len(answers), 4),
            "n_accessions_median": sorted(len(a) for a in accs)[len(accs) // 2]
            if accs else 0,
        }
    # strata rollups: grounded vs bare stability
    strata = defaultdict(lambda: defaultdict(list))
    target_strata = {t["id"]: t["stratum"] for t in
                     yaml.safe_load(PANEL.read_text())["targets"]}
    for key, m in per_group.items():
        ident, q, regime, arm = key.split("|")
        strata[(target_strata.get(ident), regime)][arm].append(m)
    rollup = {}
    for (stratum, regime), arms in sorted(strata.items()):
        rollup[f"{stratum}|{regime}"] = {
            arm: {
                "median_acc_jaccard": sorted(
                    m["accession_jaccard"] for m in ms
                    if m["accession_jaccard"] is not None
                )[max(0, len([m for m in ms
                              if m["accession_jaccard"] is not None]) // 2)]
                if any(m["accession_jaccard"] is not None for m in ms)
                else None,
                "mean_abstain_rate": round(
                    sum(m["abstain_rate"] for m in ms) / len(ms), 4),
            }
            for arm, ms in arms.items()
        }
    return {"groups": per_group, "rollup": rollup}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--regime", choices=sorted(REGIMES), default="t0")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--target", help="one panel id")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--resume", metavar="FILE")
    ap.add_argument("--analyze", metavar="ROWS",
                    help="analyze a completed rows file instead of running")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.analyze:
        print(json.dumps(analyze(Path(args.analyze)), indent=1))
        return

    panel = yaml.safe_load(PANEL.read_text())["targets"]
    if args.target:
        panel = [t for t in panel if t["id"] == args.target]
    if not panel:
        sys.exit("no targets matched")

    temperature = REGIMES[args.regime]
    shared = str(EVAL_CACHE)
    rep_cache = str(EVAL_CACHE / f"suite4_{args.regime}_rep{args.rep}")

    jev = JevClient(mock=args.mock, cache_dir=shared)
    llm_shared = LlmClient(mock=args.mock, cache_dir=shared,
                           temperature=temperature)
    # Ask generations run on the isolated per-rep client — graphs resolve
    # on the shared one so JEV/DB work is paid once.
    llm_rep = LlmClient(mock=args.mock, cache_dir=rep_cache,
                        temperature=temperature)
    resolver = Resolver(
        jev=jev,
        ncbi=NcbiClient(cache_dir=shared),
        veupathdb=VEuPathDBClient(cache_dir=shared),
        llm=llm_shared,
    )
    print(f"[config] regime={args.regime} T={temperature} rep={args.rep} "
          f"targets={len(panel)} mock={args.mock}")

    RESULTS.mkdir(exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    if args.resume:
        out_path = Path(args.resume)
        done = _done_keys(out_path)
        mode = "a"
    else:
        out_path = Path(args.out) if args.out else (
            RESULTS / f"{stamp}-suite4-{args.regime}-rep{args.rep}.jsonl")
        done = set()
        mode = "w"

    graphs: dict = {}
    with out_path.open(mode) as fh:
        if mode == "w":
            _write_row(fh, {"kind": "config", "ts": stamp,
                            "regime": args.regime, "rep": args.rep,
                            "temperature": temperature,
                            "mock": args.mock})
        else:
            _write_row(fh, {"kind": "resumed", "ts": stamp})
        for target in panel:
            ident = target["id"]
            qs = {"Q1": Q1.format(id=ident), "Q2": Q2.format(id=ident),
                  "Q3": Q3.format(id=ident), "Q4": target["q4"]}
            # full-rebuild anchors resolve under the rep cache so R1
            # measures upstream construction variance (plan §4.2)
            local_resolver = resolver
            if ident in ANCHORS:
                local_resolver = Resolver(
                    jev=JevClient(mock=args.mock, cache_dir=rep_cache),
                    ncbi=NcbiClient(cache_dir=rep_cache),
                    veupathdb=VEuPathDBClient(cache_dir=rep_cache),
                    llm=llm_rep,
                )
            for qkey, question in qs.items():
                for arm in ("grounded", "bare"):
                    key = "|".join(
                        str(x) for x in (ident, qkey, args.regime,
                                         args.rep, arm))
                    if key in done:
                        continue
                    t0 = time.monotonic()
                    try:
                        if arm == "grounded":
                            res = _ask_once(local_resolver, jev, llm_rep,
                                            target, qkey, question, graphs)
                        else:
                            res = _bare_once(llm_rep, ident, question)
                    except JevQuotaError as e:
                        _write_row(fh, {"kind": "stopped", "reason": str(e),
                                        "key": key, "ts": stamp})
                        print(f"\n[quota] JEV exhausted at {key} — "
                              f"re-run with --resume {out_path}", flush=True)
                        sys.exit(EXIT_QUOTA)
                    wall = round(time.monotonic() - t0, 1)
                    text = res.get("answer", "")
                    row = {
                        "kind": "qa", "id": ident,
                        "stratum": target["stratum"], "q": qkey,
                        "question": question, "regime": args.regime,
                        "rep": args.rep, "arm": arm,
                        "refused": res.get("refused"),
                        "answer_text": text,
                        "answer_abstains": _abstains(text),
                        "accessions": sorted(_accessions(text)),
                        "cited_nodes": res.get("cited_nodes") or [],
                        "scored": res.get("scored"),
                        "warning": res.get("warning", ""),
                        "error": res.get("error", ""),
                        "wall_s": wall,
                    }
                    _write_row(fh, row)
                    print(f"  {ident} {qkey} {arm}: {wall}s"
                          + (" [refused]" if res.get("refused") else "")
                          + (f" [ERR {res['error'][:60]}]"
                             if res.get("error") else ""), flush=True)
    print(f"[done] {out_path}")


if __name__ == "__main__":
    main()
