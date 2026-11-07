#!/usr/bin/env python3
"""Regenerate bionym claim proposals through a chat LLM and diff the
resulting question sets against the recorded asks.

  LLM_API_KEY=... LLM_BASE_URL=https://ai.tejas.tacc.utexas.edu/v1 \
      python evals/replay_generate.py --model DeepSeek-V3.2 --tag tacc-dsv32

Two generation paths are replayed, using the same prompt builders bionym
uses (imported by path so no package deps leak in):

- stage=proposals      state.source_texts {node: numbered text} ->
  proposals.build_prompt(kind, text, gene) where kind is 'dataset' for
  dataset: nodes and 'publication' otherwise. Parsed proposals get JEV
  questions via proposals.build_questions.
- stage=data_proposals state {gene, dataset{name,summary}, data} ->
  proposals.build_data_prompt + build_data_questions.

The recorded ask's own state.proposals / questions are the reference
set. Per-ask diff: question counts, proposal-type mix, and claim-text
overlap (difflib best-match in both directions).

Config: same env contract as replay_llm.py (LLM_API_KEY, LLM_BASE_URL,
LLM_MODEL, LLM_TIMEOUT; repo .env auto-loaded).
"""
from __future__ import annotations

import argparse
import difflib
import importlib.util
import json
import os
import re
import statistics as st
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    env = ROOT / ".env"
    if not env.exists():
        return
    for line in env.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip("\"'"))


_load_dotenv()

API_KEY = os.environ.get("LLM_API_KEY") or os.environ.get("OPENROUTER_API_KEY")
BASE_URL = os.environ.get("LLM_BASE_URL", "https://openrouter.ai/api/v1")
TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT", "180"))


def _load_proposals():
    """Import src/bionym/proposals.py directly (it is stdlib-only, but
    `import bionym` would run the package __init__ and its deps)."""
    spec = importlib.util.spec_from_file_location(
        "proposals", ROOT / "src" / "bionym" / "proposals.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


P = _load_proposals()

_NUM = re.compile(r"^\[\d+\]\s*")


def _units(numbered: str) -> list[str]:
    """Recorded source texts are already '[n] unit' numbered — strip the
    numbers back to plain units for citation validation."""
    return [_NUM.sub("", ln) for ln in numbered.splitlines() if ln.strip()]


def _content_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(p.get("text", "") for p in content
                       if isinstance(p, dict))
    return ""


def chat(url: str, model: str, system: str, user: str) -> str:
    """One completion -> content string. Raises on transport errors."""
    body = json.dumps({
        "model": model,
        "messages": [{"role": "system", "content": system},
                     {"role": "user", "content": user}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": 4096,
    }).encode()
    req = urllib.request.Request(
        url + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        resp = json.loads(r.read())
    return _content_text(
        (resp.get("choices") or [{}])[0].get("message", {}).get("content"))


def _kind(node: str) -> str:
    return "dataset" if node.startswith("dataset:") else "publication"


def regenerate(ask: dict) -> dict:
    """Replay the generation call(s) for one recorded ask.

    -> {"proposals": [...], "questions": {...}, "ms": int}
    or {"error": ...}. Generated proposals carry their source node."""
    t0 = time.time()
    stt = ask["state"]
    stage = ask["stage"]
    try:
        if stage == "proposals":
            gene = (stt.get("source_gene") or {}).get("symbol", "")
            props = []
            for node, text in stt.get("source_texts", {}).items():
                system, user = P.build_prompt(_kind(node), text, gene)
                out = chat(BASE_URL, MODEL, system, user)
                for p in P.parse(out, _units(text)):
                    p["node"] = node
                    props.append(p)
            questions = P.build_questions(props)
        elif stage == "data_proposals":
            meta = {
                "gene": stt.get("gene"),
                "label": (stt.get("dataset") or {}).get("name"),
                "summary": (stt.get("dataset") or {}).get("summary"),
            }
            system, user = P.build_data_prompt(meta, stt.get("data", ""))
            out = chat(BASE_URL, MODEL, system, user)
            props = P.parse(out, _units(stt.get("data", "")))
            node = (stt.get("proposals") or [{}])[0].get("node", "")
            for p in props:
                p["node"] = node
            questions = P.build_data_questions(props)
        else:
            return {"error": "unsupported_stage"}
        return {"proposals": props, "questions": questions,
                "ms": int((time.time() - t0) * 1000)}
    except urllib.error.HTTPError as e:
        return {"error": f"http_{e.code}",
                "detail": e.read()[:300].decode("utf-8", "replace")}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"error": "timeout_or_conn", "detail": str(e)[:300]}
    except Exception as e:                       # keep the run alive
        return {"error": "exception",
                "detail": f"{type(e).__name__}: {e}"[:300]}


def _norm(s: str) -> str:
    return " ".join(str(s).lower().split())


def claim_overlap(rec: list[str], gen: list[str]) -> dict:
    """Symmetric best-match similarity between claim-text sets.

    recall = for each recorded claim, its best generated match (mean).
    precision = same in the other direction. matched pairs use a greedy
    best-first assignment at >= MATCH threshold."""
    rec_n, gen_n = [_norm(c) for c in rec], [_norm(c) for c in gen]
    if not rec_n or not gen_n:
        return {"recall": 0.0, "precision": 0.0, "matched": 0,
                "pairs": []}
    scores = {}
    for i, a in enumerate(rec_n):
        for j, b in enumerate(gen_n):
            r = difflib.SequenceMatcher(None, a, b).ratio()
            scores[(i, j)] = r
    rec_best = {i: max(scores[i, j] for j in range(len(gen_n)))
                for i in range(len(rec_n))}
    gen_best = {j: max(scores[i, j] for i in range(len(rec_n)))
                for j in range(len(gen_n))}
    # greedy match: strongest pairs first, one-to-one
    pairs, used_r, used_g = [], set(), set()
    for (i, j), r in sorted(scores.items(), key=lambda kv: -kv[1]):
        if r < MATCH or i in used_r or j in used_g:
            continue
        used_r.add(i)
        used_g.add(j)
        pairs.append({"rec": rec[i], "gen": gen[j], "sim": round(r, 3)})
    return {"recall": round(st.mean(rec_best.values()), 4),
            "precision": round(st.mean(gen_best.values()), 4),
            "matched": len(pairs), "pairs": pairs}


MATCH = 0.8  # difflib ratio >= this counts as "same claim"


def diff_ask(ask: dict, gen: dict) -> dict:
    rec_props = ask["state"].get("proposals", [])
    rec_claims = [p.get("claim", "") for p in rec_props]
    gen_props = gen.get("proposals", [])
    gen_claims = [p.get("claim", "") for p in gen_props]
    ov = claim_overlap(rec_claims, gen_claims)
    return {
        "ask_i": ask["_i"], "gene": ask["gene"], "stage": ask["stage"],
        "n_recorded": len(rec_claims), "n_generated": len(gen_claims),
        "rec_types": dict(Counter(p.get("type") for p in rec_props)),
        "gen_types": dict(Counter(p.get("type") for p in gen_props)),
        "claim_recall": ov["recall"], "claim_precision": ov["precision"],
        "matched": ov["matched"],
        "unmatched_rec": [c for i, c in enumerate(rec_claims)
                          if all(p["rec"] != c for p in ov["pairs"])],
        "unmatched_gen": [c for j, c in enumerate(gen_claims)
                          if all(p["gen"] != c for p in ov["pairs"])],
        "pairs": ov["pairs"],
    }


def main() -> None:
    global MODEL
    ap = argparse.ArgumentParser()
    ap.add_argument("--asks", default=str(ROOT / "evals" / "asks-claimsplit.jsonl"))
    ap.add_argument("--model", default=os.environ.get("LLM_MODEL", ""))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out-dir", default=str(ROOT / "evals" / "results"))
    args = ap.parse_args()
    MODEL = args.model
    tag = args.tag or re.sub(r"[^A-Za-z0-9]+", "", MODEL.split("/")[-1])
    if not API_KEY:
        sys.exit("set LLM_API_KEY (or OPENROUTER_API_KEY)")
    if not MODEL:
        sys.exit("set --model or LLM_MODEL")

    asks = []
    for i, line in enumerate(open(args.asks)):
        a = json.loads(line)
        if a["stage"] in ("proposals", "data_proposals"):
            a["_i"] = i
            asks.append(a)
    if args.limit:
        asks = asks[: args.limit]
    print(f"[asks] {len(asks)} -> {BASE_URL} model={MODEL}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{date.today():%Y%m%d}-{tag}"
    diffs_path = out_dir / f"{stamp}-gen-diffs.jsonl"
    res_path = out_dir / f"{stamp}-gen-results.json"
    raw_path = out_dir / f"{stamp}-gen-raw.jsonl"

    rows, errors = [], []
    t0 = time.time()
    done = 0
    with open(diffs_path, "w") as df, open(raw_path, "w") as rf:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(regenerate, a): a for a in asks}
            for fut in as_completed(futs):
                a = futs[fut]
                gen = fut.result()
                if "error" in gen:
                    row = {"ask_i": a["_i"], "gene": a["gene"],
                           "stage": a["stage"], **gen}
                    errors.append(row)
                else:
                    row = diff_ask(a, gen)
                df.write(json.dumps(row) + "\n")
                rf.write(json.dumps({"ask_i": a["_i"], **gen}) + "\n")
                done += 1
                if done % 25 == 0 or done == len(asks):
                    print(f"[{done}/{len(asks)}]", flush=True)

    ok = [r for r in rows_or_diffs(diffs_path)]
    summ = {
        "model": MODEL, "n_asks": len(asks), "n_errors": len(errors),
        "wall_s": int(time.time() - t0),
        "by_stage": {},
    }
    for stage in ("proposals", "data_proposals"):
        sr = [r for r in ok if r["stage"] == stage]
        if not sr:
            continue
        summ["by_stage"][stage] = {
            "asks": len(sr),
            "rec_mean_n": round(st.mean(r["n_recorded"] for r in sr), 2),
            "gen_mean_n": round(st.mean(r["n_generated"] for r in sr), 2),
            "claim_recall": round(st.mean(r["claim_recall"] for r in sr), 4),
            "claim_precision": round(st.mean(r["claim_precision"] for r in sr), 4),
            "matched_total": sum(r["matched"] for r in sr),
            "empty_gen": sum(1 for r in sr if r["n_generated"] == 0),
        }
    res_path.write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ["by_stage"], indent=1))
    print(f"wrote {res_path.name}, {diffs_path.name}, {raw_path.name}")


def rows_or_diffs(path: Path):
    for line in open(path):
        r = json.loads(line)
        if "error" not in r:
            yield r


if __name__ == "__main__":
    main()
