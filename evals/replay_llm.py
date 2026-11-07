#!/usr/bin/env python3
"""Replay recorded JEV asks through an ordinary chat LLM acting as judge.

  LLM_API_KEY=... LLM_MODEL=meta-llama/llama-3.1-8b-instruct \
      python evals/replay_llm.py --tag llama8b

Reads asks.jsonl (same {gene, stage, state, questions, jev_answers}
format dump_asks.py produces and the Kaggle systemone kernels replay),
sends each ask to an OpenAI-compatible chat endpoint, instructs the
model to return JEV-shaped {"answers": {qid: {"noul"|"choice"|"score":
...}}}, then diffs typed answers against Jev's — same metrics as
kaggle/*/kernel.py (keep diff_one and the summary block in sync).

Config (env or flags; repo .env is auto-loaded):
  LLM_API_KEY / OPENROUTER_API_KEY  required
  LLM_BASE_URL    default https://openrouter.ai/api/v1
  LLM_MODEL       default meta-llama/llama-3.1-8b-instruct (128k ctx)
  LLM_TIMEOUT     per-request seconds, default 180

Unlike the systemone servers, a chat LLM's context covers the full
state (the largest asks are ~14k tokens, well inside 128k) — but it
was never trained for calibrated Noul probabilities, which is exactly
what this comparison measures.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Minimal .env loader (no dependency): KEY=VAL, comments/#, quotes."""
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
DEFAULT_MODEL = os.environ.get(
    "LLM_MODEL") or "meta-llama/llama-3.1-8b-instruct"
TIMEOUT_S = float(os.environ.get("LLM_TIMEOUT", "180"))

SYSTEM = """\
You are a System One decision model. You are given a knowledge "state"
(JSON) and a set of typed questions about it (JSON object keyed by
question id). Answer EVERY question, keyed by its exact id.

Question types and required answer shapes:
- {"type": "noul", "instructions": ...} ->
  {"noul": <float 0.0-1.0>}: probability that the proposition in
  instructions is true / should be accepted, given the state evidence.
- {"type": "choice", "instructions": ..., "criteria": [...]} ->
  {"choice": "<one of the criteria strings, verbatim>"}.
- {"type": "score", "instructions": ...} ->
  {"score": <number>}: the numeric score the instructions ask for.

Rules: judge strictly from the supplied state; do not invent evidence;
use the full 0-1 range honestly rather than clustering near 0.5;
choose exactly one criterion verbatim for choice questions.

Reply with STRICT JSON only — {"answers": {"<qid>": {...}, ...}} — no
markdown fences, no commentary."""


def build_messages(ask: dict) -> list[dict]:
    user = (
        "STATE:\n" + json.dumps(ask["state"]) +
        "\n\nQUESTIONS:\n" + json.dumps(ask["questions"]))
    return [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": user}]


def _extract_json(text: str) -> dict | None:
    """Tolerate fences/prose: grab the outermost {...} block."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def _coerce_answers(data: dict, questions: dict) -> dict:
    """Keep only plausible typed answers; coerce str->float where sane."""
    out = {}
    answers = data.get("answers", data) if isinstance(data, dict) else {}
    for qid, q in questions.items():
        a = answers.get(qid) if isinstance(answers, dict) else None
        if not isinstance(a, dict):
            continue
        if "noul" in a or "noul" == q.get("type"):
            v = a.get("noul", a.get("probability", a.get("p")))
            try:
                out[qid] = {"noul": min(1.0, max(0.0, float(v)))}
            except (TypeError, ValueError):
                continue
        elif "choice" in a or "choice" == q.get("type"):
            v = a.get("choice")
            if isinstance(v, str):
                out[qid] = {"choice": v}
        elif "score" in a or "score" == q.get("type"):
            try:
                out[qid] = {"score": float(a.get("score"))}
            except (TypeError, ValueError):
                continue
    return out


def chat_once(url: str, model: str, messages: list[dict]) -> dict:
    body = json.dumps({
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "max_tokens": 4096,
    }).encode()
    req = urllib.request.Request(
        url + "/chat/completions", data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {API_KEY}"})
    with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
        return json.loads(r.read())


def llm_ask(url: str, model: str, ask: dict) -> dict:
    """One ask -> {"answers": {...}, "ms": int, "raw": str}
    or {"error": ..., "raw": str}."""
    t0 = time.time()
    msgs = build_messages(ask)
    try:
        for attempt in range(2):
            resp = chat_once(url, model, msgs)
            raw = (resp.get("choices") or [{}])[0] \
                .get("message", {}).get("content", "")
            data = _extract_json(raw)
            if data is not None:
                answers = _coerce_answers(data, ask["questions"])
                if answers:
                    return {"answers": answers,
                            "ms": int((time.time() - t0) * 1000),
                            "raw": raw[:4000]}
            # repair pass: complain once, then give up
            msgs = msgs + [
                {"role": "assistant", "content": raw[:2000]},
                {"role": "user", "content":
                 "That was not valid JSON with an \"answers\" object "
                 "covering every question id. Return STRICT JSON only."}]
        return {"error": "bad_json", "detail": "no parseable answers",
                "raw": raw[:4000]}
    except urllib.error.HTTPError as e:
        return {"error": f"http_{e.code}",
                "detail": e.read()[:300].decode("utf-8", "replace")}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"error": "timeout_or_conn", "detail": str(e)[:300]}


def diff_one(jev_ans: dict, llm_ans: dict) -> dict:
    """Per-question deltas. Unknown/extra fields pass through raw."""
    d = {}
    if "noul" in jev_ans or "noul" in llm_ans:
        j, l = jev_ans.get("noul"), llm_ans.get("noul")
        d["qtype"] = "noul"
        d["jev"], d["laya"] = j, l
        if j is not None and l is not None:
            d["abs_delta"] = round(abs(j - l), 4)
            # 0.5 is the action threshold everywhere noul gates a pick
            d["flip_0.5"] = (j >= 0.5) != (l >= 0.5)
    elif "choice" in jev_ans or "choice" in llm_ans:
        d["qtype"] = "choice"
        d["jev"] = jev_ans.get("choice")
        d["laya"] = llm_ans.get("choice")
        d["agree"] = d["jev"] == d["laya"]
    elif "score" in jev_ans or "score" in llm_ans:
        d["qtype"] = "score"
        d["jev"] = jev_ans.get("score")
        d["laya"] = llm_ans.get("score")
        if isinstance(d["jev"], (int, float)) and isinstance(
                d["laya"], (int, float)):
            d["abs_delta"] = round(abs(d["jev"] - d["laya"]), 4)
    else:
        d["qtype"] = "other"
        d["jev"], d["laya"] = jev_ans, llm_ans
    return d


def _mean(vs):
    vs = list(vs)
    return round(sum(vs) / len(vs), 4) if vs else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asks", default="evals/asks-claimsplit.jsonl")
    ap.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"chat model id (default env LLM_MODEL or "
                         f"{DEFAULT_MODEL})")
    ap.add_argument("--tag", default=None,
                    help="result filename tag; default derived from model")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="replay only the first N asks (smoke tests)")
    ap.add_argument("--out-dir", default="evals/results")
    args = ap.parse_args()

    if not API_KEY:
        raise SystemExit(
            "LLM_API_KEY (or OPENROUTER_API_KEY) unset — get one at "
            "https://openrouter.ai/keys")

    asks_path = Path(args.asks)
    asks = [json.loads(l) for l in asks_path.read_text().splitlines()
            if l.strip()]
    if args.limit:
        asks = asks[:args.limit]
    url = BASE_URL.rstrip("/")
    tag = args.tag or re.sub(r"[^a-z0-9]+", "",
                             args.model.split("/")[-1].lower())[:16]
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f"{date.today():%Y%m%d}-{tag}-claimsplit"
    result_path = stem.with_suffix(".results.json")
    diffs_path = stem.with_suffix(".diffs.jsonl")
    raw_path = stem.with_suffix(".raw.jsonl")
    print(f"[asks] {len(asks)} -> {url} model={args.model}", flush=True)

    results_by_i: dict[int, dict] = {}
    t0 = time.time()
    done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(llm_ask, url, args.model, ask): i
                for i, ask in enumerate(asks)}
        for fut in as_completed(futs):
            i = futs[fut]
            results_by_i[i] = fut.result()
            done += 1
            if done % 25 == 0 or done == len(asks):
                print(f"[{done}/{len(asks)}]", flush=True)

    rows, raw_rows = [], []
    for i, ask in enumerate(asks):
        res = results_by_i[i]
        base = {"gene": ask["gene"], "stage": ask["stage"], "ask_i": i}
        if res.get("raw"):
            raw_rows.append({**base, "raw": res["raw"]})
        if "error" in res:
            rows.append({**base, "error": res["error"],
                         "detail": res.get("detail", "")})
            continue
        for qid, lans in res["answers"].items():
            jans = ask["jev_answers"].get(qid, {})
            rows.append({**base, "qid": qid, "ms": res["ms"],
                         **diff_one(jans, lans)})

    with diffs_path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    with raw_path.open("w") as fh:
        for r in raw_rows:
            fh.write(json.dumps(r) + "\n")

    by_stage = defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append(r)
    summary = {}
    for stage, rs in sorted(by_stage.items()):
        errs = [r for r in rs if "error" in r]
        nouls = [r for r in rs if r.get("qtype") == "noul"]
        choices = [r for r in rs if r.get("qtype") == "choice"]
        scores = [r for r in rs if r.get("qtype") == "score"]
        summary[stage] = {
            "questions": len(rs),
            "ask_errors": len(errs),
            "noul": {
                "n": len(nouls),
                "mean_abs_delta": _mean(
                    r.get("abs_delta") for r in nouls
                    if r.get("abs_delta") is not None),
                "flip_0.5_rate": _mean(float(r.get("flip_0.5", False))
                                       for r in nouls),
            },
            "choice": {
                "n": len(choices),
                "agree_rate": _mean(float(r.get("agree", False))
                                    for r in choices),
            },
            "score": {
                "n": len(scores),
                "mean_abs_delta": _mean(
                    r.get("abs_delta") for r in scores
                    if r.get("abs_delta") is not None),
            },
        }

    nouls = [r for r in rows if r.get("qtype") == "noul"]
    choices = [r for r in rows if r.get("qtype") == "choice"]
    results = {
        "n_asks": len(asks),
        "n_question_rows": len(rows),
        "n_errors": sum(1 for r in rows if "error" in r),
        "wall_s": int(time.time() - t0),
        "laya_model": args.model,
        "overall": {
            "noul_n": len(nouls),
            "noul_mean_abs_delta": _mean(
                r.get("abs_delta") for r in nouls
                if r.get("abs_delta") is not None),
            "noul_flip_0.5_rate": _mean(float(r.get("flip_0.5", False))
                                        for r in nouls),
            "choice_n": len(choices),
            "choice_agree_rate": _mean(float(r.get("agree", False))
                                       for r in choices),
        },
        "by_stage": summary,
    }
    result_path.write_text(json.dumps(results, indent=1))
    print(json.dumps(results["overall"], indent=1), flush=True)
    print(f"wrote {result_path}, {diffs_path}, {raw_path}", flush=True)


if __name__ == "__main__":
    main()
