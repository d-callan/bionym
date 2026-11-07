#!/usr/bin/env python3
"""Time JEV itself on the recorded asks.

  python evals/time_jev.py --limit 10
  python evals/time_jev.py --workers 8

Posts each ask's {state, questions} to the real systemone endpoint
exactly as bionym's JevClient does (same body shape, jev-latest model)
and records per-ask latency plus the usage counters the API returns.
Answers are kept too — cheap sanity check that JEV reproduces its own
recorded answers (cache miss vs recording-time answers may drift).

Env: TYPESAFE_API_KEY (repo .env auto-loaded). JEV_BASE_URL/JEV_MODEL
repoint like the real client.
"""
from __future__ import annotations

import argparse
import json
import os
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

API_KEY = os.environ.get("TYPESAFE_API_KEY", "")
BASE = os.environ.get("JEV_BASE_URL", "https://api.typesafe.ai").rstrip("/")
URL = BASE if BASE.endswith("/v1/systemone") else f"{BASE}/v1/systemone"
MODEL = os.environ.get("JEV_MODEL") or "jev-latest"
TIMEOUT_S = float(os.environ.get("JEV_TIMEOUT", "120"))


def jev_ask(ask: dict) -> dict:
    body = json.dumps({
        "state": ask["state"],
        "model": MODEL,
        "questions": ask["questions"],
    }).encode()
    t0 = time.time()
    try:
        req = urllib.request.Request(
            URL, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {API_KEY}"})
        with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
            data = json.loads(r.read())
        return {"ms": int((time.time() - t0) * 1000),
                "answers": data.get("answers", {}),
                "usage": data.get("usage", {})}
    except urllib.error.HTTPError as e:
        return {"error": f"http_{e.code}",
                "ms": int((time.time() - t0) * 1000),
                "detail": e.read()[:300].decode("utf-8", "replace")}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"error": "timeout_or_conn",
                "ms": int((time.time() - t0) * 1000),
                "detail": str(e)[:300]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asks", default=str(ROOT / "evals" / "asks-claimsplit.jsonl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--tag", default="jev")
    ap.add_argument("--out-dir", default=str(ROOT / "evals" / "results"))
    args = ap.parse_args()
    if not API_KEY:
        sys.exit("set TYPESAFE_API_KEY")

    asks = [json.loads(l) for l in open(args.asks)]
    if args.limit:
        asks = asks[: args.limit]
    print(f"[asks] {len(asks)} -> {URL} model={MODEL}", flush=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{date.today():%Y%m%d}-{args.tag}"
    rows_path = out_dir / f"{stamp}-timing.jsonl"
    res_path = out_dir / f"{stamp}-timing.results.json"

    rows = []
    t0 = time.time()
    done = 0
    with open(rows_path, "w") as f:
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(jev_ask, a): (i, a) for i, a in enumerate(asks)}
            for fut in as_completed(futs):
                i, a = futs[fut]
                r = fut.result()
                r.update({"ask_i": i, "gene": a["gene"], "stage": a["stage"],
                          "n_questions": len(a["questions"])})
                if "answers" in r:
                    # agreement with the recorded JEV answers
                    rec = a["jev_answers"]
                    n_same = 0
                    for qid, ans in r["answers"].items():
                        j = (rec.get(qid) or {})
                        for k in ("noul", "choice", "score"):
                            if k in ans and ans.get(k) == j.get(k):
                                n_same += 1
                    r["reproduced"] = n_same
                f.write(json.dumps(r) + "\n")
                rows.append(r)
                done += 1
                if done % 25 == 0 or done == len(asks):
                    print(f"[{done}/{len(asks)}]", flush=True)

    ms = sorted(r["ms"] for r in rows)
    ok = [r for r in rows if "error" not in r]
    by_stage: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_stage[r["stage"]].append(r)

    def _q(v, p):
        return v[int(len(v) * p)] if v else None

    summ = {
        "model": MODEL, "n_asks": len(rows),
        "n_errors": sum(1 for r in rows if "error" in r),
        "wall_s": int(time.time() - t0),
        "ms_p50": _q(ms, 0.5), "ms_p90": _q(ms, 0.9), "ms_max": ms[-1] if ms else None,
        "in_tokens_total": sum(r.get("usage", {}).get("input_tokens", 0) for r in ok),
        "out_tokens_total": sum(r.get("usage", {}).get("output_tokens", 0) for r in ok),
        "reproduced_answers": sum(r.get("reproduced", 0) for r in ok),
        "by_stage": {
            s: {"n": len(v),
                "ms_p50": _q(sorted(r["ms"] for r in v), 0.5),
                "errors": sum(1 for r in v if "error" in r)}
            for s, v in sorted(by_stage.items())
        },
    }
    res_path.write_text(json.dumps(summ, indent=1))
    print(json.dumps(summ, indent=1))
    print(f"wrote {res_path.name}, {rows_path.name}")


if __name__ == "__main__":
    main()
