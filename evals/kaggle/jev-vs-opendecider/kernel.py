"""Kaggle CPU kernel: replay recorded JEV asks through OpenDecider.

Input: asks.jsonl (one {gene, stage, state, questions, jev_answers}
per line, produced by bionym/evals/dump_asks.py and uploaded as a
Kaggle dataset — set its slug in kernel-metadata.json dataset_sources).

Steps: pip install opendecider, boot `opendecider serve`, POST every
ask to /v1/systemone, diff the typed answers against Jev's ->
results.json (summary) + diffs.jsonl (one row per question).

Context: opendecider-nano has a 2,048-token input budget — much
smaller than Laya/Von (~8k). Large states are state-truncated by the
server (answers carry "truncated": true); proposals_triage asks will
lose most of their state. That bias is part of the comparison.
"""

import json
import os
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

try:
    socket.create_connection(("huggingface.co", 443), timeout=10)
except OSError:
    raise SystemExit(
        "NO INTERNET on this kernel VM — verify phone number at "
        "kaggle.com/settings (required for internet), then re-push.")

WORK = Path("/kaggle/working")
RESULT = WORK / "results.json"
DIFFS = WORK / "diffs.jsonl"

MODEL = "opendecider-nano"   # server ignores the model field, kept for logging
# opendecider serve may or may not take --port; launch bare and probe
# common ports in order of likelihood
PORTS = [8000, 8080, 5000]
HEALTH_PATHS = ["/v1/models", "/health", "/"]
SERVE_TIMEOUT_S = 900   # first use downloads ~0.8GB weights
ASK_TIMEOUT_S = 300     # laya run hit 180s timeouts; give slower engines room

os.environ.setdefault("OMP_NUM_THREADS", "4")

asks_path = next(Path("/kaggle/input").rglob("asks.jsonl"), None)
if asks_path is None:
    raise SystemExit(
        "asks.jsonl not found under /kaggle/input — attach the "
        "dataset containing it in kernel-metadata.json")
asks = [json.loads(l) for l in asks_path.read_text().splitlines() if l.strip()]
print(f"[asks] {len(asks)} from {asks_path}", flush=True)

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "opendecider"], check=True)
# show CLI surface once so the log records the right flags if serve fails
subprocess.run(["opendecider", "serve", "--help"], check=False)

srv = subprocess.Popen(
    ["opendecider", "serve", "--host", "127.0.0.1", "--port", "8000"])
time.sleep(10)
if srv.poll() is not None:
    # flags unsupported -> retry with defaults, then probe PORTS
    print("[opendecider] flag launch failed, retrying bare `serve`",
          flush=True)
    srv = subprocess.Popen(["opendecider", "serve"])
print("[opendecider] booting (weight download on first run)…", flush=True)


def _find_port() -> int | None:
    for port in PORTS:
        for p in HEALTH_PATHS:
            try:
                urllib.request.urlopen(
                    f"http://127.0.0.1:{port}{p}", timeout=5)
                return port
            except urllib.error.HTTPError:
                return port      # HTTP status -> server is listening
            except (urllib.error.URLError, ConnectionError, OSError):
                continue
    return None


deadline = time.time() + SERVE_TIMEOUT_S
URL = None
while time.time() < deadline:
    if srv.poll() is not None:
        raise SystemExit(f"opendecider serve exited: rc={srv.returncode}")
    port = _find_port()
    if port:
        URL = f"http://127.0.0.1:{port}/v1/systemone"
        print(f"[opendecider] up on :{port}", flush=True)
        break
    time.sleep(5)
if URL is None:
    raise SystemExit(
        f"opendecider serve not responding after {SERVE_TIMEOUT_S}s")


def model_ask(ask: dict) -> dict:
    """One ask -> {"answers": {...}, "ms": int} or {"error": ...}."""
    body = json.dumps({
        "state": ask["state"],
        "model": MODEL,
        "questions": ask["questions"],
    }).encode()
    req = urllib.request.Request(
        URL, data=body,
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=ASK_TIMEOUT_S) as r:
            data = json.loads(r.read())
        return {"answers": data.get("answers", {}),
                "ms": int((time.time() - t0) * 1000)}
    except urllib.error.HTTPError as e:
        return {"error": f"http_{e.code}",
                "detail": e.read()[:300].decode("utf-8", "replace")}
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        return {"error": "timeout_or_conn", "detail": str(e)[:300]}


def diff_one(jev_ans: dict, laya_ans: dict) -> dict:
    """Per-question deltas. Unknown/extra fields pass through raw."""
    d = {}
    if "noul" in jev_ans or "noul" in laya_ans:
        j, l = jev_ans.get("noul"), laya_ans.get("noul")
        d["qtype"] = "noul"
        d["jev"], d["laya"] = j, l
        if j is not None and l is not None:
            d["abs_delta"] = round(abs(j - l), 4)
            # 0.5 is the action threshold everywhere noul gates a pick
            d["flip_0.5"] = (j >= 0.5) != (l >= 0.5)
    elif "choice" in jev_ans or "choice" in laya_ans:
        d["qtype"] = "choice"
        d["jev"] = jev_ans.get("choice")
        d["laya"] = laya_ans.get("choice")
        d["agree"] = d["jev"] == d["laya"]
    elif "score" in jev_ans or "score" in laya_ans:
        d["qtype"] = "score"
        d["jev"] = jev_ans.get("score")
        d["laya"] = laya_ans.get("score")
        if isinstance(d["jev"], (int, float)) and isinstance(
                d["laya"], (int, float)):
            d["abs_delta"] = round(abs(d["jev"] - d["laya"]), 4)
    else:
        d["qtype"] = "other"
        d["jev"], d["laya"] = jev_ans, laya_ans
    return d


rows = []
t0 = time.time()
for i, ask in enumerate(asks):
    if i % 25 == 0:
        print(f"[{i}/{len(asks)}]", flush=True)
    res = model_ask(ask)
    base = {"gene": ask["gene"], "stage": ask["stage"], "ask_i": i}
    if "error" in res:
        rows.append({**base, "error": res["error"],
                     "detail": res.get("detail", "")})
        continue
    for qid, lans in res["answers"].items():
        jans = ask["jev_answers"].get(qid, {})
        rows.append({**base, "qid": qid, "ms": res["ms"],
                     **diff_one(jans, lans)})

with DIFFS.open("w") as fh:
    for r in rows:
        fh.write(json.dumps(r) + "\n")


def _mean(vs):
    vs = list(vs)
    return round(sum(vs) / len(vs), 4) if vs else None


# per-stage + overall summary
from collections import defaultdict
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
            "mean_abs_delta": _mean(r.get("abs_delta") for r in nouls
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
            "mean_abs_delta": _mean(r.get("abs_delta") for r in scores
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
    "laya_model": MODEL,
    "overall": {
        "noul_n": len(nouls),
        "noul_mean_abs_delta": _mean(r.get("abs_delta") for r in nouls
                                     if r.get("abs_delta") is not None),
        "noul_flip_0.5_rate": _mean(float(r.get("flip_0.5", False))
                                    for r in nouls),
        "choice_n": len(choices),
        "choice_agree_rate": _mean(float(r.get("agree", False))
                                   for r in choices),
    },
    "by_stage": summary,
}
RESULT.write_text(json.dumps(results, indent=1))
print(json.dumps(results["overall"], indent=1), flush=True)
print(f"wrote {RESULT} and {DIFFS}", flush=True)

srv.terminate()
