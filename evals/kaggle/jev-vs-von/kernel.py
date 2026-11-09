"""Kaggle CPU kernel: replay recorded JEV asks through local Von.

Input: asks.jsonl (one {gene, stage, state, questions, jev_answers}
per line, produced by bionym/evals/dump_asks.py and uploaded as a
Kaggle dataset — set its slug in kernel-metadata.json dataset_sources).

Steps: pip install von-sdk, boot `von serve`, POST every ask to
/v1/systemone, diff the typed answers against Jev's -> results.json
(summary) + diffs.jsonl (one row per question).

Context: Von's encoder accepts 8k-token states (same class as Laya
Standard). Oversized states — proposals_triage runs ~12-14k — are
middle-truncated by the server (truncation field in response) or,
with --on-overflow refuse, rejected; default is truncate.
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

VON_MODEL = "von-1.3.0"
PORT = 8000
URL = f"http://127.0.0.1:{PORT}/v1/systemone"
# any HTTP response on these paths = server is up (don't care which exists)
HEALTH_PATHS = ["/v1/models", "/health", "/"]
SERVE_TIMEOUT_S = 900   # first load downloads ~3GB weights
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
                "von-sdk"], check=True)

srv = subprocess.Popen(
    ["von", "serve", "--host", "127.0.0.1", "--port", str(PORT)])
print("[von] booting (weight download on first run)…", flush=True)


def _http_up() -> bool:
    for p in HEALTH_PATHS:
        try:
            urllib.request.urlopen(
                f"http://127.0.0.1:{PORT}{p}", timeout=5)
            return True
        except urllib.error.HTTPError:
            return True          # got an HTTP status -> server is listening
        except (urllib.error.URLError, ConnectionError, OSError):
            continue
    return False


deadline = time.time() + SERVE_TIMEOUT_S
ready = False
while time.time() < deadline:
    if srv.poll() is not None:
        raise SystemExit(f"von serve exited early: rc={srv.returncode}")
    if _http_up():
        print("[von] up", flush=True)
        ready = True
        break
    time.sleep(5)
if not ready:
    raise SystemExit(f"von serve not responding after {SERVE_TIMEOUT_S}s")


def model_ask(ask: dict) -> dict:
    """One ask -> {"answers": {...}, "ms": int} or {"error": ...}."""
    body = json.dumps({
        "state": ask["state"],
        "model": VON_MODEL,
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
    "laya_model": VON_MODEL,
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
