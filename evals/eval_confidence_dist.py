#!/usr/bin/env python3
"""Suite 1a — confidence-score distribution analysis (offline, $0).

Quantifies "JEV produces a reasonable distribution of scores" vs the
open SystemOne backends and chat-LLM judges already replayed through
the same 3,592 recorded questions (evals/asks-claimsplit.jsonl).

  python evals/eval_confidence_dist.py
  python evals/eval_confidence_dist.py --diffs evals/results/*qwen32b*.diffs.jsonl

Metrics per judge x question-type (overall and per stage):

- n, distinct          answered count; unique hundredth-rounded values
- mean / p10 / p50 / p90
- entropy_norm         Shannon entropy of the value histogram normalized
                       to [0,1] (fixed 20 bins over [0,1] for noul so
                       judges compare directly; distinct-value histogram
                       for score). 1.0 = perfectly uniform — too smooth;
                       very low = degenerate.
- spike_share          largest fraction of answers at a single value —
                       catches {0,1} collapse (e.g. llama-8b judges).
- frac_0_or_1          noul only: share pinned exactly at the extremes.
- vs_jev               mean|delta| + flip@0.5 (noul) or agree rate
                       (choice) against the recorded JEV answers — the
                       same diff metrics as replay_llm.py.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ASKS = ROOT / "evals" / "asks-claimsplit.jsonl"
RESULTS = ROOT / "evals" / "results"


def _scalar(ans, qtype: str):
    """One comparable float per answer (noul/score only)."""
    if not isinstance(ans, dict):
        return ans if isinstance(ans, (int, float)) else None
    v = ans.get(qtype)
    return v if isinstance(v, (int, float)) else None


def _pct(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    i = min(len(s) - 1, max(0, round((p / 100) * (len(s) - 1))))
    return round(s[i], 4)


def _entropy_norm(counts: list[int], n_bins: int) -> float | None:
    total = sum(counts)
    if not total or n_bins < 2:
        return None
    h = -sum((c / total) * math.log(c / total) for c in counts if c)
    return round(h / math.log(n_bins), 4)


def dist_metrics(vals: list[float], noul: bool) -> dict:
    """Distribution stats for one scalar series."""
    n = len(vals)
    if not n:
        return {"n": 0}
    hundredths = [round(v, 2) for v in vals]
    if noul:
        # Fixed 20 bins over [0,1] — directly comparable across judges.
        bins = [0] * 20
        for v in vals:
            bins[min(19, max(0, int(v * 20)))] += 1
        ent = _entropy_norm(bins, 20)
    else:
        ent = _entropy_norm(list(Counter(hundredths).values()),
                            len(set(hundredths)))
    spike = Counter(hundredths).most_common(1)[0][1] / n
    out = {
        "n": n,
        "distinct": len(set(hundredths)),
        "mean": round(sum(vals) / n, 4),
        "p10": _pct(vals, 10),
        "p50": _pct(vals, 50),
        "p90": _pct(vals, 90),
        "entropy_norm": ent,
        "spike_share": round(spike, 4),
    }
    if noul:
        out["frac_0_or_1"] = round(
            sum(1 for v in vals if v in (0.0, 1.0)) / n, 4)
    return out


def jev_series(asks: list[dict]) -> dict:
    """JEV's own answers, grouped stage -> qtype -> [values]."""
    by: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for ask in asks:
        for qid, ans in (ask.get("jev_answers") or {}).items():
            qt = (ask["questions"].get(qid) or {}).get("type")
            if qt in ("noul", "score"):
                v = _scalar(ans, qt)
                if v is not None:
                    by[ask["stage"]][qt].append(v)
            elif qt == "choice":
                by[ask["stage"]]["choice_n"] = (
                    by[ask["stage"]].get("choice_n", 0) + 1)
    return by


def judge_series(rows: list[dict]) -> tuple[dict, dict]:
    """Replayed judge answers from diffs rows -> (series, vs_jev pairs)."""
    by: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    pairs: dict[str, dict[str, list]] = defaultdict(
        lambda: defaultdict(list))
    for r in rows:
        if "qid" not in r or "error" in r:
            continue
        qt = r.get("qtype")
        stage = r["stage"]
        if qt == "noul":
            jv = r.get("laya")
            if isinstance(jv, (int, float)):
                by[stage]["noul"].append(jv)
                if isinstance(r.get("jev"), (int, float)):
                    pairs[stage]["noul"].append((r["jev"], jv))
        elif qt == "score":
            jv = r.get("laya")
            if isinstance(jv, (int, float)):
                by[stage]["score"].append(jv)
                if isinstance(r.get("jev"), (int, float)):
                    pairs[stage]["score"].append((r["jev"], jv))
        elif qt == "choice":
            by[stage]["choice_n"] = by[stage].get("choice_n", 0) + 1
            pairs[stage]["choice"].append((r.get("jev"), r.get("laya")))
    return by, pairs


def _mean(vs) -> float | None:
    vs = [v for v in vs if v is not None]
    return round(sum(vs) / len(vs), 4) if vs else None


def vs_jev(pairs: dict) -> dict:
    out = {}
    nouls = [p for ps in pairs.values() for p in ps.get("noul", [])]
    if nouls:
        out["noul_mean_abs_delta"] = _mean(abs(j - l) for j, l in nouls)
        out["noul_flip_0.5_rate"] = _mean(
            float((j >= 0.5) != (l >= 0.5)) for j, l in nouls)
    choices = [p for ps in pairs.values() for p in ps.get("choice", [])]
    if choices:
        out["choice_agree_rate"] = _mean(float(j == l) for j, l in choices)
    return out


def summarize(by: dict, pairs: dict | None = None) -> dict:
    """Per-stage + overall metrics for one judge's series."""
    stages: dict[str, dict] = {}
    flat: dict[str, list] = defaultdict(list)
    for stage, qtypes in sorted(by.items()):
        row = {}
        for qt in ("noul", "score"):
            vals = qtypes.get(qt, [])
            flat[qt] += vals
            if vals:
                row[qt] = dist_metrics(vals, noul=(qt == "noul"))
        if qtypes.get("choice_n"):
            row["choice"] = {"n": qtypes["choice_n"]}
        stages[stage] = row
    overall = {qt: dist_metrics(flat[qt], noul=(qt == "noul"))
               for qt in ("noul", "score") if flat[qt]}
    out = {"overall": overall, "by_stage": stages}
    if pairs is not None:
        out["vs_jev"] = vs_jev(pairs)
    return out


def _tag(path: Path) -> str:
    name = path.name
    for pat in ("-claimsplit.diffs.jsonl", "-claimsplit-diffs.jsonl"):
        name = name.replace(pat, "")
    # Strip only the leading YYYYMMDD date — keep provider prefixes
    # (tacc-qwen32b and qwen32b are different replays).
    return name.split("-", 1)[-1] or path.stem


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asks", default=str(ASKS))
    ap.add_argument("--diffs", nargs="*", default=None,
                    help="per-question replay diffs; default all "
                         "*claimsplit*diffs*.jsonl in results/")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    asks = [json.loads(l) for l in Path(args.asks).read_text().splitlines()
            if l.strip()]
    diffs_paths = ([Path(p) for g in args.diffs for p in glob.glob(g)]
                   if args.diffs is not None
                   else sorted(RESULTS.glob("*claimsplit*diffs*.jsonl")))

    judges: dict[str, dict] = {
        "jev": summarize(jev_series(asks), pairs=None),
    }
    for path in diffs_paths:
        rows = [json.loads(l) for l in path.read_text().splitlines()
                if l.strip()]
        by, pairs = judge_series(rows)
        judges[_tag(path)] = summarize(by, pairs=pairs)

    # printed headline table — noul distributions per judge
    hdr = f"{'judge':<16}{'n':>6}{'distinct':>9}{'p50':>7}{'ent':>7}" \
          f"{'spike':>7}{'0/1':>7}{'|D|':>7}{'flip':>7}"
    print(hdr)
    for name, s in judges.items():
        o = s["overall"].get("noul", {})
        v = s.get("vs_jev") or {}
        print(f"{name:<16}{o.get('n', 0):>6}{o.get('distinct', 0):>9}"
              f"{str(o.get('p50')):>7}{str(o.get('entropy_norm')):>7}"
              f"{str(o.get('spike_share')):>7}{str(o.get('frac_0_or_1')):>7}"
              f"{str(v.get('noul_mean_abs_delta')):>7}"
              f"{str(v.get('noul_flip_0.5_rate')):>7}")

    tag = args.tag or "confidence-dist"
    out_path = RESULTS / f"{date.today():%Y%m%d}-{tag}.json"
    out = {
        "suite": "1a-confidence-distribution",
        "asks": str(Path(args.asks).relative_to(ROOT)),
        "diffs": [str(p.relative_to(ROOT)) for p in diffs_paths],
        "judges": judges,
    }
    out_path.write_text(json.dumps(out, indent=1))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
