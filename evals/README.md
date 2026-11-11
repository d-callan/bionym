# BioNym evals — plan

Real runs on real IDs, measuring correctness, latency, and cost.
Everything lives under `evals/` — in the repo but outside the shipped
`bionym` package (src-layout already excludes it).

## Goals

- **Correctness**: did it resolve to the right thing, refuse the right
  things, and not hallucinate?
- **Performance**: wall time per run (+ per stage), token usage per run
  (+ per stage via `graph.metadata["jev_usage"]`).
- **Repeatability**: corpus + config recorded per run; re-run and diff.

## Manuscript suites (branch `eval-suites`)

Five suites per `EVAL_IMPLEMENTATION_PLAN.md`; corpora live in `data/`.

### Determinism

- **Inputs are deterministic**: corpora/question shapes/prompts are
  fixed files — every run asks identical questions.
- **Live answers are stochastic**: JEV/LLM outputs and public-DB state
  can drift between runs (Suite 4 measures that variance directly).
- **Same-machine reruns replay exactly**: `evals/.cache` (diskcache,
  gitignored) stores every paid call — a rerun returns recorded answers
  free and identical. Delete it for genuinely fresh calls.
- **Analyzers are deterministic**: Suite 1A/5A compute over committed
  recorded artifacts only.
- **Suite 4 replication**: ask generations use per-(regime, rep) caches
  (`evals/.cache/suite4_<regime>_rep<k>`) so K=3 runs are genuinely
  independent; graphs share `evals/.cache` for zero re-spend.

### Environment

```bash
TYPESAFE_API_KEY=...        # JEV judge (paid — required for live runs)
LLM_API_KEY=...             # generator/judge-replay LLM
LLM_MODEL=MiniMax-M3        # or other OpenAI-compatible model id
LLM_BASE_URL=https://ai.tejas.tacc.utexas.edu/v1
```

Scripts auto-load `../.env`. `--mock` runs plumbing without model spend
(public-DB calls still run); `--dry` where present emits asks only.

### Quota contract

Any runner hitting `JevQuotaError` writes a `stopped` row, prints a
refill banner, and exits **75** (EX_TEMPFAIL). Refill, then rerun with
`--resume <rows file>` — completed cases skip, cached calls replay free.

### Run order (cheapest first)

```bash
# 0. Corpus pre-flight — all 63 ToL IDs verified live (free)
python evals/verify_corpus_ids.py

# 1A. $0 — JEV confidence distributions vs replayed judges
python evals/eval_confidence_dist.py --tag suite1a-confidence-dist

# 1B. pennies — 120-ask evidence ladder (L3>L2>L1>L0)
python evals/eval_jev_counterfactuals.py            # JEV run + metrics
python evals/eval_jev_counterfactuals.py --dry      # asks only, no spend
LLM_MODEL=<judge> python evals/replay_llm.py \
    --asks evals/data/jev_counterfactuals_asks.jsonl --tag <judge>-cf
python evals/eval_jev_counterfactuals.py --diffs evals/results/*cf*.diffs.jsonl

# 2. pennies — refusal / near-miss / wrong-entity / ambiguity traps
python evals/run.py --cases evals/cases_refusal.yaml --name suite2-refusal

# 3A. ~1.5-2.5M JEV tokens — 63-gene quick scan
python evals/eval_tree_of_life.py --stage 3a
# 3B. stratified 21-gene full graphs (propose+summarize)
python evals/eval_tree_of_life.py --stage 3b

# 4. 12 targets x 4 questions x {grounded,bare} x K=3 x {t0,t07}
for rep in 1 2 3; do
  python evals/eval_grounded_ask_vs_bare.py --regime t0  --rep $rep
  python evals/eval_grounded_ask_vs_bare.py --regime t07 --rep $rep
done
python evals/eval_grounded_ask_vs_bare.py --analyze evals/results/<rows>.jsonl

# 5A. $0 — unified judge/generator/resolve runtime table
python evals/eval_runtime.py
```

## Layout

```
evals/
  README.md        # this file
  cases.yaml       # the corpus: id + mode + expectations
  run.py           # runner -> results/<ts>-<name>.jsonl
  compare.py       # diff two runs -> markdown table
  results/         # committed JSONL rows (small — run history lives in git)
  reports/         # optional human-reviewed summaries per run
```

## Corpus (`cases.yaml`)

One entry per case:

```yaml
- id: PF3D7_0710100
  name: veupathdb-happy
  quick: false             # propose+summarize on; quick scan otherwise
  expect:
    outcome: resolved      # resolved | refused | error
    anchor_ns: veupathdb   # namespace of the resolved anchor node
    min_nodes: 10
    predicates: [ortholog_of, measured_in]   # must appear in graph

- id: abc123
  name: garbage-string
  quick: true
  expect:
    outcome: refused       # no match — pins the implicit contract

- id: K00001               # KEGG ortholog id — valid ID, wrong entity
  name: kegg-not-a-gene
  quick: true
  expect:
    outcome: refused

- id: <a VEuPathDB-only locus tag with no NCBI record>
  name: veupath-only
  quick: false
  expect:
    outcome: resolved
    anchor_ns: veupathdb
```

### Tiers

- **T1 smoke** (~6 cases, quick): garbage string, KEGG id, PubMed id,
  one happy path. Cheap, run often.
- **T2 core** (~12): happy paths across namespaces — NCBI gene id,
  gene symbol (+organism), UniProt, VEuPathDB locus tag, Ensembl;
  edge cases: VEuPath-only (no NCBI), deprecated alias, ambiguous
  symbol.
- **T3 deep** (3–5): full analysis (propose+summarize) on genes rich
  in datasets/orthologs — exercises the claim pipeline end-to-end.

## Runner (`run.py`)

```bash
python -m evals.run --tier t1 --name baseline          # one config
python -m evals.run --tier t2 --name gpt5-mini         # env-switched model
python -m evals.compare baseline gpt5-mini             # diff
```

- One JSONL row per case:
  `{case, config:{llm_model,llm_base,jev_model,propose}, outcome,
  assertions:[{check,pass,detail}], wall_s, stage_times, jev_usage,
  nodes, edges, predicates, error, git_sha}`
- Clients configured via the existing env vars (`LLM_MODEL`,
  `LLM_BASE_URL`, `LLM_API_KEY`, `TYPESAFE_API_KEY`) — a `--config
  <file>` flag sources a profile file so runs are self-describing.
- Sequential by default (NCBI/VEuPathDB etiquette); `--jobs` later if
  needed. Keep raw graphs under `results/graphs/` (gitignored) for
  debugging a failed case.

## Diffing (`compare.py`)

`A vs B` table: outcome flips, assertion pass→fail, wall/token deltas
per case, totals at the bottom. Commit `results/*.jsonl` so git
history *is* the run log; `reports/` gets a markdown summary when a
run matters.

## Known limits

- JEV/LLM nondeterminism: same case can flip across runs — that's
  signal, not bug; keep runs and diff trends.
- Claim-level truth needs eyes: assertions check structure; content
  spot-checks go in `reports/`.
- Real-API runs are slow + cost tokens: T1 for iteration, T3 sparingly.