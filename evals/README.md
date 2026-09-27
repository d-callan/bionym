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