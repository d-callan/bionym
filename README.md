# BioNym

Resolve bioinformatics identifiers into confidence-scored knowledge graphs.

Given a gene identifier (any species — NCBI gene ID, symbol, VEuPathDB
locus tag, UniProt accession, Ensembl ID, ...), BioNym gathers evidence
from public APIs (NCBI, VEuPathDB, OMA, UniProt, GEO, Expression Atlas,
...), asks [JEV](https://typesafe.ai) (TypeSafe `systemone`) an ordered
workflow of typed questions, and emits a knowledge graph where every claim
carries a confidence score and a list of supporting evidence.

## Status

All pipeline stages implemented: S0 (ID classification), S1 (entity
resolution → organism + assembly), S2 (related & newer assemblies with
species-rank lineage normalization), S3 (orthologs via OMA), S4 (UniProt
annotation: GO evidence-code confidence, KEGG, InterPro/Pfam), S5
(expression: GEO + Expression Atlas, JEV-scored), S6 (cross-assembly
presence: `annotated_in` + JEV `likely_present`).

On top of the staged pipeline, two optional LLM passes propose claims that
JEV then verifies: text claims over publication/dataset metadata, and data
claims over per-gene measurements (VEuPathDB ExpressionGraphs, GEO GDS SOFT
files, GXA baseline/differential TSVs). A third pass can synthesize a
JEV-verified gene summary into `metadata.summary`.

Interfaces: CLI (`bionym resolve`, `bionym summarize`, `bionym ask`), FastAPI backend (`backend/`), static web frontend (`web/`, D3 multi-partite network).

`ask` answers a free-form question over a resolved graph: JEV first
classifies the question into graph categories, the matching subgraph
is sliced out, the LLM answers over that slice (citing node ids), and
JEV scores accuracy/completeness — deterministic citation fidelity
checked against the full graph on top.

## Install

```bash
pip install bionym
```

From source (editable, with test/dev dependencies):

```bash
conda create -n bionym python=3.11 -y
conda activate bionym
pip install -e '.[dev]'
```

## Configure

```bash
cp .env.example .env   # then edit
```

- `TYPESAFE_API_KEY` — required for real JEV calls.
- `NCBI_API_KEY` — optional, raises NCBI rate limit 3/s → 10/s.
- `VEUPATHDB_API_KEY` — optional, enables VEuPathDB lookups (release 71+).
- `BIONYM_CACHE_DIR` — optional response/JEV cache.
- `LLM_API_KEY` + `LLM_MODEL` — optional, enables the proposal passes
  (claims, summary). Any OpenAI-compatible endpoint; `LLM_BASE_URL`
  defaults to OpenRouter. Disabled when unset — the resolver runs fine
  without them, you just get no proposed claims or summary.
- `BIONYM_PROPOSAL_MIN_SCORE` (default 0.5) — JEV triage score a
  dataset/publication needs before the LLM spends a call on it.
- `BIONYM_PROPOSAL_CONCURRENCY` (default 8) — max simultaneous LLM calls.
- `JEV_BASE_URL` — optional; repoint the systemone endpoint at any
  compatible backend (Laya, Kev, CLM all speak `POST /v1/systemone`).
- `JEV_MODEL` — optional (default `jev-latest`); model string sent in
  the request — set it when pointing at a non-jev backend.
- `LLM_TIMEOUT` — optional, seconds (default 180); whole-graph asks
  can exceed the old 60s on slow models.
- `BIONYM_ALLOW_MOCK` — server only; `=1` permits `mock_jev=true`
  on API requests (offline dev/evals). Off by default.

## Usage

```bash
bionym resolve PF3D7_0710100 -o graph.json -v
bionym resolve 672 --depth 0            # classify only
bionym resolve 672 --report             # + self-contained HTML report
bionym resolve 672 --min-confidence 0.7 # drop low-confidence edges
bionym resolve 672 --no-propose         # quick scan: skip LLM claim passes
bionym resolve 672 --summarize          # + JEV-verified summary in metadata
bionym resolve 672 --mock-jev           # offline dev, no API key

bionym summarize graph.json           # summary pass on an existing graph
bionym ask graph.json "what datasets are associated?"
                                     # classify → slice → LLM answer → JEV scores
```

> **`--mock-jev` is for development only.** It returns deterministic
> placeholder answers (first option / midpoint score / 0.5) instead of real
> JEV judgments, so the resulting graph **will not be biologically
> meaningful** — edges and confidences are arbitrary. It's a free way to
> exercise the full pipeline and surface *technical* bugs (crashes, parsing,
> wiring) without an API key or cost. Never interpret mock output as a
> result.

Output: a JSON knowledge graph — `nodes` (typed: Gene, Organism, Assembly,
IdType, ...; each with an external `url`), `edges` (claims with
`confidence`, `probabilities`, `jev_question_id`, `evidence[]`), and
`metadata` (`outcome`: `resolved|refused`, `stages` + `stage_times`,
`jev_usage` per-stage token counts, optional `summary`: JEV-verified
claim list with `claim`/`quote`/`confidence`).

## Web app

```bash
# backend (needs the package installed + .env)
cd backend && uvicorn main:app --port 8001

# frontend (static; also works opened directly or from GH Pages)
cd web && python -m http.server 8080
# open http://localhost:8080 — backend URL is set in web/index.html (BACKEND const)
```

The UI offers **Quick scan** (`propose=false` — deterministic stages + JEV
edge scoring only, seconds) vs **Full analysis** (`propose+summarize`,
minutes). The verified summary renders in a collapsible section above the
network. API: `GET /api/resolve?identifier=…&depth=…&propose=…&summarize=…`,
`POST /api/summarize` (summary pass over a previously returned graph),
and `POST /api/ask` (ask-the-graph over a previously returned graph —
powers the collapsible Q&A panel in the UI).

## Design rules

- JEV questions are Python-templated only — no LLM generates questions or
  candidate options. `criteria` always come from API result sets; every
  Choice question has a `none`/`other` escape. JEV judges, never enumerates.
- The one exception: *claims*. Free-text biological claims can't be
  enumerated from an API result set, so the proposal passes (text claims,
  data claims, summary) let the LLM propose candidates — the only place it
  generates content. The asymmetry holds: every proposal is verified by JEV
  against the serialized source (dataset metadata, per-sample values, or
  the graph itself) before it lands. The LLM proposes, JEV disposes — it
  never scores its own output.
- Every edge must carry ≥1 `Evidence` record (source, endpoint, timestamp,
  payload) so "why" is always answerable.
- Unbounded API result sets are capped before JEV/LLM stages so a single
  state can't overflow the model context or spend unbounded calls:
  ortholog candidates keep the top 100 by OMA score (S3), and
  datasets/publications keep the top 100 per kind by triage score
  (`proposals.MAX_ITEMS_PER_KIND`). These are hard-coded for now — if a
  use case needs deeper coverage they should become configurable.

## Layout

```
src/bionym/       core library + CLI (no web deps)
  jev.py          systemone client: batch questions, token logging,
                  mock mode, JEV_BASE_URL for Laya/Kev/CLM backends
  llm.py          proposal/summary/ask LLM client (mock mode for dev)
  ask.py          ask-the-graph orchestration (classify→slice→answer→score)
  evidence.py     Evidence records
  graph.py        KnowledgeGraph: nodes/edges/confidence, url + filtering
  proposals.py    LLM propose-then-JEV-verify: prompts, serializers, parsers
  questions/      staged JEV question builders (s0–s6) + ask
                  classify/slice builders
  clients/        one thin client per data source
  resolver.py     stage orchestration + proposal/summary passes
  report.py       self-contained HTML report (D3 network + tables)
backend/          FastAPI deployable
web/              static frontend (D3 multi-partite network)
evals/            corpus + runner + differ for real-ID eval runs
                  (see evals/README.md; results committed as JSONL)
tests/            pytest; JEV/LLM mocked or faked, backend via
                  TestClient
```
