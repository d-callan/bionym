# BioNym

Resolve bioinformatics identifiers into confidence-scored knowledge graphs.

Given an identifier (v1 scope: gene IDs, eukaryotic pathogens), BioNym
gathers evidence from public APIs (NCBI, VEuPathDB, OMA, UniProt, GEO,
Expression Atlas, ...), asks [JEV](https://typesafe.ai) (TypeSafe
`systemone`) an ordered workflow of typed questions, and emits a knowledge
graph where every claim carries a confidence score and a list of supporting
evidence.

## Status

All pipeline stages implemented: S0 (ID classification), S1 (entity
resolution → organism + assembly), S2 (related & newer assemblies with
species-rank lineage normalization), S3 (orthologs via OMA), S4 (UniProt
annotation: GO evidence-code confidence, KEGG, InterPro/Pfam), S5
(expression: GEO + Expression Atlas, JEV-scored), S6 (cross-assembly
presence: `annotated_in` + JEV `likely_present`).

Interfaces: CLI (`bionym resolve`), FastAPI backend (`backend/`), static web
frontend (`web/`, D3 multi-partite network).

## Install

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

## Usage

```bash
bionym resolve PF3D7_0710100 -o graph.json -v
bionym resolve 672 --depth 0            # classify only
bionym resolve 672 --report             # + self-contained HTML report
bionym resolve 672 --min-confidence 0.7 # drop low-confidence edges
bionym resolve 672 --mock-jev           # offline dev, no API key
```

Output: a JSON knowledge graph — `nodes` (typed: Gene, Organism, Assembly,
IdType, ...; each with an external `url`), `edges` (claims with
`confidence`, `probabilities`, `jev_question_id`, `evidence[]`), and
`metadata.jev_usage` (per-stage token counts for cost projection).

## Web app

```bash
# backend (needs the package installed + .env)
cd backend && uvicorn main:app --port 8001

# frontend (static; also works opened directly or from GH Pages)
cd web && python -m http.server 8080
# open http://localhost:8080 — backend URL is set in web/index.html (BACKEND const)
```

## Design rules

- JEV questions are Python-templated only — no LLM generates questions or
  candidate options. `criteria` always come from API result sets; every
  Choice question has a `none`/`other` escape. JEV judges, never enumerates.
- Every edge must carry ≥1 `Evidence` record (source, endpoint, timestamp,
  payload) so "why" is always answerable.

## Layout

```
src/bionym/       core library + CLI (no web deps)
  jev.py          systemone client: batch questions, token logging, mock mode
  evidence.py     Evidence records
  graph.py        KnowledgeGraph: nodes/edges/confidence, url + filtering
  questions/      staged JEV question builders (s0–s6)
  clients/        one thin client per data source
  resolver.py     stage orchestration
  report.py       self-contained HTML report (D3 network + tables)
backend/          FastAPI deployable
web/              static frontend (D3 multi-partite network)
tests/            pytest; JEV mocked, clients faked
```
