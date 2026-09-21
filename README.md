# idresolver

Resolve bioinformatics identifiers into confidence-scored knowledge graphs.

Given an identifier (v1 scope: gene IDs, eukaryotic pathogens), idresolver
gathers evidence from public APIs (NCBI, VEuPathDB, ...), asks
[JEV](https://docs.typesafe.ai) (TypeSafe `systemone`) an ordered workflow of
typed questions, and emits a knowledge graph where every claim carries a
confidence score and a list of supporting evidence.

## Status

All pipeline stages implemented: S0 (ID classification), S1 (entity
resolution → organism + assembly), S2 (related & newer assemblies with
species-rank lineage normalization), S3 (orthologs via OMA), S4 (UniProt
annotation: GO evidence-code confidence, KEGG, InterPro/Pfam), S5
(expression: GEO + Expression Atlas, JEV-scored), S6 (cross-assembly
presence: `annotated_in` + JEV `likely_present`).

Interfaces: CLI (`idresolver resolve`), FastAPI backend (`backend/`),
static web frontend (`web/`, D3 multi-partite network).

## Install

```bash
conda create -n idresolver python=3.11 -y
conda activate idresolver
pip install -e '.[dev]'
```

## Configure

```bash
cp .env.example .env   # then edit
```

- `TYPESAFE_API_KEY` — required for real JEV calls.
- `NCBI_API_KEY` — optional, raises NCBI rate limit 3/s → 10/s.
- `IDRESOLVER_CACHE_DIR` — optional response/JEV cache.

## Usage

```bash
idresolver resolve PF3D7_0710100 -o graph.json -v
idresolver resolve 672 --depth 0            # classify only
idresolver resolve 672 --report             # + self-contained HTML report
idresolver resolve 672 --mock-jev           # offline dev, no API key
```

Output: a JSON knowledge graph — `nodes` (typed: Gene, Organism, Assembly,
IdType, ...), `edges` (claims with `confidence`, `probabilities`,
`jev_question_id`, `evidence[]`), and `metadata.jev_usage` (per-stage token
counts for cost projection).

## Design rules

- JEV questions are Python-templated only — no LLM generates questions or
  candidate options. `criteria` always come from API result sets; every
  Choice question has a `none`/`other` escape. JEV judges, never enumerates.
- Every edge must carry ≥1 `Evidence` record (source, endpoint, timestamp,
  payload) so "why" is always answerable.

## Layout

```
src/idresolver/   core library + CLI (no web deps)
  jev.py          systemone client: batch questions, token logging, mock mode
  evidence.py     Evidence records
  graph.py        KnowledgeGraph: nodes/edges/confidence
  questions/      staged JEV question builders (s0, s1, ...)
  clients/        one thin client per data source
  resolver.py     stage orchestration
backend/          FastAPI deployable (planned, M3)
web/              static frontend (planned, M3)
tests/            pytest; JEV mocked, clients faked
```
