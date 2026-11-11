# BioNym Evaluation Implementation Plan (Self-Contained Agent Specification)

**Target Repository**: `/home/danielle/Documents/idresolver` (`bionym` package)
**Companion BRC Reference Repo (read-only reference for TACC config)**: `/home/danielle/Documents/brc-analytics/brc-analytics`
**Purpose**: This document is a complete, step-by-step engineering and execution specification for an autonomous, contextless agent to build, run, checkpoint, resume, and summarize all BioNym evaluations needed for the manuscript — safely operating under a small (~$4) initial TypeSafe JEV credit balance.

---

## 0. Ground Rules & Scientific Framing (Read Before Writing Any Code)

1. **Zero silent degradation on JEV credit exhaustion**: The user currently has **~$4 USD** in their TypeSafe JEV account (`TYPESAFE_API_KEY`). If JEV runs out of credits/tokens mid-run, the runner **MUST immediately halt**, flush all completed work to disk, leave disk caches intact so even partial in-flight stages do not re-bill on retry, print a clear `CREDIT EXHAUSTED — PAUSED` banner with instructions to refill and the exact `--resume` command, and exit with code **`75`** (`EX_TEMPFAIL`).
2. **Evidence-inclusion confidence, NOT publication quality**: Across BioNym, edge `confidence` quantifies *evidential graph-inclusion confidence* — how certain BioNym is that a retrieved database record or cited text passage supports adding that node/edge to the graph — **not** the scientific rigor, sample size, or ultimate biological truth of the upstream study. Every audit and distribution analysis in this plan evaluates *responsiveness to evidence and source-support fidelity*, never upstream study quality.
3. **Framing of the JEV evaluation**: TypeSafe `jev-latest` is treated as the state-of-the-art *System One* typed decision model. We are **not** trying to "beat" JEV or run a massive human re-annotation campaign to prove JEV is SOTA. Instead, we need quantifiable proof that:
   - JEV produces a **smooth, well-spread distribution of confidence scores** across `[0, 1]`;
   - JEV's scores **shift predictably and monotonically** with both **question type / pipeline stage** and **evidence strength** (including controlled evidence-perturbation probes where strong support → partial/hedged support → mismatched species/gene → unsupported foil); and
   - This behavior is **markedly more reasonable and calibrated** than currently available alternatives — open *System One* servers (`Laya`, `opendecider-nano`), which compress probabilities or fail on context length, and chat LLMs (`Llama-3.1-8B`, `Qwen3-32B`, `DeepSeek-V3.2`, `gpt-oss-120b`, `GLM-5.3-Flash`), which cluster on coarse verbalized round numbers (`0.80, 0.90, 0.95, 1.00`) or collapse to `{0, 1}`.
4. **Never leak secrets**: Load keys from `/home/danielle/Documents/idresolver/.env` (or environment variables). Never print, log, or commit API key values.
5. **Conda / Python environment**: Before running scripts or `pytest`, inspect existing conda environments (`conda info --envs`) or `.venv` in `/home/danielle/Documents/idresolver` so `bionym`, `httpx`, `typer`, `pyyaml`, `python-dotenv`, and `diskcache` are available; create/update a dedicated conda environment if none exists.

---

## 1. Model & Endpoint Configuration (TACC Production LLM + JEV)

### 1.1 Production Model Stack
- **Judge (`JevClient`)**:
  - Endpoint: `https://api.typesafe.ai/v1/systemone` (default)
  - Model: `jev-latest`
  - Auth env var: `TYPESAFE_API_KEY` (loaded from `.env`)
- **Generator / Summarize / Ask LLM (`LlmClient` via TACC Tejas OpenAI-compatible API)**:
  - Endpoint (`LLM_BASE_URL`): `https://ai.tejas.tacc.utexas.edu/v1`
  - Auth env var (`LLM_API_KEY`): uses the TACC key (check `LLM_API_KEY` in `/home/danielle/Documents/idresolver/.env`; if unset there, check whether `TACC_API_KEY` is exported in the shell environment and map `LLM_API_KEY=${LLM_API_KEY:-$TACC_API_KEY}` inside the runner — never copy raw secrets into tracked files).
  - **Model ID (`LLM_MODEL`) — Preflight Resolution Required**:
    - In BioNym's earlier replay (`evals/results/20261107-tacc-mm3-gen-results.json`), the TACC MiniMax model was **`MiniMax-M3`** (0.744 prose / 0.680 data overlap recall, 2× faster than `DeepSeek-V3.2`).
    - In BRC Analytics' assistant eval config (`/home/danielle/Documents/brc-analytics/brc-analytics/backend/api/evals/models.yaml.sample`), the TACC endpoint currently serves **`MiniMax-M2.7`** (alongside `gpt-oss-120b`, `Llama-4-Maverick-17B-128E-Instruct`, and `Meta-Llama-3.3-70B-Instruct`), and `evals/results/20261107-tacc-qwen32b-*` / `20261107-tacc-dsv32-*` used `Qwen3-32B` and `DeepSeek-V3.2`.
    - **Mandatory preflight step (`evals/preflight.py`)**: Query `GET https://ai.tejas.tacc.utexas.edu/v1/models` with `Authorization: Bearer $LLM_API_KEY` and test a 1-sentence JSON completion (`response_format={"type": "json_object"}`).
      1. Select the active MiniMax model served by TACC (`MiniMax-M3` if listed, else `MiniMax-M2.7` or the current `MiniMax-*` model ID returned by `/v1/models`) unless overridden by `--llm-model`.
      2. **Verify JSON-mode compatibility**: `src/bionym/llm.py` sends `"response_format": {"type": "json_object"}`. Note from BRC's `config.py` that some TACC vLLM deployments of MiniMax accept `json_object` cleanly (as `MiniMax-M3` did in `replay_generate.py`), whereas `json_schema` can 400. If the selected TACC MiniMax model returns HTTP 400 on `response_format={"type": "json_object"}` or wraps JSON in `<think>...</think>` reasoning tags, update `LlmClient.complete()` (or add a clean fallback in `llm.py`) to strip `<think>...</think>` blocks before JSON parsing and retry without `response_format` if HTTP 400 mentions `response_format`.
      3. Record the exact resolved `llm_model`, `llm_base_url`, `jev_model`, and `jev_url` in every run manifest and JSONL row.
- **Upstream Database Keys**:
  - `NCBI_API_KEY` (strongly recommended; raises NCBI rate limit from 3/s to 10/s) and `VEUPATHDB_API_KEY` (needed for direct VEuPathDB WDK calls since release 71). Verify presence (boolean only, never print values) in `preflight.py`.

---

## 2. Critical Prerequisite: Code Fixes for Credit-Exhaustion Safety & Telemetry

Before running **any** live evaluation, inspect `/home/danielle/Documents/idresolver/src/bionym/` and fix four specific issues in the codebase (accompanied by unit tests in `tests/`):

### 2.1 Fatal `JevQuotaExhaustedError` (Prevent Silent Swallowing in `Resolver`)

**Bug in current code**:
1. In `src/bionym/jev.py` (lines 170–173), any HTTP status `< 500` (other than `429`) raises a generic `JevError`, while `429` is retried 3 times and then also raises a generic `JevError`.
2. **CRITICAL**: In `src/bionym/resolver.py`, `Resolver` **catches and swallows `JevError`** in four places:
   - `_ask_chunks()` (lines 287–289): logs warning and sets `batch = {}` (used by `s3_orthologs`, `s5_expression`, `proposals_triage`!).
   - `_propose_claims()` `_verify()` (lines 2137–2140): logs warning and returns `{}` (drops all verified prose claims!).
   - `_propose_data_claims()` `_work()` (lines 2273–2278): logs warning and returns `[], {}, 0` (drops all verified data claims!).
   - `resolve()` `summarize` (lines 234–235): logs warning and leaves summary empty!
   - And in `src/bionym/ask.py` (lines 127–130): catches `JevError` and returns `scored: False`.
   - And in `evals/run.py` (lines 90–91): catches `Exception` per case, writes `"outcome": "error"`, and **continues burning calls on the remaining 90+ cases!**

If the user's ~$4 JEV balance runs out mid-resolve, `Resolver` will currently silently drop edges, produce a crippled half-empty graph marked `"outcome": "resolved"`, write it to the eval output as if it were valid data, and move on to fail every subsequent gene!

**Required fix**:
1. In `src/bionym/jev.py`, define:
   ```python
   class JevQuotaError(JevError):
       """Raised when JEV rejects a request due to exhausted credits/quota or auth failure."""
   ```
2. Add a classifier helper `_is_quota_or_auth_error(status_code: int, body_text: str) -> bool` in `src/bionym/jev.py`:
   - Return `True` immediately (no retries!) if:
     - `status_code in (401, 402, 403)`
     - OR `status_code == 429` AND `body_text.lower()` contains any of: `"insufficient"`, `"credit"`, `"balance"`, `"quota"`, `"billing"`, `"payment"`, `"exceeded your current"`, `"out of tokens"`, `"funds"` (note: plain transient rate-limit 429s without billing/quota keywords still retry normally up to 3 times; if all 3 retries still fail with 429, inspect body again).
     - OR `status_code == 400` AND `body_text.lower()` contains `"credit"`, `"balance"`, `"billing"`, or `"insufficient_quota"`.
   - When `_is_quota_or_auth_error(...)` is `True`, raise `JevQuotaError(f"JEV credit/quota or auth error ({status_code}): {body_text[:500]}")`.
3. Because `JevQuotaError` subclasses `JevError`, update **every** `except JevError` block in `src/bionym/resolver.py` and `src/bionym/ask.py` to re-raise `JevQuotaError` immediately:
   ```python
   except JevQuotaError:
       raise
   except JevError as e:
       ...
   ```
   Also inside `Resolver._map` / `ThreadPoolExecutor` blocks (`_ask_chunks`, `_propose_claims`, `_propose_data_claims`), ensure that if one worker raises `JevQuotaError`, pending futures are cancelled (`ex.shutdown(wait=False, cancel_futures=True)`) and `JevQuotaError` propagates cleanly out of `resolver.resolve()`.

### 2.2 Fix Cumulative Usage Bug & Add LLM Usage + Latency Telemetry to `Resolver`

**Bugs in current code**:
1. Look at `evals/run.py` lines 132–154 and `evals/results/20260927-214528-baseline-real.jsonl`: `run.py` reuses a **single** `JevClient` instance across all cases, and `jev.usage_log` is **never reset between resolves**! Consequently, `graph.metadata["jev_usage"]["total"]` accumulated tokens across earlier cases (`394` → `4283` → `4692` → `5160` → `100255`).
2. `LlmClient` records `self.usage_log` (`input_tokens`, `output_tokens`), but `Resolver.resolve()` does **not** record `llm_usage` or per-call latencies in `graph.metadata`!

**Required fix**:
1. Add per-call wall-clock latency (`ms`) and `cached: bool` to both `JevClient.usage_log` and `LlmClient.usage_log`:
   - In `JevClient._call()`: return `(answers, usage, ms, from_cache)` so cached disk hits record `cached=True, ms=..., billable_input_tokens=0, billable_output_tokens=0` alongside the original `input_tokens`/`output_tokens` (preserving exact reproducibility while tracking *actual remaining credit burn* separately from cached replay tokens!).
   - In `LlmClient.complete()`: record `{"input_tokens": ..., "output_tokens": ..., "ms": ..., "cached": ...}` in `self.usage_log`, and add `LlmClient.total_usage()` mirroring `JevClient.total_usage()`.
2. In `Resolver.resolve()`, snapshot `jev_start = len(self.jev.usage_log)` and `llm_start = len(self.llm.usage_log)` at the start of `resolve()` (and update after `summarize`), so `graph.metadata["jev_usage"]` and a new `graph.metadata["llm_usage"]` report **strictly the calls made for that single `resolve()` invocation**, never cumulative leakage from previous resolves on the same client!
3. Write unit tests in `tests/test_jev_quota_and_usage.py` verifying:
   - HTTP `402` / `401` / quota-`429` raises `JevQuotaError` without retrying and aborts `Resolver.resolve()` even inside `_ask_chunks`, `_propose_claims`, `_propose_data_claims`, and `summarize`.
   - Two back-to-back `resolver.resolve()` calls on the same `Resolver` instance record independent per-graph `jev_usage` and `llm_usage`.
   - Disk-cached `JevClient` hits set `cached=True` so resumed partial runs do not re-bill already-completed JEV calls.

---

## 3. Resumable Runner Architecture & ~$4 JEV Budget Plan

### 3.1 How Much Does ~$4 of JEV Buy? (Token Economics & Execution Order)
From `evals/results/20261107-jev-timing.results.json`:
- Three *deep* *P. falciparum* resolves (`ama1`, `crt`, `kelch13` — among the heaviest possible pathogen genes with hundreds of VEuPathDB datasets/abstracts) totaled **469 JEV asks = 1,123,370 input tokens + 66,566 output tokens** (~374k input / ~22k output tokens per deep hub gene).
- From `evals/results/20260927-214528-baseline-real.jsonl`:
  - A **refusal / non-gene input** (`abc123`, `K00001`, DOI, `GCA_*`) costs **0 to 1 JEV call** (~0–500 input tokens, < $0.001).
  - A **Quick Scan (`--no-propose`)** on a rich gene (`PF3D7_0710100`) costs **~41 JEV calls (~95k input tokens)**; on typical bacterial/viral/helminth/obscure genes with fewer fuzzy GEO hits, Quick Scan costs **0–15 JEV calls (~0–25k input tokens)**.
  - **Existing recorded corpus (`evals/asks-claimsplit.jsonl`)**: Already contains 469 real JEV asks / 3,592 questions at **$0 additional JEV cost**!

**Execution Priority Order (Cheapest / Highest-Leverage First)**:
To maximize manuscript value before the ~$4 balance is touched — and ensure every phase lands cleanly in git before starting the next — execute suites in this strict order:
1. **Phase A ($0 JEV cost — offline analysis of recorded data)**: Quantitative analysis of the 3,592 recorded JEV questions vs. all 9 replayed open-JEV / LLM judges + generator replays + runtime comparisons (**Suite 1A** and **Suite 5A**).
2. **Phase B (~0.05M JEV tokens — pennies)**: Controlled synthetic/semi-synthetic JEV evidence-sensitivity & counterfactual probe suite (~80–100 targeted questions across 25 states testing monotone score shifts when evidence is strong vs. hedged vs. wrong-species vs. contradicted; **Suite 1B**).
3. **Phase C (~0.05M JEV tokens — pennies)**: Dedicated Refusal, Nonsense, Wrong-Entity, and Ambiguity-Trap Benchmark (~36 inputs; mostly 0–2 JEV calls each; **Suite 2**).
4. **Phase D (~1.5M–2.5M JEV tokens — Quick Scan across the Tree of Life)**: ~64 valid gene identifiers spanning 7 taxonomic groups $\times$ 3 annotation-density tiers (`well_studied`, `moderate`, `understudied`) run in **Quick Scan (`propose=False`)** first (**Suite 3A**).
5. **Phase E (~1.2M–2.2M JEV tokens — Stratified Grounded vs. Bare-LLM `bionym ask` Accuracy & Multi-Run Stability Benchmark Across Information Tiers + Ambiguous IDs)**: Full `propose=True, summarize=True` resolve + standardized 4-question `bionym ask` battery vs. bare ungrounded TACC `MiniMax` across **12 targets** (**3 understudied** + **3 moderately studied** + **3 heavily studied** pathogen genes + **3 ambiguous identifier cases**), run across **`K = 3` independent live replicates per `(target, question)`** (under both production `temperature=0` and stochastic `temperature=0.7`, plus `K = 3` full end-to-end `resolve + ask` rebuilds on 4 anchor targets) to quantify both accuracy and run-to-run response stability, alongside a 100% edge-and-quote audit on the understudied tier (**Suite 4**; all 9 unambiguous pathogen genes share their primary Full graphs with Suite 3B so zero JEV work is duplicated!).
6. **Phase F (~1.8M–3.5M JEV tokens — Remaining Stratified Deep `propose=True, summarize=True` Genes & Cold/Warm Timing Panel)**: Complete full `propose=True, summarize=True` resolves on the remaining genes in the 21-gene balanced Tree-of-Life subset (skipping the 9 already resolved in Phase E via `evals/.cache` / shared artifacts) + run the cold/warm timing panel (**Suite 3B** & **Suite 5B**). Phases E–F are the stages most likely to trigger a mid-run $4 refill pause — and thanks to persistent `diskcache` + per-item JSONL checkpointing, pausing and resuming loses **zero** JEV calls.

### 3.2 Checkpoint, Cache, and Stop-on-Depletion Specification (`evals/harness.py`)
Build a shared runner module `evals/harness.py` used by every live evaluation script:
1. **Persistent Disk Cache (`BIONYM_CACHE_DIR`)**:
   - Default to `evals/.cache` (gitignored). Pass `cache_dir="evals/.cache"` to `JevClient`, `LlmClient`, `NcbiClient`, and `VEuPathDBClient`.
   - Why this is essential: If case `#17` completes 30 JEV calls and runs out of credit on call `#31`, those 30 calls are already committed in `evals/.cache/jev`. When the user refills their TypeSafe balance and runs `--resume`, case `#17` replays calls `1..30` from `diskcache` in `< 20 ms` at **$0 cost** and resumes live billing at call `#31`!
   - Provide `--cold-cache` (or a separate cache dir per timing run) ONLY for the small dedicated cold-latency benchmark in Suite 5B, so cold wall-time numbers are uncontaminated by cache hits.
2. **Append-Only Checkpointed JSONL + State File**:
   - Every suite writes to a deterministic run directory `evals/runs/<run_id>/`:
     - `manifest.json`: `{run_id, suite, started_at, updated_at, status: "running"|"paused_quota"|"completed", git_sha, llm_model, llm_base_url, jev_model, jev_url, total_items, completed_items, cumulative_jev_tokens_billable, cumulative_llm_tokens}`
     - `rows.jsonl`: one JSON object per completed item, flushed (`fh.flush(); os.fsync(fh.fileno())`) immediately after each item completes.
     - `graphs/<case_name>.json`: full serialized `KnowledgeGraph` per resolved case (gitignored if large, or kept in `evals/runs/<run_id>/graphs/`).
   - **Resume logic (`--resume` or re-running with same `--run-id`)**:
     - Read `rows.jsonl`, collect `completed_ids = {row["item_id"] for row in rows if row["status"] == "ok"}`.
     - Skip any item in `completed_ids`. Re-run any item not in `completed_ids` (using `evals/.cache` to recover any partial stage calls).
3. **Graceful Stop Banner on `JevQuotaError` or Optional Soft Token Ceiling (`--max-jev-input-tokens`)**:
   - Catch `JevQuotaError` at the top-level item loop:
     1. Update `manifest.json` with `"status": "paused_quota"`, `"paused_on_item": item_id`, `"pause_reason": str(err)`, and cumulative token totals.
     2. Print clearly to `stderr` and `stdout`:
        ```text
        ======================================================================
        [BIONYM EVAL PAUSED — JEV CREDITS / QUOTA EXHAUSTED]
        Suite:          <suite_name> (run_id=<run_id>)
        Completed:      <done>/<total> items saved to evals/runs/<run_id>/rows.jsonl
        Paused during:  <item_id> (partial JEV/LLM/API calls cached in evals/.cache)
        Billable JEV:   <in_tok> input / <out_tok> output tokens used this session
        Reason:         <JevQuotaError message>

        ACTION REQUIRED:
          1. Refill TypeSafe JEV credits at https://console.typesafe.ai
          2. Resume with zero lost work:
             python -m evals.<script> --run-id <run_id> --resume
        ======================================================================
        ```
     3. Exit with status code `75`.

---

## 4. Detailed Specification of the Five Evaluation Suites

### Suite 1: JEV Confidence Distribution, Evidence Responsiveness, and Comparison to Alternatives
**Goal**: Quantitatively demonstrate that (a) `jev-latest` produces a rich, non-degenerate confidence distribution, (b) its scores shift meaningfully and monotonically across *question types / pipeline stages* and *evidence quality*, and (c) this distribution and evidence responsiveness are substantially more reasonable than open *System One* checkpoints (`Laya`, `opendecider-nano`) and prompted chat LLMs (`Llama-3.1-8B`, `Qwen3-32B`, `DeepSeek-V3.2`, `gpt-oss-120b`, `GLM-5.3-Flash`).

#### Part 1A — Offline Distributional & Question-Type Analysis (`evals/eval_jev_distributions.py`, $0 JEV cost)
Use the existing `evals/asks-claimsplit.jsonl` (469 asks, 3,592 questions) and all replay files in `evals/results/2026110[67]-*`:
1. **Score distribution shape across models (`n = 3,161` `noul` questions)**:
   - Compute for `jev-latest` and every alternative (`opendecider-nano`, `laya-typed`, `laya-english`, `Llama-3.1-8B`, `GLM-5.3-Flash`, `Qwen3-32B` [TACC & OR], `gpt-oss-120b` [TACC], `DeepSeek-V3.2` [TACC]):
     - Quantiles (`min`, `p05`, `p10`, `p25`, `p50`, `p75`, `p90`, `p95`, `max`), mean, SD, IQR.
     - **Granularity / discretization metrics**: number of unique scores rounded to `0.01` (`uniq_2dp`, max 101), Shannon entropy over 20 equal-width `[0, 1]` bins ($H_{20}$, bits; higher = smoother utilization of probability scale without collapse), and **"round-number mass"** (fraction of scores landing on `{0.0, 0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0}`).
     - **Bimodality / degeneracy indicators**: fraction of exact `0.0` and `1.0` outputs (exposes `Llama-3.1-8B`'s 84.7% exact-zero collapse and `Laya`'s `0.20–0.60` compression where `data_proposals` flips 95.6% at `0.5`).
2. **Responsiveness to Question Type & Evidence Regime within `jev-latest`**:
   - Stratify JEV outputs by stage and question type, showing that JEV does *not* emit a flat prior:
     - `ncbi_link` (`noul`, $n=53$ — noisy cross-DB symbol candidates where many are false homologs): median `0.38`, IQR `[0.16, 0.65]`, rejecting most false symbol matches while passing true ortholog links.
     - `proposals` (`noul`, $n=1,639$ — LLM-proposed prose claims from abstracts, pre-triaged): median `0.92`, `p10 = 0.48`, rejecting ~12% of over-stated or off-gene prose claims while spreading uncertain inferences across `0.40–0.85`.
     - `data_proposals` (`noul`, $n=1,469$ — LLM-proposed quantitative claims backed by exact expression table rows): median `0.96`, `p10 = 0.78`, reflecting stronger direct numerical evidence than abstract prose.
     - `proposals_triage` (`score` `[0..3]`, $n=420$ — mixed relevant vs. tangential datasets/papers): bimodal/wide spread (`p10 = 0.45`, `p50 = 1.99`, `p90 = 2.88`, `uniq_2dp = 186`).
     - Also split `proposals` `noul` scores by **proposal type** (`condition` vs. `claim`), **citation count** (`1` unit vs. `2+` units), and **source kind** (`dataset` description vs. `publication` abstract).
   - Run non-parametric **Kruskal–Wallis** ($H$, $p$-value) and pairwise **Mann–Whitney U** + Cliff's $\delta$ effect sizes across regimes to report formal statistical evidence that JEV's distribution shifts strongly by question/evidence type.

#### Part 1B — Controlled Evidence-Ladder Counterfactual Benchmark (`evals/eval_jev_counterfactuals.py` + `evals/data/jev_counterfactuals.yaml`)
To prove *causally* (not just correlationally across stages) that JEV's score tracks evidence quality — and does so more smoothly than open/LLM alternatives — construct **30 controlled probe families $\times$ 4 evidence levels = 120 typed questions** (~30 small `systemone` asks, costing **< 40k JEV input tokens total, < $0.10**):
- Cover 3 BioNym `noul` tasks (10 families each):
  1. **`s1_resolve` single-candidate gene match** (`10 × 4 = 40` items)
  2. **`s5_expression` dataset relevance** (`10 × 4 = 40` items)
  3. **`proposals` claim verification against numbered source sentences** (`10 × 4 = 40` items)
- Within each family, hold the question/claim fixed and construct 4 deterministic **evidence rungs** (`L3_direct` > `L2_partial_or_hedged` > `L1_wrong_species_or_paralog` > `L0_unsupported_or_contradicted`):
  - *Example (`proposals`)*: Claim = `"erg11 expression increases upon fluconazole exposure"`.
    - `L3_direct`: Quote explicitly states *C. albicans ERG11* mRNA increased 4.8-fold after fluconazole treatment.
    - `L2_partial_or_hedged`: Quote states putative sterol pathway transcripts showed modest, non-significant upward trends under azole stress.
    - `L1_wrong_species_or_paralog`: Quote states *ERG3* (not *ERG11*) increased after fluconazole, while *ERG11* was unchanged / measured in a different species.
    - `L0_unsupported_or_contradicted`: Quote describes only culture temperature and media composition with no mention of fluconazole or *ERG11* expression change (the classic methodological-boilerplate trap).
  - *Example (`s5_expression`)*: Target gene = *Anopheles gambiae* `AGAP004707` (*Vgsc*).
    - `L3_direct`: RNA-seq of pyrethroid-resistant *A. gambiae* mosquitoes profiling *Vgsc* (`AGAP004707`).
    - `L2_partial_or_hedged`: Whole-transcriptome microarray of *A. gambiae* blood-feeding (right organism, broad assay, gene not focal in title).
    - `L1_wrong_species_or_paralog`: RNA-seq of *Drosophila melanogaster* `para` sodium channel (ortholog in wrong species).
    - `L0_unsupported_or_contradicted`: *Homo sapiens* breast cancer cell line ChIP-seq that happens to match a short symbol substring.
- **Run the exact same 120 counterfactual questions** through:
  1. `jev-latest` (`TYPESAFE_API_KEY`)
  2. TACC `MiniMax` (the production generator acting as prompted LLM judge via `replay_llm.py` logic)
  3. TACC `Qwen3-32B` (strongest TACC open LLM judge from §3.1)
  4. (Optional if local CPU runner available, else cite §3.1): `opendecider-nano` / `laya` (note: all 30 probe states are compact `< 1,200` tokens, so `opendecider-nano` **can** run all 120 without 2k-context 504s — giving `opendecider-nano` its fairest possible head-to-head test!).
- **Metrics reported**:
  - **Strict monotonic ordering rate** (% of 30 families where `score(L3) > score(L2) > score(L1) > score(L0)`) and pairwise concordance (Kendall's $\tau$ between evidence rung `{0, 1, 2, 3}` and predicted score).
  - **Separation gap**: median `score(L3) - score(L1)` (true match vs. wrong-species/paralog) and `score(L2) - score(L0)` (hedged/partial vs. unsupported), showing whether the judge distinguishes *uncertain/partial* (`L2`, ideally `0.35–0.75`) from *direct* (`L3`, `> 0.85`) rather than snapping both to `0.95` or `1.0`.
  - **AUROC** for accepting `{L3}` vs. rejecting `{L1, L0}` at standard thresholds (`0.50`, `0.85`).

---

### Suite 2: Refusal, Nonsense, Wrong-Entity, and Ambiguity-Trap Benchmark
**Goal**: Demonstrate that BioNym refuses nonsensical, non-gene, fabricated, or hopelessly ambiguous inputs cleanly (`outcome == "refused"`, zero fabricated functional/claim edges) rather than hallucinating a gene graph — while also documenting exact behavior on cross-species bare symbols.

Create `evals/data/refusal_and_ambiguity.yaml` (**40 test inputs**, costing minimal JEV tokens) across 5 categories:
1. **Pure Nonsense & Injection/Formatting Strings (`n = 8`, expected: `refused`)**:
   - `abc123`, `xyzzy_not_a_gene_99999`, `000000000`, `gene_of_unknown_function`, `SELECT * FROM genes`, `https://ncbi.nlm.nih.gov/gene/123`, `---`, `null`.
2. **Plausible-Looking Fabricated Accessions ("Near-Miss Checksum/Prefix Traps", `n = 8`, expected: `refused`)**:
   - Fake VEuPathDB locus tag: `PF3D7_9999900` (chromosome 99 does not exist in *P. falciparum*)
   - Fake Ensembl ID: `ENSG00009999999`
   - Fake UniProt accession: `Q9ZZZ9` (verify 404/empty in UniProt preflight; pick confirmed unallocated accession)
   - Fake RefSeq protein/gene: `YP_999999999.1`
   - Fake bacterial locus tag: `RV9999c` (*M. tuberculosis* H37Rv stops ~`Rv3924c`)
   - Fake *Trypanosoma* locus tag: `Tb927.99.99990`
   - Fake *Aspergillus* locus tag: `AFUA_99G99990`
   - Fake WormBase ID: `WBGene99999999`
3. **Valid Biological Identifiers of the WRONG Entity Type (`n = 12`, expected: `refused`)**:
   - Genome assemblies (`2`): `GCA_000002765.2` (*P. falciparum* 3D7), `GCF_000005845.2` (*E. coli* K-12)
   - Publications / DOIs / PMIDs (`2`): `10.1038/s41586-020-2649-2`, `PMID:12368864`
   - Pathways & Ortholog Groups (`3`): `K00001` (KEGG Orthology), `map00010` (KEGG glycolysis pathway), `OMA:HOG:E0800000` / `PF00069` (Pfam protein kinase domain)
   - Gene Ontology terms (`2`): `GO:0006915` (apoptotic process), `GO:0003677` (DNA binding)
   - Expression / Sequencing accessions (`2`): `GSE10000` (GEO series), `SRR10971381` (SRA run)
   - Taxonomy ID (`1`): `taxid:5833` (*Plasmodium falciparum* organism ID)
4. **Too-Ambiguous / Generic Short Tokens (`n = 6`, expected: `refused` OR inspected — must NEVER fabricate non-existent records)**:
   - Single-letter / ultra-generic tokens that collide across hundreds of unrelated entries or mean nothing specific without organism context: `p1`, `orf1`, `gp1`, `hypothetical protein`, `unknown`, `putative membrane protein`.
   - *Design check*: Before finalizing expectations, verify how `S0` + `S1` handles multi-word descriptive phrases like `"hypothetical protein"` and ultra-generic tokens like `"orf1"`. **Important**: If an input like `"hypothetical protein"` currently slips through `esearch` fuzzy text search and matches a random bacterium's gene, **record that finding AND tighten `S0`/`S1`** (if appropriate and general, e.g., rejecting whitespace-containing prose descriptions that are not `symbol + organism` or requiring `S1` `choice` `none` when hits span dozens of unrelated phyla with no exact symbol match) — or report exact refusal rate honestly. Remember user code convention: *avoid overly specific brittle regex hacks; keep logic clean and general*.
5. **Valid Disambiguable Cross-Species / Retired-Alias Controls (`n = 6`, positive controls so refusal isn't achieved by rejecting everything!)**:
   - Retired alias: `MAL7P1.27` → must resolve to `PF3D7_0709000` (*pfcrt*)
   - Phenotype alias: `Lps` (or `DARC`) → resolves to canonical locus (`Tlr4` / `ACKR1`)
   - NCBI-to-VEuPathDB numeric GeneID bridge: `2655077` → `PF3D7_0710100`
   - Species-qualified or canonical symbol: `PF3D7_1343700`, `P38398`, `b0002` (*E. coli* `thrA`).

**Metrics for Suite 2 (`evals/eval_refusals.py`)**:
- Overall **Refusal Specificity (True Rejection Rate)** across Categories 1–3 (`expected: refused`), broken down by whether refusal happened deterministically at `S0`/`S1` (0 JEV calls) vs. via JEV `choice="none"` / `noul < 0.5`.
- **Zero-Fabrication Check**: Verify that 100% of refused graphs contain `0` functional, orthology, dataset, publication, or claim edges (only the audit `idtype:*` node), and `0` invented accessions in resolved positive controls.
- Wall time and JEV token cost per refused input (demonstrating fail-fast cost of fractions of a second and ~0–500 tokens).

---

### Suite 3: Tree-of-Life & Information-Density Matrix (`evals/data/tree_of_life_corpus.yaml`)
**Goal**: Demonstrate comparable, reliable utility across the **entire tree of life** (all 7 BRC-relevant taxonomic clades + vertebrate hosts) AND across **genes with abundant, moderate, and sparse/minimal published information**.

#### 3.1 Corpus Design: `7 Clades × 3 Information-Density Tiers × 3 Genes = 63 Valid Genes`
Curate `evals/data/tree_of_life_corpus.yaml` with **63 verified real gene identifiers** (verify every single ID via live NCBI / UniProt / VEuPathDB API lookup in a corpus-verification script `evals/verify_corpus_ids.py` *before* spending any JEV tokens!).

Each of the **7 clades** has **9 genes** (`3 well_studied`, `3 moderate`, `3 understudied`):

1. **Bacteria** (e.g., *Mycobacterium tuberculosis*, *Escherichia coli*, *Staphylococcus aureus*, *Klebisella pneumoniae*, *Borrelia burgdorferi*, *Burkholderia pseudomallei*):
   - *Well-studied (`3`)*: e.g., `Rv1484` (*inhA*, *M. tuberculosis*), `b3067` (*rpoD*, *E. coli*), `SAOUHSC_00020` / `mecA` (*S. aureus* UniProt `P07944`).
   - *Moderate (`3`)*: metabolic/transport genes with domain annotations and a few datasets.
   - *Understudied (`3`)*: conserved hypothetical / DUF-domain proteins in *M. tuberculosis* (e.g., `Rv2140c` / `Rv0071`), *B. burgdorferi*, or *B. pseudomallei* (`BPSL*` uncharacterized locus).
2. **Viruses** (e.g., SARS-CoV-2, Influenza A, Measles morbillivirus, Ebola, Mpox / Rift Valley fever / Nipah):
   - *Well-studied (`3`)*: e.g., `GU280_gp02` /GeneID `43740568` (SARS-CoV-2 *S*), Influenza A *HA* (GeneID `956529`), Ebola *GP* (GeneID `911829`).
   - *Moderate (`3`)*: viral matrix/nucleoprotein/polymerase genes (e.g., Measles *C*/*P* or RVFV *NSs*).
   - *Understudied (`3`)*: accessory / uncharacterized ORFs in bunyaviruses, paramyxoviruses, or poxviruses (e.g., Mpox or Oropouche/SFTSV accessory locus with sparse literature).
3. **Apicomplexan & Kinetoplastid Protists (VEuPathDB)** (*Plasmodium falciparum*, *Toxoplasma gondii*, *Cryptosporidium parvum*, *Trypanosoma brucei*, *Leishmania major*):
   - *Well-studied (`3`)*: `PF3D7_1343700` (*kelch13*), `PF3D7_0709000` (*pfcrt*), `TGME49_200010` / `TGME49_221620` (*T. gondii* well-studied surface/rhoptry gene).
   - *Moderate (`3`)*: *T. brucei* (`Tb927.*`), *L. major* (`LmjF.*`), *P. vivax* (`PVX_081515`).
   - *Understudied (`3`)*: `cgd*` (*Cryptosporidium parvum* conserved unknown protein), *Babesia microti* (`BMIC*` / `BMR1_*`), or *Entamoeba histolytica* (`EHI_*`) uncharacterized locus.
4. **Pathogenic Fungi** (*Candida albicans*, *Candida auris*, *Cryptococcus neoformans*, *Aspergillus fumigatus*, *Coccidioides immitis*):
   - *Well-studied (`3`)*: *C. albicans* *ERG11* (GeneID `3641571`), *A. fumigatus* *cyp51A* (`AFUA_4G06890`), *C. neoformans* `CNAG_00047` / *CAP59*.
   - *Moderate (`3`)*: fungal transcription factors / cell-wall enzymes.
   - *Understudied (`3`)*: *Candida auris* (`B9J08_*` uncharacterized protein — emerging multidrug-resistant pathogen!), *Coccidioides* or *Histoplasma* hypothetical protein.
5. **Helminths / Parasitic Worms** (*Schistosoma mansoni*, *Brugia malayi*, *Onchocerca volvulus*) + *C. elegans* comparator:
   - *Well-studied (`3`)*: *S. mansoni* `Smp_*` (e.g. *Sm14* / praziquantel target *Sm.TRPM_PZQ*), *C. elegans* `WBGene00000898` (*daf-2*).
   - *Moderate (`3`)*: *Brugia malayi* `Bm*` / NCBI GeneID, *S. mansoni* tegumental protease.
   - *Understudied (`3`)*: *Brugia malayi* or *Schistosoma mansoni* uncharacterized `Smp_*` locus.
6. **Arthropod Vectors (VectorBase / VEuPathDB)** (*Anopheles gambiae*, *Aedes aegypti*, *Ixodes scapularis*, *Culex quinquefasciatus*):
   - *Well-studied (`3`)*: `AGAP004707` (*A. gambiae Vgsc* / *kdr*), `AAEL023844` / *Ae. aegypti* odorant receptor *Orco*, *A. gambiae* *TEP1* (`AGAP010815`).
   - *Moderate (`3`)*: *Aedes* salivary/immunity gene, *Culex* detox P450.
   - *Understudied (`3`)*: *Ixodes scapularis* (`ISCW*` / NCBI locus, deer tick vector of Lyme disease) or *Lutzomyia* / *Phlebotomus* sandfly uncharacterized gene.
7. **Vertebrate Hosts & Model Plants/Eukaryotes** (*Homo sapiens*, *Mus musculus*, *Gallus gallus*, *Arabidopsis thaliana* — confirming general tree-of-life non-pathogen utility):
   - *Well-studied (`3`)*: Human `ACE2` (`ENSG00000130234` / GeneID `59272`), Mouse `Tlr4` (GeneID `21898`), *Arabidopsis* `AT1G01010` (*NAC001*) or *FLS2* (`AT5G46330`).
   - *Moderate (`3`)*: Human `ACKR1` (`DARC`), Chicken *IFITM3*, *Arabidopsis* metabolic enzyme.
   - *Understudied (`3`)*: Human "dark genome" / uncharacterized `C*orf*` or `TMEM*` gene (e.g., poorly characterized human `TMEM` or `FAM` gene with < 5 PubMed papers), plus poorly annotated avian/plant locus.

#### 3.2 Two-Stage Execution for Budget Control (`evals/eval_tree_of_life.py`)
- **Stage 3A — All 63 genes in Quick Scan (`propose=False, summarize=False`)**:
  - Tests `S0`–`S6` across every clade and information-density tier.
  - Verifies against pre-registered ground truth (`expected_taxid`, `expected_symbol_or_locus`, `info_tier`):
    1. **`S1` exact referent accuracy** (resolved to right gene & `taxon:<expected_taxid>`).
    2. **Structural richness by clade & info tier**: median node counts and edge counts broken down by facet (`assemblies` from `S2`/`S6`, `orthologs` from `S3`, `functional terms` GO/InterPro/Pfam/KEGG from `S4`, `expression datasets & publications` from `S5`).
    3. **Graceful scaling across information density**: Show quantitatively that for `well_studied` genes BioNym surfaces rich multi-database evidence, whereas for `understudied` genes it still reliably anchors taxonomy, genome assemblies (`S2`/`S6`), sequence orthogroups (`S3` OMA/OrthoMCL), conserved protein domains (`S4` InterPro/Pfam), and any whole-genome transcriptomic datasets (`S5`) — **without fabricating missing pathways or literature**.
- **Stage 3B — Stratified 21-gene Subset (`7 clades × 3 info tiers × 1 gene`) in Full Mode (`propose=True, summarize=True`)**:
  - Uses TACC `MiniMax` (`LLM_MODEL`) + `jev-latest`.
  - Measures:
    1. Number of candidate datasets/publications triaged vs. passed to LLM vs. proposals generated vs. proposals accepted by JEV (`>= 0.50` and `>= 0.85`).
    2. Verified `Claim` and `Condition` yield across `well_studied` vs. `moderate` vs. `understudied` tiers.
    3. Citation validity rate (`100%` deterministic unit-index validity; `0%` non-verbatim quote drift — verified programmatically by checking every `quote` substring segment split on `' … '` exists verbatim in the fetched upstream abstract/table!).

---

### Suite 4: Stratified Grounded (`bionym ask`) vs. Bare-LLM Q&A Across Evidence Tiers & Ambiguous Identifiers (`evals/eval_grounded_ask_vs_bare.py`)
**Goal**: Centering on **`bionym ask` responses vs. bare ungrounded TACC `MiniMax` responses**, quantify *both* what BioNym adds and what hallucinations/conflations it prevents across **four distinct epistemic regimes** (`12 targets × 4 standardized questions = 48 paired Q&A comparisons`), while retaining a 100% edge-and-quote audit on the understudied pathogen genes as direct proof against fabrication:
- **Heavily studied genes (`n = 3`)**: Does `bionym ask` improve **precision, versioned accession accuracy, and structured completeness** (specific assemblies, dataset IDs, orthogroup members, verified quotes) even where the bare LLM already "knows" the famous gene?
- **Moderately studied genes (`n = 3`)**: Where parametric LLM memory is patchy (knows the gene family or broad role, guesses specific datasets/assemblies/orthologs), how much does `bionym ask` boost supported factual yield while eliminating guessed accessions?
- **Understudied pathogen genes (`n = 3`)**: Where public literature is near-zero, does `bionym ask` surface genuine comparative/transcriptomic clues (assemblies, orthogroups, domains, bulk RNA-seq conditions) while **fabricating zero claims, zero pathways, and zero accessions** — whereas the bare LLM invents plausible-sounding enzyme roles or guesses the wrong organism?
- **Ambiguous identifiers (`n = 3`)**: When given a bare cross-taxa symbol, a generic multi-species token, or a retired/collision-prone alias, how does BioNym's explicit `S1` resolution/refusal + `bionym ask` compare against a bare LLM that silently conflates species or answers about the wrong gene?

#### 4.1 The 12 Target Identifiers (`evals/data/ask_stratified_panel.yaml` — Share the 9 Unambiguous Genes with Suite 3B!)
Verify all 12 targets via `verify_corpus_ids.py` first. Choose the 9 unambiguous genes directly from `tree_of_life_corpus.yaml` (`deep_subset: true`) across diverse pathogen clades so their `propose=True, summarize=True` graphs are built **once** and reused in Suite 3B:
1. **Stratum A — Heavily Studied Pathogen Genes (`n = 3`)**:
   - `PF3D7_1343700` (*Plasmodium falciparum* *kelch13*, artemisinin resistance — apicomplexan protist)
   - `Rv1484` (*Mycobacterium tuberculosis* *inhA*, isoniazid target — bacterium)
   - `AFUA_4G06890` (*Aspergillus fumigatus* *cyp51A* — or *C. albicans* *ERG11* `3641571`, azole target — pathogenic fungus; or SARS-CoV-2 *S* `43740568`)
2. **Stratum B — Moderately Studied Pathogen Genes (`n = 3`)**:
   - One moderately annotated kinetoplastid or *P. vivax* locus (e.g., `PVX_081515` or a *Trypanosoma brucei* `Tb927.*` metabolic/flagellar gene with ~5–20 papers and a handful of transcriptomic datasets)
   - One moderately annotated vector or viral gene (e.g., *Aedes aegypti* salivary/immune gene or Rift Valley fever virus *NSs*)
   - One moderately annotated bacterial or helminth gene (e.g., *Schistosoma mansoni* `Smp_*` protease or *Klebsiella*/*Burkholderia* transporter)
3. **Stratum C — Understudied Genes in Understudied Pathogens (`n = 3` — also receives 100% full-graph edge & quote audit)**:
   - ***Cryptosporidium parvum* or *Babesia microti*** uncharacterized / conserved apicomplexan locus (`cgd*` or `BMR1_*` with 0–2 PubMed links, no characterized enzyme name, present in VEuPathDB RNA-seq + OrthoMCL)
   - ***Candida auris*** uncharacterized locus (`B9J08_*`, emerging multidrug-resistant fungus where most loci lack gene-specific papers and only carry sequence domains, *C. albicans* orthologs, or bulk RNA-seq rows)
   - ***Burkholderia pseudomallei* or *Borrelia burgdorferi* / *Orientia tsutsugamushi*** uncharacterized DUF/hypothetical bacterial locus
4. **Stratum D — Ambiguous Identifier Cases (`n = 3` — covering all 3 distinct ambiguity sub-types)**:
   - **D1 (Cross-pathogen conserved bare symbol)**: `ama1` — conserved apical membrane antigen across *Plasmodium falciparum*, *P. vivax*, *Toxoplasma gondii*, *Babesia*, and *Neospora* (no organism specified). Tests whether BioNym either refuses or explicitly pins a single canonical `taxon:*` + `S1` choice probability distribution in the graph so `bionym ask` answers strictly about that declared referent, vs. the bare LLM blending facts/accessions across multiple apicomplexan species.
   - **D2 (Cross-kingdom homonym / phenotype-vs-gene collision)**: `CAT` (catalase across bacteria/fungi/mammals vs. chloramphenicol acetyltransferase in bacteria) or `env` / `pol` (retroviral polyprotein across dozens of viruses) or `Lps` (bacterial lipopolysaccharide vs. mouse `Tlr4` phenotype alias).
   - **D3 (Ultra-generic unresolvable symbol that MUST refuse)**: `orf1` (or `gp1`) — hits thousands of unrelated viral/plasmid records. Note: When `resolver.resolve()` returns `outcome == "refused"`, BioNym's `bionym ask` pipeline refuses to fabricate gene claims (test both (i) BioNym's pipeline-level refusal response and (ii) running `ask_graph` over the refused graph, verifying it states the identifier did not resolve to a gene — contrasted against what the bare LLM fabricates when asked the 4 questions about `orf1`!).

#### 4.2 Standardized 4-Question Battery & Multi-Run Replicate Protocol (`K = 3` Independent Live Runs)
For every target `<ID>`, evaluate **both** (1) `ask_graph(jev, llm, question, graph)` (or explicit refusal record if `outcome == "refused"`) and (2) **Bare Ungrounded Control** using the **exact same TACC `MiniMax` model** (system prompt asking for a concise, factual bioinformatics answer citing specific accessions/databases where known, and explicitly permitting *"unknown / insufficient evidence"* so the bare model is given a completely fair chance to abstain!):
- **Q1 — Identity, Organism & Genome Assemblies**: *"Which exact gene and organism does `<ID>` refer to, what aliases or cross-database accessions map to it, and in which genome assemblies is it annotated?"* (Tests referent accuracy, alias/accession precision, and ambiguity handling on Stratum D.)
- **Q2 — Conserved Domains, Function & Orthology**: *"What conserved protein domains, functional annotations, and orthologs across other species are established for `<ID>`?"* (Tests structured functional + comparative genomics recall vs. hallucinated enzyme functions.)
- **Q3 — Expression Datasets & Experimental Conditions**: *"In which public transcriptomic or functional-genomics datasets and experimental conditions has `<ID>` been measured, and what findings were reported?"* (Tests literature/data synthesis and dataset accession grounding — where bare LLMs almost never know real GEO/GXA/VEuPathDB dataset IDs even for famous genes!)
- **Q4 — Evidence-Boundary / Absent-Fact Calibration Probe**: A pre-verified **unestablished or negative probe** tailored per tier — e.g., *"Which human host cell-surface receptor has `<ID>` been crystallized in complex with, and what PDB structure or inhibitor $K_i$ is established for that complex?"* (verify via PDB/UniProt/graph during corpus setup that the answer is genuinely absent for the understudied/moderate targets, or use a paired known-absent specific mechanism for each target — testing **abstention precision** when evidence does not exist!).

**Repeated-Run Protocol (Quantifying How Answers Change — Or Stay Invariant — Across Responses)**:
- **CRITICAL CACHE WARNING FOR IMPLEMENTING AGENT**: `LlmClient.complete()` (`src/bionym/llm.py:78-82`) and `JevClient._call()` (`src/bionym/jev.py:135-140`) hash the request body into `diskcache` whenever `cache_dir` is set! If replicates share `evals/.cache`, replicates 2 and 3 will silently return cached strings from replicate 1 and falsely measure 100% agreement. Therefore, **isolate cache directories per replicate** (`evals/.cache/suite4_t0_rep1`, `evals/.cache/suite4_t0_rep2`, `evals/.cache/suite4_t0_rep3`, etc.) — preserving checkpoint/resume safety *within* each replicate while guaranteeing `K = 3` genuinely independent live API calls!
- Run **`K = 3` independent live replicates** under two matched regimes (add a clean `temperature: float = 0.0` parameter to `LlmClient.__init__` so default production behavior remains `0.0`):
  1. **Regime R1 — Production Greedy (`temperature = 0.0`, `K = 3` replicates across all `12 × 4 = 48` pairs = `144` paired runs, PLUS `K = 3` full end-to-end `resolve(propose=True, summarize=True) + ask` rebuilds on `4` anchor targets [1 heavy, 1 moderate, 1 understudied, 1 ambiguous])**: Tests run-to-run determinism of BioNym's shipped `T = 0` configuration — capturing both query-time MoE/vLLM numerical variance and upstream graph-construction repeatability when LLM proposal enumeration re-runs from scratch.
  2. **Regime R2 — Realistic Sampling (`temperature = 0.7`, `K = 3` replicates across all `12 × 4 = 48` pairs = `144` paired runs)**: Because sequential `temperature = 0.0` argmax decoding on a quiet GPU can produce bit-identical tokens even when a model's internal distribution over guessed facts is flat, also run `K = 3` replicates at standard chat sampling (`temperature = 0.7`) for both `bionym ask` and bare `MiniMax`. This directly tests whether graph grounding locks *factual content, accessions, cited nodes, and abstentions* in place even when surface phrasing varies across runs, whereas an ungrounded LLM invents *different* accessions, species, or mechanisms on each run!

#### 4.3 Quantitative Scoring Protocol: Single-Run Accuracy AND Cross-Run Stability
Compute **six concrete, reproducible quantitative metric families** (`1`–`5` for accuracy/grounding on primary replicate `r1` and pooled across replicates; `6` for cross-response stability across `K = 3` runs):

1. **Atomic Factual Claim Decomposition & Verification (`Supported` / `Unsupported-or-Fabricated` / `Abstention`)**:
   - For both the `bionym ask` prose answer and the bare `MiniMax` prose answer on each `(target, question)` pair:
     1. Decompose the prose answer into atomic factual statements (using a deterministic sentence/clause splitter + LLM claim-enumeration pass identical in style to `proposals.py`).
     2. Build an **independent Ground-Truth Reference Bundle** for `<ID>` directly from live upstream API records (`NCBI Datasets + E-utilities`, `UniProt`, `VEuPathDB`, `OMA`, `QuickGO`/`InterPro`, `GEO`/`GXA`, plus fetched PubMed abstracts — stored in `evals/runs/<run_id>/reference_bundles/<id>.json`). *Crucial methodological point*: Judge **both** `bionym ask` and bare `MiniMax` against this **same upstream primary-record bundle** (not solely against BioNym's internal graph), so the comparison is completely fair even when the bare LLM states a true biological fact from training memory that BioNym's APIs did or did not retrieve!
     3. Score every atomic claim via `jev-latest` (`noul` support against the upstream reference bundle) AND export all atomic claims for the 3 understudied + 3 ambiguous targets to a human-checkable CSV/Markdown audit sheet:
        - **Supported Claim Count** per answer (`noul >= 0.75` against upstream records) — measures *verified information yield*.
        - **Unsupported / Hallucinated Claim Rate (%)** (`1 - supported_claims / total_asserted_claims`, counting claims with `noul < 0.35` or contradicted organism/gene/function) — measures *fabrication rate*.
2. **Deterministic Identifier & Accession Validity Rate (`0` JEV tokens, 100% programmatic)**:
   - Regex-extract every concrete bioinformatics accession mentioned in either answer (`GCA_*`/`GCF_*`, `PF3D7_*`/`cgd*`/`B9J08_*`/`Rv*`/`Tb927.*`/`Smp_*`/`AGAP*`, `IPR*`, `PF*`, `GO:*`, `GSE*`/`E-*`, `PMID:*`, UniProt accessions, NCBI GeneIDs, PDB IDs).
   - Verify each extracted accession live against the corresponding upstream API / reference bundle on two levels:
     - **Existence rate**: Does this accession exist at all in NCBI / UniProt / EBI / VEuPathDB / GO / Pfam? (Catches pure hallucinated strings.)
     - **On-target linkage rate**: Is this accession actually linked to `<ID>` (right gene and right taxon), rather than borrowed from a paralog or model organism?
   - Also report BioNym's native `citation_fidelity` (`1 - len(invalid_cited_nodes)/len(cited_nodes)`) and JEV `ask` `accuracy` and `completeness` `noul` scores across all 4 strata.
3. **Abstention Accuracy on Q4 (Absent-Evidence Trap)**:
   - Binary + rate metric per stratum (`0/3`..`3/3`, and overall across 12 targets): Did the system explicitly report that the queried complex/pathway/measurement is *not established / absent from available evidence* (`Pass = 1`), or did it guess/fabricate a receptor, pathway, or accession (`Fail = 0`)?
4. **Ambiguity Handling Score on Stratum D (`D1`, `D2`, `D3`)**:
   - Measure:
     - **Referent transparency**: Does the response explicitly name the resolved organism + canonical accession (or explicitly refuse when unresolvable like `orf1`), vs. silently answering as if only one organism existed?
     - **Cross-species chimeric conflation rate**: Fraction of claims in a single answer that mix incompatible species/ortholog attributes without attribution.
5. **100% Edge-and-Quote Primary-Source Audit on the 3 Understudied Pathogen Graphs (`understudied_audit_<gene>.md`)**:
   - Keep the complete graph-level audit for all 3 Stratum C genes: verify every `confidence == 1.0` edge against raw API JSON fields and emit a complete `[Edge/Claim | Confidence | Source URL | Verbatim Quote | Unit Indices | Exact Substring Match Pass/Fail | Human Audit]` table for every `confidence < 1.0` edge and summary bullet.
6. **Cross-Response Stability & Drift Metrics Across the `K = 3` Replicates (Reported Separately for `T = 0.0`, `T = 0.7`, and End-to-End `Resolve + Ask`)**:
   - Across all $\binom{3}{2} = 3$ replicate pairs `(r1, r2), (r1, r3), (r2, r3)` for every `(target, question)` item, compute and compare **`bionym ask` vs. bare `MiniMax`** stratified by evidence tier (`heavy`, `moderate`, `understudied`, `ambiguous`):
     - **Extracted-Accession Jaccard Stability ($J_{\text{acc}} \in [0, 1]$)**: Mean pairwise $|A_i \cap A_j| / |A_i \cup A_j|$ over regex-extracted accessions (`1.0` when both replicates validly cite the same accessions or both validly abstain with $\emptyset$; drops toward `0.0` when a model invents different accessions on each run). For `bionym ask`, also report **`cited_nodes` Jaccard ($J_{\text{nodes}}$)** over graph node IDs.
     - **Bidirectional Factual Claim Entailment / Overlap ($[0, 1]$) & Self-Contradiction Rate (%)**: Using a batched JEV `noul` check (`"Is atomic claim c from response i also asserted or entailed by response j?"`, plus `"Does response j contradict claim c?"`), compute the mean fraction of atomic claims stable across runs and the fraction of `(target, question)` items exhibiting at least one **direct self-contradiction** across the 3 runs (e.g., naming *P. falciparum* in run 1 and *T. gondii* in run 2, or reporting incompatible domain/pathway roles).
     - **Abstention & Referent Flip Rate (%)**: Fraction of Q4 trap questions where the model flips between abstaining (`Pass`) and fabricating (`Fail`) across the 3 runs, and fraction of Stratum D ambiguous queries where the primary organism/gene referent flips across runs.
     - **Surface Lexical Variation vs. Semantic Invariance (`Exact-Match %` & `Content-Word Jaccard` vs. `Claim Entailment`)**: Pairing lexical overlap alongside factual claim entailment cleanly distinguishes *benign paraphrase variation* (words change at `T = 0.7`, while `Claim Entailment` and $J_{\text{acc}}$ stay high in `bionym ask`) from *substantive factual drift* (both words and facts/accessions change across runs in bare `MiniMax`).
     - **End-to-End Graph & Score Repeatability (`4` anchor targets, `K = 3` full resolves)**: Node/edge Jaccard across the 3 independently built graphs (`conf == 1.0` deterministic core vs. `conf >= 0.85` verified claim edges) and across-run `SD` / `max - min` of JEV `ask` `accuracy` and `completeness` scores.

**Expected Quantitative Gradient Across Tiers & Replicates (Hypothesis Tested by Suite 4)**:
- On **heavily studied genes**, bare `MiniMax` recalls true high-level biology with relatively high prose consistency on Q1/Q2 *but* guesses or drifts across runs on specific assembly, GEO/GXA dataset, and orthogroup accessions (Q1/Q3), whereas `bionym ask` delivers exact versioned accessions with high cross-run $J_{\text{acc}}$ and $J_{\text{nodes}}$.
- On **moderately studied genes**, bare `MiniMax`'s unsupported claim rate and cross-run accession/claim drift both rise sharply as it interpolates from model-organism homologs, while `bionym ask` maintains near-zero unsupported claims and stable citations.
- On **understudied pathogen genes** and **ambiguous IDs**, bare `MiniMax` exhibits both high fabrication/wrong-species conflation *and* severe run-to-run self-inconsistency (inventing different roles/accessions or flipping species across replicates, especially at `T = 0.7`), whereas `bionym ask` maintains ~`0%` fabricated accessions, invariant Q4 abstentions, and transparent referent pinning or clean refusal across all runs.

---

### Suite 5: End-to-End & Per-Component Runtime, Latency, and Token Performance (`evals/eval_performance.py`)
**Goal**: Report clear, apples-to-apples timing and token metrics answering:
1. *How fast is a BioNym resolve end-to-end — **without an LLM** (Quick Scan, `propose=False`) vs. **with an LLM** (Full Analysis, `propose=True, summarize=True`), and warm-cached?*
2. *Where does the wall time go across `S0`–`S6` (public API I/O vs. JEV calls vs. LLM calls)?*
3. *How do runtimes compare across **TypeSafe JEV** vs. **open SystemOne judges** (`Laya`, `opendecider-nano`) vs. **LLM judges** (`Qwen3-32B`, `DeepSeek-V3.2`, `gpt-oss-120b`, `GLM-5.3-Flash`) and across **LLM generators** (`MiniMax` vs. `DeepSeek-V3.2` on TACC)?*

#### Part 5A — Judge & Generator Model Runtime Comparison Table ($0 JEV cost — compiled + verified from `evals/results/*` and Suite 1B)
Synthesize a unified **Judge & Generator Runtime Table** across the identical 469-ask (3,592-question) workload and the 120-question Suite 1B benchmark:
- **TypeSafe `jev-latest` (`POST /v1/systemone`)**: `5 s` wall time @ 8 workers (`p50 = 87 ms`, `p90 = 119 ms` per ask across 469 asks; `~11 ms` amortized per question!). Break down `p50`/`p90` by stage (`s4_annotate` 75 ms, `ncbi_link` 77 ms, `s2_rank` 79 ms, `proposals` 86 ms, `data_proposals` 90 ms, `s4_pick` 142 ms, `proposals_triage` 163 ms).
- **Open *System One* CPU/local servers**:
  - `opendecider-nano` (0.8 GB CPU): `11,833 s` (~3.3 h; ~25 s/ask, plus 78.9% context-limit failures).
  - `Laya` `english` / `typed-decisions` (1.7 GB CPU): `12,783 s` – `22,970 s` (~3.6–6.4 h; ~27–49 s/ask).
- **Prompted Chat LLMs acting as Judge**:
  - Cloud Flash (`GLM-5.3-Flash`, OpenRouter): `466 s` (~1.0 s/ask @ concurrency).
  - Small Open LLM (`Llama-3.1-8B`, OpenRouter): `357 s` (fast, ~0.8 s/ask, but degenerate 80.1% flip rate).
  - Open-Weight HPC (`Qwen3-32B` on TACC): `763 s` (~1.6 s/ask — **~150× slower than JEV**, **16–30× faster than CPU Laya**, and **6–10× faster than `DeepSeek-V3.2` [`4,597 s`] and `gpt-oss-120b` [`7,948 s`] on TACC**).
- **Proposal Generator LLMs on TACC (395 generation asks)**:
  - `MiniMax-M3` on TACC: `4,431 s` (~11.2 s/ask @ 8 workers) vs. `DeepSeek-V3.2` on TACC: `8,867 s` (~22.4 s/ask) — **2.0× speedup** for MiniMax, plus per-call latency from the active TACC `MiniMax` model in Suite 3B.

#### Part 5B — End-to-End Pipeline Wall Time: No-LLM (`Quick`) vs. With-LLM (`Full`) vs. Warm Cache
Across the Suite 3 corpus (using **uncached / cold-cache timing** on a representative 12-gene panel spanning `well_studied`, `moderate`, and `understudied` loci, plus full stage-time distributions across all 63 Quick and 21 Full resolves):
1. **Mode 1 — Quick Scan (`--no-propose`, No LLM, JEV + Public APIs only)**:
   - Report median, IQR (`p25–p75`), and `p90` wall time (`s`), total JEV calls, and JEV input/output tokens, stratified by gene information density (`understudied` vs. `moderate` vs. `well_studied`).
   - Report per-stage breakdown (`S0`, `S1`, `S2`, `S3`, `S4`, `S5`, `S6`) showing that network I/O to upstream bioinformatics APIs (especially OMA `S3` and GEO/GXA `S5`), *not* JEV inference (`~87 ms/call`), dominates Quick Scan latency.
2. **Mode 2 — Full Analysis (`--propose --summarize`, TACC `MiniMax` LLM + JEV)**:
   - Report median, IQR, and `p90` wall time (`s`), LLM calls + tokens + LLM wall-time share, and JEV calls + tokens + JEV wall-time share on the **exact same genes** paired against Mode 1, quantifying the exact marginal time & token cost of turning on LLM claim extraction and verified summarization.
   - Optional second-LLM comparison on 3 genes (e.g. TACC `MiniMax` vs. TACC `Qwen3-32B` or `DeepSeek-V3.2` as generator) so a compact 3-gene paired table directly compares generator LLM impact on end-to-end Full-mode wall time!
3. **Mode 3 — Warm Cache (`BIONYM_CACHE_DIR` populated)** & **`bionym ask` Latency**:
   - Re-resolve the 12-gene timing panel against warm `diskcache` (and measure `KnowledgeGraph.from_json()` load time `< 50 ms`, mirroring a cached BRC gene page hit).
   - Measure `bionym ask` end-to-end latency across the 48 `ask` calls in Suite 4 (routing JEV call + 1 TACC `MiniMax` completion + scoring JEV call), stratified by graph size (`understudied` vs. `moderate` vs. `well_studied`).

---

## 5. Exact Files to Create or Modify

| File Path | Action | Responsibility |
|---|---|---|
| `src/bionym/jev.py` | **Modify** | Add `JevQuotaError`, `_is_quota_or_auth_error()`, per-call `ms` and `cached` tracking in `usage_log`, and billable vs. cached token accounting in `total_usage()`. |
| `src/bionym/llm.py` | **Modify** | Add optional `temperature: float = 0.0` parameter on `LlmClient`, per-call `ms` and `cached` tracking in `usage_log`, `total_usage()`, `<think>` tag stripping if emitted by reasoning models, and safe JSON-mode fallback if an endpoint 400s on `response_format`. |
| `src/bionym/resolver.py` | **Modify** | Re-raise `JevQuotaError` in all `except JevError` handlers (`_ask_chunks`, `_propose_claims`, `_propose_data_claims`, `summarize`), cancel worker futures on `JevQuotaError`, and slice `jev.usage_log` / `llm.usage_log` per `resolve()` call so `graph.metadata["jev_usage"]` and `graph.metadata["llm_usage"]` are strictly per-graph (never cumulative). |
| `src/bionym/ask.py` | **Modify** | Re-raise `JevQuotaError` before the degraded `except JevError` fallback. |
| `tests/test_jev_quota_and_usage.py` | **Create** | Unit tests covering `JevQuotaError` classification (401, 402, 403, billing-429 vs. rate-limit-429), propagation out of all `Resolver` stages, per-resolve usage isolation, and `diskcache` hit accounting. |
| `evals/preflight.py` | **Create** | Checks `.env` / env vars (without printing secrets), queries TACC `/v1/models` to resolve exact `MiniMax` model ID, tests 1 TACC JSON completion and 1 tiny ~20-token JEV ping, verifies `NCBI_API_KEY` / `VEUPATHDB_API_KEY` status, and prints active config summary. |
| `evals/harness.py` | **Create** | Shared resumable runner: `evals/.cache` wiring, `evals/runs/<run_id>/manifest.json` + fsynced `rows.jsonl`, `--resume` skip logic, soft token budget flag `--max-jev-input-tokens`, and exit-code-`75` `JEV CREDITS EXHAUSTED` pause banner. |
| `evals/verify_corpus_ids.py` | **Create** | Zero-JEV-cost script that queries NCBI / UniProt / VEuPathDB APIs directly to validate every accession in `refusal_and_ambiguity.yaml`, `tree_of_life_corpus.yaml`, and `ask_stratified_panel.yaml` before any paid JEV call is made. |
| `evals/data/jev_counterfactuals.yaml` | **Create** | 30 probe families $\times$ 4 evidence rungs = 120 controlled `noul` questions (`s1_resolve`, `s5_expression`, `proposals`). |
| `evals/data/refusal_and_ambiguity.yaml` | **Create** | 40 refusal, wrong-entity, near-miss fabricated ID, ambiguity-trap, and positive-control inputs. |
| `evals/data/tree_of_life_corpus.yaml` | **Create** | 63 valid genes (`7 clades × 3 info-density tiers × 3 genes`) with `expected_taxid`, `expected_symbol_or_locus`, `clade`, `info_tier`, and `deep_subset: true|false` (21-gene balanced subset). |
| `evals/data/ask_stratified_panel.yaml` | **Create** | 12 targets (`3 heavily_studied` + `3 moderately_studied` + `3 understudied` shared with `tree_of_life_corpus.yaml`, plus `3 ambiguous_id` cases `ama1`, homonym, `orf1`) $\times$ 4 standardized Q1–Q4 prompts (including pre-verified absent-evidence Q4 traps). |
| `evals/eval_jev_distributions.py` | **Create** | Suite 1A ($0 JEV): computes quantile, `uniq_2dp`, entropy $H_{20}$, round-number mass, and Kruskal–Wallis / Mann–Whitney stage & evidence-regime stats across JEV and all 9 replayed alternatives. |
| `evals/eval_jev_counterfactuals.py` | **Create** | Suite 1B: runs `jev_counterfactuals.yaml` across `jev-latest` and TACC LLM judges (`MiniMax`, `Qwen3-32B`, plus optional local `opendecider-nano`); computes monotonicity, Kendall's $\tau$, separation gaps, and AUROC. |
| `evals/eval_refusals.py` | **Create** | Suite 2: resumable runner over `refusal_and_ambiguity.yaml`; verifies refusal accuracy, stage of refusal (`S0`/`S1` deterministic vs. JEV), zero-fabrication check, and latency/tokens. |
| `evals/eval_tree_of_life.py` | **Create** | Suite 3: resumable runner with `--mode quick` (all 63 genes) and `--mode full` (21-gene stratified subset, reusing the 9 unambiguous Full graphs already resolved in Suite 4) + automated verbatim-quote & schema assertions. |
| `evals/eval_grounded_ask_vs_bare.py` | **Create** | Suite 4: resumable runner over the 12-target (`3 heavy + 3 moderate + 3 understudied + 3 ambiguous`) $\times$ 4-question battery comparing `bionym ask` vs. bare TACC `MiniMax` across `K = 3` independent live replicates (with isolated per-replicate cache dirs at `T = 0.0` and `T = 0.7`, plus `K = 3` end-to-end `resolve + ask` rebuilds on 4 anchor targets); computes upstream-bundle atomic claim support/hallucination, accession validity, Q4 abstention, Stratum D conflation, cross-run stability ($J_{\text{acc}}$, $J_{\text{nodes}}$, claim entailment overlap, self-contradiction & flip rates), and 100% edge-and-quote audit tables for the 3 understudied genes. |
| `evals/eval_performance.py` | **Create** | Suite 5: runs 12-gene cold-cache paired `Quick` vs. `Full` (TACC `MiniMax` + 3-gene `Qwen3-32B` generator comparison) vs. `Warm` timing panel, combines with Suite 1A/3/4 logs, and outputs unified runtime/token tables. |
| `evals/build_manuscript_tables.py` | **Create** | Aggregates all completed `evals/runs/*/rows.jsonl` and `evals/results/*` into one self-contained `evals/reports/MANUSCRIPT_EVAL_SUMMARY.md` + CSVs ready to drop directly into `manuscript.md`. |

---

## 6. Step-by-Step Execution Checklist for the Implementing Agent

Execute steps strictly in order. Do **not** skip tests or preflight verification.

### Step 1: Code Hardening & Unit Tests ($0 Cost)
1. Implement the `JevQuotaError`, per-resolve `jev_usage`/`llm_usage` slicing, `<think>` stripping / JSON-mode safety, and latency/cache telemetry in `src/bionym/jev.py`, `src/bionym/llm.py`, `src/bionym/resolver.py`, and `src/bionym/ask.py`.
2. Write `tests/test_jev_quota_and_usage.py` and run `pytest` across the full test suite (`pytest tests/`) — 100% of existing and new tests must pass before proceeding.

### Step 2: Build Data Files & Harness, Run Preflight & Zero-Cost API Verification (~$0.001 JEV Cost)
1. Create `evals/harness.py` and `evals/preflight.py`. Run `python -m evals.preflight` to lock the exact TACC `MiniMax` model ID and confirm JEV + TACC connectivity.
2. Author `evals/data/jev_counterfactuals.yaml` (120 items), `evals/data/refusal_and_ambiguity.yaml` (40 items), `evals/data/tree_of_life_corpus.yaml` (63 items across all 7 clades $\times$ 3 tiers, marking the 21 balanced `deep_subset` genes), and `evals/data/ask_stratified_panel.yaml` (12 targets: 3 heavily studied + 3 moderately studied + 3 understudied drawn from `deep_subset`, plus 3 ambiguous IDs, each with Q1–Q4).
3. Run `python -m evals.verify_corpus_ids` (queries only free NCBI / UniProt / VEuPathDB / PDB APIs, **0 JEV calls**) to confirm every positive gene ID exists with the expected `tax_id`, every Q4 absent-evidence probe is genuinely absent upstream, and every fabricated accession in Suite 2 returns no real locus. Fix any stale accession before touching JEV.

### Step 3: Phase A & B — JEV Score Distributions & Counterfactual Evidence Ladder (~$0.05–$0.10 JEV Cost)
1. Run `python -m evals.eval_jev_distributions` ($0 JEV cost) → writes `evals/runs/suite1a_distributions/`.
2. Run `python -m evals.eval_jev_counterfactuals --run-id suite1b_v1 --resume` (~30 small JEV asks + TACC `MiniMax` + TACC `Qwen3-32B`).

### Step 4: Phase C — Refusal & Ambiguity Benchmark (~$0.05–$0.10 JEV Cost)
1. Run `python -m evals.eval_refusals --run-id suite2_refusals_v1 --resume` (40 inputs).
2. Inspect any unexpected non-refusal on ambiguous/descriptive inputs; if a general `S0`/`S1` gap is exposed, document it clearly (and fix cleanly if warranted, re-running with a new `run_id`).

### Step 5: Phase D — Tree-of-Life Quick Scan Across All 63 Genes (~$0.80–$1.50 Estimated JEV Cost)
1. Run `python -m evals.eval_tree_of_life --mode quick --run-id suite3_quick_v1 --resume`.
2. If exit code `75` occurs (`JEV CREDITS EXHAUSTED`), **stop immediately** and report the exact checkpoint status and resume command to the user.

### Step 6: Phase E — Stratified `bionym ask` vs. Bare TACC `MiniMax` Accuracy & Multi-Run Repeatability (~$1.20–$2.20 Estimated JEV Cost)
1. Run `python -m evals.eval_grounded_ask_vs_bare --run-id suite4_ask_vs_bare_v1 --resume` across all 12 targets $\times$ 4 questions $\times$ `K = 3` independent live replicates (`T = 0.0` and `T = 0.7`, using isolated per-replicate cache dirs `evals/.cache/suite4_<regime>_rep<k>` so replicates never read each other's cached LLM/JEV responses, plus `K = 3` end-to-end `resolve + ask` rebuilds on the 4 anchor targets). Checkpoint after every replicate item so an exit-code-`75` credit pause loses zero work.
2. Inspect `understudied_audit_<gene>.md`, single-run accuracy metrics (`supported_claims`, `unsupported_claim_rate`, `valid_accession_rate`, `q4_abstention_rate`, `ambiguity_conflation_rate`), and cross-run stability metrics (`accession_jaccard`, `cited_nodes_jaccard`, `claim_entailment_overlap`, `self_contradiction_rate`, `abstention_and_referent_flip_rate`) across all 4 strata.

### Step 7: Phase F — Remaining Stratified Full-Mode Genes & Cold/Warm Timing Panel (~$1.00–$2.20 Estimated JEV Cost)
1. Run `python -m evals.eval_performance --run-id suite5_perf_v1 --resume` (12-gene cold Quick vs. cold Full vs. warm cache panel).
2. Run `python -m evals.eval_tree_of_life --mode full --run-id suite3_full_v1 --resume` (completes the remaining 12 genes of the 21-gene `deep_subset`, reusing the 9 unambiguous Full graphs already resolved in Step 6).
3. Again, if JEV credits run out mid-way, exit code `75` preserves every completed gene in `rows.jsonl` and every partial JEV/LLM call in `evals/.cache`; notify the user to refill and resume with `--resume`.

### Step 8: Build Manuscript-Ready Summary Report ($0 Cost)
1. Run `python -m evals.build_manuscript_tables` to generate `evals/reports/MANUSCRIPT_EVAL_SUMMARY.md` containing:
   - **Table S1 / Fig 2A**: JEV vs. Open-JEV (`Laya`, `opendecider-nano`) vs. Chat-LLM Judges — confidence quantiles, `uniq_2dp`, entropy $H_{20}$, round-number clustering, stage-by-stage distributions, counterfactual monotonicity (Kendall's $\tau$), and per-ask / total wall-clock runtimes.
   - **Table S2**: Refusal & Ambiguity Benchmark — rejection rates across nonsense, near-miss fake accessions, wrong entity types (assemblies, DOIs, KEGG/GO/Pfam/GEO/SRA), generic tokens, and positive alias controls, with zero-fabrication verification and fail-fast latency.
   - **Table S3**: Tree-of-Life $\times$ Information-Density Matrix (`7 clades × 3 tiers`, $n = 63$ Quick + $n = 21$ Full) — `S1` resolution accuracy, node/edge yields by biological facet, verified claim/condition counts, and 100% verbatim quote check.
   - **Table S4 / Fig 3 (Stratified `bionym ask` vs. Bare `MiniMax` Accuracy & Multi-Run Stability + Understudied & Ambiguous Vignettes)**: Quantitative comparison across **Heavily Studied (`n=3`)**, **Moderately Studied (`n=3`)**, **Understudied (`n=3`)**, and **Ambiguous (`n=3`)** targets (`48` target-question pairs $\times$ `K = 3` independent replicates at `T = 0.0` and `T = 0.7`, plus `K = 3` end-to-end `resolve + ask` rebuilds on 4 anchors) — reporting single-run accuracy (supported atomic claims, unsupported/hallucinated claim rate, verified on-target accession rate, Q4 abstention accuracy, Stratum D species-conflation rate) AND cross-run response stability (accession Jaccard $J_{\text{acc}}$, `cited_nodes` Jaccard $J_{\text{nodes}}$, bidirectional claim entailment overlap, self-contradiction rate, abstention/referent flip rate, and lexical vs. semantic variation), alongside the 100% edge-by-edge audit table for the spotlight understudied pathogen gene and side-by-side ambiguous-ID transcript (`ama1` / `orf1`).
   - **Table S5**: End-to-End Runtime & Token Economics — Quick Scan (no LLM) vs. Full Analysis (with TACC `MiniMax` vs. comparator LLM) vs. Warm Cache + `bionym ask` latency across graph sizes, broken down by `S0`–`S6` API I/O vs. JEV time vs. LLM time.
