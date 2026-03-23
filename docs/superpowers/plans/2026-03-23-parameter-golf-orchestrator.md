# Parameter Golf Local Research Orchestrator — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a multi-agent research orchestrator that scrapes the Parameter Golf competition frontier, ranks hypotheses, and runs controlled experiments locally on an RTX 5070 to produce a competitive submission candidate.

**Architecture:** Approach B — Phase 1 uses 3 parallel research subagents (GitHub Scout, Paper Scout, Leaderboard Analyzer) that merge into a frontier map. Phase 2 ranks hypotheses. Phases 3-4-5 run as a tight sequential experiment loop (implement → evaluate → promote/rollback). All orchestrated from a single orchestrator session.

**Tech Stack:** Python, PyTorch, torchrun, sentencepiece, zstd/zlib compression, Git branching, GitHub MCP, Context7 MCP, WebSearch/WebFetch, parallel Agent subagents

**Spec:** `docs/superpowers/specs/2026-03-23-parameter-golf-orchestrator-design.md`

---

## File Structure

```
ParameterGolf/
├── BACKBONE_POINTER.md                          # Existing — skill/tool routing reference
├── docs/superpowers/
│   ├── specs/2026-03-23-*.md                    # Design spec (exists)
│   └── plans/2026-03-23-*.md                    # This plan (exists)
├── research/
│   ├── evidence/
│   │   ├── github_submissions/                  # One .md per leaderboard entry
│   │   ├── papers/                              # One .md per relevant paper
│   │   └── discussions/                         # Blog posts, forum insights
│   ├── frontier_map.md                          # Unified comparison of all methods
│   ├── technique_taxonomy.md                    # Categorized: MODEL/QUANT/COMPRESS/EVAL/THROUGHPUT
│   ├── hypothesis_backlog.md                    # Ranked experiment queue
│   ├── synergy_matrix.md                        # Which ideas combine vs conflict
│   ├── experiment_registry.jsonl                # One JSON line per experiment run
│   └── problem_docs/
│       └── EXP-NNN_<name>.md                    # One problem doc per experiment
├── runs/
│   ├── EXP-000_baseline/                        # Phase 0 baseline
│   │   ├── train_log.txt
│   │   ├── metrics.json
│   │   └── config.json
│   └── EXP-NNN_<name>/                          # One dir per experiment
├── submission_candidate/                         # PR-ready when a winner emerges
│   ├── README.md
│   ├── submission.json
│   ├── train_gpt.py
│   ├── train_log.txt
│   └── requirements.txt
└── parameter-golf/                               # Cloned official repo (working copy)
    ├── train_gpt.py                              # Main training script (edit target)
    ├── data/                                     # Datasets + tokenizers
    └── records/                                  # Existing submissions
```

---

## Task 0: Project Setup & Repo Clone

**Files:**
- Create: `research/`, `runs/`, `submission_candidate/` directories
- Create: `research/experiment_registry.jsonl`
- Clone: `parameter-golf/` from GitHub

### Steps

- [ ] **Step 0.1: Create project directory structure**

```bash
cd "c:/Users/migue/OneDrive/Desktop/Personal/AILABS/ParameterGolf"
mkdir -p research/evidence/github_submissions
mkdir -p research/evidence/papers
mkdir -p research/evidence/discussions
mkdir -p research/problem_docs
mkdir -p runs/EXP-000_baseline
mkdir -p submission_candidate
```

- [ ] **Step 0.2: Initialize experiment registry**

Create `research/experiment_registry.jsonl` with empty file (will be appended to).

- [ ] **Step 0.3: Clone official Parameter Golf repo**

```bash
cd "c:/Users/migue/OneDrive/Desktop/Personal/AILABS/ParameterGolf"
git clone https://github.com/openai/parameter-golf.git
```

- [ ] **Step 0.4: Set up Python environment and install dependencies**

```bash
cd parameter-golf
python -m venv .venv
.venv/Scripts/activate   # Windows
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install numpy sentencepiece huggingface-hub datasets tqdm zstandard
```

Verify CUDA is available:
```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"
```
Expected: `True` and `NVIDIA GeForce RTX 5070` (or similar)

- [ ] **Step 0.5: Download FineWeb data (small subset for local iteration)**

```bash
python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1
```

This downloads ~100M tokens for training + the full validation set.

- [ ] **Step 0.6: Commit setup**

```bash
cd ..
git init
git add research/ docs/ BACKBONE_POINTER.md
git commit -m "chore: initialize Parameter Golf orchestrator project structure"
```

---

## Task 1: Phase 0 — Establish Local Baseline

**Files:**
- Read: `parameter-golf/train_gpt.py`
- Create: `runs/EXP-000_baseline/metrics.json`
- Create: `runs/EXP-000_baseline/config.json`
- Append: `research/experiment_registry.jsonl`

**Purpose:** Run the official baseline train_gpt.py locally with surrogate settings. Record val_bpb, artifact size, and throughput. This becomes the reference for all future experiments.

### Steps

- [ ] **Step 1.1: Read and understand the baseline train_gpt.py**

Use `Read` tool on `parameter-golf/train_gpt.py`. Study and **document**:
- Model architecture (layers, dims, heads, vocab)
- Training loop (optimizer, LR schedule, warmup, warmdown)
- Export/quantization path (int8, zlib compression)
- Evaluation path (val_bpb calculation)
- **CRITICAL: Extract the exact environment variable names** the script reads (e.g., `os.environ.get("MAX_WALLCLOCK_SECONDS")`, `os.environ.get("DATA_PATH")`, etc.). The training commands in Steps 1.2 and 7.6 use assumed env var names — if the actual script uses different names, update ALL training commands in this plan to match.
- Record the confirmed env vars in `runs/EXP-000_baseline/config.json` for future reference

**Skills to load:** `ag-python-pro`, `ag-ai-ml`
**MCP:** `mcp__context7__resolve-library-id` → `mcp__context7__query-docs` for PyTorch API docs if needed

- [ ] **Step 1.2: Run baseline training with surrogate settings**

```bash
cd parameter-golf
RUN_ID=baseline_local \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=180 \
VAL_LOSS_EVERY=50 \
python train_gpt.py 2>&1 | tee ../runs/EXP-000_baseline/train_log.txt
```

Note: Using single GPU (no torchrun needed for 1 GPU on Windows). Wall clock capped at 3 minutes.

- [ ] **Step 1.3: Record baseline metrics**

After training completes, extract from train_log.txt:
- `val_loss` (final)
- `val_bpb` (final)
- Compressed model size in bytes
- Training throughput (tokens/sec if reported)

Write to `runs/EXP-000_baseline/metrics.json`:
```json
{
  "experiment_id": "EXP-000",
  "title": "Official baseline (surrogate)",
  "val_bpb": <extracted>,
  "artifact_bytes": <extracted>,
  "tokens_per_sec": <extracted>,
  "wall_clock_seconds": 180,
  "gpu": "RTX 5070",
  "data_shards": 1,
  "status": "baseline"
}
```

Write surrogate config to `runs/EXP-000_baseline/config.json`:
```json
{
  "max_wallclock_seconds": 180,
  "train_shards": 1,
  "gpu_count": 1,
  "gpu_model": "RTX 5070",
  "vocab_size": 1024,
  "notes": "Local surrogate baseline. Same code paths as competition, reduced data and time."
}
```

- [ ] **Step 1.4: Append to experiment registry**

Append one JSON line to `research/experiment_registry.jsonl`:
```json
{"id":"EXP-000","title":"Official baseline (surrogate)","date":"2026-03-23","val_bpb":null,"artifact_bytes":null,"status":"baseline","branch":"main","hypothesis":"N/A","category":"BASELINE","promoted":true}
```
Fill in actual values after Step 1.3.

- [ ] **Step 1.5: Commit baseline results**

```bash
git add runs/EXP-000_baseline/ research/experiment_registry.jsonl
git commit -m "feat: establish local baseline EXP-000 with surrogate settings"
```

---

## Task 2: Phase 1A — GitHub Repo Scout (Parallel Subagent)

**Files:**
- Create: `research/evidence/github_submissions/*.md` (one per leaderboard entry)
- Create: `research/evidence/github_submissions/_DONE.md`

**Purpose:** Scrape every submission in openai/parameter-golf, extract what each changed, why it worked, and how ideas interact.

### Subagent Dispatch

This task runs as a **background subagent** using the `Agent` tool. Dispatch with:

```
Agent(
  subagent_type="technical-researcher",
  description="Scrape Parameter Golf submissions",
  run_in_background=true,
  prompt=<see below>
)
```

### Subagent Prompt

The subagent must:

1. **Use GitHub MCP** to list all folders in `openai/parameter-golf` under `records/track_10min_16mb/`:
   - `mcp__github__get_file_contents(owner="openai", repo="parameter-golf", path="records/track_10min_16mb")`

2. **For each submission folder**, fetch:
   - `README.md` — extract method description, reported val_bpb, key innovations
   - `train_gpt.py` — extract architecture changes, quantization code, eval code
   - `submission.json` — extract metadata (author, score, date)

3. **For each submission**, write a structured evidence note to `research/evidence/github_submissions/<folder_name>.md`:

```markdown
---
title: <submission name>
source_url: https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/<folder>
date: <date>
category: repo
author: <author>
val_bpb: <score>
---

## Key Method
<what they changed>

## Components Changed
- [ ] Architecture (layers, dims, MLP ratio)
- [ ] Quantization (int5, int6, int8, FP16)
- [ ] Training (optimizer, LR, WD, warmdown)
- [ ] Evaluation (sliding window, context length)
- [ ] Compression (zstd, zlib, level)
- [ ] Token features (SmearGate, BigramHash)
- [ ] Initialization (orthogonal, spectral)

## Reported Metric
val_bpb: <score>

## Likely Mechanism
<why this works>

## Improvement Category
<MODEL | QUANT | COMPRESS | EVAL | THROUGHPUT> (can be multiple)

## Interactions
<which other ideas this stacks with or conflicts with>

## Implementation Complexity
<low | medium | high>

## Worth Testing Locally
<yes | no — and why>
```

4. **Also fetch and analyze:**
   - The main repo README (competition rules, FAQ)
   - All open and closed PRs via `mcp__github__list_pull_requests`
   - PR diffs for top entries via `mcp__github__get_pull_request_files`

5. **Write completion marker:** `research/evidence/github_submissions/_DONE.md`

### Skills for this subagent
- `ag-competitive-landscape` — for structuring competitive analysis
- `ag-github-ops` — batch GitHub operations

### Tools for this subagent
- **Scrapling MCP** (primary web scraper — use for rendered pages, READMEs, challenge page):
  - `mcp__scrapling__fetch` — fetch and parse web pages with full rendering
  - `mcp__scrapling__stealthy_fetch` — anti-detection fetch for protected pages
  - `mcp__scrapling__get` — lightweight GET for raw content
- **GitHub MCP** (structured repo access — use for file contents, PRs, code search):
  - `mcp__github__get_file_contents` — fetch raw files from repo
  - `mcp__github__search_code` — search for techniques across submissions
  - `mcp__github__list_pull_requests` — enumerate PRs
  - `mcp__github__get_pull_request_files` — get PR diffs
  - `mcp__github__get_pull_request_comments` — read reviewer feedback
- `WebFetch` — last-resort fallback

### Steps

- [ ] **Step 2.1: Dispatch GitHub Scout subagent in background**

Use Agent tool with `run_in_background=true`. The prompt should include all instructions above.

- [ ] **Step 2.2: Verify completion**

When the agent completes, verify:
- At least 14 evidence notes created (one per leaderboard entry)
- `_DONE.md` marker exists
- Each note follows the schema above

---

## Task 3: Phase 1B — Paper/Academic Scout (Parallel Subagent)

**Files:**
- Create: `research/evidence/papers/*.md` (one per paper)
- Create: `research/evidence/papers/_DONE.md`

**Purpose:** Find and summarize academic papers on quantization, QAT, SWA, scaling laws, compression, and token-pair features relevant to Parameter Golf.

### Subagent Dispatch

This runs as a **second background subagent** simultaneously with Task 2.

```
Agent(
  subagent_type="academic-research-synthesizer",
  description="Research papers for Parameter Golf",
  run_in_background=true,
  prompt=<see below>
)
```

### Subagent Prompt

The subagent must search for and analyze papers on these topics:

**Search queries (use WebSearch):**
1. "quantization-aware training low-bit language models 2024 2025 2026"
2. "mixed precision quantization int4 int5 int6 transformer"
3. "stochastic weight averaging generalization quantization"
4. "neural scaling laws constrained parameters compute"
5. "compression-aware training language models"
6. "bigram features token-pair augmentation transformers"
7. "parameter-efficient language model training"
8. "post-training quantization vs QAT transformers"

**For each relevant paper found**, write to `research/evidence/papers/<short_name>.md`:

```markdown
---
title: <paper title>
source_url: <arxiv/openreview URL>
date: <publication date>
category: paper
authors: <author list>
---

## Key Idea
<one paragraph summary>

## Method Details
<specific technique, equations, or algorithm>

## Reported Results
<metrics, datasets, baselines>

## Relevance to Parameter Golf
<how this applies to the 16MB / 10min constraint>

## Implementation Complexity
<low | medium | high>

## Expected Impact
- Throughput: <yes/no>
- Compressed size: <yes/no>
- Post-quant loss: <yes/no>
- Raw train loss: <yes/no>

## Key Takeaway for Implementation
<1-2 sentences: what to actually DO based on this paper>
```

**Also use Context7 MCP** to fetch latest PyTorch quantization API docs:
- `mcp__context7__resolve-library-id` for "pytorch"
- `mcp__context7__query-docs` for quantization, QAT, AMP APIs

Write completion marker: `research/evidence/papers/_DONE.md`

### Skills for this subagent
- `ag-claude-scientific-skills` — scientific research methodology
- `ag-citation-management` — systematic citation tracking
- `ag-scientific-writing` — convert papers to actionable notes
- `ag-exa-search` — semantic search for academic papers
- `ag-tavily-web` — web search for blog posts and discussions

### Tools for this subagent
- `WebSearch` — primary search tool for discovering papers
- **Scrapling MCP** (fetch paper pages and extract content):
  - `mcp__scrapling__fetch` — fetch arXiv, OpenReview, blog post pages
  - `mcp__scrapling__stealthy_fetch` — for pages with bot detection
- `WebFetch` — fallback for paper page fetching
- **Context7 MCP** (library documentation):
  - `mcp__context7__resolve-library-id` — resolve PyTorch library ID
  - `mcp__context7__query-docs` — fetch PyTorch quantization/AMP docs

### Steps

- [ ] **Step 3.1: Dispatch Paper Scout subagent in background**

Use Agent tool with `run_in_background=true`, simultaneously with Task 2.

- [ ] **Step 3.2: Verify completion**

When the agent completes, verify:
- At least 8-12 paper notes created
- Each follows the schema above
- `_DONE.md` marker exists

---

## Task 4: Phase 1C — Leaderboard Analyzer (Parallel Subagent)

**Files:**
- Create: `research/evidence/discussions/*.md`
- Create: `research/evidence/discussions/_DONE.md`

**Purpose:** Deep quantitative analysis of the leaderboard progression, identifying which technique combinations produce the best scores and which ideas appear complementary.

### Subagent Dispatch

Third background subagent, runs simultaneously with Tasks 2 and 3.

```
Agent(
  subagent_type="data-analyst",
  description="Analyze Parameter Golf leaderboard",
  run_in_background=true,
  prompt=<see below>
)
```

### Subagent Prompt

The subagent must:

1. **Build the leaderboard table** from the competition data (provided in prompt or fetched via GitHub MCP):

| Rank | Run | Score | Key Techniques | Date |
|------|-----|-------|----------------|------|
| 1 | 10L Int5-MLP + BigramHash(10240) | 1.1428 | int5 MLP, int6 attn, FP16 embed, BigramHash, SWA(0.4), WD=0.04 | 2026-03-20 |
| 2 | Int6 MLP3x + SmearGate + BigramHash | 1.1458 | 3x MLP, SmearGate, BigramHash, OrthoInit, Muon WD, SWA | 2026-03-20 |
| 3 | 11L MLP3x + Int6 QAT | 1.1502 | 11 layers, 3x MLP, int6 QAT, zstd-22, WD=0.04, sliding eval | 2026-03-20 |
| ... | ... | ... | ... | ... |
| 14 | Naive Baseline | 1.2244 | 9L 512dim 1024vocab TiedEmbed 4KV | 2026-03-18 |

2. **Analyze technique frequency** across top entries:
   - Which techniques appear in 3+ of the top 5?
   - Which techniques appear only once (potential high-variance or niche)?
   - Which combinations always co-occur?

3. **Compute improvement attribution:**
   - Baseline → fp16 Embed: what did embedding precision add?
   - fp16 Embed → int6 mixed: what did quantization add?
   - Track the cumulative contribution of each technique family

4. **Identify the frontier:**
   - What is the gap between #1 and #2? (0.003 nats)
   - What does #1 have that #2 doesn't? (int5 MLP vs int6, BigramHash(10240) vs SmearGate+BigramHash)
   - What does #2 have that #1 doesn't? (SmearGate, OrthoInit, 3x MLP)
   - Which combination of both would likely be strongest?

5. **Write analysis** to `research/evidence/discussions/leaderboard_analysis.md`

6. **Search for community discussion** using WebSearch:
   - "parameter golf competition techniques 2026"
   - "parameter golf openai leaderboard analysis"
   - "parameter golf quantization strategy"
   Save any insights to `research/evidence/discussions/community_insights.md`

Write completion marker: `research/evidence/discussions/_DONE.md`

### Skills for this subagent
- `ag-andrej-karpathy` — deep learning expertise for interpreting technique interactions
- `ag-competitor-alternatives` — structured comparison framework
- `deep-research` — in-depth synthesis

### Steps

- [ ] **Step 4.1: Dispatch Leaderboard Analyzer subagent in background**

Use Agent tool with `run_in_background=true`, simultaneously with Tasks 2 and 3.

- [ ] **Step 4.2: Verify completion**

When the agent completes, verify:
- `leaderboard_analysis.md` exists with technique frequency analysis
- `community_insights.md` exists
- `_DONE.md` marker exists

---

## Task 5: Phase 1 Merge — Build Frontier Map

**Files:**
- Create: `research/frontier_map.md`
- Create: `research/technique_taxonomy.md`

**Purpose:** After all 3 Phase 1 subagents complete, merge their findings into a unified frontier map and technique taxonomy.

**Prerequisite:** Tasks 2, 3, and 4 must all be complete (_DONE.md markers exist).

### Steps

- [ ] **Step 5.1: Wait for all Phase 1 agents to complete**

All three Phase 1 subagents were dispatched with `run_in_background=true`. The system will **automatically notify you** when each agent completes — do NOT poll or sleep. Wait for all three completion notifications before proceeding.

After all three have returned, verify their output by checking for:
- `research/evidence/github_submissions/_DONE.md`
- `research/evidence/papers/_DONE.md`
- `research/evidence/discussions/_DONE.md`

If any agent failed, review its output and re-dispatch with a corrected prompt.

- [ ] **Step 5.2: Read all evidence files**

Read every `.md` file in:
- `research/evidence/github_submissions/`
- `research/evidence/papers/`
- `research/evidence/discussions/`

- [ ] **Step 5.3: Build frontier_map.md**

**Skill to load:** `ag-reference-builder`

Create `research/frontier_map.md` with:

```markdown
# Parameter Golf Frontier Map

Generated: 2026-03-23
Sources: <N> submissions, <M> papers, <K> discussion notes

## Leaderboard Summary

| Rank | Entry | val_bpb | Key Innovations | Category |
|------|-------|---------|-----------------|----------|
| 1 | ... | ... | ... | MODEL+QUANT+EVAL |
| ... |

## Technique Frequency (Top 7 entries)

| Technique | Count | Avg Rank | Category |
|-----------|-------|----------|----------|
| Sliding window eval | 6/7 | 3.2 | EVAL |
| Int6 quantization | 5/7 | 2.8 | QUANT |
| 3x MLP | 4/7 | 2.5 | MODEL |
| SWA | 4/7 | 2.0 | MODEL+QUANT |
| BigramHash | 3/7 | 1.7 | MODEL |
| ...

## Technique Interactions

### Confirmed Synergies
- SWA + aggressive quantization (smoother weights quantize better)
- 3x MLP + lower-bit quantization (MLP compresses well, saved bytes fund more capacity)
- ...

### Likely Conflicts
- ...

## Paper Insights

| Paper | Key Takeaway | Relevance |
|-------|-------------|-----------|
| ... | ... | ... |

## Unexplored Directions
<ideas from papers or analysis that NO submission has tried yet>
```

- [ ] **Step 5.4: Build technique_taxonomy.md**

Create `research/technique_taxonomy.md` categorizing every technique by type:

```markdown
# Technique Taxonomy

## MODEL (Architecture Improvements)
- 3x MLP expansion — used by entries #2, #3, #4, #5, #6
- Extra layers (10L, 11L) — used by entries #1, #3, #5
- SmearGate — used by entries #2, #4
- BigramHash — used by entries #1, #2, #4
- ...

## QUANT (Quantization/Export)
- Int6 block weights — used by entries #1-#6
- Int5 MLP — used by entry #1 only
- FP16 embeddings — used by entries #1, #7, #13
- Int8 embeddings — used by entry #6
- STE QAT — used by entries #3, #4, #5
- ...

## COMPRESS (Compression Efficiency)
- zstd-22 — used by entries #3, #5
- zlib — used by baseline and most others
- Byte reallocation (MLP→depth tradeoff) — used by entry #1
- ...

## EVAL (Evaluation-Only)
- Sliding window (stride=64) — used by entries #3-#8
- Extended context length — used by entry #10
- ...

## THROUGHPUT (Tokens/Second)
- <any kernel or systems optimizations observed>
```

- [ ] **Step 5.5: Commit frontier map**

```bash
git add research/frontier_map.md research/technique_taxonomy.md research/evidence/
git commit -m "feat: Phase 1 complete — frontier map and technique taxonomy from 14+ submissions and papers"
```

---

## Task 6: Phase 2 — Hypothesis Generation & Ranking

**Files:**
- Create: `research/hypothesis_backlog.md`
- Create: `research/synergy_matrix.md`
- Create: `research/problem_docs/EXP-001_*.md` through `EXP-006_*.md`

**Purpose:** Convert the frontier map into a ranked backlog of experiments to run locally.

**Prerequisite:** Task 5 (frontier map) must be complete.

### Steps

- [ ] **Step 6.1: Load relevant skills**

Load: `ag-ab-test-setup`, `ag-ai-ml`, `ag-andrej-karpathy`, `ag-goal-analyzer`

- [ ] **Step 6.2: Generate hypothesis backlog**

Read `research/frontier_map.md` and `research/technique_taxonomy.md`.

For each promising direction, score on 5 dimensions (1-5 each):

| Dimension | Weight |
|-----------|--------|
| Expected BPB improvement | 3x |
| Implementation complexity (inverse) | 2x |
| Evidence strength | 2x |
| Synergy potential | 1x |
| Local testability | 2x |

Write `research/hypothesis_backlog.md`:

```markdown
# Hypothesis Backlog

Ranked by weighted score. Updated after each experiment.

| Rank | ID | Hypothesis | Category | Score | Status |
|------|----|-----------|----------|-------|--------|
| 1 | EXP-001 | ... | MODEL+QUANT | 42 | pending |
| 2 | EXP-002 | ... | QUANT | 38 | pending |
| ...
```

The initial ranked directions (from the spec) are:
1. **EXP-001: Mixed int5/int6 quantization** — strongest evidence (SOTA uses this)
2. **EXP-002: 3x MLP expansion** — used by 4 of top 7
3. **EXP-003: Sliding window evaluation** — nearly free improvement, used by 6 of top 7
4. **EXP-004: SWA (Stochastic Weight Averaging)** — improves quantization robustness
5. **EXP-005: BigramHash token features** — SOTA uses BigramHash(10240)
6. **EXP-006: QAT schedule tuning** — interaction with SWA is undertested

- [ ] **Step 6.3: Build synergy matrix**

Create `research/synergy_matrix.md`:

```markdown
# Synergy Matrix

## Compatible Combinations (evidence-based)
| Combo | Evidence | Expected Interaction |
|-------|----------|---------------------|
| SWA + Int5/Int6 | Entry #1 uses both | SWA smooths weights → better quantization |
| 3x MLP + lower bit quant | Entries #2,#3 | MLP compresses well → saved bytes fund expansion |
| BigramHash + SmearGate | Entry #2 uses both | Complementary token-pair features |
| Sliding eval + any model change | 6/7 top entries | Orthogonal: eval improvement stacks with model improvements |

## Potentially Conflicting
| Combo | Risk |
|-------|------|
| Extra layers + larger MLP | Both consume bytes → may exceed 16MB |
| Int5 everywhere + QAT | QAT may need higher precision during training |
```

- [ ] **Step 6.4: Write problem documents for top 6 experiments**

For each experiment EXP-001 through EXP-006, create `research/problem_docs/EXP-NNN_<name>.md` using the template:

```markdown
Experiment ID: EXP-001
Title: Mixed Int5 MLP / Int6 Attention Quantization
Hypothesis: Quantizing MLP weights to int5 (instead of int6 or int8) saves enough bytes to fund an extra layer, improving overall loss while staying under 16MB.
Source evidence: Entry #1 (thwu1, 1.1428) uses exactly this — int5 MLP + int6 attn + FP16 embeddings
Expected mechanism: MLP weights are more uniformly distributed and compress better at lower bits. Attention weights have more outliers and need higher precision.
Primary metric: val_bpb
Secondary metrics: artifact_bytes, tokens_per_sec
Files allowed to change: parameter-golf/train_gpt.py
Files forbidden: data/*, evaluation scripts
Implementation scope: Modify quantization function to use 5-bit for MLP, 6-bit for attention. Add byte counting per module.
Run command: python train_gpt.py (with surrogate env vars)
Evaluation command: (built into train_gpt.py)
Artifact size check: Print compressed size, verify ≤ 16,000,000
Success criteria: val_bpb < EXP-000 baseline AND artifact ≤ 16MB
Rollback criteria: val_bpb ≥ baseline OR artifact > 16MB OR training fails
Risks: int5 may lose too much precision for some MLP layers. Need to check per-layer sensitivity.
Notes: SOTA entry uses this exact approach. High confidence.
```

Repeat for EXP-002 through EXP-006 with appropriate details.

- [ ] **Step 6.5: Commit Phase 2 artifacts**

```bash
git add research/hypothesis_backlog.md research/synergy_matrix.md research/problem_docs/
git commit -m "feat: Phase 2 complete — 6 ranked hypotheses with problem documents"
```

---

## Task 7: Phase 3-4-5 — Experiment Loop (Repeatable)

**Files:**
- Modify: `parameter-golf/train_gpt.py` (per experiment)
- Create: `runs/EXP-NNN_<name>/` (per experiment)
- Append: `research/experiment_registry.jsonl`
- Update: `research/hypothesis_backlog.md` (re-rank after results)
- Update: `research/frontier_map.md` (after promotions)

**Purpose:** Execute the top-ranked experiment from the backlog. This task is repeated for each experiment in the loop.

### Pre-Loop Setup

- [ ] **Step 7.0: Read the current top hypothesis from backlog**

Read `research/hypothesis_backlog.md`. Pick the highest-ranked `pending` experiment. Read its problem document from `research/problem_docs/`.

### Phase 3: Implement (Per Experiment)

- [ ] **Step 7.1: Create experiment branch**

```bash
cd parameter-golf
git checkout -b exp/EXP-NNN-<short-name>
```

- [ ] **Step 7.2: Implement the change**

**Skills to load:** `ag-python-pro`, `ag-ai-ml`

**Subagent dispatch (optional for complex experiments):**

For complex implementations, dispatch a subagent:
```
Agent(
  subagent_type="python-expert",
  description="Implement EXP-NNN",
  prompt="Read the problem document at research/problem_docs/EXP-NNN_<name>.md.
          Implement the change described in parameter-golf/train_gpt.py.
          Rules:
          - Only modify files listed in 'Files allowed to change'
          - Make the minimal change needed
          - Add clear comments marking the experiment change
          - Do NOT change unrelated code
          Return the exact diff when done."
)
```

For simpler experiments, implement directly using Edit tool.

- [ ] **Step 7.3: Verify the diff is clean and minimal**

```bash
git diff --stat
git diff
```

Verify:
- Only allowed files changed
- No unrelated modifications
- Changes match the problem document scope

- [ ] **Step 7.4: Save patch for records**

```bash
git diff > ../runs/EXP-NNN_<name>/patch.diff
```

### Phase 4: Evaluate (Per Experiment)

- [ ] **Step 7.5: Pre-training bug check**

**Skill to load:** `ag-find-bugs`

Quick scan of the modified code for:
- Syntax errors
- Shape mismatches
- Missing imports
- Off-by-one errors in quantization bit widths
- Broken export/compression path

- [ ] **Step 7.6: Run training**

```bash
RUN_ID=EXP-NNN_<name> \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=180 \
VAL_LOSS_EVERY=50 \
python train_gpt.py 2>&1 | tee ../runs/EXP-NNN_<name>/train_log.txt
```

- [ ] **Step 7.7: Extract and record metrics**

From train_log.txt, extract:
- Final `val_loss`
- Final `val_bpb`
- Compressed artifact size
- Training throughput

Write to `runs/EXP-NNN_<name>/metrics.json`:
```json
{
  "experiment_id": "EXP-NNN",
  "title": "<title>",
  "val_bpb": <value>,
  "artifact_bytes": <value>,
  "tokens_per_sec": <value>,
  "baseline_val_bpb": <EXP-000 value>,
  "delta_bpb": <value - baseline>,
  "wall_clock_seconds": 180,
  "status": "completed"
}
```

- [ ] **Step 7.8: Run validation checklist**

**Skill to load:** `verification-before-completion` (superpowers)

Verify ALL of:
- [ ] Training completed without errors
- [ ] Export produced a valid model artifact
- [ ] Compressed artifact ≤ 16,000,000 bytes
- [ ] val_bpb was measured
- [ ] No NaN/Inf in training logs
- [ ] Throughput recorded

### Phase 5: Decide — Promote or Rollback (Per Experiment)

- [ ] **Step 7.9: Compare against baseline**

Decision logic:
```
IF val_bpb < baseline_val_bpb AND artifact_bytes ≤ 16000000:
    → PROMOTE (this is the new baseline)
ELSE:
    → ROLLBACK (discard this experiment)
```

- [ ] **Step 7.10a: If PROMOTE — merge and update**

```bash
cd parameter-golf
git checkout main
git merge exp/EXP-NNN-<short-name> --no-ff -m "feat: promote EXP-NNN — <title> (val_bpb: X.XXXX)"
```

Update `research/experiment_registry.jsonl`:
```json
{"id":"EXP-NNN","title":"<title>","date":"2026-03-23","val_bpb":<value>,"artifact_bytes":<value>,"status":"promoted","branch":"main","hypothesis":"<hypothesis>","category":"<MODEL|QUANT|etc>","promoted":true}
```

Update `research/frontier_map.md` with new local best result.
Update `research/hypothesis_backlog.md` — mark this experiment `completed`, re-rank remaining.

- [ ] **Step 7.10b: If ROLLBACK — discard and log**

```bash
cd parameter-golf
git checkout main
git branch -d exp/EXP-NNN-<short-name>
```

Update `research/experiment_registry.jsonl`:
```json
{"id":"EXP-NNN","title":"<title>","date":"2026-03-23","val_bpb":<value>,"artifact_bytes":<value>,"status":"rollback","branch":"deleted","hypothesis":"<hypothesis>","category":"<MODEL|QUANT|etc>","promoted":false,"failure_reason":"<why it didn't beat baseline>"}
```

Update `research/hypothesis_backlog.md` — mark this experiment `failed`, add failure notes.

- [ ] **Step 7.11: Commit experiment results**

```bash
cd ..
git add runs/EXP-NNN_<name>/ research/experiment_registry.jsonl research/hypothesis_backlog.md
git commit -m "result: EXP-NNN <title> — <promoted|rollback> (val_bpb: X.XXXX)"
```

- [ ] **Step 7.12: Loop — return to Step 7.0 for next experiment**

Pick the next highest-ranked `pending` experiment from the backlog and repeat.

---

## Task 8: Synthesis Experiments (After Initial Loop)

**Files:**
- Create: `research/problem_docs/EXP-1NN_synthesis_*.md`
- Modify: `parameter-golf/train_gpt.py`

**Purpose:** After running isolated experiments, combine the best promoted changes into synthesis experiments. This is where the real competitive gains happen.

### Steps

- [ ] **Step 8.1: Review promoted experiments**

Read `research/experiment_registry.jsonl`. List all experiments with `"promoted": true`.

- [ ] **Step 8.2: Design synthesis experiment**

**Skills to load:** `ag-ai-ml`, `ag-andrej-karpathy`

Consult `research/synergy_matrix.md`. Combine 2-3 promoted changes that are known compatible.

Example synthesis:
```
EXP-101: Mixed quant (EXP-001) + 3x MLP (EXP-002) + Sliding eval (EXP-003)
```

Write problem document for the synthesis experiment. Tag it `SYNTHESIS`.

- [ ] **Step 8.3: Verify all promoted experiments are merged to main**

Before creating the synthesis branch, confirm every promoted experiment has been merged:
```bash
cd parameter-golf
git log --oneline main | grep "promote"
```
Each promoted experiment from Step 7.10a should appear. If any is missing, merge it first.

- [ ] **Step 8.4: Run synthesis through the same Phase 3-4-5 loop**

Follow Task 7 steps exactly. The synthesis branch starts from `main` which now contains all promoted changes.

- [ ] **Step 8.4: If synthesis beats best isolated result → becomes new baseline**

This synthesis result is likely the submission candidate.

---

## Task 9: Prepare Submission Candidate

**Files:**
- Create: `submission_candidate/README.md`
- Create: `submission_candidate/submission.json`
- Copy: `submission_candidate/train_gpt.py`
- Copy: `submission_candidate/train_log.txt`
- Create: `submission_candidate/requirements.txt`

**Purpose:** Package the best local result as a PR-ready submission folder.

**Prerequisite:** At least one promoted experiment (ideally a synthesis).

### Steps

- [ ] **Step 9.1: Identify the best promoted experiment**

Read `research/experiment_registry.jsonl`. Find the entry with the lowest `val_bpb` that has `"promoted": true`.

- [ ] **Step 9.2: Copy training script and logs**

```bash
cp parameter-golf/train_gpt.py submission_candidate/train_gpt.py
cp runs/EXP-NNN_<best>/train_log.txt submission_candidate/train_log.txt
```

- [ ] **Step 9.3: Write submission.json**

**Skill to load:** `ag-scientific-writing`

```json
{
  "name": "<submission name>",
  "author": "<your name>",
  "github_id": "<your github>",
  "val_bpb": <best score>,
  "date": "2026-03-23",
  "description": "<one-line summary>",
  "track": "track_10min_16mb"
}
```

- [ ] **Step 9.4: Write README.md**

**Skill to load:** `ag-scientific-writing`, `ag-pr-writer`

Write a detailed README explaining:
- What techniques were used and why
- Architecture changes
- Quantization strategy
- Training hyperparameters
- Evaluation method
- Results (val_bpb, artifact size, training time)
- Ablation results from the experiment registry

- [ ] **Step 9.5: Write requirements.txt**

```
torch>=2.0
numpy
sentencepiece
zstandard
```

- [ ] **Step 9.6: Verify submission candidate**

Run the training script from the submission_candidate directory to verify it works standalone. Use the same env vars as Step 1.2, with absolute paths:
```bash
cd submission_candidate
DATA_PATH=../parameter-golf/data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=../parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 \
MAX_WALLCLOCK_SECONDS=180 \
python train_gpt.py
```

Verify:
- Runs without errors
- Produces valid model artifact
- Artifact ≤ 16,000,000 bytes
- val_bpb matches expected

- [ ] **Step 9.7: Commit submission candidate**

```bash
git add submission_candidate/
git commit -m "feat: submission candidate ready — val_bpb X.XXXX"
```

---

## Task 10: Submit PR to openai/parameter-golf

**Files:**
- Fork + push to your fork of openai/parameter-golf
- Create PR adding `records/track_10min_16mb/<date>_<name>/`

**Purpose:** Submit the final PR to the competition.

**Prerequisite:** Task 9 complete. Ideally validated on H100s.

### Steps

- [ ] **Step 10.1: Fork the competition repo (if not already done)**

```bash
gh repo fork openai/parameter-golf --clone=false
```

- [ ] **Step 10.2: Clone your fork and add submission**

```bash
git clone https://github.com/<your-username>/parameter-golf.git parameter-golf-fork
cd parameter-golf-fork
git checkout -b submission/<date>_<name>
mkdir -p records/track_10min_16mb/<date>_<name>
cp ../submission_candidate/* records/track_10min_16mb/<date>_<name>/
```

- [ ] **Step 10.3: Commit and push**

**Skill to load:** `ag-commit`

```bash
git add records/track_10min_16mb/<date>_<name>/
git commit -m "submission: <name> — val_bpb X.XXXX"
git push -u origin submission/<date>_<name>
```

- [ ] **Step 10.4: Create PR**

**Skill to load:** `ag-create-pr`, `ag-pr-writer`
**MCP:** `mcp__github__create_pull_request`

```bash
gh pr create \
  --repo openai/parameter-golf \
  --title "<submission name> — val_bpb X.XXXX" \
  --body "$(cat <<'EOF'
## Summary
<technique summary>

## Results
- val_bpb: X.XXXX (averaged over 3 runs)
- Artifact size: XX,XXX,XXX bytes
- Training time: ~X minutes on 8xH100

## Techniques
<bulleted list of techniques used>

## Ablation Results
<table from experiment registry>

## Reproducibility
<exact commands to reproduce>
EOF
)"
```

- [ ] **Step 10.5: Monitor PR for reviewer feedback**

**Skill to load:** `ag-address-github-comments`

Check for reproduction issues or reviewer questions and respond promptly.

---

## Execution Dispatch Summary

### Parallel Subagents (Phase 1) — Dispatch ALL THREE simultaneously

| Subagent | Type | Task | Background? |
|----------|------|------|-------------|
| GitHub Scout | `technical-researcher` | Task 2 | Yes |
| Paper Scout | `academic-research-synthesizer` | Task 3 | Yes |
| Leaderboard Analyzer | `data-analyst` | Task 4 | Yes |

### Sequential Tasks (Phase 0, 2, 3-5)

| Task | Phase | Prerequisites | Subagent? |
|------|-------|---------------|-----------|
| Task 0: Setup | Setup | None | No (inline) |
| Task 1: Baseline | Phase 0 | Task 0 | No (inline) |
| Tasks 2-4: Research | Phase 1 | Tasks 0 + 1 (baseline must exist before experiments, but research can run in parallel with baseline) | Yes (3 parallel) |
| Task 5: Merge | Phase 1 merge | Tasks 2-4 | No (inline) |
| Task 6: Hypotheses | Phase 2 | Task 5 | No (inline) |
| Task 7: Experiment Loop | Phases 3-4-5 | Task 6 | Optional per experiment |
| Task 8: Synthesis | Post-loop | Task 7 promoted results | Optional |
| Task 9: Package | Submission | Best result | No (inline) |
| Task 10: Submit PR | Submission | Task 9 | No (inline) |

### Complete Skill Activation Map

| Task | Skills to Activate |
|------|-------------------|
| Task 0 | (none — setup only) |
| Task 1 | `ag-python-pro`, `ag-ai-ml` |
| Task 2 | `ag-competitive-landscape`, `ag-github-ops` (+ Scrapling MCP and GitHub MCP — see MCP Usage Map) |
| Task 3 | `ag-claude-scientific-skills`, `ag-citation-management`, `ag-scientific-writing`, `ag-exa-search`, `ag-tavily-web` |
| Task 4 | `ag-andrej-karpathy`, `ag-competitor-alternatives`, `deep-research` |
| Task 5 | `ag-reference-builder` |
| Task 6 | `ag-ab-test-setup`, `ag-ai-ml`, `ag-andrej-karpathy`, `ag-goal-analyzer`, `ag-progressive-estimation`, `ag-concise-planning`, `ag-data-structure-protocol` |
| Task 7 (implement) | `ag-python-pro`, `ag-ai-ml`, `ag-python-performance-optimization`, `ag-python-patterns`, `ag-closed-loop-delivery`, `ag-differential-review` |
| Task 7 (evaluate) | `ag-find-bugs`, `ag-systematic-debugging`, `ag-backtesting-frameworks`, `verification-before-completion` |
| Task 7 (decide) | `ag-agent-memory-systems`, `ag-evolution`, `ag-commit` |
| Task 8 | `ag-ai-ml`, `ag-andrej-karpathy` |
| Task 9 | `ag-scientific-writing`, `ag-pr-writer` |
| Task 10 | `ag-commit`, `ag-create-pr`, `ag-address-github-comments` |

### Complete MCP Usage Map

| Task | MCP Server | Tools Used |
|------|-----------|------------|
| Task 2 | **Scrapling** | `fetch`, `stealthy_fetch`, `get` — scrape rendered pages, challenge docs |
| Task 2 | **GitHub** | `get_file_contents`, `search_code`, `list_pull_requests`, `get_pull_request_files`, `get_pull_request_comments` |
| Task 3 | **Scrapling** | `fetch`, `stealthy_fetch` — fetch paper pages from arXiv, OpenReview, blogs |
| Task 3 | **Context7** | `resolve-library-id`, `query-docs` — PyTorch quantization docs |
| Task 4 | **Scrapling** | `fetch` — scrape leaderboard page for latest rankings |
| Task 4 | **GitHub** | `get_file_contents` — fetch submission data |
| Task 7 | **Context7** | `query-docs` — PyTorch APIs during implementation |
| Task 10 | **GitHub** | `create_pull_request`, `create_or_update_file` |

### Complete Plugin Usage Map

| Task | Plugin | Purpose |
|------|--------|---------|
| All | `superpowers` | Verification gates, plan execution |
| Task 7 | `superpowers:verification-before-completion` | Experiment validation |
| Task 10 | `commit-commands` | Git workflow |
| Task 10 | `pr-review-toolkit` | Final review before PR |
