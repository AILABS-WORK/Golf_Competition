# Parameter Golf Local Research Orchestrator — Design Spec

**Date:** 2026-03-23
**Status:** Approved
**Approach:** B — Parallel Research + Sequential Execution

---

## Mission

Build a local multi-agent research system that prepares a strong first Parameter Golf submission before external compute credits arrive. The system uses a scraping-first, evidence-driven workflow: ingest the official repo, top leaderboard submissions, and relevant papers, convert that into ranked hypotheses, and then run tightly scoped local experiments on an RTX 5070 to improve the submission candidate.

### Competition Constraints

- **Artifact cap:** 16,000,000 bytes (code + compressed model)
- **Training budget:** 10 minutes on 8x H100 SXM
- **Evaluation budget:** 10 minutes on 8x H100 SXM (separate)
- **Metric:** val_bpb (bits per byte on FineWeb validation set, tokenizer-agnostic)
- **Submission format:** PR adding folder to `/records/track_10min_16mb/` with README.md, submission.json, train log, train_gpt.py
- **Current SOTA:** 1.1428 (thwu1 — 10L Int5-MLP + BigramHash(10240) + SWA(0.4) + WD=0.04)
- **No network calls during evaluation**
- **Must beat SOTA by ≥0.005 nats at p < 0.01 for record submission**

### Local Hardware

- Lenovo Legion i7 Pro, RTX 5070 (~12GB VRAM), 64GB RAM, 1TB SSD, Windows 11

---

## Architecture: Approach B — Parallel Research + Sequential Execution

```
┌──────────────────────────────────────────────────────────────────┐
│                        ORCHESTRATOR                              │
│  Skills: ag-concise-planning, ag-executing-plans,                │
│          ag-dispatching-parallel-agents                           │
│  Plugin: superpowers (verification gates)                        │
│  Memory: auto-memory + research/ artifacts                       │
└──┬─────────┬───────────────┬──────────────────┬──────────────────┘
   │         │               │                  │
┌──▼───┐ ┌───▼────────┐ ┌───▼──────────┐ ┌─────▼─────────┐
│PHASE │ │ PHASE 1    │ │ PHASE 2      │ │ PHASES 3-4-5  │
│  0   │ │ Evidence   │ │ Hypothesis   │ │ Experiment    │
│Basel.│ │ Ingestion  │ │ Generation   │ │ Loop          │
│      │ │ (parallel) │ │ (serial)     │ │ (serial loop) │
└──────┘ └────────────┘ └──────────────┘ └───────────────┘
```

---

## Phase 0: Baseline Establishment

Before any research or experimentation, establish a local baseline to measure all future experiments against.

**Steps:**
1. Clone `openai/parameter-golf` into `parameter-golf/`
2. Download FineWeb data: `python3 data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`
3. Run baseline training with surrogate settings (reduced wall-clock, single GPU)
4. Record local `val_bpb`, compressed artifact size, and training throughput
5. Commit as `EXP-000_baseline` in the experiment registry

**Starting point:** The official `train_gpt.py` starter code (naive baseline, ~1.2244 val_bpb at full scale). This gives us a clean reference point. Top leaderboard code will be studied in Phase 1 but the baseline is the official starter — we build improvements on top of understood code, not by copying SOTA blindly.

**Output:**
```
runs/EXP-000_baseline/
├── train_log.txt
├── metrics.json          # { local_val_bpb, artifact_bytes, tokens_per_sec }
└── config.json           # surrogate settings used
research/experiment_registry.jsonl  # first entry: EXP-000
```

---

## Phase 1: Evidence Ingestion (Parallel)

Three parallel research agents run simultaneously, merging into a unified frontier map.

### Agent 1A: GitHub Repo Scout

**Purpose:** Scrape openai/parameter-golf repo + all leaderboard record folders

| Tool Type | Tool | Purpose |
|-----------|------|---------|
| MCP | `mcp__github__get_file_contents` | Fetch train_gpt.py, README.md, submission.json per record |
| MCP | `mcp__github__search_code` | Search all submissions for techniques (SmearGate, BigramHash, int5, SWA) |
| MCP | `mcp__github__list_pull_requests` | Enumerate all PR submissions chronologically |
| MCP | `mcp__github__get_pull_request_files` | See exact diffs per submission |
| Subagent | `technical-researcher` | Analyze implementation details |
| Skill | `ag-competitive-landscape` | Structure competitive analysis |
| MCP | Scrapling (`fetch`, `stealthy_fetch`, `get`) | Primary web scraper — rendered pages, challenge docs |
| Skill | `ag-github-ops` | Batch GitHub operations |

**Target repos/folders:**
- `openai/parameter-golf` (main repo, README, FAQ)
- `records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50/`
- `records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA/`
- `records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow/`
- All other entries in `records/track_10min_16mb/`

### Agent 1B: Paper/Academic Scout

**Purpose:** Fetch and analyze papers on quantization, QAT, SWA, scaling laws, compression

| Tool Type | Tool | Purpose |
|-----------|------|---------|
| Subagent | `academic-researcher` | Search arXiv, OpenReview |
| Subagent | `academic-research-synthesizer` | Cross-reference with citations |
| Skill | `ag-claude-scientific-skills` | Scientific methodology |
| Skill | `ag-citation-management` | Citation tracking |
| Skill | `ag-scientific-writing` | Implementation-ready notes |
| Skill | `ag-exa-search` | Semantic paper search |
| Skill | `ag-tavily-web` | Web search for discussions |
| Tool | `WebSearch` / `WebFetch` | Broad search + fetch |
| MCP | `mcp__context7__query-docs` | PyTorch, FlashAttention docs |

**Target topics:**
- Constrained scaling laws (L(N) optimization)
- Mixed-precision quantization (int5, int6, int8, FP16 selective)
- Quantization-aware training (STE, progressive QAT)
- Compression-aware training
- SWA / warmdown / cooldown
- Token-pair features (SmearGate, BigramHash, n-gram augmentation)
- Efficient transformer architectures under parameter budgets

### Agent 1C: Leaderboard Analyzer

**Purpose:** Build structured frontier map comparing all submissions

| Tool Type | Tool | Purpose |
|-----------|------|---------|
| Subagent | `comprehensive-researcher` | Deep dive per entry |
| Subagent | `data-analyst` | Quantitative metric comparison |
| Skill | `ag-competitor-alternatives` | Structured comparison |
| Skill | `ag-andrej-karpathy` | Deep learning expertise |
| Skill | `ag-reference-builder` | Structured notes |
| Skill | `deep-research` | Synthesis |

### Evidence Extraction Schema

For each leaderboard submission or paper:

```
title:
source_url:
date:
category: repo | paper | issue | PR | note
key_method:
components_changed:
reported_metric_improvement:
likely_mechanism:
interactions_with_other_ideas:
implementation_complexity: low | medium | high
confidence: 1-5
worth_testing_locally: yes | no
affects:
  throughput: yes | no
  compressed_artifact_size: yes | no
  post_quantization_loss: yes | no
  raw_train_loss: yes | no
  evaluation_only_score: yes | no
improvement_category: MODEL | QUANT | COMPRESS | EVAL | THROUGHPUT
```

### Phase 1 Merge Protocol (Agent 1D: Synthesizer)

After all three parallel agents complete, the **Orchestrator** runs a synthesis step:

1. **Completion signal:** Each agent writes a `_DONE.md` marker file in its output directory. The orchestrator polls for all three markers.
2. **Conflict resolution:** If agents 1A and 1C produce conflicting analysis of the same submission, the orchestrator prefers the agent with the more detailed code-level analysis (1A > 1C for code, 1C > 1A for metrics).
3. **Merge step:** The orchestrator reads all `evidence/` outputs and writes:
   - `frontier_map.md` — unified comparison table
   - `technique_taxonomy.md` — categorized by improvement type (MODEL/QUANT/COMPRESS/EVAL/THROUGHPUT)
4. **Phase 1 complete signal:** `frontier_map.md` exists and contains at least one entry per leaderboard submission.

| Tool Type | Tool | Purpose |
|-----------|------|---------|
| Skill | `ag-reference-builder` | Merge structured notes into frontier map |
| Subagent | `data-analyst` | Quantitative cross-comparison |

### Phase 1 Output Artifacts

```
research/
├── evidence/
│   ├── github_submissions/       # One note per leaderboard entry
│   │   └── _DONE.md             # Agent 1A completion marker
│   ├── papers/                   # Implementation-ready paper notes
│   │   └── _DONE.md             # Agent 1B completion marker
│   └── discussions/              # Blog posts, forum insights
│       └── _DONE.md             # Agent 1C completion marker
├── frontier_map.md               # Normalized comparison of all methods
└── technique_taxonomy.md         # Categorized by improvement type
```

---

## Phase 2: Hypothesis Generation

Runs after Phase 1 agents merge. Converts evidence into ranked experiment backlog.

### Agent 2: Hypothesis Designer

| Tool Type | Tool | Purpose |
|-----------|------|---------|
| Skill | `ag-ab-test-setup` | Experiment design with mandatory gates |
| Skill | `ag-ai-ml` | ML reasoning about what actually improves loss |
| Skill | `ag-andrej-karpathy` | Architecture tradeoff intuition |
| Skill | `ag-data-structure-protocol` | Experiment registry schema |
| Skill | `ag-goal-analyzer` | Expected payoff analysis |
| Skill | `ag-progressive-estimation` | Complexity estimation |
| Skill | `ag-concise-planning` | Scoped experiment plans |
| Subagent | `data-analyst` | Quantitative ranking |
| Subagent | `architect` | Architectural compatibility analysis |

### Hypothesis Ranking Criteria (5 dimensions, each 1-5)

| Dimension | Measures |
|-----------|----------|
| Expected BPB improvement | How much it likely improves score |
| Implementation complexity | Lines of code / risk of breakage |
| Evidence strength | How many top entries use this idea |
| Synergy potential | Combines with other winners? |
| Local testability | Validatable on RTX 5070? |

### Improvement Category Tags

| Tag | Meaning | Example |
|-----|---------|---------|
| `MODEL` | Real architecture improvement | 3x MLP, extra layers |
| `QUANT` | Quantization/export improvement | int5 MLP, int6 attn, FP16 embed |
| `COMPRESS` | Compression efficiency | zstd-22, byte reallocation |
| `EVAL` | Evaluation-only improvement | Sliding window, longer context |
| `THROUGHPUT` | Tokens processed per budget | Faster kernels, reduced overhead |

### Ranked Starting Directions

1. **Layerwise mixed precision** — test whether sensitive layers deserve higher precision
2. **Byte reallocation across subsystems** — optimize byte spend across MLP, attention, embeddings
3. **QAT schedule tuning** — sweep when QAT begins and how it interacts with SWA
4. **Token-pair feature path** — extend SmearGate/BigramHash ideas
5. **Throughput-aware scaling** — tokens processed as first-class variable
6. **SWA and stability tuning** — checkpoint averaging frequency and late-training fraction

### Phase 2 Output Artifacts

```
research/
├── hypothesis_backlog.md         # Ranked list with scores + tags
├── synergy_matrix.md             # Which ideas combine / conflict
└── problem_docs/
    ├── EXP-001_mixed_precision.md
    ├── EXP-002_mlp_3x_expansion.md
    ├── EXP-003_qat_schedule.md
    └── ... (one per experiment using problem doc template)
```

### Problem Document Template

```
Experiment ID:
Title:
Hypothesis:
Source evidence:
Expected mechanism:
Primary metric:
Secondary metrics:
Files allowed to change:
Files forbidden:
Implementation scope:
Run command:
Evaluation command:
Artifact size check:
Success criteria:
Rollback criteria:
Risks:
Notes:
```

---

## Phases 3-4-5: Experiment Loop

Tight autoresearch-style loop. One experiment at a time, sequential on the GPU.

```
Pick top hypothesis → Implement (patch) → Train + Evaluate → Log + Decide
       ▲                                                         │
       │              PROMOTE (merge) ◀── Beats baseline? ── YES─┘
       │                                        │
       └──────── ROLLBACK ◀──────────── NO ─────┘
```

### Phase 3: Implementer Agent

| Tool Type | Tool | Purpose |
|-----------|------|---------|
| Skill | `ag-python-pro` | Idiomatic Python, PyTorch |
| Skill | `ag-python-performance-optimization` | Throughput optimization |
| Skill | `ag-python-patterns` | Clean experiment code |
| Skill | `ag-ai-ml` | ML pipeline patterns |
| Skill | `ag-async-python-patterns` | Async data loading |
| Skill | `ag-closed-loop-delivery` | Acceptance criteria gates |
| Skill | `ag-differential-review` | Diff-only review |
| Skill | `ag-sharp-edges` | Catch numerical bugs |
| Subagent | `python-expert` | Complex PyTorch implementation |
| Subagent | `ml-engineer` | ML-specific patterns |
| Subagent | `performance-engineer` | Throughput optimization |
| Skill | `verification-before-completion` (superpowers) | No experiment ships without checks |
| CLI | `git` | Branch per experiment |

**Guardrails:**
- Only edits `train_gpt.py` (and new helper files if needed)
- Each experiment = one git branch off current best baseline
- Patch summary + rationale logged before training
- Max one idea family per experiment (unless tagged `SYNTHESIS`)

### Phase 4: Auditor/Evaluator Agent

| Tool Type | Tool | Purpose |
|-----------|------|---------|
| Skill | `ag-find-bugs` | Pre-training bug detection |
| Skill | `ag-systematic-debugging` | Debug failures methodically |
| Skill | `ag-production-code-audit` | Code quality verification |
| Skill | `ag-backtesting-frameworks` | Experiment comparison |
| Subagent | `debugger` | Fix runtime errors |
| Subagent | `code-reviewer` | Code quality review |
| CLI | `python` / `torchrun` | Training + evaluation |
| CLI | `git diff` | Verify clean, minimal diffs |

**Validation Checklist (every experiment):**

- [ ] Code compiles and runs without errors
- [ ] Export path produces valid model artifact
- [ ] Quantization/compression completes successfully
- [ ] Compressed artifact ≤ 16,000,000 bytes
- [ ] Local val_bpb measured and recorded
- [ ] Post-quantization val_bpb measured (if applicable)
- [ ] Training throughput (tokens/sec) recorded
- [ ] No regressions on secondary metrics

### Phase 5: Librarian Agent

| Tool Type | Tool | Purpose |
|-----------|------|---------|
| Skill | `ag-agent-memory-systems` | Persistent experiment memory |
| Skill | `ag-context-management-context-save` | Save context for future sessions |
| Skill | `ag-evolution` | Track improvement trajectory |
| Skill | `ag-kaizen` | Continuous improvement |
| Skill | `ag-commit` | Git commits per result |
| Skill | `ag-create-pr` | Final PR creation |
| Skill | `ag-pr-writer` | PR description |
| Skill | `ag-scientific-writing` | Submission README |
| Plugin | `commit-commands` | Streamlined commit workflow |
| MCP | `mcp__github__create_pull_request` | Submit final PR |

### Phase 5 Output Artifacts

```
research/
├── experiment_registry.jsonl     # One line per experiment
├── frontier_map.md               # Updated after each promotion
└── hypothesis_backlog.md         # Re-ranked after each result

runs/
├── EXP-001_mixed_precision/
│   ├── train_log.txt
│   ├── metrics.json
│   └── patch.diff
└── ...

submission_candidate/
├── README.md
├── submission.json
├── train_gpt.py
├── train_log.txt
└── requirements.txt       # Per competition FAQ: allowed if needed for setup
```

> **Note:** `requirements.txt` is permitted per competition FAQ ("include a requirements.txt in your records folder and mention setup instructions in your README.md"). It is for dependency declaration, not for sneaking in code.

---

## Local Surrogate Strategy

| Parameter | Full Competition | Local Surrogate |
|-----------|-----------------|-----------------|
| GPUs | 8x H100 (640GB) | 1x RTX 5070 (~12GB) |
| Wall clock | 10 minutes | 2-3 minutes |
| Training data | 80 shards (8B tokens) | 1-2 shards (~200M tokens) |
| Batch size | Full | Reduced to fit VRAM |
| Validation | Full FineWeb val set | Same (fixed 50k docs) |
| Export/Quant | Same path | Same path |
| Compression | Same (zstd/zlib) | Same |
| Artifact check | ≤16MB | ≤16MB |

**Key principle:** Surrogate preserves decision logic — same tokenizer, same export, same artifact measurement. Only data volume and wall-clock are scaled.

**Throughput warning:** Experiments tagged `THROUGHPUT` are **low-confidence** when run locally. The RTX 5070 has different memory bandwidth, no NVLink, and different kernel dispatch characteristics than H100 SXM. Local throughput gains may not transfer. `THROUGHPUT`-tagged experiments should be deprioritized in the local queue and flagged for priority validation on H100s before submission.

---

## Complete Skill/Tool/MCP/Plugin Index

### Skills (28)

> **Note:** All skills below are confirmed installed via `ccpm list`. Some are from community catalogs (antigravity, buildwithclaude) rather than the BACKBONE master registry.

| Skill | Phase | Role |
|-------|-------|------|
| `ag-dispatching-parallel-agents` | 1 | Parallel research dispatch |
| `ag-competitive-landscape` | 1 | Competitive analysis |
| Scrapling MCP | 1 | Primary web scraping (`fetch`, `stealthy_fetch`, `get`) |
| `ag-exa-search` | 1 | Semantic search |
| `ag-tavily-web` | 1 | Web search |
| `ag-claude-scientific-skills` | 1 | Scientific methodology |
| `ag-citation-management` | 1 | Citation tracking |
| `ag-scientific-writing` | 1, 5 | Research notes + README |
| `ag-andrej-karpathy` | 1, 2 | Deep learning expertise |
| `ag-reference-builder` | 1 | Structured notes |
| `deep-research` | 1 | Synthesis |
| `ag-ab-test-setup` | 2 | Experiment design |
| `ag-ai-ml` | 2, 3 | ML workflow |
| `ag-goal-analyzer` | 2 | Payoff analysis |
| `ag-progressive-estimation` | 2 | Complexity estimation |
| `ag-concise-planning` | 2 | Scoped plans |
| `ag-python-pro` | 3 | Python/PyTorch |
| `ag-python-performance-optimization` | 3 | Throughput |
| `ag-closed-loop-delivery` | 3 | Acceptance gates |
| `ag-differential-review` | 3 | Diff review |
| `ag-find-bugs` | 4 | Bug detection |
| `ag-systematic-debugging` | 4 | Debug methodology |
| `ag-backtesting-frameworks` | 4 | Experiment comparison |
| `ag-agent-memory-systems` | 5 | Experiment memory |
| `ag-commit` | 5 | Git commits |
| `ag-create-pr` | 5 | PR creation |
| `ag-evolution` | 5 | Improvement tracking |
| `ag-pr-writer` | 5 | PR description writing |

### Subagents (8)

| Subagent | Phase | Role |
|----------|-------|------|
| `academic-researcher` | 1B | Paper discovery |
| `academic-research-synthesizer` | 1B | Paper synthesis |
| `comprehensive-researcher` | 1C | Deep analysis |
| `technical-researcher` | 1A | Code analysis |
| `data-analyst` | 1C, 2 | Quantitative comparison |
| `python-expert` | 3 | PyTorch implementation |
| `ml-engineer` | 3 | ML code patterns |
| `debugger` | 4 | Runtime error fixing |

### MCP Servers (4 primary, 2 optional)

| MCP | Phase | Key Tools | Role |
|-----|-------|-----------|------|
| **Scrapling** | 1A, 1B, 1C | `fetch`, `stealthy_fetch`, `get` | **Primary web scraper** — rendered pages, papers, challenge docs, leaderboard |
| **GitHub** | 1A, 5 | `get_file_contents`, `search_code`, `list_pull_requests`, `create_pull_request` | Structured repo access + PR submission |
| **Context7** | 1B, 3 | `resolve-library-id`, `query-docs` | PyTorch, FlashAttention, sentencepiece docs |
| **Playwright** | 1 (fallback) | `browser_navigate`, `browser_snapshot` | Fallback for dynamic/JS-heavy pages |
| GitNexus | 3, 4 (optional) | `context`, `impact`, `detect_changes` | Track code change impact across experiments |
| Notion | 5 (optional) | `notion-create-pages` | Publish research notes externally if desired |

### Plugins (3)

| Plugin | Phase | Purpose |
|--------|-------|---------|
| `superpowers` | All | Verification gates, brainstorming, plans, parallel dispatch |
| `commit-commands` | 5 | Git workflow |
| `pr-review-toolkit` | 5 | Final submission review |

### CLI Tools (4)

| CLI | Phase | Commands |
|-----|-------|----------|
| `git` | 3, 4, 5 | Branch, diff, commit |
| `python` / `torchrun` | 4 | Training + eval |
| `gh` | 5 | PR operations |
| `pip` | Setup | Install dependencies |

---

## Mandatory Workflow Rules

1. No agent may edit model code before Phase 1 evidence ingestion is complete
2. No experiment may change more than one idea family unless tagged `SYNTHESIS`
3. No patch promoted without: code diff summary, train/eval command, metrics report, artifact size check, rollback path
4. No paper/repo treated as truth without extracting actual claimed mechanism and expected downside
5. System must track whether improvement is: universally useful, hardware-contingent, evaluation-specific, compression-specific, or likely to break submission simplicity

---

## Directory Structure

```
ParameterGolf/
├── BACKBONE_POINTER.md
├── docs/
│   └── superpowers/
│       └── specs/
│           └── 2026-03-23-parameter-golf-orchestrator-design.md  (this file)
├── research/
│   ├── evidence/
│   │   ├── github_submissions/
│   │   ├── papers/
│   │   └── discussions/
│   ├── frontier_map.md
│   ├── technique_taxonomy.md
│   ├── hypothesis_backlog.md
│   ├── synergy_matrix.md
│   ├── experiment_registry.jsonl
│   └── problem_docs/
├── runs/
│   ├── EXP-000_baseline/         # Phase 0 baseline measurements
│   └── EXP-NNN_<name>/
├── submission_candidate/
│   ├── README.md
│   ├── submission.json
│   ├── train_gpt.py
│   ├── train_log.txt
│   └── requirements.txt
└── parameter-golf/                # Cloned official repo
```
