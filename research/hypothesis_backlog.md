# Hypothesis Backlog

Generated: 2026-03-23
Source: frontier_map.md, technique_taxonomy.md, all evidence

---

## Ranking Methodology

Each hypothesis scored on:
- **Expected BPB Impact** (0.001 = tiny, 0.050 = massive)
- **Confidence** (LOW/MED/HIGH based on evidence: paper-only vs. competition-proven)
- **Implementation Effort** (LOW/MED/HIGH based on code changes needed)
- **Risk** (LOW/MED/HIGH — chance of regression or wasted time)

Priority = Impact * Confidence / Effort

---

## EXP-001: Sliding Window Eval (stride=64)

**Hypothesis:** Evaluating with overlapping sliding windows instead of non-overlapping chunks gives each token more context, reducing val_bpb by ~0.013-0.032 BPB.

**Evidence:** Present in 9/10 top merged entries. Matthew Li's pure sliding window entry improved from baseline by 0.032 BPB with NO model changes.

**Implementation:**
- Modify `eval_val()` to use stride=64 instead of full seq_len chunking
- Each window is seq_len=1024 tokens, but only the last 64 tokens contribute to the loss
- More eval compute but well within 10-minute budget

**Expected Impact:** -0.013 to -0.032 BPB
**Confidence:** HIGH (proven in 9/10 entries)
**Effort:** LOW (eval-only change, ~30 lines)
**Risk:** LOW (cannot hurt training quality)
**Priority:** 1 (CRITICAL — implement first)

---

## EXP-002: Int6 Quantization + MLP 3x Expansion

**Hypothesis:** Switching from int8 to int6 quantization saves ~25% weight bytes, which funds expanding MLP from 2x to 3x (1024→1536). The net effect is a significantly more capable model within the same 16MB budget.

**Evidence:** Present in 8/10 top entries. The int6+MLP3x combo is the single largest contributor (-0.034 BPB combined) to the gap between baseline and merged SOTA.

**Implementation:**
- Replace int8 quantization with int6 per-row quantization (63 levels instead of 127)
- Change MLP_MULT from 2 to 3
- May need to adjust FP16 embedding handling

**Expected Impact:** -0.030 to -0.035 BPB
**Confidence:** HIGH (proven in 8/10 entries)
**Effort:** MED (quantization code rewrite + model size adjustment)
**Risk:** LOW (well-proven technique)
**Priority:** 2 (CRITICAL)

---

## EXP-003: STE QAT (Quantization-Aware Training)

**Hypothesis:** Training with Straight-Through Estimator simulates quantization during training, eliminating the post-training quantization gap (0.007-0.014 BPB with int8, potentially larger with int6).

**Evidence:** Present in 4/10 top entries. yahya010's entry achieved "zero quant gap" with STE QAT. Papers confirm plain STE is theoretically optimal (ste_improvements.md).

**Implementation:**
- During forward pass: quantize weights → compute → straight-through gradient (bypass quant in backward)
- Apply during last N% of training (after model has converged somewhat)
- Match quantization scheme (int6 per-row) to what will be used at export

**Expected Impact:** -0.007 to -0.014 BPB
**Confidence:** HIGH (proven + paper-backed)
**Effort:** MED (forward pass modification, careful gradient handling)
**Risk:** LOW (well-understood technique)
**Priority:** 3

---

## EXP-004: SmearGate + BigramHash(10240) + OrthoInit

**Hypothesis:** This cluster of lightweight token-pair features provides bigram information cheaply. SmearGate blends previous token embedding, BigramHash provides hash-based bigram features, OrthoInit improves training dynamics with Muon.

**Evidence:** Present in all top-4 merged entries. SmearGate costs ~512 params. BigramHash(10240) costs ~1.3MB but provides -0.008 BPB. OrthoInit is free.

**Implementation:**
- SmearGate: learned per-dimension gate blending token[i] with token[i-1]
- BigramHash: hash(prev_token, cur_token) → lookup in embedding table → add to residual
- OrthoInit: initialize all weight matrices with orthogonal matrices

**Expected Impact:** -0.010 to -0.015 BPB (combined)
**Confidence:** HIGH (in top-4 entries)
**Effort:** MED (new model components + init changes)
**Risk:** LOW (proven cluster)
**Priority:** 4

---

## EXP-005: 10 Layers + Muon WD=0.04 + SWA

**Hypothesis:** Adding a 10th layer funded by int5 MLP quantization, plus weight decay (tighter distributions for quant) and SWA (flat minima for quant robustness) together improve model quality significantly.

**Evidence:** 6/10 top entries use 10+ layers. Muon WD=0.04 is in 6/10. SWA in 3/10 but all top-3.

**Implementation:**
- NUM_LAYERS=10 (funded by int5/int6 byte savings)
- Add weight decay to Muon optimizer (WD=0.04)
- Implement SWA: average model weights every 50 steps, starting at 40% through training

**Expected Impact:** -0.010 to -0.015 BPB (combined)
**Confidence:** HIGH
**Effort:** MED (layer count + optimizer modification + SWA loop)
**Risk:** LOW-MED (must balance byte budget carefully)
**Priority:** 5

---

## EXP-006: zstd-22 Compression

**Hypothesis:** Replacing zlib-9 with zstd at level 22 saves ~5% (~0.8MB) on the compressed artifact, which can fund more model capacity.

**Evidence:** Present in 6/10 top entries. zstd-22 is strictly better than zlib-9 for neural weight compression.

**Implementation:**
- Replace `zlib.compress(data, level=9)` with `zstandard.compress(data, level=22)`
- Update decompression in eval to match

**Expected Impact:** ~0.8MB freed (indirect BPB impact through more capacity)
**Confidence:** HIGH
**Effort:** LOW (compression library swap)
**Risk:** LOW
**Priority:** 6

---

## EXP-007: Test-Time Training (TTT) — Cosine Variant

**Hypothesis:** Adapting model parameters on already-evaluated tokens during eval significantly reduces val_bpb. The model learns per-document patterns that generalize to subsequent tokens.

**Evidence:** THE dominant technique in unmerged frontier. Cosine TTT in PR #486 achieves 1.0887 (-0.054 from merged SOTA). Multiple independent implementations converge on 0.03-0.05 BPB improvement.

**Implementation:**
- During eval, after scoring each chunk of tokens:
  - Compute loss on those tokens
  - Backprop and update a subset of model parameters (LoRA adapters or full params)
  - Use cosine-annealed LR schedule across the eval sequence
  - Reset parameters between documents (or don't — need to test)
- Must only use already-evaluated tokens (competition rule)
- AdamW or SGD with momentum as TTT optimizer

**Expected Impact:** -0.030 to -0.054 BPB
**Confidence:** HIGH (multiple independent implementations, best unmerged PR)
**Effort:** HIGH (complex eval-time optimization loop, careful about rules)
**Risk:** MED (must verify rule compliance, eval time budget)
**Priority:** 7 (HIGH impact but needs solid base model first)

---

## EXP-008: WSD Learning Rate Schedule

**Hypothesis:** Warmup-Stable-Decay (WSD) schedule — high constant LR for ~80%, then decay for ~20% — outperforms the current linear warmdown. MiniCPM paper shows "dramatic loss drop during decay."

**Evidence:** Paper-backed (minicpm_wsd.md). Not yet tried in competition. Free to implement.

**Implementation:**
- Replace current `lr_mul()` function with WSD schedule:
  - Warmup: 0→max LR over first 500 steps
  - Stable: constant max LR from step 500 to step ~16000
  - Decay: cosine decay from max LR to 0 over last ~4000 steps

**Expected Impact:** -0.003 to -0.008 BPB
**Confidence:** MED (paper-backed, not competition-tested)
**Effort:** LOW (LR schedule change only)
**Risk:** LOW
**Priority:** 8

---

## EXP-009: Value Residual Connections

**Hypothesis:** Adding residual connections from the value projections of earlier layers to later layers improves gradient flow and representation quality.

**Evidence:** Present in top 2 unmerged PRs (1.0887 and 1.0891). Novel technique not in any merged entry.

**Implementation:**
- Store V projections from each layer
- In later layers, add a learned weighted sum of earlier V projections to the current V

**Expected Impact:** -0.005 to -0.015 BPB
**Confidence:** MED (unmerged PRs only, but in the 2 best)
**Effort:** MED (architecture change)
**Risk:** MED (unproven in isolation, may interact with other techniques)
**Priority:** 9

---

## EXP-010: TrigramHash

**Hypothesis:** Extending BigramHash to 3-token context captures richer local patterns. Hash(tok[i-2], tok[i-1], tok[i]) → embedding lookup.

**Evidence:** In best unmerged PR (#486, 1.0887). Extends the proven BigramHash approach.

**Implementation:**
- Similar to BigramHash but with 3-token window
- May need more buckets or separate embedding table
- Adds ~1.5MB to model size

**Expected Impact:** -0.002 to -0.005 BPB (incremental over BigramHash)
**Confidence:** MED (1 unmerged PR)
**Effort:** LOW (extension of BigramHash)
**Risk:** LOW
**Priority:** 10

---

## EXP-011: FP16 Tied Embedding

**Hypothesis:** Keeping the tied embedding/lm_head in FP16 instead of quantizing it prevents output head degradation.

**Evidence:** Present in 7/10 top entries. The baseline already stores small tensors in FP16, but the embedding may be large enough to get quantized.

**Implementation:**
- Ensure tok_emb.weight (which IS lm_head due to tying) is stored as FP16 in the quantized artifact
- Skip quantization for this tensor specifically

**Expected Impact:** -0.001 to -0.003 BPB
**Confidence:** HIGH
**Effort:** LOW (quantization config change)
**Risk:** LOW (but costs bytes)
**Priority:** 11

---

## EXP-012: Per-Group Quantization (group_size=64)

**Hypothesis:** Using per-group scales instead of per-row scales reduces quantization error, especially for int5/int6 with few levels.

**Evidence:** Paper-backed (qat_scaling_law.md, mixed_precision_survey.md). Recommended group size 32-128.

**Implementation:**
- Reshape weight rows into groups of 64
- Compute scale per group instead of per row
- More scale parameters but finer quantization

**Expected Impact:** -0.002 to -0.005 BPB
**Confidence:** MED (paper-backed, not competition-tested)
**Effort:** MED (quantization code change)
**Risk:** LOW (more bytes for scales, but finer quant)
**Priority:** 12

---

## Experiment Execution Order

### Phase 1: Quick Wins (EXP-001, EXP-006, EXP-011)
Low-effort, proven techniques. Establish improved baseline.
Expected combined impact: -0.015 to -0.037 BPB

### Phase 2: Architecture + Quantization (EXP-002, EXP-003, EXP-004, EXP-005)
Core model improvements that form the foundation for competitive entry.
Expected combined impact: -0.050 to -0.079 BPB

### Phase 3: Training Strategy (EXP-008)
LR schedule optimization. Quick to test.
Expected impact: -0.003 to -0.008 BPB

### Phase 4: TTT + Novel Techniques (EXP-007, EXP-009, EXP-010)
Highest-impact but highest-effort changes. Build on Phase 2 base.
Expected combined impact: -0.037 to -0.074 BPB

### Phase 5: Fine-tuning (EXP-012)
Squeeze remaining gains.
Expected impact: -0.002 to -0.005 BPB

---

## Target Trajectory

| After Phase | Expected val_bpb | vs Merged SOTA | vs Unmerged Frontier |
|-------------|-------------------|----------------|---------------------|
| Baseline | 1.2244 | +0.0816 | +0.1357 |
| Phase 1 | ~1.195 | +0.052 | +0.106 |
| Phase 2 | ~1.145 | +0.002 | +0.056 |
| Phase 3 | ~1.140 | -0.003 | +0.051 |
| Phase 4 (TTT) | ~1.095 | -0.048 | +0.006 |
| Phase 5 | ~1.090 | -0.053 | +0.001 |

**Target: val_bpb < 1.095** (competitive with unmerged frontier)
