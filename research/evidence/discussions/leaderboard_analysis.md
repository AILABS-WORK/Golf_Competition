# Parameter Golf Leaderboard Analysis
**Generated: 2026-03-23**
**Track: 10min_16mb (8xH100, 16MB artifact, FineWeb val_bpb)**

---

## 1. Complete Leaderboard Table (Merged Records)

Sorted by val_bpb (lower is better). All entries from `records/track_10min_16mb/` on the main branch.

| Rank | Run Name | val_bpb | Author | Artifact Size | Date | Key Techniques |
|------|----------|---------|--------|---------------|------|----------------|
| 1 | 10L Int5-MLP + BigramHash(10240) + SWA(0.4) + WD=0.04 | **1.14276** | thwu1 | ~15.90 MB | 2026-03-20 | Int5/Int6 mixed quant, BigramHash(10240), SWA(frac=0.4), SmearGate, OrthoInit, U-Net, 10 layers, zstd-22 |
| 2 | Int6 MLP3x + SmearGate + BigramHash + OrthoInit + Muon WD + SWA | **1.14582** | raahilshah | 15.86 MB | 2026-03-20 | Int6 quant, MLP 3x, SmearGate, BigramHash(4096), OrthoInit, Muon WD=0.04, SWA(every=50), zstd-22, Sliding Window |
| 3 | 11L MLP3x + WD=0.04 + Int6 QAT + zstd-22 + Sliding Window | **1.15015** | aruniyer | 15.43 MB | 2026-03-20 | 11 layers, MLP 3x, QAT Int6, Muon WD=0.04, zstd-22, Sliding Window, FP16 embed, Muon momentum 0.99 |
| 4 | SmearGate + OrthoInit + Muon WD + Int6 STE QAT + MLP 3x + Sliding Window | **1.15560** | notapplica (unnir) | 15.88 MB | 2026-03-19 | SmearGate, OrthoInit, Muon WD, Int6 STE QAT, MLP 3x, Sliding Window, BigramHash(4096), U-Net skip, zstd-22 |
| 5 | Int6 MLP3x Sliding Window (WarmdownQuantization) | **1.15744** | samuellarson | 15.98 MB | 2026-03-20 | Int6 PTQ, MLP 3x, Sliding Window, FP16 embed, Late-K passthrough, Train@2048 |
| 6 | 10L Int6 QAT + Zstd MLP2.6x Muon0.99 Sliding Window | **1.15862** | yahya010 | 15.56 MB | 2026-03-19 | 10 layers, Int6 STE QAT, zstd-22, MLP 2.6x (hidden=1344), FP16 embed, Muon 0.99, Sliding Window |
| 7 | Mixed Quant (int6 blocks + int8 embeddings) + Sliding Window | **1.16301** | aquariouseworkman | 15.35 MB | 2026-03-19 | MLP 3x, Int6/Int8 mixed quant, STE, zlib-9, Sliding Window |
| 8 | Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init | **1.17475** | notapplica | 15.37 MB | 2026-03-19 | 10 layers, FP16 embed, Muon WD, Overtone Init, Sliding Window |
| 9 | LoRA TTT | **1.19290** | sam (samacqua) | 15.88 MB | 2026-03-19 | LoRA test-time training (rank-8 on lm_head/Q/V), Adam lr=0.01, overlapping 256-token chunks |
| 10 | Sliding Window Eval (stride=64) | **1.19250** | mattqlf | 15.87 MB | 2026-03-19 | Sliding Window (stride=64), baseline 9x512, int8+zlib |
| 11 | Long Context Seq2048 v2 | **1.20576** | spokane-way | 15.87 MB | 2026-03-19 | Seq2048 training, tuned LR |
| 12 | Training Opt Seq4096 v1 | **1.20143** | spokane-way | 15.87 MB | 2026-03-19 | Seq4096, Muon momentum 0.99, lower LR, warmdown 3000 |
| 13 | 10L Mixed Precision | **1.21475** | nanlliu | 15.93 MB | 2026-03-19 | 10 layers, mixed int8/int6, lower LR=0.02 |
| 14 | Warmdown-Quantization | **1.21540** | samuellarson | -- | 2026-03-19 | WD=20000, FP16 embed, NTK-RoPE eval@1408, Muon backend=5 |
| 15 | FP16 Tied Embedding + LR/Warmdown Tuning | **1.21973** | chonchiog | 15.90 MB | 2026-03-18 | FP16 embed, reduced MLP(992), warmdown 3600, matrix_lr=0.06 |
| 16 | Lower LR | **1.22297** | nanlliu | 15.85 MB | 2026-03-18 | LR sweep: matrix/scalar=0.02, tied_embed=0.03 |
| 17 | Naive Baseline (OpenAI) | **1.22437** | OpenAI | 15.86 MB | 2026-03-18 | 9x512, SP-1024, KV4, int8+zlib |

**Note on Rank 9/10:** LoRA TTT (1.1929) is slightly worse than Sliding Window Eval (1.1925) despite being listed first chronologically.

---

## 2. Unmerged Frontier (Open PRs, as of 2026-03-23)

These PRs have been submitted but not yet merged. They represent the true cutting edge:

| PR# | Run Name | val_bpb | Author | Key New Techniques |
|-----|----------|---------|--------|--------------------|
| ~486 | 11L TrigramHash + ValueResidual + GradQuant + Cosine TTT | **1.0887** (best: 1.0879) | ndokutovich | TrigramHash, Value Residual, Gradient Quantization, Cosine TTT |
| ~490 | 11L Value Residual + Gated Attention + AdamW TTT | **1.0891** | ahmettrkck | Value Residual, Gated Attention, AdamW TTT |
| ~503 | 11L XSA11 + Legal Score-First AdamW TTT | **1.1218** | EthanYangTW | XSA (Cross-Selective Attention), Score-First AdamW TTT |
| ~493 | 11L EMA + Int6 + XSA + LeakyReLU^2 + Partial RoPE | **1.1309** | -- | EMA, XSA, LeakyReLU^2, Partial RoPE |
| ~492 | 11L XSA4 + EMA + Partial RoPE + Rank-8 TTT Hooks | **1.1591** | -- | XSA, EMA, Partial RoPE, TTT Hooks |

**The true frontier is now at ~1.089 val_bpb**, far below the merged leader of 1.143.

---

## 3. Technique Frequency Analysis (Top 10 Merged Entries)

### Techniques appearing in 3+ of top 10 entries:

| Technique | Count in Top 10 | Avg Rank (lower=better) | Category |
|-----------|-----------------|------------------------|----------|
| Sliding Window Eval (stride=64) | **9/10** | 4.9 | EVAL |
| Int6 Quantization (PTQ or QAT) | **8/10** | 4.1 | QUANT |
| FP16 Tied Embedding | **7/10** | 4.0 | QUANT |
| MLP 3x Expansion (hidden=1536) | **7/10** | 3.6 | MODEL |
| zstd-22 Compression | **6/10** | 3.5 | COMPRESS |
| Muon Weight Decay (0.01-0.04) | **6/10** | 3.3 | TRAINING |
| Muon Momentum 0.99 + Warmup | **6/10** | 3.8 | TRAINING |
| 10+ Layers (10 or 11) | **6/10** | 3.8 | MODEL |
| Lower Learning Rates (0.02-0.025) | **5/10** | 4.2 | TRAINING |
| BigramHash Embedding | **4/10** | 2.0 | MODEL |
| SmearGate | **4/10** | 2.0 | MODEL |
| Orthogonal Init | **4/10** | 2.0 | MODEL |
| STE QAT (train-time fake quantize) | **4/10** | 3.8 | QUANT |
| U-Net Skip Connections | **3/10** | 2.7 | MODEL |
| SWA (Stochastic Weight Averaging) | **3/10** | 1.7 | TRAINING |

### Techniques appearing only once (unique/niche):

| Technique | Entry | val_bpb | Category |
|-----------|-------|---------|----------|
| LoRA Test-Time Training | LoRA TTT | 1.1929 | EVAL |
| Int5 MLP Quantization | 10L Int5-MLP (Rank 1) | 1.1428 | QUANT |
| BigramHash 10240 buckets (vs 4096) | 10L Int5-MLP (Rank 1) | 1.1428 | MODEL |
| SWA start_frac=0.4 (vs 0.5) | 10L Int5-MLP (Rank 1) | 1.1428 | TRAINING |
| Overtone Init | SW+FP16+10L+MuonWD | 1.1748 | MODEL |
| Late-K Passthrough | Int6 MLP3x SW | 1.1574 | MODEL |
| NTK-RoPE eval@1408 | Warmdown-Quant | 1.2154 | EVAL |
| 3% Magnitude Pruning | 10L Int5-MLP (Rank 1) | 1.1428 | COMPRESS |
| Warmdown=20000 (always-decaying LR) | Warmdown-Quant | 1.2154 | TRAINING |
| MLP 2.6x (hidden=1344) | yahya010 entry | 1.1586 | MODEL |

### Co-occurring technique clusters:

**Cluster A (Top 4 entries):** SmearGate + BigramHash + OrthoInit + U-Net + Int6 + MLP3x + Muon WD + Sliding Window
- These 8 techniques ALWAYS appear together in the top 4 merged entries.

**Cluster B (Mid-tier):** Sliding Window + FP16 embed + Lower LR + Muon momentum 0.99
- Common in entries ranked 5-8.

**Cluster C (Base improvements):** Seq2048 training + LR tuning + warmdown tuning
- Appear alone or in simple combinations in entries ranked 10-15.

### Technique Family Rankings:

| Family | Avg Rank When Present | Best Entry Rank | Impact Assessment |
|--------|----------------------|-----------------|-------------------|
| MODEL (arch changes) | **2.8** | 1 | Highest impact -- MLP3x, SmearGate, BigramHash |
| QUANT (quantization) | **3.2** | 1 | Essential enabler -- Int6/Int5 funds extra capacity |
| TRAINING (optimizer) | **3.5** | 1 | Strong -- Muon WD + SWA provide measurable gains |
| COMPRESS (storage) | **3.0** | 1 | Critical -- zstd-22 saves ~1.5MB over zlib |
| EVAL (eval strategy) | **4.5** | 1 | Nearly universal -- Sliding Window is table-stakes |

---

## 4. Improvement Attribution (Marginal BPB Drops)

### Baseline to Frontier -- Cumulative Technique Contributions

Starting from the Naive Baseline (1.2244 val_bpb):

| Step | Technique Added | New val_bpb | Delta BPB | Cumulative Improvement |
|------|----------------|-------------|-----------|----------------------|
| 0 | Naive Baseline (9L, 512d, int8+zlib) | 1.2244 | -- | -- |
| 1 | Lower LR (0.02 vs 0.04) | 1.2230 | -0.0014 | -0.0014 |
| 2 | FP16 Tied Embedding | 1.2197 | -0.0033 | -0.0047 |
| 3 | Seq2048 Training | 1.2058 | -0.0139 | -0.0186 |
| 4 | Sliding Window Eval (stride=64) | 1.1925 | -0.0133 | -0.0319 |
| 5 | Int6 Quantization (enables MLP expansion) | 1.1630 | -0.0295 | -0.0614 |
| 6 | MLP 3x Expansion (hidden=1536) | 1.1586 | -0.0044 | -0.0658 |
| 7 | STE QAT (zero quant gap) | 1.1556 | -0.0030 | -0.0688 |
| 8 | SmearGate + BigramHash + OrthoInit | 1.1556 | ~0.0000 | -0.0688 |
| 9 | Muon WD=0.04 + SWA | 1.1458 | -0.0098 | -0.0786 |
| 10 | Int5 MLP + 10th layer + BigramHash(10240) | 1.1428 | -0.0030 | -0.0816 |

### Single Biggest BPB Drops (Marginal Impact):

1. **Int6 Quantization + MLP expansion** (combined): -0.0339 BPB -- the single most impactful change
2. **Sliding Window Eval**: -0.0133 BPB -- free improvement at eval time
3. **Seq2048 Training**: -0.0139 BPB -- longer context during training
4. **Muon WD + SWA**: -0.0098 BPB -- optimizer regularization
5. **FP16 Tied Embedding**: -0.0033 BPB -- prevents embedding quantization damage

### Ablation Data from #1 Entry (thwu1):

From the README ablation table:
| Change | val_bpb | Delta |
|--------|---------|-------|
| 9L int6 (PR162 base) | 1.1485 | baseline |
| + int5 MLP + 10th layer | 1.1453 | -0.0032 |
| + WD=0.04 + warmdown=3000 | 1.1452 | -0.0001 |
| + SWA_start_frac=0.4 | 1.1446 | -0.0006 |
| + bigram=8192 | 1.1434 | -0.0012 |
| + bigram=10240 | **1.1426** | **-0.0008** |

Total improvement from PR162 base: **-0.0059 BPB** through 5 incremental changes.

---

## 5. Frontier Analysis

### 5A. Gap Between #1 and #2 (Merged)

| Metric | #1 (thwu1) | #2 (raahilshah) | Delta |
|--------|-----------|-----------------|-------|
| val_bpb | 1.14276 | 1.14582 | **-0.00306** |
| Layers | 10 | 9 | +1 layer |
| MLP Quant | Int5 ([-16,15]) | Int6 ([-32,31]) | More aggressive |
| BigramHash buckets | 10240 | 4096 | 2.5x more |
| SWA start_frac | 0.4 (last 40%) | 0.5 (last 50%) | Tighter window |
| Architecture | SmearGate + OrthoInit + U-Net | SmearGate + OrthoInit + (no explicit U-Net mention) | Similar |

### 5B. What #1 Has That #2 Doesn't:
- **Int5 quantization for MLP weights** -- saves ~1.86MB vs uniform int6, funding the 10th layer
- **10 layers** (vs 9) -- extra transformer depth
- **BigramHash with 10240 buckets** -- reduced hash collisions, +0.001 BPB
- **SWA start_frac=0.4** -- fewer but more-converged checkpoints
- **3% magnitude pruning** -- additional compression

### 5C. What #2 Has That #1 Doesn't:
- **Explicit quantization penalty measurement** (0.016 BPB int6 vs fp16) -- transparency
- **fp16 last-layer key projection** -- preserves late-layer attention quality
- No int5 risk (int5 has only 31 levels vs 63 for int6)

### 5D. Theoretical Optimal Combination (from merged entries):
Combining the best of all top entries:
1. **10 layers** (from #1)
2. **Int5 MLP + Int6 attention** (from #1)
3. **MLP 3x expansion** (universal in top entries)
4. **SmearGate + BigramHash(10240)** (from #1)
5. **Orthogonal Init** (from top 4)
6. **U-Net skip connections** (from top 4)
7. **Muon WD=0.04 + momentum 0.99** (universal)
8. **SWA start_frac=0.4, every=50** (from #1)
9. **STE QAT during training** (from #3, #4, #6)
10. **zstd-22 compression** (universal)
11. **FP16 tied embedding + last-layer key** (from #2)
12. **Sliding Window eval stride=64** (universal)
13. **Seq2048 training** (from most top entries)
14. **warmdown=3000** (from #1)

**Estimated achievable BPB: ~1.138-1.140** (incremental gains from combining)

### 5E. Techniques Common in Top Entries But ABSENT from #1:
- **STE QAT (quantization-aware training)** -- #1 uses post-training quantization only. Adding QAT could reduce the int5 quantization gap further.
- **11 layers** -- #3 uses 11L and achieves 1.1502; #1 uses 10L. Could an 11L int5 model fit?
- **Higher MLP expansion** -- if int5 savings are large enough, could go to MLP 3.5x or 4x

---

## 6. Unmerged Frontier Analysis (PRs from 2026-03-23)

The unmerged PRs show a dramatic leap. The gap between merged #1 (1.1428) and the best PR (1.0887) is **0.054 BPB** -- larger than the entire merged improvement range.

### New Technique Families in Unmerged PRs:

| Technique | Description | Appears In |
|-----------|-------------|------------|
| **Test-Time Training (TTT)** | Fine-tune model parameters during evaluation on each document | PR 486, 490, 492, 503 |
| **Value Residual** | Residual connections in the value stream of attention | PR 486, 490, 493 |
| **Gated Attention** | Learnable gating on attention outputs | PR 490 |
| **TrigramHash** | Extension of BigramHash to 3-token context | PR 486 |
| **GradQuant** | Gradient quantization during training for memory savings | PR 486 |
| **Cosine TTT** | Cosine-annealed learning rate for test-time training | PR 486 |
| **XSA (Cross-Selective Attention)** | Novel attention variant | PR 492, 493, 503 |
| **EMA (Exponential Moving Average)** | EMA of model weights | PR 492, 493 |
| **Partial RoPE** | Apply RoPE to only some heads or dimensions | PR 492, 493 |
| **LeakyReLU^2** | Leaky ReLU squared activation (vs ReLU^2) | PR 493 |
| **Score-First AdamW TTT** | TTT variant using score-first optimization | PR 503 |

### TTT Impact Analysis:
The biggest jumps in the unmerged PRs all involve **test-time training (TTT)**:
- Without TTT (best merged): 1.1428
- With TTT (best unmerged): 1.0887
- **TTT contribution: ~0.054 BPB** -- this is by far the single largest technique improvement

TTT effectively uses the eval-time compute budget to adapt the model to each validation document, providing massive gains in compression quality.

---

## 7. What Could Beat 1.1428 (Current Merged SOTA)

### Immediate opportunities (techniques already proven in unmerged PRs):

1. **Add Test-Time Training (TTT)**: Expected improvement of 0.02-0.05 BPB based on unmerged evidence. This is the single highest-leverage technique not yet in the merged leader.

2. **Add STE QAT to #1's int5 approach**: #1 uses PTQ only. QAT has been shown to eliminate the quant gap (0.0000 BPB penalty in yahya010's entry). Even small quant gap reductions on int5 could yield 0.002-0.005 BPB.

3. **Try 11 layers with int5 MLP**: If int5 savings + zstd-22 can accommodate 11L, the extra depth could provide 0.003-0.005 BPB.

4. **TrigramHash (extending BigramHash)**: Moving from bigram to trigram context in the hash embedding. Unmerged PR #486 uses this.

5. **Value Residual connections**: New architectural technique in multiple unmerged PRs.

6. **XSA (Cross-Selective Attention)**: Novel attention variant appearing in 3 unmerged PRs.

7. **Partial RoPE**: Only applying positional embeddings to subset of heads.

### Speculative but high-potential:

8. **Adaptive quantization bit-width per layer**: Currently int5 MLP + int6 attention. Could go int4 for some MLP layers, int7 for critical attention layers.

9. **Larger BigramHash/TrigramHash**: Scale to 16K-32K buckets if space permits.

10. **EMA + SWA combination**: Use both exponential moving average and stochastic weight averaging.

---

## 8. Statistical Summary

### Distribution of val_bpb across merged entries:
- **Count**: 17 entries
- **Min**: 1.14276 (thwu1)
- **Max**: 1.22437 (Naive Baseline)
- **Mean**: 1.18461
- **Median**: 1.19250
- **Std Dev**: 0.02795
- **Range**: 0.08161

### Improvement velocity:
- Day 1 (March 18): Baseline at 1.2244
- Day 2 (March 19): Best at 1.1556 (improvement: 0.069 BPB in 24h)
- Day 3 (March 20): Best at 1.1428 (improvement: 0.013 BPB in 24h)
- Day 6 (March 23, unmerged): Best at 1.0887 (improvement: 0.054 BPB in 72h)

The rate of improvement is **accelerating** due to TTT techniques.

### Quantization scheme impact:
| Quant Method | Best val_bpb | Count | Avg val_bpb |
|-------------|-------------|-------|-------------|
| Int5/Int6 mixed + zstd-22 | 1.1428 | 1 | 1.1428 |
| Int6 + zstd-22 | 1.1458 | 5 | 1.1537 |
| Int6/Int8 mixed + zlib | 1.1630 | 1 | 1.1630 |
| Int8 + zlib | 1.1925 | 5 | 1.2097 |

**Moving from int8 to int6 saves ~0.056 BPB on average.**
**Moving from zlib to zstd-22 saves ~0.01-0.02 BPB** (via enabling more params).

---

## 9. Visualization Recommendations

1. **Bar chart**: val_bpb by entry, color-coded by technique family (MODEL/QUANT/TRAINING/EVAL)
2. **Waterfall chart**: Cumulative improvement from baseline, showing marginal contribution of each technique
3. **Heatmap**: Technique presence matrix (entries x techniques), sorted by rank
4. **Timeline scatter**: val_bpb vs date, with merged and unmerged entries distinguished
5. **Pareto frontier**: val_bpb vs artifact size, showing efficiency frontier
6. **Ablation funnel**: From #1 entry's ablation data, showing diminishing returns

---

## 10. Key Takeaways

1. **Quantization is the enabler, not the end goal.** Int6/Int5 quantization's main value is freeing bytes for more model capacity (MLP 3x, extra layers), not just compression.

2. **Sliding Window Eval is table-stakes.** It provides ~0.013 BPB for free. Every competitive entry uses it.

3. **The SmearGate + BigramHash + OrthoInit cluster** is the most powerful architectural innovation set, appearing in all top-4 entries.

4. **SWA + Muon WD** provide the best training-side gains (~0.01 BPB combined).

5. **Test-Time Training is the next frontier.** TTT techniques in unmerged PRs provide 3-5x the improvement of any single merged technique.

6. **The competition is moving from "compression tricks" to "eval-time adaptation."** The unmerged entries suggest the next phase of the competition is fundamentally about test-time compute.
