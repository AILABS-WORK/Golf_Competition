---
title: 10L Int5-MLP + BigramHash(10240) + SWA(frac=0.4) + WD=0.04
source_url: https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-20_10L_Int5MLP_MuonWD04_SWA50
date: 2026-03-20
category: repo
author: thwu1
val_bpb: 1.1428
---

## Key Method
CURRENT SOTA (#1 on leaderboard). Builds on PR #162 by @unnir (SmearGate, BigramHash, OrthoInit) with several targeted improvements:

### Mixed Int5/Int6 Quantization (NOVEL)
- **Int5 [-16,15]** for MLP weights (most compressible, 1.88x zstd ratio)
- **Int6 [-32,31]** for attention weights (precision-sensitive, 1.51x zstd ratio)
- **FP16** for tied embeddings and last-layer key projections
- Int5 MLP saves ~1.86MB vs uniform int6, funding a 10th transformer layer

### BigramHash(10240)
Increased from 4096 to 10240 buckets, reducing token-pair hash collisions (+0.001 bpb per bucket increase).

### SWA with start_frac=0.4 (tuned)
Collect checkpoints only from last 40% of warmdown (most converged). 24 checkpoints averaged every 50 steps. Quality over quantity: fewer but better-converged checkpoints.

### Architecture
- 10 layers, 512 dim, 8 heads, 4 KV heads (GQA)
- MLP 3x (hidden=1536), relu^2 activation
- SmearGate + BigramHash(10240, dim=128)
- Orthogonal init with muP-scaled output projections
- U-Net skip connections, tied embeddings

### Training Hyperparameters
- Muon: matrix_lr=0.02, WD=0.04, momentum=0.99
- AdamW for embeddings/scalars: WD=0.04
- warmdown=3000, warmup=20 steps
- seq_len=2048, batch=786K tokens
- grad_clip=0.3, 3% magnitude pruning
- SWA: start_frac=0.4, every=50 steps
- Sliding window eval: stride=64

## Components Changed
- Architecture: yes -- 10 layers (up from 9), MLP 3x, U-Net skips, SmearGate
- Quantization: yes -- mixed int5(MLP)/int6(attn), FP16 embed+last-K, zstd-22
- Training: yes -- WD=0.04, Muon momentum=0.99, LR=0.02, grad_clip=0.3, 3% mag pruning, SWA(frac=0.4, every=50)
- Evaluation: yes -- sliding window stride=64
- Compression: yes -- zstd-22 with mixed int5/int6
- Token features: yes -- SmearGate + BigramHash(10240, dim=128)
- Initialization: yes -- orthogonal + muP output scaling

## Reported Metric
val_bpb: 1.1428 (mean of 3 seeds: 1.1427, 1.1430, 1.1426)
std: 0.00016 (extremely tight)
Artifact: ~15.9MB (just under 16MB)

## Ablation Summary (from PR #162 base at 1.1485)
- + int5 MLP + 10th layer: 1.1453 (-0.003)
- + WD=0.04 + warmdown=3000: 1.1452 (-0.0001)
- + SWA_start_frac=0.4: 1.1446 (-0.0006)
- + bigram=8192: 1.1434 (-0.0012)
- + bigram=10240: 1.1426 (-0.0008)

## Likely Mechanism
Int5 on MLP weights (the largest tensors) saves enough bytes to add a 10th transformer layer while staying under 16MB. MLP weights are more compressible than attention weights because they have more regular distributions. Larger BigramHash table reduces collisions, giving the model more unique bigram features. Tighter SWA window (last 40%) uses only the most converged checkpoints for better weight averaging.

## Improvement Category
MODEL + QUANT + COMPRESS + EVAL + TOKEN_FEATURES + INIT

## Interactions
- Builds directly on SmearGate/BigramHash/OrthoInit from PR #162
- Int5 MLP is a novel quantization strategy -- MLP weights tolerate lower precision
- The 10th layer funded by int5 savings is the key architectural win
- BigramHash scaling (4096->8192->10240) shows diminishing but real returns
- SWA tuning (frac=0.4 vs 0.5) is a small but measurable win
- 3% magnitude pruning is mentioned but contribution unclear

## Implementation Complexity
medium-high -- mixed int5/int6 quantization requires per-layer precision logic, larger BigramHash, SWA fraction tuning

## Worth Testing Locally
yes -- THIS IS THE TARGET TO BEAT. Key innovations over #2: int5 MLP weights, 10th layer, BigramHash(10240), SWA_frac=0.4, 3% magnitude pruning. To beat this, explore: int4 MLP?, even more BigramHash buckets, different SWA strategies, LoRA TTT on top, deeper models (12L?), different architectures entirely.
