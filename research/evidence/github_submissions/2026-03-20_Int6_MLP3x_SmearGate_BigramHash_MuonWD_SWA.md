---
title: Int6 MLP3x + SmearGate + BigramHash + OrthoInit + Muon WD + SWA
source_url: https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA
date: 2026-03-20
category: repo
author: Raahil Shah (raahilshah)
val_bpb: 1.1458
---

## Key Method
Seven techniques stacked on the baseline 9-layer, 512-dim GPT. This was the #2 submission when it landed:

1. **Per-Row Int6 Quantization + zstd-22**: MLP and attention weights quantized to int6 [-32,31] with per-row scaling. Tied embeddings in FP16. Last layer key projection in FP16 (reduces late-layer attention quant penalty). zstd-22 gives ~5% better compression than zlib-9 on int6 data.

2. **3x MLP Expansion**: hidden 1024->1536. Single largest contributor to improvement.

3. **SmearGate**: Learned gate blending current/previous token embeddings. ~512 params.

4. **BigramHash Embedding**: 4096-bucket hash table (dim=128, projected to 512). Maps (prev_token * 31 + curr_token) % 4096. ~524K params.

5. **Orthogonal Weight Initialization**: orthogonal_(gain=1.0), output projections scaled by 1/sqrt(2*num_layers) following muP conventions.

6. **Muon Optimizer with Weight Decay**: WD=0.04 (swept 0.01-0.05, optimal at 0.04). AdamW WD=0.01 for embedding/scalars. Momentum warmup 0.92->0.99 over 1500 steps.

7. **Stochastic Weight Averaging (SWA)**: Every 50 steps over last 50% of training (~30 checkpoints averaged). Produces smoother weight distributions that quantize better. Swept swa_every from 200 down to 25; optimal at 50.

## Components Changed
- Architecture: yes -- 9L MLP3x (hidden=1536)
- Quantization: yes -- int6 per-row on block weights, FP16 embed + last-layer K proj, zstd-22
- Training: yes -- WD=0.04 Muon, WD=0.01 AdamW, momentum=0.99, LR=0.02/0.02/0.03, grad_clip=0.3, seq_len=2048, batch=786432, warmdown=3000
- Evaluation: yes -- sliding window stride=64
- Compression: yes -- zstd-22
- Token features: yes -- SmearGate + BigramHash(4096, dim=128)
- Initialization: yes -- orthogonal + muP output scaling

## Reported Metric
val_bpb: 1.1458 (mean of 3 seeds: 1.1460, 1.1466, 1.1449)
std: 0.0008
Pre-quant: 1.1616
Quant penalty: 0.016 bpb (int6 vs fp16)
7,379 steps at 81.3 ms/step
Model: ~22M params
Artifact: 15.86MB
Improvement over prior SOTA (1.1748): -0.0290 bpb / -0.0503 nats

## Likely Mechanism
3x MLP is the single largest contributor. SWA smooths weight distributions for better quantization. WD=0.04 is the optimal regularization strength. SmearGate+BigramHash provide free bigram features. Orthogonal init accelerates convergence with limited steps.

## Improvement Category
MODEL + QUANT + COMPRESS + EVAL + TOKEN_FEATURES + INIT

## Interactions
- This is essentially the smeargate_orthoinit_muonwd recipe with WD=0.04 (up from 0.01) and SWA added
- SWA is a novel addition that smooths weights for better quantization
- WD=0.04 is the key tuning finding (swept 0.01-0.05)
- The SOTA (#1) builds on this by adding 10th layer + int5 MLP + larger BigramHash
- Last-layer key projection in FP16 is a detail not seen in other submissions

## Implementation Complexity
medium -- all individual components are straightforward, complexity is in stacking them correctly

## Worth Testing Locally
yes -- this is the #2 submission and very close to SOTA. Key additions over smeargate submission: WD=0.04, SWA(50 steps, 50% training). Both are simple to add. Consider this the strong baseline for experimentation.
