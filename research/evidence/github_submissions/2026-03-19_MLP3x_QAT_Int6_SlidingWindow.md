---
title: 11L MLP3x + Int6 QAT + zstd-22 + Sliding Window
source_url: https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-19_MLP3x_QAT_Int6_SlidingWindow
date: 2026-03-20
category: repo
author: aruniyer
val_bpb: 1.1502
---

## Key Method
The most aggressive architecture scaling combined with every known trick:

1. **11 transformer layers** (vs 9 baseline) -- most layers of any submission
2. **Wider MLP (MLP_MULT=3)** -- 3x expansion (hidden=1536), more capacity per layer
3. **Decoupled weight decay (0.04)** -- on both Muon and AdamW
4. **QAT int6** -- STE fake-quantize simulates int6 noise during training
5. **Int6 quantization on all block weights** (layers 0-10)
6. **FP16 tied embedding export**
7. **zstd-22 compression** -- saves ~1.5MB vs zlib, critical for fitting 11L MLP3x
8. **Sliding window evaluation (stride=64)**
9. **Higher Muon momentum (0.99)** with warmup from 0.92 over 1500 steps
10. **Lower LRs**: MATRIX_LR=0.025, SCALAR_LR=0.025, TIED_EMBED_LR=0.035

Architecture: 11 blocks, 512 model dim, 8 attention heads, 4 KV heads, GQA+RoPE, relu^2 MLP (3x), tied embeddings, U-Net skip connections. 26.5M parameters, ~15.4MB compressed.

## Components Changed
- Architecture: yes -- 11 layers (most of any submission), MLP 3x, U-Net skip connections
- Quantization: yes -- int6 STE QAT on all block weights (layers 0-10), FP16 embed
- Training: yes -- WD=0.04, Muon momentum=0.99 with warmup, LR=0.025, warmdown=3000
- Evaluation: yes -- sliding window stride=64
- Compression: yes -- zstd-22
- Token features: no -- no bigram features
- Initialization: no -- standard init

## Reported Metric
val_bpb: 1.1502 (mean of 3 seeds: 1.1506, 1.1502, 1.1497)
val_loss_std: 0.00072
Roundtrip bpb: 1.1845 (standard eval)
Mean improvement: 0.1307 nats over baseline
t-statistic: 313.20 (df=2, p << 0.001)
~10,070 steps at ~59.6ms/step
Artifact: ~15.4MB
Sliding window eval: ~88s
Requires: zstandard package

## Likely Mechanism
11 layers is the deepest model in the competition, funded by aggressive int6+zstd compression. The 3x MLP at 11 layers means 26.5M params (vs 17M baseline), all fitting in 16MB through int6 QAT + zstd-22. Weight decay=0.04 keeps weights small for better quantization. U-Net skip connections help gradient flow through 11 layers.

## Improvement Category
MODEL + QUANT + COMPRESS + EVAL

## Interactions
- Does NOT use SmearGate or BigramHash (which later submissions add on top)
- The 11-layer architecture with MLP3x is the most parameter-rich approach
- Could potentially be improved by adding SmearGate + BigramHash
- Weight decay 0.04 on both Muon and AdamW is higher than earlier submissions (0.01-0.02)
- U-Net skip connections help with the deeper architecture

## Implementation Complexity
medium -- int6 QAT, zstd integration, U-Net skips

## Worth Testing Locally
yes -- this is the 3rd best submission. The key question is whether 11 layers beats 9-10 layers + SmearGate/BigramHash. Could combine this architecture depth with the token features from later submissions.
