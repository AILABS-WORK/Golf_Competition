---
title: Sliding Window + FP16 Embed + 10L + Muon WD + Overtone Init
source_url: https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit
date: 2026-03-19
category: repo
author: notapplica
val_bpb: 1.1748
---

## Key Method
Combines multiple previous wins with two novel techniques:

1. **Sliding window eval** (stride=64, seq_len=1024): standard eval improvement
2. **FP16 tied embedding export**: standard quant improvement
3. **10 transformer layers** (up from 9): Muon weight decay compresses enough to fit the extra layer
4. **Decoupled weight decay for Muon optimizer** (0.02): Improves generalization and quantization robustness
5. **Overtone spectral embedding init**: SVD power-law spectrum shaping (S_k ~ k^{-0.5}). Novel initialization that shapes the embedding's singular value spectrum.
6. **Phase-transition residual mixing**: Sigmoid-scheduled resid_mix initialization. Novel technique.

## Components Changed
- Architecture: yes -- 10 layers
- Quantization: yes -- FP16 embed, int8 block weights
- Training: yes -- Muon weight decay=0.02, compiled forward_logits
- Evaluation: yes -- sliding window stride=64
- Compression: no -- still int8+zlib (but artifact ~14.7MB, lots of headroom)
- Token features: no
- Initialization: yes -- overtone spectral embedding init, phase-transition residual mixing

## Reported Metric
val_bpb: 1.1748 (mean of 3 seeds: 1.1756, 1.1742, 1.1744)
std: 0.0008
~10,500 steps at ~57ms/step
Artifact: ~14.7MB (significant headroom under 16MB)
Eval time: ~162s (sliding window)

## Likely Mechanism
Muon weight decay regularizes weight magnitudes, producing tighter distributions that quantize better. Overtone spectral init provides a principled starting point for the embedding matrix based on power-law spectrum. Phase-transition residual mixing helps early training dynamics. Combined with 10 layers and sliding window for a solid improvement.

## Improvement Category
MODEL + QUANT + EVAL + TRAINING

## Interactions
- The 14.7MB artifact size leaves ~1.3MB headroom -- could be used for wider MLP or more layers
- Muon weight decay was later adopted by all top submissions (at WD=0.04, higher than this submission's 0.02)
- Overtone init is interesting but not picked up by later SOTA submissions
- Stacks with: int6 QAT, wider MLP, SmearGate, BigramHash

## Implementation Complexity
medium -- custom spectral init, residual mixing, weight decay integration with Muon

## Worth Testing Locally
yes -- the Muon weight decay idea proved critical. The overtone init is worth investigating as a potential additional gain not yet combined with SOTA. The large headroom (~1.3MB) suggests the model budget is underutilized.
