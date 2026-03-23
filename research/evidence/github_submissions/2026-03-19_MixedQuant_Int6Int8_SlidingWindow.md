---
title: Mixed Quant Int6/Int8 + Sliding Window
source_url: https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-19_MixedQuant_Int6Int8_SlidingWindow
date: 2026-03-19
category: repo
author: aquariouseworkman
val_bpb: 1.1630
---

## Key Method
Four orthogonal improvements stacked:

1. **Wider MLP (MLP_MULT=3)**: 3x expansion (hidden=1536), up from 2x (1024). Total params 17.1M -> 21.8M. Enabled by int6 quantization savings.

2. **Mixed-Precision Post-Training Quantization**: int6 per-row (31 levels) on all 2D block weights (attention/MLP -- these have STE protection during training), int8 per-row (127 levels) on token embedding (no STE protection, needs gentler quant). Reduces quant penalty from +0.048 to +0.0015 bpb -- a 32x improvement. During training, all CastedLinear weights get STE fake int6 quantization.

3. **Optimized Training Config**: seq_len=1024 (shorter=faster steps, 48.4ms vs 55.5ms), batch=524K (better GPU saturation). Result: 12,395 steps x 524K = ~6.5B tokens (vs ~4.25B with old config).

4. **Sliding Window Eval (stride=64)**: ~0.034 bpb free improvement.

## Components Changed
- Architecture: yes -- MLP_MULT=3 (hidden=1536)
- Quantization: yes -- int6 STE QAT for block weights, int8 for embedding, mixed export
- Training: yes -- STE fake quantization during training, optimized seq_len/batch, Muon momentum=0.99, LR=0.020
- Evaluation: yes -- sliding window stride=64
- Compression: yes -- zlib-9 (int6 values in int8 containers compress well)
- Token features: no
- Initialization: no

## Reported Metric
val_bpb: 1.1630 (sliding window, post-quant)
Pre-quant: 1.1950
Standard post-quant: 1.1965 (quant penalty: +0.0015)
Artifact: 15,353,490 bytes
12,395 steps at 48.41 ms/step

Improvement breakdown:
- Wider MLP 3x + seq1024 + 524K batch: 1.1950 (-0.0294)
- Mixed quant: +0.0015 penalty
- Sliding window: 1.1630 (-0.0335 additional)
- Total: -0.0614 over baseline

## Likely Mechanism
3x MLP provides more expressive nonlinear feature transformation. STE QAT teaches the model weight distributions that survive int6 quantization. Mixed precision (int6 blocks, int8 embed) minimizes quant damage where it matters most. Sliding window gives richer eval context.

## Improvement Category
MODEL + QUANT + EVAL

## Interactions
- This is the first submission to combine STE QAT + wider MLP + sliding window
- Foundation for all later top submissions
- Key insight: STE-protected weights can be quantized to int6 aggressively, but embedding (no STE) needs int8
- Stacks with: SmearGate, BigramHash, deeper models, SWA, weight decay tuning

## Implementation Complexity
medium -- requires STE fake quantization in forward pass, mixed precision export logic

## Worth Testing Locally
yes -- this demonstrates the core recipe that all SOTA builds on: STE QAT int6 + wider MLP + sliding window. Understanding this is essential.
