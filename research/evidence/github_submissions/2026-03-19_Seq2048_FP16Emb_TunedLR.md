---
title: 10L Int6 QAT + Zstd MLP2.6x
source_url: https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-19_Seq2048_FP16Emb_TunedLR
date: 2026-03-19
category: repo
author: yahya010
val_bpb: 1.1586
---

## Key Method
Stacked improvements on baseline with 10 key changes:

1. **10 transformer layers** (from 9)
2. **STE int6 QAT**: Straight-through estimator fake quantization during training, completely eliminating the quant gap (pre-quant = post-quant)
3. **Full int6 quantization**: All 2D block weights quantized to [-31,31] (63 levels)
4. **zstd-22 compression**: Better than zlib for int6 data
5. **MLP hidden 1344** (2.625x model_dim): Wider MLP enabled by int6+zstd savings
6. **FP16 tied embedding passthrough**
7. **Sequence length 2048**
8. **Muon momentum 0.99**, warmup from 0.92 over 1500 steps
9. **MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.04**
10. **Gradient clipping** GRAD_CLIP_NORM=0.3
11. **Sliding window evaluation** stride=64

## Components Changed
- Architecture: yes -- 10 layers, MLP hidden=1344 (2.625x)
- Quantization: yes -- int6 STE QAT, zero quant gap
- Training: yes -- seq_len=2048, Muon 0.99, LR tuned, grad_clip=0.3
- Evaluation: yes -- sliding window stride=64
- Compression: yes -- zstd-22 (saves ~1.5MB vs zlib)
- Token features: no
- Initialization: no

## Reported Metric
val_bpb: 1.1586 (mean of 3 seeds: 1.1610, 1.1598, 1.1586)
Mean val_loss (sliding): 1.9583
Quant gap: **0.0000** -- STE QAT completely eliminated quantization loss
~8,300 steps at ~72 ms/step
QAT overhead: ~28% (72ms vs 69ms without)
Sliding window eval time: ~370s
Artifact: 15,558,319 bytes

## Likely Mechanism
Full int6 QAT trains the model to be inherently robust to 6-bit quantization, completely eliminating post-training quant gap. This enables fitting more parameters (10 layers, wider MLP) within 16MB. zstd-22 provides further compression savings over zlib. Sliding window eval provides the usual ~0.033 bpb boost.

## Improvement Category
MODEL + QUANT + COMPRESS + EVAL

## Interactions
- Demonstrates zero quant gap with STE int6 QAT -- a critical finding
- zstd-22 vs zlib: important compression choice that enables larger models
- 10 layers + MLP=1344 is a different architecture point than the 9L/MLP=1536 used by other submissions
- grad_clip=0.3 appears important for stability with int6 QAT
- Stacks with: SmearGate, BigramHash, SWA, orthogonal init

## Implementation Complexity
medium -- QAT implementation, zstd integration, sliding window eval

## Worth Testing Locally
yes -- zero quant gap via STE QAT is one of the most important findings. The zstd-22 switch is also critical for enabling larger models. Consider this submission as a strong foundation recipe.
