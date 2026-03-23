---
title: 10L Mixed Precision
source_url: https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-19_10L_MixedPrecision
date: 2026-03-19
category: repo
author: Nan Liu (nanlliu)
val_bpb: 1.2147
---

## Key Method
Three improvements over baseline:
1. 10 transformer layers instead of 9 -- adds depth for better language modeling
2. Lower LRs (MATRIX_LR=0.02, SCALAR_LR=0.02, TIED_EMBED_LR=0.03)
3. Mixed int8/int6 compression -- middle layers (3,4,5,6) use int6 precision (round int8 to nearest 4) for better zlib compression, while first/last layers keep full int8

The 10-layer model has 18.9M params -> 17.6MB with standard int8+zlib (1.6MB over cap). Reducing middle layers to int6 drops compressed size to 15.9MB with only 0.0018 bpb quality loss.

Key insight: early and late layers are critical (input processing / output quality), middle layers are more compressible.

## Components Changed
- Architecture: yes -- 10 layers (from 9)
- Quantization: yes -- mixed int8/int6 (layers 3-6 int6, rest int8)
- Training: yes -- lower LRs
- Evaluation: no
- Compression: no -- still zlib, but int6 data compresses better
- Token features: no
- Initialization: no

## Reported Metric
val_bpb: 1.2147 (post-quant)
Pre-quant: 1.2129
Quant gap: 0.0018 (vs baseline 0.0093)
13,100 steps at 45.78 ms/step
Artifact: 15,928,974 bytes

## Likely Mechanism
Extra layer adds model capacity. Int6 on middle (less sensitive) layers enables fitting the extra layer within 16MB. Lower LR reduces quant gap.

## Improvement Category
MODEL + QUANT

## Interactions
- Stacks with: sliding window eval, QAT, FP16 embed
- Later submissions took this further with uniform int6 + QAT + zstd for even more aggressive compression
- The layer sensitivity analysis (early/late int8, middle int6) is useful insight

## Implementation Complexity
low -- env var config changes + minor quantization code modification

## Worth Testing Locally
no -- superseded by later submissions that use full int6 + QAT which achieves zero quant gap. The layer sensitivity insight is valuable though.
