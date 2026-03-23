---
title: Warmdown-Quantization - Training for Compression
source_url: https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-19_WarmdownQuantization
date: 2026-03-19
category: repo
author: samuellarson
val_bpb: 1.2154
---

## Key Method
Attacks the quantization bottleneck from multiple angles:

1. **Always-Decaying LR Schedule (WARMDOWN_ITERS=20000)**: Setting warmdown far beyond actual training steps (~12,200) means the entire training run is in the decay phase. LR decays linearly from 61% of peak. Post-quant penalty drops from 0.014 bpb (WD=1200) to 0.005 bpb (WD=20000). Aggressive LR decay produces tighter weight distributions with fewer outliers.

2. **FP16 Tied Embeddings**: Further reduces post-quant penalty from 0.005 to ~0.001 bpb. Costs ~500KB offset by MLP_HIDDEN=992.

3. **Optimal NTK-RoPE Extrapolation**: eval@1408 (1.375x training length) is optimal for well-trained models (+0.007 bpb). Well-trained models have precise position-dependent patterns that aggressive extrapolation distorts. (Note: eval@2048 is neutral-to-negative for well-trained models.)

4. **Optimizer-Warmdown Interaction**: MUON_BACKEND_STEPS=5 outperforms 7 when combined with aggressive warmdown. With smooth weights already from warmdown, more training steps > better per-step gradient quality.

Config: WARMDOWN_ITERS=20000, MATRIX_LR=0.06, TIED_EMBED_LR=0.07, SCALAR_LR=0.06, GRAD_CLIP_NORM=1.0, MUON_BACKEND_STEPS=5, EVAL_SEQ_LEN=1408, MLP_HIDDEN=992

## Components Changed
- Architecture: yes -- MLP_HIDDEN=992 (to fit FP16 embed)
- Quantization: yes -- FP16 embed passthrough, optimized for minimal quant gap
- Training: yes -- aggressive warmdown (WARMDOWN_ITERS=20000), higher LRs (0.06/0.07/0.06), MUON_BACKEND_STEPS=5, GRAD_CLIP_NORM=1.0
- Evaluation: yes -- NTK-RoPE extrapolation at eval_seq_len=1408
- Compression: no -- still int8+zlib
- Token features: no
- Initialization: no

## Reported Metric
val_bpb: 1.2154
Improvement: 0.009 bpb / 0.017 nats over baseline

## Likely Mechanism
The key insight is that post-training quantization penalty is the dominant bottleneck, not model quality. Aggressive LR warmdown produces tighter weight distributions that quantize with less damage. The FP16 embedding further reduces the remaining quant gap. NTK-RoPE extrapolation adds a small eval-time boost.

## Improvement Category
QUANT + TRAINING + EVAL

## Interactions
- The warmdown insight: WARMDOWN_ITERS>>total_steps produces smoother weights. This interacts with QAT -- with QAT, the model already learns to handle quantization, so the warmdown benefit may be smaller.
- FP16 embed stacks with everything.
- NTK-RoPE extrapolation to 1408 is interesting but superseded by sliding window eval in later submissions.
- The MUON_BACKEND_STEPS=5 vs 7 finding is specific to aggressive warmdown.

## Implementation Complexity
medium -- requires understanding of warmdown/quantization interaction, NTK-RoPE scaling

## Worth Testing Locally
yes -- the aggressive warmdown idea (WARMDOWN_ITERS >> actual steps) is a novel insight not fully explored by SOTA. Most SOTA submissions use WARMDOWN_ITERS=3000 which is more conventional. Testing WARMDOWN_ITERS=20000 combined with QAT could be interesting -- though QAT may make it redundant.
