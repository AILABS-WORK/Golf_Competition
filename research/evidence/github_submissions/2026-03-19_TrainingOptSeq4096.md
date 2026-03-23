---
title: Training Opt Seq4096
source_url: https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-19_TrainingOptSeq4096
date: 2026-03-19
category: repo
author: Spokane Way (spokane-way)
val_bpb: 1.2014
---

## Key Method
Combines longer training context (seq_len=4096) with aggressive Muon optimizer tuning:
1. 4x context (4096 vs 1024) per training sequence -- better signal per token
2. Higher Muon momentum (0.99 vs 0.95) with warmup from 0.92 over 1500 steps
3. Lower LRs (0.020/0.020/0.030)
4. 3/4 batch (393K vs 524K tokens) -- more optimizer updates per wallclock
5. Extended momentum warmup (1500 steps from 0.92)
6. Longer warmdown (3000 steps)

Costs 71ms/step (vs ~43ms at seq_len=1024) but quality improvement outweighs.

## Components Changed
- Architecture: no -- same 9L/512d/8H/4KV
- Quantization: no -- standard int8+zlib
- Training: yes -- seq_len=4096, Muon momentum 0.99, lower LRs, 3/4 batch, warmdown=3000
- Evaluation: no -- standard eval
- Compression: no
- Token features: no
- Initialization: no

## Reported Metric
val_bpb: 1.2014 (mean of 3 seeds: 1.2014, 1.1995, 1.2032)
std: 0.00187
8,394 steps at 71.47 ms/step
Quant penalty: 0.0034 bpb (lower LR helps)

## Likely Mechanism
4K context provides richer learning signal per token. Higher Muon momentum (0.99) with proper warmup gives better gradient smoothing. Lower LR reduces quantization damage. The combination yields 0.023 bpb improvement over baseline.

## Improvement Category
TRAINING

## Interactions
- Stacks with: sliding window eval, quantization improvements, model architecture changes
- Note: later submissions found seq_len=2048 with batch=786K to be better than seq_len=4096 with batch=393K (more tokens/step at reasonable seq length)
- The Muon momentum=0.99 with warmup became standard in all later submissions

## Implementation Complexity
low -- hyperparameter changes only

## Worth Testing Locally
no -- superseded by later submissions. However, the Muon momentum=0.99 finding is critical and used by all SOTA. seq_len=2048 appears to be the sweet spot (used by more later submissions than 4096).
