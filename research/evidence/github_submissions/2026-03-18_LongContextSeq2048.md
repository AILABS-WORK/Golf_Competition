---
title: Long Context Seq2048
source_url: https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-18_LongContextSeq2048
date: 2026-03-18
category: repo
author: Spokane Way (spokane-way)
val_bpb: 1.2058
---

## Key Method
Increased training sequence length from 1024 to 2048 tokens, with tuned learning rates for the longer context. Each training sequence sees 2x more context, improving autoregressive signal per token. Costs more ms/step (~52ms vs ~44ms) but quality improvement outweighs fewer total steps.

Learning rates tuned down: TIED_EMBED_LR=0.04, MATRIX_LR=0.032, SCALAR_LR=0.032 (vs defaults of 0.05/0.04/0.04).

## Components Changed
- Architecture: no -- same 9L/512d/8H/4KV
- Quantization: no -- standard int8+zlib
- Training: yes -- TRAIN_SEQ_LEN=2048, lower LRs (0.04/0.032/0.032)
- Evaluation: no -- standard eval (but at seq_len=2048)
- Compression: no
- Token features: no
- Initialization: no

## Reported Metric
val_bpb: 1.2058 (seed 1337), 1.2062 (seed 1338), 1.2072 (seed 1339)
Mean: 1.2064, std: 0.00072
11,564 steps at 51.89 ms/step
Quant gap: ~0.005 bpb

## Likely Mechanism
Longer context gives the autoregressive model more signal per token during training, leading to better language modeling. The model learns longer-range dependencies that improve perplexity even at the small 512-dim scale.

## Improvement Category
TRAINING

## Interactions
- Stacks with: sliding window eval, quantization improvements, deeper models
- Later submissions (Seq4096) pushed this further to 4096 seq length
- Tradeoff: longer seq = slower steps = fewer total steps. 2048 appears to be a good balance.

## Implementation Complexity
low -- just change TRAIN_SEQ_LEN and tune LRs

## Worth Testing Locally
yes -- seq_len=2048 is now standard in most top submissions. Consider testing seq_len=2048 vs 4096 tradeoffs.
