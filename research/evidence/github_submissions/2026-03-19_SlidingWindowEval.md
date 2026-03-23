---
title: Sliding Window Evaluation
source_url: https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-19_SlidingWindowEval
date: 2026-03-19
category: repo
author: Matthew Li (mattqlf)
val_bpb: 1.1925
---

## Key Method
Pure evaluation improvement. Uses overlapping sliding windows with stride=64 and seq_len=1024. Each window advances by 64 tokens, but only the rightmost 64 tokens (which have 960+ tokens of context) are scored. Every token is scored exactly once, with near-maximum context.

Baseline evaluates with non-overlapping 1024-token chunks where the first token has zero context (average context ~512 tokens). Sliding window gives every scored token 960+ context tokens.

Training is identical to naive baseline. Improvement comes entirely from evaluation.

## Components Changed
- Architecture: no
- Quantization: no -- standard int8+zlib
- Training: no -- identical to baseline
- Evaluation: yes -- sliding window at stride=64, batch_seqs=1024
- Compression: no
- Token features: no
- Initialization: no

## Reported Metric
val_bpb: 1.1925 (post-quant sliding window)
Pre-quant: 1.2196 (nearly same as baseline -- training unchanged)
Improvement: -0.0319 bpb (entirely from eval)
Eval time: 70s (vs ~16s for baseline) -- within 10-min eval budget
13,450 steps at 44.61 ms/step

## Likely Mechanism
Every token gets near-maximum context during evaluation. The model already learned to use context during training; sliding window eval simply lets it use more context for each scored position. This is "free" quality improvement with zero artifact cost.

## Improvement Category
EVAL

## Interactions
- Stacks with: ALL model/training improvements (it changes nothing about training)
- Adopted by every subsequent top submission
- stride=64 is the standard (more context per token than stride=256)
- Eval time increases from ~16s to ~70s, but well within 10-min eval budget

## Implementation Complexity
low -- add eval_val_sliding function + forward_logits compiled method

## Worth Testing Locally
yes -- this is a MUST-HAVE. ~0.032 bpb free improvement that stacks with everything else.
