---
title: Naive Baseline
source_url: https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-17_NaiveBaseline
date: 2026-03-18
category: repo
author: Baseline (OpenAI)
val_bpb: 1.2244
---

## Key Method
The official naive baseline for the Parameter Golf competition. Standard 9-layer 512-dim GPT with 1024-token BPE vocabulary, int8+zlib quantization, and tied embeddings. No tricks -- this is the starting point all other submissions improve upon.

## Components Changed
- Architecture: no -- stock 9L/512d/8H/4KV GPT with relu^2 MLP (2x expansion)
- Quantization: no -- standard int8 per-row + zlib
- Training: no -- default Muon+AdamW with MATRIX_LR=0.04, SCALAR_LR=0.04, TIED_EMBED_LR=0.05
- Evaluation: no -- non-overlapping 1024-token chunks
- Compression: no -- zlib on int8 weights
- Token features: no -- standard 1024-token BPE
- Initialization: no -- default

## Reported Metric
val_bpb: 1.2244 (post-quant int8+zlib roundtrip)
val_loss: 2.0727
pre-quant val_bpb: 1.2172 (quant gap: 0.0072)

## Likely Mechanism
Establishes the baseline performance for a small GPT model within the 16MB artifact constraint.

## Improvement Category
N/A -- baseline

## Interactions
All other submissions build upon or depart from this baseline.

## Implementation Complexity
low

## Worth Testing Locally
yes -- essential as reference point for measuring improvements
