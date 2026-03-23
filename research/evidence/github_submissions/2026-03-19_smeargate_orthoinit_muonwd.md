---
title: SmearGate + OrthoInit + Muon WD + Int6 STE QAT
source_url: https://github.com/openai/parameter-golf/tree/main/records/track_10min_16mb/2026-03-19_smeargate_orthoinit_muonwd
date: 2026-03-19
category: repo
author: aquariouseworkman
val_bpb: 1.1556
---

## Key Method
Introduces three novel architectural/training techniques on top of the STE QAT + wider MLP + sliding window foundation:

### SmearGate (NOVEL)
A learned per-dimension gate (~512 params) that blends each token's embedding with the previous token's embedding:
```
gate = sigmoid(self.gate)  # shape [dim], init via sigmoid(3.0) ~ 0.95
output = gate * current_emb + (1 - gate) * prev_token_emb
```
Injects bigram context directly into the embedding layer. Applied after embedding lookup and bigram hash addition, before RMS normalization. Near-zero parameter cost.

### Bigram Hash Embedding (NOVEL)
4096-bucket hash table (dim=128, projected to 512) maps consecutive token pairs to learned embeddings via `(prev * 92821 + cur) % 4096`. Direct access to token-pair features at ~524K params cost. Complements SmearGate with additive bigram signal.

### Orthogonal Weight Initialization (NOVEL for this competition)
All non-zero-init CastedLinear weights initialized with nn.init.orthogonal_(). All singular values equal to 1 at init = uniform gradient flow. Since Muon orthogonalizes updates, starting from orthogonal means early updates are immediately useful. Critical with only ~12k steps.

Plus all standard techniques:
- 9 layers, 512 dim, MLP 3x (1536)
- STE int6 QAT, FP16 embed passthrough
- Muon WD=0.01, momentum=0.99 with warmup
- U-Net skip connections (4 encoder + 5 decoder)
- Sliding window eval stride=64
- zstd-22 compression
- Logit softcap=30.0, QK_GAIN_INIT=1.5

## Components Changed
- Architecture: yes -- SmearGate, BigramHash(4096), 9L MLP3x, U-Net skips
- Quantization: yes -- int6 STE QAT, FP16 embed, zstd-22
- Training: yes -- Muon WD=0.01, momentum=0.99, LR=0.020, logit softcap=30.0
- Evaluation: yes -- sliding window stride=64
- Compression: yes -- zstd-22
- Token features: yes -- SmearGate + BigramHash(4096, dim=128)
- Initialization: yes -- orthogonal init for all weight matrices

## Reported Metric
val_bpb: 1.1556 (post-quant sliding window)
Standard post-quant: 1.1891
Quant gap: ~0.0001 bpb (nearly zero with STE QAT)
12,047 steps in 600s
Model: 22,368,840 params
Artifact: 15,878,809 bytes (15.1MB)
Eval time: 75s

## Likely Mechanism
SmearGate gives the model free bigram context at the embedding layer, reducing the work attention must do. BigramHash provides additive token-pair features through a different mechanism (hash-table lookup). Orthogonal init provides optimal gradient flow from step 1, critical with limited training budget. Combined with the standard int6 QAT + wider MLP foundation.

## Improvement Category
MODEL + QUANT + COMPRESS + EVAL + TOKEN_FEATURES + INIT

## Interactions
- SmearGate + BigramHash are the KEY novel contributions adopted by both later SOTA submissions
- Orthogonal init synergizes with Muon optimizer (both favor orthogonal matrices)
- SmearGate is nearly free (512 params) and provides reliable improvement
- BigramHash adds ~524K params but provides strong bigram features
- Later submissions increased BigramHash to 8192 and 10240 buckets for more gain
- WD=0.01 was later increased to 0.04 by subsequent submissions

## Implementation Complexity
medium -- SmearGate and BigramHash are simple to implement, orthogonal init straightforward

## Worth Testing Locally
yes -- SmearGate and BigramHash are ESSENTIAL components of SOTA. This is the submission that introduced them. The later #1 and #2 submissions both build directly on these techniques.
