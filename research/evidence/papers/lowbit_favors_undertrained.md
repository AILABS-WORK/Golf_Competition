---
title: "Low-Bit Quantization Favors Undertrained LLMs: Scaling Laws for Quantized LLMs with 100T Training Tokens"
source_url: https://arxiv.org/abs/2411.17691
date: 2024-11 (ACL 2025)
category: paper
authors: Xu et al.
---

## Key Idea
Models with larger sizes or fewer training tokens experience LESS quantization-induced degradation (QiD). Smaller models with extensive training suffer significant QiD. Early undertrained checkpoints have significant weight fluctuations during training, making them more robust to the small perturbations introduced by quantization.

## Method Details
- Examined over 1500 quantized LLM checkpoints of various sizes at different training levels
- Derived scaling laws relating QiD to number of training tokens, model size, and bit width
- Key finding: undertrained models are more robust to quantization because their weights are still fluctuating significantly, so quantization noise is small relative to training noise
- As models become more fully trained, weights settle into narrow loss basins where quantization perturbations cause larger degradation
- Predicted that future models trained on 100T+ tokens may have poor low-bit quantization performance

## Reported Results
- Consistent trend across all model sizes: more training tokens = more quantization degradation
- The effect is more pronounced at lower bit widths (2-bit, 3-bit)
- Larger models tolerate quantization better at the same training level

## Relevance to Parameter Golf
HIGHLY relevant -- Parameter Golf trains for only 10 minutes, producing an "undertrained" model by the paper's definition. This is GOOD news: our model will naturally be more robust to post-training quantization. However, this also means:
1. We can potentially use more aggressive quantization (lower bits) because the model is undertrained
2. If using QAT, the model learns to accommodate quantization from the start, combining both advantages
3. The key tradeoff: make the model as large as possible (within compressed 16MB) even if it means fewer effective training tokens, because larger + undertrained > smaller + well-trained when quantized

## Implementation Complexity
low

## Expected Impact
- Throughput: no
- Compressed size: yes
- Post-quant loss: yes
- Raw train loss: no

## Key Takeaway for Implementation
Favor a LARGER model trained for fewer effective steps over a smaller model trained more thoroughly. The short 10-minute training window works in our favor for quantization robustness. Consider slightly more aggressive quantization (e.g., int4 for some layers) since the undertrained model tolerates it.
