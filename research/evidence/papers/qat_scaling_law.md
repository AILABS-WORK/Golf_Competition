---
title: "Scaling Law for Quantization-Aware Training"
source_url: https://arxiv.org/abs/2505.14302
date: 2025-05
category: paper
authors: Chen et al. (OpenGVLab / related group)
---

## Key Idea
A unified scaling law for QAT that models quantization error as a function of model size, training data volume, and quantization group size. Through 268 QAT experiments, the paper reveals that quantization error DECREASES with larger model size but INCREASES with more training tokens and coarser quantization granularity.

## Method Details
- Conducted 268 controlled QAT experiments varying model size, training tokens, and quantization group size
- Decomposes quantization error into weight quantization error and activation quantization error
- Weight quantization error increases more rapidly with more training tokens
- Activation quantization error in the FC2 layer (caused by outliers) is the PRIMARY bottleneck of W4A4 QAT
- Mixed-precision quantization applied to the FC2 layer can bring weight and activation errors to similar levels
- Quantization group size has massive impact: difference between coarsest and finest granularity is nearly half the total error

## Reported Results
- Increasing training tokens from 10B to 100B causes ~22% increase in W4A4 quantization error
- Larger models (more parameters) show reduced quantization error at the same bit-width
- Mixed-precision on the FC2 layer resolves the activation outlier bottleneck

## Relevance to Parameter Golf
CRITICAL finding: for Parameter Golf, we are training a relatively small model with limited tokens (10 min on 8xH100). The scaling law suggests:
1. Fewer training tokens = LESS quantization error (favorable for our short training)
2. We should use fine-grained quantization groups (smaller group size) to minimize error
3. The FC2 (second MLP linear) layer is the activation quantization bottleneck -- this validates using higher precision (int6) for attention and different precision for MLP layers
4. Consider mixed-precision specifically targeting the FC2 layer with higher bit-width

## Implementation Complexity
low

## Expected Impact
- Throughput: no
- Compressed size: no
- Post-quant loss: yes
- Raw train loss: yes

## Key Takeaway for Implementation
Use fine-grained quantization groups (small group size) and consider giving the FC2 MLP layer higher bit-width to resolve activation outlier bottlenecks. The short 10-minute training budget actually helps since fewer tokens means less quantization error.
