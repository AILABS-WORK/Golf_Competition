---
title: "Mixed-Precision Quantization for Language Models: Techniques and Prospects"
source_url: https://arxiv.org/abs/2510.16805
date: 2025-10
category: paper (survey)
authors: (Survey authors -- arxiv 2510.16805)
---

## Key Idea
A comprehensive survey of mixed-precision quantization strategies for language models. The key insight is that different components of transformers have different sensitivity to quantization, and allocating precision non-uniformly (higher bits for sensitive layers, lower bits for robust layers) yields better accuracy-compression tradeoffs than uniform quantization.

## Method Details
- **Layer-type precision allocation:**
  - Embedding layers and LM Head: typically FP16/BF16 (most sensitive)
  - Attention layers: higher precision (INT6-INT8) due to outlier sensitivity
  - MLP layers: can tolerate lower precision (INT4-INT5)
  - Layer Normalization: FP16/BF16
- **Sensitivity analysis methods:**
  - Hessian-based: compute second-order sensitivity per layer
  - Fisher information-based: use gradient statistics
  - Reconstruction error-based: measure layer-wise output degradation
- **Common successful configurations:**
  - INT4 weights + INT8 activations (W4A8)
  - INT4+FP8 mixed schemes with scale alignment and outlier smoothing
  - Per-group symmetric quantization with low-rank compensation
- **Outlier handling:** Activation outliers in attention and MLP blocks are the main challenge; smoothing techniques (SmoothQuant-style) help redistribute outlier magnitudes

## Reported Results
- Gemma 3 27B: BF16 requires 54GB GPU memory, INT4 reduces to 14.1GB (4x reduction)
- Mixed-precision consistently outperforms uniform quantization at the same average bit-width
- INT4+FP8 with outlier smoothing achieves near-lossless compression for many architectures

## Relevance to Parameter Golf
Validates the current approach of using different bit-widths for different layer types (int5 MLP, int6 attention, FP16 embeddings). Specific insights:
1. The current allocation aligns with literature -- embeddings need highest precision, MLP can go lowest
2. Consider adding outlier smoothing (SmoothQuant-style) before quantizing attention layers
3. Per-group quantization with small group sizes is consistently better than per-tensor
4. Could explore FP8 for activations during training for potential speedup
5. Low-rank compensation could recover some quantization error in the most sensitive layers

## Implementation Complexity
medium

## Expected Impact
- Throughput: no
- Compressed size: yes
- Post-quant loss: yes
- Raw train loss: no

## Key Takeaway for Implementation
The current mixed-precision allocation (int5 MLP, int6 attention, FP16 embeddings) is well-supported by literature. Consider adding outlier smoothing for attention layers and using smaller quantization group sizes for better accuracy. Per-group quantization with groups of 32-128 elements is the sweet spot.
