---
title: "Scaling Laws for Neural Language Models"
source_url: https://arxiv.org/abs/2001.08361
date: 2020-01
category: paper
authors: Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B. Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Ilya Sutskever (OpenAI)
---

## Key Idea
Loss scales as a power-law with model size (N), dataset size (D), and compute (C), spanning over seven orders of magnitude. Under a fixed compute budget, there are optimal allocations of model size and training tokens, and larger models are increasingly sample-efficient.

## Method Details
- Loss follows power laws: L(N) ~ N^(-0.076), L(D) ~ D^(-0.095), L(C) ~ C^(-0.050)
- For compute-optimal training under budget C:
  - Optimal model size N* ~ C^0.73
  - Optimal dataset size D* ~ C^0.27
  - Most of the compute budget should go to larger models (not more data)
- Models show smooth, predictable scaling across 7+ orders of magnitude
- Batch size has a critical threshold: below it, training is compute-bound; above it, data-bound
- The optimal batch size scales with the loss achieved: B_crit ~ L^(-4.7)

## Reported Results
- Power law relationships hold remarkably consistently
- Larger models reach the same loss with fewer training tokens
- The compute-optimal frontier is well-characterized

## Relevance to Parameter Golf
Provides the theoretical framework for choosing model size given our constraints:
1. Fixed compute: 10 min on 8xH100 (~2.67e18 FLOPS assuming 50% utilization)
2. The scaling law suggests: make the model as LARGE as possible that fits in 16MB compressed, and train for fewer tokens
3. This aligns with the "undertrained models quantize better" finding
4. However, Chinchilla later updated the optimal ratio -- MiniCPM's WSD scheduler further refines this for small models
5. The key tension: model must fit in 16MB after quantization, so there is a hard parameter ceiling

## Implementation Complexity
low

## Expected Impact
- Throughput: no
- Compressed size: no
- Post-quant loss: no
- Raw train loss: yes

## Key Takeaway for Implementation
Maximize model parameters (hidden dim, layers) up to the 16MB compressed limit, accepting that the model will be undertrained. The power law says a larger undertrained model beats a smaller well-trained one at the same compute budget. Combined with the quantization-favors-undertrained insight, this is a double win.
