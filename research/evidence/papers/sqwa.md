---
title: "SQWA: Stochastic Quantized Weight Averaging for Improving the Generalization Capability of Low-Precision Deep Neural Networks"
source_url: https://arxiv.org/abs/2002.00343
date: 2020-02 (IEEE 2021)
category: paper
authors: Shin, Boo et al.
---

## Key Idea
SQWA combines Stochastic Weight Averaging (SWA) with quantization to create low-precision DNNs with superior generalization. The method captures multiple low-precision model snapshots during training with cyclical learning rates, averages them, and then re-quantizes and fine-tunes the result. This finds flatter minima in the quantized weight space.

## Method Details
1. Train a floating-point model normally
2. Directly quantize weights to low precision
3. Retrain with cyclical learning rates, capturing multiple low-precision model snapshots
4. Average the captured low-precision models (in float space)
5. Re-quantize the averaged model
6. Fine-tune with a low learning rate
The cyclical learning rate is key: it pushes the model to explore different low-precision configurations, and averaging them yields a solution near a flat minimum in the loss landscape.

## Reported Results
- Consistently improves generalization of quantized models across architectures
- The improvement of SQWA over plain quantized SGD is LARGER than the improvement of SWA over full-precision SGD
- Works effectively for low-precision (4-bit, 8-bit) weight representations

## Relevance to Parameter Golf
Directly relevant to current approach. The competition already uses SWA. SQWA extends this by:
1. Performing the averaging in the quantized domain rather than averaging then quantizing
2. Using cyclical learning rates to explore the quantized loss landscape
3. The re-quantize + fine-tune step could be added as a final phase in the 10-minute training window
This could improve the already-used SWA approach by making it quantization-aware.

## Implementation Complexity
medium

## Expected Impact
- Throughput: no
- Compressed size: no
- Post-quant loss: yes
- Raw train loss: no

## Key Takeaway for Implementation
Extend the current SWA implementation to be quantization-aware: average model snapshots in the quantized domain using cyclical learning rates, then re-quantize and fine-tune. This targets the exact gap between training loss and post-quantization loss.
