---
title: "Straight-Through Estimator Improvements: FOGZO and Theoretical Equivalences"
source_url: https://openreview.net/forum?id=3j72egd8q1
date: 2024-2025
category: paper (composite)
authors: Schoenbauer et al. (2024, STE equivalence); FOGZO authors (2025)
---

## Key Idea
Recent research shows that (1) most fancy gradient estimators for quantized training are equivalent to STE with adjusted learning rate and initialization, meaning plain STE is sufficient in most cases, and (2) FOGZO (First-Order-Guided Zeroth-Order gradient descent) can reduce STE bias for 1-8% accuracy improvements when it matters, but at higher computational cost.

## Method Details
- **STE Equivalence Result (Schoenbauer et al., 2024):**
  - A large class of custom weight gradient estimators is approximately equivalent to STE
  - After swapping in STE and adjusting weight initialization and LR in SGD, training proceeds similarly
  - For adaptive optimizers like Adam, the same holds WITHOUT modifications
  - Implication: with small base or adaptively normalized LRs, plain STE is sufficient
  - Clipped STE variant is also sufficient; complex surrogates add no value

- **FOGZO (2025):**
  - Combines first-order and zeroth-order gradient information
  - Reduces STE bias while keeping computation tractable
  - 1-8% accuracy improvement on DeiT Tiny/Small, 1-2% on ResNet 18/50
  - Up to 22 perplexity points improvement for LLaMA models
  - Tradeoff: more compute per step for better gradient accuracy

- **Decoupled Temperature (Shah et al., 2024):**
  - Independent control over sampling discreteness and gradient smoothness
  - Optimized bias-variance tradeoff for STE gradients

## Reported Results
- Plain STE is equivalent to fancy alternatives when LR is properly tuned
- FOGZO: 1-8% accuracy gain at extra compute cost
- Decoupled temperature: improved bias-variance tradeoff

## Relevance to Parameter Golf
Reassuring: the current use of plain STE for QAT gradient propagation is theoretically well-justified and does not leave accuracy on the table (given proper LR tuning). Specific implications:
1. Do NOT spend time implementing complex gradient estimators -- plain STE with proper LR is optimal
2. If QAT quality is a bottleneck, consider FOGZO but only if compute budget allows (probably not in 10 min)
3. Focus tuning effort on LR and initialization rather than gradient estimator design
4. Clipped STE (clamping gradients for out-of-range weights) can help stability

## Implementation Complexity
low

## Expected Impact
- Throughput: no
- Compressed size: no
- Post-quant loss: yes (via better LR tuning)
- Raw train loss: no

## Key Takeaway for Implementation
Stick with plain STE for QAT but ensure the learning rate is properly tuned for the quantization bit-width. The theoretical result confirms that complex gradient estimators are unnecessary. Use clipped STE if training shows instability from weights going far outside the quantization range.
