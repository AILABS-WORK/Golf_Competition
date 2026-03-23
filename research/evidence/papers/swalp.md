---
title: "SWALP: Stochastic Weight Averaging in Low-Precision Training"
source_url: https://arxiv.org/abs/1904.11943
date: 2019-04 (ICML 2019)
category: paper
authors: Guandao Yang, Tianyi Zhang, Polina Kirichenko, Junwen Bai, Andrew Gordon Wilson, Christopher De Sa
---

## Key Idea
SWALP averages low-precision SGD iterates with a modified learning rate schedule, matching full-precision SGD performance even with all numbers quantized down to 8 bits (including gradient accumulators). The key insight is that SWA's averaging effect can compensate for the noise introduced by low-precision arithmetic.

## Method Details
- Runs SGD entirely in low precision (8-bit weights, activations, gradients, accumulators)
- Uses a modified cyclical or high constant learning rate schedule
- Averages the low-precision iterates periodically
- Theoretical guarantee: converges arbitrarily close to the optimal solution for quadratic objectives
- In strongly convex settings, converges to an asymptotically smaller noise ball than low-precision SGD alone
- The averaging step effectively denoises the quantization noise accumulated during training

## Reported Results
- 8-bit SWALP matches full-precision SGD baseline on CIFAR-10/100 with PreActivation ResNet-164
- SWALP improvement over SGD-LP (low precision) is LARGER than SWA improvement over full-precision SGD
- Consistent improvements across architectures and datasets

## Relevance to Parameter Golf
Provides theoretical backing for why SWA works well in the quantized training regime used by Parameter Golf. Key implications:
1. Validates the current use of SWA -- it is theoretically justified for low-precision training
2. Suggests training could potentially use lower precision for intermediate computations (not just final weights) to speed up training
3. The averaging schedule matters: a cyclical or high constant learning rate before averaging gives better results than decaying then averaging
4. Higher learning rate during SWA exploration = better final quality after averaging

## Implementation Complexity
low

## Expected Impact
- Throughput: yes (if training precision is lowered)
- Compressed size: no
- Post-quant loss: yes
- Raw train loss: no

## Key Takeaway for Implementation
Use a higher or cyclical learning rate during the SWA collection phase (not a decayed one). The theory shows this gives better denoising of quantization artifacts. Consider whether intermediate training computations can use lower precision (BF16 or even FP8) to speed up training steps.
