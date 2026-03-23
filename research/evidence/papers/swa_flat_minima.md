---
title: "Averaging Weights Leads to Wider Optima and Better Generalization"
source_url: https://arxiv.org/abs/1803.05407
date: 2018-03
category: paper
authors: Pavel Izmailov, Dmitrii Podoprikhin, Timur Garipov, Dmitry Vetrov, Andrew Gordon Wilson
---

## Key Idea
Stochastic Weight Averaging (SWA) averages multiple SGD iterates along the training trajectory with a cyclical or high constant learning rate, finding solutions near the center of flat regions in the loss landscape. This leads to significantly better generalization than conventional training, with minimal computational overhead.

## Method Details
- **Algorithm:**
  1. Train normally until near convergence
  2. Set learning rate to a high constant or cyclical schedule
  3. Collect model snapshots at regular intervals
  4. Average all collected snapshots (simple arithmetic mean of weights)
  5. Update BatchNorm statistics on the averaged model
- **Why it works:** High/cyclical LR pushes the model to explore broadly within a loss basin. Averaging these exploratory points yields a solution near the basin center, which is necessarily a flat region (since all explored points had similar loss). Flat minima generalize better than sharp ones.
- **Computational cost:** Nearly zero -- just accumulating a running average and one BN update pass
- **Key hyperparameters:** SWA start epoch, LR during SWA phase, snapshot collection frequency

## Reported Results
- Consistent 0.5-1.5% accuracy improvement across CIFAR-10/100, ImageNet
- Finds wider optima (measured by Hessian eigenspectrum) than standard SGD
- Works across architectures: VGG, ResNets, Wide ResNets, DenseNets

## Relevance to Parameter Golf
FOUNDATIONAL for the current approach:
1. SWA is already in use in the competition -- this paper is why
2. The flat minimum found by SWA is more robust to weight perturbation, which includes quantization noise
3. For quantization specifically: a flat minimum means the loss barely changes when weights are perturbed by quantization rounding, directly reducing post-quantization loss
4. Ensure SWA is collecting snapshots during a HIGH learning rate phase (not after decay) for maximum benefit
5. Cyclical LR during collection may work better than constant LR

## Implementation Complexity
low

## Expected Impact
- Throughput: no
- Compressed size: no
- Post-quant loss: yes
- Raw train loss: no

## Key Takeaway for Implementation
Ensure SWA snapshot collection happens during a HIGH or cyclical learning rate phase, not after LR decay. The flat minimum from SWA directly improves quantization robustness. Verify that BatchNorm statistics (or equivalent normalization) are updated on the averaged model.
