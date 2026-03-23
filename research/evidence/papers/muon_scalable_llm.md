---
title: "Muon is Scalable for LLM Training"
source_url: https://arxiv.org/abs/2502.16982
date: 2025-02
category: paper
authors: Jingyuan Liu, Jianlin Su et al. (Moonshot AI / Kimi)
---

## Key Idea
Demonstrates that Muon scales effectively to large language models (3B/16B MoE trained on 5.7T tokens), achieving approximately 2x computational efficiency compared to AdamW at compute-optimal training. Two crucial techniques for scaling are identified: adding weight decay and carefully adjusting per-parameter update scale.

## Method Details
- Scaling law experiments across multiple model sizes
- Two key modifications for scaling Muon to large models:
  1. **Weight decay:** Unlike the original Muon which worked without weight decay, large-scale training requires it for stability
  2. **Per-parameter update scale:** Must be carefully tuned per layer when scaling up; wrong scaling leads to training instability
- These modifications allow Muon to work out-of-the-box at large scale without hyperparameter tuning
- Trained "Moonlight" -- a 3B/16B-parameter MoE model on 5.7T tokens
- Used by Kimi.ai for frontier lab-scale training

## Reported Results
- ~2x computational efficiency vs AdamW with compute-optimal training
- Training runs costing $500K with AdamW cost ~$260K with Muon
- Stable training at 3B and 16B parameter scales
- Moonlight model competitive with comparable-scale models trained with AdamW

## Relevance to Parameter Golf
Confirms Muon's effectiveness and provides practical guidance for tuning:
1. Weight decay is necessary even at small scale if training is intensive
2. Per-parameter update scale is the KEY hyperparameter to tune -- getting it right eliminates need for other HP tuning
3. The 2x efficiency claim means Muon effectively doubles the training compute budget within the fixed 10-minute window
4. The paper provides specific scaling recipes that can be adapted to Parameter Golf's model sizes

## Implementation Complexity
medium

## Expected Impact
- Throughput: yes
- Compressed size: no
- Post-quant loss: no
- Raw train loss: yes

## Key Takeaway for Implementation
Add weight decay to Muon and carefully tune per-parameter update scale for each layer type. This can effectively double training compute efficiency, meaning our 10-minute window yields the equivalent of 20 minutes of AdamW training.
