---
title: "Bigram Subnetworks: Mapping to Next Tokens in Transformer Language Models"
source_url: https://arxiv.org/abs/2504.15471
date: 2025-04
category: paper
authors: Tyler Chang et al.
---

## Key Idea
Identifies that transformer language models contain small subnetworks (less than 0.2% of parameters) dedicated to bigram predictions (predicting the next token based only on the current token). These subnetworks are concentrated in the first transformer MLP layer and overlap significantly with optimal pruning subnetworks, suggesting bigram computation is fundamental to transformer operation.

## Method Details
- Used structured pruning and masking to identify bigram-predicting subnetworks
- Found in fully trained models up to 1B parameters
- Bigram subnetworks are concentrated in the FIRST MLP layer (layer 0)
- These subnetworks overlap significantly with subnetworks found by optimal pruning methods
- Removing bigram subnetworks causes disproportionate performance degradation despite tiny size
- The first MLP layer essentially acts as a bigram lookup table
- Three basis functions characterize transformer weights: bigram, token-interchangeability, and context mappings

## Reported Results
- Bigram subnetworks found in models from small to 1B parameters
- Less than 0.2% of parameters, but critical for performance
- Concentrated in layer 0 MLP
- Significant overlap with pruning-derived important subnetworks

## Relevance to Parameter Golf
Validates and informs the SmearGate/BigramHash approach:
1. The first MLP layer naturally learns bigram statistics -- explicitly providing bigram features (as SmearGate/BigramHash do) gives the model this information "for free"
2. Since Parameter Golf models are tiny, the first MLP layer has limited capacity. Offloading bigram computation to explicit features frees this capacity for higher-order patterns
3. Consider allocating MORE parameters/precision to the first MLP layer since it serves a critical bigram function
4. The token-pair features in use (SmearGate, BigramHash) are well-motivated by this finding

## Implementation Complexity
low

## Expected Impact
- Throughput: no
- Compressed size: no
- Post-quant loss: no
- Raw train loss: yes

## Key Takeaway for Implementation
The first MLP layer is critical for bigram computation. The existing SmearGate/BigramHash approach is well-motivated. Consider giving the first transformer layer higher precision or more parameters, and ensure bigram features are injected BEFORE the first layer to maximize their utility.
