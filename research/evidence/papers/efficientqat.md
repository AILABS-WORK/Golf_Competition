---
title: "EfficientQAT: Efficient Quantization-Aware Training for Large Language Models"
source_url: https://arxiv.org/abs/2407.11062
date: 2024-07 (ACL 2025 Main)
category: paper
authors: Mengzhao Chen, Wenqi Shao, Peng Xu, Jiahao Wang, Peng Gao, Kaipeng Zhang, Ping Luo (OpenGVLab)
---

## Key Idea
EfficientQAT proposes a two-phase quantization-aware training method for LLMs that achieves SOTA low-bit quantization quality while being computationally efficient enough to quantize models up to 70B on a single A100 GPU. The method splits training into Block-wise training of All Parameters (Block-AP) followed by End-to-End training of Quantization Parameters (E2E-QP).

## Method Details
- **Phase 1 (Block-AP):** Trains all parameters block-by-block, similar to GPTQ-style block reconstruction but with full gradient-based optimization. This initializes quantization parameters well.
- **Phase 2 (E2E-QP):** Freezes the main weights and only trains quantization parameters (scales, zero-points) end-to-end through the full model. This is much cheaper than full QAT.
- Uses the Straight-Through Estimator (STE) for gradient propagation through quantization nodes.
- Also explores a **Scaling Law for QAT** showing how quantization error relates to model size, training data volume, and quantization group size.

## Reported Results
- 2-bit Llama-2-70B achieves 69.48 accuracy vs. 72.41 full precision (less than 3 points degradation)
- Trained on a single A100-80GB GPU in 41 hours for the 70B model
- Outperforms previous quantization methods across base LLMs, instruction-tuned LLMs, and multimodal LLMs at 7B-70B scales

## Relevance to Parameter Golf
Directly relevant -- Parameter Golf uses QAT with STE. The Block-AP initialization strategy could improve quantization quality at the start of training. The E2E-QP phase concept (training only quantization parameters) could be used as a fine-tuning step after main training to squeeze out additional quality within the 10-minute budget. The scaling law insights help predict optimal model size for a given bit-width and training budget.

## Implementation Complexity
medium

## Expected Impact
- Throughput: no
- Compressed size: no
- Post-quant loss: yes
- Raw train loss: yes

## Key Takeaway for Implementation
Consider a two-phase training approach: first train all parameters with QAT block-by-block for initialization, then switch to training only quantization scales/zero-points end-to-end. This can improve quantization quality without adding much compute overhead.
