---
title: "Muon: An Optimizer for Hidden Layers in Neural Networks"
source_url: https://kellerjordan.github.io/posts/muon/
date: 2024-10
category: blog/paper
authors: Keller Jordan
---

## Key Idea
Muon (MomentUm Orthogonalized by Newton-Schulz) runs standard SGD with Nesterov momentum, then orthogonalizes each 2D parameter's update by replacing it with the nearest orthogonal matrix via Newton-Schulz iteration. This achieves ~35% training speedup over AdamW on NanoGPT and holds current training speed records for both NanoGPT and CIFAR-10 speedrunning.

## Method Details
- Applies to 2D parameters (weight matrices) in hidden layers only
- Embedding and output layers still use AdamW
- Algorithm:
  1. Compute gradient
  2. Apply Nesterov momentum (standard SGD-momentum step)
  3. Orthogonalize the momentum buffer using 5 iterations of Newton-Schulz
  4. Newton-Schulz iteration: X_{k+1} = a*X_k + b*X_k^3 + c*X_k^5 (with specific coefficients)
  5. Use the orthogonalized matrix as the parameter update
- Newton-Schulz iteration runs stably on tensor cores in BF16
- Only 5 NS iterations needed for convergence in practice
- Per-parameter update scale must be carefully adjusted

## Reported Results
- 1.35x speedup over previous NanoGPT record (Karpathy's original)
- ~35% training speed improvement on NanoGPT speedruns vs AdamW
- Trained 1.5B parameter transformer to GPT-2 XL level on HellaSwag in 10 8xH100-hours
- Continues showing improvements at 774M and 1.5B parameter scales

## Relevance to Parameter Golf
ALREADY IN USE by top competitors. The 35% training speedup is critical when limited to 10 minutes on 8xH100. Key considerations:
1. Only applies to 2D hidden layer weights -- embeddings and output heads need separate optimizer (AdamW)
2. The orthogonalization step has negligible cost (5 NS iterations on tensor cores)
3. Must tune per-parameter update scale and weight decay when scaling
4. The orthogonal updates may interact beneficially with quantization (orthogonal matrices have bounded singular values, reducing quantization sensitivity)

## Implementation Complexity
medium

## Expected Impact
- Throughput: yes (faster convergence per step)
- Compressed size: no
- Post-quant loss: possibly (orthogonal updates may regularize)
- Raw train loss: yes

## Key Takeaway for Implementation
Use Muon for all 2D weight matrices and AdamW for embeddings/output. Ensure 5 Newton-Schulz iterations, BF16 computation, and proper per-parameter update scale tuning. The training speed gain directly translates to more effective training within the 10-minute window.
