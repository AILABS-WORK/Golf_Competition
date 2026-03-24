# Deep Dive: AttnRes, MASA, ExoFormer, TTT-LoRA, Gradient-Aligned Init, Hierarchical TTT
# Generated: 2026-03-24
# Sources: arXiv, GitHub, OpenReview, NeurIPS 2024/2025, ICLR 2025/2026

---

## PAPER 1: Attention Residuals (arXiv:2603.15031)
**Authors:** Kimi Team, MoonshotAI
**Published:** March 17, 2026
**GitHub:** https://github.com/MoonshotAI/Attention-Residuals
**Third-party impl:** https://github.com/kyegomez/attn_res

### Problem Solved
Standard residuals with PreNorm accumulate layer outputs with fixed unit weights. This causes
uncontrolled hidden-state magnitude growth with depth — each layer's contribution is progressively
diluted. Deeper layers have proportionally smaller influence.

### Core Mechanism: Full AttnRes

Replace the standard `h_l = h_{l-1} + F_l(h_{l-1})` with softmax attention over ALL preceding
layer outputs:

```
h_l = sum_{i=0}^{l-1}  alpha_{i->l} * v_i
```

where:
- `v_i` are the layer output hidden states (the "values")
- `alpha_{i->l}` are softmax-normalized attention weights computed as:

```
alpha_{i->l} = softmax_i( w_l^T * RMSNorm_no_weight(v_i) )
```

- `w_l in R^d` is a learned pseudo-query vector, ONE per layer
- Keys are RMS-normalized layer outputs (no learnable scale in the norm)
- `w_l` initializes to ZERO — this makes initial softmax weights uniform (equivalent to mean pooling), providing a stable warm-start

**Parameter overhead:** 1 vector of size d per layer = L*d total additional parameters.
For d=512, L=16: 8192 extra floats = negligible fraction of total model.

### Practical Variant: Block AttnRes

Full AttnRes requires storing ALL layer outputs simultaneously — O(L*B*T*d) memory, which is
prohibitive for large models. Block AttnRes reduces this to O(N*B*T*d) where N << L:

**Algorithm:**
1. Partition L layers into N blocks (default: N=8 blocks, so each block = L/N layers)
2. Within a block, accumulate outputs via standard addition (running sum = `partial_block`)
3. When a block boundary is crossed: append `partial_block` to `blocks` list, reset partial
4. At each layer, attention is over: `completed_blocks + [current_partial_block]`

**Memory:** O(N*d) for block summaries instead of O(L*d) for all layers.
**Quality:** N=8 blocks recovers most of full AttnRes benefits.

### Exact PyTorch Pseudocode

```python
class AttnResOperator(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        self.pseudo_query = nn.Parameter(torch.zeros(d_model))  # w_l
        self.key_norm = RMSNormNoWeight(eps=eps)               # no learnable scale

    def forward(self, sources):  # sources: [N_src, B, T, d]
        K = self.key_norm(sources)
        logits = einsum("d, n b t d -> n b t", self.pseudo_query, K)
        weights = softmax(logits, dim=0)                       # softmax over source dim
        return einsum("n b t, n b t d -> b t d", weights, sources)

# Block AttnRes forward pass (inserted BEFORE each attention + MLP pair)
blocks = [embedding_output]  # b_0 = token embeddings
partial_block = None

for i, layer in enumerate(self.layers):
    source_list = blocks + ([partial_block] if partial_block is not None else [])
    sources = torch.stack(source_list, dim=0)        # [N_src, B, T, d]
    attn_res_output = attn_res_operator[i](sources)  # per-layer operator
    raw_output = layer(attn_res_output)               # normal attention+MLP

    if i > 0 and i % block_size == 0 and partial_block is not None:
        blocks.append(partial_block)
        partial_block = None
    partial_block = raw_output if partial_block is None else partial_block + raw_output
```

### Key Hyperparameters
| Hyperparameter | Default | Notes |
|---|---|---|
| `attn_res_mode` | BLOCK | NONE / FULL / BLOCK |
| `n_blocks` (N) | 8 | Block summaries kept in memory |
| `d_model` | model-specific | Pseudo-query dimension |
| `w_l init` | zeros | Gives uniform weights at init |
| RMSNorm learnable scale | False | Key stability: no weight in norm |

### Integration with Kimi Linear (production validation)
- 48B total / 3B activated MoE model
- Trained on 1.4T tokens
- MMLU: 73.5 → 74.6
- GPQA-Diamond: 36.9 → 44.4 (+7.5 points)
- HumanEval: 59.1 → 62.2
- C-Eval: 79.6 → 82.5
- Scaling law: achieves baseline loss with 1.25x less compute

### Memory Analysis
- Full AttnRes: O(L * B * T * d) — stores all layer outputs; prohibitive for L=48+
- Block AttnRes (N=8): O(8 * B * T * d) — 8 block summaries; adds 8 tensor copies
- Additional params: L * d (pseudo-query vectors) — ~0.01% of model params for typical sizes

### Implementation Complexity
- Lines of code: ~60-80 lines (core AttnResOperator + block logic)
- Integration: insert one AttnRes call at the start of each TransformerBlock.forward()
- No changes to attention or FFN internals

### Competition Feasibility: 4/5
- Trivially composable with standard transformer
- Block AttnRes is O(N=8) overhead — memory manageable
- Risk: may interact poorly with existing techniques (need ablation)
- Pseudo-query zero-init is crucial — do not deviate
- Works for training from scratch (not just fine-tuning)

### Synergies
- Compatible with MUDDFormer (both do cross-layer aggregation, but at different levels)
- Compatible with LoRA TTT (AttnRes parameters can be frozen; only LoRA adapters updated at test time)
- Compatible with quantization (the w_l vectors are tiny, high precision can be kept)
- Potential conflict with MUDDFormer if both are applied to Q/K/V — risk of redundancy

---

## PAPER 2: Test-Time Training with LoRA — State of the Art (2024-2026)
**Key papers:**
- TTT-NN (arXiv:2305.18466) — Test-Time Training on Nearest Neighbors
- qTTT (arXiv:2512.13898) — Query-only TTT for Long Context
- TTL (arXiv:2505.20633) — Test-Time Learning for LLMs
- LoRA-TTT (arXiv:2502.02069) — LoRA Test-Time Training for VLMs
- SEAL (arXiv:2506.10943) — Self-Adapting Language Models

### 2a. TTT on Nearest Neighbors (TTT-NN)
**Citation:** "Test-Time Training on Nearest Neighbors for Large Language Models" (2023/2024)

**Mechanism:** No LoRA. Full parameter fine-tuning on 50 nearest neighbors retrieved from The Pile.
**Loss:** Standard next-token prediction (NTP) loss.
**Steps:** 1 gradient step per neighbor (50 total = 50 steps).
**Memory:** High — full model gradients during test time.
**Bits-per-byte improvements:**
- GitHub (code): 51% of original (49% reduction)
- Europarl: 68% of original (32% reduction)
- DM-Mathematics: 75% of original (25% reduction)
**Retrieval:** RoBERTa-355M embeddings, Faiss flat L2 index, 210M Pile sequences.
**Competition relevance:** LOW — retrieval infrastructure not applicable.

### 2b. Query-Only TTT (qTTT) — MOST RELEVANT FOR COMPETITION
**Citation:** "Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs"
arXiv:2512.13898, Dec 2025, Meta / Harvard / OpenAI

**EXACT MECHANISM:**
1. Full-context prefill to populate K/V cache (one-time cost)
2. K/V cache FROZEN — no updates to K,V projections
3. At test time, update ONLY query projection matrices {W_Q^l for each layer l}
4. Loss: standard NTP on random k=128 token spans from the context
5. N_TTT = 32 gradient steps
6. Optimizer: AdamW, lr swept in [1e-6, 1e-5], weight_decay=0.01, grad_clip=1.0
7. Precision: bfloat16

**Why query-only?** W_Q does not affect K/V cache computation, so the cache can be reused
without invalidation after each gradient step. This makes 32 steps feasible at inference.

**Performance (Qwen3-4B):**
- LongBench-v2: 27.0% → 39.6% (+12.6 points)
- ZeroScrolls: 18.4% → 32.5% (+14.1 points)
- Beats both "more context" and "thinking tokens" baselines

**Memory:** Single KV cache prefill + gradients only for W_Q matrices. Minimal overhead.
**LoRA variant:** The paper uses full W_Q updates. A LoRA-Q variant (rank r << d) would reduce
memory and potentially allow more steps. This is an unexplored synthesis worth trying.

### 2c. Test-Time Learning (TTL) with LoRA
**Citation:** arXiv:2505.20633 (ICML 2025)

**EXACT MECHANISM:**
- Updates LoRA matrices (A and B) at test time
- Init: A ~ Gaussian, B = 0 (standard LoRA init)
- Loss: INPUT PERPLEXITY MINIMIZATION (not output prediction)
  - `min_{Delta_Theta} P(x; Theta + Delta_Theta)` where P is exp(-NLL/|x|)
  - Only high-perplexity samples selected: weight = lambda * exp[log P(x) - log P_0] where P_0 = e^3, lambda=0.10
- LR: 5e-5 (domain/instruction tasks), 1e-6 (reasoning)
- Batch size: 1
- Performance: +20% on domain adaptation, +6.10% on GSM8K for Llama3-8B-Instruct

**Key distinction from qTTT:** TTL minimizes perplexity on the TEST INPUT itself (self-supervised),
not supervised NTP. The model adapts to the domain of the input before generating output.

### 2d. LoRA-TTT (Vision-Language, for reference)
**Citation:** arXiv:2502.02069 (Feb 2025)

**MECHANISM (VLM-specific, but principles transfer):**
- LoRA rank=16, alpha=12 (OOD) or 2 (fine-grained)
- Applied to last 2 layers only (layers 11-12), Q/K/V/O matrices
- Loss: Marginal Entropy Minimization + Masked Autoencoder MSE
  - `L = 1*L_MEM + 16*L_MAE`
- Single gradient step per test instance
- Key insight: adapt LAST LAYERS ONLY (final 2 of 12) — reduces memory 6x vs all layers

### 2e. SEAL: Self-Adapting Language Models
**Citation:** arXiv:2506.10943

**MECHANISM:**
- Model generates "self-edit instructions" (natural language specs for its own finetuning)
- Inner loop: LoRA fine-tuning (rank 64, alpha 128 for knowledge; rank 128, alpha 16 for few-shot)
- Outer loop: RL optimizes self-edit generation
- Loss: standard causal LM loss
- Not applicable during pretraining; for post-deployment adaptation

### Summary: Most Actionable TTT-LoRA Pattern for Competition

The optimal competition strategy synthesizes qTTT + LoRA:

```
1. During eval on test chunk:
   a. Prefill context into K/V cache
   b. LoRA-Q: Add low-rank adapters to W_Q only (rank r=8 or 16)
   c. Loss: NTP on random 128-token spans within context
   d. 32 gradient steps, lr=1e-5, bfloat16
   e. K/V cache preserved across gradient steps
   f. Reset LoRA adapter weights after each evaluation sequence
```

**Why LoRA-Q instead of full W_Q?**
- W_Q is d x d (for d=256: 65K params). LoRA rank-8 = 4K params = 16x reduction
- Gradient memory: d^2 (full) vs 2*d*r (LoRA) — important for small GPUs
- Allows higher lr without exploding full W_Q

**Feasibility: 3/5 in competition**
- Works only at inference/eval time (no benefit to training loss directly)
- Requires autograd during inference (not just forward pass)
- Memory: need to retain computation graph for W_Q gradients
- Risk: might overfit to test chunk distribution and degrade on diversity

---

## PAPER 3: Gradient-Aligned LoRA Initialization — Full Landscape (2024-2026)

### 3a. PiSSA — Principal Singular Values and Vectors Adaptation
**Citation:** "PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models"
arXiv:2404.02948, NeurIPS 2024 Spotlight
**GitHub:** https://github.com/GraphPKU/PiSSA

**EXACT MECHANISM:**
```
W = U S V^T  (full SVD, economy size)

W_pri = U[:, :r] @ S[:r, :r] @ V[:, :r]^T   (trainable)
W_res = U[:, r:] @ S[r:, r:] @ V[:, r:]^T   (frozen)

A_init = U[:, :r] @ S[:r, :r]^(1/2)          # shape: [m, r]
B_init = S[:r, :r]^(1/2) @ V[:, :r]^T        # shape: [r, n]
```

The frozen residual: `W_frozen = W_res` (kept in FP16/FP32, not updated).
At each step: `W_eff = W_frozen + B @ A`.

**Why better than LoRA?**
- LoRA: A~Gaussian, B=0 -> gradients start near zero (dL/dA ~ X^T * dL/dY * B^T ~ 0)
- PiSSA: A,B from principal components -> gradients immediately flow in high-variance directions
- Faster convergence: lower loss in first 100 steps

**Fast SVD:** Uses randomized SVD (Halko et al.) — takes seconds not minutes.

**Applicability:** Fine-tuning only. Requires pretrained W to have meaningful singular structure.
**Competition relevance:** LOW for pre-training, HIGH if doing post-training adaptation.

### 3b. LoRA-GA — Gradient Approximation
**Citation:** "LoRA-GA: Low-Rank Adaptation with Gradient Approximation"
arXiv:2407.05000, 2024
**URL:** https://arxiv.org/html/2407.05000v2

**EXACT MECHANISM:**
```
# Step 1: compute gradient of loss w.r.t. full weight matrix
G = dL/dW  (single batch forward+backward pass)

# Step 2: SVD of gradient
G = U_G S_G V_G^T

# Step 3: Initialize A and B from gradient singular vectors
A_init = (d_out^(1/4) / gamma) * V_G[1:r]^T     # shape: [r, d_in]
B_init = (d_out^(1/4) / gamma) * U_G[r+1:2r]     # shape: [d_out, r]

# Step 4: Adjust W_0 to ensure A_init @ B_init starts with correct gradient direction
W_init = W_0 - eta * B_init @ A_init
```

Where:
- `gamma` ~ 1.0 (scale hyperparameter)
- `eta = alpha / sqrt(r)` (LoRA scaling)
- Uses rows r+1 to 2r of U (NOT rows 1 to r) for rank stability

**Why not V_G[1:r] and U_G[1:r]?**
Using the TOP singular vectors for both A and B would cause rank collapse. The staggered
selection (V top-r, U rows r+1 to 2r) ensures A and B span different subspaces.

**Initialization cost:** One forward+backward pass on single minibatch. ~1 minute for 7B model.
**Performance:** 2-4x faster convergence, +5.69% GLUE over LoRA, +11.52% GSM8K.
**Applicability:** Fine-tuning only (gradients of pretrained W are meaningful).

### 3c. LoRA-DA — Data-Aware Initialization
**Citation:** "LoRA-DA: Data-Aware Initialization for Low-Rank Adaptation via Asymptotic Analysis"
arXiv:2510.24561, 2025

**EXACT MECHANISM:**
```
# Fisher information approximation via K-FAC:
Z_fisher = (1/|S|) * sum_j z_j @ z_j^T          # input second moments
Y_fisher = (1/|S|) * sum_j (dL/dy_j) @ (dL/dy_j)^T  # output-grad second moments
J^{-1}(W_0) approx Z_fisher^{-1} x [Y_fisher^{-1}]_{diag}  # Kronecker approx

# Bias term via Fisher-gradient:
(W_tgt - W_0) approx -J(W_0)^{-1} @ G           # Fisher-scaled gradient

# Initialization Guidance Matrix:
Omega = J(W_0)^{-1}/N - (W_tgt - W_0)(W_tgt - W_0)^T

# A_init = eigenvectors of Omega for r smallest eigenvalues
```

**Key insight:** Fisher-weighted gradient direction, not raw gradient. Fisher captures
anisotropy — scales directions by their uncertainty/information content.

**Cost:** ~6% of total training time. Requires 256 target-domain samples.
**Applicability:** Fine-tuning only. Requires target domain data and pretrained model.

### 3d. GoRA — Gradient-Driven Adaptive Rank Allocation
**Citation:** "GoRA: Gradient-driven Adaptive Low Rank Adaptation"
arXiv:2502.12171, Feb 2025

**EXACT MECHANISM:**
```
# Step 1: Compute importance per weight matrix
G = accumulated_gradients(W, n_steps=64)
I(W) = avg(|W * G|)         # element-wise product, then average

# Step 2: Normalize importance across all matrices
I_norm(W_i) = I(W_i) / sum_j I(W_j)

# Step 3: Allocate rank proportionally
r_Wi = floor(B0 * I_norm(W_i) / sqrt(m + n))
# constrained: r_min <= r_Wi <= r_max (typically 0.5*r0 to 4*r0)

# Step 4: Initialize A and B
A_0 ~ Kaiming_uniform()                          # standard random
B_0 = -(A_0^T @ A_0)^{-1} @ A_0^T @ G          # pseudo-inverse of gradient
B_0 = xi * B_0  where xi = gamma * sqrt(m) / alpha   # scaling
```

**Why pseudo-inverse for B?** A_0 @ B_0 ≈ -A_0 @ (A_0^T@A_0)^{-1} @ A_0^T @ G which is the
projection of the gradient onto the column space of A_0. This approximates a gradient descent
step aligned to A's subspace.

**Cost:** N=64 gradient accumulation steps for importance estimation (~1-2% overhead).
**Performance:** +5.13 points over standard LoRA on math reasoning (Llama3.1-8B).
**Applicability:** Fine-tuning only.

### 3e. EVA — Explained Variance Adaptation
**Citation:** "Parameter Efficient Fine-tuning via Explained Variance Adaptation"
arXiv:2410.07170, Oct 2024 (ICLR 2025)

**EXACT MECHANISM:**
```
# Pass minibatches through model, collect activation vectors X at each layer
X_i in R^{B x d_in}   (activations at layer i, weight matrix Wi)

# Incremental SVD on activations
X_i = U_i Sigma_i V_i^T    (truncated, incremental)

# Explained variance ratio per singular component
xi_j_i = (sigma_j_i^2) / ((M-1) * ||sigma_i||_1)

# Sort ALL components from ALL layers by xi
# Select top-l within rank budget

# A_init = V_i[:r, :]^T    (top-r right singular vectors of activations)
# B_init = 0
```

**Why activations, not gradients?** Activations capture which INPUT DIRECTIONS the layer
actually processes most. Right singular vectors of X are the input directions of maximum
variance — these are where the layer has most influence.

**Cost:** 0.2% of training time (incremental SVD, fast).
**Applicability:** Fine-tuning only. Requires downstream task data.

### Summary Table: Gradient-Aligned LoRA Init Methods

| Method | Signal Used | Init Cost | Rank Adaptive | Competition Applicability |
|---|---|---|---|---|
| PiSSA | SVD of W itself | seconds | No | Fine-tuning only |
| LoRA-GA | SVD of dL/dW | 1 fwd+bwd pass | No | Fine-tuning only |
| LoRA-DA | Fisher + gradient | 6% of training | No | Fine-tuning only |
| GoRA | Accumulated grads | 64 steps | YES | Fine-tuning only |
| EVA | SVD of activations | 0.2% of training | YES | Fine-tuning only |

**IMPORTANT CONCLUSION FOR COMPETITION:** None of these methods directly improve pre-training
from scratch. They all assume a pretrained model W_0 exists and meaningful task gradients
can be computed. For a competition where we train from scratch, the relevant application is:
- Use PiSSA/LoRA-GA to initialize TTT-LoRA adapters for test-time adaptation
- Or use EVA on training data at start of training to bias LoRA directions

**Speculative synthesis:** At training step 0, compute SVD of the first minibatch's gradient
(LoRA-GA style) and use that to initialize LoRA adapters applied DURING TRAINING. This could
give better gradient flow from the start even for training from scratch. This is NOT validated
in any paper but is mechanistically coherent.

---

## PAPER 4: Hierarchical Test-Time Training — State of the Art

### Finding: No dedicated paper on "hierarchical TTT" exists as of March 2026.

The closest relevant work:

**End-to-End TTT for Long Context** (test-time-training.github.io/e2e.pdf)
- Describes a two-level memory hierarchy:
  - Short-term memory: sliding window attention
  - Long-term memory: weights updated at test time (interpreted as long-term memory)
- This IS a form of hierarchical TTT — fast (window) + slow (weight update) time scales

**qTTT (arXiv:2512.13898)** — query-only updates at test time
- Only W_Q adapted — a form of "layer-partial" adaptation
- Could be extended: different adaptation granularities per layer group

**Multi-Level Framework for Training Acceleration** (arXiv:2404.07999)
- Uses coalescing/de-coalescing operators to build multi-level training
- Not TTT specific but relevant to hierarchical adaptation patterns

### Proposed Hierarchical TTT Architecture (Novel Synthesis)

Based on the gap in literature, a hierarchical TTT approach could work as follows:

```
Layer groups:
  - Lower layers (0 to L/3): Adapt slowly (lr = 1e-6), adapt only W_Q
  - Middle layers (L/3 to 2L/3): Adapt medium (lr = 1e-5), adapt W_Q + W_V
  - Upper layers (2L/3 to L): Adapt fast (lr = 1e-4), adapt all of Q,K,V,O

Rationale:
  - Lower layers encode syntax/structure — stable across contexts
  - Middle layers encode semantics — moderate adaptation helpful
  - Upper layers encode task-specific patterns — fastest adaptation needed
```

**Competition feasibility: 2/5** — no published validation, hard to tune 3 lr groups correctly.

---

## PAPER 5: MASA — Matrix Atom Sharing (arXiv:2508.04581)
**Published:** August 2025 (AAAI 2026, Singapore)

### Core Mathematical Formulation

**Weight reconstruction:**
```
W_hat_l = sum_{s=1}^{S}  c_ls * D_s

where:
  D_s in R^{d x h}      : shared matrix atoms (same shape as projection matrix)
  c_ls in R              : per-layer scalar mixing coefficients
  S = L/3               : number of atoms (1 per 3 layers)
  l = 1..L              : layer index
```

**Dictionaries:** Independent per projection type. MASA-QKVO uses 4 separate dictionaries,
one each for Q, K, V, O. MASA-QKV uses 3 (no O sharing).

### Coefficient Prediction

During training, coefficients are NOT directly learned as free scalars. Instead:
1. Each layer l has a trainable embedding vector `e_l in R^d_emb`
2. A shared 3-layer MLP: `c_l = MLP(e_l)` where c_l in R^S
3. After training completes, the MLP is DISCARDED
4. Only the final coefficient matrix C in R^{L x S} is kept for inference
5. This regularizes training and reduces gradient instability

### Compression Analysis

For projection type (e.g., W_Q) with d=768, h=64, L=12 layers:
```
Original params: L * d * h = 12 * 768 * 64 = 589,824
MASA params:     S * d * h + L * S = (L/3)*d*h + L*(L/3)
               = 4 * 768 * 64 + 12 * 4 = 196,608 + 48 ≈ 196,656
Reduction: ~66.7% (dominated by S = L/3 = 4 atoms vs 12 matrices)
```

**General formula:**
```
r = 1 - S/L  (approximately, when d*h >> L)
With S = L/3: r = 2/3 = 66.7%
```

### Pretrained Model Adaptation (Matrix PCA)

For adapting an existing model:
```
Stack all W_l for projection type into V = [W_1; W_2; ...; W_L]  # [L*d, h]
SVD of V @ V^T -> eigenvectors for S largest eigenvalues = initial atoms D_s
```
This is a closed-form initialization, no gradient needed.

### Performance vs Baselines (100M-700M models, RefinedWeb)
- Outperforms GQA at comparable parameter budgets
- Outperforms Sequential sharing and Repeat-all-over sharing
- 8.3% throughput decrease at inference (from MLP atom computation at init time,
  but MLP discarded — final inference uses pre-computed coefficients * atoms)

### Implementation Complexity
- Lines of code: ~120-150 (dictionary init + coefficient MLP + weight reconstruction)
- Key challenge: weight_reconstruction at every forward pass (W = C @ D, where C is
  the pre-computed coefficients) adds a matmul per projection per layer

**Actual inference computation:**
```python
# At inference (MLP discarded), for layer l, projection type Q:
W_Q_l = sum(c_l[s] * D_Q[s] for s in range(S))  # S matmul-like ops, but D_s are matrices
# OR precompute W_Q_l once at load time and cache — then no overhead!
```
If W_Q_l is precomputed and cached, inference is IDENTICAL to standard transformer.
The 8.3% overhead reported is from dynamic computation — precomputing eliminates this.

### Competition Feasibility: 2/5
- Requires training from scratch with MASA parameterization
- Coefficient MLP and dictionary init are non-trivial to implement correctly
- Not designed for competition's parameter-efficiency focus (reduces params but not necessarily improves val_bpb)
- The compression gain helps parameter count but may hurt expressiveness
- Note: MASA is about REDUCING parameter count — useful if budget is parameters, not if budget is compute

---

## PAPER 6: ExoFormer — Exogenous Attention (arXiv:2601.08131)
**GitHub:** Mentioned as released with code and models

### Core Mechanism

**Problem:** Cross-layer reuse of first-layer attention projections (NuResFormer, Gated Attention)
creates tension: layer 0 must simultaneously be a good reusable anchor AND a good first processor.
This structural conflict limits performance.

**Solution:** Learn anchor projections OUTSIDE the sequential layer stack, from the EMBEDDING
layer H_0 directly.

### Mathematical Formulation

**Exogenous anchor projections (computed ONCE from embeddings, used in ALL layers):**
```
Q_anc = H_0 @ W^Q_anc      # W^Q_anc are independent weights, not shared with any layer
K_anc = H_0 @ W^K_anc
V_anc = H_0 @ W^V_anc
G_anc = H_0 @ W^G_anc      # for gated attention variants
```

H_0 = token embedding (before any transformer layer). This is the "exogenous" source.

**Normalized mixing at each layer n:**
```
S_hat_n = lambda^S_{n,1} (x) RMSNorm(S_anc) + lambda^S_{n,2} (x) S_n

for S in {Q, K, V, G}
```

where:
- `lambda^S_{n,1}` and `lambda^S_{n,2}` are learnable mixing coefficients
- `(x)` denotes element-wise multiplication (with broadcasting for non-elementwise modes)
- `RMSNorm(S_anc)` normalizes the anchor to prevent distributional mismatch
- Init: all lambda = 0.5

**Three granularity modes for lambda:**
```
Scalar (S):      lambda in R^1          — 1 coefficient total per projection
Headwise (H):    lambda in R^{n_heads}  — 1 per attention head
Elementwise (E): lambda in R^{d}        — 1 per channel
```

**Dynamic mixing (context-dependent lambda):**
```
DM_n(H_{n-1}) = sigma(GELU(H_{n-1} @ W^DM_{n,1}) @ W^DM_{n,2} + b^DM_n)
```
Produces 8 scale factors {gamma^Q_{n,1}, gamma^Q_{n,2}, gamma^K_{n,1}, ...} that
modulate the base lambda parameters dynamically per token.

### External Context Buffer
The "buffer" is simply the cached output of the 4 anchor projection matmuls from H_0.
It is:
- Populated once at model initialization (or forward pass start, same as embedding computation)
- Not updated during inference — it is STATIC, purely from the original token embeddings
- Not recurrent or persistent across sequences

This makes ExoFormer SIMPLER than it sounds: it's just 4 extra linear layers applied to
embeddings, whose outputs are mixed into each layer's Q,K,V,G.

### Key Hyperparameters (~450M scale models)
| Hyperparameter | Value |
|---|---|
| Layers | 32 |
| Hidden dim | 1024 |
| Attention heads | 16 |
| d_k | 64 |
| Training tokens | 10B FineWeb-Edu |
| Optimizer | Muon (lr=0.01) + AdamW (lr=0.003) |
| Lambda init | 0.5 for all |
| Dynamic DM | 2-layer MLP, sigmoid activation |

### Critical Ablation Findings
1. RMSNorm on anchor IS ESSENTIAL — without it, distributional mismatch causes instability
2. Gate logit mixing (G) is more stable than Q or K mixing — safer starting point
3. Elementwise granularity outperforms scalar
4. Dynamic variant: achieves same val loss with 1.5x fewer training tokens than Gated Attention

### Performance
- Dynamic ExoFormer: 1.5x downstream accuracy improvement, 1.5x token efficiency vs Gated Attention
- Outperforms NuResFormer (internal anchor) across all variants

### Implementation Complexity
- Lines of code: ~100-150 (4 anchor linear layers + per-layer mixing + optional DM module)
- Integration: modify each TransformerBlock to accept anchor projections and mix them in

### Competition Feasibility: 4/5
- Clean, well-motivated mechanism
- Gate logit mixing (G) is the safest variant to start with
- The 4 anchor projections add ~4 * d^2 parameters (small for d=256: 262K params)
- Works for training from scratch
- Potential synergy with AttnRes (both address cross-layer information flow)
- Potential conflict: if both AttnRes AND ExoFormer used, redundant supervision signal

### Offloading Hypothesis (Key Insight)
"External anchors preserve essential token identity, allowing layers to specialize exclusively
in feature transformation." The anchor injected into each layer maintains a stable reference
to the ORIGINAL TOKEN, preventing the semantic drift that occurs in deep networks.

---

## SYNTHESIS: Actionability for Competition

### Priority Rankings for Implementation

**Priority 1 — AttnRes Block (Paper 1)**
- Feasibility: 4/5
- Expected gain: -0.015 to -0.030 BPB (comparable to MUDDFormer)
- Implementation: 60-80 LOC, drops into any transformer
- Status: Not in any existing competition submission
- Key risk: May conflict with MUDDFormer (test both, pick best)

**Priority 2 — ExoFormer Gate Mixing (Paper 6)**
- Feasibility: 4/5
- Expected gain: -0.010 to -0.020 BPB
- Implementation: 100-150 LOC, 4 extra linear layers from embeddings
- Status: Not in any existing competition submission
- Start with G (gate) mixing only; add Q,K,V if stable

**Priority 3 — LoRA TTT at Eval (Paper 2, qTTT synthesis)**
- Feasibility: 3/5
- Expected gain: uncertain for bits-per-byte on validation set
- Implementation: 80-100 LOC
- Key: Use LoRA-Q (rank 8) not full W_Q update
- Risk: Requires grad computation during eval pass — non-trivial engineering

**Priority 4 — MASA parameter sharing (Paper 5)**
- Feasibility: 2/5
- Expected gain: unclear for BPB (helps parameter count, may hurt expressiveness)
- Only useful if competition scoring penalizes parameter count
- Skip unless parameter budget is the binding constraint

**Priority 5 — Gradient-Aligned LoRA Init (Paper 3)**
- Feasibility: 2/5 for competition (all methods require pretrained W_0)
- Exception: LoRA-GA spectral init on FIRST minibatch gradients at step 0
  could theoretically bootstrap LoRA directions during pre-training
- Not validated; speculative

### Interaction Map

```
AttnRes + ExoFormer:  POTENTIAL CONFLICT (both do cross-layer aggregation)
AttnRes + MUDDFormer: POTENTIAL CONFLICT (MUDDFormer already does this more aggressively)
AttnRes + LoRA TTT:   SAFE (AttnRes is architecture; LoRA TTT is at eval time)
ExoFormer + LoRA TTT: SAFE (same reasoning)
MASA + anything:      REPLACES attention weight matrices — conflicts with all param-sharing techniques
PiSSA/LoRA-GA:        Only useful if doing TTT — apply as TTT-LoRA initializer
```

---

## Citation Block

1. Kimi Team. "Attention Residuals." arXiv:2603.15031 (March 2026).
   https://arxiv.org/abs/2603.15031
   GitHub: https://github.com/MoonshotAI/Attention-Residuals

2. [TTT-NN] Shi et al. "Test-Time Training on Nearest Neighbors for Large Language Models."
   arXiv:2305.18466 (2023/2024).
   https://arxiv.org/abs/2305.18466

3. [qTTT] Akyurek et al. "Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs."
   arXiv:2512.13898 (December 2025).
   https://arxiv.org/abs/2512.13898

4. [TTL] "Test-Time Learning for Large Language Models." arXiv:2505.20633 (ICML 2025).
   https://arxiv.org/abs/2505.20633

5. [LoRA-TTT] "LoRA-TTT: Low-Rank Test-Time Training for Vision-Language Models." arXiv:2502.02069 (2025).
   https://arxiv.org/abs/2502.02069

6. [SEAL] "Self-Adapting Language Models." arXiv:2506.10943 (2025).
   https://arxiv.org/abs/2506.10943

7. [PiSSA] Meng et al. "PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models."
   arXiv:2404.02948, NeurIPS 2024 Spotlight.
   https://arxiv.org/abs/2404.02948

8. [LoRA-GA] Wang et al. "LoRA-GA: Low-Rank Adaptation with Gradient Approximation."
   arXiv:2407.05000 (2024).
   https://arxiv.org/abs/2407.05000

9. [LoRA-DA] "LoRA-DA: Data-Aware Initialization for Low-Rank Adaptation via Asymptotic Analysis."
   arXiv:2510.24561 (2025).
   https://arxiv.org/abs/2510.24561

10. [GoRA] "GoRA: Gradient-driven Adaptive Low Rank Adaptation." arXiv:2502.12171 (February 2025).
    https://arxiv.org/abs/2502.12171

11. [EVA] Paischer et al. "Parameter Efficient Fine-tuning via Explained Variance Adaptation."
    arXiv:2410.07170 (ICLR 2025).
    https://arxiv.org/abs/2410.07170

12. [MASA] "Matrix Atom Sharing in Attention." arXiv:2508.04581 (AAAI 2026).
    https://arxiv.org/abs/2508.04581

13. [ExoFormer] "ExoFormer: Exogenous Attention Projection Mixing." arXiv:2601.08131 (2026).
    https://arxiv.org/abs/2601.08131
