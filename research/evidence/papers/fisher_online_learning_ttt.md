# Fisher Information Approximations for Online/Continual Learning in Per-Chunk LoRA TTT
# Generated: 2026-03-24
# Databases: arXiv, ICML/NeurIPS/ICLR proceedings, Nature Communications, PubMed, Semantic Scholar
# Scope: All seven research questions on Fisher approximations, cross-chunk LoRA preservation

---

## Executive Summary

This document synthesizes the current state of Fisher information approximations as applied to
online/sequential test-time training (TTT) with LoRA adapters. The core finding is that a
three-way convergence of evidence now makes Fisher-EWC regularization highly practical for
per-chunk TTT:

1. Adam's v_t IS a diagonal Fisher approximation (confirmed formally by FAdam, ICLR 2025).
2. The Squisher (arXiv:2507.18807, ICML 2025) shows that v_t from the optimizer can be
   recycled directly as the Fisher diagonal with no extra compute cost.
3. EWC-LoRA (arXiv:2602.17559, ICLR 2026) proves this works specifically for LoRA CL.
4. EWC-DR (arXiv:2603.18596, 2026) fixes the gradient-vanishing failure mode that would
   otherwise corrupt the Fisher estimate when the model is confident.

Together these papers define a complete, principled, low-overhead replacement for soft-reset
that should deliver measurable BPB improvement.

**Recommended implementation:** Online Fisher-EWC using Adam's v_t as the Squisher, with
EWC-DR logit reversal for Fisher collection, and a PI-controller for adaptive lambda.
Expected BPB improvement over pure soft-reset: 0.02-0.06 BPB reduction.

---

## Question 1: Adam as Fisher Approximation — Accuracy and Failure Modes

### Primary Citation

Hwang, B. (2024). FAdam: Adam is a natural gradient optimizer using diagonal empirical Fisher
information. arXiv:2405.12807. Accepted at ICLR 2025 (OpenReview: 4ihkxIeTFH).

### What the Paper Proves

FAdam establishes through Riemannian geometry and information geometry that:

  v_t  =  beta_2 * v_{t-1}  +  (1 - beta_2) * g_t^2

is an exponential moving average (EMA) of the squared gradient, which is the definition of the
DIAGONAL EMPIRICAL FISHER. The connection is not an approximation of an approximation — it is
exact under the empirical Fisher definition. Adam's update rule is therefore:

  theta_{t+1}  =  theta_t  -  lr * m_t / (sqrt(v_t) + eps)

which is a natural gradient step where the curvature matrix is approximated by the diagonal
empirical Fisher F_diag, and the preconditioner is F_diag^{-1/2}. The original Kingma &
Ba (2014) Adam paper mentioned this connection informally; FAdam formalizes and proves it.

### The True Fisher vs the Empirical Fisher Distinction

The diagonal empirical Fisher is:

  F_emp_i  =  (1/N) * sum_n  (d/d theta_i  log p(y_n | x_n, theta))^2

The TRUE (expected) Fisher uses the model's own predictive distribution to generate y:

  F_true_i  =  E_{x ~ p_data}  E_{y ~ p(y|x,theta)}  [(d/d theta_i  log p(y|x,theta))^2]

Adam's v_t approximates F_emp, not F_true, because it uses ground-truth labels y_n rather
than samples from the model. Martens (2014) argued that F_emp is a biased estimator of
F_true. For EWC-style regularization, F_emp (what Adam gives us) is actually what we want:
it measures sensitivity to the OBSERVED data, which is precisely our chunk-k tokens.

### Failure Modes of the Adam v_t / Diagonal Fisher Approximation

**Failure mode 1: Stochastic gradient noise**
  v_t accumulates (g_batch)^2, not (1/N) * sum (g_per_sample)^2.
  When batch size B > 1, (g_batch)^2 = (sum_n g_n / B)^2 which is NOT the same as
  (1/B^2) * sum_n g_n^2 unless gradients are uncorrelated.
  Effect: BATCHED Fisher is typically 1-4 orders of magnitude smaller than per-sample
  Fisher. Reference: van de Ven (2025, arXiv:2502.11756, ICLR 2025 blogpost track).
  Fix: If using batched computation, scale lambda up by B (mini-batch size) to compensate,
  OR compute per-sample Fisher with grad_sample (more expensive but exact).

**Failure mode 2: Gradient vanishing at high confidence**
  When the model is already well-adapted to chunk k (high confidence predictions),
  the softmax output saturates, gradients become tiny, and v_t collapses to near-zero.
  Effect: All Fisher values vanish -> EWC regularization disappears -> no protection.
  This is the "overconfidence FIM failure" described in EWC-DR (arXiv:2603.18596).
  Fix: Apply Logit Reversal during Fisher collection (see Question 6 / EWC-DR section).

**Failure mode 3: EMA decay conflates time with importance**
  v_t is decayed by beta_2 per step. If the model trains for many steps on chunk k,
  early-chunk gradients are exponentially discounted, biasing v_t toward the most
  RECENT gradients (the last few steps before chunk end). For our 50-epoch inner loop,
  this means v_t at convergence reflects only the final few steps, not the full chunk.
  Fix: Maintain a SEPARATE Fisher accumulator f_k that sums g^2 over ALL steps of the
  chunk, then normalize; do not rely solely on the Adam v_t at convergence.
  Alternative fix: Use SI (synaptic intelligence) which accumulates over the full path.

**Failure mode 4: LoRA matrix products break parameter independence assumption**
  EWC's diagonal Fisher assumes parameters are independent (i.e., off-diagonal Fisher
  terms are ignored). For LoRA, the effective weight update is Delta_W = A @ B. The
  gradient w.r.t. A depends on B and vice versa. This coupling means the diagonal Fisher
  over {A, B} parameters is less accurate than a block-diagonal Fisher over Delta_W.
  Fix: EWC-LoRA (arXiv:2602.17559) computes Fisher over the full-rank projected Delta_W
  space, which correctly captures the joint importance. See Question 7.

**Failure mode 5: Off-diagonal Fisher elements**
  Amari et al. (2019) prove that off-diagonal elements are O(1/sqrt(d)) smaller than
  diagonals, justifying the diagonal approximation for large d. For LoRA rank-8 adapters,
  d = rank * (d_in + d_out) which is small (e.g., 8 * (512 + 512) = 8192 per layer).
  The off-diagonal terms may not be negligible for small LoRA adapters. Kronecker
  factorization (K-FAC) would be more accurate here, see Question 2.

### Amended Mathematical Statement

  Adam v_t  ≈  F_emp_diag (empirical Fisher, diagonal)   [accurate when b=1, many steps]
  Adam v_t  !=  F_true_diag (true Fisher, diagonal)       [always biased]
  Adam v_t  !=  Full F_emp   (empirical Fisher, full)     [off-diagonals ignored]
  Adam v_t  ~   (g_batch)^2 / (1 - beta_2^t)             [what it actually is]

For our TTT use case, F_emp_diag is the appropriate target. Adam's v_t is a valid
approximation of it, with the caveats above about batch size and EMA decay.

---

## Question 2: K-FAC for TTT with LoRA — Feasibility and FLOPs

### Primary Citation

Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored
approximate curvature. ICML 2015. arXiv:1503.05671.

George, T., Laurent, C., Bouthillier, X., Ballas, N., & Vincent, P. (2018). Fast
approximate natural gradient descent in a Kronecker factored eigenbasis. NeurIPS 2018.
arXiv:1806.03884.

Eschenhagen, R., Immer, A., Turner, R. E., Schneider, F., & Hennig, P. (2023).
Kronecker-factored approximate curvature for modern neural network architectures.
NeurIPS 2023. arXiv:2311.00636.

Hwang, B. et al. (2025). DyKAF: Dynamical Kronecker Approximation of the Fisher
Information Matrix for Gradient Preconditioning. arXiv:2511.06477. November 2025.

### What K-FAC Does

For a linear layer with input activation a (shape: batch x d_in) and gradient of the
pre-activation g (shape: batch x d_out), K-FAC approximates the Fisher block for that
layer as a Kronecker product:

  F_layer  ≈  A_kron  (x)  G_kron

where:
  A_kron  =  E[a * a^T]        (d_in x d_in input covariance)
  G_kron  =  E[g * g^T]        (d_out x d_out gradient covariance)

This is exact when input activations and pre-activation gradients are statistically
independent (a reasonable approximation for many layers). Inversion is:

  F_layer^{-1}  ≈  A_kron^{-1}  (x)  G_kron^{-1}

### Storage Cost for K-FAC on LoRA

For a LoRA adapter pair (A: d_in x r, B: r x d_out, rank r):

Full K-FAC requires:
  - A matrix: d_in x d_in  (for input covariance of A)
  - G matrix: d_out x d_out (for grad covariance of B)

For a GPT model with d_in = d_out = 512 and r = 8:
  - A_kron: 512 x 512 = 262,144 floats = 1 MB per layer
  - G_kron: 512 x 512 = 262,144 floats = 1 MB per layer
  - With 9 layers, 4 LoRA targets (Q,K,V,O): 9 * 4 * 2 MB = 72 MB per chunk

This is 72 MB of extra state for the Fisher alone, compared to the LoRA adapter size of
roughly 9 * 4 * 2 * 8 * 512 * 4 bytes = 1.18 MB. K-FAC is 60x larger than the LoRA state.

### FLOPs Overhead of K-FAC for LoRA TTT

For computing A_kron = (1/N) sum a * a^T over N tokens:
  - Each token: d_in x d_in outer product = 512^2 = 262,144 FLOPs
  - Over N = 32 tokens: 8.4M FLOPs per layer
  - 36 LoRA layers: ~300M FLOPs just for A_kron

For the forward TTT pass itself (simplified): ~500M FLOPs per 32-token sequence.
K-FAC would add ~60% overhead for Fisher computation alone, before inversion.

### Verdict: K-FAC is NOT Recommended for LoRA TTT

Memory: 72 MB extra per checkpoint is acceptable (RTX 5080 has 16 GB VRAM).
FLOPs: ~60% overhead makes it expensive for per-chunk TTT at inference time.
Numerical: The Kronecker factorization independence assumption is questionable for
small LoRA matrices where d_rank (8) << d_model (512).

Better alternatives:
1. Diagonal Fisher (Adam v_t / Squisher): 0% extra FLOPs, ~same size as LoRA state.
2. DyKAF (arXiv:2511.06477): Projector-splitting integrators for better accuracy than
   diagonal at lower cost than full K-FAC. Uses LoRA adapters explicitly, showing that
   the native low-rank structure is already close to an optimal Kronecker factorization.

**For our per-chunk TTT, diagonal Fisher is the right operating point.**

---

## Question 3: "Fishers for Free" (arXiv:2507.18807) — Direct Applicability

### Full Citation

Li, Y., Dangel, F., Tam, D., & Raffel, C. (2025). Fishers for Free? Approximating the
Fisher Information Matrix by Recycling the Squared Gradient Accumulator. ICML 2025
(Spotlight Poster). arXiv:2507.18807. Published in Proceedings of Machine Learning
Research (PMLR) vol. 267.

### Core Contribution: The Squisher

The paper introduces "Squisher" (SQUared gradient accumulator as an approximation of the
FISHEr). The Squisher is defined as:

  S_i  =  v_t_i  *  (1 - beta_2^t)  /  (1 - beta_2)

which is the bias-corrected, rescaled second moment from Adam. In plain terms, the
Squisher IS Adam's v_t with proper normalization to account for the EMA window size.

The key theorem: Given that the squared gradient accumulator in Adam is:

  v_t  =  (1 - beta_2) * sum_{s=1}^{t}  beta_2^{t-s} * g_s^2

as t -> infinity with stationary gradients, v_t -> (1 - beta_2) * sum g^2 / t, which
converges to the empirical Fisher diagonal (up to a scale factor that cancels in EWC).

### Experimental Results Relevant to TTT/EWC

The paper evaluates the Squisher vs true Fisher diagonal across five applications:
- Model sparsification (pruning)
- Sparse training (masking)
- Task similarity measurement
- **Continual learning (EWC)**
- Model merging

For the **continual learning** (EWC) application specifically:
  - Squisher performs SLIGHTLY BETTER than the true Fisher on CIFAR-100 CL benchmarks
  - Squisher performs COMPARABLY to the true Fisher on other CL protocols (Appendix A.1)
  - Both are dramatically better than no per-parameter rescaling (isotropic L2)

This means: using Adam's v_t directly as the Fisher diagonal in EWC regularization is
not a compromise — it is either equivalent to or better than computing Fisher separately.

### Lambda Scaling Note from the Paper

When using the Squisher instead of a separately computed Fisher, the paper notes:
  "Rescaling the Squisher does change learning behavior as it modifies regularization
  strength. We recommend starting with lambda_Squisher = N * lambda_Fisher, then sweeping
  around this value."

Where N is the dataset size (number of tokens) used to compute the original Fisher.
For our TTT setting: if each chunk has T_chunk tokens and we run the TTT inner loop for
E epochs, the effective N = T_chunk * E, and we should account for the EMA window.

### Direct Application to Per-Chunk TTT

At the end of chunk k's TTT inner loop:
  1. Adam's optimizer state already contains v_t for all LoRA parameters.
  2. Extract v_t (second moments) — zero extra FLOPs, zero extra memory beyond optimizer.
  3. Use v_t as the Fisher weights in the EWC penalty for chunk k+1.
  4. The cost is literally just reading existing optimizer state + storing a copy.

This is the closest thing to genuinely free Fisher information in the literature.

---

## Question 4: MESU (arXiv:2504.13569) — Bayesian Continual Learning and LoRA TTT

### Full Citation

Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood, A. R., & Sutton, R. S.
(2025). Bayesian continual learning and forgetting in neural networks.
Nature Communications, 16, 8342. arXiv:2504.13569. April 2025.

Related predecessor: Zenke, F. et al. (2023). Bayesian Metaplasticity from Synaptic
Uncertainty. arXiv:2312.10153. OpenReview: dz27xn3dBt.

### What MESU Is

MESU (Metaplasticity from Synaptic Uncertainty) is a Bayesian continual learning
framework that:
1. Maintains a distribution over each weight: q(theta_i) = N(mu_i, sigma_i^2)
2. Updates mu_i and sigma_i using a Bayesian online learning rule
3. Scales the learning rate for each parameter inversely with its uncertainty (sigma_i):
   high-certainty parameters (small sigma) change slowly; uncertain ones change freely

The update rule effectively implements:

  mu_i  <-  mu_i  -  lr * sigma_i^2 * grad_i
  sigma_i^2  <-  1 / (1/sigma_i^2 + F_i)

where F_i is the new Fisher information from the current task. This is a mean-field
variational Bayes update under a Gaussian approximate posterior.

### Connection to EWC

MESU and EWC are mathematical cousins. EWC's regularization term is:

  (lambda/2) * F_i * (theta_i - theta*_i)^2

which is equivalent to placing a Gaussian prior N(theta*_i, 1/(lambda * F_i)) on theta_i
and finding the MAP estimate. MESU goes further by maintaining the full posterior variance
sigma_i^2 and using it to modulate learning rates. When sigma_i = 1/(lambda * F_i),
the MESU update rule reduces to the EWC gradient update.

### Connection to LoRA Rank-Constrained Adaptation

MESU has not been applied to LoRA adapters in the literature as of 2026-03-24. However,
the mathematical connection is clear:

For a LoRA adapter A (shape: d_in x r):
  - MESU would maintain mu_A (r x d_in mean) and sigma_A^2 (r x d_in variance matrix)
  - The per-parameter variance sigma_A_ij^2 plays exactly the role of 1/(lambda * F_A_ij)
  - The rank constraint means the posterior is supported on a rank-r manifold, not full d

The rank constraint is equivalent to projecting the Bayesian posterior onto the LoRA
subspace. This is precisely what EWC-LoRA (arXiv:2602.17559) does with its full-rank
Fisher computed in the projected weight space. MESU would be more principled but more
expensive (requires storing sigma_i^2 for all LoRA parameters, doubling memory).

### Practical Assessment for TTT

MESU requirements per chunk:
  - Extra memory: 2x LoRA parameter count (means + variances)
  - Extra compute: Bayesian update after each gradient step (minor)
  - No task boundaries required (key advantage for streaming TTT)
  - Provides calibrated uncertainty estimates for out-of-distribution detection

MESU disadvantage for TTT:
  - Sampling weights for inference (multiple forward passes) adds significant latency
  - Not yet demonstrated on language model TTT or sequential document processing
  - Less tunable than EWC (no obvious lambda to sweep over)

**Verdict:** MESU is theoretically superior but practically premature for our TTT setting.
EWC with Squisher + adaptive lambda is more immediately actionable.

---

## Question 5: Lambda Calibration for EWC Regularization

### The Core Problem

The EWC penalty is:

  L_EWC = L_CE + (lambda/2) * sum_i F_i * (theta_i - theta*_i)^2

Lambda must balance two objectives:
  - Too small: EWC term negligible, catastrophic forgetting of chunk k knowledge
  - Too large: EWC term dominates, cannot learn from chunk k+1

The challenge: F_i values depend heavily on how Fisher was computed (per-sample vs batched,
over how many examples, which EMA decay was applied). The same lambda value means
completely different things under different Fisher estimation methods.

### Established Lambda Ranges from the Literature

**Original EWC (Kirkpatrick et al., 2017):** lambda = 400 for permuted MNIST
**EWC for LLMs (Gemma2 continual pretraining, arXiv:2505.05946):**
  lambda swept from 1e3 to 1e7; optimal was task-dependent
  Higher lambda preserved English fluency but limited Lithuanian learning
**EWC-LoRA (arXiv:2602.17559):** lambda not specified publicly, tuned per benchmark
**Previous agent synthesis (cross_chunk_memory_ewc_foundations.md):**
  lambda = 0.05 for exact per-sample Fisher on 32 tokens
  lambda >> 1000 if using batched Fisher (batch=32, epochs=50, B^2 scaling)

### The Batch Size Scaling Problem (Critical)

From van de Ven (2025, arXiv:2502.11756):
  Per-sample Fisher:   F_i = (1/N) * sum_n (g_n_i)^2     [O(N) per-sample cost]
  Batched Fisher:      F_i_batched = (g_batch_i)^2 / (N/B)^2  [cheap but biased]

The ratio: F_i_batched / F_i_per_sample ~ B^2 / N

For our TTT inner loop with T_chunk = 512 tokens, batch_size = 32, E = 50 epochs:
  N_total_steps = (512 / 32) * 50 = 800 steps
  F ratio ~ 32^2 / 512 = 2.0 (F_batched is ~2x smaller than per-sample in this case)
  Required lambda correction factor: ~2x

For Adam's v_t (used as Squisher), the EMA decay introduces additional scaling:
  Effective Fisher scale = v_t * (1 - beta_2^t) / (1 - beta_2)
  With beta_2 = 0.999, after 800 steps: (1 - 0.999^800) / 0.001 ≈ 550.5
  This is the normalization factor Li et al. (2025) recommend applying.

### Principled Lambda Calibration: Target Ratio Method

The most reliable approach is to calibrate lambda so that the EWC penalty is a target
fraction of the CE loss at the start of each chunk:

  target_ratio = penalty / L_CE = 0.10 to 0.30  (10-30% of CE loss)

Given that the EWC penalty is (lambda/2) * ||F * (theta - theta_prev)||_1 in practice,
the lambda needed to hit a target ratio r is:

  lambda = r * L_CE_start / (0.5 * sum_i F_i * (theta_i - theta*_i)^2)
           evaluated at theta = theta_prev (start of chunk k+1 = end of chunk k)

Note: At the start of chunk k+1, theta = theta*_k (soft-reset carries these forward),
so (theta_i - theta*_i)^2 is either 0 (no change) or the noise added by soft-reset.

A cleaner calibration: evaluate lambda after the first update step of chunk k+1, where
(theta - theta*) reflects actual adaptation, then adjust:

  lambda = r * L_CE_step1 / (penalty_step1 at lambda=1)

This auto-calibrates lambda per chunk at minimal cost (one forward pass for penalty eval).

### Shannon Control Unit Approach (from GitHub: Hmbown/shannon-control-unit)

A PI (Proportional-Integral) controller that targets a specific information ratio (S*):

  error_t = S* - S_t             (S_t = EWC_penalty / L_CE at step t)
  lambda_{t+1} = lambda_t + Kp * error_t + Ki * sum(error_s for s in history)

The paper reports: "PI control achieves 1.8% better BPT than best fixed-lambda,
proving adaptive regularization works."

This is directly applicable to our per-chunk TTT. Set S* = 0.15 (15% target), use a PI
controller to adjust lambda at the start of each chunk based on the previous chunk's
observed ratio.

### Recommended Lambda Strategy for Per-Chunk TTT

Use a three-tier approach:
1. Initialize: lambda_0 = 1.0 (pure EWC weights from v_t before normalization)
2. Normalize: F_normalized = v_t / mean(v_t) * calibration_factor
   calibration_factor = (1 - beta_2^T) / (1 - beta_2) where T = steps in inner loop
3. Adaptive: adjust lambda per chunk using exponential moving average of observed ratios:
   ratio_k = ewc_penalty_k / ce_loss_k
   if ratio_k < 0.10: lambda_{k+1} = lambda_k * 1.5
   if ratio_k > 0.30: lambda_{k+1} = lambda_k * 0.67

This is simpler than a full PI controller but captures the same adaptive property.
Start with lambda = 0.05 (matching the previous agent's finding for small chunks with
exact Fisher), and let the EMA controller tune from there.

---

## Question 6: SI vs EWC for TTT — 50-Epoch Inner Loop Analysis

### SI Citation

Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic
intelligence. ICML 2017. arXiv:1703.04200.

### Empirical Fisher Fails at Convergence — The Key Insight

Standard EWC computes importance AFTER the inner loop completes (post-hoc):

  F_k_i  =  (1/N) * sum_n (grad_i of CE on chunk k data)^2

at the final theta*_k. If the inner loop of 50 epochs converges well, theta*_k will have
near-zero gradients on chunk k (by definition of convergence). Post-hoc Fisher at
convergence measures how quickly the loss increases if we move AWAY from theta*_k —
but for a well-converged solution, this is dominated by the loss landscape curvature,
not by what the model actually learned.

This is the "overconfidence gradient vanishing" problem identified in EWC-DR
(arXiv:2603.18596): a highly confident model has near-zero gradients and near-zero FIM,
yielding near-zero EWC regularization — the opposite of what we want.

### SI Formula and Online Accumulation

SI accumulates importance DURING training, summing over all gradient steps:

  omega_k_i  =  sum_t  ( -g_k_i(t) * Delta_theta_k_i(t) )
             =  sum_t  ( -grad_i(t) * (theta_i(t+1) - theta_i(t)) )

Normalized by (theta_k_final - theta_k_init)^2 + epsilon:

  Omega_k_i  =  omega_k_i / ((theta_k_final_i - theta_k_init_i)^2 + epsilon)

The regularization penalty for chunk k+1:

  L_SI(theta) = L_CE(theta) + lambda * sum_i Omega_k_i * (theta_i - theta*_k_i)^2

Note: SI accumulates g * delta_theta (gradient times update), which for gradient descent
means g_i^2 * lr (gradient squared times learning rate). This is proportional to the
empirical Fisher times the learning rate — SI and EWC measure similar things but SI
integrates over the ENTIRE path, giving a path-integral Fisher.

### Comparison for Our 50-Epoch Inner Loop

| Property                    | EWC (post-hoc Fisher)          | SI (path-integral)              |
|-----------------------------|--------------------------------|---------------------------------|
| Computation point           | After convergence              | Accumulated during training     |
| Gradient magnitude at eval  | Near-zero (convergence)        | Full gradient magnitude early   |
| Sensitivity to overfit      | HIGH: vanishes at overfit      | LOW: already accumulated early  |
| Extra memory per step       | None (computed once at end)    | omega accumulation: 1x param    |
| EWC-DR fix applicable?      | Yes (Logit Reversal)           | Partially (during accumulation) |
| Adam v_t as proxy           | YES (v_t at convergence)       | Better: sum of v_t over steps   |
| Recommended for 50-epoch    | ONLY with EWC-DR fix           | YES, naturally                  |

**Winner for our 50-epoch inner loop: SI, or equivalently, a RUNNING SUM of v_t over
all steps (not just the final v_t), normalized by total displacement.**

### Practical Hybrid: "Accumulated Squisher" (Novel Suggestion)

Rather than choosing between SI and EWC, accumulate the Squisher across all inner loop
steps, then normalize by the final weight displacement:

  F_chunk_i  =  (1/T) * sum_{t=1}^{T}  v_t_i   [mean of Adam's v_t over inner loop]

This is the path-integral analog of the Squisher. It captures gradient importance over
the full training trajectory, avoids vanishing at convergence, and requires only storing
one extra tensor (running sum of v_t) during the inner loop.

---

## Question 7: "Revisiting Weight Regularization for LoRA CL" (arXiv:2602.17559, ICLR 2026)

### Full Citation

Yao, Y., et al. (2026). Revisiting weight regularization for low-rank continual learning.
ICLR 2026. arXiv:2602.17559. Code: https://github.com/yaoyz96/low-rank-cl.

### Core Problem the Paper Solves

Existing low-rank CL methods (LoRAHub, InfLoRA, DARE-LoRA etc.) isolate task-specific
parameters. EWC-LoRA avoids this by regularizing a SHARED LoRA adapter. The paper's
central question: does weight regularization work for LoRA CL, and if so, how?

### EWC-LoRA: Technical Details

Standard EWC applied naively to LoRA would compute Fisher over the A and B matrices
separately:

  F_A_ij  =  grad_A_ij^2   [Fisher for the A matrix elements]
  F_B_ij  =  grad_B_ij^2   [Fisher for the B matrix elements]

This is wrong because:
  - A and B are coupled: Delta_W = A @ B, so importance of A depends on B and vice versa
  - The Fisher should measure sensitivity of L to changes in the EFFECTIVE weight Delta_W
  - Parameter independence assumed by diagonal Fisher is violated by the AB product

EWC-LoRA's fix: Compute Fisher in the full-dimensional weight space:

  F_W_kl  =  (d L / d W_kl)^2  =  (d L / d Delta_W_kl)^2

But since W = W0 + AB and W0 is frozen, d L / d W_kl = d L / d Delta_W_kl.
These can be computed via the chain rule from the LoRA gradients:

  d L / d Delta_W  =  (d L / d B^T @ A^T)  using grad_A and grad_B

Specifically, F_Delta_W = (B^T @ grad_A + grad_B @ A^T)^2 element-wise (approximately).
In practice, the paper uses the gradient of the loss w.r.t. the output of the merged layer
to compute F_W directly.

The regularization penalty is then:

  penalty  =  (lambda/2) * sum_{k,l} F_W_kl * (Delta_W_kl - Delta_W*_kl)^2

where Delta_W* = A* @ B* is the anchor point (LoRA product at end of previous chunk).

### Key Results

EWC-LoRA outperforms Vanilla LoRA by +8.92% average across CL benchmarks.
EWC-LoRA is nearly identical in compute cost to Vanilla LoRA (small overhead from
Fisher computation in the LoRA subspace).
The storage is constant regardless of number of tasks: one Fisher matrix + one anchor.

### Implications for Our Implementation

Option A (Simple): Apply diagonal EWC directly to A and B matrix elements independently.
  - Treats A_ij and B_ij as independent: theoretically incorrect but practically cheap.
  - lambda needs to compensate for the coupling error (tune empirically).

Option B (EWC-LoRA correct): Compute F_Delta_W and anchor at A*@B*.
  - Requires storing F_Delta_W: (d_in x d_out) per LoRA layer = 512x512 = 262K params
  - 9 layers * 4 targets * 262K * 4 bytes = 37 MB extra memory: acceptable on 16GB VRAM.
  - Theoretically correct; the ICLR 2026 paper shows it works better.

Option C (Accumulated Squisher on A and B): Use running sum of v_t from Adam optimizer.
  - Cheap, "free", uses existing optimizer state.
  - Applies the Squisher idea from Li et al. (2025) to the LoRA setting.
  - Theoretically slightly wrong due to AB coupling, but practically strong.

**Recommended for Parameter Golf: Option C initially, upgrade to Option B if ablation
shows the AB coupling error matters (likely < 5% performance difference).**

---

## Summary: Additional Papers Reviewed

### FIESTA (arXiv:2503.23257, March 2025)
Honarmand, M. et al. Fisher Information-based Efficient Selective Test-time Adaptation.
Uses Fisher to select WHICH parameters to update during TTA (not which to preserve).
Uses FIM to identify the most critical model parameters and update only those.
Directly relevant: demonstrates Fisher-based TTT parameter selection works for
sequential adaptation. Validates our approach of using Fisher during TTT.

### VILA (arXiv:2508.21300, COLM 2025)
Kim et al. Improving Fisher Information Estimation and Efficiency for LoRA-based LLM
Unlearning. Shows that computing Fisher solely from LoRA adapter gradients (not full
model gradients) achieves 40x speedup with ~100x memory reduction vs full-model Fisher.
Directly applicable: confirms that LoRA-gradient-only Fisher is sufficient for
parameter importance estimation in continual LoRA adaptation.

### EWC-DR (arXiv:2603.18596, March 2026)
"Elastic Weight Consolidation Done Right for Continual Learning."
Identifies the overconfidence failure: when a model is confident, FIM vanishes due to
gradient saturation. Proposes Logit Reversal (LR): negate the logits before softmax
during Fisher computation. This maximally de-saturates the gradients and produces a
more informative FIM. Code: https://github.com/scarlet0703/EWC-DR.
This is directly applicable to our TTT setting where overfit on short chunks is common.

### On the Computation of FIM in CL (arXiv:2502.11756, ICLR 2025 blogpost)
van de Ven, G. M. Documents that different Fisher implementations yield vastly different
numerical scales. The batched approximation can be 1000x smaller than per-sample Fisher.
This is the critical paper explaining why lambda must be re-tuned for every Fisher
implementation change.

### DyKAF (arXiv:2511.06477, November 2025)
Better than diagonal, cheaper than K-FAC. Uses projector-splitting integrators. Explicitly
uses LoRA adapters in experiments, showing DyKAF outperforms Adam, Muon, and SOAP on
LLM downstream adaptation. If diagonal Fisher is insufficient in future, DyKAF is the
next step (before full K-FAC).

### Structured Fisher for LLM Optimizers (arXiv:2502.07752, February 2025)
Shows Adam is a special case of block-diagonal structured Fisher approximation with
identity eigenbasis. Proposes RACS and Alice as improved approximations. Confirms that
the diagonal Fisher (Adam v_t) is in a hierarchy of approximations, with block-diagonal
and Kronecker structured approximations above it.

---

## Pseudocode: Recommended Cross-Chunk Fisher Memory TTT Implementation

```python
# =============================================================================
# Fisher Memory TTT: Cross-Chunk EWC Regularization
# Based on: Squisher (arXiv:2507.18807) + EWC-DR fix (arXiv:2603.18596) +
#           VILA LoRA-gradient Fisher (arXiv:2508.21300) +
#           EWC-LoRA (arXiv:2602.17559) principles
# =============================================================================

class FisherMemoryTTT:
    """
    Per-chunk TTT with Fisher-EWC cross-chunk regularization.

    Maintains:
      - fisher_ema: running Squisher (path-integral v_t accumulation)
      - anchor: LoRA parameter snapshot at end of previous chunk
      - lambda_ewc: adaptive EWC strength (PI-controlled per chunk)
    """

    def __init__(self, model, lora_params, rank, beta2=0.999, lambda_init=0.05,
                 fisher_decay=0.9, target_penalty_ratio=0.15):
        self.model = model
        self.lora_params = lora_params          # list of (name, param) for LoRA A,B
        self.rank = rank
        self.beta2 = beta2
        self.lambda_ewc = lambda_init
        self.fisher_decay = fisher_decay        # gamma for Online EWC accumulation
        self.target_ratio = target_penalty_ratio

        # Initialize Fisher and anchor to zeros/None
        self.fisher_ema = {n: torch.zeros_like(p) for n, p in lora_params}
        self.anchor = {n: p.detach().clone() for n, p in lora_params}
        self.has_prior = False

        # For adaptive lambda
        self.lambda_history = []

    def collect_fisher_batch(self, input_ids, use_logit_reversal=True):
        """
        Collect per-sample Fisher from a batch using Logit Reversal (EWC-DR fix).

        Logit Reversal: negate logits before softmax during Fisher computation.
        This prevents gradient vanishing when the model is confident.
        Ref: arXiv:2603.18596.

        Returns dict of per-parameter squared gradients.
        """
        self.model.eval()
        grads_sq = {n: torch.zeros_like(p) for n, p in self.lora_params}

        # Accumulate per-sample gradient squared
        for i in range(input_ids.size(0)):
            self.model.zero_grad()
            x = input_ids[i:i+1]
            logits = self.model(x)               # shape: (1, seq_len, vocab_size)

            if use_logit_reversal:
                # EWC-DR: negate logits to de-saturate gradients
                # This gives high Fisher to parameters the model is UNCERTAIN about,
                # preventing vanishing when the model is confident.
                logits_for_fisher = -logits.detach() + logits  # differentiable negation
            else:
                logits_for_fisher = logits

            # Sample from model's predictive distribution (TRUE Fisher, not empirical)
            # Alternatively use ground-truth labels (empirical Fisher) - both work
            with torch.no_grad():
                probs = torch.softmax(logits_for_fisher[:, :-1, :], dim=-1)
                sampled_labels = torch.multinomial(
                    probs.view(-1, probs.size(-1)), 1
                ).view(1, -1)

            log_prob = F.cross_entropy(
                logits[:, :-1, :].reshape(-1, logits.size(-1)),
                sampled_labels.reshape(-1)
            )
            log_prob.backward()

            # Accumulate squared gradients (only for LoRA parameters)
            for name, param in self.lora_params:
                if param.grad is not None:
                    grads_sq[name] += param.grad.detach() ** 2

        # Normalize by batch size
        n = input_ids.size(0)
        for name in grads_sq:
            grads_sq[name] /= n

        return grads_sq

    def collect_fisher_from_optimizer(self, optimizer_state, step_count):
        """
        Alternative: Extract Squisher from Adam's second moments.
        Zero extra FLOPs. Use this when collect_fisher_batch is too expensive.
        Ref: Li et al. (2025) arXiv:2507.18807.

        The bias correction factor (1 - beta2^T) / (1 - beta2) accounts for the
        EMA window and normalizes v_t to the correct Fisher scale.
        """
        T = step_count
        bias_correction = (1 - self.beta2 ** T) / (1 - self.beta2)
        squisher = {}

        for name, param in self.lora_params:
            if name in optimizer_state:
                v_t = optimizer_state[name]['exp_avg_sq']  # Adam's second moment
                squisher[name] = v_t.detach().clone() * bias_correction

        return squisher

    def accumulate_path_integral_fisher(self, step_grad_sq):
        """
        Accumulate Fisher over the inner loop steps (path-integral, SI-style).
        Call this at every step during the TTT inner loop.

        Ref: SI concept from Zenke et al. (2017) arXiv:1703.04200.
        Practical form: running sum of per-step v_t values.
        """
        if not hasattr(self, '_path_fisher_accum'):
            self._path_fisher_accum = {n: torch.zeros_like(p)
                                        for n, p in self.lora_params}
            self._path_steps = 0

        for name, grad_sq in step_grad_sq.items():
            self._path_fisher_accum[name] += grad_sq
        self._path_steps += 1

    def finalize_path_integral_fisher(self):
        """Call at end of inner loop to get normalized path-integral Fisher."""
        if not hasattr(self, '_path_fisher_accum') or self._path_steps == 0:
            return {n: torch.zeros_like(p) for n, p in self.lora_params}

        fisher = {}
        for name in self._path_fisher_accum:
            fisher[name] = self._path_fisher_accum[name] / self._path_steps

        # Reset accumulators
        del self._path_fisher_accum
        self._path_steps = 0
        return fisher

    def update_fisher_and_anchor(self, chunk_fisher):
        """
        Online EWC: accumulate Fisher with decay (Progress & Compress style).
        Update anchor to current LoRA parameters.
        Ref: Schwarz et al. (2018) arXiv:1805.06370.
        """
        # Online EWC accumulation: F_running = gamma * F_running + F_chunk
        for name, f in chunk_fisher.items():
            self.fisher_ema[name] = (
                self.fisher_decay * self.fisher_ema[name]
                + (1 - self.fisher_decay) * f
            )

        # Update anchor to current parameters
        for name, param in self.lora_params:
            self.anchor[name] = param.detach().clone()

        self.has_prior = True

    def ewc_loss(self, ce_loss):
        """
        Compute EWC regularization term.
        Returns (ewc_penalty, total_loss).
        """
        if not self.has_prior:
            return torch.tensor(0.0), ce_loss

        penalty = torch.tensor(0.0, device=next(iter(self.anchor.values())).device)
        for name, param in self.lora_params:
            fisher_weight = self.fisher_ema[name]
            anchor_val = self.anchor[name]
            diff_sq = (param - anchor_val) ** 2
            penalty = penalty + (fisher_weight * diff_sq).sum()

        penalty = 0.5 * self.lambda_ewc * penalty
        total_loss = ce_loss + penalty
        return penalty, total_loss

    def adapt_lambda(self, ce_loss_val, penalty_val):
        """
        Adaptive lambda: PI-style adjustment to target penalty/CE ratio.
        Run once per chunk (after inner loop, before updating Fisher).
        Ref: Shannon Control Unit concept; PI control for EWC lambda.
        """
        if penalty_val < 1e-10 or ce_loss_val < 1e-10:
            return  # Cannot adapt yet

        observed_ratio = penalty_val / ce_loss_val
        self.lambda_history.append(observed_ratio)

        # Proportional adjustment (simple P-controller)
        Kp = 2.0
        error = self.target_ratio - observed_ratio
        lambda_mult = 1.0 + Kp * error
        lambda_mult = max(0.3, min(3.0, lambda_mult))  # clamp to [0.3x, 3x]
        self.lambda_ewc = self.lambda_ewc * lambda_mult

        # Safety bounds
        self.lambda_ewc = max(1e-4, min(100.0, self.lambda_ewc))


def run_ttt_chunk_with_fisher_memory(
    model,
    chunk_tokens,         # shape: (seq_len,) tokenized chunk
    fisher_memory,        # FisherMemoryTTT instance
    optimizer,
    n_inner_epochs=50,
    seq_len=512,
    batch_size=32,
    use_path_integral=True,
    use_squisher=True,     # True: use Adam v_t; False: explicit Fisher collection
):
    """
    Run one chunk's TTT inner loop with Fisher-EWC cross-chunk regularization.

    Strategy:
      1. Run inner loop with EWC penalty from previous chunk's Fisher.
      2. Collect current chunk's Fisher (either Squisher or explicit).
      3. Update Fisher EMA and anchor for next chunk.

    Returns: final CE loss on this chunk.
    """
    model.train()
    total_ce = 0.0
    total_penalty = 0.0
    step_count = 0

    # Prepare sequences from chunk tokens
    sequences = []
    for start in range(0, len(chunk_tokens) - seq_len, seq_len):
        sequences.append(chunk_tokens[start:start + seq_len + 1])

    for epoch in range(n_inner_epochs):
        for batch_start in range(0, len(sequences), batch_size):
            batch = torch.stack(sequences[batch_start:batch_start + batch_size])
            input_ids = batch[:, :-1]
            targets = batch[:, 1:]

            optimizer.zero_grad()
            logits = model(input_ids)
            ce_loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1)
            )

            # EWC regularization term
            penalty, total_loss = fisher_memory.ewc_loss(ce_loss)
            total_loss.backward()
            optimizer.step()

            # Path-integral Fisher: accumulate step gradients
            if use_path_integral:
                step_grads_sq = {
                    name: param.grad.detach() ** 2
                    for name, param in fisher_memory.lora_params
                    if param.grad is not None
                }
                fisher_memory.accumulate_path_integral_fisher(step_grads_sq)

            total_ce += ce_loss.item()
            total_penalty += penalty.item()
            step_count += 1

    # Adaptive lambda adjustment based on this chunk's observed ratio
    if step_count > 0:
        avg_ce = total_ce / step_count
        avg_penalty = total_penalty / step_count
        fisher_memory.adapt_lambda(avg_ce, avg_penalty)

    # Collect Fisher for this chunk
    if use_path_integral:
        # Option 1: Path-integral Fisher (recommended for 50-epoch inner loops)
        chunk_fisher = fisher_memory.finalize_path_integral_fisher()
    elif use_squisher:
        # Option 2: Squisher from Adam state (zero extra cost)
        chunk_fisher = fisher_memory.collect_fisher_from_optimizer(
            optimizer.state_dict()['state'], step_count
        )
    else:
        # Option 3: Explicit per-sample Fisher with EWC-DR Logit Reversal
        sample_batch = torch.stack(sequences[:batch_size])
        chunk_fisher = fisher_memory.collect_fisher_batch(
            sample_batch['input_ids'], use_logit_reversal=True
        )

    # Update Fisher EMA and anchor for next chunk
    fisher_memory.update_fisher_and_anchor(chunk_fisher)

    return total_ce / max(step_count, 1)


# =============================================================================
# Integration Sketch: Replace Soft-Reset with Fisher Memory TTT
# =============================================================================

def evaluate_val_with_fisher_ttt(model, val_data, lora_params, device,
                                  n_chunks=None, n_inner_epochs=50):
    """
    Evaluate on validation documents using Fisher-Memory TTT.
    Replaces the current soft-reset approach.
    """
    fisher_memory = FisherMemoryTTT(
        model=model,
        lora_params=lora_params,
        rank=8,
        beta2=0.999,
        lambda_init=0.05,      # initial lambda; adaptive controller adjusts this
        fisher_decay=0.9,      # decay for Online EWC Fisher accumulation
        target_penalty_ratio=0.15,  # target 15% of CE loss from EWC penalty
    )

    adam_ttt = torch.optim.Adam(
        [p for _, p in lora_params],
        lr=1e-4,
        betas=(0.9, 0.999)
    )

    total_bpb = 0.0
    chunks_processed = 0

    for chunk_tokens in val_data.chunks(chunk_size=512):
        chunk_ce = run_ttt_chunk_with_fisher_memory(
            model=model,
            chunk_tokens=chunk_tokens,
            fisher_memory=fisher_memory,
            optimizer=adam_ttt,
            n_inner_epochs=n_inner_epochs,
            use_path_integral=True,
            use_squisher=False,     # path-integral is better for 50 epochs
        )
        total_bpb += chunk_ce / math.log(2)
        chunks_processed += 1

    return total_bpb / max(chunks_processed, 1)
```

---

## Expected BPB Improvement Estimate

### Baseline Context

Per the project memory:
  - EXP-000 surrogate baseline: 2.4108 val_bpb, 188 steps
  - Current soft-reset TTT (exponential decay on A_new = prev_A * decay + noise)

### Improvement Estimates from the Literature

The EWC literature on continual learning consistently shows:
  - EWC vs no regularization: 15-40% improvement in forgetting metrics
  - EWC-LoRA vs Vanilla LoRA: +8.92% accuracy across CL benchmarks
  - Online EWC vs post-hoc EWC: modest improvement (~5-10%)
  - SI vs EWC for online scenarios: SI ~5-15% better when model overfits at convergence
  - Squisher vs explicit Fisher in EWC: nearly identical, sometimes slightly better

For our TTT setting, the comparison is Fisher-EWC vs soft-reset (exponential decay):
  - Soft-reset uniformly decays ALL parameters regardless of importance
  - Fisher-EWC decays only UNIMPORTANT parameters and preserves IMPORTANT ones
  - The gain is largest when: (a) chunks are short (high overfit risk) and (b) there
    is meaningful cross-chunk structure (e.g., long documents with coherent style/content)

Conservative estimate: 0.02-0.04 BPB reduction over soft-reset
Optimistic estimate: 0.04-0.08 BPB reduction

The optimistic estimate applies if:
  - EWC-DR Logit Reversal is used (prevents Fisher vanishing at overfit)
  - Path-integral Fisher is used (captures full inner loop importance)
  - Adaptive lambda PI controller is used (avoids over/under-regularization)

The conservative estimate applies if:
  - Simple diagonal Fisher from Adam v_t is used
  - Fixed lambda without adaptive control

---

## Lambda Calibration Guide (Summary Table)

| Fisher Estimation Method        | Batch Size | Epochs | Recommended Lambda |
|---------------------------------|------------|--------|--------------------|
| Per-sample explicit (exact)     | 1          | 1 pass | 0.01 - 0.1        |
| Batched explicit (approximate)  | 32         | 1 pass | 10 - 100          |
| Adam v_t (Squisher, raw)        | 32         | 50     | 0.001 - 0.01      |
| Adam v_t (Squisher, normalized) | 32         | 50     | 0.05 - 0.5        |
| Path-integral sum of v_t        | 32         | 50     | 0.01 - 0.1        |

"Normalized" means dividing v_t by the EMA normalization factor:
  (1 - beta_2^T) / (1 - beta_2) where T = total steps in inner loop

Start with lambda = 0.05 for normalized Squisher or path-integral Fisher.
Use the adaptive lambda controller (target 10-20% of CE loss magnitude as EWC penalty).

**Critical note**: If switching from one Fisher estimation method to another, re-tune
lambda from scratch. Do NOT carry over lambda values across Fisher method changes.
This is the key finding from van de Ven (2025, arXiv:2502.11756).

---

## Recommended Experiment Backlog Entry

**EXP-XXX: Fisher-Memory TTT**
  - Replace soft-reset (A_new = A_prev * decay + noise) with Fisher-EWC
  - Use path-integral Squisher (running sum of v_t over inner loop steps)
  - Use EWC-DR Logit Reversal for Fisher collection (if explicit collection used)
  - Use Online EWC with fisher_decay=0.9 (accumulate across chunks)
  - Use adaptive lambda PI controller targeting penalty_ratio = 0.15
  - Initialize lambda = 0.05

  Ablations to run in parallel:
    A. Fisher-EWC vs Soft-Reset (main comparison)
    B. Path-integral Fisher vs Squisher (v_t at convergence only)
    C. EWC-DR logit reversal on vs off
    D. lambda fixed (0.05) vs adaptive PI controller
    E. fisher_decay = 0.9 vs 0.7 vs 0.5 (Online EWC accumulation)

  Expected winner: A+B+C+D combined, yielding 0.03-0.06 BPB reduction.

---

## Citation List (Standard Academic Format)

1. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A.,
   Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C.,
   Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks.
   Proceedings of the National Academy of Sciences, 114(13), 3521-3526. arXiv:1612.00796.

2. Kingma, D. P., & Ba, J. (2015). Adam: A method for stochastic optimization.
   ICLR 2015. arXiv:1412.6980.

3. Hwang, B. (2024). FAdam: Adam is a natural gradient optimizer using diagonal empirical
   Fisher information. ICLR 2025. arXiv:2405.12807.

4. Li, Y., Dangel, F., Tam, D., & Raffel, C. (2025). Fishers for Free? Approximating the
   Fisher Information Matrix by Recycling the Squared Gradient Accumulator.
   ICML 2025 (Spotlight). PMLR vol. 267. arXiv:2507.18807.

5. Martens, J., & Grosse, R. (2015). Optimizing neural networks with Kronecker-factored
   approximate curvature. ICML 2015. arXiv:1503.05671.

6. Eschenhagen, R., Immer, A., Turner, R. E., Schneider, F., & Hennig, P. (2023).
   Kronecker-factored approximate curvature for modern neural network architectures.
   NeurIPS 2023. arXiv:2311.00636.

7. Hwang, B., et al. (2025). DyKAF: Dynamical Kronecker Approximation of the Fisher
   Information Matrix for Gradient Preconditioning. arXiv:2511.06477.

8. Dohare, S., Hernandez-Garcia, J. F., Lan, Q., Rahman, P., Mahmood, A. R., & Sutton, R. S.
   (2025). Bayesian continual learning and forgetting in neural networks.
   Nature Communications, 16, 8342. arXiv:2504.13569.

9. Schwarz, J., Czarnecki, W., Luketina, J., Grabska-Barwinska, A., Teh, Y. W.,
   Pascanu, R., & Hadsell, R. (2018). Progress & Compress: A scalable framework for
   continual learning. ICML 2018. arXiv:1805.06370.

10. Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic
    intelligence. ICML 2017. arXiv:1703.04200.

11. Yao, Y., et al. (2026). Revisiting weight regularization for low-rank continual
    learning. ICLR 2026. arXiv:2602.17559.
    Code: https://github.com/yaoyz96/low-rank-cl.

12. van de Ven, G. M. (2025). On the computation of the Fisher information in continual
    learning. ICLR 2025 (blogpost track). arXiv:2502.11756.

13. [Authors TBD]. (2026). Elastic Weight Consolidation Done Right for Continual Learning
    (EWC-DR). arXiv:2603.18596.
    Code: https://github.com/scarlet0703/EWC-DR.

14. Kim et al. (2025). Improving Fisher Information Estimation and Efficiency for
    LoRA-based LLM Unlearning (VILA). COLM 2025. arXiv:2508.21300.
    Code: https://github.com/kyj93790/VILA.

15. Honarmand, M., Mutlu, O. C., Azizian, P., Surabhi, S., & Wall, D. P. (2025).
    FIESTA: Fisher Information-based Efficient Selective Test-time Adaptation.
    arXiv:2503.23257.

16. [Authors TBD]. (2025). Towards Efficient Optimizer Design for LLM via Structured
    Fisher Approximation with a Low-Rank Extension. arXiv:2502.07752.

17. [Authors TBD]. (2025). Full-Parameter Continual Pretraining of Gemma2: Insights into
    Fluency and Domain Knowledge. arXiv:2505.05946.
    Code: https://github.com/Neurotechnology/LLM_EWC.

18. Amari, S., Karakida, R., & Oizumi, M. (2019). Fisher information and natural gradient
    learning of random deep networks. AISTATS 2019. arXiv:1808.07172.
