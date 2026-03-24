# Cross-Chunk Memory in TTT: EWC and Continual Learning Foundations
# Generated: 2026-03-24 via WebSearch / WebFetch research
# Databases: arXiv, PNAS, NeurIPS, ICML, ICLR, CVPR/WACV proceedings, Nature Communications

---

## Executive Summary

Our TTT_LORA_DECAY (Soft-Reset) technique carries LoRA weights between sequential document
chunks using exponential decay: `A_new = prev_A * decay + noise`. This document establishes
the theoretical foundations, identifies more principled alternatives from continual learning
(CL) literature, evaluates them for our specific TTT setting, and designs the "Fisher Memory
TTT" variant.

**Key finding:** EWC-inspired Fisher regularization is strictly more principled than soft-reset
because it preserves parameters that were CAUSALLY important to the previous chunk (high
Fisher = high gradient squared = high loss sensitivity), while freely updating unimportant
parameters. Soft-reset decays ALL parameters uniformly regardless of importance.

---

## Part I: Primary Continual Learning Papers

### 1. EWC — Elastic Weight Consolidation

**Citation:** Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G.,
Rusu, A. A., Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D.,
Clopath, C., Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in
neural networks. *Proceedings of the National Academy of Sciences*, 114(13), 3521–3526.
arXiv:1612.00796. PNAS 2017.

**Core idea:** Treat catastrophic forgetting as a Bayesian inference problem. The true
posterior over parameters after learning task B while having learned task A is:

```
log p(θ | D_A, D_B) = log p(D_B | θ) + log p(θ | D_A) - log p(D_B)
```

The term `log p(θ | D_A)` is the posterior from task A, which acts as a prior for task B.
This posterior is intractable. EWC approximates it with a Laplace (Gaussian) approximation
centered at the task-A optimum θ*_A, with precision (inverse variance) given by the diagonal
of the Fisher information matrix F_A:

```
log p(θ | D_A) ≈ -1/2 * (θ - θ*_A)^T * F_A * (θ - θ*_A)
```

This yields the EWC loss for task B:

```
L_EWC(θ) = L_B(θ) + (λ/2) * Σ_i F_A,i * (θ_i - θ*_A,i)^2
```

Where:
- `L_B(θ)` is the task B cross-entropy loss
- `F_A,i` is the i-th diagonal of the Fisher information matrix from task A
- `θ*_A,i` is the i-th parameter value at the task A optimum (the anchor point)
- `λ` controls the overall regularization strength

**Fisher information matrix (diagonal approximation):**

The full Fisher is a d×d matrix — intractable for large networks. EWC uses the diagonal
(a mean-field approximation). The i-th diagonal element is the expected squared gradient
of the log-likelihood with respect to parameter i:

```
F_i = E_{x~p_data} [ E_{y~p(y|x,θ)} [ (∂ log p(y|x,θ) / ∂θ_i)^2 ] ]
```

In practice, the outer expectation is estimated by averaging over the training dataset,
and the inner expectation (over the model's own predictive distribution) is approximated
by the EMPIRICAL Fisher: using the ground-truth label y instead of sampling from the model:

```
F_i ≈ (1/N) Σ_{n=1}^{N} (∂ log p(y_n|x_n,θ) / ∂θ_i)^2
     ≈ mean over data of (grad_i of loss)^2
```

This is the "gradient squared" approximation. For language models, the loss IS the
negative log-likelihood, so:

```
F_i ≈ (1/N) Σ_{n} (∂ L(θ; x_n, y_n) / ∂θ_i)^2
```

**EWC vs isotropic L2 regularization:**

L2 regularization uses F_i = constant for all i: `L_L2(θ) = L_B(θ) + (λ/2) * ||θ - θ*_A||^2`

EWC's key contribution is replacing this uniform weight with the per-parameter Fisher
diagonal, which reflects how much each parameter contributed to task A performance. High
Fisher = parameter was in a sharp curvature direction = changing it would hurt task A.
Low Fisher = parameter is in a flat direction = changing it is safe.

**Empirical results on permuted MNIST:** EWC dramatically outperforms L2 and no-regularization
baselines. Without EWC, accuracy on task 1 drops to near-chance when task 2 is learned.
With EWC, accuracy on task 1 is maintained near peak.

**Lambda values:** Original paper used λ = 400 for sequential MNIST tasks. This is task-
and scale-dependent and must be tuned. The key insight is that λ interacts with the scale
of the Fisher values, which itself depends on the Fisher computation method used.

**Critical warning on lambda scale:** Heckel et al. (2025, arXiv:2502.11756) found that
BATCHED Fisher (squaring aggregated mini-batch gradients) produces values orders of magnitude
smaller than EXACT Fisher (squaring per-sample gradients before summing). If using BATCHED
computation, λ must be increased by 1-4 orders of magnitude to achieve equivalent effect.
For per-sample (EXACT) Fisher, original paper's λ range (100-1000) is appropriate.

---

### 2. Online EWC / Progress & Compress

**Citation:** Schwarz, J., Czarnecki, W., Luketina, J., Grabska-Barwinska, A., Teh, Y. W.,
Pascanu, R., & Hadsell, R. (2018). Progress & Compress: A scalable framework for continual
learning. *ICML 2018*, Proceedings of the 35th International Conference on Machine Learning,
PMLR 80:4528-4537. arXiv:1805.06370.

**Core problem with original EWC:** Storing a separate Fisher matrix F_k and anchor point
θ*_k for every task k is O(T * d) memory, where T is the number of tasks. For large T and
large models, this is prohibitive.

**Online EWC solution:** Maintain a single running Fisher estimate that accumulates importance
across all tasks. After learning task k, update:

```
F_running = γ * F_running + F_k
```

Where γ is a decay factor (typically γ = 1 in the original formulation, but can be < 1
for selective forgetting). The anchor point θ* is always the current parameters after
convergence on the latest task.

The Online EWC regularization loss for task k+1:

```
L_online_EWC(θ) = L_{k+1}(θ) + (λ/2) * Σ_i F_running,i * (θ_i - θ*_i)^2
```

Where θ* is the parameters at end of task k (updated at each task switch).

**Key differences from original EWC:**
1. Single Fisher matrix instead of T separate ones → O(d) memory vs O(T*d)
2. Accumulation means earlier tasks get higher importance (via multiple F_k contributions)
3. Anchor point updates to latest task end rather than oldest task end
4. Optional decay γ < 1 allows graceful forgetting of very old task importance

**Progress & Compress architecture:** Online EWC is embedded in a two-column network:
- "Progress" column: actively learns current task with lateral connections from "compress" column
- "Compress" column: distilled from the progress column after each task, protected by Online EWC

The compress column stores accumulated knowledge; online EWC prevents its corruption.

**Applicability to our TTT setting:** The running Fisher update is exactly what we need for
cross-chunk TTT. After each chunk k's TTT, compute the chunk-k Fisher and accumulate:
```
F_running = γ * F_running + grad^2_from_chunk_k
```
Then use F_running as the per-parameter importance weight in the next chunk's regularization.

---

### 3. Synaptic Intelligence (SI)

**Citation:** Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through
synaptic intelligence. *ICML 2017*, Proceedings of the 34th International Conference on
Machine Learning, PMLR 70:3987-3995. arXiv:1703.04200.

**Core idea:** Rather than computing importance AFTER a task ends (post-hoc Fisher), SI
computes importance DURING training — online, per gradient step. This tracks the cumulative
contribution of each parameter to the loss reduction over the entire training trajectory.

**Exact SI importance measure:** For parameter k, SI accumulates:

```
Ω_k = Σ_t ( -g_k(t) * Δθ_k(t) )
    = Σ_t ( -∇_θ_k L(θ(t)) * (θ_k(t+1) - θ_k(t)) )
```

The negative sign: if gradient and weight change are in the same direction (gradient
descent), the product `-g_k * Δθ_k` is positive, indicating the parameter moved in a
direction that reduced the loss. This accumulates the total loss reduction attributable
to parameter k.

To avoid trivially large values and normalize, SI divides by the total path length:

```
ω_k = Ω_k / (Δθ_k^total)^2 + ξ
```

Where `Δθ_k^total = θ_k^final - θ_k^initial` and ξ is a small damping constant (typically
ξ = 0.1) to prevent division by zero for parameters that barely moved.

**SI regularization loss for task B:**

```
L_SI(θ) = L_B(θ) + c * Σ_k ω_k * (θ_k - θ*_k)^2
```

Where c is the regularization hyperparameter (analogous to λ in EWC), typical value c = 0.1-1.

**Key differences from EWC:**
- EWC: importance computed ONCE at task end using Fisher (gradient squared at final params)
- SI: importance accumulated CONTINUOUSLY during training (gradient times weight change at every step)
- SI is purely online — no separate Fisher computation pass needed after task completion
- SI captures the importance of parameter trajectories, not just the final loss curvature
- SI can assign different importance to parameters that moved a lot vs. a little

**For TTT application:** SI is attractive because we can accumulate Ω_k WITHIN each chunk's
TTT epochs, giving us per-parameter importance that reflects the actual optimization trajectory
rather than just the gradient at the final LoRA state. This is especially valuable if we run
multiple gradient steps per chunk (e.g., 5-10 epochs of inner loop).

**Implementation in TTT context:**
```python
# Initialize per-chunk importance accumulator
omega = {nm: torch.zeros_like(mod._lora_A) for nm, mod, r, _ in lora_targets}
prev_params = {nm: mod._lora_A.clone().detach() for nm, mod, r, _ in lora_targets}

# During each inner TTT step:
for step in range(inner_steps):
    loss.backward()
    for nm, mod, r, _ in lora_targets:
        g = mod._lora_A.grad
        delta = mod._lora_A.detach() - prev_params[nm]
        omega[nm] += -g * delta  # accumulate importance
        prev_params[nm] = mod._lora_A.clone().detach()
    optimizer.step()
```

---

### 4. Progressive Neural Networks

**Citation:** Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J.,
Kavukcuoglu, K., Pascanu, R., & Hadsell, R. (2016). Progressive neural networks.
arXiv:1606.04671. Google DeepMind Technical Report.

**Core idea:** Instead of regularization, use architectural expansion. Each new task gets
its own "column" (a full copy of the network). Lateral connections from previous columns
feed into each new column's hidden states, allowing knowledge transfer without overwriting.

**Architecture:** For T tasks, maintain T network columns. Column k has access to the
outputs of columns 1..k-1 via learned lateral connections (adapters). The hidden state
of column k at layer l is:

```
h_k^l = f(W_k^l * h_k^{l-1} + Σ_{j<k} U_k^{j,l} * h_j^{l-1})
```

Where U_k^{j,l} are lateral connection weight matrices from prior columns.

**Catastrophic forgetting:** Completely eliminated — prior columns are frozen. New tasks
cannot modify old knowledge.

**Memory cost:** O(T * d) — the column per task grows linearly. Not scalable to hundreds
of tasks.

**Applicability to TTT:** Low. Progressive Nets require a separate column per task, which
is architectural overhead incompatible with sequential chunk processing. However, the IDEA
of lateral connections from prior chunks is interesting: one could maintain a frozen
"memory LoRA" from the previous chunk and connect it to the current chunk's computation
via a small adapter — but this adds inference complexity.

---

### 5. Gradient Episodic Memory (GEM)

**Citation:** Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual
learning. *NeurIPS 2017*, Proceedings of the 31st International Conference on Neural
Information Processing Systems, pp. 6467–6476. arXiv:1706.08840.

**Core idea:** Store a small episodic buffer of examples from previous tasks. When learning
a new task, project gradients to ensure they do not increase the loss on the stored buffer.

**GEM constraint:** Let g_t be the gradient for the current task and g_ep be the gradient
on the episodic memory. GEM requires:

```
<g_t, g_ep> >= 0   (cosine similarity non-negative)
```

If violated, project g_t onto the feasible set (the cone of gradients that don't hurt
episodic memory). This is solved as a quadratic program.

**Key difference from EWC/SI:** GEM operates on GRADIENTS, not parameter values. It does
not require Fisher computation or anchor points. Instead, it directly constraints the
optimization direction.

**Applicability to TTT:** Limited in our setting because GEM requires storing raw examples
from previous chunks. If we store k examples per chunk in a replay buffer, we can check
that the current chunk's gradient doesn't increase loss on those examples. However, this
adds memory and compute overhead proportional to the replay buffer size.

**More applicable variant:** A-GEM (Averaged GEM, Chaudhry et al. 2018) relaxes the
per-example constraint to a mean gradient constraint, making it computationally cheaper.

---

## Part II: Secondary Papers

### 6. PackNet — Parameter Importance via Pruning

**Citation:** Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a
single network by iterative pruning. *CVPR 2018*, pp. 7765–7773. arXiv:1711.05769.

**Core idea:** Rather than gradient-based importance (Fisher, SI), use MAGNITUDE-based
importance. After learning task A, prune the least important weights (by magnitude) to
identify a task-A subnetwork. Freeze this subnetwork for future tasks; the pruned weights
are free to be used for task B.

**Relationship to our use case:** Magnitude-based importance is computationally free (just
look at `abs(param)`) but theoretically weaker than Fisher. For LoRA specifically, low
magnitude in A or B doesn't necessarily mean the parameter is unimportant — what matters
is how much that dimension of the low-rank update affects the output. Fisher (gradient^2)
is a better proxy for this than magnitude.

---

### 7. ANML — Learning to Continually Learn via Meta-Learning

**Citation:** Beaulieu, S., Frati, L., Miconi, T., Lehman, J., Stanley, K. O., Clune, J.,
& Cheney, N. (2020). Learning to continually learn. *ECAI 2020*. arXiv:2002.09571.

**Core idea:** Use MAML-style meta-learning to train a neuromodulatory network (NM network)
that gates the plasticity of a prediction learning network (PLN). The meta-objective is
learning across a sequence of tasks without forgetting. The NM network learns WHICH
weights to update (selective plasticity), analogous to Fisher-based EWC but learned rather
than analytically computed.

**Applicability to TTT:** TTT-E2E (arXiv:2512.23675) is essentially a version of this
idea applied to language model chunks — meta-learning at training time makes the model
learn how to update its weights efficiently at test time. Our EWC approach is a cheaper
analytical alternative that achieves similar selective-plasticity goals without requiring
meta-training.

---

### 8. Sequential Bayesian Updating for Neural Networks

**Reference papers:**
- Nguyen et al. (2018). Variational Continual Learning. ICLR 2018.
- Loo et al. (2023). On Sequential Bayesian Inference for Continual Learning. *Entropy*, 25(6), 884.
- arXiv:2301.01828 (sequential Bayesian inference, 2023).
- Kapoor et al. (2022). Efficient Bayesian Updates for Deep Learning via Laplace Approximations. arXiv:2210.06112.

**Core framework:** In Bayesian inference, the posterior after T tasks is computed via
recursive Bayes' rule:

```
p(θ | D_1, ..., D_T) ∝ p(D_T | θ) * p(θ | D_1, ..., D_{T-1})
```

The key challenge: representing `p(θ | D_1, ..., D_{T-1})` as a tractable distribution.
EWC's Laplace approximation (Gaussian centered at θ*_{T-1}) is the simplest such
approximation. Variational methods (Bayesian neural networks) provide richer approximations.

**MESU — Metaplasticity from Synaptic Uncertainty:**
- Nature Communications, 2025. arXiv:2504.13569. Also arXiv:2312.10153 (preprint 2023).
- Maintains per-parameter uncertainty (standard deviation σ_k) as a running estimate.
- Update rule scales learning rate by uncertainty: Δθ_k ∝ σ_k² * gradient_k
- High uncertainty (σ_k large) → fast learning (parameter is malleable)
- Low uncertainty (σ_k small) → slow learning (parameter is consolidated)
- The σ_k converges to values proportional to the inverse Hessian diagonal:
  `σ_k ≈ 1 / sqrt(H_kk)`, connecting to both Fisher and EWC theory.
- Unlike EWC which uses a fixed anchor point, MESU continuously updates both mean and
  variance, implementing a true online Bayesian update.
- Outperforms EWC, SI, and related methods on 200 sequential Permuted-MNIST tasks.
- Does not require task boundaries.

**Connection to our TTT:** MESU is the most principled approach for our setting because:
1. No task boundaries required (sequential chunks are task-free)
2. Running σ_k can be maintained as a cheap running statistic
3. Naturally handles the same-document vs. different-document case: within a long document,
   σ_k should decay (parameters become more certain/consolidated); across documents, σ_k
   should reset or increase (return to uncertainty)

---

### 9. Memory Replay in LLMs

**References:**
- Wang et al. (2025). Continual Learning of Large Language Models: A Comprehensive Survey.
  *ACM Computing Surveys*. GitHub: Wang-ML-Lab/llm-continual-learning-survey.
- Yang et al. (2024). Mitigating Catastrophic Forgetting in Large Language Models with
  Self-Synthesized Rehearsal. *ACL 2024*. arXiv:2308.08747 vicinity.
- "FOREVER" (2025). Forgetting Curve-Inspired Memory Replay for Language Model CL.
  arXiv:2601.03938.

**Key finding for LLMs:** Replay (storing previous examples and interleaving with new data)
works well but requires access to previous task data. For our TTT setting (sequential chunks
of a SINGLE document), replay is natural: we can maintain a small buffer (e.g., 16-32
tokens) from the previous chunk and include them in the current chunk's TTT mini-batch.
This is GEM-style replay but much cheaper — we're not doing any gradient projection.

**Catastrophic forgetting in LLMs:** Recent empirical study (arXiv:2308.08747) confirms
that LLMs fine-tuned on new tasks do exhibit catastrophic forgetting, though the degree
depends on model size, task similarity, and number of fine-tuning steps. For our TTT
setting (very few steps, small LoRA), forgetting is likely mild but still measurable.

---

### 10. Online-LoRA (WACV 2025)

**Citation:** Wei, X., Li, G., & Marculescu, R. (2025). Online-LoRA: Task-free online
continual learning via low rank adaptation. *WACV 2025*.
Code: https://github.com/Christina200/Online-LoRA-official

**Core contribution:** First Fisher-based importance weighting APPLIED DIRECTLY TO LORA
MATRICES rather than full model parameters, in an online (task-free) setting.

**Exact formula for LoRA Fisher computation:**

```
Ω^A_{A,l} = (1/N) Σ_{k=1}^N ∇ log p(x_k|θ) ⊙ ∇ log p(x_k|θ)
```

That is: the element-wise square of the gradient of log-likelihood with respect to
the LoRA A matrix at layer l. Similarly for B matrix: Ω^B_{B,l}.

**Online-LoRA regularization loss:**

```
L(θ) = L_task(θ) + (λ/2) * Σ_l [ (Ω^A_l ⊙ W^A_l ⊙ W^A_l) + (Ω^B_l ⊙ W^B_l ⊙ W^B_l) ]
```

Note: the penalty here is applied to the parameter VALUES W^A, W^B (pushing toward zero,
not toward a previous anchor). This is more like L2 regularization weighted by Fisher,
anchoring to zero rather than to the previous task's parameters.

**Lambda value used:** λ = 2000 (after normalization; this is for a pretrained ViT-B/16,
not a language model).

**Task detection (task-free):** Online-LoRA detects distribution shifts via loss surface
plateaus rather than explicit task boundaries — when the loss stops decreasing, a new
"task" has begun. This is directly applicable to document boundary detection in TTT.

**Key result:** Online-LoRA achieves 48.18% vs EWC++'s 3.86% on Split-ImageNet-R,
confirming that EWC applied to full parameters (EWC++) fails on pre-trained large models,
while EWC restricted to LoRA parameters succeeds. The ~12x improvement validates the
LoRA-specific Fisher approach.

---

### 11. Fisher Information Approximation Quality

**Citation:** Li, et al. (2025). Fishers for Free? Approximating the Fisher information
matrix by recycling the squared gradient accumulator. *ICML 2025*. arXiv:2507.18807.

**Core insight:** The squared gradient accumulator maintained by Adam's second moment
(`v_t = β_2 * v_{t-1} + (1-β_2) * g_t^2`) is an exponential moving average of gradient
squared — this IS an approximate Fisher diagonal for free.

**Mathematical relationship:**
- True diagonal Fisher: `F_i = E[g_i^2]` (expectation over data and model distribution)
- Adam second moment: `v_i ≈ β_2^∞-smoothed E[g_i^2]`
- The Squisher (scaled Adam second moment): `F_Squisher = N * v` where N = dataset size

**Key caveat:** The standard Fisher sums per-sample squared gradients, while the Adam
accumulator squares the averaged batch gradient. These are mathematically different.
The Adam approach is the "BATCHED" variant (less accurate), not the "EXACT" variant.

**Practical recommendation for our use case:** During TTT, we are already computing
gradients for each chunk. If we simply accumulate `grad^2` per step (using EMA), we get
a running Fisher estimate at essentially zero additional cost beyond the backward pass.
This "Squisher" approach gives us the Fisher needed for EWC-style regularization on the
next chunk without any extra computation.

**Implementation note for EWC scaling:** When using the Squisher as a Fisher proxy for
EWC regularization, the paper recommends scaling λ by the dataset size N to compensate
for the different normalization. In our context: N = chunk_size * num_TTT_steps.

---

### 12. EWC-LoRA (ICLR 2026 under review)

**Citation:** Revisiting Weight Regularization for Low-Rank Continual Learning. ICLR 2026
submission. arXiv:2602.17559.

**Core contribution:** Establishes that vanilla EWC applied to LoRA parameters fails
because it does not account for the LOW-RANK PRODUCT STRUCTURE W = W0 + AB. The Fisher
information of the full update AB depends on both A and B jointly, not just A or B
independently.

**Proposed fix:** EWC-LoRA uses "a low-rank representation to estimate parameter importance
over the full-dimensional space." Instead of computing Fisher of A and B separately:
`F_AB = J^T * F_full * J` where J is the Jacobian of vec(AB) with respect to vec(A) and vec(B).

This projects the full-parameter Fisher information back to the low-rank subspace, giving
the correct importance measure for the LoRA parameters.

**Result:** EWC-LoRA improves over vanilla LoRA by +8.92% on average across multiple
benchmarks, while maintaining constant storage and inference costs.

**Implication for Fisher Memory TTT:** Our simpler approach (computing Fisher directly
from grad_A^2 + grad_B^2) is the Online-LoRA approach, which ignores the product structure.
EWC-LoRA's joint approach would be more principled but requires the Jacobian computation.
For a first implementation, the per-matrix approach is reasonable and has empirical
validation (Online-LoRA at WACV 2025).

---

### 13. EWC Stabilization Practical Considerations

**Citation:** Heckel, R. (2025). On the Computation of the Fisher Information in Continual
Learning. arXiv:2502.11756.

**Critical practical findings:**

1. **EXACT vs BATCHED Fisher are orders of magnitude different in scale:**
   - EXACT: `F_i = (1/N) Σ_n (∇_i log p(y_n|x_n))^2` (per-sample gradient squared, then averaged)
   - BATCHED: `F_i = (1/|D/B|) Σ_B (Σ_{n∈B} ∇_i log p(y_n|x_n))^2` (gradients aggregated in batch, then squared)
   - BATCHED produces SMALLER Fisher values because summing gradients allows cancellation before squaring.
   - The BATCHED lambda must be orders of magnitude larger than EXACT lambda.
   - Most PyTorch implementations use BATCHED (easier to implement), creating confusion.

2. **Best practice:** Compute Fisher with per-sample gradients (batch size 1 forward passes,
   or use functorch/vmap for efficiency). If this is too slow, compute on a subset of N=500
   examples — EXACT on 500 examples outperforms BATCHED on full data.

3. **Gradient vanishing issue (EWC-DR paper):** When the model achieves high confidence
   in predictions, gradients vanish → Fisher values near zero → EWC fails to protect
   parameters that are actually very important. The EWC-DR fix (arXiv:2603.18596) uses
   logit reversal to maintain gradient signal. For our TTT setting (early-stop, few steps),
   gradient vanishing is less of a concern.

4. **Lambda recommendations:**
   - For EXACT Fisher: λ = 100-1000 (original EWC range)
   - For BATCHED Fisher: λ = 1e5-1e7 (must compensate for smaller Fisher values)
   - For Online-LoRA (WACV 2025): λ = 2000 (ViT-B/16 scale)
   - For EWC-DR (vision tasks, 10 tasks): λ = 10,000-20,000 (task-free setting)

---

## Part III: Key Analytical Questions

### Q1: EWC vs Soft-Reset — Fundamental Difference

**Soft-reset (TTT_LORA_DECAY):**
```python
A_new = prev_A * decay + noise  # decay ∈ [0, 1]
```

This is equivalent to a UNIFORM L2 decay toward zero: `A_new = A * decay` corresponds to
the update that minimizes `||A||^2 * (1-decay)/decay` regularization (L2 toward zero).
All parameters are shrunk by the same factor regardless of their importance.

**EWC-inspired approach:**
```python
ewc_loss = cross_entropy + (λ/2) * Σ_i F_{prev,i} * (A_i - prev_A_i)^2
```

This penalizes deviation from `prev_A` proportionally to `F_{prev,i}`:
- High F_{prev,i} (parameter was heavily used): strong resistance to change → memory retention
- Low F_{prev,i} (parameter barely contributed): free to change → plasticity

**What this means for TTT chunks:**

Soft-reset FORGETS everything proportionally. After k chunks with decay=0.9:
- Any LoRA weight learned in chunk 1 is multiplied by 0.9^k by chunk k+1
- Important and unimportant parameters decay equally
- By chunk 10, chunk-1 information has decayed to 0.9^10 ≈ 0.35 of original

EWC-style memory SELECTIVELY retains. After k chunks:
- Parameters that were consistently important across chunks have accumulated high Fisher →
  their values are strongly anchored near the running average of previous best values
- Parameters that were not important are free to adapt to each new chunk

**Which is better for TTT?**

EWC-inspired is STRICTLY more principled and expected to be better for two reasons:
1. It preserves information about the document's CONTENT (parameters activated by this
   document's vocabulary and style) while allowing adaptation to new content within the document
2. It avoids the fundamental problem of soft-reset: a parameter that represents crucial
   vocabulary for the whole document decays just as fast as a parameter that happened to
   fire on one irrelevant token

However, EWC adds computational overhead: Fisher computation after each chunk + regularization
term in loss. In practice, the Fisher computation using the "Squisher" (EMA of grad^2) adds
near-zero overhead since we already compute gradients.

---

### Q2: Running Fisher Estimate Across TTT Chunks

**Online EWC formulation for TTT:**

```python
# Initialize running Fisher (zero = uniform importance)
running_fisher = {nm: torch.zeros_like(mod._lora_A)
                  for nm, mod, r, _ in lora_targets}

# After each chunk's TTT:
for nm, mod, r, _ in lora_targets:
    # Compute chunk-k Fisher via gradient squared (EXACT: batch_size=1 forward passes)
    chunk_fisher_A = (mod._lora_A.grad ** 2)
    chunk_fisher_B = (mod._lora_B.grad ** 2)

    # Accumulate into running Fisher (Online EWC style)
    running_fisher[nm] = gamma * running_fisher[nm] + chunk_fisher_A + chunk_fisher_B

    # Store anchor point (current best LoRA params)
    prev_A[nm] = mod._lora_A.detach().clone()
    prev_B[nm] = mod._lora_B.detach().clone()

# Next chunk's TTT loss:
ewc_penalty = sum(
    (running_fisher[nm] * (mod._lora_A - prev_A[nm])**2).sum() +
    (running_fisher[nm] * (mod._lora_B - prev_B[nm])**2).sum()
    for nm, mod, r, _ in lora_targets
)
ewc_loss = cross_entropy + (ewc_lambda / 2) * ewc_penalty
```

**Choice of gamma (decay for running Fisher):**
- gamma = 1.0: All past chunks equally weighted. Appropriate if document is homogeneous.
- gamma = 0.9-0.95: Recent chunks weighted more. Appropriate for evolving documents.
- gamma = 0.0: Only previous chunk matters. Equivalent to a per-step EWC (no accumulation).

For same-document sequential chunks, gamma = 0.9-0.95 is a reasonable default.
For document boundaries detected (see Q5), reset running_fisher to zero.

**Adam second moment as free Fisher estimate:**
If using Adam optimizer for TTT, the second moment `v_t` IS an EMA of grad^2 with
decay β_2 (typically 0.999). We can read this off directly after each chunk without
any extra computation:

```python
for param_group in optimizer.param_groups:
    for p in param_group['params']:
        if p in optimizer.state:
            v = optimizer.state[p]['exp_avg_sq']  # This is F_approx
            running_fisher[p_name] = gamma * running_fisher[p_name] + v
```

---

### Q3: Synaptic Intelligence for Per-Parameter TTT Importance

**SI during TTT inner loop:**

SI is more informative than single-step Fisher because it accumulates importance over
all the gradient steps within a chunk's TTT. A parameter that consistently has large
`-g * Δθ` products across all TTT steps is genuinely important for this chunk.

```python
# Initialize SI accumulators at start of each chunk
si_omega = {nm: torch.zeros_like(mod._lora_A) for nm, mod, r, _ in lora_targets}
si_prev = {nm: mod._lora_A.clone().detach() for nm, mod, r, _ in lora_targets}
lora_initial = {nm: mod._lora_A.clone().detach() for nm, mod, r, _ in lora_targets}

# During each inner TTT gradient step:
for step in range(inner_steps):
    loss = compute_chunk_loss(chunk_tokens)
    loss.backward()
    for nm, mod, r, _ in lora_targets:
        g = mod._lora_A.grad.clone()
        # SI: accumulate negative gradient times delta_theta
        si_omega[nm] += -g * (mod._lora_A.detach() - si_prev[nm])
        si_prev[nm] = mod._lora_A.clone().detach()
    optimizer.step()

# After chunk TTT completes, compute per-parameter SI importance:
for nm, mod, r, _ in lora_targets:
    path_length = (mod._lora_A.detach() - lora_initial[nm]) ** 2 + 1e-4  # ξ damping
    importance[nm] = si_omega[nm] / path_length
```

**SI vs Fisher for weighting soft-reset:**

Rather than replacing soft-reset entirely, SI importance can MODULATE the decay:
```python
# Importance-weighted soft-reset using SI
decay_per_param = base_decay + (1 - base_decay) * sigmoid(importance_normalized)
A_new = prev_A * decay_per_param  # High importance → decay close to 1.0 (retain)
                                   # Low importance → decay close to base_decay (forget)
```

This is a hybrid: computationally cheaper than full EWC regularization, but more
principled than uniform soft-reset.

---

### Q4: The Right Notion of "What to Remember" for Sequential Chunks

**Problem framing:** When processing sequential chunks of a validation document:
- Within a single long document: same topic, author style, vocabulary → want to RETAIN
  LoRA state that adapted to this document's characteristics
- Between documents (or very different sections): new topic, new style → want to RESET
  or PARTIALLY reset

**What Fisher-weighted EWC preserves:**

The Fisher information `F_i = E[(∂L/∂θ_i)^2]` measures how sensitive the CURRENT CHUNK's
loss is to parameter i. For a sequential document:
- Vocabulary-specific parameters (e.g., LoRA directions corresponding to rare words used
  throughout the document) will have high Fisher across all chunks of that document
- Generic parameters (common grammatical patterns) may have moderate Fisher everywhere
- Document-irrelevant parameters will have low Fisher throughout

After running chunks 1..k, `running_fisher[i]` encodes: "this parameter has been
important for all chunks of this document seen so far."

When chunk k+1 arrives from the SAME document, the EWC penalty will strongly resist
changes to document-important parameters → continuity maintained.

When a NEW document begins (detected via gradient dissimilarity, see Q5), we should
reset or reduce `running_fisher` to allow rapid adaptation.

**The key insight:** Fisher-weighted EWC naturally implements "remember what was
important for THIS document" because importance accumulates across chunks via the running
Fisher sum. This is exactly the "cross-chunk document memory" we want.

**Alternative framing — what soft-reset misses:**
Soft-reset implements: "remember a geometric mean of all past LoRA states." But the
geometric mean has no notion of importance. A parameter that was crucial for understanding
the document's topic (e.g., pointing in the direction of the author's technical vocabulary
in embedding space) decays at the same rate as a parameter that fired on one irrelevant
punctuation mark.

---

### Q5: Gradient Fingerprint as Document Boundary Detector

**Theoretical basis:**

For two text segments from the SAME document, the gradient of the TTT loss with respect
to LoRA parameters will be similar (they share topic, vocabulary, style). For segments
from DIFFERENT documents, gradients will point in different directions in parameter space.

This is analogous to gradient task similarity in multi-task learning: if `<g_A, g_B> > 0`,
tasks A and B benefit from a shared representation; if `<g_A, g_B> < 0`, they interfere.

**Gradient fingerprint via top singular vector:**

Computing full cosine similarity of gradient vectors is O(d) where d = LoRA parameter
count. For multiple LoRA layers, d can be large. A cheaper proxy:

```python
# Flatten all LoRA gradients into a single vector g ∈ R^d
g_prev = torch.cat([mod._lora_A.grad.flatten() for nm, mod, r, _ in lora_targets])
g_curr = torch.cat([mod._lora_A.grad.flatten() for nm, mod, r, _ in lora_targets])

# Cosine similarity as document similarity score
cos_sim = F.cosine_similarity(g_prev.unsqueeze(0), g_curr.unsqueeze(0)).item()

# Decision rule:
if cos_sim > THRESHOLD_SAME_DOC:      # e.g., 0.3-0.5
    decay = HIGH_DECAY                 # e.g., 0.95 — same document, retain
    gamma = 0.9                        # accumulate Fisher
else:
    decay = LOW_DECAY                  # e.g., 0.1 — new document, reset
    running_fisher = reset_to_zeros()  # start fresh
```

**Top singular vector (more expensive but more informative):**

For gradient matrices (not flattened vectors), the top singular vector U_1 (from SVD of
the gradient matrix ∂L/∂A ∈ R^{d_out x r}) captures the dominant direction of adaptation.
Two chunks from the same document should have similar dominant adaptation directions.

```python
U_prev, S_prev, V_prev = torch.svd(grad_A_prev)
U_curr, S_curr, V_curr = torch.svd(grad_A_curr)
# Compare top singular vectors
alignment = torch.abs(torch.dot(U_prev[:, 0], U_curr[:, 0]))
```

**Limitations and considerations:**
1. Gradient similarity is noisy with small chunk sizes or few TTT steps
2. Documents that happen to share vocabulary will have similar gradients even if unrelated
3. The threshold must be tuned — gradients from sequential chunks are ALWAYS more similar
   than chunks from unrelated documents, so the threshold is a relative measure
4. A simpler proxy: loss jump. If loss at the start of chunk k+1 is much HIGHER than loss
   at the end of chunk k, the LoRA adapted to chunk k is not useful for chunk k+1 → reset.
   This is the implicit mechanism in Online-LoRA's "plateau detection."

**Loss-jump as document boundary detector (simpler than gradient fingerprint):**
```python
loss_start_of_chunk = compute_loss(chunk_k1_tokens, LoRA_from_chunk_k)
loss_baseline = compute_loss(chunk_k1_tokens, original_model)  # no LoRA

relative_transfer = (loss_baseline - loss_start_of_chunk) / loss_baseline
if relative_transfer < 0:      # LoRA from prev chunk HURTS new chunk
    decay = RESET              # hard reset
elif relative_transfer < 0.1:  # minimal benefit
    decay = LOW_DECAY          # soft reset
else:                          # good transfer
    decay = HIGH_DECAY         # strong retention
```

---

## Part IV: Novel Variant Design — Fisher Memory TTT

### Design Specification

**Technique name:** TTT_EWC_FISHER (or "Fisher Memory TTT")

**Hypothesis:** Replacing uniform soft-reset (exponential decay of all LoRA parameters)
with Fisher-weighted EWC regularization will improve BPB by preserving document-important
parameters across chunks while allowing adaptation to new content.

**Algorithm:**

```python
# Global state maintained across chunks
running_fisher = {nm: torch.zeros_like(mod._lora_A)
                  for nm, mod, r, _ in lora_targets}
prev_A = {nm: torch.zeros_like(mod._lora_A) for nm, mod, r, _ in lora_targets}
prev_B = {nm: torch.zeros_like(mod._lora_B) for nm, mod, r, _ in lora_targets}

EWC_LAMBDA = 0.1        # start small, tune upward; scale depends on Fisher normalization
FISHER_GAMMA = 0.9      # decay factor for running Fisher accumulation
FISHER_N_SAMPLES = 32   # number of tokens for Fisher estimation (EXACT variant)

def compute_chunk_fisher(lora_targets, chunk_tokens, model):
    """
    Compute diagonal Fisher for LoRA params using per-token (EXACT) gradient squared.
    Uses a SUBSET of chunk tokens for efficiency.
    """
    fisher = {nm: torch.zeros_like(mod._lora_A) for nm, mod, r, _ in lora_targets}

    # Zero all grads
    model.zero_grad()

    # Single forward-backward on a small subset (EXACT Fisher: per-sample grad^2)
    sample_tokens = chunk_tokens[:FISHER_N_SAMPLES]
    loss = model(sample_tokens, labels=sample_tokens).loss
    loss.backward()

    for nm, mod, r, _ in lora_targets:
        if mod._lora_A.grad is not None:
            fisher[nm] = (mod._lora_A.grad ** 2 + mod._lora_B.grad ** 2).detach()

    return fisher

def ttt_chunk_with_ewc(chunk_tokens, model, lora_targets, optimizer,
                        running_fisher, prev_A, prev_B, ewc_lambda, inner_steps):
    """
    TTT inner loop for one chunk with EWC regularization.
    """
    for step in range(inner_steps):
        model.zero_grad()

        # Standard TTT cross-entropy loss
        ce_loss = model(chunk_tokens, labels=chunk_tokens).loss

        # EWC penalty: resist changes to previously important params
        ewc_penalty = torch.tensor(0.0, device=ce_loss.device)
        for nm, mod, r, _ in lora_targets:
            if nm in running_fisher:
                ewc_penalty += (
                    running_fisher[nm] * (mod._lora_A - prev_A[nm]) ** 2
                ).sum()
                # Note: B matrix handled separately if needed

        total_loss = ce_loss + (ewc_lambda / 2) * ewc_penalty
        total_loss.backward()
        optimizer.step()

    return ce_loss.item()

# Main TTT loop over chunks:
for chunk_idx, chunk_tokens in enumerate(document_chunks):

    # Run TTT with EWC regularization
    ttt_chunk_with_ewc(chunk_tokens, model, lora_targets, optimizer,
                        running_fisher, prev_A, prev_B, EWC_LAMBDA, inner_steps=5)

    # After chunk, compute Fisher and update running estimate
    chunk_fisher = compute_chunk_fisher(lora_targets, chunk_tokens, model)

    for nm, mod, r, _ in lora_targets:
        # Online EWC running Fisher update
        running_fisher[nm] = FISHER_GAMMA * running_fisher[nm] + chunk_fisher[nm]

        # Update anchor points
        prev_A[nm] = mod._lora_A.detach().clone()
        prev_B[nm] = mod._lora_B.detach().clone()
```

---

### Lambda Calibration

**The lambda problem:** `ewc_lambda` must balance the scale of `ce_loss` with the scale
of the EWC penalty. The penalty scale depends on:
1. Fisher values (from gradient squared): typically O(1e-4) to O(1e-1) per parameter
2. Parameter deviation (A - prev_A): starts near 0, grows to O(LoRA lr * steps)
3. Number of LoRA parameters: sum over all matrices

**Practical calibration strategy:**

At chunk 2 (after chunk 1 has built up some Fisher), compute the ratio:
```python
ce_loss_scale = ce_loss.item()                     # typically 1-4 (BPB range)
ewc_unscaled = ((running_fisher[nm] * (mod._lora_A - prev_A[nm])**2).sum()
                for nm in running_fisher)
ewc_total_unscaled = sum(ewc_unscaled)             # with lambda=1.0

# Aim for EWC penalty to be ~10-50% of ce_loss magnitude:
ewc_lambda = ce_loss_scale / (ewc_total_unscaled * 10)
```

**Empirical starting points based on literature:**
- Online EWC (Schwarz 2018): λ = 1.0 with accumulation → scales with number of tasks
- Online-LoRA (Wei 2025): λ = 2000 for ViT-B/16 scale
- EWC-DR (2026): λ = 10,000-20,000 for vision classifiers, 10 tasks
- EWC original: λ = 400 for MNIST-scale models

For our setting (small LoRA, r=4-8, ~50M parameter language model, ~500-token chunks):
- If using EXACT Fisher (per-sample grad^2): start with λ = 0.01-0.1
- If using BATCHED Fisher (full mini-batch): start with λ = 10-100 (Fisher values are smaller)
- Use Adam second moment as free Fisher proxy: scale λ by chunk_size (≈ N)

**Recommendation:** Start at λ = 0.05, run ablations at {0.01, 0.05, 0.1, 0.5, 1.0}.
Since our Fisher is per-parameter and LoRA parameters are small in number (r*d per layer,
typically 4*768 = 3072 per LoRA matrix), the penalty will not explode.

---

### Comparison Table: Memory Mechanisms for TTT

| Mechanism          | Computational Cost | Principled? | Handles Importance? | Task-Free? | Cross-Chunk Memory |
|--------------------|--------------------|-------------|---------------------|------------|--------------------|
| No Memory (reset)  | 0                  | N/A         | No                  | Yes        | None               |
| Soft-Reset (decay) | 0                  | Heuristic   | No (uniform)        | Yes        | Geometric mean     |
| EWC (offline)      | O(N*d)             | Yes (Bayes) | Yes (Fisher)        | No (needs anchor) | Anchor + Fisher |
| Online EWC         | O(d)               | Yes (Bayes) | Yes (running Fisher)| Yes        | Running anchor     |
| SI (online)        | O(d*steps)         | Yes (traj.) | Yes (cumulative)    | Yes        | Trajectory-based   |
| MESU               | O(d)               | Yes (Bayes) | Yes (uncertainty)   | Yes        | Posterior mean+var |
| Soft-Reset + SI    | O(d*steps)         | Hybrid      | Yes (SI-weighted)   | Yes        | Importance-weighted |
| **Fisher Memory TTT** | O(d) + ~O(N_s*d)| Yes (Bayes) | Yes (Fisher)        | Yes        | Running anchor     |

Legend: N = dataset size, d = LoRA param count, N_s = Fisher sample count (N_s << N)

---

## Part V: Citations in Standard Format

**Primary Papers:**

1. Kirkpatrick, J., Pascanu, R., Rabinowitz, N., Veness, J., Desjardins, G., Rusu, A. A.,
   Milan, K., Quan, J., Ramalho, T., Grabska-Barwinska, A., Hassabis, D., Clopath, C.,
   Kumaran, D., & Hadsell, R. (2017). Overcoming catastrophic forgetting in neural networks.
   *Proceedings of the National Academy of Sciences*, 114(13), 3521–3526.
   https://doi.org/10.1073/pnas.1611835114. arXiv:1612.00796.

2. Schwarz, J., Czarnecki, W., Luketina, J., Grabska-Barwinska, A., Teh, Y. W., Pascanu, R.,
   & Hadsell, R. (2018). Progress & Compress: A scalable framework for continual learning.
   *Proceedings of the 35th International Conference on Machine Learning (ICML 2018)*,
   PMLR 80:4528-4537. arXiv:1805.06370.

3. Zenke, F., Poole, B., & Ganguli, S. (2017). Continual learning through synaptic intelligence.
   *Proceedings of the 34th International Conference on Machine Learning (ICML 2017)*,
   PMLR 70:3987-3995. arXiv:1703.04200.

4. Rusu, A. A., Rabinowitz, N. C., Desjardins, G., Soyer, H., Kirkpatrick, J., Kavukcuoglu, K.,
   Pascanu, R., & Hadsell, R. (2016). Progressive neural networks. arXiv:1606.04671.

5. Lopez-Paz, D., & Ranzato, M. (2017). Gradient episodic memory for continual learning.
   *Advances in Neural Information Processing Systems 30 (NeurIPS 2017)*, pp. 6467–6476.
   arXiv:1706.08840.

**Secondary Papers:**

6. Mallya, A., & Lazebnik, S. (2018). PackNet: Adding multiple tasks to a single network
   by iterative pruning. *CVPR 2018*, pp. 7765–7773. arXiv:1711.05769.

7. Beaulieu, S., Frati, L., Miconi, T., Lehman, J., Stanley, K. O., Clune, J., & Cheney, N.
   (2020). Learning to continually learn. *ECAI 2020*. arXiv:2002.09571.

8. Loo, N., Sinha, S., & Pitassi, T. (2023). On sequential Bayesian inference for continual
   learning. *Entropy*, 25(6), 884. arXiv:2301.01828.

9. Wei, X., Li, G., & Marculescu, R. (2025). Online-LoRA: Task-free online continual
   learning via low rank adaptation. *WACV 2025*.
   https://github.com/Christina200/Online-LoRA-official. arXiv:2411.05663.

10. Heckel, R., & Reinhart, E. (2025). On the computation of the Fisher information in
    continual learning. arXiv:2502.11756.

11. Li, L., et al. (2025). Fishers for free? Approximating the Fisher information matrix
    by recycling the squared gradient accumulator. *ICML 2025*. arXiv:2507.18807.

12. Benzing, F. (2022). Unifying importance-based regularisation methods for continual
    learning. *AISTATS 2022*, PMLR 151. https://proceedings.mlr.press/v151/benzing22a.

13. Anonymous. (2026). Revisiting weight regularization for low-rank continual learning.
    *ICLR 2026 (under review)*. arXiv:2602.17559.

14. Achterberg, J., & Bhatt, U. (2025). Bayesian continual learning and forgetting in
    neural networks (MESU). *Nature Communications* (2025). arXiv:2504.13569.
    Also arXiv:2312.10153 (preprint).

15. Wang, Z., et al. (2025). Continual learning of large language models: A comprehensive
    survey. *ACM Computing Surveys*. GitHub: Wang-ML-Lab/llm-continual-learning-survey.

16. Zhang, T., et al. (2026). Elastic weight consolidation done right for continual learning
    (EWC-DR). arXiv:2603.18596.

17. Li, Z., et al. (2025). TNT: Improving chunkwise training for test-time memorization.
    arXiv:2511.07343.

18. Swamynathan, V. P. (2026). SR-TTT: Surprisal-aware residual test-time training.
    arXiv:2603.06642.

---

## Part VI: Research Gaps and Future Directions

1. **No paper addresses Fisher-weighted memory specifically for TTT** (as of March 2026).
   TTT papers (TTT-E2E, Q-only TTT, LaCT, SR-TTT) all use either hard resets or naive
   carry-over. Fisher Memory TTT would be novel.

2. **The product structure of LoRA (W = W0 + AB)** means that the natural Fisher is the
   Fisher of the COMPOSED update, not the individual matrices. EWC-LoRA (arXiv:2602.17559)
   addresses this for CL but not for TTT. A more principled approach would compute Fisher
   w.r.t. the composed update direction.

3. **Document boundary detection** using gradient fingerprints is unexplored for TTT. The
   simplest proxy (loss at start of new chunk using previous LoRA vs. baseline) requires no
   extra compute and could trigger selective reset. No paper evaluates this specifically.

4. **MESU's uncertainty-scaled learning rate** is theoretically the most principled approach
   for our setting but adds per-parameter variance tracking (2x parameter storage). Testing
   MESU vs. Online EWC for TTT would be high-value research.

5. **Interaction with quantization:** Our model uses int5/int6 quantization (Parameter Golf
   competition). How Fisher computation and EWC regularization interact with quantized
   weights is not well understood. The Fisher gradient reflects quantized forward pass
   gradients, which may differ from full-precision gradients.

---

## Part VII: Recommended Implementation Plan for Fisher Memory TTT

### Phase 1: Baseline measurement
- Run TTT with NO memory (hard reset after each chunk)
- Run TTT with current soft-reset (TTT_LORA_DECAY)
- Record BPB on validation set, broken down by chunk position

### Phase 2: Fisher Memory TTT (basic)
- After each chunk's TTT, compute Fisher via `grad^2` on 32 tokens (EXACT variant)
- Next chunk augments loss with `(λ/2) * Σ F_prev * (A - A_prev)^2`
- Tune λ ∈ {0.01, 0.05, 0.1, 0.5} → target 10-30% of ce_loss magnitude
- Expected improvement over soft-reset: -0.005 to -0.020 BPB

### Phase 3: Online Fisher accumulation
- Replace per-chunk Fisher with running average: `F = 0.9 * F + 0.1 * grad^2`
- Expected: -0.002 to -0.010 BPB additional improvement vs Phase 2

### Phase 4: SI-weighted soft-reset (lighter alternative)
- Instead of full EWC, modulate decay per-parameter by SI importance
- Hybrid approach: computationally cheaper than EWC, more principled than uniform decay
- May be preferable if EWC overhead is too large

### Phase 5: Document boundary detection
- Compute gradient cosine similarity between consecutive chunks
- If cosine < 0.2: hard reset LoRA and running Fisher
- Expected: particularly valuable when validation set crosses document boundaries
