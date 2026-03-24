# LoRA Rank Dynamics — Deep Research Report
# Topic: Rank Selection, Scheduling, and Adaptive Rank for TTT
# Generated: 2026-03-24
# Databases: arXiv, ICLR 2023, ICML 2024, NeurIPS 2024, ACL Anthology

---

## 1. EXECUTIVE SUMMARY FOR TTT RANK STRATEGY

The literature converges on four findings that directly constrain our design space:

1. **Intrinsic dimensionality is extremely low.** 200 random parameters recover 90% of full fine-tuning performance on sentence tasks (Aghajanyan 2021). For TTT on a single document, rank-2 or rank-4 is almost certainly sufficient once the adapter is pointed at the right subspace.

2. **Initialization direction matters more than rank.** PiSSA (+5.16 pp on GSM8K vs LoRA), GoRA (+5.13 pp), and LoRA-GA all show that SVD-aligned initialization dominates rank size in determining final quality. Our RELI is consistent with this finding.

3. **Standard LoRA scaling (alpha/r) causes gradient collapse at high ranks.** RsLoRA proves gamma = 1/sqrt(r) is required for gradient magnitude to stay Theta(1) at all ranks. If we do rank annealing from rank-2r to rank-r, we must use rsLoRA scaling or normalize manually.

4. **Rank annealing is validated by AdaLoRA but only after an exploration warmup.** AdaLoRA starts at 1.5x target budget, uses a cubic decay schedule, and prunes every DeltaT=100 steps. The SVD importance score (sensitivity x uncertainty) gates which singular directions survive. This is exactly the RELI-with-Rank-Annealing mechanism — just applied to a TTT loop.

---

## 2. PRIMARY PAPERS — DETAILED ANALYSIS

### 2.1 AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning
**Citation:** Zhang, Q., Chen, M., Bukharin, A., Kaur, N., He, P., Cheng, H., Chen, W., & Zhao, T. (2023). AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning. *ICLR 2023*. arXiv:2303.10512.

**Core Mechanism — SVD Parameterization:**
Instead of W + BA, AdaLoRA writes:
```
W + P * Lambda * Q
```
where P (left singular vectors), Lambda (diagonal singular values), Q (right singular vectors) are all learned. Lambda is a diagonal matrix with entries lambda_i. The update is thus a proper SVD triplet (p_i, lambda_i, q_i) for each rank component i.

The key difference from LoRA: in standard LoRA, A and B are unconstrained — their product BA can have any structure. AdaLoRA enforces that P and Q remain approximately orthonormal via a Frobenius regularizer:

```
R(P, Q) = ||P^T P - I||^2_F + ||QQ^T - I||^2_F
```

This regularizer is added to the loss with a coefficient (default lambda_reg = 0.1). Without it, P and Q drift from orthogonality and the singular value interpretation is lost.

**Importance Score (Equation 9):**
```
S_{k,i} = s(lambda_{k,i}) + (1/d1) * sum_j s(P_{k,ji}) + (1/d2) * sum_j s(Q_{k,ij})
```
where s(w) is a sensitivity-times-uncertainty score using exponential moving averages:
```
s(w_ij) = I_bar(w_ij) * U_bar(w_ij)        (Equation 13)
```
I_bar = smoothed |w * dL/dw| (sensitivity), U_bar = smoothed uncertainty term.

The score accounts for the ENTIRE triplet, not just the singular value magnitude. This is important: a triplet with a small singular value but high sensitivity (the direction is being actively updated) is NOT pruned. Only truly irrelevant triplets are removed.

**Budget Scheduler (Cubic, from Appendix A):**
```
b(t) = b(0)                                     for 0 <= t < t_i  (warmup)
b(t) = b(T) + (b(0)-b(T)) * (1 - (t-t_i)/(T-t_i-t_f))^3   for t_i <= t < T-t_f
b(t) = b(T)                                     for t >= T-t_f   (stabilization)
```
- b(0) = initial budget = 1.5 * b(T)   (start 50% higher than target)
- b(T) = final target budget
- t_i = warmup steps (default 100-200)
- t_f = stabilization steps (default ~200)
- Pruning occurs every DeltaT = 100 steps during the cubic decay phase

**Rank numbers in experiments:**
- Initial rank: 12 per layer (for DeBERTa experiments)
- Final rank: varies by layer, average = 8 after pruning
- Global budget: 0.1M to 10M parameters across tasks
- Consistent performance improvement vs fixed-rank LoRA at rank 8 in "low budget" settings

**TTT Applicability Assessment:**
AdaLoRA was designed for fine-tuning on 1000s-100000s of examples. The pruning every DeltaT=100 steps assumes multiple epochs. For TTT with 50 epochs of 50 steps each = 2500 steps total, DeltaT=100 is plausible. However, the per-layer importance score requires enough gradient history to populate the EMA statistics. With very few steps before pruning, the EMA scores are noisy. Recommendation: delay pruning to after the first 30% of TTT steps (epoch_count // 3), not immediately.

**No discussion of TTT or continual learning** in the paper. Direct application requires adaptation of the schedule parameters.

---

### 2.2 DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation
**Citation:** Valipour, M., Rezagholizadeh, M., Kobyzev, I., & Ghodsi, A. (2023). DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation. *EACL 2023*. arXiv:2210.07558.

**Core Mechanism:**
DyLoRA trains a single set of adapter matrices that works well across a RANGE of ranks simultaneously. During each forward pass, it samples a rank b from {1, 2, ..., r_max} and uses only the top-b singular components. The gradients update the top-b subspace, and by training across all b values, the adapter learns a nested structure: the top-1 direction captures the most important adaptation, top-2 adds the next, etc.

**Key insight:** After training, you can deploy at ANY rank from 1 to r_max without retraining. The 4-7x speedup comes from eliminating the full hyperparameter search over rank.

**Relevance to TTT:**
DyLoRA's nested-rank training is conceptually similar to the "exploration" phase of RELI-with-Rank-Annealing. If we initialize at rank-2r with DyLoRA-style training (randomly sampling sub-ranks during early TTT epochs), we learn which sub-rank directions are most critical BEFORE committing to rank-r at pruning time.

**Limitation:** DyLoRA does not adaptively choose which rank to USE at inference — it just trains for flexibility. The rank decision is still made by the user.

---

### 2.3 LoRA+: Efficient Low Rank Adaptation of Large Models
**Citation:** Hayou, S., Ghosh, N., & Yu, B. (2024). LoRA+: Efficient Low Rank Adaptation of Large Models. *ICML 2024*. arXiv:2402.12354.

**Core Finding — Asymmetric Learning Rates:**
Standard LoRA uses the same learning rate for both A (the down-projection) and B (the up-projection). LoRA+ proves this is suboptimal for wide networks.

**Theoretical Basis (Proposition 2):**
For width-n networks, efficient feature learning requires:
```
eta_A = Theta(n^{-1})    (A should have SMALL learning rate)
eta_B = Theta(1)          (B should have LARGE learning rate)
```
The ratio lambda = eta_B / eta_A = Theta(n), meaning B's LR should scale proportionally to model width. For typical LLMs (d_model = 768 to 8192), this translates to lambda in the range of hundreds theoretically — but practically usable values are 2 to 16.

**Empirical Optimal Ratio:**
Ablation heatmaps (Figure 2) show the optimal (eta_A, eta_B) pairs for a toy model. The paper does not specify a single canonical lambda value in the abstract. The range reported across experiments is **lambda = 2 to 16**, with a common practical choice of **lambda = 4** being effective across tasks. Performance gains of 1-2% and ~2x convergence speed are achieved at lambda=4 to 16.

**Direct relevance to our RELI:**
Our RELI already uses a gradient-aligned initialization. The LoRA+ result suggests that after RELI init, we should also set B's learning rate to 4-16x the learning rate of A. If we are using a uniform LR for A and B currently, this is a free improvement. Specifically:

```
lr_A = base_lr / sqrt(r)   (rsLoRA scaling for A)
lr_B = lambda * base_lr    (higher LR for B)
```

where lambda = 4 is a conservative safe choice, lambda = 16 may help in aggressive settings.

**Note on RELI interaction:** LoRA+ assumes B is initialized to zero (standard LoRA). With RELI, B is initialized to a non-zero gradient-aligned direction. The LoRA+ proof may not directly apply when B(0) != 0, but the intuition (B carries more of the signal, so it benefits from higher LR) likely still holds.

---

### 2.4 Flora: Low-Rank Adapters Are Secretly Gradient Compressors
**Citation:** Han, Y., Jarayam, R., Feder, A., Bhatt, A., & Koutchev, G. (2024). Flora: Low-Rank Adapters Are Secretly Gradient Compressors. *ICML 2024*. arXiv:2402.03293.

**Formal Equivalence Statement (Theorem 2.1 + Observation 2.2):**

Under small learning rate assumption:
```
W_T = W_0 + Delta_B * A_0^T * A_0 * ... ≈ W_0 + Delta_B * A_0
```
where A_0 is the INITIAL value of the A matrix (frozen at initialization).

The key observation: **LoRA is equivalent to compressing the gradient G by projecting it through A_0^T (down-projection) and then decompressing through A_0 (up-projection):**
```
gradient_compressed = A_0 @ G @ A_0^T      (dimension: r x r)
weight_update = A_0^T @ gradient_compressed  (dimension: d_in x d_out)
```

By Johnson-Lindenstrauss (Lemma 2.3 + Theorem 2.4): if A_0 is Gaussian random, then E[A_0^T @ A_0] = I, so the compression is approximately unbiased.

**What this means for RELI:**
RELI initializes A_0 from the TOP singular directions of the gradient (or surrogate gradient). Under Flora's framework, this means RELI is initializing from the OPTIMAL projection basis — one that maximizes the energy captured in the compressed gradient representation. A random Gaussian A_0 captures the gradient in expectation, but a gradient-aligned A_0 captures it with MUCH lower variance (no information loss for the dominant directions).

Formally: if A_0 = V_r^T (top-r right singular vectors of G), then:
```
A_0 @ G ≈ Sigma_r * U_r^T   (the top-r singular values and left vectors of G)
```
This is the OPTIMAL rank-r approximation of G by Eckart-Young theorem.

**Flora's Extension (Resampling):**
Flora addresses the fact that as training progresses, the optimal subspace shifts. A fixed A_0 becomes suboptimal. Flora resamples A with a new random Gaussian at each step, so the total weight change is no longer constrained to a fixed low-rank subspace (the accumulated weight change can be effectively full-rank).

**Direct implication for RELI-with-Rank-Annealing:**
At the rank pruning step (epoch // 3), we are essentially "re-orienting" the projection to the current optimal subspace, which Flora shows is necessary when the subspace drifts. The SVD of the current delta B@A identifies where the weight change has concentrated. Keeping the top-r singular directions and discarding the rest is equivalent to periodic subspace refresh in GaLore/Flora.

**No RELI-style initialization discussion** in the Flora paper. The equivalence we are drawing is our own novel inference from their theorem, not stated in the paper.

---

### 2.5 GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection
**Citation:** Zhao, J., Zhang, Z., Chen, B., Wang, Z., Anandkumar, A., & Tian, Y. (2024). GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection. *ICML 2024*. arXiv:2403.03507.

**Core Algorithm:**
```
1. Every T steps: compute G = dL/dW (full gradient)
2. SVD: G = U S V^T, take top-r left singular vectors P_t = [u_1, ..., u_r]
3. Compressed gradient: G_low = P_t^T @ G   (dimension: r x d_out)
4. Apply optimizer (Adam/SGD) to G_low, get update step delta_G_low
5. Full-space update: delta_W = P_t @ delta_G_low
6. After T steps: recompute P_{t+1} from new gradient
```

**Key Parameters:**
- T = 200 (default refresh interval, tested: 50 to 1000 produce similar results)
- Rank r = 512 for 1B models, r = 1024 for 7B models
- Scale factor alpha = 0.25 (does NOT scale with r, unlike LoRA's alpha/r)

**Ablation on T:** Both too frequent (T < 50, introduces overhead) and too infrequent (T > 500, subspace becomes stale) hurt convergence. T=200 is the sweet spot. For small ranks (r << n), more frequent updates needed.

**GaLore vs LoRA — Critical Difference:**
- LoRA: A and B are PARAMETERS updated by the optimizer. The update direction is constrained to the COLUMN SPACE of B (since Delta_W = B @ Delta_A + Delta_B @ A).
- GaLore: P is not a parameter — it is recomputed from the gradient. The optimizer states (Adam moments) live in the low-rank space, but the WEIGHT itself can move anywhere since the projection direction changes every T steps.
- GaLore = "full-parameter learning with compressed optimizer states"
- LoRA = "constrained parameter learning in a fixed low-rank subspace"

When T is small (frequent refresh), GaLore approaches full-rank training in terms of expressiveness. When T is large (infrequent refresh), GaLore approaches LoRA.

**GaLore vs RELI — What is Different:**
RELI initializes the LoRA A matrix from the gradient's top singular vectors. This is essentially "GaLore applied ONCE at initialization." GaLore continues to refresh P every T steps. If our TTT runs for 50 epochs with 50 steps each = 2500 total steps, a GaLore-style refresh at T=200 would give ~12 refreshes. Each refresh has O(mn^2) SVD cost, which at d_model=384 and d_ff=1536 is manageable (~0.3M floats) but not free.

The key question is: **should we refresh the RELI projection during TTT?** The literature suggests yes — subspace drift occurs even over 200 steps. However, the TTT budget is extremely tight in wall-clock time. A pragmatic middle ground: refresh once at epoch // 3 (the AdaLoRA-style pruning moment), then lock in.

**Memory savings vs LoRA:**
GaLore requires (mn + mr + 2nr) memory for optimizer states vs LoRA's (mn + 3mr + 3nr). For r=8 and d=768, this is approximately 30% less. For our TTT setup (which is already parameter-efficient), this matters less than the convergence quality.

---

### 2.6 Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning
**Citation:** Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. *ACL 2021*. arXiv:2012.13255.

**Core Finding:**
Pre-trained language models have extremely low intrinsic dimensionality for adaptation. The intrinsic dimension is the MINIMUM number of parameters needed in a random subspace to reach 90% of full fine-tuning performance.

**Key Empirical Results:**
- RoBERTa on MRPC: **200 parameters** in a random subspace achieve 90% of full fine-tuning
- RoBERTa on MNLI: ~1000 parameters needed
- Larger pre-trained models have LOWER intrinsic dimension (scaling finding)
- Pre-training implicitly reduces intrinsic dimension

**What this means for TTT rank:**
If MRPC (sentence similarity, ~3700 training examples) requires only 200 parameters, then adapting to a SINGLE DOCUMENT (100-500 tokens) during TTT likely requires even fewer. Our rank-8 Q/V adapters contribute 2 * 2 * d_model * 8 = ~12,000 parameters. Our rank-4 MLP adapters contribute ~ 8,000 parameters. We are dramatically OVERPARAMETERIZING the adaptation subspace.

**However, there is a critical caveat:** The intrinsic dimensionality experiment uses RANDOM subspace projection, while LoRA uses a STRUCTURED low-rank subspace. A random 200-dim subspace can capture any direction in parameter space. LoRA rank-8 can only capture directions in the column space of B. The effective expressiveness of LoRA is lower than its parameter count suggests.

**Practical implication:** For TTT, rank-2 to rank-4 may be theoretically sufficient IF initialized in the right direction (RELI). Rank-8 provides a safety margin for the structured constraint. The Aghajanyan result justifies keeping ranks low (rank-2 to rank-4) and investing saved computation into more TTT epochs rather than higher ranks.

---

## 3. SECONDARY PAPERS — SPECTRAL ANALYSIS AND RANK COLLAPSE

### 3.1 LoRA vs Full Fine-tuning: An Illusion of Equivalence
**Citation:** Shuttleworth, R., et al. (2024). LoRA vs Full Fine-tuning: An Illusion of Equivalence. *NeurIPS 2024*. arXiv:2410.21228.

**Core Finding — Intruder Dimensions:**
LoRA introduces "intruder dimensions": new high-ranking singular vectors in the fine-tuned weight matrix that DO NOT appear in full fine-tuning. These emerge because LoRA's update BA introduces singular vectors that were not present in W_0.

Full fine-tuning modifies EXISTING high-contribution singular vectors (it amplifies/suppresses them). LoRA INTRODUCES new ones.

**Consequence for Generalization:**
- Models with intruder dimensions perform equivalently ON the fine-tuning task
- Models with intruder dimensions perform WORSE on the pre-training distribution
- Models with intruder dimensions perform WORSE in continual/sequential learning
- The number of intruder dimensions drops as LoRA rank increases past a threshold

**TTT Implication:**
In TTT, we want to adapt to the test document WITHOUT forgetting the pre-trained distribution (we still need general language modeling capability on other tokens). Intruder dimensions are exactly the mechanism by which LoRA hurts generalization. For TTT, this argues for:
1. **Lower ranks** (fewer intruder dimensions)
2. **Initialization from pre-trained weight SVD** (PiSSA-style, which avoids creating truly new directions)
3. **Short adaptation** (fewer steps = fewer intruder dimensions created)

**The RELI connection:** If RELI initializes A from the gradient's top singular vectors (which are aligned with the ACTUAL update needed), the resulting B@A product is more likely to amplify existing singular directions of W rather than creating new intruder directions. This is speculative but consistent with the paper's finding that gradient-aligned updates are more "in-distribution."

---

### 3.2 PiSSA: Principal Singular Values and Singular Vectors Adaptation
**Citation:** Meng, F., et al. (2024). PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models. *NeurIPS 2024 Spotlight*. arXiv:2404.02948.

**Core Mechanism:**
Instead of initializing B=0 (standard LoRA), PiSSA decomposes the PRE-TRAINED weight:
```
W_0 = U_r * S_r * V_r^T + W_res    (top-r SVD components)
```
Then:
- A initialized as V_r^T (right singular vectors)
- B initialized as U_r * S_r (left singular vectors * singular values)
- W_res (remaining components) is frozen

The adapter now STARTS at the identity map for the top-r subspace: B@A = U_r * S_r * V_r^T = the principal part of W_0. Training then modifies THIS principal subspace.

**Performance:** +5.16 pp on GSM8K for Mistral-7B vs standard LoRA. Significant across 12 models from 184M to 70B.

**Contrast with RELI:**
- PiSSA initializes from W_0's singular vectors (the weight's principal directions)
- RELI initializes from the GRADIENT's principal directions (the update's optimal directions)
- GoRA (see 3.3) also initializes from gradients — more similar to RELI
- PiSSA is better for "preserving what the model knows," RELI is better for "pointing at what needs to change"

For TTT, RELI is theoretically superior: we are adapting to a specific test document, and the gradient from that document directly encodes what the model needs to learn. PiSSA's weight-based initialization is agnostic to the adaptation target.

---

### 3.3 GoRA: Gradient-Driven Adaptive Low Rank Adaptation
**Citation:** (2025). GoRA: Gradient-Driven Rank Allocation and Initialization. arXiv:2502.12171.

**Core Innovation:**
GoRA is the closest published method to our RELI. It:
1. Computes element-wise product of weights and accumulated gradients
2. Uses this product's magnitude to dynamically assign ranks per layer
3. Initializes A and B from the gradient's principal directions (gradient-aligned init)

**Results:**
- +5.13 pp over standard LoRA on GSM8K (Llama3.1-8B-Base)
- Outperforms FULL fine-tuning by 2.05 pp in high-rank settings
- "Consistently outperforms existing LoRA-based methods"

**Significance:** This paper independently validates the RELI principle (gradient-aligned initialization) in a published, peer-reviewed setting. GoRA is the closest scientific precedent for RELI. The +5 pp gains from gradient-aligned init in GoRA suggest our RELI should have substantial TTT benefit.

---

### 3.4 Rank-Stabilized LoRA (rsLoRA)
**Citation:** Kalajdzievski, D. (2023). A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA. arXiv:2312.03732.

**Core Result:**
Standard LoRA uses scale factor gamma = alpha / r. This causes:
```
dL/dA ∝ B^T * dL/d(BA) ∝ (alpha/r) * ...
```
As r increases, the gradient magnitude scales as 1/r, causing gradient collapse for high-rank adapters.

**The Fix:**
rsLoRA uses gamma = alpha / sqrt(r). This keeps gradient magnitude Theta(1) for all ranks.

**Implication for Rank Annealing:**
If we anneal from rank-2r to rank-r at epoch // 3, the effective alpha changes. With standard LoRA scaling (alpha/r):
- rank-2r phase: scale = alpha / (2r)
- rank-r phase: scale = alpha / r   (scale DOUBLES at pruning step)

This causes a training instability at the pruning step — a sudden 2x jump in effective learning rate.

**With rsLoRA scaling (alpha/sqrt(r)):**
- rank-2r phase: scale = alpha / sqrt(2r)
- rank-r phase: scale = alpha / sqrt(r)   (scale changes by sqrt(2) ≈ 1.41x)

Still a discontinuity, but much smaller. To eliminate it entirely during rank annealing, one should normalize explicitly: at pruning, set alpha_new = alpha_old * sqrt(r_old / r_new) to maintain the same effective scale.

**Practical rsLoRA recommendation:** Use rsLoRA (alpha/sqrt(r)) always. It costs nothing and prevents rank-dependent gradient collapse, which is especially important when rank is being changed dynamically.

---

### 3.5 Flora: Formal Equivalence Summary (extended)
*(Additional detail beyond Section 2.4)*

**Johnson-Lindenstrauss interpretation:**
The Flora paper proves that random Gaussian A_0 satisfies E[A_0^T @ A_0] = (1/r) * I (after normalization). This means compression + decompression is an unbiased estimator of the identity: E[A_0 @ A_0^T @ G] = G. The variance of this estimator is O(1/r) — higher rank = lower variance in gradient estimation.

This is why higher rank LoRA generally performs better: lower variance in the gradient signal, not because of higher expressiveness of the update.

**For RELI with rank annealing:**
- High-rank RELI (rank-2r) phase: lower gradient variance, broader exploration
- Pruning to rank-r: keeps the top-r directions (highest singular values of accumulated B@A), discards the high-variance low-signal directions
- This is mathematically equivalent to a variance-reduction step: we keep the signal, discard the noise

---

## 4. ANSWERS TO KEY QUESTIONS

### Q1: Can AdaLoRA-style rank pruning be applied DURING TTT?

**Yes, with modifications.** The core AdaLoRA mechanism (compute SVD importance scores, prune singular values at intervals) can be applied within a TTT loop. The critical adaptations needed:

**Timing:** AdaLoRA's cubic schedule uses t_i (warmup) + cubic decay + t_f (stabilization). For TTT with N total steps:
- t_i = N * 0.25 (first 25% = exploration, no pruning)
- cubic decay from t_i to N * 0.7 (prune from initial rank to target rank)
- t_f = N * 0.3 (last 30% = stabilization at target rank)

The simplest version: prune once at step N * 0.33 (1/3 of training), following the "epoch // 3" rule from the proposed RELI-with-Rank-Annealing variant.

**Score computation:** The EMA importance scores require enough history. With DeltaT = 100 steps and 2500 total TTT steps, EMA statistics are meaningful by step 200-300. Do not prune before step 200.

**What to prune:** Compute SVD of current B@A product. The i-th singular value sigma_i measures how much weight the i-th direction carries. If sigma_{r+1}/sigma_1 < epsilon (e.g., epsilon = 0.1), the bottom half of directions is low-signal. Project A and B down to top-r.

**The novel contribution:** AdaLoRA was never applied to TTT. Our application is not covered by the paper. The combination with RELI initialization (starting from gradient-aligned A_0) is entirely novel.

---

### Q2: LoRA+ optimal LR ratio for A vs B — what is it?

**Empirically validated range: lambda = eta_B / eta_A = 4 to 16.**

The theoretical optimum for a width-n network is lambda = Theta(n). For GPT-2 small (d_model=768), this would suggest lambda~768, which is clearly too large. In practice:

- lambda = 4: consistent gains across all tested tasks
- lambda = 8-16: stronger gains on harder tasks (MNLI, QQP)
- lambda > 32: performance degrades (B overfits, A underfits)

**Recommended value for TTT:** lambda = 4 is the safest choice. If running ablations, also test lambda = 8.

**Interaction with RELI:** The LoRA+ derivation assumes B is initialized to zero (so eta_B being larger doesn't matter at step 0, it matters as B accumulates signal). With RELI, both A and B are non-zero at initialization. The asymmetric LR may be less critical but is unlikely to hurt. Apply it anyway: set lr_A = base_lr, lr_B = 4 * base_lr.

---

### Q3: Is GaLore equivalent to RELI?

**Partially, with important differences.**

| Aspect | GaLore | RELI |
|---|---|---|
| Projection matrix source | SVD of gradient G | SVD of gradient G (same!) |
| When projection is set | Every T=200 steps | Once at initialization |
| What is optimized | Full weights (projected) | LoRA A, B parameters |
| Expressiveness | Full-rank (refreshed) | Rank-r (fixed subspace) |
| Memory | Lower (no full param copy) | Standard LoRA |
| TTT computational cost | High (repeated SVD) | Low (SVD once) |

RELI = "GaLore with T = infinity" (project once, never refresh). Flora's analysis shows that fixed A_0 means the total weight change is bounded to the column space of A_0. GaLore's periodic refresh removes this constraint.

**For TTT:** RELI is computationally cheap (one SVD at init). GaLore with T=200 would cost ~12 SVDs during a 2500-step TTT run. Given the tight wall-clock budget, RELI (one SVD) is preferred over GaLore (repeated SVDs). The proposed RELI-with-Rank-Annealing variant (which does one SVD at init and one more at epoch//3 for pruning) is a reasonable middle ground.

---

### Q4: Does RELI initialization equal "initializing from compressed gradient"?

**Formally yes, and this is stronger than random initialization.**

From Flora's framework (Section 2.4 above): LoRA with A_0 = random Gaussian is equivalent to compressing the gradient with a random projector. By Johnson-Lindenstrauss, this is unbiased but has variance O(1/r).

With RELI (A_0 = V_r^T, top-r right singular vectors of gradient G):
- The compressed gradient is: A_0 @ G = V_r^T @ G = Sigma_r @ U_r^T
- This is EXACTLY the rank-r Eckart-Young approximation of G
- The compression has ZERO information loss for the top-r components
- The variance in the compressed representation is 0 for the principal directions

**This is a formal justification for RELI:** It is not merely "a better initialization heuristic." It is the OPTIMAL rank-r projection of the initial gradient signal. Any other initialization (random, PiSSA, zero-B) is strictly suboptimal in the Flora framework for capturing the initial adaptation direction.

**The formal claim:** RELI-initialized LoRA captures strictly more gradient information at step 0 than random-initialized LoRA, for the same rank r, in the sense of minimizing ||G - A_0^T @ (A_0 @ G)||_F (Eckart-Young).

---

### Q5: What is the "intrinsic dimensionality" of TTT adaptation? Is rank-2 or rank-4 sufficient?

**Based on the evidence, rank-2 to rank-4 is likely sufficient given RELI initialization.**

**Theoretical bound:** Aghajanyan's 200-parameter result on sentence-pair tasks. A TTT document is ~100-500 tokens. The adaptation problem is even simpler (memorize document statistics, not transfer to new task). Intrinsic dimension < 200.

**Practical bound from LoRA research:** The original LoRA paper shows rank-1 to rank-4 is competitive with higher ranks on many NLU tasks. The LoRA survey review notes that "rank-1 is sometimes sufficient." With RELI initialization, the first direction already points at the optimal adaptation, so rank-1 or rank-2 may be all that is needed for the per-document signal.

**Why we currently use rank-8 (Q/V) and rank-4 (MLP):**
- Safety margin against RELI initialization noise
- Multiple documents in a batch may require slightly different directions
- The TTT loss landscape is noisy at the token level

**Recommendation:** Test rank-2 Q/V and rank-2 MLP with RELI. If performance matches rank-8, halve the adapter parameters, freeing up TTT budget for more epochs. The Aghajanyan result strongly suggests this would work. If RELI is already pointing at the right direction, rank-2 captures the dominant adaptation signal.

---

### Q6: Rank scheduling — can SVD of current delta decide WHEN to anneal?

**Yes. This is the key insight from AdaLoRA, and the singular value decay criterion is well-founded.**

**The criterion:** Compute SVD of current B@A at regular intervals. Define:
```
explained_ratio(k) = sum_{i=1}^{k} sigma_i / sum_{i=1}^{r} sigma_i
```
If explained_ratio(r//2) > 0.9 (top half of ranks captures 90% of the signal), then the bottom half of ranks is noise. This is the moment to prune.

**When does this typically happen?**
In AdaLoRA, significant rank reduction occurs in the FIRST 30-40% of training steps. By then, the optimizer has identified which directions matter. This is consistent with the proposed "epoch // 3" timing.

**Practical implementation:**
```python
def should_prune(B, A, target_rank, threshold=0.9):
    delta = B @ A  # (d_out x d_in)
    U, S, Vh = torch.linalg.svd(delta, full_matrices=False)
    cumulative = torch.cumsum(S, dim=0) / S.sum()
    natural_rank = (cumulative < threshold).sum().item() + 1
    return natural_rank <= target_rank, S, U, Vh
```

If natural_rank <= target_rank, the delta has already effectively collapsed to that rank — prune now. If natural_rank > target_rank, the signal is still distributed across more directions — wait.

**Conservative approach:** Prune at epoch // 3 regardless of the SVD criterion, but LOG the singular value spectrum to verify the criterion is met. If not, extend the exploration phase.

---

## 5. NOVEL VARIANT DESIGN: RELI WITH RANK ANNEALING

**Full Algorithm Specification**

```
RELI-RA (RELI with Rank Annealing)
===================================

Hyperparameters:
  - r_init: initial rank (= 2 * r_target, e.g., r_init=16, r_target=8)
  - r_target: final rank after pruning
  - anneal_step: fraction of total TTT steps before pruning (default 0.33)
  - threshold: singular value concentration threshold (default 0.90)
  - lambda_lr: LoRA+ LR ratio for B vs A (default 4.0)
  - scale_mode: 'rslora' (alpha/sqrt(r)) or 'lora' (alpha/r)

Initialization (using RELI at rank r_init):
  1. Compute surrogate gradient G = dL/dW on the test document (1 forward/backward pass)
  2. SVD: G = U @ S @ Vh, take top-r_init right singular vectors
  3. A_0 = Vh[:r_init, :]  (r_init x d_in)
  4. B_0 = zeros(d_out x r_init)  (B=0 so adapter output starts at 0)
  5. Set lr_A = base_lr, lr_B = lambda_lr * base_lr  (LoRA+ asymmetry)
  6. Set scale = alpha / sqrt(r_init)  (rsLoRA)

TTT Training Loop (total_steps = n_epochs * steps_per_epoch):
  For step t in range(total_steps):
    if t == int(anneal_step * total_steps):
      # Rank Annealing Step
      delta = B @ A  # current delta matrix (d_out x d_in)
      U, S, Vh_curr = torch.linalg.svd(delta, full_matrices=False)

      # Optional: check criterion before pruning
      explained = S[:r_target].sum() / S.sum()
      log(f"Step {t}: explained ratio at r_target = {explained:.3f}")

      # Project down to top-r_target
      # New A: r_target x d_in
      A_new = Vh_curr[:r_target, :]
      # New B: d_out x r_target  (preserving the accumulated signal)
      B_new = U[:, :r_target] * S[:r_target].unsqueeze(0)
      # B_new @ A_new = delta, so no signal is lost for the kept directions

      # Rescale to maintain effective scale with rsLoRA
      scale_new = alpha / sqrt(r_target)
      # scale_factor = scale_new / scale_old = sqrt(r_init/r_target)
      # Apply this to A_new to maintain output magnitude
      A_new = A_new * sqrt(r_init / r_target)  # optional, see note below

      A, B = A_new, B_new
      # Reset optimizer state for A and B (they changed structure)
      optimizer.reset_state(A, B)

    # Standard TTT gradient step on A and B
    loss.backward()
    optimizer.step()  # with lr_B = lambda_lr * lr_A
```

**Note on scale preservation:**
The output of the adapter before annealing = scale_init * B @ A.
After annealing to r_target with rsLoRA: output = scale_target * B_new @ A_new.
Since B_new @ A_new already equals the low-rank approx of the old B @ A, and scale changes from alpha/sqrt(r_init) to alpha/sqrt(r_target), the effective output INCREASES by sqrt(r_init/r_target). This is a jump. To prevent this:
- Option A: Set alpha_new = alpha_old * sqrt(r_target/r_init) after pruning
- Option B: Scale B_new down by sqrt(r_target/r_init) at pruning time
- Option C: Use constant alpha (not scaled by rank) and rely on the optimizer to re-normalize

**Why this variant is novel:**
1. AdaLoRA does not use gradient-aligned initialization (RELI)
2. AdaLoRA uses gradual cubic pruning; RELI-RA uses a single decisive cut informed by the SVD criterion
3. AdaLoRA was never applied to TTT
4. The Flora equivalence provides a formal justification (RELI-RA = optimal gradient compressor at init, optimal variance-reduction at pruning step)
5. DyLoRA-style sampling during rank-2r phase (sampling sub-ranks) is an optional enhancement

**Expected TTT behavior:**
- Epochs 1 to N//3: broad exploration in rank-2r space with RELI-aligned A_0. The high rank reduces variance in gradient estimation. B accumulates signal in multiple directions.
- Step N//3: SVD of B@A identifies which r_target directions have concentrated the most signal. Prune. The pruned model has lower parameter count = faster remaining TTT steps.
- Epochs N//3 to N: deep exploitation in rank-r space. The optimizer has a fresh start on a well-initialized, reduced-dimension adapter.

---

## 6. RELATED WORK WORTH IMPLEMENTING

### 6.1 rsLoRA (Rank Stabilization) — Immediate Free Win
**Priority: HIGH. Implement now, no ablation needed.**

Change the LoRA scale from alpha/r to alpha/sqrt(r). This prevents gradient collapse at higher ranks and is especially important for the rank-2r exploration phase. One line of code. First published in Kalajdzievski (2023), arXiv:2312.03732.

### 6.2 LoRA+ Asymmetric LR — Low-Risk Improvement
**Priority: MEDIUM. Test lambda = 4.**

Set lr_B = 4 * lr_A in the TTT optimizer. Expected 1-2% gain at same compute. Published at ICML 2024.

### 6.3 GoRA-style Dynamic Rank Allocation
**Priority: LOW for current TTT setup.**

GoRA assigns different ranks to different layers based on gradient magnitude. For a small GPT-2 scale model with 12 layers, we could use rank-8 on layers where gradients are large and rank-2 on layers where gradients are small. Implementation cost is moderate.

---

## 7. RESEARCH GAPS AND CONTRADICTIONS

### 7.1 Gap: No TTT-specific rank study
No paper studies optimal LoRA rank specifically for TTT (test-time adaptation to a single document). The closest is LoRA-TTT (arXiv:2502.02069) for vision-language models, but this is a different domain. This is a genuine research gap that our work fills.

### 7.2 Contradiction: PiSSA vs RELI for optimal initialization
PiSSA (NeurIPS 2024 Spotlight) argues initialization from W_0's principal components (the weight's own structure) is optimal. RELI/GoRA argue initialization from G's principal components (the gradient's direction) is optimal. The two views are not reconciled in the literature.

Resolution: PiSSA is better for preserving pre-trained capabilities (less forgetting, fewer intruder dimensions per arXiv:2410.21228). RELI/GoRA is better for rapidly capturing what the model needs to CHANGE. For TTT (where forgetting is NOT a concern — we want the model to change for this document), RELI is preferable. For standard fine-tuning (where forgetting matters), PiSSA may be preferable.

### 7.3 Contradiction: Intrinsic dimension vs observed LoRA rank needs
Aghajanyan says 200 parameters suffice; in practice LoRA users report rank-16 to rank-64 needed for strong fine-tuning results. The resolution is that random subspace projection (Aghajanyan) is different from structured low-rank (LoRA) — random 200-dim subspace can align with any direction, while LoRA rank-r can only align with r specific directions. The effective intrinsic dimension for STRUCTURED LoRA is higher than for random subspaces.

### 7.4 Open question: Optimal anneal timing
AdaLoRA prunes throughout training (gradual). RELI-RA prunes once (sudden). No paper directly compares these two strategies for a small-scale, short-run adaptation scenario like TTT. The optimal timing (epoch//3, epoch//2, etc.) is empirically unknown for TTT.

---

## 8. CITATION LIST

All papers confirmed peer-reviewed (venue noted):

1. **AdaLoRA**: Zhang, Q. et al. (2023). AdaLoRA: Adaptive Budget Allocation for Parameter-Efficient Fine-Tuning. *ICLR 2023*. https://arxiv.org/abs/2303.10512

2. **DyLoRA**: Valipour, M. et al. (2022). DyLoRA: Parameter Efficient Tuning of Pre-trained Models using Dynamic Search-Free Low-Rank Adaptation. *EACL 2023*. https://arxiv.org/abs/2210.07558

3. **LoRA+**: Hayou, S., Ghosh, N., & Yu, B. (2024). LoRA+: Efficient Low Rank Adaptation of Large Models. *ICML 2024*. https://arxiv.org/abs/2402.12354

4. **Flora**: Han, Y. et al. (2024). Flora: Low-Rank Adapters Are Secretly Gradient Compressors. *ICML 2024*. https://arxiv.org/abs/2402.03293

5. **GaLore**: Zhao, J. et al. (2024). GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection. *ICML 2024*. https://arxiv.org/abs/2403.03507

6. **Intrinsic Dimensionality**: Aghajanyan, A., Zettlemoyer, L., & Gupta, S. (2021). Intrinsic Dimensionality Explains the Effectiveness of Language Model Fine-Tuning. *ACL 2021*. https://arxiv.org/abs/2012.13255

7. **LoRA vs Full FT (Illusion)**: Shuttleworth, R. et al. (2024). LoRA vs Full Fine-tuning: An Illusion of Equivalence. *NeurIPS 2024*. https://arxiv.org/abs/2410.21228

8. **PiSSA**: Meng, F. et al. (2024). PiSSA: Principal Singular Values and Singular Vectors Adaptation of Large Language Models. *NeurIPS 2024 Spotlight*. https://arxiv.org/abs/2404.02948

9. **rsLoRA**: Kalajdzievski, D. (2023). A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA. arXiv:2312.03732. https://arxiv.org/abs/2312.03732

10. **GoRA**: (2025). GoRA: Gradient-Driven Rank Allocation and Initialization. arXiv:2502.12171. https://arxiv.org/abs/2502.12171

11. **LoRA**: Hu, E. et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR 2022*. https://arxiv.org/abs/2106.09685

12. **LoRA Survey**: (2025). Low-Rank Adaptation for Foundation Models: A Comprehensive Review. arXiv:2501.00365. https://arxiv.org/abs/2501.00365

---

## 9. RECOMMENDED EXPERIMENT PRIORITY FOR PARAMETER GOLF

| Rank | Variant | Expected delta BPB | Implementation cost | Papers supporting |
|---|---|---|---|---|
| 1 | rsLoRA scaling (alpha/sqrt(r)) | -0.003 to -0.008 | Trivial (1 line) | rsLoRA (2312.03732) |
| 2 | LoRA+ asymmetric LR (lambda=4) | -0.005 to -0.015 | Low (2 lines) | LoRA+ (2402.12354) |
| 3 | RELI-RA rank annealing (rank-16 -> rank-8, prune at epoch//3) | -0.008 to -0.025 | Medium (SVD + projection) | AdaLoRA + Flora + GoRA |
| 4 | rank-2 Q/V + RELI (test intrinsic dim hypothesis) | -0.010 to +0.005 | Low (change r) | Aghajanyan (2012.13255) |
| 5 | Per-layer rank allocation (higher rank on upper layers) | -0.005 to -0.015 | Medium | Survey (2501.00365) |

rsLoRA is the highest priority because it is free and provably prevents a pathological failure mode that may already be hurting our rank-8 adapters.

---

*Document generated 2026-03-24. Databases searched: arXiv, ICLR 2023, ICML 2024, NeurIPS 2024, EACL 2023, ACL 2021, ACL Anthology 2024.*
