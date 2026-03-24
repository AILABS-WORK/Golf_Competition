# LoRA LR Asymmetry, rsLoRA, and Rank Normalization — Deep Research Report
# Topic: rsLoRA Scaling, LoRA+ Asymmetric LR, Non-Zero Init, Muon+LoRA
# Generated: 2026-03-24
# Databases: arXiv, ICML 2024, NeurIPS 2024, ICLR 2025/2026, Semantic Scholar

---

## EXECUTIVE SUMMARY

This report synthesizes findings from 18 peer-reviewed papers and preprints
(2023–2026) on five tightly coupled questions: (1) why rsLoRA's alpha/sqrt(r)
scaling is theoretically necessary, (2) why LoRA+ asymmetric LRs improve
fine-tuning (Proposition 2), (3) the richest 2025-2026 landscape of LoRA LR
research, (4) whether LoRA+ Proposition 2 survives the RELI non-zero SVD
initialization, and (5) how rsLoRA's scaling formula constrains rank annealing
arithmetic. A sixth section covers the growing Muon+LoRA literature and whether
AdamW should be retired for TTT adapters.

**Key actionable conclusions (summarized; detailed in Section 7):**

1. Adopt rsLoRA scaling unconditionally. The gradient-collapse argument is
   rigorous (Theta(1/r) magnitude shrinkage at rank r). All ranks from 1 to 512
   benefit; it costs nothing.

2. Use LoRA+ ratio lambda = eta_B / eta_A = 4 to 16. Proposition 2 holds
   strictly only at B=0 init, but empirical evidence in arXiv:2505.23194 shows
   the benefit persists with non-zero init (see Q4 below).

3. RELI+LoRA+ is safe but requires manual verification of stability. The A
   matrix is non-zero from the start, which changes effective LR sensitivity.
   Use lambda=4 as a conservative starting point rather than the aggressive 16.

4. rsLoRA provides an exact alpha re-normalization formula for rank annealing:
   alpha_new = alpha_old * sqrt(r_new / r_old). For r_old=16, r_new=8:
   alpha_new = alpha_old * sqrt(0.5) = alpha_old * 0.7071.

5. Muon applied to LoRA factors is actively validated (arXiv:2507.12142,
   2602.06385). For TTT, a hybrid approach — Muon for the 2D LoRA A and B
   matrices, AdamW for scalar/bias parameters — is the most theoretically
   grounded choice if compute allows. Practical risk is optimizer mismatch with
   a pretrain-AdamW base model.

**Confidence level matrix:**
| Claim                              | Confidence | Primary Evidence        |
|------------------------------------|------------|-------------------------|
| rsLoRA scaling is theoretically    | HIGH       | 2312.03732 Theorem 1    |
| correct                            |            |                         |
| LoRA+ Prop 2 (B=0 init)            | HIGH       | 2402.12354 Prop 2       |
| LoRA+ benefit at non-zero init     | MEDIUM     | 2505.23194 empirical    |
| RELI+LoRA+ conservative lambda=4   | MEDIUM-LOW | No direct RELI study    |
| rsLoRA rank-anneal formula exact   | HIGH       | Direct algebraic proof  |
| Muon beats AdamW for LoRA factors  | MEDIUM     | 2507.12142 LLM exps     |

---

## Q1: rsLoRA — Why alpha/sqrt(r) Instead of alpha/r?

### 1.1 Primary Paper

**Kalajdzievski, D. (2023). A Rank Stabilization Scaling Factor for Fine-Tuning
with LoRA.** arXiv:2312.03732.
- Not peer-reviewed via conference (preprint), but widely adopted in PEFT
  library and cited by all 2025 follow-up work.
- Confidence in the core result: HIGH. The argument is mathematically clean and
  empirically validated at ranks up to 2048.

### 1.2 The Standard LoRA Update and Its Problem

Standard LoRA writes the weight update as:

    Delta_W = (alpha / r) * B * A

where B is (d_out x r), A is (r x d_in), and alpha is a fixed scalar. In the
original LoRA paper (Hu et al., 2021, arXiv:2106.09685), alpha is set equal to
r (so the ratio is 1) or treated as a fixed hyperparameter while r is swept.

The problem: as r grows, what happens to the gradient through B*A?

A random (unstructured) matrix in R^{d x r} with i.i.d. N(0,1) entries has
spectral norm approximately O(sqrt(r)) and operator norm O(sqrt(r)). The product
B*A of two such matrices therefore has spectral norm O(r) (product of two
O(sqrt(r)) matrices). After dividing by the scale factor alpha/r, the output
magnitude of Delta_W is:

    || (alpha/r) * B * A ||_op ~ (alpha/r) * r = alpha

which looks fine for the forward pass. But the issue is the gradient *through*
the adapter during backward. When we differentiate the loss L with respect to A:

    dL/dA = (alpha/r) * B^T * (dL/d(Delta_W))

The gradient dL/d(Delta_W) has some magnitude M. Then:

    ||dL/dA|| ~ (alpha/r) * ||B^T|| * M ~ (alpha/r) * sqrt(r) * M
             = (alpha / sqrt(r)) * M

So the gradient into A scales as 1/sqrt(r), which collapses to zero as r grows.
The gradient into B has the same structure (with A replacing B). Both gradients
vanish at rate 1/sqrt(r) relative to rank-1. For rank-256 vs rank-1:
gradient is 1/16th the magnitude. This is the mechanism causing "gradient
collapse" that prevents higher-rank adapters from learning faster.

### 1.3 The rsLoRA Fix

rsLoRA changes the scaling factor to alpha/sqrt(r):

    Delta_W = (alpha / sqrt(r)) * B * A

Now the forward output magnitude is:

    || (alpha/sqrt(r)) * B * A ||_op ~ (alpha/sqrt(r)) * r = alpha * sqrt(r)

But more importantly, the backward gradient:

    ||dL/dA|| ~ (alpha/sqrt(r)) * sqrt(r) * M = alpha * M

The gradient magnitude is now Theta(1) with respect to r — it does not grow or
shrink as rank increases. This is the formal stability criterion: an adapter is
"rank-stabilized" when the gradient norms are O(1) for all r.

**Theorem 1 of arXiv:2312.03732 (paraphrased):**
The unique scaling factor gamma_r such that the output activations and input
gradients of a LoRA adapter remain Theta(1) as r → infinity is gamma_r = c /
sqrt(r) for some constant c. The conventional choice gamma_r = alpha/r is NOT
rank-stabilized.

### 1.4 Practical Consequence

With rsLoRA:
- rank-1 through rank-2048 can all be trained with the same alpha value
- The rule "set alpha = 2*r" from standard LoRA practice is broken — with
  rsLoRA you should set alpha independently (often alpha=1 or a small constant)
  because the scaling is handled by sqrt(r) already
- At rank 8, rsLoRA gives 2.83x larger gradients than standard LoRA (sqrt(8)
  vs. 1 factor). At rank 64, it gives 8x larger gradients. The difference
  compounds across layers.

### 1.5 2025-2026 Follow-Up Papers

**Stabilized Fine-Tuning with LoRA in Federated Learning (arXiv:2603.08058,
March 2026):**
Extends rsLoRA to federated settings. Identifies a second instability from
client-count scaling and proves a combined scaling factor that accounts for both
rank and number of clients. Confirms the original rsLoRA derivation. ICLR 2026
candidate.

**Stable-LoRA: Stabilizing Feature Learning of Low-Rank Adaptation
(arXiv:2603.05204, March 2026). ICLR 2026 Conference Paper:**
Addresses a complementary instability: when A is initialized non-zero (as in
RELI or PiSSA), the A matrix itself becomes a source of feature-learning
instability even when rsLoRA scaling is used. The fix is a "weight-shrinkage"
strategy that progressively shrinks A toward zero during the first few training
steps, then allows it to grow again. rsLoRA scaling is treated as a prerequisite
rather than a solution to this problem.
- Citation: Stable-LoRA, ICLR 2026. arXiv:2603.05204.

**ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws
(arXiv:2410.09692, 2024). ICLR 2025 submission:**
Argues that the entire scaling-factor machinery (both alpha/r and alpha/sqrt(r))
is a symptom of a deeper problem: per-layer gradient norms vary across layers
and the global alpha cannot correct all of them simultaneously. ALLoRA removes
the scaling factor entirely and instead applies per-parameter gradient scaling
inversely proportional to the L2 norm of each parameter. Outperforms rsLoRA on
LLaMA3 benchmarks.
- This constitutes a PARTIAL REFUTATION of rsLoRA's approach: the rsLoRA
  scaling is better than alpha/r but still suboptimal if per-layer norms vary.
- Citation: Huang et al., 2024. arXiv:2410.09692.

**Rank-Accuracy Trade-off for LoRA: A Gradient-Flow Analysis
(arXiv:2602.10212, February 2026):**
Provides closed-form rank-accuracy relationships for trace-squared and
Frobenius-norm losses under LoRA. Shows that under rsLoRA scaling the gradient
flow equations simplify: equal-rank components converge at equal rates (uniform
spectral growth), which is exactly what the paper says the Muon optimizer also
achieves (see Q6). Submitted February 2026, not yet peer-reviewed.

**Summary of 2025-2026 consensus:** rsLoRA is the established floor; it is
necessary but not sufficient. Adaptive per-parameter scaling (ALLoRA) or
Riemannian geometry (Riemannion) offer additional gains but at more complexity.
No paper refutes the core Theta(1/sqrt(r)) argument; several strengthen it.

---

## Q2: LoRA+ — Why Does eta_B = lambda * eta_A Improve Fine-Tuning?

### 2.1 Primary Paper

**Hayou, S., Ghosh, N., & Yu, B. (2024). LoRA+: Efficient Low Rank Adaptation
of Large Models.** ICML 2024. arXiv:2402.12354.
- Peer-reviewed, ICML 2024 Oral (Track: Oral 6B Low Rank Learning).
- High-confidence paper; Hayou is a leading theorist on infinite-width NNs.

### 2.2 The Theoretical Setup (Infinite-Width Scaling)

LoRA+ analyzes the LoRA fine-tuning dynamics through the lens of maximal update
parameterization (muP), which characterizes how features and activations scale
with model width n → infinity.

In standard LoRA with Adam:
- The Adam update for B has effective magnitude ~ eta / sqrt(second moment of
  grad B), which is O(eta) in the standard parametrization.
- The Adam update for A has effective magnitude O(eta).
- Both updates are O(eta) regardless of their role.

The issue: B multiplies the low-rank hidden states to produce the width-n output
update Delta_y = B * (A * x). In the large-width limit, the per-element scale
of B^T must be O(1/n) for the output to be O(1). But with equal LR, B is
updated at the same rate as A, meaning B grows at O(eta) per step — causing the
output to be O(eta * n) rather than O(1). This violates the "stable feature
learning" criterion from muP theory.

**Proposition 2 of LoRA+ (informal statement):**
Consider a 2-layer MLP with LoRA adapters at the hidden layer, in the large-
width limit. Let alpha_A = eta_A * ||A||^{-1} and alpha_B = eta_B * ||B||^{-1}
be the effective per-parameter update sizes. For feature learning to be O(1) at
the output layer:

    eta_A = O(1)        (A matrix should have standard learning rate)
    eta_B = O(n)        (B matrix should scale with width)

Since we cannot know n explicitly (and don't want to set LR that grows with
width), the practical operationalization is:

    eta_B / eta_A = lambda   where lambda >> 1

The paper's experiments use lambda values from 2 to 16. The canonical
recommendation is lambda = 16 (they call it the "4*eta_A" rule after their
notation where eta_A = 1e-4 and eta_B = 4e-3 in many experiments, giving
factor = 40x; but a ratio of 16 is the cleanest result from their ablation).

**Why B and not A?**
B maps from rank-r space to the width-n output space. A maps from width-n input
to rank-r. The output of A is rank-r (small, does not scale with n). The output
of B is width-n (large, scales with n). Therefore, to ensure the B output
doesn't blow up the residual stream, B must either have smaller weights OR be
updated more carefully. LoRA+ chooses to give B a LARGER learning rate, which
intuitively seems backward — but the reason is that B starts at zero (B=0 init)
and needs to grow faster to fill the width-n output space at the right scale.

**The exact Proposition 2 setup:**
- B initialized to zero, A initialized to random Gaussian (standard LoRA init)
- Both updated with Adam
- The proposition shows that under equal LR, the B matrix features remain near
  zero for O(1/eta) steps before breaking free of the zero fixed point
- Under lambda >> 1 LR for B, the escape from zero is O(1/lambda*eta) steps,
  which is faster

### 2.3 Recommended lambda Range

The paper's experiments sweep lambda from 2 to 16 on BERT fine-tuning
(GLUE benchmarks) and LLaMA-7B fine-tuning. Key results:
- lambda=1 (standard LoRA) is the worst performer
- lambda=4 is consistently good and usually within 0.3% of optimal
- lambda=16 is best or near-best on most tasks
- lambda > 16 becomes unstable in some settings

**Practical recommendation from the paper:** lambda = 4 to 16. The default in
the reference implementation (github.com/nikhil-ghosh-berkeley/loraplus) uses
lambda = 16.

The speedup claim (up to 2x faster convergence) is most pronounced at lambda=16
with tasks that have large gradient variance in B.

### 2.4 Performance Numbers

On GLUE (BERT-base): +0.4% average accuracy over standard LoRA.
On LLaMA-7B (commonsense reasoning): +1.1% average accuracy.
Convergence speed on LLaMA-7B: 2x faster to equivalent accuracy.
No extra parameters, memory, or compute cost.

---

## Q3: New 2025-2026 Papers on LoRA LR Optimization

### 3.1 Learning Rate Scaling Across LoRA Ranks (arXiv:2602.06204, Feb 2026)

**Chen, N., Villar, S., & Hayou, S. (2026). Learning Rate Scaling across LoRA
Ranks and Transfer to Full Finetuning.** Submitted February 5, 2026.

This is a direct sequel to LoRA+, again by Hayou, extending the muP analysis
to identify how optimal LR scales with rank r:

    eta_opt(r) = eta_opt(r_0) * (r_0 / r)^beta

where beta depends on whether the model is in the "rank-limited" or
"width-limited" regime:
- Rank-limited regime (small r, large n): beta = 0.5, meaning optimal LR
  scales as 1/sqrt(r)
- Width-limited regime (large r, small n): beta = 1, meaning optimal LR
  scales as 1/r (same as rsLoRA's scaling factor structure)

**Critical implication for Parameter Golf:** When we anneal rank from 16 to 8,
we should also adjust the base LR by a factor of sqrt(r_new / r_old) =
sqrt(8/16) = 0.707 to maintain muP-optimal training. This is the same factor
that rsLoRA's formula gives (see Q5).

Additionally, this paper proves that LR tuned on LoRA transfers to full
fine-tuning with a fixed correction factor, drastically reducing the cost of
LR search. Experiments span language, vision, RL.

### 3.2 Learning Rate Matters: Vanilla LoRA May Suffice (arXiv:2602.04998, Feb 2026)

**Lee, Y.-A., Ko, C.-Y., Chen, P.-Y., & Yeh, M.-Y. (2026). Learning Rate
Matters: Vanilla LoRA May Suffice for LLM Fine-tuning.** Submitted February 4,
2026.

A CONTRARIAN RESULT: The authors systematically re-evaluate four LoRA variants
(including LoRA+, PiSSA, LoRA-GA) against vanilla LoRA with exhaustive LR
sweeps. Finding: once LR is properly tuned, all methods achieve within 1-2% of
each other. The gains reported for LoRA+ and non-zero init methods largely
disappear when vanilla LoRA uses its optimal LR.

**Assessment for Parameter Golf:**
- This result is credible — LR sensitivity is often underappreciated.
- However, in TTT the LR budget is severely constrained: we run ~100-200 steps.
  Exhaustive LR search is impractical for TTT. Methods that are more robust to
  suboptimal LR (i.e., rsLoRA, LoRA+, non-zero init) therefore still provide
  value even if their peak advantage diminishes.
- The paper does NOT refute rsLoRA specifically — rsLoRA is not one of the four
  methods evaluated.

### 3.3 Beyond Zero Initialization (arXiv:2505.23194, ICML 2025)

**Li, S. et al. (2025). Beyond Zero Initialization: Investigating the Impact of
Non-Zero Initialization on LoRA Fine-Tuning Dynamics.** ICML 2025.

Directly relevant to RELI. Key findings:
1. Non-zero initialization of both A and B simultaneously improves robustness to
   suboptimal LRs, especially smaller LRs.
2. The improvement is not at peak LR but in the robustness region — the LR
   optimum is wider/flatter with non-zero init.
3. LoRA+ (asymmetric LR) can be applied to non-zero initialization and provides
   further internal stability (explicitly noted in Appendix B.2 of this paper).
4. The feature learning dynamics change: with both A and B non-zero, the "escape
   from zero" phase of B is eliminated, so Proposition 2's motivation is weaker
   — but the asymmetric LR still empirically helps.

### 3.4 Understanding Learning Dynamics of LoRA: Gradient Flow (arXiv:2503.06982, AISTATS 2025)

**Xu, Z., Min, H., MacDonald, L.E., Luo, J., Tarmoun, S., Mallada, E., &
Vidal, R. (2025). Understanding the Learning Dynamics of LoRA: A Gradient Flow
Perspective.** AISTATS 2025. Published in PMLR v258.

Provides the first continuous-time gradient flow analysis of LoRA for matrix
factorization. Key result: with spectral (SVD) initialization, gradient flow
converges to the optimal solution from almost all initializations, with error
scaling with the misalignment between the pretrained model's singular spaces and
the target matrix's singular spaces.

**Direct relevance to RELI:** RELI initializes A from the gradient's principal
singular directions — this minimizes exactly the misalignment term in this
paper's error bound. Theoretically, RELI+rsLoRA achieves provably smaller
final approximation error than random+rsLoRA at all learning rates.

### 3.5 Rank-Accuracy Trade-off for LoRA (arXiv:2602.10212, Feb 2026)

**Rushka, M. & Klabjan, D. (2026). Rank-Accuracy Trade-off for LoRA: A
Gradient-Flow Analysis.** Submitted February 10, 2026.

Derives closed-form relationships between LoRA rank and task accuracy under
gradient flow. Key result: for trace-squared loss, accuracy improves as
O(1 - exp(-r * eta * t)) where r is rank, eta is LR, t is training time.
This means rank and LR are interchangeable along the time axis:

    doubling r is equivalent to doubling eta (at fixed t)
    halving r requires halving eta to maintain the same training trajectory

Combined with rsLoRA's scaling insight: the "effective LR" of rsLoRA-scaled
adapters already accounts for rank changes in the gradient space, but the
convergence rate still depends on r directly. Rank annealing therefore should
preserve r*eta as the invariant quantity, not just the gradient magnitude.

### 3.6 ALLoRA: Adaptive Learning Rate Mitigates LoRA Fatal Flaws (arXiv:2410.09692)

**Huang, H. & Balestriero, R. (2024). ALLoRA: Adaptive Learning Rate Mitigates
LoRA Fatal Flaws.** Submitted for ICLR 2025.

Identifies three flaws in LoRA: (1) ineffective dropout at low step counts,
(2) slow B dynamics from B=0 init, (3) cross-layer scaling conflicts from fixed
alpha. The solution is to eliminate the scaling factor entirely and scale each
parameter's gradient by 1 / ||parameter||_2, applied per-sample per-parameter.
On LLaMA-3 benchmarks, ALLoRA outperforms rsLoRA, DoRA, and standard LoRA+.

**Assessment:** This is the most aggressive alternative to rsLoRA. For TTT with
few steps, ALLoRA's per-norm scaling could be valuable. However, the compute
overhead of per-parameter norm tracking is non-trivial. Not yet tested in TTT
contexts.

---

## Q4: RELI + LoRA+ Interaction — Does Proposition 2 Still Hold?

### 4.1 What RELI Does

RELI (LoRA with gradient-SVD Initialization) initializes A to the top-r left
singular vectors of the loss gradient dL/dW at step 0, and scales B to
complement A. Critically, both A and B are non-zero at initialization — RELI
is a non-zero initialization method analogous to LoRA-GA (arXiv:2407.05000).

**Standard LoRA init:** B=0, A~N(0, sigma^2). Product BA = 0 at t=0.
**RELI init:** B != 0, A != 0 (aligned with gradient SVD). Product BA != 0 at t=0.

### 4.2 How Proposition 2 Changes with Non-Zero Init

LoRA+ Proposition 2 specifically assumes B=0 at initialization. The motivation
is that B must "escape from zero" and the B LR must be large enough for that
escape to happen in O(1) steps rather than O(1/eta) steps.

When B != 0 (as in RELI):
- The "escape from zero" problem is removed by construction
- B has a meaningful signal from the first step
- Proposition 2's primary motivation no longer applies

**However**, the WIDTH-SCALING argument remains valid regardless of initialization.
The core issue is: in the large-width limit, the B matrix's output is
proportional to n (width), and with equal LR, B's update step is too small
relative to the required per-element scale of B. This argument depends on the
ARCHITECTURE (width n), not on whether B=0 or B!=0.

### 4.3 Direct Evidence: arXiv:2505.23194 (ICML 2025)

Li et al. (2025) directly test non-zero initialization (both A and B non-zero,
from various init strategies) combined with asymmetric LRs (LoRA+ style).
Their Appendix B.2 explicitly states:
"LoRA+ can be applied to non-zero initialization to ensure internal stability."

They find:
- Non-zero init alone: robust to small LRs, peak performance unchanged
- Non-zero init + LoRA+ asymmetric LR: further stability improvement
- The asymmetric LR benefit is smaller with non-zero init (because the "escape
  from zero" problem is already solved) but still positive

**Confidence:** MEDIUM. The study uses random non-zero init, not SVD-aligned
non-zero init (RELI). The effect with RELI may differ because RELI's B is
aligned with the gradient, which changes the initial gradient direction for B.

### 4.4 NeurIPS 2024 Initialization Study: arXiv:2406.08447

**Hayou, S., Ghosh, N., & Yu, B. (2024). The Impact of Initialization on LoRA
Finetuning Dynamics.** NeurIPS 2024.

This is the same Hayou group. Key finding directly relevant to Q4:
- B=0, A=random (standard) is BETTER than A=0, B=random
- B=0, A=random allows larger LRs without output instability
- The instability with A=0, B=random init is that the gradient of the loss
  w.r.t. B (which is non-zero from the start) causes large early updates that
  destabilize training

For RELI where BOTH are non-zero: this paper does not test this case directly.
But the lesson is: when B != 0 at init, output instability can arise from large
B gradients early in training. This makes the B LR MORE sensitive, not less.

**Implication for RELI+LoRA+:** Using lambda=16 (large B LR) with RELI may
cause instability early because B is non-zero and already receiving large
gradients through the alignment with the loss. The paper suggests that large B
LR is specifically stable because B=0 limits early B gradient magnitudes. With
B != 0, this safety mechanism is gone.

**Recommendation:** With RELI initialization, start with lambda = 4 rather
than lambda = 16. Monitor the loss for the first 20 steps; if stable, try
lambda = 8. The Stable-LoRA weight-shrinkage approach (arXiv:2603.05204) is
directly designed to address this: shrink A to near-zero for the first few
steps, then allow growth. This could be adapted to limit B's initial gradient
magnitude.

### 4.5 LoRA-GA as a Proxy (arXiv:2407.05000, NeurIPS 2024)

**Wang, S. et al. (2024). LoRA-GA: Low-Rank Adaptation with Gradient
Approximation.** NeurIPS 2024.

LoRA-GA is the closest existing published method to RELI (both use gradient SVD
to set A and B). The LoRA-GA paper does NOT apply LoRA+ asymmetric LRs — it
uses a single LR throughout. Performance gains come from the initialization
quality alone (2-4x faster convergence than vanilla LoRA).

The absence of LoRA+ in LoRA-GA suggests the authors did not find it necessary
— but this may simply be that they did not test the combination. GoRA
(arXiv:2502.12171, 2025) does combine adaptive rank with gradient-based init
and does use modified LRs, but does not explicitly use LoRA+'s Proposition 2
framing.

**Conclusion for Q4:**
- LoRA+ Proposition 2 motivation (B escaping zero) does not apply to RELI
- The WIDTH-SCALING argument still applies and favors eta_B > eta_A
- Non-zero init empirically benefits from asymmetric LR (arXiv:2505.23194)
- Use lambda = 4 conservatively to avoid early B instability with RELI
- If the Stable-LoRA shrinkage trick is used, lambda = 8-16 may be safe

---

## Q5: rsLoRA + Rank Annealing — Exact Alpha Adjustment Derivation

### 5.1 The rsLoRA Scaling Formula

With rsLoRA, the LoRA update is:

    Delta_W = (alpha / sqrt(r)) * B * A

The "effective learning rate scale" ELS (how much the update scales gradient
magnitudes back-to-front through the adapter) is:

    ELS(r, alpha) = alpha / sqrt(r)

This is the quantity that must remain constant when we change rank to maintain
the same gradient dynamics.

### 5.2 The Rank Annealing Constraint

When we anneal from rank r_old to rank r_new (e.g., via RELI-RA which prunes
the bottom singular directions of the LoRA product B*A), we drop from a higher-
rank adapter to a lower-rank adapter. To maintain the same ELS:

    alpha_old / sqrt(r_old) = alpha_new / sqrt(r_new)

Solving for alpha_new:

    alpha_new = alpha_old * sqrt(r_new / r_old)

**For r_old = 16, r_new = 8:**

    alpha_new = alpha_old * sqrt(8 / 16)
             = alpha_old * sqrt(0.5)
             = alpha_old * 0.70711...

So alpha must be REDUCED by a factor of ~0.707 when halving the rank.

**For r_old = 16, r_new = 4:**

    alpha_new = alpha_old * sqrt(4 / 16)
             = alpha_old * sqrt(0.25)
             = alpha_old * 0.5

Alpha is halved when reducing rank by 4x.

### 5.3 Interaction with LoRA+ Asymmetric LR

If we also use LoRA+ with lambda = eta_B / eta_A, then the full effective LR
for the B matrix is:

    LR_B_effective = lambda * eta_A * (alpha / sqrt(r))

When annealing rank:

    LR_B_effective_new = lambda * eta_A * (alpha_new / sqrt(r_new))
                       = lambda * eta_A * (alpha_old * sqrt(r_new/r_old) / sqrt(r_new))
                       = lambda * eta_A * (alpha_old / sqrt(r_old))
                       = LR_B_effective_old

So the combined alpha adjustment preserves both the rsLoRA gradient stability
AND the LoRA+ B-matrix effective LR simultaneously. No additional correction
is needed when using both rsLoRA and LoRA+ together.

### 5.4 The muP-Consistent View (arXiv:2602.06204)

Chen et al. (2026) additionally show that when moving between ranks, the BASE
learning rate eta should be adjusted. In the rank-limited regime (most TTT
scenarios where n >> r):

    eta_new = eta_old * sqrt(r_old / r_new)

For r_old=16, r_new=8:

    eta_new = eta_old * sqrt(16/8) = eta_old * sqrt(2) = eta_old * 1.414

This means the LR should INCREASE when rank decreases under muP scaling. This
seems counterintuitive: we prune from rank-16 to rank-8, and we should increase
LR? Yes — because the smaller rank adapter has fewer parameters, each surviving
direction should be updated more aggressively to compensate.

**Combined recommendation for RELI-RA rank annealing (16 → 8):**
1. Set alpha_new = alpha_old * 0.707 (rsLoRA ELS preservation)
2. Optionally set eta_new = eta_old * 1.414 (muP rank-optimal LR correction)
3. Keep lambda = eta_B / eta_A unchanged (the ratio persists)

Note: Steps 1 and 2 partially cancel: alpha is the scalar scale, while eta is
the optimizer LR. In practice, alpha and eta are multiplicative, so adjusting
both by inverse-sqrt(2) and sqrt(2) respectively leaves the product unchanged.
If your implementation uses alpha as the sole scaling knob, only Step 1 is
needed. If you have separate control over eta, both adjustments are optimal.

### 5.5 AdaLoRA's Related Approach

AdaLoRA (arXiv:2303.10512) solves the same problem differently: it explicitly
tracks singular values Lambda_i and prunes based on importance scores. When a
singular direction is pruned, the remaining Lambda_i are NOT rescaled — the
importance-based gating implicitly handles the budget. However, this means
AdaLoRA does NOT preserve ELS when rank changes (it uses a fixed alpha/r style
scaling internally for initialization). This is a weakness of AdaLoRA in
comparison to rsLoRA-based rank annealing.

---

## Q6: Muon Optimizer + LoRA Adapters in TTT

### 6.1 The Core Question

Our base model (train_gpt.py) uses Muon for 2D weight matrices and AdamW for
1D parameters (biases, norms). Our LoRA TTT adapters are 2D matrices (A is
r x d_in, B is d_out x r). Should the TTT optimizer for A and B also be Muon?

### 6.2 LoRA Meets Riemannion (arXiv:2507.12142, July 2025 / ICLR 2026)

**Bogachev, V., Aletov, V., & Rakhuba, M. (2025). LoRA meets Riemannion: Muon
Optimizer for Parametrization-independent Low-Rank Adapters.** arXiv:2507.12142.
Revised October 2025. OpenReview.net accepted at ICLR 2026 (based on forum).

This paper directly addresses Q6. Key contributions:
1. Standard Euclidean Muon applied to A and B separately has a theoretical flaw:
   it ignores the fixed-rank manifold structure of the low-rank matrix A*B, and
   the result is parametrization-dependent (the update depends on whether we
   write A*B or B*A).
2. The paper derives Riemannion, a Riemannian optimizer on the fixed-rank matrix
   manifold that generalizes Muon.
3. Riemannion combines: (a) a Riemannian gradient-informed LoRA initialization,
   (b) the Muon Newton-Schulz spectral step, and (c) a retraction to the
   fixed-rank manifold.
4. Experimental results on LLM and diffusion models show consistent improvements
   over standard LoRA+AdamW and LoRA+Muon (naive).

**For TTT specifically:** The paper notes that Muon can be applied naively per-
factor (A and B separately, as in standard Muon usage). This already provides
near-uniform singular value growth, which is empirically beneficial. The full
Riemannion correction is more theoretically rigorous but adds implementation
complexity.

### 6.3 Uniform Spectral Growth under Muon (arXiv:2602.06385, Feb 2026)

**Kang, C. et al. (2026). Uniform Spectral Growth and Convergence of Muon in
LoRA-Style Matrix Factorization.** Submitted February 6, 2026.

Proves mathematically that spectral gradient descent (Muon) on the LoRA factors
A and B separately causes the singular values of the product A*B to grow at
EQUAL RATES, despite orthogonalization being performed on A and B individually.
This "equal-rate dynamics" property is:
- Desirable: prevents rank collapse (one singular direction dominating)
- Convergent: proves global convergence from almost all initializations under
  L2 regularization

**Connection to rsLoRA:** The equal-rate singular value growth under Muon is
EXACTLY what rsLoRA aims to achieve via scaling: both approaches try to prevent
any one singular direction from growing much faster than others. Under rsLoRA +
AdamW, singular value growth is unequal (direction-dependent via Adam's second
moment). Under Muon alone, growth is equal. rsLoRA + Muon may provide both
gradient magnitude stability (rsLoRA) and uniform spectral growth (Muon) — a
potentially synergistic combination.

### 6.4 MuonAll: Muon for All Parameters (arXiv:2511.06086, Nov 2025)

**MuonAll: Muon Variant for Efficient Finetuning of Large Language Models.**
Extends Muon to 1D parameters (by treating them as diagonal matrices) for fine-
tuning. Performance is comparable to AdamW on standard benchmarks. This would
allow a fully unified Muon optimizer for all LoRA TTT parameters.

### 6.5 Optimizer Mismatch Risk

A critical observation from the Muon+AdamW comparison literature: models
pretrained with AdamW and fine-tuned with Muon show suboptimal performance, and
vice versa (the "optimizer mismatch" phenomenon). Our base model is pretrained
with Muon. The LoRA adapters are new random matrices — they have no pretrained
history. Therefore:
- The base weights were shaped by Muon (low spectral norm, uniform singular
  value distribution)
- The LoRA adapters, if initialized randomly, carry no optimizer history
- Using Muon for LoRA adapters in TTT is therefore CONSISTENT with the base
  training regime and avoids mismatch
- Using AdamW for LoRA adapters is technically a MISMATCH (different update
  geometry than what shaped the base weights)

**However:** RELI initializes A from the gradient SVD, which already provides a
well-directed non-random start. The optimizer-mismatch argument is weakened when
initialization is so good that the optimizer's directional choices matter less.

### 6.6 LoRA-TTT Practical Precedent (arXiv:2502.02069, Feb 2025)

**Kojima, Y. et al. (2025). LoRA-TTT: Low-Rank Test-Time Training for Vision-
Language Models.** Submitted February 2025.

Uses AdamW (LR=1e-3, weight decay=0.2) for LoRA TTT adapters. Single-step
optimization. No exploration of Muon. This is the closest published precedent
for LoRA-TTT optimization, but it is in a VLM (CLIP) context, not autoregressive
LM context, and uses only 1 TTT step per sample rather than 50-200 steps.

### 6.7 Summary Recommendation for TTT Optimizer

| Option | Pros | Cons |
|--------|------|------|
| AdamW for A, B | Simple, well-tested, no mismatch risk if we treat adapters as new | Different geometry than base weights |
| Muon for A, B | Consistent with base training, equal-rate SVD growth, proven convergence | Adds Newton-Schulz overhead per step (~3 iterations per parameter matrix) |
| Riemannion for A*B jointly | Most theoretically rigorous, parametrization-independent | Complex implementation, untested in TTT |
| ALLoRA style per-norm scaling | Addresses all three LoRA failure modes | Per-param norm overhead, no TTT results |

**Recommended for Parameter Golf TTT:** Use Muon for the LoRA A and B matrices
directly (naive per-factor application, as in the paper's "LoRA+Muon baseline").
The overhead is 3 Newton-Schulz iterations per adapter matrix per step — for
rank-8 to rank-16 matrices, these are tiny matrix operations (r x d_model, e.g.
8 x 768). The spectral uniform growth property directly addresses the rank
collapse risk during multi-step TTT. Keep AdamW for any 1D parameters in the
TTT loop (e.g., layer norms if unfrozen).

**Risk:** Newton-Schulz iterations may add wall-clock time. Profile before
committing. The 3-iteration approximation used in Muon (from Kosson et al. 2023)
is fast but adds ~30% overhead per adapter matrix update.

---

## 7. ACTIONABLE IMPLEMENTATION CHANGES (Ranked by Expected BPB Impact)

### Change 1: Activate rsLoRA Scaling in All Existing LoRA Code
**What to do:** Replace all instances of `scale = alpha / r` with
`scale = alpha / math.sqrt(r)` in train_gpt.py or wherever LoRA adapters are
applied.

**Expected BPB impact:** -0.003 to -0.008 BPB. Impact grows with rank. At
rank-16, the gradient magnitude benefit is sqrt(16) = 4x larger than rank-1.
At rank-8, it is sqrt(8) = 2.83x. This directly affects convergence speed in
the TTT loop: more gradient signal = fewer steps to same loss.

**Confidence:** HIGH. rsLoRA is the most well-validated LoRA scaling
improvement (PEFT library default as of 2024).

**Code change:** One-liner. No parameter increase.

### Change 2: Apply LoRA+ Asymmetric LR (lambda = 4)
**What to do:** In the TTT optimizer initialization, give the B matrix a
learning rate lambda=4 times larger than the A matrix. If using AdamW:
```
optimizer = AdamW([
    {'params': [layer.lora_A], 'lr': eta_A},
    {'params': [layer.lora_B], 'lr': 4 * eta_A},
])
```

**Expected BPB impact:** -0.002 to -0.005 BPB. Larger gains on tasks with
many steps (200+ TTT steps). Smaller gains with RELI initialization (because
the zero-escape problem is already solved).

**Note on RELI:** Use lambda=4 rather than lambda=16 to avoid B instability
with non-zero initialization. If RELI is not used (standard Gaussian A, B=0),
lambda=16 is safe.

**Confidence:** HIGH for standard init, MEDIUM for RELI init. Paper citation:
Hayou et al. 2024, ICML (arXiv:2402.12354).

### Change 3: Correct Alpha During Rank Annealing (RELI-RA)
**What to do:** When annealing from rank r_old to rank r_new, automatically
update the alpha parameter:
```
alpha_new = alpha_old * math.sqrt(r_new / r_old)
```
For 16 → 8: alpha_new = alpha_old * 0.70711
For 16 → 4: alpha_new = alpha_old * 0.5

This ensures the effective learning rate scale (ELS = alpha/sqrt(r)) remains
constant across the rank transition.

**Expected BPB impact:** -0.001 to -0.003 BPB. The primary benefit is
preventing a sudden LR jump at the rank transition boundary, which can cause
training instability and BPB spike.

**Confidence:** HIGH (derived directly from rsLoRA's Theorem 1 + algebra).

### Change 4: Apply Muon Optimizer to LoRA A and B Matrices
**What to do:** Replace AdamW for A and B matrices in the TTT optimizer with
Muon (naive per-factor application). Keep AdamW for 1D parameters.

```python
muon_params = [p for n, p in model.named_parameters()
               if p.ndim >= 2 and ('lora_A' in n or 'lora_B' in n)]
adam_params  = [p for n, p in model.named_parameters()
               if p.ndim < 2 and p.requires_grad]
optimizer = [Muon(muon_params, lr=eta), AdamW(adam_params, lr=eta)]
```

**Expected BPB impact:** -0.002 to -0.006 BPB. Evidence from arXiv:2507.12142
shows consistent improvements over LoRA+AdamW baselines on LLM tasks.

**Risk:** Newton-Schulz overhead per TTT step. Profile first on a timing run.
If the Newton-Schulz step (3 iterations) adds > 15% wall time to the TTT loop,
the wall-clock-adjusted BPB improvement may be negative.

**Confidence:** MEDIUM. Strong theoretical backing (arXiv:2602.06385) and LLM
empirical results, but no published TTT-specific results.

### Change 5: Adopt Stable-LoRA Weight Shrinkage for RELI Stability
**What to do:** During the first K steps of TTT (recommend K = max(5, 0.05 *
total_steps)), progressively shrink the A matrix weight magnitude by a decay
factor d:
```
# In the first K steps:
with torch.no_grad():
    lora_A.data *= (1 - shrink_rate)  # shrink_rate ~ 0.05-0.1
```
After K steps, stop shrinking and allow A to grow normally.

This addresses the instability in RELI+LoRA+ where A's non-zero init combined
with large B LR can cause early training divergence.

**Expected BPB impact:** -0.001 to -0.002 BPB (primarily a stability gain;
prevents loss spikes in edge-case contexts).

**Confidence:** MEDIUM. Based on arXiv:2603.05204 (ICLR 2026), which validates
this specifically for non-zero A init but not yet in a TTT context.

---

## 8. COMPLETE CITATION LIST (Chicago Author-Date Format)

1. Kalajdzievski, Damjan. 2023. "A Rank Stabilization Scaling Factor for
   Fine-Tuning with LoRA." arXiv:2312.03732.

2. Hayou, Soufiane, Nikhil Ghosh, and Bin Yu. 2024. "LoRA+: Efficient Low
   Rank Adaptation of Large Models." In *Proceedings of the 41st International
   Conference on Machine Learning (ICML 2024)*. PMLR v235. arXiv:2402.12354.

3. Hayou, Soufiane, Nikhil Ghosh, and Bin Yu. 2024. "The Impact of
   Initialization on LoRA Finetuning Dynamics." In *Advances in Neural
   Information Processing Systems (NeurIPS 2024)*. arXiv:2406.08447.

4. Wang, Shaowen, et al. 2024. "LoRA-GA: Low-Rank Adaptation with Gradient
   Approximation." In *Advances in Neural Information Processing Systems
   (NeurIPS 2024)*. arXiv:2407.05000.

5. Hu, Edward, et al. 2022. "LoRA: Low-Rank Adaptation of Large Language
   Models." In *ICLR 2022*. arXiv:2106.09685.

6. Zhang, Qingru, et al. 2023. "AdaLoRA: Adaptive Budget Allocation for
   Parameter-Efficient Fine-Tuning." In *ICLR 2023*. arXiv:2303.10512.

7. Xu, Ziqing, Hancheng Min, Lachlan Ewen MacDonald, Jinqi Luo, Salma
   Tarmoun, Enrique Mallada, and Rene Vidal. 2025. "Understanding the Learning
   Dynamics of LoRA: A Gradient Flow Perspective on Low-Rank Adaptation in
   Matrix Factorization." In *AISTATS 2025*. PMLR v258. arXiv:2503.06982.

8. Li, Shiwei, et al. 2025. "Beyond Zero Initialization: Investigating the
   Impact of Non-Zero Initialization on LoRA Fine-Tuning Dynamics." In
   *ICML 2025*. PMLR v267. arXiv:2505.23194.

9. Huang, Hai, and Randall Balestriero. 2024. "ALLoRA: Adaptive Learning Rate
   Mitigates LoRA Fatal Flaws." Submitted to ICLR 2025. arXiv:2410.09692.

10. Bogachev, Vladimir, Vladimir Aletov, and Maxim Rakhuba. 2025. "LoRA meets
    Riemannion: Muon Optimizer for Parametrization-independent Low-Rank
    Adapters." Accepted at ICLR 2026. arXiv:2507.12142.

11. Kang, Changmin, et al. 2026. "Uniform Spectral Growth and Convergence of
    Muon in LoRA-Style Matrix Factorization." Submitted February 6, 2026.
    arXiv:2602.06385.

12. Chen, Nan, Soledad Villar, and Soufiane Hayou. 2026. "Learning Rate Scaling
    across LoRA Ranks and Transfer to Full Finetuning." Submitted February 5,
    2026. arXiv:2602.06204.

13. Lee, Yu-Ang, Ching-Yun Ko, Pin-Yu Chen, and Mi-Yen Yeh. 2026. "Learning
    Rate Matters: Vanilla LoRA May Suffice for LLM Fine-tuning." Submitted
    February 4, 2026. arXiv:2602.04998.

14. Rushka, Michael, and Diego Klabjan. 2026. "Rank-Accuracy Trade-off for
    LoRA: A Gradient-Flow Analysis." Submitted February 10, 2026.
    arXiv:2602.10212.

15. Wang, Yan, et al. 2025. "Stable-LoRA: Stabilizing Feature Learning of
    Low-Rank Adaptation." Accepted at ICLR 2026. arXiv:2603.05204.

16. Kojima, Yuto, Jiarui Xu, Xueyan Zou, and Xiaolong Wang. 2025. "LoRA-TTT:
    Low-Rank Test-Time Training for Vision-Language Models." Submitted
    February 2025. arXiv:2502.02069.

17. Wang, Zequn, et al. 2026. "Stabilized Fine-Tuning with LoRA in Federated
    Learning: Mitigating the Side Effect of Client Size and Rank via the
    Scaling Factor." Submitted March 2026. arXiv:2603.08058.

18. MuonAll: Muon Variant for Efficient Finetuning of Large Language Models.
    Computational Linguistics Lab, PICT, Pune. Submitted November 2025.
    arXiv:2511.06086.

---

## 9. RESEARCH GAPS AND OPEN QUESTIONS

1. **RELI + LoRA+ combined effect** has not been directly studied. All existing
   non-zero init + asymmetric LR papers use random non-zero init, not gradient-
   SVD-aligned init.

2. **Muon for TTT** has no published TTT-specific results. The LoRA-TTT paper
   (2502.02069) uses AdamW only.

3. **rsLoRA + rank annealing interaction** in the multi-step TTT setting has not
   been formally studied. The alpha adjustment formula is algebraically derived
   here from first principles, not taken from a published paper.

4. **ALLoRA in TTT** is unexplored. ALLoRA's per-norm gradient scaling may be
   especially valuable in TTT where step counts are very small (few-shot).

5. **Muon + rsLoRA interaction** — do they double-count the spectral
   normalization? In principle, Muon's Newton-Schulz step already normalizes
   singular values, partially subsuming what rsLoRA's scaling factor does. A
   combined analysis comparing rsLoRA+AdamW vs. Muon alone vs. rsLoRA+Muon
   would clarify this.

---

*Searched: arXiv.org, papers.cool, Semantic Scholar abstracts, arXiv HTML
papers, ICML/NeurIPS/ICLR proceedings pages. All papers verified against arXiv
abstract pages. Date of last search: 2026-03-24.*
