# Anti-Forgetting and Calibration Preservation in Per-Document TTT
# Generated: 2026-03-24 via WebSearch research
# Databases: arXiv, NeurIPS, ICML, ICLR, CVPR/ACCV proceedings, ACL Anthology, IEEE Xplore

---

## Executive Summary

Our TTT system performs per-document LoRA adaptation. The risk profile is twofold:

1. **Overconfidence on the test document:** The LoRA learns to predict that document's
   specific tokens with near-zero entropy, collapsing the output distribution.
2. **Calibration drift on other distributions:** Even though LoRA is low-rank, the inner
   loop's optimizer (Adam on LoRA params) can drive the adapted model far from the
   pretrained distribution in representation space, degrading out-of-distribution BPB.

This document surveys the full regularization toolkit from continual learning, calibration,
and test-time adaptation literatures, and provides concrete implementation recommendations
ranked by expected BPB impact and implementation cost.

**Bottom line:**
- **Highest-signal, lowest-effort additions:** (1) KL-distillation regularizer against the
  frozen base model (3 lines), (2) Mixout replacing dropout in TTT (2 lines), (3) label
  smoothing alpha=0.05–0.1 in the TTT cross-entropy loss (1 line).
- **Higher-effort, higher-ceiling:** UQ4CT ensemble-of-LoRAs, EATA-C min-max entropy
  calibration, Fisher-weighted EWC on LoRA params.
- **Do NOT combine label smoothing + KL-distillation naively:** they partially conflict
  (see Section 4 below). Pick one primary calibration signal.

---

## Part I: Knowledge Distillation as TTT Regularizer

### 1. Hinton et al. — Distilling the Knowledge in a Neural Network (Foundational)

**Citation:** Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a
Neural Network. *arXiv:1503.02531*. NIPS 2015 Deep Learning Workshop.

**URL:** https://arxiv.org/abs/1503.02531

**Core mechanism:** Train a student model by minimizing a convex combination of:

```
L_total = (1 - alpha) * L_CE(y_hard, p_student)
        + alpha * T^2 * KL(p_teacher(T) || p_student(T))
```

Where:
- `T` is the "distillation temperature" applied to both teacher and student softmax
- `T^2` is a normalization factor that keeps gradient magnitudes scale-invariant across T
- `p_teacher(T)` = softmax(logits_teacher / T) — the frozen base model at test time
- `alpha` controls the balance between hard-label CE and soft-label distillation
- At T=1 this is standard KL divergence from student to teacher

**Why this prevents overconfidence during TTT:** The teacher (frozen base model) always
assigns non-negligible probability to alternative tokens. Minimizing KL from student to
teacher forces the student to preserve that "dark knowledge" — the relative probabilities
over the vocabulary that encode inter-token similarity. The student cannot collapse to a
delta distribution on any token without incurring a large KL penalty.

**Key finding on temperature interaction:** At high T, all token probabilities become more
uniform, making the soft targets carry more gradient signal about off-peak probabilities.
At T=1, the soft-label KL is still meaningful but the peak token dominates. For TTT where
the goal is calibration preservation, T=2–4 is recommended to amplify the regularization
signal from non-top tokens.

**Implementation for TTT inner loop:**

```python
# Inside TTT forward pass
with torch.no_grad():
    base_logits = frozen_base_model(x)  # frozen, no grad
T = 2.0  # distillation temperature
alpha = 0.3  # weight of KL regularizer
kl_loss = F.kl_div(
    F.log_softmax(adapted_logits / T, dim=-1),
    F.softmax(base_logits / T, dim=-1),
    reduction='batchmean'
) * T**2
L_total = (1 - alpha) * ce_loss + alpha * kl_loss
```

**Important caveat from Muller et al. (2019):** If the teacher is trained with label
smoothing, the KD soft labels lose their inter-class similarity information, making
the KL regularizer LESS effective. If our base model was trained with label smoothing,
the KD regularizer is weakened. Use T > 1 to compensate.

**Complexity:** 3 lines if the frozen base model forward pass is already in the codebase
(it is, for baseline BPB measurement). **Cost: negligible at inference; one extra frozen
forward per training step.**

---

### 2. Calibration Transfer via Knowledge Distillation

**Citation:** Hebbalaguppe, R., et al. (2024). Calibration Transfer via Knowledge
Distillation. *Proceedings of ACCV 2024*.

**URL:** https://openaccess.thecvf.com/content/ACCV2024/papers/Hebbalaguppe_Calibration_Transfer_via_Knowledge_Distillation_ACCV_2024_paper.pdf

**Key finding directly relevant to TTT:** The paper proves that an uncalibrated (overconfident)
teacher provides poor KD supervision — the KL divergence signal is dominated by the
teacher's own calibration error, and the student inherits that overconfidence. Conversely, a
well-calibrated teacher transfers calibration to the student. Critical distinction:

- **Static label smoothing on teacher:** Reduces teacher confidence but results in LESS
  accurate students (accuracy drops) and only moderate calibration improvement.
- **Dynamic label smoothing on teacher (sample-level):** Yields better-calibrated AND more
  accurate student models.

**Implication for TTT:** Our frozen base model IS the teacher. If the base model is well-
calibrated (low ECE), the KL regularizer is a strong anti-overconfidence signal. If the base
model is itself overconfident (common after pretraining without explicit calibration), the KL
regularizer will still push the adapted model toward the base distribution, which is a weaker
but still useful form of anti-forgetting.

**Recommendation:** Use temperature T=2–4 on both sides of KL to "soften" the potentially
overconfident teacher signal before using it as a regularization target.

---

### 3. The Role of Teacher Calibration in Knowledge Distillation

**Citation:** Anonymous. (2025). The Role of Teacher Calibration in Knowledge Distillation.
*arXiv:2508.20224*.

**URL:** https://arxiv.org/abs/2508.20224

**Key finding:** "Overconfident error diminishes the influence of the KD loss." This
quantifies the risk: if the teacher's top-1 probability is near 1.0 (very overconfident),
the soft labels are nearly identical to hard labels and the KL term provides no additional
regularization beyond standard CE. Using T > 1 on the teacher is ESSENTIAL in this case.

**Practical rule derived from this paper:** If the base model's mean top-1 probability on
the test document exceeds 0.90, use T >= 4. If it is 0.70–0.90, T=2 is sufficient.

---

### 4. Self-Distillation Enables Continual Learning (SDFT)

**Citation:** Shenfeld, I., Damani, M., Hubotter, J., & Agrawal, P. (2026). Self-Distillation
Enables Continual Learning. *arXiv:2601.19897*.

**URL:** https://arxiv.org/abs/2601.19897

**Core mechanism:** SDFT uses the same model in dual roles — a "teacher" conditioned on
demonstration context, and a "student" without context. The student is trained on on-policy
trajectories to match teacher distributions via per-token KL divergence. The KL constraint
to the initial distribution for generic (non-demonstration) inputs prevents catastrophic
forgetting.

**Most relevant result for TTT:** Distributions close to the pretrained distribution suffer
significantly less catastrophic forgetting. The KL regularizer anchors the adapted model
to its initialization, and the effect is stronger when the distillation constraint is applied
on-policy (using the model's own generations rather than ground-truth tokens).

**TTT adaptation:** In our per-document TTT, the "student" role is the LoRA-adapted model
trained on the document, and the "teacher" is the frozen base model. The SDFT principle
suggests computing KL on the model's OWN output distribution, not just the teacher-forced
positions:

```python
# Self-distillation variant: use student's own generations as sampling points
# This is stronger than teacher-forcing the KL at training positions only
sampled_positions = torch.randint(0, seq_len, (n_sample,))
kl_loss = KL(student_logits[sampled_positions] || teacher_logits[sampled_positions])
```

**Key result:** SDFT consistently outperforms SFT with higher new-task accuracy while
substantially reducing catastrophic forgetting. No quantified BPB numbers available
(the paper targets instruction following, not language modeling), but the principle is
directly applicable.

---

## Part II: Calibration-Aware Fine-Tuning

### 5. When Does Label Smoothing Help?

**Citation:** Muller, R., Kornblith, S., & Hinton, G. E. (2019). When Does Label Smoothing
Help? *Advances in Neural Information Processing Systems 32 (NeurIPS 2019)*. arXiv:1906.02629.

**URL:** https://arxiv.org/abs/1906.02629

**Core mechanisms and calibration findings:**

**Label smoothing replaces hard targets y in CE loss:**
```
y_smooth_i = (1 - alpha) * y_i + alpha / K
```
Where K is the vocabulary size and alpha is the smoothing coefficient (typically 0.05–0.1).
The effective target for the correct class becomes `(1 - alpha + alpha/K)` and all other
classes get `alpha/K` instead of 0.

**Calibration result:** Label smoothing achieves similar calibration to temperature scaling
post-hoc. Specifically, training with alpha=0.05 produces a model similarly calibrated to
training without smoothing and then applying temperature scaling. This is a direct substitution.

**Key insight on tight clusters:** Label smoothing encourages representations of same-class
examples to form tight clusters. This improves calibration (predictions between classes
become more meaningful) but REDUCES knowledge distillation effectiveness — the teacher
trained with smoothing has less useful soft-label information about inter-class relationships
because representations collapse into tight same-class clusters.

**Critical warning for TTT:** Do NOT simultaneously apply label smoothing AND a KL
distillation regularizer from a base model trained with label smoothing. The two interact
destructively: (1) the teacher's soft labels are already less informative (tight clusters),
(2) the student's smoothed targets prevent it from fitting the teacher's distribution
tightly. Use one or the other:
- **Label smoothing only:** alpha=0.05–0.1 in the TTT CE loss. Simpler, 1 line.
- **KL distillation only:** From frozen base model at T=2–4. More principled.

**Recommended configuration for TTT:** Use label smoothing alpha=0.05 as the default
(1 line change). Switch to KD regularizer if the base model is known to be well-calibrated.

**Complexity:** 1 line. In PyTorch: `F.cross_entropy(logits, targets, label_smoothing=0.05)`

---

### 6. Calibrated Language Models and Label Smoothing (2025)

**Citation:** Anonymous. (2025). Calibrated Language Models and How to Find Them with Label
Smoothing. *arXiv:2508.00264 / OpenReview*.

**URL:** https://arxiv.org/abs/2508.00264

**Key findings specifically for language models:**

1. Label smoothing reduces calibration error while having negligible effects on downstream
   task performance accuracy.
2. The effectiveness of label smoothing diminishes significantly in smaller models.
3. Label smoothing can maintain calibration throughout the SFT process, but its effectiveness
   relates directly to model architecture characteristics including hidden size and vocabulary size.

**Implication for our GPT-2 scale model:** Our model is relatively small. This paper warns
that label smoothing effectiveness degrades at smaller scales. This weakens the case for
label smoothing and strengthens the case for explicit KL distillation (which is architecture-
size agnostic).

---

### 7. Restoring Calibration for Aligned LLMs: A Calibration-Aware Fine-Tuning Approach

**Citation:** Anonymous. (2025). Restoring Calibration for Aligned Large Language Models:
A Calibration-Aware Fine-Tuning Approach. *arXiv:2505.01997*.

**URL:** https://arxiv.org/abs/2505.01997

**Core problem addressed:** Instruction tuning (SFT) causes significant calibration
degradation. The paper proposes Calibration-aware Fine-Tuning (CFT) using EM-algorithm-
based ECE regularization.

**Key theoretical finding:** When models are over-fine-tuned, they shift into a "non-
calibratable regime" where there is a fundamental trade-off between calibration error and
performance — temperature scaling can no longer recover calibration without hurting
accuracy. **This is the regime we must avoid in TTT.**

**CFT loss function:**
```
L_CFT = L_CE + beta * ECE_differentiable
```
Where ECE_differentiable is a binning-based approximation of Expected Calibration Error,
differentiable via bin membership soft-assignment.

**Implementation complexity:** Moderate. Requires computing a differentiable ECE estimate
per batch. Not trivial for a TTT inner loop (adds ~20% compute overhead). More practical
for longer adaptation runs.

---

### 8. Functional-Level Uncertainty Quantification for Calibrated Fine-Tuning (UQ4CT)

**Citation:** Anonymous. (2024). Functional-level Uncertainty Quantification for Calibrated
Fine-tuning on LLMs. *arXiv:2410.06431*.

**URL:** https://arxiv.org/abs/2410.06431

**Core approach:** Instead of a single LoRA adapter, train an ensemble of LoRA modules at
each layer. Use a Mixture-of-Experts (MoE) gating to combine them. During fine-tuning,
jointly learn LoRA expert parameters and calibrate the prompt-dependent mixture to align
functional-level uncertainty with predictive correctness.

**Results:** UQ4CT achieves over 25% reduction in Expected Calibration Error (ECE) while
preserving high accuracy across five benchmarks. Maintains superior ECE performance under
distribution shift.

**Adaptation to TTT context:** This is our most expensive option but potentially the most
powerful. Instead of one TTT LoRA per document, train K=3–5 LoRA adapters initialized
differently (different random seeds for A, same zero init for B). Their disagreement at
inference time is a direct uncertainty estimate. The gating MoE selects or weights them.

**Complexity:** Substantial (K x LoRA parameter overhead, MoE gating network). Not
recommended for EXP priority unless simpler methods stall.

**Simpler variant (MC-LoRA):** Use standard dropout INSIDE the LoRA forward pass (between A
and B). Sample K=5–10 forward passes with dropout enabled at test time. Average predictions.
The variance across passes is your uncertainty estimate. This is 2 lines.

---

### 9. Calibrating Language Models with Adaptive Temperature Scaling (ATS)

**Citation:** Anonymous. (2024). Calibrating Language Models with Adaptive Temperature
Scaling. *arXiv:2409.19817*.

**URL:** https://ar5iv.labs.arxiv.org/html/2409.19817

**Core approach:** Post-hoc calibration via temperature scaling where the temperature is
input-dependent (computed from a small auxiliary network) rather than a fixed global scalar.

**Relevance to TTT:** Temperature scaling is ORTHOGONAL to TTT LoRA adaptation. After each
document's TTT, apply a per-document temperature T* that minimizes NLL on a held-out
portion of that document. Cost: essentially zero. This can be combined with any of the
other anti-overconfidence techniques as a post-hoc correction layer.

**Key interaction finding:** ATS is MORE effective after label smoothing (which narrows the
calibration gap that ATS needs to close) and LESS necessary after KD regularization (which
already addresses calibration during training).

---

## Part III: Anti-Forgetting LoRA Variants

### 10. DARE — Drop and REscale (Language Models are Super Mario)

**Citation:** Yu, L., Yu, B., Yu, H., Huang, F., & Li, Y. (2023). Language Models are Super
Mario: Absorbing Abilities from Homologous Models as a Free Lunch. *arXiv:2311.03099*.
Accepted at ICML 2024.

**URL:** https://arxiv.org/abs/2311.03099

**DARE mechanism:**
```
delta_W = W_finetuned - W_pretrained  # task vector
mask = Bernoulli(1 - p)               # drop p fraction of deltas
delta_W_dare = mask * delta_W / (1 - p)  # rescale surviving deltas
W_merged = W_pretrained + delta_W_dare
```

The key insight is that SFT delta parameters are nearly redundant: up to 90–99% can be
zeroed without measurable performance loss on the fine-tuned task. The rescaling by `1/(1-p)`
ensures the expected value of delta_W is preserved (unbiased estimator).

**Why this prevents forgetting:** By randomly zeroing most delta parameters and rescaling,
DARE keeps the merged model CLOSE to the pretrained weights in L2 norm. The surviving
deltas carry the task-specific information but the total displacement from the pretrained
weights is dramatically reduced.

**How this differs from LoRA-TIES:**
- TIES-Merging (Yadav et al., arXiv:2306.01708) trims low-magnitude delta parameters
  (keeps top 20% by magnitude), resolves sign conflicts by majority vote, then merges.
- DARE uses random dropping (not magnitude-based), which has a different sparsity structure
  and is easier to implement as a training regularizer rather than a post-hoc merging step.

**Adaptation for TTT regularization (not model merging):** Apply DARE-style random masking
during TTT training as a delta-parameter regularizer:

```python
# After each TTT step, randomly zero p fraction of LoRA deltas
# (Applied as an explicit regularizer, not for merging)
with torch.no_grad():
    for name, param in lora_params.items():
        mask = torch.bernoulli(torch.full_like(param, 1 - p))
        param.data = param.data * mask / (1 - p)
```

This is a post-step stochastic perturbation that keeps LoRA weights near-sparse. With p=0.5,
it is equivalent to DropConnect on the LoRA delta.

**Important note:** DARE was designed for model MERGING post-training, not as a training-time
regularizer. Its use here is a novel adaptation not directly validated in the paper.

**Complexity:** 5 lines. Low risk.

---

### 11. TIES-Merging: Resolving Interference When Merging Models

**Citation:** Yadav, P., Tam, D., Choshen, L., Raffel, C., & Bansal, M. (2023).
TIES-MERGING: Resolving Interference When Merging Models. *arXiv:2306.01708*.
NeurIPS 2023.

**URL:** https://arxiv.org/abs/2306.01708

**Three-step mechanism:**
1. **TrIm:** Zero out parameters in the task vector with magnitude below threshold
   (keep top 20% by magnitude).
2. **Elect sign:** For each parameter, take the sign favored by the majority of models
   being merged.
3. **Merge:** Average only models that agree with the elected sign.

**Catastrophic forgetting prevention:** Merging algorithms that aggressively change weights
and move outside the base model's loss basin cause observed catastrophic forgetting. TIES
stays close to the base by keeping only the highest-magnitude deltas.

**TTT application:** Use TIES-inspired trimming as a regularizer: after TTT converges,
zero out the bottom 50–80% of LoRA parameters by magnitude. This is a deterministic
version of DARE. The surviving high-magnitude parameters are the ones that mattered most
for the document adaptation.

**Complexity:** 3 lines.

---

### 12. Merge Before Forget: Continual LoRA via Continual Merging

**Citation:** Anonymous. (2025). Merge before Forget: A Single LoRA Continual Learning via
Continual Merging. *arXiv:2512.23017*. OpenReview 2025.

**URL:** https://arxiv.org/abs/2512.23017

**Core idea:** Orthogonally initialize LoRA updates for each new task, then sequentially
merge them into a single unified LoRA. The orthogonal initialization minimizes inter-task
interference in the LoRA subspace.

**Direct relevance to per-document TTT:** For sequential document processing, each
document's LoRA update is initialized orthogonally to all previous documents' updates
(using QR decomposition of previous B matrices). This ensures new document adaptation
does not overwrite previous document-specific knowledge encoded in the LoRA.

**Implementation complexity:** Moderate. Requires tracking and QR-decomposing previous
LoRA updates. Feasible if we process documents sequentially and maintain state.

---

### 13. LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank Adaptation

**Citation:** Anonymous. (2025). LoRI: Reducing Cross-Task Interference in Multi-Task
Low-Rank Adaptation. *arXiv:2504.07448*.

**URL:** https://arxiv.org/abs/2504.07448

**Key finding relevant to forgetting:** TIES-Merging prunes low-magnitude parameters and
merges those with consistent signs. DARE applies random pruning with rescaling. By ensuring
approximate orthogonality between adapters, LoRI minimizes interference and preserves task-
specific performance.

**Simpler takeaway for TTT:** Sparse LoRA deltas (via DARE or TIES trimming) and orthogonal
LoRA subspaces (via QR projection) are the two main anti-interference levers. The sparsity
approach is simpler to implement.

---

## Part IV: Mixout Regularization

### 14. Mixout: Effective Regularization to Fine-tune Large-scale Pretrained Language Models

**Citation:** Lee, K., Mehta, D., Zhang, Z., Ganesan, K., Hwang, S., & Paul, S. (2020).
Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models.
*arXiv:1909.11299*. ICLR 2020.

**URL:** https://arxiv.org/abs/1909.11299

**Core mechanism:** Mixout is motivated by dropout but operates on parameters rather than
activations. During each forward pass, each parameter is stochastically either the fine-tuned
value OR reset to its pretrained value:

```
p_effective = Mixout(p_finetuned, p_pretrained, mix_prob)
            = p_finetuned   with probability (1 - mix_prob)
            = p_pretrained  with probability mix_prob
```

The effective parameter expectation is:
```
E[p_effective] = (1 - mix_prob) * p_finetuned + mix_prob * p_pretrained
```

At training time, the model is randomly "soft-reset" on individual parameters. The
regularization strength adapts along the optimization trajectory: as `p_finetuned`
drifts far from `p_pretrained`, the stochastic resets create larger gradient signals
pulling back toward the pretrained values.

**Why this is strictly more principled than our exponential-decay soft-reset:**
Our current `A_new = prev_A * decay + noise` decays ALL parameters uniformly. Mixout
decays parameters INDIVIDUALLY and STOCHASTICALLY, which means:
1. Parameters that are consistently important (large gradients) drift far from pretrained
   values and get larger restoring forces.
2. Parameters that are not important for the current document stay close to pretrained.
3. The noise is structured (pretrained values) not random Gaussian.

**Key results from original paper:** On GLUE benchmarks with only small training sets
(few-shot regime), Mixout prevents degenerate fine-tuning performance more effectively
than L2 regularization and standard dropout. Specifically on unstable fine-tuning settings
(small datasets), Mixout reduces variance of outcomes substantially.

**Implementation for TTT LoRA:**

```python
class MixoutLoRA(nn.Module):
    def __init__(self, lora_layer, p_mix=0.3):
        super().__init__()
        self.lora = lora_layer
        self.p_mix = p_mix
        # Store pretrained LoRA values (typically zeros for B, random for A)
        self.A_pretrained = lora_layer.lora_A.data.clone()
        self.B_pretrained = lora_layer.lora_B.data.clone()

    def forward(self, x):
        if self.training:
            # Stochastic mixing per element
            mask_A = torch.bernoulli(
                torch.full_like(self.lora.lora_A.data, self.p_mix)
            )
            mask_B = torch.bernoulli(
                torch.full_like(self.lora.lora_B.data, self.p_mix)
            )
            A_eff = torch.where(
                mask_A.bool(), self.A_pretrained, self.lora.lora_A
            )
            B_eff = torch.where(
                mask_B.bool(), self.B_pretrained, self.lora.lora_B
            )
            # Rescale to maintain expected output magnitude
            A_eff = A_eff / (1 - self.p_mix)
            # ... use A_eff, B_eff in the LoRA forward pass
        else:
            # Standard LoRA at inference
            ...
```

**Practical simplification:** Since our LoRA B is initialized to zero, `B_pretrained = 0`,
so the Mixout mask on B just zeros out some fine-tuned B elements with rescaling — identical
to DropConnect. The interesting part is on A, where pretrained values are non-zero random.

**Complexity:** Moderate (requires a wrapper around LoRA layers). Core logic is 10–15 lines.

---

### 15. Revisiting Mixout: An Overlooked Path to Robust Fine-tuning (2025)

**Citation:** Anonymous. (2025). Revisiting Mixout: An Overlooked Path to Robust Finetuning.
*arXiv:2510.06982*.

**URL:** https://arxiv.org/abs/2510.06982

**Key finding:** Mixout is described as "a stochastic regularizer that intermittently
replaces finetuned weights with their pretrained reference." The 2025 revisit confirms
Mixout remains competitive with modern alternatives (DoRA, LoRA+, GaLore) for few-shot
and unstable fine-tuning settings, and highlights that it was underutilized in the 2022–2024
literature due to being published before the PEFT (Parameter-Efficient Fine-Tuning) wave.

**New finding:** The effectiveness of Mixout degrades when the fine-tuned model diverges
very far from pretrained (e.g., after full fine-tuning on large datasets). For small LoRA
adaptation (our TTT case, typically 10–50 steps), Mixout is in its most effective regime —
the mixing probability p_mix=0.3–0.5 is well-matched to small perturbations from pretrained.

---

## Part V: Dropout as TTT Regularization

### 16. Dropout as a Bayesian Approximation — MC Dropout (Gal & Ghahramani 2016)

**Citation:** Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation:
Representing Model Uncertainty in Deep Learning. *Proceedings of ICML 2016*, Vol. 48,
1050–1059. arXiv:1506.02142.

**URL:** https://arxiv.org/abs/1506.02142

**Core theoretical result:** Training with dropout is equivalent to approximate variational
Bayesian inference in a deep Gaussian process. At test time, applying dropout and taking
K stochastic forward passes produces samples from the approximate posterior p(y|x, data).
The mean of these samples is a more calibrated predictor; the variance gives uncertainty.

**Mechanism for implicit regularization during fine-tuning:** Dropout during training
explicitly penalizes representations that rely on specific neurons. During few-shot fine-
tuning, this forces the model to distribute information across many pathways rather than
sharply memorizing specific token patterns. This prevents the overconfident collapse that
TTT risks.

**Key interaction with TTT:** Standard transformers have dropout layers in attention and
MLP. These are usually disabled during inference. For TTT, keeping dropout ENABLED during
the inner loop (train mode) provides implicit regularization. This is already the default
if dropout_rate > 0 in the model config, but it is worth verifying explicitly.

**MC-Dropout for uncertainty at inference:** After TTT adaptation, use K=10–20 stochastic
forward passes with dropout to estimate predictive uncertainty. High-variance positions
suggest the adapted model is uncertain — these are candidates for temperature scaling or
entropy regularization.

---

### 17. Implicit Regularization of Dropout (2024)

**Citation:** Anonymous. (2024). Implicit Regularization of Dropout. *IEEE Transactions
on Pattern Analysis and Machine Intelligence*, 2024.

**URL:** https://dl.acm.org/doi/abs/10.1109/TPAMI.2024.3357172

**Key finding:** Dropout introduces two distinct but entangled regularization effects:
1. **Explicit effect:** Adds noise to parameters, analogous to L2 weight decay on the
   expected output.
2. **Implicit effect:** Modifies the loss landscape curvature — it implicitly penalizes
   parameter configurations where nearby points have very different loss values
   (sharp minima). This is analogous to sharpness-aware minimization (SAM).

**TTT relevance:** The implicit sharpness penalty is exactly what prevents overconfident
fine-tuning. Sharp loss minima correspond to overfit solutions. By keeping dropout active
during TTT, we are implicitly applying SAM-like regularization without any additional
compute.

---

### 18. Fine-tuning with Very Large Dropout

**Citation:** Zhang, J., & Bottou, L. (2024). Fine-tuning with Very Large Dropout.
*arXiv:2403.00946*. ICLR 2024.

**URL:** https://arxiv.org/abs/2403.00946

**Core finding:** Using very large dropout rates (p=0.5–0.9) during fine-tuning
substantially improves out-of-distribution (OOD) performance — exceeding both ensembles
and weight averaging methods. A 90% dropout rate (masking 90% of representation units)
reliably produces good OOD performance across four tasks.

**Mechanism:** Very large dropout forces the model to form redundant representations:
any small subset of features must be sufficient to make good predictions. This redundancy
generalizes better OOD.

**For TTT:** Very large dropout (p=0.5) on the ACTIVATIONS passing through the LoRA
adapter (not on the LoRA parameters themselves) would be a strong regularizer. But this
also slows convergence. Recommended: p=0.2–0.3 as a compromise for the short TTT inner
loop (10–50 steps).

**Critical note on LoRA + very large dropout:** LoRA itself is a low-rank bottleneck. If
you apply very large activation dropout ON TOP of the LoRA bottleneck, you might kill the
signal entirely (dropout on a rank-4 projection leaves very few degrees of freedom).
Scale dropout rate down proportionally to LoRA rank: `p_dropout ~ 1 - rank/hidden_dim * C`
where C is a coverage constant.

---

## Part VI: Uncertainty-Calibrated TTT without Forgetting

### 19. EATA / EATA-C: Uncertainty-Calibrated Test-Time Model Adaptation without Forgetting

**Citation:** Tang, S., et al. (2024). Uncertainty-Calibrated Test-Time Model Adaptation
without Forgetting. *arXiv:2403.11491*. IEEE Transactions on Neural Networks and Learning
Systems, 2024.

**URL:** https://arxiv.org/abs/2403.11491

**This is the most directly relevant paper to our exact problem.** It addresses BOTH
anti-forgetting AND calibration within a TTT framework simultaneously.

**Two distinct contributions:**

**Contribution 1 — EATA (Efficient Anti-Forgetting Test-Time Adaptation):**
- Implements Fisher regularization estimated FROM TEST SAMPLES (not training samples):
```
L_EATA = L_entropy(x_test) + lambda * Σ_i F_i * (theta_i - theta_0_i)^2
```
Where `F_i` is computed from the test batch itself (recent test samples), and `theta_0`
is the initial model before adaptation begins. This is EWC applied online to TTT.

- Sample selection: only uses "reliable" samples (low entropy = high confidence) and
  "non-redundant" samples (high divergence from previously seen test samples) for the
  entropy minimization objective. This prevents the optimizer from spending capacity on
  ambiguous or redundant inputs.

**Contribution 2 — EATA-C (EATA with Calibration):**
The key insight is that data uncertainty is often overlooked by entropy minimization.
Entropy minimization alone causes the model to become overconfident (low entropy = high
certainty) even on genuinely ambiguous inputs. EATA-C introduces a MIN-MAX entropy
regularizer:

```
L_EATA_C = L_min_entropy(reliable samples)      # minimize entropy for confident preds
           + L_max_entropy(unreliable samples)   # maximize entropy for uncertain preds
           + lambda * Fisher_regularizer
```

Where reliable/unreliable is determined by comparing model uncertainty (disagreement
between full network and sub-network predictions) with inherent data uncertainty
(disagreement among predicted labels for augmented views).

**Mechanism for measuring data vs model uncertainty:**
- **Model uncertainty:** KL(p_full_network || p_sub_network) where the sub-network is
  created by dropping some blocks or heads.
- **Data uncertainty:** Disagreement of predicted labels across augmented views of x.
- If model uncertainty is HIGH but data uncertainty is LOW: the model is genuinely
  uncertain about an easy example — force it to increase confidence (minimize entropy).
- If both are HIGH: the data is genuinely ambiguous — force the model to maintain
  high entropy (maximize entropy).

**Results:** Achieves significant improvements on image classification and semantic
segmentation OOD benchmarks versus entropy-only TTA baselines.

**Implementation complexity for TTT on language models:** Moderate-to-high. The
reliable/unreliable sample discrimination requires either sub-network evaluation (expensive)
or a proxy metric. For TTT on text, a simpler proxy for data uncertainty:
- **High data uncertainty proxy:** If the same position's top-2 tokens are semantically
  related (e.g., synonym pair), the position is inherently uncertain.
- **Simple version:** Use entropy threshold: positions with entropy < E_min get entropy
  minimized; positions with entropy > E_max get entropy maximized. This is 5 lines.

```python
# Simplified EATA-C for TTT language modeling
per_token_entropy = -(p * p.log()).sum(-1)  # [batch, seq_len]
reliable_mask = per_token_entropy < entropy_threshold_low
uncertain_mask = per_token_entropy > entropy_threshold_high

loss = (
    -per_token_entropy[reliable_mask].mean()   # minimize entropy for confident
    + per_token_entropy[uncertain_mask].mean() # maximize entropy for uncertain
    + lambda_ewc * fisher_regularizer
)
```

---

## Part VII: Contrastive Learning for TTT Calibration

### 20. Contrastive Test-Time Adaptation (CTA)

**Citation:** Chen, D., et al. (2022). Contrastive Test-Time Adaptation. *CVPR 2022*.

**URL:** https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Contrastive_Test-Time_Adaptation_CVPR_2022_paper.pdf

**Core mechanism:** During TTT, apply InfoNCE contrastive loss between augmented views
of the same input (positive pairs) and different inputs (negative pairs). The contrastive
objective prevents representation collapse that entropy minimization alone risks — if all
inputs map to the same confident prediction, entropy is minimized but representations are
useless. InfoNCE forces different inputs to remain separated.

**InfoNCE loss:**
```
L_InfoNCE = -log( exp(sim(z_i, z_j+) / tau) /
                  Σ_k exp(sim(z_i, z_k) / tau) )
```
Where `z_i` is the representation of input i, `z_j+` is its augmented positive, and
all other `z_k` in the batch are negatives.

**For language model TTT:** Text augmentation is non-trivial (unlike image augmentation).
Viable augmentations for language TTT:
1. **Token dropout:** Randomly mask 15% of tokens (BERT-style MLM masks).
2. **Span permutation:** Shuffle short spans (2–4 tokens).
3. **Prefix truncation:** Use different prefix lengths to generate views.

The contrastive loss on [CLS]-equivalent representations (or mean-pooled hidden states)
maintains diversity across the document's different subregions.

**Complexity:** Moderate (requires augmentation pipeline + contrastive loss). More suited
to the outer training loop than the inner TTT loop. Not recommended as the first experiment
to try.

---

### 21. Open-World Test-Time Training with Contrastive Learning

**Citation:** Anonymous. (2024). Open-World Test-Time Training: Self-Training with
Contrast Learning. *arXiv:2409.09591*.

**URL:** https://arxiv.org/abs/2409.09591

**Key contribution:** Open-World Dynamic Contrastive Learning (OWDCL) applies NT-XENT
contrastive loss during TTT specifically to handle out-of-distribution inputs that
interfere with in-distribution predictions (preventing what they call "premature
classification of classes as strong OOD").

**NT-XENT loss (normalized temperature-scaled cross entropy):**
```
L_NT-XENT = -log( exp(sim(z_i, z_j) / tau) /
                  Σ_k≠i exp(sim(z_i, z_k) / tau) )
```
This is SimCLR's version of InfoNCE.

**Forgetting prevention mechanism:** The contrastive objective on data-augmented views
maintains the structure of the representation space. Even if the model's classifier head
shifts toward the test document distribution, the encoder's representations remain
diverse and well-separated, preserving the base model's ability to handle other inputs.

**For our TTT:** This targets the cross-document forgetting problem rather than within-
document overconfidence. Worth considering if sequential document processing shows
degradation across document boundaries.

---

## Part VIII: EWC-LoRA and Weight Regularization in Context

### 22. Revisiting Weight Regularization for Low-Rank Continual Learning

**Citation:** Anonymous. (2025). Revisiting Weight Regularization for Low-Rank Continual
Learning. *OpenReview 2025*.

**URL:** https://openreview.net/forum?id=pZj2DhfaVD

**Key finding:** EWC-LoRA leverages a low-rank representation to estimate parameter
importance over the full-dimensional space. Unlike existing low-rank continual learning
methods, it mitigates task interference by regularizing a shared low-rank update through
EWC, keeping storage constant regardless of the number of tasks. EWC-LoRA improves over
vanilla LoRA by 8.92% on continual learning benchmarks.

**How EWC-LoRA compares to DARE/Mixout/KD for TTT:**

| Method          | Mechanism              | Complexity | BPB Impact |
|-----------------|------------------------|------------|------------|
| EWC-LoRA        | Fisher * (delta)^2     | Moderate   | High       |
| Mixout          | Stochastic param reset | Low-Med    | Medium-High|
| KD Regularizer  | KL to frozen base      | Low        | Medium-High|
| DARE masking    | Random delta zeroing   | Low        | Medium     |
| Label Smoothing | Soft targets           | Trivial    | Low-Medium |
| Large Dropout   | Activation masking     | Low        | Medium     |
| EATA-C          | Min-max entropy + EWC  | High       | High       |
| UQ4CT Ensemble  | Ensemble LoRA + MoE    | Very High  | Very High  |

---

## Part IX: Synthesis and Experiment Recommendations

### Priority Tier 1 — Implement First (1–5 lines each)

**TTT-KD: KL Distillation Regularizer**

Add a term to the TTT loss that penalizes KL divergence from the frozen base model:

```python
with torch.no_grad():
    base_logits = base_model(tokens)
T = 2.0; alpha = 0.2
kl = F.kl_div(
    F.log_softmax(adapted_logits / T, dim=-1),
    F.softmax(base_logits / T, dim=-1),
    reduction='batchmean'
) * T**2
ttt_loss = (1 - alpha) * ce_loss + alpha * kl
```

**Expected BPB impact:** Moderate improvement (-0.02 to -0.05 BPB vs unregularized TTT),
primarily from reduced overconfidence on the test document which improves held-out BPB.

**Key risk:** Adds one extra frozen forward pass per TTT step. If the base model cannot
fit in VRAM alongside the adapted model, this is not free.

---

**TTT-LS: Label Smoothing in TTT CE loss**

Replace `F.cross_entropy(logits, targets)` with
`F.cross_entropy(logits, targets, label_smoothing=0.05)` in the TTT inner loop.

**Expected BPB impact:** Small (-0.01 to -0.02 BPB). The 2025 paper warns effectiveness
diminishes at smaller model scales. Still worth doing as a no-cost baseline.

**Caveat:** Do not combine with KD regularizer from a label-smoothing-trained base model
(destructive interaction per Muller et al. 2019).

---

**TTT-DARE: Sparse LoRA Delta Regularization**

After each TTT optimizer step, randomly zero p=0.3 of LoRA delta parameters and rescale:

```python
with torch.no_grad():
    for p in lora_params:
        mask = torch.bernoulli(torch.full_like(p.data, 0.7))
        p.data = p.data * mask / 0.7
```

**Expected BPB impact:** Small-to-moderate. Keeps LoRA deltas sparse and close to the
pretrained subspace. Most useful for long documents where LoRA can drift significantly.

---

### Priority Tier 2 — Medium Complexity, High Potential

**TTT-Mixout: Stochastic Parameter Reset**

Wrap LoRA layers with Mixout regularizer (mix_prob=0.3–0.4). At each forward pass during
TTT, 30–40% of LoRA parameters are stochastically reset to their pretrained values.

**Expected BPB impact:** Medium-high. Most effective for short adaptation runs (our 10–50
step TTT) in the regime where Mixout is well-validated. Better than L2 regularization for
preventing degenerate fine-tuning.

**Caution:** The implementation needs to correctly handle the rescaling so that the EXPECTED
LoRA output is preserved (otherwise convergence is disrupted).

---

**TTT-EATA-C (simplified): Min-Max Entropy with Fisher Regularizer**

Add EATA-C style entropy regularization based on per-token confidence:

```python
entropy = -(probs * probs.log()).sum(-1)
hi_conf = entropy < low_thresh    # minimize entropy here
lo_conf = entropy > high_thresh   # maximize entropy here
eata_loss = -entropy[hi_conf].mean() + entropy[lo_conf].mean()
ttt_loss = ce_loss + beta * eata_loss + lambda_ewc * fisher_reg
```

**Expected BPB impact:** High. This directly addresses overconfidence (entropy maximization
on uncertain positions) and underconfidence (entropy minimization on confident ones).

---

### Priority Tier 3 — High Complexity, Validate After Tier 1-2

**MC-LoRA / UQ4CT Lite:** Train 3–5 LoRA adapters per document with different random seeds
for A initialization. Use their prediction variance as an uncertainty signal during
inference. Average predictions for better calibration.

**Expected BPB impact:** Very high potential, but 3–5x LoRA compute overhead per TTT step.
Best implemented after single-LoRA optimizations plateau.

**Contrastive TTT:** Add NT-XENT loss over masked/augmented views of the document to
maintain representation diversity. Worth trying after all simpler regularizers are validated.

---

### Critical Warning: Interaction Between Techniques

The following combinations are PROBLEMATIC:

1. **Label smoothing + KD regularizer on a base model trained with label smoothing:**
   The teacher's soft labels lose inter-class similarity (tight cluster effect from
   Muller et al. 2019). The KD term provides no useful signal beyond hard labels.
   Solution: Use either label smoothing OR KD, not both.

2. **Very large dropout (p>0.5) + Low LoRA rank (r=4):**
   Dropping 50%+ of a rank-4 projection kills the signal (effectively reduces rank to 2).
   Solution: Scale dropout inversely with LoRA rank, or apply dropout to the BASE model
   residual stream, not inside the LoRA path.

3. **EATA-C entropy maximization + KD regularizer:**
   Entropy maximization at uncertain positions conflicts with KD which pushes toward the
   teacher's potentially sharper distribution. Solution: Disable KD at positions where
   EATA-C applies entropy maximization (use the same uncertainty mask).

4. **EWC-LoRA Fisher regularization + DARE masking:**
   DARE randomly zeros LoRA parameters regardless of their Fisher importance. Fisher-
   important parameters should NOT be zeroed. Solution: Apply DARE only to LOW Fisher-
   importance parameters (invert the EWC selection logic).

---

## Appendix: Complete Bibliography

1. Hinton, G., Vinyals, O., & Dean, J. (2015). Distilling the Knowledge in a Neural
   Network. arXiv:1503.02531. https://arxiv.org/abs/1503.02531

2. Muller, R., Kornblith, S., & Hinton, G. E. (2019). When Does Label Smoothing Help?
   NeurIPS 2019. arXiv:1906.02629. https://arxiv.org/abs/1906.02629

3. Gal, Y., & Ghahramani, Z. (2016). Dropout as a Bayesian Approximation: Representing
   Model Uncertainty in Deep Learning. ICML 2016. arXiv:1506.02142.
   https://arxiv.org/abs/1506.02142

4. Lee, K., et al. (2020). Mixout: Effective Regularization to Finetune Large-scale
   Pretrained Language Models. ICLR 2020. arXiv:1909.11299.
   https://arxiv.org/abs/1909.11299

5. Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks.
   PNAS 114(13):3521–3526. arXiv:1612.00796.

6. Yadav, P., et al. (2023). TIES-MERGING: Resolving Interference When Merging Models.
   NeurIPS 2023. arXiv:2306.01708. https://arxiv.org/abs/2306.01708

7. Yu, L., et al. (2023). Language Models are Super Mario: Absorbing Abilities from
   Homologous Models as a Free Lunch. ICML 2024. arXiv:2311.03099.
   https://arxiv.org/abs/2311.03099

8. Tang, S., et al. (2024). Uncertainty-Calibrated Test-Time Model Adaptation without
   Forgetting. IEEE TNNLS 2024. arXiv:2403.11491. https://arxiv.org/abs/2403.11491

9. Zhang, J., & Bottou, L. (2024). Fine-tuning with Very Large Dropout. ICLR 2024.
   arXiv:2403.00946. https://arxiv.org/abs/2403.00946

10. Anonymous. (2024). Functional-level Uncertainty Quantification for Calibrated
    Fine-tuning on LLMs. arXiv:2410.06431. https://arxiv.org/abs/2410.06431

11. Anonymous. (2024). Calibration Transfer via Knowledge Distillation. ACCV 2024.
    https://openaccess.thecvf.com/content/ACCV2024/papers/Hebbalaguppe_Calibration_Transfer_via_Knowledge_Distillation_ACCV_2024_paper.pdf

12. Anonymous. (2024). Calibrating Language Models with Adaptive Temperature Scaling.
    arXiv:2409.19817. https://ar5iv.labs.arxiv.org/html/2409.19817

13. Anonymous. (2024). Open-World Test-Time Training: Self-Training with Contrast Learning.
    arXiv:2409.09591. https://arxiv.org/abs/2409.09591

14. Anonymous. (2024). Implicit Regularization of Dropout. IEEE TPAMI 2024.
    https://dl.acm.org/doi/abs/10.1109/TPAMI.2024.3357172

15. Anonymous. (2025). Restoring Calibration for Aligned LLMs. arXiv:2505.01997.
    https://arxiv.org/abs/2505.01997

16. Anonymous. (2025). Calibrated Language Models and How to Find Them with Label
    Smoothing. arXiv:2508.00264. https://arxiv.org/abs/2508.00264

17. Anonymous. (2025). The Role of Teacher Calibration in Knowledge Distillation.
    arXiv:2508.20224. https://arxiv.org/abs/2508.20224

18. Anonymous. (2025). Revisiting Mixout: An Overlooked Path to Robust Finetuning.
    arXiv:2510.06982. https://arxiv.org/abs/2510.06982

19. Anonymous. (2025). Merge before Forget: A Single LoRA Continual Learning via
    Continual Merging. arXiv:2512.23017. https://arxiv.org/abs/2512.23017

20. Anonymous. (2025). Revisiting Weight Regularization for Low-Rank Continual Learning.
    OpenReview 2025. https://openreview.net/forum?id=pZj2DhfaVD

21. Anonymous. (2025). LoRI: Reducing Cross-Task Interference in Multi-Task Low-Rank
    Adaptation. arXiv:2504.07448. https://arxiv.org/abs/2504.07448

22. Shenfeld, I., Damani, M., Hubotter, J., & Agrawal, P. (2026). Self-Distillation
    Enables Continual Learning. arXiv:2601.19897. https://arxiv.org/abs/2601.19897

23. Anonymous. (2026). Test-Time Learning for Large Language Models. arXiv:2505.20633.
    ICML 2025. https://arxiv.org/abs/2505.20633

24. Chen, D., et al. (2022). Contrastive Test-Time Adaptation. CVPR 2022.
    https://openaccess.thecvf.com/content/CVPR2022/papers/Chen_Contrastive_Test-Time_Adaptation_CVPR_2022_paper.pdf
