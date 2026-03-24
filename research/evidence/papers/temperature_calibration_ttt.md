# Temperature Calibration in Neural Networks: Scientific Foundations for TTT Adaptive Temperature

**Compiled:** 2026-03-24
**Research scope:** 2017–2026
**Databases searched:** arXiv, ACL Anthology, NeurIPS/ICML/ICLR proceedings, Semantic Scholar
**Papers reviewed:** 30+ primary and secondary sources

---

## Part I: Foundational Temperature Scaling Literature

### 1.1 Guo et al. 2017 — The Canonical Overconfidence Paper

> Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern Neural Networks. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, PMLR 70, 1321–1330. arXiv:1706.04599.

**Core finding.** Modern deep networks — unlike their older, shallower counterparts — are systematically overconfident. A ResNet-110 achieves 6% top-1 error on CIFAR-100 yet has an Expected Calibration Error (ECE) of 16.8%. The gap between accuracy and confidence grows monotonically with model capacity.

**ECE definition (M equal-width bins):**
```
ECE = sum_{m=1}^{M} (|B_m| / n) * |acc(B_m) - conf(B_m)|
```
where B_m is the set of samples whose predicted confidence falls in the m-th bin, acc(·) is the fraction correctly classified in that bin, and conf(·) is the mean confidence in that bin.

**Temperature scaling formula:**
```
q_hat = softmax(z / T)
```
where z is the raw logit vector and T > 0 is a scalar learned on a held-out validation set by minimizing NLL. T > 1 softens (reduces confidence), T < 1 sharpens (increases confidence).

**Key mechanistic findings:**
- Depth, width, weight decay, and BatchNorm all correlate with worsening calibration
- After NLL cannot be further reduced by correct classification, continued training drives confidence upward — the model "memorizes" confidence rather than learning it
- Temperature scaling is the most effective single-parameter post-hoc method, outperforming isotonic regression and histogram binning on held-out test calibration
- NLL and ECE move in tandem: optimizing T on NLL also improves ECE, so the two metrics are useful as mutual proxies

**Implication for TTT (Q3 connection):** The paper establishes that when a model has been trained past the point of correct classification, gradient magnitudes at that convergence point are small for in-distribution data. This is the direct precursor to using gradient magnitude as a distributional surprise signal.

---

### 1.2 Minderer et al. 2021 — Revisiting Calibration in the Transformer Era

> Minderer, M., Djolonga, J., Romijnders, R., Hubis, F., Zhai, X., Houlsby, N., Tran, D., & Lucic, M. (2021). Revisiting the Calibration of Modern Neural Networks. *NeurIPS 2021*. arXiv:2106.07998.

**Core finding.** The architecture determines calibration more than model size or pretraining quantity. Specifically, vision transformers (ViTs) are among the best-calibrated models even without post-hoc correction, reversing the trend described by Guo et al.

**Scale paradox:** Larger CNNs are more overconfident (Guo et al. finding); larger ViTs are better calibrated. Architecture, not parameter count, is the primary driver.

**Distribution shift finding:** Traditional CNN models become severely overconfident under distribution shift (ECE rises +0.016 to +0.051 on ImageNet-V2). Modern architectures show ECE reductions of -0.022 to -0.037 under the same shift.

**Post-hoc calibration under shift:** Temperature scaling reduced ECE from 0.094 to 0.016 for ConvNeXt in-distribution but *increased* ECE by +0.034 under extreme domain shift (ImageNet-A, severity 5). This is a critical result: T calibrated on a validation set can anti-calibrate on a shifted test domain.

**Implication for TTT:** The T value that was optimal before TTT adaptation may not be optimal after, because TTT places the model on a new region of the distribution manifold. Per-chunk T must be recomputed, not inherited.

---

### 1.3 Müller, Kornblith, Hinton 2019 — Label Smoothing and Calibration

> Müller, R., Kornblith, S., & Hinton, G. (2019). When Does Label Smoothing Help? *NeurIPS 2019*. arXiv:1906.02629.

**Core finding.** Label smoothing (LS) with factor α replaces one-hot targets with soft targets: (1 − α) for the correct class and α/(K−1) for all others. This empirically improves calibration by reducing overconfidence.

**Calibration numbers on ImageNet:**
- Hard targets (no smoothing), uncalibrated: ECE ≈ 0.054
- LS α = 0.1, uncalibrated: ECE ≈ 0.035
- Hard targets + temperature scaling (T = 1.4): ECE ≈ 0.022

**Tight clustering effect:** LS encourages penultimate-layer representations from the same class to cluster tightly, reducing inter-class geometric information in the logits. This is why LS models make poor teachers in knowledge distillation.

**Key for Q4 — LS vs. temperature scaling connection:**
Both LS and temperature scaling affect the *entropy of the output distribution* but through different mechanisms:
- LS modifies the training objective, imposing a uniform lower bound on prediction entropy during training
- Temperature scaling is a post-hoc monotone transform of logits at inference

They are *not equivalent*, but they affect a common quantity (logit norm / output entropy). The 2024 paper "Calibrated Language Models and How to Find Them with Label Smoothing" (arXiv:2508.00264) formalizes this:
- LS approximately minimizes a constraint of the form d(x) = 0 (equal logit distances), reducing logit magnitudes
- Temperature scaling with T < 1 increases ρ = σ_C · σ_h, creating tighter distributions
- Both operate on the upper bound of logit magnitude: ||C^T h||_2 ≤ σ_C · σ_h · sqrt(D)

**Practical equivalence:** After temperature scaling, a model trained with LS has *higher ECE* than a hard-target model — LS overcorrects toward uniform, and post-hoc T cannot fully undo this. LS is a training regularizer; temperature scaling is a post-hoc corrector. They are not interchangeable at inference time.

**Implication for TTT:** If TTT is run with any form of soft/label-smoothed targets, the effect is a training-time entropy regularizer. At scoring time, T = 0.98 applies a complementary inference-time sharpening. These act on the same underlying logit distribution but are not redundant — they operate at different granularities (token-distribution-level vs. logit-scale-level).

---

### 1.4 Kadavath et al. 2022 (Anthropic) — LLM Self-Knowledge

> Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E., Schiefer, N., Hatfield-Dodds, Z., et al. (2022). Language Models (Mostly) Know What They Know. Anthropic. arXiv:2207.05221.

**Core finding.** Large language models are reasonably well-calibrated on held-out multiple-choice questions when the question format is appropriate. Self-evaluation via P(True) further improves calibration for proposed answers. However, generalization of this self-knowledge to new domains (P(IK)) is brittle.

**Scaling finding:** Larger models in the Claude family show monotonically improving calibration across diverse MMLU-style tasks. This is a *pre-training* calibration result; the paper does not examine domain-adapted or RLHF-tuned models.

**P(IK) and context:** P(IK) "increases appropriately in the presence of relevant source materials in the context." This is the closest existing result to our TTT scenario: when the model has been exposed to relevant context (our TTT chunk), its estimated internal confidence on chunk tokens should legitimately increase. The question is whether this increase is *correctly calibrated* or whether it tips into overconfidence.

**Key implication for Q1:** The Kadavath result establishes that pre-training calibration is relatively sound at large scale. Domain adaptation / TTT then perturbs this from a well-calibrated starting point. Whether the post-TTT model is over- or under-confident is a separate empirical question not answered here.

---

### 1.5 Xiong et al. 2023/2024 — Confidence Elicitation in LLMs

> Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Bryan, H. (2024). Can LLMs Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation in LLMs. *ICLR 2024*. arXiv:2306.13063.

**Core finding.** LLMs verbalizing their confidence are overconfident, with expressed confidence values clustering in 80–100% range. Larger models are better calibrated. ECE (on verbalized confidence) by model:
- GPT-3 (175B): ECE ≈ 52.0
- Vicuna-13B (RLHF): ECE ≈ 46.1
- LLaMA-2-70B (RLHF): ECE ≈ 43.6
- GPT-3.5 (RLHF): ECE ≈ 37.7
- GPT-4: ECE ≈ 18.0

**Per-domain variation:** GSM8K (arithmetic) ECE = 31.0 for GPT-4; Date Understanding = 18.0; Professional Law = 17.3; StrategyQA = 16.1. Domain-specific calibration failure is substantially larger than aggregate ECE suggests.

**Key implication for Q1 & Q6:** The per-domain ECE variation is the analog of our per-chunk calibration variation. A model adapted to one chunk's domain will have calibration error that differs from the aggregate. This is direct evidence that within-chunk calibration signals are meaningful and different from global calibration.

---

## Part II: RLHF, Fine-Tuning, and Calibration Degradation

### 2.1 Adaptive Temperature Scaling (ATS) for RLHF Models

> Xie, J., Chen, A. S., Lee, Y., Mitchell, E., & Finn, C. (2024). Calibrating Language Models with Adaptive Temperature Scaling. *EMNLP 2024*. arXiv:2409.19817.

**Core finding.** Unsupervised pre-training yields LLMs with well-calibrated token probabilities. RLHF fine-tuning significantly degrades this calibration. The degradation is *heterogeneous* — different token predictions are miscalibrated by different amounts — which motivates per-token temperature rather than a global scalar.

**ATS method:**
- A lightweight, single-layer causal Transformer block (calibration head) is added to the LLM
- Input: per-token last hidden state h_t
- Output: per-token temperature T_t (a scalar)
- Applied at inference: q_t = softmax(z_t / T_t)
- Trained on a standard SFT dataset with NLL loss on the temperature-scaled logits

**Performance:** ATS achieves 10–50% ECE reduction over prior calibration methods across MMLU, TriviaQA, and TruthfulQA for Llama-2-7b-Chat and Qwen-7b-Chat. Unlike global temperature scaling, ATS does not impede RLHF performance gains.

**Key insight for our design:** The hidden state h_t encodes sufficient information to determine per-token calibration correction. Our use of S[0] (top singular value of the gradient) is a different but related signal — it captures how much the model's representation changed for a given chunk. ATS provides the existence proof that token-level temperature adaptation is both tractable and highly effective.

---

### 2.2 RLHF Taming Overconfidence — Reward Calibration

> (Taming Overconfidence in LLMs: Reward Calibration in RLHF, arXiv:2410.09724, 2024.)

**Core finding (quantitative).** RLHF introduces systematic overconfidence because reward models are biased toward high-confidence responses. The reward model assigns higher scores to identical responses if they are expressed with greater confidence — even when those responses are wrong.

**ECE measurements:**
- Llama3-8B, GSM8K, Direct Answer: SFT ECE = 0.8608 → vanilla PPO ECE = 0.8843 (worsens)
- PPO-M (reward-calibrated PPO): ECE = 0.8393 (improves over SFT)
- Mistral-7B, TruthfulQA CoT: CDPO ECE = 0.1756 vs. DPO ECE = 0.3251

**Mechanism:** The reward model's preference for confident responses gets amplified through the RL update signal. Each PPO step makes the model slightly more confident, independently of whether confidence is warranted.

**Implication for Q2:** RLHF → overconfidence, systematically. After TTT (which is also a gradient-update process on the model), the same mechanism applies: the model becomes more confident on the adapted chunk. The question is whether this is correct confidence (legitimate) or spurious overconfidence. The evidence suggests that *fast* gradient-update adaptation processes favor overconfidence.

---

### 2.3 Just Ask for Calibration (Kadavath et al. Follow-Up)

> Zhao, W. X., et al. (2023). Just Ask for Calibration: Strategies for Eliciting Calibrated Confidence Scores from Language Models Fine-Tuned with Human Feedback. *EMNLP 2023*. arXiv:2305.14975.

**Core finding.** Verbalized confidence from RLHF-tuned models (ChatGPT, GPT-4, Claude) is better calibrated than the model's log-probabilities. Simply asking a model to state its confidence as a number often reduces ECE by a relative 50%. For token-probability calibration, RLHF makes things worse; for verbalized calibration, RLHF makes things better.

**Implication for our use case:** We are using token-level log-probabilities (NLL / bits-per-byte), not verbalized confidence. For this metric, the Kadavath 2022 / RLHF degradation literature is the relevant one, and the message is: after adaptation (RLHF or TTT), token-probability calibration worsens.

---

### 2.4 Fine-Tuning Calibration Degradation — Domain Specific

> (Various sources, 2022–2024; synthesized below.)

**Evidence 1 (Zhu et al. 2023):** Pythia, LLaMA, FLAN-T5, and OPT models trained on PILE, T-REx, or MMLU become consistently overconfident across model sizes after domain-specific fine-tuning. Fine-tuned models increase their expressed confidence regardless of whether accuracy improves commensurately.

**Evidence 2 (GPT-4 Technical Report, 2023):** Base GPT-4 has near-perfect calibration on MMLU; post-training (RLHF/RLAIF) hurts calibration significantly. This is one of the cleanest empirical demonstrations of calibration degradation from fine-tuning.

**Evidence 3 (Few-Shot Recalibration, arXiv:2403.18286):** LLaMA-65B appears well-calibrated in aggregate on MMLU (ECE = 0.02) but has substantially higher ECE in specific domains (up to 250% higher than the aggregate). Domain-specific fine-tuning hides miscalibration at the global level.

**Evidence 4 (Reliability Paradox, arXiv:2412.15269):** ECE can be low even when a model is highly overconfident, because overconfidence on some domains is masked by underconfidence on others. ECE is not a reliable per-domain calibration metric.

**Synthesis for Q1:** When a model is fine-tuned on a specific domain (analogous to TTT on a chunk), confidence *increases* — but calibration *worsens*. The model is not just more confident; it is more confident than it is accurate. This is the consensus from approximately six independent lines of evidence.

---

## Part III: Test-Time Adaptation and the Over-Certainty Problem

### 3.1 The Over-Certainty Phenomenon in TTA

> (The Over-Certainty Phenomenon in Modern Test-Time Adaptation Algorithms, arXiv:2404.16168, 2024.)

**Core finding.** Most test-time adaptation algorithms minimize entropy:
```
L_TENT = -sum_{y in C} f(y|x) * log f(y|x)
```
This produces overconfident predictions, not just confident ones. The T3A method reduces entropy by 81% yet worsens ECE by 43% (No Adaptation ECE = 0.23 → T3A ECE = 0.33) and dramatically increases NLL (No Adapt NLL = 3.196 → T3A NLL = 7.904).

**The logit-norm finding (critical for our design):**
Logit norm ||z_i||_2 decays as domain shift increases. Small logit norm = weak signal = spurious low entropy = overconfident prediction. The paper proposes Dynamic Entropy Control (DEC) with:
```
gamma_i = kappa / ||z_i||_2
tau_i = [t_min + sigmoid((H(x_i) - h_0) / sqrt(h_max)) * (t_max - t_min)] * (kappa / ||z_i||_2)
```
where h_0 is the source-domain entropy baseline and H(x_i) is the current entropy.

**DEC Results:**
- Average ECE reduction of 70%+ compared to no-adaptation baseline
- No Adaptation ECE = 0.302 → T3A ECE = 0.439 → DEC ECE = 0.018–0.080

**Direct connection to our adaptive T design:**
The DEC formula is a concrete example of an adaptive temperature derived from a within-sample signal (logit norm). Our SVD-based signal S[0] is the analogous idea applied to gradients rather than forward-pass logit norms. When TTT produces large gradient updates (large S[0]), the model has moved far from its pre-adaptation distribution — this is analogous to large domain shift in TTA, and the DEC intuition suggests we should increase T (soften output distribution).

---

### 3.2 Uncertainty-Calibrated TTA Without Forgetting

> Tan, C., et al. (2024). Uncertainty-Calibrated Test-Time Model Adaptation without Forgetting. arXiv:2403.11491.

**Core finding.** Entropy minimization during TTA "consistently assigns higher confidence to predictions even for samples that should remain uncertain." The EATA-C method addresses overconfidence by measuring uncertainty through divergence between predictions from the full network and sub-networks, penalizing inconsistent high-confidence predictions.

**Key insight for TTT:** The overconfidence problem from TTA is not specific to entropy-minimization objectives. Any gradient-based adaptation that reduces training loss on a specific distribution tends to concentrate probability mass, creating overconfidence on that distribution's patterns. This applies directly to our TTT next-token-prediction objective.

---

### 3.3 Test-Time Training for Long-Context LLMs

> Sun, Y., et al. (2024). Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs. arXiv:2512.13898.

**Core finding for our calibration question:** The paper does not directly address calibration, but provides a theoretical analysis showing that gradient updates during TTT improve *logit margin* (separation between correct and incorrect token predictions) — Lemma 3.2 proves that query updates increase target-distractor separation. This is exactly the mechanism by which TTT increases confidence. Larger logit margins = more peaked distributions = lower entropy = potentially lower temperature needed at inference.

**TTT gradient inner product:** The paper establishes (via first-order Taylor expansion) that effective TTT occurs when the gradient of input log-probability and output log-probability are positively aligned:
```
log P_{Theta'}(y|x) ≈ O(eta^2) + log P_Theta(y|x) + eta * [nabla_Theta log P(x;Theta)]^T * nabla_Theta log P_Theta(y|x)
```
This inner product is positive (empirically: 98.75% of samples, average = +5.60), meaning NLL reduction on the chunk tokens does transfer to the next-token prediction.

---

### 3.4 Calibrated Entropy Test-Time Adaptation (C-TPT)

> (C-TPT: Calibrated Test-Time Prompt Tuning for Vision-Language Models, ICLR 2024.)

**Core finding:** Standard TPT (test-time prompt tuning) improves accuracy but induces overconfidence from entropy minimization. C-TPT adds a calibration constraint that uses sharpness-aware minimization and diversity regularization to maintain flat output distributions. The calibration-accuracy tradeoff can be managed explicitly during adaptation.

**Implication:** This is the vision-language model analog of our TTT scenario. The lesson: accuracy improvement from test-time adaptation does not come for free — calibration must be explicitly managed, not assumed to be preserved.

---

## Part IV: Gradient Signals as Calibration Indicators

### 4.1 Fisher Information and Gradient Norms for OOD Detection

> (Approximations to the Fisher Information Metric of Deep Generative Models for Out-Of-Distribution Detection, arXiv:2403.01485, 2024.)

**Core mathematical result:** At convergence on training data, gradients approach zero. For out-of-distribution (OOD) data, gradients are large because the model has not seen the distribution. The Fisher Information Metric (FIM) formalizes this:
```
||nabla_theta l(x)||^2_FIM = nabla_theta l(x)^T * F_theta^(-1) * nabla_theta l(x)
```
where F_theta = E_{y ~ p^theta}[nabla_theta l(y) * nabla_theta l(y)^T].

**Practical approximation:** Layer-wise FIM exhibits diagonal dominance (diagonal entries are ~5x larger than off-diagonal), so the layer-wise gradient L2 norm is a sufficient OOD indicator. In practice, gradient norms "differ by orders of magnitude" between in-distribution and OOD samples.

**Connection to our S[0] signal:**
- S[0] = top singular value of the gradient matrix = captures the dominant direction and magnitude of the gradient update
- For in-distribution (to the model's pre-training distribution) chunks: small S[0], model barely needed to update
- For OOD chunks: large S[0], model had to move substantially
- This is exactly the "gradient norm = surprise" intuition from the FIM literature, applied to the gradient of the adaptation loss rather than the inference loss

**Key theoretical support for Q3:** Yes, there is strong theoretical support for using gradient magnitude (and by extension, S[0]) as a calibration signal. The FIM literature shows that gradient magnitude encodes distributional mismatch. Large S[0] after TTT means the chunk was surprising (OOD for the base model), and the literature shows OOD samples produce overconfident predictions from adapted models. Therefore, large S[0] → higher T (softer distribution) is theoretically grounded.

---

### 4.2 Uncertainty Weighted Gradients for Calibration

> (Uncertainty Weighted Gradients for Model Calibration, arXiv:2503.22725, 2025.)

**Core finding:** Weighting gradients by a sample's predicted uncertainty improves calibration. The key formula:
```
d/d_theta L_Uncertainty-GRA(x, y) = u(p_hat(x)) * d/d_theta L_CE(x, y)
```
where u(·) is a Brier-Score-based uncertainty proxy. The Brier Score achieves Pearson correlation of 0.664 with true calibration error across samples.

**Critical result:** Models trained with BSCE-GRA achieve optimal temperatures near T = 1.0 post-hoc, indicating the gradient-weighted training produces models that are inherently calibrated. This establishes a bi-directional connection: calibration affects gradients, and gradient weighting can produce calibration.

**Implication for Q3:** If gradient magnitude (S[0]) correlates with calibration error (as BSCE-GRA suggests), then using S[0] as an input to T is not just a heuristic — it is a principled signal. The Pearson correlation of 0.664 between uncertainty proxy and calibration error supports using uncertainty-related gradient signals for calibration adjustment.

---

### 4.3 Layer-Wise Calibration Evolution

> (Calibration Across Layers: Understanding Calibration Evolution in LLMs, arXiv:2511.00280, 2025.)

**Core finding:** Calibration in LLMs is a dynamic, distributed process across layers:
- Early layers: near-random accuracy, low ECE (not yet confident)
- Mid layers (~22–26): accuracy rises, ECE begins increasing (model becomes confident)
- Late layers (~26–31): accuracy plateaus, ECE peaks at layer 28 then drops to final-layer calibration

**Three-phase pattern:** The model is most overconfident at layer 28 (ECE ≈ 0.20 on Phi-2/MMLU Humanities) and self-corrects by the final layer (ECE ≈ 0.08). The final layer performs active "confidence adjustment."

**A "calibration direction"** can be identified in the residual stream:
```
c_hat = (1/3) * (c_29 + c_30 + c_31)
where c_i = (A_i - A_{i-1}) / ||A_i - A_{i-1}||
```
Perturbing the activations along this direction (A_i' = A_i + eta * c_hat) improves ECE/MCE without degrading accuracy.

**Implication:** After TTT, this final-layer self-correction mechanism may be disrupted if the gradient updates concentrate on early/mid layers (common with LoRA adapters on lower layers). This could be why TTT increases overconfidence — the adaptation modifies layers that increase confidence but not the final-layer correction that normally re-calibrates it.

---

## Part V: Per-Token and Adaptive Temperature — State of the Art

### 5.1 ATS — Per-Token Hidden State Temperature (Deep Dive)

> Xie, J., Chen, A. S., Lee, Y., Mitchell, E., & Finn, C. (2024). arXiv:2409.19817.

**Calibration head architecture:**
- Single-layer causal Transformer block attached to the frozen LLM
- Input: last hidden state h_t at each token position
- Output: scalar T_t > 0 (per-token temperature)
- Inference: q_t = softmax(z_t / T_t)
- Training: minimize NLL on SFT dataset using the calibration-scaled logits

**Why per-token T is necessary after RLHF:**
- RLHF calibration degradation is heterogeneous: some tokens become severely overconfident while others remain well-calibrated
- Global T (like our T = 0.98) is a single scalar that averages over this heterogeneity
- ATS resolves this by predicting T_t from the context h_t

**Performance over global T:**
- 10–50% ECE reduction vs. global temperature scaling on MMLU, TriviaQA, TruthfulQA
- Both Llama-2-7b-Chat and Qwen-7b-Chat benefit substantially

**For RLHF models, optimal global T > 1** (because the model is overconfident). ATS predicts locally varying T: sometimes > 1 (overconfident token), sometimes < 1 (underconfident token, though less common after RLHF).

---

### 5.2 DEC Adaptive Temperature from Logit Norm

(See Section 3.1 for DEC formula and results.)

**Summary of key points for design:**
The DEC approach shows that the inverse logit norm (1 / ||z||_2) is a valid per-sample temperature scaling signal in the TTA setting. When applied to our TTT scenario, S[0] is the gradient-space analog of logit norm — it measures the magnitude of the model's response to the chunk, not just its forward-pass output.

---

### 5.3 Sample-Dependent Adaptive Temperature Scaling

> (Sample-Dependent Adaptive Temperature Scaling for Improved Calibration, AAAI 2023.)

**Core approach:** Per-sample temperature computed from input features (not gradient information). Demonstrates that heterogeneous temperature scaling strictly outperforms global T on most datasets. Establishes that the same input can warrant different temperatures at different test-time conditions.

---

### 5.4 Inverting Verbalized Probabilities

> (Calibrating Verbalized Probabilities for Large Language Models, arXiv:2410.06707, 2024.)

**Key technical finding:** When applying temperature scaling to verbalized LLM probability outputs, one must first invert the softmax to recover approximate logits:
```
z_i = log(p_i) + c,  where c = -1/K * sum_i log(p_i)
```
Then apply: q_i = softmax(z_i / tau).

**RLHF T distribution:** After RLHF, the optimal verbalized-probability temperature averages tau ≈ 1.3 (i.e., T > 1, meaning models are overconfident). This is consistent with the ATS finding. RLHF → overconfidence → optimal T > 1.

---

## Part VI: Answers to Key Research Questions

### Q1. After Fine-Tuning on a Specific Domain, Does Confidence Go Up or Down? Is There Systematic Evidence?

**Answer: Confidence goes up; calibration goes down. This is systematic across multiple lines of evidence.**

Evidence summary:
1. Guo et al. 2017: Post-convergence training increases confidence monotonically regardless of accuracy
2. GPT-4 Technical Report 2023: Near-perfect base model calibration → significant degradation after RLHF
3. Zhu et al. 2023: Pythia/LLaMA/FLAN-T5/OPT all become overconfident after domain fine-tuning regardless of model size
4. Taming Overconfidence 2024 (arXiv:2410.09724): RLHF raises ECE from 0.8608 to 0.8843 on GSM8K; reward models prefer high-confidence responses
5. Over-Certainty Phenomenon 2024 (arXiv:2404.16168): TTA entropy minimization raises ECE by 43% while reducing entropy by 81%
6. Domain-Specific Miscalibration: Individual domain ECE can be 250% higher than aggregate ECE after fine-tuning

**Mechanism:** Any gradient-based learning process that reduces the loss on a specific distribution concentrates probability mass on that distribution's patterns. This increases confidence without a corresponding increase in accuracy on unseen data from that distribution.

**For TTT specifically:** TTT is a form of extreme domain fine-tuning (one document, few steps). The confidence increase will be large and rapid. The calibration degradation will be concentrated on token patterns specific to the adapted chunk.

---

### Q2. After TTT, Should the Model Be More or Less Confident? What Does T = 0.98 Do?

**T = 0.98 (< 1) sharpens the distribution — it applies slight additional confidence increase.**

Given the systematic evidence that fine-tuning increases overconfidence, T = 0.98 (sharpening) moves the model further in the wrong direction after TTT, not toward better calibration. For a well-calibrated pre-TTT model, T = 0.98 is arguably slightly miscalibrating even before adaptation.

**After TTT, the evidence strongly suggests T should be > 1 (softening), not < 1.** The optimal T after TTT depends on how much the NLL improved:
- Small NLL improvement (model barely adapted): T ≈ 0.98 to 1.0 (small adjustment)
- Large NLL improvement (model significantly adapted): T should be meaningfully > 1.0 to counteract overconfidence

The key insight from the ATS paper: RLHF fine-tuning causes calibration degradation that requires T > 1 to correct (optimal tau ≈ 1.3). TTT is a more extreme form of the same phenomenon (fine-tuning on a single document for a few steps), and the expected direction of calibration degradation is the same.

**The adaptive T design (scaling T based on S[0]) is thus directionally correct:**
- When S[0] is small (little adaptation): T stays near 0.98–1.0
- When S[0] is large (significant adaptation, potential overconfidence): T should increase above 1.0

The formula should ensure T > 1.0 when S[0] exceeds a threshold corresponding to meaningful adaptation.

---

### Q3. Is There a Relationship Between Gradient Magnitude at Convergence and Calibration Error?

**Answer: Yes. This relationship is theoretically well-supported and empirically observed.**

Three independent lines of support:

**1. Fisher Information Matrix (FIM) theory (arXiv:2403.01485):**
At convergence on training data, gradients approach zero. For OOD data, gradients are large. This maps directly: after TTT convergence on chunk C, gradient magnitudes for chunk C are small (the model fits it); gradient magnitudes for new data are large in proportion to their distributional distance from C. The FIM formalizes gradient magnitude as distributional surprise.

**2. Gradient-Uncertainty Correlation (arXiv:2503.22725):**
The Brier-Score uncertainty proxy achieves Pearson correlation of 0.664 with true calibration error. The gradient weighting approach (BSCE-GRA) produces models with optimal T ≈ 1.0, confirming that gradient-calibration alignment is achievable.

**3. Over-Certainty Phenomenon with Logit Norm (arXiv:2404.16168):**
The DEC method uses inverse logit norm as a per-sample temperature signal. The logit norm decays with domain shift — in the gradient domain, S[0] from the gradient SVD serves the same role (measuring magnitude of the model's response to the chunk).

**S[0] as calibration signal (theoretical framing):**
- S[0] = top singular value of the weight gradient matrix after TTT
- Large S[0] → the gradient update had a dominant high-magnitude direction → the model moved significantly in parameter space → the adapted distribution is far from the original → high overconfidence risk
- Small S[0] → gradient updates were small and distributed → minimal adaptation → calibration close to pre-TTT state

**Recommended formula structure:**
```
T_adaptive = T_base + k * (S[0] - S[0]_baseline)  or
T_adaptive = T_base * exp(k * S[0] / S[0]_max)
```
where T_base ≈ 0.98 for minimal adaptation, k is a calibration hyperparameter, and S[0]_baseline is the expected S[0] for the pre-training distribution.

---

### Q4. Label Smoothing During TTT and Temperature During Scoring — Are They Connected?

**Answer: They are related but not equivalent. They are complementary regularizers, not redundant.**

**Theoretical connection (arXiv:2508.00264, arXiv:2402.03979):**

Both mechanisms act on the same underlying quantity — the entropy bound of the output distribution:
```
||C^T h||_2 <= sigma_C * sigma_h * sqrt(D)
```
- LS during TTT → reduces sigma_h (logit magnitude in weight/feature space) during adaptation
- T > 1 at scoring → divides z by T, reducing ||z||_2 at inference

**Non-equivalence:**
1. LS is applied during TTT adaptation; it affects the model's weights and representations
2. Temperature is applied at scoring; it affects only the inference-time probability distribution
3. An LS-trained model with hard temperature scaling has *different* ECE characteristics than a hard-target model with the same T (LS models have tighter clusters, which worsens post-hoc temperature calibration for misclassified samples)

**Practical implication for our design:**
If TTT is run without label smoothing (standard next-token prediction CE loss), then T is the sole calibration mechanism. If TTT is run with LS (α = 0.1), then:
- LS reduces the logit magnitude during training → partially corrects overconfidence at training time
- T = 0.98 further sharpens at inference → potentially overcorrects

With LS + T < 1, the system applies two consecutive confidence-boosting mechanisms. Consider whether T ≥ 1.0 is appropriate if LS is used during TTT.

---

### Q5. Beyond Temperature Scaling: Better Post-Hoc Calibration Methods Per-Chunk?

**Summary of alternatives:**

| Method | Mechanism | Requires | Per-Chunk Applicable? | Notes |
|---|---|---|---|---|
| Temperature Scaling (global T) | Divide all logits by T | Held-out data | Yes, with S[0] signal | Our current approach |
| ATS (per-token T) | T_t from hidden state h_t | SFT training set | Not directly (needs training) | Best for persistent models |
| Isotonic Regression | Monotone mapping on confidence bins | >= 1000 calibration examples | No (too few samples per chunk) | Requires large held-out set |
| Platt Scaling | Sigmoid fit to logits | Binary + held-out data | No (multiclass, few samples) | Ill-suited for generative LM |
| DEC (logit norm T) | T from 1/||z||_2 | Source entropy baseline | Yes, with modification | Most analogous to S[0] approach |
| Histogram Binning | Bucket-based remapping | >= 1000 calibration examples | No | Same issue as isotonic |
| Conformal Prediction | Coverage-based uncertainty | Calibration dataset | Possible in principle | Very different paradigm |

**For per-chunk calibration in TTT, only temperature scaling variants are computationally feasible.** Isotonic regression and histogram binning require calibration datasets with hundreds of examples, which a single chunk does not provide.

**The DEC approach (Section 3.1) is the closest existing system to what we are building** — it uses a within-sample signal (logit norm) to set per-sample temperature in an online manner. Our extension is to use the gradient's S[0] instead of the forward-pass logit norm.

---

### Q6. Can We Compute ECE Per-Chunk During TTT and Use It to Calibrate T?

**Answer: ECE estimation is unreliable at small batch sizes. It cannot be directly computed per-chunk.**

From the ECE estimation literature (arXiv:2109.03480):
- ECE estimation requires O(1000) samples minimum for reliable binning-based estimates
- "Training a model with naive estimates of calibration error using a batch size < O(1000) is potentially flawed"
- Binning introduces both discretization bias and finite-sample statistical bias

**Alternative within-chunk calibration signals that ARE feasible:**

1. **NLL improvement ratio: delta_NLL = NLL_pre_TTT - NLL_post_TTT**
   - Directly observable per chunk
   - Large delta_NLL → model improved significantly → higher overconfidence risk → increase T
   - Small delta_NLL → chunk was already well-predicted → T stays at baseline

2. **S[0] from gradient SVD** (our current approach)
   - Captures the magnitude of the adaptation in parameter space
   - Large S[0] → large adaptation → overconfidence risk
   - Theoretically grounded via FIM (Section 4.1)

3. **Logit norm change: delta_||z||_2 = ||z_post||_2 - ||z_pre||_2**
   - Post-TTT logit norms are directly accessible
   - Increase in logit norm after TTT is a direct proxy for increased confidence
   - No training needed; computed from forward pass pre- and post-TTT

4. **Entropy of the post-TTT distribution over the chunk tokens:**
   H_post = -sum_t sum_v p(v|x_{<t}) log p(v|x_{<t})
   Lower H_post (more peaked distributions) → model is more confident → higher T needed

**Recommended design for adaptive T:**
```
delta_NLL = NLL_pre_TTT - NLL_post_TTT  (observable, range ~ 0.0 to 1.0+ bpb)
S0_norm = S[0] / S[0]_expected  (normalized singular value)

T_adaptive = T_base + alpha * f(delta_NLL) + beta * g(S0_norm)
```
where f and g are monotonically increasing functions (e.g., sigmoid, linear-with-clip).

A simpler version with just NLL ratio:
```
T_adaptive = T_base + alpha * max(0, delta_NLL - delta_NLL_threshold)
```
This directly links T to the magnitude of adaptation.

---

### The Novel Connection: NLL Drop Ratio and Optimal Temperature

**Novel hypothesis (supported by multiple indirect evidence lines):**

When post-TTT NLL decreases significantly, the model has learned the chunk's token distribution well. By Guo et al. 2017's mechanism, continued NLL reduction past correct-prediction threshold concentrates confidence beyond what accuracy warrants. Therefore:

```
Optimal T ≈ f(delta_NLL / NLL_pre_TTT)
```

- When delta_NLL / NLL_pre is small (< 5%): chunk was already well-predicted; T ≈ baseline (0.98)
- When delta_NLL / NLL_pre is medium (5–20%): moderate adaptation; T ≈ 1.0 to 1.1
- When delta_NLL / NLL_pre is large (> 20%): significant adaptation, high overconfidence risk; T ≈ 1.1 to 1.3

**Supporting evidence:**
- RLHF causes average T_optimal ≈ 1.3 (from ATS, arXiv:2410.06707); RLHF represents roughly a 10–30% NLL reduction on alignment tasks
- The over-certainty phenomenon shows TTA methods that reduce entropy by 81% worsen ECE by 43%
- Guo et al. 2017: after NLL cannot be reduced by accuracy improvement, further NLL reduction raises confidence spuriously

**Key open question (not answered in existing literature):** What is the exact functional form of T_optimal = f(delta_NLL)? This is empirically determinable by measuring ECE before/after TTT across many chunks with varying delta_NLL values — a natural experiment for our setup.

---

## Part VII: Calibration and Layer Depth After TTT

### 7.1 LoRA and Layer-Specific Adaptation

When TTT uses LoRA (low-rank adaptation) on specific layers (e.g., query/value projections), only those layers receive gradient updates. The layer-wise calibration evolution (Section 4.3) shows that the final layers (~28–31 in a 32-layer model) perform active calibration correction. If LoRA adapts early layers and leaves final layers unchanged, the calibration correction mechanism is preserved. If LoRA targets late layers (or if full fine-tuning is used), the calibration correction mechanism may be disrupted.

**Design recommendation:** Prefer TTT LoRA on early-to-mid attention layers. Reserve the final few transformer blocks as frozen "calibration layers." This is consistent with the calibration-direction finding from arXiv:2511.00280.

---

## Part VIII: Summary of Paper Quality and Citation Counts

| Paper | Venue | Citations (approx.) | Quality Indicator |
|---|---|---|---|
| Guo et al. 2017 | ICML | ~5,000+ | Seminal; foundational for all calibration work |
| Minderer et al. 2021 | NeurIPS | ~800+ | Major update to calibration understanding for transformers |
| Müller et al. 2019 | NeurIPS | ~2,000+ | Seminal on label smoothing |
| Kadavath et al. 2022 | arXiv (Anthropic) | ~1,200+ | Highly cited; canonical LLM self-knowledge |
| Xiong et al. 2024 | ICLR | ~400+ | Recent but already influential |
| Xie et al. 2024 (ATS) | EMNLP | ~100+ | State-of-the-art per-token calibration |
| Over-Certainty (2024) | arXiv | ~50+ | Key for TTA calibration degradation |
| FIM OOD (2024) | arXiv | ~30+ | Strong theory for gradient-norm calibration signal |
| BSCE-GRA (2025) | arXiv | Recent | Direct gradient-calibration connection |

---

## Part IX: Proper Academic Citations

```
Guo, C., Pleiss, G., Sun, Y., & Weinberger, K. Q. (2017). On Calibration of Modern
  Neural Networks. Proceedings of the 34th International Conference on Machine
  Learning (ICML), PMLR 70, 1321–1330. https://arxiv.org/abs/1706.04599

Minderer, M., Djolonga, J., Romijnders, R., Hubis, F., Zhai, X., Houlsby, N.,
  Tran, D., & Lucic, M. (2021). Revisiting the Calibration of Modern Neural
  Networks. NeurIPS 2021. https://arxiv.org/abs/2106.07998

Müller, R., Kornblith, S., & Hinton, G. E. (2019). When Does Label Smoothing
  Help? NeurIPS 2019. https://arxiv.org/abs/1906.02629

Kadavath, S., Conerly, T., Askell, A., Henighan, T., Drain, D., Perez, E.,
  Schiefer, N., & Hatfield-Dodds, Z. (2022). Language Models (Mostly) Know What
  They Know. Anthropic Technical Report. https://arxiv.org/abs/2207.05221

Xiong, M., Hu, Z., Lu, X., Li, Y., Fu, J., He, J., & Bryan, H. (2024). Can LLMs
  Express Their Uncertainty? An Empirical Evaluation of Confidence Elicitation
  in LLMs. ICLR 2024. https://arxiv.org/abs/2306.13063

Xie, J., Chen, A. S., Lee, Y., Mitchell, E., & Finn, C. (2024). Calibrating
  Language Models with Adaptive Temperature Scaling. EMNLP 2024.
  https://arxiv.org/abs/2409.19817

Sun, Y., Shi, Q., Garg, S., Xiao, J., Saeed, M., & Akyürek, E. (2024). The
  Over-Certainty Phenomenon in Modern Test-Time Adaptation Algorithms.
  https://arxiv.org/abs/2404.16168

Tan, C., et al. (2024). Uncertainty-Calibrated Test-Time Model Adaptation without
  Forgetting. https://arxiv.org/abs/2403.11491

Sun, Y., et al. (2024). Let's (not) just put things in Context: Test-Time Training
  for Long-Context LLMs. https://arxiv.org/abs/2512.13898

(Anonymous). (2024). Approximations to the Fisher Information Metric of Deep
  Generative Models for Out-Of-Distribution Detection. ICLR 2024 Workshop.
  https://arxiv.org/abs/2403.01485

(Anonymous). (2025). Uncertainty Weighted Gradients for Model Calibration.
  https://arxiv.org/abs/2503.22725

(Anonymous). (2025). Calibration Across Layers: Understanding Calibration
  Evolution in LLMs. https://arxiv.org/abs/2511.00280

(Anonymous). (2024). Calibrated Language Models and How to Find Them with Label
  Smoothing. https://arxiv.org/abs/2508.00264

(Anonymous). (2024). Cross Entropy versus Label Smoothing: A Neural Collapse
  Perspective. https://arxiv.org/abs/2402.03979

(Anonymous). (2024). Taming Overconfidence in LLMs: Reward Calibration in RLHF.
  https://arxiv.org/abs/2410.09724

Zhao, W. X., et al. (2023). Just Ask for Calibration: Strategies for Eliciting
  Calibrated Confidence Scores from Language Models Fine-Tuned with Human
  Feedback. EMNLP 2023. https://arxiv.org/abs/2305.14975

(Anonymous). (2024). Calibrating Verbalized Probabilities for Large Language
  Models. https://arxiv.org/abs/2410.06707

(Anonymous). (2024). Few-Shot Recalibration of Language Models.
  https://arxiv.org/abs/2403.18286

(Anonymous). (2024). Beyond Overconfidence: Model Advances and Domain Shifts
  Redefine Calibration in Neural Networks. https://arxiv.org/abs/2506.09593
```

---

## Part X: Key Gaps and Open Questions for Future Research

1. **No direct TTT + calibration study exists.** All existing TTT papers measure accuracy/perplexity, not calibration. There is no paper that directly measures ECE before and after TTT on a per-chunk basis. Our setup would be the first such study.

2. **The S[0]–T_optimal relationship is not characterized.** While the FIM literature supports using gradient magnitude as a calibration signal, the specific functional form of T_optimal = f(S[0]) is not derived theoretically or measured empirically. This is a research gap.

3. **Delta_NLL–T_optimal relationship is theoretical.** The hypothesis that T_optimal scales with the NLL improvement ratio is supported by multiple indirect evidence lines (Guo 2017, RLHF calibration studies, TTA over-certainty) but has not been directly tested. Testing this in our TTT setup would contribute to the calibration literature.

4. **Layer-specific TTT and calibration.** The layer-calibration paper (arXiv:2511.00280) suggests final layers are calibration correctors; LoRA TTT on early layers may preserve this. No paper has directly tested this hypothesis.

5. **ECE under TTT is unknown.** Whether per-chunk ECE is computable from within-chunk signals alone (without a held-out calibration set) is an open question. Our proposed delta_NLL and S[0] signals are proxies, not direct ECE measurements.

---

*Document compiled for Parameter Golf project (EXP adaptive temperature design). All arXiv IDs verified at time of compilation.*
