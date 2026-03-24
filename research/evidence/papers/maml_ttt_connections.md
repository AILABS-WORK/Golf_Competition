# MAML / Meta-Learning — Test-Time Training (TTT) Connections
## Deep Research Synthesis for Parameter Golf

**Date:** 2026-03-24
**Scope:** Formal connection between MAML-family algorithms and per-document LoRA TTT;
actionable modifications to our TTT inner loop.

---

## Search Summary

**Databases searched:** arXiv (via web search), PubMed/PMC, ACL Anthology, NeurIPS/ICML/ICLR
proceedings (OpenReview), ACM DL, MIT Press TACL, Amazon Science, Semantic Scholar,
SSRN.

**Queries issued (representative):** MAML inner-loop TTT; implicit meta-learning pretraining
transformer; gradient-aligned initialization fast adaptation; LoRA gradient approximation
convergence; Reptile first-order convergence theory; TTT nearest-neighbors LLMs; in-place
TTT ICLR 2026; LoRA-TTT VLMs; test-time meta-adaptation self-synthesis; TNT chunkwise
training; end-to-end TTT long context.

**Papers reviewed (primary sources confirmed):** ~30 papers directly examined; ~20
additional corroborating results reviewed.

---

## Section 1 — MAML's Inner Loop and Its Structural Identity with Our TTT

### The Canonical MAML Algorithm

Finn et al. (2017), arXiv:1703.03400, introduced MAML as a bilevel optimisation:

- **Outer loop:** Updates the global parameter initialisation theta such that *k* gradient
  steps on any sampled task leads to low test loss on that task.
- **Inner loop:** For each task tau_i, run `theta_i' = theta - alpha * grad_theta L_tau_i(theta)`
  for *k* steps.

The meta-objective is `min_theta sum_i L_tau_i(theta_i')`. MAML therefore optimises
explicitly for *good initialisations for fast adaptation*.

### Structural Identity with Our Per-Document LoRA TTT

Our system does the following at inference time for each document:

1. Reset LoRA adapters to zero (or a soft-reset value).
2. Run *n* gradient steps on the document's tokens with next-token-prediction loss.
3. Forward-pass the test prefix through the adapted model.

This is, by construction, identical to MAML's **inner loop** run on a single "task" (the
document). The key difference is that our **outer loop** is standard language model
pretraining, NOT a MAML meta-objective. Standard pretraining does not explicitly optimise
for "the initialisation that adapts fastest in few steps." MAML pretraining does.

**Consequence:** If our model were meta-trained with MAML (or a first-order approximant),
the base initialisation theta would be explicitly chosen to minimise loss *after* k gradient
steps. We would expect faster convergence per step and better final quality from the same
number of inner-loop steps.

---

## Section 2 — Does Standard LM Pretraining Induce Implicit MAML Properties?

### The ICL-as-Gradient-Descent Literature (Seminal)

Two landmark 2022-2023 papers establish that Transformer pretraining on next-token
prediction induces implicit meta-learning properties that closely resemble MAML's inner
loop:

**Von Oswald et al. (2023).** "Transformers Learn In-Context by Gradient Descent."
*ICML 2023*, Proceedings of the 40th ICML.
- Constructs exact weight matrices for a linear self-attention layer such that one forward
  pass implements one step of gradient descent on a regression loss.
- Shows empirically that trained self-attention-only transformers on regression tasks become
  *mesa-optimisers*: they implement gradient descent in their forward pass.
- **Key implication for us:** Even without explicit MAML training, a pretrained LM's
  attention layers are already performing implicit gradient-like updates as they process
  context. TTT makes this implicit process explicit by updating weights.

**Dai et al. (2023).** "Why Can GPT Learn In-Context? Language Models Secretly Perform
Gradient Descent as Meta-Optimizers." *ACL Findings 2023*, arXiv:2212.10559.
- Shows that Transformer attention has a *dual form* of gradient descent. GPT computes
  "meta-gradients" from demonstration examples and applies them.
- Frames ICL as: pretraining = MAML outer loop; ICL inference = MAML inner loop.
- **Key implication for us:** LM pretraining naturally produces a parameter point theta that
  is adapted to being adapted. The inner loop of our TTT is walking the same gradient
  landscape that the model's implicit in-context adaptation already navigates.

**Akyurek et al. (2022).** "What Learning Algorithm is In-Context Learning? Investigations
with Linear Models." *ICLR 2023*, arXiv:2211.15661.
- Proves by construction that transformers can implement gradient descent and ridge
  regression for linear models.
- Shows algorithmic phase transitions: 1-layer transformers approximate one step of GD;
  deeper transformers match OLS.

### Quantitative Evidence: Does MAML Pretraining Help Over Standard Pretraining?

**Bhatt et al. (2025).** "Exploring the Efficacy of Meta-Learning: Unveiling Superior Data
Diversity Utilization of MAML Over Pre-training." arXiv:2501.08506.
- On 12 visual datasets and 5 model configurations, HO-MAML (5-10 inner steps) shows
  significantly stronger positive correlation between dataset diversity and performance
  relative to pretrain+finetune baselines (R^2 = 0.15–0.42 vs. weaker for PT).
- Suggests MAML has an *advantage specifically when data is diverse*, which is precisely
  the setting of language modelling across heterogeneous documents.

**Hou et al. (2022).** "Is Pre-training Truly Better Than Meta-Learning?" arXiv:2306.13841.
- Controlled comparison (same architecture, same optimiser, trained to convergence).
- Key finding: when dataset diversity is **low**, pretraining wins; when diversity is
  **high**, meta-learning is competitive or better. Effect size is near zero overall.
- **For us:** document-level language modelling is high diversity. Meta-learning is likely
  to help.

**Rupe et al. (2024/2025).** "Learning Dynamics of Meta-Learning in Small Model Pretraining."
arXiv:2508.02189.
- Integrates FO-MAML episodes with standard next-token prediction pretraining in LLaMA-
  style decoder models.
- Reaches the same training loss **1.6x sooner** (measured in compute, not wall-clock).
- Phase-like dynamics: representations first diversify, then compress — aligns with
  improved episodic accuracy.
- **Key actionable finding:** hybrid pretraining (standard NTP + periodic MAML episodes)
  is feasible at small model scale and produces faster adapters.

---

## Section 3 — iMAML and Online/Efficient Meta-Learning During Inference

### iMAML: Implicit MAML (NeurIPS 2019)

**Rajeswaran, Finn et al. (2019).** "Meta-Learning with Implicit Gradients."
*NeurIPS 2019*, arXiv:1909.04630.
- iMAML avoids differentiating through the inner loop unrolling entirely. Instead, it
  solves for the implicit gradient using the implicit function theorem:
  `grad_theta L_meta = (I - alpha * H_inner)^{-1} * grad_theta' L_test`.
- Memory is **independent of the number of inner-loop steps** (requires only
  Hessian-vector products via conjugate gradient).
- Achieves better results than MAML while using comparable resources; MAML requires either
  more outer steps or expensive long backprop chains.
- **For our TTT:** if we ever wanted to meta-train our model to produce a better
  initialisation for TTT (outer loop), iMAML is the right tool. It scales to many inner
  steps (20-50) without memory blowup.

### MAML Applied to LLMs for In-Context Learning (2024)

**Khanna et al. (2024).** "MAML-en-LLM: Model Agnostic Meta-Training of LLMs for Improved
In-Context Learning." *KDD 2024*, arXiv:2405.11446.
- Integrates first-order MAML into LLaMA-style training. Inner loop: adapts model to
  diverse tasks. Outer loop: MAML meta-gradient update.
- +2% average gain on unseen domains; +4% on adaptation performance vs. standard fine-tuning.
- MAML's inductive bias may **favour optimisation over regularisation**, accelerating
  convergence but occasionally degrading out-of-task fluency at large scales.
- **Key finding:** explicit MAML training does improve fast adaptation, quantified and
  reproducible.

### Test-Time Meta-Adaptation with Self-Synthesis (MASS, 2026)

**Anonymous (2026).** "Test-Time Meta-Adaptation with Self-Synthesis." arXiv:2603.03524.
- Most direct overlap with our system in the literature.
- MASS is a meta-learning framework where LLMs: (a) generate problem-specific synthetic
  training examples; (b) adapt on those examples via inner-loop gradient descent; (c) an
  outer loop (meta-trained offline) learns data-attribution signals to weight examples.
- The bilevel optimisation: inner loop adapts on self-generated data; outer loop rewards
  post-update task performance via scalable meta-gradients.
- Tested on mathematical reasoning; learns "per-instance curricula" that yield
  data-efficient TTT.
- **Actionable idea for us:** instead of adapting on the raw document chunks, we could
  meta-train a curriculum/weighting signal that selects which tokens or chunks to
  prioritise in the inner loop.

---

## Section 4 — TTT Papers That Explicitly Use or Analyse Meta-Learning Structure

### End-to-End TTT (TTT-E2E, 2025)

**Tandon, Dalal et al. (2025).** "End-to-End Test-Time Training for Long Context."
arXiv:2512.23675. Submitted December 2025.
- Frames long-context LM as continual learning. Uses Transformer + sliding-window
  attention; model updates weights via NTP on the given context.
- **Explicitly improves initialisation for TTT via meta-learning at training time.**
  The outer loop meta-trains the model so the initialisation is maximally suited to fast
  in-context gradient updates. This is a direct MAML outer loop applied to TTT.
- For 3B models trained on 164B tokens, scales with context length identically to full
  attention, while Mamba 2 and Gated DeltaNet do not.
- Inference latency: **2.7x faster than full attention at 128K context** due to constant
  per-step cost.
- This paper is the most direct empirical confirmation that **explicitly meta-training
  the TTT initialisation improves TTT performance**. The structure is: pretraining = MAML
  outer loop; TTT at inference = MAML inner loop.

### TTT on Nearest Neighbours for LLMs (ICLR 2024)

**Hardt and Sun (2024).** "Test-Time Training on Nearest Neighbors for Large Language
Models." *ICLR 2024*, arXiv:2305.18466.
- At each test step, retrieves 20 nearest neighbours from the Pile using a distributed
  dense index; fine-tunes on their text using the standard NTP objective.
- As few as 20 neighbours, each for **1 gradient step**, drastically improves perplexity
  across 20+ Pile domains.
- Narrows the gap between GPT-2 and GPT-Neo (10x larger model) significantly.
- This is a **memory-augmented TTT** approach: instead of only adapting on the current
  document, the model is shown semantically related documents. This is analogous to
  prototype networks using the retrieved documents as task exemplars.
- **Actionable idea for us:** chunk-level nearest-neighbour retrieval before the inner
  loop — use the embedding of the current chunk to retrieve similar past chunks or held-out
  training segments, then include them in the TTT batch.

### Learning to (Learn at Test Time): TTT Layers (Sun et al., 2024)

**Sun et al. (2024).** "Learning to (Learn at Test Time): RNNs with Expressive Hidden
States." *ICML 2025*, arXiv:2407.04620.
- Makes the hidden state itself a machine learning model (TTT-Linear: linear model;
  TTT-MLP: two-layer MLP). The update rule is one step of self-supervised learning.
- Outer loop (pretraining) trains all parameters including the meta-learning rule embedded
  in the model. Inner loop (test time) updates only the hidden-state model via GD.
- TTT-Linear and TTT-MLP keep reducing perplexity as context grows beyond 16K; Mamba
  cannot.
- This is the most principled architectural instantiation of the MAML-TTT connection.
  The entire architecture is trained end-to-end to optimise the *fast-weight update rule*.

### TTT Provably Improves Transformers as In-Context Learners (2025)

**Gozeten et al. (2025).** "Test-Time Training Provably Improves Transformers as In-Context
Learners." *ICML 2025*, arXiv:2503.11842.
- Provides theoretical characterisation of linear transformers under single-step gradient
  TTT:
  (a) alignment between pretraining distribution and target task governs TTT effectiveness;
  (b) TTT alleviates distribution shift;
  (c) TTT **significantly reduces sample complexity** of ICL (3–5x fewer samples needed
      for same classification performance on TabPFN).
- **Key theoretical result for us:** TTT's benefit is maximised when the pretraining
  distribution and the test document share structure. For language modelling, this means
  TTT is most effective on documents similar to training data. For out-of-distribution
  documents, the alignment is weaker and the inner loop may need more steps.

### TTT Enhances ICL of Nonlinear Functions (2025)

**Kuwataka et al. (2025).** "Test Time Training Enhances In-Context Learning of Nonlinear
Functions." *ICLR 2025 (OpenReview)*, arXiv:2509.25741.
- Single-layer transformers with gradient-based TTT adapt to both the feature vector and
  the link function (nonlinear task structure) across tasks.
- ICL alone cannot adapt to link-function shifts. TTT can.
- Convergence rate bound: prediction error driven to noise level as context length and
  network width grow.
- **For us:** this provides convergence guarantees even in the nonlinear regime, validating
  that our per-document LoRA adaptation is theoretically sound for heterogeneous documents
  with differing distributional structures.

### In-Place TTT (ICLR 2026)

**Anonymous (2026).** "In-Place Test-Time Training." *ICLR 2026 (conference paper)*,
OpenReview id: dTWfCLSoyl.
- Treats the final projection matrix of MLP blocks as "fast weights" — a drop-in
  enhancement for any LLM without full retraining.
- Replaces generic reconstruction objective with a Next-Token-Prediction aligned objective.
- Enables 4B-parameter model to achieve superior performance on contexts up to 128K.
- **Directly relevant:** their NTP-aligned TTT objective is closely related to our
  inner-loop loss. They confirm that alignment between the TTT objective and the NTP
  pretraining objective is crucial — a generic reconstruction loss is suboptimal.

### TTT for Long-Context LLMs (2512.13898)

**Lou et al. (2025).** "Let's (not) just put things in Context: Test-Time Training for
Long-Context LLMs." arXiv:2512.13898.
- Query-only TTT variant: adapts only on the query tokens, not the full context.
- Shows that selective adaptation (query-focused) can be more efficient than full-document
  adaptation.

---

## Section 5 — Meta-Learning the Initialisation: RELI Connection

### Gradient-Aligned LoRA Initialisation (LoRA-GA, NeurIPS 2024)

**Wang et al. (2024).** "LoRA-GA: Low-Rank Adaptation with Gradient Approximation."
*NeurIPS 2024*, arXiv:2407.05000.
- Initialises LoRA adapter matrices from the **eigenvectors of the gradient matrix** (via
  SVD on the full gradient at step 0), rather than random initialisation.
- This ensures that the low-rank subspace at initialisation *aligns with the principal
  gradient directions* — the same principle as RELI.
- Results: **2–4x faster convergence** vs. vanilla LoRA; +5.69% on GLUE with T5-Base;
  +11.52% on GSM8K with LLaMA-2 7B.
- **Direct theoretical support for RELI:** LoRA-GA is a published, peer-reviewed
  instantiation of gradient-principal-direction initialisation for low-rank adaptation,
  with large empirical gains at NeurIPS.

### Theoretical Analysis: Why Gradient-Aligned Init is Optimal

**Xu et al. (2025).** "Understanding the Learning Dynamics of LoRA: A Gradient Flow
Perspective." *AISTATS 2025*, arXiv:2503.06982.
- Theoretically analyses LoRA under gradient flow for matrix factorisation.
- **Key theorem:** smaller initialisation scale + alignment with singular spaces of the
  target update matrix leads to lower final error.
- The *misalignment between the pre-trained model's singular spaces and the target matrix*
  is the dominant source of LoRA's convergence penalty vs. full fine-tuning.
- When LoRA matrices are initialised in the gradient's principal subspace, this
  misalignment is minimised.

**Mu and Klabjan (2024).** "On the Convergence Rate of LoRA Gradient Descent."
arXiv:2512.18248.
- Proves that vanilla LoRA converges at O(1/log T) to a stationary point due to the
  recursive interaction between adapter norms and gradients.
- If adapter norms are **bounded** (e.g. by cosine schedule, weight decay, or explicit
  norm clipping), the classical O(1/T) rate is recovered.
- **For our TTT inner loop:** bounding the norm of LoRA adapters during the k-step TTT
  is theoretically motivated to avoid the O(1/log T) trap. This would be a one-line
  implementation (add `max_norm` to the optimizer or add `weight_decay` on adapters).

### MAML Initialisation as Task-Manifold Interpolation

**Nichol and Schulman (2018).** "On First-Order Meta-Learning Algorithms (Reptile)."
arXiv:1803.02999.
- Both MAML and Reptile contain the same leading-order terms in their gradient:
  (1) minimises expected loss (joint training signal);
  (2) **maximises within-task generalisation** by maximising the inner product between
      gradients on different minibatches from the same task.
- The second term is the key MAML insight: the initialisation is pushed to a point where
  successive gradient steps on the same task point in compatible directions.
- **For our TTT:** this means a MAML-trained initialisation would exhibit *gradient
  alignment across chunks of the same document*. Our k inner-loop steps would accumulate
  consistently rather than oscillating. This is the mechanistic reason MAML init helps TTT.

---

## Section 6 — Reptile vs. MAML for TTT: When Is First-Order Sufficient?

### First-Order Approximation Theory

**Finn, Abbeel, Levine (2017) + Nichol, Schulman (2018)** together establish:
- FOMAML ignores second-order terms by setting them to zero. This is a linear approximation
  of the meta-gradient.
- The approximation is accurate when the loss landscape has **low local curvature**
  (near-zero Hessian eigenvalues at the initialisation point).
- Reptile is mathematically similar to FOMAML and has identical leading-order terms.
- Empirically: Reptile and MAML achieve similar performance on Omniglot and Mini-ImageNet.
- Reptile converges **faster** (lower-variance gradient updates) and uses less memory.

**Zhou et al. (2020, cited in convergence analysis):**
- "Multi-step MAML may not improve significantly from increasing inner-loop steps" —
  consistent with our finding that diminishing returns apply to TTT inner steps.

**Key practical conclusion for our TTT:**
- Our gradient-based LoRA inner loop is equivalent to Reptile/FOMAML in structure.
- Full MAML (second-order) would require differentiating through our inner loop steps,
  which is expensive and likely unnecessary given the near-linear loss landscape of
  pretrained LMs at inference time.
- First-order is theoretically justified and empirically sufficient for our use case.

### When Does Second-Order Help?

- Second-order MAML adds value when the task loss landscape has **high curvature**, i.e.
  when the Hessian is large at the initialisation. This occurs in:
  (a) shallow networks on non-convex tasks;
  (b) tasks requiring rapid representation change (not just weight scaling);
  (c) regression on nonlinear targets with short contexts.
- For LM fine-tuning with LoRA (low-rank, near-pretrained-loss landscape), curvature is
  low. First-order is sufficient.

---

## Section 7 — Prototype Networks and Memory-Based TTT

### ProtoNets as Memory for TTT

Prototype networks (Snell et al., 2017) learn a per-class embedding by averaging support
set representations. For TTT, this translates to: maintain a running average of chunk
embeddings as a "document prototype," and use the distance from this prototype to gate
or weight the TTT gradient.

**Titans (Google, 2025, arXiv:2501.00663):**
- Introduces a neural long-term memory module that learns to *memorize historical context
  at test time*. Combines recurrent memory with attention.
- The memory is updated via gradient descent on past tokens — structurally a TTT inner
  loop on the memory weights.
- Handles sequences beyond 2M tokens with constant memory footprint.
- **Connection to prototype TTT:** Titans' memory module is a continuous, differentiable
  analogue of a prototype — it aggregates document information into a compressed
  representation that is updated online.

**TNT: Improving Chunkwise Training for Test-Time Memorization (2025), arXiv:2511.07343:**
- Introduces a **periodic state reset** (resets local memory to a shared learnable initial
  state at each segment boundary). The learnable initial state is essentially a meta-
  learned prototype for the "expected document state."
- This periodic reset is what enables context parallelism — directly analogous to our
  soft-reset approach.
- **Key finding:** the learnable initial state (shared across all documents) is a
  trainable prototype that captures average document start-of-chunk statistics.

**LoRA-TTT for Vision-Language Models (2025), arXiv:2502.02069:**
- Applies LoRA exclusively to image encoder at test time with a reconstruction loss.
- LoRA initialisation using image-text pairs (warm-start from relevant examples) improves
  performance by >1% on fine-grained benchmarks.
- Confirms that **warm-start initialisation from related examples** (the nearest-neighbour
  spirit) combines with LoRA-TTT effectively.

**Test-Time Training Done Right / LaCT (2025), arXiv:2505.23884:**
- Proposes Large Chunk TTT (LaCT): uses chunk sizes from 2K to 1M tokens.
- Very large chunks improve GPU utilisation (from <5% to near-full) and enable scaling
  the fast-weight state to 40% of model parameters.
- **Actionable for us:** our current chunk size (512 tokens with 32 overlap from the
  literature) may be too small for efficient GPU usage. LaCT suggests 2K+ chunks.

---

## Section 8 — SSRN and Economics/Finance Literature

The SSRN search returned no methodologically novel TTT papers from finance or social
science. The relevant SSRN papers are:
- Survey papers on LLM post-training and RLHF (not TTT-specific).
- "Chronologically Consistent Large Language Models" (He et al., 2025, SSRN:5159615) —
  temporal adaptation of LLMs, but via fine-tuning, not TTT.
- "Contrastive Domain Adaptation with Test-Time Training for News Detection"
  (Gu et al., 2024, SSRN:5006088) — applies TTT to OOD detection in NLP; no MAML
  connection.

**SSRN verdict:** No novel methodology relevant to MAML-TTT was found in the finance or
social-science literature. The relevant theory is entirely in the CS/ML venue.

---

## Section 9 — Contradictions and Open Debates

### 1. Does Explicit MAML Training Beat Implicit Pretraining for TTT?

The ICL-as-GD literature (von Oswald, Dai) argues pretraining already induces MAML-like
properties. MAML-en-LLM and TTT-E2E argue explicit meta-training improves things further.
These are not truly contradictory — pretraining gives an approximate MAML initialisation;
explicit meta-training refines it. The question is cost vs. benefit.

### 2. Pretraining vs. Meta-Learning (Hou et al. 2023)

The effect size is near zero when dataset diversity is averaged. This means that for
**homogeneous** document sets (low diversity), pretraining is better; for **diverse**
corpora (e.g. The Pile), meta-learning wins. Language model TTT over diverse documents
should favour meta-trained initialisations.

### 3. Number of Inner Steps

Zhou et al.'s analysis suggests diminishing returns after several inner steps; Gozeten et al.'s
theoretical bounds show error decreases with more context (more tokens = more steps).
These are reconciled by the distinction between "inner loop steps given fixed context" vs.
"context length growth."

### 4. Objective Function Alignment

In-Place TTT (ICLR 2026) and Dai et al. both emphasise that the TTT objective must align
with the pretraining objective (NTP) to work well. Papers using reconstruction/denoising
objectives get smaller gains. This is an important design constraint.

---

## Section 10 — Actionable Modifications to Our TTT Inner Loop

The following are ranked by estimated impact and implementation effort:

### Priority 1 (High Impact, Low Effort)

**A. Bound LoRA Adapter Norms During TTT**
- Theoretical basis: Mu & Klabjan (2024) proves bounding adapter norms recovers O(1/T)
  convergence vs. O(1/log T) unbounded.
- Implementation: add `weight_decay=1e-4` to the LoRA-specific optimizer group, OR
  clip adapter norms via `clip_grad_norm_` with a small max_norm (e.g. 0.1–0.5) after
  each TTT gradient step.
- Expected gain: faster convergence per step; less degradation on short documents.

**B. Verify TTT Objective Alignment with NTP**
- Basis: In-Place TTT (ICLR 2026) and Dai et al. (2023) both show that using a
  reconstruction/denoising loss for TTT underperforms the NTP objective.
- If we are already using NTP loss for TTT, confirm no auxiliary objective is weakening it.
- If we are using any masking or denoising variant, switch to clean next-token prediction.

### Priority 2 (High Impact, Medium Effort)

**C. Gradient-Aligned LoRA Initialisation (RELI-style)**
- Basis: LoRA-GA (NeurIPS 2024) shows 2–4x faster convergence from SVD-based gradient
  initialisation vs. random. Xu et al. (2025) provides the theoretical foundation.
- Implementation: before the TTT inner loop, compute one forward+backward pass on the
  first 128–256 tokens of the document. SVD the gradient of the first LoRA-affected
  weight matrix. Initialise LoRA-A from the top-k right singular vectors; LoRA-B from
  the top-k left singular vectors (scaled to ensure B@A is near zero initially).
- This is essentially RELI, now with peer-reviewed theoretical and empirical support.
- Estimated implementation: 15–30 lines of code; one extra backward pass per document.

**D. Increase Chunk Size to 2K+ Tokens**
- Basis: LaCT (arXiv:2505.23884) shows that <512-token chunks give <5% GPU utilisation
  for the TTT inner loop. Chunks of 2K–8K improve hardware utilisation by orders of
  magnitude and enable larger state capacity.
- For our per-document setup (documents ~2K–8K tokens), this may mean doing fewer but
  larger gradient steps rather than many small steps.
- Implementation: adjust `chunk_size` parameter; combine with gradient accumulation if
  memory-constrained on RTX 5080 (16GB).

### Priority 3 (Medium Impact, Medium Effort)

**E. Soft-Reset as Reptile Outer Loop**
- Basis: Our current soft-reset (partial decay of LoRA weights between documents) is
  structurally identical to one step of Reptile on the document distribution.
- The Reptile theory (Nichol & Schulman, 1803.02999) says this implicitly maximises
  within-task gradient alignment, pushing the initialisation toward a point that adapts
  consistently across chunks of similar documents.
- Actionable change: tune the soft-reset decay coefficient (lambda) explicitly as the
  Reptile step-size: `theta_reset = (1 - lambda) * theta_0 + lambda * theta_adapted`.
  Sweep lambda in {0.05, 0.1, 0.2, 0.5}. Reptile theory predicts an optimal lambda that
  trades off task-specific retention vs. mean task performance.

**F. Sample Efficient Token Selection for TTT Gradient Steps**
- Basis: TLM (Test-Time Learning for LLMs, arXiv:2505.20633) shows that high-perplexity
  tokens carry more gradient signal. Prioritising them for inner-loop updates reduces
  gradient noise.
- Implementation: after a no-grad forward pass, sort tokens by perplexity. Take the
  top-P% (e.g. top 25%) for the actual gradient step. This reduces computation and
  focuses adaptation signal.
- This is compatible with gradient-aligned LoRA initialisation (Priority 2C).

### Priority 4 (Medium Impact, Higher Effort)

**G. Chunk-Level Nearest-Neighbour Retrieval for TTT Batch Augmentation**
- Basis: Hardt & Sun (2024) show that 20 nearest neighbours + 1 gradient step per
  neighbour drastically reduces perplexity. Bridges small GPT-2 to 10x larger GPT-Neo.
- Implementation: embed each document chunk with a frozen sentence encoder (or the model's
  own last-layer mean pooling). Retrieve top-K similar chunks from a pre-built index of
  training data. Include retrieved chunks in the TTT minibatch.
- Caveat: requires an inference-time vector index. At small scale, can use flat FAISS.
  At document level, a per-domain sub-index is sufficient.

**H. MAML-Aware Pretraining (Periodic MAML Episodes)**
- Basis: Rupe et al. (2025, arXiv:2508.02189) reaches same loss 1.6x sooner via hybrid
  pretraining that alternates NTP and FO-MAML episodes.
- Implementation: add a MAML episode every N pretraining steps. Each episode: sample a
  batch of documents, split into support and query halves, run 1 inner step on support,
  compute query loss, backprop through the inner step (FOMAML: stop gradient through
  the inner gradient).
- Cost: ~2x per MAML step (one extra forward+backward). At 10% frequency (1 in 10
  steps), overall training cost increases ~10%.
- This would be our most impactful long-run change but requires training from scratch.

---

## Section 11 — Research Gaps Identified

1. **No paper explicitly studies MAML-pretrained LoRA-TTT for language modelling.** The
   closest is TTT-E2E (2512.23675) which meta-trains a TTT-layer model, not a LoRA-TTT
   model. A direct ablation of standard pretraining vs. MAML pretraining for LoRA-TTT
   would be novel.

2. **No theoretical analysis of the optimal soft-reset decay rate for Reptile-equivalent
   TTT.** The Reptile paper gives step-size intuitions but not a closed-form optimum for
   the language modelling setting.

3. **LoRA-GA has not been applied to TTT specifically.** All published LoRA-GA results are
   on standard fine-tuning tasks. Applying it to the per-document TTT inner loop is an
   open experiment.

4. **Combining nearest-neighbour retrieval with gradient-aligned initialisation is
   unexplored.** Hardt & Sun use flat initialisation; gradient-aligned init is not in
   their setup.

---

## Proper Citations

1. Finn, C., Abbeel, P., and Levine, S. (2017). "Model-Agnostic Meta-Learning for Fast
   Adaptation of Deep Networks." *ICML 2017*. arXiv:1703.03400.

2. Nichol, A. and Schulman, J. (2018). "On First-Order Meta-Learning Algorithms."
   arXiv:1803.02999.

3. Rajeswaran, A., Finn, C., Kakade, S., and Levine, S. (2019). "Meta-Learning with
   Implicit Gradients." *NeurIPS 2019*. arXiv:1909.04630.

4. Akyurek, E., Schuurmans, D., Andreas, J., Ma, T., and Zhou, D. (2022). "What Learning
   Algorithm is In-Context Learning? Investigations with Linear Models." *ICLR 2023*.
   arXiv:2211.15661.

5. Von Oswald, J. et al. (2023). "Transformers Learn In-Context by Gradient Descent."
   *ICML 2023*. Proceedings of the 40th ICML.

6. Dai, D. et al. (2023). "Why Can GPT Learn In-Context? Language Models Secretly Perform
   Gradient Descent as Meta-Optimizers." *ACL Findings 2023*. arXiv:2212.10559.

7. Hou, Z., Salazar, J., and Polovets, G. (2022). "Meta-Learning the Difference: Preparing
   Large Language Models for Efficient Adaptation." *TACL 2022*.
   DOI:10.1162/tacl_a_00517.

8. Hou, Z. et al. (2023). "Is Pre-training Truly Better Than Meta-Learning?"
   arXiv:2306.13841.

9. Sun, Y. et al. (2024). "Learning to (Learn at Test Time): RNNs with Expressive Hidden
   States." *ICML 2025*. arXiv:2407.04620.

10. Hardt, M. and Sun, Y. (2024). "Test-Time Training on Nearest Neighbors for Large
    Language Models." *ICLR 2024*. arXiv:2305.18466.

11. Wang, S. et al. (2024). "LoRA-GA: Low-Rank Adaptation with Gradient Approximation."
    *NeurIPS 2024*. arXiv:2407.05000.

12. Khanna, A. et al. (2024). "MAML-en-LLM: Model Agnostic Meta-Training of LLMs for
    Improved In-Context Learning." *KDD 2024*. arXiv:2405.11446.

13. Gozeten, H.A. et al. (2025). "Test-Time Training Provably Improves Transformers as
    In-Context Learners." *ICML 2025*. arXiv:2503.11842.

14. Tandon, A., Dalal, K. et al. (2025). "End-to-End Test-Time Training for Long Context."
    arXiv:2512.23675.

15. Xu, M. et al. (2025). "Understanding the Learning Dynamics of LoRA: A Gradient Flow
    Perspective on Low-Rank Adaptation in Matrix Factorization." *AISTATS 2025*.
    arXiv:2503.06982.

16. Mu, S. and Klabjan, D. (2024). "On the Convergence Rate of LoRA Gradient Descent."
    arXiv:2512.18248.

17. Anonymous (2026). "In-Place Test-Time Training." *ICLR 2026 (conference paper)*.
    OpenReview:dTWfCLSoyl.

18. Anonymous (2026). "Test-Time Meta-Adaptation with Self-Synthesis (MASS)."
    arXiv:2603.03524.

19. Bhatt, M. et al. (2025). "Exploring the Efficacy of Meta-Learning: Unveiling Superior
    Data Diversity Utilization of MAML Over Pre-training." arXiv:2501.08506.

20. Rupe, E. et al. (2025). "Learning Dynamics of Meta-Learning in Small Model Pretraining."
    arXiv:2508.02189.

21. Li, Z. et al. (2025). "TNT: Improving Chunkwise Training for Test-Time Memorization."
    arXiv:2511.07343.

22. Zhang, T. et al. (2025). "Test-Time Training Done Right (LaCT)." arXiv:2505.23884.

23. Kuwataka, K. et al. (2025). "Test Time Training Enhances In-Context Learning of
    Nonlinear Functions." *ICLR 2025*. arXiv:2509.25741.

24. Behrouz, A. et al. (2025). "Titans: Learning to Memorize at Test Time."
    arXiv:2501.00663.

25. Kojima, Y. et al. (2025). "LoRA-TTT: Low-Rank Test-Time Training for Vision-Language
    Models." arXiv:2502.02069.

26. Lou, Y. et al. (2025). "Let's (not) just put things in Context: Test-Time Training for
    Long-Context LLMs." arXiv:2512.13898.

---

## Further Reading Recommendations

1. **TTT-E2E full paper** (arXiv:2512.23675) — most important: the meta-training recipe
   for TTT initialisation.
2. **LoRA-GA full paper** (arXiv:2407.05000) — gradient-aligned init implementation
   details; source code on GitHub at Outsider565/LoRA-GA.
3. **MASS full paper** (arXiv:2603.03524) — the bilevel TTT meta-adaptation system that
   is most analogous to a production version of our system.
4. **iMAML** (arXiv:1909.04630) — if we ever want to meta-train our initialisation,
   iMAML is the right tool (memory-efficient outer loop).
5. **LaCT** (arXiv:2505.23884 + GitHub: a1600012888/LaCT) — chunk size engineering for
   GPU-efficient TTT.
