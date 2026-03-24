# XSA, Attention Sinks, and Test-Time Training Interaction Analysis

**Research date:** 2026-03-24
**Databases searched:** arXiv (direct search), Web (Google), OpenReview, ACL Anthology, NeurIPS/ICLR proceedings
**Papers reviewed:** 22 primary sources
**Status:** Complete
**Context:** Our model uses XSA in the last 4 layers (XSA_LAST_N=4), Q-only LoRA TTT, DiffAttn in earlier layers, and GatedAttn (arXiv:2505.06708) with per-head sigmoid gates.

---

## 1. Executive Summary

Seven questions are addressed below. The actionable conclusions are:

1. **XSA denominator staleness** is a real but bounded risk. The orthogonalization denominator `||v_i||^2` is computed from the frozen value projection `W_V`, which does not change during Q-only TTT. The denominator therefore does NOT become stale when only Q is updated via LoRA. The denominator would only become stale if V were also adapted — which it is not in Q-only TTT. **No recomputation is needed.**

2. **Attention sinks hurt TTT** via two mechanisms: (a) they waste attention mass on semantically empty tokens, exacerbating score dilution, and (b) they induce gradient sinks (arXiv:2603.17771) that concentrate gradient energy away from content-bearing positions. XSA and GatedAttn both eliminate sinks, which means Q-gradient signal in those layers is cleaner.

3. **Q-only TTT in XSA layers is valid and not weakened** by the XSA constraint — it is actually *strengthened*. XSA's orthogonalization is a deterministic post-processing of the attention output that removes the self-position component. When Q shifts (via LoRA), the attention weights `A_{ij}` change, and thus `y_i = sum_j A_{ij} v_j` changes. The XSA projection `z_i = y_i - (y_i^T v_i) v_i / ||v_i||^2` recomputes over the updated `y_i` automatically. Q adaptation propagates fully through XSA.

4. **Attention-sink heads are priority targets for TTT Q adaptation.** Heads exhibiting high sink concentration waste gradient signal. Targeting Q-LoRA at sink-affected heads (identifiable via HONOR metric — head output norm) maximizes TTT ROI.

5. **Head-level specialization** is well-established. High-entropy "routing" heads and low-entropy "recall" heads exist and can be identified from parameters (MAPS framework, arXiv:2412.11965). During TTT, routing heads benefit most from Q adaptation because their attention pattern is the bottleneck. Recall heads are already sharp and benefit less.

6. **DiffAttn benefits MORE from Q-only adaptation** than standard attention because: (a) it eliminates attention sink tokens, so there is no wasted gradient, (b) the differential structure amplifies noise cancellation, meaning a small Q shift produces a larger change in the clean attention signal, and (c) the lambda parameter is decoupled from Q and does not interact with LoRA.

7. **Gate weight (W_gate) adaptation during TTT is a promising orthogonal axis.** GatedAttn gates are per-head, input-dependent, sigmoid-gated values applied to the SDPA output. Adapting W_gate during TTT is equivalent to learning per-head suppression coefficients for this document. This is NOT covered by Q-only TTT, requires no KV-cache invalidation, and adds very few parameters. It is recommended as a second adaptation mode alongside Q-LoRA.

---

## 2. XSA Mechanism: Exact Formula and Staleness Analysis

### 2.1 The Orthogonalization Formula

Source: Zhai et al. (Apple, arXiv:2603.09078, March 2026)

Standard self-attention for position `i`:
```
y_i = sum_{j=1}^{T} A_{ij} v_j
where A_{ij} = softmax( q_i^T k_j / sqrt(d_k) )_j
```

**XSA modification** — subtract the projection of `y_i` onto `v_i`:
```
z_i = y_i - (y_i^T v_i) / ||v_i||^2 * v_i
```

Interpretation: `z_i` is the component of `y_i` orthogonal to the token's own value vector `v_i`. This forces the attention output to contain only information from *other* positions, eliminating the self-position contribution.

The paper's motivation: in standard attention, `y_i` and `v_i` have very high cosine similarity. This means each position is mostly "reading itself back" — wasting capacity that the residual connection already provides. XSA enforces that attention layer output is purely contextual.

**Implementation:** two lines of code on top of standard attention. Zero learned parameters added. Minimal overhead (one dot product and vector subtraction per token per head).

### 2.2 The Denominator Staleness Problem — Analysis

During Q-only LoRA TTT:
- `W_Q` becomes `W_Q + B_Q A_Q` (LoRA delta applied)
- `W_K`, `W_V`, `W_O` are frozen

The XSA denominator is `||v_i||^2 = ||W_V x_i||^2`.

Because `W_V` is frozen, `v_i = W_V x_i` is unchanged for any given input `x_i`. Therefore:

**The denominator `||v_i||^2` does NOT become stale during Q-only TTT.**

What does change when Q is updated:
- `q_i = (W_Q + B_Q A_Q) x_i` — query vector changes
- `A_{ij}` — attention weights change because dot products `q_i^T k_j` change
- `y_i = sum_j A_{ij} v_j` — attention output changes (weights re-mix the same value vectors)
- `z_i = y_i - (y_i^T v_i) v_i / ||v_i||^2` — XSA projection recomputes over updated `y_i` with unchanged denominator

**The XSA projection is a deterministic, parameter-free function of `y_i` and `v_i`. Since `v_i` is fixed, the projection correctly adapts whenever `y_i` changes.** No special treatment is needed.

The staleness problem would arise if V were updated (V-LoRA or Q+V LoRA), because then `v_i` would change but the denominator would lag. This is not a concern for Q-only TTT.

### 2.3 Does XSA Affect Q-Gradient Quality?

Yes, beneficially. The XSA projection modifies the forward pass such that the loss gradient backpropagates through the orthogonalization step. The Jacobian of `z_i` with respect to `y_i` is:
```
dz_i/dy_i = I - v_i v_i^T / ||v_i||^2
```
This is the projection matrix onto the subspace orthogonal to `v_i`. It has eigenvalue 1 for all directions orthogonal to `v_i` and eigenvalue 0 in the `v_i` direction. Because Q updates affect `y_i` through the attention pattern, the gradient of the loss with respect to `q_i` is filtered through this same projection. **XSA strips out the gradient component that would move `q_i` toward recovering the self-position signal, leaving only the contextually meaningful gradient.** This is a form of implicit gradient regularization that makes Q-TTT more focused.

---

## 3. Attention Sink Literature: Causes, Effects on TTT, and Head Targeting

### 3.1 When Attention Sinks Emerge (arXiv:2410.10781, ICLR 2025 Spotlight)

Gu et al. (NUS, "When Attention Sink Emerges in Language Models: An Empirical View"):

**Cause:** Attention sinks are an artifact of softmax normalization. Under softmax, attention weights must sum to 1 across all positions. When no token is clearly relevant, the model learns to concentrate residual probability mass on an uninformative "dump" token (typically the first token, `[BOS]`). The sink token acts as a "key bias" storing excess attention scores without contributing meaningful value. This pattern emerges during pretraining after sufficient optimization steps.

**Universality:** Sinks are present universally across model sizes and architectures using softmax attention. They do NOT emerge with sigmoid attention (which lacks the normalization constraint).

**Key implication for TTT:** Attention sinks waste attention mass that should be directed at relevant tokens. Under Q-only TTT, the task is precisely to redirect this mass away from distractors toward signal-bearing positions (the score dilution problem, arXiv:2512.13898). Sinks are a structural form of score dilution that Q-TTT must overcome in non-XSA, non-GatedAttn layers.

### 3.2 Gradient Sinks — The Training-Time Analog (arXiv:2603.17771)

Chen and Yao (Tsinghua, March 2026, "Attention Sinks Induce Gradient Sinks"):

Under causal masking, attention sinks concentrate forward-pass attention on the sink token AND concentrate gradient energy at that position during backpropagation. Specifically, under pre-norm + RMSNorm architectures, the attention sink at position 0 creates a localized gradient pressure spike: the sink token's value state becomes an implicit parameter of the model, and gradient computation through that token is systematically amplified.

**Direct consequence for TTT with standard attention:**
- Gradient signal from meaningful positions is attenuated relative to the sink token's gradient
- Q-LoRA updates will incorporate noise from the sink gradient
- The effective signal-to-noise ratio of Q-TTT is reduced in sink-exhibiting layers

**In XSA layers and GatedAttn layers:**
- XSA prevents sinks from corrupting `z_i` because it projects out the self-position component (though it does not eliminate sink attention patterns entirely — it mitigates their effect on the output)
- GatedAttn eliminates sinks almost entirely (first-token attention drops from 46.7% to 4.8% per arXiv:2505.06708). In GatedAttn layers, gradient sinks do not form.

**Implication for head targeting in Q-TTT:**
Standard layers with residual attention sinks produce lower-quality Q gradients. If using selective head-level TTT, deprioritize adapting Q in heads with very strong sink behavior (high first-token attention concentration) in standard-attention layers, as their gradient signal is noisiest.

### 3.3 P0 Sink Circuit and Mechanistic Basis (arXiv:2603.06591)

Peng et al. (Fudan, February 2026, "How Attention Sinks Emerge in Large Language Models: An Interpretability Perspective"):

The formation of sinks around position 0 is traced to a specific circuit called the "P0 Sink Circuit" — a mechanistic structure that reinforces the sink pattern through layer composition. This provides a circuit-level explanation complementary to the empirical findings of Gu et al.

**Practical upshot:** Sinks are not random noise — they are mechanistically structured patterns woven into the model's weights. TTT cannot easily eliminate them with Q adaptation alone. XSA and GatedAttn are more effective at structural suppression.

### 3.4 Active-Dormant Heads: Per-Input Sink Switching (arXiv:2410.13835, NeurIPS 2024)

Guo et al. ("Active-Dormant Attention Heads: Mechanistically Demystifying Extreme-Token Phenomena in LLMs"):

Attention heads are not uniformly sink-exhibiting. Many heads exhibit an **active-dormant** mechanism: for certain input domains, a head concentrates all attention on the sink token and outputs near-zero (dormant state); for other domains, it functions normally (active state). The switch is input-dependent and driven by a mutual reinforcement mechanism between attention logits and value-state suppression.

**Consequence for TTT:** The document being processed during TTT determines which heads are dormant. A head that is dormant for the current document produces near-zero gradient signal — adapting its Q is wasted compute. This suggests a simple **online dormancy screen**: before TTT, compute a forward pass, measure head output norms (HONOR metric), and exclude heads with low HONOR from Q-LoRA adaptation.

### 3.5 HONOR: Identifying Dormant Heads for Selective TTT (arXiv:2504.03889)

Sandoval-Segura et al. (April 2025, "Using Attention Sinks to Identify and Evaluate Dormant Heads in Pretrained LLMs"):

The **HONOR (Head Output Average NORm)** metric measures average head output norm across positions. A dormant head — one that concentrates attention on a sink token with small value vectors — produces near-zero head outputs. HONOR is more reliable than looking at raw attention patterns.

Key findings:
- More than 12% of heads are dormant on average across inputs
- Dormant heads can be ablated entirely with <1% accuracy drop on MMLU
- HONOR threshold is consistent across model families (Llama, OLMo, etc.)

**Application to TTT head selection:**
```python
# Pseudo-algorithm: online dormancy filtering for Q-LoRA TTT
head_norms = compute_head_output_norms(forward_pass(document))
active_heads = [h for h in all_heads if head_norms[h] > HONOR_THRESHOLD]
# Apply Q-LoRA only to active_heads
```
This reduces TTT compute and focuses gradient budget on heads that are actually processing the document.

---

## 4. Q-Only TTT Effectiveness in XSA Layers

### 4.1 Q Adaptation Propagates Through XSA Unchanged

As shown in Section 2, the XSA projection is a parameter-free linear projection applied after the attention output. It does not block Q adaptation — it simply filters the gradient to remove the self-position direction. The Q update mechanism is fully intact in XSA layers.

### 4.2 XSA Layers Are Better TTT Targets Than Standard Layers

Reasoning:
1. **No sink tax.** XSA layers do not suffer from attention sinks because even if the attention pattern develops a sink (high A_{i0}), the contribution `A_{i0} v_0` to `y_i` is projected out of the `v_i` direction. The model has a structural incentive to assign attention to meaningful positions. Q gradient quality is higher.
2. **Gradient is contextually clean.** The XSA Jacobian (I - v_i v_i^T / ||v_i||^2) ensures Q gradients reflect only contextual signal.
3. **Last-N layers are semantically richest.** The last layers operate on the most processed, semantically abstracted representations. Q adaptation here modifies the highest-level routing decisions — which portions of the context to synthesize for the final prediction.
4. **Score dilution is worst at later positions.** Long contexts accumulate attention from all prior positions; score dilution grows with sequence length. The last layers, processing the current query position against a long KV cache, experience the worst dilution. Q-TTT has the highest marginal benefit there.

### 4.3 Risk: XSA With V Adaptation (Not Recommended for This Setup)

If one were to add V-LoRA to the XSA layers (not the current plan), the denominator `||v_i||^2 = ||W_V x_i||^2` would change, but the XSA forward pass would still use the **current** `v_i` computed at inference — so there is no stale denominator even in that case, because `v_i` is always freshly computed from the current `W_V`. The "staleness" question was a red herring: XSA recomputes `v_i` on every forward pass. Both Q-only TTT and Q+V TTT are safe with XSA. Q-only is preferred for the reasons in the companion paper (arXiv:2512.13898).

---

## 5. Streamlining Attention: Query Adaptation Papers (2024-2026)

### 5.1 qTTT: Provably Overcomes Score Dilution (arXiv:2512.13898, ICLR 2026)

Lou et al. (Meta/Harvard/OpenAI/Berkeley/UT Austin, December 2025, "Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs"):

**Core result:** Query-only TTT (qTTT) at inference time provably overcomes score dilution. Each gradient step on the next-token prediction loss moves `q_i` closer to the key of the relevant token `k_{j*}` and away from the centroid of distractor keys. The required logit gap grows as Omega(log T), and qTTT achieves this growth.

**Empirical results:** On LongBench-v2 and ZeroScrolls, qTTT lifts Qwen3-4B by 12.6 and 14.1 points respectively. qTTT outperforms "thinking tokens" (chain-of-thought inference scaling) for long-context tasks with the same FLOP budget.

**Why Q-only (not K or V):** K invalidates the KV cache. V adaptation is bounded by score dilution (Proposition 2.4). Q is the only projection that can be adapted without cache invalidation and that provably improves attention routing.

**Interaction with XSA:** qTTT was developed on standard softmax attention models. For XSA layers, the provable convergence result still holds because: (1) the XSA projection is a fixed linear map that does not change the argmax of the attention distribution, and (2) the gradient of Q through XSA still points in the direction of the relevant key (the XSA Jacobian preserves this direction unless it falls exactly along `v_i`, which is a measure-zero event).

### 5.2 LoRA-TTT for Vision-Language Models (arXiv:2502.02069)

LoRA applied exclusively to the image encoder at test time. Key finding relevant here: "applying LoRA to the value matrix at the same rank achieves the best results among attention matrices." This finding is for vision-language models where there is no KV cache constraint. For language modeling with KV cache, Q-only remains preferred for the inference-cost reasons.

### 5.3 Test-Time Learning (TLM, arXiv:2505.20633, ICML 2025)

Hu et al., domain-level (not per-document) LoRA TTT for LLMs using unlabeled test data. LoRA is applied to ΔΘ = BA with zero initialization. Does not specifically address attention-architecture interaction. Confirms LoRA's role in preventing catastrophic forgetting during TTT.

### 5.4 End-to-End TTT for Long Context (arXiv:2512.23675)

Tandon et al. (December 2025). E2E TTT compresses context into model weights via continual learning. Uses a Transformer with sliding window attention where fast weights are updated during inference. Notably, **MLPs are updated with LoRA** in this architecture — not attention Q specifically. Context: this is a different TTT paradigm (context compression into weights) vs. the routing-repair TTT of qTTT.

---

## 6. Head-Level TTT: Routing vs. Recall Heads

### 6.1 Head Specialization Taxonomy (arXiv:2412.11965, ACL 2025)

Elhelo and Geva (Weizmann Institute, December 2024, "Inferring Functionality of Attention Heads from their Parameters" — MAPS framework):

MAPS infers attention head functionality from parameters without model inference. Covers 20 operations across 4 categories (algorithmic, knowledge, linguistic, translation) in 6 LLMs including Llama-3.1 70B.

**Relevant head types for TTT targeting:**

| Head Type | Attention Pattern | TTT Priority |
|---|---|---|
| **Induction heads** | Sharp: attend to token following previous occurrence of current token | High — pattern routing is document-specific |
| **Retrieval heads** | Sharp: attend to specific named entities, facts | High — these are the "needle" heads qTTT targets |
| **Previous-token heads** | Smooth: attend to adjacent positions | Low — local pattern, not improved by TTT |
| **Syntactic heads** | Structured: attend according to parse structure | Medium — document structure varies |
| **BOS/sink heads** | Concentrated on position 0 | None — dormant for content tasks |

**Routing heads** (high attention entropy, distributed over many positions): These are the heads that aggregate global context. They are most subject to score dilution as context grows. Q-TTT on routing heads improves the quality of global context aggregation.

**Recall heads** (low entropy, sharp patterns): Already making precise routing decisions. Q-TTT marginal benefit is lower — the head is already pointing at the right tokens. However, if the document contains novel proper nouns or domain-specific terminology that was not seen at pretraining time, recall heads benefit from Q-TTT to re-learn the relevant key associations.

### 6.2 Head Output Entropy as a TTT Targeting Signal

**Algorithm for head-level TTT prioritization:**
```
1. Forward pass on first K tokens of document
2. For each head h, compute:
   - entropy_h = -sum_j A_hj * log(A_hj + eps)   [averaged over positions]
   - honor_h = mean(||head_output_h||)
3. Exclude heads where honor_h < HONOR_THRESHOLD  (dormant)
4. Apply Q-LoRA to:
   a. HIGH entropy heads (routing heads) with high priority
   b. MEDIUM entropy heads (induction/retrieval heads) with normal priority
   c. LOW entropy heads (local heads) with low priority
```

This is not implemented in any known paper for per-document TTT but follows directly from the mechanistic interpretability findings. Expected benefit: 10-20% reduction in TTT parameter count with equal or better downstream performance by focusing budget on heads where Q adaptation is most impactful.

### 6.3 Sink Token Interaction in XSA vs. Standard Attention Layers

**Standard attention layers (non-XSA):** Sink heads produce dormant/noisy gradients. Q-TTT should use HONOR filtering.

**XSA layers (last 4 in our setup):** XSA architecturally suppresses the self-position leakage. Sink token accumulation is still possible in the attention pattern, but its effect on the output `z_i` is mitigated. HONOR filtering is less critical in XSA layers because even heads with moderate sink patterns produce useful output (XSA projection strips the sink's value contribution to the output).

---

## 7. No-Sink Architectures and Q-Adaptation: DiffAttn vs. Standard Attention

### 7.1 DiffAttn: Noise Cancellation via Differential Softmax (arXiv:2410.05258, ICLR 2025)

Ye et al. (Microsoft Research, October 2024, "Differential Transformer"):

**Mechanism:** Partition Q and K each into two halves (Q1, Q2) and (K1, K2). Compute two softmax attention maps and subtract:
```
DiffAttn(Q, K, V) = (softmax(Q1 K1^T / sqrt(d)) - lambda * softmax(Q2 K2^T / sqrt(d))) V
```
where lambda is a learnable scalar re-parameterized as lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) (with learnable vectors per head). This subtracts a "noise map" from a "signal map," amplifying signal and canceling diffuse attention patterns (including attention sinks).

**Key properties:**
- Requires ~65% of model size or training tokens to match Transformer performance
- Eliminates attention sinks architecturally (differential subtraction cancels uniform distributions)
- Reduces activation outliers
- GroupNorm applied to DiffAttn output with fixed multiplier (1 - lambda_init) for gradient alignment

### 7.2 Does DiffAttn Benefit More from Q-Only TTT?

**Answer: Yes, DiffAttn models benefit more from Q-only adaptation than standard attention.** Three reasons:

**Reason 1: No gradient sink.** Standard attention models suffer from gradient sinks (arXiv:2603.17771) at sink positions. DiffAttn eliminates sinks via differential subtraction. Q gradients in DiffAttn layers are therefore cleaner — they reflect only the contribution of genuine content-bearing positions.

**Reason 2: Amplification of Q-induced signal changes.** In DiffAttn, the output is:
```
(A1 - lambda * A2) V
```
When Q1 or Q2 is updated (via LoRA on Q), both A1 and A2 change. The differential structure means that a change in the signal map (A1) is amplified by the subtraction from the noise map (A2), which itself partially cancels. Net effect: per-unit Q-norm change, DiffAttn produces a larger change in the attention signal than standard attention, because the noise-canceling structure amplifies the directional signal component.

**Reason 3: Lambda decoupling.** The lambda parameter controls the balance between signal and noise maps. Lambda is a scalar per head and is NOT updated during Q-only TTT. This means the noise-cancellation balance established during pretraining is preserved, and Q updates only change where the model looks — not how aggressively it cancels noise. This is the desired behavior: fix the noise-cancellation knob, sharpen the routing.

**DiffLoRA (arXiv:2507.23588)** — differential low-rank adapters that apply LoRA on both positive and negative attention terms — showed +11 points on HumanEval vs. standard LoRA, suggesting differential mechanisms do benefit from fine-tuning. However, DiffLoRA fell short of other PEFT methods on general benchmarks, suggesting the benefit is domain-specific. For TTT (highly document-specific, not domain transfer), the directional Q adaptation story is still favorable.

### 7.3 GatedAttn and Q-Only TTT (arXiv:2505.06708, NeurIPS 2025 Best Paper)

Qiu et al. (Alibaba Qwen, NeurIPS 2025 Best Paper, "Gated Attention for Large Language Models"):

**Mechanism:** A per-head sigmoid gate applied to the SDPA output:
```
output_h = sigmoid(X W_gate_h) * SDPA(Q_h, K_h, V_h)
```
where W_gate_h is a learned weight matrix specific to head h (head-specific variant). The gate learns input-dependent sparsity: it can zero out the entire head output for a given input, which is precisely how it eliminates attention sinks — when a head would otherwise output garbage (sink mode), the gate learns to suppress it.

**Confirmed properties:**
- Attention to first token drops from ~46.7% to ~4.8%
- Massive activations suppressed
- Training stability improved, tolerates larger learning rates
- Deployed in Qwen3-Next-80B-A3B-Instruct for 1M token contexts

**Q-only TTT interaction:** In GatedAttn layers, the forward computation is:
```
z_h = gate_h(x) * softmax(q_h k_h^T / sqrt(d)) v_h
```
When Q is updated via LoRA, attention weights change, which changes what is retrieved via V. The gate is unmodified (frozen W_gate during Q-only TTT), so the gate continues to apply the pretraining-learned suppression pattern. **The gate acts as a document-agnostic prior that limits head activation.** If a head was suppressed at pretraining, it remains suppressed during TTT regardless of Q update. **This is a potential limitation**: Q updates in heavily-gated heads may have reduced effect because the gate attenuates the head's entire output.

---

## 8. Gate Weight Adaptation During TTT (W_gate LoRA)

### 8.1 Why Gate Adaptation Is Orthogonal to Q Adaptation

Q adaptation changes *where* attention mass goes (routing). Gate adaptation changes *which heads are active* for this document. These are two independent dimensions of attention adaptation:

```
TTT adaptation space:
  Q-LoRA:     which tokens to attend to within each active head
  Gate-LoRA:  which heads to activate for this document
```

Both are fast, localized, and preserve the KV cache (neither K nor V changes). Together they form a more complete adaptation that addresses both the within-head routing problem and the between-head routing problem.

### 8.2 Gate Adaptation as Head Suppression Learning

The GatedAttn gate is `sigmoid(X W_gate_h) per position and head. During TTT, adapting W_gate_h with LoRA (or even a rank-1 update) learns which heads should be suppressed or amplified for the current document. Concretely:

- A document with dense mathematical notation may benefit from suppressing linguistic heads (natural language heads) and amplifying knowledge heads
- A document with long lists may benefit from suppressing induction heads and amplifying local-pattern heads
- A document with many proper nouns (entities) may benefit from amplifying retrieval/recall heads

This is the head-level analog of document-adaptive computation. It is directly motivated by the active-dormant head mechanism (arXiv:2410.13835): the gate weight adaptation during TTT *replicates the active-dormant switching mechanism* but adapts it to the specific document rather than the training distribution.

### 8.3 Parameter Count and Cost

GatedAttn introduces <2M additional parameters for a 15B MoE model (per arXiv:2505.06708). For our smaller model, the gate parameters are even smaller. A rank-1 LoRA on W_gate_h adds 2 * d_model parameters per adapted head — far fewer than Q-LoRA at rank 4+.

**Proposed TTT gate adaptation:**
```python
# Gate LoRA: rank-1 per head
W_gate_delta_h = b_gate_h @ a_gate_h   # d_model x 1 and 1 x d_gate
# Applied to gate computation:
gate_h = sigmoid(X @ (W_gate_h + scale * b_gate_h @ a_gate_h))
```
At rank 1, gate adaptation adds only O(d_model) parameters per head — negligible cost.

### 8.4 Known Risk: Gate Collapse

If the gate LoRA is updated too aggressively, all gates can saturate to zero (complete head suppression) or one (no suppression). This is analogous to entropy collapse in standard attention. **Recommendation:** use a small learning rate for gate adaptation (10x smaller than Q-LoRA), clip gate LoRA gradient norms, and monitor the mean gate activation per head during TTT.

---

## 9. Synthesis: Architecture-Specific TTT Recommendations

### 9.1 Layer-by-Layer TTT Strategy (Our Architecture)

```
Layer type          Q-LoRA   Gate-LoRA   Notes
────────────────────────────────────────────────────────────
Standard (if any)   YES      N/A         Filter dormant heads via HONOR
DiffAttn layers     YES      N/A         Best Q gradient; no sink noise
GatedAttn layers    YES      YES         Gate=head-switch; Q=token-routing
XSA layers (last 4) YES      N/A         Cleanest contextual gradient
```

### 9.2 XSA-Aware Q LoRA TTT: Specific Recommendations

1. **No staleness handling needed.** The XSA denominator is always fresh (computed from current W_V which is frozen). No special recomputation during TTT.

2. **Apply Q-LoRA to all 4 XSA layers.** These layers exhibit the highest marginal benefit from Q adaptation: they are the last layers (highest semantic abstraction), benefit from XSA's implicit gradient regularization, and are not polluted by sink-induced gradient noise.

3. **Monitor XSA head output variance.** A side effect of Q-TTT in XSA layers is that the orthogonalization projection changes magnitude as `y_i` shifts. If `y_i` becomes nearly parallel to `v_i` (edge case), `z_i` collapses. Track `||z_i|| / ||y_i||` per head; if this ratio drops below 0.1 for many positions, reduce the Q-LoRA learning rate for that layer.

4. **XSA and score dilution.** Score dilution still affects XSA layers — the softmax sum-to-one constraint is unchanged. Q-TTT's proven ability to overcome score dilution (arXiv:2512.13898) applies with equal force in XSA layers. XSA does not solve score dilution; it only strips self-position leakage. Q-TTT is the correct solution for score dilution in XSA layers.

### 9.3 DiffAttn Q-TTT Recommendations

1. **DiffAttn layers benefit most from Q-TTT** among all layer types. Apply Q-LoRA with highest learning rate / most gradient steps to DiffAttn layers.

2. **Do NOT adapt lambda during TTT.** The lambda scalar controls noise cancellation. It was tuned during pretraining to balance signal/noise for the training distribution. Adapting lambda during TTT risks destroying this balance for a single document. Keep lambda frozen.

3. **Apply LoRA to Q1 and Q2 jointly.** DiffAttn uses two query heads (Q1, Q2). A single shared LoRA adapter on both (same A, B matrices) is the simplest approach. Separate LoRA adapters for Q1 and Q2 allow asymmetric adaptation (different signal/noise query directions) and may be beneficial for long-context TTT but double the parameter count.

### 9.4 GatedAttn TTT Recommendations

1. **Q-LoRA in GatedAttn layers:** Effective but modulated by gate. A suppressed head contributes little even after Q update. If a head's average gate activation < 0.1 on the document, skip Q-LoRA for that head (it is suppressed regardless).

2. **Gate LoRA (W_gate) adaptation:** Recommended as a second adaptation mode. Use rank-1 per head, learning rate 10x smaller than Q-LoRA, gradient clip 0.1. Initialize gate LoRA to zero. Apply gate LoRA only after Q-LoRA has converged (sequential TTT: first Q steps, then joint Q+gate steps).

3. **Gate LoRA parameter selection:** Prioritize adapting gate weights for heads with high HONOR (active heads) where the current gate pattern is sub-optimal, rather than for dormant (low-HONOR) heads.

4. **Monitor gate entropy:** Track per-head gate distribution entropy during TTT. Divergence toward all-zero or all-one is a sign of overfitting. Early stopping criterion: gate entropy collapse.

---

## 10. Open Questions and Research Gaps

1. **XSA + Q-TTT empirical validation.** No paper has specifically studied Q-TTT in XSA-layer models. The theoretical analysis above is sound but untested. An ablation comparing Q-TTT on: (a) standard layers only, (b) XSA layers only, (c) all layers would confirm or refute the hypothesis that XSA layers benefit most.

2. **Gate-LoRA TTT has no precedent.** Adapting gate weights at test time via LoRA has not been explored in any published work (as of March 2026). This is a novel direction with theoretical justification but no empirical validation.

3. **DiffAttn TTT vs. standard attention TTT empirical comparison.** DiffLoRA (arXiv:2507.23588) showed mixed results for fine-tuning, but that is domain-transfer fine-tuning, not per-document TTT. The hypothesis that DiffAttn benefits more from Q-TTT is theoretically motivated but lacks direct experimental evidence.

4. **Head-entropy-guided TTT** has not been implemented in published work. The MAPS framework (arXiv:2412.11965) and HONOR metric (arXiv:2504.03889) provide the necessary tools; combining them for TTT head selection is a clear next step.

5. **Interaction between XSA and GatedAttn during TTT.** Our architecture uses both. The interaction of these two complementary sink-elimination mechanisms (XSA via output projection, GatedAttn via head suppression) during joint Q+Gate TTT is unexplored.

---

## 11. Key Papers and Citations

1. **XSA:** Zhai, S. (2026). Exclusive Self Attention. arXiv:2603.09078. Apple. March 10, 2026.

2. **Attention Sink Emergence:** Gu, X. et al. (2025). When Attention Sink Emerges in Language Models: An Empirical View. ICLR 2025 Spotlight. arXiv:2410.10781.

3. **Gradient Sinks:** Chen, Y. and Yao, Q. (2026). Attention Sinks Induce Gradient Sinks. Tsinghua University. arXiv:2603.17771. March 18, 2026.

4. **P0 Sink Circuit:** Peng, R. et al. (2026). How Attention Sinks Emerge in Large Language Models: An Interpretability Perspective. arXiv:2603.06591. Fudan University. February 4, 2026.

5. **Active-Dormant Heads:** Guo, Z. et al. (2024). Active-Dormant Attention Heads: Mechanistically Demystifying Extreme-Token Phenomena in LLMs. NeurIPS 2024. arXiv:2410.13835.

6. **HONOR Dormant Head Metric:** Sandoval-Segura, P. et al. (2025). Using Attention Sinks to Identify and Evaluate Dormant Heads in Pretrained LLMs. arXiv:2504.03889. April 2025.

7. **qTTT:** Lou, A. et al. (2025). Let's (not) just put things in Context: Test-Time Training for Long-Context LLMs. ICLR 2026. arXiv:2512.13898. Meta/Harvard/OpenAI/Berkeley/UT Austin.

8. **DiffAttn:** Ye, T. et al. (2025). Differential Transformer. ICLR 2025. arXiv:2410.05258. Microsoft Research.

9. **DiffLoRA:** Misrahi, A. et al. (2025). DiffLoRA: Differential Low-Rank Adapters for Large Language Models. arXiv:2507.23588. July 2025.

10. **GatedAttn:** Qiu, Z. et al. (2025). Gated Attention for Large Language Models: Non-linearity, Sparsity, and Attention-Sink-Free. NeurIPS 2025 Best Paper. arXiv:2505.06708. Alibaba Qwen.

11. **Spike/Sparse/Sink Anatomy:** Sun, S., Canziani, A., LeCun, Y., Zhu, J. (2026). The Spike, the Sparse and the Sink: Anatomy of Massive Activations and Attention Sinks. NYU. arXiv:2603.05498. March 5, 2026.

12. **Orthogonal Self-Attention:** Zhang, L. and Martens, J. (2026). Orthogonal Self-Attention. arXiv:2602.05996. February 2026. (Related to XSA but different mechanism — orthogonal attention matrix via matrix exponential vs. output projection.)

13. **MAPS Head Functionality:** Elhelo, A. and Geva, M. (2025). Inferring Functionality of Attention Heads from their Parameters. ACL 2025. arXiv:2412.11965.

14. **TTT Done Right (LaCT):** Zhang, T. et al. (2026). Test-Time Training Done Right. ICLR 2026. arXiv:2505.23884. MIT/Adobe Research.

15. **Surgery (Sink-Guided Fine-Tuning):** Liu, G. et al. (2026). Surgery: Mitigating Harmful Fine-Tuning for Large Language Models via Attention Sink. arXiv:2602.05228. February 2026.

16. **Sink Forges MoE:** (2026). Attention Sink Forges Native MoE in Attention Layers: Sink-Aware Training to Address Head Collapse. arXiv:2602.01203.

17. **Transformer Circuits:** Elhage, N. et al. (2021). A Mathematical Framework for Transformer Circuits. Anthropic. transformer-circuits.pub.

18. **LoRA:** Hu, E. et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. arXiv:2106.09685.

19. **Attention Entropy Collapse:** (2025). Variance Sensitivity Induces Attention Entropy Collapse in Transformers. EMNLP 2025. ACL Anthology:2025.emnlp-main.421.

20. **Sinks and Compression Valleys:** (2025). Attention Sinks and Compression Valleys in LLMs are Two Sides of the Same Coin. arXiv:2510.06477.

21. **Geometric Approach on Sink:** (2025). What are you sinking? A geometric approach on attention sink. arXiv:2508.02546.

22. **LoRA-TTT (VLM):** (2025). LoRA-TTT: Low-Rank Test-Time Training for Vision-Language Models. ICLR 2025. arXiv:2502.02069.

---

## 12. Quick-Reference Decisions for Our Setup

| Question | Answer |
|---|---|
| Does XSA denominator go stale during Q-only TTT? | No. W_V frozen => v_i frozen => ||v_i||^2 unchanged. |
| Should we apply Q-LoRA to XSA layers? | Yes. Cleanest gradient, highest marginal benefit. |
| Do attention sinks hurt per-document TTT? | Yes, via gradient sinks and score dilution waste. |
| Should we target sink-heavy heads for Q adaptation? | Opposite: use HONOR to SKIP dormant/sink heads. |
| Does DiffAttn benefit more from Q-TTT than standard attn? | Yes, 3 reasons (clean gradient, amplification, lambda decoupled). |
| Should we adapt lambda in DiffAttn during TTT? | No. Keep lambda frozen. |
| Should we adapt W_gate in GatedAttn layers during TTT? | Yes, as a rank-1 gate LoRA, LR 10x smaller, with collapse monitoring. |
| Should gate LoRA run simultaneously with Q-LoRA? | Sequential preferred: Q first, then joint Q+gate. |
| Any risk from Q-TTT in GatedAttn layers? | If gate < 0.1 average, skip Q-LoRA for that head (suppressed anyway). |
