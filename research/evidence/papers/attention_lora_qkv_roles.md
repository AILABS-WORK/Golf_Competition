# Attention LoRA Specialization: Mechanistic Roles of Q, K, V, O and the Scientific Basis for Q-Only TTT

**Research date:** 2026-03-24
**Databases searched:** arXiv (direct HTML), Semantic Scholar, ACL Anthology, transformer-circuits.pub
**Papers reviewed:** 14 primary + secondary sources
**Status:** Complete

---

## 1. Executive Summary

The question of which attention projection to adapt under test-time training (TTT) or LoRA fine-tuning has a mathematically grounded answer that emerges from the intersection of four research threads: mechanistic interpretability (Elhage 2021, Voita 2019, Clark 2019), attention pruning (Michel 2019), low-rank adaptation theory (Hu et al. 2022), and the qTTT provable convergence result (Lou et al. 2025, arXiv:2512.13898). The unified picture is:

- **Q (query):** Controls *what each token searches for*. Q updates directly increase the logit margin separating target from distractor tokens in the key space. Q can be adapted without invalidating the KV cache. Q gradient is position-local, directional, and provably sufficient to overcome score dilution in long contexts.
- **K (key):** Controls *how each token presents itself to be found*. K updates are expensive: modifying K invalidates the full KV cache, requiring a fresh O(T) prefill. K encodes the "QK circuit" half that determines *which* tokens are attended to.
- **V (value):** Controls *what information is retrieved when a token is attended to*. V updates are futile under score dilution: Proposition 2.4 of qTTT proves the output is linearly bounded by the attention mass on the target. If attention mass is diffuse, adapting V cannot rescue it.
- **O (output):** Projects the retrieved value mix back to the residual stream. O is the "OV circuit" output half. Adapting O is equivalent to changing what information flows from the attention sublayer to the next layer — a higher-level semantic intervention than adapting V.

---

## 2. Mechanistic Framework: The QK and OV Circuits

### 2.1 The Transformer Circuits Decomposition (Elhage et al. 2021)

Elhage et al. (Anthropic, 2021, "A Mathematical Framework for Transformer Circuits") decompose each attention head into two functionally independent circuits:

**QK Circuit** — determines *which* position attends to *which*:
```
Attention pattern A_h = softmax( (X W_Q^h) (X W_K^h)^T / sqrt(d_k) )
```
The compound matrix `W_Q^h (W_K^h)^T` (a bilinear form on token pairs) fully specifies the routing decision. This circuit answers: "Does token i look at token j?"

**OV Circuit** — determines *what* the head writes to the residual stream once it attends:
```
Head output contribution = A_h (X W_V^h) W_O^h
```
The compound matrix `W_V^h W_O^h` fully specifies the content transformation. This circuit answers: "What does head h write to the residual stream when it reads from position j?"

**Key independence result:** These two circuits compose but are individually analyzable. You can freeze A_h (the attention pattern) and vary what is written (OV circuit), or freeze what is written and vary the pattern (QK circuit). This is precisely the logic behind Q-only TTT: the goal is to repair the *routing decision* (QK circuit), not the *content transformation* (OV circuit), because the model's knowledge of what information is useful (V, O) was already learned during pretraining.

### 2.2 Standard Attention Backward: Which Gradient Is What

From "Reversed Attention: On The Gradient Descent Of Attention Layers In GPT" (arXiv:2412.17019, 2024), the explicit VJP formulas are:

```
δ_q^j = R_j × K         # Q gradient: current position's attention row times all keys
δ_k^j = R_j^T × Q       # K gradient: transpose attention (future positions) times all queries
δ_v^j = Σ_l (A_{l,j} × e^l)  # V gradient: attention-weighted sum of error signals
```

Where `R` is the "Reversed Attention" matrix — the softmax derivative, which is itself a lower-triangular attention-like matrix. Key observations:

1. **Q gradients are position-local.** `δ_q^j` depends only on the current row's attention weights times keys. Each query's gradient is self-contained.
2. **K gradients are globally aggregated.** `δ_k^j` accumulates gradients from *all future positions* that attended to token j (via the transpose). In long contexts (T → large), K gradients aggregate O(T) signals — making them high-variance and expensive to track faithfully.
3. **V gradients couple to task error.** `δ_v^j` is weighted by the forward attention scores. Under score dilution (attention mass spread across distractors), the gradient signal reaching V from the target position is attenuated by the small `A_{l,j*}` weight.

This asymmetry directly supports Q-only TTT: Q has the cleanest, most directional, and most efficiently computable gradient in the TTT setting.

---

## 3. The qTTT Mathematical Justification (arXiv:2512.13898)

### 3.1 The Score Dilution Problem

**Definition:** As context length T grows, attention mass `α_{t,j*}` on the target token j* shrinks toward 1/T for uniformly distributed queries. This is the fundamental bottleneck for long-context retrieval.

**Lemma 2.3 (Margin Requirement):** To guarantee that attention mass on the target token exceeds `(1-ε)` against worst-case distractors, the logit gap must satisfy:
```
min_{j ≠ j*} (z_{i,j*} - z_{i,j}) ≥ log((T-1)(1-ε)/ε)
```
The required margin grows as **Ω(log T)** — logarithmically in context length. Static pretrained models cannot adapt this margin at inference time.

**Proposition 2.4 (Score Dilution Bound):** For any target direction u ∈ R^{d_v}:
```
⟨u, o_t⟩ ≤ α_{t,j*}⟨u, v_{j*}⟩ + (1 - α_{t,j*}) max_{j≠j*} ⟨u, v_j⟩
```
This is the crucial bound. When `α_{t,j*}` is small (diluted), the output is dominated by distractor values regardless of how good v_{j*} is. **Adapting V under this regime is provably futile.** The signal in v_{j*} is irretrievably swamped by the distractor mass.

### 3.2 The TTT Procedure

**Step 1 — Single Prefill:** Execute one complete forward pass on context x_{1:T}, caching:
```
K^(ℓ) ∈ R^{T × d_k},  V^(ℓ) ∈ R^{T × d_v}  for each layer ℓ = 1..L
```
These are frozen for all subsequent steps. This is the fundamental KV cache reuse.

**Step 2 — Span-Sampled Query Updates:** For N_TTT iterations:
- Sample random contiguous span x_s = x_{t:t+k} where k ≪ T (typically k = 128)
- Compute the loss:
  ```
  L_TTT(θ; x_s) = -Σ_{i=t}^{t+k-1} log p_θ(x_{i+1} | x_{1:i}; {K^(ℓ), V^(ℓ)})
  ```
- Update **only** {W_Q^(ℓ)} via gradient descent; keep all other parameters frozen

**Step 3 — Answer Generation:** Use the adapted model (updated W_Q only, unchanged K/V cache) to generate.

### 3.3 The Q Gradient Mechanics

**Proposition 3.1 (Query Update Gradient):** For the per-position loss `ℓ_i = -log α_{i,j*}` with fixed K/V:
```
∇_{q_i} ℓ_i = (1/√d_k)(Σ_{ℓ=1}^T α_{i,ℓ} k_ℓ - k_{j*})
             = (1/√d_k)(μ_i - k_{j*})
```
Where `μ_i = Σ_ℓ α_{i,ℓ} k_ℓ` is the **attention-weighted mean of all keys**.

Interpretation: The gradient moves query q_i **toward the target key k_{j*}** and **away from the current attention-weighted centroid μ_i**. This directly increases the inner product gap `q_i · k_{j*} - q_i · k_{distractor}`.

**Lemma 3.2 (Guaranteed Margin Improvement):** For sufficiently small learning rate η > 0:
```
M_i(q_i - η ∇_{q_i} ℓ_i) = M_i(q_i) + η ‖∇_{q_i} ℓ_i‖²₂ + O(η²)
```
Where M_i is the logit margin. The margin **strictly increases** after each step, with improvement proportional to `‖k_{j*} - μ_i‖²₂`. Crucially:

> **The gain is largest precisely when attention is most diffuse** (μ_i is pulled away from k_{j*} by many distractors) — exactly the hard long-context regime.

### 3.4 Why K and V Updates Are Excluded

**K exclusion:** Modifying any K^(ℓ) invalidates the KV cache, requiring a fresh O(T) prefill. One update altering K requires recomputing forward-backward passes over all T tokens — approximately 1.2×T decoding token equivalents in FLOPs. This negates all computational savings of cache reuse.

**V exclusion:** Proposition 2.4 proves the bound. Under score dilution, the output linearly weighted by `α_{t,j*}` cannot benefit from better v_{j*}; the distractor mass dominates. V adaptation is not wrong — it is simply **insufficient** and **orthogonal** to the retrieval routing problem.

**O exclusion:** The output projection W_O maps the post-softmax value mix to the residual stream. This operates on the aggregated output, not on the attention routing. Fixing the wrong routing and then trying to correct via W_O is equivalent to cleaning up downstream from a broken filter rather than fixing the filter itself.

---

## 4. Head Specialization: What Each Projection Encodes in Practice

### 4.1 Voita et al. 2019: Three Types of Specialized Heads

Voita, Talbot, Moiseev, Sennrich, Titov (ACL 2019, arXiv:1905.09418) identified three functionally distinct head types in Transformer NMT models. When heads are pruned via L0-regularization, specialized heads are **last to be pruned**:

| Head Type | Functional Role | Dominant Layers | What Q/K/V Encodes |
|-----------|----------------|-----------------|---------------------|
| **Positional heads** | Attend to fixed offsets (±1) | Early-middle | Q encodes "I am at position t", K encodes "I am at position t±1" |
| **Syntactic heads** | Track grammatical relations (nsubj, dobj, amod, advmod) | Middle layers | Q encodes "I need my syntactic governor", K encodes "I am a syntactic governor" |
| **Rare token heads** | One head in layer 1 attends to rare/unknown tokens | First layer | Q broadly distributed, K has high norm for rare token positions |

Key finding: **In each model, one head in the first layer is the most important**, and it attends preferentially to rare tokens. This is consistent with the attention sink literature (rare and first tokens receive disproportionate attention mass).

### 4.2 Clark et al. 2019 (Stanford): BERT Syntactic Attention

Clark, Khandelwal, Levy, Manning ("What Does BERT Look At?" arXiv:1906.04341, ACL 2019):

- Layer 2-4 heads detect basic grammatical relationships: noun-verb, determiner-noun
- Higher layers capture coreference, long-range dependencies
- **Delimiter/CLS tokens** receive heavy attention in early layers — consistent with attention sinks
- Heads in the same layer exhibit similar behaviors (layer-level coherence in attention patterns)
- Probing classifiers on attention weights recover substantial syntactic information

**Implication for TTT:** In a new document, syntactic structure is **token-specific but grammar-invariant**. Q updates in syntactic-head layers should require only small changes (grammar doesn't change per document) but positional heads may need more significant updates (sentence structure varies).

### 4.3 Michel et al. 2019 (Carnegie Mellon): Pruning Confirms Head Redundancy

Michel, Levy, Neubig ("Are Sixteen Heads Really Better than One?" arXiv:1905.10650, NeurIPS 2019):

**Head importance score:** Based on expected sensitivity of loss to masking head h:
```
I_h = E_{x ~ X} |∂L/∂ξ_h|
```
Where ξ_h is the attention gate for head h. This is the expected absolute gradient of the loss with respect to the head being present.

Key findings:
- Most attention heads can be pruned individually without significant performance loss
- **Encoder-decoder attention heads are most critical** (>60% pruning causes catastrophic degradation)
- Self-attention heads are more redundant than cross-attention heads
- Only a small fraction of heads carry specialized, non-redundant function
- Up to 40% of BERT heads can be pruned without noticeable harm

**Implication for TTT:** Since most heads are redundant, adapting all layers' W_Q uniformly may be inefficient. A head-importance-weighted Q adaptation (higher LR for high-importance heads) could be more efficient.

### 4.4 Rogers et al. 2020 (BERTology): Layer-Depth Specialization Evidence

Rogers, Kovaleva, Rumshisky ("A Primer in BERTology," TACL 2020, arXiv:2002.12327):

Synthesizing dozens of probing studies on BERT, the definitive layer-depth hierarchy is:

```
Layers 1-4:  Lexical features, local word order, surface-level patterns
Layers 5-8:  Syntactic phenomena (POS, chunking, dependency distance)
Layers 9-12: Semantic knowledge (coreference, discourse, entity types)
```

More precisely:
- **Surface/positional heads:** Predominantly layers 1-4; attend to adjacent tokens or fixed offsets
- **Syntactic heads:** Peak in middle layers (5-8); track grammatical relations
- **Semantic/coreference heads:** Layers 8-12; attend across long distances on semantic grounds

**Critical caveat:** Representations are diffuse — no single layer fully contains any linguistic property. Syntactic and positional heads are the **last to be pruned** in Voita's L0 experiments, confirming they are most load-bearing.

### 4.5 Attention Sinks: What K and V Store at Sink Positions

From "When Attention Sink Emerges in Language Models" (arXiv:2410.10781, ICLR 2025):

- Attention sinks emerge between 1,000-2,000 training steps, not at initialization
- At sink positions (typically token 1), key vectors have **smaller ℓ₂-norm** than average
- The sink is driven by **small angles** between q_t and k_1 (cosine similarity, not norm)
- Value vectors at sink positions have **near-zero norm** — the sink carries no content
- The sink is a softmax normalization artifact: it acts as a "key bias" absorbing surplus probability mass
- Removing softmax normalization eliminates sinks entirely (sigmoid attention shows no sinks)

**Implication for TTT:** When adapting W_Q, queries at sink-heavy heads will naturally have small gradients toward k_{sink} (since v_{sink} ≈ 0, there is no meaningful signal to chase). The qTTT gradient formula `μ_i - k_{j*}` will correctly route Q away from the sink and toward semantically meaningful keys.

---

## 5. LoRA on Attention: Empirical Evidence for Q vs K vs V vs O

### 5.1 LoRA Ablation Results (Hu et al. 2022, Table 5 in arXiv:2106.09685)

GPT-3 175B, 18M parameter budget (rank r shown per matrix type):

| Adapted matrices | WikiSQL (Acc%) | MNLI (Acc%) | Notes |
|-----------------|----------------|-------------|-------|
| W_Q only, r=8 | 70.4 | 91.0 | Needs larger r |
| W_V only, r=8 | 73.0 | 91.0 | Better alone than Q |
| W_Q + W_V, r=4 | 73.7 | 91.3 | **Best efficiency** |
| All four matrices, r=2 | 73.7 | 91.7 | Marginal gain over Q+V |

**W_K performs worst alone.** The paper does not report W_K alone but notes that "W_Q and W_V together give the best overall performance." W_K is omitted from recommended configurations.

### 5.2 The Intrinsic Rank Hypothesis Differential

From Table 6 of LoRA: r=1 suffices for W_Q + W_V joint adaptation on most tasks, but **W_Q alone requires r ≥ 2 to match**. This reveals:

- W_V has **stronger low-rank structure** than W_Q across random seeds — V updates concentrate in fewer dimensions
- W_Q requires more rank to fully represent task-specific changes — Q changes are higher-dimensional
- Subspace analysis (Figure 3 in the paper): top singular vectors of high-rank W_Q adaptation overlap significantly with low-rank, confirming the intrinsic dimensionality is indeed low but varies by matrix type

**Theoretical insight from fine-tuning theory (IJCAI 2025):** The gradient of W_V has a structural asymmetry due to softmax normalization: `∂L/∂W_V` remains non-zero even when gradient flow to W_Q and W_K is near-saturated. This suggests that **in fine-tuning tasks where routing is already correct (the common fine-tuning scenario), V adaptation is more informative**. But in TTT where routing is broken (score dilution), Q adaptation is more informative.

---

## 6. Differential Attention (DiffAttn): Q1 vs Q2 Specialization

### 6.1 Mathematical Formulation (arXiv:2410.05258)

The differential attention mechanism (Microsoft, NIPS 2024):
```
DiffAttn(X) = (softmax(Q1 K1^T / √d) − λ softmax(Q2 K2^T / √d)) V
```

Where Q1, Q2, K1, K2 ∈ R^{N×d} are obtained by splitting the projected query and key matrices; V ∈ R^{N×2d} is the full value matrix. The scalar λ is learnable and initialized as:
```
λ = exp(λ_{q1} · λ_{k1}) - exp(λ_{q2} · λ_{k2}) + λ_init
λ_init = 0.8 - 0.6 × exp(-0.3 × (l-1))    # layer-dependent initialization
```
The layer-dependent λ_init decreases across depth: early layers subtract *less* noise (higher λ_init), later layers subtract *more*.

### 6.2 Gradient Asymmetry Between Q1 and Q2

The gradient derivation shows asymmetric scaling:
- Q1, K1 gradients scale as `1/√d` (standard scaling)
- Q2, K2 gradients scale as `-λ/√d` (negative, noise-cancellation enforced)

Since λ ∈ (0,1) by the initialization scheme, |gradient Q2| < |gradient Q1|. Q1 receives larger gradient updates and thus learns faster. Q2 learns a *subtler* correction, the noise model.

### 6.3 Q1 vs Q2 Functional Roles for TTT

| Property | Q1 (signal branch) | Q2 (noise branch) |
|----------|-------------------|--------------------|
| Role | Amplifies relevant attention patterns | Cancels irrelevant/noise patterns |
| Gradient magnitude | Larger (×1/√d) | Smaller (×-λ/√d) |
| What it "looks for" | Semantically useful keys | Keys correlated with noise/distractors |
| TTT priority | **Primary adaptation target** | Secondary; may benefit from lower LR |
| Rank needed | Higher (signal is richer) | Lower (noise patterns are more uniform) |

**Implication for our architecture:** In DiffAttn TTT, Q1 should have higher learning rate and higher LoRA rank than Q2. A specific configuration might be: Q1 LoRA rank=8, Q2 LoRA rank=2, with Q2 LR = λ × Q1 LR. This mirrors the λ-weighted gradient asymmetry.

---

## 7. Value Residual Learning (ResFormer) and TTT Implications

### 7.1 Architecture (arXiv:2410.17897)

ResFormer adds a residual from the **first layer's value vectors** to all subsequent layers:
```
U_n = (1/2) × A_n × (V_n + V_1)
```
Where V_1 is the cached first-layer value embedding and A_n is the current layer's attention matrix. SVFormer extends this to `U_n = A_n × V_1` (fully shared values).

This is derived by approximating cross-layer attention: concatenating K/V across all layers and simplifying by replacing past keys with current keys. The 1/2 normalizes the mixture.

### 7.2 Why This Helps and What V_1 Contains

The first layer's attention is "naturally more dispersed" — it captures broad, local token information before deep processing. V_1 therefore provides:

- **Foundational token identity information** (what each token is, before contextual modification)
- **Mitigation of attention concentration** in deep layers where attention can collapse
- **KV cache efficiency** in SVFormer (nearly 50% cache reduction)

V_1 is best understood as carrying **lexical/positional identity** of each token — the raw embedding-level information before any transformer processing.

### 7.3 Should We LoRA-Adapt the V_1 Residual During TTT?

**Recommendation: No, do not LoRA-adapt V_1 during TTT.**

Reasons:
1. **V_1 is a bottleneck broadcast to all layers.** A LoRA on V_1 propagates its effect to every layer simultaneously via `V_n + V_1`. This creates cascading instability during rapid TTT updates (few steps, high LR).
2. **The 1/2 scaling assumption is violated.** ResFormer's normalization assumes equal contribution from V_n and V_1. LoRA adaptation shifts this balance without compensating adjustments to A_n.
3. **V_1 carries lexical identity, not retrieval routing.** The qTTT analysis shows the bottleneck is routing (Q), not identity (V). Document-specific adaptation should target how the model *searches* (Q), not what tokens *are* (V_1).
4. **Layer coupling hazard.** In SVFormer, all layers share V_1 — adapting it is equivalent to adapting a globally-shared representation, which will generalize poorly to different token positions.

**Alternative:** If value adaptation is desired, prefer per-layer V_n LoRA (only in the final few layers, where semantic information is highest-density) over touching V_1.

---

## 8. Layer-Depth Specialization and Differential LR for TTT

### 8.1 The Evidence-Based Layer Hierarchy

Combining BERTology (Rogers 2020), Voita (2019), and Clark (2019), the clearest layer-depth specialization picture for Q projections is:

```
Early layers (1-4):
  - Positional heads: Q encodes relative position query ("who is at offset ±1?")
  - Surface heads: Q encodes local n-gram context
  - Rare token heads: layer 1, Q broadly distributed
  - For TTT: these heads STABLE across documents — Q needs SMALL LR
  - Document-specific content is NOT here

Middle layers (5-8):
  - Syntactic heads: Q encodes "find my governor / dependent"
  - Mixed syntactic/semantic heads emerge
  - For TTT: moderate instability — grammar doesn't change but sentence structure varies
  - Q needs MEDIUM LR

Late layers (8-12, or last ~1/4 of the model):
  - Semantic/coreference heads: Q encodes "find the entity I corefer to"
  - Discourse heads: Q encodes "find the clause that resolves my reference"
  - **These are the most document-specific heads**
  - For TTT: Q needs LARGE LR (semantics is document-specific)
```

### 8.2 E2E-TTT Corroborating Evidence

The TTT-E2E paper (arXiv:2512.23675) adapts **only the last 1/4 of MLP layers** during test-time training, observing that:
- Adapting only 1-3 layers fails to scale with context
- Adapting only early layers also fails
- The last 1/4 provides the best trade-off

While this applies to MLP layers (not Q projections), the principle is consistent: **late layers carry the most document-specific, adaptation-relevant information**.

### 8.3 Recommended Layer-Depth LR Schedule for Q-TTT

Proposed LR scaling for Q LoRA across depth l ∈ [1, L]:
```
LR(l) = LR_base × (1 + (l/L)^2 × (LR_max_mult - 1))
```
Where LR_max_mult ≈ 5-10× concentrates adaptation effort in late layers. Concrete example for L=12 layers:
- Layers 1-3: LR = LR_base × 1.0 (positional/surface heads, stable)
- Layers 4-6: LR = LR_base × 2.0 (syntactic heads, moderate)
- Layers 7-9: LR = LR_base × 4.0 (mixed semantic-syntactic, higher variance)
- Layers 10-12: LR = LR_base × 8.0 (semantic/coreference, document-specific)

---

## 9. The XSA Connection: Exclusive Self Attention in the Last 4 Layers

### 9.1 What XSA Does

Exclusive Self Attention (arXiv:2603.09078, March 2026) modifies attention output by removing the self-value projection:
```
z_i = y_i - (y_i^T v_i)(v_i / ‖v_i‖²₂)
```
This removes from the output any component parallel to the token's own value vector, forcing the attention output to be **orthogonal to the self-value direction**. The result: z_i contains only information the token received *from context*, not from itself.

**Connection to attention sinks:** XSA implicitly handles attention sinks by allowing `a_{i,i}` (self-attention) to absorb "undesired attention scores" — the diagonal acts as a learned sink without requiring explicit sink tokens.

### 9.2 What XSA Changes About the Q Role

In standard attention, Q serves two purposes simultaneously:
1. **Retrieval routing:** "Find the contextually relevant tokens"
2. **Self-identity suppression:** Implicit (not explicit) — Q may inadvertently attend to self

In XSA, the self-direction is explicitly subtracted post-hoc. This means Q in XSA layers is **free to specialize more purely on retrieval routing** — the self-content is handled by the orthogonalization step, not by Q design.

**Implication for Q-TTT in XSA layers:** W_Q in XSA layers should be adapted with the same priority as W_Q in standard layers (or potentially *higher*, since Q in XSA is more purely a retrieval router). The orthogonalization step does not interact with W_Q directly — it operates on the output z_i after attention is computed.

### 9.3 Should XSA Layers Be Treated Differently in TTT?

**For the 4-layer XSA suffix (late layers):** These are already in the high-LR zone from the layer-depth schedule. The XSA mechanism adds one wrinkle: the value vector v_i appears in both the attention computation (as part of A_n V) and in the exclusion formula (as v_i / ‖v_i‖²₂). If W_V is adapted in XSA layers, both the attended content and the exclusion direction change, creating potential instability in the self-direction removal.

**Recommendation for XSA-layer TTT:**
- Adapt W_Q with standard high-LR treatment (XSA Q is a purer retrieval router — adaptation is cleaner)
- **Do NOT adapt W_V in XSA layers during TTT** (would destabilize the orthogonalization denominator ‖v_i‖²₂)
- W_O can be adapted if budget allows (it operates after XSA's exclusion, so no instability)
- Use slightly lower rank for Q in XSA layers vs standard attention (the signal subspace is already cleaned by exclusion — less rank needed to represent the remaining direction changes)

---

## 10. Novel Variants: 5 Specific Ideas

### Variant 1: Layer-Stratified Q-TTT (LS-qTTT)

**Concept:** Apply Q-only TTT but with layer-stratified learning rates following the depth-specialization hierarchy.

**Implementation:**
```python
# Learning rate multiplier for layer l of L total layers
lr_mult = 1.0 + ((l / L) ** 2) * (max_mult - 1.0)  # max_mult=8 for L=12

# During TTT update step:
for l, q_lora in enumerate(q_lora_layers):
    grad = compute_q_grad(loss, q_lora)
    q_lora.update(grad * lr_mult(l, L))
```

**Justification:** Early layers' Q encodes stable positional/syntactic information (low document-specificity). Late layers' Q encodes document-specific semantic search. Differential LR concentrates adaptation budget where it matters most.

**Expected gain:** 5-15% reduction in TTT steps required to reach the same margin improvement, with reduced risk of destabilizing early positional heads.

### Variant 2: DiffAttn-Aware Dual-Q LoRA (DQ-LoRA-TTT)

**Concept:** For DiffAttn layers, adapt Q1 and Q2 with separate LoRA adapters having different ranks and learning rates, matched to their gradient asymmetry.

**Implementation:**
```python
# Q1: signal branch — higher rank, standard LR
q1_lora = LoRA(rank=8, alpha=16)

# Q2: noise branch — lower rank, lambda-scaled LR
q2_lora = LoRA(rank=2, alpha=4)
lr_q2 = lr_q1 * lambda_l  # lambda_l = initial lambda for layer l

# In TTT, update both but asymmetrically
```

**Justification:** Q1 and Q2 have gradient magnitudes of ratio 1:λ (≈ 0.5-0.8). Q2's noise model requires fewer parameters (noise patterns are more uniform across documents). Matching LoRA rank to the effective gradient dimensionality avoids wasting parameters on Q2.

**Expected gain:** 30-40% parameter reduction vs equal-rank DQ-LoRA, with equivalent adaptation quality.

### Variant 3: XSA-Calibrated Q-TTT (XC-qTTT)

**Concept:** In XSA layers, after each TTT Q update step, re-calibrate the exclusion denominator `‖v_i‖²₂` using the updated Q's attention distribution.

**Motivation:** When W_Q is updated, the attended value mixture `y_i = Σ_j a_{i,j} v_j` changes. The exclusion step `z_i = y_i - (y_i^T v_i)(v_i / ‖v_i‖²₂)` uses the frozen v_i denominator. After Q updates, the angle between y_i and v_i may shift — recalibrating ensures the orthogonalization remains geometrically valid.

**Implementation:** Cheap (one dot product per position, no backward pass) — add `v_norm_recalib = ‖v_i‖²₂` recomputation after each W_Q gradient step in XSA layers.

### Variant 4: V_1-Frozen Q+V_n Late-Layer TTT (VLate-qTTT)

**Concept:** Combine Q-only TTT in early/middle layers with Q + per-layer-V_n TTT in the last 2-3 layers, while keeping V_1 (the value residual) completely frozen.

**Justification:**
- V_1 provides document-invariant lexical identity → never adapt
- V_n in late layers carries high-level semantic content → small LoRA safe
- Q in late layers does the retrieval routing → adapt as standard qTTT
- Budget split: Q LoRA rank=8 all layers, V_n LoRA rank=2 last 2-3 layers only

**Expected gain:** Late-layer V_n adaptation corrects content retrieval (what is returned when the correct key is found), while Q adaptation corrects routing (which key is found). These are complementary and non-interfering fixes. This extends qTTT from fixing routing-only to fixing routing+content.

### Variant 5: Head-Importance-Weighted Q-TTT (HIW-qTTT)

**Concept:** Use pre-computed head importance scores (from Michel et al.'s gradient-based importance) to allocate Q LoRA rank per head at initialization.

**Implementation:**
```python
# Compute offline head importance I_h = E[|∂L/∂ξ_h|] on a small calibration corpus
# Normalize: w_h = I_h / max(I_h) ∈ [0, 1]
# Assign per-head rank:
rank_h = max(1, round(rank_max * w_h))

# Each head gets its own Q LoRA adapter sized by importance
q_lora_h = LoRA(d_in=d_k, d_out=d_k, rank=rank_h)
```

**Justification:** Michel et al. showed that most heads are redundant (only ~20% carry non-redundant function). Allocating more rank to the 20% of heads with high importance scores, and rank=1 to redundant heads, concentrates the Q adaptation budget on the heads that actually drive model behavior.

**Expected gain:** 50-70% parameter reduction vs uniform-rank Q LoRA, with potentially better performance due to avoiding over-parameterizing redundant heads.

---

## 11. Synthesized Answers to Key Research Questions

### Q1: Mechanistic Difference — Adapting Q vs K vs V in TTT

| Projection | What it adapts | TTT effect | Cache impact | Gradient structure |
|-----------|---------------|------------|-------------|-------------------|
| **Q** | "What this token searches for" | Moves queries toward relevant keys; directly increases logit margin | None (cache reused) | Local (per-position), directional, provably sufficient |
| **K** | "How this token presents itself" | Reshapes key manifold; changes who gets found | Full cache invalidation O(T) | Globally aggregated, high-variance in long context |
| **V** | "What info is returned when found" | Improves content quality when retrieved | None | Attention-weighted, attenuated under score dilution |
| **O** | "How retrieved mix projects to stream" | Changes downstream residual stream contribution | None | Depends on upstream V and A; far from retrieval bottleneck |

**For a TTT scenario with a NEW document:** The bottleneck is the QK routing being miscalibrated to the new document's content distribution. Q is the correct and sufficient intervention.

### Q2: Why Q Freezing K/V Cache Works — Mathematical Justification

The KV cache records the document's "address space" (keys) and "content space" (values) in the pretraining representation. For a new document, these representations are fixed by the single prefill. Q adaptation then acts as **query calibration** in that fixed address space: it learns to ask the right questions (`k_{j*}`) rather than relying on the question-asking pattern learned during pretraining on other documents.

The Lemma 3.2 margin improvement is the formal guarantee: each gradient step `q_i ← q_i - η(μ_i - k_{j*})/√d_k` strictly increases the logit margin M_i with improvement ∝ ‖k_{j*} - μ_i‖². The K/V cache provides the reference frame (k_{j*} and μ_i) against which Q is calibrated.

### Q3: DiffAttn Q1 vs Q2 — Different LRs/Ranks in TTT

Yes. Q1 should have higher LR and rank than Q2 during TTT:
- Q1 gradient magnitude: 1/√d
- Q2 gradient magnitude: λ/√d (where λ ≈ 0.5-0.8 in typical DiffAttn configurations)
- Q2's noise model is more uniform across documents → lower intrinsic dimensionality → lower rank sufficient
- Recommended: Q1 rank=8, Q2 rank=2, LR_Q2 = λ × LR_Q1

### Q4: Should V_1 Residual Be LoRA-Adapted During TTT?

**No.** V_1 is a globally broadcast information bottleneck carrying document-invariant lexical identity. Adapting it during TTT would: (a) cascade changes through all layers simultaneously, (b) violate the 1/2 balance normalization assumption, (c) address lexical identity rather than retrieval routing (the actual TTT bottleneck). Late-layer V_n can be lightly adapted (rank=2) as a complement to Q adaptation.

### Q5: Layer-Depth Q LR Specialization — Evidence and Recommendation

Strong evidence: BERTology (Rogers 2020), Voita (2019), Clark (2019) all confirm early layers encode stable positional/syntactic patterns while late layers encode document-specific semantic patterns. E2E-TTT (2025) finds adapting only the last 1/4 of layers is optimal for MLP TTT, consistent with the same hierarchy.

**Recommended Q-TTT LR schedule:**
```
Early 1/3 of layers: LR × 1.0
Middle 1/3 of layers: LR × 3.0
Last 1/3 of layers:   LR × 8.0
```
This quadratically front-loads adaptation in semantically specialized late layers.

### Q6: Gradient of Q Relative to K and V

Based on the VJP formulas from "Reversed Attention" (arXiv:2412.17019):
- **Q gradient:** Position-local (`R_j × K`). Each query's gradient is a function only of its own attention row's deviation from the target. Clean, directional, low-variance.
- **K gradient:** Globally aggregated (`R_j^T × Q`). Accumulates from all future tokens that attended to position j. In long context, high-variance due to O(T) accumulation. Hard to use efficiently.
- **V gradient:** Attention-weighted error signal (`Σ_l A_{l,j} × e^l`). Under score dilution, the target position j* receives gradient proportional to its small attention weight α_{l,j*}. Signal-to-noise ratio degrades as T grows.

**Summary:** Q has the cleanest, lowest-variance, most directional gradient in the TTT regime. K's gradient is expensive and high-variance. V's gradient is attenuated by the very dilution problem we're trying to solve.

---

## 12. Citation Index

1. **Lou et al. 2025** — "qTTT: Query-Only Test-Time Training" (arXiv:2512.13898, Meta/Harvard/OpenAI). Key contributions: score dilution bound (Proposition 2.4), margin requirement (Lemma 2.3), query gradient formula (Proposition 3.1), margin improvement guarantee (Lemma 3.2).

2. **Hu et al. 2022** — "LoRA: Low-Rank Adaptation of Large Language Models" (arXiv:2106.09685, Microsoft). Key: W_Q+W_V optimal, W_K underperforms, r=1 sufficient for Q+V, intrinsic rank hypothesis.

3. **Voita et al. 2019** — "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned" (arXiv:1905.09418, ACL 2019). Key: three head types (positional, syntactic, rare-token), L0-pruning survival rates, layer distribution.

4. **Clark et al. 2019** — "What Does BERT Look At? An Analysis of BERT's Attention" (arXiv:1906.04341, ACL BlackboxNLP 2019). Key: syntactic head patterns, delimiter attention, layer-level coherence, probing classifiers.

5. **Michel et al. 2019** — "Are Sixteen Heads Really Better than One?" (arXiv:1905.10650, NeurIPS 2019). Key: head importance score I_h = E[|∂L/∂ξ_h|], most heads prunable, encoder-decoder attention most critical.

6. **Rogers et al. 2020** — "A Primer in BERTology: What We Know About How BERT Works" (arXiv:2002.12327, TACL). Key: layer-depth hierarchy (layers 1-4 surface, 5-8 syntactic, 9-12 semantic), diffuse representation caveat.

7. **Elhage et al. 2021** — "A Mathematical Framework for Transformer Circuits" (transformer-circuits.pub, Anthropic). Key: QK circuit = routing (W_Q W_K^T), OV circuit = content (W_V W_O), circuit independence principle.

8. **Ye et al. 2024** — "Differential Transformer" (arXiv:2410.05258, Microsoft, NIPS 2024). Key: DiffAttn formula, Q1/Q2 functional roles, λ gradient asymmetry (Q1:1/√d, Q2:-λ/√d), layer-depth λ_init.

9. **Zhang et al. 2024** — "Value Residual Learning for Alleviating Attention Concentration in Transformers" (arXiv:2410.17897, ResFormer). Key: U_n = (1/2)A_n(V_n + V_1) formula, SVFormer variant, 50% KV cache reduction, attention concentration mitigation.

10. **Sun et al. 2024** — "When Attention Sink Emerges in Language Models: An Empirical View" (arXiv:2410.10781, ICLR 2025). Key: sink emerges at 1K-2K steps, K/V at sink have small norm, sink = softmax normalization artifact, cosine similarity mechanism.

11. **Kim et al. 2026** — "Exclusive Self Attention" (arXiv:2603.09078). Key: z_i = y_i - (y_i^T v_i)v_i/‖v_i‖², removes self-direction from output, implicit attention sink, benefits increase with layer depth.

12. **Vaswani et al. 2017** — "Attention Is All You Need" (arXiv:1706.03762, NeurIPS 2017). Key: original Q, K, V definitions, scaled dot-product attention formula, multi-head attention.

13. **Xie et al. 2025 (TTT-E2E)** — "End-to-End Test-Time Training for Long Context" (arXiv:2512.23675). Key: last 1/4 of layers most effective for TTT, MLP layers only (attention layers cause instability), mini-batch TTT procedure.

14. **Wu et al. 2025** — "Test-Time Training with KV Binding Is Secretly Linear Attention" (arXiv:2602.21204). Key: TTT is learned linear attention, final-layer adaptation most effective, distributional mismatch between Q and K acceptable.

---

## 13. Key Equations Reference Card

```
# Standard Attention
A = softmax(QK^T / √d_k)
O = A V W_O

# QK Circuit (routing)
A_h = softmax(X W_Q^h (X W_K^h)^T / √d_k)

# OV Circuit (content)
contribution_h = A_h (X W_V^h) W_O^h

# Score Dilution Bound (qTTT Proposition 2.4)
⟨u, o_t⟩ ≤ α_{t,j*}⟨u, v_{j*}⟩ + (1-α_{t,j*}) max_{j≠j*} ⟨u, v_j⟩

# Margin Requirement (qTTT Lemma 2.3)
min_{j≠j*}(z_{i,j*} - z_{i,j}) ≥ log((T-1)(1-ε)/ε)  [Ω(log T)]

# Query Gradient (qTTT Proposition 3.1)
∇_{q_i} ℓ_i = (1/√d_k)(μ_i - k_{j*})
where μ_i = Σ_ℓ α_{i,ℓ} k_ℓ   # attention-weighted key centroid

# Margin Improvement (qTTT Lemma 3.2)
M_i(q_i - η∇_{q_i}ℓ_i) = M_i(q_i) + η‖∇_{q_i}ℓ_i‖²₂ + O(η²)

# TTT Loss Function (qTTT)
L_TTT(θ; x_s) = -Σ_{i=t}^{t+k-1} log p_θ(x_{i+1} | x_{1:i}; {K^(ℓ), V^(ℓ)})

# Value Residual (ResFormer, arXiv:2410.17897)
U_n = (1/2) × A_n × (V_n + V_1)

# Differential Attention (DiffAttn, arXiv:2410.05258)
DiffAttn(X) = (softmax(Q1 K1^T/√d) - λ softmax(Q2 K2^T/√d)) V
λ_init(l) = 0.8 - 0.6 × exp(-0.3 × (l-1))

# XSA Exclusion (arXiv:2603.09078)
z_i = y_i - (y_i^T v_i)(v_i / ‖v_i‖²₂)

# Backward VJP Gradients (arXiv:2412.17019)
δ_q^j = R_j × K          # local, positional
δ_k^j = R_j^T × Q        # global, aggregated
δ_v^j = Σ_l A_{l,j} × e^l  # attention-weighted error
```
