# MoLE Deep Dive: Mixture of Lookup Experts (arXiv:2503.15798)
# ICML 2025 Oral — Full Technical Analysis for Parameter Golf Integration
# Researched: 2026-03-24

---

## 1. Paper Overview

**Title:** Mixture of Lookup Experts
**Venue:** ICML 2025 (Oral Presentation)
**arXiv:** https://arxiv.org/abs/2503.15798 (v2 available)
**GitHub:** https://github.com/JieShibo/MoLE
**OpenReview:** https://openreview.net/forum?id=wUEp13rqXP

**Core Claim:** A Mixture-of-Experts variant where the routed experts take the token embedding
(output of the embedding layer, i.e., a function of token ID only) as input rather than the
intermediate transformer hidden state. This single change makes the expert computation
context-free and therefore pre-computable as a lookup table indexed by token ID. The result:
zero FLOPs per expert at inference, with the LUTs offloaded to storage.

---

## 2. Architecture: Full MoLE Layer

### 2.1 Training-Time Structure

Each MoLE block replaces the standard FFN with a combined layer containing:

  (a) One shared (dense) FFN expert that takes intermediate features (the hidden state h)
      as input — identical to a standard transformer FFN. This expert handles context-dependent
      information.

  (b) K routed experts, each a small FFN, that take the token EMBEDDING e_i (the output of
      the token embedding table for token t_i) as input — NOT the hidden state h.

  (c) A router/gate network that also takes intermediate features (h) as input and produces
      per-token mixing weights over the K routed experts.

The final output of the expert layer is:

    y_i = FFN_shared(h_i) + sum_{k=1}^{K} g_{i,k} * f_k(e_i)

Where:
    h_i  = intermediate hidden state at position i (shape: [model_dim])
    e_i  = token embedding of token t_i (shape: [embed_dim], often equal to model_dim)
    g_{i,k} = routing weight for expert k at position i (scalar)
    f_k  = k-th routed expert (FFN applied to e_i)
    y_i  = output of the combined expert layer (shape: [model_dim])

### 2.2 Routing (Gating) Mechanism

The router is a linear layer applied to the intermediate hidden state h_i:

    g_i = Softmax(W_r * h_i)     shape: [K]

Where:
    W_r  = routing weight matrix (shape: [K, model_dim])
    g_i  = soft routing weights (sums to 1 across K experts)

NOTE: The router is context-dependent (depends on h_i, not just token ID). Only f_k(e_i)
is token-ID-only (and therefore pre-computable). The routing weights g_i CANNOT be
pre-computed — they are computed at inference time from the hidden state.

ALL K experts are activated at training time (dense routing, no sparsity). No auxiliary
load balancing losses or z-losses are used — MoLE is fully differentiable. Ablation studies
show that adding auxiliary losses HARMS MoLE performance because they misalign the
optimization objective with inference requirements.

### 2.3 Expert Structure

Each routed expert f_k is a small FFN applied to the token embedding e_i:

    f_k(e_i) = W_k2 * activation(W_k1 * e_i)

Where:
    W_k1: [intermediate_dim, embed_dim]
    W_k2: [model_dim, intermediate_dim]
    activation: typically SiLU or similar

Key hyperparameter: intermediate_dim. The paper tests:
    - intermediate_dim = embed_dim (1x)
    - intermediate_dim = 4 * embed_dim (4x)  <-- sweet spot
    - intermediate_dim = 16 * embed_dim (16x) <-- saturation, no further gain

The insight: increasing intermediate_dim during training does NOT increase the LUT size at
inference (the LUT is always vocab_size x model_dim). Diminishing returns kick in early
because the LUT capacity (vocab_size entries) is the true bottleneck.

### 2.4 Inference-Time Re-parameterization

Since e_i = E[t_i] where E is the token embedding table, e_i takes only vocab_size distinct
values. Therefore, for each expert f_k, we can pre-compute:

    LUT_k[v] = f_k(E[v])     for all v in {0, 1, ..., vocab_size - 1}

Shape of each LUT: [vocab_size, model_dim]

At inference, f_k(e_i) becomes simply LUT_k[t_i] — a single table lookup with zero FLOPs.

The K LUTs can be concatenated into one tensor: [K, vocab_size, model_dim], offloaded to
CPU/SSD, and only the relevant rows (one per token per expert) are loaded into VRAM.

The router (g_i computation) still runs at inference since it depends on h_i.

### 2.5 Full Inference Forward Pass

    # At layer l for sequence of tokens t_0..t_{T-1}:
    # h: [B, T, model_dim] - current hidden states
    # t: [B, T] - input token IDs

    # 1. Shared FFN (standard, context-dependent)
    shared_out = FFN_shared(h)       # [B, T, model_dim]

    # 2. Router applied to hidden state
    g = Softmax(h @ W_r.T)           # [B, T, K]

    # 3. Routed expert lookup (replaces FFN at inference)
    expert_outs = LUT_k[t]           # [B, T, K, model_dim] — pure lookup, zero FLOPs

    # 4. Weighted combination
    mole_out = (g.unsqueeze(-1) * expert_outs).sum(dim=-2)   # [B, T, model_dim]

    # 5. Final output
    y = shared_out + mole_out

---

## 3. Relationship to BigramHash and the Codebase's Simpler Interpretation

### 3.1 How the Codebase Uses MoLE (Simplified / Token-Only Variant)

The Parameter Golf codebase implements a simplified version of MoLE that diverges from the
paper in one key respect:

**Paper's MoLE:** The router g_i uses the HIDDEN STATE h_i (context-dependent).
**Codebase's MoLE:** The router uses the TOKEN ID directly via an Embedding table.

Codebase implementation (train_gpt.py:1290-1304):

```python
class MixtureOfLookupExperts(nn.Module):
    def __init__(self, vocab_size, num_experts, expert_dim, model_dim):
        self.experts = nn.ModuleList([nn.Embedding(vocab_size, expert_dim) for _ in num_experts])
        self.gate    = nn.Embedding(vocab_size, num_experts)   # <-- token-ID gated, NOT h_i
        self.proj    = CastedLinear(expert_dim, model_dim)

    def forward(self, input_ids):  # (B, T)
        gates      = softmax(self.gate(input_ids), dim=-1)  # [B, T, E] — token-ID only
        expert_outs = stack([e(input_ids) for e in self.experts], dim=-2)  # [B, T, E, D]
        combined    = (gates.unsqueeze(-1) * expert_outs).sum(dim=-2)     # [B, T, D]
        return self.proj(combined)   # [B, T, model_dim]
```

The codebase module is then ADDED to the residual stream at the embedding stage:

    x = tok_emb(input_ids)
    x = x + mole(input_ids)   # added before the transformer blocks

This is a fully context-free operation: both the expert outputs AND the routing weights
depend only on token ID, making the entire thing equivalent to a single token-ID-indexed
lookup after training.

### 3.2 Algebraic Equivalence to BigramHash

BigramHashEmbed computes for token at position i:
    bigram_idx = hash(token[i-1], token[i]) mod num_buckets
    output = Embedding[bigram_idx] @ W_proj

The codebase's MoLE computes for token at position i:
    gates_i      = softmax(gate_table[token[i]])          # K-dim vector
    expert_outs_i = stack(expert_table_k[token[i]] for k)  # K x D matrix
    combined_i   = gates_i @ expert_outs_i                 # D-dim vector
    output_i     = combined_i @ W_proj                     # model_dim-dim vector

Key differences:
1. MoLE is UNIGRAM-based (only current token ID), not bigram-based
2. MoLE uses a learned soft mixture of K tables; BigramHash uses a single hash table
3. BigramHash encodes bigram statistics (captures short-range context); MoLE encodes
   richer per-token statistics via multiple learned basis functions

MoLE generalizes BigramHash in the sense that multiple lookup tables + learned mixing
weights can represent a richer function of the current token than any single table can.
However, BigramHash captures prev-token context which the token-only MoLE CANNOT.

### 3.3 True Paper MoLE vs. Codebase MoLE: Comparison Table

| Aspect                | Paper MoLE                          | Codebase MoLE                    |
|-----------------------|-------------------------------------|----------------------------------|
| Router input          | Hidden state h_i (context-aware)   | Token ID only (token_id-indexed) |
| Expert input          | Token embedding e_i                 | Token ID only                    |
| Placement             | Inside each transformer block (MoE layer replacement) | Before transformer blocks (at embedding) |
| Shared FFN            | Yes, alongside routed experts       | No                               |
| Re-parameterizable    | Experts yes, router NO              | Entire module YES                |
| Relationship to paper | Full paper implementation           | Simplified/token-only variant    |
| torch.compile safe    | Yes (no Python loops)               | Yes (nn.ModuleList uses stack)   |

The codebase variant is essentially a "pre-computed per-token feature augmentation" module.
Both the experts AND the gate are fully parameterized by token ID, meaning the entire
combined output is a fixed function of the input token: the whole thing is equivalent to
a single learned embedding of shape [vocab_size, model_dim] but factored through K basis
functions with a mixing head to get more expressive power per parameter.

---

## 4. Exact PyTorch Pseudocode: Paper-Faithful Implementation

### 4.1 Per-Layer MoLE (Paper's Approach)

This is what the paper actually proposes — replacing standard FFN layers inside transformer
blocks. The router depends on the hidden state; only the expert lookup is token-id-based.

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MoLEExpertLayer(nn.Module):
    """
    Paper-faithful MoLE layer (arXiv:2503.15798).

    Replaces a standard FFN inside a transformer block.
    Output: shared_expert(h) + sum_k( gate_k(h) * lookup_expert_k(token_id) )

    At inference, lookup_expert_k can be replaced by a precomputed LUT:
        LUT[k, v] = f_k(E[v])  for all v in vocab
    """

    def __init__(
        self,
        vocab_size: int,
        model_dim: int,
        num_routed_experts: int,
        expert_intermediate_dim: int,   # typically 4 * embed_dim
        ffn_intermediate_dim: int,      # shared FFN intermediate dim
        embed_dim: int,                 # token embedding dimension
    ):
        super().__init__()

        # Shared (dense) FFN — context-dependent, not re-parameterizable
        self.shared_ffn_w1 = nn.Linear(model_dim, ffn_intermediate_dim, bias=False)
        self.shared_ffn_w2 = nn.Linear(ffn_intermediate_dim, model_dim, bias=False)

        # Router — applied to hidden state, NOT re-parameterizable
        self.router = nn.Linear(model_dim, num_routed_experts, bias=False)

        # Routed experts — applied to token embedding e_i
        # Each f_k: embed_dim -> model_dim (via intermediate_dim)
        # At inference, replaced by LUT of shape [vocab_size, model_dim]
        self.expert_w1 = nn.ModuleList([
            nn.Linear(embed_dim, expert_intermediate_dim, bias=False)
            for _ in range(num_routed_experts)
        ])
        self.expert_w2 = nn.ModuleList([
            nn.Linear(expert_intermediate_dim, model_dim, bias=False)
            for _ in range(num_routed_experts)
        ])

        self.num_routed_experts = num_routed_experts

    def forward(
        self,
        h: Tensor,           # [B, T, model_dim] — hidden state
        token_embeds: Tensor # [B, T, embed_dim]  — token embeddings (E[input_ids])
    ) -> Tensor:
        # 1. Shared FFN output (context-dependent)
        shared_out = self.shared_ffn_w2(F.silu(self.shared_ffn_w1(h)))  # [B, T, model_dim]

        # 2. Routing weights (context-dependent via h)
        gates = F.softmax(self.router(h), dim=-1)  # [B, T, K]

        # 3. All K routed experts applied to token embeddings (token-ID-only)
        # Each expert: [B, T, embed_dim] -> [B, T, model_dim]
        expert_outs = torch.stack([
            self.expert_w2[k](F.silu(self.expert_w1[k](token_embeds)))
            for k in range(self.num_routed_experts)
        ], dim=-2)  # [B, T, K, model_dim]

        # 4. Weighted combination
        routed_out = (gates.unsqueeze(-1) * expert_outs).sum(dim=-2)  # [B, T, model_dim]

        return shared_out + routed_out

    @torch.no_grad()
    def build_luts(self, token_embedding_weight: Tensor) -> Tensor:
        """
        Pre-compute lookup tables for all routed experts.
        Call once before inference; store result on CPU/disk.

        token_embedding_weight: [vocab_size, embed_dim]
        Returns: luts [K, vocab_size, model_dim]
        """
        vocab_size = token_embedding_weight.shape[0]
        luts = []
        for k in range(self.num_routed_experts):
            # f_k applied to every token embedding
            lut_k = self.expert_w2[k](F.silu(self.expert_w1[k](token_embedding_weight)))
            luts.append(lut_k)  # [vocab_size, model_dim]
        return torch.stack(luts, dim=0)  # [K, vocab_size, model_dim]
```

### 4.2 Codebase-Style Token-Only MoLE (Matching Current train_gpt.py Pattern)

This variant is what the codebase already implements: everything is indexed by token ID,
placed BEFORE the transformer blocks, added to the residual stream at the embedding stage.

```python
class MixtureOfLookupExperts(nn.Module):
    """
    Token-ID-only MoLE (codebase variant, NOT paper-faithful for per-layer usage).

    Placed before transformer blocks. Both experts AND gate are indexed by token ID.
    Equivalent at inference to a single learned embedding [vocab_size, model_dim]
    factored via K basis functions + learned mixing for richer parameterization.

    Args:
        vocab_size:   size of token vocabulary
        num_experts:  K — number of lookup expert tables
        expert_dim:   D — dimension of each expert's embedding table
        model_dim:    output dimension for residual stream addition

    Usage:
        mole = MixtureOfLookupExperts(vocab_size=50257, num_experts=8,
                                       expert_dim=64, model_dim=768)
        x = tok_emb(input_ids)
        x = x + mole(input_ids)   # add to residual before transformer blocks
    """

    def __init__(self, vocab_size: int, num_experts: int, expert_dim: int, model_dim: int):
        super().__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([
            nn.Embedding(vocab_size, expert_dim) for _ in range(num_experts)
        ])
        for e in self.experts:
            nn.init.normal_(e.weight, std=0.01)

        # Gate: token-ID -> mixing weights over K experts
        # Zero init -> uniform softmax at start -> all experts equally weighted at t=0
        self.gate = nn.Embedding(vocab_size, num_experts)
        nn.init.zeros_(self.gate.weight)

        # Project expert_dim -> model_dim; zero-init for residual stability
        self.proj = nn.Linear(expert_dim, model_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: [B, T]

        # Gate lookup: token_id -> soft mixing weights
        gates = F.softmax(self.gate(input_ids).float(), dim=-1)  # [B, T, K]

        # Expert lookups: each expert_k[token_id] -> [B, T, D]
        # Stack into [B, T, K, D] — compile-safe, no Python branching per element
        expert_outs = torch.stack(
            [e(input_ids) for e in self.experts], dim=-2
        )  # [B, T, K, D]

        # Weighted combination: gates * expert_outs, sum over K
        combined = (gates.unsqueeze(-1).to(expert_outs.dtype) * expert_outs).sum(dim=-2)
        # combined: [B, T, D]

        return self.proj(combined)  # [B, T, model_dim]
```

### 4.3 Fused Single-Tensor Implementation (Memory-Efficient Alternative)

Instead of K separate nn.Embedding tables, fuse into one 3D tensor:

```python
class MixtureOfLookupExpertsFused(nn.Module):
    """
    Memory-efficient version: fuses K expert tables into one tensor.
    Avoids K separate Python objects; more cache-friendly indexing.
    Identical computation to MixtureOfLookupExperts above.
    """

    def __init__(self, vocab_size: int, num_experts: int, expert_dim: int, model_dim: int):
        super().__init__()
        # Fused expert table: [K, vocab_size, expert_dim]
        self.expert_table = nn.Parameter(
            torch.randn(num_experts, vocab_size, expert_dim) * 0.01
        )
        # Gate table: [vocab_size, K]
        self.gate_table = nn.Parameter(torch.zeros(vocab_size, num_experts))
        # Projection: [expert_dim, model_dim]
        self.proj = nn.Linear(expert_dim, model_dim, bias=False)
        nn.init.zeros_(self.proj.weight)

    def forward(self, input_ids: Tensor) -> Tensor:
        # input_ids: [B, T]

        # Gate: [B, T, K]
        gates = F.softmax(self.gate_table[input_ids].float(), dim=-1)

        # Expert lookup: expert_table[:, input_ids, :] -> [K, B, T, D]
        # Permute to [B, T, K, D]
        expert_outs = self.expert_table[:, input_ids, :].permute(1, 2, 0, 3)

        # Weighted sum: [B, T, D]
        combined = (gates.unsqueeze(-1).to(expert_outs.dtype) * expert_outs).sum(dim=-2)

        return self.proj(combined)
```

---

## 5. Exact Paper Equations (Reconstructed from Paper Text)

Based on the paper description, the MoLE layer computes:

### Equation 1: Expert Layer Output

    y_i = FFN_shared(h_i) + sum_{k=1}^{K} g_{i,k} * f_k(e_i)

Where:
- y_i in R^{model_dim}: output of MoLE layer at token position i
- h_i in R^{model_dim}: hidden state at position i
- e_i in R^{embed_dim}: token embedding of t_i (= E[t_i], from embedding table)
- g_{i,k} in R: routing weight for expert k at token i (scalar)
- f_k: k-th routed expert (FFN), mapping R^{embed_dim} -> R^{model_dim}
- K: number of routed experts

### Equation 2: Routing Weights

    g_i = Softmax(W_r * h_i)       g_i in R^K, W_r in R^{K x model_dim}

Key property: g_i depends on h_i (context-dependent), NOT on token ID alone.

### Equation 3: Routed Expert FFN

    f_k(e_i) = W_{k2} * sigma(W_{k1} * e_i)

Where:
- W_{k1} in R^{d_int x embed_dim}: up-projection for expert k
- W_{k2} in R^{model_dim x d_int}: down-projection for expert k
- sigma: activation function (paper uses SiLU-based gating)
- d_int: intermediate dimension (optimal at 4 * embed_dim per ablations)

### Equation 4: Inference Re-parameterization

Since e_i = E[t_i] (fixed for each token ID), pre-compute:

    LUT_k[v] = f_k(E[v])     for v in {0, ..., V-1}

At inference, replace f_k(e_i) with LUT_k[t_i].

### Equation 5: Shared FFN

    FFN_shared(h_i) = W_{s2} * sigma(W_{s1} * h_i)

Standard FFN, identical to dense transformer FFN. NOT re-parameterizable.

---

## 6. Key Hyperparameters

From the paper's experiments (evaluated at 160M, 410M, 1B activated parameters):

| Hyperparameter         | Paper Values Tested  | Sweet Spot     | Notes                              |
|------------------------|---------------------|----------------|-------------------------------------|
| K (num_routed_experts) | 4, 16               | 16             | More experts = better, diminishing |
| d_int / embed_dim      | 1x, 4x, 16x         | 4x             | Beyond 4x: LUT capacity saturates  |
| Routing               | Dense (all K active) | Dense           | No sparsity — no collapse issues   |
| Auxiliary losses       | Load balance, z-loss | None           | Harms performance                  |
| LR schedule            | Pythia (3e-4)        | Same as base   | No changes needed                  |
| Vocabulary size        | 50k (GPT-NeoX)       | N/A            | LUT size = K * V * model_dim bytes |

For Parameter Golf (vocab_size=1024):
- LUT size = K * 1024 * model_dim = tiny (e.g., 16 * 1024 * 64 * 4 bytes = 4 MB)
- Sweet spot K: 8-16 experts
- Sweet spot expert_dim: 64-256 (equivalent to 4x embed_dim if embed = 64)

---

## 7. Performance vs. Dense Baseline

From the paper's experiments (Pile dataset, 100B tokens, GPT-NeoX tokenizer):

- "Both MoE and MoLE significantly improve performance over the dense baseline"
- "In the comparison of five pairs of MoLE and MoE models with the same number of
  training parameters, MoLE outperforms MoE in FOUR out of five comparisons"
- Inference efficiency: MoLE-16E loads only 1/1500 to 1/2000 of the per-token parameters
  that MoE requires

Quantitative perplexity/BPB improvement: The paper presents results in a table comparing
dense, MoE-10E, MoE-34E, MoLE-4E, and MoLE-16E. MoLE-16E consistently matches or
beats MoE-34E performance at far lower inference cost.

---

## 8. Relationship to L3 (Large Lookup Layers, arXiv:2601.21461)

L3 (January 2026) extends the lookup-layer idea further:
- "Generalizes embedding tables to model decoder layers"
- Uses "static token-based routing to aggregate a set of learned embeddings per token
  in a context-dependent way"
- Positioned as a follow-on to MoLE in the lineage of token-ID-indexed computation

This suggests MoLE's design was influential in the subsequent L3 work.

---

## 9. Critical Analysis: Codebase Implementation vs. Paper

### What the codebase gets right:
1. Multiple learned expert tables indexed by token ID — correct
2. Softmax mixing over experts — correct
3. Token-ID-only indexing for experts — correct
4. Zero-init on gate and projection — good for residual stability
5. torch.compile compatible (no Python control flow per batch element) — correct

### What the codebase diverges from the paper:

**Divergence 1: Gate is token-ID-based, not hidden-state-based**

Paper: g_i = Softmax(W_r * h_i)        — depends on hidden state (context)
Codebase: gates = Softmax(gate_table[token_id])  — depends only on token ID

Impact: The codebase gate cannot differentiate between the same token appearing in different
contexts. The paper's gate CAN do this (same token gets different expert weights depending
on what surrounds it). This is a significant capability difference.

However, consequence: the codebase variant IS fully re-parameterizable (the entire MoLE
output is a fixed function of token ID, algebraically equivalent to a single embedding).

**Divergence 2: Placement at embedding stage, not inside transformer blocks**

Paper: MoLE replaces the FFN inside each transformer block (alongside a shared FFN)
Codebase: Single MoLE module added to residual stream before block 0

Impact: The paper's placement means the MoLE output at layer l informs attention at layers
l+1, l+2, ... The codebase placement only informs layer 0's attention.

From the paper: "the output of the expert layer is part of the input to subsequent attention
layers, allowing the experts to influence the behavior of later attentions."

**Divergence 3: No shared FFN in codebase**

Paper: y = FFN_shared(h) + sum_k( g_k * f_k(e) )
Codebase: y = sum_k( gate_k(id) * expert_k(id) ) then projected to model_dim

The codebase adds a new module; the standard transformer blocks remain unchanged. The paper
integrates MoLE with the existing FFN.

### For Parameter Golf Competition Context:

The codebase's simplified token-only variant is actually well-suited to the competition:
1. It is ADDITIVE to the existing architecture (no block structure changes needed)
2. It is fully re-parameterizable (no inference overhead)
3. At vocab_size=1024, the entire module is only K*1024*D parameters
4. torch.compile works correctly
5. It directly targets the "first-token predictions are hard" problem by giving the model
   explicit per-token feature vectors

The main architectural argument for whether this helps: the transformer already has a
token embedding table. MoLE adds K additional "basis" embeddings per token, mixed with
a learned per-token gate. This gives the residual stream richer starting information
before any attention is computed.

---

## 10. Expected BPB Improvement Estimate

The codebase already has BigramHash. MoLE (token-only variant) replaces or supplements it.

From the novel_2025_2026_survey.md: "-0.015 to -0.030 BPB (replacing BigramHash)"

More precise estimate for the Parameter Golf setting:
- BigramHash contribution: captures bigram statistics (prev_token, cur_token) — encodes
  ~bigram language model in an embedding. This is powerful.
- MoLE (token-only): captures richer UNIGRAM statistics via multi-basis decomposition.
  Cannot capture prev-token context that BigramHash provides.
- Running BOTH: MoLE adds incremental gain over BigramHash by capturing per-token
  distributional patterns beyond what the main embedding table already represents.

Conservative estimate (MoLE on top of BigramHash, same parameter budget):
- If MOLE_NUM_EXPERTS=8, MOLE_DIM=64: ~+0.5M extra parameters
- Expected: -0.005 to -0.015 BPB over BigramHash-only baseline

The larger gains from MoLE would come from implementing the paper-faithful per-layer
version with context-dependent routing, but this requires architectural changes to the
transformer blocks.

---

## 11. Integration Guide for train_gpt.py

### Current Configuration Variables (already in codebase):

    mole_num_experts = int(os.environ.get("MOLE_NUM_EXPERTS", 0))   # 0 = disabled
    mole_dim         = int(os.environ.get("MOLE_DIM", 64))          # expert_dim

### Suggested Experiment Configurations:

    # EXP-A: Baseline MoLE (token-only, small)
    MOLE_NUM_EXPERTS=4 MOLE_DIM=64

    # EXP-B: MoLE at sweet spot (paper K=16)
    MOLE_NUM_EXPERTS=16 MOLE_DIM=64

    # EXP-C: MoLE replacing BigramHash entirely
    BIGRAM_HASH_BUCKETS=0 MOLE_NUM_EXPERTS=16 MOLE_DIM=128

    # EXP-D: MoLE + BigramHash combo
    BIGRAM_HASH_BUCKETS=32768 MOLE_NUM_EXPERTS=8 MOLE_DIM=64

### Parameter Count Analysis (vocab_size=1024, model_dim=512):

    MOLE_NUM_EXPERTS=4,  MOLE_DIM=64:  4 * 1024 * 64 + 1024 * 4 + 64 * 512   =   299K params
    MOLE_NUM_EXPERTS=8,  MOLE_DIM=64:  8 * 1024 * 64 + 1024 * 8 + 64 * 512   =   565K params
    MOLE_NUM_EXPERTS=16, MOLE_DIM=64:  16 * 1024 * 64 + 1024 * 16 + 64 * 512  =  1.09M params
    MOLE_NUM_EXPERTS=16, MOLE_DIM=128: 16 * 1024 * 128 + 1024 * 16 + 128 * 512 = 2.15M params

---

## 12. Potential Improvement to Codebase Implementation

The codebase implementation (lines 1290-1304) is algorithmically correct for the simplified
variant but has one potential improvement for torch.compile efficiency:

**Current:** Uses `nn.ModuleList` with a Python list comprehension inside `forward()`.
The `torch.stack([e(input_ids) for e in self.experts], dim=-2)` creates a Python loop
that torch.compile traces but must unroll at compile time.

**Alternative:** Use a single `nn.Embedding` with shape `[K * vocab_size, expert_dim]`
or `[K, vocab_size, expert_dim]` as a 3D parameter and index with advanced indexing.
This avoids the Python loop entirely and may be more compile-friendly.

```python
# Alternative: fused single-parameter expert table
self.expert_table = nn.Parameter(
    torch.randn(num_experts, vocab_size, expert_dim) * 0.01
)
# In forward:
# expert_outs = self.expert_table[:, input_ids, :]  # [K, B, T, D]
# expert_outs = expert_outs.permute(1, 2, 0, 3)    # [B, T, K, D]
```

However, the current implementation is already torch.compile-safe because torch.compile
traces and unrolls the fixed-size loop. For small K (4-16), this is fine in practice.

---

## 13. Citation

Shibo Jie, Zhi-Hong Deng, Yehui Tang, Kai Han, Yunhe Wang.
"Mixture of Lookup Experts."
International Conference on Machine Learning (ICML), 2025. Oral Presentation.
arXiv:2503.15798 [cs.LG]. https://arxiv.org/abs/2503.15798

---

## 14. References

- arXiv:2503.15798 — https://arxiv.org/abs/2503.15798
- arXiv:2503.15798 HTML — https://arxiv.org/html/2503.15798v1
- GitHub Official Code — https://github.com/JieShibo/MoLE
- OpenReview ICML — https://openreview.net/forum?id=wUEp13rqXP
- ar5iv HTML — https://ar5iv.labs.arxiv.org/html/2503.15798
- alphaXiv — https://www.alphaxiv.org/overview/2503.15798
- Literature Review — https://www.themoonlight.io/en/review/mixture-of-lookup-experts
- L3 (follow-on work) — https://arxiv.org/abs/2601.21461
