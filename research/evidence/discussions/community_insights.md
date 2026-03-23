# Parameter Golf Community Insights
**Collected: 2026-03-23**

---

## Source Summary

Insights gathered from:
- GitHub repository (openai/parameter-golf) -- merged records and open PRs
- Web searches across news sites, forums, social media
- DeepWiki analysis pages
- Official OpenAI announcement

---

## 1. Official Guidance (from OpenAI)

OpenAI explicitly listed these technique categories as expected directions:
- **Unique architectures**: test-time compute, aggressive parameter tying, depth recurrence, low-rank training
- **Compression schemes**: low precision, QAT, bitnets, novel tokenizers
- **Creative approaches**: test-time training, long context, megakernels

Source: [OpenAI Parameter Golf announcement](https://openai.com/index/parameter-golf/)

Key constraint insight: "The binding constraint is really the 16MB artifact, not the 10-minute clock." The training budget is generous relative to model size, so the bottleneck is fitting quality into 16MB.

---

## 2. Community Technique Evolution (from GitHub PRs)

### Phase 1: Days 1-2 (March 18-19) -- Low-hanging fruit
- LR tuning, warmdown tuning, FP16 embeddings
- Sliding window eval (stride=64) -- discovered as "free" 0.013 BPB
- Long context training (seq2048, seq4096)
- Basic int8 quantization with zlib

### Phase 2: Days 2-3 (March 19-20) -- Compression revolution
- Int6 quantization (unlocking MLP 3x expansion)
- STE QAT (eliminating quantization gap)
- zstd-22 replacing zlib (saving ~1.5MB)
- SmearGate + BigramHash + OrthoInit cluster emerges
- U-Net skip connections
- Muon Weight Decay + SWA

### Phase 3: Days 4-6 (March 21-23) -- Eval-time innovation
- Test-Time Training (TTT) with LoRA/AdamW
- Value Residual connections
- XSA (Cross-Selective Attention)
- TrigramHash (extending BigramHash to 3-grams)
- EMA model weights
- Partial RoPE
- LeakyReLU^2 activation

### Emerging Phase 4 (expected):
- Combined TTT + compression stacking
- Multi-strategy TTT (different TTT methods per document type)
- Adaptive eval strategies

---

## 3. Key Community Findings

### 3A. Quantization Gap is the Hidden Bottleneck

From samuellarson's Warmdown-Quantization entry (README):
> "On 8xH100, the dominant bottleneck isn't model quality -- it's quantization quality. The post-training int8 quantization penalty (0.014 BPB with default settings) is larger than most hyperparameter improvements combined."

Key numbers:
- Default int8 PTQ penalty: 0.014 BPB
- With warmdown=20000: 0.005 BPB
- With FP16 embed: 0.001 BPB
- With STE QAT: 0.000 BPB (complete elimination)

### 3B. SmearGate Design Rationale

From unnir's entry (PR #162, the foundation for top entries):
> "A learned per-dimension gate (~512 params) that blends each token's embedding with the previous token's embedding before the transformer processes anything. Normally a transformer must discover token-pair relationships through self-attention; SmearGate provides this signal for free."

Gate initialization: sigmoid(3.0) = 0.95 (starts near-identity).
Cost: ~512 parameters. Impact: significant when combined with BigramHash.

### 3C. BigramHash Scaling

From thwu1's entry (#1):
- BigramHash(4096): baseline
- BigramHash(8192): -0.0012 BPB
- BigramHash(10240): -0.0008 BPB additional
- Total from scaling: -0.0020 BPB

There appears to be diminishing returns beyond 10240 buckets, but TrigramHash (in unmerged PRs) suggests the next step is extending context length rather than bucket count.

### 3D. SWA Tuning Matters

From thwu1's entry:
- SWA start_frac=0.5 (default): good baseline
- SWA start_frac=0.4: -0.0006 BPB improvement
- "Quality over quantity: fewer but better-converged checkpoints"
- Optimal SWA every=50 steps (swept from 25-200)

### 3E. Int5 vs Int6 Trade-off

From thwu1's ablation:
- Int5 MLP saves ~1.86MB vs uniform int6
- This funds a 10th transformer layer
- Net effect: -0.0032 BPB
- Int5 has 31 levels (vs 63 for int6), so precision loss is real
- The trick: apply int5 ONLY to MLP weights (most compressible, 1.88x zstd ratio)
- Keep int6 for attention weights (precision-sensitive, 1.51x zstd ratio)

### 3F. Weight Decay Enables Better Quantization

From raahilshah's entry (#2):
> "Weight decay regularizes magnitudes, directly improving int6 quantization quality."
- Muon WD=0.04 was optimal (swept 0.01-0.05)
- "Tighter weight distributions quantize into fewer int6 buckets with less error and compress better with zstd"

### 3G. Orthogonal Init + Muon Synergy

From unnir's entry:
> "Orthogonal matrices have all singular values equal to 1, meaning gradients flow uniformly through the network at initialization. Since Muon's Newton-Schulz step orthogonalizes updates, starting from an already-orthogonal matrix means early updates are immediately useful rather than spent correcting a random initialization."

This is particularly important with only ~12k training steps.

---

## 4. Test-Time Training (TTT) Deep Dive

TTT is the most impactful new technique class. Key variants observed:

### 4A. LoRA TTT (from samacqua, merged entry)
- Rank-8 LoRA on lm_head/Q/V projections
- Adam lr=0.01
- Overlapping 256-token chunks in 1024-token context windows
- Result: 1.1929 BPB (modest improvement, not combined with other techniques)

### 4B. AdamW TTT (from unmerged PRs)
- Full AdamW optimization at test time
- Applied to Value Residual + Gated Attention models
- Result: ~1.089 BPB (massive improvement)

### 4C. Cosine TTT (from unmerged PR #486)
- Cosine-annealed learning rate during test-time adaptation
- Combined with TrigramHash + GradQuant
- Result: 1.0887 BPB (best known)

### 4D. Score-First AdamW TTT (from unmerged PR #503)
- Score-first variant of AdamW for TTT
- Combined with XSA attention
- Result: 1.1218 BPB

**TTT estimated impact: 0.03-0.05 BPB** depending on the base model and TTT variant.

---

## 5. Discord and Community Channels

- OpenAI Discord has dedicated channels: #parameter-golf-discussions, #parameter-golf-announcements
- Active collaboration between competitors (e.g., thwu1 built on unnir's PR #162)
- Community leaderboard monitor tool created (Issue #158)
- External leaderboard tracker: https://parameter-golf.github.io/

---

## 6. Competition Meta-Strategy Insights

### 6A. Build on Existing PRs
The most successful entrants build on merged work rather than starting from scratch:
- thwu1 (#1) explicitly credits unnir's PR #162
- raahilshah (#2) independently developed similar techniques
- The ablation-based approach (try one thing at a time) dominates

### 6B. Compression Budget Allocation
The 16MB budget is best allocated as:
- ~15.8-15.9 MB for model weights
- ~50-56 KB for code
- The winning formula: aggressive quantization (int5/int6) frees bytes for more model capacity

### 6C. Eval-Time Compute is Unconstrained
The competition rules constrain training time (10 min on 8xH100) and artifact size (16MB), but eval-time compute has a generous budget. TTT exploits this by doing per-document adaptation during evaluation.

### 6D. Diminishing Returns in Compression
Each successive compression technique yields less improvement:
- Int8 -> Int6: massive (0.03+ BPB via more capacity)
- Int6 -> Int5 (MLP only): moderate (0.003 BPB)
- BigramHash 4K -> 10K: small (0.002 BPB)
- SWA tuning: small (0.0006 BPB)

The next big gains are from fundamentally different approaches (TTT, new architectures) rather than incremental compression.

---

## 7. Recommended Next Steps for Competitive Entry

Based on community insights, a competitive entry should:

1. **Start from the thwu1 / raahilshah foundation** (SmearGate + BigramHash + OrthoInit + Int6 + MLP3x + SWA + Muon WD + zstd-22)
2. **Add STE QAT** to eliminate quantization gap
3. **Implement TTT** (AdamW or Cosine variant) for eval-time adaptation
4. **Explore Value Residual + Gated Attention** from the unmerged frontier
5. **Try TrigramHash** as an extension of BigramHash
6. **Consider XSA and Partial RoPE** as attention improvements
7. **Target val_bpb below 1.09** to be competitive with the current unmerged frontier

Sources:
- [GitHub - openai/parameter-golf](https://github.com/openai/parameter-golf)
- [OpenAI Model Craft: Parameter Golf](https://openai.com/index/parameter-golf/)
- [The Decoder - Parameter Golf analysis](https://the-decoder.com/openai-turns-model-compression-into-a-talent-hunt-with-its-16-mb-parameter-golf-challenge/)
- [DeepWiki - parameter-golf](https://deepwiki.com/openai/parameter-golf)
- [Parameter Golf Leaderboard](https://parameter-golf.github.io/)
- [GitHub Issue #158 - Community Leaderboard Monitor](https://github.com/openai/parameter-golf/issues/158)
- [GitHub PR #342 - SmearGate + BigramHash discussion](https://github.com/openai/parameter-golf/pull/342)
