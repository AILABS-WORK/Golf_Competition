# Paper Scout: Research Complete

**Date:** 2026-03-23
**Total papers/posts analyzed:** 15
**Search queries executed:** 14

---

## Papers Found (by category)

### Quantization-Aware Training (QAT)
1. **efficientqat.md** -- EfficientQAT: Two-phase QAT (Block-AP + E2E-QP) for efficient LLM quantization. ACL 2025.
2. **qat_scaling_law.md** -- Scaling Law for QAT: 268 experiments showing quantization error scales with model size, tokens, and group size. Key finding: FC2 layer is the activation quantization bottleneck.
3. **lowbit_favors_undertrained.md** -- Low-Bit Quantization Favors Undertrained LLMs: Undertrained models are MORE robust to quantization. ACL 2025. CRITICAL for Parameter Golf strategy.
4. **ste_improvements.md** -- STE Improvements: Plain STE is theoretically equivalent to fancy alternatives. Just tune LR properly.

### Stochastic Weight Averaging + Quantization
5. **swa_flat_minima.md** -- Foundational SWA paper: averaging weights finds flat minima that generalize better and are more robust to perturbation.
6. **sqwa.md** -- SQWA: Quantization-aware SWA with cyclical LR, averaging in quantized domain. Extends SWA for low-precision networks.
7. **swalp.md** -- SWALP: SWA in low-precision training. Matches full-precision SGD with 8-bit training. Theoretical guarantees.

### Optimizer
8. **muon_optimizer.md** -- Muon: Newton-Schulz orthogonalized momentum. 35% speedup over AdamW. NanoGPT speed record holder.
9. **muon_scalable_llm.md** -- Muon Scalability: 2x compute efficiency vs AdamW at scale. Key: add weight decay + tune per-parameter update scale.

### Model Architecture & Features
10. **bigram_subnetworks.md** -- Bigram Subnetworks in Transformers: <0.2% of params in first MLP layer are critical for bigram predictions. Validates SmearGate/BigramHash approach.
11. **mixed_precision_quantization_survey.md** -- Mixed-Precision Quantization Survey: Validates int5 MLP / int6 attention / FP16 embeddings allocation. Recommends per-group quantization and outlier smoothing.

### Scaling Laws & Training Strategy
12. **scaling_laws_neural_lm.md** -- Kaplan et al. Scaling Laws: Loss ~ N^(-0.076). Maximize model size within constraints.
13. **minicpm_wsd.md** -- MiniCPM / WSD Scheduler: Warmup-Stable-Decay LR schedule. Dramatic loss drop during decay phase. Small models may need more data than Chinchilla predicts.

### Compression
14. **zipnn_neural_weight_compression.md** -- ZipNN + NWC: Lossless compression tailored to neural weights. Zstd vs Huffman auto-selection. 62%+ savings on Llama 3.

### Training Optimization Compendium
15. **nanogpt_speedrun_techniques.md** -- NanoGPT Speedrun techniques: Muon, trapezoidal LR, ReLU-squared, QK-Norm, logit softcap, RoPE, test-time training.

---

## Top Actionable Insights (ranked by expected impact)

### Tier 1: High-Impact, Low-Effort
1. **WSD Learning Rate Schedule** (minicpm_wsd.md): Run high constant LR for ~80% of training, decay for final ~20%. Essentially free BPB improvement.
2. **SWA during high-LR phase** (swa_flat_minima.md, swalp.md): Collect SWA snapshots during the stable/high-LR phase, NOT after decay. This maximizes flat-minimum benefit.
3. **Maximize model size** (scaling_laws_neural_lm.md, lowbit_favors_undertrained.md): Larger undertrained model > smaller well-trained model, especially when quantized.
4. **Per-group quantization with small groups** (qat_scaling_law.md, mixed_precision_quantization_survey.md): Use group size 32-128 for quantization. Reduces error significantly.

### Tier 2: Medium-Impact, Medium-Effort
5. **SQWA extension** (sqwa.md): Make SWA quantization-aware by averaging in quantized domain with cyclical LR.
6. **FC2 layer mixed precision** (qat_scaling_law.md): Give the second MLP linear layer higher bit-width to address activation outlier bottleneck.
7. **Muon weight decay + scale tuning** (muon_scalable_llm.md): Ensure Muon has weight decay enabled and per-parameter update scale is tuned.
8. **Test-time training / parameter nudging** (nanogpt_speedrun_techniques.md): Adapt model parameters on early tokens of eval documents. Potentially large BPB gain if allowed by rules.

### Tier 3: Worth Investigating
9. **ReLU-squared activation** (nanogpt_speedrun_techniques.md): May outperform GELU for small models.
10. **Per-block compression selection** (zipnn_neural_weight_compression.md): Test zstd vs Huffman per weight block for optimal compression.
11. **First-layer precision** (bigram_subnetworks.md): Consider higher precision for the first MLP layer given its critical bigram role.
12. **Sparsity regularization** (zipnn_neural_weight_compression.md): Encourage zeros in weights to improve compression ratio.

---

## Key Tensions / Open Questions
- **Model size vs training tokens:** Kaplan scaling says bigger model; MiniCPM says small models need more data. The optimum depends on the specific 16MB/10min constraints and must be found experimentally.
- **SWA schedule:** Should SWA collection happen during stable phase (high LR) or with cyclical LR? SQWA suggests cyclical may be better.
- **Test-time training legality:** Parameter nudging at eval time could be powerful but needs rule verification.
- **Compression budget allocation:** Every byte saved on compression = more parameters. The interplay between quantization bit-width, compression ratio, and model quality is a 3-way optimization.

---

## Sources

- [EfficientQAT](https://arxiv.org/abs/2407.11062) -- ACL 2025
- [Scaling Law for QAT](https://arxiv.org/abs/2505.14302) -- 2025
- [Low-Bit Quantization Favors Undertrained LLMs](https://arxiv.org/abs/2411.17691) -- ACL 2025
- [SQWA](https://arxiv.org/abs/2002.00343) -- IEEE 2021
- [SWALP](https://arxiv.org/abs/1904.11943) -- ICML 2019
- [Muon Optimizer](https://kellerjordan.github.io/posts/muon/) -- Keller Jordan, 2024
- [Muon is Scalable for LLM Training](https://arxiv.org/abs/2502.16982) -- Moonshot AI, 2025
- [Mixed-Precision Quantization Survey](https://arxiv.org/abs/2510.16805) -- 2025
- [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) -- OpenAI, 2020
- [MiniCPM / WSD Scheduler](https://arxiv.org/abs/2404.06395) -- Tsinghua, 2024
- [Bigram Subnetworks](https://arxiv.org/abs/2504.15471) -- 2025
- [NanoGPT Speedrun](https://github.com/KellerJordan/modded-nanogpt) -- 2024-2026
- [STE Equivalence](https://openreview.net/forum?id=3j72egd8q1) -- 2024
- [ZipNN](https://arxiv.org/abs/2411.05239) -- 2024
- [SWA Flat Minima](https://arxiv.org/abs/1803.05407) -- 2018
