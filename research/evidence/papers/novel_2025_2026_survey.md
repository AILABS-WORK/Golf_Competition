# Novel Papers Survey — Parameter Golf 2026
# Generated: 2026-03-23 via arXiv/OpenReview search
# Databases: arXiv, NeurIPS 2024/2025, ICLR 2025/2026, ICML 2025

## TIER 1: HIGH IMPACT — Novel, not in any competition submission

### MUDDFormer (ICML 2025)
- arXiv: 2502.12170
- Dynamic dense connections across ALL previous layers for Q, K, V, residual streams
- 1.8-2.4x effective compute multiplier — in 10-min budget this is massive
- Only +0.23% parameters, +0.4% compute overhead
- Code: github.com/Caiyun-AI/MUDDFormer
- Expected: -0.020 to -0.040 BPB over merged SOTA
- Status: NOT in any competition submission

### Mixture of Lookup Experts / MoLE (ICML 2025 ORAL)
- arXiv: 2503.15798
- Replaces MoE experts with precomputed lookup tables indexed by token ID
- At vocab=1024: each expert LUT has only 1024 entries = tiny storage
- ZERO compute at inference (table lookup, not matmul)
- Generalizes and replaces BigramHash — multiple learned experts vs one hash table
- Expected: -0.015 to -0.030 BPB (replacing BigramHash)
- Status: NOT in any competition submission

### LOTION (arXiv:2510.08757)
- Replaces STE with differentiable optimization of quantized loss expectation
- Provably convergent to local minimum (STE has no convergence guarantee)
- Validated on 150M and 300M LMs, outperforms standard QAT
- Drop-in replacement for STE
- Expected: -0.005 to -0.015 BPB
- Status: NOT tried

### Compute-Optimal QAT (arXiv:2509.22935)
- Scaling laws for optimal FP-to-QAT compute allocation ratio
- Derives exact formula for competition setup (200M params, int5/int6, 10 min)
- Cooldown+QAT fusion: LR decay happens jointly with QAT activation
- Expected: -0.005 to -0.010 BPB from timing optimization
- Status: NOT tried

### NuMuon (arXiv:2603.03597) — March 2026
- Nuclear-norm constrained Muon optimizer
- Pushes weights toward low-rank structure during training
- Low-rank weights compress better with zstd-22 and int quantization
- Single-line modification to existing Muon Newton-Schulz step
- Expected: -0.003 to -0.008 BPB
- Status: NOT tried

### QuEST (arXiv:2502.05003)
- Gaussian grid fitting for quantization instead of uniform grid
- Matches grid to actual weight distribution (sub-Gaussian with heavy tails)
- Pareto-optimal frontier moves to ~4-bit at same quality
- Expected: -0.003 to -0.008 BPB
- Status: NOT tried

### TWEO / Anti-Outlier Training (ICLR 2026)
- arXiv: 2511.23225
- Colinearity penalty eliminates extreme activation outliers (10000+ → <20)
- Enables per-tensor static quantization (previously impossible due to outliers)
- Eliminates per-row scale storage overhead in compressed artifact
- Expected: -0.002 to -0.005 BPB
- Status: NOT tried

### Frac-Connections (arXiv:2503.14125)
- Fractional extension of Hyper-Connections
- Learnable inter-depth connectivity without width expansion
- Complementary to MUDD (apply to residual stream while MUDD handles KV)
- Expected: -0.005 to -0.015 BPB
- Status: NOT tried

---

## TIER 2: HIGH-MED IMPACT

### TTT-E2E (arXiv:2512.23675) — MIT/Stanford
- Meta-learning during training makes model "learn to learn" at test time
- More effective adaptation per gradient step at eval vs standard TTT
- Expected: -0.010 to -0.020 BPB incremental over cosine TTT
- Status: Cosine TTT is in unmerged frontier; TTT-E2E NOT tried

### Q-only TTT (arXiv:2512.13898) — MIT/Berkeley
- Update ONLY query (Q) projection matrices at test time
- Dramatically cheaper than full-model TTT
- More gradient steps possible within same eval time budget
- Expected: -0.010 to -0.020 BPB (cheaper alternative to cosine TTT)
- Status: NOT tried

### ResFormer / Value Residual (arXiv:2410.17897)
- First-layer V projection residual to all subsequent layers
- 16.11% fewer params for equivalent quality (= room for bigger model)
- IN unmerged PRs #486 (1.0887) and #490 (1.0891)
- Status: In unmerged only. Integrate into merged SOTA stack.

### DeepCrossAttention / DCA (arXiv:2502.06785) — Google
- Dynamic combination of ALL previous layer outputs (vs ResFormer's first-layer only)
- "3x training speedup equivalent" at negligible parameter overhead
- Expected: -0.010 to -0.020 BPB
- Status: NOT tried

### WSM: Checkpoint Merging Instead of LR Decay (arXiv:2507.17634)
- Maintain constant LR through stable phase, merge checkpoint window at end
- Outperforms WSD: +3.5% MATH, +2.9% HumanEval, +5.5% MMLU-Pro
- Expected: -0.003 to -0.005 BPB (replace warmdown with checkpoint merge)
- Status: NOT tried

### DNQ + SWA (arXiv:2511.01462)
- Differential Noise Injection: add calibrated noise matching quantization step size
- +1.39% over baseline vs +0.75% for SWA alone (+0.64% incremental over SWA)
- Expected: -0.002 to -0.005 BPB incremental over SWA
- Status: NOT tried

---

## TIER 3: MED-LOW IMPACT

### EfficientQAT E2E-QP (arXiv:2407.11062)
- Optimize only scale+zero-point globally after per-layer convergence
- Cheap addition to existing QAT training loop
- Expected: -0.002 to -0.005 BPB

### Neural Weight Compression NWC (arXiv:2510.11234) — Google/DeepMind
- Neural codec trained on transformer weight distributions
- Outperforms zstd-22 for neural weights specifically
- Challenge: codec itself must be external or tiny
- Expected: -0.002 to -0.005 BPB

### Bigram Subnetworks (arXiv:2504.15471)
- <0.2% of first-MLP params are critical bigram subnetwork
- Justifies int8 for first layer, int5 elsewhere (mixed precision)
- Expected: -0.001 to -0.003 BPB

---

## Strategic Path to val_bpb < 1.08

| Step | Technique | Cumul. Expected BPB | Effort |
|------|-----------|---------------------|--------|
| Merged SOTA | — | 1.1428 | — |
| +ValueResidual into merged stack | ResFormer | ~1.128 | MED |
| +MUDD over static V-residual | MUDDFormer | ~1.115 | MED |
| +MoLE replacing BigramHash | MoLE | ~1.095 | MED |
| +LOTION replacing STE | LOTION | ~1.085 | MED |
| +QuEST grid + Optimal QAT timing | QuEST+DremovScaling | ~1.080 | LOW |
| +TWEO + NuMuon | TWEO+NuMuon | ~1.075 | LOW |
| +TTT-E2E meta-learning | TTT-E2E | ~1.060 | HIGH |

**Target <1.08 achievable without TTT-E2E using Steps 1-5 only.**
