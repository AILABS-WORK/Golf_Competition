# Parameter Golf Technique Taxonomy

Generated: 2026-03-23
Source: frontier_map.md + all evidence files

---

## Category: MODEL (Architecture Changes)

| Technique | Impact (BPB) | Byte Cost | Present In | Notes |
|-----------|-------------|-----------|------------|-------|
| MLP 3x Expansion (1536) | -0.015~0.020 | +50% MLP bytes | 7/10 top | Largest single arch contributor |
| 10+ Layers | -0.008~0.012 | ~1.5MB per layer | 6/10 top | Funded by int5/int6 savings |
| U-Net Skip Connections | -0.003~0.005 | ~0 (reuse) | 5/10 top | Encoder stores, decoder reuses |
| SmearGate | -0.004~0.006 | ~512 params | 4/10 top | Learned prev-token blend gate |
| BigramHash(10240) | -0.006~0.008 | ~1.3MB | 4/10 top | Token-pair hash embedding |
| TrigramHash | -0.002~0.004 | ~1.5MB | Unmerged only | Extension to 3-token context |
| Value Residual | -0.005~0.010 | Minimal | Unmerged only | In top 2 unmerged PRs |
| XSA (Cross-Selective Attn) | -0.005~0.010 | Minimal | 3 unmerged PRs | Novel attention variant |
| Partial RoPE | -0.002~0.004 | None | Unmerged | Apply RoPE to subset of heads |
| GQA (8H/4KV) | Baseline | Saves KV bytes | Baseline | Already in baseline |
| Tied Embeddings | Baseline | Saves ~1MB | Baseline | Already in baseline |
| LeakyReLU^2 | Unknown | None | 1 unmerged PR | Alternative to ReLU^2 |
| Logit Softcap (30) | Baseline | None | Baseline | Already in baseline |

## Category: QUANT (Quantization)

| Technique | Impact (BPB) | Byte Savings | Present In | Notes |
|-----------|-------------|-------------|------------|-------|
| Int6 PTQ (uniform) | -0.020~0.030 | ~25% vs int8 | 8/10 top | Enables MLP expansion |
| Int5 MLP / Int6 Attn | -0.003 additional | ~1.86MB vs uniform int6 | 1/10 (SOTA) | MLP more compressible |
| STE QAT | -0.007~0.014 | None | 4/10 top | Eliminates quant gap entirely |
| FP16 Tied Embedding | -0.001~0.003 | Small cost | 7/10 top | Prevents output head degradation |
| Per-group Quant (g=32-128) | Unknown | None | Not tried | Paper-backed, reduces quant error |
| FC2 Mixed Precision | Unknown | Small cost | Not tried | FC2 is quant bottleneck (paper) |
| GPTQ-lite | Unknown | Unknown | 1 unmerged PR | Alternative PTQ approach |
| GradQuant | Unknown | None | 1 unmerged PR | In best unmerged PR (#486) |

## Category: COMPRESS (Weight Compression)

| Technique | Impact (BPB) | Byte Savings | Present In | Notes |
|-----------|-------------|-------------|------------|-------|
| zstd-22 (over zlib-9) | 0 direct | ~5% = ~0.8MB | 6/10 top | More bytes for model |
| Per-block compression | Unknown | Potential | Not tried | ZipNN paper suggests auto-select |
| Sparsity regularization | Unknown | Potential | Not tried | Encourage zeros for compression |

## Category: TRAIN (Training Strategy)

| Technique | Impact (BPB) | Cost | Present In | Notes |
|-----------|-------------|------|------------|-------|
| Muon Optimizer | Baseline | None | Baseline | Newton-Schulz orthogonalized momentum |
| Muon WD=0.04 | -0.005~0.008 | None | 6/10 top | Tighter weight dist = better quant |
| SWA (start=0.4, every=50) | -0.003~0.006 | Minimal compute | 3/10 top | Flat minima for quant robustness |
| Orthogonal Init | -0.002~0.004 | None | 4/10 top | Synergy with Muon |
| WSD LR Schedule | Unknown | None | Not tried | MiniCPM paper: dramatic loss drop |
| Muon momentum=0.99 | Baseline | None | Baseline | With warmup |
| seq_len=2048 | -0.003~0.005 | None | Most top | Longer context during training |
| grad_clip=0.3 | Baseline | None | Baseline | Already in top entries |
| EMA(0.997) | Unknown | Minimal | 1 unmerged PR | May outperform SWA |
| SQWA (quant-aware SWA) | Unknown | Minimal | Not tried | Paper-backed extension of SWA |

## Category: EVAL (Evaluation Strategy)

| Technique | Impact (BPB) | Cost | Present In | Notes |
|-----------|-------------|------|------------|-------|
| Sliding Window (stride=64) | -0.013~0.032 | Eval compute | 9/10 top | Essentially free with 10min eval budget |
| Train@2048 / Eval@longer | -0.002~0.005 | None | Some top | Leverage longer context at eval |

## Category: TTT (Test-Time Training)

| Technique | Impact (BPB) | Cost | Present In | Notes |
|-----------|-------------|------|------------|-------|
| LoRA TTT (rank-8) | -0.032 | Eval compute | 1 merged | Modest; not combined with other techs |
| AdamW Full TTT | -0.040~0.054 | Eval compute | 2 unmerged | Massive improvement |
| Cosine TTT | -0.054 | Eval compute | 1 unmerged (#486) | Best known variant |
| Score-First AdamW TTT | -0.021 | Eval compute | 1 unmerged (#503) | Combined with XSA |
| GradQuant TTT | Unknown | Eval compute | 1 unmerged (#486) | Gradient quantization during TTT |

---

## Priority Stack (for new entry)

### Must-Have (in every competitive entry)
1. Sliding Window Eval (stride=64)
2. Int6 Quantization (or mixed int5/int6)
3. FP16 Tied Embedding
4. MLP 3x Expansion
5. zstd-22 Compression
6. Muon WD=0.04
7. STE QAT
8. SmearGate + BigramHash
9. Orthogonal Init
10. 10+ Layers
11. SWA (start=0.4)

### High Priority (proven in unmerged frontier)
12. **TTT (Cosine or AdamW variant)** — Largest single improvement opportunity
13. Value Residual connections
14. TrigramHash

### Medium Priority (paper-backed, untested in competition)
15. WSD LR Schedule
16. Per-group quantization (g=32-128)
17. FC2 mixed precision
18. SQWA
19. EMA vs SWA comparison

### Low Priority (speculative)
20. XSA attention
21. Partial RoPE
22. LeakyReLU^2
23. Per-block compression selection
24. Sparsity regularization

---

## Technique Dependency Graph

```
Baseline (1.2244)
├── Sliding Window Eval (-0.032) → 1.1925
├── Int6 Quant (-0.030) → enables MLP 3x
│   ├── MLP 3x (-0.020) → 1.163
│   ├── +1 Layer (-0.010) → funded by int5 MLP
│   └── STE QAT (-0.014) → eliminates quant gap
├── FP16 Embed (-0.003)
├── Muon WD=0.04 (-0.008) → improves quant quality
├── SmearGate + BigramHash (-0.010)
├── OrthoInit (-0.004) → synergy with Muon
├── SWA (-0.006) → synergy with quant
└── TTT (-0.030~0.054) → ORTHOGONAL to all above
```

**Key insight:** TTT stacks on top of ALL other improvements because it operates at eval time, not training time. This means the expected BPB of a fully-stacked entry with TTT is approximately:

- Merged SOTA (1.1428) - TTT (~0.040) = ~1.103 BPB
- With additional novel techniques: potentially sub-1.09 BPB
