# Parameter Golf Frontier Map

Generated: 2026-03-23
Sources: 17 merged submissions, 5 unmerged PRs, 15 academic papers, community discussions

---

## Leaderboard Summary (Merged)

| Rank | Entry | val_bpb | Key Innovations | Category |
|------|-------|---------|-----------------|----------|
| 1 | 10L Int5-MLP + BigramHash(10240) + SWA(0.4) | 1.1428 | Int5/Int6 mixed, BigramHash, SWA, SmearGate, OrthoInit | MODEL+QUANT+TRAIN |
| 2 | Int6 MLP3x + SmearGate + BigramHash + SWA | 1.1458 | Int6, MLP3x, SmearGate, BigramHash, SWA, WD=0.04 | MODEL+QUANT+TRAIN |
| 3 | 11L MLP3x + Int6 QAT + zstd-22 | 1.1502 | 11L, QAT, zstd-22, sliding eval | MODEL+QUANT+COMPRESS |
| 4 | SmearGate + OrthoInit + Int6 STE QAT + MLP3x | 1.1556 | SmearGate, OrthoInit, STE QAT, BigramHash | MODEL+QUANT |
| 5 | Int6 MLP3x Sliding Window (WarmdownQuant) | 1.1574 | Late-K passthrough, Train@2048 | MODEL+QUANT+EVAL |
| 6 | 10L Int6 QAT + Zstd MLP2.6x | 1.1586 | 10L, STE QAT, zero quant gap | QUANT+MODEL |
| 7 | Mixed Quant (int6+int8) + Sliding | 1.1630 | Mixed quant, STE, MLP3x | QUANT |
| 8 | SW + FP16 + 10L + MuonWD + OvertoneInit | 1.1748 | Overtone init, 10L | MODEL+TRAIN |
| 9 | Sliding Window Eval (stride=64) | 1.1925 | Pure eval improvement | EVAL |
| 10 | LoRA TTT | 1.1929 | Test-time training (LoRA rank-8) | EVAL+TTT |
| ... | Baseline | 1.2244 | 9L 512dim SP-1024 int8+zlib | BASELINE |

## UNMERGED FRONTIER (Open PRs — THE REAL COMPETITION)

| PR | val_bpb | Key New Techniques | Delta vs Merged SOTA |
|----|---------|--------------------|--------------------|
| ~486 | **1.0887** | TrigramHash + ValueResidual + GradQuant + Cosine TTT | **-0.0541** |
| ~490 | **1.0891** | ValueResidual + GatedAttn + AdamW TTT | **-0.0537** |
| ~503 | **1.1218** | XSA + Score-First AdamW TTT | **-0.0210** |
| ~493 | **1.1309** | EMA + XSA + LeakyReLU² + Partial RoPE | **-0.0119** |

**The true frontier is ~1.089 val_bpb, not 1.143.** TTT is the dominant technique driving this gap.

---

## Technique Frequency (Top 10 Merged)

| Technique | Count | Avg Rank | Category |
|-----------|-------|----------|----------|
| Sliding Window Eval (stride=64) | 9/10 | 4.9 | EVAL |
| Int6 Quantization (PTQ or QAT) | 8/10 | 4.1 | QUANT |
| FP16 Tied Embedding | 7/10 | 4.0 | QUANT |
| MLP 3x Expansion | 7/10 | 3.6 | MODEL |
| zstd-22 Compression | 6/10 | 3.5 | COMPRESS |
| Muon WD (0.01-0.04) | 6/10 | 3.3 | TRAIN |
| 10+ Layers | 6/10 | 3.8 | MODEL |
| BigramHash | 4/10 | 2.0 | MODEL |
| SmearGate | 4/10 | 2.0 | MODEL |
| Orthogonal Init | 4/10 | 2.0 | MODEL |
| SWA | 3/10 | 1.7 | TRAIN |
| STE QAT | 4/10 | 3.8 | QUANT |

---

## Technique Interactions

### Confirmed Synergies
- **SWA + aggressive quantization**: SWA smooths weights → flat minima → better post-quant quality (paper: swa_flat_minima.md)
- **Int6 quant + MLP 3x**: Int6 saves bytes → fund MLP expansion → biggest single combo impact (-0.034 BPB)
- **Muon WD + quantization**: WD regularizes weight magnitudes → tighter distribution → lower quant error
- **SmearGate + BigramHash**: Complementary token-pair features (gate + hash embedding)
- **OrthoInit + Muon**: Orthogonal start + orthogonalized updates → stable early training
- **TTT + any model improvement**: TTT is orthogonal to model/quant — stacks on top (paper: nanogpt_speedrun)

### Likely Conflicts
- **Extra layers + larger MLP**: Both consume bytes → may exceed 16MB
- **Int5 everywhere + high QAT quality**: Very few quantization levels → hard for QAT to eliminate gap
- **Multiple TTT methods simultaneously**: Eval-time compute budget may be a practical limit

---

## Paper Insights

| Paper | Key Takeaway | Relevance |
|-------|-------------|-----------|
| lowbit_favors_undertrained | Undertrained models are MORE robust to quantization | Maximize model size, don't overtrain |
| qat_scaling_law | FC2 (second MLP linear) is the activation quant bottleneck | Give FC2 higher precision |
| minicpm_wsd | WSD LR schedule: high constant → decay | May outperform current warmdown |
| sqwa | SWA in quantized domain with cyclical LR | Extend SWA for better quant |
| swalp | SWA works in low-precision training, matches FP32 | Validates SWA approach |
| muon_scalable_llm | Muon 2x more compute-efficient than AdamW with WD | Confirms WD is critical |
| bigram_subnetworks | <0.2% of first-MLP params critical for bigram | Validates SmearGate/BigramHash |
| zipnn | Per-block compression selection | Test zstd vs alternatives per block |
| nanogpt_speedrun | TTT provides significant eval-time improvement | Confirms TTT is highest-leverage |

---

## Unexplored Directions (from papers + analysis)

1. **Test-Time Training (TTT)** — The #1 opportunity. Unmerged PRs show 0.03-0.05 BPB improvement. Must implement.
2. **WSD Learning Rate Schedule** — MiniCPM paper shows dramatic loss drop during decay. Free improvement.
3. **Per-group quantization (group_size=32-128)** — Mixed precision survey recommends small groups for lower quant error.
4. **SQWA (quantization-aware SWA)** — Averaging in quantized domain, not FP32. Could beat standard SWA.
5. **Value Residual connections** — Present in top unmerged PRs. Novel arch change.
6. **TrigramHash** — Extension of BigramHash to 3-token context. In best unmerged PR.
7. **XSA (Cross-Selective Attention)** — Novel attention variant in 3 unmerged PRs.
8. **Partial RoPE** — Only apply positional encoding to some heads/dimensions.
9. **LeakyReLU²** — Alternative to ReLU² activation in MLP.
10. **FC2-aware mixed precision** — Give FC2 layer higher precision (quant bottleneck per QAT scaling law paper).

---

## Strategic Priority

**To beat the unmerged frontier (~1.089):**
1. Start from thwu1/raahilshah foundation (all merged techniques)
2. Add TTT (highest-leverage single addition: ~0.03-0.05 BPB)
3. Add ValueResidual + TrigramHash (from best unmerged PR)
4. Add STE QAT (eliminate quant gap, missing from #1)
5. Try WSD schedule (paper-backed, free improvement)

**To beat merged SOTA (1.1428):**
Any single addition from the unexplored list should suffice. The gap between merged and unmerged suggests massive low-hanging fruit.
