---
title: GitHub Scout Completion Report
date: 2026-03-23
status: COMPLETE
---

# GitHub Submission Scraping Complete

## Scope
- Repository: openai/parameter-golf
- Track: records/track_10min_16mb/
- Total submission folders found: 17
- Evidence files written: 16 (1 incomplete -- int6_STE QAT folder has no README/submission.json)
- PR landscape file: 1 (covering 376+ open PRs and merged records)

## Evidence Files Created

### Leaderboard Submissions (ranked by val_bpb)

| Rank | File | val_bpb | Author | Key Innovation |
|------|------|---------|--------|----------------|
| 1 | 2026-03-20_10L_Int5MLP_MuonWD04_SWA50.md | 1.1428 | thwu1 | Mixed int5/int6, BigramHash(10240), SWA(0.4) |
| 2 | 2026-03-20_Int6_MLP3x_SmearGate_BigramHash_MuonWD_SWA.md | 1.1458 | Raahil Shah | SmearGate+BigramHash+SWA+WD=0.04 |
| 3 | 2026-03-19_MLP3x_QAT_Int6_SlidingWindow.md | 1.1502 | aruniyer | 11L MLP3x, int6 QAT, zstd-22, U-Net |
| 4 | 2026-03-19_smeargate_orthoinit_muonwd.md | 1.1556 | aquariouseworkman | SmearGate, OrthoInit, BigramHash intro |
| 5 | 2026-03-19_Seq2048_FP16Emb_TunedLR.md | 1.1586 | yahya010 | Full int6 QAT (zero quant gap), zstd |
| 6 | 2026-03-19_MixedQuant_Int6Int8_SlidingWindow.md | 1.1630 | aquariouseworkman | STE QAT, mixed int6/int8, MLP3x |
| 7 | 2026-03-19_SlidingWindow_FP16Emb_10L_MuonWD_OvertoneInit.md | 1.1748 | notapplica | Muon WD, overtone init, 10L |
| 8 | 2026-03-19_SlidingWindowEval.md | 1.1925 | Matthew Li | Pure sliding window eval |
| 9 | 2026-03-17_LoRA_TTT.md | 1.1928 | samacqua | LoRA test-time training |
| 10 | 2026-03-19_TrainingOptSeq4096.md | 1.2014 | Spokane Way | seq4096, Muon 0.99 |
| 11 | 2026-03-18_LongContextSeq2048.md | 1.2058 | Spokane Way | seq2048 |
| 12 | 2026-03-19_10L_MixedPrecision.md | 1.2147 | Nan Liu | 10L, mixed int8/int6 |
| 13 | 2026-03-19_WarmdownQuantization.md | 1.2154 | samuellarson | WARMDOWN_ITERS=20000, NTK-RoPE |
| 14 | 2026-03-18_FP16Embed_WD3600.md | 1.2197 | Renier Velazco | FP16 tied embedding |
| 15 | 2026-03-18_LowerLR.md | 1.2230 | Nan Liu | LR sweep, optimal at 0.02 |
| 16 | 2026-03-17_NaiveBaseline.md | 1.2244 | Baseline | Official baseline |
| N/A | 2026-03-19_int6_STE_QAT_MLP_bigram_U_Net.md | unknown | unknown | Incomplete (log only) |

### Supplementary Files
| File | Description |
|------|-------------|
| PR_landscape.md | Analysis of 376+ open PRs and competition trajectory |

## Key Findings

### 1. CURRENT SOTA RECIPE (1.1428 bpb)
The winning formula combines ALL of these:
- **Architecture**: 10 layers, 512 dim, MLP 3x (1536), U-Net skips, tied embeddings
- **Token features**: SmearGate + BigramHash(10240, dim=128)
- **Quantization**: Mixed int5(MLP)/int6(attn), FP16 embed, STE QAT during training
- **Compression**: zstd-22
- **Training**: Muon WD=0.04, momentum=0.99, LR=0.02, grad_clip=0.3, seq_len=2048, batch=786K
- **Weight averaging**: SWA every 50 steps, last 40% of training
- **Evaluation**: Sliding window stride=64
- **Init**: Orthogonal + muP output scaling

### 2. TECHNIQUES THAT CONSISTENTLY WORK
1. Sliding window eval (stride=64): ~0.032 bpb free
2. STE int6 QAT: eliminates quant gap (0.007 -> 0.000 bpb)
3. MLP 3x expansion: largest single architecture contributor
4. FP16 tied embedding: prevents output head degradation
5. SmearGate + BigramHash: lightweight bigram features
6. Muon WD=0.04: optimal regularization for quantization
7. Muon momentum=0.99 with warmup: better convergence
8. SWA/EMA: smoother weights for quantization
9. zstd-22 over zlib: ~5% better compression

### 3. FRONTIER TECHNIQUES FROM OPEN PRs (NOT YET ON LEADERBOARD)
The most exciting developments are in open PRs:
1. **Cosine TTT (10-100 epochs)**: Multiple PRs claim 1.06-1.09 bpb -- MASSIVE gains
2. **XSA (Cross-Segment Attention)**: Consistent ~0.005-0.010 improvement
3. **Value Residual + Gated Attention**: ~0.015 bpb combined
4. **Partial RoPE + LN Scale**: Appears in many top PR submissions
5. **GPTQ-lite**: Alternative post-training quantization approach
6. **FlashAttention 3**: Enables more training steps per wallclock
7. **EMA(0.997)**: May outperform SWA for weight averaging
8. **SwiGLU**: Viable when combined with faster implementations
9. **Larger BigramHash (12288 buckets)**: Diminishing but real returns
10. **TrigramHash**: Mixed results, may complement BigramHash

### 4. COMPETITION TRAJECTORY
Phase 1 (Mar 17-18): Hyperparameter tuning (1.2244 -> 1.2058)
Phase 2 (Mar 19): Quantization+eval revolution (1.2058 -> 1.1630)
Phase 3 (Mar 19-20): Architecture enrichment (1.1630 -> 1.1428)
Phase 4 (Mar 20-23): TTT explosion -- scores crashing toward 1.06-1.09

### 5. WHAT WOULD BEAT 1.1428 bpb
Based on PR analysis, the most promising paths to beat current SOTA:
1. **Add Cosine TTT** to SOTA recipe (potential: sub-1.10 bpb)
2. **Add XSA4** cross-segment attention (potential: ~1.135 bpb without TTT)
3. **Add Value Residual + Gated Attention** (potential: ~1.128 bpb without TTT)
4. **Use EMA instead of SWA** (marginal improvement)
5. **Switch to FlashAttention 3** for more training steps
6. **Try SwiGLU** activation (viable with modern implementations)
7. **Explore 11-12 layers** with int5 MLP quantization
8. **BigramHash(12288+)** for more unique bigram features
9. **GPTQ-lite** post-training quantization
10. **Combine ALL of the above** -- the competition shows techniques stack

### 6. COMPETITION RULES SUMMARY
- 16MB artifact (code + compressed model), decimal 16,000,000 bytes
- 10 min training on 8xH100 SXM
- 10 min eval time (separate from training)
- No external downloads or network during eval
- New records must beat SOTA by >= 0.005 nats at p < 0.01
- Can import any package (FlashAttention etc OK)
- Eval at any sequence length allowed
- Cannot access validation data during training
- Test-time training only on already-evaluated tokens
- Submissions accepted chronologically by PR creation time

## Data Sources
- GitHub MCP: get_file_contents for all README.md and submission.json files
- WebFetch: PR listings (pages 1-3 of open and closed PRs)
- Total API calls: ~50
- Rate limiting encountered: 1 time (GitHub API)
- PR listing MCP denied: used WebFetch as fallback

## Notes
- The int6_STE_QAT_MLP_bigram_U_Net folder only contains train.log with no README or submission.json
- The LowerLR submission ran on 8xH200 (slightly faster than H100), which may account for a small portion of its improvement
- The WarmdownQuantization submission.json shows a different name ("Int6 MLP3x Sliding Window") and score (1.1574) than the README's claimed 1.2154 -- possible the submission.json was updated after initial README was written
- The TrainingOptSeq4096 submission.json was not fetched due to rate limiting
