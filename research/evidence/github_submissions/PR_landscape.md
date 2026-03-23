---
title: PR Landscape Analysis - Parameter Golf Competition
source_url: https://github.com/openai/parameter-golf/pulls
date: 2026-03-23
category: repo
author: various
val_bpb: varies
---

## Overview

As of 2026-03-23, the parameter-golf repo has 376 open PRs and many closed/merged ones. The competition is extremely active with rapid innovation. Below are the most significant emerging techniques from PRs NOT yet on the leaderboard.

## Merged Record PRs (in leaderboard)
- PR #180 (thwu1): CURRENT SOTA, 1.1428 bpb -- merged
- PR #162 (raahilshah): #2, 1.1458 bpb -- merged
- PR #86 (aruniyer): #3, 1.1502 bpb -- merged
- PR #65 (aquariouseworkman): SmearGate introduction, 1.1556 bpb -- merged
- PR #63 (yahya010): Int6 QAT + zstd, 1.1586 bpb -- merged
- PR #60 (notapplica): Muon WD + Overtone Init, 1.1748 bpb -- merged
- PR #77 (samacqua): LoRA TTT, 1.1928 bpb -- merged
- PR #61 (saml212): Warmdown quantization, 1.2154 bpb -- merged

## KEY EMERGING TECHNIQUES FROM OPEN/RECENT PRs

### 1. XSA (Cross-Segment Attention) -- MAJOR NEW TECHNIQUE
Multiple PRs reference "XSA" or "XSA4" or "XSA11":
- PR #452: 10L XSA + EMA + Partial RoPE + LN Scale -> 1.1366 bpb
- PR #434: 10L XSA + LeakyReLU^2 + Partial RoPE -> 1.1370 bpb
- PR #415/410: 11L XSA4 + Tight SWA + FA3 + Two-Phase TTT -> 1.1216 bpb
- PR #372: 11L + XSA4 + EMA + seq2048 + Int5-MLP -> 1.1361 bpb
- PR #359: 11L MLP3x + Int6 QAT + XSA + EMA + BigramHash + FA3 -> 1.1345 bpb
XSA appears to be a cross-segment attention mechanism that consistently yields improvement.

### 2. TTT (Test-Time Training) Evolution -- DRAMATIC GAINS
TTT has evolved significantly beyond the original LoRA TTT:
- PR #518: 11L XSA4 + LeakyReLU(0.5)^2 + Cosine TTT 50ep -> 1.0622 bpb (!!)
- PR #517: Goldfish ML -> 0.978 bpb (100ep Cosine TTT -- needs verification)
- PR #512: PROTEUS v7 -- 11L INT6 + LoRA TTT -> 0.9968 bpb
- PR #490: 11L Value Residual + Gated Attention + AdamW TTT -> 1.0891 bpb
- PR #486: TrigramHash + ValueResidual + GradQuant + Cosine TTT -> 1.0887 bpb
- PR #484: Sequential TTT + Memorization Analysis -> 1.0476 bpb (non-record)
- PR #481: Cosine TTT scheduling with per-layer lr -> 1.0970 bpb
- PR #462: SwiGLU + XSA4 + U-Net + AdamW TTT -> 1.0672 bpb
- PR #442: 11L EMA + AdamW TTT 10ep -> 1.1027 bpb
- PR #415: 11L XSA4 + Two-Phase TTT -> 1.1216 bpb
- PR #390: Sponge Bath -- TTT 8ep eval-only -> 1.1295 bpb

**CRITICAL INSIGHT**: Cosine TTT with many epochs (10-100) at eval time is yielding MASSIVE gains, with multiple PRs claiming sub-1.10 bpb. This is the frontier.

### 3. Value Residual / Gated Attention -- NEW ARCHITECTURE COMPONENTS
- PR #487: Value Residual (-0.015 BPB) + Gated Attention (-0.003 BPB)
- PR #490: Value Residual + Gated Attention + AdamW TTT -> 1.0891 bpb
- PR #388: 11L + Tight SWA + VE128 + Partial RoPE + LN Scale + TTT -> 1.1231 bpb
Value Residual appears to be a residual connection from value projections. VE128 suggests 128-dim value embeddings.

### 4. EMA (Exponential Moving Average) as SWA replacement
Multiple PRs use EMA instead of SWA:
- PR #372: EMA(0.997) -> consistently appears in top submissions
- PR #466: EMA + BigramHash(12288) + Mixed Int5 + FA3 -> 1.1354 bpb

### 5. Partial RoPE / LN Scale
- PR #452, #434: Partial RoPE + LN Scale appears as a consistent improvement
- Modifies rotary position embeddings to apply only partially

### 6. SwiGLU Activation
- PR #373: SwiGLU + BigramHash + SWA -> 1.1634 bpb
- PR #505: SwiGLU + VE128 + NoTTT -> 1.1181 bpb
- PR #462: SwiGLU + XSA4 + U-Net + AdamW TTT -> 1.0672 bpb
SwiGLU was dismissed early (slower per step) but appears viable when combined with other improvements.

### 7. GPTQ-lite / Advanced Quantization
- PR #478: GPTQ-lite + XSA + EMA + Late QAT -> 1.1268 bpb
- PR #508: GPTQ + Early QAT + Legal TTT -> 1.1215 bpb

### 8. FlashAttention 3 (FA3)
- PR #359, #415, #466: FA3 appears in multiple top submissions
- Likely enables faster attention computation -> more training steps in 10 min

### 9. TrigramHash
- PR #504: TrigramHash (iso-parametric bigram(96)+trigram(32)) -> 1.5275 (poor)
- PR #486: TrigramHash + ValueResidual + Cosine TTT -> 1.0887 bpb
- Mixed results -- may only work with TTT

### 10. Larger BigramHash
- PR #466: BigramHash(12288) -- pushing beyond SOTA's 10240 buckets
- PR #474: BigramHash(10240) in a 12L model

### 11. Catalytic Residuals
- PR #507: Catalytic + SwiGLU -> 1.1558 bpb
- PR #450: 12L + Catalytic Residuals + BigramHash(10240) -> 1.1466 bpb
- PR #474: Catalytic Residuals in non-record

### 12. Depth Recurrence
- PR #109/112: Depth Recurrence 5x3 d672 SwiGLU (closed, likely didn't work well)
- PR #521: RIOM v1 shared-depth recurrence (draft)
- PR #461: 11L Depth Recurrence + Legal TTT -> 1.14458 bpb

### 13. Paid Prefix (DISQUALIFIED)
- PR #278: 8L Paid Prefix + Sparse Hard Blocks -> 1.0365 bpb
- PR #275: Paid Prefix Research -> 1.0539 bpb
- These compress validation data into the 16MB artifact. The rules now explicitly forbid this.

## SUSPICIOUS/LIKELY INVALID CLAIMS
- PR #475: 0.50 bpb at 34KB -- almost certainly invalid
- PR #517: 0.978 bpb -- may be legitimate with 100ep Cosine TTT but needs verification
- PR #512: 0.9968 bpb -- needs careful verification
- PR #120: val_bpb=0.9588 -- closed, likely invalid (Val Only method)

## COMPETITION TRAJECTORY SUMMARY

The competition is evolving in clear phases:
1. **Phase 1 (Mar 17-18)**: Baseline tuning (LR, seq_len, FP16 embed)
2. **Phase 2 (Mar 19)**: Quantization revolution (int6 QAT, zstd, sliding window)
3. **Phase 3 (Mar 19-20)**: Architecture enrichment (SmearGate, BigramHash, MLP3x, SWA)
4. **Phase 4 (Mar 20-21)**: Advanced TTT begins (LoRA TTT -> AdamW TTT -> Cosine TTT)
5. **Phase 5 (Mar 21-23)**: TTT explosion + new attention (XSA, Value Residual, Gated Attention, Partial RoPE)

## TOP PRIORITY TECHNIQUES TO INVESTIGATE

1. **Cosine TTT with many epochs** -- the single biggest opportunity (claiming sub-1.10 bpb)
2. **XSA (Cross-Segment Attention)** -- consistent ~0.005-0.010 improvement
3. **Value Residual + Gated Attention** -- ~0.015-0.018 bpb combined
4. **EMA instead of SWA** -- may be better for weight averaging
5. **Partial RoPE** -- appears in many top PRs
6. **SwiGLU** -- may now be viable with faster implementations
7. **FlashAttention 3** -- enables more training steps
8. **GPTQ-lite post-training quantization** -- better than pure int5/int6?
