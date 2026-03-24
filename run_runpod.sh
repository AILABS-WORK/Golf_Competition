#!/bin/bash
# Parameter Golf — RunPod 8×H100 Benchmark Runner
# Runs variants at TRUE COMPETITION CONDITIONS:
#   • torchrun --nproc_per_node=8   (all 8 H100s)
#   • TRAIN_BATCH_TOKENS=524288     (competition batch)
#   • MAX_WALLCLOCK_SECONDS=600     (10-minute wall clock)
#   • ~7100 steps per run           (matches leaderboard exactly)
#
# Usage (from parameter-golf/ directory):
#   bash run_runpod.sh tier1        # 6 highest-priority variants (~1 hr, ~$7)
#   bash run_runpod.sh tier2        # next 6 variants (~1 hr, ~$7)
#   bash run_runpod.sh tier3        # sweep remaining variants (~2 hr, ~$14)
#   bash run_runpod.sh tier4        # WSM variants (~1 hr, ~$7)
#   bash run_runpod.sh tier5        # DiffAttn variants (~1 hr, ~$7)
#   bash run_runpod.sh tier6        # Peri-LN variants (~30 min, ~$3)
#   bash run_runpod.sh tier19       # Tier 19: rsLoRA/LoRA+/EWC/RELI-RA/LayerLR (~1 hr, ~$7)
#   bash run_runpod.sh tier20       # Tier 20: norm-bound/KD/label-smooth/chunk2k (~1 hr, ~$8)
#   bash run_runpod.sh tier21       # Tier 21: top-N layers + MLP-only TTT (~1 hr, ~$6)
#   bash run_runpod.sh tier22       # Tier 22: Muon optimizer for LoRA adapters (~1 hr, ~$5)
#   bash run_runpod.sh all          # everything (~5 hrs, ~$35)
#   bash run_runpod.sh V47          # single variant by ID
#
# Cost: ~$1 per variant on 8×H100 @ $6/hr
# Each run: exactly 10 minutes wall-clock, then auto-stops.
#
# DO NOT ADD:
#   TORCHDYNAMO_DISABLE=1     (not needed on Linux CUDA)
#   PYTORCH_SDP_BACKEND=...   (not needed on Linux)
#   TRAIN_SEQ_LEN=1024        (already default)

set -e

# ─── RUNPOD BASE CONFIG ────────────────────────────────────────────────────────
# Competition-accurate settings. No Windows workarounds.
BASE_VARS="DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  ITERATIONS=20000 \
  TRAIN_BATCH_TOKENS=524288 \
  MAX_WALLCLOCK_SECONDS=600 \
  VAL_LOSS_EVERY=500 \
  TRAIN_LOG_EVERY=100 \
  MUON_WEIGHT_DECAY=0.04 \
  MATRIX_LR=0.02 \
  SCALAR_LR=0.02 \
  WARMDOWN_ITERS=3500"

# Full SOTA stack base (11L, all proven techniques from 1.1233 submission)
SOTA_BASE="NUM_LAYERS=11 MLP_MULT=3 SMEARGATE=1 \
  BIGRAM_HASH_BUCKETS=2048 BIGRAM_HASH_DIM=128 \
  WARMDOWN_ITERS=3500"

# SOTA + quantization (full 1.1233 stack)
SOTA_QUANT="GPTQ_LITE=1 QUANT_BITS=6 COMPRESS_METHOD=zstd"

# ─── RUN FUNCTION ─────────────────────────────────────────────────────────────
run_rp() {
  local run_id=$1
  local extra_vars=$2
  local log="logs/${run_id}.txt"

  # Skip if log exists and has a final val result
  if [ -f "$log" ] && grep -q "step:[0-9]*/[0-9]* val_bpb:" "$log" 2>/dev/null; then
    local cached=$(grep "val_bpb:" "$log" | tail -1 | grep -o "val_bpb:[0-9.]*" | sed "s/val_bpb://")
    echo "  SKIP (cached): $run_id → val_bpb=$cached"
    return 0
  fi

  echo ""
  echo "╔══════════════════════════════════════════════════════════════════╗"
  echo "║  RUNPOD RUN: $run_id"
  printf "║  %-65s║\n" "Vars: $extra_vars"
  echo "╚══════════════════════════════════════════════════════════════════╝"

  mkdir -p logs
  eval "env $BASE_VARS RUN_ID=$run_id $extra_vars \
    torchrun --nproc_per_node=8 train_gpt.py" 2>&1 | tee "$log"

  local last_val=$(grep "val_bpb:" "$log" 2>/dev/null | grep "^step:" | tail -1 | grep -o "val_bpb:[0-9.]*" | sed "s/val_bpb://")
  echo ""
  echo "  ✓ DONE: $run_id → final val_bpb=$last_val"
  echo ""
}

TARGET="${1:-tier1}"

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 1 — Must run first (~6 variants, ~1 hr, ~$7)
# Goal: replicate SOTA baseline + test the two highest-impact novel techniques
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V47" ]]; then
  # Ground-truth SOTA replication (should match leaderboard 1.1233)
  run_rp "V47_full_sota" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V61" ]]; then
  # LeakyReLU² on SOTA stack — easy win from PR #518 (no TTT cost)
  run_rp "V61_leaky_relu2" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_ALPHA=0.5"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V73" ]]; then
  # TTT + LeakyReLU² — replicates PR #518 (1.0622 on leaderboard)
  run_rp "V73_ttt_leaky" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_ALPHA=0.5 \
     TTT_EPOCHS=10 TTT_LR=0.0001"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V82" ]]; then
  # HybridNorm + SSNorm — Layer 9 Tier S stack (both low-risk, one-line changes)
  run_rp "V82_hybrid_ss" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     HYBRID_NORM=1 SSNORM=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V83" ]]; then
  # Full SOTA (quantized) + HybridNorm + SSNorm — best submission candidate
  run_rp "V83_sota_layer9" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT HYBRID_NORM=1 SSNORM=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier1" || "$TARGET" == "V70" ]]; then
  # Cosine TTT 8 epochs (fastest TTT, baseline for the technique)
  run_rp "V70_ttt_8ep" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TTT_EPOCHS=8 TTT_LR=0.0001"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 2 — High value ablations (~6 variants, ~1 hr, ~$7)
# Goal: isolate which Layer 9 components contribute, test TTT scaling
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier2" || "$TARGET" == "V80" ]]; then
  # HybridNorm alone (isolate FFN Post-Norm contribution)
  run_rp "V80_hybridnorm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     HYBRID_NORM=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier2" || "$TARGET" == "V81" ]]; then
  # SSNorm alone (isolate outlier suppression contribution)
  run_rp "V81_ssnorm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     SSNORM=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier2" || "$TARGET" == "V71" ]]; then
  # TTT 10 epochs (PR #442 replication target: 1.1027)
  run_rp "V71_ttt_10ep" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TTT_EPOCHS=10 TTT_LR=0.0001"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier2" || "$TARGET" == "V74" ]]; then
  # Full SOTA + TTT 10ep (stack on 1.1233 foundation)
  run_rp "V74_sota_ttt10" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT TTT_EPOCHS=10 TTT_LR=0.0001"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier2" || "$TARGET" == "V84" ]]; then
  # Full SOTA + SSNorm (quantization benefit of outlier suppression)
  run_rp "V84_sota_ssnorm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT SSNORM=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier2" || "$TARGET" == "V85" ]]; then
  # Full SOTA + HybridNorm (Post-Norm quality on quantized model)
  run_rp "V85_sota_hybridnorm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT HYBRID_NORM=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 3 — Sweep: combos, activation variants, TTT scaling (~12 variants, ~2 hr)
# Run after Tier 1+2 results tell you which direction to invest in
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V62" ]]; then
  # SwiGLU mlp_mult=2 (same params as relu² at mult=3, smooth gating)
  run_rp "V62_swiglu_m2" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MLP_ACTIVATION=swiglu MLP_MULT=2"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V63" ]]; then
  # SwiGLU mlp_mult=3 (more params, tests capacity trade-off)
  run_rp "V63_swiglu_m3" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MLP_ACTIVATION=swiglu"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V72" ]]; then
  # TTT 20 epochs (more adaptation, slower eval)
  run_rp "V72_ttt_20ep" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TTT_EPOCHS=20 TTT_LR=0.0001"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V75" ]]; then
  # TTT higher LR (1e-3 vs default 1e-4)
  run_rp "V75_ttt_lr1e3" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TTT_EPOCHS=10 TTT_LR=0.001"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V50" ]]; then
  # Value Residual on SOTA stack (ResFormer arXiv:2410.17897)
  run_rp "V50_value_residual" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     VALUE_RESIDUAL=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V51" ]]; then
  # MoLE on SOTA stack (Mixture of Lookup Experts)
  run_rp "V51_mole" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     MOLE_NUM_EXPERTS=4 MOLE_DIM=64"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V52" ]]; then
  # TWEO on SOTA stack (colinearity penalty, arXiv:2511.23225)
  run_rp "V52_tweo" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TWEO_LAMBDA=0.0001"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V53" ]]; then
  # Tight SWA on SOTA stack (threshold-triggered weight averaging)
  run_rp "V53_tight_swa" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     TIGHT_SWA=1 TIGHT_SWA_THRESHOLD=0.2 TIGHT_SWA_INTERVAL=50"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier3" || "$TARGET" == "V59" ]]; then
  # Kitchen sink: SOTA + VR + MoLE + TWEO + Layer 9
  run_rp "V59_sota_all_novel" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT \
     VALUE_RESIDUAL=1 MOLE_NUM_EXPERTS=4 MOLE_DIM=64 TWEO_LAMBDA=0.0001 \
     HYBRID_NORM=1 SSNORM=1 \
     TTT_EPOCHS=10 TTT_LR=0.0001 MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_ALPHA=0.5"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 4 — WSM + NuMuon variants (from research agent findings, 2026-03-24)
#   V86: WSM alone (no warmdown, stable-LR checkpoint merge)
#   V87: WSM + SOTA stack (replaces warmdown with merge)
#   V88: WSM + HybridNorm + SSNorm (all no-cost improvements stacked)
#   V89: WSM + TTT + LeakyReLU² (frontier: every technique stacked)
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier4" || "$TARGET" == "V86" ]]; then
  # WSM alone on SOTA stack: no warmdown, merge last 30% of checkpoints
  run_rp "V86_wsm" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     WSM=1 WSM_MERGE_FRACTION=0.3 SWA_INTERVAL=50"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier4" || "$TARGET" == "V87" ]]; then
  # WSM + full quantized SOTA (replaces WARMDOWN_ITERS with stable-phase merge)
  run_rp "V87_wsm_sota" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT WSM=1 WSM_MERGE_FRACTION=0.3 SWA_INTERVAL=50"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier4" || "$TARGET" == "V88" ]]; then
  # WSM + HybridNorm + SSNorm + SOTA (stack all zero-cost improvements)
  run_rp "V88_wsm_layer9" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT HYBRID_NORM=1 SSNORM=1 WSM=1 WSM_MERGE_FRACTION=0.3 SWA_INTERVAL=50"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier4" || "$TARGET" == "V89" ]]; then
  # Full frontier stack: WSM + Layer9 + TTT + LeakyReLU² (kitchen sink v2)
  run_rp "V89_frontier_stack" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT HYBRID_NORM=1 SSNORM=1 \
     WSM=1 WSM_MERGE_FRACTION=0.3 SWA_INTERVAL=50 \
     MLP_ACTIVATION=leaky_relu2 LEAKY_RELU_ALPHA=0.5 \
     TTT_EPOCHS=10 TTT_LR=0.0001"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 5 — Differential Transformer (arXiv:2410.05258, ICLR 2025 Oral)
#   Halves head_dim, two sub-heads per head (Q1/Q2, K1/K2) sharing same V.
#   Lambda scalar: attn = attn1 - λ·attn2 damps attention noise.
#   Zero parameter overhead — projection shapes identical to baseline.
#   NOTE: incompatible with VALUE_RESIDUAL (different V shape).
#   V90: DiffAttn alone (ablation — does the mechanism help at all?)
#   V91: DiffAttn + full quantized SOTA (main submission candidate)
#   V92: DiffAttn + HybridNorm + SSNorm + SOTA (maximum stacking)
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier5" || "$TARGET" == "V90" ]]; then
  # DiffAttn alone on SOTA stack: isolates the differential attention mechanism
  run_rp "V90_diff_attn" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     DIFF_TRANSFORMER=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier5" || "$TARGET" == "V91" ]]; then
  # DiffAttn + full quantized SOTA (XSA auto-disabled by DiffAttn incompatibility)
  run_rp "V91_diff_attn_sota" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT DIFF_TRANSFORMER=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier5" || "$TARGET" == "V92" ]]; then
  # DiffAttn + Layer 9 (HybridNorm + SSNorm) + full SOTA (maximum novel stack)
  run_rp "V92_diff_layer9_sota" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT DIFF_TRANSFORMER=1 HYBRID_NORM=1 SSNORM=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 6 — Peri-LN (arXiv:2502.02732, Gemma/OLMo 2 families)
#   Pre-Norm + Post-Norm on BOTH attention and FFN sublayers.
#   Superset of HybridNorm (which Post-Norms FFN only). Mutually exclusive.
#   V93: Peri-LN alone on SOTA stack (ablation vs HybridNorm)
#   V94: Peri-LN + SSNorm + SOTA (replaces HybridNorm in best layer-9 stack)
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier6" || "$TARGET" == "V93" ]]; then
  # Peri-LN alone on SOTA stack (compare directly to V80_hybridnorm)
  run_rp "V93_peri_ln" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     PERI_LN=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier6" || "$TARGET" == "V94" ]]; then
  # Peri-LN + SSNorm + full SOTA (replaces HybridNorm in V83 for direct comparison)
  run_rp "V94_peri_ln_ssnorm_sota" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT PERI_LN=1 SSNORM=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 7 — Power-law warmdown (arXiv:2602.06797 optimal LR scaling laws)
#   Linear decay (α=1.0) is the default. For severely undertrained models
#   (≪Chinchilla optimal), scaling theory predicts steeper decay (α>1) is better.
#   V95: α=2.0 (quadratic) on full SOTA — tests steeper warmdown shape
#   V96: α=1.5 (mid-power) on full SOTA — moderate steepening
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier7" || "$TARGET" == "V95" ]]; then
  run_rp "V95_wsd_power2" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT WSD_POWER=2.0"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier7" || "$TARGET" == "V96" ]]; then
  run_rp "V96_wsd_power15" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT WSD_POWER=1.5"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 8 — Adaptive Group Gradient Clipping (arXiv:2601.11864)
# ═══════════════════════════════════════════════════════════════════════════════
#   V97: AGGC β=0.99 on full SOTA — tests per-group adaptive gradient clipping

if [[ "$TARGET" == "all" || "$TARGET" == "tier8" || "$TARGET" == "V97" ]]; then
  run_rp "V97_aggc" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT AGGC_BETA=0.99 AGGC_THRESHOLD=3.0"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 9 — DenseFormer: Depth-Weighted Averaging (arXiv:2402.02622, NeurIPS 2024)
# ═══════════════════════════════════════════════════════════════════════════════
#   V98: DenseFormer on full SOTA — replaces U-Net skips with learned DWA (66 scalars)

if [[ "$TARGET" == "all" || "$TARGET" == "tier9" || "$TARGET" == "V98" ]]; then
  run_rp "V98_denseformer" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT DENSEFORMER=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 10 — NuMuon: Nuclear-Norm Constrained Muon (arXiv:2603.03597)
# ═══════════════════════════════════════════════════════════════════════════════
#   V99: NuMuon weight=1e-4 on full SOTA — promotes low-rank weights → better int6

if [[ "$TARGET" == "all" || "$TARGET" == "tier10" || "$TARGET" == "V99" ]]; then
  run_rp "V99_numuon" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT NUMUON_WEIGHT=1e-4"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 11 — MUDDFormer: Dynamic Dense Connections (arXiv:2502.12170, ICML 2025)
# ═══════════════════════════════════════════════════════════════════════════════
#   V100: MUDD V-stream only (highest per-ablation benefit, ~45K params)
#   V101: MUDD Q+K+V streams (full 3-stream variant, ~137K params)

if [[ "$TARGET" == "all" || "$TARGET" == "tier11" || "$TARGET" == "V100" ]]; then
  run_rp "V100_mudd_v" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT MUDD_STREAMS=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier11" || "$TARGET" == "V101" ]]; then
  run_rp "V101_mudd_qkv" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT MUDD_STREAMS=3"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 12 — LOTION: Smooth Quantization via Stochastic Noise (arXiv:2510.08757)
# ═══════════════════════════════════════════════════════════════════════════════
#   V102: LOTION QAT from step 0 — replaces STE with calibrated noise injection.
#         σ=sB*sqrt(Δ(1-Δ)) noise convergence guarantee. STE is biased; LOTION is not.

if [[ "$TARGET" == "all" || "$TARGET" == "tier12" || "$TARGET" == "V102" ]]; then
  run_rp "V102_lotion" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT LOTION=1 QAT_START_FRACTION=0.0"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 13 — Gated Attention (arXiv:2505.06708, NeurIPS 2025 Best Paper, Qwen3)
# ═══════════════════════════════════════════════════════════════════════════════
#   V103: Post-SDPA sigmoid gate — eliminates attention sinks, ~4K extra params.
#         gate = sigmoid(W_gate @ x) ∈ R^{B,T,H}, applied as y ← y * gate.
if [[ "$TARGET" == "all" || "$TARGET" == "tier13" || "$TARGET" == "V103" ]]; then
  run_rp "V103_gated_attn" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT GATED_ATTN=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 14 — Muon-VS: Variance-Adaptive Muon (arXiv:2601.14603)
# ═══════════════════════════════════════════════════════════════════════════════
#   V104: Variance scaling before Newton-Schulz. 1.36x faster convergence.
#         Zero new hyperparameters beyond existing Muon momentum.
if [[ "$TARGET" == "all" || "$TARGET" == "tier14" || "$TARGET" == "V104" ]]; then
  run_rp "V104_muon_vs" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT MUON_VS=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 15 — LoRA TTT: Rank-8 Q/V Adapters at Test-Time (competition frontier)
# ═══════════════════════════════════════════════════════════════════════════════
# PR #596 (0.6430 BPB), PR #605 (0.7227), PR #611 Chimera (0.5601 BPB)
# Key insight: ~125–200× fewer trainable params than full TTT → more epochs per
# chunk in the same 600s wall clock. Per-chunk LoRA reset = document adaptation.
#
#   V105: LoRA TTT baseline — rank-8 Q/V, 50 epochs, cosine LR, SOTA stack.
#         Expected ~0.65–0.75 BPB (vs 1.0622 full-param TTT baseline).
#
#   V106: LoRA TTT "Chimera" — V105 + K-LoRA (0.3× LR) + min-NLL selection +
#         temperature calibration (T=0.98). Mirrors PR #611 configuration.
#         Expected ~0.56–0.64 BPB (competition frontier).
#
#   V107: V106 + 100 epochs. LoRA efficiency lets us run 2× more epochs
#         in the same wall clock as full-param 50-epoch TTT.
#         Expected ~0.54–0.60 BPB.

if [[ "$TARGET" == "all" || "$TARGET" == "tier15" || "$TARGET" == "V105" ]]; then
  run_rp "V105_lora_ttt_base" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT MLP_ACTIVATION=leaky_relu2 \
     TTT_LORA=1 TTT_EPOCHS=50 TTT_LR=3e-4 TTT_LORA_RANK_QV=8"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier15" || "$TARGET" == "V106" ]]; then
  run_rp "V106_lora_ttt_chimera" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT MLP_ACTIVATION=leaky_relu2 \
     TTT_LORA=1 TTT_EPOCHS=50 TTT_LR=3e-4 TTT_LORA_RANK_QV=8 TTT_LORA_RANK_LM=16 \
     TTT_K_LORA=1 TTT_MIN_NLL=1 TTT_TEMPERATURE=0.98"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier15" || "$TARGET" == "V107" ]]; then
  run_rp "V107_lora_ttt_chimera_100ep" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT MLP_ACTIVATION=leaky_relu2 \
     TTT_LORA=1 TTT_EPOCHS=100 TTT_LR=3e-4 TTT_LORA_RANK_QV=8 TTT_LORA_RANK_LM=16 \
     TTT_K_LORA=1 TTT_MIN_NLL=1 TTT_TEMPERATURE=0.98"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 16 — Novel LoRA TTT Extensions (V108–V115)
# ═══════════════════════════════════════════════════════════════════════════════
# Five novel extensions beyond PR #611 Chimera, pushing toward 0.44–0.51 BPB:
#
#  V108: MLP LoRA — rank-4 LoRA on MLP fc (0.5× LR) + proj (3× LR). Adapts
#        factual knowledge alongside attention routing. Expected ~0.49–0.53 BPB.
#
#  V109: Gate Adaptation — unfreeze attn_scale/mlp_scale per block (5.6K total
#        params, 0.05× LR). Adjusts layer contribution profile per document.
#        Expected ~0.51–0.56 BPB (alone), likely synergistic with V108.
#
#  V110: RELI — Retroactive Gradient-Aligned LoRA Init. SVD of pre-TTT gradient
#        gives A/B init aligned with the loss surface. Expected 3–5× faster
#        convergence per chunk → ~0.49–0.54 BPB.
#
#  V111: Soft-Reset (decay=0.5) — cross-chunk LoRA memory instead of hard reset.
#        Retains document context across adjacent chunks in same document.
#        Expected ~0.50–0.55 BPB (depends on document boundary alignment).
#
#  V112: Difficulty-Adaptive Epochs — pre-score chunk, allocate more epochs to
#        high-NLL (hard) chunks. Better compute allocation → ~0.50–0.55 BPB.
#
#  V113: Full kitchen sink — Chimera + MLP LoRA + Gates + RELI + Soft-Reset.
#        Expected ~0.44–0.51 BPB if all techniques are additive.
#
#  V114: RELI + MLP LoRA (no gates/soft-reset) — test RELI x MLP synergy.
#        Expected ~0.47–0.52 BPB.
#
#  V115: Chimera + Difficulty-Adaptive Epochs + RELI.
#        Expected ~0.46–0.52 BPB.

CHIMERA_BASE="$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
  $SOTA_QUANT MLP_ACTIVATION=leaky_relu2 \
  TTT_LORA=1 TTT_EPOCHS=50 TTT_LR=3e-4 TTT_LORA_RANK_QV=8 TTT_LORA_RANK_LM=16 \
  TTT_K_LORA=1 TTT_MIN_NLL=1 TTT_TEMPERATURE=0.98"

if [[ "$TARGET" == "all" || "$TARGET" == "tier16" || "$TARGET" == "V108" ]]; then
  run_rp "V108_mlp_lora" \
    "$CHIMERA_BASE TTT_MLP_LORA=1 TTT_LORA_RANK_MLP=4"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier16" || "$TARGET" == "V109" ]]; then
  run_rp "V109_gate_adapt" \
    "$CHIMERA_BASE TTT_GATE_ADAPT=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier16" || "$TARGET" == "V110" ]]; then
  run_rp "V110_reli_init" \
    "$CHIMERA_BASE TTT_RELI=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier16" || "$TARGET" == "V111" ]]; then
  run_rp "V111_soft_reset" \
    "$CHIMERA_BASE TTT_LORA_DECAY=0.5"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier16" || "$TARGET" == "V112" ]]; then
  run_rp "V112_difficulty_adaptive" \
    "$CHIMERA_BASE TTT_DIFFICULTY=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier16" || "$TARGET" == "V113" ]]; then
  run_rp "V113_full_novel_stack" \
    "$CHIMERA_BASE TTT_MLP_LORA=1 TTT_LORA_RANK_MLP=4 TTT_GATE_ADAPT=1 \
     TTT_RELI=1 TTT_LORA_DECAY=0.5"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier16" || "$TARGET" == "V114" ]]; then
  run_rp "V114_reli_plus_mlp" \
    "$CHIMERA_BASE TTT_MLP_LORA=1 TTT_LORA_RANK_MLP=4 TTT_RELI=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier16" || "$TARGET" == "V115" ]]; then
  run_rp "V115_difficulty_reli" \
    "$CHIMERA_BASE TTT_RELI=1 TTT_DIFFICULTY=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 17 — qTTT-inspired and Architecture Extensions (V116–V120)
# ═══════════════════════════════════════════════════════════════════════════════
# New variants from deep paper research (2025–2026):
#
#  V116: Q-only LoRA TTT (qTTT-inspired, arXiv:2512.13898)
#        Update only c_q adapters, not V/K/lm_head. Hypothesis: Q controls
#        "what to retrieve" without corrupting key/value document encodings.
#        Expected ~0.52–0.58 BPB (fewer params, cleaner adaptation signal).
#
#  V117: Q-only + RELI. RELI gradient init is especially valuable for Q-only
#        since the gradient SVD directly shows which query directions matter.
#        Expected ~0.50–0.55 BPB.
#
#  V118: Full Chimera + Q-only + RELI + MLP LoRA.
#        Q-only for attention (cleaner) + MLP LoRA for factual knowledge.
#        Orthogonal: attention routing (Q) and factual recall (MLP).
#        Expected ~0.46–0.52 BPB.
#
#  V119: RELI (staggered LoRA-GA init) ablation vs V110.
#        V110 used simple init; V119 uses proven staggered init (LoRA-GA NeurIPS 2024).
#        Same config as V110 — pure ablation of init quality.
#        Expected improvement: +1–3% faster convergence per chunk.
#
#  V120: Kitchen sink V2 — Chimera + MLP LoRA + RELI (staggered) + Difficulty + Q-not-only.
#        Keeps V/K LoRA but adds all research-validated improvements.
#        Expected ~0.44–0.50 BPB.

if [[ "$TARGET" == "all" || "$TARGET" == "tier17" || "$TARGET" == "V116" ]]; then
  run_rp "V116_q_only_lora" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT MLP_ACTIVATION=leaky_relu2 \
     TTT_LORA=1 TTT_EPOCHS=50 TTT_LR=3e-4 TTT_LORA_RANK_QV=8 \
     TTT_MIN_NLL=1 TTT_TEMPERATURE=0.98 TTT_Q_ONLY=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier17" || "$TARGET" == "V117" ]]; then
  run_rp "V117_q_only_reli" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT MLP_ACTIVATION=leaky_relu2 \
     TTT_LORA=1 TTT_EPOCHS=50 TTT_LR=3e-4 TTT_LORA_RANK_QV=8 \
     TTT_MIN_NLL=1 TTT_TEMPERATURE=0.98 TTT_Q_ONLY=1 TTT_RELI=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier17" || "$TARGET" == "V118" ]]; then
  run_rp "V118_q_only_reli_mlp" \
    "$CHIMERA_BASE TTT_Q_ONLY=1 TTT_RELI=1 TTT_MLP_LORA=1 TTT_LORA_RANK_MLP=4"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier17" || "$TARGET" == "V119" ]]; then
  # RELI with staggered LoRA-GA init (ablation vs V110 which had simple init)
  run_rp "V119_reli_staggered_ablation" \
    "$CHIMERA_BASE TTT_RELI=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier17" || "$TARGET" == "V120" ]]; then
  run_rp "V120_kitchen_sink_v2" \
    "$CHIMERA_BASE TTT_MLP_LORA=1 TTT_LORA_RANK_MLP=4 TTT_RELI=1 TTT_DIFFICULTY=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 18 — Novel Cross-Technique Connections (V121–V130)
# ═══════════════════════════════════════════════════════════════════════════════
# Four deep connections found by analyzing all variants together:
#
# CONNECTION 1: RELI S[0] → free difficulty + temperature signal
#   RELI already computes SVD of gradient; S[0] = "model surprise". This is a
#   better difficulty proxy than a separate NLL forward pass AND calibrates T.
#   V121 = RELI-Difficulty fusion (no extra forward pass, S[0] as difficulty)
#   V123 = Adaptive Temperature (T scales from 0.98→1.0 as S[0] increases)
#
# CONNECTION 2: Two-Phase RELI — residual gradient exploration
#   Phase 1: RELI init → TTT → min-NLL checkpoint.
#   Phase 2: RELI on RESIDUAL gradient from checkpoint → 50% more epochs.
#   Key insight: gradient at min-NLL encodes what the model STILL doesn't know.
#   Phase 2 explores the orthogonal subspace phase 1 missed.
#   V122 = Two-Phase RELI only
#   V127 = Two-Phase RELI + Token-Selective (compound signal quality)
#
# CONNECTION 3: Token-Selective TTT — importance sampling over tokens
#   Hard tokens carry ~80% of gradient signal; easy tokens add noise.
#   Train on top-50% highest-loss tokens per epoch → same wall-clock, better SNR.
#   V124 = Token-Selective only (TTT_TOKEN_K=0.5)
#   V125 = Token-Selective + RELI (both target high-information signal)
#
# CONNECTION 4: Q+MLP "routing+knowledge" separation
#   Q-only adapts attention routing; MLP LoRA adapts factual knowledge.
#   Together with RELI: gradient init is better because attention and MLP
#   have different gradient structures → orthogonal SVD directions.
#   V118 already covers this (already in tier17).
#
# KITCHEN SINK VARIANTS (all techniques together):
#   V128 = Two-Phase RELI + Token-Selective + Adaptive-T + MLP LoRA
#   V129 = Full stack + Q-only + Two-Phase RELI + Token-Selective
#   V130 = Max ablation: every novel technique in one run

if [[ "$TARGET" == "all" || "$TARGET" == "tier18" || "$TARGET" == "V121" ]]; then
  # RELI-Difficulty fusion: S[0] as difficulty proxy (no extra forward pass)
  run_rp "V121_reli_diff_fusion" \
    "$CHIMERA_BASE TTT_RELI=1 TTT_DIFFICULTY=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier18" || "$TARGET" == "V122" ]]; then
  # Two-Phase RELI: residual gradient exploration after phase-1 min-NLL
  run_rp "V122_two_phase_reli" \
    "$CHIMERA_BASE TTT_RELI=1 TTT_RELI_PHASES=2 TTT_MIN_NLL=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier18" || "$TARGET" == "V123" ]]; then
  # Adaptive Temperature: S[0] calibrates T per-chunk
  run_rp "V123_adaptive_temp" \
    "$CHIMERA_BASE TTT_RELI=1 TTT_ADAPTIVE_TEMP=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier18" || "$TARGET" == "V124" ]]; then
  # Token-Selective TTT: train only on top-50% hardest tokens
  run_rp "V124_token_selective_50" \
    "$CHIMERA_BASE TTT_TOKEN_K=0.5"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier18" || "$TARGET" == "V125" ]]; then
  # Token-Selective + RELI: both target high-information signal
  run_rp "V125_token_sel_reli" \
    "$CHIMERA_BASE TTT_RELI=1 TTT_TOKEN_K=0.5"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier18" || "$TARGET" == "V126" ]]; then
  # Two-Phase RELI + Adaptive Temp: compound gradient-signal improvements
  run_rp "V126_twophase_adaptive" \
    "$CHIMERA_BASE TTT_RELI=1 TTT_RELI_PHASES=2 TTT_ADAPTIVE_TEMP=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier18" || "$TARGET" == "V127" ]]; then
  # Two-Phase RELI + Token-Selective: both attack gradient quality
  run_rp "V127_twophase_toksel" \
    "$CHIMERA_BASE TTT_RELI=1 TTT_RELI_PHASES=2 TTT_TOKEN_K=0.5"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier18" || "$TARGET" == "V128" ]]; then
  # Compound: Two-Phase RELI + Token-Selective + Adaptive-T + MLP LoRA
  run_rp "V128_compound_reli_mlp" \
    "$CHIMERA_BASE TTT_RELI=1 TTT_RELI_PHASES=2 TTT_ADAPTIVE_TEMP=1 \
     TTT_TOKEN_K=0.5 TTT_MLP_LORA=1 TTT_LORA_RANK_MLP=4"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier18" || "$TARGET" == "V129" ]]; then
  # Q-only path: Q + MLP LoRA + Two-Phase RELI + Token-Selective
  # Clean routing/knowledge separation with best gradient techniques
  run_rp "V129_qonly_twophase_toksel" \
    "$SOTA_BASE XSA_LAST_N=4 EMA=1 EMA_DECAY=0.997 PARTIAL_ROPE_DIMS=16 LN_SCALE=1 \
     $SOTA_QUANT MLP_ACTIVATION=leaky_relu2 \
     TTT_LORA=1 TTT_EPOCHS=50 TTT_LR=3e-4 TTT_LORA_RANK_QV=8 TTT_LORA_RANK_LM=16 \
     TTT_MIN_NLL=1 TTT_TEMPERATURE=0.98 \
     TTT_Q_ONLY=1 TTT_MLP_LORA=1 TTT_LORA_RANK_MLP=4 \
     TTT_RELI=1 TTT_RELI_PHASES=2 TTT_TOKEN_K=0.5 TTT_ADAPTIVE_TEMP=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier18" || "$TARGET" == "V130" ]]; then
  # Maximum stack: every novel technique enabled
  # Expected: 0.41-0.48 BPB if all techniques are additive
  run_rp "V130_maximum_novel_stack" \
    "$CHIMERA_BASE \
     TTT_MLP_LORA=1 TTT_LORA_RANK_MLP=4 TTT_GATE_ADAPT=1 \
     TTT_RELI=1 TTT_RELI_PHASES=2 TTT_ADAPTIVE_TEMP=1 \
     TTT_TOKEN_K=0.5 TTT_DIFFICULTY=1 TTT_LORA_DECAY=0.5"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 19 — Research-validated LoRA mechanics improvements (V131–V138)
# All derived from 6 deep-research agents (2026-03-24 synthesis):
#
#  V131: rsLoRA + LoRA+ — two mechanical optimizer fixes:
#        rsLoRA (arXiv:2312.03732): LR *= sqrt(r) so rank changes don't cause LR jumps.
#        LoRA+ (arXiv:2402.12354, ICML 2024): lr_B = 4 × lr_A. B maps FROM learned
#        subspace (needs large steps), A projects INTO it (needs stable steps).
#        Expected: -0.010 to -0.025 BPB. Zero architecture change, 2 lines.
#
#  V132: Fisher Memory TTT — EWC cross-chunk regularization using Adam second moment
#        as diagonal Fisher approximation (zero extra compute: exp_avg_sq is already
#        computed). Principled replacement for soft-reset: preserves high-curvature
#        parameters, allows free update of flat directions.
#        Expected: -0.008 to -0.020 BPB over soft-reset baseline.
#
#  V133: RELI-RA (Rank Annealing) — start at rank-2r for wider subspace exploration,
#        SVD-compress to rank-r at epoch//3 (retaining top-r signal directions).
#        rsLoRA ensures LR doesn't jump 1.41× at the prune step.
#        Based on DyLoRA (arXiv:2210.07558) + GaLore (arXiv:2403.03507).
#        Expected: -0.008 to -0.025 BPB (richer init → faster convergence).
#
#  V134: Layer-Stratified Q LR — per-depth LR multipliers for Q LoRA:
#        Early layers (0-3): 1×  — syntax/positional, stable across docs
#        Middle (4-7):       3×  — semantic routing, moderate variance
#        Late (8-10):        8×  — per-doc recall heads, highest variance
#        Based on E2E-TTT (arXiv:2512.23675) + Rogers 2020 probing studies.
#        Expected: -0.005 to -0.015 BPB (concentrates compute where it matters).
#
#  V135: Proj-only MLP LoRA — freeze W_fc (key side), concentrate rank budget on
#        W_proj (output side) at rank=8 and 3× LR. Justified by ROME/MEMIT:
#        factual updates exclusively require W_proj (vocab promotion layer).
#        Expected: -0.005 to -0.012 BPB over fc+proj split.
#
#  V136: Extended Adaptive Temperature — T_max = 1.3 instead of 1.0.
#        Research (arXiv:2409.19817): optimal T ≈ 1.3 after heavy fine-tuning.
#        Prior range [0.98, 1.0] still undercorrected overconfidence on hard chunks.
#        Expected: -0.002 to -0.008 BPB (calibration improvement).
#
#  V137: Tier 19 compound — rsLoRA + LoRA+ + Fisher Memory + RELI-RA + Layer LR
#        All five mechanical improvements stacked. Expected: -0.025 to -0.060 BPB.
#
#  V138: Max Tier 19 — V137 + Extended AdaptiveTemp + Proj-only MLP + Two-Phase RELI
#        Every research-validated improvement from Tiers 18 + 19.
#        Expected: -0.035 to -0.080 BPB vs Chimera baseline.
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier19" || "$TARGET" == "V131" ]]; then
  # rsLoRA + LoRA+ asymmetric LR on Chimera baseline
  run_rp "V131_rslora_loraplus" \
    "$CHIMERA_BASE TTT_RS_LORA=1 TTT_LORA_PLUS=1 TTT_LORA_PLUS_LAMBDA=4.0"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier19" || "$TARGET" == "V132" ]]; then
  # Fisher Memory EWC: cross-chunk regularization (light lambda=0.05)
  run_rp "V132_fisher_ewc" \
    "$CHIMERA_BASE TTT_EWC_LAMBDA=0.05"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier19" || "$TARGET" == "V133" ]]; then
  # RELI-RA: rank annealing (start 2r=16, prune to r=8 at epoch//3)
  run_rp "V133_reli_ra" \
    "$CHIMERA_BASE TTT_RELI=1 TTT_RELI_RA=1 TTT_RS_LORA=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier19" || "$TARGET" == "V134" ]]; then
  # Layer-stratified Q LR: early=1x, mid=3x, late=8x
  run_rp "V134_layer_lr" \
    "$CHIMERA_BASE TTT_LAYER_LR=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier19" || "$TARGET" == "V135" ]]; then
  # Proj-only MLP LoRA: rank-8 on W_proj only (freeze W_fc)
  run_rp "V135_proj_only_mlp" \
    "$CHIMERA_BASE TTT_MLP_LORA=1 TTT_PROJ_ONLY_MLP=1 TTT_LORA_RANK_QV=8"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier19" || "$TARGET" == "V136" ]]; then
  # Extended Adaptive Temperature: T range [0.98, 1.3] based on calibration research
  run_rp "V136_ext_adaptive_temp" \
    "$CHIMERA_BASE TTT_RELI=1 TTT_ADAPTIVE_TEMP=1 TTT_ADAPTIVE_TEMP_MAX=1.3"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier19" || "$TARGET" == "V137" ]]; then
  # Tier 19 compound: rsLoRA + LoRA+ + Fisher Memory + RELI-RA + Layer LR
  run_rp "V137_tier19_compound" \
    "$CHIMERA_BASE \
     TTT_RS_LORA=1 TTT_LORA_PLUS=1 TTT_LORA_PLUS_LAMBDA=4.0 \
     TTT_EWC_LAMBDA=0.05 \
     TTT_RELI=1 TTT_RELI_RA=1 \
     TTT_LAYER_LR=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier19" || "$TARGET" == "V138" ]]; then
  # Max Tier 19: V137 + Extended AdaptiveTemp + Proj-only MLP + Two-Phase RELI
  run_rp "V138_max_tier19" \
    "$CHIMERA_BASE \
     TTT_RS_LORA=1 TTT_LORA_PLUS=1 TTT_LORA_PLUS_LAMBDA=4.0 \
     TTT_EWC_LAMBDA=0.05 \
     TTT_RELI=1 TTT_RELI_RA=1 TTT_RELI_PHASES=2 \
     TTT_LAYER_LR=1 \
     TTT_MLP_LORA=1 TTT_PROJ_ONLY_MLP=1 \
     TTT_ADAPTIVE_TEMP=1 TTT_ADAPTIVE_TEMP_MAX=1.3 \
     TTT_TOKEN_K=0.5 TTT_DIFFICULTY=1"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 20: Convergence stabilizers + anti-forgetting (V139–V146)
# Expected: ~0.01–0.05 BPB gains individually, 0.03–0.08 compounded
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier20" || "$TARGET" == "V139" ]]; then
  # LoRA norm bounding (Mu & Klabjan 2024): recover O(1/T) convergence rate.
  # Unconstrained LoRA growth → O(1/log T) rate; clamping ||A||,||B|| ≤ 1.0 fixes it.
  run_rp "V139_norm_bound" \
    "$CHIMERA_BASE TTT_NORM_BUDGET=1.0"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier20" || "$TARGET" == "V140" ]]; then
  # Larger chunk size (LaCT arXiv:2505.23884): 1024→2048 tokens per TTT chunk.
  # Default 1024 leaves GPU at <5% utilization during TTT; 2048 → >50% utilization.
  # Fewer chunks but more context per chunk: better cross-token gradient signal.
  run_rp "V140_chunk2048" \
    "$CHIMERA_BASE TTT_CHUNK_SIZE=2048"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier20" || "$TARGET" == "V141" ]]; then
  # Label smoothing in TTT inner loop (Müller et al. 2019 + anti-forgetting research).
  # alpha=0.05 prevents collapse to single-token predictions; equivalent to post-hoc T scaling.
  run_rp "V141_label_smooth" \
    "$CHIMERA_BASE TTT_LABEL_SMOOTH=0.05"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier20" || "$TARGET" == "V142" ]]; then
  # KD regularization from frozen base (Hinton 2015 + TTT anti-forgetting).
  # Captures base logits before TTT (B=0 → pure base), penalises KL divergence during inner loop.
  # T=2 softens targets; prevents overconfidence; -0.02 to -0.05 BPB expected.
  run_rp "V142_kd_reg" \
    "$CHIMERA_BASE TTT_KD_ALPHA=0.1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier20" || "$TARGET" == "V143" ]]; then
  # Norm bound + label smoothing compound: two orthogonal anti-degradation techniques.
  run_rp "V143_norm_smooth" \
    "$CHIMERA_BASE TTT_NORM_BUDGET=1.0 TTT_LABEL_SMOOTH=0.05"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier20" || "$TARGET" == "V144" ]]; then
  # KD + label smooth: full anti-forgetting stack (KD + calibration).
  run_rp "V144_kd_smooth" \
    "$CHIMERA_BASE TTT_KD_ALPHA=0.1 TTT_LABEL_SMOOTH=0.05"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier20" || "$TARGET" == "V145" ]]; then
  # Tier 20 compound: norm bound + KD + label smooth + larger chunk.
  run_rp "V145_tier20_compound" \
    "$CHIMERA_BASE \
     TTT_NORM_BUDGET=1.0 \
     TTT_KD_ALPHA=0.1 \
     TTT_LABEL_SMOOTH=0.05 \
     TTT_CHUNK_SIZE=2048"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier20" || "$TARGET" == "V146" ]]; then
  # Max Tier 20: V145 + best Tier 19 techniques (rsLoRA + LoRA+ + RELI + layer LR).
  # Full convergence + anti-forgetting + init alignment stack.
  run_rp "V146_max_tier20" \
    "$CHIMERA_BASE \
     TTT_RS_LORA=1 TTT_LORA_PLUS=1 TTT_LORA_PLUS_LAMBDA=4.0 \
     TTT_RELI=1 TTT_LAYER_LR=1 \
     TTT_NORM_BUDGET=1.0 \
     TTT_KD_ALPHA=0.1 \
     TTT_LABEL_SMOOTH=0.05 \
     TTT_CHUNK_SIZE=2048"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 21: Layer targeting — top-N restriction + MLP-only (V147–V151)
# Research: E2E-TTT, "Attention Retrieves MLP Memorizes", In-Place TTT (ICLR 2026)
# Expected: -0.02–0.06 BPB (layer restriction is high-confidence)
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier21" || "$TARGET" == "V147" ]]; then
  # Top-3 layer restriction: only layers 8-10 of 11 receive LoRA adapters.
  # E2E-TTT + FLoE: 25% layer adaptation recovers 93%+ of full-model quality.
  # Hypothesis: lower layers encode stable syntax → noise source in current setup.
  run_rp "V147_top3_layers" \
    "$CHIMERA_BASE TTT_TOP_LAYERS=3"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier21" || "$TARGET" == "V148" ]]; then
  # MLP-only TTT: freeze all attention Q/V in inner loop; adapt only MLP.
  # E2E-TTT key finding: attention updates destabilize inner loop.
  # "Attention Retrieves, MLP Memorizes": MLP is the TTT adaptation target.
  run_rp "V148_mlp_only" \
    "$CHIMERA_BASE TTT_MLP_ONLY=1 TTT_LORA_RANK_MLP=8"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier21" || "$TARGET" == "V149" ]]; then
  # In-Place TTT (ICLR 2026): W_down (proj) only — the MLP write-head.
  # Combines V148 (MLP-only) + V135 (proj-only). Minimum rank budget, maximum signal.
  run_rp "V149_wdown_only" \
    "$CHIMERA_BASE TTT_MLP_ONLY=1 TTT_PROJ_ONLY_MLP=1 TTT_LORA_RANK_QV=8"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier21" || "$TARGET" == "V150" ]]; then
  # Top-3 layers + MLP-only: most targeted TTT variant.
  # Only W_fc+W_proj in layers 8-10 get LoRA. All other params frozen.
  run_rp "V150_top3_mlp_only" \
    "$CHIMERA_BASE TTT_MLP_ONLY=1 TTT_LORA_RANK_MLP=8 TTT_TOP_LAYERS=3"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier21" || "$TARGET" == "V151" ]]; then
  # Max Tier 21: Top-3 + W_down only + RELI + rsLoRA + KD + norm bound.
  # Full research-validated stack from layer importance agent.
  run_rp "V151_max_tier21" \
    "$CHIMERA_BASE \
     TTT_MLP_ONLY=1 TTT_PROJ_ONLY_MLP=1 TTT_LORA_RANK_QV=8 \
     TTT_TOP_LAYERS=3 \
     TTT_RELI=1 TTT_RS_LORA=1 TTT_LORA_PLUS=1 \
     TTT_NORM_BUDGET=1.0 TTT_KD_ALPHA=0.1 TTT_LABEL_SMOOTH=0.05"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# TIER 22: Muon optimizer for LoRA adapters (V152–V154)
# Research: arXiv:2507.12142 (LoRA meets Riemannian), arXiv:2602.06385 (equal spectral growth)
# Base model trained with Muon — AdamW creates geometric mismatch for adapters.
# Expected: -0.002 to -0.006 BPB via rank collapse prevention
# ═══════════════════════════════════════════════════════════════════════════════

if [[ "$TARGET" == "all" || "$TARGET" == "tier22" || "$TARGET" == "V152" ]]; then
  # Muon for LoRA A+B adapters — Newton-Schulz orthogonalization, no DDP all-reduce.
  # Equal singular-value growth prevents rank collapse in multi-step TTT.
  run_rp "V152_muon_lora" \
    "$CHIMERA_BASE TTT_MUON=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier22" || "$TARGET" == "V153" ]]; then
  # Muon + RELI: RELI aligns init to gradient principal dirs; Muon keeps them orthogonal.
  run_rp "V153_muon_reli" \
    "$CHIMERA_BASE TTT_MUON=1 TTT_RELI=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier22" || "$TARGET" == "V155" ]]; then
  # Path-integral EWC Fisher (SI-style): accumulate v_t over ALL epochs, not just at convergence.
  # Fixes V132 critical bug: at convergence gradients ≈ 0 → Fisher ≈ 0 → EWC has no effect.
  # Accumulated Squisher (arXiv:2507.18807): running mean of Adam v_t captures peak signal.
  run_rp "V155_pi_ewc" \
    "$CHIMERA_BASE TTT_EWC_LAMBDA=0.05 TTT_EWC_PATH_INTEGRAL=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "tier22" || "$TARGET" == "V154" ]]; then
  # Max Tier 22: Muon + top-3 MLP-only + RELI + rsLoRA + norm bound + KD.
  # Full research-validated stack: geometry + layer targeting + convergence + anti-forgetting.
  run_rp "V154_max_tier22" \
    "$CHIMERA_BASE \
     TTT_MUON=1 \
     TTT_MLP_ONLY=1 TTT_PROJ_ONLY_MLP=1 TTT_LORA_RANK_QV=8 \
     TTT_TOP_LAYERS=3 \
     TTT_RELI=1 TTT_RS_LORA=1 \
     TTT_NORM_BUDGET=1.0 TTT_KD_ALPHA=0.1 TTT_LABEL_SMOOTH=0.05"
fi

# ═══════════════════════════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║  RUNPOD RESULTS SUMMARY — val_bpb (lower = better)             ║"
echo "║  Competition conditions: 8×H100, 600s, 524K batch/step         ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
printf "%-35s %-12s %-10s\n" "RUN_ID" "final_bpb" "step"
echo "────────────────────────────────────────────────────────────────────"
for log in logs/V*.txt; do
  [ -f "$log" ] || continue
  id=$(basename "$log" .txt)
  # Match actual log output lines (start with "step:")
  last=$(grep "^step:[0-9]" "$log" 2>/dev/null | grep "val_bpb:" | tail -1)
  [ -z "$last" ] && continue
  bpb=$(echo "$last" | grep -o "val_bpb:[0-9.]*" | sed "s/val_bpb://")
  step=$(echo "$last" | grep -o "step:[0-9]*/" | sed "s/step://;s/\///")
  [ -n "$bpb" ] && printf "%-35s %-12s %-10s\n" "$id" "$bpb" "$step"
done | sort -k2 -n
echo ""
echo "Leaderboard reference: 1.2244 (baseline) | 1.1233 (SOTA) | 1.0622 (PR#518 frontier)"
