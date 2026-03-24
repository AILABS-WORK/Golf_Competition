#!/bin/bash
# Safe local smoke tests — laptop-friendly (RTX 5080, 16GB VRAM, 64GB RAM)
# Modes:
#   quick  (default) — 1000 steps, ~7 min/run, for fast screening
#   medium           — 2000 steps, ~14 min/run, for better EMA + power-law fit
#
# Run from: parameter-golf/ directory
# Usage: bash run_smoke_tests.sh [--medium] [group|id|all]
#   bash run_smoke_tests.sh A              # group A, 1000 steps
#   bash run_smoke_tests.sh --medium A     # group A, 2000 steps
#   bash run_smoke_tests.sh A2             # single test
#   bash run_smoke_tests.sh all            # everything, 1000 steps
#   bash run_smoke_tests.sh --medium all   # everything, 2000 steps (long!)

set -e

PYTHON="/c/Python314/python.exe"

# Parse --medium flag
MODE="quick"
ARGS=()
for arg in "$@"; do
  if [[ "$arg" == "--medium" ]]; then
    MODE="medium"
  else
    ARGS+=("$arg")
  fi
done

if [[ "$MODE" == "medium" ]]; then
  ITERATIONS=2000
  TRAIN_LOG_EVERY=20   # denser logging = smoother EMA curve
  echo "Mode: MEDIUM (2000 steps, log every 20)"
else
  ITERATIONS=1000
  TRAIN_LOG_EVERY=50
  echo "Mode: QUICK (1000 steps, log every 50)"
fi

BASE_VARS="TORCHDYNAMO_DISABLE=1 \
  DATA_PATH=./data/datasets/fineweb10B_sp1024 \
  TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
  VOCAB_SIZE=1024 \
  ITERATIONS=$ITERATIONS \
  TRAIN_BATCH_TOKENS=8192 \
  SKIP_ALL_VAL=1 \
  TRAIN_LOG_EVERY=$TRAIN_LOG_EVERY"

run_smoke() {
  local run_id=$1
  local extra_vars=$2
  # In medium mode, suffix run_id to avoid overwriting quick results
  if [[ "$MODE" == "medium" ]]; then
    run_id="${run_id}_m"
  fi
  echo ""
  echo "═══════════════════════════════════════"
  echo "  STARTING: $run_id  [mode=$MODE, iters=$ITERATIONS]"
  echo "  Extra: $extra_vars"
  echo "═══════════════════════════════════════"
  eval "env $BASE_VARS RUN_ID=$run_id $extra_vars $PYTHON train_gpt.py"
  local result=$(grep "^step:${ITERATIONS}" "logs/$run_id.txt" 2>/dev/null | grep -o "train_loss:[0-9.]*" | sed "s/train_loss://")
  local ema=$(grep "^step:${ITERATIONS}" "logs/$run_id.txt" 2>/dev/null | grep -o "ema_loss:[0-9.]*" | sed "s/ema_loss://")
  echo "  RESULT: $run_id → step:${ITERATIONS} train_loss=$result ema_loss=$ema (baseline=3.4587)"
  echo ""
}

TARGET="${ARGS[0]:-all}"

echo "Parameter Golf Smoke Tests"
echo "Baseline A0: raw=3.4587 (step:1000)"
echo "Target: $TARGET"

# ─────────────────────────────────────────────────────
# GROUP A: Env-var only, no code changes
# ─────────────────────────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "A" || "$TARGET" == "A0" ]]; then
  run_smoke "A0_baseline" ""
fi

if [[ "$TARGET" == "all" || "$TARGET" == "A" || "$TARGET" == "A1" ]]; then
  run_smoke "A1_smear" "SMEARGATE=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "A" || "$TARGET" == "A2" ]]; then
  run_smoke "A2_bigram" "BIGRAM_HASH_BUCKETS=10240 BIGRAM_HASH_DIM=128"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "A" || "$TARGET" == "A3" ]]; then
  run_smoke "A3_ortho" "ORTHO_INIT=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "A" || "$TARGET" == "A5" ]]; then
  run_smoke "A5_10L" "NUM_LAYERS=10"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "A" || "$TARGET" == "A6" ]]; then
  run_smoke "A6_10L_muonWD" "NUM_LAYERS=10 MUON_WEIGHT_DECAY=0.04"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "A" || "$TARGET" == "A7" ]]; then
  run_smoke "A7_mlp3x" "MLP_MULT=3"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "A" || "$TARGET" == "A8" ]]; then
  run_smoke "A8_everything" \
    "SMEARGATE=1 BIGRAM_HASH_BUCKETS=10240 BIGRAM_HASH_DIM=128 ORTHO_INIT=1 NUM_LAYERS=10 MUON_WEIGHT_DECAY=0.04 MLP_MULT=3"
fi

# ─────────────────────────────────────────────────────
# GROUP B: Architecture/training variants
# ─────────────────────────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "B" || "$TARGET" == "B1" ]]; then
  run_smoke "B1_wsd_lr" "WSD_LR=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "B" || "$TARGET" == "B2" ]]; then
  run_smoke "B2_value_residual" "VALUE_RESIDUAL=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "B" || "$TARGET" == "B3" ]]; then
  run_smoke "B3_trigram" "TRIGRAM_HASH_BUCKETS=10240 TRIGRAM_HASH_DIM=64"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "B" || "$TARGET" == "B4" ]]; then
  run_smoke "B4_ste_qat" "STE_QAT=1 QAT_START_FRACTION=0.0 QUANT_BITS=6"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "B" || "$TARGET" == "B5" ]]; then
  run_smoke "B5_swa" "SWA=1 SWA_START_FRACTION=0.6"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "B" || "$TARGET" == "B6" ]]; then
  run_smoke "B6_mole4" "MOLE_NUM_EXPERTS=4 MOLE_DIM=64"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "B" || "$TARGET" == "B7" ]]; then
  run_smoke "B7_tweo" "TWEO_LAMBDA=0.0001"
fi

# ─────────────────────────────────────────────────────
# GROUP C: Stacked combos (best A + best B techniques)
# ─────────────────────────────────────────────────────
if [[ "$TARGET" == "all" || "$TARGET" == "C" || "$TARGET" == "C1" ]]; then
  run_smoke "C1_bigram_vr" \
    "BIGRAM_HASH_BUCKETS=10240 BIGRAM_HASH_DIM=128 VALUE_RESIDUAL=1"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "C" || "$TARGET" == "C2" ]]; then
  run_smoke "C2_bigram_vr_swa" \
    "BIGRAM_HASH_BUCKETS=10240 BIGRAM_HASH_DIM=128 VALUE_RESIDUAL=1 SWA=1 SWA_START_FRACTION=0.6"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "C" || "$TARGET" == "C3" ]]; then
  run_smoke "C3_bigram_trigram" \
    "BIGRAM_HASH_BUCKETS=10240 BIGRAM_HASH_DIM=128 TRIGRAM_HASH_BUCKETS=10240 TRIGRAM_HASH_DIM=64"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "C" || "$TARGET" == "C4" ]]; then
  run_smoke "C4_mole_vr_swa" \
    "MOLE_NUM_EXPERTS=4 MOLE_DIM=64 VALUE_RESIDUAL=1 SWA=1 SWA_START_FRACTION=0.6"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "C" || "$TARGET" == "C5" ]]; then
  run_smoke "C5_full_stack" \
    "SMEARGATE=1 BIGRAM_HASH_BUCKETS=10240 BIGRAM_HASH_DIM=128 TRIGRAM_HASH_BUCKETS=10240 TRIGRAM_HASH_DIM=64 ORTHO_INIT=1 NUM_LAYERS=10 MUON_WEIGHT_DECAY=0.04 MLP_MULT=3 VALUE_RESIDUAL=1 SWA=1 SWA_START_FRACTION=0.6 WSD_LR=1"
fi

# ─────────────────────────────────────────────────────
# NOVEL PAPER VARIANTS (individual isolation tests)
# ─────────────────────────────────────────────────────
if [[ "$TARGET" == "novel" || "$TARGET" == "N1" ]]; then
  # ResFormer / Value Residual (arXiv:2410.17897)
  run_smoke "N1_value_residual" "VALUE_RESIDUAL=1"
fi

if [[ "$TARGET" == "novel" || "$TARGET" == "N2" ]]; then
  # MoLE 4-expert (arXiv:2503.15798)
  run_smoke "N2_mole4" "MOLE_NUM_EXPERTS=4 MOLE_DIM=64"
fi

if [[ "$TARGET" == "novel" || "$TARGET" == "N3" ]]; then
  # MoLE 8-expert larger
  run_smoke "N3_mole8" "MOLE_NUM_EXPERTS=8 MOLE_DIM=64"
fi

if [[ "$TARGET" == "novel" || "$TARGET" == "N4" ]]; then
  # TrigramHash (from top unmerged PR)
  run_smoke "N4_trigram" "TRIGRAM_HASH_BUCKETS=10240 TRIGRAM_HASH_DIM=64"
fi

if [[ "$TARGET" == "novel" || "$TARGET" == "N5" ]]; then
  # STE QAT from step 0 with int6 (arXiv:2509.22935)
  run_smoke "N5_ste_qat_int6" "STE_QAT=1 QAT_START_FRACTION=0.0 QUANT_BITS=6"
fi

if [[ "$TARGET" == "novel" || "$TARGET" == "N6" ]]; then
  # SWA starting at 60% (proven in top-3 competition entries)
  run_smoke "N6_swa" "SWA=1 SWA_START_FRACTION=0.6"
fi

if [[ "$TARGET" == "novel" || "$TARGET" == "N7" ]]; then
  # WSD cosine LR schedule (arXiv:2507.17634 WSM-inspired)
  run_smoke "N7_wsd_lr" "WSD_LR=1"
fi

if [[ "$TARGET" == "novel" || "$TARGET" == "N8" ]]; then
  # TWEO colinearity penalty (arXiv:2511.23225)
  run_smoke "N8_tweo" "TWEO_LAMBDA=0.0001"
fi

if [[ "$TARGET" == "novel" || "$TARGET" == "N9" ]]; then
  # Value Residual + MoLE stack
  run_smoke "N9_vr_mole" "VALUE_RESIDUAL=1 MOLE_NUM_EXPERTS=4 MOLE_DIM=64"
fi

if [[ "$TARGET" == "novel" || "$TARGET" == "N10" ]]; then
  # Full novel stack: VR + MoLE + SWA + WSD
  run_smoke "N10_novel_stack" \
    "VALUE_RESIDUAL=1 MOLE_NUM_EXPERTS=4 MOLE_DIM=64 SWA=1 SWA_START_FRACTION=0.6 WSD_LR=1 TWEO_LAMBDA=0.0001"
fi

# ─────────────────────────────────────────────────────
# SUMMARY (bash quick table + python chart)
# ─────────────────────────────────────────────────────
echo ""
echo "═══════════════════════════════════════════════════════════════════"
echo "SUMMARY — final-step train_loss / ema_loss (lower = better)"
echo "Baseline ref: A0_baseline  raw=3.4587"
echo "═══════════════════════════════════════════════════════════════════"
printf "%-35s %-12s %-12s\n" "RUN_ID" "train_loss" "ema_loss"
echo "-------------------------------------------------------------------"
for log in logs/A*.txt logs/B*.txt logs/C*.txt logs/N*.txt; do
  [ -f "$log" ] || continue
  id=$(basename "$log" .txt)
  # get last step:N line (handles both 1000 and 2000 step runs)
  last_line=$(grep "^step:[0-9]*/[0-9]* train_loss" "$log" 2>/dev/null | tail -1)
  [ -z "$last_line" ] && continue
  raw=$(echo "$last_line" | grep -o "train_loss:[0-9.]*" | sed "s/train_loss://")
  ema=$(echo "$last_line" | grep -o "ema_loss:[0-9.]*" | sed "s/ema_loss://")
  [ -z "$ema" ] && ema="(no ema)"
  printf "%-35s %-12s %-12s\n" "$id" "$raw" "$ema"
done
echo ""
# Generate chart if matplotlib available
if $PYTHON -c "import matplotlib" 2>/dev/null; then
  echo "Generating results chart..."
  $PYTHON plot_results.py --list
  $PYTHON plot_results.py all
else
  echo "(install matplotlib for charts: pip install matplotlib)"
fi
echo "Extrapolation note: ~4000 iterations on H100 clusters approximates competition scale."
echo "Rank techniques by delta vs A0_baseline to identify best H100 candidates."
