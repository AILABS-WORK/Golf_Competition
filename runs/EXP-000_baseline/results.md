# EXP-000: Surrogate Baseline Results

**Date:** 2026-03-23
**Environment:** Local laptop, RTX 5080 (hybrid mode — GPU NOT used), CPU-bound eager mode
**Script:** parameter-golf/train_gpt.py (unmodified baseline)

## Config
- TRAIN_BATCH_TOKENS=65536 (reduced 8x for CPU speed)
- WARMUP_STEPS=2
- MAX_WALLCLOCK_SECONDS=300 (5 min)
- VAL_LOSS_EVERY=25
- TORCHDYNAMO_DISABLE=1 (Triton unavailable on Windows)

## Results
| Step | train_loss | val_loss | val_bpb | train_time |
|------|-----------|----------|---------|------------|
| 0 | — | 6.9357 | 4.1077 | 0ms |
| 25 | 5.6949 | 5.6448 | 3.3432 | 44.7s |
| 50 | 5.1461 | 5.1385 | 3.0433 | 84.4s |
| 75 | 4.5821 | 4.5625 | 2.7022 | 123.9s |
| 100 | 4.3874 | 4.3524 | 2.5777 | 162.3s |
| 125 | 4.2831 | 4.2340 | 2.5076 | 199.8s |
| 150 | 4.1163 | 4.1458 | 2.4554 | 239.3s |
| 175 | 4.1136 | 4.0810 | 2.4170 | 279.2s |
| 188 | — | 4.0706 | 2.4108 | 300.2s |

**Final val_bpb: 2.4108** (188/20000 steps, wallclock cap)
**Artifact size:** 7.18MB (int8+zlib), well under 16MB
**Peak memory:** 5,980 MiB allocated, 9,774 MiB reserved

## Notes
- Training was CPU-bound (hybrid GPU mode, 0% GPU utilization)
- Only 188 of 20,000 steps completed (<1%)
- This is NOT representative of 8xH100 performance
- Official baseline achieves 1.2244 val_bpb with full 20K steps on 8xH100
- Quantization gap NOT measured (process killed before roundtrip eval)
