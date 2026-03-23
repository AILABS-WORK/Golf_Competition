---
title: "NanoGPT Speedrun: Training Optimization Techniques Compendium"
source_url: https://github.com/KellerJordan/modded-nanogpt
date: 2024-2026 (ongoing)
category: blog/leaderboard
authors: Keller Jordan, community contributors
---

## Key Idea
The NanoGPT speedrun is a competitive benchmark to train GPT-2 (124M) to 3.28 val loss on FineWeb as fast as possible on 8xH100. The leaderboard has driven innovations in optimizer design, architecture, and training efficiency that are directly applicable to Parameter Golf.

## Method Details
Key techniques from the speedrun community:
- **Muon optimizer:** ~35% speedup over AdamW for hidden layer weights
- **Trapezoidal LR schedule:** Similar to WSD; high constant LR then linear decay
- **Logit softcap:** Prevents logit explosion, stabilizes training
- **ReLU-squared activation:** Better than GELU for small models
- **Rotary Position Embeddings (RoPE):** More parameter-efficient than learned positional embeddings
- **QK-Norm:** Normalizes query and key vectors before attention, preventing attention entropy collapse
- **Architecture modernizations:** Pre-norm, removed bias terms, tied embeddings
- **Test-time training (TTT):** Parameter nudging -- updating model on early tokens of a document improves predictions on later tokens (from @samacqua, Jan 2026)
- **TokenMonster tokenizer:** Better tokenization can improve BPB but may be outside Parameter Golf rules

## Reported Results
- Record as of Jan 2025: 3.14 minutes (then 2.77 min with TokenMonster, outside rules)
- Muon alone dropped time from ~7 min to ~4.53 hours for 1.5B (different benchmark)
- Techniques compound: each innovation stacks with others

## Relevance to Parameter Golf
DIRECTLY relevant -- many Parameter Golf competitors come from the NanoGPT speedrun community:
1. Muon optimizer: already in use
2. Trapezoidal/WSD LR schedule: should verify current schedule matches this pattern
3. ReLU-squared: check if current activation function is optimal
4. QK-Norm: essential for training stability at speed
5. Logit softcap: prevents training instability
6. TTT/parameter nudging: novel idea -- could improve eval BPB by adapting at test time within the sliding window evaluation. This is a potential breakthrough if allowed by competition rules.
7. RoPE: more parameter-efficient positional encoding

## Implementation Complexity
low-medium (individual techniques are simple, but composing them requires care)

## Expected Impact
- Throughput: yes
- Compressed size: no
- Post-quant loss: no
- Raw train loss: yes

## Key Takeaway for Implementation
Audit the current training setup against the NanoGPT speedrun best practices: ensure Muon, trapezoidal/WSD LR, QK-Norm, logit softcap, and ReLU-squared are all in use. Investigate test-time training (parameter nudging) as a potential BPB improvement at evaluation time.
