# Leaderboard Analysis Complete
**Timestamp: 2026-03-23**

## Files Generated

1. **leaderboard_analysis.md** -- Full quantitative analysis including:
   - Complete leaderboard table (17 merged entries ranked by val_bpb)
   - Unmerged frontier PRs (5 entries, best at 1.0887)
   - Technique frequency analysis across top 10 entries
   - Improvement attribution with marginal BPB drops
   - Frontier analysis (#1 vs #2 comparison, theoretical optimum)
   - Statistical summary of the leaderboard
   - Visualization recommendations

2. **community_insights.md** -- Community discussion findings including:
   - Official OpenAI guidance on expected techniques
   - Community technique evolution (4 phases)
   - Key findings on quantization gaps, SmearGate, BigramHash scaling, SWA tuning
   - TTT deep dive (4 variants: LoRA, AdamW, Cosine, Score-First)
   - Competition meta-strategy insights
   - Recommended next steps

## Key Numbers

- **Merged SOTA**: 1.14276 val_bpb (thwu1, 10L Int5-MLP + BigramHash(10240) + SWA)
- **Unmerged SOTA**: ~1.0887 val_bpb (ndokutovich, 11L TrigramHash + ValueResidual + GradQuant + Cosine TTT)
- **Baseline**: 1.22437 val_bpb (OpenAI Naive Baseline)
- **Total merged improvement**: 0.0816 BPB (6.7% reduction)
- **Total frontier improvement**: 0.1357 BPB (11.1% reduction)

## Data Sources

- GitHub: openai/parameter-golf (main branch, 17 merged record directories)
- GitHub: open pull requests (#486, #490, #492, #493, #503)
- Web: OpenAI official page, The Decoder, DeepWiki, community discussions
- All submission.json and README.md files from top entries
