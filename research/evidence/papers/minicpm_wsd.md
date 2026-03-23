---
title: "MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies"
source_url: https://arxiv.org/abs/2404.06395
date: 2024-04
category: paper
authors: Shengding Hu, Yuge Tu, Xu Han, Ganqu Cui, Chaoqun He, Weilin Zhao, Xiang Long, Zhi Zheng, Yewei Fang, Yuxiang Huang, Xinrong Zhang, Zheng Leng Thai, Kaihuo Zhang, Chongyi Wang, Yuan Yao, Chenyang Zhao, Jie Zhou, Jie Cai, Zhongwu Zhai, Ning Ding, Chao Jia, Guoyang Zeng, Dahai Li, Zhiyuan Liu, Maosong Sun (Tsinghua / ModelBest)
---

## Key Idea
MiniCPM introduces the Warmup-Stable-Decay (WSD) learning rate scheduler that enables efficient scaling law discovery and continuous training. The WSD scheduler splits training into three phases: warmup, stable (high LR), and decay. The decay phase causes a dramatic loss decrease, and intermediate checkpoints from the stable phase can be reused, enabling efficient data-model scaling law studies.

## Method Details
- **WSD Scheduler:**
  1. Warmup phase: standard linear warmup
  2. Stable phase: constant high learning rate -- loss decreases gradually
  3. Decay phase: rapid LR decay -- loss drops dramatically (the "free lunch")
- **Key insight:** The decay phase discovers how much latent quality the model has accumulated during stable training. Checkpoints from any point in the stable phase can be branched into decay.
- **Scaling law advantage:** Can study data-model scaling with linear effort (not quadratic) because you run one stable phase and branch multiple decay experiments
- **Derived higher compute-optimal data-model ratio than Chinchilla** -- small models benefit from more data relative to parameters than Chinchilla predicts
- MiniCPM 1.2B and 2.4B models match 7B-13B LLMs in capability

## Reported Results
- MiniCPM-2.4B matches Llama-2-7B and Mistral-7B on many benchmarks
- WSD scheduler enables efficient hyperparameter search
- Higher data-model ratio found optimal for small models vs. Chinchilla scaling

## Relevance to Parameter Golf
HIGHLY relevant for training schedule optimization:
1. The WSD scheduler could replace or complement current LR schedule -- the dramatic loss drop during decay is essentially free
2. For Parameter Golf: spend most of the 10 minutes in stable phase (high LR, fast exploration), then decay in the final 1-2 minutes for a big loss improvement
3. The finding that small models benefit from higher data-model ratios suggests training on MORE tokens (larger batch, more steps) rather than making the model larger
4. TENSION with scaling law paper: Kaplan says bigger model, MiniCPM says more data for small models. The optimal is likely somewhere in between, tuned by experiment.
5. SWA collection could happen during the stable phase, with final averaging + decay as the closing move

## Implementation Complexity
low

## Expected Impact
- Throughput: no
- Compressed size: no
- Post-quant loss: no
- Raw train loss: yes

## Key Takeaway for Implementation
Adopt a WSD learning rate schedule: run at high constant LR for ~80% of training time (stable phase), then aggressively decay LR for the final ~20%. This extracts maximum quality from the training budget. Combine with SWA by collecting snapshots during the stable phase.
