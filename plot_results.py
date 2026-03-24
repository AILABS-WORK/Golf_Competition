#!/usr/bin/env python3
"""
Parameter Golf — val_bpb Plotter & Token-Scaled Extrapolator
=============================================================
Plots val_bpb (the actual competition metric) vs training steps for all variants.
No smoothing — raw data points only. Power-law fit on val_bpb vs cumulative tokens.

Usage:
    python plot_results.py                # all V*.txt logs
    python plot_results.py A B            # groups by first letter prefix
    python plot_results.py V0 V8          # specific run IDs
    python plot_results.py --val V        # val_bpb mode, V* logs
    python plot_results.py --train V      # train_loss mode
    python plot_results.py --list         # table only, no chart
    python plot_results.py --list --val V # val_bpb table only

Competition reference:
    Best known: 1.1233 val_bpb (sliding window stride=64) at ~7,101 H100 steps
    = 7101 × 786432 = 5.58B tokens (8×H100, seq_len=2048, batch=786K)
    Our runs: each step = 8192 tokens (TRAIN_BATCH_TOKENS on single RTX 5080)
"""
from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import numpy as np

# ─── Constants ────────────────────────────────────────────────────────────────
LOGS_DIR = Path(__file__).parent / "logs"

# Competition reference point (sliding-window val_bpb, best known submission)
COMP_BPB_TARGET = 1.1233
# H100 8× runs: 7101 steps × 786,432 tokens/step (seq_len=2048, batch=384 seqs)
COMP_TOKENS = 7_101 * 786_432   # ≈ 5.58B tokens

# Our local runs
LOCAL_TOKENS_PER_STEP = 8_192   # TRAIN_BATCH_TOKENS in benchmark runs

# Leaderboard milestones (val_bpb, label) — merged competition scores + unmerged frontier
MILESTONES = [
    (1.2244, "NaiveBaseline"),
    (1.1928, "SlidingWindowEval"),
    (1.1458, "Int6+MLP3x+SWA"),
    (1.1428, "10L+Int5+SWA0.4"),
    (1.1307, "11L+XSA4"),
    (1.1271, "11L+XSA4+EMA"),
    (1.1248, "11L+PartRoPE+LNScale"),
    (1.1233, "★ SOTA: GPTQ-lite+EMA (1.1233)"),
    # Unmerged PRs (frontier) — dashed reference
    (1.1027, "PR#442: EMA+TTT10ep"),
    (1.0891, "PR#490: VR+GatedAttn+TTT"),
    (1.0622, "PR#518: LeakyReLU²+TTT50ep"),
]

# ─── Log Parsing ──────────────────────────────────────────────────────────────

def parse_log(path: Path) -> dict:
    """
    Parse a train_gpt.py log file.
    Returns:
        steps:    array of logged training steps
        tokens:   cumulative tokens at each logged step
        val_bpb:  val_bpb values (None where not logged)
        train_loss: train_loss values at each logged step
        ema_loss:   ema_loss values (None if not logged)
    """
    steps, tokens_arr, val_bpb_arr, train_loss_arr, ema_arr = [], [], [], [], []
    for line in path.read_text(errors="replace").splitlines():
        # val_bpb line: "step:N/M val_loss:X val_bpb:Y ..."
        mv = re.search(r"^step:(\d+)/\d+\s+val_loss:[\d.]+\s+val_bpb:([\d.]+)", line)
        if mv:
            s = int(mv.group(1))
            bpb = float(mv.group(2))
            steps.append(s)
            tokens_arr.append(s * LOCAL_TOKENS_PER_STEP)
            val_bpb_arr.append(bpb)
            train_loss_arr.append(None)
            ema_arr.append(None)
            continue
        # train_loss line: "step:N/M train_loss:X ema_loss:Y ..."
        mt = re.search(r"^step:(\d+)/\d+\s+train_loss:([\d.]+)(?:\s+ema_loss:([\d.]+))?", line)
        if mt:
            s = int(mt.group(1))
            steps.append(s)
            tokens_arr.append(s * LOCAL_TOKENS_PER_STEP)
            val_bpb_arr.append(None)
            train_loss_arr.append(float(mt.group(2)))
            ema_arr.append(float(mt.group(3)) if mt.group(3) else None)

    return {
        "steps": np.array(steps, dtype=float),
        "tokens": np.array(tokens_arr, dtype=float),
        "val_bpb": val_bpb_arr,        # list with None entries
        "train_loss": train_loss_arr,   # list with None entries
        "ema_loss": ema_arr,
    }


def collect_runs(targets: list[str]) -> dict[str, dict]:
    all_logs = sorted(LOGS_DIR.glob("*.txt"))
    runs = {}
    for log in all_logs:
        name = log.stem
        include = any(
            t == "all"
            or (len(t) == 1 and name.startswith(t))
            or (len(t) <= 3 and t.isupper() and name.startswith(t))
            or name.startswith(t)
            for t in targets
        )
        if include:
            data = parse_log(log)
            if len(data["steps"]) > 0:
                runs[name] = data
    return runs


# ─── Power-Law Fitting (token-scaled) ─────────────────────────────────────────

def fit_power_law_tokens(tokens: np.ndarray, bpb: np.ndarray, min_tokens: int = 500_000) -> tuple:
    """
    Fit BPB = a * tokens^(-b) using log-linear regression.
    Returns (a, b) or (nan, nan) if insufficient data.
    min_tokens: skip early noisy points (warmup region).
    """
    mask = (tokens >= min_tokens) & np.isfinite(bpb)
    if mask.sum() < 2:
        return float("nan"), float("nan")
    x = np.log(tokens[mask])
    y = np.log(bpb[mask])
    coeffs = np.polyfit(x, y, 1)
    b = -coeffs[0]
    a = math.exp(coeffs[1])
    return a, b


def extrapolate_to_tokens(a: float, b: float, target_tokens: int) -> float:
    if math.isnan(a):
        return float("nan")
    return a * (target_tokens ** -b)


# ─── Summary Table ────────────────────────────────────────────────────────────

def get_val_series(data: dict) -> tuple[np.ndarray, np.ndarray]:
    """Extract (tokens, val_bpb) where val_bpb is not None."""
    toks = [data["tokens"][i] for i, v in enumerate(data["val_bpb"]) if v is not None]
    bpbs = [v for v in data["val_bpb"] if v is not None]
    return np.array(toks, dtype=float), np.array(bpbs, dtype=float)


def print_summary(runs: dict[str, dict], mode: str = "val"):
    """Print ranked summary table."""
    rows = []
    baseline_extrap = None

    for name, d in runs.items():
        if len(d["steps"]) == 0:
            continue

        if mode == "val":
            tok_arr, bpb_arr = get_val_series(d)
            if len(bpb_arr) == 0:
                final_metric = float("nan")
                extrap = float("nan")
            else:
                final_metric = float(bpb_arr[-1])
                a, b = fit_power_law_tokens(tok_arr, bpb_arr)
                extrap = extrapolate_to_tokens(a, b, COMP_TOKENS)
        else:
            train_vals = [v for v in d["train_loss"] if v is not None]
            ema_vals = [v for v in d["ema_loss"] if v is not None]
            final_metric = float(ema_vals[-1]) if ema_vals else (float(train_vals[-1]) if train_vals else float("nan"))
            extrap = float("nan")

        final_step = int(d["steps"][-1]) if len(d["steps"]) > 0 else 0
        rows.append((name, final_step, final_metric, extrap))

        if name in ("V0_baseline", "A0_baseline"):
            baseline_extrap = extrap

    rows.sort(key=lambda r: r[2] if not math.isnan(r[2]) else 99)

    metric_label = "val_bpb@N" if mode == "val" else "ema_loss@N"

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    print()
    print("=" * 90)
    print(f"  Parameter Golf Results ({metric_label}, lower = better)")
    print(f"  Competition target: {COMP_BPB_TARGET} val_bpb ({COMP_TOKENS/1e9:.2f}B tokens on H100 8x)")
    print("=" * 90)
    print(f"{'RUN':<34} {'steps':>6} {metric_label:>12} {'extrap_7K_H100':>15} {'delta':>10}")
    print("-" * 90)
    for name, final_step, final_metric, extrap in rows:
        delta = (extrap - baseline_extrap) if (baseline_extrap and not math.isnan(extrap) and not math.isnan(baseline_extrap)) else float("nan")
        extrap_s = f"{extrap:.4f}" if not math.isnan(extrap) else "      n/a"
        metric_s = f"{final_metric:.4f}" if not math.isnan(final_metric) else "     n/a"
        delta_s = f"{delta:+.4f}" if not math.isnan(delta) else "      n/a"
        win = " **" if not math.isnan(extrap) and extrap < COMP_BPB_TARGET else ""
        beat = " *" if not math.isnan(delta) and delta < -0.005 else ""
        print(f"{name:<34} {final_step:>6} {metric_s:>12} {extrap_s:>15} {delta_s:>10}{win}{beat}")
    print()
    if baseline_extrap and not math.isnan(baseline_extrap):
        print(f"  Baseline extrapolated to competition scale: {baseline_extrap:.4f}")
        print(f"  Competition best known: {COMP_BPB_TARGET}   Gap: {baseline_extrap - COMP_BPB_TARGET:+.4f}")
        print(f"  * = better than baseline by >0.005   ** = beats competition target!")
    print()
    if mode == "val":
        print("  Milestones from local competition records:")
        for bpb, label in MILESTONES:
            print(f"    {bpb:.4f}  {label}")
    print()


# ─── Plotting ─────────────────────────────────────────────────────────────────

def plot(runs: dict[str, dict], mode: str = "val"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("matplotlib not available. Run: pip install matplotlib")
        print_summary(runs, mode)
        return

    NRUNS = len(runs)
    colors = cm.tab20(np.linspace(0, 1, max(NRUNS, 1)))

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    ax_curve, ax_bar = axes

    bar_names, bar_finals, bar_extrap = [], [], []
    all_final_metrics = []

    for i, (name, d) in enumerate(sorted(runs.items())):
        color = colors[i % len(colors)]

        if mode == "val":
            tok_arr, bpb_arr = get_val_series(d)
            if len(bpb_arr) == 0:
                continue
            # Plot raw val_bpb (no smoothing — as requested)
            ax_curve.plot(tok_arr / 1e6, bpb_arr, "o-", label=name, color=color,
                          linewidth=1.8, markersize=5, alpha=0.9)
            # Power-law fit extrapolation
            a, b = fit_power_law_tokens(tok_arr, bpb_arr)
            if not math.isnan(a):
                ext_toks = np.linspace(tok_arr[max(0, len(tok_arr)//2)],
                                       tok_arr[-1] * 3, 300)
                ext_bpb = a * ext_toks ** -b
                ax_curve.plot(ext_toks / 1e6, ext_bpb, "--", color=color, alpha=0.35, linewidth=1.0)
                extrap_val = extrapolate_to_tokens(a, b, COMP_TOKENS)
            else:
                extrap_val = float("nan")
            final_val = float(bpb_arr[-1])
        else:
            # train_loss mode
            train_idx = [i_ for i_, v in enumerate(d["train_loss"]) if v is not None]
            ema_idx = [i_ for i_, v in enumerate(d["ema_loss"]) if v is not None]
            if not train_idx:
                continue
            toks_t = d["tokens"][train_idx] / 1e6
            raw_t = np.array([d["train_loss"][i_] for i_ in train_idx])
            ax_curve.plot(toks_t, raw_t, alpha=0.2, color=color, linewidth=0.7)
            if ema_idx:
                toks_e = d["tokens"][ema_idx] / 1e6
                ema_t = np.array([d["ema_loss"][i_] for i_ in ema_idx])
                ax_curve.plot(toks_e, ema_t, label=name, color=color, linewidth=1.8)
                final_val = float(ema_t[-1])
            else:
                final_val = float(raw_t[-1])
            extrap_val = float("nan")

        bar_names.append(name)
        bar_finals.append(final_val)
        bar_extrap.append(extrap_val)
        all_final_metrics.append(final_val)

    # ── Learning curve axes ──
    y_label = "val_bpb (raw, no smoothing)" if mode == "val" else "train loss (EMA)"
    ax_curve.set_xlabel("Cumulative Tokens (millions)", fontsize=12)
    ax_curve.set_ylabel(y_label, fontsize=12)
    ax_curve.set_title("Learning Curves — raw data points (solid) + power-law extrapolation (dashed)", fontsize=11)
    ax_curve.legend(fontsize=7, ncol=2, loc="upper right")
    ax_curve.grid(True, alpha=0.3)

    if mode == "val" and all_final_metrics:
        y_min = min(all_final_metrics) - 0.05
        y_max = max(all_final_metrics) + 0.1
        ax_curve.set_ylim(y_min, y_max)
        # Draw competition milestone lines
        for bpb_ref, label in MILESTONES:
            if y_min <= bpb_ref <= y_max:
                ax_curve.axhline(bpb_ref, color="red", alpha=0.3, linewidth=0.8, linestyle=":")
                ax_curve.text(ax_curve.get_xlim()[0], bpb_ref, f" {label}", fontsize=6,
                              color="red", alpha=0.7, va="center")
        # Marker for competition token target
        ax_curve.axvline(COMP_TOKENS / 1e6, color="darkred", alpha=0.4, linewidth=1.2,
                         linestyle="--", label=f"H100 7K steps ({COMP_TOKENS/1e9:.2f}B tok)")

    # ── Bar chart axes ──
    if bar_names:
        sorted_idx = np.argsort([v if not math.isnan(v) else 99 for v in bar_finals])
        s_names = [bar_names[i] for i in sorted_idx]
        s_finals = [bar_finals[i] for i in sorted_idx]
        s_extrap = [bar_extrap[i] for i in sorted_idx]
        bar_colors = [colors[list(sorted(runs.keys())).index(n) % len(colors)]
                      if n in runs else "gray" for n in s_names]

        x_pos = np.arange(len(s_names))
        ax_bar.barh(x_pos, s_finals, color=bar_colors, alpha=0.8)
        ax_bar.set_yticks(x_pos)
        ax_bar.set_yticklabels(s_names, fontsize=8)
        ax_bar.set_xlabel(f"Final {y_label} (lower = better)", fontsize=12)
        ax_bar.set_title(f"Ranking @ step 3000 | diamonds = extrapolated to 7K H100 steps", fontsize=11)

        # Extrapolation diamonds
        for j, (xe, nm) in enumerate(zip(s_extrap, s_names)):
            if not math.isnan(xe):
                ax_bar.plot(xe, j, "D", color="black", markersize=5, alpha=0.8)

        if mode == "val":
            for bpb_ref, label in MILESTONES:
                ax_bar.axvline(bpb_ref, color="red", alpha=0.3, linewidth=0.8, linestyle=":")
            ax_bar.axvline(COMP_BPB_TARGET, color="darkred", linewidth=1.5, linestyle="--",
                           label=f"competition target {COMP_BPB_TARGET}")
            ax_bar.legend(fontsize=8)

        ax_bar.grid(True, alpha=0.3, axis="x")

    plt.tight_layout()
    mode_tag = "val_bpb" if mode == "val" else "train_loss"
    out_path = LOGS_DIR.parent / f"results_{mode_tag}.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\n  Chart saved -> {out_path}\n")
    print_summary(runs, mode)


# ─── Entry Point ──────────────────────────────────────────────────────────────

def main():
    args = sys.argv[1:]
    mode = "val"
    list_only = False
    targets = []

    for a in args:
        if a == "--val":
            mode = "val"
        elif a == "--train":
            mode = "train"
        elif a == "--list":
            list_only = True
        else:
            targets.append(a)

    if not targets:
        targets = ["V", "A0_baseline"]  # default: all benchmark + reference baseline

    runs = collect_runs(targets)
    if not runs:
        print(f"No logs found in {LOGS_DIR} matching: {targets}")
        return

    print(f"  Loaded {len(runs)} runs: {', '.join(sorted(runs))}")

    if list_only:
        print_summary(runs, mode)
    else:
        plot(runs, mode)


if __name__ == "__main__":
    main()
