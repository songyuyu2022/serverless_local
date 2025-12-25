# picture.py
# -*- coding: utf-8 -*-
"""
Paper-ready minimal figures for ICWS (Improved Version).
Run (PowerShell):
  python .\picture.py --csv .\metrics.csv --out_dir .\figures_icws --col 1 --smooth 10 --fmt pdf,png
"""

import os
import argparse
import math
import numpy as np
import pandas as pd
import matplotlib as mlp
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


# -------------------------
# Paper style
# -------------------------
def set_paper_style(font="Times New Roman", base_fontsize=10):
    # Use a style that looks good in papers
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [font, "Times New Roman", "Times", "DejaVu Serif"],
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize + 1,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize - 1,
        "ytick.labelsize": base_fontsize - 1,
        "legend.fontsize": base_fontsize - 1,
        "figure.titlesize": base_fontsize + 2,
        "axes.linewidth": 0.8,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "lines.linewidth": 1.5,
        "grid.linewidth": 0.5,
        "grid.alpha": 0.5,
        "grid.linestyle": "--",
        "savefig.dpi": 300,
    })


def fig_size(col=1, aspect=0.618):  # Golden ratio
    # IEEE two-column typical widths: single ~3.5", double ~7.16"
    w = 3.5 if col == 1 else 7.16
    h = w * aspect
    return (w, h)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def rolling_mean(s: pd.Series, window: int):
    if window is None or window <= 1:
        return s
    return s.rolling(window=window, min_periods=1).mean()


def save_fig(fig, out_dir: str, name: str, fmt=("pdf", "png"), dpi=600):
    fig.tight_layout()
    for f in fmt:
        f = f.lower()
        path = os.path.join(out_dir, f"{name}.{f}")
        if f == "png":
            fig.savefig(path, dpi=dpi, bbox_inches="tight")
        else:
            fig.savefig(path, bbox_inches="tight")
        print(f"[saved] {path}")
    plt.close(fig)


def get_phase(df: pd.DataFrame, phase: str):
    if "phase" not in df.columns:
        return None
    sub = df[df["phase"].astype(str).str.lower() == phase.lower()].copy()
    return sub if len(sub) > 0 else None


def fallback_train(df: pd.DataFrame):
    train = get_phase(df, "train")
    if train is None:
        train = df.copy()
    return train


def to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


# -------------------------
# Plots
# -------------------------

COLORS = {
    'train': '#1f77b4',  # Blue
    'val': '#ff7f0e',  # Orange
    'loss': '#d62728',  # Red
    'ppl': '#2ca02c',  # Green
}


# -------------------------
# Plots (minimal set)
# -------------------------
def plot_val_loss_ppl(df, out_dir, x="step", col=1, smooth=10, dpi=600, fmt=("pdf", "png")):
    val = get_phase(df, "val")
    train = get_phase(df, "train")

    if val is None and train is None:
        print("[skip] loss/ppl: no phase found.")
        return

    fig = plt.figure(figsize=fig_size(col, aspect=0.78))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twinx()

    def _plot(sub, prefix):
        if sub is None or "loss" not in sub.columns:
            return
        loss = rolling_mean(sub["loss"], smooth)
        ppl = loss.apply(lambda v: math.exp(v) if pd.notna(v) else np.nan)
        ax1.plot(sub[x], loss, label=f"{prefix}-loss")
        ax2.plot(sub[x], ppl, linestyle="--", label=f"{prefix}-ppl")

    # 论文里更常用 val（但你可以同时画 train 做参考）
    _plot(train, "train")
    _plot(val, "val")

    ax1.set_xlabel(x)
    ax1.set_ylabel("Loss")
    ax2.set_ylabel("Perplexity (exp(loss))")
    ax1.set_title("Loss / PPL vs Step")
    ax1.grid(True)

    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, frameon=False, ncol=2)

    save_fig(fig, out_dir, "loss_ppl_vs_step", fmt=fmt, dpi=dpi)


def plot_acc5_only(df, out_dir, x="step", col=1, smooth=10, dpi=600, fmt=("pdf", "png")):
    if "acc_top5" not in df.columns:
        print("[skip] acc@5: acc_top5 not found.")
        return

    train = get_phase(df, "train")
    val = get_phase(df, "val")

    fig, ax = plt.subplots(figsize=fig_size(col))

    if train is not None:
        # Plot smoothed training accuracy
        ax.plot(train[x], rolling_mean(train["acc_top5"], smooth), color=COLORS['train'], label="Train Acc@5",
                linewidth=1.5)

    if val is not None:
        # Plot validation with markers and line
        ax.plot(val[x], val["acc_top5"], color=COLORS['val'], label="Val Acc@5", marker="o", markersize=2,
                linestyle="--", linewidth=1.5)

    ax.set_xlabel(x.capitalize())
    ax.set_ylabel("Accuracy")

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    # Set y-axis limits
    ax.set_ylim(bottom=0.0)
    max_val = df["acc_top5"].max()
    if max_val < 0.9:
        ax.set_ylim(top=1.0)

    ax.set_title("Top-5 Accuracy")
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='lower right', frameon=True)

    save_fig(fig, out_dir, "acc5_vs_step", fmt=fmt, dpi=dpi)


def plot_grad_mode_fractions(df, out_dir, x="step", col=1, smooth=10, dpi=600, fmt=("pdf", "png")):
    train = fallback_train(df)

    cols = ["grad_mode_hot_frac", "grad_mode_cold_frac", "grad_mode_http_frac"]
    cols = [c for c in cols if c in train.columns]

    if not cols:
        return

    fig, ax = plt.subplots(figsize=fig_size(col))

    label_map = {
        "grad_mode_hot_frac": "Hot",
        "grad_mode_cold_frac": "Cold",
        "grad_mode_http_frac": "HTTP",
    }

    xs = train[x]
    ys = [rolling_mean(train[c], smooth) for c in cols]
    labels = [label_map.get(c, c) for c in cols]

    stack_colors = [ '#ff7f0e', '#1f77b4', '#2ca02c']

    ax.stackplot(xs, *ys, labels=labels, colors=stack_colors, alpha=0.8)

    ax.set_xlabel(x.capitalize())
    ax.set_ylabel("Fraction")
    ax.set_title("Gradient Apply Modes")
    ax.set_ylim(0, 1)
    ax.margins(0, 0)
    ax.grid(True, linestyle='--', alpha=0.3)
    ax.legend(loc='center right', frameon=True, fontsize=8)

    save_fig(fig, out_dir, "grad_mode_fractions_vs_step", fmt=fmt, dpi=dpi)


def plot_cost_breakdown(df, out_dir, x="step", col=1, smooth=10, dpi=600, fmt=("pdf", "png")):
    train = fallback_train(df)

    comps = [
        "cost_usd_pre_fwd", "cost_usd_expert_fwd", "cost_usd_post_fwd",
        "cost_usd_pre_bwd", "cost_usd_post_bwd", "cost_usd_grad_apply"
    ]

    comp_labels = [
        "Pre-Fwd", "Expert Fwd", "Post-Fwd",
        "Pre-Bwd", "Post-Bwd", "Grad Apply"
    ]

    existing_comps = []
    existing_labels = []
    for c, l in zip(comps, comp_labels):
        if c in train.columns:
            existing_comps.append(c)
            existing_labels.append(l)

    if not existing_comps:
        return

    xs = train[x]
    ys = [rolling_mean(train[c].fillna(0.0), smooth) for c in existing_comps]

    fig, ax = plt.subplots(figsize=fig_size(col))

    try:
        pal = mlp.colormaps['Set2'].colors
    except AttributeError:
        pal = plt.cm.get_cmap('Set2').colors

    ax.stackplot(xs, *ys, labels=existing_labels, colors=pal, alpha=0.9)

    ax.set_xlabel(x.capitalize())
    ax.set_ylabel("Cost (USD / step)")
    ax.set_title("Cost Breakdown")
    ax.grid(True, linestyle='--', alpha=0.3)

    ax.legend(loc='upper left', frameon=True, fontsize=8, ncol=2)

    save_fig(fig, out_dir, "cost_breakdown_vs_step", fmt=fmt, dpi=dpi)


def plot_deadline(df, out_dir, x="step", col=1, smooth=10, dpi=600, fmt=("pdf", "png")):
    train = fallback_train(df)

    fig, ax1 = plt.subplots(figsize=fig_size(col))

    if "deadline_miss" in train.columns:
        miss = rolling_mean(train["deadline_miss"].fillna(0.0), smooth)
        ax1.plot(train[x], miss, color=COLORS['loss'], label="Miss Rate")
        ax1.set_ylabel("Miss Rate")
        ax1.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))

    if "deadline_slack_ms" in train.columns:
        ax2 = ax1.twinx()
        slack = rolling_mean(train["deadline_slack_ms"], smooth)
        ax2.plot(train[x], slack, color=COLORS['train'], linestyle="--", label="Slack (ms)")
        ax2.set_ylabel("Slack (ms)")
        ax2.axhline(0, color='gray', linestyle=':', linewidth=1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = (ax2.get_legend_handles_labels() if "deadline_slack_ms" in train.columns else ([], []))
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='best', frameon=True)

    ax1.set_xlabel(x.capitalize())
    ax1.set_title("Deadline Miss Rate & Slack")
    ax1.grid(True, linestyle='--', alpha=0.5)

    save_fig(fig, out_dir, "deadline_vs_step", fmt=fmt, dpi=dpi)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="metrics.csv")
    ap.add_argument("--out_dir", type=str, default="figures_icws")
    ap.add_argument("--col", type=int, default=1, choices=[1, 2])
    ap.add_argument("--smooth", type=int, default=10)
    ap.add_argument("--dpi", type=int, default=600)
    ap.add_argument("--fmt", type=str, default="pdf,png")
    ap.add_argument("--x", type=str, default="step", choices=["step", "epoch", "step_in_epoch"])
    ap.add_argument("--font", type=str, default="Times New Roman")
    ap.add_argument("--fontsize", type=int, default=10)
    args = ap.parse_args()

    set_paper_style(font=args.font, base_fontsize=args.fontsize)
    safe_mkdir(args.out_dir)

    df = pd.read_csv(args.csv)

    needed = [
        args.x, "loss", "acc_top5",
        "grad_mode_hot_frac", "grad_mode_cold_frac", "grad_mode_http_frac",
        "deadline_miss", "deadline_slack_ms",
        "cost_usd_pre_fwd", "cost_usd_post_fwd", "cost_usd_expert_fwd",
        "cost_usd_pre_bwd", "cost_usd_post_bwd", "cost_usd_grad_apply",
    ]
    to_numeric(df, needed)

    if args.x in df.columns:
        df = df.sort_values([args.x]).reset_index(drop=True)

    fmt = tuple([s.strip().lower() for s in args.fmt.split(",") if s.strip()])

    # Generate plots
    plot_val_loss_ppl(df, args.out_dir, x=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    plot_acc5_only(df, args.out_dir, x=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    plot_grad_mode_fractions(df, args.out_dir, x=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    plot_cost_breakdown(df, args.out_dir, x=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    plot_deadline(df, args.out_dir, x=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)


if __name__ == "__main__":
    main()