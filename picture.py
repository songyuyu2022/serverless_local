# picture.py
# -*- coding: utf-8 -*-
"""
ICWS / IEEE SERVICES-style (two-column) paper-ready plotting for metrics.csv.

Usage (Windows PowerShell):
  python .\picture.py --csv .\metrics.csv --out_dir .\figures_icws --col 1 --smooth 5

Outputs:
  figures_icws/*.pdf   (vector, best for papers)
  figures_icws/*.png   (high-res, default 600 dpi)

Key options:
  --col 1              single-column width (~3.5 in)
  --col 2              double-column width (~7.16 in)
  --smooth 5           rolling mean smoothing window (1 = no smoothing)
  --x step|epoch|step_in_epoch
  --fmt pdf,png        output formats
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------
# Paper-style configuration
# -------------------------
def set_paper_style(font="Times New Roman", base_fontsize=8):
    """
    A simple IEEE-friendly style:
    - Serif font (Times-like)
    - Small font size (8pt is common for captions)
    - Thin axes and ticks
    """
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": [font, "Times New Roman", "Times", "DejaVu Serif"],
        "font.size": base_fontsize,
        "axes.titlesize": base_fontsize,
        "axes.labelsize": base_fontsize,
        "xtick.labelsize": base_fontsize,
        "ytick.labelsize": base_fontsize,
        "legend.fontsize": base_fontsize,
        "figure.titlesize": base_fontsize,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "lines.linewidth": 1.2,
        "grid.linewidth": 0.3,
        "grid.alpha": 0.5,
    })


def fig_size(col=1, aspect=0.62):
    """
    IEEE-ish width:
      single-column ~3.5 in
      double-column ~7.16 in
    aspect: height/width
    """
    w = 3.5 if col == 1 else 7.16
    h = w * aspect
    return (w, h)


def safe_mkdir(path: str):
    os.makedirs(path, exist_ok=True)


def to_numeric(df: pd.DataFrame, cols):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")


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
    """
    Return a df subset by phase, or None if phase column missing or empty result.
    """
    if "phase" not in df.columns:
        return None
    sub = df[df["phase"].astype(str).str.lower() == phase.lower()].copy()
    return sub if len(sub) > 0 else None


def fallback_train(df: pd.DataFrame):
    """
    Safe fallback without using `or` on DataFrame (avoids ambiguous truth value).
    """
    train = get_phase(df, "train")
    if train is None:
        train = df.copy()
    return train


def pick_x(df: pd.DataFrame, xaxis: str):
    if xaxis not in df.columns:
        raise ValueError(f"x-axis '{xaxis}' not found in csv columns.")
    return xaxis


# -------------------------
# Plot functions
# -------------------------
def plot_loss(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    train = get_phase(df, "train")
    val = get_phase(df, "val")

    if train is None and val is None:
        print("[skip] loss: no phase=train/val found.")
        return
    if (train is not None and "loss" not in train.columns) and (val is not None and "loss" not in val.columns):
        print("[skip] loss: 'loss' not found.")
        return

    x = pick_x(df, xaxis)
    fig = plt.figure(figsize=fig_size(col))
    ax = fig.add_subplot(111)

    if train is not None and "loss" in train.columns:
        ax.plot(train[x], rolling_mean(train["loss"], smooth), label="train")
    if val is not None and "loss" in val.columns:
        ax.plot(val[x], rolling_mean(val["loss"], smooth), label="val")

    ax.set_xlabel(x)
    ax.set_ylabel("Loss")
    ax.set_title("Loss vs " + x)
    ax.grid(True)
    ax.legend(frameon=False)

    save_fig(fig, out_dir, f"loss_vs_{x}", fmt=fmt, dpi=dpi)


def plot_accuracy(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    train = get_phase(df, "train")
    val = get_phase(df, "val")

    acc_col = None
    for c in ("acc_top1", "acc_top5"):
        if c in df.columns:
            acc_col = c
            break
    if acc_col is None:
        print("[skip] accuracy: acc_top1/acc_top5 not found.")
        return
    if train is None and val is None:
        print("[skip] accuracy: no phase=train/val found.")
        return

    x = pick_x(df, xaxis)
    fig = plt.figure(figsize=fig_size(col))
    ax = fig.add_subplot(111)

    if train is not None and acc_col in train.columns:
        ax.plot(train[x], rolling_mean(train[acc_col], smooth), label="train")
    if val is not None and acc_col in val.columns:
        ax.plot(val[x], rolling_mean(val[acc_col], smooth), label="val")

    ax.set_xlabel(x)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"{acc_col} vs {x}")
    ax.grid(True)
    ax.legend(frameon=False)

    save_fig(fig, out_dir, f"{acc_col}_vs_{x}", fmt=fmt, dpi=dpi)


def plot_step_time(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    train = fallback_train(df)
    if "step_time_ms" not in train.columns:
        print("[skip] step time: step_time_ms not found.")
        return

    x = pick_x(df, xaxis)
    fig = plt.figure(figsize=fig_size(col))
    ax = fig.add_subplot(111)

    ax.plot(train[x], rolling_mean(train["step_time_ms"], smooth))
    ax.set_xlabel(x)
    ax.set_ylabel("Step Time (ms)")
    ax.set_title("Step Time vs " + x)
    ax.grid(True)

    save_fig(fig, out_dir, f"step_time_ms_vs_{x}", fmt=fmt, dpi=dpi)


def plot_stage_breakdown(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    train = fallback_train(df)

    stages = ["pre_fwd_ms", "post_fwd_ms", "expert_comm_ms", "pre_bwd_ms", "post_bwd_ms"]
    stages = [c for c in stages if c in train.columns]
    if not stages:
        print("[skip] stage breakdown: no stage columns found.")
        return

    x = pick_x(df, xaxis)
    fig = plt.figure(figsize=fig_size(col, aspect=0.75))
    ax = fig.add_subplot(111)

    for c in stages:
        ax.plot(train[x], rolling_mean(train[c], smooth), label=c)

    ax.set_xlabel(x)
    ax.set_ylabel("Time (ms)")
    ax.set_title("Latency Breakdown vs " + x)
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)

    save_fig(fig, out_dir, f"latency_breakdown_vs_{x}", fmt=fmt, dpi=dpi)


def plot_comm_ratio(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    train = fallback_train(df)
    if "expert_comm_ms" not in train.columns or "step_time_ms" not in train.columns:
        print("[skip] comm ratio: need expert_comm_ms & step_time_ms.")
        return

    ratio = train["expert_comm_ms"] / train["step_time_ms"]
    ratio = ratio.replace([np.inf, -np.inf], np.nan)

    x = pick_x(df, xaxis)
    fig = plt.figure(figsize=fig_size(col))
    ax = fig.add_subplot(111)

    ax.plot(train[x], rolling_mean(ratio, smooth))
    ax.set_xlabel(x)
    ax.set_ylabel("Comm / Step")
    ax.set_title("Communication Ratio vs " + x)
    ax.grid(True)

    save_fig(fig, out_dir, f"comm_ratio_vs_{x}", fmt=fmt, dpi=dpi)


def plot_hot_cold(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    train = fallback_train(df)

    cols = ["hot_ratio", "cold_skip_ratio", "mode_hot_frac", "mode_cold_frac", "mode_http_frac"]
    cols = [c for c in cols if c in train.columns]
    if not cols:
        print("[skip] hot/cold: no related columns found.")
        return

    x = pick_x(df, xaxis)
    fig = plt.figure(figsize=fig_size(col, aspect=0.75))
    ax = fig.add_subplot(111)

    for c in cols:
        ax.plot(train[x], rolling_mean(train[c], smooth), label=c)

    ax.set_xlabel(x)
    ax.set_ylabel("Ratio / Fraction")
    ax.set_title("Scheduling Dynamics vs " + x)
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)

    save_fig(fig, out_dir, f"scheduling_dynamics_vs_{x}", fmt=fmt, dpi=dpi)


def plot_inst_entropy(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    train = fallback_train(df)
    if "inst_entropy" not in train.columns:
        print("[skip] inst_entropy: not found.")
        return

    x = pick_x(df, xaxis)
    fig = plt.figure(figsize=fig_size(col))
    ax = fig.add_subplot(111)

    ax.plot(train[x], rolling_mean(train["inst_entropy"], smooth))
    ax.set_xlabel(x)
    ax.set_ylabel("Entropy")
    ax.set_title("Instance Selection Entropy vs " + x)
    ax.grid(True)

    save_fig(fig, out_dir, f"inst_entropy_vs_{x}", fmt=fmt, dpi=dpi)


def plot_throughput(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    train = fallback_train(df)

    cols = [c for c in ["samples_per_s", "tokens_per_s"] if c in train.columns]
    if not cols:
        print("[skip] throughput: samples_per_s/tokens_per_s not found.")
        return

    x = pick_x(df, xaxis)
    fig = plt.figure(figsize=fig_size(col))
    ax = fig.add_subplot(111)

    for c in cols:
        ax.plot(train[x], rolling_mean(train[c], smooth), label=c)

    ax.set_xlabel(x)
    ax.set_ylabel("Throughput")
    ax.set_title("Throughput vs " + x)
    ax.grid(True)
    ax.legend(frameon=False)

    save_fig(fig, out_dir, f"throughput_vs_{x}", fmt=fmt, dpi=dpi)


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default="metrics.csv")
    ap.add_argument("--out_dir", type=str, default="figures_icws")
    ap.add_argument("--col", type=int, default=1, choices=[1, 2], help="1=single-column, 2=double-column width")
    ap.add_argument("--smooth", type=int, default=1, help="rolling mean window; 1=no smoothing")
    ap.add_argument("--dpi", type=int, default=600, help="PNG dpi")
    ap.add_argument("--fmt", type=str, default="pdf,png", help="comma-separated formats, e.g., pdf,png or pdf")
    ap.add_argument("--x", type=str, default="step", choices=["step", "epoch", "step_in_epoch"], help="x-axis column")
    ap.add_argument("--font", type=str, default="Times New Roman")
    ap.add_argument("--fontsize", type=int, default=8)
    args = ap.parse_args()

    set_paper_style(font=args.font, base_fontsize=args.fontsize)
    safe_mkdir(args.out_dir)

    df = pd.read_csv(args.csv)

    # numeric conversion
    numeric_cols = [
        "epoch", "step", "step_in_epoch",
        "loss", "acc_top1", "acc_top5",
        "step_time_ms", "pre_fwd_ms", "post_fwd_ms",
        "expert_comm_ms", "pre_bwd_ms", "post_bwd_ms",
        "samples_per_s", "tokens_per_s",
        "hot_ratio", "cold_skip_ratio",
        "mode_hot_frac", "mode_cold_frac", "mode_http_frac",
        "inst_entropy",
    ]
    to_numeric(df, numeric_cols)

    # sort by x for stable lines
    if args.x in df.columns:
        df = df.sort_values([args.x]).reset_index(drop=True)

    fmt = tuple([s.strip().lower() for s in args.fmt.split(",") if s.strip()])

    # Core: convergence + performance
    plot_loss(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    plot_accuracy(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    plot_step_time(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)

    # # Overhead + breakdown
    # plot_stage_breakdown(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    # plot_comm_ratio(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    # plot_throughput(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    #
    # # Scheduling behavior (your key contribution)
    # plot_hot_cold(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    # plot_inst_entropy(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)


if __name__ == "__main__":
    main()
