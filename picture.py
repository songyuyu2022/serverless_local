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
    if "phase" not in df.columns:
        return None
    sub = df[df["phase"].astype(str).str.lower() == phase.lower()].copy()
    return sub if len(sub) > 0 else None


def fallback_train(df: pd.DataFrame):
    train = get_phase(df, "train")
    if train is None:
        train = df.copy()
    return train


def pick_x(df: pd.DataFrame, xaxis: str):
    if xaxis not in df.columns:
        raise ValueError(f"x-axis '{xaxis}' not found in csv columns.")
    return xaxis


def _has_any(df: pd.DataFrame, cols):
    return any((c in df.columns) for c in cols)


def _safe_series(df: pd.DataFrame, col: str):
    if col not in df.columns:
        return None
    s = df[col].replace([np.inf, -np.inf], np.nan)
    return s


# -------------------------
# Plot helpers
# -------------------------
def plot_lines(train: pd.DataFrame, out_dir: str, name: str, x: str, cols, ylabel: str, title: str,
               col=1, smooth=1, dpi=600, fmt=("pdf", "png"), ncol=2, aspect=0.75):
    cols = [c for c in cols if c in train.columns]
    if not cols:
        print(f"[skip] {name}: no columns found.")
        return

    fig = plt.figure(figsize=fig_size(col, aspect=aspect))
    ax = fig.add_subplot(111)

    for c in cols:
        ax.plot(train[x], rolling_mean(train[c], smooth), label=c)

    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    ax.legend(frameon=False, ncol=ncol)

    save_fig(fig, out_dir, name, fmt=fmt, dpi=dpi)


def plot_ratio(train: pd.DataFrame, out_dir: str, name: str, x: str, numerator: str, denom: str,
               ylabel: str, title: str, col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    if numerator not in train.columns or denom not in train.columns:
        print(f"[skip] {name}: need {numerator} and {denom}.")
        return
    r = train[numerator] / train[denom]
    r = r.replace([np.inf, -np.inf], np.nan)

    fig = plt.figure(figsize=fig_size(col))
    ax = fig.add_subplot(111)
    ax.plot(train[x], rolling_mean(r, smooth))
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True)
    save_fig(fig, out_dir, name, fmt=fmt, dpi=dpi)


# -------------------------
# Core plots (已有但加强)
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

    # 同时画 top1/top5（如果都有）
    acc_cols = [c for c in ["acc_top1", "acc_top5"] if c in df.columns]
    if not acc_cols:
        print("[skip] accuracy: acc_top1/acc_top5 not found.")
        return

    if train is None and val is None:
        print("[skip] accuracy: no phase=train/val found.")
        return

    x = pick_x(df, xaxis)
    fig = plt.figure(figsize=fig_size(col))
    ax = fig.add_subplot(111)

    for c in acc_cols:
        if train is not None and c in train.columns:
            ax.plot(train[x], rolling_mean(train[c], smooth), label=f"train-{c}")
        if val is not None and c in val.columns:
            ax.plot(val[x], rolling_mean(val[c], smooth), label=f"val-{c}")

    ax.set_xlabel(x)
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Accuracy vs {x}")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)

    save_fig(fig, out_dir, f"accuracy_vs_{x}", fmt=fmt, dpi=dpi)


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


# -------------------------
# NEW (ICWS strong) plots
# -------------------------
def plot_throughput(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    train = fallback_train(df)
    cols = [c for c in ["samples_per_s", "tokens_per_s"] if c in train.columns]
    if not cols:
        print("[skip] throughput: samples_per_s/tokens_per_s not found.")
        return
    x = pick_x(df, xaxis)
    plot_lines(train, out_dir, f"throughput_vs_{x}", x, cols, "Throughput", f"Throughput vs {x}",
               col=col, smooth=smooth, dpi=dpi, fmt=fmt, ncol=2, aspect=0.62)


def plot_end2end_decomposition(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    """
    端到端分解（论文常用）：
      step_time_ms vs (fwd approx + bwd_total_ms + grad_inv_total_ms)
    """
    train = fallback_train(df)
    need = ["step_time_ms"]
    if not _has_any(train, need):
        print("[skip] e2e_decomp: step_time_ms not found.")
        return

    x = pick_x(df, xaxis)

    # forward 近似：pre_fwd + post_fwd + expert_comm（如果都有）
    fwd_cols = [c for c in ["pre_fwd_ms", "post_fwd_ms", "expert_comm_ms"] if c in train.columns]
    bwd_col = "bwd_total_ms" if "bwd_total_ms" in train.columns else None
    grad_col = "grad_inv_total_ms" if "grad_inv_total_ms" in train.columns else None

    if not fwd_cols and not bwd_col and not grad_col:
        print("[skip] e2e_decomp: no fwd/bwd/grad columns found.")
        return

    fig = plt.figure(figsize=fig_size(col, aspect=0.75))
    ax = fig.add_subplot(111)

    ax.plot(train[x], rolling_mean(train["step_time_ms"], smooth), label="step_time_ms")

    if fwd_cols:
        fwd = None
        for c in fwd_cols:
            fwd = train[c] if fwd is None else (fwd + train[c])
        ax.plot(train[x], rolling_mean(fwd, smooth), label="fwd(pre+post+expert_comm)")

    if bwd_col:
        ax.plot(train[x], rolling_mean(train[bwd_col], smooth), label=bwd_col)

    if grad_col:
        ax.plot(train[x], rolling_mean(train[grad_col], smooth), label=grad_col)

    ax.set_xlabel(x)
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"End-to-End Decomposition vs {x}")
    ax.grid(True)
    ax.legend(frameon=False, ncol=2)

    save_fig(fig, out_dir, f"e2e_decomposition_vs_{x}", fmt=fmt, dpi=dpi)


def plot_invoke_breakdown(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png"),
                          prefix="inv", title_prefix="Forward Invoke"):
    """
    Serverless 特征图：queue/cold/net/compute 分解
    prefix:
      inv_*  (forward invoke)
      bwd_inv_* (backward invoke)
      grad_inv_* (grad apply invoke)
    """
    train = fallback_train(df)
    x = pick_x(df, xaxis)

    mapping = {
        f"{prefix}_queue_ms": "queue",
        f"{prefix}_cold_ms": "cold",
        f"{prefix}_net_ms": "net",
        f"{prefix}_compute_ms": "compute",
    }
    cols = [k for k in mapping.keys() if k in train.columns]
    if not cols:
        print(f"[skip] {prefix}_breakdown: no breakdown columns found.")
        return

    fig = plt.figure(figsize=fig_size(col, aspect=0.75))
    ax = fig.add_subplot(111)

    for c in cols:
        ax.plot(train[x], rolling_mean(train[c], smooth), label=mapping[c])

    ax.set_xlabel(x)
    ax.set_ylabel("Time (ms)")
    ax.set_title(f"{title_prefix} Breakdown vs {x}")
    ax.grid(True)
    ax.legend(frameon=False, ncol=4)

    save_fig(fig, out_dir, f"{prefix}_breakdown_vs_{x}", fmt=fmt, dpi=dpi)


def plot_overflow(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    train = fallback_train(df)
    x = pick_x(df, xaxis)

    cols = [c for c in ["capacity", "overflow_total_assignments", "overflow_dropped_assignments"] if c in train.columns]
    if not cols and "overflow_drop_ratio" not in train.columns:
        print("[skip] overflow: related columns not found.")
        return

    if cols:
        plot_lines(train, out_dir, f"capacity_overflow_counts_vs_{x}", x, cols,
                   ylabel="Count", title=f"Capacity / Overflow Counts vs {x}",
                   col=col, smooth=smooth, dpi=dpi, fmt=fmt, ncol=2, aspect=0.75)

    if "overflow_drop_ratio" in train.columns:
        fig = plt.figure(figsize=fig_size(col))
        ax = fig.add_subplot(111)
        ax.plot(train[x], rolling_mean(train["overflow_drop_ratio"], smooth))
        ax.set_xlabel(x)
        ax.set_ylabel("Drop Ratio")
        ax.set_title(f"Overflow Drop Ratio vs {x}")
        ax.grid(True)
        save_fig(fig, out_dir, f"overflow_drop_ratio_vs_{x}", fmt=fmt, dpi=dpi)


def plot_hot_dynamics(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    """
    论文贡献点：证明热/冷识别在变化
    """
    train = fallback_train(df)
    x = pick_x(df, xaxis)

    cols = [c for c in ["hot_ratio", "active_hot_ratio", "hot_flip_cnt"] if c in train.columns]
    if not cols:
        print("[skip] hot_dynamics: hot_ratio/active_hot_ratio/hot_flip_cnt not found.")
        return

    # hot_flip_cnt 是计数，数值可能较小；仍然画线即可
    plot_lines(train, out_dir, f"hot_dynamics_vs_{x}", x, cols,
               ylabel="Ratio / Count", title=f"Hot/Cold Dynamics vs {x}",
               col=col, smooth=smooth, dpi=dpi, fmt=fmt, ncol=2, aspect=0.75)


def plot_cold_update(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    """
    解释冷专家延迟更新的强度（否则 cold_skip_ratio 难解释）
    """
    train = fallback_train(df)
    x = pick_x(df, xaxis)

    cols = [c for c in [
        "cold_skip_ratio",
        "cold_pending_steps_avg",
        "cold_apply_steps_avg",
        "cold_grad_scale_avg",
        "cold_update_hit_cnt",
    ] if c in train.columns]

    if not cols:
        print("[skip] cold_update: related columns not found.")
        return

    plot_lines(train, out_dir, f"cold_update_dynamics_vs_{x}", x, cols,
               ylabel="Ratio / Steps / Count", title=f"Cold Update Dynamics vs {x}",
               col=col, smooth=smooth, dpi=dpi, fmt=fmt, ncol=2, aspect=0.85)


def plot_mode_fractions(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    """
    Forward mode fractions（按 expert 计数 or token 计数）
    你当前 csv 里若有 *_tok 列，会优先画 token 口径。
    """
    train = fallback_train(df)
    x = pick_x(df, xaxis)

    tok_cols = [c for c in ["fwd_mode_hot_frac_tok", "fwd_mode_cold_frac_tok", "fwd_mode_local_frac_tok"] if c in train.columns]
    exp_cols = [c for c in ["fwd_mode_hot_frac", "fwd_mode_cold_frac", "fwd_mode_local_frac"] if c in train.columns]

    if tok_cols:
        plot_lines(train, out_dir, f"fwd_mode_token_fractions_vs_{x}", x, tok_cols,
                   ylabel="Fraction", title=f"Forward Mode Fractions (Token) vs {x}",
                   col=col, smooth=smooth, dpi=dpi, fmt=fmt, ncol=3, aspect=0.62)
    elif exp_cols:
        plot_lines(train, out_dir, f"fwd_mode_expert_fractions_vs_{x}", x, exp_cols,
                   ylabel="Fraction", title=f"Forward Mode Fractions (Expert) vs {x}",
                   col=col, smooth=smooth, dpi=dpi, fmt=fmt, ncol=3, aspect=0.62)
    else:
        print("[skip] fwd_mode_fractions: not found.")


def plot_grad_modes(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    """
    Gradient 模式分布（ICWS: 强支撑“调度/优化策略真实生效”）
    """
    train = fallback_train(df)
    x = pick_x(df, xaxis)

    cols = [c for c in [
        "grad_mode_hot_frac",
        "grad_mode_cold_frac",
        "grad_mode_http_frac",
        "grad_mode_local_frac",
        "grad_mode_fallback_frac",
    ] if c in train.columns]

    if not cols:
        print("[skip] grad_modes: grad_mode_* not found.")
        return

    plot_lines(train, out_dir, f"grad_mode_fractions_vs_{x}", x, cols,
               ylabel="Fraction", title=f"Gradient Apply Mode Fractions vs {x}",
               col=col, smooth=smooth, dpi=dpi, fmt=fmt, ncol=2, aspect=0.75)

    # 可解释字段：feasible / fallback cnt（若有）
    extra_cols = [c for c in ["grad_nsga2_feasible", "grad_fallback_cnt", "grad_total"] if c in train.columns]
    if extra_cols:
        plot_lines(train, out_dir, f"grad_scheduler_debug_vs_{x}", x, extra_cols,
                   ylabel="Value", title=f"Gradient Scheduler Diagnostics vs {x}",
                   col=col, smooth=smooth, dpi=dpi, fmt=fmt, ncol=3, aspect=0.75)


def plot_comm_share(df, out_dir, xaxis="step", col=1, smooth=1, dpi=600, fmt=("pdf", "png")):
    """
    通信/开销占比（论文常用一张图）
    """
    train = fallback_train(df)
    x = pick_x(df, xaxis)

    # expert_comm_ms / step_time_ms
    plot_ratio(train, out_dir, f"comm_share_expert_vs_{x}", x,
               numerator="expert_comm_ms", denom="step_time_ms",
               ylabel="ExpertComm / Step", title=f"Expert Comm Share vs {x}",
               col=col, smooth=smooth, dpi=dpi, fmt=fmt)

    # invoke_total / step_time（如果有）
    if "inv_total_ms" in train.columns and "step_time_ms" in train.columns:
        plot_ratio(train, out_dir, f"comm_share_invoke_vs_{x}", x,
                   numerator="inv_total_ms", denom="step_time_ms",
                   ylabel="InvokeTotal / Step", title=f"Invoke Share vs {x}",
                   col=col, smooth=smooth, dpi=dpi, fmt=fmt)

    # grad_inv_total / step_time（如果有）
    if "grad_inv_total_ms" in train.columns and "step_time_ms" in train.columns:
        plot_ratio(train, out_dir, f"comm_share_grad_inv_vs_{x}", x,
                   numerator="grad_inv_total_ms", denom="step_time_ms",
                   ylabel="GradInv / Step", title=f"Grad Apply Invoke Share vs {x}",
                   col=col, smooth=smooth, dpi=dpi, fmt=fmt)


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

    # numeric conversion (尽量覆盖你 csv 的字段，缺失也没关系)
    numeric_cols = [
        "epoch", "step", "step_in_epoch",
        "loss", "acc_top1", "acc_top5",
        "step_time_ms", "pre_fwd_ms", "post_fwd_ms", "expert_comm_ms",
        "bwd_total_ms", "pre_bwd_ms", "post_bwd_ms",
        "samples_per_s", "tokens_per_s",
        "hot_ratio", "active_hot_ratio", "hot_flip_cnt",
        "cold_skip_ratio", "cold_pending_steps_avg", "cold_apply_steps_avg", "cold_grad_scale_avg", "cold_update_hit_cnt",
        "capacity", "overflow_total_assignments", "overflow_dropped_assignments", "overflow_drop_ratio",
        "inv_total_ms", "inv_queue_ms", "inv_cold_ms", "inv_net_ms", "inv_compute_ms", "inv_retry_cnt",
        "bwd_inv_total_ms", "bwd_inv_queue_ms", "bwd_inv_cold_ms", "bwd_inv_net_ms", "bwd_inv_compute_ms", "bwd_inv_retry_cnt",
        "grad_total", "grad_bytes",
        "grad_mode_hot_frac", "grad_mode_cold_frac", "grad_mode_http_frac", "grad_mode_local_frac", "grad_mode_fallback_frac",
        "grad_inv_total_ms", "grad_inv_queue_ms", "grad_inv_cold_ms", "grad_inv_net_ms", "grad_inv_compute_ms", "grad_inv_retry_cnt",
        "grad_nsga2_feasible", "grad_fallback_cnt",
        "fwd_mode_hot_frac", "fwd_mode_cold_frac", "fwd_mode_local_frac",
        "fwd_mode_hot_frac_tok", "fwd_mode_cold_frac_tok", "fwd_mode_local_frac_tok",
    ]
    to_numeric(df, numeric_cols)

    # sort by x for stable lines
    if args.x in df.columns:
        df = df.sort_values([args.x]).reset_index(drop=True)

    fmt = tuple([s.strip().lower() for s in args.fmt.split(",") if s.strip()])

    # =========================
    # (A) 训练收敛（论文必备）
    # =========================
    plot_loss(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    plot_accuracy(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)

    # =========================
    # (B) 性能（论文必备）
    # =========================
    plot_step_time(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    plot_throughput(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)

    # =========================
    # (C) 系统分解（ICWS 强图）
    # =========================
    plot_end2end_decomposition(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    plot_comm_share(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)

    # =========================
    # (D) Serverless invoke 特征分解（关键）
    # =========================
    plot_invoke_breakdown(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt,
                          prefix="inv", title_prefix="Forward Invoke")
    plot_invoke_breakdown(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt,
                          prefix="bwd_inv", title_prefix="Backward Invoke")
    plot_invoke_breakdown(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt,
                          prefix="grad_inv", title_prefix="Grad Apply Invoke")

    # =========================
    # (E) MoE 正确性：capacity / overflow
    # =========================
    plot_overflow(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)

    # =========================
    # (F) 你的方法贡献点：hot/cold 动态 & 冷更新强度
    # =========================
    plot_hot_dynamics(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    plot_cold_update(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)

    # Forward/Grad 模式分布（证明调度在起作用）
    plot_mode_fractions(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)
    plot_grad_modes(df, args.out_dir, xaxis=args.x, col=args.col, smooth=args.smooth, dpi=args.dpi, fmt=fmt)


if __name__ == "__main__":
    main()
