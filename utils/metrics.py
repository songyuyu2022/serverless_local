# utils/metrics.py
import csv
import os
from dataclasses import dataclass, asdict
import matplotlib.pyplot as plt
from collections.abc import Iterable
from typing import Dict, Optional, Tuple, List


@dataclass
class StepMetrics:
    # -------- 基础训练信息 --------
    epoch: int = 0  # epoch 号（如果未使用 epoch 可为 0）
    step: int = 0  # 全局 step
    step_in_epoch: int = 0  # epoch 内的 step 序号
    phase: str = "train"  # "train" or "val"

    # -------- 模型效果指标 --------
    loss: Optional[float] = None
    acc_top1: Optional[float] = None
    acc_top5: Optional[float] = None

    # -------- 规模相关信息（可选）--------
    batch_size: Optional[int] = None
    seq_len: Optional[int] = None
    tokens: Optional[int] = None

    # -------- 性能指标：端到端及各阶段时间（毫秒）--------
    step_time_ms: Optional[float] = None
    pre_fwd_ms: Optional[float] = None
    post_fwd_ms: Optional[float] = None
    post_bwd_ms: Optional[float] = None
    pre_bwd_ms: Optional[float] = None
    expert_comm_ms: Optional[float] = None

    # -------- 吞吐 --------
    samples_per_s: Optional[float] = None
    tokens_per_s: Optional[float] = None

    # -------- 通信指标 --------
    grad_bytes: Optional[float] = None  # 梯度字节数
    dispatch_count: Optional[int] = None  # 调度/触发专家次数
    expert_inst_cnt: Optional[int] = None  # [新增] 专家实例数量 (修复 TypeError)

    # -------- 热/冷专家行为 --------
    hot_ratio: Optional[float] = None  # 热专家比例
    cold_skip_ratio: Optional[float] = None  # 冷专家跳过比例

    # -------- 通信模式分布 --------
    mode_hot_frac: Optional[float] = None
    mode_cold_frac: Optional[float] = None
    mode_http_frac: Optional[float] = None


class MetricsLogger:
    """
    简单 CSV 记录器：
    - 初次写入写 header
    - 后续 append 追加
    """

    def __init__(self, path: str = "metrics.csv"):
        self.path = path
        self._initialized = os.path.exists(path)

    def log(self, m: StepMetrics):
        row = asdict(m)

        # 如果 CSV 不存在或第一次写入 → 写 header
        need_header = not os.path.exists(self.path) or not self._initialized

        # 确保目录存在
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)

        # 以 append 模式写
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if need_header:
                writer.writeheader()
                self._initialized = True
            writer.writerow(row)


# -----------------------------------------------------------------------------
# 下面是独立的辅助函数，不应该放在 MetricsLogger 类内部，或者是静态方法
# -----------------------------------------------------------------------------

def _safe_float(value: Optional[str]) -> Optional[float]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def _safe_int(value: Optional[str]) -> Optional[int]:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return int(float(value))  # 处理 "1.0" 这种可能的情况
    except ValueError:
        return None


def _extract_series(rows: Iterable[Dict[str, str]]) -> Tuple[
    Tuple[List[int], List[float]],
    Tuple[List[int], List[float]],
    Tuple[List[int], List[float]]
]:
    loss_steps, loss_vals = [], []
    acc1_steps, acc1_vals = [], []
    acc5_steps, acc5_vals = [], []

    for idx, row in enumerate(rows):
        step = _safe_int(row.get("step"))
        if step is None:
            step = idx

        loss = _safe_float(row.get("loss"))
        acc_top1 = _safe_float(row.get("acc_top1"))
        acc_top5 = _safe_float(row.get("acc_top5"))

        if loss is not None:
            loss_steps.append(step)
            loss_vals.append(loss)

        if acc_top1 is not None:
            acc1_steps.append(step)
            acc1_vals.append(acc_top1)

        if acc_top5 is not None:
            acc5_steps.append(step)
            acc5_vals.append(acc_top5)

    return (loss_steps, loss_vals), (acc1_steps, acc1_vals), (acc5_steps, acc5_vals)


def plot_loss_and_acc(csv_path: str, output_path: str = "metrics.png", title: str = "Training Metrics") -> None:
    """读取 CSV 训练日志，绘制 loss、acc_top1/acc_top5 曲线并保存成图片。"""

    if not os.path.exists(csv_path):
        print(f"Metrics file {csv_path} not found, skipping plot.")
        return

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = list(csv.DictReader(f))

    (loss_steps, loss_vals), (acc1_steps, acc1_vals), (acc5_steps, acc5_vals) = _extract_series(reader)

    if not loss_vals and not acc1_vals and not acc5_vals:
        print(f"No valid metrics found in {csv_path}, skipping plot.")
        return

    fig, ax1 = plt.subplots(figsize=(10, 6))
    ax1.set_title(title)
    ax1.set_xlabel("step")

    plotted = []

    # 绘制 Loss (左轴)
    if loss_vals:
        loss_line = ax1.plot(loss_steps, loss_vals, label="loss", color="tab:red", linewidth=2)
        ax1.set_ylabel("loss", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red")
        plotted.extend(loss_line)

    # 绘制 Accuracy (右轴)
    if acc1_vals or acc5_vals:
        ax2 = ax1.twinx()
        ax2.set_ylabel("accuracy", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        if acc1_vals:
            acc1_line = ax2.plot(acc1_steps, acc1_vals, label="acc_top1", color="tab:blue", linestyle="--", linewidth=2)
            plotted.extend(acc1_line)

        if acc5_vals:
            acc5_line = ax2.plot(acc5_steps, acc5_vals, label="acc_top5", color="tab:green", linestyle=":", linewidth=2)
            plotted.extend(acc5_line)

    if plotted:
        labels = [line.get_label() for line in plotted]
        ax1.legend(plotted, labels, loc="best")

    fig.tight_layout()
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    try:
        fig.savefig(output_path, dpi=300)
        print(f"Metrics plot saved to {output_path}")
    except Exception as e:
        print(f"Failed to save plot: {e}")
    finally:
        plt.close(fig)