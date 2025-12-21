# utils/metrics.py
import csv
import os
from dataclasses import dataclass, asdict
from typing import Optional


@dataclass
class StepMetrics:
    epoch: int = 0
    step: int = 0
    step_in_epoch: int = 0
    phase: str = "train"

    loss: Optional[float] = None
    acc_top1: Optional[float] = None
    acc_top5: Optional[float] = None

    batch_size: Optional[int] = None
    seq_len: Optional[int] = None
    tokens: Optional[int] = None

    step_time_ms: Optional[float] = None
    pre_fwd_ms: Optional[float] = None
    post_fwd_ms: Optional[float] = None
    expert_comm_ms: Optional[float] = None

    bwd_total_ms: Optional[float] = None
    pre_bwd_ms: Optional[float] = None
    post_bwd_ms: Optional[float] = None

    bwd_inv_total_ms: Optional[float] = None
    bwd_inv_queue_ms: Optional[float] = None
    bwd_inv_cold_ms: Optional[float] = None
    bwd_inv_net_ms: Optional[float] = None
    bwd_inv_compute_ms: Optional[float] = None
    bwd_inv_retry_cnt: Optional[int] = None

    samples_per_s: Optional[float] = None
    tokens_per_s: Optional[float] = None

    grad_bytes: Optional[float] = None
    grad_total: Optional[int] = None
    grad_mode_hot_frac: Optional[float] = None
    grad_mode_cold_frac: Optional[float] = None
    grad_mode_http_frac: Optional[float] = None
    grad_mode_local_frac: Optional[float] = None
    grad_mode_fallback_frac: Optional[float] = None  # ✅新增

    grad_inv_total_ms: Optional[float] = None
    grad_inv_queue_ms: Optional[float] = None
    grad_inv_cold_ms: Optional[float] = None
    grad_inv_net_ms: Optional[float] = None
    grad_inv_compute_ms: Optional[float] = None
    grad_inv_retry_cnt: Optional[int] = None

    grad_nsga2_feasible: Optional[int] = None  # ✅新增：NSGA2 是否成功输出可行解
    grad_fallback_cnt: Optional[int] = None    # ✅新增：fallback 次数（更可解释）

    dispatch_count: Optional[int] = None
    expert_inst_cnt: Optional[int] = None

    hot_ratio: Optional[float] = None
    active_expert_cnt: Optional[int] = None
    active_hot_ratio: Optional[float] = None
    hot_flip_cnt: Optional[int] = None

    cold_total_cnt: Optional[int] = None
    cold_skipped_cnt: Optional[int] = None
    cold_updated_cnt: Optional[int] = None
    cold_skip_ratio: Optional[float] = None
    cold_apply_steps_avg: Optional[float] = None
    cold_grad_scale_avg: Optional[float] = None
    cold_pending_steps_avg: Optional[float] = None
    cold_update_hit_cnt: Optional[int] = None

    fwd_mode_hot_frac: Optional[float] = None
    fwd_mode_cold_frac: Optional[float] = None
    fwd_mode_local_frac: Optional[float] = None

    fwd_mode_hot_frac_tok: Optional[float] = None
    fwd_mode_cold_frac_tok: Optional[float] = None
    fwd_mode_local_frac_tok: Optional[float] = None

    capacity: Optional[int] = None
    overflow_total_assignments: Optional[int] = None
    overflow_dropped_assignments: Optional[int] = None
    overflow_drop_ratio: Optional[float] = None

    inv_total_ms: Optional[float] = None
    inv_queue_ms: Optional[float] = None
    inv_cold_ms: Optional[float] = None
    inv_net_ms: Optional[float] = None
    inv_compute_ms: Optional[float] = None
    inv_retry_cnt: Optional[int] = None


class MetricsLogger:
    def __init__(self, path: str = "metrics.csv"):
        self.path = path
        self._initialized = os.path.exists(path)
        self._fieldnames = None

        if self._initialized:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        self._fieldnames = header
            except Exception:
                self._fieldnames = None

    def log(self, m: StepMetrics):
        row = asdict(m)
        latest_fields = list(row.keys())

        if os.path.exists(self.path) and self._fieldnames and self._fieldnames != latest_fields:
            try:
                with open(self.path, "r", newline="", encoding="utf-8") as f:
                    old_reader = list(csv.DictReader(f))
                with open(self.path, "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=latest_fields)
                    writer.writeheader()
                    for r in old_reader:
                        merged = {k: r.get(k, "") for k in latest_fields}
                        writer.writerow(merged)
                self._fieldnames = latest_fields
            except Exception:
                self._fieldnames = latest_fields

        need_header = (not os.path.exists(self.path)) or (not self._initialized) or (self._fieldnames is None)

        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=latest_fields)
            if need_header:
                writer.writeheader()
                self._initialized = True
                self._fieldnames = latest_fields

            safe_row = {k: row.get(k, "") for k in latest_fields}
            writer.writerow(safe_row)
