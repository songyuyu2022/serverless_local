# utils/metrics.py
from __future__ import annotations

import csv
import os
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Any

import numpy as np


@dataclass
class StepMetrics:
    # ---- identity ----
    epoch: int = 0
    step: int = 0
    step_in_epoch: int = 0
    phase: str = "train"

    # ---- quality ----
    loss: Optional[float] = None
    acc_top1: Optional[float] = None
    acc_top5: Optional[float] = None

    # ---- basic setting ----
    batch_size: Optional[int] = None
    seq_len: Optional[int] = None
    tokens: Optional[int] = None

    # ---- end-to-end perf ----
    step_time_ms: Optional[float] = None
    samples_per_s: Optional[float] = None
    tokens_per_s: Optional[float] = None

    # ---- breakdown ----
    pre_fwd_ms: Optional[float] = None
    post_fwd_ms: Optional[float] = None
    expert_comm_ms: Optional[float] = None

    bwd_total_ms: Optional[float] = None
    pre_bwd_ms: Optional[float] = None
    post_bwd_ms: Optional[float] = None

    # ---- invocation breakdown (forward) ----
    inv_total_ms: Optional[float] = None
    inv_queue_ms: Optional[float] = None
    inv_cold_ms: Optional[float] = None
    inv_net_ms: Optional[float] = None
    inv_compute_ms: Optional[float] = None
    inv_retry_cnt: Optional[int] = None

    # ---- invocation breakdown (backward pre/post) ----
    bwd_inv_total_ms: Optional[float] = None
    bwd_inv_queue_ms: Optional[float] = None
    bwd_inv_cold_ms: Optional[float] = None
    bwd_inv_net_ms: Optional[float] = None
    bwd_inv_compute_ms: Optional[float] = None
    bwd_inv_retry_cnt: Optional[int] = None

    # ---- gradient apply / grad communication ----
    grad_bytes: Optional[float] = None
    grad_total: Optional[int] = None

    grad_mode_hot_frac: Optional[float] = None
    grad_mode_cold_frac: Optional[float] = None
    grad_mode_http_frac: Optional[float] = None
    grad_mode_local_frac: Optional[float] = None
    grad_mode_fallback_frac: Optional[float] = None

    grad_inv_total_ms: Optional[float] = None
    grad_inv_queue_ms: Optional[float] = None
    grad_inv_cold_ms: Optional[float] = None
    grad_inv_net_ms: Optional[float] = None
    grad_inv_compute_ms: Optional[float] = None
    grad_inv_retry_cnt: Optional[int] = None

    grad_nsga2_feasible: Optional[int] = None
    grad_fallback_cnt: Optional[int] = None

    # ---- mode-specific grad latency/bytes (解释性更强) ----
    grad_lat_hot_ms: Optional[float] = None
    grad_lat_cold_ms: Optional[float] = None
    grad_lat_http_ms: Optional[float] = None

    grad_bytes_hot: Optional[int] = None
    grad_bytes_cold: Optional[int] = None
    grad_bytes_http: Optional[int] = None

    # ---- dispatch / hotcold ----
    dispatch_count: Optional[int] = None
    expert_inst_cnt: Optional[int] = None

    hot_ratio: Optional[float] = None
    active_expert_cnt: Optional[int] = None
    active_hot_ratio: Optional[float] = None
    hot_flip_cnt: Optional[int] = None

    # ---- hot/cold dynamics (论文证据链关键) ----
    hot_set_size: Optional[int] = None
    hot_set_jaccard: Optional[float] = None
    expert_load_entropy: Optional[float] = None

    # ---- cold update stats ----
    cold_total_cnt: Optional[int] = None
    cold_skipped_cnt: Optional[int] = None
    cold_updated_cnt: Optional[int] = None
    cold_skip_ratio: Optional[float] = None
    cold_apply_steps_avg: Optional[float] = None
    cold_grad_scale_avg: Optional[float] = None
    cold_pending_steps_avg: Optional[float] = None
    cold_update_hit_cnt: Optional[int] = None

    # ---- fwd mode ----
    fwd_mode_hot_frac: Optional[float] = None
    fwd_mode_cold_frac: Optional[float] = None
    fwd_mode_local_frac: Optional[float] = None

    fwd_mode_hot_frac_tok: Optional[float] = None
    fwd_mode_cold_frac_tok: Optional[float] = None
    fwd_mode_local_frac_tok: Optional[float] = None

    # ---- overflow ----
    capacity: Optional[int] = None
    overflow_total_assignments: Optional[int] = None
    overflow_dropped_assignments: Optional[int] = None
    overflow_drop_ratio: Optional[float] = None

    # ---- tail latency / SLO ----
    p95_step_time_ms: Optional[float] = None
    p95_inv_total_ms: Optional[float] = None

    deadline_ms: Optional[float] = None
    deadline_miss: Optional[int] = None
    deadline_slack_ms: Optional[float] = None

    # ---- ✅ COST breakdown (NEW) ----
    cost_usd_pre_fwd: Optional[float] = None
    cost_usd_post_fwd: Optional[float] = None
    cost_usd_expert_fwd: Optional[float] = None
    cost_usd_pre_bwd: Optional[float] = None
    cost_usd_post_bwd: Optional[float] = None
    cost_usd_grad_apply: Optional[float] = None

    # total
    cost_usd_step: Optional[float] = None


class MetricsLogger:
    """
    - 自动维护滚动窗口并写入 p95_step_time_ms / p95_inv_total_ms
    - 兼容旧字段：如果文件已存在会对齐新表头
    """
    def __init__(self, path: str = "metrics.csv", tail_window: int = 50):
        self.path = path
        self.tail_window = max(5, int(tail_window))
        self._initialized = os.path.exists(path)
        self._fieldnames: Optional[List[str]] = None

        self._buf_step_time: List[float] = []
        self._buf_inv_total: List[float] = []

        if self._initialized:
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    if header:
                        self._fieldnames = header
            except Exception:
                self._fieldnames = None

    def _push_tail(self, step_time_ms: Optional[float], inv_total_ms: Optional[float]):
        if step_time_ms is not None:
            self._buf_step_time.append(float(step_time_ms))
            if len(self._buf_step_time) > self.tail_window:
                self._buf_step_time.pop(0)

        if inv_total_ms is not None:
            self._buf_inv_total.append(float(inv_total_ms))
            if len(self._buf_inv_total) > self.tail_window:
                self._buf_inv_total.pop(0)

    def _p95(self, xs: List[float]) -> float:
        if not xs:
            return 0.0
        return float(np.percentile(np.asarray(xs, dtype=np.float64), 95))

    def log(self, m: StepMetrics):
        self._push_tail(m.step_time_ms, m.inv_total_ms)

        if m.p95_step_time_ms is None:
            m.p95_step_time_ms = self._p95(self._buf_step_time)
        if m.p95_inv_total_ms is None:
            m.p95_inv_total_ms = self._p95(self._buf_inv_total)

        row: Dict[str, Any] = asdict(m)
        latest_fields = list(row.keys())

        # header 变化 -> 对齐旧文件
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
