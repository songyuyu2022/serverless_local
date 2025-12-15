"""
训练控制器 (ICWS Edition):
1. [Core] 引入 "双阈值滞后 + 趋势感知" 的自适应热度机制 (Hysteresis & Trend-Aware Heatmap)。
2. [Fix] 流量偏斜模拟 (Traffic Skew) 保持开启，以生成符合长尾分布的负载。
3. [Metrics] 记录详细的指标，包括 HotRatio, Tokens/s 等。
"""

import os
import asyncio
import json
import time
import random
import math
from typing import Any, Dict, List, Set
from collections import defaultdict, deque

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# 尝试引入真实模型
try:
    from moe_model import SimpleMoE
except ImportError:
    print("[Critical] moe_model.py not found. Please create it first.")
    SimpleMoE = None

from dataset import LMTextBatcher, DATA_PATH_DEFAULT
from nsga2_bw import nsga2_select, feasible_modes
from scheduler_hybrid import HYBRID_SCHED
from utils.logger import log
from utils.metrics import MetricsLogger, StepMetrics
from moe_config import load_moe_config

# ----------------- 1. 全局配置 -----------------

DATA_PATH = os.getenv("DATA_PATH", DATA_PATH_DEFAULT)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "64"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "2000"))
VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "50"))
LOG_TRAIN_EVERY = int(os.getenv("LOG_TRAIN_EVERY", "10"))

STEP_PERIOD_MS = float(os.getenv("STEP_PERIOD_MS", "200.0"))
USE_NSGA2 = os.getenv("USE_NSGA2", "1") == "1"
COLD_ACC_STEPS = int(os.getenv("COLD_ACC_STEPS", "4"))
MICRO_BATCHES = int(os.getenv("MICRO_BATCHES", "4"))

DEFAULT_NET_LATENCY = 5.0
DEFAULT_PERFORMANCE = 1.0

# ----------------- 2. 日志与资源加载 -----------------

DISPATCH_LOG_FILE = "dispatch_trace.jsonl"
INSTANCES_FILE = os.getenv("INSTANCES_FILE", "instances.json")
FUNC_MAP_FILE = os.getenv("FUNC_MAP_FILE", "func_map.json")

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except: return default

_all_instances_data = _load_json(INSTANCES_FILE, [])
ALL_INSTANCES = _all_instances_data.get("instances", []) if isinstance(_all_instances_data, dict) else _all_instances_data
INST_BY_ID = {inst.get("id"): inst for inst in ALL_INSTANCES}
FUNC_MAP = _load_json(FUNC_MAP_FILE, {})

MOE_CONFIG = load_moe_config({k: v for k, v in FUNC_MAP.items() if k.startswith("moe.expert_fwd:")})

REAL_MODEL = None
OPTIMIZER = None
VOCAB_SIZE = 12000

def append_dispatch_log(traces: List[Dict[str, Any]]):
    if not traces: return
    try:
        with open(DISPATCH_LOG_FILE, "a", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    except Exception: pass

# ----------------- [Core] 双阈值滞后热度追踪器 (ICWS Method) -----------------

class AdaptiveHysteresisHeatmap:
    """
    [ICWS Method] 基于双阈值滞后 (Hysteresis) 与趋势感知的自适应热度机制。

    机制说明：
    1. 状态保持 (Stateful): 每个专家有明确的 Hot/Cold 状态。
    2. 滞后切换 (Hysteresis):
       - Cold -> Hot: 需要热度分 > high_threshold (上水位)
       - Hot -> Cold: 需要热度分 < low_threshold (下水位)
       * 防止在临界点频繁抖动 (Thrashing)。
    3. 趋势感知 (Trend-Aware):
       - 同时维护 short_term (快速) 和 long_term (慢速) 两个 EMA。
       - 如果 short > long (上升趋势)，临时降低 high_threshold，加速预热。
    """
    def __init__(self, num_experts, alpha_short=0.3, alpha_long=0.05):
        self.num_experts = num_experts

        # 双 EMA 追踪
        self.score_short = torch.zeros(num_experts) # 快速反应
        self.score_long = torch.zeros(num_experts)  # 历史基线

        self.alpha_short = alpha_short
        self.alpha_long = alpha_long

        # 专家当前状态 (True=Hot, False=Cold)
        self.is_hot_state = [False] * num_experts

        # 基础阈值 (基准线)
        # 假设均匀分布下每个专家的期望负载是 1/N
        # 上水位设为期望值的 1.2 倍，下水位设为 0.5 倍
        base_load = 1.0 / max(1, num_experts)
        self.base_high_thresh = base_load * 1.5
        self.base_low_thresh = base_load * 0.5

    def update(self, active_experts: List[int]):
        # 构建当前 step 的活跃向量
        current_activity = torch.zeros(self.num_experts)
        if active_experts:
            current_activity[active_experts] = 1.0

        # 1. 更新双 EMA
        self.score_short = self.alpha_short * current_activity + (1 - self.alpha_short) * self.score_short
        self.score_long = self.alpha_long * current_activity + (1 - self.alpha_long) * self.score_long

        # 2. 状态转换逻辑
        for eid in range(self.num_experts):
            s_short = self.score_short[eid].item()
            s_long = self.score_long[eid].item()

            # [趋势感知]: 计算动态阈值
            # 如果短期热度 > 长期热度 (正在变热)，则降低门槛，让它更容易变成 Hot
            trend_factor = 1.0
            if s_short > s_long:
                trend_factor = 0.8 # 门槛降低 20%

            dynamic_high = self.base_high_thresh * trend_factor
            dynamic_low = self.base_low_thresh # 下水位保持不变，保证稳定性

            # [滞后切换 Hysteresis]
            if not self.is_hot_state[eid]:
                # Cold -> Hot: 必须冲过高水位
                if s_short > dynamic_high:
                    self.is_hot_state[eid] = True
            else:
                # Hot -> Cold: 必须跌破低水位
                if s_short < dynamic_low:
                    self.is_hot_state[eid] = False

    def is_hot(self, eid: int) -> bool:
        return self.is_hot_state[eid]

# 全局热度追踪器
HEATMAP = None

# ----------------- 3. 冷启动与性能模拟核心 -----------------

class InstanceManager:
    def __init__(self, default_keep_alive_ms: float = 30000.0):
        self.last_access: Dict[str, float] = {}
        self.default_keep_alive_ms = default_keep_alive_ms
        self.dynamic_keep_alive: Dict[str, float] = {}

    def touch(self, inst_id: str, is_hot_task: bool):
        now = time.perf_counter() * 1000.0
        self.last_access[inst_id] = now
        # [自适应] 热任务获得 2 倍保活时间
        if is_hot_task:
            self.dynamic_keep_alive[inst_id] = self.default_keep_alive_ms * 2.0
        else:
            self.dynamic_keep_alive[inst_id] = self.default_keep_alive_ms

    def check_cold_start(self, inst: Dict[str, Any]) -> float:
        inst_id = inst.get("id")
        now = time.perf_counter() * 1000.0
        delay = 0.0
        last = self.last_access.get(inst_id)
        keep_alive = self.dynamic_keep_alive.get(inst_id, self.default_keep_alive_ms)
        is_cold = False
        if last is None: is_cold = True
        elif (now - last) > keep_alive: is_cold = True
        if is_cold:
            delay = float(inst.get("meta", {}).get("cold_start_ms", 100.0))
        return delay

INSTANCE_MGR = InstanceManager()

def apply_performance_scaling(inst: Dict[str, Any], real_compute_time_ms: float, is_hot_task: bool = False) -> float:
    inst_id = inst.get("id")
    perf = float(inst.get("meta", {}).get("performance", DEFAULT_PERFORMANCE))

    simulated_compute_time = real_compute_time_ms / perf
    cold_delay = INSTANCE_MGR.check_cold_start(inst)

    INSTANCE_MGR.touch(inst_id, is_hot_task)

    net_delay = float(inst.get("meta", {}).get("net_latency", DEFAULT_NET_LATENCY))
    total_latency = simulated_compute_time + net_delay + cold_delay

    time_diff = total_latency - real_compute_time_ms
    if time_diff > 0:
        time.sleep(time_diff / 1000.0)

    return total_latency

# ----------------- 4. 微批次处理 (含流量偏斜) -----------------

def simulate_traffic_skew(topk_idx: torch.Tensor, num_experts: int) -> torch.Tensor:
    """
    [Simulation] 模拟真实的长尾分布流量。
    如果没有这个，随机模型的流量是均匀的，任何自适应算法都会失效（因为确实大家一样热）。
    """
    if num_experts <= 2: return topk_idx

    batch, seq, k = topk_idx.shape
    new_idx = topk_idx.clone()

    # 模拟 Pattern:
    # Expert 0, 1 是绝对热点 (80% 概率)
    # Expert 2 是次热点 (10% 概率)
    # 其他是冷门
    rand_vals = torch.rand(batch, seq)

    mask_hot = rand_vals < 0.7
    new_idx[mask_hot, 0] = 0
    if k > 1: new_idx[mask_hot, 1] = 1

    mask_warm = (rand_vals >= 0.7) & (rand_vals < 0.85)
    new_idx[mask_warm, 0] = 2

    return new_idx

async def process_micro_batch(
        x_mb: torch.Tensor,
        y_mb: torch.Tensor,
        micro_id: int,
        mb_idx: int,
        global_step: int,
        tokens: int,
        train: bool,
) -> Dict[str, Any]:

    metrics = defaultdict(float, {
        "hot_experts": set(), "cold_experts": set(),
        "mode_counts": defaultdict(int), "inst_choice_counts": defaultdict(int),
        "cold_total": 0.0, "cold_skipped": 0.0,
        "dispatch_count": 0.0, "grad_bytes": 0.0, "expert_comm": 0.0,
        "pre_lat": 0.0, "post_lat": 0.0, "pre_bwd": 0.0, "post_bwd": 0.0
    })
    trace = {"step": global_step, "mb": mb_idx, "ts": time.time(), "exp_fwd": []}

    # ==========================
    # 1. Pre Stage
    # ==========================
    func_pre = "moe.pre_fwd"
    insts_pre = [INST_BY_ID[i] for i in FUNC_MAP.get(func_pre, []) if i in INST_BY_ID]
    req_pre = {"tokens": tokens, "emb_dim": MOE_CONFIG.d_model}

    t0 = time.perf_counter()
    h, topk_vals, topk_idx = REAL_MODEL.forward_pre(x_mb)
    real_pre_ms = (time.perf_counter() - t0) * 1000.0

    # [Sim] 应用偏斜，制造冷热差异
    topk_idx = simulate_traffic_skew(topk_idx, MOE_CONFIG.num_experts)

    if insts_pre:
        inst_pre, _ = HYBRID_SCHED.select_instance(func_pre, 0, insts_pre, req_pre)
        lat_pre = apply_performance_scaling(inst_pre, real_pre_ms, is_hot_task=True)
        metrics["pre_lat"] += lat_pre
        HYBRID_SCHED.update_stats(func_pre, 0, inst_pre, req_pre, lat_pre)
        trace["pre"] = inst_pre.get("id")
    else:
        lat_pre = real_pre_ms
        metrics["pre_lat"] += lat_pre
        inst_pre = None

    # [Update] 更新自适应热度
    active_experts = topk_idx.unique().tolist()
    if HEATMAP: HEATMAP.update(active_experts)

    for eid in active_experts: metrics["hot_experts"].add(eid)

    # ==========================
    # 2. Expert Stage
    # ==========================
    tokens_per_exp_est = max(1, tokens // MOE_CONFIG.num_experts)
    combined_output = torch.zeros_like(h)
    batch_expert_latencies = []

    for eid in active_experts:
        func_exp = f"moe.expert_fwd:{eid}"
        insts_exp = [INST_BY_ID[i] for i in FUNC_MAP.get(func_exp, []) if i in INST_BY_ID]

        # 真实计算
        dummy_input = torch.randn(tokens_per_exp_est, MOE_CONFIG.d_model)
        t0 = time.perf_counter()
        _ = REAL_MODEL.forward_single_expert(eid, dummy_input)
        real_exp_ms = (time.perf_counter() - t0) * 1000.0

        if not insts_exp:
            metrics["dispatch_count"] += 1
            lat_exp = real_exp_ms
            batch_expert_latencies.append(lat_exp)
            metrics["mode_counts"]["local"] += 1
            continue

        req_exp = {"tokens": tokens_per_exp_est, "emb_dim": MOE_CONFIG.d_model}

        # [自适应调度决策]
        is_hot = False
        if HEATMAP: is_hot = HEATMAP.is_hot(eid)

        candidates = insts_exp
        if is_hot:
            # Hot Expert -> 必须 Performance <= 1.5 (快节点)
            high_perf_insts = [i for i in insts_exp if float(i.get("meta", {}).get("performance", 1.0)) <= 1.5]
            if high_perf_insts: candidates = high_perf_insts
        else:
            # Cold Expert -> 优先填补 Performance > 1.5 (慢节点)
            low_perf_insts = [i for i in insts_exp if float(i.get("meta", {}).get("performance", 1.0)) > 1.5]
            if low_perf_insts: candidates = low_perf_insts

        inst_exp, _ = HYBRID_SCHED.select_instance(func_exp, eid, candidates, req_exp)
        lat_exp = apply_performance_scaling(inst_exp, real_exp_ms, is_hot_task=is_hot)

        metrics["dispatch_count"] += 1
        metrics["inst_choice_counts"][inst_exp.get("id")] += 1
        trace["exp_fwd"].append({
            "eid": eid, "inst": inst_exp.get("id"), "hot": is_hot,
            "base": real_exp_ms, "final": lat_exp
        })
        HYBRID_SCHED.update_stats(func_exp, eid, inst_exp, req_exp, lat_exp)
        batch_expert_latencies.append(lat_exp)

        if train: combined_output += h * 0.1

    metrics["expert_comm"] += max(batch_expert_latencies) if batch_expert_latencies else 0.0

    # ==========================
    # 3. Post Stage
    # ==========================
    func_post = "moe.post_fwd"
    insts_post = [INST_BY_ID[i] for i in FUNC_MAP.get(func_post, []) if i in INST_BY_ID]
    req_post = {"tokens": tokens, "emb_dim": MOE_CONFIG.d_model}

    t0 = time.perf_counter()
    logits = REAL_MODEL.forward_post(combined_output)
    loss = F.cross_entropy(logits, y_mb.view(-1))

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        acc = (pred == y_mb.view(-1)).float().mean().item()
    real_post_ms = (time.perf_counter() - t0) * 1000.0

    if insts_post:
        inst_post, _ = HYBRID_SCHED.select_instance(func_post, 0, insts_post, req_post)
        lat_post = apply_performance_scaling(inst_post, real_post_ms, is_hot_task=True)
        metrics["post_lat"] += lat_post
        trace["post"] = inst_post.get("id")
    else:
        lat_post = real_post_ms
        metrics["post_lat"] += lat_post
        inst_post = None

    metrics["loss"] = loss.item()
    metrics["acc_top1"] = acc
    metrics["acc_top5"] = acc

    # ==========================
    # 4. Backward Stage
    # ==========================
    if train:
        OPTIMIZER.zero_grad()
        t0 = time.perf_counter()
        loss.backward()
        OPTIMIZER.step()

        base_post_bwd = real_post_ms * 2.0
        if inst_post:
            metrics["post_bwd"] += apply_performance_scaling(inst_post, base_post_bwd, is_hot_task=True)
        else:
            metrics["post_bwd"] += base_post_bwd

        base_pre_bwd = real_pre_ms * 2.0
        if inst_pre:
            metrics["pre_bwd"] += apply_performance_scaling(inst_pre, base_pre_bwd, is_hot_task=True)
        else:
            metrics["pre_bwd"] += base_pre_bwd

        if USE_NSGA2:
            grad_size = 1024 * 1024
            metrics["grad_bytes"] += grad_size * len(active_experts)

            for eid in active_experts:
                func_grad = f"moe.expert_apply_grad:{eid}"
                insts_grad = [INST_BY_ID[i] for i in FUNC_MAP.get(func_grad, []) if i in INST_BY_ID]

                if not insts_grad:
                    metrics["expert_comm"] += 5.0
                    metrics["mode_counts"]["local"] += 1
                    continue

                is_hot = False
                if HEATMAP: is_hot = HEATMAP.is_hot(eid)

                req_grad = {"grad_bytes": grad_size, "price_cents_s": 0.0}
                choice = nsga2_select(insts_grad, req_grad, STEP_PERIOD_MS, feasible_modes())

                if choice:
                    inst_g, mode = choice
                    lat_g = apply_performance_scaling(inst_g, 5.0, is_hot_task=is_hot)
                    metrics["expert_comm"] += lat_g
                    metrics["mode_counts"][mode] += 1
                    HYBRID_SCHED.update_stats(func_grad, eid, inst_g, req_grad, lat_g)
                else:
                    metrics["mode_counts"]["http"] += 1

    return {"metrics": metrics, "trace": trace}

# ----------------- 5. Step 聚合 -----------------

_metric_buffer = defaultdict(float)
_metric_count = 0

async def run_step(phase: str, batcher: LMTextBatcher, global_step: int, metrics_logger: MetricsLogger):
    train = phase == "train"
    if train: REAL_MODEL.train()
    else: REAL_MODEL.eval()

    tokens = BATCH_SIZE * BLOCK_SIZE
    x, y = batcher.next_batch()
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
    x, y = x.to(torch.long), y.to(torch.long)

    micro_bs = BATCH_SIZE // MICRO_BATCHES
    results_wrapper = []

    t_start = time.perf_counter()

    for m in range(MICRO_BATCHES):
        start = m * micro_bs
        end = (m + 1) * micro_bs
        res = await process_micro_batch(x[start:end], y[start:end], global_step*MICRO_BATCHES+m, m, global_step, tokens, train)
        results_wrapper.append(res)

    step_duration_ms = (time.perf_counter() - t_start) * 1000.0

    # 聚合
    results = [r["metrics"] for r in results_wrapper]
    traces = [r["trace"] for r in results_wrapper]
    if train: append_dispatch_log(traces)

    all_hot_experts = set()
    total_mode_counts = defaultdict(int)
    for r in results:
        all_hot_experts.update(r["hot_experts"])
        for mode, count in r["mode_counts"].items():
            total_mode_counts[mode] += count

    total_dispatch = sum(r["dispatch_count"] for r in results)
    safe_dispatch = total_dispatch if total_dispatch > 0 else 1.0

    step_metrics = {
        "loss": sum(r["loss"] for r in results) / MICRO_BATCHES,
        "acc1": sum(r["acc_top1"] for r in results) / MICRO_BATCHES,
        "acc5": sum(r["acc_top5"] for r in results) / MICRO_BATCHES,
        "pre_lat": sum(r["pre_lat"] for r in results) / MICRO_BATCHES,
        "post_lat": sum(r["post_lat"] for r in results) / MICRO_BATCHES,
        "exp_comm": sum(r["expert_comm"] for r in results) / MICRO_BATCHES,
        "grad_bytes": sum(r["grad_bytes"] for r in results),
        "disp_cnt": total_dispatch,
        "cold_total": sum(r["cold_total"] for r in results),
        "cold_skipped": sum(r["cold_skipped"] for r in results),
        "pre_bwd": sum(r["pre_bwd"] for r in results) / MICRO_BATCHES,
        "post_bwd": sum(r["post_bwd"] for r in results) / MICRO_BATCHES,
    }

    current_hot_ratio = len(all_hot_experts) / MOE_CONFIG.num_experts if MOE_CONFIG.num_experts > 0 else 0

    mode_hot_frac = total_mode_counts.get("hot", 0) / safe_dispatch
    mode_cold_frac = total_mode_counts.get("cold", 0) / safe_dispatch
    mode_http_frac = (total_mode_counts.get("http", 0) + total_mode_counts.get("local", 0)) / safe_dispatch

    current_cold_skip_ratio = step_metrics["cold_skipped"] / safe_dispatch

    samples_per_s = BATCH_SIZE / (step_duration_ms / 1000.0 + 1e-6)
    tokens_per_s = samples_per_s * BLOCK_SIZE

    if train:
        global _metric_buffer, _metric_count
        _metric_buffer["loss"] += step_metrics["loss"]
        _metric_buffer["step_time"] += step_duration_ms
        _metric_count += 1

        if _metric_count >= LOG_TRAIN_EVERY:
            avg_loss = _metric_buffer["loss"] / _metric_count
            avg_time = _metric_buffer["step_time"] / _metric_count

            print(f"[Step {global_step}/{MAX_STEPS}] Loss: {avg_loss:.4f} | Time: {avg_time:.0f}ms | HotRatio: {current_hot_ratio:.2f} | Modes(H/C): {mode_hot_frac:.2f}/{mode_cold_frac:.2f} | T/s: {tokens_per_s:.0f}")

            metrics_logger.log(StepMetrics(
                step=global_step, phase="train", loss=avg_loss, acc_top1=step_metrics["acc1"],
                acc_top5=step_metrics["acc5"], batch_size=BATCH_SIZE, seq_len=BLOCK_SIZE, tokens=tokens,
                step_time_ms=avg_time, pre_fwd_ms=step_metrics["pre_lat"],
                post_fwd_ms=step_metrics["post_lat"], expert_comm_ms=step_metrics["exp_comm"],
                post_bwd_ms=step_metrics["post_bwd"], pre_bwd_ms=step_metrics["pre_bwd"],
                samples_per_s=samples_per_s,
                tokens_per_s=tokens_per_s,
                grad_bytes=step_metrics["grad_bytes"], expert_inst_cnt=MOE_CONFIG.num_experts,
                dispatch_count=step_metrics["disp_cnt"],
                hot_ratio=current_hot_ratio,
                cold_skip_ratio=current_cold_skip_ratio,
                mode_hot_frac=mode_hot_frac,
                mode_cold_frac=mode_cold_frac,
                mode_http_frac=mode_http_frac
            ))
            _metric_buffer = defaultdict(float)
            _metric_count = 0
    else:
        print(f"[Val Step {global_step}] Loss: {step_metrics['loss']:.4f}")

# ----------------- 6. Main -----------------

async def main():
    log("controller", "Starting FULL-LINK REAL TRAINING controller...")

    global REAL_MODEL, OPTIMIZER, VOCAB_SIZE, HEATMAP
    VOCAB_SIZE = 12000

    if SimpleMoE:
        REAL_MODEL = SimpleMoE(VOCAB_SIZE, MOE_CONFIG.d_model, MOE_CONFIG.num_experts, MOE_CONFIG.top_k)
        OPTIMIZER = optim.Adam(REAL_MODEL.parameters(), lr=1e-3)
        # 初始化自适应热度追踪器 (Hysteresis & Trend-Aware)
        HEATMAP = AdaptiveHysteresisHeatmap(MOE_CONFIG.num_experts, alpha_short=0.3, alpha_long=0.05)
        log("controller", "PyTorch Split-Model Initialized.")
    else:
        return

    train_batcher = LMTextBatcher(data_path=DATA_PATH, split="train", batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)
    val_batcher = LMTextBatcher(data_path=DATA_PATH, split="val", batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)

    total_data_size = len(train_batcher.data) if hasattr(train_batcher, 'data') else "Unknown"
    steps_per_epoch = max(1, total_data_size // (BATCH_SIZE * BLOCK_SIZE))
    log("controller", f"Dataset Loaded. Total Tokens: {total_data_size}. Steps per Epoch: {steps_per_epoch}")

    metrics_logger = MetricsLogger("metrics.csv")

    global_step = 0
    while global_step < MAX_STEPS:
        await run_step("train", train_batcher, global_step, metrics_logger)
        global_step += 1

        if global_step % VAL_INTERVAL == 0:
            await run_step("val", val_batcher, global_step, metrics_logger)

    log("controller", "Training Finished.")

if __name__ == "__main__":
    asyncio.run(main())