"""
训练控制器 (ICWS Edition) - TRUE MoE Version (Two-Optimizer, Hot/Cold Update + Capacity)

关键修复：
- Two-Optimizer：
  * OPT_SHARED：embed + gate + norm + lm_head/head（共享参数，每 step 更新）
  * OPT_EXPERT：experts（专家参数，hot 每 step 更新；cold 跨 step 累积，周期更新）
- HotRatio：来自 HEATMAP.is_hot_state（不再几乎恒为 1）
  * 强制 top_k < num_experts（否则 MoE 退化为“所有专家都被路由”，必然全热）
  * 热化阈值加入 baseline 下限：l_eff=max(long, 1/num_experts 或 BASELINE_FLOOR)
- Traffic skew：当 K>2 时清零 k>=2 的权重，避免“尾部路由”激活更多专家导致全热
- FwdModes：local 分支也计数，避免全 0
- 冷专家延迟更新：COLD_ACC_STEPS 周期更新 + 梯度缩放(1/apply_steps)，避免累积过大
- Capacity factor + overflow drop：限制单专家每步 token assignment 数量，trace 记录 overflow

说明：
- StepMetrics 的 mode_*_frac 口径：仅统计 expert_fwd 前向的 hot/cold/local 分布。
- grad(NSGA2) 的 mode 分布写入 dispatch_trace.jsonl 的 trace["grad"] 里，便于论文后处理。
"""

import os
import asyncio
import json
import time
from typing import Any, Dict, List, Set, Tuple
from collections import defaultdict

import torch
import torch.nn.functional as F
import torch.optim as optim

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

MICRO_BATCHES = int(os.getenv("MICRO_BATCHES", "4"))
COLD_ACC_STEPS = int(os.getenv("COLD_ACC_STEPS", "4"))

VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "12000"))
LR_SHARED = float(os.getenv("LR_SHARED", os.getenv("LR", "1e-3")))
LR_EXPERT = float(os.getenv("LR_EXPERT", os.getenv("LR", "1e-3")))

DEFAULT_NET_LATENCY = 5.0
DEFAULT_PERFORMANCE = 1.0

# MoE capacity（论文必备点）
CAPACITY_FACTOR = float(os.getenv("CAPACITY_FACTOR", "1.25"))  # e.g. 1.0/1.25/1.5/2.0
OVERFLOW_DROP = os.getenv("OVERFLOW_DROP", "1") == "1"

DISPATCH_LOG_FILE = "dispatch_trace.jsonl"
INSTANCES_FILE = os.getenv("INSTANCES_FILE", "instances.json")
FUNC_MAP_FILE = os.getenv("FUNC_MAP_FILE", "func_map.json")


# ----------------- 2. 资源加载 -----------------

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


_all_instances_data = _load_json(INSTANCES_FILE, [])
ALL_INSTANCES = _all_instances_data.get("instances", []) if isinstance(_all_instances_data, dict) else _all_instances_data
INST_BY_ID = {inst.get("id"): inst for inst in ALL_INSTANCES}
FUNC_MAP = _load_json(FUNC_MAP_FILE, {})

MOE_CONFIG = load_moe_config({k: v for k, v in FUNC_MAP.items() if k.startswith("moe.expert_fwd:")})

REAL_MODEL = None
OPT_SHARED = None
OPT_EXPERT = None


def append_dispatch_log(traces: List[Dict[str, Any]]):
    if not traces:
        return
    try:
        with open(DISPATCH_LOG_FILE, "a", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    except Exception:
        pass


# ----------------- [Core] 双时间尺度 + 趋势感知 + 滞后双阈值 热度机制 -----------------

class AdaptiveHysteresisHeatmap:
    """
    双时间尺度负载追踪 + 趋势感知动态阈值 + 双阈值滞后 (Hysteresis Buffer)

    负载定义（每 step）:
      load[e] = sum_{token,k routed->e} topk_vals / (B*T)
    注意：load 的总和约等于 1（因为每 token 的 topk_vals 在 K 上归一化）。
    """

    def __init__(
        self,
        num_experts: int,
        alpha_short: float = 0.30,
        alpha_long: float = 0.05,
        high_mul: float = 1.50,
        low_mul: float = 0.70,
        trend_discount: float = 0.80,
        eps: float = 1e-9,
        baseline_floor: float = None,  # 若 None，则用 1/num_experts
    ):
        self.num_experts = num_experts
        self.alpha_short = alpha_short
        self.alpha_long = alpha_long
        self.high_mul = high_mul
        self.low_mul = low_mul
        self.trend_discount = trend_discount
        self.eps = eps

        self.short = torch.zeros(num_experts)
        self.long = torch.zeros(num_experts)
        self.is_hot_state = [False] * num_experts

        self.baseline_floor = baseline_floor

    @torch.no_grad()
    def update_from_routing(self, topk_idx: torch.Tensor, topk_vals: torch.Tensor):
        if topk_idx is None or topk_vals is None:
            return
        if topk_idx.ndim != 3 or topk_vals.ndim != 3:
            return

        B, T, K = topk_idx.shape
        denom = float(B * T) + self.eps

        idx = topk_idx.reshape(-1, K)   # [N,K]
        vals = topk_vals.reshape(-1, K) # [N,K]
        load = torch.zeros(self.num_experts, device=idx.device, dtype=vals.dtype)

        for kk in range(K):
            load.scatter_add_(0, idx[:, kk], vals[:, kk])

        load = load / denom  # [E], sum(load) ~ 1

        self.short = self.alpha_short * load + (1.0 - self.alpha_short) * self.short
        self.long = self.alpha_long * load + (1.0 - self.alpha_long) * self.long

        base = self.baseline_floor
        if base is None:
            base = 1.0 / max(1, self.num_experts)

        for e in range(self.num_experts):
            s = float(self.short[e].item())
            l = float(self.long[e].item())

            l_eff = max(l, base)  # 防止 early-stage long≈0 导致阈值≈0 -> 全热
            high_th = max(self.eps, l_eff * self.high_mul)
            low_th = max(self.eps, l_eff * self.low_mul)

            if s > l:  # 趋势上升：提前热化
                high_th *= self.trend_discount

            if not self.is_hot_state[e]:
                if s >= high_th:
                    self.is_hot_state[e] = True
            else:
                if s <= low_th:
                    self.is_hot_state[e] = False

    def is_hot(self, eid: int) -> bool:
        return self.is_hot_state[eid]

    def hot_ratio(self) -> float:
        if self.num_experts <= 0:
            return 0.0
        return sum(1 for x in self.is_hot_state if x) / float(self.num_experts)


HEATMAP = None


# ----------------- 3. 冷启动与性能模拟 -----------------

class InstanceManager:
    def __init__(self, default_keep_alive_ms: float = 30000.0):
        self.last_access: Dict[str, float] = {}
        self.default_keep_alive_ms = default_keep_alive_ms
        self.dynamic_keep_alive: Dict[str, float] = {}

    def touch(self, inst_id: str, is_hot_task: bool):
        now = time.perf_counter() * 1000.0
        self.last_access[inst_id] = now
        self.dynamic_keep_alive[inst_id] = self.default_keep_alive_ms * (2.0 if is_hot_task else 1.0)

    def check_cold_start(self, inst: Dict[str, Any]) -> float:
        inst_id = inst.get("id")
        now = time.perf_counter() * 1000.0
        last = self.last_access.get(inst_id)
        keep_alive = self.dynamic_keep_alive.get(inst_id, self.default_keep_alive_ms)

        is_cold = (last is None) or ((now - last) > keep_alive)
        if is_cold:
            return float(inst.get("meta", {}).get("cold_start_ms", 100.0))
        return 0.0


INSTANCE_MGR = InstanceManager()


def apply_performance_scaling(inst: Dict[str, Any], real_compute_time_ms: float, is_hot_task: bool = False) -> float:
    perf = float(inst.get("meta", {}).get("performance", DEFAULT_PERFORMANCE))
    simulated_compute_time = real_compute_time_ms / max(perf, 1e-6)

    cold_delay = INSTANCE_MGR.check_cold_start(inst)
    INSTANCE_MGR.touch(inst.get("id"), is_hot_task)

    net_delay = float(inst.get("meta", {}).get("net_latency", DEFAULT_NET_LATENCY))
    total_latency = simulated_compute_time + net_delay + cold_delay

    time_diff = total_latency - real_compute_time_ms
    if time_diff > 0:
        time.sleep(time_diff / 1000.0)

    return total_latency


# ----------------- 4. Traffic Skew（修正 idx/vals） -----------------

def simulate_traffic_skew(
    topk_idx: torch.Tensor,
    topk_vals: torch.Tensor,
    num_experts: int,
    hot_prob: float = 0.70,
    warm_prob: float = 0.15,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    修复点：
    - 若 K>2：清零 k>=2 的权重，避免尾部路由激活更多专家 -> 全热
    """
    if num_experts <= 2:
        return topk_idx, topk_vals

    is_2d = (topk_idx.ndim == 2)
    if is_2d:
        topk_idx_3d = topk_idx.unsqueeze(1)
        topk_vals_3d = topk_vals.unsqueeze(1)
    else:
        topk_idx_3d = topk_idx
        topk_vals_3d = topk_vals

    B, T, K = topk_idx_3d.shape
    device = topk_idx_3d.device

    new_idx = topk_idx_3d.clone()
    new_vals = topk_vals_3d.clone()

    rand_vals = torch.rand((B, T), device=device)
    mask_hot = rand_vals < hot_prob
    mask_warm = (rand_vals >= hot_prob) & (rand_vals < hot_prob + warm_prob)

    if K >= 1:
        t0 = new_idx[..., 0]
        t0[mask_hot] = 0
        t0[mask_warm] = 2
        new_idx[..., 0] = t0

        v0 = new_vals[..., 0]
        v0[mask_hot] = 0.8
        v0[mask_warm] = 1.0
        new_vals[..., 0] = v0

    if K >= 2:
        t1 = new_idx[..., 1]
        t1[mask_hot] = 1
        new_idx[..., 1] = t1

        v1 = new_vals[..., 1]
        v1[mask_hot] = 0.2
        v1[mask_warm] = 0.0
        new_vals[..., 1] = v1

    if K > 2:
        new_vals[..., 2:] = 0.0

    denom = new_vals.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    new_vals = new_vals / denom
    new_idx.clamp_(0, num_experts - 1)

    if is_2d:
        return new_idx.squeeze(1), new_vals.squeeze(1)
    return new_idx, new_vals


# ----------------- 5. TRUE MoE Dispatch（前向 + capacity） -----------------

def moe_dispatch_and_compute(
    h: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_vals: torch.Tensor,
    model: SimpleMoE,
    metrics: Dict[str, Any],
    trace: Dict[str, Any],
) -> torch.Tensor:
    """
    前向 MoE：
    - per-expert token dispatch + compute
    - 加权写回 combined
    - capacity factor + overflow drop
    - mode_counts：用于 CSV 的前向分布（hot/cold/local）
    """
    B, T, D = h.shape
    K = topk_idx.shape[-1]

    flat_h = h.reshape(B * T, D)        # [N,D]
    N = flat_h.shape[0]
    flat_idx = topk_idx.reshape(N, K)   # [N,K]
    flat_val = topk_vals.reshape(N, K)  # [N,K]

    combined = torch.zeros_like(flat_h)

    active_experts: Set[int] = set(flat_idx.unique().tolist())

    overflow_dropped = 0
    overflow_total_assign = 0

    cap = int((CAPACITY_FACTOR * (N / max(1, MOE_CONFIG.num_experts))) + 0.999999)

    for eid in active_experts:
        eid = int(eid)
        if HEATMAP and HEATMAP.is_hot(eid):
            metrics["hot_experts"].add(eid)
        else:
            metrics["cold_experts"].add(eid)

    for eid in sorted(active_experts):
        eid = int(eid)
        func_exp = f"moe.expert_fwd:{eid}"
        insts_exp = [INST_BY_ID[i] for i in FUNC_MAP.get(func_exp, []) if i in INST_BY_ID]

        mask = (flat_idx == eid)
        if not mask.any():
            continue

        token_ids, k_ids = torch.nonzero(mask, as_tuple=True)
        m = int(token_ids.numel())
        if m == 0:
            continue

        overflow_total_assign += m

        if OVERFLOW_DROP and m > cap:
            token_ids = token_ids[:cap]
            k_ids = k_ids[:cap]
            overflow_dropped += (m - cap)
            m = cap

        x_e = flat_h[token_ids]
        w_e = flat_val[token_ids, k_ids].unsqueeze(-1)

        t0 = time.perf_counter()
        y_e = model.forward_single_expert(eid, x_e)
        real_exp_ms = (time.perf_counter() - t0) * 1000.0

        is_hot = HEATMAP.is_hot(eid) if HEATMAP else False

        if not insts_exp:
            metrics["dispatch_count"] += 1
            metrics["expert_comm"] += real_exp_ms
            metrics["mode_counts"]["local"] += 1

            trace["exp_fwd"].append({
                "eid": eid, "inst": None, "hot": is_hot,
                "tokens": int(m),
                "base": real_exp_ms, "final": real_exp_ms,
                "mode": "local",
            })
            combined.index_add_(0, token_ids, y_e * w_e)
            continue

        candidates = insts_exp
        if is_hot:
            high_perf = [i for i in insts_exp if float(i.get("meta", {}).get("performance", 1.0)) <= 1.5]
            if high_perf:
                candidates = high_perf
        else:
            low_perf = [i for i in insts_exp if float(i.get("meta", {}).get("performance", 1.0)) > 1.5]
            if low_perf:
                candidates = low_perf

        req_exp = {"tokens": int(m), "emb_dim": D}
        inst_exp, _ = HYBRID_SCHED.select_instance(func_exp, eid, candidates, req_exp)
        lat_exp = apply_performance_scaling(inst_exp, real_exp_ms, is_hot_task=is_hot)

        metrics["dispatch_count"] += 1
        metrics["inst_choice_counts"][inst_exp.get("id")] += 1
        metrics["expert_comm"] += lat_exp
        metrics["mode_counts"]["hot" if is_hot else "cold"] += 1

        trace["exp_fwd"].append({
            "eid": eid, "inst": inst_exp.get("id"), "hot": is_hot,
            "tokens": int(m),
            "base": real_exp_ms, "final": lat_exp,
            "mode": "hot" if is_hot else "cold",
        })
        HYBRID_SCHED.update_stats(func_exp, eid, inst_exp, req_exp, lat_exp)

        combined.index_add_(0, token_ids, y_e * w_e)

    trace["overflow"] = {
        "capacity": cap,
        "total_assignments": int(overflow_total_assign),
        "dropped_assignments": int(overflow_dropped),
        "drop_ratio": float(overflow_dropped / max(1, overflow_total_assign)),
    }

    return combined.reshape(B, T, D)


# ----------------- 6. 梯度工具（只对 expert 做延迟更新） -----------------

def _iter_params(module: torch.nn.Module):
    for p in module.parameters():
        if p.requires_grad:
            yield p

@torch.no_grad()
def _scale_grads(module: torch.nn.Module, scale: float):
    for p in _iter_params(module):
        if p.grad is not None:
            p.grad.mul_(scale)

@torch.no_grad()
def _stash_grads(module: torch.nn.Module) -> List[torch.Tensor]:
    stash = []
    for p in _iter_params(module):
        stash.append(p.grad)
        p.grad = None
    return stash

@torch.no_grad()
def _restore_grads(module: torch.nn.Module, stash: List[torch.Tensor]):
    i = 0
    for p in _iter_params(module):
        p.grad = stash[i]
        i += 1

@torch.no_grad()
def _zero_grads(module: torch.nn.Module):
    for p in _iter_params(module):
        p.grad = None


# cold expert 累积计数（跨 step / micro）
_COLD_ACC_COUNTER: Dict[int, int] = defaultdict(int)


# ----------------- 7. Micro-batch（全链路：只 forward+backward，不 step） -----------------

async def process_micro_batch(
    x_mb: torch.Tensor,
    y_mb: torch.Tensor,
    micro_id: int,
    mb_idx: int,
    global_step: int,
    train: bool,
) -> Dict[str, Any]:

    B = x_mb.shape[0]
    T = x_mb.shape[1]
    tokens_mb = B * T

    metrics = defaultdict(float, {
        "hot_experts": set(),
        "cold_experts": set(),

        "mode_counts": defaultdict(int),
        "inst_choice_counts": defaultdict(int),

        "cold_total": 0.0,
        "cold_skipped": 0.0,

        "dispatch_count": 0.0,
        "grad_bytes": 0.0,

        "expert_comm": 0.0,
        "pre_lat": 0.0,
        "post_lat": 0.0,
        "pre_bwd": 0.0,
        "post_bwd": 0.0,
    })

    trace = {
        "step": global_step,
        "mb": mb_idx,
        "ts": time.time(),
        "exp_fwd": [],
        "grad": {
            "mode_counts": defaultdict(int),
            "total": 0,
            "skipped": 0,
        }
    }

    # ==========================
    # 1) Pre
    # ==========================
    func_pre = "moe.pre_fwd"
    insts_pre = [INST_BY_ID[i] for i in FUNC_MAP.get(func_pre, []) if i in INST_BY_ID]
    req_pre = {"tokens": tokens_mb, "emb_dim": MOE_CONFIG.d_model}

    t0 = time.perf_counter()
    h, topk_vals, topk_idx = REAL_MODEL.forward_pre(x_mb)
    real_pre_ms = (time.perf_counter() - t0) * 1000.0

    topk_idx, topk_vals = simulate_traffic_skew(topk_idx, topk_vals, MOE_CONFIG.num_experts)

    if HEATMAP:
        HEATMAP.update_from_routing(topk_idx, topk_vals)

    if insts_pre:
        inst_pre, _ = HYBRID_SCHED.select_instance(func_pre, 0, insts_pre, req_pre)
        lat_pre = apply_performance_scaling(inst_pre, real_pre_ms, is_hot_task=True)
        metrics["pre_lat"] += lat_pre
        HYBRID_SCHED.update_stats(func_pre, 0, inst_pre, req_pre, lat_pre)
        trace["pre"] = inst_pre.get("id")
    else:
        metrics["pre_lat"] += real_pre_ms
        inst_pre = None

    # ==========================
    # 2) Expert (TRUE MoE + capacity)
    # ==========================
    combined_output = moe_dispatch_and_compute(h, topk_idx, topk_vals, REAL_MODEL, metrics, trace)

    # ==========================
    # 3) Post + loss/acc
    # ==========================
    func_post = "moe.post_fwd"
    insts_post = [INST_BY_ID[i] for i in FUNC_MAP.get(func_post, []) if i in INST_BY_ID]
    req_post = {"tokens": tokens_mb, "emb_dim": MOE_CONFIG.d_model}

    t0 = time.perf_counter()
    logits = REAL_MODEL.forward_post(combined_output)  # [B*T, vocab]
    loss = F.cross_entropy(logits, y_mb.reshape(-1))

    with torch.no_grad():
        pred1 = logits.argmax(dim=-1)
        acc1 = (pred1 == y_mb.reshape(-1)).float().mean().item()

        k = min(5, logits.size(-1))
        topk_pred = torch.topk(logits, k=k, dim=-1).indices
        target = y_mb.reshape(-1).unsqueeze(-1)
        acc5 = (topk_pred == target).any(dim=-1).float().mean().item()

    real_post_ms = (time.perf_counter() - t0) * 1000.0

    if insts_post:
        inst_post, _ = HYBRID_SCHED.select_instance(func_post, 0, insts_post, req_post)
        lat_post = apply_performance_scaling(inst_post, real_post_ms, is_hot_task=True)
        metrics["post_lat"] += lat_post
        trace["post"] = inst_post.get("id")
    else:
        metrics["post_lat"] += real_post_ms
        inst_post = None

    metrics["loss"] = float(loss.item())
    metrics["acc_top1"] = float(acc1)
    metrics["acc_top5"] = float(acc5)

    # ==========================
    # 4) Backward：只累积梯度（不 step）
    # ==========================
    if train:
        # micro-batch 梯度累积：loss / MICRO_BATCHES
        (loss / max(1, MICRO_BATCHES)).backward()

        base_post_bwd = real_post_ms * 2.0
        metrics["post_bwd"] += apply_performance_scaling(inst_post, base_post_bwd, is_hot_task=True) if inst_post else base_post_bwd

        base_pre_bwd = real_pre_ms * 2.0
        metrics["pre_bwd"] += apply_performance_scaling(inst_pre, base_pre_bwd, is_hot_task=True) if inst_pre else base_pre_bwd

        if USE_NSGA2:
            grad_size = 1024 * 1024
            active_eids = [t["eid"] for t in trace["exp_fwd"]]
            metrics["grad_bytes"] += grad_size * len(active_eids)

            for eid in active_eids:
                eid = int(eid)
                is_hot = HEATMAP.is_hot(eid) if HEATMAP else False

                func_grad = f"moe.expert_apply_grad:{eid}"
                insts_grad = [INST_BY_ID[i] for i in FUNC_MAP.get(func_grad, []) if i in INST_BY_ID]
                req_grad = {"grad_bytes": grad_size, "price_cents_s": 0.0}

                if not insts_grad:
                    metrics["expert_comm"] += 5.0
                    trace["grad"]["mode_counts"]["local"] += 1
                    continue

                choice = nsga2_select(insts_grad, req_grad, STEP_PERIOD_MS, feasible_modes())
                if choice:
                    inst_g, mode = choice
                    lat_g = apply_performance_scaling(inst_g, 5.0, is_hot_task=is_hot)
                    metrics["expert_comm"] += lat_g
                    trace["grad"]["mode_counts"][mode] += 1
                    HYBRID_SCHED.update_stats(func_grad, eid, inst_g, req_grad, lat_g)
                else:
                    trace["grad"]["mode_counts"]["http"] += 1

    trace["grad"]["mode_counts"] = dict(trace["grad"]["mode_counts"])
    return {"metrics": metrics, "trace": trace}


# ----------------- 8. Step 聚合 + Two-Optimizer 更新 -----------------

_metric_buffer = defaultdict(float)
_metric_count = 0


def _get_shared_modules(model: SimpleMoE):
    """
    兼容你的 SimpleMoE：head 可能叫 head 或 lm_head
    """
    head = getattr(model, "head", None)
    if head is None:
        head = getattr(model, "lm_head", None)
    if head is None:
        raise RuntimeError("SimpleMoE missing output head layer (expected model.head or model.lm_head).")

    embed = getattr(model, "embed", None)
    gate = getattr(model, "gate", None)
    norm = getattr(model, "norm", None)

    if embed is None or gate is None or norm is None:
        raise RuntimeError("SimpleMoE missing shared modules (expected embed/gate/norm).")

    return embed, gate, norm, head


async def run_step(phase: str, batcher: LMTextBatcher, global_step: int, metrics_logger: MetricsLogger, steps_per_epoch: int):
    train = (phase == "train")
    REAL_MODEL.train() if train else REAL_MODEL.eval()

    epoch = global_step // max(1, steps_per_epoch)
    step_in_epoch = global_step % max(1, steps_per_epoch)

    x, y = batcher.next_batch()
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x, dtype=torch.long)
        y = torch.tensor(y, dtype=torch.long)
    x = x.to(torch.long)
    y = y.to(torch.long)

    micro_bs = max(1, BATCH_SIZE // max(1, MICRO_BATCHES))
    results_wrapper = []

    # ====== 梯度准备（shared 每步清；expert：hot 每步清，cold 不清以累积）======
    if train:
        # shared grads 每 step 都要清
        OPT_SHARED.zero_grad(set_to_none=True)

        # expert grads：hot 清空；cold 保留（用于跨 step 累积）
        for eid, expert in enumerate(REAL_MODEL.experts):
            is_hot = HEATMAP.is_hot(eid) if HEATMAP else False
            if is_hot:
                _zero_grads(expert)  # hot 不跨步累积

    t_start = time.perf_counter()

    for m in range(MICRO_BATCHES):
        start = m * micro_bs
        end = min((m + 1) * micro_bs, x.shape[0])
        if start >= end:
            break

        res = await process_micro_batch(
            x_mb=x[start:end],
            y_mb=y[start:end],
            micro_id=global_step * MICRO_BATCHES + m,
            mb_idx=m,
            global_step=global_step,
            train=train,
        )
        results_wrapper.append(res)

    step_duration_ms = (time.perf_counter() - t_start) * 1000.0

    results = [r["metrics"] for r in results_wrapper]
    traces = [r["trace"] for r in results_wrapper]
    if train:
        append_dispatch_log(traces)

    # ====== Two-Optimizer 参数更新（只在 train）======
    cold_total = 0.0
    cold_skipped = 0.0

    if train:
        # 1) shared：每步更新
        OPT_SHARED.step()

        # 2) expert：hot 每步更新；cold 每 COLD_ACC_STEPS 更新
        update_eids: Set[int] = set()
        for eid in range(MOE_CONFIG.num_experts):
            is_hot = HEATMAP.is_hot(eid) if HEATMAP else False
            if is_hot:
                _COLD_ACC_COUNTER[eid] = 0
                update_eids.add(eid)
            else:
                _COLD_ACC_COUNTER[eid] += 1
                cold_total += 1.0
                if COLD_ACC_STEPS > 1 and (_COLD_ACC_COUNTER[eid] % COLD_ACC_STEPS != 0):
                    cold_skipped += 1.0
                    continue
                update_eids.add(eid)

        # 对需要更新的 cold expert：做 1/apply_steps 缩放
        for eid in update_eids:
            if HEATMAP and (not HEATMAP.is_hot(eid)):
                apply_steps = max(1, _COLD_ACC_COUNTER[eid])
                if apply_steps > 1:
                    _scale_grads(REAL_MODEL.experts[eid], 1.0 / float(apply_steps))

        # 为了不更新“未到周期的 cold”，我们临时把它们 grad stash 掉
        stashes: Dict[int, List[torch.Tensor]] = {}
        for eid, expert in enumerate(REAL_MODEL.experts):
            if eid not in update_eids:
                stashes[eid] = _stash_grads(expert)

        # 执行一次 expert optimizer step（只会更新有 grad 的专家）
        OPT_EXPERT.step()

        # 恢复未更新 cold 的 grads（继续累积）
        for eid, stash in stashes.items():
            _restore_grads(REAL_MODEL.experts[eid], stash)

        # 对已更新的专家：清 grad，并对 cold 计数归零
        for eid in update_eids:
            _zero_grads(REAL_MODEL.experts[eid])
            if HEATMAP and (not HEATMAP.is_hot(eid)):
                _COLD_ACC_COUNTER[eid] = 0

    # ====== 指标聚合（仅聚合 forward 的 mode_counts 用于 CSV）======
    total_mode_counts_fwd = defaultdict(int)
    total_dispatch_fwd = 0.0

    for r in results:
        for mode, count in r["mode_counts"].items():
            total_mode_counts_fwd[mode] += int(count)
        total_dispatch_fwd += float(r["dispatch_count"])

    safe_dispatch_fwd = total_dispatch_fwd if total_dispatch_fwd > 0 else 1.0

    step_metrics = {
        "loss": sum(r["loss"] for r in results) / max(1, len(results)),
        "acc1": sum(r["acc_top1"] for r in results) / max(1, len(results)),
        "acc5": sum(r["acc_top5"] for r in results) / max(1, len(results)),
        "pre_lat": sum(r["pre_lat"] for r in results) / max(1, len(results)),
        "post_lat": sum(r["post_lat"] for r in results) / max(1, len(results)),
        "exp_comm": sum(r["expert_comm"] for r in results) / max(1, len(results)),
        "grad_bytes": sum(r["grad_bytes"] for r in results),
        "disp_cnt": total_dispatch_fwd,
        "pre_bwd": sum(r["pre_bwd"] for r in results) / max(1, len(results)),
        "post_bwd": sum(r["post_bwd"] for r in results) / max(1, len(results)),
    }

    current_hot_ratio = HEATMAP.hot_ratio() if HEATMAP else 0.0

    mode_hot_frac = total_mode_counts_fwd.get("hot", 0) / safe_dispatch_fwd
    mode_cold_frac = total_mode_counts_fwd.get("cold", 0) / safe_dispatch_fwd
    mode_http_frac = total_mode_counts_fwd.get("local", 0) / safe_dispatch_fwd

    current_cold_skip_ratio = (cold_skipped / cold_total) if cold_total > 0 else 0.0

    samples_per_s = BATCH_SIZE / (step_duration_ms / 1000.0 + 1e-6)
    tokens_per_s = samples_per_s * BLOCK_SIZE

    global _metric_buffer, _metric_count

    if train:
        _metric_buffer["loss"] += step_metrics["loss"]
        _metric_buffer["step_time"] += step_duration_ms
        _metric_count += 1

        if _metric_count >= LOG_TRAIN_EVERY:
            avg_loss = _metric_buffer["loss"] / _metric_count
            avg_time = _metric_buffer["step_time"] / _metric_count

            print(
                f"[Step {global_step}/{MAX_STEPS}] "
                f"Loss: {avg_loss:.4f} | Time: {avg_time:.0f}ms | "
                f"HotRatio: {current_hot_ratio:.2f} | "
                f"FwdModes(H/C/Fallback): {mode_hot_frac:.2f}/{mode_cold_frac:.2f}/{mode_http_frac:.2f} | "
                f"ColdSkip: {current_cold_skip_ratio:.2f} | "
                f"T/s: {tokens_per_s:.0f}"
            )

            metrics_logger.log(StepMetrics(
                epoch=epoch,
                step=global_step,
                step_in_epoch=step_in_epoch,
                phase="train",

                loss=avg_loss,
                acc_top1=step_metrics["acc1"],
                acc_top5=step_metrics["acc5"],

                batch_size=BATCH_SIZE,
                seq_len=BLOCK_SIZE,
                tokens=BATCH_SIZE * BLOCK_SIZE,

                step_time_ms=avg_time,
                pre_fwd_ms=step_metrics["pre_lat"],
                post_fwd_ms=step_metrics["post_lat"],
                post_bwd_ms=step_metrics["post_bwd"],
                pre_bwd_ms=step_metrics["pre_bwd"],
                expert_comm_ms=step_metrics["exp_comm"],

                samples_per_s=samples_per_s,
                tokens_per_s=tokens_per_s,

                grad_bytes=step_metrics["grad_bytes"],
                dispatch_count=int(step_metrics["disp_cnt"]),
                expert_inst_cnt=MOE_CONFIG.num_experts,

                hot_ratio=current_hot_ratio,
                cold_skip_ratio=current_cold_skip_ratio,

                mode_hot_frac=mode_hot_frac,
                mode_cold_frac=mode_cold_frac,
                mode_http_frac=mode_http_frac,
            ))

            _metric_buffer = defaultdict(float)
            _metric_count = 0

    else:
        print(f"[Val Step {global_step}] Loss: {step_metrics['loss']:.4f}")

        metrics_logger.log(StepMetrics(
            epoch=epoch,
            step=global_step,
            step_in_epoch=step_in_epoch,
            phase="val",

            loss=step_metrics["loss"],
            acc_top1=step_metrics["acc1"],
            acc_top5=step_metrics["acc5"],

            batch_size=BATCH_SIZE,
            seq_len=BLOCK_SIZE,
            tokens=BATCH_SIZE * BLOCK_SIZE,

            step_time_ms=step_duration_ms,
            pre_fwd_ms=step_metrics["pre_lat"],
            post_fwd_ms=step_metrics["post_lat"],
            post_bwd_ms=step_metrics["post_bwd"],
            pre_bwd_ms=step_metrics["pre_bwd"],
            expert_comm_ms=step_metrics["exp_comm"],

            samples_per_s=samples_per_s,
            tokens_per_s=tokens_per_s,

            grad_bytes=step_metrics["grad_bytes"],
            dispatch_count=int(step_metrics["disp_cnt"]),
            expert_inst_cnt=MOE_CONFIG.num_experts,

            hot_ratio=current_hot_ratio,
            cold_skip_ratio=current_cold_skip_ratio,

            mode_hot_frac=mode_hot_frac,
            mode_cold_frac=mode_cold_frac,
            mode_http_frac=mode_http_frac,
        ))


# ----------------- 9. Main -----------------

async def main():
    log("controller", "Starting FULL-LINK REAL TRAINING controller (ICWS, Two-Optimizer, Hot/Cold Update + Capacity)...")

    global REAL_MODEL, OPT_SHARED, OPT_EXPERT, HEATMAP

    if not SimpleMoE:
        return

    if MOE_CONFIG.top_k >= MOE_CONFIG.num_experts:
        raise RuntimeError(
            f"[ConfigError] Invalid MoE config: top_k({MOE_CONFIG.top_k}) must be < num_experts({MOE_CONFIG.num_experts})."
        )

    print(f"[DEBUG] num_experts={MOE_CONFIG.num_experts}, top_k={MOE_CONFIG.top_k}, capacity_factor={CAPACITY_FACTOR}")

    REAL_MODEL = SimpleMoE(
        vocab_size=VOCAB_SIZE,
        d_model=MOE_CONFIG.d_model,
        num_experts=MOE_CONFIG.num_experts,
        top_k=MOE_CONFIG.top_k,
    )

    # ===== Two optimizers =====
    embed, gate, norm, head = _get_shared_modules(REAL_MODEL)

    shared_params = (
        list(embed.parameters()) +
        list(gate.parameters()) +
        list(norm.parameters()) +
        list(head.parameters())
    )
    expert_params = []
    for expert in REAL_MODEL.experts:
        expert_params += list(expert.parameters())

    OPT_SHARED = optim.Adam(shared_params, lr=LR_SHARED)
    OPT_EXPERT = optim.Adam(expert_params, lr=LR_EXPERT)

    # baseline floor 解析：空 -> None；否则 float
    bf_env = os.getenv("BASELINE_FLOOR", "")
    baseline_floor = None
    if bf_env.strip() != "":
        try:
            baseline_floor = float(bf_env)
        except Exception:
            baseline_floor = None

    HEATMAP = AdaptiveHysteresisHeatmap(
        MOE_CONFIG.num_experts,
        alpha_short=float(os.getenv("ALPHA_SHORT", "0.30")),
        alpha_long=float(os.getenv("ALPHA_LONG", "0.05")),
        high_mul=float(os.getenv("HOT_HIGH_MUL", "1.50")),
        low_mul=float(os.getenv("HOT_LOW_MUL", "0.70")),
        trend_discount=float(os.getenv("TREND_DISCOUNT", "0.80")),
        baseline_floor=baseline_floor,
    )

    log("controller", "PyTorch Split-Model Initialized.")

    train_batcher = LMTextBatcher(data_path=DATA_PATH, split="train", batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)
    val_batcher = LMTextBatcher(data_path=DATA_PATH, split="val", batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)

    total_data_size = len(train_batcher.data) if hasattr(train_batcher, "data") else "Unknown"
    steps_per_epoch = max(1, (int(total_data_size) if isinstance(total_data_size, int) else 1) // (BATCH_SIZE * BLOCK_SIZE))
    log("controller", f"Dataset Loaded. Total Tokens: {total_data_size}. Steps per Epoch: {steps_per_epoch}")

    metrics_logger = MetricsLogger("metrics.csv")

    global_step = 0
    while global_step < MAX_STEPS:
        await run_step("train", train_batcher, global_step, metrics_logger, steps_per_epoch)
        global_step += 1

        if global_step % VAL_INTERVAL == 0:
            await run_step("val", val_batcher, global_step, metrics_logger, steps_per_epoch)

    log("controller", "Training Finished.")


if __name__ == "__main__":
    asyncio.run(main())
