# controller.py
"""
训练控制器 (ICWS Edition) - TRUE MoE Version (Two-Optimizer, Hot/Cold Update + Capacity)
[ICWS Strong Metrics Version]
- 每 step 记录真实值（metrics.csv 每 step 一行）
- 指标增强（论文友好）：
  1) Forward/Backward 分离记录：
     - fwd：按专家次数 hot/cold/local 分布 + 按 token 流量 hot/cold/local 分布
     - bwd：pre_bwd_ms / post_bwd_ms（模拟“反向阶段 serverless invocations”）
     - grad：NSGA2 选择的通信 mode 分布（按次数）+ grad_inv_total/queue/cold/net/compute 分解
  2) capacity/overflow：按 step token 口径算 capacity，避免 micro 并行导致 overflow 假高
  3) Active Hot Ratio + Hot flip 动态（证明热/冷识别在变化）
  4) 冷专家延迟更新强度：skip/update、apply_steps_avg、grad_scale_avg + cold_pending_steps_avg + cold_update_hit_cnt

【建议环境变量（不用改代码先提升“论文现象”）】
- HOTSPOT_DRIFT_EVERY=50
- ALPHA_SHORT=0.45
- HOT_HIGH_MUL=1.25
- HOT_LOW_MUL=0.80
- COLD_ACC_STEPS=2
- CAPACITY_FACTOR=2.0
"""

import os
import asyncio
import json
import time
import random
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


# ----------------- 1) Config -----------------
DATA_PATH = os.getenv("DATA_PATH", DATA_PATH_DEFAULT)
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "64"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "500"))
VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "100"))
LOG_TRAIN_EVERY = int(os.getenv("LOG_TRAIN_EVERY", "10"))

# micro-batch 并行
MICRO_BATCHES = int(os.getenv("MICRO_BATCHES", "4"))
PARALLEL_DEGREE = int(os.getenv("PARALLEL_DEGREE", str(MICRO_BATCHES)))

# MoE
VOCAB_SIZE = int(os.getenv("VOCAB_SIZE", "2000"))
LR = float(os.getenv("LR", "1e-3"))
CAPACITY_FACTOR = float(os.getenv("CAPACITY_FACTOR", "2.0"))
OVERFLOW_DROP = os.getenv("OVERFLOW_DROP", "1") == "1"

# NSGA2 / comm
USE_NSGA2 = os.getenv("USE_NSGA2", "1") == "1"
STEP_PERIOD_MS = float(os.getenv("STEP_PERIOD_MS", "200.0"))
INVOKE_RETRIES = int(os.getenv("INVOKE_RETRIES", "3"))

# cold delayed update
COLD_ACC_STEPS = int(os.getenv("COLD_ACC_STEPS", "2"))

# traffic drift（让 hot/cold 真变化）
HOTSPOT_DRIFT_EVERY = int(os.getenv("HOTSPOT_DRIFT_EVERY", "50"))
HOTSPOT_SPAN = int(os.getenv("HOTSPOT_SPAN", "2"))
HOT_PROB = float(os.getenv("HOT_PROB", "0.70"))
WARM_PROB = float(os.getenv("WARM_PROB", "0.15"))

# bwd 模拟系数（让 bwd 有稳定可解释的量级）
BWD_MULT_PRE = float(os.getenv("BWD_MULT_PRE", "2.0"))
BWD_MULT_POST = float(os.getenv("BWD_MULT_POST", "2.0"))
GRAD_BASE_MS = float(os.getenv("GRAD_BASE_MS", "8.0"))  # apply_grad 的“基础计算量”模拟

DEFAULT_NET_LATENCY = float(os.getenv("DEFAULT_NET_LATENCY_MS", "5.0"))
DEFAULT_PERFORMANCE = float(os.getenv("DEFAULT_PERFORMANCE", "1.0"))

# resource pool
INSTANCES_FILE = os.getenv("INSTANCES_FILE", "instances.json")
FUNC_MAP_FILE = os.getenv("FUNC_MAP_FILE", "func_map.json")
DISPATCH_LOG_FILE = os.getenv("DISPATCH_LOG_FILE", "dispatch_trace.jsonl")


# ----------------- 2) Load instances / func_map -----------------
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

REAL_MODEL: SimpleMoE = None
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


# ----------------- 3) Hot/Cold Heatmap (Adaptive Hysteresis) -----------------
class AdaptiveHysteresisHeatmap:
    def __init__(
        self,
        num_experts: int,
        alpha_short: float,
        alpha_long: float,
        high_mul: float,
        low_mul: float,
        trend_discount: float = 0.80,
        eps: float = 1e-9,
        baseline_floor: float = None,
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
        self._flip_accum = 0

    @torch.no_grad()
    def update_from_routing(self, topk_idx: torch.Tensor, topk_vals: torch.Tensor):
        if topk_idx is None or topk_vals is None:
            return
        if topk_idx.ndim != 3 or topk_vals.ndim != 3:
            return

        B, T, K = topk_idx.shape
        denom = float(B * T) + self.eps

        idx = topk_idx.reshape(-1, K)
        vals = topk_vals.reshape(-1, K)
        load = torch.zeros(self.num_experts, device=idx.device, dtype=vals.dtype)
        for kk in range(K):
            load.scatter_add_(0, idx[:, kk], vals[:, kk])
        load = load / denom

        self.short = self.alpha_short * load + (1.0 - self.alpha_short) * self.short
        self.long = self.alpha_long * load + (1.0 - self.alpha_long) * self.long

        base = self.baseline_floor
        if base is None:
            base = 1.0 / max(1, self.num_experts)

        flips = 0
        for e in range(self.num_experts):
            s = float(self.short[e].item())
            l = float(self.long[e].item())

            l_eff = max(l, base)
            high_th = max(self.eps, l_eff * self.high_mul)
            low_th = max(self.eps, l_eff * self.low_mul)

            if s > l:
                high_th *= self.trend_discount

            prev = self.is_hot_state[e]
            if not prev:
                if s >= high_th:
                    self.is_hot_state[e] = True
            else:
                if s <= low_th:
                    self.is_hot_state[e] = False

            if self.is_hot_state[e] != prev:
                flips += 1

        self._flip_accum += flips

    def consume_flip_count(self) -> int:
        c = int(self._flip_accum)
        self._flip_accum = 0
        return c

    def is_hot(self, eid: int) -> bool:
        return self.is_hot_state[eid]

    def hot_ratio(self) -> float:
        if self.num_experts <= 0:
            return 0.0
        return sum(1 for x in self.is_hot_state if x) / float(self.num_experts)


HEATMAP: AdaptiveHysteresisHeatmap = None
HEATMAP_LOCK = asyncio.Lock()


# ----------------- 4) FaaS-like invoke simulation with breakdown -----------------
class InstanceManager:
    def __init__(self, default_keep_alive_ms: float = 30000.0):
        self.last_access: Dict[str, float] = {}
        self.default_keep_alive_ms = default_keep_alive_ms

    def touch(self, inst_id: str):
        self.last_access[inst_id] = time.perf_counter() * 1000.0

    def check_cold_start(self, inst: Dict[str, Any]) -> float:
        inst_id = inst.get("id")
        now = time.perf_counter() * 1000.0
        last = self.last_access.get(inst_id)
        if last is None:
            return float(inst.get("meta", {}).get("cold_start_ms", 100.0))
        if (now - last) > self.default_keep_alive_ms:
            return float(inst.get("meta", {}).get("cold_start_ms", 100.0))
        return 0.0


INSTANCE_MGR = InstanceManager()
INSTANCE_LOCK = asyncio.Lock()

# 简单的“并发->排队”模拟：每个实例一个 semaphore
INSTANCE_SEM: Dict[str, asyncio.Semaphore] = {}
INSTANCE_MAX_CONC_DEFAULT = int(os.getenv("INSTANCE_MAX_CONC_DEFAULT", "1"))


def _get_inst_max_conc(inst: Dict[str, Any]) -> int:
    meta = inst.get("meta", {}) or {}
    mc = meta.get("max_concurrency", None)
    if mc is None:
        # fallback：用 cpu_cores
        cpu = meta.get("cpu_cores", 1)
        try:
            mc = int(cpu)
        except Exception:
            mc = 1
    return max(1, int(mc))


def _get_inst_sem(inst: Dict[str, Any]) -> asyncio.Semaphore:
    iid = inst.get("id")
    if iid not in INSTANCE_SEM:
        INSTANCE_SEM[iid] = asyncio.Semaphore(_get_inst_max_conc(inst) or INSTANCE_MAX_CONC_DEFAULT)
    return INSTANCE_SEM[iid]


async def simulate_invoke_with_breakdown(
    inst: Dict[str, Any],
    base_compute_ms: float,
    *,
    is_hot_task: bool,
) -> Tuple[float, float, float, float, float]:
    """
    返回 (total, queue, cold, net, compute)
    """
    meta = inst.get("meta", {}) or {}

    perf = float(meta.get("performance", DEFAULT_PERFORMANCE) or DEFAULT_PERFORMANCE)
    compute_ms = float(base_compute_ms) / max(perf, 1e-6)

    net_ms = float(meta.get("net_latency", DEFAULT_NET_LATENCY) or DEFAULT_NET_LATENCY)

    async with INSTANCE_LOCK:
        cold_ms = float(INSTANCE_MGR.check_cold_start(inst))
        INSTANCE_MGR.touch(inst.get("id"))

    # queue
    sem = _get_inst_sem(inst)
    t0 = time.perf_counter()
    await sem.acquire()
    queue_ms = (time.perf_counter() - t0) * 1000.0
    try:
        total_ms = queue_ms + cold_ms + net_ms + compute_ms
        await asyncio.sleep(total_ms / 1000.0)
        return float(total_ms), float(queue_ms), float(cold_ms), float(net_ms), float(compute_ms)
    finally:
        sem.release()


async def invoke_with_retry(
    func_name: str,
    logical_id: int,
    candidates: List[Dict[str, Any]],
    req: Dict[str, Any],
    base_compute_ms: float,
    *,
    is_hot_task: bool,
    max_tries: int,
) -> Tuple[Dict[str, Any], Tuple[float, float, float, float, float], int]:
    """
    返回 (inst, (total,queue,cold,net,compute), retry_cnt)
    """
    cand = list(candidates)
    tries = 0
    last_err = None

    while tries < max_tries and cand:
        tries += 1
        inst, _ = HYBRID_SCHED.select_instance(func_name, logical_id, cand, req)
        try:
            breakdown = await simulate_invoke_with_breakdown(inst, base_compute_ms, is_hot_task=is_hot_task)
            retry_cnt = max(0, tries - 1)
            HYBRID_SCHED.update_stats(func_name, logical_id, inst, req, breakdown[0])
            return inst, breakdown, retry_cnt
        except Exception as e:
            last_err = e
            bad = inst.get("id")
            cand = [x for x in cand if x.get("id") != bad]

    raise RuntimeError(f"invoke_with_retry failed: func={func_name} id={logical_id} err={last_err}")


# ----------------- 5) Hotspot drift: make routing dynamic -----------------
def simulate_traffic_skew(
    topk_idx: torch.Tensor,
    topk_vals: torch.Tensor,
    num_experts: int,
    global_step: int,
    hot_prob: float,
    warm_prob: float,
    drift_every: int,
    hot_span: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
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

    phase = max(0, int(global_step // max(1, drift_every)))
    hot0 = (phase * max(1, hot_span)) % num_experts
    hot_set = [(hot0 + i) % num_experts for i in range(max(1, hot_span))]
    warm_e = (hot0 + max(1, hot_span)) % num_experts

    new_idx = topk_idx_3d.clone()
    new_vals = topk_vals_3d.clone()

    rand_vals = torch.rand((B, T), device=device)
    mask_hot = rand_vals < hot_prob
    mask_warm = (rand_vals >= hot_prob) & (rand_vals < hot_prob + warm_prob)

    if K >= 1:
        t0 = new_idx[..., 0]
        if mask_hot.any():
            choices = torch.randint(0, len(hot_set), (int(mask_hot.sum().item()),), device=device)
            hot_ids = torch.tensor(hot_set, device=device, dtype=t0.dtype)[choices]
            t0[mask_hot] = hot_ids
        if mask_warm.any():
            t0[mask_warm] = warm_e
        new_idx[..., 0] = t0

        v0 = new_vals[..., 0]
        v0[mask_hot] = 0.8
        v0[mask_warm] = 1.0
        new_vals[..., 0] = v0

    if K >= 2:
        t1 = new_idx[..., 1]
        if mask_hot.any() and len(hot_set) >= 2:
            t1[mask_hot] = hot_set[1]
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


# ----------------- 6) TRUE MoE dispatch (capacity uses step token scale) -----------------
async def moe_dispatch_and_compute_async(
    h: torch.Tensor,
    topk_idx: torch.Tensor,
    topk_vals: torch.Tensor,
    metrics: Dict[str, Any],
    trace: Dict[str, Any],
    *,
    step_tokens_total: int,
) -> torch.Tensor:
    B, T, D = h.shape
    K = topk_idx.shape[-1]

    flat_h = h.reshape(B * T, D)
    N = flat_h.shape[0]

    flat_idx = topk_idx.reshape(N, K)
    flat_val = topk_vals.reshape(N, K)

    combined = torch.zeros_like(flat_h)

    active_experts: Set[int] = set(flat_idx.unique().tolist())

    overflow_dropped = 0
    overflow_total_assign = 0

    cap = int((CAPACITY_FACTOR * (step_tokens_total / max(1, MOE_CONFIG.num_experts))) + 0.999999)
    metrics["capacity"] = int(cap)

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

        # 真正的 expert 前向（本地算）
        t0 = time.perf_counter()
        y_e = REAL_MODEL.forward_single_expert(eid, x_e)
        real_exp_ms = (time.perf_counter() - t0) * 1000.0

        is_hot = HEATMAP.is_hot(eid) if HEATMAP else False
        req_exp = {"tokens": int(m), "emb_dim": D}

        if not insts_exp:
            metrics["dispatch_count"] += 1
            metrics["expert_comm"] += real_exp_ms
            metrics["mode_counts_expert"]["local"] += 1
            metrics["mode_counts_token"]["local"] += int(m)

            trace["exp_fwd"].append({
                "eid": eid, "inst": None, "hot": is_hot,
                "tokens": int(m),
                "final_ms": real_exp_ms,
                "mode": "local",
            })
            combined.index_add_(0, token_ids, y_e * w_e)
            continue

        # hot/cold 模式（这里将“无服务通信”只分类为 hot/cold/local；你若要 http 也能加）
        mode_key = "hot" if is_hot else "cold"

        inst_exp, (tot, q, cold, net, comp), retry_cnt = await invoke_with_retry(
            func_exp, eid, insts_exp, req_exp, real_exp_ms,
            is_hot_task=is_hot,
            max_tries=INVOKE_RETRIES,
        )

        metrics["dispatch_count"] += 1
        metrics["expert_comm"] += tot

        # forward invoke 分解
        metrics["inv_total_ms"] += tot
        metrics["inv_queue_ms"] += q
        metrics["inv_cold_ms"] += cold
        metrics["inv_net_ms"] += net
        metrics["inv_compute_ms"] += comp
        metrics["inv_retry_cnt"] += int(retry_cnt)

        metrics["mode_counts_expert"][mode_key] += 1
        metrics["mode_counts_token"][mode_key] += int(m)

        trace["exp_fwd"].append({
            "eid": eid, "inst": inst_exp.get("id"), "hot": is_hot,
            "tokens": int(m),
            "final_ms": tot,
            "queue_ms": q, "cold_ms": cold, "net_ms": net, "compute_ms": comp,
            "retry": int(retry_cnt),
            "mode": mode_key,
        })

        combined.index_add_(0, token_ids, y_e * w_e)

    metrics["overflow_total_assignments"] += int(overflow_total_assign)
    metrics["overflow_dropped_assignments"] += int(overflow_dropped)

    return combined.reshape(B, T, D)


# ----------------- 7) Micro-batch forward (returns loss_tensor; bwd simulated in step) -----------------
async def process_micro_batch(
    x_mb: torch.Tensor,
    y_mb: torch.Tensor,
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

        "mode_counts_expert": defaultdict(int),
        "mode_counts_token": defaultdict(int),

        "dispatch_count": 0.0,

        "expert_comm": 0.0,
        "pre_lat": 0.0,
        "post_lat": 0.0,

        "capacity": 0,
        "overflow_total_assignments": 0,
        "overflow_dropped_assignments": 0,

        # forward invoke breakdown
        "inv_total_ms": 0.0,
        "inv_queue_ms": 0.0,
        "inv_cold_ms": 0.0,
        "inv_net_ms": 0.0,
        "inv_compute_ms": 0.0,
        "inv_retry_cnt": 0.0,
    })

    trace = {"step": global_step, "mb": mb_idx, "ts": time.time(), "exp_fwd": []}

    # --- Pre forward ---
    func_pre = "moe.pre_fwd"
    insts_pre = [INST_BY_ID[i] for i in FUNC_MAP.get(func_pre, []) if i in INST_BY_ID]
    req_pre = {"tokens": tokens_mb, "emb_dim": MOE_CONFIG.d_model}

    t0 = time.perf_counter()
    h, topk_vals, topk_idx = REAL_MODEL.forward_pre(x_mb)
    real_pre_ms = (time.perf_counter() - t0) * 1000.0

    # traffic drift -> make heatmap dynamic
    topk_idx, topk_vals = simulate_traffic_skew(
        topk_idx, topk_vals, MOE_CONFIG.num_experts,
        global_step=global_step,
        hot_prob=HOT_PROB, warm_prob=WARM_PROB,
        drift_every=HOTSPOT_DRIFT_EVERY,
        hot_span=HOTSPOT_SPAN,
    )

    if HEATMAP:
        async with HEATMAP_LOCK:
            HEATMAP.update_from_routing(topk_idx, topk_vals)

    if insts_pre:
        inst_pre, (tot, q, cold, net, comp), retry_cnt = await invoke_with_retry(
            func_pre, 0, insts_pre, req_pre, real_pre_ms,
            is_hot_task=True,
            max_tries=INVOKE_RETRIES,
        )
        metrics["pre_lat"] += tot
        metrics["inv_total_ms"] += tot
        metrics["inv_queue_ms"] += q
        metrics["inv_cold_ms"] += cold
        metrics["inv_net_ms"] += net
        metrics["inv_compute_ms"] += comp
        metrics["inv_retry_cnt"] += int(retry_cnt)
    else:
        metrics["pre_lat"] += real_pre_ms
        metrics["inv_compute_ms"] += real_pre_ms

    # --- Expert forward ---
    step_tokens_total = int(BATCH_SIZE * BLOCK_SIZE)
    combined_output = await moe_dispatch_and_compute_async(
        h, topk_idx, topk_vals,
        metrics, trace,
        step_tokens_total=step_tokens_total,
    )

    # --- Post forward + loss ---
    func_post = "moe.post_fwd"
    insts_post = [INST_BY_ID[i] for i in FUNC_MAP.get(func_post, []) if i in INST_BY_ID]
    req_post = {"tokens": tokens_mb, "emb_dim": MOE_CONFIG.d_model}

    t0 = time.perf_counter()
    logits = REAL_MODEL.forward_post(combined_output)
    loss_tensor = F.cross_entropy(logits, y_mb.reshape(-1))
    real_post_ms = (time.perf_counter() - t0) * 1000.0

    with torch.no_grad():
        pred1 = logits.argmax(dim=-1)
        acc1 = (pred1 == y_mb.reshape(-1)).float().mean().item()
        k = min(5, logits.size(-1))
        topk_pred = torch.topk(logits, k=k, dim=-1).indices
        target = y_mb.reshape(-1).unsqueeze(-1)
        acc5 = (topk_pred == target).any(dim=-1).float().mean().item()

    if insts_post:
        inst_post, (tot, q, cold, net, comp), retry_cnt = await invoke_with_retry(
            func_post, 0, insts_post, req_post, real_post_ms,
            is_hot_task=True,
            max_tries=INVOKE_RETRIES,
        )
        metrics["post_lat"] += tot
        metrics["inv_total_ms"] += tot
        metrics["inv_queue_ms"] += q
        metrics["inv_cold_ms"] += cold
        metrics["inv_net_ms"] += net
        metrics["inv_compute_ms"] += comp
        metrics["inv_retry_cnt"] += int(retry_cnt)
    else:
        metrics["post_lat"] += real_post_ms
        metrics["inv_compute_ms"] += real_post_ms

    metrics["loss"] = float(loss_tensor.detach().item())
    metrics["acc_top1"] = float(acc1)
    metrics["acc_top5"] = float(acc5)

    # 为 bwd 模拟保留 forward compute 量级（更稳定）
    metrics["real_pre_ms"] = float(real_pre_ms)
    metrics["real_post_ms"] = float(real_post_ms)

    return {"metrics": metrics, "trace": trace, "loss_tensor": loss_tensor}


# ----------------- 8) Step runner: backward simulation + grad_inv breakdown -----------------
_metric_buffer = defaultdict(float)
_metric_count = 0

# cold pending 统计用（每个 cold expert 当前 pending 了多少 step）
_COLD_PENDING: Dict[int, int] = defaultdict(int)


async def run_step(
    phase: str,
    batcher: LMTextBatcher,
    global_step: int,
    metrics_logger: MetricsLogger,
    steps_per_epoch: int,
):
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

    if train:
        OPT_SHARED.zero_grad(set_to_none=True)
        OPT_EXPERT.zero_grad(set_to_none=True)

    sem = asyncio.Semaphore(max(1, PARALLEL_DEGREE))

    async def _guarded_micro(m: int):
        async with sem:
            start = m * micro_bs
            end = min((m + 1) * micro_bs, x.shape[0])
            if start >= end:
                return None
            return await process_micro_batch(
                x_mb=x[start:end],
                y_mb=y[start:end],
                mb_idx=m,
                global_step=global_step,
                train=train,
            )

    t_start = time.perf_counter()
    micro_results = await asyncio.gather(*[_guarded_micro(m) for m in range(MICRO_BATCHES)])
    micro_results = [r for r in micro_results if r is not None]
    step_duration_ms = (time.perf_counter() - t_start) * 1000.0

    results = [r["metrics"] for r in micro_results]
    traces = [r["trace"] for r in micro_results]
    if train:
        append_dispatch_log(traces)

    # ---- (A) real autograd backward (optional) ----
    bwd_total_ms = 0.0
    if train:
        t0 = time.perf_counter()
        total_loss = None
        for r in micro_results:
            lt = r["loss_tensor"] / max(1, MICRO_BATCHES)
            total_loss = lt if total_loss is None else (total_loss + lt)
        if total_loss is not None:
            total_loss.backward()
        bwd_total_ms = (time.perf_counter() - t0) * 1000.0

    # ---- (B) simulate serverless bwd invocations: pre_bwd / post_bwd ----
    # 目的：让 pre_bwd_ms/post_bwd_ms 不为 NaN，并且论文上可解释（“反向阶段也有 serverless 开销”）
    pre_bwd_ms = 0.0
    post_bwd_ms = 0.0
    bwd_inv_total_ms = 0.0
    bwd_inv_queue_ms = 0.0
    bwd_inv_cold_ms = 0.0
    bwd_inv_net_ms = 0.0
    bwd_inv_compute_ms = 0.0
    bwd_inv_retry_cnt = 0

    if train:
        func_pre_bwd = "moe.pre_bwd"
        func_post_bwd = "moe.post_bwd"
        insts_pre_bwd = [INST_BY_ID[i] for i in FUNC_MAP.get(func_pre_bwd, []) if i in INST_BY_ID]
        insts_post_bwd = [INST_BY_ID[i] for i in FUNC_MAP.get(func_post_bwd, []) if i in INST_BY_ID]

        # 基于 forward compute 构造 bwd “计算量”
        avg_pre_real = sum(float(r.get("real_pre_ms", 0.0)) for r in results) / max(1, len(results))
        avg_post_real = sum(float(r.get("real_post_ms", 0.0)) for r in results) / max(1, len(results))
        base_pre_bwd = avg_pre_real * BWD_MULT_PRE
        base_post_bwd = avg_post_real * BWD_MULT_POST

        # pre_bwd
        if insts_pre_bwd:
            inst, (tot, q, cold, net, comp), retry = await invoke_with_retry(
                func_pre_bwd, 0, insts_pre_bwd,
                {"tokens": BATCH_SIZE * BLOCK_SIZE, "stage": "pre_bwd"},
                base_pre_bwd,
                is_hot_task=True,
                max_tries=INVOKE_RETRIES,
            )
            pre_bwd_ms += tot
            bwd_inv_total_ms += tot
            bwd_inv_queue_ms += q
            bwd_inv_cold_ms += cold
            bwd_inv_net_ms += net
            bwd_inv_compute_ms += comp
            bwd_inv_retry_cnt += int(retry)
        else:
            pre_bwd_ms += base_pre_bwd
            bwd_inv_compute_ms += base_pre_bwd

        # post_bwd
        if insts_post_bwd:
            inst, (tot, q, cold, net, comp), retry = await invoke_with_retry(
                func_post_bwd, 0, insts_post_bwd,
                {"tokens": BATCH_SIZE * BLOCK_SIZE, "stage": "post_bwd"},
                base_post_bwd,
                is_hot_task=True,
                max_tries=INVOKE_RETRIES,
            )
            post_bwd_ms += tot
            bwd_inv_total_ms += tot
            bwd_inv_queue_ms += q
            bwd_inv_cold_ms += cold
            bwd_inv_net_ms += net
            bwd_inv_compute_ms += comp
            bwd_inv_retry_cnt += int(retry)
        else:
            post_bwd_ms += base_post_bwd
            bwd_inv_compute_ms += base_post_bwd

    # ---- (C) grad apply + grad_inv breakdown ----
    grad_mode_counts = defaultdict(int)
    grad_total = 0
    grad_bytes = 0.0

    grad_inv_total_ms = 0.0
    grad_inv_queue_ms = 0.0
    grad_inv_cold_ms = 0.0
    grad_inv_net_ms = 0.0
    grad_inv_compute_ms = 0.0
    grad_inv_retry_cnt = 0

    if train and USE_NSGA2:
        grad_size = 1024 * 1024  # mock
        active_eids_set: Set[int] = set()
        for tr in traces:
            for t in tr.get("exp_fwd", []):
                if "eid" in t:
                    active_eids_set.add(int(t["eid"]))

        modes = feasible_modes()

        for eid in sorted(active_eids_set):
            grad_total += 1
            grad_bytes += grad_size

            func_grad = f"moe.expert_apply_grad:{eid}"
            insts_grad = [INST_BY_ID[i] for i in FUNC_MAP.get(func_grad, []) if i in INST_BY_ID]
            req_grad = {"grad_bytes": grad_size}

            if not insts_grad:
                grad_mode_counts["local"] += 1
                # local 情况也给一个可比的 compute 量
                grad_inv_compute_ms += GRAD_BASE_MS
                continue

            choice = nsga2_select(
                insts_grad,
                req_grad,
                STEP_PERIOD_MS,
                pop_size=int(os.getenv("NSGA2_POP_SIZE", "30")),
                generations=int(os.getenv("NSGA2_GENS", "8")),
                modes=modes,
            )
            if not choice:
                grad_mode_counts["http"] += 1
                # http fallback：用“等价 compute+net”粗略计
                grad_inv_net_ms += DEFAULT_NET_LATENCY
                grad_inv_compute_ms += GRAD_BASE_MS
                continue

            inst_g, mode = choice
            grad_mode_counts[mode] += 1

            # 对 apply_grad 也做 “invoke breakdown”
            inst, (tot, q, cold, net, comp), retry = await invoke_with_retry(
                func_grad, eid, insts_grad, req_grad,
                base_compute_ms=GRAD_BASE_MS,
                is_hot_task=(HEATMAP.is_hot(eid) if HEATMAP else False),
                max_tries=INVOKE_RETRIES,
            )

            grad_inv_total_ms += tot
            grad_inv_queue_ms += q
            grad_inv_cold_ms += cold
            grad_inv_net_ms += net
            grad_inv_compute_ms += comp
            grad_inv_retry_cnt += int(retry)

    # ---- (D) Two-optimizer updates + cold pending stats ----
    cold_total = 0
    cold_skipped = 0
    cold_updated = 0
    cold_apply_steps_sum = 0.0
    cold_grad_scale_sum = 0.0

    cold_pending_steps_sum = 0.0
    cold_pending_cnt = 0
    cold_update_hit_cnt = 0

    if train:
        # shared
        OPT_SHARED.step()

        # decide which experts to update this step
        update_eids: Set[int] = set()
        for eid in range(MOE_CONFIG.num_experts):
            is_hot = HEATMAP.is_hot(eid) if HEATMAP else False
            if is_hot:
                _COLD_PENDING[eid] = 0
                update_eids.add(eid)
            else:
                cold_total += 1
                _COLD_PENDING[eid] += 1  # pending++

                # pending strength metrics
                cold_pending_steps_sum += float(_COLD_PENDING[eid])
                cold_pending_cnt += 1

                # delayed update decision
                if COLD_ACC_STEPS > 1 and (_COLD_PENDING[eid] % COLD_ACC_STEPS != 0):
                    cold_skipped += 1
                    continue

                # we will update cold now
                update_eids.add(eid)
                cold_update_hit_cnt += 1

        # expert update
        if update_eids:
            # scale cold grads if accumulated
            for eid in update_eids:
                if HEATMAP and (not HEATMAP.is_hot(eid)):
                    apply_steps = max(1, _COLD_PENDING[eid])
                    cold_updated += 1
                    cold_apply_steps_sum += float(apply_steps)
                    cold_grad_scale_sum += float(1.0 / float(apply_steps))
                    if apply_steps > 1:
                        for p in REAL_MODEL.experts[eid].parameters():
                            if p.grad is not None:
                                p.grad.mul_(1.0 / float(apply_steps))

            # mask non-updated experts grads
            stashes: Dict[int, List[torch.Tensor]] = {}
            for eid, expert in enumerate(REAL_MODEL.experts):
                if eid not in update_eids:
                    stash = []
                    for p in expert.parameters():
                        stash.append(p.grad)
                        p.grad = None
                    stashes[eid] = stash

            OPT_EXPERT.step()

            # restore masked grads
            for eid, stash in stashes.items():
                i = 0
                for p in REAL_MODEL.experts[eid].parameters():
                    p.grad = stash[i]
                    i += 1

            # clear & reset pending
            for eid in update_eids:
                for p in REAL_MODEL.experts[eid].parameters():
                    p.grad = None
                if HEATMAP and (not HEATMAP.is_hot(eid)):
                    _COLD_PENDING[eid] = 0

    # ----------------- Aggregate forward metrics -----------------
    fwd_mode_counts_expert = defaultdict(int)
    fwd_mode_counts_token = defaultdict(int)
    dispatch_fwd = 0

    capacity_val = 0
    overflow_total_assign = 0
    overflow_dropped_assign = 0

    active_hot_set: Set[int] = set()
    active_cold_set: Set[int] = set()

    pre_lat = 0.0
    post_lat = 0.0
    exp_comm = 0.0

    inv_total_ms = 0.0
    inv_queue_ms = 0.0
    inv_cold_ms = 0.0
    inv_net_ms = 0.0
    inv_compute_ms = 0.0
    inv_retry_cnt = 0

    loss = 0.0
    acc1 = 0.0
    acc5 = 0.0

    for r in results:
        for mode, count in r["mode_counts_expert"].items():
            fwd_mode_counts_expert[mode] += int(count)
        for mode, tok in r["mode_counts_token"].items():
            fwd_mode_counts_token[mode] += int(tok)

        dispatch_fwd += int(r["dispatch_count"])

        capacity_val = int(r.get("capacity", capacity_val)) or capacity_val
        overflow_total_assign += int(r.get("overflow_total_assignments", 0))
        overflow_dropped_assign += int(r.get("overflow_dropped_assignments", 0))

        active_hot_set |= set(r["hot_experts"])
        active_cold_set |= set(r["cold_experts"])

        pre_lat += float(r.get("pre_lat", 0.0))
        post_lat += float(r.get("post_lat", 0.0))
        exp_comm += float(r.get("expert_comm", 0.0))

        inv_total_ms += float(r.get("inv_total_ms", 0.0))
        inv_queue_ms += float(r.get("inv_queue_ms", 0.0))
        inv_cold_ms += float(r.get("inv_cold_ms", 0.0))
        inv_net_ms += float(r.get("inv_net_ms", 0.0))
        inv_compute_ms += float(r.get("inv_compute_ms", 0.0))
        inv_retry_cnt += int(r.get("inv_retry_cnt", 0))

        loss += float(r.get("loss", 0.0))
        acc1 += float(r.get("acc_top1", 0.0))
        acc5 += float(r.get("acc_top5", 0.0))

    n_mb = max(1, len(results))
    loss /= n_mb
    acc1 /= n_mb
    acc5 /= n_mb

    safe_dispatch = dispatch_fwd if dispatch_fwd > 0 else 1
    safe_tok = (
        fwd_mode_counts_token.get("hot", 0)
        + fwd_mode_counts_token.get("cold", 0)
        + fwd_mode_counts_token.get("local", 0)
    )
    safe_tok = safe_tok if safe_tok > 0 else 1

    fwd_mode_hot_frac = fwd_mode_counts_expert.get("hot", 0) / safe_dispatch
    fwd_mode_cold_frac = fwd_mode_counts_expert.get("cold", 0) / safe_dispatch
    fwd_mode_local_frac = fwd_mode_counts_expert.get("local", 0) / safe_dispatch

    fwd_mode_hot_frac_tok = fwd_mode_counts_token.get("hot", 0) / safe_tok
    fwd_mode_cold_frac_tok = fwd_mode_counts_token.get("cold", 0) / safe_tok
    fwd_mode_local_frac_tok = fwd_mode_counts_token.get("local", 0) / safe_tok

    active_expert_cnt = len(active_hot_set | active_cold_set)
    active_hot_ratio = (len(active_hot_set) / max(1, active_expert_cnt))

    current_hot_ratio = HEATMAP.hot_ratio() if HEATMAP else 0.0
    hot_flip_cnt = HEATMAP.consume_flip_count() if HEATMAP else 0

    overflow_drop_ratio = float(overflow_dropped_assign / max(1, overflow_total_assign))

    samples_per_s = BATCH_SIZE / (step_duration_ms / 1000.0 + 1e-6)
    tokens_per_s = samples_per_s * BLOCK_SIZE

    current_cold_skip_ratio = (cold_skipped / cold_total) if cold_total > 0 else 0.0
    cold_apply_steps_avg = (cold_apply_steps_sum / cold_updated) if cold_updated > 0 else 0.0
    cold_grad_scale_avg = (cold_grad_scale_sum / cold_updated) if cold_updated > 0 else 0.0

    cold_pending_steps_avg = (cold_pending_steps_sum / cold_pending_cnt) if cold_pending_cnt > 0 else 0.0

    grad_mode_hot_frac = grad_mode_counts.get("hot", 0) / max(1, grad_total)
    grad_mode_cold_frac = grad_mode_counts.get("cold", 0) / max(1, grad_total)
    grad_mode_http_frac = grad_mode_counts.get("http", 0) / max(1, grad_total)
    grad_mode_local_frac = grad_mode_counts.get("local", 0) / max(1, grad_total)

    # --------- write CSV (方案A：每 step 真实值) ----------
    metrics_logger.log(StepMetrics(
        epoch=epoch,
        step=global_step,
        step_in_epoch=step_in_epoch,
        phase=phase,

        loss=float(loss),
        acc_top1=float(acc1),
        acc_top5=float(acc5),

        batch_size=BATCH_SIZE,
        seq_len=BLOCK_SIZE,
        tokens=BATCH_SIZE * BLOCK_SIZE,

        step_time_ms=float(step_duration_ms),
        pre_fwd_ms=float(pre_lat / n_mb),
        post_fwd_ms=float(post_lat / n_mb),
        expert_comm_ms=float(exp_comm / n_mb),

        # ✅补齐 bwd（不再 NaN）
        bwd_total_ms=float(bwd_total_ms),
        pre_bwd_ms=float(pre_bwd_ms),
        post_bwd_ms=float(post_bwd_ms),
        bwd_inv_total_ms=float(bwd_inv_total_ms),
        bwd_inv_queue_ms=float(bwd_inv_queue_ms),
        bwd_inv_cold_ms=float(bwd_inv_cold_ms),
        bwd_inv_net_ms=float(bwd_inv_net_ms),
        bwd_inv_compute_ms=float(bwd_inv_compute_ms),
        bwd_inv_retry_cnt=int(bwd_inv_retry_cnt),

        samples_per_s=float(samples_per_s),
        tokens_per_s=float(tokens_per_s),

        grad_bytes=float(grad_bytes),
        grad_total=int(grad_total),
        grad_mode_hot_frac=float(grad_mode_hot_frac),
        grad_mode_cold_frac=float(grad_mode_cold_frac),
        grad_mode_http_frac=float(grad_mode_http_frac),
        grad_mode_local_frac=float(grad_mode_local_frac),

        # ✅补齐 grad invoke breakdown（不再 NaN）
        grad_inv_total_ms=float(grad_inv_total_ms),
        grad_inv_queue_ms=float(grad_inv_queue_ms),
        grad_inv_cold_ms=float(grad_inv_cold_ms),
        grad_inv_net_ms=float(grad_inv_net_ms),
        grad_inv_compute_ms=float(grad_inv_compute_ms),
        grad_inv_retry_cnt=int(grad_inv_retry_cnt),

        dispatch_count=int(dispatch_fwd),
        expert_inst_cnt=int(MOE_CONFIG.num_experts),

        hot_ratio=float(current_hot_ratio),
        active_expert_cnt=int(active_expert_cnt),
        active_hot_ratio=float(active_hot_ratio),
        hot_flip_cnt=int(hot_flip_cnt),

        cold_total_cnt=int(cold_total),
        cold_skipped_cnt=int(cold_skipped),
        cold_updated_cnt=int(cold_updated),
        cold_skip_ratio=float(current_cold_skip_ratio),
        cold_apply_steps_avg=float(cold_apply_steps_avg),
        cold_grad_scale_avg=float(cold_grad_scale_avg),

        # ✅新增：cold pending 强度（让 cold_skip_ratio 可解释）
        cold_pending_steps_avg=float(cold_pending_steps_avg),
        cold_update_hit_cnt=int(cold_update_hit_cnt),

        fwd_mode_hot_frac=float(fwd_mode_hot_frac),
        fwd_mode_cold_frac=float(fwd_mode_cold_frac),
        fwd_mode_local_frac=float(fwd_mode_local_frac),

        fwd_mode_hot_frac_tok=float(fwd_mode_hot_frac_tok),
        fwd_mode_cold_frac_tok=float(fwd_mode_cold_frac_tok),
        fwd_mode_local_frac_tok=float(fwd_mode_local_frac_tok),

        capacity=int(capacity_val),
        overflow_total_assignments=int(overflow_total_assign),
        overflow_dropped_assignments=int(overflow_dropped_assign),
        overflow_drop_ratio=float(overflow_drop_ratio),

        inv_total_ms=float(inv_total_ms),
        inv_queue_ms=float(inv_queue_ms),
        inv_cold_ms=float(inv_cold_ms),
        inv_net_ms=float(inv_net_ms),
        inv_compute_ms=float(inv_compute_ms),
        inv_retry_cnt=int(inv_retry_cnt),
    ))

    # 控制台打印（窗口平均）
    global _metric_buffer, _metric_count
    if train:
        _metric_buffer["loss"] += float(loss)
        _metric_buffer["step_time"] += float(step_duration_ms)
        _metric_count += 1

        if _metric_count >= LOG_TRAIN_EVERY:
            avg_loss = _metric_buffer["loss"] / _metric_count
            avg_time = _metric_buffer["step_time"] / _metric_count
            print(
                f"[Step {global_step}/{MAX_STEPS}] "
                f"Loss(avg{LOG_TRAIN_EVERY}): {avg_loss:.4f} | "
                f"Time(avg{LOG_TRAIN_EVERY}): {avg_time:.0f}ms | "
                f"HotRatio: {current_hot_ratio:.2f} | ActiveHot: {active_hot_ratio:.2f} | Flip: {hot_flip_cnt} | "
                f"OverflowDrop: {overflow_drop_ratio:.3f} | "
                f"ColdSkip: {current_cold_skip_ratio:.2f} | ColdPendingAvg: {cold_pending_steps_avg:.2f} | "
                f"GradInv(total/queue/cold/net/comp): "
                f"{grad_inv_total_ms:.1f}/{grad_inv_queue_ms:.1f}/{grad_inv_cold_ms:.1f}/{grad_inv_net_ms:.1f}/{grad_inv_compute_ms:.1f}"
            )
            _metric_buffer = defaultdict(float)
            _metric_count = 0


# ----------------- 9) Main -----------------
async def main():
    log("controller", "Starting controller (ICWS Strong Metrics)...")

    global REAL_MODEL, OPT_SHARED, OPT_EXPERT, HEATMAP

    if not SimpleMoE:
        return

    if MOE_CONFIG.top_k >= MOE_CONFIG.num_experts:
        raise RuntimeError(
            f"[ConfigError] top_k({MOE_CONFIG.top_k}) must be < num_experts({MOE_CONFIG.num_experts})."
        )

    # build model
    REAL_MODEL = SimpleMoE(
        vocab_size=VOCAB_SIZE,
        d_model=MOE_CONFIG.d_model,
        num_experts=MOE_CONFIG.num_experts,
        top_k=MOE_CONFIG.top_k,
    )

    # two optimizer（shared + experts）
    shared_params = []
    for name in ["embed", "gate", "norm", "head", "lm_head"]:
        m = getattr(REAL_MODEL, name, None)
        if m is not None:
            shared_params += list(m.parameters())
    expert_params = []
    for expert in REAL_MODEL.experts:
        expert_params += list(expert.parameters())

    OPT_SHARED = optim.Adam(shared_params, lr=LR)
    OPT_EXPERT = optim.Adam(expert_params, lr=LR)

    # heatmap sensitivity (env-tunable)
    alpha_short = float(os.getenv("ALPHA_SHORT", "0.45"))
    alpha_long = float(os.getenv("ALPHA_LONG", "0.05"))
    high_mul = float(os.getenv("HOT_HIGH_MUL", "1.25"))
    low_mul = float(os.getenv("HOT_LOW_MUL", "0.80"))
    trend_discount = float(os.getenv("TREND_DISCOUNT", "0.80"))

    HEATMAP = AdaptiveHysteresisHeatmap(
        MOE_CONFIG.num_experts,
        alpha_short=alpha_short,
        alpha_long=alpha_long,
        high_mul=high_mul,
        low_mul=low_mul,
        trend_discount=trend_discount,
    )

    # dataset
    train_batcher = LMTextBatcher(data_path=DATA_PATH, split="train", batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)
    val_batcher = LMTextBatcher(data_path=DATA_PATH, split="val", batch_size=BATCH_SIZE, block_size=BLOCK_SIZE)

    total_data_size = len(train_batcher.data) if hasattr(train_batcher, "data") else 1
    steps_per_epoch = max(1, int(total_data_size) // (BATCH_SIZE * BLOCK_SIZE))

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
