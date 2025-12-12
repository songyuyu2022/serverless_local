"""
训练控制器 (Final Version)：
- 架构：并行微批次 (Parallel Micro-batches) + 混合调度 (Heuristic + Online NN)
- 功能：
  1. 从本地文本构造 LM 批次
  2. 异步并发调度 pre/post/expert 实例
  3. 在线反馈真实 Latency 给调度器进行实时训练
  4. 记录详细的调度轨迹到 dispatch_trace.jsonl
"""

import os
import asyncio
import json
import time
import math
from typing import Any, Dict, List, Tuple, Set
from collections import defaultdict

import httpx
import torch
import torch.nn.functional as F
import numpy as np

from dataset import LMTextBatcher, DATA_PATH_DEFAULT
from shared import dumps, loads, tensor_to_pack, pack_to_tensor
from nsga2_bw import nsga2_select, feasible_modes
from scheduler_hybrid import HYBRID_SCHED  # 核心调度器
from utils.logger import log
from utils.metrics import MetricsLogger, StepMetrics
from comm import CommManager
from moe_config import load_moe_config

# ----------------- 全局配置 -----------------

DATA_PATH = os.getenv("DATA_PATH", DATA_PATH_DEFAULT)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "8"))
BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "128"))
VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "50"))
LOG_TRAIN_EVERY = int(os.getenv("LOG_TRAIN_EVERY", "20"))

STEP_PERIOD_MS = float(os.getenv("STEP_PERIOD_MS", "200.0"))
USE_NSGA2 = os.getenv("USE_NSGA2", "1") == "1"
COLD_ACC_STEPS = int(os.getenv("COLD_ACC_STEPS", "4"))
DEFAULT_PRICE_CENTS_S = float(os.getenv("DEFAULT_PRICE_CENTS_S", "0.0"))

MICRO_BATCHES = int(os.getenv("MICRO_BATCHES", "1"))

MOE_CONFIG = None
TOP_K_DEFAULT = 2

# ----------------- 日志记录工具 -----------------

DISPATCH_LOG_FILE = "dispatch_trace.jsonl"


def append_dispatch_log(traces: List[Dict[str, Any]]):
    """将调度轨迹追加写入 JSONL 文件，供后续分析"""
    if not traces:
        return
    try:
        with open(DISPATCH_LOG_FILE, "a", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to write dispatch log: {e}")


# ----------------- 统一函数实例池 & 函数映射 -----------------

INSTANCES_FILE = os.getenv("INSTANCES_FILE", "instances.json")
FUNC_MAP_FILE = os.getenv("FUNC_MAP_FILE", "func_map.json")


def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log("controller", f"Loaded {path}")
        return data
    except FileNotFoundError:
        log("controller", f"WARNING: file {path} not found, using default")
        return default
    except Exception as e:
        log("controller", f"ERROR loading {path}: {e}")
        return default


_all_instances_data = _load_json(INSTANCES_FILE, [])
if isinstance(_all_instances_data, dict):
    ALL_INSTANCES: List[Dict[str, Any]] = _all_instances_data.get("instances", [])
elif isinstance(_all_instances_data, list):
    ALL_INSTANCES = _all_instances_data
else:
    raise RuntimeError(
        f"instances.json 格式错误，期望 dict 或 list，实际是 {type(_all_instances_data)}"
    )

INST_BY_ID: Dict[str, Dict[str, Any]] = {
    inst.get("id"): inst for inst in ALL_INSTANCES
}
FUNC_MAP: Dict[str, List[str]] = _load_json(FUNC_MAP_FILE, {})

PRE_STEP_INSTANCE_IDS: Set[str] = set(FUNC_MAP.get("moe.pre_fwd", []))
POST_STEP_INSTANCE_IDS: Set[str] = set(FUNC_MAP.get("moe.post_fwd", []))

EXPERT_INSTANCE_IDS: Set[str] = set()
for fn_name, ids in FUNC_MAP.items():
    if fn_name.startswith("moe.expert_apply_grad:"):
        for iid in ids:
            if iid in INST_BY_ID:
                EXPERT_INSTANCE_IDS.add(iid)

log("controller", f"Loaded {len(ALL_INSTANCES)} instances from {INSTANCES_FILE}")
log("controller", f"Loaded {len(FUNC_MAP)} function mappings from {FUNC_MAP_FILE}")

MOE_CONFIG = load_moe_config(
    {k: v for k, v in FUNC_MAP.items() if k.startswith("moe.expert_fwd:")}
)
TOP_K_DEFAULT = MOE_CONFIG.top_k

# ----------------- 函数候选实例获取 & 调度封装 -----------------


def get_candidate_instances_for_func(func_name: str) -> List[Dict[str, Any]]:
    """根据函数名获取所有候选实例对象"""
    ids = FUNC_MAP.get(func_name, [])
    inst_list: List[Dict[str, Any]] = []
    for iid in ids:
        inst = INST_BY_ID.get(iid)
        if inst is not None:
            inst_list.append(inst)
    return inst_list


def select_instance_for_func(
    func_name: str,
    tokens: int,
    emb_dim: int,
    logical_id: int = 0,
) -> Dict[str, Any]:
    """
    通用调度入口：调用 HybridScheduler 选择最优实例
    """
    instances = get_candidate_instances_for_func(func_name)
    if not instances:
        raise RuntimeError(f"No instances configured for func={func_name}")

    req = {"tokens": int(tokens), "emb_dim": int(emb_dim)}
    try:
        # 使用混合调度器选择实例（Heuristic + NN）
        inst, _ = HYBRID_SCHED.select_instance(
            func_type=func_name,
            logical_id=logical_id,
            instances=instances,
            req=req,
        )
    except Exception as e:
        # 降级策略：直接选列表第一个
        log("controller", f"Scheduler failed for {func_name}: {e}, using fallback.")
        inst = instances[0]

    return inst


# ----------------- 前向调用封装 (含在线学习反馈) -----------------


async def call_pre_fwd(
    client: httpx.AsyncClient,
    x_ids_pack: Dict[str, Any],
    micro_id: int,
    tokens: int,
    emb_dim: int,
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    func_name = "moe.pre_fwd"
    # 1. 调度选择
    inst = select_instance_for_func(
        func_name=func_name,
        tokens=tokens,
        emb_dim=emb_dim,
        logical_id=0,
    )
    url = inst.get("url", "").rstrip("/") + "/fwd"
    payload = {
        "x": x_ids_pack,
        "micro_id": micro_id,
        "tokens": tokens,
        "emb_dim": emb_dim,
    }

    # 2. 执行调用并计时
    t0 = time.perf_counter()
    resp = await client.post(
        url,
        content=dumps(payload),
        headers={"Content-Type": "application/msgpack"},
        timeout=30.0,
    )
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    if resp.status_code != 200:
        raise RuntimeError(f"pre_fn HTTP {resp.status_code}: {resp.text[:200]}")

    pre_resp = loads(resp.content)

    # 3. [核心] 在线更新 NN 调度器
    # 将真实的 Latency 反馈给调度器，触发一次 SGD 更新
    try:
        HYBRID_SCHED.update_stats(
            func_type=func_name,
            logical_id=0,
            inst=inst,
            req={"tokens": int(tokens), "emb_dim": int(emb_dim)},
            latency_ms=latency_ms,
        )
    except Exception as e:
        log("controller", f"Failed to update NN stats: {e}")

    return pre_resp, latency_ms, inst


async def call_post_fwd(
    client: httpx.AsyncClient,
    y_pack: Dict[str, Any],
    targets_pack: Dict[str, Any],
    micro_id: int,
    tokens: int,
    emb_dim: int,
    train: bool,
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    func_name = "moe.post_fwd"
    inst = select_instance_for_func(
        func_name=func_name,
        tokens=tokens,
        emb_dim=emb_dim,
        logical_id=0,
    )
    url = inst.get("url", "").rstrip("/") + "/fwd"

    t0 = time.perf_counter()
    resp = await client.post(
        url,
        content=dumps(
            {
                "y": y_pack,
                "targets": targets_pack,
                "micro_id": micro_id,
                "tokens": tokens,
                "emb_dim": emb_dim,
                "train": train,
            }
        ),
        headers={"Content-Type": "application/msgpack"},
    )
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    # 在线更新 NN
    try:
        HYBRID_SCHED.update_stats(
            func_type=func_name,
            logical_id=0,
            inst=inst,
            req={"tokens": int(tokens), "emb_dim": int(emb_dim)},
            latency_ms=latency_ms,
        )
    except Exception:
        pass

    data = loads(resp.content)
    return data, latency_ms, inst


async def call_expert_fwd_for_eid(
    client: httpx.AsyncClient,
    eid: int,
    x_e: torch.Tensor,
    emb_dim: int,
) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    func_name = f"moe.expert_fwd:{eid}"
    insts = get_candidate_instances_for_func(func_name)
    if not insts:
        # 无可用实例时直接返回原数据
        return x_e, {}, 0.0

    req = {"tokens": int(x_e.shape[0]), "emb_dim": int(emb_dim)}

    # 调度选择
    inst, _ = HYBRID_SCHED.select_instance(
        func_type=func_name,
        logical_id=eid,
        instances=insts,
        req=req,
    )
    url = inst.get("url", "").rstrip("/") + "/fwd"

    payload = {"x": tensor_to_pack(x_e.cpu())}
    t0 = time.perf_counter()
    resp = await client.post(
        url,
        content=dumps(payload),
        headers={"Content-Type": "application/msgpack"},
    )
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    if resp.status_code != 200:
        raise RuntimeError(
            f"expert_fwd HTTP {resp.status_code} for eid={eid}, text={resp.text[:200]}"
        )

    obj = loads(resp.content)
    y_e = pack_to_tensor(obj["y"])

    # 在线更新 NN
    try:
        HYBRID_SCHED.update_stats(
            func_type="moe.expert_fwd",
            logical_id=eid,
            inst=inst,
            req=req,
            latency_ms=latency_ms,
        )
    except Exception:
        pass

    return y_e, inst, latency_ms


def _est_grad_bytes(grad_dict: Dict[str, Any]) -> int:
    """估算梯度大小，用于计算通信开销特征"""
    total = 0
    dtype_size = {
        "float32": 4, "float16": 2, "bfloat16": 2, "float64": 8,
        "int8": 1, "int16": 2, "int32": 4, "int64": 8,
    }
    for name, pack in grad_dict.items():
        shape = pack.get("shape", [])
        dtype = pack.get("dtype", "float32")
        numel = 1
        for d in shape:
            numel *= int(d)
        total += numel * dtype_size.get(dtype, 4)
    return total


# ----------------- 单个微批次处理逻辑 (并行化核心) -----------------


async def process_micro_batch(
    client: httpx.AsyncClient,
    comm_manager: CommManager,
    x_mb: torch.Tensor,
    y_mb: torch.Tensor,
    micro_id: int,
    micro_batch_index: int,  # 当前是第几个微批次
    global_step: int,        # 全局步数
    tokens: int,
    train: bool,
) -> Dict[str, Any]:
    """
    处理单个微批次的完整流程：Pre -> Expert(s) -> Post -> Grad(Back)
    返回包含 metrics 和 trace (调度轨迹) 的字典
    """
    metrics = {
        "loss": 0.0, "acc_top1": 0.0, "acc_top5": 0.0,
        "pre_lat": 0.0, "post_lat": 0.0,
        "pre_bwd": 0.0, "post_bwd": 0.0,
        "expert_comm": 0.0, "grad_bytes": 0,
        "dispatch_count": 0,
        "hot_experts": set(), "cold_experts": set(),
        "cold_total": 0, "cold_skipped": 0,
        "mode_counts": defaultdict(int),
        "inst_choice_counts": defaultdict(int),
    }

    # 调度轨迹结构，用于 jsonl 日志
    trace = {
        "step": global_step,
        "mb_idx": micro_batch_index,
        "ts": time.time(),
        "pre": None,
        "experts_fwd": [],
        "post": None,
        "experts_bwd": [],
    }

    # ---------- 1. pre_fn / fwd ----------
    pre_resp, pre_lat_ms, pre_inst = await call_pre_fwd(
        client=client,
        x_ids_pack=tensor_to_pack(x_mb),
        micro_id=micro_id,
        tokens=tokens,
        emb_dim=0,
    )
    metrics["pre_lat"] += pre_lat_ms
    trace["pre"] = pre_inst.get("id")

    hidden_pack = pre_resp["hidden"]
    gate_pack = pre_resp["gate"]
    if "hot" in pre_resp:
        metrics["hot_experts"].update(pre_resp["hot"])
        metrics["cold_experts"].update(pre_resp.get("cold", []))

    # ---------- 2. Controller Router Logic ----------
    h = pack_to_tensor(hidden_pack).float()
    router_logits = pack_to_tensor(gate_pack).float()

    B_mb, T, D = h.shape
    num_experts = router_logits.shape[-1]
    top_k = max(1, min(TOP_K_DEFAULT, num_experts))

    topk_vals, topk_idx = torch.topk(router_logits, k=top_k, dim=-1)
    gates = F.softmax(topk_vals, dim=-1)

    expert_to_tokens = {}
    topk_idx_np = topk_idx.cpu().numpy()
    gates_np = gates.cpu().numpy()

    for b in range(B_mb):
        for t in range(T):
            for k_id in range(top_k):
                eid = int(topk_idx_np[b, t, k_id])
                gw = float(gates_np[b, t, k_id])
                expert_to_tokens.setdefault(eid, []).append((b, t, k_id, gw))

    merged_h = torch.zeros_like(h)

    # ---------- 3. Parallel Expert Forward ----------
    expert_tasks = []
    expert_eids = []

    for eid, items in expert_to_tokens.items():
        idx_b = [b for (b, t, k_id, gw) in items]
        idx_t = [t for (b, t, k_id, gw) in items]
        x_e = h[idx_b, idx_t, :]
        expert_eids.append((eid, items))
        expert_tasks.append(call_expert_fwd_for_eid(client, eid, x_e, D))

    if expert_tasks:
        # 并发请求所有专家
        expert_results = await asyncio.gather(*expert_tasks)

        for i, (y_e, inst_e, lat_ms) in enumerate(expert_results):
            eid, items = expert_eids[i]
            metrics["expert_comm"] += lat_ms
            inst_id = inst_e.get("id") or inst_e.get("url") or str(inst_e)

            trace["experts_fwd"].append({"eid": eid, "inst": inst_id, "lat": lat_ms})

            # 结果聚合
            cnt = 0
            for b, t, k_id, gw in items:
                merged_h[b, t, :] += gw * y_e[cnt]
                cnt += 1

            if inst_e:
                metrics["dispatch_count"] += 1
                metrics["inst_choice_counts"][inst_id] += 1
    else:
        merged_h = h

    # ---------- 4. post_fn / fwd ----------
    hidden_after_expert_pack = tensor_to_pack(merged_h.cpu())
    targets_pack = tensor_to_pack(y_mb)

    post_resp, post_lat_ms, post_inst = await call_post_fwd(
        client=client,
        y_pack=hidden_after_expert_pack,
        targets_pack=targets_pack,
        micro_id=micro_id,
        tokens=tokens,
        emb_dim=0,
        train=train,
    )
    metrics["post_lat"] += post_lat_ms
    metrics["loss"] = float(post_resp["loss"])
    metrics["acc_top1"] = float(post_resp.get("acc_top1", 0.0))
    metrics["acc_top5"] = float(post_resp.get("acc_top5", 0.0))
    trace["post"] = post_inst.get("id")

    # ---------- 5. Backward Path (Train Only) ----------
    if train:
        # post_fn bwd
        t0 = time.perf_counter()
        grads_pack = post_resp["grads"]
        resp = await client.post(
            post_inst.get("url", "").rstrip("/") + "/bwd",
            content=dumps({"grads": grads_pack, "micro_id": micro_id}),
            headers={"Content-Type": "application/msgpack"},
        )
        metrics["post_bwd"] += (time.perf_counter() - t0) * 1000.0

        rb = loads(resp.content)
        pre_grads = rb.get("pre_grads")

        # pre_fn bwd
        if pre_grads is not None:
            t0 = time.perf_counter()
            await client.post(
                pre_inst.get("url", "").rstrip("/") + "/bwd",
                content=dumps({"grads": pre_grads, "micro_id": micro_id}),
                headers={"Content-Type": "application/msgpack"},
            )
            metrics["pre_bwd"] += (time.perf_counter() - t0) * 1000.0

        # Expert Grads Communication (NSGA-II)
        if USE_NSGA2 and "expert_grads" in rb:
            grads_map = rb["expert_grads"]
            if grads_map:
                grad_bytes = _est_grad_bytes(grads_map)
                metrics["grad_bytes"] += grad_bytes

                step_hot = metrics["hot_experts"]
                step_cold = metrics["cold_experts"]

                for eid_str, g_data in grads_map.items():
                    eid_int = int(eid_str)
                    func_grad_name = f"moe.expert_apply_grad:{eid_str}"
                    inst_list = get_candidate_instances_for_func(func_grad_name)
                    if not inst_list:
                        continue

                    # 冷专家降频更新策略
                    if eid_int in step_cold:
                        metrics["cold_total"] += 1
                        if (micro_id % COLD_ACC_STEPS) != 0:
                            metrics["cold_skipped"] += 1
                            continue

                    # 通信模式选择
                    all_modes = feasible_modes()
                    if eid_int in step_hot:
                        cand_modes = [m for m in all_modes if m in ("hot", "http")]
                    elif eid_int in step_cold:
                        cand_modes = [m for m in all_modes if m in ("cold", "http")]
                    else:
                        cand_modes = list(all_modes)

                    if not cand_modes:
                        continue

                    # NSGA-II 选实例和模式
                    req = {
                        "grad_bytes": grad_bytes,
                        "price_cents_s": DEFAULT_PRICE_CENTS_S,
                    }
                    choice = nsga2_select(
                        inst_list,
                        req,
                        deadline_ms=STEP_PERIOD_MS,
                        pop_size=8,
                        generations=3,
                        seed=42 + micro_id, # 确保并行时种子不同
                        modes=cand_modes,
                    )

                    if choice:
                        inst, mode = choice
                        url = inst.get("url", "").rstrip("/")

                        t_comm0 = time.perf_counter()
                        if mode == "hot":
                            comm_manager.send_hot(eid_str, {eid_str: g_data})
                        elif mode == "cold":
                            comm_manager.send_cold(eid_str, {eid_str: g_data})
                        else:
                            await client.post(
                                url + "/grad/apply",
                                content=dumps({"grads": {eid_str: g_data}}),
                                headers={"Content-Type": "application/msgpack"},
                            )

                        comm_lat = (time.perf_counter() - t_comm0) * 1000.0
                        metrics["expert_comm"] += comm_lat
                        metrics["dispatch_count"] += 1
                        metrics["mode_counts"][mode] += 1
                        inst_id = inst.get("id") or inst.get("url")
                        metrics["inst_choice_counts"][inst_id] += 1

                        trace["experts_bwd"].append({
                            "eid": eid_int,
                            "inst": inst_id,
                            "mode": mode,
                            "lat": comm_lat
                        })

                        # [关键] 更新 NN：把通信时间也作为该实例处理 apply_grad 的性能指标
                        try:
                            HYBRID_SCHED.update_stats(
                                func_type=func_grad_name,
                                logical_id=eid_int,
                                inst=inst,
                                req=req,
                                latency_ms=comm_lat,
                            )
                        except Exception:
                            pass

    return {"metrics": metrics, "trace": trace}


# ----------------- 聚合微批次并执行 Step -----------------


async def run_step(
    phase: str,
    batcher: LMTextBatcher,
    global_step: int,
    metrics_logger: MetricsLogger,
) -> None:
    train = phase == "train"
    tokens = BATCH_SIZE * BLOCK_SIZE

    # 1. 获取 Batch
    x, y = batcher.next_batch()

    micro_batches = MICRO_BATCHES
    micro_bs = BATCH_SIZE // micro_batches

    COMM = CommManager()
    t_step0 = time.perf_counter()

    async with httpx.AsyncClient() as client:
        # 2. 并行执行所有微批次
        tasks = []
        for m in range(micro_batches):
            x_mb = x[m * micro_bs : (m + 1) * micro_bs]
            y_mb = y[m * micro_bs : (m + 1) * micro_bs]

            tasks.append(
                process_micro_batch(
                    client=client,
                    comm_manager=COMM,
                    x_mb=x_mb,
                    y_mb=y_mb,
                    micro_id=global_step * micro_batches + m, # 唯一标识
                    micro_batch_index=m,
                    global_step=global_step,
                    tokens=tokens,
                    train=train,
                )
            )

        # 核心并发点: 等待所有微批次完成
        results_wrapper = await asyncio.gather(*tasks)

    # 3. 结果分离 & 记录 Trace
    results = [r["metrics"] for r in results_wrapper]
    traces = [r["trace"] for r in results_wrapper]

    if train:
        append_dispatch_log(traces)

    # 4. 聚合指标 (求平均或求和)
    agg_loss = sum(r["loss"] for r in results)
    agg_top1 = sum(r["acc_top1"] for r in results)
    agg_top5 = sum(r["acc_top5"] for r in results)

    pre_lat_all = sum(r["pre_lat"] for r in results)
    post_lat_all = sum(r["post_lat"] for r in results)
    post_bwd_all = sum(r["post_bwd"] for r in results)
    pre_bwd_all = sum(r["pre_bwd"] for r in results)
    expert_comm_ms = sum(r["expert_comm"] for r in results)
    grad_bytes = sum(r["grad_bytes"] for r in results)
    expert_inst_cnt = len(EXPERT_INSTANCE_IDS)

    # 集合合并
    hot_experts_step = set().union(*[r["hot_experts"] for r in results])
    cold_experts_step = set().union(*[r["cold_experts"] for r in results])

    dispatch_count = sum(r["dispatch_count"] for r in results)
    cold_total = sum(r["cold_total"] for r in results)
    cold_skipped = sum(r["cold_skipped"] for r in results)

    mode_counts = defaultdict(int)
    inst_choice_counts = defaultdict(int)
    for r in results:
        for k, v in r["mode_counts"].items():
            mode_counts[k] += v
        for k, v in r["inst_choice_counts"].items():
            inst_choice_counts[k] += v

    t_step1 = time.perf_counter()
    step_time_ms = (t_step1 - t_step0) * 1000.0

    # 5. 计算统计指标
    if hot_experts_step or cold_experts_step:
        denom = len(hot_experts_step) + len(cold_experts_step)
        hot_ratio = len(hot_experts_step) / denom
    else:
        hot_ratio = 0.0

    if cold_total > 0:
        cold_skip_ratio = cold_skipped / cold_total
    else:
        cold_skip_ratio = 0.0

    total_dispatch = sum(mode_counts.values()) or 1
    mode_hot_frac = mode_counts["hot"] / total_dispatch
    mode_cold_frac = mode_counts["cold"] / total_dispatch
    mode_http_frac = mode_counts["http"] / total_dispatch

    inst_total = sum(inst_choice_counts.values())
    if inst_total > 0:
        probs = [c / inst_total for c in inst_choice_counts.values()]
        inst_entropy = -sum(p * math.log(p + 1e-12) for p in probs)
    else:
        inst_entropy = 0.0

    # 6. 记录 Step 指标
    step_metrics = StepMetrics(
        step=global_step,
        phase=phase,
        loss=agg_loss / micro_batches,
        acc_top1=agg_top1 / micro_batches,
        acc_top5=agg_top5 / micro_batches,
        batch_size=BATCH_SIZE,
        seq_len=BLOCK_SIZE,
        tokens=tokens,
        pre_fwd_ms=pre_lat_all / micro_batches,
        post_fwd_ms=post_lat_all / micro_batches,
        post_bwd_ms=post_bwd_all / micro_batches,
        pre_bwd_ms=pre_bwd_all / micro_batches,
        step_time_ms=step_time_ms,
        expert_comm_ms=expert_comm_ms,
        grad_bytes=grad_bytes,
        expert_inst_cnt=expert_inst_cnt,
        dispatch_count=dispatch_count,
        hot_ratio=hot_ratio,
        cold_skip_ratio=cold_skip_ratio,
        mode_hot_frac=mode_hot_frac,
        mode_cold_frac=mode_cold_frac,
        mode_http_frac=mode_http_frac,
        inst_entropy=inst_entropy,
    )

    global _train_acc, _train_count

    if phase == "train":
        # 参数更新 (Step)
        # 注意：这里我们模拟“所有微批次完成后，统一进行一次参数更新”
        async with httpx.AsyncClient() as client_step:
            for iid in PRE_STEP_INSTANCE_IDS:
                inst = INST_BY_ID.get(iid)
                if inst:
                    try:
                        await client_step.post(inst.get("url", "").rstrip("/") + "/step")
                    except Exception:
                        pass
            for iid in POST_STEP_INSTANCE_IDS:
                inst = INST_BY_ID.get(iid)
                if inst:
                    try:
                        await client_step.post(inst.get("url", "").rstrip("/") + "/step")
                    except Exception:
                        pass

        # 累积窗口平均值
        _train_count += 1
        _train_acc["loss"] += step_metrics.loss
        _train_acc["acc_top1"] += step_metrics.acc_top1
        _train_acc["acc_top5"] += step_metrics.acc_top5
        _train_acc["pre_fwd_ms"] += step_metrics.pre_fwd_ms
        _train_acc["post_fwd_ms"] += step_metrics.post_fwd_ms
        _train_acc["post_bwd_ms"] += step_metrics.post_bwd_ms
        _train_acc["pre_bwd_ms"] += step_metrics.pre_bwd_ms
        _train_acc["step_time_ms"] += step_metrics.step_time_ms
        _train_acc["expert_comm_ms"] += step_metrics.expert_comm_ms
        _train_acc["grad_bytes"] += step_metrics.grad_bytes
        _train_acc["dispatch_count"] += step_metrics.dispatch_count
        _train_acc["hot_ratio"] += step_metrics.hot_ratio
        _train_acc["cold_skip_ratio"] += step_metrics.cold_skip_ratio
        _train_acc["mode_hot_frac"] += step_metrics.mode_hot_frac
        _train_acc["mode_cold_frac"] += step_metrics.mode_cold_frac
        _train_acc["mode_http_frac"] += step_metrics.mode_http_frac

        if _train_count >= LOG_TRAIN_EVERY:
            avg = 1.0 / _train_count
            avg_metrics = StepMetrics(
                step=global_step,
                phase="train",
                loss=_train_acc["loss"] * avg,
                acc_top1=_train_acc["acc_top1"] * avg,
                acc_top5=_train_acc["acc_top5"] * avg,
                pre_fwd_ms=_train_acc["pre_fwd_ms"] * avg,
                post_fwd_ms=_train_acc["post_fwd_ms"] * avg,
                post_bwd_ms=_train_acc["post_bwd_ms"] * avg,
                pre_bwd_ms=_train_acc["pre_bwd_ms"] * avg,
                step_time_ms=_train_acc["step_time_ms"] * avg,
                expert_comm_ms=_train_acc["expert_comm_ms"] * avg,
                grad_bytes=_train_acc["grad_bytes"] * avg,
                expert_inst_cnt=step_metrics.expert_inst_cnt,
                dispatch_count=_train_acc["dispatch_count"] * avg,
                hot_ratio=_train_acc["hot_ratio"] * avg,
                cold_skip_ratio=_train_acc["cold_skip_ratio"] * avg,
                mode_hot_frac=_train_acc["mode_hot_frac"] * avg,
                mode_cold_frac=_train_acc["mode_cold_frac"] * avg,
                mode_http_frac=_train_acc["mode_http_frac"] * avg,
                inst_entropy=step_metrics.inst_entropy,
                batch_size=BATCH_SIZE,
                seq_len=BLOCK_SIZE,
                tokens=tokens,
            )
            metrics_logger.log(avg_metrics)
            _train_acc = defaultdict(float)
            _train_count = 0
            return
        return

    # val 阶段
    metrics_logger.log(step_metrics)


# ----------------- 主训练循环 -----------------


async def main() -> None:
    log("controller", "Starting training controller")

    # 清空旧的调度日志
    if os.path.exists(DISPATCH_LOG_FILE):
        try:
            os.remove(DISPATCH_LOG_FILE)
            log("controller", f"Removed old dispatch log: {DISPATCH_LOG_FILE}")
        except:
            pass

    train_batcher = LMTextBatcher(
        data_path=DATA_PATH,
        split="train",
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
    )
    val_batcher = LMTextBatcher(
        data_path=DATA_PATH,
        split="val",
        batch_size=BATCH_SIZE,
        block_size=BLOCK_SIZE,
    )

    metrics_logger = MetricsLogger("metrics.csv")

    global_step = 0
    max_steps = int(os.getenv("MAX_STEPS", "4200"))

    while global_step < max_steps:
        phase = "train"
        await run_step(phase, train_batcher, global_step, metrics_logger)
        global_step += 1

        if global_step % VAL_INTERVAL == 0:
            await run_step("val", val_batcher, global_step, metrics_logger)

    log("controller", "Training finished")


_train_acc = defaultdict(float)
_train_count = 0


if __name__ == "__main__":
    asyncio.run(main())