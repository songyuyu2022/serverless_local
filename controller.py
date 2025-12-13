"""
训练控制器 (Final Corrected Metrics Version):
- 核心机制：并行微批次 + 混合调度
- 实验特性：冷启动模拟 + 在线学习闭环 + 资源竞争模拟
- 修复内容：
  1. 修复了 metrics.csv 中 hot/cold 为 0 的问题 (不再硬编码，而是真实统计)
  2. 实现了对 hot_experts/cold_experts 集合的跨微批次聚合
  3. 实现了对通信模式 (hot/cold/http) 的统计
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
from scheduler_hybrid import HYBRID_SCHED
from utils.logger import log
from utils.metrics import MetricsLogger, StepMetrics
from comm import CommManager
from moe_config import load_moe_config

# ----------------- 1. 全局配置 -----------------

DATA_PATH = os.getenv("DATA_PATH", DATA_PATH_DEFAULT)

# [安全配置] 默认 Batch=16 防止 OOM
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "16"))
BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "64"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "1000"))

VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "100"))
LOG_TRAIN_EVERY = int(os.getenv("LOG_TRAIN_EVERY", "10"))

STEP_PERIOD_MS = float(os.getenv("STEP_PERIOD_MS", "200.0"))
USE_NSGA2 = os.getenv("USE_NSGA2", "1") == "1"
COLD_ACC_STEPS = int(os.getenv("COLD_ACC_STEPS", "4"))
DEFAULT_PRICE_CENTS_S = float(os.getenv("DEFAULT_PRICE_CENTS_S", "0.0"))
MICRO_BATCHES = int(os.getenv("MICRO_BATCHES", "4"))

MOE_CONFIG = None
TOP_K_DEFAULT = 2

# ----------------- 2. 日志工具 -----------------

DISPATCH_LOG_FILE = "dispatch_trace.jsonl"

def append_dispatch_log(traces: List[Dict[str, Any]]):
    if not traces: return
    try:
        with open(DISPATCH_LOG_FILE, "a", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    except Exception: pass

# ----------------- 3. 资源加载 -----------------

INSTANCES_FILE = os.getenv("INSTANCES_FILE", "instances.json")
FUNC_MAP_FILE = os.getenv("FUNC_MAP_FILE", "func_map.json")

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f: return json.load(f)
    except: return default

_all_instances_data = _load_json(INSTANCES_FILE, [])
if isinstance(_all_instances_data, dict):
    ALL_INSTANCES = _all_instances_data.get("instances", [])
else:
    ALL_INSTANCES = _all_instances_data

INST_BY_ID = {inst.get("id"): inst for inst in ALL_INSTANCES}
FUNC_MAP = _load_json(FUNC_MAP_FILE, {})

PRE_STEP_INSTANCE_IDS = set(FUNC_MAP.get("moe.pre_fwd", []))
POST_STEP_INSTANCE_IDS = set(FUNC_MAP.get("moe.post_fwd", []))
EXPERT_INSTANCE_IDS = set()
for fn_name, ids in FUNC_MAP.items():
    if fn_name.startswith("moe.expert_apply_grad:"):
        for iid in ids: EXPERT_INSTANCE_IDS.add(iid)

MOE_CONFIG = load_moe_config({k: v for k, v in FUNC_MAP.items() if k.startswith("moe.expert_fwd:")})
TOP_K_DEFAULT = MOE_CONFIG.top_k

log("controller", f"Loaded {len(ALL_INSTANCES)} instances, MicroBatches={MICRO_BATCHES}, BatchSize={BATCH_SIZE}")

# ----------------- 4. 冷启动管理 -----------------

class InstanceManager:
    def __init__(self, keep_alive_ms: float = 30000.0):
        self.last_access: Dict[str, float] = {}
        self.keep_alive_ms = keep_alive_ms
        self._lock = asyncio.Lock()

    async def check_and_warmup(self, inst: Dict[str, Any]) -> float:
        inst_id = inst.get("id")
        now = time.perf_counter() * 1000.0
        delay = 0.0
        async with self._lock:
            last = self.last_access.get(inst_id)
            is_cold = False
            if last is None: is_cold = True
            elif (now - last) > self.keep_alive_ms: is_cold = True

            if is_cold:
                cold_ms = float(inst.get("meta", {}).get("cold_start_ms", 100.0))
                await asyncio.sleep(cold_ms / 1000.0)
                delay = cold_ms
            self.last_access[inst_id] = time.perf_counter() * 1000.0
        return delay

INSTANCE_MGR = InstanceManager()

# ----------------- 5. 调度封装 -----------------

def get_candidate_instances_for_func(func_name: str) -> List[Dict[str, Any]]:
    ids = FUNC_MAP.get(func_name, [])
    return [INST_BY_ID[i] for i in ids if i in INST_BY_ID]

def select_instance_for_func(func_name: str, tokens: int, emb_dim: int, logical_id: int = 0) -> Dict[str, Any]:
    insts = get_candidate_instances_for_func(func_name)
    if not insts: raise RuntimeError(f"No instances for {func_name}")
    req = {"tokens": int(tokens), "emb_dim": int(emb_dim)}
    try:
        inst, _ = HYBRID_SCHED.select_instance(func_name, logical_id, insts, req)
        return inst
    except: return insts[0]

async def call_pre_fwd(client, x_ids_pack, micro_id, tokens, emb_dim):
    func_name = "moe.pre_fwd"
    inst = select_instance_for_func(func_name, tokens, emb_dim)
    cold_delay = await INSTANCE_MGR.check_and_warmup(inst)

    url = inst.get("url", "").rstrip("/") + "/fwd"
    payload = {"x": x_ids_pack, "micro_id": micro_id, "tokens": tokens, "emb_dim": emb_dim}

    try:
        t0 = time.perf_counter()
        resp = await client.post(url, content=dumps(payload), headers={"Content-Type": "application/msgpack"})
        latency_ms = (time.perf_counter() - t0) * 1000.0 + cold_delay

        if resp.status_code != 200:
            raise RuntimeError(f"pre_fn {inst['id']} failed: {resp.status_code}")

        HYBRID_SCHED.update_stats(func_name, 0, inst, {"tokens": tokens, "emb_dim": emb_dim}, latency_ms)
        return loads(resp.content), latency_ms, inst
    except Exception as e:
        print(f"[CRITICAL] call_pre_fwd failed: {e}")
        raise e

async def call_post_fwd(client, y_pack, targets_pack, micro_id, tokens, emb_dim, train):
    func_name = "moe.post_fwd"
    inst = select_instance_for_func(func_name, tokens, emb_dim)
    cold_delay = await INSTANCE_MGR.check_and_warmup(inst)

    url = inst.get("url", "").rstrip("/") + "/fwd"
    payload = {"y": y_pack, "targets": targets_pack, "micro_id": micro_id, "tokens": tokens, "emb_dim": emb_dim, "train": train}

    try:
        t0 = time.perf_counter()
        resp = await client.post(url, content=dumps(payload), headers={"Content-Type": "application/msgpack"})
        latency_ms = (time.perf_counter() - t0) * 1000.0 + cold_delay

        if resp.status_code != 200:
            raise RuntimeError(f"post_fn {inst['id']} failed: {resp.status_code}")

        HYBRID_SCHED.update_stats(func_name, 0, inst, {"tokens": tokens, "emb_dim": emb_dim}, latency_ms)
        return loads(resp.content), latency_ms, inst
    except Exception as e:
        print(f"[CRITICAL] call_post_fwd failed: {e}")
        raise e

async def call_expert_fwd_for_eid(client, eid, x_e, emb_dim):
    func_name = f"moe.expert_fwd:{eid}"
    insts = get_candidate_instances_for_func(func_name)
    if not insts: return x_e, {}, 0.0

    req = {"tokens": int(x_e.shape[0]), "emb_dim": int(emb_dim)}
    inst, _ = HYBRID_SCHED.select_instance(func_name, eid, insts, req)
    cold_delay = await INSTANCE_MGR.check_and_warmup(inst)

    url = inst.get("url", "").rstrip("/") + "/fwd"
    payload = {"x": tensor_to_pack(x_e.cpu())}

    try:
        t0 = time.perf_counter()
        resp = await client.post(url, content=dumps(payload), headers={"Content-Type": "application/msgpack"})
        latency_ms = (time.perf_counter() - t0) * 1000.0 + cold_delay

        if resp.status_code != 200:
            raise RuntimeError(f"expert {eid} failed: {resp.status_code}")

        HYBRID_SCHED.update_stats("moe.expert_fwd", eid, inst, req, latency_ms)
        return pack_to_tensor(loads(resp.content)["y"]), inst, latency_ms
    except Exception as e:
        print(f"[CRITICAL] expert {eid} failed: {e}")
        return x_e, inst, 0.0

def _est_grad_bytes(grad_dict):
    return sum(np.prod(p["shape"]) * 4 for p in grad_dict.values())

# ----------------- 6. 微批次处理 (统计收集) -----------------

async def process_micro_batch(
    client: httpx.AsyncClient,
    comm_manager: CommManager,
    x_mb: torch.Tensor,
    y_mb: torch.Tensor,
    micro_id: int,
    mb_idx: int,
    global_step: int,
    tokens: int,
    train: bool,
) -> Dict[str, Any]:

    # 统计容器
    metrics = defaultdict(float, {
        "hot_experts": set(), "cold_experts": set(),
        "mode_counts": defaultdict(int), "inst_choice_counts": defaultdict(int)
    })

    trace = {"step": global_step, "mb": mb_idx, "ts": time.time(), "pre": None, "post": None, "exp_fwd": [], "exp_bwd": []}

    # 1. Pre
    pre_resp, pre_lat, pre_inst = await call_pre_fwd(client, tensor_to_pack(x_mb), micro_id, tokens, 0)
    metrics["pre_lat"] += pre_lat
    trace["pre"] = pre_inst.get("id")

    # [关键] 收集冷热信息
    if "hot" in pre_resp: metrics["hot_experts"].update(pre_resp["hot"])
    metrics["cold_experts"].update(pre_resp.get("cold", []))

    # 2. Router
    h = pack_to_tensor(pre_resp["hidden"]).float()
    gate_pack = pack_to_tensor(pre_resp["gate"]).float()
    gates = F.softmax(torch.topk(gate_pack, k=TOP_K_DEFAULT, dim=-1)[0], dim=-1)
    indices = torch.topk(gate_pack, k=TOP_K_DEFAULT, dim=-1)[1]

    expert_map = defaultdict(list)
    B, T, _ = h.shape
    idx_np, gate_np = indices.cpu().numpy(), gates.cpu().numpy()

    for b in range(B):
        for t in range(T):
            for k in range(TOP_K_DEFAULT):
                expert_map[int(idx_np[b,t,k])].append((b,t,k, float(gate_np[b,t,k])))

    # 3. Expert Fwd
    merged_h = torch.zeros_like(h)
    tasks, eids = [], []
    for eid, items in expert_map.items():
        idx_b, idx_t = [i[0] for i in items], [i[1] for i in items]
        tasks.append(call_expert_fwd_for_eid(client, eid, h[idx_b, idx_t], h.shape[-1]))
        eids.append((eid, items))

    if tasks:
        results = await asyncio.gather(*tasks)
        for i, (y_e, inst, lat) in enumerate(results):
            metrics["expert_comm"] += lat
            metrics["dispatch_count"] += 1
            inst_id = inst.get("id")
            metrics["inst_choice_counts"][inst_id] += 1
            trace["exp_fwd"].append({"eid": eids[i][0], "inst": inst_id, "lat": lat})
            items = eids[i][1]
            for j, (b, t, k, gw) in enumerate(items):
                merged_h[b, t] += gw * y_e[j]
    else: merged_h = h

    # 4. Post
    post_resp, post_lat, post_inst = await call_post_fwd(client, tensor_to_pack(merged_h.cpu()), tensor_to_pack(y_mb), micro_id, tokens, 0, train)
    metrics["post_lat"] += post_lat
    metrics["loss"] = post_resp["loss"]
    metrics["acc_top1"] = post_resp.get("acc_top1", 0)
    trace["post"] = post_inst.get("id")

    # 5. Backward & Comm
    if train:
        t0 = time.perf_counter()
        resp = await client.post(post_inst.get("url")+"/bwd", content=dumps({"grads": post_resp["grads"]}), headers={"Content-Type": "application/msgpack"})
        metrics["post_bwd"] += (time.perf_counter()-t0)*1000

        rb = loads(resp.content)
        if "pre_grads" in rb:
             t0 = time.perf_counter()
             await client.post(pre_inst.get("url")+"/bwd", content=dumps({"grads": rb["pre_grads"]}), headers={"Content-Type": "application/msgpack"})
             metrics["pre_bwd"] += (time.perf_counter()-t0)*1000

        if USE_NSGA2 and "expert_grads" in rb:
            grad_bytes = _est_grad_bytes(rb["expert_grads"])
            metrics["grad_bytes"] += grad_bytes

            for eid_str, g_data in rb["expert_grads"].items():
                eid = int(eid_str)
                func_grad = f"moe.expert_apply_grad:{eid}"
                insts = get_candidate_instances_for_func(func_grad)
                if not insts: continue

                is_cold_exp = eid in metrics["cold_experts"]
                if is_cold_exp and (micro_id % COLD_ACC_STEPS) != 0:
                    metrics["cold_skipped"] += 1
                    continue

                modes = feasible_modes()
                if eid in metrics["hot_experts"]: modes = [m for m in modes if m in ("hot", "http")]
                elif is_cold_exp: modes = [m for m in modes if m in ("cold", "http")]

                req = {"grad_bytes": grad_bytes, "price_cents_s": DEFAULT_PRICE_CENTS_S}
                choice = nsga2_select(insts, req, STEP_PERIOD_MS, modes=modes)

                if choice:
                    inst, mode = choice
                    cold_delay = await INSTANCE_MGR.check_and_warmup(inst)
                    url = inst.get("url").rstrip("/")
                    t0 = time.perf_counter()

                    if mode == "hot": comm_manager.send_hot(eid_str, {eid_str: g_data})
                    elif mode == "cold": comm_manager.send_cold(eid_str, {eid_str: g_data})
                    else: await client.post(url+"/grad/apply", content=dumps({"grads": {eid_str: g_data}}), headers={"Content-Type": "application/msgpack"})

                    lat = (time.perf_counter()-t0)*1000 + cold_delay
                    metrics["expert_comm"] += lat
                    metrics["dispatch_count"] += 1
                    metrics["mode_counts"][mode] += 1  # [关键] 记录模式选择
                    inst_id = inst.get("id")
                    metrics["inst_choice_counts"][inst_id] += 1
                    trace["exp_bwd"].append({"eid": eid, "inst": inst.get("id"), "mode": mode, "lat": lat})
                    HYBRID_SCHED.update_stats(func_grad, eid, inst, req, lat)

    return {"metrics": metrics, "trace": trace}


# ----------------- 7. 指标聚合 (Step级) -----------------

_metric_buffer = defaultdict(float)
_metric_count = 0

async def run_step(phase, batcher, global_step, metrics_logger):
    train = phase == "train"
    tokens = BATCH_SIZE * BLOCK_SIZE
    x, y = batcher.next_batch()
    micro_bs = BATCH_SIZE // MICRO_BATCHES
    comm = CommManager()
    t_start = time.perf_counter()

    async with httpx.AsyncClient(timeout=120.0) as client:
        tasks = []
        for m in range(MICRO_BATCHES):
            x_mb = x[m*micro_bs : (m+1)*micro_bs]
            y_mb = y[m*micro_bs : (m+1)*micro_bs]
            tasks.append(process_micro_batch(client, comm, x_mb, y_mb, global_step*MICRO_BATCHES+m, m, global_step, tokens, train))

        results_wrapper = await asyncio.gather(*tasks)

    results = [r["metrics"] for r in results_wrapper]
    if train: append_dispatch_log([r["trace"] for r in results_wrapper])

    # 1. 基础指标平均
    step_metrics = {
        "loss": sum(r["loss"] for r in results) / MICRO_BATCHES,
        "acc1": sum(r["acc_top1"] for r in results) / MICRO_BATCHES,
        "pre_lat": sum(r["pre_lat"] for r in results) / MICRO_BATCHES,
        "post_lat": sum(r["post_lat"] for r in results) / MICRO_BATCHES,
        "pre_bwd": sum(r["pre_bwd"] for r in results) / MICRO_BATCHES,
        "post_bwd": sum(r["post_bwd"] for r in results) / MICRO_BATCHES,
        "exp_comm": sum(r["expert_comm"] for r in results) / MICRO_BATCHES,
        "grad_bytes": sum(r["grad_bytes"] for r in results),
        "disp_cnt": sum(r["dispatch_count"] for r in results),
        "step_time": (time.perf_counter() - t_start) * 1000.0
    }

    # 2. [关键修复] 高级指标聚合 (Hot/Cold Ratio)
    # 合并所有微批次中出现的 hot/cold 专家集合
    all_hot = set().union(*[r["hot_experts"] for r in results])
    all_cold = set().union(*[r["cold_experts"] for r in results])
    total_experts_seen = len(all_hot) + len(all_cold)

    step_metrics["hot_ratio"] = len(all_hot) / total_experts_seen if total_experts_seen > 0 else 0.0

    # 聚合通信模式计数
    mode_counts_total = defaultdict(int)
    for r in results:
        for m, c in r["mode_counts"].items():
            mode_counts_total[m] += c

    total_modes = sum(mode_counts_total.values()) or 1
    step_metrics["mode_hot_frac"] = mode_counts_total["hot"] / total_modes
    step_metrics["mode_cold_frac"] = mode_counts_total["cold"] / total_modes
    step_metrics["mode_http_frac"] = mode_counts_total["http"] / total_modes

    if train:
        async with httpx.AsyncClient(timeout=120.0) as client:
            for iid in PRE_STEP_INSTANCE_IDS | POST_STEP_INSTANCE_IDS:
                if iid in INST_BY_ID:
                    try: await client.post(INST_BY_ID[iid]["url"].rstrip("/")+"/step")
                    except: pass

        global _metric_buffer, _metric_count
        for k, v in step_metrics.items():
            _metric_buffer[k] += v
        _metric_count += 1

        if _metric_count >= LOG_TRAIN_EVERY:
            avg = {k: v / _metric_count for k, v in _metric_buffer.items()}

            log("controller", f"Step {global_step}: Loss={avg['loss']:.4f} Time={avg['step_time']:.1f}ms HotRatio={avg['hot_ratio']:.2f}")

            # [关键修复] 将聚合后的 hot/cold 数据传入 StepMetrics
            metrics_logger.log(StepMetrics(
                step=global_step, phase="train",
                loss=avg["loss"],
                acc_top1=avg["acc1"],
                acc_top5=0,
                batch_size=BATCH_SIZE, seq_len=BLOCK_SIZE, tokens=tokens,
                pre_fwd_ms=avg["pre_lat"],
                post_fwd_ms=avg["post_lat"],
                post_bwd_ms=avg["post_bwd"],
                pre_bwd_ms=avg["pre_bwd"],
                step_time_ms=avg["step_time"],
                expert_comm_ms=avg["exp_comm"],
                grad_bytes=avg["grad_bytes"],
                expert_inst_cnt=len(EXPERT_INSTANCE_IDS),
                dispatch_count=avg["disp_cnt"],
                # 填入计算出的平均值
                hot_ratio=avg["hot_ratio"],
                cold_skip_ratio=0, # 简化
                mode_hot_frac=avg["mode_hot_frac"],
                mode_cold_frac=avg["mode_cold_frac"],
                mode_http_frac=avg["mode_http_frac"],
                inst_entropy=0
            ))

            _metric_buffer = defaultdict(float)
            _metric_count = 0
    else:
        metrics_logger.log(StepMetrics(
            step=global_step, phase="val",
            loss=step_metrics["loss"],
            acc_top1=step_metrics["acc1"],
            acc_top5=0,
            batch_size=BATCH_SIZE, seq_len=BLOCK_SIZE, tokens=tokens,
            pre_fwd_ms=step_metrics["pre_lat"],
            post_fwd_ms=step_metrics["post_lat"],
            post_bwd_ms=step_metrics["post_bwd"],
            pre_bwd_ms=step_metrics["pre_bwd"],
            step_time_ms=step_metrics["step_time"],
            expert_comm_ms=step_metrics["exp_comm"],
            grad_bytes=step_metrics["grad_bytes"],
            expert_inst_cnt=len(EXPERT_INSTANCE_IDS),
            dispatch_count=step_metrics["disp_cnt"],
            hot_ratio=step_metrics["hot_ratio"],
            cold_skip_ratio=0,
            mode_hot_frac=step_metrics["mode_hot_frac"],
            mode_cold_frac=step_metrics["mode_cold_frac"],
            mode_http_frac=step_metrics["mode_http_frac"],
            inst_entropy=0
        ))

async def main() -> None:
    log("controller", "Starting training controller")

    # 清空旧的调度日志
    if os.path.exists(DISPATCH_LOG_FILE):
        try:
            os.remove(DISPATCH_LOG_FILE)
            # log("controller", f"Removed old dispatch log: {DISPATCH_LOG_FILE}")
        except:
            pass

    # [修复] 使用关键字参数实例化，避免位置参数错位
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
    # 确保 max_steps 从环境变量读取
    max_steps = int(os.getenv("MAX_STEPS", "1000"))

    while global_step < max_steps:
        phase = "train"
        await run_step(phase, train_batcher, global_step, metrics_logger)
        global_step += 1

        if global_step % VAL_INTERVAL == 0:
            await run_step("val", val_batcher, global_step, metrics_logger)

    log("controller", "Training finished")

if __name__ == "__main__":
    asyncio.run(main())