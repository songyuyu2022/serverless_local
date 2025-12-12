"""
训练控制器 (Final Optimized Version)：
- 架构：并行微批次 (Parallel Micro-batches) + 混合调度 (Heuristic + Online NN)
- 优化：
  1. 默认参数调整为 Batch=32, MicroBatch=4, Steps=500，大幅缩短实验时间
  2. 实现了 Step 级指标聚合 (Buffer)，减少 CSV 写入频率
  3. 保留了详细的调度轨迹日志 (JSONL) 用于论文分析
  4. 包含冷启动模拟和在线学习反馈闭环
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

# ----------------- 1. 全局配置与初始化 -----------------

DATA_PATH = os.getenv("DATA_PATH", DATA_PATH_DEFAULT)

# [优化配置] 针对本地模拟调整的默认参数
# 增大 Batch Size 以减少总 Step 数，同时利用 Micro Batches 并行
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
BLOCK_SIZE = int(os.getenv("BLOCK_SIZE", "64"))
MAX_STEPS = int(os.getenv("MAX_STEPS", "500"))       # 500步即可跑完约1MB数据

# 日志与验证频率
VAL_INTERVAL = int(os.getenv("VAL_INTERVAL", "100")) # 每100步验证一次
LOG_TRAIN_EVERY = int(os.getenv("LOG_TRAIN_EVERY", "10")) # 每10步聚合记录一次指标

# 系统模拟参数
STEP_PERIOD_MS = float(os.getenv("STEP_PERIOD_MS", "200.0"))
USE_NSGA2 = os.getenv("USE_NSGA2", "1") == "1"
COLD_ACC_STEPS = int(os.getenv("COLD_ACC_STEPS", "4"))
DEFAULT_PRICE_CENTS_S = float(os.getenv("DEFAULT_PRICE_CENTS_S", "0.0"))
MICRO_BATCHES = int(os.getenv("MICRO_BATCHES", "4")) # 默认开启4路并行

MOE_CONFIG = None
TOP_K_DEFAULT = 2

# ----------------- 2. 日志工具 -----------------

DISPATCH_LOG_FILE = "dispatch_trace.jsonl"

def append_dispatch_log(traces: List[Dict[str, Any]]):
    """将详细的调度轨迹追加写入 JSONL 文件"""
    if not traces:
        return
    try:
        with open(DISPATCH_LOG_FILE, "a", encoding="utf-8") as f:
            for t in traces:
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[ERROR] Failed to write dispatch log: {e}")

# ----------------- 3. 实例资源加载 -----------------

INSTANCES_FILE = os.getenv("INSTANCES_FILE", "instances.json")
FUNC_MAP_FILE = os.getenv("FUNC_MAP_FILE", "func_map.json")

def _load_json(path: str, default: Any) -> Any:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        log("controller", f"ERROR loading {path}: {e}")
        return default

_all_instances_data = _load_json(INSTANCES_FILE, [])
if isinstance(_all_instances_data, dict):
    ALL_INSTANCES = _all_instances_data.get("instances", [])
else:
    ALL_INSTANCES = _all_instances_data

INST_BY_ID: Dict[str, Dict[str, Any]] = {
    inst.get("id"): inst for inst in ALL_INSTANCES
}
FUNC_MAP = _load_json(FUNC_MAP_FILE, {})

# 预先缓存各类实例ID
PRE_STEP_INSTANCE_IDS = set(FUNC_MAP.get("moe.pre_fwd", []))
POST_STEP_INSTANCE_IDS = set(FUNC_MAP.get("moe.post_fwd", []))
EXPERT_INSTANCE_IDS = set()
for fn_name, ids in FUNC_MAP.items():
    if fn_name.startswith("moe.expert_apply_grad:"):
        for iid in ids:
            if iid in INST_BY_ID:
                EXPERT_INSTANCE_IDS.add(iid)

# 加载 MoE 配置
MOE_CONFIG = load_moe_config(
    {k: v for k, v in FUNC_MAP.items() if k.startswith("moe.expert_fwd:")}
)
TOP_K_DEFAULT = MOE_CONFIG.top_k

log("controller", f"Loaded {len(ALL_INSTANCES)} instances, MicroBatches={MICRO_BATCHES}")

# ----------------- 4. 实例生命周期管理 (冷启动模拟) -----------------

class InstanceManager:
    """
    模拟 Serverless 平台的实例冷启动与回收机制。
    """
    def __init__(self, keep_alive_ms: float = 30000.0): # 30秒无请求则回收(模拟高频冷启动)
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

            if last is None:
                is_cold = True # 首次调用
            elif (now - last) > self.keep_alive_ms:
                is_cold = True # 超时回收

            if is_cold:
                # 获取配置中的冷启动时间，默认 100ms
                cold_ms = float(inst.get("meta", {}).get("cold_start_ms", 100.0))
                # 真实阻塞，模拟容器启动耗时
                await asyncio.sleep(cold_ms / 1000.0)
                delay = cold_ms

            # 更新保活时间
            self.last_access[inst_id] = time.perf_counter() * 1000.0

        return delay

# 全局单例
INSTANCE_MGR = InstanceManager()

# ----------------- 5. 调度与调用封装 -----------------

def get_candidate_instances_for_func(func_name: str) -> List[Dict[str, Any]]:
    ids = FUNC_MAP.get(func_name, [])
    return [INST_BY_ID[i] for i in ids if i in INST_BY_ID]

def select_instance_for_func(func_name: str, tokens: int, emb_dim: int, logical_id: int = 0) -> Dict[str, Any]:
    """通过 Hybrid Scheduler 选择最佳实例"""
    insts = get_candidate_instances_for_func(func_name)
    if not insts:
        raise RuntimeError(f"No instances for {func_name}")

    req = {"tokens": int(tokens), "emb_dim": int(emb_dim)}
    try:
        inst, _ = HYBRID_SCHED.select_instance(func_name, logical_id, insts, req)
        return inst
    except Exception:
        return insts[0] # 降级

async def call_pre_fwd(client, x_ids_pack, micro_id, tokens, emb_dim):
    func_name = "moe.pre_fwd"
    inst = select_instance_for_func(func_name, tokens, emb_dim)

    # 1. 冷启动模拟
    cold_delay = await INSTANCE_MGR.check_and_warmup(inst)

    url = inst.get("url", "").rstrip("/") + "/fwd"
    payload = {"x": x_ids_pack, "micro_id": micro_id, "tokens": tokens, "emb_dim": emb_dim}

    t0 = time.perf_counter()
    resp = await client.post(url, content=dumps(payload), headers={"Content-Type": "application/msgpack"}, timeout=30.0)

    # 2. 计算总延迟 (含冷启动惩罚)
    latency_ms = (time.perf_counter() - t0) * 1000.0 + cold_delay

    # 3. 在线反馈 (Update NN)
    HYBRID_SCHED.update_stats(func_name, 0, inst, {"tokens": tokens, "emb_dim": emb_dim}, latency_ms)

    return loads(resp.content), latency_ms, inst

async def call_post_fwd(client, y_pack, targets_pack, micro_id, tokens, emb_dim, train):
    func_name = "moe.post_fwd"
    inst = select_instance_for_func(func_name, tokens, emb_dim)

    cold_delay = await INSTANCE_MGR.check_and_warmup(inst)

    url = inst.get("url", "").rstrip("/") + "/fwd"
    payload = {"y": y_pack, "targets": targets_pack, "micro_id": micro_id, "tokens": tokens, "emb_dim": emb_dim, "train": train}

    t0 = time.perf_counter()
    resp = await client.post(url, content=dumps(payload), headers={"Content-Type": "application/msgpack"})

    latency_ms = (time.perf_counter() - t0) * 1000.0 + cold_delay

    HYBRID_SCHED.update_stats(func_name, 0, inst, {"tokens": tokens, "emb_dim": emb_dim}, latency_ms)
    return loads(resp.content), latency_ms, inst

async def call_expert_fwd_for_eid(client, eid, x_e, emb_dim):
    func_name = f"moe.expert_fwd:{eid}"
    insts = get_candidate_instances_for_func(func_name)
    if not insts: return x_e, {}, 0.0

    req = {"tokens": int(x_e.shape[0]), "emb_dim": int(emb_dim)}
    inst, _ = HYBRID_SCHED.select_instance(func_name, eid, insts, req)

    cold_delay = await INSTANCE_MGR.check_and_warmup(inst)

    url = inst.get("url", "").rstrip("/") + "/fwd"
    payload = {"x": tensor_to_pack(x_e.cpu())}

    t0 = time.perf_counter()
    resp = await client.post(url, content=dumps(payload), headers={"Content-Type": "application/msgpack"})

    latency_ms = (time.perf_counter() - t0) * 1000.0 + cold_delay

    HYBRID_SCHED.update_stats("moe.expert_fwd", eid, inst, req, latency_ms)
    return pack_to_tensor(loads(resp.content)["y"]), inst, latency_ms

def _est_grad_bytes(grad_dict):
    return sum(np.prod(p["shape"]) * 4 for p in grad_dict.values())

# ----------------- 6. 微批次处理逻辑 (并行核心) -----------------

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

    metrics = defaultdict(float, {
        "hot_experts": set(), "cold_experts": set(),
        "mode_counts": defaultdict(int), "inst_choice_counts": defaultdict(int)
    })

    # 调度轨迹对象
    trace = {
        "step": global_step, "mb": mb_idx, "ts": time.time(),
        "pre": None, "post": None,
        "exp_fwd": [], "exp_bwd": []
    }

    # 1. Pre Processing
    pre_resp, pre_lat, pre_inst = await call_pre_fwd(client, tensor_to_pack(x_mb), micro_id, tokens, 0)
    metrics["pre_lat"] += pre_lat
    trace["pre"] = pre_inst.get("id")

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
                eid = int(idx_np[b,t,k])
                gw = float(gate_np[b,t,k])
                expert_map[eid].append((b,t,k, gw))

    # 3. Parallel Expert Forward
    merged_h = torch.zeros_like(h)
    tasks, eids = [], []
    for eid, items in expert_map.items():
        idx_b, idx_t = [i[0] for i in items], [i[1] for i in items]
        tasks.append(call_expert_fwd_for_eid(client, eid, h[idx_b, idx_t], h.shape[-1]))
        eids.append((eid, items))

    if tasks:
        # 并发执行专家计算
        results = await asyncio.gather(*tasks)
        for i, (y_e, inst, lat) in enumerate(results):
            eid = eids[i][0]
            metrics["expert_comm"] += lat
            metrics["dispatch_count"] += 1
            inst_id = inst.get("id")
            metrics["inst_choice_counts"][inst_id] += 1

            trace["exp_fwd"].append({"eid": eid, "inst": inst_id, "lat": lat})

            # Merge
            items = eids[i][1]
            for j, (b, t, k, gw) in enumerate(items):
                merged_h[b, t] += gw * y_e[j]
    else:
        merged_h = h

    # 4. Post Processing
    post_resp, post_lat, post_inst = await call_post_fwd(client, tensor_to_pack(merged_h.cpu()), tensor_to_pack(y_mb), micro_id, tokens, 0, train)
    metrics["post_lat"] += post_lat
    metrics["loss"] = post_resp["loss"]
    metrics["acc_top1"] = post_resp.get("acc_top1", 0)
    trace["post"] = post_inst.get("id")

    # 5. Backward & Comm (Training only)
    if train:
        # Post Bwd
        t0 = time.perf_counter()
        resp = await client.post(post_inst.get("url")+"/bwd", content=dumps({"grads": post_resp["grads"]}), headers={"Content-Type": "application/msgpack"})
        metrics["post_bwd"] += (time.perf_counter()-t0)*1000

        # Pre Bwd
        rb = loads(resp.content)
        if "pre_grads" in rb:
             t0 = time.perf_counter()
             await client.post(pre_inst.get("url")+"/bwd", content=dumps({"grads": rb["pre_grads"]}), headers={"Content-Type": "application/msgpack"})
             metrics["pre_bwd"] += (time.perf_counter()-t0)*1000

        # Expert Grads (Using NSGA-II to select Mode & Instance)
        if USE_NSGA2 and "expert_grads" in rb:
            grad_bytes = _est_grad_bytes(rb["expert_grads"])
            metrics["grad_bytes"] += grad_bytes

            for eid_str, g_data in rb["expert_grads"].items():
                eid = int(eid_str)
                func_grad = f"moe.expert_apply_grad:{eid}"
                insts = get_candidate_instances_for_func(func_grad)
                if not insts: continue

                # 简单冷热判断
                is_cold_exp = eid in metrics["cold_experts"]
                if is_cold_exp and (micro_id % COLD_ACC_STEPS) != 0:
                    metrics["cold_skipped"] += 1
                    continue

                # 模式过滤
                modes = feasible_modes()
                if eid in metrics["hot_experts"]: modes = [m for m in modes if m in ("hot", "http")]
                elif is_cold_exp: modes = [m for m in modes if m in ("cold", "http")]

                # NSGA-II 决策
                req = {"grad_bytes": grad_bytes, "price_cents_s": DEFAULT_PRICE_CENTS_S}
                choice = nsga2_select(insts, req, STEP_PERIOD_MS, modes=modes)

                if choice:
                    inst, mode = choice

                    # 模拟反向传播的冷启动延迟
                    cold_delay = await INSTANCE_MGR.check_and_warmup(inst)

                    url = inst.get("url").rstrip("/")
                    t0 = time.perf_counter()

                    # 执行通信
                    if mode == "hot":
                        comm_manager.send_hot(eid_str, {eid_str: g_data})
                    elif mode == "cold":
                        comm_manager.send_cold(eid_str, {eid_str: g_data})
                    else:
                        await client.post(url+"/grad/apply", content=dumps({"grads": {eid_str: g_data}}), headers={"Content-Type": "application/msgpack"})

                    # 总耗时
                    lat = (time.perf_counter()-t0)*1000 + cold_delay

                    metrics["expert_comm"] += lat
                    metrics["dispatch_count"] += 1
                    metrics["mode_counts"][mode] += 1
                    inst_id = inst.get("id")
                    metrics["inst_choice_counts"][inst_id] += 1

                    trace["exp_bwd"].append({"eid": eid, "inst": inst.get("id"), "mode": mode, "lat": lat})

                    # 反馈给 NN (通信时间也是性能的一部分)
                    HYBRID_SCHED.update_stats(func_grad, eid, inst, req, lat)

    return {"metrics": metrics, "trace": trace}


# ----------------- 7. Step 级聚合逻辑 -----------------

# 全局 Buffer 用于聚合 LOG_TRAIN_EVERY 步的指标
_metric_buffer = defaultdict(float)
_metric_count = 0

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
    comm = CommManager()

    t_start = time.perf_counter()

    # [核心] 并行执行 Micro Batches
    async with httpx.AsyncClient() as client:
        tasks = []
        for m in range(micro_batches):
            x_mb = x[m * micro_bs : (m + 1) * micro_bs]
            y_mb = y[m * micro_bs : (m + 1) * micro_bs]

            # 唯一 micro_id 保证通信不冲突
            u_id = global_step * MICRO_BATCHES + m

            tasks.append(
                process_micro_batch(
                    client, comm, x_mb, y_mb,
                    u_id, m, global_step, tokens, train
                )
            )

        # 等待所有并行任务完成
        results_wrapper = await asyncio.gather(*tasks)

    # 提取结果
    results = [r["metrics"] for r in results_wrapper]
    traces = [r["trace"] for r in results_wrapper]

    # 记录调度日志
    if train:
        append_dispatch_log(traces)

    # 聚合当前 Step 的统计值 (平均)
    agg_loss = sum(r["loss"] for r in results) / micro_batches
    agg_acc1 = sum(r["acc_top1"] for r in results) / micro_batches
    step_lat = (time.perf_counter() - t_start) * 1000.0

    # 训练模式：触发参数更新 & 累积指标
    if train:
        # 通知所有 Worker 更新参数
        async with httpx.AsyncClient() as client:
            for iid in PRE_STEP_INSTANCE_IDS | POST_STEP_INSTANCE_IDS:
                if iid in INST_BY_ID:
                    try: await client.post(INST_BY_ID[iid]["url"].rstrip("/")+"/step")
                    except: pass

        # 累积到 Buffer
        global _metric_buffer, _metric_count
        _metric_buffer["loss"] += agg_loss
        _metric_buffer["acc"] += agg_acc1
        _metric_buffer["time"] += step_lat
        _metric_count += 1

        # 达到记录周期，写入 CSV 并打印
        if _metric_count >= LOG_TRAIN_EVERY:
            avg_loss = _metric_buffer['loss'] / _metric_count
            avg_time = _metric_buffer['time'] / _metric_count

            log("controller", f"Step {global_step}: Loss={avg_loss:.4f} Time={avg_time:.1f}ms")

            # 构造 Metrics 对象写入 CSV
            metrics_logger.log(StepMetrics(
                step=global_step, phase="train",
                loss=avg_loss,
                acc_top1=_metric_buffer['acc']/_metric_count,
                acc_top5=0,
                batch_size=BATCH_SIZE, seq_len=BLOCK_SIZE, tokens=0,
                pre_fwd_ms=0, post_fwd_ms=0, post_bwd_ms=0, pre_bwd_ms=0,
                step_time_ms=avg_time,
                expert_comm_ms=0, grad_bytes=0, expert_inst_cnt=0,
                dispatch_count=0, hot_ratio=0, cold_skip_ratio=0,
                mode_hot_frac=0, mode_cold_frac=0, mode_http_frac=0, inst_entropy=0
            ))

            # 清空 Buffer
            _metric_buffer = defaultdict(float)
            _metric_count = 0

    # 验证模式：直接记录，不累积
    else:
        metrics_logger.log(StepMetrics(
            step=global_step, phase="val",
            loss=agg_loss, acc_top1=agg_acc1, acc_top5=0,
            batch_size=BATCH_SIZE, seq_len=BLOCK_SIZE, tokens=0,
            pre_fwd_ms=0, post_fwd_ms=0, post_bwd_ms=0, pre_bwd_ms=0,
            step_time_ms=step_lat, expert_comm_ms=0, grad_bytes=0, expert_inst_cnt=0,
            dispatch_count=0, hot_ratio=0, cold_skip_ratio=0,
            mode_hot_frac=0, mode_cold_frac=0, mode_http_frac=0, inst_entropy=0
        ))


# ----------------- 主程序入口 -----------------

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
    max_steps = int(os.getenv("MAX_STEPS", "500")) # 默认改为500

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