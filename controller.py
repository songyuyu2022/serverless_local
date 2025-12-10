"""
训练控制器：
- 从本地文本构造 LM 批次
- 调度 pre_fn / post_fn / expert_app 等无服务器函数（基于统一实例池 + 函数映射）
- 在前向阶段插入真正的 MoE 专家前向：
  pre_fn -> (h, router_logits)
  controller -> 根据 router_logits 做 top-k、分发到 expert_app:/fwd
  expert_app -> ExpertMLP 前向
  controller -> 将专家输出按 gate 加权合并，再送入 post_fn
- 在反向阶段根据 Hot/Cold 专家集合，区分通信模式：
  - 热专家优先走 Redis 等 "hot" 通道（低延迟），或退回 http
  - 冷专家优先走 OSS 等 "cold" 通道（高延迟），或退回 http
- 指标记录：
  - train：每 LOG_TRAIN_EVERY 个 step 记录一次“窗口平均”指标
  - val：每次验证 step 都记录（由 VAL_INTERVAL 控制频率）
"""

import os
import asyncio
import json
import time
import math
from typing import Any, Dict, List, Tuple, Set
from collections import defaultdict

import httpx
import torch  # 主要用于 dataset / shared 中的张量处理
import torch.nn.functional as F

from dataset import LMTextBatcher, DATA_PATH_DEFAULT
from shared import dumps, loads, tensor_to_pack, pack_to_tensor
from nsga2_bw import nsga2_select, feasible_modes
from scheduler_hybrid import HYBRID_SCHED
from scheduler_lgbm import record_lgb_training_sample
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

# MoE 路由超参（用于 controller 端的 expert fwd）
MOE_CONFIG = None  # 稍后在加载完函数映射后初始化
TOP_K_DEFAULT = 2

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


# 实例池：平台视角，只知道 runtime/mem/cpu/依赖/价格 等
_all_instances_data = _load_json(INSTANCES_FILE, [])

# 兼容两种格式：
#   1) {"instances": [ ... ]}
#   2) [ ... ]
if isinstance(_all_instances_data, dict):
    ALL_INSTANCES: List[Dict[str, Any]] = _all_instances_data.get("instances", [])
elif isinstance(_all_instances_data, list):
    ALL_INSTANCES = _all_instances_data
else:
    raise RuntimeError(
        f"instances.json 格式错误，期望 dict 或 list，实际是 {type(_all_instances_data)}"
    )

INST_BY_ID: Dict[str, Dict[str, Any]] = {inst.get("id"): inst for inst in ALL_INSTANCES}

# 函数映射：应用视角，知道“moe.pre_fwd”可以在哪些实例上执行
FUNC_MAP: Dict[str, List[str]] = _load_json(FUNC_MAP_FILE, {})

# 预先统计：哪些实例参与 pre/post/expert_apply_grad，用于 /step 和指标
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
log(
    "controller",
    f"PRE_STEP_INSTANCES={len(PRE_STEP_INSTANCE_IDS)}, "
    f"POST_STEP_INSTANCES={len(POST_STEP_INSTANCE_IDS)}, "
    f"EXPERT_INSTANCES={len(EXPERT_INSTANCE_IDS)}",
)

# 初始化 MoE 配置（优先读取 moe_config 中的默认值，再结合 experts 映射）
MOE_CONFIG = load_moe_config(
    {k: v for k, v in FUNC_MAP.items() if k.startswith("moe.expert_fwd:")}
)
TOP_K_DEFAULT = MOE_CONFIG.top_k

# ----------------- 函数候选实例获取 & 调度封装 -----------------


def get_candidate_instances_for_func(func_name: str) -> List[Dict[str, Any]]:
    """
    根据函数逻辑名（如 'moe.pre_fwd', 'moe.post_fwd', 'moe.expert_fwd:0'）从统一实例池中取出候选实例列表。
    """
    ids = FUNC_MAP.get(func_name, [])
    inst_list: List[Dict[str, Any]] = []
    for iid in ids:
        inst = INST_BY_ID.get(iid)
        if inst is not None:
            inst_list.append(inst)

    if not inst_list:
        log("controller", f"[warn] No instances for func={func_name}, ids={ids}")
    return inst_list


def select_instance_for_func(
    func_name: str,
    tokens: int,
    emb_dim: int,
    logical_id: int = 0,
) -> Dict[str, Any]:
    """
    使用 HYBRID_SCHED（LightGBM + NN）在候选实例中选出 cost 最低的实例。
    """
    instances = get_candidate_instances_for_func(func_name)
    if not instances:
        raise RuntimeError(f"No instances configured for func={func_name}")

    req = {"tokens": int(tokens), "emb_dim": int(emb_dim)}

    try:
        chosen, scores = HYBRID_SCHED.select_instances(
            func_type=func_name,
            logical_id=logical_id,
            instances=instances,
            req=req,
            top_k=1,
        )
        inst = chosen[0]
    except Exception as e:
        log(
            "controller",
            f"HYBRID_SCHED.select_instances failed for func={func_name}: {e}, "
            f"fallback to first candidate instance",
        )
        inst = instances[0]

    return inst


# ----------------- 前向调用封装 -----------------


async def call_pre_fwd(
    client: httpx.AsyncClient,
    x_ids_pack: Dict[str, Any],
    micro_id: int,
    tokens: int,
    emb_dim: int,
) -> Tuple[Dict[str, Any], float, Dict[str, Any]]:
    """
    调用 pre_fn /fwd：
      - 通过调度器从 func_map.json + instances.json 中为 'moe.pre_fwd' 选择实例
      - 向该实例的 /fwd 发送 msgpack 请求
    """
    func_name = "moe.pre_fwd"
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

    req_bytes = dumps(payload)
    t0 = time.perf_counter()
    resp = await client.post(
        url,
        content=req_bytes,
        headers={"Content-Type": "application/msgpack"},
        timeout=30.0,
    )
    t1 = time.perf_counter()
    latency_ms = (t1 - t0) * 1000.0

    if resp.status_code != 200:
        print(
            "[controller] pre_fn HTTP error:",
            resp.status_code,
            "text:",
            resp.text[:200],
        )
        raise RuntimeError(f"pre_fn HTTP {resp.status_code}")

    try:
        pre_resp = loads(resp.content)
    except Exception as e:
        print("[controller] pre_fn decode error:", repr(e))
        print("[controller] raw response bytes[:200] =", resp.content[:200])
        raise

    # LightGBM 在线样本：pre
    try:
        record_lgb_training_sample(
            func_type=func_name,
            logical_id=0,
            inst=inst,
            req={"tokens": int(tokens), "emb_dim": int(emb_dim)},
            latency_ms=latency_ms,
        )
    except Exception as e:
        log("controller", f"[warn] record_lgb_training_sample(pre) failed: {e}")

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
    """
    调用 post_fn 前向，并在结束后记录一条 LightGBM 在线训练样本。
    """
    func_name = "moe.post_fwd"
    inst = select_instance_for_func(
        func_name=func_name,
        tokens=tokens,
        emb_dim=emb_dim,
        logical_id=0,
    )
    url = inst.get("url", "").rstrip("/") + "/fwd"
    req_features = {"tokens": int(tokens), "emb_dim": int(emb_dim)}

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

    # LightGBM 在线训练样本（post）
    try:
        record_lgb_training_sample(
            func_type=func_name,
            logical_id=0,
            inst=inst,
            req=req_features,
            latency_ms=latency_ms,
        )
    except Exception as e:
        log("controller", f"[warn] record_lgb_training_sample({func_name}) failed: {e}")

    data = loads(resp.content)
    return data, latency_ms, inst


async def call_expert_fwd_for_eid(
    client: httpx.AsyncClient,
    eid: int,
    x_e: torch.Tensor,
    emb_dim: int,
) -> Tuple[torch.Tensor, Dict[str, Any], float]:
    """
    对某个逻辑专家 eid 执行一次前向：
      - 通过 func_name = f"moe.expert_fwd:{eid}" 从 FUNC_MAP 取候选实例
      - 使用 HYBRID_SCHED 选择实例
      - 调用 /fwd
    """
    func_name = f"moe.expert_fwd:{eid}"
    insts = get_candidate_instances_for_func(func_name)
    if not insts:
        # 没有配置时，直接返回原输入，表示不经过专家变换
        log("controller", f"[expert_fwd] no instances for {func_name}, skip")
        return x_e, {}, 0.0

    inst, _ = HYBRID_SCHED.select_instance(
        func_type=func_name,
        logical_id=eid,
        instances=insts,
        req={"tokens": int(x_e.shape[0]), "emb_dim": int(emb_dim)},
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
    y_e = pack_to_tensor(obj["y"])  # (N,D)

    # LightGBM 在线训练样本（expert fwd）
    try:
        record_lgb_training_sample(
            func_type="moe.expert_fwd",
            logical_id=eid,
            inst=inst,
            req={"tokens": int(x_e.shape[0]), "emb_dim": int(emb_dim)},
            latency_ms=latency_ms,
        )
    except Exception as e:
        log("controller", f"[warn] record_lgb_training_sample(expert_fwd) failed: {e}")

    return y_e, inst, latency_ms


# ----------------- 工具：估算梯度大小 -----------------


def _est_grad_bytes(grad_dict: Dict[str, Any]) -> int:
    """
    根据 packed tensor 估算梯度总字节大小。
    格式约定同 shared.tensor_to_pack：
    - {'shape': [...], 'dtype': 'float32', 'data': b'...'}
    """
    total = 0
    dtype_size = {
        "float32": 4,
        "float16": 2,
        "bfloat16": 2,
        "float64": 8,
        "int8": 1,
        "int16": 2,
        "int32": 4,
        "int64": 8,
    }
    for name, pack in grad_dict.items():
        shape = pack.get("shape", [])
        dtype = pack.get("dtype", "float32")
        numel = 1
        for d in shape:
            numel *= int(d)
        total += numel * dtype_size.get(dtype, 4)
    return total


# ----------------- 单个 step 的完整前向 + 反向 + 通信 -----------------


async def run_step(
    phase: str,
    batcher: LMTextBatcher,
    global_step: int,
    metrics_logger: MetricsLogger,
) -> None:
    train = phase == "train"
    tokens = BATCH_SIZE * BLOCK_SIZE

    # 从数据集中取一个 batch
    x, y = batcher.next_batch()
    assert x.shape == (BATCH_SIZE, BLOCK_SIZE)
    assert y.shape == (BATCH_SIZE, BLOCK_SIZE)

    x_ids = x
    target_ids = y

    # 微批次拆分
    micro_batches = MICRO_BATCHES
    micro_bs = BATCH_SIZE // micro_batches

    COMM = CommManager()

    t_step0 = time.perf_counter()

    # 调度/通信行为统计（论文指标）
    hot_experts_global: List[int] = []
    cold_experts_global: List[int] = []
    hot_experts_step = set()
    cold_experts_step = set()
    cold_total = 0
    cold_skipped = 0
    mode_counts = {"hot": 0, "cold": 0, "http": 0}
    inst_choice_counts = defaultdict(int)
    dispatch_count = 0  # 本 step 实际触发的 expert dispatch 次数

    async with httpx.AsyncClient() as client:
        agg_loss = 0.0
        agg_top1 = 0.0
        agg_top5 = 0.0

        pre_lat_all = 0.0
        post_lat_all = 0.0
        post_bwd_all = 0.0
        pre_bwd_all = 0.0
        expert_comm_ms = 0.0
        grad_bytes = 0
        expert_inst_cnt = len(EXPERT_INSTANCE_IDS)

        for m in range(micro_batches):
            x_mb = x_ids[m * micro_bs : (m + 1) * micro_bs]
            y_mb = target_ids[m * micro_bs : (m + 1) * micro_bs]

            # ---------- pre_fn / fwd ----------
            pre_resp, pre_lat_ms, pre_inst = await call_pre_fwd(
                client=client,
                x_ids_pack=tensor_to_pack(x_mb),
                micro_id=global_step,
                tokens=tokens,
                emb_dim=0,  # 这里暂时不用 emb_dim，可根据模型实际扩展
            )
            pre_lat_all += pre_lat_ms

            hidden_pack = pre_resp["hidden"]
            gate_pack = pre_resp["gate"]  # router logits
            route_info = pre_resp.get("route", None)

            # 记录热/冷专家（根据 pre_fn 的统计）
            if "hot" in pre_resp:
                hot_experts_micro = pre_resp["hot"]
                cold_experts_micro = pre_resp.get("cold", [])
                hot_experts_step.update(hot_experts_micro)
                cold_experts_step.update(cold_experts_micro)
                hot_experts_global = list(hot_experts_step)
                cold_experts_global = list(cold_experts_step)

            # ---------- controller 内部执行真正 MoE 专家前向 ----------
            # 将 hidden_pack 解包成 tensor
            h = pack_to_tensor(hidden_pack).float()  # (B_mb, T, D)
            router_logits = pack_to_tensor(gate_pack).float()  # (B_mb, T, E)

            B_mb, T, D = h.shape
            num_experts = router_logits.shape[-1]
            top_k = max(1, min(TOP_K_DEFAULT, num_experts))

            # top-k + softmax
            topk_vals, topk_idx = torch.topk(router_logits, k=top_k, dim=-1)  # (B,T,K)
            gates = F.softmax(topk_vals, dim=-1)  # (B,T,K)

            # 收集每个 expert 对应的 token 列表
            expert_to_tokens: Dict[int, List[Tuple[int, int, int, float]]] = {}
            topk_idx_np = topk_idx.cpu().numpy()
            gates_np = gates.cpu().numpy()

            for b in range(B_mb):
                for t in range(T):
                    for k_id in range(top_k):
                        eid = int(topk_idx_np[b, t, k_id])
                        gw = float(gates_np[b, t, k_id])
                        expert_to_tokens.setdefault(eid, []).append((b, t, k_id, gw))

            merged_h = torch.zeros_like(h)

            for eid, items in expert_to_tokens.items():
                # 该 expert 对应的输入子 batch
                idx_b = [b for (b, t, k_id, gw) in items]
                idx_t = [t for (b, t, k_id, gw) in items]
                x_e = h[idx_b, idx_t, :]  # (N, D)

                y_e, inst_e, lat_ms = await call_expert_fwd_for_eid(
                    client=client,
                    eid=eid,
                    x_e=x_e,
                    emb_dim=D,
                )
                expert_comm_ms += lat_ms

                # 将专家输出按 gate 写回 merged_h
                i = 0
                for (b, t, k_id, gw) in items:
                    merged_h[b, t, :] += gw * y_e[i]
                    i += 1

                if inst_e:
                    dispatch_count += 1
                    inst_id = inst_e.get("id") or inst_e.get("url") or str(inst_e)
                    inst_choice_counts[inst_id] += 1

            # 如果没有任何 expert 实例可用，就退回原始 h
            if not expert_to_tokens:
                merged_h = h

            # 将专家后的 hidden 打包，送给 post_fn
            hidden_after_expert_pack = tensor_to_pack(merged_h.cpu())

            # ---------- post_fn / fwd ----------
            targets_pack = tensor_to_pack(y_mb)
            post_resp, post_lat_ms, post_inst = await call_post_fwd(
                client=client,
                y_pack=hidden_after_expert_pack,
                targets_pack=targets_pack,
                micro_id=global_step,
                tokens=tokens,
                emb_dim=0,
                train=train,
            )
            post_lat_all += post_lat_ms

            loss = float(post_resp["loss"])
            acc_top1 = float(post_resp.get("acc_top1", 0.0))
            acc_top5 = float(post_resp.get("acc_top5", 0.0))
            agg_loss += loss
            agg_top1 += acc_top1
            agg_top5 += acc_top5

            # ---------- post_fn / bwd ----------
            if train:
                t0 = time.perf_counter()
                grads_pack = post_resp["grads"]
                resp = await client.post(
                    post_inst.get("url", "").rstrip("/") + "/bwd",
                    content=dumps(
                        {
                            "grads": grads_pack,
                            "micro_id": global_step,
                        }
                    ),
                    headers={"Content-Type": "application/msgpack"},
                )
                t1 = time.perf_counter()
                post_bwd_all += (t1 - t0) * 1000.0

                rb = loads(resp.content)
                # 这里假设 rb 里包含 pre_grads / expert_grads 等
                if "pre_grads" in rb:
                    pre_grads = rb["pre_grads"]
                else:
                    pre_grads = None

                # ---------- pre_fn / bwd ----------
                if pre_grads is not None:
                    t0 = time.perf_counter()
                    await client.post(
                        pre_inst.get("url", "").rstrip("/") + "/bwd",
                        content=dumps(
                            {
                                "grads": pre_grads,
                                "micro_id": global_step,
                            }
                        ),
                        headers={"Content-Type": "application/msgpack"},
                    )
                    t1 = time.perf_counter()
                    pre_bwd_all += (t1 - t0) * 1000.0

                # ---------- 专家梯度通信 (NSGA-II + 热/冷模式区分) ----------
                if USE_NSGA2 and "expert_grads" in rb:
                    grads = rb["expert_grads"]
                    if grads:
                        grad_bytes = _est_grad_bytes(grads)
                        log(
                            "controller",
                            f"[train] expert_grads size ≈ {grad_bytes / 1e6:.3f} MB, "
                            f"hot={hot_experts_global}, cold={cold_experts_global}",
                        )

                        all_modes = feasible_modes()

                        # 本 step 参与路由的逻辑专家集合（来自 hot/cold）
                        expert_ids = set()
                        for e in hot_experts_step:
                            expert_ids.add(e)
                        for e in cold_experts_step:
                            expert_ids.add(e)

                        for eid_int in sorted(expert_ids):
                            eid_str = str(eid_int)

                            # 根据函数名从实例池 + func_map 获取候选实例
                            func_name_grad = f"moe.expert_apply_grad:{eid_str}"
                            inst_list = get_candidate_instances_for_func(func_name_grad)
                            if not inst_list:
                                continue

                            # 冷专家统计总数（用于 cold_skip_ratio）
                            if eid_int in cold_experts_step:
                                cold_total += 1

                            # 冷专家降频：仅在部分 step 更新（发送梯度）
                            if eid_int in cold_experts_step and (global_step % COLD_ACC_STEPS) != 0:
                                cold_skipped += 1
                                log(
                                    "controller",
                                    f"[train] skip cold expert eid={eid_str} at step={global_step} "
                                    f"due to COLD_ACC_STEPS={COLD_ACC_STEPS}",
                                )
                                continue

                            # 候选模式集合
                            if eid_int in hot_experts_step:
                                candidate_modes = [m for m in all_modes if m in ("hot", "http")]
                            elif eid_int in cold_experts_step:
                                candidate_modes = [m for m in all_modes if m in ("cold", "http")]
                            else:
                                candidate_modes = list(all_modes)

                            if not candidate_modes:
                                continue

                            req = {
                                "grad_bytes": grad_bytes,
                                "price_cents_s": DEFAULT_PRICE_CENTS_S,
                            }

                            log(
                                "controller",
                                f"[train] NSGA-II for expert {eid_str}, modes={candidate_modes}",
                            )
                            choice = nsga2_select(
                                inst_list,
                                req,
                                deadline_ms=STEP_PERIOD_MS,
                                pop_size=8,
                                generations=3,
                                seed=42,
                                modes=candidate_modes,
                            )
                            log(
                                "controller",
                                f"[train] NSGA-II result for eid={eid_str}: {choice}",
                            )

                            if choice is None:
                                continue

                            inst, mode = choice
                            url = inst.get("url", "").rstrip("/")

                            # 计时：将所有模式的通信时间统一计入 expert_comm_ms
                            t_comm0 = time.perf_counter()
                            if mode == "hot":
                                COMM.send_hot(eid_str, grads)
                                log(
                                    "controller",
                                    f"[train] send_hot to expert {eid_str} (mode=hot), inst={inst.get('id')}",
                                )
                            elif mode == "cold":
                                COMM.send_cold(eid_str, grads)
                                log(
                                    "controller",
                                    f"[train] send_cold to expert {eid_str} (mode=cold), inst={inst.get('id')}",
                                )
                            else:
                                await client.post(
                                    url + "/grad/apply",
                                    content=dumps({"grads": grads}),
                                    headers={"Content-Type": "application/msgpack"},
                                )
                                log(
                                    "controller",
                                    f"[train] /grad/apply via http eid={eid_str}, inst={inst.get('id')}",
                                )
                            t_comm1 = time.perf_counter()
                            comm_latency_ms = (t_comm1 - t_comm0) * 1000.0
                            expert_comm_ms += comm_latency_ms

                            # LightGBM 在线样本记录：expert 通信延迟
                            try:
                                record_lgb_training_sample(
                                    func_type="moe.expert_apply_grad",
                                    logical_id=eid_int,
                                    inst=inst,
                                    req={"tokens": int(tokens), "emb_dim": int(0)},
                                    latency_ms=comm_latency_ms,
                                )
                            except Exception as e:
                                log(
                                    "controller",
                                    f"[warn] record_lgb_training_sample(expert) failed: {e}",
                                )

                            # 调度行为统计
                            dispatch_count += 1
                            if mode in mode_counts:
                                mode_counts[mode] += 1
                            else:
                                mode_counts[mode] = 1
                            inst_id = inst.get("id") or inst.get("url") or str(inst)
                            inst_choice_counts[inst_id] += 1

        # ---------- pre / post：所有共享参数模块 step ----------
        if train:
            # pre / post：对所有实例做 step（单 step 内累积的所有微批梯度在此统一更新）
            async with httpx.AsyncClient() as client_step:
                for iid in PRE_STEP_INSTANCE_IDS:
                    inst = INST_BY_ID.get(iid)
                    if inst is None:
                        continue
                    url = inst.get("url", "").rstrip("/")
                    await client_step.post(url + "/step")

                for iid in POST_STEP_INSTANCE_IDS:
                    inst = INST_BY_ID.get(iid)
                    if inst is None:
                        continue
                    url = inst.get("url", "").rstrip("/")
                    await client_step.post(url + "/step")

            # NOTE:
            # 专家参数的 step 不再由 controller 触发，
            # 而是由 expert_app 内的 grad_poll_loop 自主决定。

    # ---------- 端到端 step 时间 ----------
    t_step1 = time.perf_counter()
    step_time_ms = (t_step1 - t_step0) * 1000.0

    # ---------- 论文级调度行为指标计算 ----------
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

    # ---------- 指标记录 ----------
    mb = micro_batches

    step_metrics = StepMetrics(
        step=global_step,
        phase=phase,
        # 模型效果类
        loss=agg_loss / mb,
        acc_top1=agg_top1 / mb,
        acc_top5=agg_top5 / mb,
        # 规模相关信息
        batch_size=BATCH_SIZE,
        seq_len=BLOCK_SIZE,
        tokens=tokens,
        # 性能类
        pre_fwd_ms=pre_lat_all / mb,
        post_fwd_ms=post_lat_all / mb,
        post_bwd_ms=post_bwd_all / mb,
        pre_bwd_ms=pre_bwd_all / mb,
        step_time_ms=step_time_ms,
        # 通信 & 调度类
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
        # ===== 窗口累积，用于窗口平均 =====
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

        # ===== 到达窗口边界，写一条“窗口平均”记录 =====
        if _train_count >= LOG_TRAIN_EVERY:
            avg = 1.0 / _train_count
            avg_metrics = StepMetrics(
                step=global_step,  # 使用窗口末尾的 step 作为横坐标
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

        # 未到窗口边界：暂不写 train step
        return

    # val 阶段：每次验证 step 都直接写（由 VAL_INTERVAL 控制调用频率）
    metrics_logger.log(step_metrics)


# ----------------- 主训练循环 -----------------


async def main() -> None:
    log("controller", "Starting training controller")

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
    max_steps = int(os.getenv("MAX_STEPS", "1000"))

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
